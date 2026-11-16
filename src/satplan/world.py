"""Minimal SimPy co-simulation world with Store-and-Run, PowerThermalModel, and r3 actuation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import simpy

from satplan.actions import BackgroundFlowSnapshot, MigrationOrder, MigrationSessionSnapshot, PlanAction
from satplan.net import FluidNet
from satplan.power_thermal import PowerThermalModel
from satplan.timing import iter_time, kv_bytes_per_tok
from satplan.trace import INF, Trace
from satplan.views import ForecastView, LocalStateView

if TYPE_CHECKING:
    from collections.abc import Callable

    from satplan.metrics import Recorder
    from satplan.planners import CompositePolicy, RuntimePlanner

CHUNK_S = 30.0


@dataclass(eq=False)
class Request:
    i: int
    origin: int
    t: float
    long: bool
    pin: int
    pout: int
    upload_bytes: float
    deadline: float
    node: int = -1
    state: str = "wait"  # wait | upload | decode | deliver | done
    gen: float = 0.0
    t_admit: float | None = None
    t_decoded: float | None = None
    t_delivered: float | None = None
    stall_s: float = 0.0
    upload_fail: int = 0
    proc: object = None
    trail: list[int] = field(default_factory=list)
    output_budget: int | None = None
    paused: bool = False
    decode_done: simpy.Event | None = None
    mig_dst: int | None = None
    mig_target_bytes: float = 0.0
    mig_received_bytes: float = 0.0
    mig_in_flight: bool = False
    mig_phase: str = "idle"
    mig_prop_end_t: float | None = None
    mig_path_edges: tuple[int, ...] = ()
    mig_path_nodes: tuple[int, ...] = ()
    n_mig: int = 0

    @property
    def reserved_tokens(self) -> int:
        return self.pin + (self.pout if self.output_budget is None else self.output_budget)

    @property
    def kv(self) -> float:
        return self.pin + self.gen

    @property
    def remaining(self) -> float:
        return max(0.0, self.pout - self.gen)

    @property
    def is_active_decode(self) -> bool:
        return self.state == "decode" and not self.paused and self.mig_dst is None and self.remaining > 1e-9


class CoWorld:
    def __init__(  # noqa: PLR0913
        self,
        env: simpy.Environment,
        trace: Trace,
        cfg: dict,
        *,
        policy: CompositePolicy | None = None,
        placer: Callable[[Any, Request], int | None] | None = None,
        planner: RuntimePlanner | None = None,
        recorder: Recorder | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.trace = trace
        self.cfg = cfg
        if policy is None:
            if placer is None or planner is None:
                raise ValueError("CoWorld requires either policy or (placer, planner)")
            from satplan.planners import CompositePolicy, RebalanceInvariants

            policy = CompositePolicy(placer, planner, RebalanceInvariants.from_cfg(cfg))
        self.policy = policy
        self.rec = recorder
        self.rng = random.Random(seed ^ 0x5EED)
        self.gpu = cfg["gpu"]
        self.bpt = kv_bytes_per_tok(cfg["model"])
        self.kv_cap = float(self.gpu["kv_gib"]) * 2**30
        self.pt = PowerThermalModel.from_config(cfg, trace)
        self.temp_C = self.pt.temp_C
        caps = cfg["net"]["link_gbps"]
        self.net = FluidNet(env, [float(caps[k]) * 1e9 / 8 for k in trace.kind])
        self.hop_lat = float(cfg["net"]["hop_latency_s"])
        self.period = float(cfg["control_period_s"])
        self.stream = cfg["delivery"] == "stream"
        self.rs_downlink = bool(cfg.get("rs_downlink", False))
        self.stall_weight = float(cfg["slo"].get("stall_weight", 1.0))
        self.nodes: dict[int, list[Request]] = {j: [] for j in trace.compute}
        self._node_reserved_tokens: dict[int, int] = {j: 0 for j in trace.compute}
        self.reserved = {j: 0.0 for j in trace.compute}
        self.reqs: list[Request] = []
        self._req_by_id: dict[int, Request] = {}
        self.topo_event = env.event()
        self._node_wake: dict[int, simpy.Event] = {j: env.event() for j in trace.compute}
        self._node_clock_f: dict[int, float] = {j: 0.0 for j in trace.compute}
        self._pt_last_t: dict[int, float] = {j: 0.0 for j in trace.compute}
        self._inflight_mig = 0
        self._mig_flows: dict[int, Any] = {}
        self.acc = {"upload_fail": 0}
        self.local_view = LocalStateView(self)
        self.forecast_view = ForecastView(self)

        env.process(self._changes())
        env.process(self._control())
        for j in trace.compute:
            env.process(self._node_decode_loop(j))

    @property
    def peak_temp_C(self) -> float:
        return self.pt.peak_temp_C

    def node_load(self, j: int) -> int:
        return len(self.nodes[j])

    def net_edge_rates(self) -> list[tuple[int, float]]:
        self.net.advance(self.env.now)
        return sorted(self.net.edge_rates().items())

    def upload_edge_rates(self) -> dict[int, float]:
        self.net.advance(self.env.now)
        out: dict[int, float] = {}
        for f in self.net.flows:
            if f.tag == "mig":
                continue
            for _, _, e in f.hops:
                out[e] = out.get(e, 0.0) + f.rate
        return out

    def background_flow_snapshots(self) -> tuple[BackgroundFlowSnapshot, ...]:
        self.net.advance(self.env.now)
        snaps: list[BackgroundFlowSnapshot] = []
        for f in self.net.flows:
            if f.tag == "mig":
                continue
            req_id = f.meta.get("req")
            req = self._req_by_id.get(req_id) if req_id is not None else None
            total = float(req.upload_bytes) if req is not None else float(f.sent + f.remaining)
            snaps.append(
                BackgroundFlowSnapshot(
                    req_id=req_id,
                    hops=tuple((int(a), int(b), int(e)) for a, b, e in f.hops),
                    remaining_bytes=max(0.0, float(f.remaining)),
                    total_bytes=max(0.0, total),
                )
            )
        return tuple(snaps)

    def migration_snapshot(self, req_id: int) -> MigrationSessionSnapshot | None:
        r = self._req_by_id.get(req_id)
        if r is None or r.mig_dst is None:
            return None
        f = self._mig_flows.get(req_id)
        if f is not None:
            self.net.advance(self.env.now)
            chunk_sent = max(0.0, float(f.sent))
            chunk_rem = max(0.0, float(f.remaining))
        else:
            chunk_sent = 0.0
            chunk_rem = 0.0
        prop_rem = 0.0 if r.mig_prop_end_t is None else max(0.0, r.mig_prop_end_t - self.env.now)
        return MigrationSessionSnapshot(
            phase=r.mig_phase,
            dst=r.mig_dst,
            target_bytes=r.mig_target_bytes,
            received_bytes=r.mig_received_bytes,
            chunk_sent_bytes=chunk_sent,
            chunk_rem_bytes=chunk_rem,
            prop_rem_s=prop_rem,
            path_edges=r.mig_path_edges,
            path_nodes=r.mig_path_nodes,
        )

    @property
    def upload_fail_count(self) -> int:
        return self.acc["upload_fail"]

    @property
    def link_bytes_hops(self) -> float:
        return self.net.bytes_hops

    @property
    def arrival_window_s(self) -> float:
        return float(self.cfg["workload"]["arrival_window_s"])

    def _active_reqs(self, j: int) -> list[Request]:
        return [r for r in self.nodes[j] if r.is_active_decode]

    def _active_count(self, j: int) -> int:
        return sum(1 for r in self.nodes[j] if r.is_active_decode)

    def f_of(self, j: int, t: float | None = None) -> float:
        now = self.env.now if t is None else t
        ecl = self.trace.is_eclipse(j, now)
        op = self.pt.current_op(j, self._active_count(j), is_eclipse=ecl)
        return op.clock_f

    def kv_used(self, j: int) -> float:
        return self._node_reserved_tokens[j] * self.bpt + self.reserved[j]

    def fits(self, j: int, r: Request) -> bool:
        return self.kv_used(j) + r.reserved_tokens * self.bpt <= self.kv_cap

    def inflight_migrations(self) -> int:
        return self._inflight_mig

    def req_by_id(self, req_id: int) -> Request | None:
        return self._req_by_id.get(req_id)

    def iter_of(self, j: int, f: float | None = None) -> float:
        rs = self._active_reqs(j)
        nd = max(len(rs), 1)
        ctx = sum(r.kv for r in rs)
        cf = self.f_of(j) if f is None else f
        if cf <= 0.0:
            return INF
        return self.pt.params.iter_time(nd, ctx, cf)

    def wake_node(self, j: int) -> None:
        ev = self._node_wake.get(j)
        if ev is not None and not ev.triggered:
            ev.succeed()

    def _attach_node(self, j: int, r: Request) -> None:
        self.nodes[j].append(r)
        self._node_reserved_tokens[j] += r.reserved_tokens

    def _detach_node(self, j: int, r: Request) -> None:
        self.nodes[j].remove(r)
        self._node_reserved_tokens[j] -= r.reserved_tokens

    def log(self, kind: str, **kw) -> None:
        if self.rec is not None:
            self.rec.event(self.env.now, kind, **kw)

    def apply_plan(self, plan: PlanAction) -> None:
        for j, tier_idx in plan.tier_decisions.items():
            if j in self.nodes:
                self.pt.set_explicit_tier(j, tier_idx)
                self.wake_node(j)
        for rid in plan.pause_ids:
            r = self._req_by_id.get(rid)
            if r is not None and r.state == "decode" and r.mig_dst is None and not r.paused:
                r.paused = True
                self.wake_node(r.node)
        for rid in plan.resume_ids:
            r = self._req_by_id.get(rid)
            if r is not None and r.state == "decode" and r.mig_dst is None and r.paused:
                r.paused = False
                self.wake_node(r.node)
        for order in plan.migrations:
            self._start_migration(order)

    def _start_migration(self, order: MigrationOrder) -> None:
        r = self._req_by_id.get(order.req_id)
        if r is None or r.state != "decode" or r.node == order.dst or order.dst not in self.nodes:
            return
        if r.mig_in_flight or (r.mig_dst is not None and r.mig_dst != order.dst):
            return
        p = self.trace.path(r.node, order.dst, self.env.now)
        if p is None:
            return
        if r.mig_dst is None:
            if not self.fits(order.dst, r):
                return
            self.reserved[order.dst] += r.reserved_tokens * self.bpt
            r.mig_dst = order.dst
            r.mig_target_bytes = max(r.kv * self.bpt, 1.0)
            r.mig_received_bytes = 0.0
            self.wake_node(r.node)
        rem = max(0.0, r.mig_target_bytes - r.mig_received_bytes)
        nbytes = rem if order.nbytes is None else min(max(float(order.nbytes), 0.0), rem)
        r.mig_in_flight = True
        self._inflight_mig += 1
        self.env.process(self._run_migration(r, p, nbytes))

    def _clear_migration(self, r: Request) -> None:
        if r.mig_dst is not None:
            self.reserved[r.mig_dst] -= r.reserved_tokens * self.bpt
        r.mig_dst = None
        r.mig_target_bytes = 0.0
        r.mig_received_bytes = 0.0
        r.mig_phase = "idle"
        r.mig_prop_end_t = None
        r.mig_path_edges = ()
        r.mig_path_nodes = ()

    def _run_migration(self, r: Request, p: tuple[list[int], list[int]], nbytes: float):
        try:
            f = self.net.add("mig", p[0], p[1], nbytes, req=r.i)
            self._mig_flows[r.i] = f
            r.mig_phase = "sending"
            r.mig_path_edges = tuple(p[1])
            r.mig_path_nodes = tuple(p[0])
            self.log("mig_start", req=r.i, src=r.node, dst=r.mig_dst, bytes=nbytes)
            yield f.done
            self._mig_flows.pop(r.i, None)
            r.mig_received_bytes += f.sent if f.failed else nbytes
            if not f.failed and p[1] and self.hop_lat > 0.0:
                prop_s = self.hop_lat * len(p[1])
                r.mig_phase = (
                    "propagating_unlock"
                    if r.mig_received_bytes >= r.mig_target_bytes - 1e-6
                    else "propagating_chunk"
                )
                r.mig_prop_end_t = self.env.now + prop_s
                yield self.env.timeout(prop_s)
                r.mig_prop_end_t = None
            if r.state != "decode":
                self._clear_migration(r)
                return
            if r.mig_dst is not None and r.mig_received_bytes >= r.mig_target_bytes - 1e-6:
                src, dst = r.node, r.mig_dst
                self._detach_node(src, r)
                self._clear_migration(r)
                self._attach_node(dst, r)
                r.node = dst
                r.trail.append(dst)
                r.n_mig += 1
                r.paused = False
                self.wake_node(src)
                self.wake_node(dst)
                self.log("mig_unlock", req=r.i, src=src, dst=dst)
        finally:
            self._mig_flows.pop(r.i, None)
            r.mig_prop_end_t = None
            r.mig_in_flight = False
            self._inflight_mig -= 1
            if r.mig_dst is not None:
                r.mig_phase = "resumable"

    def _settle_pt(self, j: int, now: float, *, is_eclipse: bool) -> None:
        dt = now - self._pt_last_t[j]
        if dt > 0.0:
            self.pt.step_node(j, dt, self._active_count(j), is_eclipse=is_eclipse)
            self._pt_last_t[j] = now

    def _sync_node_clock(self, j: int, *, is_eclipse: bool) -> None:
        next_op = self.pt.current_op(j, self._active_count(j), is_eclipse=is_eclipse)
        if next_op.clock_f != self._node_clock_f[j]:
            self._node_clock_f[j] = next_op.clock_f
            self.wake_node(j)

    def _thermal_tick(self) -> None:
        now = self.env.now
        for j in self.trace.compute:
            ecl = self.trace.is_eclipse(j, now)
            self._settle_pt(j, now, is_eclipse=ecl)
            self._sync_node_clock(j, is_eclipse=ecl)

    def _changes(self):
        prev_eclipse = {j: self.trace.is_eclipse(j, 0.0) for j in self.trace.compute}
        for t in self.trace.change_times():
            if t > self.env.now:
                yield self.env.timeout(t - self.env.now)
            now = self.env.now
            if self.net.flows:
                for f in self.net.abort_dead(lambda e: self.trace.edge_up(e, now)):
                    self.log("cut", tag=f.tag, req=f.meta.get("req"))
            for j in self.trace.compute:
                ecl = self.trace.is_eclipse(j, now)
                if ecl != prev_eclipse[j]:
                    self._settle_pt(j, now, is_eclipse=prev_eclipse[j])
                    prev_eclipse[j] = ecl
                    self.pt.on_eclipse_change(j, ecl)
                    self._sync_node_clock(j, is_eclipse=ecl)
            ev, self.topo_event = self.topo_event, self.env.event()
            ev.succeed()

    def _control(self):
        while True:
            yield self.env.timeout(self.period)
            self._thermal_tick()
            self.policy.rebalance(self)

    def submit(self, r: Request) -> None:
        self.reqs.append(r)
        self._req_by_id[r.i] = r
        r.proc = self.env.process(self._life(r))

    def _life(self, r: Request):
        if r.t > self.env.now:
            yield self.env.timeout(r.t - self.env.now)
        self.log("arrive", req=r.i, o=r.origin, long=r.long)
        j = yield from self._admit_and_upload(r)
        r.node = j
        r.trail.append(j)
        r.t_admit = self.env.now
        r.state = "decode"
        r.decode_done = self.env.event()
        self._attach_node(j, r)
        self.wake_node(j)
        self.log("admit", req=r.i, node=j)
        yield r.decode_done
        yield from self._deliver(r)

    def _admit_and_upload(self, r: Request):
        while True:
            j = self.policy.place(self, r)
            if j is None:
                yield self.topo_event | self.env.timeout(self.period)
                continue
            p = self.trace.path(r.origin, j, self.env.now)
            if p is None:
                yield self.env.timeout(self.period)
                continue
            need = r.reserved_tokens * self.bpt
            self.reserved[j] += need
            r.state = "upload"
            f = self.net.add("up", p[0], p[1], r.upload_bytes, req=r.i)
            self.log("upload", req=r.i, path=p[0])
            yield f.done
            if f.failed:
                self.reserved[j] -= need
                r.upload_fail += 1
                self.acc["upload_fail"] += 1
                r.state = "wait"
                continue
            yield self.env.timeout(self.hop_lat * len(p[1]))
            self.reserved[j] -= need
            return j

    def _advance_active(self, j: int, active: list[Request], dt: float, it: float, t0: float) -> None:
        got_base = dt / it
        now = self.env.now
        for r in active:
            if r.state != "decode":
                continue
            r.gen += min(r.remaining, got_base)
            if self.stream and self.rs_downlink:
                r.stall_s += self.trace.disconnected_time(r.origin, j, t0, now)
            if r.remaining <= 1e-9:
                r.gen = float(r.pout)
                if r.decode_done is not None and not r.decode_done.triggered:
                    r.decode_done.succeed()

    def _node_decode_loop(self, j: int):
        while True:
            active = self._active_reqs(j)
            op = self.pt.current_op(j, len(active), is_eclipse=self.trace.is_eclipse(j, self.env.now))
            self._node_clock_f[j] = op.clock_f
            if not active or op.clock_f <= 0.0:
                if self._node_wake[j].triggered:
                    self._node_wake[j] = self.env.event()
                yield self._node_wake[j]
                continue
            ctx = sum(r.kv for r in active)
            it = self.pt.params.iter_time(len(active), ctx, op.clock_f)
            dt_fin = min(r.remaining * it for r in active)
            step = min(CHUNK_S, max(dt_fin, 1e-9))
            t0 = self.env.now
            if self._node_wake[j].triggered:
                self._node_wake[j] = self.env.event()
            yield self.env.timeout(step) | self._node_wake[j]
            dt = self.env.now - t0
            if dt > 0.0 and it > 0.0:
                self._advance_active(j, active, dt, it, t0)

    def _deliver(self, r: Request):
        r.state = "deliver"
        r.t_decoded = self.env.now
        j = r.node
        self._detach_node(j, r)
        self.wake_node(j)
        if not self.rs_downlink:
            r.t_delivered = self.env.now
            r.state = "done"
            self.log("deliver", req=r.i, node=j, o=r.origin, path=[j])
            return
        t_conn = self.trace.next_conn(r.origin, j, self.env.now)
        if t_conn == INF:
            r.state = "done"
            self.log("lost", req=r.i, node=j)
            return
        if t_conn > self.env.now:
            yield self.env.timeout(t_conn - self.env.now)
        p = self.trace.path(j, r.origin, self.env.now)
        hops = len(p[1]) if p else 1
        yield self.env.timeout(self.hop_lat * hops)
        r.t_delivered = self.env.now
        r.state = "done"
        self.log("deliver", req=r.i, node=j, o=r.origin, path=p[0] if p else [j, r.origin])

    def good(self, r: Request) -> bool:
        if r.t_delivered is None or r.t_delivered > r.deadline:
            return False
        if self.stream and self.rs_downlink:
            dur = (r.t_decoded or r.t_delivered) - (r.t_admit or r.t)
            if dur > 0 and r.stall_s / dur > float(self.cfg["slo"]["max_stall_frac"]):
                return False
        return True


def deadline_for(cfg: dict, t: float, pin: int, pout: int, long: bool) -> float:
    fixed = cfg["slo"].get("long_deadline_s" if long else "short_deadline_s")
    if fixed is not None:
        return t + float(fixed)
    ideal = pout * iter_time(1, pin + pout / 2, 1.0, cfg["gpu"])
    k = float(cfg["slo"]["deadline_factor_long" if long else "deadline_factor_short"])
    return t + max(k * ideal, float(cfg["slo"]["min_deadline_s"]))


def make_requests(cfg: dict, trace: Trace, seed: int) -> list[Request]:
    wl = cfg["workload"]
    rng = random.Random(seed)
    out: list[Request] = []
    window = float(wl["arrival_window_s"])
    rates = wl.get("source_rates_per_min")
    if rates is not None and len(rates) != len(trace.rs):
        raise ValueError("source_rates_per_min must cover each remote-sensing satellite")
    for source_index, o in enumerate(trace.rs):
        rate = float(rates[source_index] if rates is not None else wl["rate_per_rs_per_min"]) / 60.0
        if rate < 0:
            raise ValueError("arrival rate must be nonnegative")
        if rate == 0:
            continue
        t = float(wl.get("start_s", 0.0))
        while True:
            t += rng.expovariate(rate)
            if t >= window:
                break
            long = rng.random() < float(wl["p_long"])
            lo, hi = wl["long_out" if long else "short_out"]
            pout = rng.randint(int(lo), int(hi))
            pin = int(wl["long_pin" if long else "short_pin"])
            up = float(wl["long_upload_MB" if long else "short_upload_MB"]) * 1e6
            budget = hi if wl.get("reserve_output_budget") else None
            out.append(Request(0, o, t, long, pin, pout, up, deadline_for(cfg, t, pin, pout, long), output_budget=budget))
    out.sort(key=lambda r: (r.t, r.origin))
    for i, r in enumerate(out):
        r.i = i
    return out
