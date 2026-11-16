"""Minimal SimPy co-simulation world with Store-and-Run no-drop semantics."""
from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import simpy

from satplan.net import FluidNet
from satplan.timing import clock_factor, iter_time, kv_bytes_per_tok
from satplan.trace import INF, Trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from satplan.metrics import Recorder
    from satplan.planners import RuntimePlanner

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
    decoding: bool = False

    @property
    def reserved_tokens(self) -> int:
        return self.pin + (self.pout if self.output_budget is None else self.output_budget)

    @property
    def kv(self) -> float:
        return self.pin + self.gen

    @property
    def remaining(self) -> float:
        return max(0.0, self.pout - self.gen)


class CoWorld:
    def __init__(
        self,
        env: simpy.Environment,
        trace: Trace,
        cfg: dict,
        *,
        placer: Callable[[CoWorld, Request], int | None],
        planner: RuntimePlanner,
        recorder: Recorder | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.trace = trace
        self.cfg = cfg
        self.placer = placer
        self.planner = planner
        self.rec = recorder
        self.rng = random.Random(seed ^ 0x5EED)
        self.gpu = cfg["gpu"]
        self.power = cfg["power"]
        self.bpt = kv_bytes_per_tok(cfg["model"])
        self.kv_cap = float(self.gpu["kv_gib"]) * 2**30
        self.f_eclipse = clock_factor(float(self.power["P_dark"]), self.power)
        caps = cfg["net"]["link_gbps"]
        self.net = FluidNet(env, [float(caps[k]) * 1e9 / 8 for k in trace.kind])
        self.hop_lat = float(cfg["net"]["hop_latency_s"])
        self.period = float(cfg["control_period_s"])
        self.stream = cfg["delivery"] == "stream"
        self.nodes: dict[int, list[Request]] = {j: [] for j in trace.compute}
        self._node_reserved_tokens: dict[int, int] = {j: 0 for j in trace.compute}
        self.reserved = {j: 0.0 for j in trace.compute}
        self.reqs: list[Request] = []
        self.topo_event = env.event()
        self.acc = {"upload_fail": 0}

        th = cfg.get("thermal", {})
        t0 = float(th.get("T0_C", 25.0))
        self.t_amb_C = float(th.get("T_amb_C", 20.0))
        self.r_th = float(th.get("R_th", 0.2))
        self.c_th = float(th.get("C_th", 150.0))
        self.temp_C = [t0] * trace.n
        self.peak_temp_C = t0

        env.process(self._changes())
        env.process(self._control())

    def node_load(self, j: int) -> int:
        return len(self.nodes[j])

    def net_edge_rates(self) -> list[tuple[int, float]]:
        self.net.advance(self.env.now)
        return sorted(self.net.edge_rates().items())

    @property
    def upload_fail_count(self) -> int:
        return self.acc["upload_fail"]

    @property
    def link_bytes_hops(self) -> float:
        return self.net.bytes_hops

    @property
    def arrival_window_s(self) -> float:
        return float(self.cfg["workload"]["arrival_window_s"])

    def f_of(self, j: int, t: float | None = None) -> float:
        now = self.env.now if t is None else t
        return self.f_eclipse if self.trace.is_eclipse(j, now) else 1.0

    def kv_used(self, j: int) -> float:
        return self._node_reserved_tokens[j] * self.bpt + self.reserved[j]

    def fits(self, j: int, r: Request) -> bool:
        return self.kv_used(j) + r.reserved_tokens * self.bpt <= self.kv_cap

    def iter_of(self, j: int, f: float | None = None) -> float:
        rs = self.nodes[j]
        nd = max(len(rs), 1)
        ctx = sum(r.kv for r in rs)
        return iter_time(nd, ctx, self.f_of(j) if f is None else f, self.gpu)

    def poke(self, j: int) -> None:
        for r in list(self.nodes[j]):
            if r.state == "decode" and r.decoding and r.proc is not None and r.proc.is_alive:
                with contextlib.suppress(RuntimeError):
                    r.proc.interrupt()

    def _attach_node(self, j: int, r: Request) -> None:
        self.nodes[j].append(r)
        self._node_reserved_tokens[j] += r.reserved_tokens

    def _detach_node(self, j: int, r: Request) -> None:
        self.nodes[j].remove(r)
        self._node_reserved_tokens[j] -= r.reserved_tokens

    def log(self, kind: str, **kw) -> None:
        if self.rec is not None:
            self.rec.event(self.env.now, kind, **kw)

    def _thermal_tick(self) -> None:
        tau = self.r_th * self.c_th
        decay = math.exp(-self.period / tau) if tau > 0 else 0.0
        p_idle = float(self.power["P_idle"])
        p_max = float(self.power["P_max"])
        gamma = float(self.power["gamma"])
        for j in self.trace.compute:
            pwr = p_idle if not self.nodes[j] else p_idle + (p_max - p_idle) * (self.f_of(j) ** gamma)
            t_eq = self.t_amb_C + self.r_th * pwr
            self.temp_C[j] = t_eq + (self.temp_C[j] - t_eq) * decay
            if self.temp_C[j] > self.peak_temp_C:
                self.peak_temp_C = self.temp_C[j]

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
                    prev_eclipse[j] = ecl
                    self.poke(j)
            ev, self.topo_event = self.topo_event, self.env.event()
            ev.succeed()

    def _control(self):
        while True:
            yield self.env.timeout(self.period)
            self._thermal_tick()
            self.planner.rebalance(self)

    def submit(self, r: Request) -> None:
        self.reqs.append(r)
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
        self._attach_node(j, r)
        self.poke(j)
        self.log("admit", req=r.i, node=j)
        yield from self._decode_loop(r)
        yield from self._deliver(r)

    def _admit_and_upload(self, r: Request):
        while True:
            j = self.placer(self, r)
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

    def _decode_loop(self, r: Request):
        while r.remaining > 1e-9:
            j = r.node
            it = self.iter_of(j)
            step = min(CHUNK_S, it * r.remaining)
            t0, g0 = self.env.now, r.gen
            r.decoding = True
            with contextlib.suppress(simpy.Interrupt):
                yield self.env.timeout(max(step, 1e-9))
            r.decoding = False
            dt = self.env.now - t0
            got = min(r.remaining, dt / it) if it > 0 else 0.0
            r.gen = g0 + got
            if self.stream and dt > 0:
                r.stall_s += self.trace.disconnected_time(r.origin, j, t0, self.env.now)

    def _deliver(self, r: Request):
        r.state = "deliver"
        r.t_decoded = self.env.now
        j = r.node
        self._detach_node(j, r)
        self.poke(j)
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
        if self.stream:
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
