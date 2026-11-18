"""Shared worlds, planners, metrics, and diagnostics for LAB-2082."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import simpy
from experiments.lab2046_common import ReplayPlacer, ZeroCostAtomicWorld
from experiments.lab2046_two_stage_proof.exp2_congestion_tradeoff import NoShareDiagWorld
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, _pct, summarize
from satplan.net import allocate_max_min_rates
from satplan.planners import CompositePolicy, RebalanceInvariants
from satplan.trace import INF, Trace
from satplan.world import CoWorld, Request


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class DiagnosticWorld(CoWorld):
    """CoWorld with Queue-Reroute support, causal request tracking, and resource checks."""

    def __init__(self, *args: Any, allow_queue_reroute: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_queue_reroute = allow_queue_reroute
        self._upload_flows: dict[int, tuple[int, Any]] = {}
        self._reroute_dst: dict[int, int] = {}
        self.reroute_count: int = 0
        self.thermal_violations: int = 0
        self.dod_violations: int = 0
        self.decision_logs: list[dict[str, Any]] = []
        self.env.process(self._monitor_resources())

    def _monitor_resources(self):
        while True:
            for j in self.trace.compute:
                if self.pt.temp_C[j] > self.pt.params.t_max_C + 1e-5:
                    self.thermal_violations += 1
                if self.pt.dod_remaining_j[j] < -1e-5:
                    self.dod_violations += 1
            yield self.env.timeout(self.period)

    def _admit_and_upload(self, r: Request):
        while True:
            if r.i in self._reroute_dst:
                j: int | None = self._reroute_dst.pop(r.i)
                if not self.fits(j, r):
                    j = None
            else:
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
            self._upload_flows[r.i] = (j, f)
            self.log("upload", req=r.i, path=p[0], dst=j)
            yield f.done
            self._upload_flows.pop(r.i, None)
            if r.i in self._reroute_dst:
                # Cancelled by Queue-Reroute; release old reservation and retry to new target
                self.reserved[j] -= need
                r.state = "wait"
                continue
            if f.failed:
                self.reserved[j] -= need
                r.upload_fail += 1
                self.acc["upload_fail"] += 1
                r.state = "wait"
                continue
            yield self.env.timeout(self.hop_lat * len(p[1]))
            if r.i in self._reroute_dst:
                self.reserved[j] -= need
                r.state = "wait"
                continue
            self.reserved[j] -= need
            return j

    def reroute_upload(self, req_id: int, new_dst: int) -> bool:
        """Reroute a request that has NOT started decoding yet (state == 'upload')."""
        if not self.allow_queue_reroute:
            return False
        r = self._req_by_id.get(req_id)
        if r is None or r.state != "upload" or req_id not in self._upload_flows:
            return False
        old_dst, f = self._upload_flows[req_id]
        if old_dst == new_dst or not self.fits(new_dst, r):
            return False
        if self.trace.path(r.origin, new_dst, self.env.now) is None:
            return False
        self._reroute_dst[req_id] = new_dst
        self.reroute_count += 1
        self.log("queue_reroute", req=req_id, src=old_dst, dst=new_dst)
        self.net.cancel(f)
        if not f.done.triggered:
            f.failed = True
            f.done.succeed()
        return True


class ContentionFreeDiagnosticWorld(DiagnosticWorld, NoShareDiagWorld):
    """Dyn-CF: No inter-flow link contention for migrations, preserves single-flow cost & constraints."""


class ZeroCostDiagnosticWorld(DiagnosticWorld, ZeroCostAtomicWorld):
    """Dyn-Zero: Zero-cost atomic migration, preserves connectivity, KV, thermal/DoD."""


def estimate_causal_finish(view: Any, node: int, req: Request, t_start: float, *, joining: bool = True) -> float:
    """Estimate decode finish time using ONLY causal predicted_remaining (never true pout)."""
    extra = req if joining else None
    it_lit = view.iter_of_at(node, 1.0, extra) if hasattr(view, "iter_of_at") else view.iter_of(node, extra)
    if hasattr(view, "power_segments"):
        it_ecl = view.iter_of_at(node, view.f_eclipse, extra)
        left = req.predicted_remaining
        if left <= 0:
            return t_start
        for a, b, ecl in view.power_segments(node, t_start):
            it = it_ecl if ecl else it_lit
            if it == INF:
                continue
            span = b - a
            if left * it <= span:
                return a + left * it
            left -= span / it
        return INF
    # LocalStateView fallback (no future eclipse lookahead)
    it_cur = view.iter_of(node, extra)
    if it_cur == INF:
        return INF
    return t_start + req.predicted_remaining * it_cur


def estimate_shared_mig_time(view: Any, src: int, dst: int, nbytes: float) -> float | None:
    """Estimate migration transfer + propagation time accounting for active flows on path."""
    p = view.path(src, dst)
    if p is None:
        return None
    edges = p[1]
    if not edges:
        return 0.0
    # Use edge_utilization() from LocalStateView to estimate available bandwidth
    util = view.edge_utilization()
    edge_rates = view.edge_rates()
    min_bw = INF
    for e in edges:
        # Approximate capacity from edge_rates / util or transfer_estimate
        u = util.get(e, 0.0)
        if u > 1e-9 and e in edge_rates:
            cap = edge_rates[e] / u
        else:
            # Fallback: use transfer_estimate for single edge or 0.6 Gbps default
            cap = view._w.net.cap[e] if hasattr(view, "_w") else 0.6e9 / 8
        # Available share when adding 1 flow alongside existing load: max(cap * (1 - u), cap / (est_users + 1))
        avail = max(cap * max(0.05, 1.0 - u), cap * 0.1)
        min_bw = min(min_bw, avail)
    if min_bw <= 0 or min_bw == INF:
        return INF
    return nbytes / min_bw + view.hop_latency_s * len(edges)


class CausalReactivePlanner:
    """Runtime planner supporting eclipse + K1 causal overload triggers, with Naive or Net-Aware selection."""

    requires_forecast: bool = False

    def __init__(
        self,
        *,
        mode: str = "naive",  # "naive" | "aware"
        enable_eclipse_trigger: bool = True,
        enable_k1_trigger: bool = True,
        max_orders_per_round: int = 4,
    ) -> None:
        self.mode = mode
        self.enable_eclipse_trigger = enable_eclipse_trigger
        self.enable_k1_trigger = enable_k1_trigger
        self.max_orders_per_round = max_orders_per_round
        self.decision_log: list[dict[str, Any]] = []

    def _causal_rem_decode_s(self, view: Any, node: int, req: Request, *, extra_load: int = 0, extra_kv: float = 0.0, joining: bool = False) -> float:
        it = view.iter_of(node, req if joining else None, extra_load=extra_load, extra_kv=extra_kv)
        if it == INF:
            return INF
        return req.predicted_remaining * it

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        inflight = view.inflight_migrations
        if inflight >= invariants.max_inflight:
            return PlanAction()

        rem_kv = {j: view.kv_headroom(j) for j in view.compute_nodes}
        extra_load: dict[int, int] = defaultdict(int)
        extra_kv: dict[int, float] = defaultdict(float)
        orders: list[MigrationOrder] = []

        for src in view.compute_nodes:
            src_ecl = view.is_eclipse(src)
            src_iter = view.iter_of(src)
            for req in view.node_requests(src):
                if inflight >= invariants.max_inflight or len(orders) >= self.max_orders_per_round:
                    return PlanAction(migrations=tuple(orders))
                if (
                    req.state != "decode"
                    or req.mig_in_flight
                    or req.mig_dst is not None
                    or (invariants.max_mig_per_req is not None and req.n_mig >= invariants.max_mig_per_req)
                ):
                    continue

                # Check trigger
                trigger_reason: str | None = None
                if self.enable_eclipse_trigger and src_ecl:
                    trigger_reason = "eclipse"
                elif self.enable_k1_trigger:
                    est_rem = req.predicted_remaining * src_iter if src_iter < INF else INF
                    if view.now + est_rem > req.deadline:
                        trigger_reason = "k1_overload"

                if trigger_reason is None:
                    continue

                need = req.reserved_tokens * view.bpt
                kv_bytes = max(req.kv * view.bpt, 1.0)
                src_finish = view.now + (req.predicted_remaining * src_iter if src_iter < INF else INF)

                reachable_cnt = 0
                feasible_cnt = 0
                reject_reasons: dict[str, int] = defaultdict(int)
                best_dst: int | None = None
                best_key: tuple[float, ...] | None = None

                for dst in view.compute_nodes:
                    if dst == src:
                        continue
                    hops = view.hop_count(src, dst)
                    if hops is None:
                        reject_reasons["no_contact"] += 1
                        continue
                    reachable_cnt += 1
                    if view.is_eclipse(dst):
                        reject_reasons["eclipse_dst"] += 1
                        continue
                    if rem_kv[dst] < need:
                        reject_reasons["no_memory"] += 1
                        continue
                    dst_iter = view.iter_of(dst, req, extra_load=extra_load[dst], extra_kv=extra_kv[dst])
                    if dst_iter >= src_iter:
                        reject_reasons["compute_too_slow"] += 1
                        continue

                    if self.mode == "aware":
                        mig_t = estimate_shared_mig_time(view, src, dst, kv_bytes)
                        if mig_t is None or mig_t == INF:
                            reject_reasons["network_too_slow"] += 1
                            continue
                        est_finish = view.now + mig_t + req.predicted_remaining * dst_iter
                        if trigger_reason == "k1_overload" and est_finish >= src_finish:
                            reject_reasons["network_too_slow"] += 1
                            continue
                        feasible_cnt += 1
                        key = (est_finish, mig_t, dst)
                    else:
                        feasible_cnt += 1
                        key = (dst_iter, float(hops), float(dst))

                    if best_key is None or key < best_key:
                        best_key = key
                        best_dst = dst

                self.decision_log.append({
                    "t": round(view.now, 2),
                    "req": req.i,
                    "src": src,
                    "trigger": trigger_reason,
                    "reachable_targets": reachable_cnt,
                    "feasible_targets": feasible_cnt,
                    "chosen_dst": best_dst,
                    "reject_reasons": dict(reject_reasons),
                })

                if best_dst is not None:
                    orders.append(MigrationOrder(req_id=req.i, dst=best_dst))
                    rem_kv[best_dst] -= need
                    extra_load[best_dst] += 1
                    extra_kv[best_dst] += req.kv
                    inflight += 1

        return PlanAction(migrations=tuple(orders))


class QueueReroutePlanner:
    """Only reroutes requests still in 'upload' state before decode starts."""

    requires_forecast: bool = False

    def __init__(self, world_ref: DiagnosticWorld) -> None:
        self._w = world_ref

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        for req in list(self._w.reqs):
            if req.state != "upload" or req.i not in self._w._upload_flows:
                continue
            cur_dst, _ = self._w._upload_flows[req.i]
            cur_iter = view.iter_of(cur_dst, req)
            if cur_iter == INF or view.now + req.predicted_remaining * cur_iter > req.deadline or view.is_eclipse(cur_dst):
                best_dst: int | None = None
                best_iter = cur_iter
                for dst in view.compute_nodes:
                    if dst == cur_dst or view.is_eclipse(dst) or not view.fits(dst, req):
                        continue
                    if view.hop_count(req.origin, dst) is None:
                        continue
                    d_iter = view.iter_of(dst, req)
                    if d_iter < best_iter * 0.95:
                        best_iter = d_iter
                        best_dst = dst
                if best_dst is not None:
                    self._w.reroute_upload(req.i, best_dst)
        return PlanAction()


def extended_metrics(w: CoWorld) -> dict[str, Any]:
    base = summarize(w)
    rs = w.reqs
    shorts = [r for r in rs if not r.long]
    longs = [r for r in rs if r.long]
    good_short = [r for r in shorts if w.good(r)]
    good_long = [r for r in longs if w.good(r)]
    window = w.arrival_window_s
    horizon = w.trace.horizon
    lat_all = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in rs]
    lat_short = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in shorts]
    lat_long = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in longs]
    return {
        **base,
        "n_short": len(shorts),
        "good_short": len(good_short),
        "good_short_frac": len(good_short) / max(1, len(shorts)),
        "throughput_req_s": base["delivered"] / window,
        "e2e_p50_all_s": _pct(lat_all, 50),
        "e2e_p99_all_s": _pct(lat_all, 99),
        "short_p95_all_s": _pct(lat_short, 95),
        "n_migrations": sum(r.n_mig for r in rs),
        "reroute_count": getattr(w, "reroute_count", 0),
        "thermal_violations": getattr(w, "thermal_violations", 0),
        "dod_violations": getattr(w, "dod_violations", 0),
    }
