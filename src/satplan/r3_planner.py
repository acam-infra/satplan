"""ProblemBuilder and R3RuntimePlanner wiring R3Solver behind ForecastView and CompositePolicy."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from satplan.actions import MigrationOrder, PlanAction
from satplan.r3_solver import (
    R3EdgeSpec,
    R3NodeSpec,
    R3Problem,
    R3RequestSpec,
    R3Result,
    R3Solver,
    R3TierSpec,
)

if TYPE_CHECKING:
    from satplan.planners import RebalanceInvariants
    from satplan.power_thermal import PhysicsParams
    from satplan.views import ForecastView
    from satplan.world import Request


def _tier_specs(params: PhysicsParams, nd_rep: int, ctx_rep: float) -> tuple[R3TierSpec, ...]:
    if params.discrete:
        raw_tiers = [(t.index, t.power_w, t.clock_f) for t in params.tiers]
    else:
        raw_tiers = [(0, 0.0, 0.0), (1, params.p_max, 1.0)]
    specs: list[R3TierSpec] = []
    for idx, power_w, clock_f in raw_tiers:
        rate = 0.0 if clock_f <= 0.0 else 1.0 / max(1e-9, params.iter_time(nd_rep, ctx_rep, clock_f))
        specs.append(R3TierSpec(index=idx, power_w=power_w, rate_tok_s=rate))
    return tuple(specs)


def _node_tier_rates(params: PhysicsParams, nd_j: int, ctx_j: float) -> tuple[float, ...]:
    if params.discrete:
        clocks = [t.clock_f for t in params.tiers]
    else:
        clocks = [0.0, 1.0]
    return tuple(0.0 if cf <= 0.0 else 1.0 / max(1e-9, params.iter_time(max(1, nd_j), ctx_j, cf)) for cf in clocks)


def _req_kv_ub(params: PhysicsParams, r: Request, window_s: float) -> float:
    clocks = [t.clock_f for t in params.tiers if t.clock_f > 0.0] if params.discrete else [1.0]
    max_rate = max((1.0 / max(1e-9, params.iter_time(1, r.kv, cf)) for cf in clocks), default=0.0)
    return min(float(r.pin + r.pout), float(r.kv) + min(float(r.remaining), max_rate * window_s))


class ProblemBuilder:
    """Maps a ForecastView snapshot and PhysicsParams into a pure-data R3Problem."""

    def __init__(
        self,
        *,
        horizon_slots: int = 6,
        dt_s: float | None = None,
        cvar_alpha: float = 0.95,
        cvar_weight: float = 1e-3,
        stage2_eps: float = 1e-5,
        time_limit_s: float = 0.5,
    ) -> None:
        if horizon_slots <= 0:
            raise ValueError("horizon_slots must be positive")
        self.horizon_slots = horizon_slots
        self.dt_s = dt_s
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.stage2_eps = stage2_eps
        self.time_limit_s = time_limit_s

    def build(self, view: ForecastView, invariants: RebalanceInvariants) -> R3Problem:
        dt_s = float(self.dt_s if self.dt_s is not None else invariants.interval_s)
        rem_horizon = max(dt_s, view.horizon_s - view.now)
        horizon_slots = max(1, min(self.horizon_slots, int(math.ceil(rem_horizon / dt_s))))
        params = view.physics_params

        decode_reqs, decode_by_node = self._collect_decode_reqs(view)
        window_s = horizon_slots * dt_s
        kv_ubs = {r.i: _req_kv_ub(params, r, window_s) for r in decode_reqs}
        nd_rep, ctx_rep = self._rep_batch(view, decode_reqs)
        tiers = _tier_specs(params, nd_rep, ctx_rep)
        nodes = self._build_nodes(view, params, decode_reqs, decode_by_node, horizon_slots, dt_s, ctx_rep, kv_ubs)
        edges, routes = self._build_edges_and_routes(view, horizon_slots, dt_s)
        requests = self._build_requests(view, decode_reqs, tiers, horizon_slots, dt_s, kv_ubs)
        kv_sorted = sorted(kv_ubs.values(), reverse=True)
        batch_tier_rates = (
            tuple(_node_tier_rates(params, k, sum(kv_sorted[:k])) for k in range(1, len(decode_reqs) + 1))
            if decode_reqs
            else None
        )

        return R3Problem(
            horizon_slots=horizon_slots,
            dt_s=dt_s,
            nodes=nodes,
            tiers=tiers,
            edges=edges,
            requests=requests,
            t_amb_C=params.t_amb_C,
            t_max_C=params.t_max_C,
            r_th=params.r_th,
            c_th=params.c_th,
            eclipse_dod_budget_j=params.eclipse_dod_budget_j,
            max_inflight=invariants.max_inflight,
            max_mig_per_req=invariants.max_mig_per_req,
            hop_latency_s=view.hop_latency_s,
            routes=routes,
            batch_tier_rates=batch_tier_rates,
            cvar_alpha=self.cvar_alpha,
            cvar_weight=self.cvar_weight,
            stage2_eps=self.stage2_eps,
            time_limit_s=self.time_limit_s,
        )

    @staticmethod
    def _collect_decode_reqs(view: ForecastView) -> tuple[list[Request], dict[int, list[Request]]]:
        decode_reqs: list[Request] = []
        decode_by_node: dict[int, list[Request]] = {}
        for j in view.compute_nodes:
            reqs_j = [r for r in view.node_requests(j) if r.state == "decode" and r.remaining > 1e-9]
            decode_by_node[j] = reqs_j
            decode_reqs.extend(reqs_j)
        decode_reqs.sort(key=lambda r: r.i)
        return decode_reqs, decode_by_node

    @staticmethod
    def _rep_batch(view: ForecastView, decode_reqs: list[Request]) -> tuple[int, float]:
        if not decode_reqs:
            return 1, 0.0
        active_nodes = sum(1 for j in view.compute_nodes if view.active_load(j) > 0)
        denom = max(1, active_nodes)
        return max(1, round(len(decode_reqs) / denom)), sum(r.kv for r in decode_reqs) / denom

    @staticmethod
    def _build_nodes(  # noqa: PLR0913
        view: ForecastView,
        params: PhysicsParams,
        decode_reqs: list[Request],
        decode_by_node: dict[int, list[Request]],
        horizon_slots: int,
        dt_s: float,
        ctx_rep: float,
        kv_ubs: dict[int, float],
    ) -> tuple[R3NodeSpec, ...]:
        incoming_mig_bytes: dict[int, float] = {j: 0.0 for j in view.compute_nodes}
        for r in decode_reqs:
            if r.mig_dst is not None and r.mig_dst in incoming_mig_bytes:
                incoming_mig_bytes[r.mig_dst] += r.reserved_tokens * view.bpt

        nodes: list[R3NodeSpec] = []
        for j in view.compute_nodes:
            res_decode_bytes = sum(r.reserved_tokens * view.bpt for r in decode_by_node[j])
            kv_cap_bytes = view.kv_headroom(j) + res_decode_bytes + incoming_mig_bytes[j]
            is_eclipse = tuple(
                view.is_eclipse_at(j, view.now + (t + 0.5) * dt_s) for t in range(horizon_slots)
            )
            dod_rem = None if params.eclipse_dod_budget_j is None else view.dod_remaining_j(j)
            active_j = [r for r in decode_by_node[j] if r.is_active_decode]
            nd_j = len(active_j) if active_j else len(decode_by_node[j])
            ctx_j = sum(kv_ubs[r.i] for r in active_j) if active_j else sum(kv_ubs[r.i] for r in decode_by_node[j])
            if nd_j == 0:
                nd_j, ctx_j = 1, ctx_rep
            nodes.append(
                R3NodeSpec(
                    node_id=j,
                    init_temp_C=view.temp_C(j),
                    kv_cap_bytes=kv_cap_bytes,
                    is_eclipse=is_eclipse,
                    init_dod_remaining_j=dod_rem,
                    tier_rates_tok_s=_node_tier_rates(params, nd_j, ctx_j),
                )
            )
        return tuple(nodes)

    @staticmethod
    def _build_edges_and_routes(
        view: ForecastView, horizon_slots: int, dt_s: float
    ) -> tuple[tuple[R3EdgeSpec, ...], dict[tuple[int, int, int], tuple[int, ...]]]:
        compute_set = set(view.compute_nodes)
        cur_rates = view.upload_edge_rates()
        edges: list[R3EdgeSpec] = []
        dir_edge_idx: dict[tuple[int, int], int] = {}
        for e_idx, (u, v) in enumerate(view.edges):
            if u not in compute_set or v not in compute_set:
                continue
            cap_bps = view.edge_cap_Bps(e_idx) * 8.0
            tot_bps = tuple(
                cap_bps if view.edge_up_at(e_idx, view.now + (t + 0.5) * dt_s) else 0.0
                for t in range(horizon_slots)
            )
            bg_0 = min(tot_bps[0], cur_rates.get(e_idx, 0.0) * 8.0)
            bg_bps = (bg_0,) + (0.0,) * (horizon_slots - 1)
            dir_edge_idx[(u, v)] = len(edges)
            edges.append(R3EdgeSpec(src=u, dst=v, tot_bps=tot_bps, bg_bps=bg_bps))
            dir_edge_idx[(v, u)] = len(edges)
            edges.append(R3EdgeSpec(src=v, dst=u, tot_bps=tot_bps, bg_bps=bg_bps))

        routes: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for t in range(horizon_slots):
            t_query = view.now if t == 0 else view.now + (t + 0.5) * dt_s
            for s in view.compute_nodes:
                for d in view.compute_nodes:
                    if s == d:
                        continue
                    p = view.path_at(s, d, t_query)
                    if p is None or not p[1]:
                        continue
                    p_nodes = p[0]
                    if all(pn in compute_set for pn in p_nodes):
                        routes[(s, d, t)] = tuple(
                            dir_edge_idx[(p_nodes[k], p_nodes[k + 1])] for k in range(len(p_nodes) - 1)
                        )
        return tuple(edges), routes

    @staticmethod
    def _build_requests(  # noqa: PLR0913
        view: ForecastView,
        decode_reqs: list[Request],
        tiers: tuple[R3TierSpec, ...],
        horizon_slots: int,
        dt_s: float,
        kv_ubs: dict[int, float],
    ) -> tuple[R3RequestSpec, ...]:
        max_rate = max((t.rate_tok_s for t in tiers), default=0.0)
        horizon_cap_tok = max(1.0, 0.8 * max_rate * horizon_slots * dt_s)
        req_specs: list[R3RequestSpec] = []
        for r in decode_reqs:
            rem_tok = float(r.remaining)
            raw_dl = max(0.0, r.deadline - view.now)
            if rem_tok > horizon_cap_tok:
                scale = horizon_cap_tok / rem_tok
                eff_tok = horizon_cap_tok
                eff_dl = max(dt_s, raw_dl * scale)
            else:
                eff_tok = rem_tok
                eff_dl = raw_dl
            snap = view.migration_snapshot(r.i)
            kv_bytes = max(snap.target_bytes, 1.0) if snap is not None else max(kv_ubs[r.i] * view.bpt, 1.0)
            req_specs.append(
                R3RequestSpec(
                    req_id=r.i,
                    init_node=r.node,
                    kv_bytes=kv_bytes,
                    remaining_tokens=eff_tok,
                    deadline_s=eff_dl,
                    weight=1.0,
                    elapsed_s=max(0.0, view.now - r.t),
                    reserved_kv_bytes=float(r.reserved_tokens * view.bpt),
                    currently_paused=r.paused,
                    init_mig_dst=r.mig_dst,
                    init_mig_received_bytes=float(r.mig_received_bytes),
                    init_mig_in_flight=r.mig_in_flight,
                    init_mig=snap,
                    n_mig=r.n_mig,
                )
            )
        return tuple(req_specs)


def _validate_action(
    action: PlanAction,
    view: ForecastView,
    invariants: RebalanceInvariants,
    req_map: dict[int, Request],
) -> str | None:
    avail_slots = max(0, invariants.max_inflight - view.inflight_migrations)
    if len(action.migrations) > avail_slots:
        return f"too_many_migrations: {len(action.migrations)} > {avail_slots}"
    rem_kv = {j: view.kv_headroom(j) for j in view.compute_nodes}
    seen_reqs: set[int] = set()
    for order in action.migrations:
        if order.req_id in seen_reqs:
            return f"duplicate_migration_req: {order.req_id}"
        seen_reqs.add(order.req_id)
        req = req_map.get(order.req_id)
        if req is None or req.state != "decode":
            return f"invalid_migration_req: {order.req_id}"
        if req.mig_in_flight:
            return f"req_already_in_flight: {order.req_id}"
        if order.dst not in rem_kv or order.dst == req.node:
            return f"invalid_migration_dst: {order.dst}"
        if view.path(req.node, order.dst) is None:
            return f"no_path: {req.node}->{order.dst}"
        if req.mig_dst is not None:
            if order.dst != req.mig_dst:
                return f"mig_dst_mismatch: {order.dst} != {req.mig_dst}"
        else:
            if invariants.max_mig_per_req is not None and req.n_mig >= invariants.max_mig_per_req:
                return f"mig_budget_exhausted: {order.req_id}"
            need = req.reserved_tokens * view.bpt
            if rem_kv[order.dst] < need - 1e-6:
                return f"insufficient_kv_headroom: dst={order.dst}"
            rem_kv[order.dst] -= need
    return None


def _safe_fallback_action(
    view: ForecastView,
    invariants: RebalanceInvariants,
    req_map: dict[int, Request],
) -> PlanAction:
    tier_decisions: dict[int, int | None] = (
        {j: None for j in view.compute_nodes} if view.physics_params.discrete else {}
    )
    resume_ids = tuple(
        r.i
        for r in req_map.values()
        if r.state == "decode" and r.paused and r.mig_dst is None
    )
    avail_slots = max(0, invariants.max_inflight - view.inflight_migrations)
    cont_migs: list[MigrationOrder] = []
    for r in sorted(req_map.values(), key=lambda x: x.i):
        if len(cont_migs) >= avail_slots:
            break
        if r.state == "decode" and r.mig_dst is not None and not r.mig_in_flight and view.path(r.node, r.mig_dst) is not None:
            cont_migs.append(MigrationOrder(req_id=r.i, dst=r.mig_dst))
    return PlanAction(
        tier_decisions=tier_decisions,
        pause_ids=(),
        resume_ids=resume_ids,
        migrations=tuple(cont_migs),
    )


class R3RuntimePlanner:
    """Periodic R3 MILP runtime planner constrained to ForecastView and RebalanceInvariants."""

    requires_forecast: bool = True

    def __init__(
        self,
        builder: ProblemBuilder | None = None,
        solver: R3Solver | None = None,
    ) -> None:
        self.builder = builder or ProblemBuilder()
        self.solver = solver or R3Solver()
        self.last_result: R3Result | None = None
        self.last_diag: dict[str, Any] = {}

    def plan(self, view: ForecastView, invariants: RebalanceInvariants) -> PlanAction:
        problem = self.builder.build(view, invariants)
        result = self.solver.solve(problem)
        self.last_result = result
        req_map = {r.i: r for j in view.compute_nodes for r in view.node_requests(j)}

        fallback_reason = result.fallback_reason
        if result.status in ("timeout_no_incumbent", "proven_infeasible", "error"):
            action = _safe_fallback_action(view, invariants, req_map)
        else:
            val_err = _validate_action(result.action, view, invariants, req_map)
            if val_err is not None:
                fallback_reason = f"actuation_validation_failed: {val_err}"
                action = _safe_fallback_action(view, invariants, req_map)
            else:
                tier_decisions = result.action.tier_decisions if view.physics_params.discrete else {}
                action = PlanAction(
                    tier_decisions=tier_decisions,
                    pause_ids=result.action.pause_ids,
                    resume_ids=result.action.resume_ids,
                    migrations=result.action.migrations,
                )

        self.last_diag = {
            "now_s": view.now,
            "status": result.status,
            "wall_time_s": result.wall_time_s,
            "mip_gap": result.mip_gap,
            "fallback_reason": fallback_reason,
            "stage1_status": result.stage1_status,
            "stage1_wall_time_s": result.stage1_wall_time_s,
            "stage1_mip_gap": result.stage1_mip_gap,
            "stage2_status": result.stage2_status,
            "stage2_wall_time_s": result.stage2_wall_time_s,
            "stage2_mip_gap": result.stage2_mip_gap,
            "n_migrations": len(action.migrations),
        }
        return action
