"""ProblemBuilder and R3RuntimePlanner wiring R3Solver behind ForecastView and CompositePolicy."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from satplan.actions import PlanAction
from satplan.r3_solver import (
    R3EdgeSpec,
    R3NodeSpec,
    R3Problem,
    R3RequestSpec,
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
        time_limit_s: float = 0.25,
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

        decode_reqs: list[Request] = []
        decode_by_node: dict[int, list[Request]] = {}
        for j in view.compute_nodes:
            reqs_j = [r for r in view.node_requests(j) if r.state == "decode" and r.remaining > 1e-9]
            decode_by_node[j] = reqs_j
            decode_reqs.extend(reqs_j)
        decode_reqs.sort(key=lambda r: r.i)

        if decode_reqs:
            active_nodes = sum(1 for j in view.compute_nodes if view.active_load(j) > 0)
            denom = max(1, active_nodes)
            nd_rep = max(1, round(len(decode_reqs) / denom))
            ctx_rep = sum(r.kv for r in decode_reqs) / denom
        else:
            nd_rep = 1
            ctx_rep = 0.0

        tiers = _tier_specs(params, nd_rep, ctx_rep)

        nodes: list[R3NodeSpec] = []
        for j in view.compute_nodes:
            res_decode_bytes = sum(r.reserved_tokens * view.bpt for r in decode_by_node[j])
            kv_cap_bytes = view.kv_headroom(j) + res_decode_bytes
            is_eclipse = tuple(
                view.is_eclipse_at(j, view.now + (t + 0.5) * dt_s) for t in range(horizon_slots)
            )
            dod_rem = None if params.eclipse_dod_budget_j is None else view.dod_remaining_j(j)
            nodes.append(
                R3NodeSpec(
                    node_id=j,
                    init_temp_C=view.temp_C(j),
                    kv_cap_bytes=kv_cap_bytes,
                    is_eclipse=is_eclipse,
                    init_dod_remaining_j=dod_rem,
                )
            )

        compute_set = set(view.compute_nodes)
        cur_rates = view.edge_rates()
        edges: list[R3EdgeSpec] = []
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
            edges.append(R3EdgeSpec(src=u, dst=v, tot_bps=tot_bps, bg_bps=bg_bps))
            edges.append(R3EdgeSpec(src=v, dst=u, tot_bps=tot_bps, bg_bps=bg_bps))

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
            req_specs.append(
                R3RequestSpec(
                    req_id=r.i,
                    init_node=r.node,
                    kv_bytes=max(r.kv * view.bpt, 1.0),
                    remaining_tokens=eff_tok,
                    deadline_s=eff_dl,
                    weight=1.0,
                    elapsed_s=max(0.0, view.now - r.t),
                    reserved_kv_bytes=float(r.reserved_tokens * view.bpt),
                    currently_paused=r.paused,
                    init_mig_dst=r.mig_dst,
                    init_mig_received_bytes=float(r.mig_received_bytes),
                )
            )
        requests = tuple(req_specs)

        return R3Problem(
            horizon_slots=horizon_slots,
            dt_s=dt_s,
            nodes=tuple(nodes),
            tiers=tiers,
            edges=tuple(edges),
            requests=requests,
            t_amb_C=params.t_amb_C,
            t_max_C=params.t_max_C,
            r_th=params.r_th,
            c_th=params.c_th,
            eclipse_dod_budget_j=params.eclipse_dod_budget_j,
            cvar_alpha=self.cvar_alpha,
            cvar_weight=self.cvar_weight,
            stage2_eps=self.stage2_eps,
            time_limit_s=self.time_limit_s,
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

    def plan(self, view: ForecastView, invariants: RebalanceInvariants) -> PlanAction:
        problem = self.builder.build(view, invariants)
        result = self.solver.solve(problem)
        if result.status == "infeasible":
            return PlanAction()

        req_map = {r.i: r for j in view.compute_nodes for r in view.node_requests(j)}
        avail_slots = max(0, invariants.max_inflight - view.inflight_migrations)
        filtered_migrations = []
        for order in result.action.migrations:
            if len(filtered_migrations) >= avail_slots:
                break
            req = req_map.get(order.req_id)
            if req is None or req.mig_in_flight or req.n_mig >= invariants.max_mig_per_req:
                continue
            filtered_migrations.append(order)

        tier_decisions = result.action.tier_decisions if view.physics_params.discrete else {}
        return PlanAction(
            tier_decisions=tier_decisions,
            pause_ids=result.action.pause_ids,
            resume_ids=result.action.resume_ids,
            migrations=tuple(filtered_migrations),
        )
