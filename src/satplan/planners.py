"""Periodic runtime reallocation planners (RuntimePlanner) and CompositePolicy."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from satplan.r3_planner import R3RuntimePlanner
from satplan.world import MigrationOrder, PlanAction

if TYPE_CHECKING:
    from collections.abc import Callable

    from satplan.views import LocalStateView
    from satplan.world import CoWorld, Request


@dataclass(frozen=True)
class RebalanceInvariants:
    """Uniform environment-owned cadence and concurrency limits for runtime planners."""

    interval_s: float = 5.0
    max_inflight: int = 24
    max_mig_per_req: int | None = 4

    @classmethod
    def from_cfg(cls, cfg: dict) -> RebalanceInvariants:
        raw = cfg.get("rebalance_invariants") or {}
        if not isinstance(raw, dict):
            raise ValueError("rebalance_invariants must be a mapping")
        if "max_mig_per_req" not in raw:
            max_mig: int | None = 4
        elif raw["max_mig_per_req"] is None:
            max_mig = None
        else:
            max_mig = int(raw["max_mig_per_req"])
            if max_mig < 0:
                raise ValueError("max_mig_per_req must be nonnegative or None")
        return cls(
            interval_s=float(raw.get("interval_s", 5.0)),
            max_inflight=int(raw.get("max_inflight", 24)),
            max_mig_per_req=max_mig,
        )


class RuntimePlanner(Protocol):
    requires_forecast: bool

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction: ...


class NonePlanner:
    """No-op runtime planner (baseline without migration or tier override)."""

    requires_forecast: bool = False

    def plan(self, view: LocalStateView, invariants: RebalanceInvariants) -> PlanAction:
        return PlanAction()


class ReactivePlanner:
    """Eclipse-reactive planner strictly constrained to LocalStateView and RebalanceInvariants.

    When a resident request's compute node is currently in eclipse, migrates to a currently
    sunlit reachable compute node with KV headroom and strictly lower current step delay.
    """

    requires_forecast: bool = False

    def _best_sunlit_dst(
        self,
        view: LocalStateView,
        src: int,
        src_iter: float,
        req: Request,
        rem_kv: dict[int, float],
        extra_load: dict[int, int],
        extra_kv: dict[int, float],
    ) -> int | None:
        need = req.reserved_tokens * view.bpt
        best_dst: int | None = None
        best_key: tuple[float, int, int] | None = None
        for dst in view.compute_nodes:
            if dst == src or view.is_eclipse(dst) or rem_kv[dst] < need:
                continue
            hops = view.hop_count(src, dst)
            if hops is None:
                continue
            dst_iter = view.iter_of(dst, req, extra_load=extra_load[dst], extra_kv=extra_kv[dst])
            if dst_iter >= src_iter:
                continue
            key = (dst_iter, hops, dst)
            if best_key is None or key < best_key:
                best_key = key
                best_dst = dst
        return best_dst

    def plan(self, view: LocalStateView, invariants: RebalanceInvariants) -> PlanAction:
        inflight = view.inflight_migrations
        if inflight >= invariants.max_inflight:
            return PlanAction()

        rem_kv = {j: view.kv_headroom(j) for j in view.compute_nodes}
        extra_load: dict[int, int] = defaultdict(int)
        extra_kv: dict[int, float] = defaultdict(float)
        orders: list[MigrationOrder] = []

        for src in view.compute_nodes:
            if not view.is_eclipse(src):
                continue
            src_iter = view.iter_of(src)
            for req in view.node_requests(src):
                if inflight >= invariants.max_inflight:
                    return PlanAction(migrations=tuple(orders))
                if (
                    req.state != "decode"
                    or req.mig_in_flight
                    or req.mig_dst is not None
                    or (invariants.max_mig_per_req is not None and req.n_mig >= invariants.max_mig_per_req)
                ):
                    continue
                dst = self._best_sunlit_dst(view, src, src_iter, req, rem_kv, extra_load, extra_kv)
                if dst is None:
                    continue
                orders.append(MigrationOrder(req_id=req.i, dst=dst))
                rem_kv[dst] -= req.reserved_tokens * view.bpt
                extra_load[dst] += 1
                extra_kv[dst] += req.kv
                inflight += 1
        return PlanAction(migrations=tuple(orders))


class CompositePolicy:
    """Binds an outer Placer, an inner RuntimePlanner, and uniform RebalanceInvariants."""

    def __init__(
        self,
        placer: Callable[[Any, Request], int | None],
        planner: RuntimePlanner,
        invariants: RebalanceInvariants | None = None,
    ) -> None:
        self.placer = placer
        self.planner = planner
        self.invariants = invariants or RebalanceInvariants()
        self._last_rebalance_t = -1e18

    def place(self, world: CoWorld, req: Request) -> int | None:
        view = world.forecast_view if getattr(self.placer, "requires_forecast", False) else world.local_view
        return self.placer(view, req)

    def rebalance(self, world: CoWorld) -> None:
        now = world.env.now
        if now - self._last_rebalance_t < self.invariants.interval_s - 1e-9:
            return
        self._last_rebalance_t = now
        view = world.forecast_view if getattr(self.planner, "requires_forecast", False) else world.local_view
        plan = self.planner.plan(view, self.invariants)
        filtered: list[MigrationOrder] = []
        avail_slots = max(0, self.invariants.max_inflight - world.inflight_migrations())
        for order in plan.migrations:
            if len(filtered) >= avail_slots:
                break
            req = world.req_by_id(order.req_id)
            if req is None or (
                req.mig_dst is None
                and self.invariants.max_mig_per_req is not None
                and req.n_mig >= self.invariants.max_mig_per_req
            ):
                continue
            filtered.append(order)
        world.apply_plan(
            PlanAction(
                tier_decisions=plan.tier_decisions,
                pause_ids=plan.pause_ids,
                resume_ids=plan.resume_ids,
                migrations=tuple(filtered),
            )
        )


PLANNERS: dict[str, type[RuntimePlanner]] = {
    "none": NonePlanner,
    "reactive": ReactivePlanner,
    "r3_milp": R3RuntimePlanner,
}


def get_planner(name: str, cfg: dict | None = None) -> RuntimePlanner:
    if name not in PLANNERS:
        raise ValueError(f"unknown planner {name!r}; available: {sorted(PLANNERS)}")
    if name == "r3_milp" and cfg is not None:
        from satplan.demand_forecast import DemandForecastConfig
        from satplan.r3_planner import ProblemBuilder

        r3_cfg = cfg.get("r3") or {}
        df_cfg = DemandForecastConfig.from_cfg(cfg)
        builder = ProblemBuilder(
            horizon_slots=int(r3_cfg.get("horizon_slots", 6)),
            tail_enabled=bool(r3_cfg.get("tail_enabled", True)),
            tail_slots=int(r3_cfg.get("tail_slots", 4)),
            tail_dt_s=float(r3_cfg.get("tail_dt_s", 30.0)),
            time_limit_s=float(r3_cfg.get("time_limit_s", 0.5)),
            demand_forecast=df_cfg,
        )
        return R3RuntimePlanner(builder=builder)
    return PLANNERS[name]()
