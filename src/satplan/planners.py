"""Periodic runtime reallocation planners (RuntimePlanner) and CompositePolicy."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from satplan.demand_forecast import CausalDemandForecaster, DemandForecastConfig
from satplan.r3_planner import R3RuntimePlanner, _slot_bounds
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


class ForecastAwarePlanner:
    """Simple causal forecast-aware heuristic using the exact same ForecastView & PlanAction as R3.

    Combines ReactivePlanner's eclipse-reactive migration with proactive KV-headroom
    reservation for near-term predicted arrivals (no MILP optimization).
    """

    requires_forecast: bool = True

    def __init__(
        self,
        df_cfg: DemandForecastConfig | None = None,
        *,
        horizon_slots: int = 4,
        dt_s: float = 5.0,
    ) -> None:
        self.df_cfg = df_cfg or DemandForecastConfig(enabled=True)
        self.horizon_slots = horizon_slots
        self.dt_s = dt_s
        self.forecaster = CausalDemandForecaster()
        self._reactive = ReactivePlanner()

    @staticmethod
    def _best_proactive_dst(
        view: Any,
        hot_j: int,
        req: Request,
        rem_kv: dict[int, float],
        reserved_kv: dict[int, float],
        extra_load: dict[int, int],
        extra_kv: dict[int, float],
    ) -> int | None:
        req_need = req.reserved_tokens * view.bpt
        best_dst: int | None = None
        best_key: tuple[int, float, int, int] | None = None
        for dst in view.compute_nodes:
            if dst == hot_j or rem_kv[dst] - reserved_kv[dst] < req_need:
                continue
            hops = view.hop_count(hot_j, dst)
            if hops is None:
                continue
            dst_iter = view.iter_of(dst, req, extra_load=extra_load[dst], extra_kv=extra_kv[dst])
            key = (1 if view.is_eclipse(dst) else 0, dst_iter, hops, dst)
            if best_key is None or key < best_key:
                best_key = key
                best_dst = dst
        return best_dst

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        base_action = self._reactive.plan(view, invariants)
        orders = list(base_action.migrations)
        inflight = view.inflight_migrations + len(orders)
        if not self.df_cfg.enabled or inflight >= invariants.max_inflight:
            return PlanAction(migrations=tuple(orders))

        migrated_ids = {o.req_id for o in orders}
        rem_kv = {j: view.kv_headroom(j) for j in view.compute_nodes}
        extra_load: dict[int, int] = defaultdict(int)
        extra_kv: dict[int, float] = defaultdict(float)
        req_by_id = {r.i: r for j in view.compute_nodes for r in view.node_requests(j)}
        for o in orders:
            r = req_by_id.get(o.req_id)
            if r is not None:
                rem_kv[o.dst] -= r.reserved_tokens * view.bpt
                extra_load[o.dst] += 1
                extra_kv[o.dst] += r.kv

        bounds = _slot_bounds(view.now, (self.dt_s,) * self.horizon_slots)
        entries = self.forecaster.forecast(view, bounds, self.df_cfg)
        near_entries = sorted(
            (e for e in entries if e.arrival_slot <= 2),
            key=lambda e: (-e.weight * e.expected_requests, e.arrival_slot, e.origin),
        )

        reserved_kv: dict[int, float] = defaultdict(float)
        for entry in near_entries:
            if inflight >= invariants.max_inflight:
                break
            cands = [j for j in view.compute_nodes if view.path_at(entry.origin, j, entry.arrival_time_s) is not None]
            if not cands:
                continue
            hot_j = min(
                cands,
                key=lambda j: (len(view.path_at(entry.origin, j, entry.arrival_time_s) or ()), j),
            )
            reserved_kv[hot_j] += entry.expected_requests * entry.kv_bytes_per_req
            if rem_kv[hot_j] >= reserved_kv[hot_j]:
                continue
            for req in view.node_requests(hot_j):
                if inflight >= invariants.max_inflight or rem_kv[hot_j] >= reserved_kv[hot_j]:
                    break
                if (
                    req.i in migrated_ids
                    or req.state != "decode"
                    or req.mig_in_flight
                    or req.mig_dst is not None
                    or (invariants.max_mig_per_req is not None and req.n_mig >= invariants.max_mig_per_req)
                ):
                    continue
                best_dst = self._best_proactive_dst(view, hot_j, req, rem_kv, reserved_kv, extra_load, extra_kv)
                if best_dst is None:
                    continue
                req_need = req.reserved_tokens * view.bpt
                orders.append(MigrationOrder(req_id=req.i, dst=best_dst))
                migrated_ids.add(req.i)
                rem_kv[best_dst] -= req_need
                rem_kv[hot_j] += req_need
                extra_load[best_dst] += 1
                extra_kv[best_dst] += req.kv
                inflight += 1
        return PlanAction(migrations=tuple(orders))


PLANNERS: dict[str, type[RuntimePlanner]] = {
    "none": NonePlanner,
    "reactive": ReactivePlanner,
    "forecast_aware": ForecastAwarePlanner,
    "r3_milp": R3RuntimePlanner,
}


def get_planner(name: str, cfg: dict | None = None) -> RuntimePlanner:
    if name not in PLANNERS:
        raise ValueError(f"unknown planner {name!r}; available: {sorted(PLANNERS)}")
    if name == "forecast_aware" and cfg is not None:
        df_cfg = DemandForecastConfig.from_cfg(cfg)
        inv = RebalanceInvariants.from_cfg(cfg)
        return ForecastAwarePlanner(df_cfg=df_cfg, dt_s=inv.interval_s)
    if name == "r3_milp" and cfg is not None:
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
