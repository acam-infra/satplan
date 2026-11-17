"""Causal demand forecasting: history-mean estimator and published demand plans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from satplan.views import ForecastView


@dataclass(frozen=True)
class PublishedDemandPlan:
    publish_time_s: float
    arrival_time_s: float
    origin: int
    expected_requests: float
    tokens_per_req: float
    kv_bytes_per_req: float
    deadline_s: float
    weight: float = 1.0
    demand_class: str = "short"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PublishedDemandPlan:
        return cls(
            publish_time_s=float(raw["publish_time_s"]),
            arrival_time_s=float(raw["arrival_time_s"]),
            origin=int(raw["origin"]),
            expected_requests=float(raw["expected_requests"]),
            tokens_per_req=float(raw["tokens_per_req"]),
            kv_bytes_per_req=float(raw["kv_bytes_per_req"]),
            deadline_s=float(raw["deadline_s"]),
            weight=float(raw.get("weight", 1.0)),
            demand_class=str(raw.get("demand_class", "short")),
        )


@dataclass(frozen=True)
class ObservedArrivalRecord:
    arrival_time_s: float
    origin: int
    tokens: float
    kv_bytes: float
    deadline_s: float
    demand_class: str = "short"


@dataclass(frozen=True)
class DemandForecastConfig:
    enabled: bool = False
    obs_window_s: float = 60.0
    prior_rate_per_s: float = 0.0
    prior_tokens_per_req: float = 200.0
    prior_kv_bytes_per_req: float = 2_000_000.0
    prior_deadline_s: float = 40.0
    max_predicted_requests_per_slot: float = 4.0
    calibration_interval_s: float | None = None

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> DemandForecastConfig:
        if not cfg or "demand_forecast" not in cfg:
            return cls()
        raw = cfg["demand_forecast"] or {}
        cal = raw.get("calibration_interval_s")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            obs_window_s=float(raw.get("obs_window_s", 60.0)),
            prior_rate_per_s=float(raw.get("prior_rate_per_s", 0.0)),
            prior_tokens_per_req=float(raw.get("prior_tokens_per_req", 200.0)),
            prior_kv_bytes_per_req=float(raw.get("prior_kv_bytes_per_req", 2_000_000.0)),
            prior_deadline_s=float(raw.get("prior_deadline_s", 40.0)),
            max_predicted_requests_per_slot=float(raw.get("max_predicted_requests_per_slot", 4.0)),
            calibration_interval_s=None if cal is None else float(cal),
        )


@dataclass(frozen=True)
class DemandForecastEntry:
    origin: int
    arrival_slot: int
    arrival_time_s: float
    expected_requests: float
    tokens_per_req: float
    kv_bytes_per_req: float
    deadline_s: float
    weight: float
    demand_class: str
    source: str  # "history_mean" | "published_plan"


class CausalDemandForecaster:
    """Causal demand forecaster using only arrivals <= now and plans with publish_time_s <= now."""

    def forecast(
        self,
        view: ForecastView,
        slot_bounds: tuple[tuple[float, float], ...],
        cfg: DemandForecastConfig,
    ) -> tuple[DemandForecastEntry, ...]:
        if not cfg.enabled or len(slot_bounds) < 2:
            return ()
        now = view.now
        entries: list[DemandForecastEntry] = []
        entries.extend(self._forecast_published_plans(view, slot_bounds, now, cfg))
        entries.extend(self._forecast_history_mean(view, slot_bounds, now, cfg))
        return tuple(entries)

    @staticmethod
    def _forecast_published_plans(
        view: ForecastView,
        slot_bounds: tuple[tuple[float, float], ...],
        now: float,
        cfg: DemandForecastConfig,
    ) -> list[DemandForecastEntry]:
        plans = view.published_plans(now)
        out: list[DemandForecastEntry] = []
        horizon_end = slot_bounds[-1][1]
        for plan in plans:
            if plan.arrival_time_s <= now or plan.arrival_time_s >= horizon_end:
                continue
            slot_idx = next(
                (t for t, (sa, sb) in enumerate(slot_bounds) if t >= 1 and sa - 1e-9 <= plan.arrival_time_s < sb),
                1 if plan.arrival_time_s < slot_bounds[1][0] else None,
            )
            if slot_idx is None:
                continue
            exp_req = min(cfg.max_predicted_requests_per_slot, max(0.0, plan.expected_requests))
            if exp_req <= 1e-9:
                continue
            out.append(
                DemandForecastEntry(
                    origin=plan.origin,
                    arrival_slot=slot_idx,
                    arrival_time_s=plan.arrival_time_s,
                    expected_requests=exp_req,
                    tokens_per_req=max(1.0, plan.tokens_per_req),
                    kv_bytes_per_req=max(1.0, plan.kv_bytes_per_req),
                    deadline_s=max(0.0, (plan.arrival_time_s - now) + plan.deadline_s),
                    weight=max(0.0, plan.weight),
                    demand_class=plan.demand_class,
                    source="published_plan",
                )
            )
        return out

    @staticmethod
    def _forecast_history_mean(
        view: ForecastView,
        slot_bounds: tuple[tuple[float, float], ...],
        now: float,
        cfg: DemandForecastConfig,
    ) -> list[DemandForecastEntry]:
        obs = view.observed_arrivals(now)
        win_start = max(0.0, now - cfg.obs_window_s)
        eff_win = min(now, cfg.obs_window_s) if now > 1e-9 else cfg.obs_window_s
        if cfg.calibration_interval_s is not None and cfg.calibration_interval_s > 0.0:
            eff_win = max(eff_win, cfg.calibration_interval_s)

        by_origin: dict[int, list[ObservedArrivalRecord]] = {}
        for rec in obs:
            if rec.arrival_time_s >= win_start - 1e-9:
                by_origin.setdefault(rec.origin, []).append(rec)

        origins = sorted(set(by_origin) | (set(view.rs_nodes) if cfg.prior_rate_per_s > 0.0 else set()))
        out: list[DemandForecastEntry] = []
        for origin in origins:
            recs = by_origin.get(origin, [])
            if recs and eff_win > 1e-9:
                rate = len(recs) / eff_win
                tok = sum(r.tokens for r in recs) / len(recs)
                kv = sum(r.kv_bytes for r in recs) / len(recs)
                dl = sum(r.deadline_s for r in recs) / len(recs)
                d_cls = "long" if sum(1 for r in recs if r.demand_class == "long") >= len(recs) / 2 else "short"
            elif cfg.prior_rate_per_s > 0.0:
                rate = cfg.prior_rate_per_s
                tok = cfg.prior_tokens_per_req
                kv = cfg.prior_kv_bytes_per_req
                dl = cfg.prior_deadline_s
                d_cls = "short"
            else:
                continue

            for t in range(1, len(slot_bounds)):
                sa, sb = slot_bounds[t]
                exp_req = min(cfg.max_predicted_requests_per_slot, rate * (sb - sa))
                if exp_req <= 1e-6:
                    continue
                out.append(
                    DemandForecastEntry(
                        origin=origin,
                        arrival_slot=t,
                        arrival_time_s=sa,
                        expected_requests=exp_req,
                        tokens_per_req=max(1.0, tok),
                        kv_bytes_per_req=max(1.0, kv),
                        deadline_s=max(0.0, (sa - now) + dl),
                        weight=1.0,
                        demand_class=d_cls,
                        source="history_mean",
                    )
                )
        return out
