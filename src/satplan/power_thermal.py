"""Discrete power-tier, linear RC thermal, and Eclipse DoD budget model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satplan.timing import clock_factor, iter_time

if TYPE_CHECKING:
    from satplan.trace import Trace

INF = math.inf


@dataclass(frozen=True)
class PowerTier:
    index: int
    name: str
    power_w: float
    clock_f: float


@dataclass(frozen=True)
class OperatingPoint:
    tier_idx: int
    power_w: float
    clock_f: float


@dataclass(frozen=True)
class PhysicsParams:
    discrete: bool
    tiers: tuple[PowerTier, ...]
    t0_C: float
    t_amb_C: float
    t_max_C: float
    r_th: float
    c_th: float
    p_idle: float
    p_max: float
    gamma: float
    f_eclipse: float
    eclipse_dod_budget_j: float | None

    @property
    def tau_s(self) -> float:
        return self.r_th * self.c_th


def _parse_tiers(power: dict) -> tuple[bool, tuple[PowerTier, ...]]:
    raw = power.get("tiers")
    if not raw:
        return False, ()
    tiers: list[PowerTier] = []
    has_zero = False
    for item in raw:
        pw = float(item["power_w"])
        cf = float(item["clock_f"])
        name = str(item.get("name", f"tier_{len(tiers)}"))
        if pw <= 0.0:
            has_zero = True
            tiers.append(PowerTier(0, name, 0.0, 0.0))
        else:
            tiers.append(PowerTier(len(tiers) + (0 if has_zero else 1), name, pw, cf))
    if not has_zero:
        tiers.insert(0, PowerTier(0, "idle_0w", 0.0, 0.0))
    tiers.sort(key=lambda t: (t.power_w, t.clock_f))
    indexed = tuple(PowerTier(i, t.name, t.power_w, t.clock_f) for i, t in enumerate(tiers))
    return True, indexed


def _parse_params(cfg: dict) -> PhysicsParams:
    power = cfg["power"]
    th = cfg.get("thermal", {})
    discrete, tiers = _parse_tiers(power)
    dod_raw = power.get("eclipse_dod_budget_j")
    dod = None if dod_raw is None else float(dod_raw)
    if dod is not None and dod < 0.0:
        raise ValueError("eclipse_dod_budget_j must be nonnegative")
    return PhysicsParams(
        discrete=discrete,
        tiers=tiers,
        t0_C=float(th.get("T0_C", 25.0)),
        t_amb_C=float(th.get("T_amb_C", 20.0)),
        t_max_C=float(th.get("T_max_C", 85.0)),
        r_th=float(th.get("R_th", 0.2)),
        c_th=float(th.get("C_th", 150.0)),
        p_idle=float(power["P_idle"]),
        p_max=float(power["P_max"]),
        gamma=float(power["gamma"]),
        f_eclipse=clock_factor(float(power["P_dark"]), power),
        eclipse_dod_budget_j=dod,
    )


class PowerThermalModel:
    """Tracks per-node temperature, Eclipse DoD energy budget, and discrete tier actuation."""

    def __init__(self, cfg: dict, trace: Trace):
        self.trace = trace
        self.gpu = cfg["gpu"]
        self.params = _parse_params(cfg)
        self.period = float(cfg.get("control_period_s", 2.0))
        p = self.params
        self.temp_C: list[float] = [p.t0_C] * trace.n
        self.peak_temp_C: float = p.t0_C
        dod0 = INF if p.eclipse_dod_budget_j is None else p.eclipse_dod_budget_j
        self.dod_remaining_j: dict[int, float] = {j: dod0 for j in trace.compute}
        self._in_eclipse: dict[int, bool] = {j: trace.is_eclipse(j, 0.0) for j in trace.compute}
        default_tier = len(p.tiers) - 1 if p.discrete else 0
        self.tier_idx: dict[int, int] = {j: 0 for j in trace.compute}
        self.explicit_tier_idx: dict[int, int | None] = {j: None for j in trace.compute}
        self._default_work_tier = default_tier
        self.zero_power_s: float = 0.0
        self.energy_j: float = 0.0
        self.dod_exhausted_s: float = 0.0

    def _decay(self, dt: float) -> float:
        tau = self.params.tau_s
        return math.exp(-dt / tau) if tau > 0.0 else 0.0

    def _next_temp(self, temp: float, power_w: float, dt: float) -> float:
        t_eq = self.params.t_amb_C + self.params.r_th * power_w
        return t_eq + (temp - t_eq) * self._decay(dt)

    def _select_discrete_op(
        self,
        temp: float,
        dod_rem: float,
        dt: float,
        active_batch: int,
        *,
        is_eclipse: bool,
        tier_override: int | None,
    ) -> OperatingPoint:
        tiers = self.params.tiers
        if active_batch <= 0 or tier_override == 0:
            return OperatingPoint(0, 0.0, 0.0)
        if is_eclipse and self.params.eclipse_dod_budget_j is not None and dod_rem <= 1e-9:
            return OperatingPoint(0, 0.0, 0.0)
        top = self._default_work_tier if tier_override is None else min(max(tier_override, 0), len(tiers) - 1)
        for idx in range(top, 0, -1):
            tier = tiers[idx]
            if self._next_temp(temp, tier.power_w, dt) > self.params.t_max_C + 1e-9:
                continue
            if is_eclipse and self.params.eclipse_dod_budget_j is not None and tier.power_w * dt > dod_rem + 1e-9:
                continue
            return OperatingPoint(idx, tier.power_w, tier.clock_f)
        return OperatingPoint(0, 0.0, 0.0)

    def _select_continuous_op(self, active_batch: int, *, is_eclipse: bool) -> OperatingPoint:
        p = self.params
        f = p.f_eclipse if is_eclipse else 1.0
        if active_batch <= 0:
            return OperatingPoint(0, p.p_idle, f)
        pwr = p.p_idle + (p.p_max - p.p_idle) * (f**p.gamma)
        return OperatingPoint(1, pwr, f)

    def current_op(self, node: int, active_batch: int, *, is_eclipse: bool, dt: float | None = None) -> OperatingPoint:
        step = self.period if dt is None else dt
        if not self.params.discrete:
            return self._select_continuous_op(active_batch, is_eclipse=is_eclipse)
        return self._select_discrete_op(
            self.temp_C[node],
            self.dod_remaining_j[node],
            step,
            active_batch,
            is_eclipse=is_eclipse,
            tier_override=self.explicit_tier_idx[node],
        )

    def set_explicit_tier(self, node: int, tier_idx: int | None) -> None:
        if tier_idx is not None and self.params.discrete and not (0 <= tier_idx < len(self.params.tiers)):
            raise ValueError(f"invalid tier index {tier_idx} for node {node}")
        self.explicit_tier_idx[node] = tier_idx

    def is_dod_exhausted(self, node: int, *, is_eclipse: bool) -> bool:
        return is_eclipse and self.params.eclipse_dod_budget_j is not None and self.dod_remaining_j[node] <= 1e-9

    def on_eclipse_change(self, node: int, is_eclipse: bool) -> None:
        if self.params.eclipse_dod_budget_j is not None and (not is_eclipse or not self._in_eclipse[node]):
            self.dod_remaining_j[node] = self.params.eclipse_dod_budget_j
        self._in_eclipse[node] = is_eclipse

    def step_node(self, node: int, dt: float, active_batch: int, *, is_eclipse: bool) -> OperatingPoint:
        if self.params.eclipse_dod_budget_j is not None and (not is_eclipse or not self._in_eclipse[node]):
            self.dod_remaining_j[node] = self.params.eclipse_dod_budget_j
        self._in_eclipse[node] = is_eclipse

        op = self.current_op(node, active_batch, is_eclipse=is_eclipse, dt=dt)
        self.tier_idx[node] = op.tier_idx
        self.temp_C[node] = self._next_temp(self.temp_C[node], op.power_w, dt)
        if self.temp_C[node] > self.peak_temp_C:
            self.peak_temp_C = self.temp_C[node]

        if dt > 0.0:
            self.energy_j += op.power_w * dt
            if op.power_w <= 0.0:
                self.zero_power_s += dt
            if is_eclipse and self.params.eclipse_dod_budget_j is not None:
                self.dod_remaining_j[node] = max(0.0, self.dod_remaining_j[node] - op.power_w * dt)
                if active_batch > 0 and op.clock_f <= 0.0:
                    self.dod_exhausted_s += dt
        return op

    def project_finish(
        self,
        node: int,
        start: float,
        batch: int,
        context: float,
        tokens: float,
        *,
        tier_override: int | None = None,
    ) -> tuple[float, float]:
        """Non-mutating forward projection of decode completion time and temperature."""
        rem = float(tokens)
        temp = self.temp_C[node]
        dod = self.dod_remaining_j[node]
        if rem <= 0.0:
            return float(start), temp
        t = float(start)
        horizon = self.trace.horizon
        eval_batch = max(int(batch), 1)
        prev_ecl = self.trace.is_eclipse(node, t)
        step_base = max(self.period, 0.5)
        while t < horizon - 1e-9 and rem > 1e-9:
            dt = min(step_base, horizon - t)
            ecl = self.trace.is_eclipse(node, t)
            if self.params.eclipse_dod_budget_j is not None and (not ecl or (ecl and not prev_ecl)):
                dod = self.params.eclipse_dod_budget_j
            prev_ecl = ecl
            if self.params.discrete:
                op = self._select_discrete_op(temp, dod, dt, eval_batch, is_eclipse=ecl, tier_override=tier_override)
            else:
                op = self._select_continuous_op(eval_batch, is_eclipse=ecl)
            temp = self._next_temp(temp, op.power_w, dt)
            if ecl and self.params.eclipse_dod_budget_j is not None:
                dod = max(0.0, dod - op.power_w * dt)
            if op.clock_f > 0.0:
                it = iter_time(eval_batch, context, op.clock_f, self.gpu)
                cap = dt / it
                if cap >= rem - 1e-9:
                    return t + rem * it, temp
                rem -= cap
            t += dt
        return INF, temp
