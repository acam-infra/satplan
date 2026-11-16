"""Seam tests for PowerThermalModel: 0W cooling, T_max cap, Eclipse DoD budget, and non-mutating project_finish."""
from __future__ import annotations

import math

from satplan.power_thermal import PowerThermalModel
from satplan.trace import SCHEMA, Trace


def _two_node_trace(*, eclipse: list[list[float]] | None = None) -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "pt-synth",
        "horizon_s": 200.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
        ],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 200.0]]}],
        "dark": [[], eclipse if eclipse is not None else []],
    })


def _discrete_cfg(*, t0: float = 25.0, t_max: float = 85.0, dod_j: float | None = None) -> dict:
    power: dict = {
        "P_idle": 0.0,
        "P_max": 300.0,
        "gamma": 2.4,
        "f_min": 0.35,
        "P_dark": 100.0,
        "tiers": [
            {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
            {"name": "low", "power_w": 100.0, "clock_f": 0.5},
            {"name": "high", "power_w": 300.0, "clock_f": 1.0},
        ],
    }
    if dod_j is not None:
        power["eclipse_dod_budget_j"] = dod_j
    return {
        "power": power,
        "thermal": {
            "T0_C": t0,
            "T_amb_C": 20.0,
            "T_max_C": t_max,
            "R_th": 0.25,
            "C_th": 100.0,
        },
        "gpu": {
            "kv_gib": 24.0,
            "t_fixed_s": 0.002,
            "t_weights_s": 0.005,
            "t_kv_tok_s": 1.0e-7,
            "t_seq_s": 5.0e-5,
            "mem_freq_exp": 0.6,
        },
    }


def test_zero_watt_cooling_when_empty_or_all_suspended() -> None:
    tr = _two_node_trace()
    cfg = _discrete_cfg(t0=60.0)
    pt = PowerThermalModel(cfg, tr)

    # Empty node -> 0W idle cooling toward T_amb_C = 20.0
    # tau = R_th * C_th = 25.0 s, dt = 25.0 s -> T = 20 + (60 - 20) * exp(-1)
    op = pt.step_node(1, dt=25.0, active_batch=0, is_eclipse=False)
    assert op.power_w == 0.0
    assert op.clock_f == 0.0
    expected_t = 20.0 + 40.0 * math.exp(-1.0)
    assert math.isclose(pt.temp_C[1], expected_t, rel_tol=1e-6)


def test_tmax_cap_steps_down_tier() -> None:
    tr = _two_node_trace()
    # At T0=84.0 C and T_max=85.0 C:
    # high tier (300W) -> T_eq = 20 + 0.25*300 = 95 C -> would exceed 85 C in 5s.
    # low tier (100W) -> T_eq = 20 + 0.25*100 = 45 C -> cools below 85 C.
    cfg = _discrete_cfg(t0=84.0, t_max=85.0)
    pt = PowerThermalModel(cfg, tr)

    op = pt.step_node(1, dt=5.0, active_batch=2, is_eclipse=False)
    assert op.power_w == 100.0
    assert op.clock_f == 0.5
    assert pt.temp_C[1] <= 85.0


def test_eclipse_dod_exhaustion_and_sunlit_restoration() -> None:
    tr = _two_node_trace(eclipse=[[10.0, 50.0]])
    # DoD budget = 600 J for the eclipse window [10, 50].
    cfg = _discrete_cfg(t0=25.0, dod_j=600.0)
    pt = PowerThermalModel(cfg, tr)

    # Sunlit at t=0..10: unlimited energy, runs high tier (300W) -> 3000 J, DoD untouched.
    op0 = pt.step_node(1, dt=10.0, active_batch=1, is_eclipse=False)
    assert op0.power_w == 300.0
    assert math.isclose(pt.dod_remaining_j[1], 600.0)

    # Eclipse t=10..12 (dt=2s): 300W * 2s = 600 J -> consumes entire budget.
    op1 = pt.step_node(1, dt=2.0, active_batch=1, is_eclipse=True)
    assert op1.power_w == 300.0
    assert math.isclose(pt.dod_remaining_j[1], 0.0, abs_tol=1e-9)

    # Eclipse t=12..14 (dt=2s): DoD exhausted -> compute halts (0W, clock_f=0.0).
    op2 = pt.step_node(1, dt=2.0, active_batch=1, is_eclipse=True)
    assert op2.power_w == 0.0
    assert op2.clock_f == 0.0

    # Sunlit restoration at t=50..52: DoD budget resets and compute resumes at 300W.
    op3 = pt.step_node(1, dt=2.0, active_batch=1, is_eclipse=False)
    assert op3.power_w == 300.0
    assert math.isclose(pt.dod_remaining_j[1], 600.0)


def test_project_finish_is_non_mutating_and_respects_tier_override() -> None:
    tr = _two_node_trace()
    cfg = _discrete_cfg(t0=30.0)
    pt = PowerThermalModel(cfg, tr)

    t_before = pt.temp_C[1]
    dod_before = pt.dod_remaining_j[1]

    fin_high, _ = pt.project_finish(1, start=0.0, batch=1, context=500.0, tokens=1000.0, tier_override=2)
    fin_low, _ = pt.project_finish(1, start=0.0, batch=1, context=500.0, tokens=1000.0, tier_override=1)

    assert fin_high < fin_low
    assert math.isfinite(fin_low)
    assert pt.temp_C[1] == t_before
    assert pt.dod_remaining_j[1] == dod_before
