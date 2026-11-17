"""P0-P2 power invariants; moved without changing test bodies."""
from __future__ import annotations

import pytest
import simpy
from factories import _base_cfg, _exec_returned_schedule, _two_cs_trace
from satplan.placers import get_placer
from satplan.planners import RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder
from satplan.r3_solver import (
    R3Solver,
)
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


@pytest.mark.parametrize(
    ("eclipse_start_s", "pout", "expect_ontime"),
    [
        (6.5, 50, False),   # before slot midpoint (7.501)
        (7.5, 50, False),   # at slot midpoint
        (8.0, 50, False),   # after slot midpoint (R8 repro)
        (10.0, 49, True),   # slot boundary: [5.001, 10.0) has 4.999s sunlit -> 49 tok fits
        (8.0, 25, True),    # after midpoint, smaller request fits in [5.001, 8.0)
        (20.0, 50, True),   # continuous sunlit
    ],
)
def test_r1c_slot_interior_eclipse_entry_and_sunlit_positive(
    eclipse_start_s: float, pout: int, expect_ontime: bool
) -> None:
    """R1c: slot-interior eclipse entry bounds compute to sunlit duration; positive cases still deliver."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1c-eclipse-entry",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], [[0.0, 5.0], [eclipse_start_s, 60.0]]],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, False, 1, pout, 1.0, 10.001)
    world.submit(req)
    start = 5.001
    env.run(until=start)
    problem = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start, slots=1)
    actual_ontime = {0} if (req.t_delivered is not None and req.t_delivered <= req.deadline) else set()
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



@pytest.mark.parametrize(
    ("dark_intervals", "pout", "expect_ontime"),
    [
        ([[0.0, 7.0], [10.0, 60.0]], 25, True),   # exit at 7s -> 3s sunlit -> 25 tok fits
        ([[0.0, 7.0], [10.0, 60.0]], 40, False),  # exit at 7s -> 3s sunlit -> 40 tok infeasible
        ([[0.0, 7.0], [9.0, 60.0]], 15, True),    # short 2s sunlit [7, 9) then re-entry -> 15 tok fits
        ([[0.0, 7.0], [9.0, 60.0]], 25, False),   # short 2s sunlit [7, 9) then re-entry -> 25 tok infeasible
    ],
)
def test_r1c_slot_interior_eclipse_exit_and_short_sunlit_reentry(
    dark_intervals: list[list[float]], pout: int, expect_ontime: bool
) -> None:
    """R1c: slot-interior eclipse exit and short sunlit window before re-entry."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1c-exit-reentry",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], dark_intervals],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, False, 1, pout, 1.0, 10.0)
    world.submit(req)
    env.run(until=5.0)
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 5.0, slots=1)
    actual_ontime = {0} if (req.t_delivered is not None and req.t_delivered <= req.deadline) else set()
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



@pytest.mark.parametrize(
    ("dod_budget_j", "dark_intervals", "pre_consume_until", "pout", "deadline_s", "expect_ontime"),
    [
        (375.0, [[5.0, 60.0]], 5.0, 20, 10.1, True),                  # finite DoD: 75W*5s=375J -> 25 tok max (5 tok/s)
        (375.0, [[5.0, 60.0]], 5.0, 35, 10.1, False),                 # finite DoD: 300W blocked by period*P > DoD
        (None, [[5.0, 60.0]], 5.0, 45, 10.1, True),                   # unlimited DoD: 300W runs -> 50 tok max (10 tok/s)
        (375.0, [[0.0, 7.0], [10.5, 60.0]], 5.0, 40, 15.1, True),     # cross-window: w0 exhausted, [7,10.5) recovers, w1=375J
        (375.0, [[0.0, 7.0], [10.5, 60.0]], 5.0, 75, 15.1, False),    # cross-window: 75 > 35 + 30 = 65 tok
    ],
)
def test_r1c_dod_modes_and_cross_window_consumed_budget(
    dod_budget_j: float | None,
    dark_intervals: list[list[float]],
    pre_consume_until: float,
    pout: int,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R1c: finite/unlimited DoD, control_period tier admission, and cross-window consumed DoD."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1c-dod-modes",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], dark_intervals],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = dod_budget_j
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    # Exhaust w0 DoD (375 J = 75W * 5s) before t=5.0
    if dark_intervals[0][0] == 0.0:
        world.submit(Request(99, 0, 0.0, False, 1, 60, 1.0, 15.0))
    req = Request(0, 0, pre_consume_until, False, 1, pout, 1.0, deadline_s)
    world.submit(req)
    env.run(until=pre_consume_until + 0.05)
    now = env.now
    if dark_intervals[0][0] == 0.0:
        assert world.forecast_view.dod_remaining_j(1) == pytest.approx(0.0, abs=1e-6)
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, now, slots=2)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert 0 in result.goodput_req_ids
        assert 0 in actual_ontime
    else:
        assert 0 not in result.goodput_req_ids



def test_r1c_migrate_from_eclipse_to_sunlit_positive() -> None:
    """R1c positive: eclipse node with DoD=0 migrates request to sunlit node and finishes on time."""
    tr = _two_cs_trace(c1_dark=[[0.0, 60.0]], c2_dark=[])
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"] = {"inter": 0.16, "rs": 10.0}
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    r = Request(0, 0, 0.0, False, 4800, 50, 1.0, 20.0)
    w.submit(r)
    env.run(until=20.0)
    assert r.node == 2 and r.n_mig == 1 and r.t_delivered is not None and r.t_delivered <= 20.0



@pytest.mark.parametrize(
    ("budget_j", "pout", "expect_ontime"),
    [
        (1600.0, 68, False),  # R9 repro: 1500 + 375 = 1875 > 1600 -> 2nd slot 75W rejected
        (1874.0, 68, False),  # just below 1500 + 375 = 1875 threshold
        (1875.0, 68, True),   # exact 1500 + 375 threshold -> [2, 1] admissible
        (2000.0, 68, True),   # R9 positive control -> finishes on time
        (750.0, 60, True),    # 375 + 375 = 750 -> two low-tier slots (65.97 tok >= 60)
        (750.0, 68, False),   # two low-tier slots insufficient for 68 tok
    ],
)
def test_r1d_finite_dod_repeated_admission_thresholds(
    budget_j: float, pout: int, expect_ontime: bool
) -> None:
    """R1d: finite DoD tracks per-slot consumption without fractional u_ecl under-accounting."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1d-finite-dod",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], [[0.0, 60.0]]],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = budget_j
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 5.0, False, 1, pout, 1.0, 15.05)
    world.submit(req)
    start = 5.05
    env.run(until=start)
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start, slots=2)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



@pytest.mark.parametrize(
    ("dark_intervals", "budget_j", "pout", "deadline_s", "expect_ontime"),
    [
        ([[0.0, 5.0], [8.0, 60.0]], 400.0, 42, 15.05, True),   # 3s*10 (30) + 2s*6.597 (13.19) = 43.19 >= 42
        ([[0.0, 5.0], [8.0, 60.0]], 400.0, 50, 15.05, False),  # 150J used in [8,10) -> 250J < 375J in slot 1 -> 50 infeasible
        ([[0.0, 7.0], [10.5, 60.0]], 400.0, 45, 15.05, True),  # w0=400J (1.95s low + 3.5s high), w1=400J resets after sunlit
    ],
)
def test_r1d_mixed_sunlit_eclipse_finite_dod_and_reset(
    dark_intervals: list[list[float]],
    budget_j: float,
    pout: int,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R1d: mixed sunlit/eclipse slots with finite DoD account fallback consumption and real sunlit reset."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1d-mixed-dod",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], dark_intervals],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = budget_j
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 5.0, False, 1, pout, 1.0, deadline_s)
    world.submit(req)
    start = 5.05
    env.run(until=start)
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start, slots=2)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



@pytest.mark.parametrize(
    ("eclipse_start_s", "t_max_C", "pout", "expect_ontime"),
    [
        (8.0, 41.0, 50, False),    # R10 repro: after midpoint, t=8s lookahead 43.77 > 41.0 -> []
        (6.5, 41.0, 50, False),    # before midpoint: t=6.5s lookahead > 41.0 -> []
        (10.5, 41.0, 49, True),    # slot boundary/after: no intra-slot event in [5.001, 10.001), 39.96 <= 41.0 -> [0]
        (8.0, 41.0, 30, True),     # low tier 75W (T_eq=35<=T0) delivers 32.99 >= 30 -> [0]
        (8.0, 85.0, 50, True),     # R10 positive control: wide thermal headroom -> [0]
    ],
)
def test_r1e_intraslot_thermal_admission_thresholds_and_positive(
    eclipse_start_s: float, t_max_C: float, pout: int, expect_ontime: bool
) -> None:
    """R1e: intra-slot power change re-runs current_op thermal admission over full control_period_s."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1e-thermal-admission",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], [[0.0, 5.0], [eclipse_start_s, 60.0]]],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = None
    cfg["thermal"].update(T0_C=35.0, T_amb_C=20.0, T_max_C=t_max_C, R_th=0.2, C_th=150.0)
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 5.0, False, 1, pout, 1.0, 10.001)
    world.submit(req)
    start = 5.001
    env.run(until=start)
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start, slots=1)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



@pytest.mark.parametrize(
    ("c_th", "control_period_s", "dod_budget_j", "dark_intervals", "pout", "deadline_s", "expect_ontime"),
    [
        (150.0, 5.0, 2000.0, [[0.0, 7.0], [12.0, 60.0]], 50, 10.001, False),  # exit at 7s, finite DoD: high blocked
        (150.0, 5.0, 2000.0, [[0.0, 7.0], [10.001, 60.0]], 75, 15.001, True),   # slot 0 low (33), slot 1 high (50) -> 83>=75
        (150.0, 10.0, None, [[0.0, 5.0], [20.0, 60.0]], 50, 10.001, False),   # control_period=10s > dt=5s blocks high tier
        (300.0, 5.0, None, [[0.0, 5.0], [8.0, 60.0]], 50, 10.001, True),      # larger C_th=300 stays <= 41 C at t=8s
        (150.0, 5.0, None, [[0.0, 60.0]], 50, 15.001, True),                  # warm T0=35C cools in slot 0 or low+high by 15s
    ],
)
def test_r1e_thermal_params_dod_and_cooling_positive(
    c_th: float,
    control_period_s: float,
    dod_budget_j: float | None,
    dark_intervals: list[list[float]],
    pout: int,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R1e: combined thermal+DoD admission across C_th, control_period_s, intra-slot exit, and cooling."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1e-params",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], dark_intervals],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = control_period_s
    cfg["power"]["eclipse_dod_budget_j"] = dod_budget_j
    cfg["thermal"].update(T0_C=35.0, T_amb_C=20.0, T_max_C=41.0, R_th=0.2, C_th=c_th)
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 5.0, False, 1, pout, 1.0, deadline_s)
    world.submit(req)
    start = 5.001
    env.run(until=start)
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start, slots=2)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



def test_r1e_migrate_from_thermally_constrained_node_positive() -> None:
    """R1e positive: node 1 thermally constrained under T_max=41C migrates to cool node 2 and delivers."""
    tr = _two_cs_trace(c1_dark=[], c2_dark=[])
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 5.0
    cfg["thermal"].update(T0_C=40.5, T_amb_C=20.0, T_max_C=41.0, R_th=0.2, C_th=150.0)
    cfg["net"]["link_gbps"] = {"inter": 0.16, "rs": 10.0}
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("r3_milp"), seed=1)
    w.pt.temp_C[2] = 20.0
    r = Request(0, 0, 0.0, False, 4800, 132, 1.0, 25.0)
    w.submit(r)
    env.run(until=30.0)
    assert r.node == 2 and r.n_mig == 1 and r.t_delivered is not None and r.t_delivered <= 25.0



@pytest.mark.parametrize(
    ("control_period_s", "budget_j", "start_s", "pout", "expect_ontime"),
    [
        (2.0, 1500.0, 2.05, 50, False),  # R11 repro: ctrl_dt=2 < dt=5, 300*(2+5)=2100 > 1500 -> []
        (2.0, 2099.0, 2.05, 50, False),  # just below 2100 J threshold
        (2.0, 2100.0, 2.05, 50, True),   # R11 positive control: exact 2100 J -> [0]
        (2.0, 1500.0, 2.55, 50, False),  # different tick phase -> still []
        (2.0, 1500.0, 2.05, 30, True),   # low tier 75W (525 <= 1500) delivers 32.99 >= 30 -> [0]
        (1.7, 1500.0, 2.05, 50, False),  # non-integer ratio 5/1.7: 300*(1.7+5)=2010 > 1500 -> []
        (1.7, 2100.0, 2.05, 50, True),   # non-integer ratio 5/1.7: >= 4*510=2040 -> [0]
        (5.0, 1500.0, 5.001, 49, True),  # ctrl_dt == dt: 300*5=1500 <= 1500 -> finishes before t=10 tick -> [0]
        (7.0, 2100.0, 7.001, 50, True),  # ctrl_dt > dt (aligned): 300*7=2100 <= 2100 -> [0]
        (7.0, 1500.0, 2.05, 50, False),  # ctrl_dt > dt: 300*7=2100 > 1500 -> []
    ],
)
def test_r1f_multi_tick_per_slot_dod_admission_and_phases(
    control_period_s: float, budget_j: float, start_s: float, pout: int, expect_ontime: bool
) -> None:
    """R1f: multiple control ticks inside a planning slot require DoD >= P*(ctrl_dt + dur)."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1f-multi-tick",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], [[0.0, 60.0]]],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = control_period_s
    cfg["power"]["eclipse_dod_budget_j"] = budget_j
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    arr_t = 2.0 if start_s < 5.0 else start_s - 0.001
    req = Request(0, 0, arr_t, False, 1, pout, 1.0, arr_t + 5.05)
    world.submit(req)
    env.run(until=start_s)
    problem = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start_s, slots=1)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



@pytest.mark.parametrize(
    ("control_period_s", "budget_j", "dark_intervals", "pout", "deadline_s", "expect_ontime"),
    [
        (2.0, None, [[0.0, 60.0]], 50, 7.05, True),                        # unlimited DoD under ctrl_dt < dt
        (2.0, 1500.0, [[0.0, 4.5], [10.0, 60.0]], 40, 7.05, True),         # exit at 4.5s: dur=2.45 -> 300*(2+2.45)=1335<=1500
        (2.0, 0.0, [[0.0, 4.5], [10.0, 60.0]], 35, 12.05, True),           # DoD=0 waits for exit at 4.5s and finishes in slot 1
    ],
)
def test_r1f_multi_tick_intraslot_exit_and_unlimited_dod(
    control_period_s: float,
    budget_j: float | None,
    dark_intervals: list[list[float]],
    pout: int,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R1f positive: unlimited DoD, intra-slot eclipse exit, and waiting for sunlit under ctrl_dt < dt."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r1f-exit-unlimited",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], dark_intervals],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = control_period_s
    cfg["power"]["eclipse_dod_budget_j"] = budget_j
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 2.0, False, 1, pout, 1.0, deadline_s)
    world.submit(req)
    start_s = 2.05
    env.run(until=start_s)
    problem = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, start_s, slots=2)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime

