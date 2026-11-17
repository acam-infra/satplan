"""P0-P2 cadence invariants; moved without changing test bodies."""
from __future__ import annotations

import pytest
import simpy
from factories import _base_cfg, _two_cs_trace
from satplan.placers import get_placer
from satplan.planners import RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


@pytest.mark.parametrize(
    ("control_period_s", "interval_s", "expected_dt_s", "arrival_t", "deadline_s", "expect_ontime"),
    [
        (2.0, 5.0, 6.0, 2.001, 18.0, False),  # R12 repro: 6s cadence (2,8,14,20) -> no false promise
        (2.0, 5.0, 6.0, 2.001, 19.0, False),  # conservative [] <= {0}
        (2.0, 5.0, 6.0, 2.001, 26.5, True),   # R12 positive (wide deadline): cools then finishes -> [0]
        (5.0, 5.0, 5.0, 2.001, 23.0, True),   # period == interval: 5s cadence -> [0]
        (7.0, 5.0, 7.0, 2.001, 22.0, True),   # period > interval: 7s cadence -> [0]
        (1.7, 5.0, 5.1, 2.001, 28.0, True),   # non-integer ratio 5/1.7: 5.1s cadence -> [0]
        (2.0, 5.0, 6.0, 3.5, 26.5, True),     # different arrival phase -> [0]
    ],
)
def test_r1g_periodic_closed_loop_actuation_cadence(
    control_period_s: float,
    interval_s: float,
    expected_dt_s: float,
    arrival_t: float,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R1g: real CompositePolicy -> R3RuntimePlanner -> CoWorld closed loop aligns dt_s to actuation cadence."""
    from satplan.planners import CompositePolicy
    from satplan.r3_planner import R3RuntimePlanner

    class _RecordingR3Planner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))
            self.records: list[tuple[float, tuple[int, ...], float]] = []

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            self.records.append((view.now, res.goodput_req_ids, self.inner.builder.build(view, invariants).dt_s))
            return action

    tr = Trace({
        "schema": SCHEMA,
        "name": "r1g-periodic-loop",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = control_period_s
    cfg["rebalance_invariants"]["interval_s"] = interval_s
    cfg["power"]["eclipse_dod_budget_j"] = None
    cfg["thermal"].update(T0_C=50.0, T_max_C=50.0)
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    rec = _RecordingR3Planner()
    policy = CompositePolicy(placer=lambda v, r: 1, planner=rec, invariants=RebalanceInvariants.from_cfg(cfg))
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, policy=policy, seed=1)
    req = Request(0, 0, arrival_t, False, 1, 80, 1.0, deadline_s)
    world.submit(req)
    env.run(until=max(30.0, deadline_s + 2.0))
    actual_ontime = {0} if (req.t_delivered is not None and req.t_delivered <= deadline_s) else set()
    call_times = [t for t, _, _ in rec.records]
    assert len(call_times) >= 2
    for t_a, t_b in zip(call_times, call_times[1:], strict=False):
        assert t_b - t_a == pytest.approx(expected_dt_s, abs=1e-6)
    for _t, pred_ids, model_dt in rec.records:
        assert model_dt == pytest.approx(expected_dt_s, abs=1e-6)
        assert set(pred_ids).issubset(actual_ontime)
    if expect_ontime:
        assert 0 in actual_ontime
        assert any(0 in pred_ids for _, pred_ids, _ in rec.records)



def test_r1g_periodic_closed_loop_migration_positive() -> None:
    """R1g positive: periodic CompositePolicy -> R3RuntimePlanner -> CoWorld migrates under 6s cadence."""
    tr = _two_cs_trace(c1_dark=[[0.0, 60.0]], c2_dark=[])
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 2.0
    cfg["rebalance_invariants"]["interval_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"] = {"inter": 0.16, "rs": 10.0}
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    r = Request(0, 0, 0.0, False, 4800, 60, 1.0, 22.0)
    w.submit(r)
    env.run(until=22.0)
    assert r.node == 2 and r.n_mig == 1 and r.t_delivered is not None and r.t_delivered <= 22.0



def test_r1g_runtime_planner_rejects_horizon_bypass() -> None:
    """R1g-b: R3RuntimePlanner rejects unaligned dt_s even when control_period_s >= horizon_s."""
    tr = _two_cs_trace()
    cfg = _base_cfg()
    cfg["control_period_s"] = 100.0
    cfg["rebalance_invariants"]["interval_s"] = 5.0
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    w.submit(Request(0, 0, 0.0, False, 1, 80, 1.0, 20.0))
    env.run(until=1.0)
    planner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))
    planner.plan(w.forecast_view, RebalanceInvariants.from_cfg(cfg))
    assert planner.last_result is not None
    assert planner.last_result.status == "error"
    assert planner.last_result.goodput_req_ids == ()
    assert planner.last_diag["fallback_reason"].startswith("unaligned_planning_dt:")



@pytest.mark.parametrize(
    ("dt_override", "deadline_s", "expect_ontime", "expect_rejected"),
    [
        (5.0, 18.0, False, True),   # R13 repro: too fine (< 6.0s effective cadence) -> rejected
        (8.0, 18.0, False, True),   # non-multiple of 6.0s effective cadence -> rejected
        (6.0, 26.5, True, False),   # valid aligned override (1x effective cadence) -> positive
        (12.0, 32.5, True, False),  # valid coarser aligned override (2x effective cadence) -> positive
    ],
)
def test_r1g_b_explicit_dt_override_in_periodic_closed_loop(
    dt_override: float,
    deadline_s: float,
    expect_ontime: bool,
    expect_rejected: bool,
) -> None:
    """R1g-b: explicit ProblemBuilder(dt_s=...) must be validated at R3RuntimePlanner runtime boundary."""
    from satplan.planners import CompositePolicy

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(
                builder=ProblemBuilder(horizon_slots=4, dt_s=dt_override, time_limit_s=5.0)
            )
            self.records: list[tuple[str, tuple[int, ...], str | None]] = []

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            self.records.append((res.status, res.goodput_req_ids, self.inner.last_diag["fallback_reason"]))
            return action

    tr = Trace({
        "schema": SCHEMA,
        "name": "r1gb-explicit-dt",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["rebalance_invariants"]["interval_s"] = 5.0
    cfg["power"]["eclipse_dod_budget_j"] = None
    cfg["thermal"].update(T0_C=50.0, T_max_C=50.0)
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.1, t_kv_tok_s=0.0, t_seq_s=0.0)
    rec = _RecPlanner()
    policy = CompositePolicy(placer=lambda v, r: 1, planner=rec, invariants=RebalanceInvariants.from_cfg(cfg))
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, policy=policy, seed=1)
    req = Request(0, 0, 2.001, False, 1, 80, 1.0, deadline_s)
    world.submit(req)
    env.run(until=max(30.0, deadline_s + 2.0))
    actual_ontime = {0} if (req.t_delivered is not None and req.t_delivered <= deadline_s) else set()
    assert len(rec.records) >= 2
    for status, pred_ids, fb_reason in rec.records:
        assert set(pred_ids).issubset(actual_ontime)
        if expect_rejected:
            assert status == "error"
            assert pred_ids == ()
            assert fb_reason is not None and fb_reason.startswith("unaligned_planning_dt:")
        else:
            assert status == "optimal"
            assert fb_reason is None
    if expect_ontime:
        assert 0 in actual_ontime
        assert any(0 in pred_ids for _, pred_ids, _ in rec.records)



def test_r1g_b_injected_custom_builder_grid_validation() -> None:
    """R1g-b: injected custom builder returning unaligned dt_s is rejected at R3RuntimePlanner boundary."""
    from dataclasses import replace

    from satplan.planners import CompositePolicy

    class _CustomUnalignedBuilder:
        def __init__(self) -> None:
            self._base = ProblemBuilder(horizon_slots=4, dt_s=6.0, time_limit_s=5.0)

        def build(self, view, invariants):
            return replace(self._base.build(view, invariants), dt_s=5.0)

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=_CustomUnalignedBuilder())  # type: ignore[arg-type]
            self.records: list[tuple[str, tuple[int, ...], str | None]] = []

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            self.records.append((res.status, res.goodput_req_ids, self.inner.last_diag["fallback_reason"]))
            return action

    tr = Trace({
        "schema": SCHEMA,
        "name": "r1gb-injected-builder",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["rebalance_invariants"]["interval_s"] = 5.0
    rec = _RecPlanner()
    policy = CompositePolicy(placer=lambda v, r: 1, planner=rec, invariants=RebalanceInvariants.from_cfg(cfg))
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, policy=policy, seed=1)
    world.submit(Request(0, 0, 2.001, False, 1, 80, 1.0, 18.0))
    env.run(until=16.0)
    assert len(rec.records) >= 2
    for status, pred_ids, fb_reason in rec.records:
        assert status == "error"
        assert pred_ids == ()
        assert fb_reason is not None and fb_reason.startswith("unaligned_planning_dt:")



def test_r1g_b_rejected_dt_fallback_preserves_inflight_migration_and_paused_recovery() -> None:
    """R1g-b: unaligned dt_s fallback protects pending/in-flight migrations and resumes paused requests."""
    tr = _two_cs_trace()
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["rebalance_invariants"]["interval_s"] = 5.0
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    r_mig = Request(0, 0, 0.0, False, 100, 100, 1.0, 60.0)
    r_paused = Request(1, 0, 0.0, False, 100, 100, 1.0, 60.0)
    w.submit(r_mig)
    w.submit(r_paused)
    env.run(until=1.0)
    r_mig.state = "decode"
    r_mig.mig_dst = 2
    r_mig.mig_in_flight = False
    r_paused.state = "decode"
    r_paused.paused = True
    r_paused.mig_dst = None

    planner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0))
    action = planner.plan(w.forecast_view, RebalanceInvariants.from_cfg(cfg))
    assert planner.last_result is not None
    assert planner.last_result.status == "error"
    assert planner.last_result.goodput_req_ids == ()
    assert planner.last_diag["fallback_reason"].startswith("unaligned_planning_dt:")
    assert action.tier_decisions == {1: None, 2: None}
    assert action.resume_ids == (1,)
    assert len(action.migrations) == 1 and action.migrations[0].req_id == 0 and action.migrations[0].dst == 2

