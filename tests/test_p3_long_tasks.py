"""P3 tests: real unscaled long tasks, near-fine + far-coarse tail slots, and tail on/off ablation."""
from __future__ import annotations

import pytest
import simpy
from factories import _base_cfg
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner
from satplan.r3_solver import R3NodeSpec, R3Problem, R3RequestSpec, R3Solver, R3TierSpec
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


def test_p3_no_fake_scaling_real_remaining_and_deadline() -> None:
    """P3: ProblemBuilder preserves real remaining_tokens and deadline-now (both tail on and tail off)."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p3-no-scaling",
        "horizon_s": 120.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 120.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["gpu"].update(t_fixed_s=0.05, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, True, 100, 1000, 1.0, 70.0)
    w.submit(req)
    env.run(until=2.0)

    inv = RebalanceInvariants.from_cfg(cfg)
    prob_on = ProblemBuilder(horizon_slots=4, tail_enabled=True, tail_slots=4, tail_dt_s=30.0).build(
        w.forecast_view, inv
    )
    prob_off = ProblemBuilder(horizon_slots=4, tail_enabled=False).build(w.forecast_view, inv)

    for prob in (prob_on, prob_off):
        assert len(prob.requests) == 1
        assert prob.requests[0].remaining_tokens == pytest.approx(req.remaining, rel=1e-9)
        assert prob.requests[0].deadline_s == pytest.approx(70.0 - 2.0, abs=1e-9)
    assert prob_on.near_slots == 4
    assert prob_on.slot_dts_s == (6.0, 6.0, 6.0, 6.0, 30.0, 30.0, 30.0, 30.0)
    assert prob_off.slot_dts_s == (6.0, 6.0, 6.0, 6.0)


def test_p3_single_completion_value_across_near_and_tail() -> None:
    """P3: long request completing in coarse tail slot earns weight once and respects shared DoD."""
    tiers = (
        R3TierSpec(index=0, power_w=0.0, rate_tok_s=0.0),
        R3TierSpec(index=1, power_w=75.0, rate_tok_s=10.0),
        R3TierSpec(index=2, power_w=300.0, rate_tok_s=20.0),
    )
    # Near: 2 x 6s = 12s; Far coarse: 2 x 30s = 60s (total 72s).
    # 800 tokens @ 20 tok/s = 40s <= 42s (slot 2 end) <= 50s deadline.
    prob = R3Problem(
        horizon_slots=4,
        dt_s=6.0,
        slot_dts_s=(6.0, 6.0, 30.0, 30.0),
        near_slots=2,
        nodes=(
            R3NodeSpec(
                node_id=1,
                init_temp_C=25.0,
                kv_cap_bytes=1e9,
                is_eclipse=(False, False, False, False),
                tier_rates_tok_s=(0.0, 10.0, 20.0),
            ),
        ),
        tiers=tiers,
        edges=(),
        requests=(
            R3RequestSpec(
                req_id=0,
                init_node=1,
                kv_bytes=1e6,
                remaining_tokens=800.0,
                deadline_s=50.0,
                weight=2.5,
            ),
        ),
        t_amb_C=20.0,
        t_max_C=85.0,
        r_th=0.1,
        c_th=150.0,
        control_period_s=2.0,
        time_limit_s=5.0,
    )
    res = R3Solver().solve(prob)
    assert res.status == "optimal"
    assert res.goodput_req_ids == (0,)
    assert res.goodput_weight == pytest.approx(2.5, abs=1e-6)

    # Same problem with eclipse & 0 DoD in slot 2 blocks tail completion.
    prob_ecl = R3Problem(
        horizon_slots=4,
        dt_s=6.0,
        slot_dts_s=(6.0, 6.0, 30.0, 30.0),
        near_slots=2,
        nodes=(
            R3NodeSpec(
                node_id=1,
                init_temp_C=25.0,
                kv_cap_bytes=1e9,
                is_eclipse=(False, False, True, True),
                init_dod_remaining_j=0.0,
                tier_rates_tok_s=(0.0, 10.0, 20.0),
            ),
        ),
        tiers=tiers,
        edges=(),
        requests=prob.requests,
        t_amb_C=20.0,
        t_max_C=85.0,
        r_th=0.1,
        c_th=150.0,
        eclipse_dod_budget_j=0.0,
        control_period_s=2.0,
        time_limit_s=5.0,
    )
    res_ecl = R3Solver().solve(prob_ecl)
    assert res_ecl.status == "optimal"
    assert res_ecl.goodput_req_ids == ()
    assert res_ecl.goodput_weight == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("tail_enabled", [True, False])
def test_p3_long_task_multi_window_progress_and_tail_ablation(tail_enabled: bool) -> None:
    """P3 closed-loop ablation: long task advances monotonically across >=3 near windows under tail on/off."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p3-long-ablation",
        "horizon_s": 120.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 120.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 120.0]]},
        ],
        "dark": [[], [], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.08, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.05, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)  # 20 tok/s at high
    records: list[dict] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(
                builder=ProblemBuilder(
                    horizon_slots=4,
                    tail_enabled=tail_enabled,
                    tail_slots=4,
                    tail_dt_s=30.0,
                    time_limit_s=5.0,
                )
            )

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            reqs = view.node_requests(1) + view.node_requests(2)
            records.append({
                "time_s": view.now,
                "remaining": {r.i: r.remaining for r in reqs},
                "pred_ids": res.goodput_req_ids,
                "goodput_weight": res.goodput_weight,
                "tiers": action.tier_decisions,
                "pause_ids": action.pause_ids,
                "migrations": [(m.req_id, m.dst) for m in action.migrations],
            })
            return action

    env = simpy.Environment()
    policy = CompositePolicy(
        placer=lambda v, r: 1,
        planner=_RecPlanner(),
        invariants=RebalanceInvariants.from_cfg(cfg),
    )
    world = CoWorld(env, tr, cfg, policy=policy, seed=1)
    long_req = Request(0, 0, 0.0, True, 200, 900, 1.0, 65.0)  # 45s @ 20 tok/s
    world.submit(long_req)
    env.run(until=70.0)

    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert len(records) >= 4
    # Monotonic progress across >= 3 near windows while active
    rems = [rec["remaining"][0] for rec in records if 0 in rec["remaining"]]
    assert len(rems) >= 4
    assert all(b < a for a, b in zip(rems, rems[1:], strict=False))
    assert all(rec["pause_ids"] == () for rec in records)
    assert all(set(rec["pred_ids"]).issubset(actual_ontime) for rec in records)
    if tail_enabled:
        assert 0 in actual_ontime
        assert 0 in records[0]["pred_ids"]
    else:
        # With tail disabled (24s near window), 900 tok (45s) cannot finish in near window at t=2s.
        assert 0 not in records[0]["pred_ids"]
        assert long_req.gen > 0.0


@pytest.mark.parametrize("tail_enabled", [True, False])
def test_p3_tail_enables_eclipse_migration_for_long_task(tail_enabled: bool) -> None:
    """P3 ablation: tail_enabled=True sees tail completion value and migrates long task out of eclipse."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p3-eclipse-mig-ablation",
        "horizon_s": 120.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 120.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 120.0]]},
        ],
        "dark": [[], [[0.0, 120.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.08, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.05, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    planner = R3RuntimePlanner(
        builder=ProblemBuilder(
            horizon_slots=4, tail_enabled=tail_enabled, tail_slots=4, tail_dt_s=30.0, time_limit_s=5.0
        )
    )
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=planner, seed=1)
    long_req = Request(0, 0, 0.0, True, 200, 900, 1.0, 65.0)
    world.submit(long_req)
    env.run(until=70.0)
    if tail_enabled:
        assert long_req.node == 2 and long_req.t_delivered is not None and long_req.t_delivered <= 65.0
    else:
        assert long_req.t_delivered is None


def test_p3_truly_infeasible_long_task_not_falsely_promised() -> None:
    """P3: genuinely infeasible long task (needs 100s, deadline 60s) is never falsely promised."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p3-infeasible-long",
        "horizon_s": 120.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 120.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["gpu"].update(t_fixed_s=0.05, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    preds: list[tuple[int, ...]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(
                builder=ProblemBuilder(horizon_slots=4, tail_enabled=True, tail_slots=4, tail_dt_s=30.0, time_limit_s=5.0)
            )

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            if self.inner.last_result is not None:
                preds.append(self.inner.last_result.goodput_req_ids)
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=_RecPlanner(), seed=1)
    req = Request(0, 0, 0.0, True, 200, 2000, 1.0, 60.0)  # needs 100s > 60s
    world.submit(req)
    env.run(until=65.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert actual_ontime == set()
    assert all(p == () for p in preds)
    assert req.gen > 0.0
