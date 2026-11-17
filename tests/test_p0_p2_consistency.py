"""Independent seam tests for P0-P2: state machine, budgets, execution semantics, and safe fallback."""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import satplan.r3_solver as solver_module
import simpy
from satplan.actions import MigrationOrder, PlanAction
from satplan.placers import get_placer
from satplan.planners import RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner
from satplan.r3_solver import (
    R3EdgeSpec,
    R3NodeSpec,
    R3Problem,
    R3RequestSpec,
    R3Solver,
    R3TierSpec,
)
from satplan.runner import run_simulation
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request

TIERS_STD = (
    R3TierSpec(index=0, power_w=0.0, rate_tok_s=0.0),
    R3TierSpec(index=1, power_w=100.0, rate_tok_s=10.0),
    R3TierSpec(index=2, power_w=300.0, rate_tok_s=20.0),
)


def _base_cfg(*, max_mig_per_req: int | None = 4, kv_gib: float = 0.5, hop_latency_s: float = 0.0) -> dict:
    inv: dict[str, object] = {"interval_s": 5.0, "max_inflight": 8}
    if max_mig_per_req is not None:
        inv["max_mig_per_req"] = max_mig_per_req
    else:
        inv["max_mig_per_req"] = None
    return {
        "name": "p0_p2_seam",
        "delivery": "final",
        "control_period_s": 5.0,
        "rebalance_invariants": inv,
        "model": {"layers": 4, "kv_heads": 4, "head_dim": 64, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": kv_gib,
            "t_fixed_s": 0.005,
            "t_weights_s": 0.015,
            "t_kv_tok_s": 1e-7,
            "t_seq_s": 1e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {
            "P_idle": 60,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 80,
            "eclipse_dod_budget_j": 1500.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
                {"name": "low_75w", "power_w": 75.0, "clock_f": 0.5},
                {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
            ],
        },
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.2, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 1.0, "inter": 1.0, "rs": 10.0}, "hop_latency_s": hop_latency_s},
        "workload": {
            "arrival_window_s": 10.0,
            "rate_per_rs_per_min": 0.0,
            "p_long": 0.0,
            "long_out": [100, 100],
            "short_out": [100, 100],
            "long_pin": 100,
            "short_pin": 100,
            "long_upload_MB": 0.1,
            "short_upload_MB": 0.1,
        },
        "slo": {"deadline_factor_long": 2.0, "deadline_factor_short": 4.0, "min_deadline_s": 20.0},
    }


def _two_cs_trace(*, c1_dark: list[list[float]] | None = None, c2_dark: list[list[float]] | None = None) -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "two-cs",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
            {"id": 2, "name": "C-2", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], c1_dark or [], c2_dark or []],
    })


def test_bridge_r3_problem_solver_plan_action_coworld_first_slot() -> None:
    """R3Problem -> R3Solver -> PlanAction -> CoWorld first control period consistency."""
    tr = _two_cs_trace(c1_dark=[[0.0, 60.0]], c2_dark=[])
    cfg = _base_cfg()
    cfg["power"]["eclipse_dod_budget_j"] = 0.0  # Node 1 has 0 J DoD -> must migrate to Node 2
    env = simpy.Environment()
    planner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4))
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=planner, seed=1)

    r0 = Request(0, 0, 0.0, False, 1000, 200, 1000.0, 30.0)
    r1 = Request(1, 0, 0.0, False, 1000, 200, 1000.0, 30.0)
    w.submit(r0)
    w.submit(r1)
    env.run(until=4.9)  # Before first control tick at 5.0s; both uploaded to node 1
    assert r0.node == 1 and r1.node == 1
    assert r0.state == "decode" and r1.state == "decode"

    inv = RebalanceInvariants.from_cfg(cfg)
    prob = planner.builder.build(w.forecast_view, inv)
    res = planner.solver.solve(prob)
    assert res.status in ("optimal", "feasible_time_limit")

    # Apply PlanAction at t=5.0s and advance one control slot (to 10.0s)
    env.run(until=5.001)
    assert planner.last_result is not None
    last_res = planner.last_result

    # Check node 1 tier & temperature prediction vs CoWorld after 1 slot
    env.run(until=10.001)
    for node_id in (1, 2):
        assert w.temp_C[node_id] == pytest.approx(last_res.node_temps_C[node_id][0], abs=1e-6)
    # Any migrating request must have had zero decode progress during migration
    for order in last_res.action.migrations:
        req = w.req_by_id(order.req_id)
        assert req is not None
        assert req.node == order.dst  # Unlocked within 5s slot at 1 Gbps
        assert req.gen == pytest.approx(0.0, abs=1e-6)
        assert w.kv_used(order.dst) >= req.reserved_tokens * w.bpt - 1e-6


def test_same_slot_compute_and_transfer_conflict_rejected() -> None:
    """Cannot compute on src AND complete transfer in the same slot to meet an impossible deadline.

    Node 0 has power (10 tok/s) so old model could compute 50 tok on node 0 in slot 0 WHILE
    transferring 100 MB to node 1, then compute 100 tok on node 1 (20 tok/s) in slot 1 = 150 tok by 10s.
    With migration-compute mutual exclusion, migrating in slot 0 yields 0 tok, so 150 tok by 10s is impossible.
    """
    dt_s = 5.0
    bps = 100e6 * 8.0 / dt_s  # 100 MB per 5s slot
    tiers_asym = (
        R3TierSpec(index=0, power_w=0.0, rate_tok_s=0.0),
        R3TierSpec(index=1, power_w=100.0, rate_tok_s=10.0),
        R3TierSpec(index=2, power_w=300.0, rate_tok_s=20.0),
    )
    prob_impossible = R3Problem(
        horizon_slots=3,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 3, tier_rates_tok_s=(0.0, 10.0, 10.0)),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 3, tier_rates_tok_s=(0.0, 10.0, 20.0)),
        ),
        tiers=tiers_asym,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 3),),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=150.0, deadline_s=10.0),),
    )
    res_imp = R3Solver().solve(prob_impossible)
    assert res_imp.goodput_req_ids == ()

    prob_possible = R3Problem(
        horizon_slots=3,
        dt_s=dt_s,
        nodes=prob_impossible.nodes,
        tiers=tiers_asym,
        edges=prob_impossible.edges,
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),),
    )
    res_pos = R3Solver().solve(prob_possible)
    assert res_pos.goodput_req_ids == (0,)


def test_dest_kv_contention_and_multislot_continuation_and_no_byte_reuse() -> None:
    """Test destination KV pre-reservation, multi-slot continuation under max_mig=1, and no byte reuse."""
    dt_s = 5.0
    bps = 100e6 * 8.0 / dt_s  # 100 MB/slot

    # 1) Destination KV contention: node 1 has 250 MB cap; r0 (200 MB) takes 2 slots to migrate 0->1.
    # r1 (200 MB) cannot reserve node 1 simultaneously while r0 is migrating!
    prob_kv = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 4),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=250e6, is_eclipse=(False,) * 4),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 4),),
        requests=(
            R3RequestSpec(req_id=0, init_node=0, kv_bytes=200e6, remaining_tokens=50.0, deadline_s=15.0),
            R3RequestSpec(req_id=1, init_node=0, kv_bytes=200e6, remaining_tokens=50.0, deadline_s=15.0),
        ),
        eclipse_dod_budget_j=0.0,
    )
    res_kv = R3Solver().solve(prob_kv)
    # At most 1 of r0/r1 can migrate to node 1 at t=0 because node 1 only fits 250 MB
    assert len(res_kv.action.migrations) <= 1

    # 2) Continuation of partial migration is NOT blocked by max_mig_per_req=1 and does not double-count n_mig
    tr = _two_cs_trace(c1_dark=[[0.0, 60.0]], c2_dark=[])
    cfg = _base_cfg(max_mig_per_req=1)
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    r = Request(0, 0, 0.0, False, 1000, 100, 1000.0, 50.0)
    w.submit(r)
    env.run(until=4.0)
    # Simulate an interrupted partial migration session (mig_dst=2, received half, not in_flight)
    w.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=r.kv * w.bpt * 0.5),)))
    env.run(until=4.9)
    assert r.mig_dst == 2
    assert r.n_mig == 0
    assert w.kv_used(2) == pytest.approx(r.reserved_tokens * w.bpt)
    # Control tick at 5.0s continues the migration even with max_mig_per_req=1
    env.run(until=10.0)
    assert r.node == 2
    assert r.n_mig == 1
    assert w.reserved[2] == pytest.approx(0.0)

    # 3) No reusing old bytes on return migration (0 -> 1 -> 0 must pay 100 MB each way = 200 MB)
    # Node 0 sunlit only in slots 3,4 (200 tok max); Node 1 sunlit in slots 1,2 (200 tok max).
    # To finish 300 tok by 25s, r0 must migrate 0->1 (slot 0), decode 200 tok (slots 1,2), migrate 1->0 (slot 3), decode 100 tok (slot 4).
    prob_aba = R3Problem(
        horizon_slots=5,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True, True, True, False, False)),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False, False, False, True, True)),
        ),
        tiers=TIERS_STD,
        edges=(
            R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 5),
            R3EdgeSpec(src=1, dst=0, tot_bps=(bps,) * 5),
        ),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=300.0, deadline_s=25.0),),
        eclipse_dod_budget_j=0.0,
        max_mig_per_req=None,
    )
    res_aba = R3Solver().solve(prob_aba)
    assert res_aba.goodput_req_ids == (0,)
    assert res_aba.migration_bytes >= 200e6 - 1e-3


def test_budget_exhausted_vs_null_unlimited_uniform_across_planners() -> None:
    """max_mig_per_req omitted defaults to 4; None allows >1 relay migrations uniformly for reactive & r3_milp."""
    assert RebalanceInvariants.from_cfg({}).max_mig_per_req == 4
    assert RebalanceInvariants.from_cfg({"rebalance_invariants": {"max_mig_per_req": None}}).max_mig_per_req is None

    # C-1 sunlit ONLY [19, 26] (333 tok < 350); C-2 sunlit ONLY [0, 12.6] (301 tok < 350 after 1.28s transfer).
    # Request (pin=10000, pout=350, deadline=26.0) MUST do 1->2 at t=5 and 2->1 at t=15/20 (2 migrations) to finish.
    tr = _two_cs_trace(c1_dark=[[0.0, 19.0], [26.0, 60.0]], c2_dark=[[12.6, 60.0]])
    for planner_name in ("reactive", "r3_milp"):
        cfg_lim = _base_cfg(max_mig_per_req=1)
        cfg_lim["power"]["eclipse_dod_budget_j"] = 0.0
        cfg_lim["net"]["link_gbps"]["inter"] = 0.25
        env_lim = simpy.Environment()
        w_lim = CoWorld(env_lim, tr, cfg_lim, placer=get_placer("nearest"), planner=get_planner(planner_name), seed=1)
        r_lim = Request(0, 0, 0.0, False, 10000, 350, 100.0, 26.0)
        w_lim.submit(r_lim)
        env_lim.run(until=60.0)
        assert r_lim.n_mig <= 1

        cfg_inf = _base_cfg(max_mig_per_req=None)
        cfg_inf["power"]["eclipse_dod_budget_j"] = 0.0
        cfg_inf["net"]["link_gbps"]["inter"] = 0.25
        env_inf = simpy.Environment()
        w_inf = CoWorld(env_inf, tr, cfg_inf, placer=get_placer("nearest"), planner=get_planner(planner_name), seed=1)
        r_inf = Request(0, 0, 0.0, False, 10000, 350, 100.0, 26.0)
        w_inf.submit(r_inf)
        env_inf.run(until=60.0)
        assert r_inf.n_mig >= 2


def test_dual_flow_max_min_bottleneck_and_heterogeneous_context_batch() -> None:
    """Dual flows sharing bottleneck edge get cap/2 (max-min); heterogeneous context batch shares iter_time."""
    dt_s = 5.0
    bps = 100e6 * 8.0 / dt_s  # 100 MB/slot on edge 0->1
    # Two 100 MB requests both migrating 0->1 simultaneously get 50 MB/slot each -> need 2 slots to unlock!
    # So deadline_s=10.0s (finish in slot 1 after 1-slot unlock) is IMPOSSIBLE if both migrate in slot 0.
    # Only ONE can finish by 10.0s (staggered), the other must finish at 15.0s!
    prob_dual = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 4),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 4),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 4),),
        requests=(
            R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),
            R3RequestSpec(req_id=1, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),
        ),
        eclipse_dod_budget_j=0.0,
    )
    res_dual = R3Solver().solve(prob_dual)
    assert len(res_dual.goodput_req_ids) == 1

    # Heterogeneous context batch in CoWorld + ProblemBuilder: both active requests advance at same batch rate
    tr = _two_cs_trace()
    cfg = _base_cfg()
    cfg["gpu"]["t_kv_tok_s"] = 1e-4  # Make context sensitivity visible
    env = simpy.Environment()
    builder = ProblemBuilder(horizon_slots=2)
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=R3RuntimePlanner(builder=builder), seed=1)
    r_small = Request(0, 0, 0.0, False, 100, 500, 100.0, 40.0)
    r_large = Request(1, 0, 0.0, False, 2000, 500, 100.0, 40.0)
    w.submit(r_small)
    w.submit(r_large)
    env.run(until=2.0)
    prob_batch = builder.build(w.forecast_view, RebalanceInvariants.from_cfg(cfg))
    node1_spec = next(n for n in prob_batch.nodes if n.node_id == 1)
    assert node1_spec.tier_rates_tok_s is not None
    # Both requests on node 1 should advance at the exact same per-request batch rate in CoWorld
    g0_before, g1_before = r_small.gen, r_large.gen
    env.run(until=5.0)
    assert (r_small.gen - g0_before) == pytest.approx(r_large.gen - g1_before, rel=1e-6)


def test_solver_status_classification_and_safe_fallback_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinguish proven_infeasible, timeout_no_incumbent, feasible_time_limit, error; fallback resumes paused reqs."""
    # 1) Proven infeasible: init_temp_C > T_max in slot 1 (impossible thermal bound)
    prob_inf = R3Problem(
        horizon_slots=2,
        dt_s=5.0,
        nodes=(R3NodeSpec(node_id=0, init_temp_C=120.0, kv_cap_bytes=500e6, is_eclipse=(False, False)),),
        tiers=TIERS_STD,
        edges=(),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=10e6, remaining_tokens=10.0, deadline_s=10.0),),
        t_max_C=85.0,
    )
    res_inf = R3Solver().solve(prob_inf)
    assert res_inf.status == "proven_infeasible"
    assert res_inf.fallback_reason is not None

    # 2) Mock timeout_no_incumbent, feasible_time_limit, and error
    prob_ok = R3Problem(
        horizon_slots=2,
        dt_s=5.0,
        nodes=(R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False, False)),),
        tiers=TIERS_STD,
        edges=(),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=10e6, remaining_tokens=10.0, deadline_s=10.0),),
    )
    monkeypatch.setattr("satplan.r3_solver.milp", lambda **kw: SimpleNamespace(status=1, x=None, message="time limit"))
    res_to = R3Solver().solve(prob_ok)
    assert res_to.status == "timeout_no_incumbent"

    monkeypatch.setattr("satplan.r3_solver.milp", lambda **kw: SimpleNamespace(status=4, x=None, message="numeric err"))
    res_err = R3Solver().solve(prob_ok)
    assert res_err.status == "error"

    # 3) Safe fallback in R3RuntimePlanner clears explicit tier=0 and resumes paused non-migrating requests
    monkeypatch.undo()
    tr = _two_cs_trace()
    cfg = _base_cfg()
    env = simpy.Environment()
    planner = R3RuntimePlanner()
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=planner, seed=1)
    r = Request(0, 0, 0.0, False, 100, 100, 100.0, 30.0)
    w.submit(r)
    env.run(until=2.0)
    w.apply_plan(PlanAction(tier_decisions={1: 0}, pause_ids=(0,)))
    assert r.paused is True
    assert w.pt.explicit_tier_idx[1] == 0

    monkeypatch.setattr("satplan.r3_solver.milp", lambda **kw: SimpleNamespace(status=2, x=None, message="infeasible"))
    fb_action = planner.plan(w.forecast_view, RebalanceInvariants.from_cfg(cfg))
    assert fb_action.tier_decisions == {1: None, 2: None}
    assert 0 in fb_action.resume_ids
    w.apply_plan(fb_action)
    assert r.paused is False
    assert w.pt.explicit_tier_idx[1] is None


def test_regression_local_view_0w_idle_and_store_and_run() -> None:
    """LocalStateView hides future; discrete 0W idle uses 0W; Store-and-Run drops zero requests."""
    tr = _two_cs_trace(c1_dark=[[10.0, 30.0]])
    cfg = _base_cfg()
    cfg["workload"]["rate_per_rs_per_min"] = 12.0
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    assert not hasattr(w.local_view, "power_segments")
    assert not hasattr(w.local_view, "next_conn")
    assert not hasattr(w.local_view, "path_at")

    res = run_simulation(tr, cfg, placer="nearest", planner="r3_milp", seed=1)
    assert res.metrics["n_req"] > 0
    assert res.metrics["delivered"] == res.metrics["n_req"]
    assert res.metrics["peak_temp_C"] <= cfg["thermal"]["T_max_C"] + 1e-6
    assert math.isfinite(res.metrics["energy_kj"])


def _exec_returned_schedule(world: CoWorld, env: simpy.Environment, result, start: float, dt: float = 5.0, slots: int = 2) -> None:
    world.apply_plan(result.action)
    for slot in range(1, slots):
        env.run(until=start + slot * dt)
        world.apply_plan(PlanAction(tier_decisions={j: tiers[slot] for j, tiers in result.node_tiers.items()}))
    env.run(until=start + slots * dt + 1e-6)


def test_r1_frozen_batch_after_join_must_not_promise_impossible_goodput() -> None:
    """R1 counterexample: two requests joining an empty node double batch iter_time (0.1*2=0.2s)."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "review-frozen-batch",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}] + [{"id": j, "role": "compute"} for j in (1, 2, 3)],
        "edges": [
            {"u": s, "v": d, "kind": "inter" if s else "rs", "up": [[0.0, 60.0]]}
            for s, d in ((0, 1), (0, 2), (1, 3), (2, 3))
        ],
        "dark": [[], [[0.0, 60.0]], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.1)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda view, req: req.i + 1, planner=get_planner("none"), seed=1)
    reqs = [Request(i, 0, 0.0, False, 1, 50, 1.0, 11.0) for i in (0, 1)]
    for req in reqs:
        world.submit(req)
    env.run(until=1.0)
    problem = ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 1.0)
    actual_ontime_ids = {r.i for r in reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    # Every request predicted as goodput MUST actually finish on time in CoWorld
    assert set(result.goodput_req_ids) <= actual_ontime_ids


def test_r2_latency_with_fair_share_and_cross_slot_propagation() -> None:
    """R2 counterexample: 2 * 102.4 MB / 125 MB/s + 4s prop = 5.6384s > 5s slot; cannot unlock in slot 0."""
    cfg = _base_cfg(hop_latency_s=4.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(
        env,
        _two_cs_trace(c1_dark=[[0.0, 60.0]]),
        cfg,
        placer=lambda view, req: 1,
        planner=get_planner("none"),
        seed=1,
    )
    reqs = [Request(i, 0, 0.0, False, 25000, 50, 1.0, 15.0) for i in (0, 1)]
    for req in reqs:
        world.submit(req)
    env.run(until=5.0)
    problem = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 5.0)
    actual_ontime_ids = {r.i for r in reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids) <= actual_ontime_ids

    # Also verify cross-slot propagation (> dt_s) does NOT freeze bytes to zero
    dt_s = 5.0
    bps = 100e6 * 8.0 / dt_s
    prob_long_prop = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 4),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 4),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 4),),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=50e6, remaining_tokens=50.0, deadline_s=20.0),),
        eclipse_dod_budget_j=0.0,
        hop_latency_s=6.0,
    )
    res_lp = R3Solver().solve(prob_long_prop)
    # 50MB takes 2.5s + 6.0s prop = 8.5s -> unlocks at t=2 (10s), decodes 50 tok in slot 2 -> finishes 15s <= 20s
    assert res_lp.goodput_req_ids == (0,)


def test_r3_secondary_solver_failure_must_not_report_optimal() -> None:
    """R3 counterexample: Stage 2 error/timeout must degrade status & record fallback_reason."""
    problem = R3Problem(
        horizon_slots=2,
        dt_s=5.0,
        nodes=(R3NodeSpec(0, 25.0, 1000.0, (False, False)),),
        tiers=(R3TierSpec(0, 0.0, 0.0), R3TierSpec(1, 1.0, 10.0)),
        edges=(),
        requests=(R3RequestSpec(0, 0, 10.0, 10.0, 10.0),),
    )
    backend = solver_module.milp
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return SimpleNamespace(status=4, x=None, message="controlled secondary numerical error")
        return backend(**kwargs)

    with patch.object(solver_module, "milp", side_effect=fail_second):
        result = R3Solver().solve(problem)
    assert result.status != "optimal"
    assert result.fallback_reason is not None
    assert "stage2" in result.fallback_reason
    assert result.stage1_status == "optimal"
    assert result.stage2_status == "error"

    # Stage 2 timeout with incumbent -> feasible_time_limit (not optimal)
    calls = 0

    def timeout_second_with_x(**kwargs):
        nonlocal calls
        calls += 1
        res = backend(**kwargs)
        if calls == 2:
            return SimpleNamespace(status=1, x=res.x, mip_gap=0.05, message="stage2 time limit")
        return res

    with patch.object(solver_module, "milp", side_effect=timeout_second_with_x):
        res_s2_to = R3Solver().solve(problem)
    assert res_s2_to.status == "feasible_time_limit"
    assert res_s2_to.stage2_status == "feasible_time_limit"

    # Stage 1 timeout with incumbent -> feasible_time_limit
    calls = 0

    def timeout_first_with_x(**kwargs):
        nonlocal calls
        calls += 1
        res = backend(**kwargs)
        if calls == 1:
            return SimpleNamespace(status=1, x=res.x, mip_gap=0.1, message="stage1 time limit")
        return res

    with patch.object(solver_module, "milp", side_effect=timeout_first_with_x):
        res_s1_to = R3Solver().solve(problem)
    assert res_s1_to.status == "feasible_time_limit"
    assert res_s1_to.stage1_status == "feasible_time_limit"


def test_r2_multihop_heterogeneous_paths_and_inflight_session() -> None:
    """Multi-hop heterogeneous paths (0->1->2 vs 0->3->2) and existing in-flight session."""
    dt_s = 5.0
    bps_fast = 100e6 * 8.0 / dt_s
    bps_slow = 50e6 * 8.0 / dt_s
    prob = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(0, 25.0, 500e6, (True,) * 4),
            R3NodeSpec(1, 25.0, 0.0, (True,) * 4),
            R3NodeSpec(2, 25.0, 500e6, (False,) * 4),
        ),
        tiers=TIERS_STD,
        edges=(
            R3EdgeSpec(0, 1, (bps_fast,) * 4),
            R3EdgeSpec(1, 2, (bps_slow,) * 4),
        ),
        requests=(
            R3RequestSpec(
                0,
                0,
                100e6,
                50.0,
                20.0,
                init_mig_dst=2,
                init_mig_received_bytes=50e6,
                init_mig_in_flight=True,
            ),
        ),
        eclipse_dod_budget_j=0.0,
        hop_latency_s=0.5,
        max_inflight=1,
    )
    res = R3Solver().solve(prob)
    assert res.status == "optimal"
    # Existing in-flight session is not re-emitted in slot 0, completes remaining 50 MB + 1.0s prop in slot 0,
    # unlocks at t=1 (5s), decodes 50 tok in slot 1 -> finishes at 10s <= 20s
    assert res.action.migrations == ()
    assert res.goodput_req_ids == (0,)


def test_r1a_skipped_small_batch_breakpoint_must_be_conservative() -> None:
    """R1a counterexample: sparse batch breakpoint intervals must use slowest rate in interval."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "review-small-batch-breakpoint",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}] + [{"id": j, "role": "compute"} for j in (1, 2, 3)],
        "edges": [
            {"u": s, "v": d, "kind": "inter" if s else "rs", "up": [[0.0, 60.0]]}
            for s, d in ((0, 1), (0, 2), (1, 3), (2, 3))
        ],
        "dark": [[], [[0.0, 60.0]], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.0997, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0003)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda view, req: 2 if req.i == 1 else 1, planner=get_planner("none"), seed=1)
    reqs = [Request(i, 0, 0.0, False, 1, 50, 1.0, 11.0) for i in (0, 1)]
    reqs.append(Request(2, 0, 0.0, False, 1, 1000, 1.0, 10000.0))
    for req in reqs:
        world.submit(req)
    env.run(until=1.0)
    problem = ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 1.0)
    actual_ontime_ids = {r.i for r in reqs[:2] if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids) <= actual_ontime_ids


def test_r1b_growing_context_within_horizon_must_be_conservative() -> None:
    """R1b counterexample: context grows as tokens are generated within horizon (1 -> 501+)."""
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 100.0
    cfg["gpu"].update(t_fixed_s=0.0, t_weights_s=0.0, t_kv_tok_s=0.01, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(env, _two_cs_trace(), cfg, placer=lambda view, req: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, False, 1, 600, 1.0, 10.000001)
    world.submit(req)
    env.run(until=0.000001)
    problem = ProblemBuilder(horizon_slots=2, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 0.000001)
    actual_ontime_ids = {req.i} if (req.t_delivered is not None and req.t_delivered <= req.deadline) else set()
    assert set(result.goodput_req_ids) <= actual_ontime_ids


def test_r2a_repeated_propagation_completion_blocked() -> None:
    """R2a counterexample: tx_just_done must fire at most once per session; 20s prop > 15s deadline."""
    problem = R3Problem(
        horizon_slots=5,
        dt_s=5.0,
        nodes=(R3NodeSpec(0, 25.0, 500e6, (True,) * 5), R3NodeSpec(1, 25.0, 500e6, (False,) * 5)),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(0, 1, (160e6,) * 5),),
        requests=(R3RequestSpec(0, 0, 50e6, 50.0, 15.0),),
        eclipse_dod_budget_j=0.0,
        hop_latency_s=20.0,
    )
    result = R3Solver().solve(problem)
    assert result.goodput_req_ids == ()

    # With 30s deadline (2.5s tx + 20s prop -> unlocks at t=25s, finishes 50 tok at 27.5s <= 30s), succeeds!
    prob_ok = R3Problem(
        horizon_slots=6,
        dt_s=5.0,
        nodes=(R3NodeSpec(0, 25.0, 500e6, (True,) * 6), R3NodeSpec(1, 25.0, 500e6, (False,) * 6)),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(0, 1, (160e6,) * 6),),
        requests=(R3RequestSpec(0, 0, 50e6, 50.0, 30.0),),
        eclipse_dod_budget_j=0.0,
        hop_latency_s=20.0,
    )
    res_ok = R3Solver().solve(prob_ok)
    assert res_ok.goodput_req_ids == (0,)
    # Node 1 cannot run high tier before slot 5 (t=25s) when propagation finishes
    assert res_ok.node_tiers[1][:5] == (0, 0, 0, 0, 0)


def test_r2b_partial_session_in_propagation_must_not_promise_impossible_16s() -> None:
    """R2b counterexample: partial chunk in propagation wait (t=6s, prop ends 9.4s) cannot resume at t=6s."""
    cfg = _base_cfg(hop_latency_s=4.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(
        env,
        _two_cs_trace(c1_dark=[[0.0, 60.0]]),
        cfg,
        placer=lambda view, req: 1,
        planner=get_planner("none"),
        seed=1,
    )
    req = Request(0, 0, 0.0, False, 25000, 50, 1.0, 16.0)
    world.submit(req)
    env.run(until=5.0)
    world.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=50_000_000.0),)))
    # At t=6.0s: 50 MB finished sending at 5.4s, now waiting 4s propagation until 9.4s
    env.run(until=6.0)
    assert req.mig_in_flight is True
    assert req.mig_received_bytes == pytest.approx(50_000_000.0)
    assert world.local_view.edge_rates() == {}

    problem = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    assert result.action.migrations == ()
    # Earliest physical finish is 9.4 + 0.4192 + 4.0 + 5.0 = 18.8192s > 16.0s
    assert result.goodput_req_ids == ()

    # Positive case with wide deadline (26s) under real periodic control: resumes and finishes on time
    cfg_ok = _base_cfg(hop_latency_s=4.0)
    cfg_ok["rebalance_invariants"]["interval_s"] = 3.0
    cfg_ok["control_period_s"] = 3.0
    cfg_ok["power"]["eclipse_dod_budget_j"] = 0.0
    cfg_ok["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env_ok = simpy.Environment()
    w_ok = CoWorld(
        env_ok,
        _two_cs_trace(c1_dark=[[0.0, 60.0]]),
        cfg_ok,
        placer=get_placer("nearest"),
        planner=get_planner("r3_milp"),
        seed=1,
    )
    r_ok = Request(0, 0, 0.0, False, 25000, 50, 1.0, 26.0)
    w_ok.submit(r_ok)
    env_ok.run(until=4.5)
    w_ok.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=50_000_000.0),)))
    env_ok.run(until=26.0)
    assert r_ok.node == 2
    assert r_ok.n_mig == 1
    assert r_ok.t_delivered is not None and r_ok.t_delivered <= 26.0


def test_r2b_snapshots_sending_and_propagating_unlock_across_ticks() -> None:
    """Verify sending and propagating_unlock snapshots preserve KV/inflight and do not resend."""
    cfg = _base_cfg(hop_latency_s=4.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"]["inter"] = 0.2  # 25 MB/s -> 102.4 MB takes 4.096s + 4s prop = 8.096s
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(
        env,
        _two_cs_trace(c1_dark=[[0.0, 60.0]]),
        cfg,
        placer=lambda view, req: 1,
        planner=get_planner("none"),
        seed=1,
    )
    req = Request(0, 0, 0.0, False, 25000, 50, 1.0, 25.0)
    world.submit(req)
    env.run(until=5.0)
    world.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))

    # At t=7.0s: still sending (50 MB sent, 52.4 MB remaining)
    env.run(until=7.0)
    snap_send = world.local_view.migration_snapshot(0)
    assert snap_send is not None and snap_send.phase == "sending"
    assert snap_send.chunk_rem_bytes == pytest.approx(52_400_000.0, rel=1e-4)
    prob_send = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    res_send = R3Solver().solve(prob_send)
    assert res_send.action.migrations == ()
    assert world.local_view.inflight_migrations == 1

    # At t=11.0s: all 102.4 MB received (finished at 9.096s), now in propagating_unlock until 13.096s
    env.run(until=11.0)
    snap_prop = world.local_view.migration_snapshot(0)
    assert snap_prop is not None and snap_prop.phase == "propagating_unlock"
    assert snap_prop.prop_rem_s == pytest.approx(2.096, abs=1e-2)
    prob_prop = ProblemBuilder(horizon_slots=3, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    res_prop = R3Solver().solve(prob_prop)
    assert res_prop.action.migrations == ()
    assert res_prop.goodput_req_ids == (0,)  # Unlocks by t=16s, decodes 5s -> finishes 21s <= 25s
    assert res_prop.node_tiers[2][0] == 0  # Cannot decode in slot 0 [11, 16] while propagating


def test_r2c_sending_on_frozen_path_cannot_reroute_to_shortcut() -> None:
    """R2c counterexample: in-flight sending flow on 1->3->2 (5 MB/s) cannot jump to new 1->2 (125 MB/s) shortcut."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "review-frozen-path",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "intra", "up": [[0.0, 60.0]]},  # 5 MB/s
            {"u": 3, "v": 2, "kind": "intra", "up": [[0.0, 60.0]]},  # 5 MB/s
            {"u": 1, "v": 2, "kind": "inter", "up": [[6.0, 60.0]]},  # 125 MB/s shortcut appears at t=6s
        ],
        "dark": [[], [[0.0, 60.0]], [], [[0.0, 60.0]]],
    })
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"] = {"intra": 0.04, "inter": 1.0, "rs": 10.0}  # intra=5 MB/s, inter=125 MB/s
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda view, req: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, False, 25000, 50, 1.0, 22.0)
    world.submit(req)
    env.run(until=5.0)
    world.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))
    # At t=7.0s: 10 MB sent on 1->3->2 (edges (1, 2)), 92.4 MB remaining; 1->2 shortcut is up since t=6s
    env.run(until=7.0)
    snap = world.local_view.migration_snapshot(0)
    assert snap is not None and snap.phase == "sending"
    assert snap.path_edges == (1, 2)
    assert snap.chunk_rem_bytes == pytest.approx(92_400_000.0, rel=1e-4)

    problem = ProblemBuilder(horizon_slots=5, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    assert result.action.migrations == ()
    # Physical lower bound on 1->3->2 is 5 + 102.4/5 + 5 = 30.48s > 22.0s
    assert result.goodput_req_ids == ()

    # Positive case with wide deadline (35.0s): old path completes and delivers on time
    req.deadline = 35.0
    prob_ok = ProblemBuilder(horizon_slots=6, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    res_ok = R3Solver().solve(prob_ok)
    assert res_ok.goodput_req_ids == (0,)
    _exec_returned_schedule(world, env, res_ok, 7.0, slots=6)
    assert req.node == 2 and req.t_delivered is not None and req.t_delivered <= 35.0


@pytest.mark.parametrize(
    ("hop_lat_s", "pout", "pin", "chunk1_bytes", "wide_deadline"),
    [
        (4.0, 50, 150_000, 500_000_000.0, 30.0),       # 614.4 MB total, 500 MB chunk1 -> 9s data, 13s prop end
        (0.005, 50, 335_449, 750_000_000.0, 30.0),     # 1,373,999,104 B total, 750 MB chunk1 -> 11s data, 11.005s prop end
    ],
)
def test_r2d_sending_partial_chunk_then_propagating(
    hop_lat_s: float, pout: int, pin: int, chunk1_bytes: float, wide_deadline: float
) -> None:
    """R2d counterexample: sending partial chunk must pay its own propagation wait before continuation."""
    cfg = _base_cfg(hop_latency_s=hop_lat_s, kv_gib=2.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(
        env,
        _two_cs_trace(c1_dark=[[0.0, 60.0]]),
        cfg,
        placer=lambda view, req: 1,
        planner=get_planner("none"),
        seed=1,
    )
    req = Request(0, 0, 0.0, False, pin, pout, 1.0, 21.0)
    world.submit(req)
    env.run(until=5.0)
    world.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=chunk1_bytes),)))
    # At t=6.0s: 125 MB sent, chunk1 still sending
    env.run(until=6.0)
    snap = world.local_view.migration_snapshot(0)
    assert snap is not None and snap.phase == "sending"
    assert snap.chunk_sent_bytes == pytest.approx(125_000_000.0, rel=1e-4)

    problem = ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    assert result.action.migrations == ()
    # Earliest physical finish is > 21.0s (22.9152s for 4s prop; 21.00199s for 5ms prop)
    assert result.goodput_req_ids == ()

    # Wide deadline positive case under periodic R3RuntimePlanner control
    cfg_ok = _base_cfg(hop_latency_s=hop_lat_s, kv_gib=2.0)
    cfg_ok["power"]["eclipse_dod_budget_j"] = 0.0
    cfg_ok["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env_ok = simpy.Environment()
    w_ok = CoWorld(
        env_ok,
        _two_cs_trace(c1_dark=[[0.0, 60.0]]),
        cfg_ok,
        placer=get_placer("nearest"),
        planner=get_planner("r3_milp"),
        seed=1,
    )
    r_ok = Request(0, 0, 0.0, False, pin, pout, 1.0, wide_deadline)
    w_ok.submit(r_ok)
    env_ok.run(until=4.5)
    w_ok.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=chunk1_bytes),)))
    env_ok.run(until=wide_deadline)
    assert r_ok.node == 2 and r_ok.t_delivered is not None and r_ok.t_delivered <= wide_deadline


def _three_cs_shortcut_trace(
    *,
    intra_up: list[list[float]] | None = None,
    shortcut_up: list[list[float]] | None = None,
) -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "three-cs-shortcut",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "intra", "up": intra_up if intra_up is not None else [[0.0, 60.0]]},
            {"u": 3, "v": 2, "kind": "intra", "up": intra_up if intra_up is not None else [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": shortcut_up if shortcut_up is not None else [[6.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], [[0.0, 60.0]]],
    })


def test_r2e_newly_issued_long_flow_cannot_reroute() -> None:
    """R2e counterexample: newly issued nbytes=None flow at t=5s on 1->3->2 (5 MB/s) cannot reroute to 1->2 at t=6s."""
    tr = _three_cs_shortcut_trace(intra_up=[[0.0, 60.0]], shortcut_up=[[6.0, 60.0]])
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"] = {"intra": 0.04, "inter": 0.128, "rs": 10.0}  # intra=5 MB/s, inter=16 MB/s
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda view, req: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, False, 25000, 50, 1.0, 20.0)  # 102.4 MB KV, 5s decode
    world.submit(req)
    env.run(until=5.0)
    assert world.local_view.migration_snapshot(0) is None

    problem = ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    # Apply the returned action in CoWorld and verify predicted goodput <= actual on-time IDs
    world.apply_plan(result.action)
    env.run(until=20.0)
    actual_ontime = {0} if (req.t_delivered is not None and req.t_delivered <= 20.0) else set()
    assert set(result.goodput_req_ids).issubset(actual_ontime)


def test_r2e_static_shortcut_wait_and_link_break_lifecycle() -> None:
    """R2e lifecycle: (1) static path wide-deadline positive, (2) wait for shortcut at t=10s, (3) link break -> resumable."""
    # (1) Static path (no shortcut), wide deadline=35.0s -> completes on 1->3->2
    tr_static = _three_cs_shortcut_trace(intra_up=[[0.0, 60.0]], shortcut_up=[])
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"] = {"intra": 0.04, "inter": 0.128, "rs": 10.0}
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)

    env_s = simpy.Environment()
    w_s = CoWorld(env_s, tr_static, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    r_s = Request(0, 0, 0.0, False, 25000, 50, 1.0, 40.0)
    w_s.submit(r_s)
    env_s.run(until=40.0)
    assert r_s.node == 2 and r_s.t_delivered is not None and r_s.t_delivered <= 40.0

    # (2) Shortcut appears at t=10s (slot 1): waiting at t=5s and starting on 1->2 at t=10s
    # finishes migration at 16.4s, next tick t=20s decodes 5s -> delivers at ~25.0s <= 26.0s
    tr_wait = _three_cs_shortcut_trace(intra_up=[[0.0, 60.0]], shortcut_up=[[10.0, 60.0]])
    env_w = simpy.Environment()
    w_w = CoWorld(env_w, tr_wait, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    r_w = Request(0, 0, 0.0, False, 25000, 50, 1.0, 26.0)
    w_w.submit(r_w)
    env_w.run(until=5.0)
    assert r_w.node == 1
    env_w.run(until=26.0)
    assert r_w.node == 2 and r_w.t_delivered is not None and r_w.t_delivered <= 26.0

    # (3) Old path 1->3->2 breaks at t=7s while 1->2 appears at t=10s:
    # At t=5s solver must NOT promise impossible completion via free reroute;
    # in CoWorld, starting at t=5s fails at t=7s -> resumable -> resumes on 1->2 at t=10s.
    tr_brk = _three_cs_shortcut_trace(intra_up=[[0.0, 7.0]], shortcut_up=[[10.0, 60.0]])
    cfg_manual = dict(cfg, control_period_s=100.0)
    env_b = simpy.Environment()
    w_b = CoWorld(env_b, tr_brk, cfg_manual, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    r_b = Request(0, 0, 0.0, False, 25000, 50, 1.0, 22.0)
    w_b.submit(r_b)
    env_b.run(until=5.0)
    prob_b = ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0).build(
        w_b.forecast_view, RebalanceInvariants.from_cfg(cfg_manual)
    )
    res_b = R3Solver().solve(prob_b)
    # If solver issues at t=5s, 1->3->2 breaks at t=7s so it cannot promise goodput at t<=22s
    if res_b.action.migrations:
        assert res_b.goodput_req_ids == ()
    w_b.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))
    env_b.run(until=8.0)
    snap_brk = w_b.local_view.migration_snapshot(0)
    assert snap_brk is not None and snap_brk.phase == "resumable"
    assert snap_brk.received_bytes == pytest.approx(10_000_000.0, rel=1e-4)
    env_b.run(until=10.0)
    w_b.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))
    env_b.run(until=25.0)
    assert r_b.node == 2 and r_b.t_delivered is not None and r_b.t_delivered <= 25.0


@pytest.mark.parametrize(
    ("up_intervals", "deadline_s", "expect_ontime"),
    [
        ([[0.0, 7.0]], 15.0, False),                  # breaks before midpoint (t=7.0 < 7.5)
        ([[0.0, 8.0]], 15.0, False),                  # R2f counterexample: breaks after midpoint (t=8.0 > 7.5)
        ([[0.0, 10.0]], 15.0, True),                  # breaks exactly at slot boundary t=10.0: [5,10) fully up
        ([[0.0, 8.0], [12.0, 60.0]], 15.0, False),    # short break + mid-slot recovery: [5,10) & [10,15) unusable
        ([[12.0, 60.0]], 20.0, False),                # comes up mid-slot at t=12.0: unusable at t=10.0 boundary
        ([[0.0, 60.0]], 15.0, True),                  # continuously up positive case
    ],
)
def test_r2f_intraslot_link_break_and_boundary_conservatism(
    up_intervals: list[list[float]], deadline_s: float, expect_ontime: bool
) -> None:
    """R2f: slot capacity and route lookahead must be conservative across full [t_start, t_end) intervals."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2f-link-intervals",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": up_intervals},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg(hop_latency_s=0.0)
    cfg["control_period_s"] = 100.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"] = {"inter": 0.04, "rs": 10.0}  # 5 MB/s inter-sat
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1)
    req = Request(0, 0, 0.0, False, 4800, 50, 1.0, deadline_s)  # 19.6608 MB KV, 5s decode
    world.submit(req)
    env.run(until=5.0)

    problem = ProblemBuilder(horizon_slots=4, dt_s=5.0, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 5.0, slots=4)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert set(result.goodput_req_ids).issubset(actual_ontime)
    if expect_ontime:
        assert result.goodput_req_ids == (0,)
        assert 0 in actual_ontime
    else:
        assert result.goodput_req_ids == ()



def test_r2f_recovery_and_future_link_up_periodic_positive() -> None:
    """R2f positive: after short break/recovery at t=12s or link-up at t=12s, periodic planner waits for t=15s and delivers."""
    for up_iv in ([[0.0, 8.0], [12.0, 60.0]], [[12.0, 60.0]]):
        tr = Trace({
            "schema": SCHEMA,
            "name": "r2f-recovery-positive",
            "horizon_s": 60.0,
            "nodes": [
                {"id": 0, "role": "rs"},
                {"id": 1, "role": "compute"},
                {"id": 2, "role": "compute"},
            ],
            "edges": [
                {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
                {"u": 1, "v": 2, "kind": "inter", "up": up_iv},
            ],
            "dark": [[], [[0.0, 60.0]], []],
        })
        cfg = _base_cfg(hop_latency_s=0.0)
        cfg["power"]["eclipse_dod_budget_j"] = 0.0
        cfg["net"]["link_gbps"] = {"inter": 0.04, "rs": 10.0}
        cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
        env = simpy.Environment()
        w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
        r = Request(0, 0, 0.0, False, 4800, 50, 1.0, 30.0)
        w.submit(r)
        env.run(until=30.0)
        assert r.node == 2 and r.t_delivered is not None and r.t_delivered <= 30.0


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


@pytest.mark.parametrize(
    ("upload_bytes", "expect_migrant_ontime"),
    [
        (100_000_000.0, False),  # R14 repro: 100MB upload occupies CS1->CS2 across slots 0..2
        (1.0, True),             # 1B upload finishes before t=2s -> migrant delivers on time
    ],
)
def test_r2g_known_upload_across_slots_negative_and_positive(
    upload_bytes: float,
    expect_migrant_ontime: bool,
) -> None:
    """R2g: known upload occupancy persists across forecast slots without false goodput."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-known-upload",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 1, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    first_pred, first_bg = records[0]
    assert set(first_pred).issubset(actual_ontime)
    if expect_migrant_ontime:
        assert first_bg[(1, 2)] == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-3)
        assert 1 in actual_ontime
        assert 1 in first_pred
    else:
        assert first_bg[(1, 2)] == pytest.approx((40e6, 40e6, 40e6, 0.0), rel=1e-5)
        assert first_bg[(2, 1)] == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-3)
        assert 1 not in first_pred


def test_r2g_shared_bottleneck_speedup_and_completion_release() -> None:
    """R2g: when a competing upload finishes mid-slot, surviving upload speeds up then releases."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-shared-speedup",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(rs=0.08, inter=0.08)  # 10 MB/s on both RS and inter
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(
        env,
        tr,
        cfg,
        placer=lambda v, r: {0: 2, 1: 3, 2: 1}[r.i],
        planner=get_planner("r3_milp"),
        seed=1,
    )
    # Req 0 (45MB -> CS2) & Req 1 (15MB -> CS3) share 0->1 (5 MB/s each for 3s -> Req 1 ends at t=3s;
    # Req 0 speeds up to 10 MB/s at t=3s and finishes remaining 30MB at t=6s, releasing CS1->CS2 for slot 1).
    world.submit(Request(0, 0, 0.0, False, 1, 1, 45_000_000.0, 50.0))
    world.submit(Request(1, 0, 0.0, False, 1, 1, 15_000_000.0, 50.0))
    migrant = Request(2, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=2.0)
    prob = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    bg_12 = next(e.bg_bps for e in prob.edges if e.src == 1 and e.dst == 2)
    assert bg_12 is not None
    # At t=2s: rem(0)=35MB, rem(1)=5MB. Slot 0 [2,8): 5MB*1s + 10MB*3s = 35MB -> 46.67 Mbps (>40Mbps initial!),
    # then slot 1+ released (0.0 Mbps).
    assert bg_12[0] == pytest.approx(35_000_000.0 * 8.0 / 6.0, rel=1e-4)
    assert bg_12[1] == pytest.approx(0.0, abs=1e-3)
    env.run(until=25.0)
    assert migrant.node == 2 and migrant.t_delivered is not None and migrant.t_delivered <= 20.0


def test_r2g_link_break_and_recovery_reserves_retry_slot() -> None:
    """R2g: link break aborts upload and recovery re-reserves retry occupancy in later slot."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-link-break-retry",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 5.0], [14.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"].update(rs=10.0, inter=0.04)  # 5 MB/s
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2, planner=get_planner("none"), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 60_000_000.0, 50.0))
    env.run(until=2.0)
    prob = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    bg_12 = next(e.bg_bps for e in prob.edges if e.src == 1 and e.dst == 2)
    assert bg_12 is not None
    # Slot 0 [2,8): active [2,5) -> 20 Mbps; Slot 1 [8,14): RS down -> 0; Slot 2 [14,20): retried 60MB -> 40 Mbps.
    assert bg_12[0] == pytest.approx(20e6, rel=1e-4)
    assert bg_12[1] == pytest.approx(0.0, abs=1e-3)
    assert bg_12[2] == pytest.approx(40e6, rel=1e-4)


def test_r2g_disjoint_path_migration_positive() -> None:
    """R2g positive: long upload on CS1->CS2 does not block migration on disjoint CS3->CS4."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-disjoint-path",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
            {"id": 4, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 0, "v": 3, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
            {"u": 3, "v": 4, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    preds: list[tuple[int, ...]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            if self.inner.last_result is not None:
                preds.append(self.inner.last_result.goodput_req_ids)
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 3, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 100_000_000.0, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=25.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert 1 in actual_ontime
    assert migrant.node == 4
    assert any(1 in p for p in preds)
    assert all(set(p).issubset(actual_ontime) for p in preds)


@pytest.mark.parametrize(
    ("upload_bytes", "hop_latency_s", "deadline_s", "expect_ontime"),
    [
        (30_000_000.0, 1.1, 14.001, False),  # R15 repro: 10MB bg + ~40MB mig + 1.1s prop > 6s slot
        (1.0, 1.1, 14.001, True),            # 1B bg -> unlocks in slot 0, delivers at 14.0s
        (30_000_000.0, 0.5, 14.001, True),   # shorter 0.5s prop: 5.024s + 0.5s <= 6.0s -> delivers on time
        (30_000_000.0, 1.1, 20.5, True),     # wider deadline: unlocks in slot 1, delivers at 20.0s
    ],
)
def test_r2h_background_average_unlock_timing(
    upload_bytes: float,
    hop_latency_s: float,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R2h: tx_frac must account for background bytes + migration bytes over total link capacity."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2h-unlock-timing",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg(hop_latency_s=hop_latency_s)
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.08, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    preds: list[tuple[int, ...]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            if self.inner.last_result is not None:
                preds.append(self.inner.last_result.goodput_req_ids)
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 1, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 9765, 60, 1.0, deadline_s)
    world.submit(migrant)
    env.run(until=30.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert preds
    assert all(set(p).issubset(actual_ontime) for p in preds)
    if expect_ontime:
        assert 1 in actual_ontime
        assert any(1 in p for p in preds)
    else:
        assert 1 not in preds[0]


@pytest.mark.parametrize(
    ("upload_bytes", "expect_ontime"),
    [
        (100_000_000.0, False),  # R15 repro: 0->1->2 breaks at t=3s, retries on 0->3->2
        (1.0, True),             # 1B bg -> delivers at 19.0s
    ],
)
def test_r2i_background_retry_new_path(upload_bytes: float, expect_ontime: bool) -> None:
    """R2i: failed upload re-routes onto new path via ForecastView.path_at and reserves retry occupancy."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2i-retry-new-path",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}] + [{"id": j, "role": "compute"} for j in (1, 2, 3)],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 3.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 3.0]]},
            {"u": 1, "v": 3, "kind": "intra", "up": [[0.0, 60.0]]},
            {"u": 0, "v": 3, "kind": "rs", "up": [[3.0, 60.0]]},
            {"u": 3, "v": 2, "kind": "inter", "up": [[3.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], [[0.0, 60.0]]],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, intra=10.0, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 3, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    assert all(set(p).issubset(actual_ontime) for p, _ in records)
    first_pred, first_bg = records[0]
    if expect_ontime:
        assert 1 in actual_ontime
        assert 1 in first_pred
    else:
        assert first_bg[(3, 2)] == pytest.approx((0.0, 40e6, 40e6, 20e6), rel=1e-4)
        assert 1 not in first_pred


@pytest.mark.parametrize(
    ("upload_bytes", "expect_ontime"),
    [
        (100_000_000.0, False),  # R16 repro: 0->2 breaks at t=3s, placer switches dst 2->3 (0->1->3)
        (1.0, True),             # 1B bg -> finishes before t=2s, migrant delivers at 13.0s
    ],
)
def test_r2j_background_retry_new_destination(upload_bytes: float, expect_ontime: bool) -> None:
    """R2j: failed upload re-entering external placer must conservatively bound candidate destinations."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2j-retry-new-dst",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}] + [{"id": j, "role": "compute"} for j in (1, 2, 3)],
        "edges": [
            {"u": 0, "v": 2, "kind": "inter", "up": [[0.0, 3.0]]},
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []
    placements: list[tuple[float, int, int]] = []

    def placer(view, req):
        dst = (2 if view.connected(req.origin, 2) else 3) if req.i == 0 else 1
        placements.append((view.now, req.i, dst))
        return dst

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=placer, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 14.001)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    assert all(set(p).issubset(actual_ontime) for p, _ in records)
    first_pred, first_bg = records[0]
    if expect_ontime:
        assert 1 in actual_ontime
        assert 1 in first_pred
    else:
        assert (0.0, 0, 2) in placements and (3.0, 0, 3) in placements
        assert first_bg[(1, 3)] == pytest.approx((33.333333e6, 40e6, 40e6, 20e6), rel=1e-4)
        assert 1 not in first_pred


def test_r2j_waiting_upload_before_placement_reserves_occupancy() -> None:
    """R2j: arrived request in state=='wait' waiting for link recovery still reserves future occupancy."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2j-waiting-upload",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[6.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2, planner=get_planner("none"), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 60_000_000.0, 50.0))
    env.run(until=2.0)
    prob = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    bg_12 = next(e.bg_bps for e in prob.edges if e.src == 1 and e.dst == 2)
    assert bg_12 is not None
    # Wait request starts when RS 0->1 comes up at t=6.0 (slot 0 [2,8): 2s -> 13.33Mbps; slot 1 [8,14): 40Mbps).
    assert bg_12[0] == pytest.approx(10_000_000.0 * 8.0 / 6.0, rel=1e-4)
    assert bg_12[1] == pytest.approx(40e6, rel=1e-4)
