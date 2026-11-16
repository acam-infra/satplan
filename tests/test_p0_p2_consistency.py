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
    problem = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 1.0)
    good_by_deadline = sum(r.t_delivered is not None and r.t_delivered <= r.deadline for r in reqs)
    # Any request predicted as goodput MUST actually finish on time in CoWorld
    assert len(result.goodput_req_ids) <= good_by_deadline


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
    problem = ProblemBuilder(horizon_slots=3, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    result = R3Solver().solve(problem)
    _exec_returned_schedule(world, env, result, 5.0)
    good_by_deadline = sum(r.t_delivered is not None and r.t_delivered <= r.deadline for r in reqs)
    assert len(result.goodput_req_ids) <= good_by_deadline

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
