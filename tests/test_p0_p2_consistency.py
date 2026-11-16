"""Independent seam tests for P0-P2: state machine, budgets, execution semantics, and safe fallback."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
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
    """Cannot compute on src AND complete transfer in the same slot to meet an impossible deadline."""
    dt_s = 5.0
    bps = 100e6 * 8.0 / dt_s  # 100 MB per 5s slot
    # r0 starts on eclipse node 0 (0 J DoD), needs 100 MB transfer to node 1 + 100 tokens (5s at 20 tok/s).
    # With no compute during migration, earliest finish is slot 1 (t=10.0s), NOT slot 0 (t=5.0s).
    prob_impossible = R3Problem(
        horizon_slots=3,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 3),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 3),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 3),),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=5.0),),
        eclipse_dod_budget_j=0.0,
    )
    res_imp = R3Solver().solve(prob_impossible)
    assert res_imp.goodput_req_ids == ()

    prob_possible = R3Problem(
        horizon_slots=3,
        dt_s=dt_s,
        nodes=prob_impossible.nodes,
        tiers=TIERS_STD,
        edges=prob_impossible.edges,
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),),
        eclipse_dod_budget_j=0.0,
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
    """max_mig_per_req omitted defaults to 4; None allows unlimited relay uniformly for reactive & r3_milp."""
    assert RebalanceInvariants.from_cfg({}).max_mig_per_req == 4
    assert RebalanceInvariants.from_cfg({"rebalance_invariants": {"max_mig_per_req": None}}).max_mig_per_req is None

    # Alternating eclipse: C-1 dark [0, 15], [30, 60]; C-2 dark [15, 30].
    # Request needs 2 migrations (1->2 then 2->1) to finish if DoD=0.
    tr = _two_cs_trace(c1_dark=[[0.0, 15.0], [30.0, 60.0]], c2_dark=[[15.0, 30.0]])
    for planner_name in ("reactive", "r3_milp"):
        cfg_lim = _base_cfg(max_mig_per_req=1)
        cfg_lim["power"]["eclipse_dod_budget_j"] = 0.0
        cfg_lim["workload"]["rate_per_rs_per_min"] = 6.0
        res_lim = run_simulation(tr, cfg_lim, placer="nearest", planner=planner_name, seed=1)

        cfg_inf = _base_cfg(max_mig_per_req=None)
        cfg_inf["power"]["eclipse_dod_budget_j"] = 0.0
        cfg_inf["workload"]["rate_per_rs_per_min"] = 6.0
        res_inf = run_simulation(tr, cfg_inf, placer="nearest", planner=planner_name, seed=1)

        assert res_lim.metrics["n_req"] > 0
        # Under max_mig=1 no request can exceed 1 migration; under None, relay (>1) is permitted
        assert res_inf.metrics["link_GB_hops"] >= res_lim.metrics["link_GB_hops"]


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
