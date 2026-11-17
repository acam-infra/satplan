"""P0-P2 migration invariants; moved without changing test bodies."""
from __future__ import annotations

import pytest
import simpy
from factories import TIERS_STD, _base_cfg, _exec_returned_schedule, _three_cs_shortcut_trace, _two_cs_trace
from satplan.actions import MigrationOrder, PlanAction
from satplan.placers import get_placer
from satplan.planners import RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder
from satplan.r3_solver import (
    R3EdgeSpec,
    R3NodeSpec,
    R3Problem,
    R3RequestSpec,
    R3Solver,
)
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


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

