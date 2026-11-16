"""Seam tests for CoWorld actuation: 0W suspend, Eclipse DoD Store-and-Run, and Cumulative Unlock."""
from __future__ import annotations

import simpy
from satplan.placers import nearest
from satplan.planners import NonePlanner
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, MigrationOrder, PlanAction, Request


def _two_compute_trace(*, eclipse: list[list[float]] | None = None) -> Trace:
    dark = [[], eclipse if eclipse is not None else [], []]
    return Trace({
        "schema": SCHEMA,
        "name": "actuation-synth",
        "horizon_s": 200.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
            {"id": 2, "name": "C-2", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 200.0]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0.0, 200.0]]},
        ],
        "dark": dark,
    })


def _actuation_cfg(*, t0: float = 60.0, dod_j: float | None = None, kv_gib: float = 1.0) -> dict:
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
        "name": "act_test",
        "delivery": "final",
        "control_period_s": 2.0,
        "frame_s": 10.0,
        "model": {"layers": 2, "kv_heads": 2, "head_dim": 64, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": kv_gib,
            "t_fixed_s": 0.002,
            "t_weights_s": 0.005,
            "t_kv_tok_s": 1.0e-7,
            "t_seq_s": 5.0e-5,
            "mem_freq_exp": 0.6,
        },
        "power": power,
        "thermal": {"T0_C": t0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.2, "C_th": 100.0},
        "net": {"link_gbps": {"intra": 0.008, "inter": 0.008, "rs": 5.0}, "hop_latency_s": 0.0},
        "workload": {"arrival_window_s": 100.0},
        "slo": {"max_stall_frac": 0.1},
    }


def test_suspend_holds_kv_and_cools_at_zero_watts() -> None:
    tr = _two_compute_trace()
    cfg = _actuation_cfg(t0=60.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=False, pin=100, pout=2000, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    env.run(until=2.0)
    assert r.state == "decode"
    kv_before = w.kv_used(1)
    assert kv_before > 0.0

    # Suspend request 0 -> KV remains reserved, decode pauses, node cools at 0W.
    w.apply_plan(PlanAction(pause_ids=(0,)))
    env.run(until=4.0)
    gen_at_pause = r.gen
    t_at_pause = w.temp_C[1]
    env.run(until=12.0)
    assert r.gen == gen_at_pause
    assert w.kv_used(1) == kv_before
    assert w.temp_C[1] < t_at_pause

    # Resume request 0 -> finishes decoding and delivers with 0 drops.
    w.apply_plan(PlanAction(resume_ids=(0,)))
    env.run(until=200.0)
    assert r.state == "done"
    assert r.t_delivered is not None


def test_eclipse_dod_exhaustion_suspends_and_sunlit_resumes() -> None:
    # Eclipse window [10, 50]; DoD budget = 600 J (2s at 300W).
    tr = _two_compute_trace(eclipse=[[10.0, 50.0]])
    cfg = _actuation_cfg(t0=25.0, dod_j=600.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    r = Request(i=0, origin=0, t=10.0, long=False, pin=100, pout=10000, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    # Run through t=20 (well past DoD exhaustion at t=12) and record progress.
    env.run(until=20.0)
    gen_exhausted = r.gen
    kv_during_halt = w.kv_used(1)
    assert kv_during_halt > 0.0

    # Still in eclipse at t=40 -> Store-and-Run: zero progress, zero drop.
    env.run(until=40.0)
    assert r.gen == gen_exhausted
    assert r.state == "decode"

    # Sunlit restored at t=50 -> resumes and finishes cleanly.
    env.run(until=200.0)
    assert r.state == "done"
    assert r.t_delivered is not None


def test_eclipse_dod_exact_budget_no_overshoot() -> None:
    # Eclipse window [0, 50]; DoD budget = 600 J (exactly 2.0s at 300W).
    tr = _two_compute_trace(eclipse=[[0.0, 50.0]])
    cfg = _actuation_cfg(t0=25.0, dod_j=600.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=False, pin=100, pout=20000, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    env.run(until=2.01)
    gen_at_exhaust = r.gen
    energy_at_exhaust = w.pt.energy_j
    assert energy_at_exhaust <= 600.0 + 1e-6

    # Must NOT decode for an extra control period after t=2.0 exhaustion.
    env.run(until=10.0)
    assert r.gen == gen_at_exhaust
    assert w.pt.energy_j == energy_at_exhaust


def test_sunlit_to_eclipse_boundary_zero_dod_leakage() -> None:
    # Sunlit [0, 10), Eclipse [10, 50]; control_period_s = 2.0 aligns with t=10.0.
    tr = _two_compute_trace(eclipse=[[10.0, 50.0]])
    cfg = _actuation_cfg(t0=25.0, dod_j=600.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=False, pin=100, pout=50000, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    # Just after entering eclipse at t=10.001, [8, 10] sunlit energy must NOT leak into eclipse DoD.
    env.run(until=10.001)
    assert w.pt.dod_remaining_j[1] >= 600.0 - 1e-6


def test_migration_unreachable_path_does_not_lock_or_leak() -> None:
    # Edge 1->2 is down during [0, 50].
    tr = Trace({
        "schema": SCHEMA,
        "name": "mig-down",
        "horizon_s": 200.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
            {"id": 2, "name": "C-2", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 200.0]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[50.0, 200.0]]},
        ],
        "dark": [[], [], []],
    })
    cfg = _actuation_cfg(t0=25.0, kv_gib=1.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=False, pin=100, pout=1000, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    env.run(until=0.5)
    w.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))
    assert w.kv_used(2) == 0.0
    assert r.mig_dst is None

    # Request continues decoding on node 1 and completes normally.
    env.run(until=200.0)
    assert r.state == "done"
    assert r.node == 1
    assert w.kv_used(2) == 0.0


def test_migration_source_completes_mid_flight_releases_dst_kv() -> None:
    tr = _two_compute_trace()
    cfg = _actuation_cfg(t0=25.0, kv_gib=1.0)
    cfg["net"]["hop_latency_s"] = 0.5
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=False, pin=500, pout=500, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    env.run(until=0.6)
    assert r.state == "decode"
    # Start migration, then simulate source finishing/delivering before migration finishes.
    w.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))
    assert w.kv_used(2) > 0.0
    r.gen = float(r.pout)
    r.decode_done.succeed()

    env.run(until=200.0)
    assert r.state == "done"
    assert w.kv_used(1) == 0.0
    assert w.kv_used(2) == 0.0


def test_cumulative_unlock_multi_slot_migration_and_kv_accounting() -> None:
    tr = _two_compute_trace()
    # Link intra = 0.008 Gbps = 1 MB/s = 1e6 B/s.
    cfg = _actuation_cfg(t0=25.0, kv_gib=1.0)
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=nearest, planner=NonePlanner(), seed=1)
    # pin=500, pout=500 -> bpt = 2*2*2*64*2 = 1024 B/tok -> reserved = 1000 * 1024 = 1,024,000 B.
    r = Request(i=0, origin=0, t=0.0, long=False, pin=500, pout=500, upload_bytes=1e4, deadline=200.0)
    w.submit(r)

    env.run(until=0.6)
    assert r.state == "decode"
    assert r.node == 1
    need = r.reserved_tokens * w.bpt
    assert w.kv_used(1) == need
    assert w.kv_used(2) == 0.0

    # Pause on node 1, then send HALF of S_r^kv in slot 1.
    target_kv = r.kv * w.bpt
    half = target_kv * 0.5
    w.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=half),)))

    # During/after first half: KV is accounted on BOTH src (1) and dst (2), and r stays on node 1 (locked).
    env.run(until=2.0)
    assert r.node == 1
    assert w.kv_used(1) == need
    assert w.kv_used(2) == need
    # Attempting to resume on dst before cumulative unlock must not run on dst.
    w.apply_plan(PlanAction(resume_ids=(0,)))
    assert r.node == 1

    # Send remaining bytes in slot 2 -> unlocks on node 2, frees node 1 without double-freeing node 2.
    w.apply_plan(PlanAction(migrations=(MigrationOrder(req_id=0, dst=2, nbytes=None),)))
    env.run(until=200.0)
    assert r.node == 2
    assert r.trail == [1, 2]
    assert r.state == "done"
    assert w.kv_used(1) == 0.0
    assert w.kv_used(2) == 0.0
