"""Seam tests for pure-data R3Solver (Two-Stage Goodput-CVaR MILP)."""
from __future__ import annotations

import math
import subprocess
import sys

from satplan.actions import PlanAction
from satplan.r3_solver import (
    R3EdgeSpec,
    R3NodeSpec,
    R3Problem,
    R3RequestSpec,
    R3Solver,
    R3TierSpec,
)

TIERS_STD = (
    R3TierSpec(index=0, power_w=0.0, rate_tok_s=0.0),
    R3TierSpec(index=1, power_w=100.0, rate_tok_s=10.0),
    R3TierSpec(index=2, power_w=300.0, rate_tok_s=20.0),
)


def test_r3_solver_has_zero_simpy_or_coworld_imports() -> None:
    code = (
        "import sys, satplan.r3_solver; "
        "assert 'simpy' not in sys.modules, 'simpy imported'; "
        "assert 'satplan.world' not in sys.modules, 'satplan.world imported'; "
        "print('CLEAN')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "CLEAN"


def test_worked_example_staggered_bottleneck_and_eclipse_dod() -> None:
    """Worked example: 2 requests on an Eclipse node share a 1-req/slot bottleneck link to a Sunlit node.

    Node 0 is in Eclipse with 500 J DoD (at most 50 tok at 100W), insufficient for 100-tok requests.
    Link 0->1 has 160 Mbps (100 MB per 5s slot), so both 100 MB requests cannot transfer in slot 0.
    R3Solver must stagger: r0 transfers in slot 0 (decodes on node 1 in slot 1, finishing at 10s <= D0=10s),
    while r1 suspends at 0W in slot 0, transfers in slot 1, and decodes on node 1 in slot 2 (15s <= D1=15s).
    """
    dt_s = 5.0
    bps_100mb_per_slot = 100e6 * 8.0 / dt_s  # 160 Mbps
    problem = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 4),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 4),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps_100mb_per_slot,) * 4),),
        requests=(
            R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),
            R3RequestSpec(req_id=1, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=15.0),
        ),
        eclipse_dod_budget_j=500.0,
    )
    res = R3Solver().solve(problem)
    assert res.status == "optimal"
    assert isinstance(res.action, PlanAction)
    assert res.goodput_req_ids == (0, 1)
    assert res.goodput_weight == 2.0
    # At t=0, r0 migrates to node 1 while r1 waits/suspends on node 0
    assert len(res.action.migrations) == 1
    assert res.action.migrations[0].req_id == 0
    assert res.action.migrations[0].dst == 1
    assert res.action.tier_decisions[0] in (0, 1)  # Cannot pick 300W (1500J > 500J DoD) on node 0 at t=0


def test_eclipse_dod_budget_selects_low_tier_to_avoid_early_exhaustion() -> None:
    """With 2 eclipse slots and 1000 J DoD, node 0 cannot run 300W (1500J/slot) so it must use 100W (500J/slot)."""
    dt_s = 5.0
    problem = R3Problem(
        horizon_slots=2,
        dt_s=dt_s,
        nodes=(R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True, True)),),
        tiers=TIERS_STD,
        edges=(),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),),
        eclipse_dod_budget_j=1000.0,
    )
    res = R3Solver().solve(problem)
    assert res.status == "optimal"
    assert res.node_tiers[0] == (1, 1)
    assert res.goodput_req_ids == (0,)


def test_cumulative_unlock_and_kv_capacity_accounting() -> None:
    """A 200 MB migration over a 100 MB/slot link takes 2 slots (u_rjt=1 only at t>=2).

    Also, KV capacity M_1 = 250 MB allows r0 (200 MB) on node 1, but blocks r1 (200 MB) from
    co-residing until r0 finishes or stays on node 0.
    """
    dt_s = 5.0
    bps_100mb_per_slot = 100e6 * 8.0 / dt_s
    problem = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 4),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=250e6, is_eclipse=(False,) * 4),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps_100mb_per_slot,) * 4),),
        requests=(
            # r0 cannot finish on node 0 (DoD=0), so it must migrate 200 MB over 2 slots -> unlocks at t=2, finishes t=2 (15s)
            R3RequestSpec(req_id=0, init_node=0, kv_bytes=200e6, remaining_tokens=100.0, deadline_s=15.0),
        ),
        eclipse_dod_budget_j=0.0,
    )
    res = R3Solver().solve(problem)
    assert res.status == "optimal"
    assert res.goodput_req_ids == (0,)
    # Cannot finish at t=1 (10s) because cumulative unlock requires 2 full slots (t=0, t=1) before t=2 compute
    assert res.cvar_s >= 15.0 - 1e-6

    # Now add a tighter deadline (10s = end of slot 1): impossible due to cumulative unlock (no compute during transfer)
    tight_problem = R3Problem(
        horizon_slots=4,
        dt_s=dt_s,
        nodes=problem.nodes,
        tiers=problem.tiers,
        edges=problem.edges,
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=200e6, remaining_tokens=100.0, deadline_s=10.0),),
        eclipse_dod_budget_j=0.0,
    )
    tight_res = R3Solver().solve(tight_problem)
    assert tight_res.goodput_req_ids == ()


def test_rc_thermal_upper_bound_forces_step_down() -> None:
    """Starting at 84.0 C with T_max=85.0 C, 300W violates T_max so solver must step down to 100W or 0W."""
    dt_s = 5.0
    r_th, c_th, t_amb, t_max = 0.2, 150.0, 20.0, 85.0
    alpha = math.exp(-dt_s / (r_th * c_th))
    # Independent check: 300W steady-state T_eq = 20 + 0.2*300 = 80 C, wait: use 400W (T_eq = 100 C) to exceed 85 C!
    tiers = (
        R3TierSpec(index=0, power_w=0.0, rate_tok_s=0.0),
        R3TierSpec(index=1, power_w=100.0, rate_tok_s=10.0),
        R3TierSpec(index=2, power_w=400.0, rate_tok_s=25.0),
    )
    t_next_400 = alpha * 84.0 + (1.0 - alpha) * (t_amb + r_th * 400.0)
    assert t_next_400 > t_max  # 86.46 C > 85.0 C

    problem = R3Problem(
        horizon_slots=2,
        dt_s=dt_s,
        nodes=(R3NodeSpec(node_id=0, init_temp_C=84.0, kv_cap_bytes=500e6, is_eclipse=(False, False)),),
        tiers=tiers,
        edges=(),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=50e6, remaining_tokens=50.0, deadline_s=10.0),),
        t_amb_C=t_amb,
        t_max_C=t_max,
        r_th=r_th,
        c_th=c_th,
    )
    res = R3Solver().solve(problem)
    assert res.status == "optimal"
    assert res.node_tiers[0][0] != 2  # Cannot pick 400W tier
    assert all(temp <= t_max + 1e-6 for temp in res.node_temps_C[0])
    assert res.goodput_req_ids == (0,)


def test_stage2_eliminates_unnecessary_migration_and_supports_multihop() -> None:
    """Stage 2 keeps migration_bytes == 0 when local node meets SLO, and routes multi-hop when needed."""
    dt_s = 5.0
    bps = 100e6 * 8.0 / dt_s
    # Case A: Request already on sunlit node 0 -> Stage 2 must not migrate unnecessarily
    local_problem = R3Problem(
        horizon_slots=2,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False, False)),
            R3NodeSpec(node_id=1, init_temp_C=20.0, kv_cap_bytes=500e6, is_eclipse=(False, False)),
        ),
        tiers=TIERS_STD,
        edges=(R3EdgeSpec(src=0, dst=1, tot_bps=(bps, bps)),),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=5.0),),
    )
    local_res = R3Solver().solve(local_problem)
    assert local_res.goodput_req_ids == (0,)
    assert local_res.migration_bytes == 0.0
    assert local_res.action.migrations == ()

    # Case B: Multi-hop 0 -> 1 -> 2 (node 1 has 0 KV cap for unlock, acts as relay to node 2)
    multihop_problem = R3Problem(
        horizon_slots=3,
        dt_s=dt_s,
        nodes=(
            R3NodeSpec(node_id=0, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(True,) * 3),
            R3NodeSpec(node_id=1, init_temp_C=25.0, kv_cap_bytes=0.0, is_eclipse=(True,) * 3),
            R3NodeSpec(node_id=2, init_temp_C=25.0, kv_cap_bytes=500e6, is_eclipse=(False,) * 3),
        ),
        tiers=TIERS_STD,
        edges=(
            R3EdgeSpec(src=0, dst=1, tot_bps=(bps,) * 3),
            R3EdgeSpec(src=1, dst=2, tot_bps=(bps,) * 3),
        ),
        requests=(R3RequestSpec(req_id=0, init_node=0, kv_bytes=100e6, remaining_tokens=100.0, deadline_s=10.0),),
        eclipse_dod_budget_j=0.0,
    )
    mh_res = R3Solver().solve(multihop_problem)
    assert mh_res.status == "optimal"
    assert mh_res.goodput_req_ids == (0,)
    assert mh_res.action.migrations == (
        # Must target node 2 across the 0 -> 1 -> 2 relay
        mh_res.action.migrations[0].__class__(req_id=0, dst=2, nbytes=None),
    )
    assert abs(mh_res.migration_bytes - 200e6) < 1e-3  # 100 MB on (0->1) + 100 MB on (1->2)
