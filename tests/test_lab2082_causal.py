"""Regression tests for causal length prediction (predicted_pout vs actual pout)."""
from __future__ import annotations

import copy
import simpy
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


def _two_star_trace(horizon: float = 200.0) -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "causal-2star",
        "horizon_s": horizon,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1", "role": "compute"},
            {"id": 2, "name": "CS-2", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, horizon]]},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0.0, horizon]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, horizon]]},
        ],
        "dark": [[], [[30.0, 120.0]], []],
    })


def _base_cfg() -> dict:
    return {
        "delivery": "final",
        "rs_downlink": False,
        "control_period_s": 2.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4},
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7, "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.25, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 0.6, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {"arrival_window_s": 25.0},
        "slo": {"max_stall_frac": 0.1},
    }


def test_orbit_thermal_uses_predicted_pout_not_actual() -> None:
    """Changing actual pout while holding predicted_pout & output_budget fixed must not alter initial placement."""
    tr = _two_star_trace()
    cfg = _base_cfg()
    choices = []
    for actual_pout in (2000, 4000, 8000):
        env = simpy.Environment()
        policy = CompositePolicy(get_placer("orbit_thermal"), get_planner("none", cfg), RebalanceInvariants.from_cfg(cfg))
        w = CoWorld(env, tr, copy.deepcopy(cfg), policy=policy, seed=1)
        r = Request(
            i=0, origin=0, t=0.0, long=True, pin=500,
            pout=actual_pout, upload_bytes=10e6, deadline=180.0,
            output_budget=2000, predicted_pout=2000,
        )
        w.submit(r)
        env.run(until=20.0)
        choices.append((r.trail[0], w.observed_arrivals()[0].tokens))
    assert len(set(choices)) == 1, f"Placement or observed arrival leaked actual pout: {choices}"
    assert choices[0][1] == 2000.0
