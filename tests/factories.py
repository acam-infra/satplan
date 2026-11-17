"""Shared pure test builders and schedule helpers; fixtures live in conftest.py."""
from __future__ import annotations

import simpy
from satplan.actions import PlanAction
from satplan.r3_solver import (
    R3TierSpec,
)
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld

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



def _exec_returned_schedule(world: CoWorld, env: simpy.Environment, result, start: float, dt: float = 5.0, slots: int = 2) -> None:
    world.apply_plan(result.action)
    for slot in range(1, slots):
        env.run(until=start + slot * dt)
        world.apply_plan(PlanAction(tier_decisions={j: tiers[slot] for j, tiers in result.node_tiers.items()}))
    env.run(until=start + slots * dt + 1e-6)



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

