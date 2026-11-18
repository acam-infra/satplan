"""Shared worlds, traces, planners, and accounting for LAB-2084."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import random
import sys
from pathlib import Path
from typing import Any

import simpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Load LAB-2082 common module cleanly
_LAB2082_COMMON = REPO_ROOT / "experiments" / "LAB-2082" / "common.py"
_spec = importlib.util.spec_from_file_location("lab2082_common", _LAB2082_COMMON)
assert _spec and _spec.loader
_c82 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c82)

CausalReactivePlanner = _c82.CausalReactivePlanner
ContentionFreeDiagnosticWorld = _c82.ContentionFreeDiagnosticWorld
DiagnosticWorld = _c82.DiagnosticWorld
digest = _c82.digest
extended_metrics = _c82.extended_metrics

from experiments.lab2046_common import ReplayPlacer
from satplan.metrics import Recorder
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, NonePlanner, RebalanceInvariants
from satplan.trace import SCHEMA, Trace
from satplan.world import Request


def make_base_cfg(spec: dict[str, Any], *, max_inflight: int | None = None) -> dict[str, Any]:
    reb = spec["rebalance"]
    return {
        "name": "lab2084-demand-validation",
        "delivery": "final",
        "rs_downlink": False,
        "horizon_s": spec["horizon_s"],
        "control_period_s": 2.0,
        "frame_s": 5.0,
        "rebalance_invariants": {
            "interval_s": reb["interval_s"],
            "max_inflight": max_inflight if max_inflight is not None else reb["max_inflight"],
            "max_mig_per_req": reb["max_mig_per_req"],
        },
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 24.0,
            "t_fixed_s": 0.003,
            "t_weights_s": 0.01,
            "t_kv_tok_s": 1e-07,
            "t_seq_s": 5e-05,
            "mem_freq_exp": 0.6,
        },
        "power": {
            "P_idle": 60,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 80,
            "eclipse_dod_budget_j": 1000.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
                {"name": "low_20w", "power_w": 20.0, "clock_f": 0.35},
                {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
            ],
        },
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.25, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 0.6, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {"arrival_window_s": spec["goodput_window_s"]},
        "slo": {"max_stall_frac": 0.1},
    }


def make_trace(
    nodes: int,
    *,
    staggered: bool = True,
    horizon_s: float = 320.0,
    cut_inter_group_at_s: float | None = None,
) -> Trace:
    half = nodes // 2
    edges = [{"u": 0, "v": j, "kind": "rs", "up": [[0.0, horizon_s]]} for j in range(1, nodes + 1)]
    for u, v in itertools.combinations(range(1, nodes + 1), 2):
        cross_group = (u <= half < v) or (v <= half < u)
        if cut_inter_group_at_s is not None and cross_group:
            up = [[0.0, cut_inter_group_at_s]]
        else:
            up = [[0.0, horizon_s]]
        edges.append({"u": u, "v": v, "kind": "inter", "up": up})

    if staggered:
        dark = [[]] + [[[85.0, 210.0]]] * half + [[[0.0, 90.0], [220.0, horizon_s]]] * half
    else:
        dark = [[]] + [[[85.0, 210.0]]] * nodes
    return Trace({
        "schema": SCHEMA,
        "name": f"lab2084-{nodes}sat-{'staggered' if staggered else 'sync'}",
        "horizon_s": float(horizon_s),
        "nodes": [{"id": 0, "name": "RS-0", "role": "rs"}]
        + [{"id": j, "name": f"CS-{j}", "role": "compute"} for j in range(1, nodes + 1)],
        "edges": edges,
        "dark": dark,
    })


def make_dataset(
    seed: int,
    n_short: int,
    n_long: int,
    *,
    length_ratio: float = 1.0,
    dl_long_off_s: float = 195.0,
    dl_short_off_s: float = 45.0,
    post_start_k: int = 0,
) -> tuple[list[Request], dict[str, Any]]:
    rng = random.Random(seed)
    items: list[tuple[float, bool, int, int, int, float, float]] = []
    for _ in range(n_short):
        t = round(rng.uniform(0.0, 20.0), 2)
        actual_pout = rng.randint(800, 1200)
        pred_pout = max(1, round(actual_pout / length_ratio))
        items.append((t, False, 200, actual_pout, pred_pout, 10.0, t + dl_short_off_s))
    for _ in range(n_long):
        t = round(rng.uniform(0.0, 20.0), 2)
        actual_pout = rng.randint(9500, 10800)
        pred_pout = max(1, round(actual_pout / length_ratio))
        items.append((t, True, 500, actual_pout, pred_pout, 30.0, t + dl_long_off_s))
    items.sort(key=lambda x: x[0])
    reqs = [
        Request(
            i=idx,
            origin=0,
            t=t,
            long=is_long,
            pin=pin,
            pout=pout,
            upload_bytes=up_mb * 1e6,
            deadline=dl,
            output_budget=pred_pout,
            predicted_pout=pred_pout,
        )
        for idx, (t, is_long, pin, pout, pred_pout, up_mb, dl) in enumerate(items)
    ]
    modified_ids: list[int] = []
    if post_start_k > 0:
        short_idxs = [r.i for r in reqs if not r.long][-post_start_k:]
        for off, idx in enumerate(short_idxs):
            reqs[idx].t = 11.0 + 0.5 * off
            reqs[idx].deadline = reqs[idx].t + dl_short_off_s
            modified_ids.append(idx)
    pert_info = {
        "length_ratio": length_ratio,
        "underestimated_req_ids": [r.i for r in reqs if r.predicted_pout != r.pout],
        "post_start_req_ids": modified_ids,
    }
    return reqs, pert_info


def compute_frozen_initial_targets(
    cfg: dict[str, Any],
    trace: Trace,
    reqs: list[Request],
    *,
    placer_name: str = "orbit_thermal",
    force_src_ids: set[int] | None = None,
    force_src_node: int = 1,
) -> dict[int, int]:
    env = simpy.Environment()
    inv = RebalanceInvariants.from_cfg(cfg)
    placer = get_placer(placer_name)
    w = DiagnosticWorld(env, trace, copy.deepcopy(cfg), policy=CompositePolicy(placer, NonePlanner(), inv), seed=1)
    for r in reqs:
        w.submit(copy.deepcopy(r))
    env.run(until=cfg["horizon_s"])
    targets = {r.i: r.trail[0] for r in w.reqs if r.trail}
    if force_src_ids:
        for rid in force_src_ids:
            targets[rid] = force_src_node
    return targets


def _classify_static_miss(r: Request, horizon_s: float) -> str:
    if r.t_delivered is None:
        return "unfinished_horizon"
    if r.t_delivered > r.deadline + 1e-9:
        return "deadline_miss_late_finish"
    return "on_time"


def run_single_arm(
    arm: str,
    cfg: dict[str, Any],
    trace: Trace,
    reqs: list[Request],
    targets: dict[int, int],
    seed: int,
    *,
    max_orders_per_round: int = 4,
    include_events: bool = False,
) -> dict[str, Any]:
    env = simpy.Environment()
    rec = Recorder(float(cfg.get("frame_s", 5.0)))
    inv = RebalanceInvariants.from_cfg(cfg)
    if arm == "Static":
        world_cls = DiagnosticWorld
        planner = NonePlanner()
    elif arm == "Dyn-CF":
        world_cls = ContentionFreeDiagnosticWorld
        planner = CausalReactivePlanner(mode="naive", max_orders_per_round=max_orders_per_round)
    else:
        raise ValueError(f"Unsupported arm in LAB-2084: {arm}")

    w = world_cls(
        env,
        trace,
        copy.deepcopy(cfg),
        policy=CompositePolicy(ReplayPlacer(targets), planner, inv),
        recorder=rec,
        seed=seed,
        allow_queue_reroute=False,
    )
    rec.attach(w)
    for r in reqs:
        w.submit(copy.deepcopy(r))
    env.run(until=cfg["horizon_s"])

    m = extended_metrics(w)
    delivered_toks = sum(r.pout for r in w.reqs if r.t_delivered is not None)
    m["throughput_tok_s_horizon"] = delivered_toks / float(cfg["horizon_s"])
    m["actual_admitted"] = sum(1 for r in w.reqs if r.t_admit is not None)

    miss_reasons: dict[str, int] = {}
    for r in w.reqs:
        if r.long and not w.good(r):
            reason = _classify_static_miss(r, float(cfg["horizon_s"]))
            miss_reasons[reason] = miss_reasons.get(reason, 0) + 1

    trig_counts = {"eclipse": 0, "k1_overload": 0}
    feas_dist: list[int] = []
    reject_totals: dict[str, int] = {}
    decision_log: list[dict[str, Any]] = []
    if hasattr(w.policy.planner, "decision_log"):
        decision_log = w.policy.planner.decision_log
        for d in decision_log:
            trig_counts[d["trigger"]] = trig_counts.get(d["trigger"], 0) + 1
            feas_dist.append(d["feasible_targets"])
            for k, v in d["reject_reasons"].items():
                reject_totals[k] = reject_totals.get(k, 0) + v

    req_rows = [
        {
            "id": r.i,
            "long": r.long,
            "t": r.t,
            "pin": r.pin,
            "pout": r.pout,
            "pred_pout": r.predicted_pout,
            "deadline": r.deadline,
            "t_admit": r.t_admit,
            "t_decoded": r.t_decoded,
            "t_delivered": r.t_delivered,
            "good": w.good(r),
            "n_mig": r.n_mig,
            "trail": r.trail,
            "state": r.state,
        }
        for r in w.reqs
    ]
    out: dict[str, Any] = {
        "arm": arm,
        "metrics": m,
        "miss_reasons": miss_reasons,
        "triggers": trig_counts,
        "feasible_targets_mean": (sum(feas_dist) / len(feas_dist)) if feas_dist else 0.0,
        "reject_totals": reject_totals,
        "requests": req_rows,
    }
    if include_events:
        out["events"] = rec.events
        out["decision_log"] = decision_log
    return out
