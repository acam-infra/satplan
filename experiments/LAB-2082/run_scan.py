"""LAB-2082 parameter scan & paired validation runner (L0, L1 K1xK2/K3/Oracle, L2 5-seed + 24-sat)."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import simpy

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from common import (
    CausalReactivePlanner,
    ContentionFreeDiagnosticWorld,
    DiagnosticWorld,
    QueueReroutePlanner,
    ZeroCostDiagnosticWorld,
    digest,
    extended_metrics,
)
from experiments.lab2046_common import ReplayPlacer, git_head_sha
from satplan.metrics import Recorder
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, NonePlanner, RebalanceInvariants
from satplan.trace import SCHEMA, Trace
from satplan.world import Request

SPEC = json.loads((HERE / "spec.json").read_text())


def make_base_cfg(*, max_inflight: int = 24, bytes_per_elem: int = 2) -> dict[str, Any]:
    return {
        "name": "lab2082-scan",
        "delivery": "final",
        "rs_downlink": False,
        "horizon_s": SPEC["horizon_s"],
        "control_period_s": 2.0,
        "frame_s": 5.0,
        "rebalance_invariants": {
            "interval_s": SPEC["rebalance"]["interval_s"],
            "max_inflight": max_inflight,
            "max_mig_per_req": SPEC["rebalance"]["max_mig_per_req"],
        },
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": bytes_per_elem},
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
        "workload": {"arrival_window_s": SPEC["goodput_window_s"]},
        "slo": {"max_stall_frac": 0.1},
    }


def make_trace(nodes: int, *, staggered: bool = True, horizon_s: float = 320.0) -> Trace:
    half = nodes // 2
    edges = [{"u": 0, "v": j, "kind": "rs", "up": [[0.0, horizon_s]]} for j in range(1, nodes + 1)]
    edges.extend(
        {"u": u, "v": v, "kind": "inter", "up": [[0.0, horizon_s]]}
        for u, v in itertools.combinations(range(1, nodes + 1), 2)
    )
    if staggered:
        dark = [[]] + [[[85.0, 210.0]]] * half + [[[0.0, 90.0], [220.0, horizon_s]]] * half
    else:
        dark = [[]] + [[[85.0, 210.0]]] * nodes
    return Trace({
        "schema": SCHEMA,
        "name": f"lab2082-{nodes}sat-{'staggered' if staggered else 'sync'}",
        "horizon_s": float(horizon_s),
        "nodes": [{"id": 0, "name": "RS-0", "role": "rs"}] + [{"id": j, "name": f"CS-{j}", "role": "compute"} for j in range(1, nodes + 1)],
        "edges": edges,
        "dark": dark,
    })


def make_dataset(
    seed: int,
    n_short: int,
    n_long: int,
    *,
    length_ratio: float = 1.0,
    post_start_k: int = 0,
) -> list[Request]:
    rng = random.Random(seed)
    items: list[tuple[float, bool, int, int, int, float, float]] = []
    for _ in range(n_short):
        t = round(rng.uniform(0.0, 20.0), 2)
        actual_pout = rng.randint(800, 1200)
        pred_pout = max(1, round(actual_pout / length_ratio))
        items.append((t, False, 200, actual_pout, pred_pout, 10.0, t + 45.0))
    for _ in range(n_long):
        t = round(rng.uniform(0.0, 20.0), 2)
        actual_pout = rng.randint(9500, 10800)
        pred_pout = max(1, round(actual_pout / length_ratio))
        items.append((t, True, 500, actual_pout, pred_pout, 30.0, t + 195.0))
    items.sort(key=lambda x: x[0])
    reqs = [
        Request(
            i=idx, origin=0, t=t, long=is_long, pin=pin, pout=pout,
            upload_bytes=up_mb * 1e6, deadline=dl,
            output_budget=pred_pout, predicted_pout=pred_pout,
        )
        for idx, (t, is_long, pin, pout, pred_pout, up_mb, dl) in enumerate(items)
    ]
    if post_start_k > 0:
        # Move the last post_start_k short requests to arrive at t=11..14s (post-start arrivals)
        short_idxs = [r.i for r in reqs if not r.long][-post_start_k:]
        for off, idx in enumerate(short_idxs):
            reqs[idx].t = 11.0 + 0.5 * off
            reqs[idx].deadline = reqs[idx].t + 45.0
    return reqs


def compute_frozen_initial_targets(
    cfg: dict,
    trace: Trace,
    reqs: list[Request],
    *,
    force_src_ids: set[int] | None = None,
    force_src_node: int = 1,
) -> tuple[dict[int, int], int, dict[int, int]]:
    """Compute initial placement using ONLY ground-visible predicted_pout & orbit_thermal."""
    env = simpy.Environment()
    inv = RebalanceInvariants.from_cfg(cfg)
    placer = get_placer("orbit_thermal")
    w = DiagnosticWorld(env, trace, copy.deepcopy(cfg), policy=CompositePolicy(placer, NonePlanner(), inv), seed=1)
    for r in reqs:
        rc = copy.deepcopy(r)
        w.submit(rc)
    env.run(until=cfg["horizon_s"])
    targets = {r.i: r.trail[0] for r in w.reqs if r.trail}
    if force_src_ids:
        for rid in force_src_ids:
            targets[rid] = force_src_node
    local_overload = sum(1 for r in w.reqs if r.t_delivered is None or r.t_delivered > r.deadline + 1e-9)
    return targets, local_overload, targets


def _arm_spec(arm: str, world_ref: Any = None) -> tuple[type[DiagnosticWorld], Any, bool]:
    max_ord = SPEC["rebalance"]["max_orders_per_round"]
    if arm == "Static":
        return DiagnosticWorld, NonePlanner(), False
    if arm == "Queue-Reroute":
        return DiagnosticWorld, None, True
    if arm == "Dyn-Zero":
        return ZeroCostDiagnosticWorld, CausalReactivePlanner(mode="naive", max_orders_per_round=max_ord), False
    if arm == "Dyn-CF":
        return ContentionFreeDiagnosticWorld, CausalReactivePlanner(mode="naive", max_orders_per_round=max_ord), False
    if arm == "Dyn-Sh-Naive":
        return DiagnosticWorld, CausalReactivePlanner(mode="naive", max_orders_per_round=max_ord), False
    if arm == "Dyn-Sh-Aware":
        return DiagnosticWorld, CausalReactivePlanner(mode="aware", max_orders_per_round=max_ord), False
    raise ValueError(f"Unknown arm: {arm}")


def run_single_arm(
    arm: str,
    cfg: dict,
    trace: Trace,
    reqs: list[Request],
    targets: dict[int, int],
    seed: int,
    *,
    bg_specs: list[tuple[float, list[int], list[int], float]] | None = None,
) -> dict[str, Any]:
    env = simpy.Environment()
    rec = Recorder(float(cfg.get("frame_s", 5.0)))
    inv = RebalanceInvariants.from_cfg(cfg)
    world_cls, planner, allow_qr = _arm_spec(arm)
    w = world_cls(
        env, trace, copy.deepcopy(cfg),
        policy=CompositePolicy(ReplayPlacer(targets), planner if planner is not None else NonePlanner(), inv),
        recorder=rec, seed=seed, allow_queue_reroute=allow_qr,
    )
    if arm == "Queue-Reroute":
        w.policy.planner = QueueReroutePlanner(w)
    rec.attach(w)
    if bg_specs:
        def _bg():
            t_cur = 0.0
            for t_start, nodes, eids, nbytes in sorted(bg_specs):
                if t_start > t_cur:
                    yield env.timeout(t_start - t_cur)
                    t_cur = t_start
                w.net.add("bg", nodes, eids, nbytes, req=-1)
        env.process(_bg())
    for r in reqs:
        w.submit(copy.deepcopy(r))
    env.run(until=cfg["horizon_s"])
    m = extended_metrics(w)
    trig_counts = {"eclipse": 0, "k1_overload": 0}
    feas_dist: list[int] = []
    reject_totals: dict[str, int] = {}
    if hasattr(w.policy.planner, "decision_log"):
        for d in w.policy.planner.decision_log:
            trig_counts[d["trigger"]] = trig_counts.get(d["trigger"], 0) + 1
            feas_dist.append(d["feasible_targets"])
            for k, v in d["reject_reasons"].items():
                reject_totals[k] = reject_totals.get(k, 0) + v
    req_rows = [
        {
            "id": r.i, "long": r.long, "t": r.t, "pin": r.pin, "pout": r.pout,
            "pred_pout": r.predicted_pout, "deadline": r.deadline,
            "t_admit": r.t_admit, "t_decoded": r.t_decoded, "t_delivered": r.t_delivered,
            "good": w.good(r), "n_mig": r.n_mig, "trail": r.trail, "state": r.state,
        }
        for r in w.reqs
    ]
    return {
        "arm": arm,
        "metrics": m,
        "triggers": trig_counts,
        "feasible_targets_mean": (sum(feas_dist) / len(feas_dist)) if feas_dist else 0.0,
        "reject_totals": reject_totals,
        "requests": req_rows,
    }


def run_k1k2_cell(job: tuple) -> dict[str, Any]:
    seed, ratio, spare, post_k = job
    cfg = make_base_cfg()
    tr = make_trace(6, staggered=(spare == "available"), horizon_s=SPEC["horizon_s"])
    reqs = make_dataset(seed, 24, 24, length_ratio=ratio, post_start_k=post_k)
    force_ids = {r.i for r in reqs if not r.long and r.t >= 11.0} if post_k > 0 else None
    targets, local_ov, _ = compute_frozen_initial_targets(cfg, tr, reqs, force_src_ids=force_ids, force_src_node=1)
    arms_out = {}
    for arm in SPEC["l1_k1k2"]["arms"]:
        arms_out[arm] = run_single_arm(arm, cfg, tr, reqs, targets, seed)
    return {
        "seed": seed, "ratio": ratio, "spare": spare, "post_k": post_k,
        "input_sha": digest([(r.i, r.t, r.pout, r.predicted_pout, r.deadline) for r in reqs]),
        "targets_sha": digest(targets),
        "static_overload": local_ov,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "requests"} for k, v in arms_out.items()},
        "requests_by_arm": {k: v["requests"] for k, v in arms_out.items()},
    }


def _find_edge_id(tr: Trace, u: int, v: int) -> int:
    for idx, (a, b) in enumerate(tr.edges):
        if (a == u and b == v) or (a == v and b == u):
            return idx
    raise ValueError(f"No edge ({u}, {v})")


def run_k3_cell(job: tuple) -> dict[str, Any]:
    seed, ratio, bg_frac, pattern, kv_scale, max_inf = job
    bpe = {0.5: 1, 1.0: 2, 2.0: 4}[kv_scale]
    cfg = make_base_cfg(max_inflight=max_inf, bytes_per_elem=bpe)
    tr = make_trace(6, staggered=True, horizon_s=SPEC["horizon_s"])
    reqs = make_dataset(seed, 24, 24, length_ratio=ratio, post_start_k=0)
    targets, _, _ = compute_frozen_initial_targets(cfg, tr, reqs)

    # Background flow over [85s, 210s] (125s window) on 0.6 Gbps (75 MB/s) inter-group links
    cap_Bps = 0.6e9 / 8.0
    bg_specs: list[tuple[float, list[int], list[int], float]] = []
    if bg_frac > 0.0:
        # Total bytes per link at bg_frac over 125s: bg_frac * cap_Bps * 125
        # Inject in 25s chunks so max-min rate tracks bg_frac smoothly
        if pattern == "concentrated":
            edges_to_load = [(1, 4), (2, 4), (3, 4)]
        else:
            edges_to_load = [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)]
        scale = 3.0 / len(edges_to_load) if pattern == "dispersed" else 1.0
        chunk_bytes = bg_frac * scale * cap_Bps * 25.0
        for t0 in (85.0, 110.0, 135.0, 160.0, 185.0):
            for u, v in edges_to_load:
                eid = _find_edge_id(tr, u, v)
                bg_specs.append((t0, [u, v], [eid], chunk_bytes))

    arms_out = {}
    for arm in SPEC["l1_k3"]["arms"]:
        arms_out[arm] = run_single_arm(arm, cfg, tr, reqs, targets, seed, bg_specs=bg_specs)
    return {
        "seed": seed, "ratio": ratio, "bg_frac": bg_frac, "pattern": pattern,
        "kv_scale": kv_scale, "max_inflight": max_inf,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "requests"} for k, v in arms_out.items()},
    }


def run_24sat_cell(seed: int) -> dict[str, Any]:
    cfg = make_base_cfg(max_inflight=24)
    tr = make_trace(24, staggered=True, horizon_s=SPEC["horizon_s"])
    reqs = make_dataset(seed, 96, 96, length_ratio=2.0, post_start_k=0)
    targets, _, _ = compute_frozen_initial_targets(cfg, tr, reqs)
    arms_out = {}
    for arm in SPEC["l2_24sat"]["arms"]:
        arms_out[arm] = run_single_arm(arm, cfg, tr, reqs, targets, seed)
    return {
        "seed": seed,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "requests"} for k, v in arms_out.items()},
        "requests_by_arm": {k: v["requests"] for k, v in arms_out.items()},
    }


def run_offline_oracle_subproblem(seed: int = 101) -> dict[str, Any]:
    """Exact enumeration oracle on a frozen 6-req 6-sat subproblem at t=85s eclipse."""
    cfg = make_base_cfg()
    tr = make_trace(6, staggered=True, horizon_s=240.0)
    # 6 long requests placed on CS-1..3 (2 per star), deadline=195s
    reqs = [
        Request(i=k, origin=0, t=float(k) * 0.5, long=True, pin=500, pout=10000, upload_bytes=10e6, deadline=195.0, output_budget=10000, predicted_pout=10000)
        for k in range(6)
    ]
    targets = {k: 1 + (k % 3) for k in range(6)}

    class FixedMigSchedulePlanner:
        requires_forecast = False
        def __init__(self, sched: tuple[int, ...]) -> None:
            self.sched = sched
            self.done = False
        def plan(self, view, invariants):
            from satplan.actions import MigrationOrder, PlanAction
            if self.done or view.now < 85.0:
                return PlanAction()
            self.done = True
            orders = tuple(MigrationOrder(req_id=k, dst=dst) for k, dst in enumerate(self.sched) if dst in (4, 5, 6))
            return PlanAction(migrations=orders)

    best_good = -1
    best_sched: tuple[int, ...] | None = None
    best_toks = -1
    # Enumerate 4^6 = 4096 schedules (stay on src or migrate at t=85s to CS-4, CS-5, or CS-6)
    # Keep top-level enumeration fast by testing symmetric subset or full 4^6
    # Wait: 6 requests x 4 choices = 4096, at ~1ms each = ~4s!
    enum_count = 0
    for sched in itertools.product((0, 4, 5, 6), repeat=6):
        enum_count += 1
        env = simpy.Environment()
        inv = RebalanceInvariants.from_cfg(cfg)
        w = DiagnosticWorld(env, tr, copy.deepcopy(cfg), policy=CompositePolicy(ReplayPlacer(targets), FixedMigSchedulePlanner(sched), inv), seed=seed)
        for r in reqs:
            w.submit(copy.deepcopy(r))
        env.run(until=240.0)
        g = sum(1 for r in w.reqs if w.good(r))
        toks = sum(r.pout for r in w.reqs if w.good(r))
        if (g, toks) > (best_good, best_toks):
            best_good, best_toks, best_sched = g, toks, sched

    # Compare against Static, Dyn-Sh-Naive, Dyn-Sh-Aware
    comp = {}
    for arm in ("Static", "Dyn-Zero", "Dyn-CF", "Dyn-Sh-Naive", "Dyn-Sh-Aware"):
        res = run_single_arm(arm, cfg, tr, reqs, targets, seed)
        comp[arm] = {"good": res["metrics"]["good"], "good_frac": res["metrics"]["good_frac"]}
    return {
        "seed": seed,
        "n_requests": 6,
        "enum_scope": enum_count,
        "oracle_best_good": best_good,
        "oracle_best_frac": best_good / 6.0,
        "oracle_best_sched": best_sched,
        "arms": comp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["l1", "l2", "oracle", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    if args.phase in ("l1", "all"):
        k1k2_jobs = [
            (seed, ratio, spare, 0)
            for seed in SPEC["dev_seeds"]
            for ratio in SPEC["l1_k1k2"]["length_ratios"]
            for spare in SPEC["l1_k1k2"]["spare_targets"]
        ] + [
            (seed, 1.0, "available", pk)
            for seed in SPEC["dev_seeds"]
            for pk in SPEC["l1_k1k2"]["post_start_arrivals_at_1x"]
            if pk > 0
        ]
        k1k2_res = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_k1k2_cell, j) for j in k1k2_jobs]
            for f in as_completed(futs):
                k1k2_res.append(f.result())
        k1k2_res.sort(key=lambda x: (x["seed"], x["ratio"], x["spare"], x["post_k"]))
        (args.out / "l1_k1k2.json").write_text(json.dumps(k1k2_res, indent=2))
        print(f"[L1 K1xK2] {len(k1k2_res)} cells done in {time.monotonic() - t0:.1f}s", flush=True)

        # Pre-registered rule: pick first ratio (in [1.0, 2.0, 4.0]) with Static misses > 0 & Dyn-CF rescue > 0
        chosen_ratio = 2.0
        for ratio in SPEC["l1_k1k2"]["length_ratios"]:
            cells = [c for c in k1k2_res if c["ratio"] == ratio and c["spare"] == "available" and c["post_k"] == 0]
            st_miss = sum(48 - c["arms"]["Static"]["metrics"]["good"] for c in cells)
            cf_gain = sum(c["arms"]["Dyn-CF"]["metrics"]["good"] - c["arms"]["Static"]["metrics"]["good"] for c in cells)
            if st_miss > 0 and cf_gain > 0:
                chosen_ratio = ratio
                break

        k3_jobs = []
        for seed in SPEC["dev_seeds"]:
            for bg_frac in SPEC["l1_k3"]["bg_load_fractions"]:
                patterns = ["concentrated"] if bg_frac == 0.0 else SPEC["l1_k3"]["bg_patterns"]
                for pat in patterns:
                    k3_jobs.append((seed, chosen_ratio, bg_frac, pat, 1.0, 24))
            # Near transition (bg_frac=0.6, concentrated): sweep kv_scale & max_inflight
            for kv_s in SPEC["l1_k3"]["kv_scales"]:
                if kv_s != 1.0:
                    k3_jobs.append((seed, chosen_ratio, 0.6, "concentrated", kv_s, 24))
            for mi in SPEC["l1_k3"]["mig_concurrencies"]:
                k3_jobs.append((seed, chosen_ratio, 0.6, "concentrated", 1.0, mi))

        k3_res = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_k3_cell, j) for j in k3_jobs]
            for f in as_completed(futs):
                k3_res.append(f.result())
        k3_res.sort(key=lambda x: (x["seed"], x["bg_frac"], x["pattern"], x["kv_scale"], x["max_inflight"]))
        (args.out / "l1_k3.json").write_text(json.dumps({"chosen_ratio": chosen_ratio, "cells": k3_res}, indent=2))
        print(f"[L1 K3] {len(k3_res)} cells (ratio={chosen_ratio}) done in {time.monotonic() - t0:.1f}s", flush=True)

    if args.phase in ("oracle", "all"):
        ora = run_offline_oracle_subproblem(101)
        (args.out / "l1_oracle.json").write_text(json.dumps(ora, indent=2))
        print(f"[Oracle] scope={ora['enum_scope']} best={ora['oracle_best_good']}/6 in {time.monotonic() - t0:.1f}s", flush=True)

    if args.phase in ("l2", "all"):
        # Frozen 5-seed paired validation on 6-sat (ratio=2.0, bg=0.0 and bg=0.6 concentrated) + 24-sat
        val_jobs_6sat = [
            (seed, 2.0, "available", 0) for seed in SPEC["val_seeds"]
        ]
        val_6sat = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_k1k2_cell, j) for j in val_jobs_6sat]
            for f in as_completed(futs):
                val_6sat.append(f.result())
        # Also run Dyn-Sh-Aware on val_6sat
        for cell in val_6sat:
            cfg = make_base_cfg()
            tr = make_trace(6, staggered=True, horizon_s=SPEC["horizon_s"])
            reqs = make_dataset(cell["seed"], 24, 24, length_ratio=2.0, post_start_k=0)
            targets, _, _ = compute_frozen_initial_targets(cfg, tr, reqs)
            aw = run_single_arm("Dyn-Sh-Aware", cfg, tr, reqs, targets, cell["seed"])
            cell["arms"]["Dyn-Sh-Aware"] = {kk: vv for kk, vv in aw.items() if kk != "requests"}
            cell["requests_by_arm"]["Dyn-Sh-Aware"] = aw["requests"]
        val_6sat.sort(key=lambda x: x["seed"])
        (args.out / "l2_val_6sat.json").write_text(json.dumps(val_6sat, indent=2))

        val_24sat = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_24sat_cell, s) for s in SPEC["val_seeds"]]
            for f in as_completed(futs):
                val_24sat.append(f.result())
        val_24sat.sort(key=lambda x: x["seed"])
        (args.out / "l2_val_24sat.json").write_text(json.dumps(val_24sat, indent=2))
        print(f"[L2] 6-sat & 24-sat 5-seed validation done in {time.monotonic() - t0:.1f}s", flush=True)

    manifest = {
        "issue": "LAB-2082",
        "git_sha": git_head_sha(REPO_ROOT),
        "host": socket.gethostname(),
        "wall_s": round(time.monotonic() - t0, 2),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
