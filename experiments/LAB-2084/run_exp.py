"""LAB-2084 runner: anchor reproduction, dev budget grid, held-out 5-seed validation, and 24-sat direction check."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from common import (
    compute_frozen_initial_targets,
    digest,
    make_base_cfg,
    make_dataset,
    make_trace,
    run_single_arm,
)
from experiments.lab2046_common import git_head_sha

SPEC = json.loads((HERE / "spec.json").read_text())


def run_6sat_cell(job: tuple[int, dict[str, Any], bool]) -> dict[str, Any]:
    seed, sc, include_events = job
    cfg = make_base_cfg(SPEC)
    neg = sc["neg_mode"]
    tr = make_trace(
        6,
        staggered=(neg != "no_spare"),
        horizon_s=SPEC["horizon_s"],
        cut_inter_group_at_s=80.0 if neg == "no_contact" else None,
    )
    reqs, pert_info = make_dataset(
        seed,
        sc["n_short"],
        sc["n_long"],
        length_ratio=sc["ratio"],
        dl_long_off_s=sc["dl_off_s"],
        post_start_k=sc["post_k"],
    )
    force_ids = set(pert_info["post_start_req_ids"]) if sc["post_k"] > 0 else None
    targets = compute_frozen_initial_targets(cfg, tr, reqs, force_src_ids=force_ids, force_src_node=1)
    arms_out = {}
    for arm in SPEC["arms"]:
        arms_out[arm] = run_single_arm(
            arm,
            cfg,
            tr,
            reqs,
            targets,
            seed,
            max_orders_per_round=SPEC["rebalance"]["max_orders_per_round"],
            include_events=include_events,
        )
    return {
        "scenario_id": sc["id"],
        "seed": seed,
        "scenario": sc,
        "input_sha": digest([(r.i, r.t, r.pout, r.predicted_pout, r.deadline) for r in reqs]),
        "targets_sha": digest(targets),
        "pert_info": {
            "length_ratio": pert_info["length_ratio"],
            "n_underestimated": len(pert_info["underestimated_req_ids"]),
            "post_start_req_ids": pert_info["post_start_req_ids"],
        },
        "arms": {k: {kk: vv for kk, vv in v.items() if kk not in ("requests", "events", "decision_log")} for k, v in arms_out.items()},
        "requests_by_arm": {k: v["requests"] for k, v in arms_out.items()},
        **({"events_by_arm": {k: v.get("events", []) for k, v in arms_out.items()},
            "decision_log_dyn": arms_out["Dyn-CF"].get("decision_log", []),
            "initial_targets": targets} if include_events else {}),
    }


def run_24sat_cell(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    seed, sc = job
    cfg = make_base_cfg(SPEC)
    tr = make_trace(sc["nodes"], staggered=True, horizon_s=SPEC["horizon_s"])
    reqs, _ = make_dataset(
        seed,
        sc["n_short"],
        sc["n_long"],
        length_ratio=sc["ratio"],
        dl_long_off_s=sc["dl_off_s"],
        post_start_k=0,
    )
    targets = compute_frozen_initial_targets(cfg, tr, reqs)
    arms_out = {}
    for arm in SPEC["arms"]:
        arms_out[arm] = run_single_arm(
            arm,
            cfg,
            tr,
            reqs,
            targets,
            seed,
            max_orders_per_round=SPEC["rebalance"]["max_orders_per_round"],
        )
    return {
        "scenario_id": sc["id"],
        "seed": seed,
        "scenario": sc,
        "input_sha": digest([(r.i, r.t, r.pout, r.predicted_pout, r.deadline) for r in reqs]),
        "targets_sha": digest(targets),
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "requests"} for k, v in arms_out.items()},
        "requests_by_arm": {k: v["requests"] for k, v in arms_out.items()},
    }


def run_placer_check(seed: int = 101) -> list[dict[str, Any]]:
    """Verify orbit_thermal against least_loaded and nearest on ground-visible inputs."""
    cfg = make_base_cfg(SPEC)
    tr = make_trace(6, staggered=True, horizon_s=SPEC["horizon_s"])
    rows: list[dict[str, Any]] = []
    for sc_id in ("mod_1.0x", "mod_2.0x", "hp_2.0x"):
        sc = next(x for x in SPEC["budget_grid_6sat"] if x["id"] == sc_id)
        reqs, _ = make_dataset(
            seed,
            sc["n_short"],
            sc["n_long"],
            length_ratio=sc["ratio"],
            dl_long_off_s=sc["dl_off_s"],
        )
        for pname in ("orbit_thermal", "least_loaded", "nearest"):
            targets = compute_frozen_initial_targets(cfg, tr, reqs, placer_name=pname)
            st = run_single_arm("Static", cfg, tr, reqs, targets, seed)
            cf = run_single_arm("Dyn-CF", cfg, tr, reqs, targets, seed)
            rows.append({
                "scenario_id": sc_id,
                "seed": seed,
                "placer": pname,
                "Static": st["metrics"],
                "Dyn-CF": cf["metrics"],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["anchor", "dev", "val", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    if args.phase in ("anchor", "all"):
        anc = SPEC["anchor_repro"]
        sc_6 = {
            "id": "anchor_6sat_s142",
            "n_short": anc["6sat_n_short"],
            "n_long": anc["6sat_n_long"],
            "dl_off_s": anc["6sat_dl_off_s"],
            "ratio": anc["6sat_ratio"],
            "post_k": 0,
            "neg_mode": "none",
        }
        r6 = run_6sat_cell((anc["6sat_seed"], sc_6, False))
        sc_24 = {
            "id": "anchor_24sat",
            "nodes": 24,
            "n_short": anc["24sat_n_short"],
            "n_long": anc["24sat_n_long"],
            "dl_off_s": anc["24sat_dl_off_s"],
            "ratio": anc["24sat_ratio"],
        }
        r24 = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_24sat_cell, (s, sc_24)) for s in anc["24sat_seeds"]]
            for f in as_completed(futs):
                r24.append(f.result())
        r24.sort(key=lambda x: x["seed"])
        (args.out / "anchor_repro.json").write_text(
            json.dumps({"sat6": r6, "sat24": r24}, indent=2)
        )
        print(f"[Anchor] done in {time.monotonic() - t0:.1f}s", flush=True)

    if args.phase in ("dev", "all"):
        dev_jobs = [
            (seed, sc, False)
            for seed in SPEC["dev_seeds"]
            for sc in SPEC["budget_grid_6sat"]
        ]
        dev_res = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_6sat_cell, j) for j in dev_jobs]
            for f in as_completed(futs):
                dev_res.append(f.result())
        dev_res.sort(key=lambda x: (x["scenario_id"], x["seed"]))
        placer_rows = run_placer_check(SPEC["dev_seeds"][0])
        (args.out / "dev_6sat.json").write_text(
            json.dumps({"cells": dev_res, "placer_check": placer_rows}, indent=2)
        )
        print(f"[Dev] {len(dev_res)} cells done in {time.monotonic() - t0:.1f}s", flush=True)

    if args.phase in ("val", "all"):
        val_jobs = [
            (seed, sc, (seed == SPEC["val_seeds"][0] and sc["id"] in ("mod_1.0x", "mod_2.0x", "mod_2.0x_no_contact", "hp_2.0x")))
            for seed in SPEC["val_seeds"]
            for sc in SPEC["budget_grid_6sat"]
        ]
        val_res = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_6sat_cell, j) for j in val_jobs]
            for f in as_completed(futs):
                val_res.append(f.result())
        val_res.sort(key=lambda x: (x["scenario_id"], x["seed"]))

        sample_logs = [
            {
                "scenario_id": c["scenario_id"],
                "seed": c["seed"],
                "initial_targets": c["initial_targets"],
                "events_by_arm": c["events_by_arm"],
                "decision_log_dyn": c["decision_log_dyn"],
                "requests_by_arm": c["requests_by_arm"],
            }
            for c in val_res
            if "events_by_arm" in c
        ]
        for c in val_res:
            c.pop("events_by_arm", None)
            c.pop("decision_log_dyn", None)
            c.pop("initial_targets", None)

        (args.out / "sample_logs_seed142.json").write_text(json.dumps(sample_logs, indent=2))
        (args.out / "val_6sat.json").write_text(json.dumps(val_res, indent=2))

        val24_jobs = [
            (seed, sc)
            for seed in SPEC["val_seeds"]
            for sc in SPEC["direction_24sat"]
        ]
        val24_res = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_24sat_cell, j) for j in val24_jobs]
            for f in as_completed(futs):
                val24_res.append(f.result())
        val24_res.sort(key=lambda x: (x["scenario_id"], x["seed"]))
        (args.out / "val_24sat.json").write_text(json.dumps(val24_res, indent=2))
        print(f"[Val] {len(val_res)} 6-sat + {len(val24_res)} 24-sat done in {time.monotonic() - t0:.1f}s", flush=True)

    try:
        sha = git_head_sha(REPO_ROOT)
    except Exception:
        sha = SPEC.get("base_commit", "unknown")
    manifest = {
        "issue": "LAB-2084",
        "git_sha": sha,
        "host": socket.gethostname(),
        "wall_s": round(time.monotonic() - t0, 2),
        "effective_cfg": make_base_cfg(SPEC),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
