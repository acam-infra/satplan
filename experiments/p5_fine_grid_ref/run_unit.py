"""Run a single bounded all-fine-grid MILP vs mixed-grid reference unit."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from satplan.runner import load_config, run_simulation
from satplan.trace import Trace
from satplan.world import Request

EXP_DIR = Path(__file__).resolve().parent

UNITS: tuple[str, ...] = (
    "small_120s_tl5__mixed_6p4",
    "small_120s_tl5__all_fine_26",
    "w53_600s_tl0p5__mixed_6p4",
    "w53_600s_tl0p5__all_fine_26",
)


def _load_requests(horizon_s: float) -> list[Request]:
    raw = json.loads((EXP_DIR / "inputs" / "requests_seed_1.json").read_text(encoding="utf-8"))
    return [
        Request(
            i=int(r["i"]),
            origin=int(r["origin"]),
            t=float(r["t"]),
            long=bool(r["long"]),
            pin=int(r["pin"]),
            pout=int(r["pout"]),
            upload_bytes=float(r["upload_MB"]) * 1e6,
            deadline=float(r["deadline"]),
        )
        for r in raw
        if float(r["t"]) < horizon_s
    ]


def run_unit(unit_id: str) -> dict[str, Any]:
    if unit_id not in UNITS:
        raise ValueError(f"Unknown unit: {unit_id}")
    case_key, grid_key = unit_id.split("__")
    spec_path = EXP_DIR / "spec.yaml"
    cfg = load_config(spec_path)
    case_cfg = cfg["cases"][case_key]
    grid_cfg = cfg["grids"][grid_key]

    run_cfg = copy.deepcopy(cfg)
    run_cfg["workload"]["arrival_window_s"] = float(case_cfg["arrival_window_s"])
    run_cfg["r3"] = {
        "horizon_slots": int(grid_cfg["horizon_slots"]),
        "tail_enabled": bool(grid_cfg["tail_enabled"]),
        "tail_slots": int(grid_cfg["tail_slots"]),
        "tail_dt_s": float(grid_cfg["tail_dt_s"]),
        "time_limit_s": float(case_cfg["time_limit_s"]),
    }

    trace_path = EXP_DIR / cfg["trace"]
    tr_dict = json.loads(trace_path.read_text(encoding="utf-8"))
    tr_dict["horizon_s"] = float(case_cfg["horizon_s"])
    tr = Trace(tr_dict)
    reqs = _load_requests(float(case_cfg["horizon_s"]))

    t0 = time.perf_counter()
    res = run_simulation(tr, run_cfg, str(cfg["placer"]), "r3_milp", int(cfg["seed"]), record=False, requests=reqs)
    wall_s = time.perf_counter() - t0

    return {
        "unit": unit_id,
        "case": case_key,
        "grid": grid_key,
        "horizon_s": float(case_cfg["horizon_s"]),
        "time_limit_s": float(case_cfg["time_limit_s"]),
        "planning_window_s": float(grid_cfg["planning_window_s"]),
        "total_slots": int(grid_cfg["total_slots"]),
        "slot_dt_desc": "6x6s + 4x30s" if grid_cfg["tail_enabled"] else "26x6s",
        "wall_time_s": wall_s,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "metrics": res.metrics,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_unit.py <list|unit_id>")
    cmd = sys.argv[1]
    if cmd == "list":
        for u in UNITS:
            print(u)
        return
    out_dir = Path(os.environ.get("OUT", str(EXP_DIR / "out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_unit(cmd)
    (out_dir / f"{cmd}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
