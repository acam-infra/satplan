"""Verification 1: Structural limit of single-shot placement (Minimal 3-node + W53 mapping)."""
from __future__ import annotations

import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import simpy
from experiments.lab2046_common import ReplayPlacer, ZeroCostAtomicWorld, git_head_sha, sha256_file
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, request_table, summarize
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.runner import load_config
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request, make_requests


def _min_trace(horizon_s: float = 300.0) -> Trace:
    """CS-1 sunlit [0,85] & [210,300]; CS-2 eclipse [0,95] & sunlit [95,300]."""
    data = {
        "schema": SCHEMA,
        "name": "lab2046-exp1-staggered-eclipse",
        "horizon_s": float(horizon_s),
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1-EarlySun", "role": "compute"},
            {"id": 2, "name": "CS-2-LateSun", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, float(horizon_s)]]},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0.0, float(horizon_s)]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, float(horizon_s)]]},
        ],
        "dark": [[], [[85.0, 210.0]], [[0.0, 95.0]]],
    }
    return Trace(data)


def _min_cfg(*, upload_mb: float = 50.0) -> dict:
    return {
        "name": "exp1-minimal",
        "delivery": "final",
        "rs_downlink": False,
        "control_period_s": 2.0,
        "frame_s": 5.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 8, "max_mig_per_req": 4},
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7, "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
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
        "net": {"link_gbps": {"intra": 10.0, "inter": 1.5, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {"arrival_window_s": 20.0},
        "slo": {"max_stall_frac": 0.1},
        "_upload_mb": upload_mb,
    }


class _HandoffAt80Planner:
    """Move req 0 from CS-1 to CS-2 at t >= 80s (just before CS-1 enters eclipse at 85s)."""

    requires_forecast = True

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        if view.now < 80.0:
            return PlanAction()
        for r in view.node_requests(1):
            if r.state == "decode" and r.n_mig == 0 and view.fits(2, r):
                return PlanAction(migrations=(MigrationOrder(req_id=r.i, dst=2),))
        return PlanAction()


def _run_min_arm(
    arm: str,
    cfg: dict,
    trace: Trace,
    target_cs: int,
    planner_obj: Any,
    *,
    world_cls: type[CoWorld] = CoWorld,
    upload_mb: float = 50.0,
) -> dict[str, Any]:
    env = simpy.Environment()
    rec = Recorder(5.0)
    inv = RebalanceInvariants.from_cfg(cfg)
    w = world_cls(env, trace, cfg, policy=CompositePolicy(ReplayPlacer({0: target_cs}), planner_obj, inv), recorder=rec, seed=1)
    rec.attach(w)
    r = Request(i=0, origin=0, t=0.0, long=True, pin=500, pout=10500, upload_bytes=upload_mb * 1e6, deadline=178.0)
    timeline: list[dict[str, Any]] = []

    def _probe():
        while True:
            timeline.append(
                {
                    "t": round(env.now, 1),
                    "node": r.node,
                    "state": r.state,
                    "gen_tok": round(r.gen, 1),
                    "cs1_f": round(w.f_of(1), 2),
                    "cs1_ecl": trace.is_eclipse(1, env.now),
                    "cs1_T": round(w.pt.temp_C[1], 1),
                    "cs2_f": round(w.f_of(2), 2),
                    "cs2_ecl": trace.is_eclipse(2, env.now),
                    "cs2_T": round(w.pt.temp_C[2], 1),
                }
            )
            yield env.timeout(10.0)

    env.process(_probe())
    w.submit(r)
    env.run(until=trace.horizon)
    m = summarize(w)
    return {"arm": arm, "metrics": m, "requests": request_table(w), "timeline": timeline, "events": rec.events}


def run_minimal(out_dir: Path) -> None:
    tr = _min_trace(300.0)
    cfg = _min_cfg(upload_mb=50.0)
    # Check T_amb + R_th * P_low <= T0
    assert cfg["thermal"]["T_amb_C"] + cfg["thermal"]["R_th"] * 20.0 <= cfg["thermal"]["T0_C"]

    arms = [
        _run_min_arm("Single_CS1", cfg, tr, 1, get_planner("none", cfg), upload_mb=50.0),
        _run_min_arm("Single_CS2", cfg, tr, 2, get_planner("none", cfg), upload_mb=50.0),
        _run_min_arm("Single_Upper_CS1_0up", cfg, tr, 1, get_planner("none", cfg), upload_mb=0.0),
        _run_min_arm("Single_Upper_CS2_0up", cfg, tr, 2, get_planner("none", cfg), upload_mb=0.0),
        _run_min_arm("Dynamic_RealMig", cfg, tr, 1, _HandoffAt80Planner(), world_cls=CoWorld, upload_mb=50.0),
        _run_min_arm("Dynamic_ZeroCost", cfg, tr, 1, _HandoffAt80Planner(), world_cls=ZeroCostAtomicWorld, upload_mb=50.0),
    ]
    rows = []
    for a in arms:
        m = a["metrics"]
        r0 = a["requests"][0]
        rows.append(
            {
                "arm": a["arm"],
                "goodput_tok_s": m["goodput_tok_s"],
                "long_slo_pct": m["good_long_frac"] * 100.0,
                "badput_tok_s": m["badput_tok_s"],
                "rescued_long_count": m["good_long"],
                "t_delivered_s": r0[6],
                "trail": "->".join(str(x) for x in r0[8]),
                "e2e_p95_all_s": m["e2e_p95_all_s"],
            }
        )
    with (out_dir / "exp1_minimal_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "exp1_minimal_raw.json").write_text(json.dumps(arms, indent=2), encoding="utf-8")

    print("=== Exp 1 Minimal 3-Node (1 long req, pout=10500, deadline=178s) ===")
    for r in rows:
        print(
            f"  {r['arm']:22s} | Good={r['goodput_tok_s']:6.1f} | SLO={r['long_slo_pct']:5.1f}% | "
            f"Bad={r['badput_tok_s']:6.1f} | Rescued={r['rescued_long_count']} | t_del={r['t_delivered_s']}s | trail={r['trail']}"
        )
    print("\n--- Exp 1 Timeline: Single_CS1 vs Dynamic_RealMig ---")
    t_b = {x["t"]: x for x in arms[0]["timeline"]}
    t_d = {x["t"]: x for x in arms[4]["timeline"]}
    for t in sorted(t_d):
        if t <= 180.0:
            xb, xd = t_b[t], t_d[t]
            print(
                f"  t={t:5.1f}s | Single_CS1(node={xb['node']}, gen={xb['gen_tok']:7.1f}, f1={xb['cs1_f']}) "
                f"| Dynamic(node={xd['node']}, gen={xd['gen_tok']:7.1f}, f1={xd['cs1_f']}, f2={xd['cs2_f']})"
            )


def run_w53_single_vs_realloc(out_dir: Path, here: Path) -> None:
    """Compare single-shot placers (nearest, least_loaded, strong orbit_thermal) vs reallocation on W53."""
    base_cfg_path = here.parent.parent / "configs" / "w53_baseline.yaml"
    cfg = load_config(base_cfg_path)
    cfg["workload"]["arrival_window_s"] = 400.0
    cfg["rebalance_invariants"] = {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4}
    cfg["power"]["eclipse_dod_budget_j"] = 100000.0
    cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
        {"name": "low_20w", "power_w": 20.0, "clock_f": 0.35},
        {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
    ]
    cfg["thermal"].update({"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.25})
    cfg["net"]["link_gbps"]["inter"] = 0.35

    trace_path = (Path(cfg["_dir"]) / cfg["trace"]).resolve()
    tr_raw = json.loads(trace_path.read_text(encoding="utf-8"))
    tr_raw["horizon_s"] = 3600.0
    trace = Trace(tr_raw)
    inv = RebalanceInvariants.from_cfg(cfg)

    w53_rows: list[dict[str, Any]] = []
    for seed in (1, 2, 3):
        reqs = make_requests(cfg, trace, seed)
        for arm_name, placer_name in [
            ("Single_nearest", "nearest"),
            ("Single_least_loaded", "least_loaded"),
            ("Single_orbit_thermal", "orbit_thermal"),
        ]:
            env = simpy.Environment()
            w = CoWorld(env, trace, copy.deepcopy(cfg), policy=CompositePolicy(get_placer(placer_name), get_planner("none", cfg), inv), seed=seed)
            for r in reqs:
                w.submit(
                    Request(
                        i=r.i,
                        origin=r.origin,
                        t=r.t,
                        long=r.long,
                        pin=r.pin,
                        pout=r.pout,
                        upload_bytes=r.upload_bytes,
                        deadline=r.deadline,
                        output_budget=r.output_budget,
                    )
                )
            env.run(until=trace.horizon)
            m = summarize(w)
            w53_rows.append(
                {
                    "arm": arm_name,
                    "seed": seed,
                    "goodput_tok_s": m["goodput_tok_s"],
                    "long_slo_pct": m["good_long_frac"] * 100.0,
                    "badput_tok_s": m["badput_tok_s"],
                    "good_long_count": m["good_long"],
                    "e2e_p95_all_s": m["e2e_p95_all_s"],
                }
            )

    with (out_dir / "exp1_w53_single_placers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(w53_rows[0].keys()))
        w.writeheader()
        w.writerows(w53_rows)

    print("\n=== Exp 1 W53 Constrained (3-seed mean: Single-shot placers vs Realloc) ===")
    b_g = statistics.mean(r["goodput_tok_s"] for r in w53_rows if r["arm"] == "Single_nearest")
    b_l = statistics.mean(r["good_long_count"] for r in w53_rows if r["arm"] == "Single_nearest")
    for arm in ("Single_nearest", "Single_least_loaded", "Single_orbit_thermal"):
        rs = [r for r in w53_rows if r["arm"] == arm]
        gm = statistics.mean(r["goodput_tok_s"] for r in rs)
        sm = statistics.mean(r["long_slo_pct"] for r in rs)
        bm = statistics.mean(r["badput_tok_s"] for r in rs)
        lm = statistics.mean(r["good_long_count"] for r in rs)
        print(
            f"  {arm:22s} | Good={gm:7.1f} ({(gm/b_g-1)*100:+5.1f}%) | LongSLO={sm:5.2f}% | "
            f"Bad={bm:7.1f} | GoodLong={lm:5.1f} (net={lm-b_l:+5.1f})"
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_minimal(out_dir)
    run_w53_single_vs_realloc(out_dir, here)
    mf = {
        "pre_run_head_sha": git_head_sha(here.parent.parent),
        "runner_sha256": sha256_file(Path(__file__)),
        "rerun_cmd": "PYTHONPATH=. uv run --frozen python experiments/lab2046_two_stage_proof/exp1_single_placement_limit.py",
    }
    (out_dir / "exp1_manifest.json").write_text(json.dumps(mf, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
