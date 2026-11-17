"""Verification 1: Structural limit of single-shot placement (Minimal 3-node + W53 mapping)."""
from __future__ import annotations

import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import simpy
import yaml
from experiments.lab2046_common import ReplayPlacer, ZeroCostAtomicWorld, git_head_sha, sha256_bytes, sha256_file
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, request_table, summarize
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.runner import load_config
from satplan.timing import iter_time
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request, make_requests


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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


def _analytical_single_star_bounds(cfg: dict) -> dict[str, Any]:
    """Model-derived single-star capacity upper bound (0 upload, 0 context, no thermal limit, arbitrary tier mixing)."""
    gpu = cfg["gpu"]
    req = cfg["request"]
    dl = float(req["deadline"])
    pout = float(req["pout"])
    dod_j = float(cfg["power"]["eclipse_dod_budget_j"])
    tiers = [t for t in cfg["power"]["tiers"] if float(t["power_w"]) > 0.0]
    rates = [(t["name"], float(t["power_w"]), 1.0 / iter_time(1, 0.0, float(t["clock_f"]), gpu)) for t in tiers]
    max_rate_name, _, max_rate_tok_s = max(rates, key=lambda x: x[2])
    best_eff_name, best_eff_pw, best_eff_rate = max(rates, key=lambda x: x[2] / x[1])
    best_eff_tok_per_j = best_eff_rate / best_eff_pw
    cs1_sun_s = 85.0
    cs2_sun_s = max(0.0, dl - 95.0)
    cs1_ub = cs1_sun_s * max_rate_tok_s + dod_j * best_eff_tok_per_j
    cs2_ub = cs2_sun_s * max_rate_tok_s + dod_j * best_eff_tok_per_j
    return {
        "relaxation": "0_upload, 0_context_overhead, ignore_thermal_Tmax, arbitrary_tier_mixing",
        "max_rate_tier": max_rate_name,
        "max_rate_tok_s": round(max_rate_tok_s, 3),
        "best_eff_tier": best_eff_name,
        "best_eff_tok_per_j": round(best_eff_tok_per_j, 3),
        "eclipse_dod_budget_j": dod_j,
        "required_pout_tokens": pout,
        "deadline_s": dl,
        "cs1_sunlit_s_before_dl": cs1_sun_s,
        "cs1_upper_bound_tokens": round(cs1_ub, 1),
        "cs2_sunlit_s_before_dl": cs2_sun_s,
        "cs2_upper_bound_tokens": round(cs2_ub, 1),
        "feasible_single_star_exists": (cs1_ub >= pout or cs2_ub >= pout),
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
    rec = Recorder(float(cfg.get("frame_s", 5.0)))
    inv = RebalanceInvariants.from_cfg(cfg)
    w = world_cls(env, trace, cfg, policy=CompositePolicy(ReplayPlacer({0: target_cs}), planner_obj, inv), recorder=rec, seed=1)
    rec.attach(w)
    req_spec = cfg["request"]
    r = Request(
        i=int(req_spec["i"]),
        origin=int(req_spec["origin"]),
        t=float(req_spec["t"]),
        long=bool(req_spec["long"]),
        pin=int(req_spec["pin"]),
        pout=int(req_spec["pout"]),
        upload_bytes=upload_mb * 1e6,
        deadline=float(req_spec["deadline"]),
    )
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
    return {"arm": arm, "metrics": summarize(w), "requests": request_table(w), "timeline": timeline, "events": rec.events}


def run_minimal(spec: dict, out_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(spec["exp1_minimal"])
    cfg["name"] = "exp1-minimal"
    cfg["delivery"] = "final"
    cfg["rs_downlink"] = False
    tr = _min_trace(float(cfg["horizon_s"]))
    assert cfg["thermal"]["T_amb_C"] + cfg["thermal"]["R_th"] * 20.0 <= cfg["thermal"]["T0_C"]
    bounds = _analytical_single_star_bounds(cfg)
    up_mb = float(cfg["request"]["upload_MB"])

    arms = [
        _run_min_arm("Single_CS1", cfg, tr, 1, get_planner("none", cfg), upload_mb=up_mb),
        _run_min_arm("Single_CS2", cfg, tr, 2, get_planner("none", cfg), upload_mb=up_mb),
        _run_min_arm("Single_0Upload_CS1", cfg, tr, 1, get_planner("none", cfg), upload_mb=0.0),
        _run_min_arm("Single_0Upload_CS2", cfg, tr, 2, get_planner("none", cfg), upload_mb=0.0),
        _run_min_arm("Dynamic_RealMig", cfg, tr, 1, _HandoffAt80Planner(), world_cls=CoWorld, upload_mb=up_mb),
        _run_min_arm("Dynamic_ZeroCost", cfg, tr, 1, _HandoffAt80Planner(), world_cls=ZeroCostAtomicWorld, upload_mb=up_mb),
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

    print("=== Exp 1 Analytical Single-Star Bounds ===")
    print(" ", bounds)
    print("=== Exp 1 Minimal 3-Node ===")
    for r in rows:
        print(
            f"  {r['arm']:20s} | Good={r['goodput_tok_s']:6.1f} | SLO={r['long_slo_pct']:5.1f}% | "
            f"Bad={r['badput_tok_s']:6.1f} | Rescued={r['rescued_long_count']} | t_del={r['t_delivered_s']}s | trail={r['trail']}"
        )
    return bounds


def run_w53_single_vs_realloc(spec: dict, out_dir: Path, here: Path) -> dict[str, Any]:
    raw_dir = out_dir / "raw_w53_exp1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_cfg_path = (here / spec["base_w53_config"]).resolve()
    base_cfg = load_config(base_cfg_path)
    base_cfg["workload"]["arrival_window_s"] = float(spec["w53_arrival_window_s"])
    base_cfg["rebalance_invariants"] = copy.deepcopy(spec["w53_rebalance_invariants"])
    cfg = _deep_merge(base_cfg, spec["w53_constrained_overrides"])

    trace_path = (Path(base_cfg["_dir"]) / base_cfg["trace"]).resolve()
    tr_raw = json.loads(trace_path.read_text(encoding="utf-8"))
    tr_raw["horizon_s"] = float(spec["w53_horizon_s"])
    trace = Trace(tr_raw)
    inv = RebalanceInvariants.from_cfg(cfg)

    w53_rows: list[dict[str, Any]] = []
    for seed in spec["w53_seeds"]:
        reqs = make_requests(cfg, trace, int(seed))
        for arm_name, placer_name in [
            ("Single_nearest", "nearest"),
            ("Single_least_loaded", "least_loaded"),
            ("Single_orbit_thermal", "orbit_thermal"),
        ]:
            env = simpy.Environment()
            w = CoWorld(env, trace, copy.deepcopy(cfg), policy=CompositePolicy(get_placer(placer_name), get_planner("none", cfg), inv), seed=int(seed))
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
            m["censored_dl_gt_h"] = sum(1 for r in w.reqs if r.deadline > trace.horizon)
            m["censored_completion_count"] = sum(1 for r in w.reqs if r.t_delivered is None)
            rt = request_table(w)
            (raw_dir / f"{arm_name}__s{seed}.json").write_text(
                json.dumps({"arm": arm_name, "seed": int(seed), "metrics": m, "requests": rt}, indent=2),
                encoding="utf-8",
            )
            w53_rows.append(
                {
                    "arm": arm_name,
                    "seed": int(seed),
                    "goodput_tok_s": m["goodput_tok_s"],
                    "long_slo_pct": m["good_long_frac"] * 100.0,
                    "badput_tok_s": m["badput_tok_s"],
                    "good_long_count": m["good_long"],
                    "e2e_p95_all_s": m["e2e_p95_all_s"],
                    "censored_dl_gt_h": m["censored_dl_gt_h"],
                    "censored_completion_count": m["censored_completion_count"],
                }
            )

    with (out_dir / "exp1_w53_single_placers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(w53_rows[0].keys()))
        w.writeheader()
        w.writerows(w53_rows)

    print("\n=== Exp 1 W53 Constrained (3-seed mean: Single-shot placers) ===")
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
    return {
        "base_cfg_sha256": sha256_file(base_cfg_path),
        "trace_sha256": sha256_file(trace_path),
        "effective_w53_config": {k: v for k, v in cfg.items() if k != "_dir"},
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = here / "spec.yaml"
    spec_raw = spec_path.read_bytes()
    spec = yaml.safe_load(spec_raw)

    bounds = run_minimal(spec, out_dir)
    w53_meta = run_w53_single_vs_realloc(spec, out_dir, here)
    mf = {
        "base_main_sha": "04bfed1f675fcb899dad352bdba7b47c79383e7d",
        "pre_run_head_sha": git_head_sha(here.parent.parent),
        "runner_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(here.parent / "lab2046_common.py"),
        "spec_sha256": sha256_bytes(spec_raw),
        "analytical_single_star_bounds": bounds,
        **w53_meta,
        "rerun_cmd": "PYTHONPATH=. uv run --frozen python experiments/lab2046_two_stage_proof/exp1_single_placement_limit.py",
    }
    (out_dir / "exp1_manifest.json").write_text(json.dumps(mf, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
