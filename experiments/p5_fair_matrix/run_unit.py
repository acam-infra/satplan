"""Run a single P5 experiment unit from frozen spec and request manifest."""
from __future__ import annotations

import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import simpy
from satplan.demand_forecast import DemandForecastConfig, PublishedDemandPlan
from satplan.metrics import summarize
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner
from satplan.runner import load_config, run_simulation
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request

EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parents[1]

PLACERS = ("nearest", "least_loaded", "orbit_thermal")
SEEDS = (1, 2, 3)


def list_units() -> list[str]:
    units: list[str] = []
    # Main 3x3x3-seed x {mig4, mignull} = 54 units
    for mig_tag in ("mig4", "mignull"):
        for placer in PLACERS:
            for planner in ("none", "reactive", "r3_milp"):
                for seed in SEEDS:
                    units.append(f"main__{mig_tag}__{placer}__{planner}__s{seed}")
    # Same-info forecast_aware & r3_milp_fc x 3 placers x 3 seeds x {mig4, mignull} = 36 units
    for mig_tag in ("mig4", "mignull"):
        for placer in PLACERS:
            for planner in ("forecast_aware", "r3_milp_fc"):
                for seed in SEEDS:
                    units.append(f"fc__{mig_tag}__{placer}__{planner}__s{seed}")
    # Minimal ablation (nearest, mig4, 3 seeds x 3 configs = 9 units)
    for abl in ("tail_off_fc_off", "tail_on_fc_off", "tail_on_fc_on"):
        for seed in SEEDS:
            units.append(f"ablation__{abl}__s{seed}")
    # Diagnostic subset + bounded oracle reference
    units.append("diag__subset")
    return units


def _load_manifest_requests(seed: int) -> tuple[list[Request], str]:
    manifest_p = EXP_DIR / "manifests" / f"requests_seed_{seed}.json"
    meta_p = EXP_DIR / "manifests" / "manifest_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    raw = json.loads(manifest_p.read_text(encoding="utf-8"))
    reqs = [
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
    ]
    return reqs, str(meta["seeds"][str(seed)]["sha256"])


def _run_diag_subset(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Fixed small diagnostic subset covering hotspot presence, reachability, forecast error, false hotspot, and oracle_ref."""
    scenarios: list[dict[str, Any]] = []

    def make_trace(*, inter_up: list[list[float]], name: str) -> Trace:
        return Trace({
            "schema": SCHEMA,
            "name": name,
            "horizon_s": 50.0,
            "nodes": [
                {"id": 0, "role": "rs"},
                {"id": 1, "role": "compute"},
                {"id": 2, "role": "compute"},
                {"id": 3, "role": "rs"},
            ],
            "edges": [
                {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 50.0]]},
                {"u": 1, "v": 2, "kind": "inter", "up": inter_up},
                {"u": 3, "v": 1, "kind": "rs", "up": [[0.0, 50.0]]},
            ],
            "dark": [[], [], [], []],
        })

    def run_case(
        name: str,
        tr: Trace,
        *,
        plans: list[PublishedDemandPlan],
        actual_reqs: list[Request],
        policy_name: str,
    ) -> dict[str, Any]:
        c = copy.deepcopy(base_cfg)
        c["control_period_s"] = 2.0
        c["gpu"].update(kv_gib=0.03, t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
        c["net"]["link_gbps"].update(inter=0.08, rs=10.0)
        c["rebalance_invariants"] = {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4}
        c["demand_forecast"] = {"enabled": policy_name != "r3_no_fc", "obs_window_s": 1.0}
        c["r3"] = {"horizon_slots": 4, "tail_enabled": False, "time_limit_s": 2.0}

        if policy_name == "oracle_ref":
            # Bounded-horizon / time-limited reference with exact published plans matching actual future arrivals.
            pl = R3RuntimePlanner(
                builder=ProblemBuilder(
                    horizon_slots=4,
                    tail_enabled=False,
                    demand_forecast=DemandForecastConfig(enabled=True, obs_window_s=1.0),
                    time_limit_s=2.0,
                )
            )
        elif policy_name == "forecast_aware":
            pl = get_planner("forecast_aware", c)
        else:
            pl = get_planner("r3_milp", c)

        def placer(view: Any, req: Request) -> int | None:
            if req.origin == 3:
                return 1 if view.kv_headroom(1) >= req.reserved_tokens * view.bpt - 1e-6 else None
            return 1

        env = simpy.Environment()
        pol = CompositePolicy(placer=placer, planner=pl, invariants=RebalanceInvariants.from_cfg(c))
        w = CoWorld(env, tr, c, policy=pol, seed=1, published_plans=plans)
        for r in actual_reqs:
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
                )
            )
        env.run(until=tr.horizon)
        m = summarize(w)
        if hasattr(pl, "stats_summary"):
            m.update(pl.stats_summary())
        return {"scenario": name, "policy": policy_name, "metrics": m}

    req0 = Request(0, 0, 0.0, False, 4900, 150, 1.0, 40.0)
    req1 = Request(1, 3, 8.0, False, 4950, 50, 1.0, 20.0)
    true_plan = PublishedDemandPlan(0.0, 8.0, 3, 1.0, 50.0, 5000 * 4096.0, 12.0, weight=5.0)
    err_plan = PublishedDemandPlan(0.0, 10.0, 3, 2.0, 50.0, 5000 * 4096.0, 12.0, weight=5.0)

    # 1. Reachable hotspot present (inter link [0, 8])
    tr_reach = make_trace(inter_up=[[0.0, 8.0]], name="diag-reachable")
    for p in ("r3_no_fc", "forecast_aware", "r3_fc", "oracle_ref"):
        plans = [true_plan] if p == "oracle_ref" else ([true_plan] if p != "r3_no_fc" else [])
        scenarios.append(run_case("hotspot_reachable", tr_reach, plans=plans, actual_reqs=[req0, req1], policy_name=p))

    # 2. Unreachable hotspot (no inter link)
    tr_unreach = make_trace(inter_up=[], name="diag-unreachable")
    for p in ("r3_no_fc", "forecast_aware", "r3_fc"):
        scenarios.append(
            run_case("hotspot_unreachable", tr_unreach, plans=[true_plan], actual_reqs=[req0, req1], policy_name=p)
        )

    # 3. Forecast magnitude/timing error
    for p in ("r3_no_fc", "forecast_aware", "r3_fc", "oracle_ref"):
        plans = [true_plan] if p == "oracle_ref" else [err_plan]
        scenarios.append(run_case("forecast_error", tr_reach, plans=plans, actual_reqs=[req0, req1], policy_name=p))

    # 4. False hotspot (no actual future arrival)
    for p in ("r3_no_fc", "forecast_aware", "r3_fc"):
        scenarios.append(run_case("false_hotspot", tr_reach, plans=[true_plan], actual_reqs=[req0], policy_name=p))

    return {"unit": "diag__subset", "scenarios": scenarios}


def run_unit(unit_id: str) -> dict[str, Any]:
    spec_path = EXP_DIR / "spec.yaml"
    cfg = load_config(spec_path)
    meta = json.loads((EXP_DIR / "manifests" / "manifest_meta.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()

    if unit_id == "diag__subset":
        payload = _run_diag_subset(cfg)
        payload["wall_time_s"] = time.perf_counter() - t0
        payload["host"] = platform.node()
        return payload

    parts = unit_id.split("__")
    kind = parts[0]
    if kind in ("main", "fc"):
        _, mig_tag, placer, planner_tag, seed_s = parts
        seed = int(seed_s[1:])
        max_mig = 4 if mig_tag == "mig4" else None
        fc_enabled = kind == "fc"
        tail_enabled = True
        planner = "r3_milp" if planner_tag in ("r3_milp", "r3_milp_fc") else planner_tag
    elif kind == "ablation":
        _, abl, seed_s = parts
        seed = int(seed_s[1:])
        placer = "nearest"
        planner = "r3_milp"
        planner_tag = abl
        max_mig = 4
        tail_enabled = abl != "tail_off_fc_off"
        fc_enabled = abl == "tail_on_fc_on"
    else:
        raise ValueError(f"Unknown unit_id: {unit_id}")

    run_cfg = copy.deepcopy(cfg)
    run_cfg["rebalance_invariants"]["max_mig_per_req"] = max_mig
    run_cfg["r3"]["tail_enabled"] = tail_enabled
    run_cfg["demand_forecast"]["enabled"] = fc_enabled

    tr = Trace.load((EXP_DIR / run_cfg["trace"]).resolve())
    reqs, manifest_sha = _load_manifest_requests(seed)
    res = run_simulation(tr, run_cfg, placer, planner, seed, record=False, requests=reqs)
    wall_s = time.perf_counter() - t0

    return {
        "unit": unit_id,
        "group": kind,
        "placer": placer,
        "planner": planner_tag,
        "seed": seed,
        "max_mig_per_req": max_mig,
        "tail_enabled": tail_enabled,
        "forecast_enabled": fc_enabled,
        "wall_time_s": wall_s,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "spec_sha256": meta["spec_sha256"],
        "trace_sha256": meta["trace_sha256"],
        "manifest_sha256": manifest_sha,
        "metrics": res.metrics,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_unit.py <list|unit_id>")
    cmd = sys.argv[1]
    if cmd == "list":
        for u in list_units():
            print(u)
        return
    out_dir = Path(os.environ.get("OUT", str(EXP_DIR / "out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_unit(cmd)
    (out_dir / f"{cmd}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
