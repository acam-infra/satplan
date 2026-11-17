"""Reproducible runner for LAB-2046 minimal 3-node mechanism (B vs C vs O)."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import simpy
import yaml
from experiments.lab2046_common import ZeroCostAtomicWorld, git_head_sha, sha256_bytes, sha256_file
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, request_table, summarize
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


def _trace_data(horizon_s: float, *, with_eclipse: bool = True) -> dict:
    return {
        "schema": SCHEMA,
        "name": "lab2046-minimal-3node",
        "horizon_s": float(horizon_s),
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1-Hotspot", "role": "compute"},
            {"id": 2, "name": "CS-2-CoolSunlit", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, float(horizon_s)]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, float(horizon_s)]]},
        ],
        "dark": [[], [[90.0, 210.0]], []] if with_eclipse else [[], [], []],
    }


def build_frozen_requests(spec: dict) -> list[Request]:
    return [
        Request(
            i=int(item["i"]),
            origin=int(item["origin"]),
            t=float(item["t"]),
            long=bool(item["long"]),
            pin=int(item["pin"]),
            pout=int(item["pout"]),
            upload_bytes=float(item["upload_MB"]) * 1e6,
            deadline=float(item["deadline"]),
        )
        for item in spec["requests"]
    ]


class IdealSchedulePlanner:
    """Explicit, hand-verifiable ideal schedule for the 3-node minimal scenario."""

    requires_forecast = True

    def __init__(self, mig_map: dict[int, int]) -> None:
        self.mig_map = dict(mig_map)

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        orders: list[MigrationOrder] = []
        for j in view.compute_nodes:
            for r in view.node_requests(j):
                if r.state == "decode" and r.i in self.mig_map:
                    dst = self.mig_map[r.i]
                    if r.node != dst and view.fits(dst, r) and r.n_mig < (invariants.max_mig_per_req or 1 << 30):
                        orders.append(MigrationOrder(req_id=r.i, dst=dst))
                        if len(orders) >= invariants.max_inflight:
                            return PlanAction(migrations=tuple(orders))
        return PlanAction(migrations=tuple(orders))


def run_arm(  # noqa: PLR0913
    arm_id: str,
    label: str,
    spec: dict,
    trace: Trace,
    *,
    world_cls: type[CoWorld],
    planner_obj: Any,
    invariants: RebalanceInvariants,
) -> dict[str, Any]:
    env = simpy.Environment()
    rec = Recorder(float(spec.get("frame_s", 10.0)))
    policy = CompositePolicy(get_placer("nearest"), planner_obj, invariants)
    w = world_cls(env, trace, spec, policy=policy, recorder=rec, seed=1)
    rec.attach(w)
    cs_telemetry: list[dict[str, Any]] = []

    def _sample_pt():
        while True:
            cs_telemetry.append(
                {
                    "t": round(env.now, 2),
                    "cs1_temp_C": round(w.pt.temp_C[1], 2),
                    "cs1_tier": w.pt.tier_idx[1],
                    "cs1_dod_rem_j": round(w.pt.dod_remaining_j[1], 1),
                    "cs1_eclipse": trace.is_eclipse(1, env.now),
                    "cs2_temp_C": round(w.pt.temp_C[2], 2),
                    "cs2_tier": w.pt.tier_idx[2],
                    "cs2_eclipse": trace.is_eclipse(2, env.now),
                }
            )
            yield env.timeout(10.0)

    env.process(_sample_pt())
    for r in build_frozen_requests(spec):
        w.submit(r)
    env.run(until=trace.horizon)
    kv_chk = w.check_kv_invariants(at_end=True) if isinstance(w, ZeroCostAtomicWorld) else None
    return {
        "arm": arm_id,
        "label": label,
        "metrics": summarize(w),
        "requests": request_table(w),
        "events": rec.events,
        "cs_telemetry": cs_telemetry,
        "kv_conservation": kv_chk,
    }


def main() -> None:  # noqa: PLR0915
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = here / "spec.yaml"
    spec_raw = spec_path.read_bytes()
    spec = yaml.safe_load(spec_raw)

    h_s = float(spec["horizon_s"])
    trace = Trace(_trace_data(h_s, with_eclipse=True))
    trace_no_ecl = Trace(_trace_data(h_s, with_eclipse=False))
    assert trace.is_eclipse(1, 100.0)
    assert not trace_no_ecl.is_eclipse(1, 100.0)

    inv_default = RebalanceInvariants.from_cfg(spec)
    inv_unlim = RebalanceInvariants(interval_s=5.0, max_inflight=999, max_mig_per_req=999)

    arms: list[dict[str, Any]] = []
    arms.append(
        run_arm(
            "B",
            "B: Static (nearest + none)",
            copy.deepcopy(spec),
            trace,
            world_cls=CoWorld,
            planner_obj=get_planner("none", spec),
            invariants=inv_default,
        )
    )

    spec_ct = copy.deepcopy(spec)
    spec_ct["thermal"]["T_max_C"] = 999.0
    arms.append(
        run_arm(
            "C_thermal",
            "C_thermal: Lift T_max (999C), keep DoD & eclipse",
            spec_ct,
            trace,
            world_cls=CoWorld,
            planner_obj=get_planner("none", spec_ct),
            invariants=inv_default,
        )
    )

    spec_cdod = copy.deepcopy(spec)
    spec_cdod["power"]["eclipse_dod_budget_j"] = None
    arms.append(
        run_arm(
            "C_dod_only",
            "C_dod_only: Lift DoD budget only, keep eclipse & T_max=85C",
            spec_cdod,
            trace,
            world_cls=CoWorld,
            planner_obj=get_planner("none", spec_cdod),
            invariants=inv_default,
        )
    )

    arms.append(
        run_arm(
            "C_eclipse_all",
            "C_eclipse_all: Rebuild Trace without eclipse & lift DoD, keep T_max=85C",
            spec_cdod,
            trace_no_ecl,
            world_cls=CoWorld,
            planner_obj=get_planner("none", spec_cdod),
            invariants=inv_default,
        )
    )

    spec_cb = copy.deepcopy(spec_ct)
    spec_cb["power"]["eclipse_dod_budget_j"] = None
    arms.append(
        run_arm(
            "C_both",
            "C_both: Lift T_max & all eclipse/DoD constraints",
            spec_cb,
            trace_no_ecl,
            world_cls=CoWorld,
            planner_obj=get_planner("none", spec_cb),
            invariants=inv_default,
        )
    )

    best_o: dict[str, Any] | None = None
    best_key: tuple[float, float] = (-1.0, -1.0)
    for mask in range(16):
        mig_map = {i: 2 for i in range(4) if (mask >> i) & 1}
        res = run_arm(
            f"O_mask_{mask}",
            f"O_mask_{mask}",
            copy.deepcopy(spec),
            trace,
            world_cls=ZeroCostAtomicWorld,
            planner_obj=IdealSchedulePlanner(mig_map),
            invariants=inv_default,
        )
        m = res["metrics"]
        key = (float(m["good_long_frac"]), float(m["goodput_tok_s"]))
        if key > best_key:
            best_key = key
            best_o = res
            best_o["mig_map"] = mig_map

    assert best_o is not None
    best_o["arm"] = "O_quota"
    best_o["label"] = f"O: Ideal Zero-Cost Schedule (mig->CS2={sorted(best_o['mig_map'])}, quota)"
    arms.append(best_o)

    o_unlim = run_arm(
        "O_unlim",
        f"O_unlim: Ideal Zero-Cost Schedule (mig->CS2={sorted(best_o['mig_map'])}, unlim)",
        copy.deepcopy(spec),
        trace,
        world_cls=ZeroCostAtomicWorld,
        planner_obj=IdealSchedulePlanner(best_o["mig_map"]),
        invariants=inv_unlim,
    )
    arms.append(o_unlim)

    b_m = arms[0]["metrics"]
    summary_rows: list[dict[str, Any]] = []
    for a in arms:
        m = a["metrics"]
        summary_rows.append(
            {
                "arm": a["arm"],
                "label": a["label"],
                "goodput_tok_s": m["goodput_tok_s"],
                "long_slo_pct": m["good_long_frac"] * 100.0,
                "badput_tok_s": m["badput_tok_s"],
                "undelivered_pct": m["undelivered_frac"] * 100.0,
                "e2e_p95_all_s": m["e2e_p95_all_s"],
                "peak_temp_C": m["peak_temp_C"],
                "dod_exhausted_s": m["dod_exhausted_s"],
                "energy_kj": m["energy_kj"],
            }
        )

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    b_reqs = {r[0]: r for r in arms[0]["requests"]}
    o_reqs = {r[0]: r for r in best_o["requests"]}
    spec_deadlines = {int(item["i"]): float(item["deadline"]) for item in spec["requests"]}
    mig_events = {ev[2]["req"]: ev[0] for ev in best_o["events"] if len(ev) >= 3 and ev[1] == "mig_atomic"}
    timeline_rows: list[dict[str, Any]] = []
    for a in arms:
        for r in a["requests"]:
            ri = r[0]
            timeline_rows.append(
                {
                    "arm": a["arm"],
                    "req_i": ri,
                    "t_arr": r[3],
                    "t_admit": r[4],
                    "deadline": spec_deadlines[ri],
                    "trail": "->".join(str(x) for x in r[8]),
                    "t_mig": mig_events.get(ri, "") if a["arm"].startswith("O") else "",
                    "t_delivered": r[6],
                    "good": bool(r[7]),
                    "rescued_vs_B": (a["arm"].startswith("O") and not bool(b_reqs[ri][7]) and bool(r[7])),
                }
            )
    with (out_dir / "per_request_timeline.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(timeline_rows[0].keys()))
        w.writeheader()
        w.writerows(timeline_rows)

    (out_dir / "arms_raw.json").write_text(json.dumps(arms, indent=2), encoding="utf-8")

    manifest = {
        "git_sha": git_head_sha(here.parent.parent),
        "runner_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(here.parent / "lab2046_common.py"),
        "spec_sha256": sha256_bytes(spec_raw),
        "requests_sha256": sha256_bytes(json.dumps(spec["requests"], sort_keys=True).encode()),
        "best_o_mig_map": best_o["mig_map"],
        "kv_conservation_O_quota": best_o["kv_conservation"],
        "rerun_cmd": "PYTHONPATH=. uv run --frozen python experiments/lab2046_minimal_mechanism/run.py",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Minimal 3-Node Mechanism Summary ===")
    for row in summary_rows:
        dg = ((row["goodput_tok_s"] / b_m["goodput_tok_s"] - 1.0) * 100.0) if b_m["goodput_tok_s"] > 0 else 0.0
        print(
            f"  {row['arm']:14s} | Good={row['goodput_tok_s']:7.2f} ({dg:+6.1f}%) | "
            f"SLO={row['long_slo_pct']:5.1f}% | Bad={row['badput_tok_s']:7.2f} | "
            f"Undel={row['undelivered_pct']:5.1f}% | P95={row['e2e_p95_all_s']:5.1f}s | "
            f"Tmax={row['peak_temp_C']:5.1f}C | DoD_exh={row['dod_exhausted_s']:5.1f}s"
        )
    print("O_quota KV conservation:", best_o["kv_conservation"])
    print("\n=== Rescued Requests (O_quota vs B) ===")
    for i in sorted(b_reqs):
        print(
            f"  req {i}: B(good={bool(b_reqs[i][7])}, t_del={b_reqs[i][6]}, trail={b_reqs[i][8]}) "
            f"-> O(good={bool(o_reqs[i][7])}, t_del={o_reqs[i][6]}, trail={o_reqs[i][8]})"
        )


if __name__ == "__main__":
    main()
