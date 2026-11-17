"""Reproducible runner for LAB-2046 W53 validation (B vs C vs O with frozen initial placement)."""
from __future__ import annotations

import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import simpy
import yaml
from experiments.lab2046_common import (
    ReplayPlacer,
    ZeroCostAtomicWorld,
    git_head_sha,
    sha256_bytes,
    sha256_file,
)
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, request_table, summarize
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.runner import load_config
from satplan.trace import INF, Trace
from satplan.world import CoWorld, Request, make_requests


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class OracleLookaheadPlanner:
    """Future-aware ideal heuristic using ForecastView + zero-cost atomic migration."""

    requires_forecast = True

    def __init__(self, lookahead_s: float = 90.0) -> None:
        self.lookahead_s = lookahead_s

    def _node_stressed(self, view: Any, j: int) -> bool:
        if view.is_eclipse(j) or view.temp_C(j) >= view.t_max_C - 4.0 or view.node_load(j) >= 3:
            return True
        now = view.now
        return any(view.is_eclipse_at(j, now + dt) for dt in (15.0, 30.0, 60.0))

    def _pick_dst(
        self,
        view: Any,
        r: Request,
        src: int,
        added_load: dict[int, int],
        added_bytes: dict[int, float],
        r_bytes: float,
    ) -> int | None:
        now = view.now
        best_dst: int | None = None
        best_score = INF
        for dst in view.compute_nodes:
            if dst == src or not view.fits(dst, r):
                continue
            if view.kv_headroom(dst) - added_bytes.get(dst, 0.0) < r_bytes:
                continue
            if view.path(src, dst) is None:
                continue
            if view.is_eclipse(dst) or view.temp_C(dst) >= view.t_max_C - 4.0:
                continue
            ecl_pen = sum(60.0 for dt in (20.0, 45.0, 90.0) if view.is_eclipse_at(dst, now + dt))
            load_dst = view.node_load(dst) + added_load.get(dst, 0)
            t_pred = view.predict_decode_end(dst, r, now, joining=True)
            score = (t_pred - now) + ecl_pen + 15.0 * load_dst + max(0.0, view.temp_C(dst) - 55.0)
            if score < best_score:
                best_score = score
                best_dst = dst
        return best_dst

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        now = view.now
        max_mig = invariants.max_mig_per_req if invariants.max_mig_per_req is not None else (1 << 30)
        candidates: list[tuple[float, Request, int]] = []
        for j in view.compute_nodes:
            if not self._node_stressed(view, j):
                continue
            reqs = [r for r in view.node_requests(j) if r.state == "decode" and r.n_mig < max_mig and not r.paused]
            reqs.sort(key=lambda r: (0 if r.long else 1, r.deadline - now))
            candidates.extend((r.deadline - now, r, j) for r in reqs)

        candidates.sort(key=lambda x: x[0])
        orders: list[MigrationOrder] = []
        added_load: dict[int, int] = {}
        added_bytes: dict[int, float] = {}
        bpt = view.bpt
        for _, r, src in candidates:
            if len(orders) >= invariants.max_inflight:
                break
            r_bytes = r.reserved_tokens * bpt
            best_dst = self._pick_dst(view, r, src, added_load, added_bytes, r_bytes)
            if best_dst is not None:
                orders.append(MigrationOrder(req_id=r.i, dst=best_dst))
                added_load[best_dst] = added_load.get(best_dst, 0) + 1
                added_bytes[best_dst] = added_bytes.get(best_dst, 0.0) + r_bytes
        return PlanAction(migrations=tuple(orders))


def clone_requests(reqs: list[Request]) -> list[Request]:
    return [
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
        for r in reqs
    ]


def run_arm(  # noqa: PLR0913
    cfg: dict,
    trace: Trace,
    frozen_reqs: list[Request],
    *,
    placer_obj: Any,
    planner_obj: Any,
    world_cls: type[CoWorld],
    invariants: RebalanceInvariants,
    seed: int,
    expected_targets: dict[int, int] | None = None,
) -> dict[str, Any]:
    env = simpy.Environment()
    rec = Recorder(float(cfg.get("frame_s", 10.0)))
    policy = CompositePolicy(placer_obj, planner_obj, invariants)
    w = world_cls(env, trace, cfg, policy=policy, recorder=rec, seed=seed)
    rec.attach(w)
    for r in clone_requests(frozen_reqs):
        w.submit(r)
    env.run(until=trace.horizon)
    kv_chk = w.check_kv_invariants(at_end=True) if isinstance(w, ZeroCostAtomicWorld) else None
    if expected_targets is not None:
        for r in w.reqs:
            if r.trail:
                assert r.trail[0] == expected_targets[r.i], (
                    f"req {r.i} initial target {r.trail[0]} != frozen {expected_targets[r.i]}"
                )
    m = summarize(w)
    m["censored_dl_gt_h"] = sum(1 for r in w.reqs if r.deadline > trace.horizon)
    m["censored_completion_count"] = sum(1 for r in w.reqs if r.t_delivered is None)
    initial_targets = {r.i: r.trail[0] for r in w.reqs if r.trail}
    mig_events = [
        {"t": ev[0], "req": ev[2]["req"], "src": ev[2]["src"], "dst": ev[2]["dst"]}
        for ev in rec.events
        if len(ev) >= 3 and ev[1] == "mig_atomic"
    ]
    return {
        "metrics": m,
        "requests": request_table(w),
        "initial_targets": initial_targets,
        "mig_events": mig_events,
        "kv_conservation": kv_chk,
    }


def _build_arm_specs(  # noqa: PLR0913
    sc_cfg: dict,
    trace: Trace,
    trace_no_ecl: Trace,
    inv_quota: RebalanceInvariants,
    inv_unlim: RebalanceInvariants,
    lookahead_s: float,
    b_near_targets: dict[int, int],
    b_least_targets: dict[int, int],
) -> list[tuple[str, dict, Trace, Any, Any, type[CoWorld], RebalanceInvariants, dict[int, int] | None]]:
    cfg_ct = _deep_merge(sc_cfg, {"thermal": {"T_max_C": 999.0}})
    cfg_cdod = _deep_merge(sc_cfg, {"power": {"eclipse_dod_budget_j": None}})
    cfg_cboth = _deep_merge(cfg_ct, {"power": {"eclipse_dod_budget_j": None}})
    return [
        ("C_thermal", cfg_ct, trace, get_placer("nearest"), get_planner("none", cfg_ct), CoWorld, inv_quota, None),
        ("C_dod_only", cfg_cdod, trace, get_placer("nearest"), get_planner("none", cfg_cdod), CoWorld, inv_quota, None),
        (
            "C_eclipse_all",
            cfg_cdod,
            trace_no_ecl,
            get_placer("nearest"),
            get_planner("none", cfg_cdod),
            CoWorld,
            inv_quota,
            None,
        ),
        (
            "C_both",
            cfg_cboth,
            trace_no_ecl,
            get_placer("nearest"),
            get_planner("none", cfg_cboth),
            CoWorld,
            inv_quota,
            None,
        ),
        (
            "O_nearest_frozen_quota",
            sc_cfg,
            trace,
            ReplayPlacer(b_near_targets),
            OracleLookaheadPlanner(lookahead_s),
            ZeroCostAtomicWorld,
            inv_quota,
            b_near_targets,
        ),
        (
            "O_nearest_frozen_unlim",
            sc_cfg,
            trace,
            ReplayPlacer(b_near_targets),
            OracleLookaheadPlanner(lookahead_s),
            ZeroCostAtomicWorld,
            inv_unlim,
            b_near_targets,
        ),
        (
            "O_least_frozen_quota",
            sc_cfg,
            trace,
            ReplayPlacer(b_least_targets),
            OracleLookaheadPlanner(lookahead_s),
            ZeroCostAtomicWorld,
            inv_quota,
            b_least_targets,
        ),
        (
            "O_nearest_feedback_quota",
            sc_cfg,
            trace,
            get_placer("nearest"),
            OracleLookaheadPlanner(lookahead_s),
            ZeroCostAtomicWorld,
            inv_quota,
            None,
        ),
        (
            "O_least_feedback_quota",
            sc_cfg,
            trace,
            get_placer("least_loaded"),
            OracleLookaheadPlanner(lookahead_s),
            ZeroCostAtomicWorld,
            inv_quota,
            None,
        ),
    ]


def main() -> None:  # noqa: PLR0915
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spec_raw = (here / "spec.yaml").read_bytes()
    spec = yaml.safe_load(spec_raw)

    base_cfg_path = (here / spec["base_config"]).resolve()
    base_cfg = load_config(base_cfg_path)
    base_cfg["workload"]["arrival_window_s"] = float(spec["arrival_window_s"])
    base_cfg["rebalance_invariants"] = copy.deepcopy(spec["rebalance_invariants"])

    trace_path = (Path(base_cfg["_dir"]) / base_cfg["trace"]).resolve()
    trace_raw = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_raw["horizon_s"] = float(spec["horizon_s"])
    trace = Trace(trace_raw)
    trace_raw_no_ecl = copy.deepcopy(trace_raw)
    trace_raw_no_ecl["dark"] = [[] for _ in trace_raw_no_ecl["dark"]]
    trace_no_ecl = Trace(trace_raw_no_ecl)
    assert not any(trace_no_ecl.is_eclipse(j, 100.0) for j in trace_no_ecl.compute)

    inv_quota = RebalanceInvariants.from_cfg(spec)
    inv_unlim = RebalanceInvariants.from_cfg({"rebalance_invariants": spec["unlim_invariants"]})
    lookahead_s = float(spec.get("oracle_lookahead_s", 90.0))

    effective_configs: dict[str, dict] = {}
    all_rows: list[dict[str, Any]] = []
    kv_checks: dict[str, Any] = {}

    for sc_name, sc_spec in spec["scenarios"].items():
        sc_cfg = _deep_merge(base_cfg, sc_spec.get("overrides", {}))
        effective_configs[sc_name] = {k: v for k, v in sc_cfg.items() if k != "_dir"}
        for seed in spec["seeds"]:
            frozen_reqs = make_requests(sc_cfg, trace, int(seed))
            max_dl = max(r.deadline for r in frozen_reqs)
            assert max_dl <= trace.horizon, f"deadline {max_dl} > horizon {trace.horizon}"

            # Run B_nearest and B_least first to capture exact frozen initial targets
            b_near = run_arm(
                sc_cfg,
                trace,
                frozen_reqs,
                placer_obj=get_placer("nearest"),
                planner_obj=get_planner("none", sc_cfg),
                world_cls=CoWorld,
                invariants=inv_quota,
                seed=int(seed),
            )
            b_least = run_arm(
                sc_cfg,
                trace,
                frozen_reqs,
                placer_obj=get_placer("least_loaded"),
                planner_obj=get_planner("none", sc_cfg),
                world_cls=CoWorld,
                invariants=inv_quota,
                seed=int(seed),
            )

            arm_results: list[tuple[str, dict[str, Any]]] = [("B_nearest", b_near), ("B_least", b_least)]
            specs = _build_arm_specs(
                sc_cfg,
                trace,
                trace_no_ecl,
                inv_quota,
                inv_unlim,
                lookahead_s,
                b_near["initial_targets"],
                b_least["initial_targets"],
            )
            for arm_name, a_cfg, a_tr, p_obj, pl_obj, w_cls, inv, exp_t in specs:
                res = run_arm(
                    a_cfg,
                    a_tr,
                    frozen_reqs,
                    placer_obj=p_obj,
                    planner_obj=pl_obj,
                    world_cls=w_cls,
                    invariants=inv,
                    seed=int(seed),
                    expected_targets=exp_t,
                )
                arm_results.append((arm_name, res))

            for arm_name, res in arm_results:
                m = res["metrics"]
                if res["kv_conservation"] is not None:
                    kv_checks[f"{sc_name}__s{seed}__{arm_name}"] = res["kv_conservation"]
                raw_file = raw_dir / f"{sc_name}__s{seed}__{arm_name}.json"
                raw_file.write_text(
                    json.dumps(
                        {
                            "scenario": sc_name,
                            "seed": int(seed),
                            "arm": arm_name,
                            "metrics": m,
                            "kv_conservation": res["kv_conservation"],
                            "mig_events": res["mig_events"],
                            "requests": res["requests"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                all_rows.append(
                    {
                        "scenario": sc_name,
                        "arm": arm_name,
                        "seed": int(seed),
                        "n_req": m["n_req"],
                        "max_deadline_s": round(max_dl, 2),
                        "goodput_tok_s": m["goodput_tok_s"],
                        "long_slo_pct": m["good_long_frac"] * 100.0,
                        "badput_tok_s": m["badput_tok_s"],
                        "undelivered_pct": m["undelivered_frac"] * 100.0,
                        "e2e_p95_all_s": m["e2e_p95_all_s"],
                        "long_p95_all_s": m["long_p95_all_s"],
                        "peak_temp_C": m["peak_temp_C"],
                        "dod_exhausted_s": m["dod_exhausted_s"],
                        "censored_dl_gt_h": m["censored_dl_gt_h"],
                        "censored_completion_count": m["censored_completion_count"],
                    }
                )

    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    repo_dir = here.parent.parent
    manifest = {
        "base_main_sha": "04bfed1f675fcb899dad352bdba7b47c79383e7d",
        "git_head_sha": git_head_sha(repo_dir),
        "runner_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(here.parent / "lab2046_common.py"),
        "spec_sha256": sha256_bytes(spec_raw),
        "base_cfg_sha256": sha256_file(base_cfg_path),
        "trace_sha256": sha256_file(trace_path),
        "effective_configs": effective_configs,
        "kv_conservation_checks": kv_checks,
        "rerun_cmd": "PYTHONPATH=. uv run --frozen python experiments/lab2046_w53_validation/run.py",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    arm_order = [
        "B_nearest",
        "B_least",
        "C_thermal",
        "C_dod_only",
        "C_eclipse_all",
        "C_both",
        "O_nearest_frozen_quota",
        "O_nearest_frozen_unlim",
        "O_least_frozen_quota",
        "O_nearest_feedback_quota",
        "O_least_feedback_quota",
    ]
    for sc_name in spec["scenarios"]:
        print(f"\n=== {sc_name} (3-seed mean, horizon={trace.horizon}s, arrival={spec['arrival_window_s']}s) ===")
        sc_rows = [r for r in all_rows if r["scenario"] == sc_name]
        b_by_seed = {r["seed"]: r for r in sc_rows if r["arm"] == "B_nearest"}
        bl_by_seed = {r["seed"]: r for r in sc_rows if r["arm"] == "B_least"}
        for arm in arm_order:
            rs = [r for r in sc_rows if r["arm"] == arm]
            g_m = statistics.mean(r["goodput_tok_s"] for r in rs)
            s_m = statistics.mean(r["long_slo_pct"] for r in rs)
            bad_m = statistics.mean(r["badput_tok_s"] for r in rs)
            u_m = statistics.mean(r["undelivered_pct"] for r in rs)
            p95_m = statistics.mean(r["e2e_p95_all_s"] for r in rs)
            cc_sum = sum(r["censored_completion_count"] for r in rs)
            bg_m = statistics.mean(b_by_seed[s]["goodput_tok_s"] for s in b_by_seed)
            dg_pct = (g_m / bg_m - 1.0) * 100.0 if bg_m > 0 else 0.0
            extra = ""
            if arm in ("O_least_frozen_quota", "O_least_feedback_quota"):
                bl_g = statistics.mean(bl_by_seed[s]["goodput_tok_s"] for s in bl_by_seed)
                extra = f" | vs B_least={(g_m/bl_g - 1.0)*100.0:+6.1f}%"
            print(
                f"  {arm:25s} | Good={g_m:7.1f} ({dg_pct:+6.1f}%) | LongSLO={s_m:5.1f}% | "
                f"Bad={bad_m:7.1f} | Undel={u_m:5.1f}% (comp_cens={cc_sum}) | P95={p95_m:6.1f}s{extra}"
            )


if __name__ == "__main__":
    main()
