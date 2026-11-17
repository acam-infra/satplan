"""Reproducible runner for LAB-2046 W53 validation (B vs C vs O across seeds [1, 2, 3])."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import simpy
import yaml
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import request_table, summarize
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.runner import load_config
from satplan.trace import INF, Trace
from satplan.world import CoWorld, Request, make_requests


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ZeroCostAtomicWorld(CoWorld):
    """Idealized oracle world: migration itself has 0 transfer bytes, 0 hop latency, 0 ISL bandwidth."""

    def _start_migration(self, order: MigrationOrder) -> None:
        r = self._req_by_id.get(order.req_id)
        if r is None or r.state != "decode" or r.node == order.dst or order.dst not in self.nodes:
            return
        if self.trace.path(r.node, order.dst, self.env.now, via_rs=False) is None:
            return
        if not self.fits(order.dst, r):
            return
        src = r.node
        kv_bytes = r.reserved_tokens * self.bpt
        gen_before = r.gen
        total_res_before = sum(self.reserved.values())
        self._detach_node(src, r)
        self.reserved[src] -= kv_bytes
        self.reserved[order.dst] += kv_bytes
        r.node = order.dst
        self._attach_node(order.dst, r)
        r.trail.append(order.dst)
        r.n_mig += 1
        assert r.gen == gen_before, "decode progress altered during atomic migration"
        assert abs(sum(self.reserved.values()) - total_res_before) < 1e-6, "KV reservation not conserved"
        self.log("mig_atomic", req=r.i, src=src, dst=order.dst)
        self.wake_node(src)
        self.wake_node(order.dst)


class OracleLookaheadPlanner:
    """Future-aware ideal heuristic using ForecastView + zero-cost atomic migration.

    Not claimed to be globally optimal; demonstrates feasible reallocation headroom
    under identical physical/KV/reachability constraints.
    """

    requires_forecast = True

    def __init__(self, lookahead_s: float = 90.0) -> None:
        self.lookahead_s = lookahead_s

    def _node_stressed(self, view: Any, j: int) -> bool:
        if view.is_eclipse(j):
            return True
        if view.temp_C(j) >= view.t_max_C - 4.0:
            return True
        if view.node_load(j) >= 3:
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
    placer: str,
    planner_obj: Any,
    world_cls: type[CoWorld],
    invariants: RebalanceInvariants,
    seed: int,
) -> dict[str, Any]:
    env = simpy.Environment()
    policy = CompositePolicy(get_placer(placer), planner_obj, invariants)
    w = world_cls(env, trace, cfg, policy=policy, seed=seed)
    for r in clone_requests(frozen_reqs):
        w.submit(r)
    env.run(until=trace.horizon)
    m = summarize(w)
    m["n_censored_deadline_gt_horizon"] = sum(1 for r in w.reqs if r.deadline > trace.horizon)
    m["n_delivered_total"] = sum(1 for r in w.reqs if r.t_delivered is not None)
    rows = request_table(w)
    return {"metrics": m, "requests": rows}


def make_constrained_cfg(base_cfg: dict) -> dict:
    c = copy.deepcopy(base_cfg)
    c["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
        {"name": "low_20w", "power_w": 20.0, "clock_f": 0.35},
        {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
    ]
    c["power"]["eclipse_dod_budget_j"] = 100000.0
    c["thermal"]["T0_C"] = 25.0
    c["thermal"]["T_amb_C"] = 20.0
    c["thermal"]["T_max_C"] = 85.0
    c["thermal"]["R_th"] = 0.25
    c["net"]["link_gbps"]["inter"] = 0.35
    return c


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_raw = (here / "spec.yaml").read_bytes()
    spec = yaml.safe_load(spec_raw)

    base_cfg_path = here.parent.parent / "configs" / "w53_baseline.yaml"
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

    inv_quota = RebalanceInvariants.from_cfg(spec)
    inv_unlim = RebalanceInvariants(interval_s=5.0, max_inflight=999, max_mig_per_req=999)

    scenarios = {
        "w53_original": base_cfg,
        "w53_constrained": make_constrained_cfg(base_cfg),
    }

    all_rows: list[dict[str, Any]] = []
    for sc_name, sc_cfg in scenarios.items():
        for seed in spec["seeds"]:
            frozen_reqs = make_requests(sc_cfg, trace, int(seed))
            max_dl = max(r.deadline for r in frozen_reqs)
            assert max_dl <= trace.horizon, f"deadline {max_dl} > horizon {trace.horizon}"

            arms_spec = [
                ("B_nearest", sc_cfg, trace, "nearest", get_planner("none", sc_cfg), CoWorld, inv_quota),
                ("B_least", sc_cfg, trace, "least_loaded", get_planner("none", sc_cfg), CoWorld, inv_quota),
                (
                    "C_thermal",
                    {**copy.deepcopy(sc_cfg), "thermal": {**sc_cfg["thermal"], "T_max_C": 999.0}},
                    trace,
                    "nearest",
                    get_planner("none", sc_cfg),
                    CoWorld,
                    inv_quota,
                ),
                (
                    "C_eclipse",
                    {**copy.deepcopy(sc_cfg), "power": {**sc_cfg["power"], "eclipse_dod_budget_j": None}},
                    trace_no_ecl,
                    "nearest",
                    get_planner("none", sc_cfg),
                    CoWorld,
                    inv_quota,
                ),
                (
                    "C_both",
                    {
                        **copy.deepcopy(sc_cfg),
                        "thermal": {**sc_cfg["thermal"], "T_max_C": 999.0},
                        "power": {**sc_cfg["power"], "eclipse_dod_budget_j": None},
                    },
                    trace_no_ecl,
                    "nearest",
                    get_planner("none", sc_cfg),
                    CoWorld,
                    inv_quota,
                ),
                (
                    "O_nearest_quota",
                    sc_cfg,
                    trace,
                    "nearest",
                    OracleLookaheadPlanner(),
                    ZeroCostAtomicWorld,
                    inv_quota,
                ),
                (
                    "O_nearest_unlim",
                    sc_cfg,
                    trace,
                    "nearest",
                    OracleLookaheadPlanner(),
                    ZeroCostAtomicWorld,
                    inv_unlim,
                ),
                (
                    "O_least_quota",
                    sc_cfg,
                    trace,
                    "least_loaded",
                    OracleLookaheadPlanner(),
                    ZeroCostAtomicWorld,
                    inv_quota,
                ),
            ]

            for arm_name, a_cfg, a_tr, placer, planner_obj, w_cls, inv in arms_spec:
                res = run_arm(
                    a_cfg,
                    a_tr,
                    frozen_reqs,
                    placer=placer,
                    planner_obj=planner_obj,
                    world_cls=w_cls,
                    invariants=inv,
                    seed=int(seed),
                )
                m = res["metrics"]
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
                        "censored_dl_gt_h": m["n_censored_deadline_gt_horizon"],
                    }
                )

    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here.parent.parent, text=True).strip()
    manifest = {
        "git_sha": git_sha,
        "spec_sha256": sha256_bytes(spec_raw),
        "base_cfg_sha256": sha256_bytes(base_cfg_path.read_bytes()),
        "trace_sha256": sha256_bytes(trace_path.read_bytes()),
        "rerun_cmd": "uv run --frozen python experiments/lab2046_w53_validation/run.py",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Print paired summary
    for sc_name in scenarios:
        print(f"\n=== {sc_name} (3-seed mean, horizon={trace.horizon}s, arrival={spec['arrival_window_s']}s) ===")
        sc_rows = [r for r in all_rows if r["scenario"] == sc_name]
        b_by_seed = {r["seed"]: r for r in sc_rows if r["arm"] == "B_nearest"}
        bl_by_seed = {r["seed"]: r for r in sc_rows if r["arm"] == "B_least"}
        arm_order = [
            "B_nearest",
            "B_least",
            "C_thermal",
            "C_eclipse",
            "C_both",
            "O_nearest_quota",
            "O_nearest_unlim",
            "O_least_quota",
        ]
        for arm in arm_order:
            rs = [r for r in sc_rows if r["arm"] == arm]
            g_m = statistics.mean(r["goodput_tok_s"] for r in rs)
            s_m = statistics.mean(r["long_slo_pct"] for r in rs)
            bad_m = statistics.mean(r["badput_tok_s"] for r in rs)
            u_m = statistics.mean(r["undelivered_pct"] for r in rs)
            p95_m = statistics.mean(r["e2e_p95_all_s"] for r in rs)
            tmax_m = max(r["peak_temp_C"] for r in rs)
            dod_m = statistics.mean(r["dod_exhausted_s"] for r in rs)
            bg_m = statistics.mean(b_by_seed[s]["goodput_tok_s"] for s in b_by_seed)
            dg_pct = (g_m / bg_m - 1.0) * 100.0 if bg_m > 0 else 0.0
            extra = ""
            if arm == "O_least_quota":
                bl_g = statistics.mean(bl_by_seed[s]["goodput_tok_s"] for s in bl_by_seed)
                extra = f" | vs B_least={(g_m/bl_g - 1.0)*100.0:+6.1f}%"
            print(
                f"  {arm:16s} | Good={g_m:7.1f} ({dg_pct:+6.1f}%) | LongSLO={s_m:5.1f}% | "
                f"Bad={bad_m:7.1f} | Undel={u_m:5.1f}% | P95={p95_m:5.1f}s | Tmax={tmax_m:5.1f}C | DoD={dod_m:6.1f}s{extra}"
            )


if __name__ == "__main__":
    main()
