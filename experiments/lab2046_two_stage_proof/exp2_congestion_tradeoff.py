"""Verification 2: Shared-link congestion cost of simple reallocation & comm-aware recovery."""
from __future__ import annotations

import copy
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import simpy
import yaml
from experiments.lab2046_common import ReplayPlacer, git_head_sha, sha256_bytes, sha256_file
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, _pct, request_table, summarize
from satplan.net import FluidNet, allocate_max_min_rates
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.runner import load_config
from satplan.trace import INF, SCHEMA, Trace
from satplan.world import CoWorld, Request, make_requests


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class NoShareMigFluidNet(FluidNet):
    """Diagnostic privilege: migration flows neither share with each other nor contend with upload flows."""

    def _allocate(self) -> None:
        for f in self.flows:
            f.rate = 0.0
        if not self.flows:
            return
        up_items = [(i, f.hops) for i, f in enumerate(self.flows) if f.tag != "mig"]
        if up_items:
            up_rates = allocate_max_min_rates(up_items, lambda e: self.cap[e])
            for i, rate in up_rates.items():
                self.flows[i].rate = rate
        for f in self.flows:
            if f.tag == "mig" and f.hops:
                f.rate = min(self.cap[e] for _, _, e in f.hops)


class NoShareDiagWorld(CoWorld):
    """CoWorld using NoShareMigFluidNet (privileged: cancels mig-mig and mig-up contention)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net.__class__ = NoShareMigFluidNet


def _exp2_trace(horizon_s: float = 200.0) -> Trace:
    data = {
        "schema": SCHEMA,
        "name": "lab2046-exp2-shared-bottleneck",
        "horizon_s": float(horizon_s),
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1-Hotspot", "role": "compute"},
            {"id": 2, "name": "CS-2-CoolPrimary", "role": "compute"},
            {"id": 3, "name": "CS-3-CoolAlt", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, float(horizon_s)]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, float(horizon_s)]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, float(horizon_s)]]},
        ],
        "dark": [[], [[50.0, float(horizon_s)]], [], []],
    }
    return Trace(data)


def _exp2_reqs(cfg: dict, *, count: int) -> list[Request]:
    tpl = cfg["requests_template"]
    return [
        Request(
            i=i,
            origin=int(tpl["origin"]),
            t=float(i),
            long=bool(tpl["long"]),
            pin=int(tpl["pin"]),
            pout=int(tpl["pout"]),
            upload_bytes=float(tpl["upload_MB"]) * 1e6,
            deadline=float(tpl["base_deadline"]) + i,
        )
        for i in range(count)
    ]


class _SimpleAllToCS2Planner:
    def __init__(self, *, max_reqs: int = 4) -> None:
        self.max_reqs = max_reqs

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        if not view.is_eclipse(1):
            return PlanAction()
        orders: list[MigrationOrder] = []
        for r in view.node_requests(1):
            if r.state == "decode" and r.n_mig == 0 and r.mig_dst is None and view.fits(2, r):
                orders.append(MigrationOrder(req_id=r.i, dst=2))
                if len(orders) >= min(self.max_reqs, invariants.max_inflight):
                    break
        return PlanAction(migrations=tuple(orders))


class _CommAwareSpreadPlanner:
    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        if not view.is_eclipse(1):
            return PlanAction()
        reqs = [r for r in view.node_requests(1) if r.state == "decode" and r.n_mig == 0 and r.mig_dst is None]
        reqs.sort(key=lambda r: r.deadline)
        orders: list[MigrationOrder] = []
        added_load = {2: 0, 3: 0}
        for r in reqs:
            if len(orders) >= invariants.max_inflight:
                break
            cands = [dst for dst in (2, 3) if view.fits(dst, r)]
            if not cands:
                continue
            dst = min(cands, key=lambda d: (view.node_load(d) + added_load[d], d))
            orders.append(MigrationOrder(req_id=r.i, dst=dst))
            added_load[dst] += 1
        return PlanAction(migrations=tuple(orders))


def _mig_stats_from_events(events: list[list[Any]]) -> dict[str, Any]:
    """Track per-attempt migration durations cleanly using (req_id, attempt_idx)."""
    active_attempt: dict[int, tuple[int, float]] = {}
    attempt_counts: dict[int, int] = defaultdict(int)
    attempts: list[dict[str, Any]] = []
    unlocked_durs: list[float] = []
    for ev in events:
        if len(ev) < 3:
            continue
        t, kind, kw = float(ev[0]), ev[1], ev[2]
        if kind == "mig_start":
            rid = int(kw["req"])
            attempt_counts[rid] += 1
            active_attempt[rid] = (attempt_counts[rid], t)
        elif kind == "mig_unlock":
            rid = int(kw["req"])
            att, t0 = active_attempt.pop(rid, (attempt_counts[rid], t))
            dur = t - t0
            unlocked_durs.append(dur)
            attempts.append({"req": rid, "attempt": att, "t_start": t0, "t_end": t, "status": "unlocked", "dur_s": dur})
        elif kind in ("mig_fail", "mig_abort"):
            rid = int(kw["req"])
            att, t0 = active_attempt.pop(rid, (attempt_counts[rid], t))
            attempts.append({"req": rid, "attempt": att, "t_start": t0, "t_end": t, "status": kind, "dur_s": t - t0})
    for rid, (att, t0) in active_attempt.items():
        attempts.append({"req": rid, "attempt": att, "t_start": t0, "t_end": None, "status": "unfinished", "dur_s": None})
    unfinished_or_failed = sum(1 for a in attempts if a["status"] != "unlocked")
    return {
        "mig_unlocked_count": len(unlocked_durs),
        "mig_unfinished_or_failed_count": unfinished_or_failed,
        "mig_unlocked_p95_s": _pct(unlocked_durs, 95) if unlocked_durs else None,
        "attempts": attempts,
    }


def _run_min_exp2(
    arm: str,
    cfg: dict,
    tr: Trace,
    reqs: list[Request],
    planner_obj: Any,
    *,
    world_cls: type[CoWorld] = CoWorld,
) -> dict[str, Any]:
    env = simpy.Environment()
    rec = Recorder(float(cfg.get("frame_s", 2.0)))
    inv = RebalanceInvariants.from_cfg(cfg)
    targets = {r.i: 1 for r in reqs}
    w = world_cls(env, tr, cfg, policy=CompositePolicy(ReplayPlacer(targets), planner_obj, inv), recorder=rec, seed=1)
    rec.attach(w)
    bn_timeline: list[dict[str, Any]] = []
    peak_mig_flows_e1 = 0

    def _probe():
        nonlocal peak_mig_flows_e1
        while True:
            mig_e1 = [f for f in w.net.flows if f.tag == "mig" and any(e == 1 for _, _, e in f.hops)]
            up_e1 = [f for f in w.net.flows if f.tag == "up" and any(e == 1 for _, _, e in f.hops)]
            peak_mig_flows_e1 = max(peak_mig_flows_e1, len(mig_e1))
            if 48.0 <= env.now <= 90.0:
                bn_timeline.append(
                    {
                        "t": round(env.now, 1),
                        "mig_flows_e1": len(mig_e1),
                        "up_flows_e1": len(up_e1),
                        "per_flow_MBps": round(mig_e1[0].rate / 1e6, 2) if mig_e1 else 0.0,
                        "total_mig_rem_MB": round(sum(f.remaining for f in mig_e1) / 1e6, 1),
                    }
                )
            yield env.timeout(2.0)

    env.process(_probe())
    for r in reqs:
        w.submit(r)
    env.run(until=tr.horizon)
    m = summarize(w)
    m["censored_completion_count"] = sum(1 for r in w.reqs if r.t_delivered is None)
    ms = _mig_stats_from_events(rec.events)
    return {
        "arm": arm,
        "metrics": m,
        "mig_stats": ms,
        "peak_mig_flows_e1": peak_mig_flows_e1,
        "bn_timeline": bn_timeline,
        "requests": request_table(w),
    }


def run_minimal_exp2(spec: dict, out_dir: Path) -> None:
    cfg = copy.deepcopy(spec["exp2_minimal"])
    cfg["name"] = "exp2-congestion"
    cfg["delivery"] = "final"
    cfg["rs_downlink"] = False
    tr = _exp2_trace(float(cfg["horizon_s"]))
    solo = _run_min_exp2("Solo_1_Mig", cfg, tr, _exp2_reqs(cfg, count=1), _SimpleAllToCS2Planner(max_reqs=1))
    main_arms = [
        _run_min_exp2("B_none", cfg, tr, _exp2_reqs(cfg, count=4), get_planner("none", cfg)),
        _run_min_exp2("Simple_Greedy", cfg, tr, _exp2_reqs(cfg, count=4), _SimpleAllToCS2Planner(max_reqs=4)),
        _run_min_exp2(
            "Diag_NoShare",
            cfg,
            tr,
            _exp2_reqs(cfg, count=4),
            _SimpleAllToCS2Planner(max_reqs=4),
            world_cls=NoShareDiagWorld,
        ),
        _run_min_exp2("CommAware_Spread", cfg, tr, _exp2_reqs(cfg, count=4), _CommAwareSpreadPlanner()),
    ]
    all_arms = [solo, *main_arms]
    rows = []
    for a in all_arms:
        m = a["metrics"]
        ms = a["mig_stats"]
        rows.append(
            {
                "table_group": "single_flow_diag" if a["arm"] == "Solo_1_Mig" else "main_4req",
                "arm": a["arm"],
                "n_req": m["n_req"],
                "goodput_tok_s": m["goodput_tok_s"],
                "long_slo_pct": m["good_long_frac"] * 100.0,
                "badput_tok_s": m["badput_tok_s"],
                "mig_unlocked_p95_s": round(ms["mig_unlocked_p95_s"], 2) if ms["mig_unlocked_p95_s"] is not None else "",
                "mig_unfinished_or_failed": ms["mig_unfinished_or_failed_count"],
                "peak_bn_mig_flows": a["peak_mig_flows_e1"],
                "censored_completion_count": m["censored_completion_count"],
                "e2e_p95_all_s": m["e2e_p95_all_s"],
            }
        )
    with (out_dir / "exp2_minimal_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "exp2_minimal_raw.json").write_text(json.dumps(all_arms, indent=2), encoding="utf-8")

    print("=== Exp 2 Minimal Shared-Bottleneck Congestion ===")
    for r in rows:
        print(
            f"  [{r['table_group']:16s}] {r['arm']:17s} | Good={r['goodput_tok_s']:6.1f} | SLO={r['long_slo_pct']:5.1f}% | "
            f"MigP95={str(r['mig_unlocked_p95_s']):>5s}s | PeakFlows={r['peak_bn_mig_flows']} | "
            f"CompCens={r['censored_completion_count']} | P95={r['e2e_p95_all_s']:.2f}s"
        )


class _CommAwareReactivePlanner:
    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        now = view.now
        max_mig = invariants.max_mig_per_req if invariants.max_mig_per_req is not None else (1 << 30)
        in_flight = view.inflight_migrations
        if in_flight >= invariants.max_inflight:
            return PlanAction()
        busy_edges = {e for e, rate in view.edge_rates().items() if rate > 1e-3}
        cands: list[tuple[float, Request, int]] = []
        for j in view.compute_nodes:
            if not view.is_eclipse(j):
                continue
            for r in view.node_requests(j):
                if r.state == "decode" and r.mig_dst is None and r.n_mig < max_mig:
                    cands.append((r.deadline - now, r, j))
        cands.sort(key=lambda x: x[0])
        orders: list[MigrationOrder] = []
        added_load: dict[int, int] = defaultdict(int)
        used_edges = set(busy_edges)
        for _, r, src in cands:
            if in_flight + len(orders) >= invariants.max_inflight:
                break
            best_dst: int | None = None
            best_score = INF
            for dst in view.compute_nodes:
                if dst == src or view.is_eclipse(dst) or not view.fits(dst, r):
                    continue
                p = view.path(src, dst)
                if p is None:
                    continue
                overlap = sum(1 for e in p[1] if e in used_edges)
                score = overlap * 100.0 + len(p[1]) * 10.0 + (view.node_load(dst) + added_load[dst]) * 20.0 + max(0.0, view.temp_C(dst) - 55.0)
                if score < best_score:
                    best_score = score
                    best_dst = dst
            if best_dst is not None:
                p = view.path(src, best_dst)
                assert p is not None
                if any(e in used_edges for e in p[1]) and len(orders) >= 2:
                    continue
                used_edges.update(p[1])
                added_load[best_dst] += 1
                orders.append(MigrationOrder(req_id=r.i, dst=best_dst))
        return PlanAction(migrations=tuple(orders))


def run_w53_congestion(spec: dict, out_dir: Path, here: Path) -> dict[str, Any]:
    raw_dir = out_dir / "raw_w53_exp2"
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
        for arm_name, planner_obj, w_cls in [
            ("W53_none", get_planner("none", cfg), CoWorld),
            ("W53_Simple_Reactive", get_planner("reactive", cfg), CoWorld),
            ("W53_Diag_NoShare", get_planner("reactive", cfg), NoShareDiagWorld),
            ("W53_CommAware", _CommAwareReactivePlanner(), CoWorld),
        ]:
            env = simpy.Environment()
            rec = Recorder(10.0)
            w = w_cls(env, trace, copy.deepcopy(cfg), policy=CompositePolicy(get_placer("nearest"), planner_obj, inv), recorder=rec, seed=int(seed))
            rec.attach(w)
            peak_edge_mig = 0
            edge_timeline: list[dict[str, Any]] = []

            def _probe_net(world=w, tl=edge_timeline):
                nonlocal peak_edge_mig
                while True:
                    cnt_mig: dict[int, int] = defaultdict(int)
                    cnt_up: dict[int, int] = defaultdict(int)
                    for f in world.net.flows:
                        target = cnt_mig if f.tag == "mig" else cnt_up
                        for _, _, e in f.hops:
                            target[e] += 1
                    if cnt_mig:
                        top_e, top_c = max(cnt_mig.items(), key=lambda x: x[1])
                        peak_edge_mig = max(peak_edge_mig, top_c)
                        tl.append({"t": round(world.env.now, 1), "top_edge": top_e, "mig_flows": top_c, "up_flows": cnt_up[top_e]})
                    yield world.env.timeout(5.0)

            env.process(_probe_net())
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
            ms = _mig_stats_from_events(rec.events)
            (raw_dir / f"{arm_name}__s{seed}.json").write_text(
                json.dumps(
                    {
                        "arm": arm_name,
                        "seed": int(seed),
                        "metrics": m,
                        "mig_unlocked_count": ms["mig_unlocked_count"],
                        "mig_unfinished_or_failed_count": ms["mig_unfinished_or_failed_count"],
                        "mig_unlocked_p95_s": ms["mig_unlocked_p95_s"],
                        "mig_attempts": ms["attempts"],
                        "edge_timeline": edge_timeline,
                        "requests": request_table(w),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            w53_rows.append(
                {
                    "arm": arm_name,
                    "seed": int(seed),
                    "goodput_tok_s": m["goodput_tok_s"],
                    "long_slo_pct": m["good_long_frac"] * 100.0,
                    "badput_tok_s": m["badput_tok_s"],
                    "mig_unlocked_count": ms["mig_unlocked_count"],
                    "mig_unfinished_or_failed": ms["mig_unfinished_or_failed_count"],
                    "mig_unlocked_p95_s": round(ms["mig_unlocked_p95_s"], 2) if ms["mig_unlocked_p95_s"] is not None else 0.0,
                    "peak_edge_mig_flows": peak_edge_mig,
                    "censored_dl_gt_h": m["censored_dl_gt_h"],
                    "censored_completion_count": m["censored_completion_count"],
                }
            )

    with (out_dir / "exp2_w53_congestion.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(w53_rows[0].keys()))
        w.writeheader()
        w.writerows(w53_rows)

    print("\n=== Exp 2 W53 Constrained (3-seed mean) ===")
    b_g = statistics.mean(r["goodput_tok_s"] for r in w53_rows if r["arm"] == "W53_none")
    for arm in ("W53_none", "W53_Simple_Reactive", "W53_Diag_NoShare", "W53_CommAware"):
        rs = [r for r in w53_rows if r["arm"] == arm]
        gm = statistics.mean(r["goodput_tok_s"] for r in rs)
        sm = statistics.mean(r["long_slo_pct"] for r in rs)
        bm = statistics.mean(r["badput_tok_s"] for r in rs)
        mp = statistics.mean(r["mig_unlocked_p95_s"] for r in rs)
        pf = max(r["peak_edge_mig_flows"] for r in rs)
        mc = statistics.mean(r["mig_unlocked_count"] for r in rs)
        mf = statistics.mean(r["mig_unfinished_or_failed"] for r in rs)
        print(
            f"  {arm:20s} | Good={gm:7.1f} ({(gm/b_g-1)*100:+5.1f}%) | LongSLO={sm:5.2f}% | "
            f"Bad={bm:7.1f} | MigOK={mc:5.1f} Fail={mf:4.1f} P95={mp:5.1f}s | PeakFlows={pf}"
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

    run_minimal_exp2(spec, out_dir)
    w53_meta = run_w53_congestion(spec, out_dir, here)
    mf = {
        "base_main_sha": "04bfed1f675fcb899dad352bdba7b47c79383e7d",
        "pre_run_head_sha": git_head_sha(here.parent.parent),
        "runner_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(here.parent / "lab2046_common.py"),
        "spec_sha256": sha256_bytes(spec_raw),
        "diag_noshare_note": "Privileged: cancels both mig-mig and mig-up sharing, and allows dynamic re-decision (not fixed migration plan).",
        **w53_meta,
        "rerun_cmd": "PYTHONPATH=. uv run --frozen python experiments/lab2046_two_stage_proof/exp2_congestion_tradeoff.py",
    }
    (out_dir / "exp2_manifest.json").write_text(json.dumps(mf, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
