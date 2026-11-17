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
from experiments.lab2046_common import ReplayPlacer, git_head_sha, sha256_file
from satplan.actions import MigrationOrder, PlanAction
from satplan.metrics import Recorder, _pct, request_table, summarize
from satplan.net import FluidNet, allocate_max_min_rates
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.runner import load_config
from satplan.trace import INF, SCHEMA, Trace
from satplan.world import CoWorld, Request, make_requests


class NoShareMigFluidNet(FluidNet):
    """Diagnostic privilege: migration flows do not share bandwidth with each other.

    Upload flows still share normally; each migration flow receives its full path bottleneck
    capacity as if no other migration flow contended on shared links.
    """

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
    """CoWorld using NoShareMigFluidNet for shared-link contention attribution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net.__class__ = NoShareMigFluidNet


def _exp2_trace(horizon_s: float = 200.0) -> Trace:
    """RS-0 -> CS-1 (eclipse [50,200]); CS-1 -> CS-2 (e1) and CS-1 -> CS-3 (e2)."""
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
        "dark": [[], [[50.0, 200.0]], [], []],
    }
    return Trace(data)


def _exp2_cfg() -> dict:
    return {
        "name": "exp2-congestion",
        "delivery": "final",
        "rs_downlink": False,
        "control_period_s": 2.0,
        "frame_s": 2.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 8, "max_mig_per_req": 4},
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7, "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "P_idle": 60,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 80,
            "eclipse_dod_budget_j": 800.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
                {"name": "low_20w", "power_w": 20.0, "clock_f": 0.35},
                {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
            ],
        },
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.25, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 0.5, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {"arrival_window_s": 10.0},
        "slo": {"max_stall_frac": 0.1},
    }


def _exp2_reqs(*, count: int = 4) -> list[Request]:
    return [
        Request(i=i, origin=0, t=float(i), long=True, pin=500, pout=6000, upload_bytes=10e6, deadline=102.0 + i)
        for i in range(count)
    ]


class _SimpleAllToCS2Planner:
    """Naive realloc: only checks CS-2 headroom, ignores in-flight link sharing -> sends all to CS-2."""

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
    """Comm-aware realloc: accounts for link sharing & load, splits across CS-2 (e1) and CS-3 (e2)."""

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
    starts: dict[int, float] = {}
    durs: list[float] = []
    for ev in events:
        if len(ev) < 3:
            continue
        t, kind, kw = float(ev[0]), ev[1], ev[2]
        if kind == "mig_start":
            starts[int(kw["req"])] = t
        elif kind == "mig_unlock":
            rid = int(kw["req"])
            if rid in starts:
                durs.append(t - starts[rid])
    return {"mig_count": len(durs), "mig_done_p95_s": _pct(durs, 95) if durs else None}


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
    rec = Recorder(2.0)
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
            if env.now >= 48.0 and env.now <= 90.0:
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
    ms = _mig_stats_from_events(rec.events)
    return {
        "arm": arm,
        "metrics": m,
        "mig_stats": ms,
        "peak_mig_flows_e1": peak_mig_flows_e1,
        "bn_timeline": bn_timeline,
        "requests": request_table(w),
    }


def run_minimal_exp2(out_dir: Path) -> None:
    tr = _exp2_trace(200.0)
    cfg = _exp2_cfg()
    arms = [
        _run_min_exp2("Solo_1_Mig", cfg, tr, _exp2_reqs(count=1), _SimpleAllToCS2Planner(max_reqs=1)),
        _run_min_exp2("B_none", cfg, tr, _exp2_reqs(count=4), get_planner("none", cfg)),
        _run_min_exp2("Simple_Greedy", cfg, tr, _exp2_reqs(count=4), _SimpleAllToCS2Planner(max_reqs=4)),
        _run_min_exp2(
            "Diag_NoShare",
            cfg,
            tr,
            _exp2_reqs(count=4),
            _SimpleAllToCS2Planner(max_reqs=4),
            world_cls=NoShareDiagWorld,
        ),
        _run_min_exp2("CommAware_Spread", cfg, tr, _exp2_reqs(count=4), _CommAwareSpreadPlanner()),
    ]
    rows = []
    for a in arms:
        m = a["metrics"]
        ms = a["mig_stats"]
        rows.append(
            {
                "arm": a["arm"],
                "n_req": m["n_req"],
                "goodput_tok_s": m["goodput_tok_s"],
                "long_slo_pct": m["good_long_frac"] * 100.0,
                "badput_tok_s": m["badput_tok_s"],
                "mig_done_p95_s": ms["mig_done_p95_s"] if ms["mig_done_p95_s"] is not None else "",
                "peak_bn_mig_flows": a["peak_mig_flows_e1"],
                "e2e_p95_all_s": m["e2e_p95_all_s"],
            }
        )
    with (out_dir / "exp2_minimal_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "exp2_minimal_raw.json").write_text(json.dumps(arms, indent=2), encoding="utf-8")

    print("=== Exp 2 Minimal Shared-Bottleneck Congestion ===")
    for r in rows:
        print(
            f"  {r['arm']:18s} | n={r['n_req']} | Good={r['goodput_tok_s']:6.1f} | SLO={r['long_slo_pct']:5.1f}% | "
            f"Bad={r['badput_tok_s']:6.1f} | MigP95={str(r['mig_done_p95_s']):>5s}s | PeakFlows={r['peak_bn_mig_flows']}"
        )
    print("\n--- Exp 2 Congestion Timeline (e_12 bottleneck): Simple_Greedy vs CommAware_Spread ---")
    ts = {x["t"]: x for x in arms[2]["bn_timeline"]}
    tc = {x["t"]: x for x in arms[4]["bn_timeline"]}
    for t in sorted(ts):
        if 50.0 <= t <= 76.0:
            xs, xc = ts[t], tc[t]
            print(
                f"  t={t:4.1f}s | Simple(flows={xs['mig_flows_e1']}, rate={xs['per_flow_MBps']:5.2f} MB/s, rem={xs['total_mig_rem_MB']:6.1f} MB) "
                f"| CommAware(flows={xc['mig_flows_e1']}, rate={xc['per_flow_MBps']:5.2f} MB/s, rem={xc['total_mig_rem_MB']:6.1f} MB)"
            )


class _CommAwareReactivePlanner:
    """Reactive planner with link-congestion & deadline awareness."""

    def plan(self, view: Any, invariants: RebalanceInvariants) -> PlanAction:
        now = view.now
        max_mig = invariants.max_mig_per_req if invariants.max_mig_per_req is not None else (1 << 30)
        in_flight = view.inflight_migrations
        if in_flight >= invariants.max_inflight:
            return PlanAction()
        # Count active flows per edge from current edge_rates
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
                # Avoid launching a flow onto an already-congested edge if we can stagger
                if any(e in used_edges for e in p[1]) and len(orders) >= 2:
                    continue
                used_edges.update(p[1])
                added_load[best_dst] += 1
                orders.append(MigrationOrder(req_id=r.i, dst=best_dst))
        return PlanAction(migrations=tuple(orders))


def run_w53_congestion(out_dir: Path, here: Path) -> None:
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
        for arm_name, planner_obj, w_cls in [
            ("W53_none", get_planner("none", cfg), CoWorld),
            ("W53_Simple_Reactive", get_planner("reactive", cfg), CoWorld),
            ("W53_Diag_NoShare", get_planner("reactive", cfg), NoShareDiagWorld),
            ("W53_CommAware", _CommAwareReactivePlanner(), CoWorld),
        ]:
            env = simpy.Environment()
            rec = Recorder(10.0)
            w = w_cls(env, trace, copy.deepcopy(cfg), policy=CompositePolicy(get_placer("nearest"), planner_obj, inv), recorder=rec, seed=seed)
            rec.attach(w)
            peak_edge_mig = 0

            def _probe_net(world=w):
                nonlocal peak_edge_mig
                while True:
                    cnt: dict[int, int] = defaultdict(int)
                    for f in world.net.flows:
                        if f.tag == "mig":
                            for _, _, e in f.hops:
                                cnt[e] += 1
                    if cnt:
                        peak_edge_mig = max(peak_edge_mig, max(cnt.values()))
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
            ms = _mig_stats_from_events(rec.events)
            w53_rows.append(
                {
                    "arm": arm_name,
                    "seed": seed,
                    "goodput_tok_s": m["goodput_tok_s"],
                    "long_slo_pct": m["good_long_frac"] * 100.0,
                    "badput_tok_s": m["badput_tok_s"],
                    "mig_count": ms["mig_count"],
                    "mig_done_p95_s": ms["mig_done_p95_s"] if ms["mig_done_p95_s"] is not None else 0.0,
                    "peak_edge_mig_flows": peak_edge_mig,
                }
            )

    with (out_dir / "exp2_w53_congestion.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(w53_rows[0].keys()))
        w.writeheader()
        w.writerows(w53_rows)

    print("\n=== Exp 2 W53 Constrained (3-seed mean: Simple Reactive vs NoShare Diag vs CommAware) ===")
    b_g = statistics.mean(r["goodput_tok_s"] for r in w53_rows if r["arm"] == "W53_none")
    for arm in ("W53_none", "W53_Simple_Reactive", "W53_Diag_NoShare", "W53_CommAware"):
        rs = [r for r in w53_rows if r["arm"] == arm]
        gm = statistics.mean(r["goodput_tok_s"] for r in rs)
        sm = statistics.mean(r["long_slo_pct"] for r in rs)
        bm = statistics.mean(r["badput_tok_s"] for r in rs)
        mp = statistics.mean(r["mig_done_p95_s"] for r in rs)
        pf = max(r["peak_edge_mig_flows"] for r in rs)
        print(
            f"  {arm:20s} | Good={gm:7.1f} ({(gm/b_g-1)*100:+5.1f}%) | LongSLO={sm:5.2f}% | "
            f"Bad={bm:7.1f} | MigP95={mp:5.1f}s | PeakEdgeMigFlows={pf}"
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_minimal_exp2(out_dir)
    run_w53_congestion(out_dir, here)
    mf = {
        "pre_run_head_sha": git_head_sha(here.parent.parent),
        "runner_sha256": sha256_file(Path(__file__)),
        "rerun_cmd": "PYTHONPATH=. uv run --frozen python experiments/lab2046_two_stage_proof/exp2_congestion_tradeoff.py",
    }
    (out_dir / "exp2_manifest.json").write_text(json.dumps(mf, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
