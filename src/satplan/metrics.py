"""Playback recording, canonical metrics, and satsim.cosim_playback/v1 export."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from satplan.trace import Trace
    from satplan.world import CoWorld


@dataclass
class RunResult:
    metrics: dict[str, Any]
    frames: list[dict]
    events: list[list]
    good_curve: list[list[float]]
    requests: list[list]


class Recorder:
    def __init__(self, frame_s: float):
        self.frame_s = float(frame_s)
        self.frames: list[dict] = []
        self.events: list[list] = []

    def event(self, t: float, kind: str, **kw) -> None:
        self.events.append([round(t, 2), kind, kw])

    def attach(self, w: CoWorld) -> None:
        w.env.process(self._sample(w))

    def _sample(self, w: CoWorld):
        n = w.trace.n
        while True:
            load = [0] * n
            kv = [0.0] * n
            for j in w.trace.compute:
                load[j] = len(w.nodes[j])
                kv[j] = round(min(1.0, w.kv_used(j) / w.kv_cap), 3)
            w.net.advance(w.env.now)
            rates = sorted(w.net.edge_rates().items())
            self.frames.append({
                "t": round(w.env.now, 1),
                "load": load,
                "kv": kv,
                "e": [[e, round(r * 8 / 1e9, 3)] for e, r in rates if r > 0],
            })
            yield w.env.timeout(self.frame_s)


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(w: CoWorld) -> dict[str, Any]:
    rs = w.reqs
    longs = [r for r in rs if r.long]
    delivered = [r for r in rs if r.t_delivered is not None]
    good = [r for r in rs if w.good(r)]
    bad = [r for r in rs if not w.good(r)]
    window = float(w.cfg["workload"]["arrival_window_s"])
    goodput_gen = sum(r.gen for r in good)
    badput_gen = sum(r.gen for r in bad)
    horizon = w.trace.horizon
    lat_all = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in rs]
    lat_long_all = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in longs]
    return {
        "n_req": len(rs),
        "n_long": len(longs),
        "delivered": len(delivered),
        "good": len(good),
        "good_long": sum(1 for r in good if r.long),
        "good_frac": len(good) / max(1, len(rs)),
        "good_long_frac": sum(1 for r in good if r.long) / max(1, len(longs)),
        "goodput_tok_s": sum(r.pout for r in good) / window,
        "badput_tok_s": badput_gen / window,
        "waste_ratio": badput_gen / (goodput_gen + badput_gen) if goodput_gen + badput_gen > 0 else 0.0,
        "undelivered_frac": sum(1 for r in rs if r.t_delivered is None) / max(1, len(rs)),
        "e2e_p95_all_s": _pct(lat_all, 95),
        "long_p95_all_s": _pct(lat_long_all, 95),
        "peak_temp_C": w.peak_temp_C,
        "upload_fail": w.acc["upload_fail"],
        "link_GB_hops": w.net.bytes_hops / 1e9,
    }


def cumulative_good(w: CoWorld, step: float) -> list[list[float]]:
    ts = sorted(r.t_delivered for r in w.reqs if w.good(r))
    out, k, t = [], 0, 0.0
    while t <= w.trace.horizon + 1e-9:
        while k < len(ts) and ts[k] <= t:
            k += 1
        out.append([round(t, 1), k])
        t += step
    return out


def request_table(w: CoWorld) -> list[list]:
    def r2(x: float | None) -> float | None:
        return None if x is None else round(x, 1)

    return [
        [r.i, r.origin, int(r.long), r2(r.t), r2(r.t_admit), r2(r.t_decoded), r2(r.t_delivered), int(w.good(r)), r.trail]
        for r in w.reqs
    ]


def _trace_payload(cfg: dict, trace: Trace) -> dict:
    if "_dir" in cfg and "trace" in cfg:
        raw = json.loads((Path(cfg["_dir"]) / cfg["trace"]).resolve().read_text(encoding="utf-8"))
        return {k: raw[k] for k in ("nodes", "edges", "dark", "horizon_s", "stats") if k in raw}
    edges = [{"u": u, "v": v, "kind": k, "up": up} for (u, v), k, up in zip(trace.edges, trace.kind, trace.up, strict=True)]
    return {"nodes": trace.nodes, "edges": edges, "dark": trace.eclipse, "horizon_s": trace.horizon, "stats": {}}


def _provenance(cfg: dict, seed: int) -> dict:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    prov: dict[str, Any] = {
        "config_sha256": hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest(),
        "seeds": cfg.get("seeds", [seed]),
        "record_seed": seed,
    }
    if "_dir" in cfg and "trace" in cfg:
        tp = (Path(cfg["_dir"]) / cfg["trace"]).resolve()
        if tp.is_file():
            prov["trace_sha256"] = hashlib.sha256(tp.read_bytes()).hexdigest()
    if "_dir" in cfg and "scenario" in cfg:
        sp = (Path(cfg["_dir"]) / cfg["scenario"]).resolve()
        if sp.is_file():
            prov["scenario_sha256"] = hashlib.sha256(sp.read_bytes()).hexdigest()
    return prov


def write_playback(
    cfg: dict,
    trace: Trace,
    result: RunResult,
    policy_name: str,
    viz_dir: Path,
    seed: int = 1,
) -> Path:
    scenario = {}
    if "_dir" in cfg and "scenario" in cfg:
        sp = (Path(cfg["_dir"]) / cfg["scenario"]).resolve()
        if sp.is_file():
            scenario = json.loads(sp.read_text(encoding="utf-8"))
    runs = {
        policy_name: {
            "summary": result.metrics,
            "frames": result.frames,
            "events": [e for e in result.events if e[1] not in ("arrive", "admit")],
            "good_curve": result.good_curve,
        }
    }
    body = {
        "schema": "satsim.cosim_playback/v1",
        "name": cfg["name"],
        "provenance": _provenance(cfg, seed),
        "title": cfg.get("title", cfg["name"]),
        "description": cfg.get("description", ""),
        "takeaway": cfg.get("takeaway", ""),
        "delivery": cfg["delivery"],
        "frame_s": float(cfg.get("frame_s", 10.0)),
        "link_gbps": cfg["net"]["link_gbps"],
        "scenario": scenario,
        "trace": _trace_payload(cfg, trace),
        "policies": [policy_name],
        "mean": {policy_name: result.metrics},
        "runs": runs,
    }
    viz_dir.mkdir(parents=True, exist_ok=True)
    out = viz_dir / f"{cfg['name']}.json"
    out.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    idx_path = viz_dir / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    idx = [e for e in idx if e["name"] != cfg["name"]]
    idx.append({"name": cfg["name"], "title": body["title"], "policies": body["policies"]})
    idx.sort(key=lambda e: e["name"])
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
