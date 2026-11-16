"""Simulation entry point: load_config, run_simulation, and CLI main."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import simpy
import yaml

from satplan.metrics import Recorder, RunResult, cumulative_good, request_table, summarize, write_playback
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.trace import Trace
from satplan.world import CoWorld, make_requests

REQUIRED = {
    "name",
    "trace",
    "model",
    "gpu",
    "power",
    "net",
    "workload",
    "slo",
    "delivery",
    "control_period_s",
}


def load_config(path: Path, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = REQUIRED - set(cfg)
    if missing:
        raise SystemExit(f"satplan config missing {sorted(missing)}")
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *head, last = key.split(".")
        for k in head:
            node = node[k]
        node[last] = yaml.safe_load(raw)
    if cfg["delivery"] not in ("final", "stream"):
        raise SystemExit("delivery must be final | stream")
    cfg["_dir"] = str(path.parent.resolve())
    return cfg


def run_simulation(
    trace: Trace,
    cfg: dict,
    placer: str,
    planner: str,
    seed: int,
    *,
    record: bool = False,
) -> RunResult:
    env = simpy.Environment()
    frame_s = float(cfg.get("frame_s", 10.0))
    rec = Recorder(frame_s) if record else None
    policy = CompositePolicy(get_placer(placer), get_planner(planner), RebalanceInvariants.from_cfg(cfg))
    w = CoWorld(
        env,
        trace,
        cfg,
        policy=policy,
        recorder=rec,
        seed=seed,
    )
    if rec is not None:
        rec.attach(w)
    for r in make_requests(cfg, trace, seed):
        w.submit(r)
    env.run(until=trace.horizon)
    return RunResult(
        metrics=summarize(w),
        frames=rec.frames if rec is not None else [],
        events=rec.events if rec is not None else [],
        good_curve=cumulative_good(w, frame_s * 3) if rec is not None else [],
        requests=request_table(w) if rec is not None else [],
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="satplan tracer-bullet runner")
    ap.add_argument("config", help="YAML config path")
    ap.add_argument("--out", default="", help="output directory for summary.json")
    ap.add_argument("--viz", default="", help="directory for satsim.cosim_playback/v1 JSON")
    ap.add_argument("--set", action="append", default=[], help="override key=val")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config), args.set)
    trace = Trace.load((Path(cfg["_dir"]) / cfg["trace"]).resolve())
    placer = str(cfg.get("placer", "nearest"))
    planner = str(cfg.get("planner", "none"))
    seed = int(cfg.get("seed", cfg.get("seeds", [1])[0]))
    result = run_simulation(trace, cfg, placer, planner, seed, record=bool(args.viz))
    print(json.dumps(result.metrics, indent=2))
    if args.out:
        dest = Path(args.out)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    if args.viz:
        pb = write_playback(cfg, trace, result, f"{placer}+{planner}", Path(args.viz), seed=seed)
        print(f"wrote playback: {pb}")


if __name__ == "__main__":
    main()
