"""Seam tests for run_simulation: end-to-end tracer bullet, Store-and-Run no-drop, and playback JSON."""
from __future__ import annotations

import json
from pathlib import Path

from satplan.metrics import write_playback
from satplan.runner import run_simulation
from satplan.trace import SCHEMA, Trace

CANONICAL_KEYS = {
    "n_req",
    "n_long",
    "delivered",
    "good",
    "good_long",
    "good_frac",
    "good_long_frac",
    "goodput_tok_s",
    "badput_tok_s",
    "waste_ratio",
    "undelivered_frac",
    "e2e_p95_all_s",
    "long_p95_all_s",
    "peak_temp_C",
}


def _synthetic_trace() -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "synthetic-line",
        "horizon_s": 300.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
        ],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 300.0]]}],
        "dark": [[], [[50.0, 80.0]]],
    })


def _synthetic_cfg(*, kv_gib: float = 4.0, rate_per_min: float = 4.0) -> dict:
    return {
        "name": "synth_test",
        "delivery": "final",
        "control_period_s": 2.0,
        "frame_s": 10.0,
        "model": {"layers": 2, "kv_heads": 2, "head_dim": 64, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": kv_gib,
            "t_fixed_s": 0.002,
            "t_weights_s": 0.005,
            "t_kv_tok_s": 1.0e-7,
            "t_seq_s": 5.0e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.2, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {
            "arrival_window_s": 120.0,
            "rate_per_rs_per_min": rate_per_min,
            "p_long": 0.5,
            "long_out": [400, 500],
            "short_out": [100, 150],
            "long_pin": 200,
            "short_pin": 100,
            "long_upload_MB": 5.0,
            "short_upload_MB": 1.0,
        },
        "slo": {
            "deadline_factor_long": 3.0,
            "deadline_factor_short": 4.0,
            "min_deadline_s": 60.0,
            "max_stall_frac": 0.1,
        },
    }


def test_run_simulation_end_to_end_and_playback(tmp_path: Path) -> None:
    tr = _synthetic_trace()
    cfg = _synthetic_cfg()
    res = run_simulation(tr, cfg, placer="nearest", planner="none", seed=1, record=True)

    assert CANONICAL_KEYS.issubset(res.metrics)
    assert res.metrics["n_req"] > 0
    assert res.metrics["delivered"] == res.metrics["n_req"]
    assert res.metrics["undelivered_frac"] == 0.0
    assert res.metrics["peak_temp_C"] >= 25.0

    pb_path = write_playback(cfg, tr, res, "nearest+none", tmp_path, seed=1)
    pb = json.loads(pb_path.read_text(encoding="utf-8"))
    assert pb["schema"] == "satsim.cosim_playback/v1"
    assert pb["scenario"]["epoch_iso"]
    assert len(pb["scenario"]["satellites"]) == tr.n
    assert "nearest+none" in pb["runs"]
    assert len(pb["runs"]["nearest+none"]["frames"]) > 0
    assert (tmp_path / "index.json").is_file()


def test_store_and_run_no_drop_under_tight_kv() -> None:
    tr = _synthetic_trace()
    # Tight KV (~1.2 MB): only 1 request fits at a time; queued requests must wait, never drop.
    cfg = _synthetic_cfg(kv_gib=0.0012, rate_per_min=8.0)
    res = run_simulation(tr, cfg, placer="nearest", planner="none", seed=7)

    assert res.metrics["n_req"] >= 5
    assert res.metrics["delivered"] == res.metrics["n_req"]
    assert res.metrics["undelivered_frac"] == 0.0
