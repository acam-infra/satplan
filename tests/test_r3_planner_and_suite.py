"""Seam and end-to-end tests for ProblemBuilder, R3RuntimePlanner (r3_milp), and 3x3 run_suite."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
import simpy
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner
from satplan.runner import run_simulation
from satplan.suite import PLACERS_3X3, PLANNERS_3X3, run_suite
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


def _discrete_3node_trace(*, downlink_gap: bool = False) -> Trace:
    rs_up = [[0.0, 15.0], [95.0, 120.0]] if downlink_gap else [[0.0, 120.0]]
    return Trace({
        "schema": SCHEMA,
        "name": "synth-3x3",
        "horizon_s": 120.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
            {"id": 2, "name": "C-2", "role": "compute"},
            {"id": 3, "name": "C-3", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": rs_up},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0.0, 120.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 120.0]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0.0, 120.0]]},
        ],
        "dark": [[], [[10.0, 90.0]], [], [[20.0, 70.0]]],
    })


def _discrete_cfg() -> dict:
    return {
        "name": "synth_3x3",
        "delivery": "final",
        "control_period_s": 2.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 8, "max_mig_per_req": 2},
        "model": {"layers": 4, "kv_heads": 4, "head_dim": 64, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 0.5,
            "t_fixed_s": 0.005,
            "t_weights_s": 0.015,
            "t_kv_tok_s": 1e-7,
            "t_seq_s": 1e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {
            "P_idle": 60,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 80,
            "eclipse_dod_budget_j": 1500.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
                {"name": "low_75w", "power_w": 75.0, "clock_f": 0.5},
                {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
            ],
        },
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.2, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 5.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {
            "arrival_window_s": 30.0,
            "rate_per_rs_per_min": 8.0,
            "p_long": 0.3,
            "long_out": [800, 1200],
            "short_out": [150, 300],
            "long_pin": 200,
            "short_pin": 80,
            "long_upload_MB": 2.0,
            "short_upload_MB": 0.5,
        },
        "slo": {"deadline_factor_long": 2.0, "deadline_factor_short": 4.0, "min_deadline_s": 40.0},
    }


def test_problem_builder_maps_forecast_view_and_excludes_rs_edges() -> None:
    tr = _discrete_3node_trace()
    cfg = _discrete_cfg()
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=get_placer("nearest"), planner=get_planner("r3_milp"), seed=1)
    r = Request(0, 0, 0.0, False, 100, 400, 1e4, 80.0)
    w.submit(r)
    env.run(until=5.0)

    builder = ProblemBuilder(horizon_slots=4)
    inv = RebalanceInvariants.from_cfg(cfg)
    prob = builder.build(w.forecast_view, inv)

    assert prob.horizon_slots == 4
    assert prob.dt_s == 6.0  # ceil(5.0 / 2.0) * 2.0 = 6.0s effective CompositePolicy cadence
    assert [n.node_id for n in prob.nodes] == [1, 2, 3]
    assert len(prob.tiers) == 3
    # 2 compute-compute edges (1-2, 2-3) -> 4 directed R3EdgeSpecs (RS-0 edges excluded)
    assert len(prob.edges) == 4
    assert all(e.src in (1, 2, 3) and e.dst in (1, 2, 3) for e in prob.edges)
    assert len(prob.requests) == 1
    assert prob.requests[0].req_id == 0
    assert prob.eclipse_dod_budget_j == 1500.0


@pytest.mark.parametrize("placer", PLACERS_3X3)
def test_r3_milp_composes_with_all_placers_and_respects_invariants(placer: str) -> None:
    planner = get_planner("r3_milp")
    assert isinstance(planner, R3RuntimePlanner)
    assert planner.requires_forecast is True

    tr = _discrete_3node_trace()
    cfg = _discrete_cfg()
    cfg["rebalance_invariants"] = {"interval_s": 5.0, "max_inflight": 1, "max_mig_per_req": 1}
    res = run_simulation(tr, cfg, placer=placer, planner="r3_milp", seed=1)
    assert res.metrics["n_req"] > 0
    assert res.metrics["delivered"] == res.metrics["n_req"]


def test_r3_milp_improves_goodput_under_eclipse_dod_bottleneck() -> None:
    tr = _discrete_3node_trace()
    cfg = _discrete_cfg()
    # Tight DoD on node 1 (eclipse [10, 140]) vs sunlit node 2
    cfg["power"]["eclipse_dod_budget_j"] = 600.0
    res_none = run_simulation(tr, cfg, placer="nearest", planner="none", seed=1)
    res_r3 = run_simulation(tr, cfg, placer="nearest", planner="r3_milp", seed=1)
    assert res_r3.metrics["good_frac"] >= res_none.metrics["good_frac"]


def test_run_suite_3x3_multiseed_outputs_csv_and_summary_md(tmp_path: Path) -> None:
    tr = _discrete_3node_trace()
    cfg = _discrete_cfg()
    inv = RebalanceInvariants(interval_s=5.0, max_inflight=6, max_mig_per_req=2)
    suite_res = run_suite(tr, cfg, tmp_path, seeds=[1, 2], invariants=inv)

    assert suite_res.csv_path.is_file()
    assert suite_res.markdown_path.is_file()
    assert len(suite_res.rows) == len(PLACERS_3X3) * len(PLANNERS_3X3) * 2
    assert len(suite_res.summary_rows) == 9

    with suite_res.csv_path.open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 18
    assert all(r["rs_downlink"] == "False" for r in csv_rows)
    assert all(float(r["interval_s"]) == 5.0 and int(r["max_inflight"]) == 6 for r in csv_rows)

    md = suite_res.markdown
    assert "SLO 达标率(%) ↑" in md
    assert "Badput(tok/s) ↓" in md
    assert "r3_milp" in md
    assert "%" in md
    # Ensure CompositePolicy view isolation remains intact
    assert not hasattr(CompositePolicy(get_placer("nearest"), get_planner("r3_milp")).placer, "requires_forecast")


def test_run_suite_rs_downlink_sensitivity(tmp_path: Path) -> None:
    tr = _discrete_3node_trace(downlink_gap=True)
    cfg = _discrete_cfg()
    res_off = run_suite(
        tr,
        cfg,
        tmp_path / "off",
        seeds=[1],
        matrix=(("nearest",), ("none", "r3_milp")),
        rs_downlink=False,
    )
    res_on = run_suite(
        tr,
        cfg,
        tmp_path / "on",
        seeds=[1],
        matrix=(("nearest",), ("none", "r3_milp")),
        rs_downlink=True,
    )
    off_p95 = res_off.summary_rows[0]["e2e_p95_all_s"]
    on_p95 = res_on.summary_rows[0]["e2e_p95_all_s"]
    assert on_p95 > off_p95
