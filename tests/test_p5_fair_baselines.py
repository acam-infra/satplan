"""P5 tests: ForecastAwarePlanner (same-info causal baseline), manifest injection, solver stats, and suite baseline."""
from __future__ import annotations

import simpy
from factories import _base_cfg
from satplan.demand_forecast import PublishedDemandPlan
from satplan.placers import get_placer
from satplan.planners import CompositePolicy, ForecastAwarePlanner, RebalanceInvariants, get_planner
from satplan.runner import run_simulation
from satplan.suite import run_suite
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request, make_requests


def test_p5_forecast_aware_planner_same_info_and_non_anticipativity() -> None:
    """ForecastAwarePlanner uses ForecastView causally and triggers proactive KV-freeing migration."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p5-fa",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "rs"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
            {"u": 3, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [], [], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["gpu"]["kv_gib"] = 0.03
    cfg["demand_forecast"] = {"enabled": True, "obs_window_s": 1.0}
    plan = PublishedDemandPlan(
        publish_time_s=0.0,
        arrival_time_s=6.0,
        origin=3,
        expected_requests=1.0,
        tokens_per_req=50.0,
        kv_bytes_per_req=5000 * 4096.0,
        deadline_s=20.0,
        weight=2.0,
    )
    pl = get_planner("forecast_aware", cfg)
    assert isinstance(pl, ForecastAwarePlanner)
    assert pl.requires_forecast is True

    def run_at_t2(extra_future: list[Request]):
        env = simpy.Environment()
        w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1, published_plans=[plan])
        w.submit(Request(0, 0, 0.0, False, 4900, 150, 1.0, 40.0))
        for r in extra_future:
            w.submit(r)
        env.run(until=2.0)
        return pl.plan(w.forecast_view, RebalanceInvariants.from_cfg(cfg))

    act_a = run_at_t2([Request(1, 3, 10.0, False, 100, 100, 1.0, 30.0)])
    act_b = run_at_t2([Request(i, 3, 12.0 + i, True, 500, 500, 1.0, 40.0) for i in range(1, 6)])
    assert act_a == act_b
    assert len(act_a.migrations) == 1
    assert act_a.migrations[0].req_id == 0 and act_a.migrations[0].dst == 2


def test_p5_manifest_injection_and_solver_stats(tmp_path) -> None:
    """run_simulation accepts frozen request manifests and records R3 solver diagnostics."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p5-manifest",
        "horizon_s": 40.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 40.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 40.0]]},
        ],
        "dark": [[], [[5.0, 25.0]], []],
    })
    cfg = _base_cfg()
    cfg["workload"]["rate_per_rs_per_min"] = 6.0
    reqs = make_requests(cfg, tr, 1)
    res_r3 = run_simulation(tr, cfg, "nearest", "r3_milp", 1, requests=reqs)
    assert res_r3.metrics["n_req"] == len(reqs)
    assert res_r3.metrics["solver_calls"] >= 1
    assert "zero_power_frac" in res_r3.metrics

    suite_res = run_suite(tr, cfg, tmp_path, seeds=[1], matrix=(("nearest", "least_loaded"), ("none", "reactive")))
    assert "nearest+none" in suite_res.markdown
    assert "0W 空闲(%) ↑" in suite_res.markdown
    assert not hasattr(CompositePolicy(get_placer("nearest"), get_planner("forecast_aware")).placer, "requires_forecast")
