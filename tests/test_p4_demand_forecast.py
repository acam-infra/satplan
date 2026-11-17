"""P4 tests: causal demand forecasting (history mean + published plans), non-anticipativity, and joint R3 actuation."""
from __future__ import annotations

import pytest
import simpy
from satplan.demand_forecast import CausalDemandForecaster, DemandForecastConfig, PublishedDemandPlan
from satplan.planners import CompositePolicy, RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner, _slot_bounds
from satplan.runner import run_simulation
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request
from test_p0_p2_consistency import _base_cfg


def test_p4_causal_non_anticipativity_hidden_future_invariance() -> None:
    """P4: hidden future requests (> now) and unpublished plans (publish_time_s > now) do not affect t=now."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p4-causal",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [], []],
    })
    df_cfg = DemandForecastConfig(
        enabled=True,
        obs_window_s=30.0,
        prior_rate_per_s=0.1,
        max_predicted_requests_per_slot=2.0,
    )
    plans = [
        PublishedDemandPlan(
            publish_time_s=1.0, arrival_time_s=10.0, origin=0, expected_requests=1.0,
            tokens_per_req=100.0, kv_bytes_per_req=1e6, deadline_s=25.0, weight=2.0,
        ),
        PublishedDemandPlan(
            publish_time_s=5.0, arrival_time_s=15.0, origin=0, expected_requests=2.0,
            tokens_per_req=200.0, kv_bytes_per_req=2e6, deadline_s=30.0, weight=4.0,
        ),
    ]

    def run_until_t2(extra_future: list[Request]) -> tuple[tuple, object, tuple]:
        cfg = _base_cfg()
        cfg["control_period_s"] = 2.0
        env = simpy.Environment()
        builder = ProblemBuilder(horizon_slots=4, tail_enabled=False, demand_forecast=df_cfg, time_limit_s=5.0)
        planner = R3RuntimePlanner(builder=builder)
        w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1, published_plans=plans)
        w.submit(Request(0, 0, 0.5, False, 100, 100, 1.0, 30.0))
        for req in extra_future:
            w.submit(req)
        env.run(until=2.0)
        bounds = _slot_bounds(w.forecast_view.now, (6.0, 6.0, 6.0, 6.0))
        fc = CausalDemandForecaster().forecast(w.forecast_view, bounds, df_cfg)
        act = planner.plan(w.forecast_view, RebalanceInvariants.from_cfg(cfg))
        assert not hasattr(w.local_view, "observed_arrivals")
        assert not hasattr(w.local_view, "published_plans")
        return fc, act, w.forecast_view.published_plans(2.0)

    fc_a, act_a, vis_a = run_until_t2([Request(1, 0, 15.0, False, 100, 100, 1.0, 40.0)])
    fc_b, act_b, vis_b = run_until_t2([Request(i, 0, 12.0 + i, True, 500, 500, 1.0, 50.0) for i in range(1, 10)])
    assert fc_a == fc_b
    assert act_a == act_b
    assert len(vis_a) == 1 and vis_a[0].publish_time_s == 1.0
    assert vis_a == vis_b


def test_p4_history_mean_cold_start_and_no_double_counting() -> None:
    """P4: cold-start prior, history-mean update, truncation cap, and arrived plan exclusion."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p4-hist",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    plans = [
        PublishedDemandPlan(
            publish_time_s=0.0, arrival_time_s=2.0, origin=0, expected_requests=1.0,
            tokens_per_req=100.0, kv_bytes_per_req=1e6, deadline_s=20.0,
        ),
    ]
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, placer=lambda v, r: 1, planner=get_planner("none"), seed=1, published_plans=plans)
    df_cfg = DemandForecastConfig(
        enabled=True, obs_window_s=20.0, prior_rate_per_s=0.5, max_predicted_requests_per_slot=2.0
    )
    bounds0 = _slot_bounds(0.0, (6.0, 6.0))
    fc0 = CausalDemandForecaster().forecast(w.forecast_view, bounds0, df_cfg)
    assert any(e.source == "published_plan" for e in fc0)
    assert any(e.source == "history_mean" and e.expected_requests == pytest.approx(2.0) for e in fc0)

    # Advance to t=4.0: plan with arrival_time_s=2.0 <= now=4.0 must not be re-predicted.
    w.submit(Request(0, 0, 2.0, False, 100, 100, 1.0, 20.0))
    env.run(until=4.0)
    bounds4 = _slot_bounds(4.0, (6.0, 6.0))
    fc4 = CausalDemandForecaster().forecast(w.forecast_view, bounds4, df_cfg)
    assert not any(e.source == "published_plan" for e in fc4)
    hist_entries = [e for e in fc4 if e.source == "history_mean"]
    assert hist_entries and hist_entries[0].expected_requests == pytest.approx(min(2.0, (1 / 4.0) * 6.0))


@pytest.mark.parametrize("forecast_enabled", [True, False])
def test_p4_forecast_drives_proactive_migration_closed_loop(forecast_enabled: bool) -> None:
    """P4 closed-loop ablation: causal published plan triggers proactive migration to free Node 1 KV."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p4-proactive-mig",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "rs"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 8.0]]},
            {"u": 3, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [], [], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["gpu"].update(kv_gib=0.04, t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    cfg["net"]["link_gbps"].update(inter=0.08, rs=10.0)  # 10 MB/s
    # 5000 tokens * 4096 B/tok = 20.48 MB (~0.019 GiB); kv_gib=0.04 GiB (~42.9 MB) fits 1 or 2?
    # Let's make kv_gib=0.03 GiB (32.2 MB): fits 1 req (20.48 MB), NOT 2 (40.96 MB)!
    cfg["gpu"]["kv_gib"] = 0.03
    plan = PublishedDemandPlan(
        publish_time_s=0.0,
        arrival_time_s=8.0,
        origin=3,  # only directly connected to Node 1 via RS-3
        expected_requests=1.0,
        tokens_per_req=50.0,
        kv_bytes_per_req=5000 * 4096.0,
        deadline_s=12.0,
        weight=5.0,
    )
    df_cfg = DemandForecastConfig(enabled=forecast_enabled)
    actions_at_t2 = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(
                builder=ProblemBuilder(horizon_slots=4, tail_enabled=False, demand_forecast=df_cfg, time_limit_s=5.0)
            )

        def plan(self, view, invariants):
            act = self.inner.plan(view, invariants)
            if view.now <= 2.5:
                actions_at_t2.append((act, self.inner.last_result, self.inner.last_diag))
            # Predicted/virtual IDs must never appear in real PlanAction or goodput_req_ids
            assert all(m.req_id in (0, 1) for m in act.migrations)
            assert all(pid in (0, 1) for pid in act.pause_ids)
            assert all(rid in (0, 1) for rid in act.resume_ids)
            if self.inner.last_result is not None:
                assert all(gid in (0, 1) for gid in self.inner.last_result.goodput_req_ids)
            return act

    def placer(view, req):
        if req.origin == 3:
            return 1 if view.kv_headroom(1) >= req.reserved_tokens * view.bpt - 1e-6 else None
        return 1

    env = simpy.Environment()
    policy = CompositePolicy(placer=placer, planner=_RecPlanner(), invariants=RebalanceInvariants.from_cfg(cfg))
    w = CoWorld(env, tr, cfg, policy=policy, seed=1, published_plans=[plan])
    req0 = Request(0, 0, 0.0, False, 4900, 150, 1.0, 40.0)
    req1 = Request(1, 3, 8.0, False, 4950, 50, 1.0, 20.0)
    w.submit(req0)
    w.submit(req1)
    env.run(until=35.0)

    act2, res2, diag2 = actions_at_t2[0]
    assert res2 is not None and res2.status == "optimal"
    if forecast_enabled:
        assert diag2["n_predicted_demands"] >= 1
        assert len(act2.migrations) == 1 and act2.migrations[0].req_id == 0 and act2.migrations[0].dst == 2
        assert req0.t_delivered is not None and req0.t_delivered <= req0.deadline
        assert req1.t_delivered is not None and req1.t_delivered <= req1.deadline
    else:
        assert diag2["n_predicted_demands"] == 0
        assert act2.migrations == ()


def test_p4_yaml_config_and_runner_wiring() -> None:
    """P4: runner/get_planner reads demand_forecast and r3.tail_enabled declaratively (default off)."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "p4-cfg",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]}],
        "dark": [[], []],
    })
    cfg = _base_cfg()
    cfg["trace"] = "unused"
    cfg["workload"] = {
        "arrival_window_s": 20.0,
        "rate_per_rs_per_min": 6.0,
        "p_long": 0.0,
        "long_out": [200, 300],
        "short_out": [40, 60],
        "long_pin": 100,
        "short_pin": 40,
        "long_upload_MB": 0.1,
        "short_upload_MB": 0.1,
    }
    cfg["slo"] = {"deadline_factor_long": 2.0, "deadline_factor_short": 4.0, "min_deadline_s": 30.0}
    assert DemandForecastConfig.from_cfg(cfg).enabled is False

    pl_default = get_planner("r3_milp", cfg)
    assert isinstance(pl_default, R3RuntimePlanner)
    assert pl_default.builder.demand_forecast.enabled is False
    assert pl_default.builder.tail_enabled is True

    cfg_on = dict(cfg)
    cfg_on["r3"] = {"tail_enabled": False, "horizon_slots": 4}
    cfg_on["demand_forecast"] = {"enabled": True, "obs_window_s": 25.0, "prior_rate_per_s": 0.05}
    pl_on = get_planner("r3_milp", cfg_on)
    assert isinstance(pl_on, R3RuntimePlanner)
    assert pl_on.builder.tail_enabled is False
    assert pl_on.builder.demand_forecast.enabled is True
    res = run_simulation(tr, cfg_on, placer="nearest", planner="r3_milp", seed=1)
    assert res.metrics["n_req"] > 0
