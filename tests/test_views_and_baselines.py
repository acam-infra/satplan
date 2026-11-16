"""Seam tests for LocalStateView vs ForecastView, placers, ReactivePlanner, and rs_downlink."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import simpy
from satplan.placers import get_placer, least_loaded, nearest, orbit_thermal
from satplan.planners import CompositePolicy, ReactivePlanner, RebalanceInvariants, get_planner
from satplan.runner import run_simulation
from satplan.trace import INF, SCHEMA, Trace
from satplan.views import ForecastView, LocalStateView
from satplan.world import CoWorld, PlanAction, Request

FORBIDDEN_LOCAL_ATTRS = (
    "power_segments",
    "next_conn",
    "trace",
    "eclipse",
    "conn_until",
    "disconnected_time",
    "edge_up_until",
    "horizon_s",
)


@dataclass
class _StubRequest:
    i: int
    origin: int = 0
    reserved_tokens: int = 100
    kv: float = 100.0
    state: str = "decode"
    mig_in_flight: bool = False
    mig_dst: int | None = None
    paused: bool = False
    n_mig: int = 0


@dataclass
class _LocalViewStub:
    now: float = 10.0
    compute_nodes: tuple[int, ...] = (1, 2, 3)
    bpt: int = 1024
    kv_cap: float = 1_000_000.0
    t_max_C: float = 85.0
    t_amb_C: float = 20.0
    stream: bool = False
    rs_downlink: bool = False
    inflight_migrations: int = 0
    loads: dict[int, int] = field(default_factory=lambda: {1: 2, 2: 0, 3: 1})
    headrooms: dict[int, float] = field(default_factory=lambda: {1: 500_000.0, 2: 500_000.0, 3: 500_000.0})
    eclipsed: dict[int, bool] = field(default_factory=lambda: {1: True, 2: False, 3: False})
    iters: dict[int, float] = field(default_factory=lambda: {1: 0.05, 2: 0.02, 3: 0.06})
    reqs_by_node: dict[int, tuple[_StubRequest, ...]] = field(default_factory=dict)
    hops_from_origin: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (2, 3))

    def node_load(self, j: int) -> int:
        return self.loads[j]

    def node_requests(self, j: int) -> tuple[_StubRequest, ...]:
        return self.reqs_by_node.get(j, ())

    def kv_headroom(self, j: int) -> float:
        return self.headrooms[j]

    def fits(self, j: int, req: _StubRequest) -> bool:
        return self.headrooms[j] >= req.reserved_tokens * self.bpt

    def dod_remaining_j(self, j: int) -> float:
        return INF

    def is_eclipse(self, j: int) -> bool:
        return self.eclipsed[j]

    def compute_by_hops(self, origin: int) -> tuple[tuple[int, int], ...]:
        return self.hops_from_origin

    def hop_count(self, s: int, d: int) -> int | None:
        return 1 if s != d else 0

    def connected(self, o: int, j: int) -> bool:
        return True

    def iter_of(
        self,
        j: int,
        extra: _StubRequest | None = None,
        *,
        extra_load: int = 0,
        extra_kv: float = 0.0,
    ) -> float:
        return self.iters[j] + 0.005 * extra_load


def _multi_node_trace() -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "synthetic-3compute",
        "horizon_s": 300.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
            {"id": 2, "name": "C-2", "role": "compute"},
            {"id": 3, "name": "C-3", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 300.0]]},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0.0, 300.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 300.0]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0.0, 300.0]]},
        ],
        "dark": [[], [[10.0, 180.0]], [], [[20.0, 120.0]]],
    })


def _multi_node_cfg() -> dict:
    return {
        "name": "synth_3x2",
        "delivery": "final",
        "rs_downlink": False,
        "control_period_s": 2.0,
        "frame_s": 10.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4},
        "model": {"layers": 2, "kv_heads": 2, "head_dim": 64, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 4.0,
            "t_fixed_s": 0.002,
            "t_weights_s": 0.005,
            "t_kv_tok_s": 1.0e-7,
            "t_seq_s": 5.0e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.2, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 5.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {
            "arrival_window_s": 100.0,
            "rate_per_rs_per_min": 6.0,
            "p_long": 0.5,
            "long_out": [800, 1200],
            "short_out": [100, 200],
            "long_pin": 200,
            "short_pin": 100,
            "long_upload_MB": 2.0,
            "short_upload_MB": 0.5,
        },
        "slo": {
            "deadline_factor_long": 3.0,
            "deadline_factor_short": 4.0,
            "min_deadline_s": 60.0,
            "max_stall_frac": 0.1,
        },
    }


def test_local_state_view_exposes_zero_future_attrs() -> None:
    tr = _multi_node_trace()
    cfg = _multi_node_cfg()
    env = simpy.Environment()
    policy = CompositePolicy(get_placer("nearest"), get_planner("none"), RebalanceInvariants.from_cfg(cfg))
    w = CoWorld(env, tr, cfg, policy=policy, seed=1)

    local = w.local_view
    assert isinstance(local, LocalStateView)
    assert not isinstance(local, ForecastView)
    assert local.rs_downlink is False
    assert local.dod_remaining_j(1) == INF
    for attr in FORBIDDEN_LOCAL_ATTRS:
        assert not hasattr(local, attr), f"LocalStateView leaked forbidden attribute {attr!r}"

    forecast = w.forecast_view
    assert isinstance(forecast, ForecastView)
    assert isinstance(forecast, LocalStateView)
    assert hasattr(forecast, "power_segments")
    assert hasattr(forecast, "next_conn")
    assert hasattr(forecast, "disconnected_time")
    assert not hasattr(forecast, "trace")


def test_nearest_least_loaded_and_reactive_on_local_view_stub() -> None:
    stub = _LocalViewStub()
    r_new = _StubRequest(i=99)

    assert nearest(stub, r_new) == 1  # type: ignore[arg-type]
    assert least_loaded(stub, r_new) == 2  # type: ignore[arg-type]

    r1 = _StubRequest(i=10, n_mig=0)
    r2 = _StubRequest(i=11, n_mig=4)
    stub.reqs_by_node = {1: (r1, r2)}
    planner = ReactivePlanner()
    inv = RebalanceInvariants(interval_s=5.0, max_inflight=2, max_mig_per_req=4)
    plan = planner.plan(stub, inv)  # type: ignore[arg-type]

    assert isinstance(plan, PlanAction)
    assert len(plan.migrations) == 1
    assert plan.migrations[0].req_id == 10
    assert plan.migrations[0].dst == 2


def test_idle_discrete_tier_node_iter_of_avoids_zero_division() -> None:
    tr = _multi_node_trace()
    cfg = _multi_node_cfg()
    cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
        {"name": "low", "power_w": 100.0, "clock_f": 0.5},
        {"name": "high", "power_w": 300.0, "clock_f": 1.0},
    ]
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    env = simpy.Environment()
    policy = CompositePolicy(get_placer("nearest"), get_planner("none"), RebalanceInvariants.from_cfg(cfg))
    w = CoWorld(env, tr, cfg, policy=policy, seed=1)

    # Empty sunlit node 2 has f_of(2) == 0.0 when idle, yet candidate iter_of must not ZeroDivisionError.
    it_sunlit = w.local_view.iter_of(2)
    assert 0.0 < it_sunlit < INF

    # Node 1 at t=15.0 is in eclipse with 0 J DoD budget -> candidate_clock_f == 0.0 -> INF.
    env.run(until=15.0)
    assert w.local_view.iter_of(1) == INF


def test_composite_policy_enforces_narrow_view_seam_and_sync_inflight() -> None:
    seen_placer_types: list[type] = []
    seen_planner_types: list[type] = []

    def spy_placer(view: LocalStateView, req: Request) -> int | None:
        seen_placer_types.append(type(view))
        return nearest(view, req)

    class _SpyPlanner:
        requires_forecast = False

        def plan(self, view: LocalStateView, invariants: RebalanceInvariants) -> PlanAction:
            seen_planner_types.append(type(view))
            return PlanAction()

    tr = _multi_node_trace()
    cfg = _multi_node_cfg()
    env = simpy.Environment()
    policy = CompositePolicy(spy_placer, _SpyPlanner(), RebalanceInvariants.from_cfg(cfg))
    w = CoWorld(env, tr, cfg, policy=policy, seed=1)
    req = Request(0, 0, 0.0, False, 100, 100, 1e5, 100.0)
    assert policy.place(w, req) == 1
    env.run(until=6.0)

    assert seen_placer_types == [LocalStateView]
    assert seen_planner_types == [LocalStateView]
    assert getattr(orbit_thermal, "requires_forecast", False) is True

    # Synchronous max_inflight cap within a single control tick
    cfg2 = _multi_node_cfg()
    cfg2["rebalance_invariants"] = {"interval_s": 5.0, "max_inflight": 1, "max_mig_per_req": 4}
    env2 = simpy.Environment()
    pol2 = CompositePolicy(get_placer("nearest"), get_planner("reactive"), RebalanceInvariants.from_cfg(cfg2))
    w2 = CoWorld(env2, tr, cfg2, policy=pol2, seed=1)
    r_a = Request(0, 0, 0.0, False, 100, 2000, 1e4, 200.0)
    r_b = Request(1, 0, 0.0, False, 100, 2000, 1e4, 200.0)
    w2.submit(r_a)
    w2.submit(r_b)
    env2.run(until=10.5)  # node 1 enters eclipse at t=10.0; control ticks at t=10.0
    assert w2.inflight_migrations() <= 1


def test_rs_downlink_toggle_decouples_return_wait() -> None:
    # Return link RS-0 <-> C-1 drops at t=5s and reconnects at t=80s.
    tr = Trace({
        "schema": SCHEMA,
        "name": "downlink-gap",
        "horizon_s": 120.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
        ],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 5.0], [80.0, 120.0]]}],
        "dark": [[], []],
    })
    cfg_off = _multi_node_cfg()
    cfg_off["net"]["hop_latency_s"] = 0.0
    cfg_off["rs_downlink"] = False
    env_off = simpy.Environment()
    w_off = CoWorld(env_off, tr, cfg_off, placer=nearest, planner=get_planner("none"), seed=1)
    r_off = Request(0, 0, 0.0, False, 100, 1000, 1e4, 120.0)
    w_off.submit(r_off)
    env_off.run(until=120.0)
    assert r_off.t_decoded is not None and r_off.t_decoded < 30.0
    assert r_off.t_delivered == r_off.t_decoded

    cfg_on = _multi_node_cfg()
    cfg_on["net"]["hop_latency_s"] = 0.0
    cfg_on["rs_downlink"] = True
    env_on = simpy.Environment()
    w_on = CoWorld(env_on, tr, cfg_on, placer=nearest, planner=get_planner("none"), seed=1)
    r_on = Request(0, 0, 0.0, False, 100, 1000, 1e4, 120.0)
    w_on.submit(r_on)
    env_on.run(until=120.0)
    assert r_on.t_decoded is not None and r_on.t_decoded < 30.0
    assert r_on.t_delivered is not None and r_on.t_delivered >= 80.0


@pytest.mark.parametrize("placer", ["nearest", "least_loaded", "orbit_thermal"])
@pytest.mark.parametrize("planner", ["none", "reactive"])
def test_3x2_cross_matrix_end_to_end(placer: str, planner: str) -> None:
    tr = _multi_node_trace()
    cfg = _multi_node_cfg()
    res = run_simulation(tr, cfg, placer=placer, planner=planner, seed=1)
    assert res.metrics["n_req"] > 0
    assert res.metrics["delivered"] == res.metrics["n_req"]
    assert res.metrics["undelivered_frac"] == 0.0
