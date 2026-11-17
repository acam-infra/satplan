"""P0-P2 background invariants; moved without changing test bodies."""
from __future__ import annotations

import pytest
import simpy
from factories import _base_cfg
from satplan.planners import RebalanceInvariants, get_planner
from satplan.r3_planner import ProblemBuilder, R3RuntimePlanner
from satplan.trace import SCHEMA, Trace
from satplan.world import CoWorld, Request


@pytest.mark.parametrize(
    ("upload_bytes", "expect_migrant_ontime"),
    [
        (100_000_000.0, False),  # R14 repro: 100MB upload occupies CS1->CS2 across slots 0..2
        (1.0, True),             # 1B upload finishes before t=2s -> migrant delivers on time
    ],
)
def test_r2g_known_upload_across_slots_negative_and_positive(
    upload_bytes: float,
    expect_migrant_ontime: bool,
) -> None:
    """R2g: known upload occupancy persists across forecast slots without false goodput."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-known-upload",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 1, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    first_pred, first_bg = records[0]
    assert set(first_pred).issubset(actual_ontime)
    if expect_migrant_ontime:
        assert first_bg[(1, 2)][:4] == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-3)
        assert 1 in actual_ontime
        assert 1 in first_pred
    else:
        assert first_bg[(1, 2)][:4] == pytest.approx((40e6, 40e6, 40e6, 0.0), rel=1e-5)
        assert first_bg[(2, 1)][:4] == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-3)
        assert 1 not in first_pred



def test_r2g_shared_bottleneck_speedup_and_completion_release() -> None:
    """R2g: when a competing upload finishes mid-slot, surviving upload speeds up then releases."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-shared-speedup",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(rs=0.08, inter=0.08)  # 10 MB/s on both RS and inter
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    env = simpy.Environment()
    world = CoWorld(
        env,
        tr,
        cfg,
        placer=lambda v, r: {0: 2, 1: 3, 2: 1}[r.i],
        planner=get_planner("r3_milp"),
        seed=1,
    )
    # Req 0 (45MB -> CS2) & Req 1 (15MB -> CS3) share 0->1 (5 MB/s each for 3s -> Req 1 ends at t=3s;
    # Req 0 speeds up to 10 MB/s at t=3s and finishes remaining 30MB at t=6s, releasing CS1->CS2 for slot 1).
    world.submit(Request(0, 0, 0.0, False, 1, 1, 45_000_000.0, 50.0))
    world.submit(Request(1, 0, 0.0, False, 1, 1, 15_000_000.0, 50.0))
    migrant = Request(2, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=2.0)
    prob = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    bg_12 = next(e.bg_bps for e in prob.edges if e.src == 1 and e.dst == 2)
    assert bg_12 is not None
    # At t=2s: rem(0)=35MB, rem(1)=5MB. Slot 0 [2,8): 5MB*1s + 10MB*3s = 35MB -> 46.67 Mbps (>40Mbps initial!),
    # then slot 1+ released (0.0 Mbps).
    assert bg_12[0] == pytest.approx(35_000_000.0 * 8.0 / 6.0, rel=1e-4)
    assert bg_12[1] == pytest.approx(0.0, abs=1e-3)
    env.run(until=25.0)
    assert migrant.node in (2, 3) and migrant.t_delivered is not None and migrant.t_delivered <= 20.0



def test_r2g_link_break_and_recovery_reserves_retry_slot() -> None:
    """R2g: link break aborts upload and recovery re-reserves retry occupancy in later slot."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-link-break-retry",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 5.0], [14.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"].update(rs=10.0, inter=0.04)  # 5 MB/s
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2, planner=get_planner("none"), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 60_000_000.0, 50.0))
    env.run(until=2.0)
    prob = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    bg_12 = next(e.bg_bps for e in prob.edges if e.src == 1 and e.dst == 2)
    assert bg_12 is not None
    # Slot 0 [2,8): active [2,5) -> 20 Mbps; Slot 1 [8,14): RS down -> 0; Slot 2 [14,20): retried 60MB -> 40 Mbps.
    assert bg_12[0] == pytest.approx(20e6, rel=1e-4)
    assert bg_12[1] == pytest.approx(0.0, abs=1e-3)
    assert bg_12[2] == pytest.approx(40e6, rel=1e-4)



def test_r2g_disjoint_path_migration_positive() -> None:
    """R2g positive: long upload on CS1->CS2 does not block migration on disjoint CS3->CS4."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2g-disjoint-path",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
            {"id": 4, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 0, "v": 3, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
            {"u": 3, "v": 4, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    preds: list[tuple[int, ...]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            if self.inner.last_result is not None:
                preds.append(self.inner.last_result.goodput_req_ids)
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 3, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 100_000_000.0, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=25.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert 1 in actual_ontime
    assert migrant.node == 4
    assert any(1 in p for p in preds)
    assert all(set(p).issubset(actual_ontime) for p in preds)



@pytest.mark.parametrize(
    ("upload_bytes", "hop_latency_s", "deadline_s", "expect_ontime"),
    [
        (30_000_000.0, 1.1, 14.001, False),  # R15 repro: 10MB bg + ~40MB mig + 1.1s prop > 6s slot
        (1.0, 1.1, 14.001, True),            # 1B bg -> unlocks in slot 0, delivers at 14.0s
        (30_000_000.0, 0.5, 14.001, True),   # shorter 0.5s prop: 5.024s + 0.5s <= 6.0s -> delivers on time
        (30_000_000.0, 1.1, 20.5, True),     # wider deadline: unlocks in slot 1, delivers at 20.0s
    ],
)
def test_r2h_background_average_unlock_timing(
    upload_bytes: float,
    hop_latency_s: float,
    deadline_s: float,
    expect_ontime: bool,
) -> None:
    """R2h: tx_frac must account for background bytes + migration bytes over total link capacity."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2h-unlock-timing",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg(hop_latency_s=hop_latency_s)
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.08, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    preds: list[tuple[int, ...]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            if self.inner.last_result is not None:
                preds.append(self.inner.last_result.goodput_req_ids)
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 1, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 9765, 60, 1.0, deadline_s)
    world.submit(migrant)
    env.run(until=30.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert preds
    assert all(set(p).issubset(actual_ontime) for p in preds)
    if expect_ontime:
        assert 1 in actual_ontime
        assert any(1 in p for p in preds)
    else:
        assert 1 not in preds[0]



@pytest.mark.parametrize(
    ("upload_bytes", "expect_ontime"),
    [
        (100_000_000.0, False),  # R15 repro: 0->1->2 breaks at t=3s, retries on 0->3->2
        (1.0, True),             # 1B bg -> delivers at 19.0s
    ],
)
def test_r2i_background_retry_new_path(upload_bytes: float, expect_ontime: bool) -> None:
    """R2i: failed upload re-routes onto new path via ForecastView.path_at and reserves retry occupancy."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2i-retry-new-path",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}] + [{"id": j, "role": "compute"} for j in (1, 2, 3)],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 3.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 3.0]]},
            {"u": 1, "v": 3, "kind": "intra", "up": [[0.0, 60.0]]},
            {"u": 0, "v": 3, "kind": "rs", "up": [[3.0, 60.0]]},
            {"u": 3, "v": 2, "kind": "inter", "up": [[3.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [], [[0.0, 60.0]]],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, intra=10.0, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2 if r.i == 0 else 3, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 20.0)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    assert all(set(p).issubset(actual_ontime) for p, _ in records)
    first_pred, first_bg = records[0]
    if expect_ontime:
        assert 1 in actual_ontime
        assert 1 in first_pred
    else:
        assert first_bg[(3, 2)][:4] == pytest.approx((0.0, 40e6, 40e6, 20e6), rel=1e-4)
        assert 1 not in first_pred



@pytest.mark.parametrize(
    ("upload_bytes", "expect_ontime"),
    [
        (100_000_000.0, False),  # R16 repro: 0->2 breaks at t=3s, placer switches dst 2->3 (0->1->3)
        (1.0, True),             # 1B bg -> finishes before t=2s, migrant delivers at 13.0s
    ],
)
def test_r2j_background_retry_new_destination(upload_bytes: float, expect_ontime: bool) -> None:
    """R2j: failed upload re-entering external placer must conservatively bound candidate destinations."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2j-retry-new-dst",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}] + [{"id": j, "role": "compute"} for j in (1, 2, 3)],
        "edges": [
            {"u": 0, "v": 2, "kind": "inter", "up": [[0.0, 3.0]]},
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []
    placements: list[tuple[float, int, int]] = []

    def placer(view, req):
        dst = (2 if view.connected(req.origin, 2) else 3) if req.i == 0 else 1
        placements.append((view.now, req.i, dst))
        return dst

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=placer, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(1, 0, 0.0, False, 5000, 50, 1.0, 14.001)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    assert all(set(p).issubset(actual_ontime) for p, _ in records)
    first_pred, first_bg = records[0]
    if expect_ontime:
        assert 1 in actual_ontime
        assert 1 in first_pred
    else:
        assert (0.0, 0, 2) in placements and (3.0, 0, 3) in placements
        assert first_bg[(1, 3)][:4] == pytest.approx((33.333333e6, 40e6, 40e6, 20e6), rel=1e-4)
        assert 1 not in first_pred



def test_r2j_waiting_upload_before_placement_reserves_occupancy() -> None:
    """R2j: arrived request in state=='wait' waiting for link recovery still reserves future occupancy."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2j-waiting-upload",
        "horizon_s": 60.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[6.0, 60.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["net"]["link_gbps"].update(inter=0.04, rs=10.0)
    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=lambda v, r: 2, planner=get_planner("none"), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 60_000_000.0, 50.0))
    env.run(until=2.0)
    prob = ProblemBuilder(horizon_slots=4, time_limit_s=5.0).build(
        world.forecast_view, RebalanceInvariants.from_cfg(cfg)
    )
    bg_12 = next(e.bg_bps for e in prob.edges if e.src == 1 and e.dst == 2)
    assert bg_12 is not None
    # Wait request starts when RS 0->1 comes up at t=6.0 (slot 0 [2,8): 2s -> 13.33Mbps; slot 1 [8,14): 40Mbps).
    assert bg_12[0] == pytest.approx(10_000_000.0 * 8.0 / 6.0, rel=1e-4)
    assert bg_12[1] == pytest.approx(40e6, rel=1e-4)



@pytest.mark.parametrize(
    ("background_count", "upload_bytes", "break_times", "expect_migrant_ontime"),
    [
        (1, 9_000_000.0, (3.0,), True),
        (2, 9_000_000.0, (3.0, 3.0), False),
        (3, 9_000_000.0, (3.0, 3.0, 3.0), False),
        (3, 12_000_000.0, (3.0, 3.0, 3.0), False),
        (2, 9_000_000.0, (3.0, 4.0), False),
    ],
)
def test_r2k_simultaneous_and_staggered_background_retries(
    background_count: int,
    upload_bytes: float,
    break_times: tuple[float, ...],
    expect_migrant_ontime: bool,
) -> None:
    """R2k: joint retry occupancy sums concurrent/staggered retries and carries spillover across slots."""
    nodes = [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}]
    edges = [
        {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
        {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 60.0]]},
    ]
    dark = [[], [[0.0, 60.0]], []]
    fail_nodes: dict[int, int] = {}
    for idx_b, bt in enumerate(break_times):
        nid = 3 + idx_b
        nodes.append({"id": nid, "role": "compute"})
        edges.append({"u": 0, "v": nid, "kind": "intra", "up": [[0.0, bt]]})
        dark.append([[0.0, 60.0]])
        fail_nodes[idx_b] = nid

    tr = Trace({"schema": SCHEMA, "name": "r2k-joint-retries", "horizon_s": 60.0, "nodes": nodes, "edges": edges, "dark": dark})
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, intra=0.008, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    records: list[tuple[tuple[int, ...], dict[tuple[int, int], tuple[float, ...]]]] = []

    def placer(view, req):
        if req.i < background_count:
            nid = fail_nodes[req.i]
            return nid if view.connected(req.origin, nid) else 2
        return 1

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            res = self.inner.last_result
            assert res is not None
            prob = self.inner.builder.build(view, invariants)
            bg_map = {(e.src, e.dst): e.bg_bps for e in prob.edges if e.bg_bps is not None}
            records.append((res.goodput_req_ids, bg_map))
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=placer, planner=_RecPlanner(), seed=1)
    for i in range(background_count):
        world.submit(Request(i, 0, 0.0, False, 1, 1, upload_bytes, 50.0))
    migrant = Request(background_count, 0, 0.0, False, 5000, 50, 1.0, 14.001)
    world.submit(migrant)
    env.run(until=40.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert records
    assert all(set(p).issubset(actual_ontime) for p, _ in records)
    first_pred, first_bg = records[0]
    if expect_migrant_ontime:
        assert background_count in actual_ontime
        assert background_count in first_pred
        assert first_bg[(1, 2)][:4] == pytest.approx((12e6, 0.0, 0.0, 0.0), rel=1e-4)
    else:
        assert background_count not in first_pred
        if background_count == 2:
            assert first_bg[(1, 2)][:4] == pytest.approx((24e6, 0.0, 0.0, 0.0), rel=1e-4)
        elif upload_bytes == 9_000_000.0:
            assert first_bg[(1, 2)][:4] == pytest.approx((36e6, 0.0, 0.0, 0.0), rel=1e-4)
        else:
            # 3 * 12MB = 36MB > 30MB slot-0 cap: 30MB (40Mbps) in slot 0, 6MB carryover (8Mbps) in slot 1.
            assert first_bg[(1, 2)][0] == pytest.approx(40e6, rel=1e-4)
            assert first_bg[(1, 2)][1] == pytest.approx(6_000_000.0 * 8.0 / 6.0, rel=1e-4)



def test_r2k_joint_retry_disjoint_path_positive() -> None:
    """R2k positive: 2 simultaneous retries on 0->1->3 do not block disjoint 4->5 migration."""
    tr = Trace({
        "schema": SCHEMA,
        "name": "r2k-disjoint-positive",
        "horizon_s": 60.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
            {"id": 4, "role": "compute"},
            {"id": 5, "role": "compute"},
            {"id": 6, "role": "rs"},
        ],
        "edges": [
            {"u": 0, "v": 2, "kind": "intra", "up": [[0.0, 3.0]]},
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, 60.0]]},
            {"u": 6, "v": 4, "kind": "rs", "up": [[0.0, 60.0]]},
            {"u": 4, "v": 5, "kind": "inter", "up": [[0.0, 60.0]]},
        ],
        "dark": [[], [[0.0, 60.0]], [[0.0, 60.0]], [], [[0.0, 60.0]], [], []],
    })
    cfg = _base_cfg()
    cfg["control_period_s"] = 2.0
    cfg["power"]["eclipse_dod_budget_j"] = 0.0
    cfg["thermal"]["T0_C"] = 35.0
    cfg["net"]["link_gbps"].update(inter=0.04, intra=0.008, rs=10.0)
    cfg["gpu"].update(t_fixed_s=0.1, t_weights_s=0.0, t_kv_tok_s=0.0, t_seq_s=0.0)
    preds: list[tuple[int, ...]] = []

    def placer(view, req):
        if req.i < 2:
            return 2 if view.connected(req.origin, 2) else 3
        return 4

    class _RecPlanner:
        requires_forecast = True

        def __init__(self) -> None:
            self.inner = R3RuntimePlanner(builder=ProblemBuilder(horizon_slots=4, time_limit_s=5.0))

        def plan(self, view, invariants):
            action = self.inner.plan(view, invariants)
            if self.inner.last_result is not None:
                preds.append(self.inner.last_result.goodput_req_ids)
            return action

    env = simpy.Environment()
    world = CoWorld(env, tr, cfg, placer=placer, planner=_RecPlanner(), seed=1)
    world.submit(Request(0, 0, 0.0, False, 1, 1, 9_000_000.0, 50.0))
    world.submit(Request(1, 0, 0.0, False, 1, 1, 9_000_000.0, 50.0))
    migrant = Request(2, 6, 0.0, False, 5000, 50, 1.0, 14.001)
    world.submit(migrant)
    env.run(until=25.0)
    actual_ontime = {r.i for r in world.reqs if r.t_delivered is not None and r.t_delivered <= r.deadline}
    assert 2 in actual_ontime
    assert migrant.node == 5
    assert any(2 in p for p in preds)
    assert all(set(p).issubset(actual_ontime) for p in preds)

