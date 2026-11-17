"""Regression tests for LAB-2046 ZeroCostAtomicWorld KV ledger, Trace eclipse rebuild, and ReplayPlacer."""
from __future__ import annotations

import simpy
from experiments.lab2046_common import ReplayPlacer, ZeroCostAtomicWorld
from satplan.actions import MigrationOrder, PlanAction
from satplan.placers import nearest
from satplan.planners import CompositePolicy, RebalanceInvariants
from satplan.trace import SCHEMA, Trace
from satplan.world import Request


def _cfg() -> dict:
    return {
        "name": "test-atomic-kv",
        "delivery": "final",
        "rs_downlink": False,
        "control_period_s": 2.0,
        "frame_s": 10.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 8, "max_mig_per_req": 4},
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7, "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "P_idle": 60,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 80,
            "eclipse_dod_budget_j": 1500.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
                {"name": "low_20w", "power_w": 20.0, "clock_f": 0.35},
                {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
            ],
        },
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.25, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": 1.5, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {"arrival_window_s": 20.0},
        "slo": {"max_stall_frac": 0.1},
    }


def _trace_data(*, with_eclipse: bool = True) -> dict:
    return {
        "schema": SCHEMA,
        "name": "test-3node",
        "horizon_s": 200.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1", "role": "compute"},
            {"id": 2, "name": "CS-2", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 200.0]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 200.0]]},
        ],
        "dark": [[], [[30.0, 80.0]], []] if with_eclipse else [[], [], []],
    }


class _MoveOncePlanner:
    requires_forecast = True

    def plan(self, view, invariants: RebalanceInvariants) -> PlanAction:
        orders = []
        for r in view.node_requests(1):
            if r.state == "decode" and r.n_mig == 0 and view.fits(2, r):
                orders.append(MigrationOrder(req_id=r.i, dst=2))
        return PlanAction(migrations=tuple(orders[: invariants.max_inflight]))


def test_atomic_migration_preserves_per_node_kv_and_zeroes_on_finish() -> None:
    cfg = _cfg()
    tr = Trace(_trace_data(with_eclipse=True))
    env = simpy.Environment()
    inv = RebalanceInvariants.from_cfg(cfg)
    w = ZeroCostAtomicWorld(env, tr, cfg, policy=CompositePolicy(nearest, _MoveOncePlanner(), inv), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=True, pin=500, pout=2000, upload_bytes=5e6, deadline=150.0)
    w.submit(r)
    env.run(until=tr.horizon)

    assert r.trail == [1, 2]
    assert r.t_delivered is not None
    chk = w.check_kv_invariants(at_end=True)
    assert chk["final_reserved_sum"] == 0.0
    assert chk["final_resident_sum"] == 0.0
    assert w.kv_used(1) == 0.0
    assert w.kv_used(2) == 0.0


def test_trace_rebuild_clears_eclipse_and_replay_placer_pins_target() -> None:
    tr_ecl = Trace(_trace_data(with_eclipse=True))
    tr_no_ecl = Trace(_trace_data(with_eclipse=False))
    assert tr_ecl.is_eclipse(1, 50.0)
    assert not tr_no_ecl.is_eclipse(1, 50.0)

    env = simpy.Environment()
    cfg = _cfg()
    inv = RebalanceInvariants.from_cfg(cfg)
    w = ZeroCostAtomicWorld(env, tr_ecl, cfg, policy=CompositePolicy(ReplayPlacer({0: 2}), _MoveOncePlanner(), inv), seed=1)
    r = Request(i=0, origin=0, t=0.0, long=False, pin=100, pout=200, upload_bytes=1e6, deadline=100.0)
    w.submit(r)
    env.run(until=tr_ecl.horizon)
    assert r.trail[0] == 2
