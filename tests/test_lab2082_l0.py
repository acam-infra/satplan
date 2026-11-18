"""L0 deterministic counterexamples for K1 (batch/mispredict), K2 (spare capacity), and K3 (ISL contention)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import simpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "LAB-2082"))

from common import (
    CausalReactivePlanner,
    ContentionFreeDiagnosticWorld,
    DiagnosticWorld,
    QueueReroutePlanner,
    ZeroCostDiagnosticWorld,
)
from experiments.lab2046_common import ReplayPlacer
from satplan.metrics import Recorder
from satplan.net import allocate_max_min_rates
from satplan.planners import CompositePolicy, NonePlanner, RebalanceInvariants
from satplan.timing import iter_time, kv_bytes_per_tok
from satplan.trace import SCHEMA, Trace
from satplan.world import Request


def _l0_cfg(*, inter_gbps: float = 0.6) -> dict:
    return {
        "delivery": "final",
        "rs_downlink": False,
        "horizon_s": 320.0,
        "control_period_s": 2.0,
        "rebalance_invariants": {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4},
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 24.0,
            "t_fixed_s": 0.003,
            "t_weights_s": 0.01,
            "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {
            "P_idle": 60,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 80,
            "eclipse_dod_budget_j": 1000.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "clock_f": 0.0},
                {"name": "low_20w", "power_w": 20.0, "clock_f": 0.35},
                {"name": "high_300w", "power_w": 300.0, "clock_f": 1.0},
            ],
        },
        "thermal": {"T0_C": 25.0, "T_amb_C": 20.0, "T_max_C": 85.0, "R_th": 0.25, "C_th": 150.0},
        "net": {"link_gbps": {"intra": 10.0, "inter": inter_gbps, "rs": 5.0}, "hop_latency_s": 0.005},
        "workload": {"arrival_window_s": 25.0},
        "slo": {"max_stall_frac": 0.1},
    }


def _two_compute_trace(horizon: float = 320.0, *, dark: list | None = None, inter_up: list | None = None) -> Trace:
    return Trace({
        "schema": SCHEMA,
        "name": "l0-2star",
        "horizon_s": horizon,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1", "role": "compute"},
            {"id": 2, "name": "CS-2", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, horizon]]},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0.0, horizon]]},
            {"u": 1, "v": 2, "kind": "inter", "up": inter_up if inter_up is not None else [[0.0, horizon]]},
        ],
        "dark": dark if dark is not None else [[], [], []],
    })


def test_A_batch_runtime_mispredict() -> None:
    """K1: Post-start arrivals + output length underestimation slow decode TPOT on CS-1.

    Analytical check:
      - Solo R0 (ctx=500..4500, nd=1, f=1.0): iter_time ~ 0.01310..0.01350 s/tok.
        4000 tokens -> ~53.2s; 8000 tokens -> ~108.0s.
      - With 6 co-running requests (nd=7, ctx=30000, f=1.0): iter_time = 0.01635 s/tok (+24% slower!).
        8000 tokens at 0.01635 s/tok -> ~131s > deadline 120s!
      - Static leaves all on CS-1 -> R0 misses deadline=120s.
      - Queue-Reroute cannot reroute requests already in decode -> R0 still misses.
      - Dyn-CF migrates R0 (or co-runners) to idle CS-2 -> R0 finishes < 120s!
    """
    cfg = _l0_cfg()
    gpu = cfg["gpu"]
    it_solo = iter_time(1, 2500.0, 1.0, gpu)
    it_crowd = iter_time(7, 30000.0, 1.0, gpu)
    assert it_crowd > it_solo * 1.15, f"Expected >=15% batch slowdown, got {it_crowd / it_solo:.3f}"

    tr = _two_compute_trace(320.0)
    # R0 arrives at t=0 (predicted=4000, actual=8000, deadline=120s) on CS-1.
    # 6 co-running requests arrive at t=6.0s (right after t=5s tick, so they enter decode before t=10s tick!)
    # and all map initially to CS-1.
    def make_reqs():
        reqs = [
            Request(
                i=0, origin=0, t=0.0, long=True, pin=500,
                pout=8000, upload_bytes=10e6, deadline=125.0,
                output_budget=4000, predicted_pout=4000,
            )
        ]
        for k in range(1, 7):
            reqs.append(
                Request(
                    i=k, origin=0, t=6.0 + 0.05 * k, long=True, pin=500,
                    pout=8000, upload_bytes=10e6, deadline=130.0,
                    output_budget=8000, predicted_pout=8000,
                )
            )
        return reqs

    targets = {k: 1 for k in range(7)}
    inv = RebalanceInvariants.from_cfg(cfg)

    def run_case(arm: str, planner_factory, world_cls=DiagnosticWorld, allow_reroute=False):
        env = simpy.Environment()
        rec = Recorder(2.0)
        reqs = make_reqs()
        w = world_cls(env, tr, copy.deepcopy(cfg), policy=CompositePolicy(ReplayPlacer(targets), NonePlanner(), inv), recorder=rec, seed=1, allow_queue_reroute=allow_reroute)
        w.policy.planner = planner_factory(w)
        for r in reqs:
            w.submit(r)
        env.run(until=200.0)
        return w, w.reqs[0]

    _, r0_static = run_case("static", lambda w: NonePlanner())
    _, r0_qr = run_case("queue_reroute", lambda w: QueueReroutePlanner(w), allow_reroute=True)
    w_cf, r0_cf = run_case("dyn_cf", lambda w: CausalReactivePlanner(mode="naive", enable_eclipse_trigger=False, enable_k1_trigger=True), world_cls=ContentionFreeDiagnosticWorld)

    assert r0_static.t_delivered is not None and r0_static.t_delivered > r0_static.deadline, f"Static R0={r0_static.t_delivered}"
    assert r0_qr.t_delivered is not None and r0_qr.t_delivered > r0_qr.deadline, f"QR R0={r0_qr.t_delivered}"
    assert r0_cf.t_delivered is not None and r0_cf.t_delivered <= r0_cf.deadline, f"CF R0={r0_cf.t_delivered}"
    assert any(r.n_mig > 0 for r in w_cf.reqs), "Expected at least one K1-triggered migration"


def test_B_spare_capacity() -> None:
    """K2: CS-1 enters eclipse at t=15s; CS-2 is sunlit. Test positive + no_contact & no_memory negatives."""
    cfg = _l0_cfg()
    inv = RebalanceInvariants.from_cfg(cfg)

    def run_b(inter_up, cs2_prefill_kv: bool):
        tr = _two_compute_trace(320.0, dark=[[], [[15.0, 200.0]], []], inter_up=inter_up)
        reqs = [
            Request(i=0, origin=0, t=0.0, long=True, pin=500, pout=10000, upload_bytes=10e6, deadline=180.0, output_budget=10000, predicted_pout=10000)
        ]
        targets = {0: 1}
        if cs2_prefill_kv:
            # Fill CS-2 (24 GiB = 196,608 tokens at bpt=131072) with a long-lived resident request
            reqs.append(Request(i=1, origin=0, t=0.0, long=True, pin=145000, pout=50000, upload_bytes=1e6, deadline=300.0, output_budget=50000, predicted_pout=50000))
            targets[1] = 2

        env = simpy.Environment()
        planner = CausalReactivePlanner(mode="naive", enable_eclipse_trigger=True, enable_k1_trigger=True)
        w = ContentionFreeDiagnosticWorld(env, tr, copy.deepcopy(cfg), policy=CompositePolicy(ReplayPlacer(targets), planner, inv), seed=1)
        for r in reqs:
            w.submit(r)
        env.run(until=300.0)
        return w.reqs[0], planner.decision_log

    # Positive: CS-2 reachable & has KV headroom -> rescued
    r0_pos, log_pos = run_b([[0.0, 320.0]], False)
    assert r0_pos.t_delivered is not None and r0_pos.t_delivered <= r0_pos.deadline
    assert r0_pos.n_mig == 1
    assert any(d["feasible_targets"] == 1 and d["chosen_dst"] == 2 for d in log_pos)

    # Negative 1: contact window closed after t=10s -> no_contact
    r0_nocontact, log_nocontact = run_b([[0.0, 10.0]], False)
    assert r0_nocontact.n_mig == 0
    assert r0_nocontact.t_delivered is None or r0_nocontact.t_delivered > r0_nocontact.deadline
    assert any(d["reachable_targets"] == 0 and d["reject_reasons"].get("no_contact", 0) >= 1 for d in log_nocontact)

    # Negative 2: CS-2 KV full -> no_memory
    r0_nomem, log_nomem = run_b([[0.0, 320.0]], True)
    assert r0_nomem.n_mig == 0
    assert r0_nomem.t_delivered is None or r0_nomem.t_delivered > r0_nomem.deadline
    assert any(d["reachable_targets"] == 1 and d["feasible_targets"] == 0 and d["reject_reasons"].get("no_memory", 0) >= 1 for d in log_nomem)


def test_C_isl_contention() -> None:
    """K3: Shared ISL contention. CS-2 & CS-3 have identical compute; edge (CS-1, CS-2) has background flow.

    Hand calculation:
      - bpt = 2 * 32 * 8 * 128 * 2 = 131,072 bytes/tok (Wait: 32*8*128*2*2 = 131,072 B/tok!).
      - Request R0 has pin=500, and by t=20s generates ~1500 tokens -> kv ~ 2000 tokens -> 262.14 MB.
      - Actually, with pin=4000 and ~1500 tokens -> kv ~ 5500 tokens -> 720.9 MB.
      - On 0.6 Gbps (75 MB/s): exclusive transfer = 720.9 / 75 = 9.61s (+ 0.005s hop).
      - With 1 background flow sharing (CS-1, CS-2) max-min rate = 37.5 MB/s -> transfer = 19.22s!
      - That 9.6s difference determines whether R0 meets a tight deadline!
    """
    cfg = _l0_cfg(inter_gbps=0.6)
    bpt = kv_bytes_per_tok(cfg["model"])
    assert bpt == 131072
    # Verify allocate_max_min_rates directly
    rates = allocate_max_min_rates(
        [(0, [(1, 2, 10)]), (1, [(1, 2, 10)])],
        lambda e: 75e6,
    )
    assert abs(rates[0] - 37.5e6) < 1.0 and abs(rates[1] - 37.5e6) < 1.0

    horizon = 250.0
    tr = Trace({
        "schema": SCHEMA,
        "name": "l0-4star-k3",
        "horizon_s": horizon,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "CS-1", "role": "compute"},
            {"id": 2, "name": "CS-2", "role": "compute"},
            {"id": 3, "name": "CS-3", "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, horizon]]},
            {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, horizon]]},
            {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, horizon]]},
        ],
        "dark": [[], [[20.0, 200.0]], [], []],
    })
    inv = RebalanceInvariants.from_cfg(cfg)

    def run_c(mode: str, congest_both: bool = False):
        env = simpy.Environment()
        planner = CausalReactivePlanner(mode=mode, enable_eclipse_trigger=True, enable_k1_trigger=False)
        # R0: pin=6000, pout=8000, arrives t=0 on CS-1.
        # At t=20s (when CS-1 enters eclipse), R0 has generated ~1300 toks -> kv ~ 7300 toks = 956.8 MB.
        # Exclusive on CS-1->CS-3 (75 MB/s): ~12.76s. Remaining ~6700 toks on CS-3 (~0.0138s/tok) -> ~92.5s.
        # Total finish ~ 20 + 12.8 + 92.5 = 125.3s.
        # Shared on CS-1->CS-2 (37.5 MB/s): ~25.5s -> finish ~ 20 + 25.5 + 92.5 = 138.0s.
        # Tight deadline = 132.0s separates Aware (CS-3, <=132s) from Naive (CS-2, >132s)!
        r0 = Request(i=0, origin=0, t=0.0, long=True, pin=6000, pout=8000, upload_bytes=10e6, deadline=132.0, output_budget=8000, predicted_pout=8000)
        w = DiagnosticWorld(env, tr, copy.deepcopy(cfg), policy=CompositePolicy(ReplayPlacer({0: 1}), planner, inv), seed=1)

        def bg_injector():
            yield env.timeout(19.0)
            w.net.add("bg", [1, 2], [1], 3.0e9, req=-1)
            if congest_both:
                w.net.add("bg", [1, 3], [2], 3.0e9, req=-2)

        env.process(bg_injector())
        w.submit(r0)
        env.run(until=200.0)
        return w.reqs[0]

    r0_naive = run_c("naive", congest_both=False)
    r0_aware = run_c("aware", congest_both=False)
    r0_neg = run_c("aware", congest_both=True)

    assert r0_naive.trail == [1, 2] and r0_naive.t_delivered is not None and r0_naive.t_delivered > 132.0, f"Naive={r0_naive.t_delivered}, trail={r0_naive.trail}"
    assert r0_aware.trail == [1, 3] and r0_aware.t_delivered is not None and r0_aware.t_delivered <= 132.0, f"Aware={r0_aware.t_delivered}, trail={r0_aware.trail}"
    assert r0_neg.t_delivered is not None and r0_neg.t_delivered > 132.0, f"Negative control should also miss deadline: {r0_neg.t_delivered}"
