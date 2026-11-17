"""Pure-data R3 time-expanded Two-Stage Goodput-CVaR MILP solver (zero SimPy/CoWorld dependency)."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from satplan.actions import MigrationOrder, MigrationSessionSnapshot, PlanAction


@dataclass(frozen=True)
class R3TierSpec:
    index: int
    power_w: float
    rate_tok_s: float


@dataclass(frozen=True)
class R3EclipseWindowSpec:
    budget_j: float | None
    slot_durations_s: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class R3NodeSpec:
    node_id: int
    init_temp_C: float
    kv_cap_bytes: float
    is_eclipse: tuple[bool, ...]
    init_dod_remaining_j: float | None = None
    tier_rates_tok_s: tuple[float, ...] | None = None
    sunlit_s: tuple[float, ...] | None = None
    eclipse_windows: tuple[R3EclipseWindowSpec, ...] | None = None
    max_event_offset_s: tuple[float, ...] | None = None


@dataclass(frozen=True)
class R3EdgeSpec:
    src: int
    dst: int
    tot_bps: tuple[float, ...]
    bg_bps: tuple[float, ...] | None = None


@dataclass(frozen=True)
class R3RequestSpec:
    req_id: int
    init_node: int
    kv_bytes: float
    remaining_tokens: float
    deadline_s: float
    weight: float = 1.0
    elapsed_s: float = 0.0
    reserved_kv_bytes: float | None = None
    currently_paused: bool = False
    init_mig_dst: int | None = None
    init_mig_received_bytes: float = 0.0
    init_mig_in_flight: bool = False
    init_mig: MigrationSessionSnapshot | None = None
    init_mig_solver_edges: tuple[int, ...] = ()
    n_mig: int = 0


@dataclass(frozen=True)
class R3PredictedDemandSpec:
    demand_id: int
    origin: int
    candidate_nodes: tuple[int, ...]
    arrival_slot: int
    expected_requests: float
    tokens_per_req: float
    kv_bytes_per_req: float
    deadline_s: float
    weight: float = 1.0
    arrival_offset_s: float | None = None


@dataclass(frozen=True)
class R3Problem:
    horizon_slots: int
    dt_s: float
    nodes: tuple[R3NodeSpec, ...]
    tiers: tuple[R3TierSpec, ...]
    edges: tuple[R3EdgeSpec, ...]
    requests: tuple[R3RequestSpec, ...]
    t_amb_C: float = 20.0
    t_max_C: float = 85.0
    r_th: float = 0.2
    c_th: float = 150.0
    eclipse_dod_budget_j: float | None = None
    control_period_s: float | None = None
    max_inflight: int | None = None
    max_mig_per_req: int | None = None
    hop_latency_s: float = 0.0
    routes: dict[tuple[int, int, int], tuple[int, ...]] | None = None
    batch_tier_rates: tuple[tuple[float, ...], ...] | None = None
    cvar_alpha: float = 0.95
    cvar_weight: float = 1e-3
    stage2_eps: float = 1e-5
    time_limit_s: float = 5.0
    slot_dts_s: tuple[float, ...] | None = None
    near_slots: int | None = None
    predicted_demands: tuple[R3PredictedDemandSpec, ...] = ()

    def slot_dt(self, t: int) -> float:
        return self.dt_s if self.slot_dts_s is None else self.slot_dts_s[t]


@dataclass(frozen=True)
class R3Result:
    action: PlanAction
    status: str
    stage1_obj: float
    goodput_weight: float
    cvar_s: float
    migration_bytes: float
    node_tiers: dict[int, tuple[int, ...]]
    node_temps_C: dict[int, tuple[float, ...]]
    goodput_req_ids: tuple[int, ...]
    wall_time_s: float = 0.0
    mip_gap: float | None = None
    fallback_reason: str | None = None
    stage1_status: str = "optimal"
    stage1_wall_time_s: float = 0.0
    stage1_mip_gap: float | None = None
    stage2_status: str = "not_run"
    stage2_wall_time_s: float = 0.0
    stage2_mip_gap: float | None = None


def _compute_slot_bounds(slot_dts: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    starts: list[float] = []
    ends: list[float] = []
    acc = 0.0
    for dt_t in slot_dts:
        starts.append(acc)
        acc += dt_t
        ends.append(acc)
    return tuple(starts), tuple(ends), acc


class _VarLayout:
    """Contiguous variable index layout for the R3 MILP."""

    def __init__(self, problem: R3Problem) -> None:
        self.H = problem.horizon_slots
        self.slot_dts: tuple[float, ...] = (
            problem.slot_dts_s if problem.slot_dts_s is not None else (problem.dt_s,) * self.H
        )
        self.slot_starts, self.slot_ends, self.total_horizon_s = _compute_slot_bounds(self.slot_dts)
        self.J = len(problem.nodes)
        self.L = len(problem.tiers)
        self.E = len(problem.edges)
        self.R = len(problem.requests)
        if self.R == 0 or self.E == 0:
            self.K_max = 0
        elif problem.max_inflight is not None:
            self.K_max = max(0, min(self.R, problem.max_inflight))
        else:
            self.K_max = self.R
        self.B_ks: tuple[int, ...] = self._compute_batch_breakpoints(problem)
        self.B_max = len(self.B_ks)

        sizes = (
            ("off_z", self.J * self.L * self.H),
            ("off_theta", self.J * self.H),
            ("off_x", self.R * self.J * self.H),
            ("off_h", self.R * self.J * self.H),
            ("off_u", self.R * self.J * self.H),
            ("off_m", self.R * self.J * self.H),
            ("off_tx", self.R * self.H),
            ("off_start_flow", self.R * self.H),
            ("off_cont_flow", self.R * self.H),
            ("off_tx_complete", self.R * self.H),
            ("off_init_chunk_done", self.R * self.H),
            ("off_init_chunk_active", self.R * self.H),
            ("off_comp", self.R * self.H),
            ("off_y", self.R),
            ("off_w", self.E * self.H * self.K_max),
            ("off_b_ge", self.J * self.H * self.B_max),
            ("off_g", self.R * self.J * self.H),
            ("off_unlock", self.R * self.J * self.H),
            ("off_tx_just_done", self.R * self.H),
            ("off_init_chunk_just_done", self.R * self.H),
            ("off_b", self.R * self.H),
            ("off_tx_frac", self.R * self.H),
            ("off_prop_rem", self.R * self.H),
            ("off_init_prop_rem", self.R * self.H),
            ("off_d", self.R * self.J * self.H),
            ("off_arc", self.R * self.J * self.J * self.H),
            ("off_arc_frozen", self.R * self.H),
            ("off_e_use", self.R * self.E * self.H),
            ("off_b_edge", self.R * self.E * self.H),
            ("off_flow_hops", self.R * self.H),
            ("off_eta", 1),
            ("off_xi", self.R),
        )
        off = 0
        for name, count in sizes:
            setattr(self, name, off)
            off += count
        self.sunlit_s: list[list[float]] = []
        self.ecl_windows: list[tuple[R3EclipseWindowSpec, ...]] = []
        self._a_ecl_map: dict[tuple[int, int, int, int], int] = {}
        self._dod_start_map: dict[tuple[int, int, int], int] = {}
        self._a_ecl_by_slot: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
        self._ecl_win_slots: dict[tuple[int, int], list[tuple[int, float]]] = {}
        ctrl_dt = self.slot_dts[0] if problem.control_period_s is None else problem.control_period_s
        for j, node in enumerate(problem.nodes):
            s_s, e_wins = _resolve_node_power_windows(problem, node)
            sun_j = list(s_s)
            self.sunlit_s.append(sun_j)
            self.ecl_windows.append(e_wins)
            for w_idx, win in enumerate(e_wins):
                budget = win.budget_j
                if budget is None:
                    for t, dur in win.slot_durations_s:
                        if dur > 1e-12:
                            sun_j[t] += dur
                    continue
                if budget <= 1e-9:
                    continue
                slots = [(t, dur) for t, dur in win.slot_durations_s if dur > 1e-12]
                if not slots:
                    continue
                self._ecl_win_slots[(j, w_idx)] = slots
                for k, (t, dur) in enumerate(slots):
                    self._dod_start_map[(j, w_idx, k)] = off
                    off += 1
                    for l_i, tier in enumerate(problem.tiers):
                        if not (tier.power_w > 0.0 or tier.rate_tok_s > 0.0):
                            continue
                        extra_dur = dur if ctrl_dt < self.slot_dts[t] - 1e-9 else 0.0
                        if tier.power_w * (ctrl_dt + extra_dur) > budget + 1e-9:
                            continue
                        key = (j, w_idx, k, l_i)
                        self._a_ecl_map[key] = off
                        self._a_ecl_by_slot.setdefault((j, t, l_i), []).append((off, dur))
                        off += 1
        self.D = len(problem.predicted_demands)
        self._p_place: dict[tuple[int, int], int] = {}
        self._p_active: dict[tuple[int, int, int], int] = {}
        self._p_g: dict[tuple[int, int, int], int] = {}
        self._p_comp: dict[tuple[int, int], int] = {}
        self._p_y: dict[int, int] = {}
        off = self._init_pred_vars(problem, off)
        self.total = off

    def _init_pred_vars(self, problem: R3Problem, off: int) -> int:
        node_pos = {n.node_id: j for j, n in enumerate(problem.nodes)}
        for d_i, dem in enumerate(problem.predicted_demands):
            self._p_y[d_i] = off
            off += 1
            for t in range(max(1, dem.arrival_slot), self.H):
                self._p_comp[(d_i, t)] = off
                off += 1
            for nid in dem.candidate_nodes:
                j = node_pos.get(nid)
                if j is None:
                    continue
                self._p_place[(d_i, j)] = off
                off += 1
                for t in range(max(1, dem.arrival_slot), self.H):
                    self._p_active[(d_i, j, t)] = off
                    off += 1
                    self._p_g[(d_i, j, t)] = off
                    off += 1
        return off

    @staticmethod
    def _compute_batch_breakpoints(problem: R3Problem) -> tuple[int, ...]:
        if problem.batch_tier_rates is None or len(problem.requests) < 2:
            return ()
        rates = problem.batch_tier_rates
        r_max = len(problem.requests)
        ref = rates[0]
        max_ref = max(ref, default=0.0)
        if max_ref <= 0.0:
            return ()
        # Only add binary breakpoints when batch degradation exceeds 1%
        if all((ref[l_i] - rates[-1][l_i]) <= 0.01 * max_ref for l_i in range(len(ref))):
            return ()
        ks: list[int] = []
        last_r = ref
        for k in range(2, r_max + 1):
            cur = rates[min(k - 1, len(rates) - 1)]
            if any((last_r[l_i] - cur[l_i]) > 0.01 * max_ref for l_i in range(len(ref))):
                ks.append(k)
                last_r = cur
        return tuple(ks)

    def z(self, j: int, l_idx: int, t: int) -> int:
        return self.off_z + (j * self.L + l_idx) * self.H + t

    def theta(self, j: int, t: int) -> int:
        return self.off_theta + j * self.H + t

    def x(self, r: int, j: int, t: int) -> int:
        return self.off_x + (r * self.J + j) * self.H + t

    def h(self, r: int, j: int, t: int) -> int:
        return self.off_h + (r * self.J + j) * self.H + t

    def u(self, r: int, j: int, t: int) -> int:
        return self.off_u + (r * self.J + j) * self.H + t

    def m(self, r: int, j: int, t: int) -> int:
        return self.off_m + (r * self.J + j) * self.H + t

    def unlock(self, r: int, j: int, t: int) -> int:
        return self.off_unlock + (r * self.J + j) * self.H + t

    def tx(self, r: int, t: int) -> int:
        return self.off_tx + r * self.H + t

    def start_flow(self, r: int, t: int) -> int:
        return self.off_start_flow + r * self.H + t

    def cont_flow(self, r: int, t: int) -> int:
        return self.off_cont_flow + r * self.H + t

    def tx_complete(self, r: int, t: int) -> int:
        return self.off_tx_complete + r * self.H + t

    def init_chunk_done(self, r: int, t: int) -> int:
        return self.off_init_chunk_done + r * self.H + t

    def init_chunk_active(self, r: int, t: int) -> int:
        return self.off_init_chunk_active + r * self.H + t

    def tx_just_done(self, r: int, t: int) -> int:
        return self.off_tx_just_done + r * self.H + t

    def init_chunk_just_done(self, r: int, t: int) -> int:
        return self.off_init_chunk_just_done + r * self.H + t

    def comp(self, r: int, t: int) -> int:
        return self.off_comp + r * self.H + t

    def y(self, r: int) -> int:
        return self.off_y + r

    def w(self, e: int, t: int, k: int) -> int:
        return self.off_w + ((e * self.H + t) * self.K_max) + (k - 1)

    def b_ge(self, j: int, t: int, b_idx: int) -> int:
        # b_idx in 0..B_max-1
        return self.off_b_ge + ((j * self.H + t) * self.B_max) + b_idx

    def g(self, r: int, j: int, t: int) -> int:
        return self.off_g + (r * self.J + j) * self.H + t

    def b(self, r: int, t: int) -> int:
        return self.off_b + r * self.H + t

    def tx_frac(self, r: int, t: int) -> int:
        return self.off_tx_frac + r * self.H + t

    def prop_rem(self, r: int, t: int) -> int:
        return self.off_prop_rem + r * self.H + t

    def init_prop_rem(self, r: int, t: int) -> int:
        return self.off_init_prop_rem + r * self.H + t

    def d(self, r: int, j: int, t: int) -> int:
        return self.off_d + (r * self.J + j) * self.H + t

    def arc(self, r: int, s: int, d: int, t: int) -> int:
        return self.off_arc + (((r * self.J + s) * self.J + d) * self.H) + t

    def arc_frozen(self, r: int, t: int) -> int:
        return self.off_arc_frozen + r * self.H + t

    def e_use(self, r: int, e: int, t: int) -> int:
        return self.off_e_use + (r * self.E + e) * self.H + t

    def b_edge(self, r: int, e: int, t: int) -> int:
        return self.off_b_edge + (r * self.E + e) * self.H + t

    def flow_hops(self, r: int, t: int) -> int:
        return self.off_flow_hops + r * self.H + t

    def eta(self) -> int:
        return self.off_eta

    def xi(self, r: int) -> int:
        return self.off_xi + r


class _ConstraintBuilder:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []

    def add(self, terms: list[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(self.lb)
        for col, val in terms:
            if val != 0.0:
                self.rows.append(row)
                self.cols.append(col)
                self.vals.append(float(val))
        self.lb.append(float(lb))
        self.ub.append(float(ub))

    def build(self, n_vars: int) -> LinearConstraint:
        mat = coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.lb), n_vars)).tocsc()
        return LinearConstraint(mat, np.asarray(self.lb, dtype=float), np.asarray(self.ub, dtype=float))


def _eclipse_windows(is_eclipse: tuple[bool, ...]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for t, ecl in enumerate(is_eclipse):
        if ecl and start is None:
            start = t
        elif not ecl and start is not None:
            windows.append((start, t))
            start = None
    if start is not None:
        windows.append((start, len(is_eclipse)))
    return windows


def _resolve_node_power_windows(
    problem: R3Problem, node: R3NodeSpec
) -> tuple[tuple[float, ...], tuple[R3EclipseWindowSpec, ...]]:
    sunlit_s = (
        node.sunlit_s
        if node.sunlit_s is not None
        else tuple(0.0 if ecl else problem.slot_dt(t) for t, ecl in enumerate(node.is_eclipse))
    )
    if node.eclipse_windows is not None:
        return sunlit_s, node.eclipse_windows
    ecl_windows = tuple(
        R3EclipseWindowSpec(
            budget_j=(
                node.init_dod_remaining_j
                if (w_start == 0 and node.init_dod_remaining_j is not None)
                else problem.eclipse_dod_budget_j
            ),
            slot_durations_s=tuple((t, problem.slot_dt(t)) for t in range(w_start, w_end)),
        )
        for w_start, w_end in _eclipse_windows(node.is_eclipse)
    )
    return sunlit_s, ecl_windows


def _shortest_path_edges(
    problem: R3Problem, node_pos: dict[int, int], src_j: int, dst_j: int, t: int
) -> tuple[int, ...] | None:
    if src_j == dst_j:
        return ()
    adj: list[list[tuple[int, int]]] = [[] for _ in range(len(problem.nodes))]
    for e_i, edge in enumerate(problem.edges):
        bg = 0.0 if edge.bg_bps is None else edge.bg_bps[t]
        if edge.tot_bps[t] - bg <= 0.0:
            continue
        if edge.src in node_pos and edge.dst in node_pos:
            adj[node_pos[edge.src]].append((node_pos[edge.dst], e_i))
    dist: dict[int, int] = {src_j: 0}
    prev: dict[int, tuple[int, int]] = {}
    pq: list[tuple[int, int]] = [(0, src_j)]
    while pq:
        d_cur, u = heapq.heappop(pq)
        if u == dst_j:
            path: list[int] = []
            cur = dst_j
            while cur != src_j:
                p_u, e_i = prev[cur]
                path.append(e_i)
                cur = p_u
            path.reverse()
            return tuple(path)
        if d_cur > dist.get(u, 10**9):
            continue
        for v, e_i in adj[u]:
            nd = d_cur + 1
            if nd < dist.get(v, 10**9):
                dist[v] = nd
                prev[v] = (u, e_i)
                heapq.heappush(pq, (nd, v))
    return None


def _resolve_routes(
    problem: R3Problem, node_pos: dict[int, int]
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    routes: dict[tuple[int, int, int], tuple[int, ...]] = {}
    n_nodes = len(problem.nodes)
    for t in range(problem.horizon_slots):
        for s_j in range(n_nodes):
            s_id = problem.nodes[s_j].node_id
            for d_j in range(n_nodes):
                if s_j == d_j:
                    continue
                d_id = problem.nodes[d_j].node_id
                eids = (
                    problem.routes.get((s_id, d_id, t))
                    if problem.routes is not None
                    else _shortest_path_edges(problem, node_pos, s_j, d_j, t)
                )
                if eids:
                    routes[(s_j, d_j, t)] = tuple(eids)
    return routes


def _effective_init_mig(req: R3RequestSpec) -> MigrationSessionSnapshot | None:
    if req.init_mig is not None:
        return None if req.init_mig.phase == "idle" else req.init_mig
    if req.init_mig_dst is not None:
        phase = "sending" if req.init_mig_in_flight else "resumable"
        rem = max(0.0, req.kv_bytes - req.init_mig_received_bytes)
        return MigrationSessionSnapshot(
            phase=phase,
            dst=req.init_mig_dst,
            target_bytes=req.kv_bytes,
            received_bytes=req.init_mig_received_bytes,
            chunk_rem_bytes=rem if req.init_mig_in_flight else 0.0,
        )
    return None


def _classify_milp_status(res: object) -> tuple[str, str]:
    st = getattr(res, "status", None)
    msg = str(getattr(res, "message", ""))
    has_x = getattr(res, "x", None) is not None
    if st == 0 and has_x:
        return "optimal", ""
    if st == 1 and has_x:
        return "feasible_time_limit", f"time_limit_with_incumbent: {msg}"
    if st == 1:
        return "timeout_no_incumbent", f"time_limit_no_incumbent: {msg}"
    if st == 2:
        return "proven_infeasible", f"proven_infeasible: {msg}"
    return "error", f"solver_error(status={st}): {msg}"


class R3Solver:
    """Two-stage Goodput-CVaR MILP solver using scipy.optimize.milp (HiGHS)."""

    def solve(self, problem: R3Problem) -> R3Result:
        t_start = time.perf_counter()
        try:
            self._validate(problem)
        except ValueError as exc:
            return self._failure_result(problem, "error", f"invalid_problem: {exc}", time.perf_counter() - t_start)

        if not problem.requests and not problem.predicted_demands:
            idle_idx = min(problem.tiers, key=lambda t: t.power_w).index
            wall = time.perf_counter() - t_start
            return R3Result(
                action=PlanAction(tier_decisions={n.node_id: idle_idx for n in problem.nodes}),
                status="optimal",
                stage1_obj=0.0,
                goodput_weight=0.0,
                cvar_s=0.0,
                migration_bytes=0.0,
                node_tiers={n.node_id: (idle_idx,) * problem.horizon_slots for n in problem.nodes},
                node_temps_C={n.node_id: (n.init_temp_C,) * problem.horizon_slots for n in problem.nodes},
                goodput_req_ids=(),
                wall_time_s=wall,
                mip_gap=0.0,
                stage1_status="optimal",
                stage1_wall_time_s=wall,
                stage1_mip_gap=0.0,
            )

        idx = _VarLayout(problem)
        node_pos = {n.node_id: j for j, n in enumerate(problem.nodes)}
        routes = _resolve_routes(problem, node_pos)
        integrality, lb, ub = self._build_bounds(problem, idx, node_pos, routes)
        cb = _ConstraintBuilder()

        self._add_power_thermal_constraints(problem, idx, cb)
        self._add_request_compute_constraints(problem, idx, cb)
        self._add_predicted_demand_constraints(problem, idx, cb)
        self._add_migration_state_constraints(problem, idx, node_pos, routes, cb)
        self._add_network_constraints(problem, idx, routes, cb)
        self._add_cvar_constraints(problem, idx, cb)

        c1 = self._stage1_cost(problem, idx)
        t1_0 = time.perf_counter()
        try:
            res1 = milp(
                c=c1,
                integrality=integrality,
                bounds=Bounds(lb, ub),
                constraints=cb.build(idx.total),
                options={"time_limit": problem.time_limit_s},
            )
        except Exception as exc:  # noqa: BLE001
            return self._failure_result(problem, "error", f"milp_exception: {exc}", time.perf_counter() - t_start)
        s1_wall = time.perf_counter() - t1_0

        s1_status, s1_msg = _classify_milp_status(res1)
        if getattr(res1, "x", None) is None:
            return self._failure_result(problem, s1_status, s1_msg, time.perf_counter() - t_start)

        s1_gap_raw = getattr(res1, "mip_gap", None)
        s1_gap = None if s1_gap_raw is None else float(s1_gap_raw)
        v1_star = float(-np.dot(c1, res1.x))

        cb.add([(col, float(c1[col])) for col in np.nonzero(c1)[0]], -math.inf, -v1_star + problem.stage2_eps)
        c2 = np.zeros(idx.total, dtype=float)
        for r, req in enumerate(problem.requests):
            g_tie = -1e-6 / max(1.0, req.remaining_tokens)
            for e in range(idx.E):
                for t in range(idx.H):
                    c2[idx.b_edge(r, e, t)] = 1.0
            for j in range(idx.J):
                for t in range(idx.H):
                    # Break exact ties against holding destination KV / migrating earlier than needed,
                    # and against idling/pausing uncompleted work when y(r) == 0.
                    c2[idx.m(r, j, t)] = 1e-4 * (idx.H - t)
                    c2[idx.h(r, j, t)] = 1e-3
                    c2[idx.g(r, j, t)] = g_tie

        t2_0 = time.perf_counter()
        res2 = None
        s2_exc: str | None = None
        try:
            res2 = milp(
                c=c2,
                integrality=integrality,
                bounds=Bounds(lb, ub),
                constraints=cb.build(idx.total),
                options={"time_limit": problem.time_limit_s, "mip_rel_gap": 0.0},
            )
        except Exception as exc:  # noqa: BLE001
            s2_exc = f"stage2_exception: {exc}"
        s2_wall = time.perf_counter() - t2_0

        s2_status, s2_msg = ("error", s2_exc) if s2_exc is not None else _classify_milp_status(res2)
        s2_gap_raw = None if res2 is None else getattr(res2, "mip_gap", None)
        s2_gap = None if s2_gap_raw is None else float(s2_gap_raw)

        use_res2 = res2 is not None and getattr(res2, "x", None) is not None
        x_sol = res2.x if use_res2 else res1.x
        if s1_status == "optimal" and s2_status == "optimal":
            overall_status, overall_gap, fb_reason = "optimal", s1_gap, None
        elif use_res2:
            overall_status = "feasible_time_limit"
            overall_gap = s2_gap if s2_gap is not None else s1_gap
            fb_reason = s1_msg or s2_msg or "suboptimal_time_limit"
        else:
            overall_status = "feasible_stage2_fallback"
            overall_gap = None
            fb_reason = f"stage2_{s2_status}: {s2_msg}"

        return self._extract_result(
            problem,
            idx,
            node_pos,
            x_sol,
            c1,
            overall_status,
            time.perf_counter() - t_start,
            overall_gap,
            fb_reason,
            (s1_status, s1_wall, s1_gap),
            (s2_status, s2_wall, s2_gap),
        )

    @staticmethod
    def _validate_node(problem: R3Problem, n: R3NodeSpec) -> None:
        if len(n.is_eclipse) != problem.horizon_slots:
            raise ValueError(f"node {n.node_id} is_eclipse length must match horizon_slots")
        if n.sunlit_s is not None and len(n.sunlit_s) != problem.horizon_slots:
            raise ValueError(f"node {n.node_id} sunlit_s length must match horizon_slots")
        if n.max_event_offset_s is not None and len(n.max_event_offset_s) != problem.horizon_slots:
            raise ValueError(f"node {n.node_id} max_event_offset_s length must match horizon_slots")
        if n.eclipse_windows is not None:
            for win in n.eclipse_windows:
                for t, dur in win.slot_durations_s:
                    if not (0 <= t < problem.horizon_slots) or dur < 0.0:
                        raise ValueError(f"node {n.node_id} invalid eclipse window slot ({t}, {dur})")
        if n.tier_rates_tok_s is not None and len(n.tier_rates_tok_s) != len(problem.tiers):
            raise ValueError(f"node {n.node_id} tier_rates_tok_s length must match tiers")

    def _validate(self, problem: R3Problem) -> None:
        if problem.horizon_slots <= 0 or problem.dt_s <= 0.0:
            raise ValueError("horizon_slots and dt_s must be positive")
        if problem.slot_dts_s is not None and (
            len(problem.slot_dts_s) != problem.horizon_slots or any(d <= 0.0 for d in problem.slot_dts_s)
        ):
            raise ValueError("slot_dts_s length must match horizon_slots and all entries must be positive")
        if problem.control_period_s is not None and problem.control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive")
        if not problem.nodes or not problem.tiers:
            raise ValueError("nodes and tiers must be non-empty")
        node_ids = {n.node_id for n in problem.nodes}
        for n in problem.nodes:
            self._validate_node(problem, n)
        for e in problem.edges:
            if len(e.tot_bps) != problem.horizon_slots:
                raise ValueError(f"edge ({e.src}, {e.dst}) tot_bps length must match horizon_slots")
            if e.bg_bps is not None and len(e.bg_bps) != problem.horizon_slots:
                raise ValueError(f"edge ({e.src}, {e.dst}) bg_bps length must match horizon_slots")
        for req in problem.requests:
            if req.init_node not in node_ids:
                raise ValueError(f"request {req.req_id} init_node {req.init_node} not in nodes")
            if req.init_mig_dst is not None and req.init_mig_dst not in node_ids:
                raise ValueError(f"request {req.req_id} init_mig_dst {req.init_mig_dst} not in nodes")

    def _build_bounds(
        self,
        problem: R3Problem,
        idx: _VarLayout,
        node_pos: dict[int, int],
        routes: dict[tuple[int, int, int], tuple[int, ...]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        integrality = np.zeros(idx.total, dtype=int)
        lb = np.zeros(idx.total, dtype=float)
        ub = np.full(idx.total, math.inf, dtype=float)

        for start, end in ((idx.off_z, idx.off_theta), (idx.off_x, idx.off_g)):
            integrality[start:end] = 1
            ub[start:end] = 1.0

        ub[idx.off_unlock : idx.off_b] = 1.0
        ub[idx.off_tx_frac : idx.off_prop_rem] = 1.0
        ub[idx.off_arc : idx.off_b_edge] = 1.0
        integrality[idx.off_e_use : idx.off_b_edge] = 1

        tau = problem.r_th * problem.c_th
        alpha0 = math.exp(-idx.slot_dts[0] / tau) if tau > 0.0 else 0.0
        for j, node in enumerate(problem.nodes):
            t_cool_0 = alpha0 * node.init_temp_C + (1.0 - alpha0) * problem.t_amb_C
            for t in range(idx.H):
                col = idx.theta(j, t)
                lb[col] = -math.inf
                ub[col] = max(problem.t_max_C, t_cool_0) if t == 0 else problem.t_max_C

        for r, req in enumerate(problem.requests):
            self._apply_request_bounds(problem, idx, node_pos, routes, lb, ub, r, req)

        for var_id in idx._a_ecl_map.values():
            integrality[var_id] = 1
            ub[var_id] = 1.0
        for var_id in (*idx._p_place.values(), *idx._p_comp.values(), *idx._p_y.values()):
            integrality[var_id] = 1
            ub[var_id] = 1.0
        for var_id in idx._p_active.values():
            ub[var_id] = 1.0
        for (j, w_idx, k), var_id in idx._dod_start_map.items():
            budget = idx.ecl_windows[j][w_idx].budget_j
            assert budget is not None
            lb[var_id] = budget if k == 0 else 0.0
            ub[var_id] = budget

        lb[idx.eta()] = -math.inf
        return integrality, lb, ub

    def _apply_request_bounds(  # noqa: PLR0913
        self,
        problem: R3Problem,
        idx: _VarLayout,
        node_pos: dict[int, int],
        routes: dict[tuple[int, int, int], tuple[int, ...]],
        lb: np.ndarray,
        ub: np.ndarray,
        r: int,
        req: R3RequestSpec,
    ) -> None:
        j0 = node_pos[req.init_node]
        ub[idx.cont_flow(r, 0)] = 0.0
        for j in range(idx.J):
            ub[idx.unlock(r, j, 0)] = 0.0
            col_u = idx.u(r, j, 0)
            lb[col_u] = 1.0 if j == j0 else 0.0
            ub[col_u] = 1.0 if j == j0 else 0.0
        snap = _effective_init_mig(req)
        if snap is not None:
            d0 = node_pos[snap.dst]
            for j in range(idx.J):
                col_m = idx.m(r, j, 0)
                lb[col_m] = 1.0 if j == d0 else 0.0
                ub[col_m] = 1.0 if j == d0 else 0.0
            self._apply_init_mig_phase_bounds(problem, idx, lb, ub, r, req, snap)
        else:
            for t in range(idx.H):
                ub[idx.init_chunk_done(r, t)] = 0.0
                ub[idx.init_chunk_just_done(r, t)] = 0.0
                ub[idx.init_chunk_active(r, t)] = 0.0
                ub[idx.init_prop_rem(r, t)] = 0.0
                ub[idx.arc_frozen(r, t)] = 0.0
            if problem.max_mig_per_req is not None and req.n_mig >= problem.max_mig_per_req:
                for j in range(idx.J):
                    for t in range(idx.H):
                        ub[idx.m(r, j, t)] = 0.0
        for t in range(idx.H):
            for s_j in range(idx.J):
                for d_j in range(idx.J):
                    if (s_j, d_j, t) not in routes:
                        ub[idx.arc(r, s_j, d_j, t)] = 0.0

    @staticmethod
    def _apply_init_mig_phase_bounds(  # noqa: PLR0913
        problem: R3Problem,
        idx: _VarLayout,
        lb: np.ndarray,
        ub: np.ndarray,
        r: int,
        req: R3RequestSpec,
        snap: MigrationSessionSnapshot,
    ) -> None:
        if snap.phase == "sending":
            lb[idx.tx(r, 0)] = 1.0
            lb[idx.init_chunk_active(r, 0)] = 1.0
            ub[idx.init_chunk_active(r, 0)] = 1.0
            ub[idx.b(r, 0)] = min(req.kv_bytes, max(0.0, snap.chunk_rem_bytes))
            return
        for t in range(idx.H):
            ub[idx.init_chunk_done(r, t)] = 0.0
            ub[idx.init_chunk_just_done(r, t)] = 0.0
            ub[idx.init_chunk_active(r, t)] = 0.0
            ub[idx.init_prop_rem(r, t)] = 0.0
            ub[idx.arc_frozen(r, t)] = 0.0
        if snap.phase == "propagating_chunk":
            block_slots = next(
                (k + 1 for k, end_s in enumerate(idx.slot_ends) if end_s >= snap.prop_rem_s - 1e-9),
                idx.H,
            )
            for t in range(min(idx.H, block_slots)):
                ub[idx.tx(r, t)] = 0.0
        elif snap.phase == "propagating_unlock":
            ub[idx.tx(r, 0)] = 0.0
            lb[idx.tx_complete(r, 0)] = 1.0
            ub[idx.tx_just_done(r, 0)] = 0.0
            lb[idx.prop_rem(r, 0)] = max(0.0, snap.prop_rem_s - idx.slot_dts[0])

    @staticmethod
    def _add_eclipse_dod_constraints(
        problem: R3Problem, idx: _VarLayout, j: int, cb: _ConstraintBuilder
    ) -> None:
        ctrl_dt = idx.slot_dts[0] if problem.control_period_s is None else problem.control_period_s
        for w_idx, win in enumerate(idx.ecl_windows[j]):
            slots = idx._ecl_win_slots.get((j, w_idx))
            if not slots or win.budget_j is None:
                continue
            budget = win.budget_j
            for k, (t, dur) in enumerate(slots):
                dt_t = idx.slot_dts[t]
                d_k = idx._dod_start_map[(j, w_idx, k)]
                a_vars = [
                    (l_i, problem.tiers[l_i], idx._a_ecl_map[(j, w_idx, k, l_i)])
                    for l_i in range(idx.L)
                    if (j, w_idx, k, l_i) in idx._a_ecl_map
                ]
                if a_vars:
                    cb.add([(a_var, 1.0) for _, _, a_var in a_vars], -math.inf, 1.0)
                if idx.sunlit_s[j][t] <= 1e-12 or ctrl_dt < dt_t - 1e-9:
                    for l_i, tier in enumerate(problem.tiers):
                        if not (tier.power_w > 0.0 or tier.rate_tok_s > 0.0):
                            continue
                        a_var = idx._a_ecl_map.get((j, w_idx, k, l_i))
                        if a_var is None:
                            cb.add([(idx.z(j, l_i, t), 1.0)], 0.0, 0.0)
                        else:
                            cb.add([(idx.z(j, l_i, t), 1.0), (a_var, -1.0)], -math.inf, 0.0)
                extra_dur = dur if ctrl_dt < dt_t - 1e-9 else 0.0
                for a_pos, (l_i, tier, a_var) in enumerate(a_vars):
                    req_dod = tier.power_w * (ctrl_dt + extra_dur)
                    cb.add([(a_var, 1.0)] + [(idx.z(j, m, t), -1.0) for m in range(l_i, idx.L)], -math.inf, 0.0)
                    cb.add([(a_var, req_dod), (d_k, -1.0)], -math.inf, 0.0)
                    # Mandatory admission: if selected tier >= l_i and dod_start >= req_dod, some tier >= l_i must run.
                    mand_terms = [(d_k, 1.0)]
                    mand_terms.extend((idx.z(j, m, t), budget) for m in range(l_i, idx.L))
                    mand_terms.extend((a_var2, -budget) for _, _, a_var2 in a_vars[a_pos:])
                    cb.add(mand_terms, -math.inf, max(0.0, req_dod - 1e-6) + budget)
                if k + 1 < len(slots):
                    d_next = idx._dod_start_map[(j, w_idx, k + 1)]
                    bal_terms = [(d_next, 1.0), (d_k, -1.0)] + [
                        (a_var, tier.power_w * dur) for _, tier, a_var in a_vars
                    ]
                    cb.add(bal_terms, 0.0, 0.0)
                else:
                    end_terms = [(d_k, 1.0)] + [(a_var, -tier.power_w * dur) for _, tier, a_var in a_vars]
                    cb.add(end_terms, 0.0, math.inf)

    def _add_power_thermal_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        tau = problem.r_th * problem.c_th
        ctrl_dt = idx.slot_dts[0] if problem.control_period_s is None else problem.control_period_s
        alpha_ctrl = math.exp(-ctrl_dt / tau) if tau > 0.0 else 0.0
        max_t_eq = max((problem.t_amb_C + problem.r_th * t.power_w for t in problem.tiers), default=problem.t_amb_C)
        zero_tiers = [l_i for l_i, t in enumerate(problem.tiers) if t.power_w <= 0.0 and t.rate_tok_s <= 0.0]

        for j, node in enumerate(problem.nodes):
            m_th = max(0.0, max(node.init_temp_C, problem.t_max_C, max_t_eq) - problem.t_max_C) + 1.0
            for t in range(idx.H):
                dt_t = idx.slot_dts[t]
                alpha = math.exp(-dt_t / tau) if tau > 0.0 else 0.0
                one_minus_alpha = 1.0 - alpha
                cb.add([(idx.z(j, l_i, t), 1.0) for l_i in range(idx.L)], 1.0, 1.0)
                terms: list[tuple[int, float]] = [(idx.theta(j, t), 1.0)]
                if t > 0:
                    terms.append((idx.theta(j, t - 1), -alpha))
                for l_i, tier in enumerate(problem.tiers):
                    terms.append((idx.z(j, l_i, t), -one_minus_alpha * problem.r_th * tier.power_w))
                rhs = one_minus_alpha * problem.t_amb_C + (alpha * node.init_temp_C if t == 0 else 0.0)
                cb.add(terms, rhs, rhs)

                # Enforce current_op full-control-period thermal admission at slot start and worst intra-slot event.
                offset_s = (
                    node.max_event_offset_s[t]
                    if node.max_event_offset_s is not None
                    else (dt_t if ctrl_dt < dt_t - 1e-9 else 0.0)
                )
                alpha_tot = math.exp(-(ctrl_dt + offset_s) / tau) if tau > 0.0 else 0.0
                for l_i, tier in enumerate(problem.tiers):
                    if tier.power_w <= 0.0:
                        continue
                    t_eq = problem.t_amb_C + problem.r_th * tier.power_w
                    for a_eff in (alpha_ctrl, alpha_tot) if offset_s > 1e-12 or ctrl_dt > dt_t + 1e-9 else ():
                        rhs_adm = problem.t_max_C - (1.0 - a_eff) * t_eq + m_th
                        if t == 0:
                            if a_eff * node.init_temp_C + (1.0 - a_eff) * t_eq > problem.t_max_C + 1e-9:
                                cb.add([(idx.z(j, l_i, t), 1.0)], 0.0, 0.0)
                        else:
                            cb.add([(idx.theta(j, t - 1), a_eff), (idx.z(j, l_i, t), m_th)], -math.inf, rhs_adm)

                for z0 in zero_tiers:
                    for r in range(idx.R):
                        cb.add([(idx.x(r, j, t), 1.0), (idx.z(j, z0, t), 1.0)], -math.inf, 1.0)

                if zero_tiers and (idx.R > 0 or idx.D > 0):
                    self._add_non_idle_active_gate(problem, idx, cb, j, t)

            self._add_eclipse_dod_constraints(problem, idx, j, cb)

    @staticmethod
    def _add_non_idle_active_gate(
        problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder, j: int, t: int
    ) -> None:
        active_terms = [(idx.x(r, j, t), -1.0) for r in range(idx.R)]
        for d_i in range(idx.D):
            pa = idx._p_active.get((d_i, j, t))
            if pa is not None:
                active_terms.append((pa, -1.0))
        if not active_terms:
            return
        for l_i, tier in enumerate(problem.tiers):
            if tier.power_w > 0.0:
                cb.add([(idx.z(j, l_i, t), 1.0), *active_terms], -math.inf, 0.0)

    @staticmethod
    def _add_slot_rate_cap(  # noqa: PLR0913
        idx: _VarLayout,
        cb: _ConstraintBuilder,
        j: int,
        t: int,
        r: int,
        rates: tuple[float, ...],
        extra_lhs: tuple[tuple[int, float], ...] = (),
        rhs: float = 0.0,
    ) -> None:
        terms: list[tuple[int, float]] = [(idx.g(r, j, t), 1.0), *extra_lhs]
        for l_i in range(idx.L):
            rate = rates[l_i]
            if rate <= 0.0:
                continue
            terms.append((idx.z(j, l_i, t), -rate * idx.sunlit_s[j][t]))
            for a_var, dur in idx._a_ecl_by_slot.get((j, t, l_i), ()):
                terms.append((a_var, -rate * dur))
        cb.add(terms, -math.inf, rhs)

    def _add_request_compute_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        node_rates = [
            n.tier_rates_tok_s if n.tier_rates_tok_s is not None else tuple(t.rate_tok_s for t in problem.tiers)
            for n in problem.nodes
        ]
        # Base interval [1, k_first - 1] (or [1, R] if no breakpoints) must use the slowest rate in that interval.
        cons_floor: tuple[float, ...] | None = None
        if problem.batch_tier_rates:
            floor_k = (idx.B_ks[0] - 1) if idx.B_ks else len(problem.batch_tier_rates)
            cons_floor = problem.batch_tier_rates[min(max(floor_k - 1, 0), len(problem.batch_tier_rates) - 1)]
        for j in range(idx.J):
            rates_j = (
                tuple(min(node_rates[j][l_i], cons_floor[l_i]) for l_i in range(idx.L))
                if cons_floor is not None
                else node_rates[j]
            )
            max_rate_j = max(rates_j, default=0.0)
            for t in range(idx.H):
                dt_t = idx.slot_dts[t]
                kv_terms = []
                for r, req in enumerate(problem.requests):
                    res_bytes = req.kv_bytes if req.reserved_kv_bytes is None else req.reserved_kv_bytes
                    kv_terms.append((idx.u(r, j, t), res_bytes))
                    kv_terms.append((idx.m(r, j, t), res_bytes))
                for d_i, dem in enumerate(problem.predicted_demands):
                    pa = idx._p_active.get((d_i, j, t))
                    if pa is not None:
                        kv_terms.append((pa, dem.expected_requests * dem.kv_bytes_per_req))
                cb.add(kv_terms, -math.inf, problem.nodes[j].kv_cap_bytes)

                for r, req in enumerate(problem.requests):
                    self._add_slot_rate_cap(idx, cb, j, t, r, rates_j)
                    cb.add(
                        [(idx.g(r, j, t), 1.0), (idx.x(r, j, t), -min(req.remaining_tokens, max_rate_j * dt_t))],
                        -math.inf,
                        0.0,
                    )

                if idx.B_ks and problem.batch_tier_rates is not None:
                    for b_idx, k in enumerate(idx.B_ks):
                        cb.add(
                            [(idx.x(r, j, t), 1.0) for r in range(idx.R)] + [(idx.b_ge(j, t, b_idx), -float(idx.R))],
                            -math.inf,
                            float(k - 1),
                        )
                        k_end = (idx.B_ks[b_idx + 1] - 1) if (b_idx + 1 < len(idx.B_ks)) else idx.R
                        rates_k = problem.batch_tier_rates[min(k_end - 1, len(problem.batch_tier_rates) - 1)]
                        m_big = max(max_rate_j, max(rates_k, default=0.0)) * dt_t
                        for r in range(idx.R):
                            self._add_slot_rate_cap(
                                idx, cb, j, t, r, rates_k, extra_lhs=((idx.b_ge(j, t, b_idx), m_big),), rhs=m_big
                            )

        for r, req in enumerate(problem.requests):
            cb.add([(idx.comp(r, t), 1.0) for t in range(idx.H)], -math.inf, 1.0)
            valid_slots = [t for t in range(idx.H) if idx.slot_ends[t] <= req.deadline_s + 1e-9]
            if valid_slots:
                cb.add([(idx.y(r), 1.0)] + [(idx.comp(r, t), -1.0) for t in valid_slots], -math.inf, 0.0)
            else:
                cb.add([(idx.y(r), 1.0)], 0.0, 0.0)

            for t in range(idx.H):
                tok_terms = [(idx.comp(r, tau), req.remaining_tokens) for tau in range(t + 1)]
                tok_terms.extend((idx.g(r, j, tau), -1.0) for tau in range(t + 1) for j in range(idx.J))
                cb.add(tok_terms, -math.inf, 0.0)

    def _add_predicted_demand_constraints(
        self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder
    ) -> None:
        if idx.D == 0:
            return
        self._add_predicted_node_sharing(problem, idx, cb)
        for d_i, dem in enumerate(problem.predicted_demands):
            self._add_predicted_demand_lifecycle(idx, cb, d_i, dem)

    @staticmethod
    def _add_predicted_node_sharing(
        problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder
    ) -> None:
        node_rates = [
            n.tier_rates_tok_s if n.tier_rates_tok_s is not None else tuple(t.rate_tok_s for t in problem.tiers)
            for n in problem.nodes
        ]
        for j in range(idx.J):
            rates_j = node_rates[j]
            max_rate_j = max(rates_j, default=0.0)
            for t in range(idx.H):
                dt_t = idx.slot_dts[t]
                p_terms: list[tuple[int, float]] = [(idx.g(r, j, t), 1.0) for r in range(idx.R)]
                has_p = False
                for d_i, dem in enumerate(problem.predicted_demands):
                    pg = idx._p_g.get((d_i, j, t))
                    pa = idx._p_active.get((d_i, j, t))
                    if pg is not None and pa is not None:
                        has_p = True
                        p_terms.append((pg, dem.expected_requests))
                        eff_dt_t = dt_t
                        if dem.arrival_offset_s is not None:
                            eff_dt_t = max(0.0, min(dt_t, idx.slot_ends[t] - dem.arrival_offset_s))
                        cb.add([(pg, 1.0), (pa, -max_rate_j * eff_dt_t)], -math.inf, 0.0)
                        if eff_dt_t < dt_t - 1e-9:
                            tier_cap = [(pg, 1.0)]
                            for l_i, rate in enumerate(rates_j):
                                if rate > 0.0:
                                    tier_cap.append((idx.z(j, l_i, t), -rate * eff_dt_t))
                            cb.add(tier_cap, -math.inf, 0.0)
                if not has_p:
                    continue
                for l_i in range(idx.L):
                    rate = rates_j[l_i]
                    if rate <= 0.0:
                        continue
                    p_terms.append((idx.z(j, l_i, t), -rate * idx.sunlit_s[j][t]))
                    for a_var, dur in idx._a_ecl_by_slot.get((j, t, l_i), ()):
                        p_terms.append((a_var, -rate * dur))
                cb.add(p_terms, -math.inf, 0.0)

    @staticmethod
    def _add_predicted_demand_lifecycle(
        idx: _VarLayout, cb: _ConstraintBuilder, d_i: int, dem: R3PredictedDemandSpec
    ) -> None:
        t0 = max(1, dem.arrival_slot)
        place_vars = [idx._p_place[(d_i, j)] for j in range(idx.J) if (d_i, j) in idx._p_place]
        if not place_vars or t0 >= idx.H:
            cb.add([(idx._p_y[d_i], 1.0)], 0.0, 0.0)
            return
        cb.add([(v, 1.0) for v in place_vars], -math.inf, 1.0)
        comp_vars = [idx._p_comp[(d_i, t)] for t in range(t0, idx.H)]
        cb.add([(v, 1.0) for v in comp_vars] + [(pv, -1.0) for pv in place_vars], -math.inf, 0.0)
        valid_comp = [idx._p_comp[(d_i, t)] for t in range(t0, idx.H) if idx.slot_ends[t] <= dem.deadline_s + 1e-9]
        if valid_comp:
            cb.add([(idx._p_y[d_i], 1.0)] + [(cv, -1.0) for cv in valid_comp], -math.inf, 0.0)
        else:
            cb.add([(idx._p_y[d_i], 1.0)], 0.0, 0.0)
        for j in range(idx.J):
            pv = idx._p_place.get((d_i, j))
            if pv is None:
                continue
            for t in range(t0, idx.H):
                pa = idx._p_active[(d_i, j, t)]
                cb.add([(pa, 1.0), (pv, -1.0)], -math.inf, 0.0)
                prev_comps = [(idx._p_comp[(d_i, tau)], 1.0) for tau in range(t0, t)]
                cb.add([(pa, 1.0), (pv, -1.0), *prev_comps], 0.0, math.inf)
        for t in range(t0, idx.H):
            tok_terms = [(idx._p_comp[(d_i, tau)], dem.tokens_per_req) for tau in range(t0, t + 1)]
            for tau in range(t0, t + 1):
                for j in range(idx.J):
                    pg = idx._p_g.get((d_i, j, tau))
                    if pg is not None:
                        tok_terms.append((pg, -1.0))
            cb.add(tok_terms, -math.inf, 0.0)

    def _add_migration_state_constraints(
        self,
        problem: R3Problem,
        idx: _VarLayout,
        node_pos: dict[int, int],
        routes: dict[tuple[int, int, int], tuple[int, ...]],
        cb: _ConstraintBuilder,
    ) -> None:
        max_hops = max((len(eids) for eids in routes.values()), default=0)
        max_slot_dt = max(idx.slot_dts)
        m_prop = max(max_slot_dt, max_hops * problem.hop_latency_s) + max_slot_dt
        for t in range(idx.H):
            if problem.max_inflight is not None:
                inf_terms = [(idx.m(r, j, t), 1.0) for r in range(idx.R) for j in range(idx.J)]
                cb.add(inf_terms, -math.inf, float(problem.max_inflight))

        for r, req in enumerate(problem.requests):
            init_mig_j = node_pos.get(req.init_mig_dst)
            if problem.max_mig_per_req is not None:
                rem_mig = max(0, problem.max_mig_per_req - req.n_mig)
                mig_terms = [(idx.unlock(r, j, t), 1.0) for j in range(idx.J) for t in range(1, idx.H)]
                cb.add(mig_terms, -math.inf, float(rem_mig))

            for t in range(idx.H):
                self._add_req_slot_migration(problem, idx, routes, cb, r, req, init_mig_j, m_prop, t)

    def _add_req_slot_migration(  # noqa: PLR0913
        self,
        problem: R3Problem,
        idx: _VarLayout,
        routes: dict[tuple[int, int, int], tuple[int, ...]],
        cb: _ConstraintBuilder,
        r: int,
        req: R3RequestSpec,
        init_mig_j: int | None,
        m_prop: float,
        t: int,
    ) -> None:
        rem_active = [(idx.comp(r, tau), 1.0) for tau in range(t)]
        cb.add([(idx.u(r, j, t), 1.0) for j in range(idx.J)] + rem_active, 1.0, 1.0)
        cb.add([(idx.m(r, j, t), 1.0) for j in range(idx.J)] + rem_active, -math.inf, 1.0)
        cb.add(
            [(idx.x(r, j, t), 1.0) for j in range(idx.J)] + [(idx.m(r, j, t), 1.0) for j in range(idx.J)] + rem_active,
            -math.inf,
            1.0,
        )
        act_terms = [(idx.x(r, j, t), 1.0) for j in range(idx.J)] + [(idx.h(r, j, t), 1.0) for j in range(idx.J)]
        cb.add(act_terms + rem_active, 1.0, 1.0)

        # tx_complete and tx_just_done (bidirectional one-shot per session)
        snap = _effective_init_mig(req)
        cb.add([(idx.tx_complete(r, t), 1.0)] + [(idx.m(r, j, t), -1.0) for j in range(idx.J)], -math.inf, 0.0)
        cb.add([(idx.tx_just_done(r, t), 1.0), (idx.tx_complete(r, t), -1.0)], -math.inf, 0.0)
        if t == 0:
            if snap is None or snap.phase != "propagating_unlock":
                cb.add([(idx.tx_just_done(r, 0), 1.0), (idx.tx_complete(r, 0), -1.0)], 0.0, 0.0)
                if snap is None or snap.received_bytes < req.kv_bytes - 1e-6:
                    cb.add([(idx.tx_just_done(r, 0), 1.0), (idx.tx(r, 0), -1.0)], -math.inf, 0.0)
        else:
            prev_terms = [(idx.tx_just_done(r, t), 1.0), (idx.tx_complete(r, t), -1.0), (idx.tx_complete(r, t - 1), 1.0)]
            prev_terms.extend((idx.unlock(r, j, t), -1.0) for j in range(idx.J))
            cb.add(prev_terms, 0.0, math.inf)
            # Upper bound: cannot fire tx_just_done again while tx_complete[t-1]==1 in same session
            ub_terms = [(idx.tx_just_done(r, t), 1.0), (idx.tx_complete(r, t - 1), 1.0)]
            ub_terms.extend((idx.unlock(r, j, t), -1.0) for j in range(idx.J))
            cb.add(ub_terms, -math.inf, 1.0)
            cb.add([(idx.tx_just_done(r, t), 1.0), (idx.tx(r, t), -1.0)], -math.inf, 0.0)

        # Propagation remaining time
        if problem.hop_latency_s > 0.0:
            dt_t = idx.slot_dts[t]
            prop_terms = [
                (idx.prop_rem(r, t), 1.0),
                (idx.tx_frac(r, t), -dt_t),
                (idx.tx_just_done(r, t), -m_prop),
                (idx.flow_hops(r, t), -problem.hop_latency_s),
            ]
            cb.add(prop_terms, -dt_t - m_prop, math.inf)
            if t >= 1:
                cb.add(
                    [
                        (idx.prop_rem(r, t), 1.0),
                        (idx.prop_rem(r, t - 1), -1.0),
                        (idx.tx_complete(r, t), -m_prop),
                        (idx.tx_just_done(r, t), m_prop),
                    ],
                    -dt_t - m_prop,
                    math.inf,
                )

        if t < idx.H - 1:
            unlock_next = [(idx.unlock(r, j, t + 1), 1.0) for j in range(idx.J)]
            cb.add(unlock_next + [(idx.tx_complete(r, t), -1.0)], -math.inf, 0.0)
            cb.add([(idx.prop_rem(r, t), 1.0)] + [(c, m_prop) for c, _ in unlock_next], -math.inf, m_prop)

        for j in range(idx.J):
            cb.add([(idx.x(r, j, t), 1.0), (idx.h(r, j, t), 1.0), (idx.u(r, j, t), -1.0)], -math.inf, 0.0)
            cb.add([(idx.u(r, j, t), 1.0), (idx.m(r, j, t), 1.0)], -math.inf, 1.0)
            if t < idx.H - 1:
                cb.add([(idx.m(r, j, t), 1.0), (idx.u(r, j, t + 1), -1.0), (idx.m(r, j, t + 1), -1.0)], -math.inf, 0.0)
                cb.add([(idx.u(r, j, t + 1), 1.0), (idx.u(r, j, t), -1.0), (idx.m(r, j, t), -1.0)], -math.inf, 0.0)
            if t >= 1:
                cb.add([(idx.unlock(r, j, t), 1.0), (idx.u(r, j, t), -1.0), (idx.u(r, j, t - 1), 1.0)], 0.0, math.inf)
                cb.add([(idx.unlock(r, j, t), 1.0), (idx.u(r, j, t), -1.0)], -math.inf, 0.0)
                cb.add([(idx.unlock(r, j, t), 1.0), (idx.u(r, j, t - 1), 1.0)], -math.inf, 1.0)
            init_recv = (
                (snap.received_bytes + snap.chunk_sent_bytes)
                if (snap is not None and j == init_mig_j)
                else (req.init_mig_received_bytes if j == init_mig_j else 0.0)
            )
            byte_terms = [(idx.tx_complete(r, t), req.kv_bytes), (idx.m(r, j, t), req.kv_bytes)]
            byte_terms.extend((idx.unlock(r, j, tau), req.kv_bytes) for tau in range(1, t + 1))
            byte_terms.extend((idx.d(r, j, tau), -1.0) for tau in range(t + 1))
            cb.add(byte_terms, -math.inf, init_recv + req.kv_bytes)

    def _add_network_constraints(
        self,
        problem: R3Problem,
        idx: _VarLayout,
        routes: dict[tuple[int, int, int], tuple[int, ...]],
        cb: _ConstraintBuilder,
    ) -> None:
        edge_routes_by_t: list[list[list[tuple[int, int]]]] = [[[] for _ in range(idx.E)] for _ in range(idx.H)]
        for (s_j, d_j, t), eids in routes.items():
            for e_i in eids:
                edge_routes_by_t[t][e_i].append((s_j, d_j))

        frozen_by_req: list[tuple[int, ...]] = []
        for req in problem.requests:
            snap = _effective_init_mig(req)
            if snap is not None and snap.phase == "sending":
                j0 = problem.nodes.index(next(n for n in problem.nodes if n.node_id == req.init_node))
                d0 = problem.nodes.index(next(n for n in problem.nodes if n.node_id == snap.dst))
                frozen_by_req.append(req.init_mig_solver_edges or routes.get((j0, d0, 0), ()))
            else:
                frozen_by_req.append(())
        m_hops = float(max([len(e) for e in routes.values()] + [len(fe) for fe in frozen_by_req] + [0]))
        for r, req in enumerate(problem.requests):
            snap = _effective_init_mig(req)
            self._add_request_net_constraints(
                problem, idx, routes, edge_routes_by_t, cb, r, req, snap, frozen_by_req[r], m_hops
            )
        for e_i, edge in enumerate(problem.edges):
            self._add_edge_fair_constraints(problem, idx, cb, e_i, edge)

    def _add_request_net_constraints(  # noqa: PLR0913
        self,
        problem: R3Problem,
        idx: _VarLayout,
        routes: dict[tuple[int, int, int], tuple[int, ...]],
        edge_routes_by_t: list[list[list[tuple[int, int]]]],
        cb: _ConstraintBuilder,
        r: int,
        req: R3RequestSpec,
        snap: MigrationSessionSnapshot | None,
        frozen_edges: tuple[int, ...],
        m_hops: float,
    ) -> None:
        is_sending = snap is not None and snap.phase == "sending"
        for t in range(idx.H):
            cb.add([(idx.tx(r, t), 1.0)] + [(idx.m(r, j, t), -1.0) for j in range(idx.J)], -math.inf, 0.0)
            cb.add([(idx.b(r, t), 1.0), (idx.tx(r, t), -req.kv_bytes)], -math.inf, 0.0)
            cb.add([(idx.d(r, j, t), 1.0) for j in range(idx.J)] + [(idx.b(r, t), -1.0)], 0.0, 0.0)
            for j in range(idx.J):
                cb.add([(idx.d(r, j, t), 1.0), (idx.m(r, j, t), -req.kv_bytes)], -math.inf, 0.0)

            cb.add([(idx.arc_frozen(r, t), 1.0), (idx.init_chunk_active(r, t), -1.0)], 0.0, 0.0)
            cb.add(
                [(idx.arc_frozen(r, t), 1.0), (idx.start_flow(r, t), 1.0), (idx.cont_flow(r, t), 1.0), (idx.tx(r, t), -1.0)],
                0.0,
                0.0,
            )
            arc_all = [(idx.arc(r, s, d, t), 1.0) for s in range(idx.J) for d in range(idx.J) if s != d]
            cb.add(arc_all + [(idx.start_flow(r, t), -1.0)], 0.0, 0.0)
            if t >= 1:
                cb.add([(idx.cont_flow(r, t), 1.0), (idx.start_flow(r, t - 1), -1.0), (idx.cont_flow(r, t - 1), -1.0)], -math.inf, 0.0)
                cb.add([(idx.cont_flow(r, t), 1.0), (idx.tx_complete(r, t - 1), 1.0)], -math.inf, 1.0)
                cb.add(
                    [(idx.cont_flow(r, t), 1.0), (idx.start_flow(r, t - 1), -1.0), (idx.cont_flow(r, t - 1), -1.0), (idx.tx_complete(r, t - 1), 1.0)],
                    0.0,
                    math.inf,
                )
            if is_sending and snap is not None:
                self._add_init_chunk_slot_constraints(problem, idx, cb, r, req, snap, frozen_edges, t)
            self._add_flow_path_locking_constraints(idx, routes, edge_routes_by_t, cb, r, frozen_edges, m_hops, t)
            for s in range(idx.J):
                cb.add(
                    [(idx.arc(r, s, d, t), 1.0) for d in range(idx.J) if d != s] + [(idx.u(r, s, t), -1.0)],
                    -math.inf,
                    0.0,
                )
            for d in range(idx.J):
                cb.add(
                    [(idx.arc(r, s, d, t), 1.0) for s in range(idx.J) if s != d] + [(idx.m(r, d, t), -1.0)],
                    -math.inf,
                    0.0,
                )

    @staticmethod
    def _add_flow_path_locking_constraints(  # noqa: PLR0913
        idx: _VarLayout,
        routes: dict[tuple[int, int, int], tuple[int, ...]],
        edge_routes_by_t: list[list[list[tuple[int, int]]]],
        cb: _ConstraintBuilder,
        r: int,
        frozen_edges: tuple[int, ...],
        m_hops: float,
        t: int,
    ) -> None:
        cb.add([(idx.flow_hops(r, t), 1.0), (idx.tx(r, t), -m_hops)], -math.inf, 0.0)
        cb.add([(idx.flow_hops(r, t), 1.0), (idx.arc_frozen(r, t), -float(len(frozen_edges)))], 0.0, math.inf)
        hop_arc = [
            (idx.arc(r, s_j, d_j, t), -float(len(eids)))
            for (s_j, d_j, t_r), eids in routes.items()
            if t_r == t
        ]
        cb.add([(idx.flow_hops(r, t), 1.0)] + hop_arc, 0.0, math.inf)
        cb.add([(idx.flow_hops(r, t), 1.0), (idx.start_flow(r, t), m_hops)] + hop_arc, -math.inf, m_hops)
        if t >= 1:
            cb.add([(idx.flow_hops(r, t), 1.0), (idx.flow_hops(r, t - 1), -1.0), (idx.cont_flow(r, t), -m_hops)], -m_hops, math.inf)
            cb.add([(idx.flow_hops(r, t), 1.0), (idx.flow_hops(r, t - 1), -1.0), (idx.cont_flow(r, t), m_hops)], -math.inf, m_hops)

        frozen_set = set(frozen_edges)
        for e_i in range(idx.E):
            cb.add([(idx.e_use(r, e_i, t), 1.0), (idx.tx(r, t), -1.0)], -math.inf, 0.0)
            if e_i in frozen_set:
                cb.add([(idx.e_use(r, e_i, t), 1.0), (idx.arc_frozen(r, t), -1.0)], 0.0, math.inf)
            else:
                cb.add([(idx.e_use(r, e_i, t), 1.0), (idx.arc_frozen(r, t), 1.0)], -math.inf, 1.0)
            arc_terms = [(idx.arc(r, s_j, d_j, t), -1.0) for s_j, d_j in edge_routes_by_t[t][e_i]]
            cb.add([(idx.e_use(r, e_i, t), 1.0)] + arc_terms, 0.0, math.inf)
            cb.add([(idx.e_use(r, e_i, t), 1.0), (idx.start_flow(r, t), 1.0)] + arc_terms, -math.inf, 1.0)
            if t >= 1:
                cb.add([(idx.e_use(r, e_i, t), 1.0), (idx.e_use(r, e_i, t - 1), -1.0), (idx.cont_flow(r, t), -1.0)], -1.0, math.inf)
                cb.add([(idx.e_use(r, e_i, t), 1.0), (idx.e_use(r, e_i, t - 1), -1.0), (idx.cont_flow(r, t), 1.0)], -math.inf, 1.0)

    @staticmethod
    def _add_init_chunk_slot_constraints(  # noqa: PLR0913
        problem: R3Problem,
        idx: _VarLayout,
        cb: _ConstraintBuilder,
        r: int,
        req: R3RequestSpec,
        snap: MigrationSessionSnapshot,
        frozen_edges: tuple[int, ...],
        t: int,
    ) -> None:
        c_rem = max(0.0, snap.chunk_rem_bytes)
        l_prop = problem.hop_latency_s * len(frozen_edges)
        max_slot_dt = max(idx.slot_dts)
        m_prop = max(max_slot_dt, l_prop) + max_slot_dt
        sum_b = [(idx.b(r, tau), 1.0) for tau in range(t + 1)]
        cb.add(sum_b + [(idx.init_chunk_done(r, t), -c_rem)], 0.0, math.inf)
        if t == 0:
            cb.add(sum_b, -math.inf, c_rem)
            cb.add([(idx.init_chunk_just_done(r, 0), 1.0), (idx.init_chunk_done(r, 0), -1.0)], 0.0, 0.0)
        else:
            cb.add(sum_b + [(idx.init_chunk_done(r, t - 1), -req.kv_bytes)], -math.inf, c_rem)
            cb.add([(idx.init_chunk_active(r, t), 1.0), (idx.init_chunk_done(r, t - 1), 1.0)], -math.inf, 1.0)
            cb.add([(idx.init_chunk_active(r, t), 1.0), (idx.tx(r, t), -1.0)], -math.inf, 0.0)
            unlock_cum = [(idx.unlock(r, j, tau), 1.0) for j in range(idx.J) for tau in range(1, t + 1)]
            cb.add([(idx.init_chunk_active(r, t), 1.0), (idx.init_chunk_done(r, t - 1), 1.0)] + unlock_cum, 1.0, math.inf)
            cb.add(
                [(idx.init_chunk_just_done(r, t), 1.0), (idx.init_chunk_done(r, t), -1.0), (idx.init_chunk_done(r, t - 1), 1.0)],
                0.0,
                math.inf,
            )
            cb.add([(idx.init_chunk_just_done(r, t), 1.0), (idx.init_chunk_done(r, t), -1.0)], -math.inf, 0.0)
            cb.add([(idx.init_chunk_just_done(r, t), 1.0), (idx.init_chunk_done(r, t - 1), 1.0)], -math.inf, 1.0)
            # Continuation flow cannot start until initial chunk data AND its propagation complete
            cb.add([(idx.tx(r, t), 1.0), (idx.init_chunk_active(r, t), -1.0), (idx.init_chunk_done(r, t - 1), -1.0)], -math.inf, 0.0)
            if l_prop > 0.0:
                cb.add(
                    [(idx.init_prop_rem(r, t - 1), 1.0), (idx.tx(r, t), m_prop), (idx.init_chunk_active(r, t), -m_prop)],
                    -math.inf,
                    m_prop,
                )
        if l_prop > 0.0:
            dt_t = idx.slot_dts[t]
            cb.add(
                [(idx.init_prop_rem(r, t), 1.0), (idx.tx_frac(r, t), -dt_t), (idx.init_chunk_just_done(r, t), -m_prop)],
                l_prop - dt_t - m_prop,
                math.inf,
            )
            if t >= 1:
                cb.add(
                    [
                        (idx.init_prop_rem(r, t), 1.0),
                        (idx.init_prop_rem(r, t - 1), -1.0),
                        (idx.init_chunk_done(r, t), -m_prop),
                        (idx.init_chunk_just_done(r, t), m_prop),
                    ],
                    -dt_t - m_prop,
                    math.inf,
                )

    def _add_edge_fair_constraints(
        self,
        problem: R3Problem,
        idx: _VarLayout,
        cb: _ConstraintBuilder,
        e_i: int,
        edge: R3EdgeSpec,
    ) -> None:
        for t in range(idx.H):
            dt_t = idx.slot_dts[t]
            bg = 0.0 if edge.bg_bps is None else edge.bg_bps[t]
            tot_cap_bytes = (dt_t / 8.0) * edge.tot_bps[t]
            cap_bytes = (dt_t / 8.0) * max(0.0, edge.tot_bps[t] - bg)
            bg_frac = min(1.0, bg / edge.tot_bps[t]) if edge.tot_bps[t] > 0.0 else 0.0
            cb.add([(idx.b_edge(r, e_i, t), 1.0) for r in range(idx.R)], -math.inf, cap_bytes)

            if idx.K_max >= 2:
                cb.add([(idx.w(e_i, t, k), 1.0) for k in range(1, idx.K_max + 1)], -math.inf, 1.0)
                count_terms = [(idx.w(e_i, t, k), float(k)) for k in range(1, idx.K_max + 1)]
                count_terms.extend((idx.e_use(r, e_i, t), -1.0) for r in range(idx.R))
                cb.add(count_terms, 0.0, 0.0)

            for r, req in enumerate(problem.requests):
                use_terms = [(idx.e_use(r, e_i, t), 1.0)]
                cb.add([(idx.b_edge(r, e_i, t), 1.0), (idx.e_use(r, e_i, t), -req.kv_bytes)], -math.inf, 0.0)
                cb.add(
                    [(idx.b_edge(r, e_i, t), 1.0), (idx.b(r, t), -1.0), (idx.e_use(r, e_i, t), -req.kv_bytes)],
                    -req.kv_bytes,
                    math.inf,
                )
                if idx.K_max >= 2:
                    fair_terms = [(idx.b_edge(r, e_i, t), 1.0), (idx.e_use(r, e_i, t), req.kv_bytes)]
                    fair_terms.extend((idx.w(e_i, t, k), -(cap_bytes / k)) for k in range(1, idx.K_max + 1))
                    cb.add(fair_terms, -math.inf, req.kv_bytes)
                if cap_bytes > 0.0:
                    self._add_tx_frac_bound(idx, cb, e_i, t, r, tot_cap_bytes, bg_frac, use_terms)

    @staticmethod
    def _add_tx_frac_bound(  # noqa: PLR0913
        idx: _VarLayout,
        cb: _ConstraintBuilder,
        e_i: int,
        t: int,
        r: int,
        tot_cap_bytes: float,
        bg_frac: float,
        use_terms: list[tuple[int, float]],
    ) -> None:
        if idx.K_max >= 2:
            for k in range(1, idx.K_max + 1):
                terms = [(idx.tx_frac(r, t), 1.0), (idx.b_edge(r, e_i, t), -(k / tot_cap_bytes))]
                terms.append((idx.w(e_i, t, k), -float(k)))
                terms.extend((c, -float(k)) for c, _ in use_terms)
                cb.add(terms, bg_frac - 2.0 * float(k), math.inf)
        else:
            terms = [(idx.tx_frac(r, t), 1.0), (idx.b_edge(r, e_i, t), -(1.0 / tot_cap_bytes))]
            terms.extend((c, -1.0) for c, _ in use_terms)
            cb.add(terms, bg_frac - 1.0, math.inf)

    def _add_cvar_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        miss_lat = idx.total_horizon_s + idx.slot_dts[-1]
        for r, req in enumerate(problem.requests):
            terms: list[tuple[int, float]] = [(idx.xi(r), 1.0), (idx.eta(), 1.0)]
            for t in range(idx.H):
                coeff = idx.slot_ends[t] - miss_lat
                terms.append((idx.comp(r, t), -coeff))
            cb.add(terms, req.elapsed_s + miss_lat, math.inf)

    def _stage1_cost(self, problem: R3Problem, idx: _VarLayout) -> np.ndarray:
        c = np.zeros(idx.total, dtype=float)
        for r, req in enumerate(problem.requests):
            c[idx.y(r)] = -req.weight
        for d_i, dem in enumerate(problem.predicted_demands):
            c[idx._p_y[d_i]] = -dem.weight * dem.expected_requests
        if problem.cvar_weight > 0.0:
            denom = max(1e-9, (1.0 - problem.cvar_alpha) * max(1, idx.R))
            c[idx.eta()] = problem.cvar_weight
            for r in range(idx.R):
                c[idx.xi(r)] = problem.cvar_weight / denom
        return c

    def _failure_result(self, problem: R3Problem, status: str, reason: str, wall_time_s: float) -> R3Result:
        return R3Result(
            action=PlanAction(),
            status=status,
            stage1_obj=-math.inf,
            goodput_weight=0.0,
            cvar_s=math.inf,
            migration_bytes=0.0,
            node_tiers={n.node_id: () for n in problem.nodes},
            node_temps_C={n.node_id: () for n in problem.nodes},
            goodput_req_ids=(),
            wall_time_s=wall_time_s,
            mip_gap=None,
            fallback_reason=reason,
            stage1_status=status,
            stage1_wall_time_s=wall_time_s,
            stage1_mip_gap=None,
            stage2_status="not_run",
        )

    def _extract_result(  # noqa: PLR0913
        self,
        problem: R3Problem,
        idx: _VarLayout,
        node_pos: dict[int, int],
        x_sol: np.ndarray,
        c1: np.ndarray,
        status: str,
        wall_time_s: float,
        mip_gap: float | None,
        fallback_reason: str | None,
        stage1_diag: tuple[str, float, float | None],
        stage2_diag: tuple[str, float, float | None],
    ) -> R3Result:
        tier_decisions: dict[int, int | None] = {}
        node_tiers: dict[int, tuple[int, ...]] = {}
        node_temps_C: dict[int, tuple[float, ...]] = {}

        for j, node in enumerate(problem.nodes):
            t_list: list[int] = []
            temp_list: list[float] = []
            for t in range(idx.H):
                best_l = max(range(idx.L), key=lambda l_i: float(x_sol[idx.z(j, l_i, t)]))
                t_list.append(problem.tiers[best_l].index)
                temp_list.append(float(x_sol[idx.theta(j, t)]))
            tier_decisions[node.node_id] = t_list[0]
            node_tiers[node.node_id] = tuple(t_list)
            node_temps_C[node.node_id] = tuple(temp_list)

        pause_ids: list[int] = []
        resume_ids: list[int] = []
        migrations: list[MigrationOrder] = []
        goodput_ids: list[int] = []

        for r, req in enumerate(problem.requests):
            if x_sol[idx.y(r)] > 0.5:
                goodput_ids.append(req.req_id)
            j0 = node_pos[req.init_node]
            is_mig_0 = any(x_sol[idx.m(r, j, 0)] > 0.5 for j in range(idx.J))
            if not is_mig_0:
                if x_sol[idx.h(r, j0, 0)] > 0.5 and not req.currently_paused:
                    pause_ids.append(req.req_id)
                elif x_sol[idx.x(r, j0, 0)] > 0.5 and req.currently_paused:
                    resume_ids.append(req.req_id)

            snap = _effective_init_mig(req)
            in_flight_0 = snap is not None and snap.phase in ("sending", "propagating_chunk", "propagating_unlock")
            if x_sol[idx.tx(r, 0)] > 0.5 and not in_flight_0:
                dst_j = max(range(idx.J), key=lambda j: float(x_sol[idx.m(r, j, 0)]))
                if x_sol[idx.m(r, dst_j, 0)] > 0.5 and dst_j != j0:
                    migrations.append(
                        MigrationOrder(req_id=req.req_id, dst=problem.nodes[dst_j].node_id, nbytes=None)
                    )

        mig_bytes = sum(
            float(x_sol[idx.b_edge(r, e, t)]) for r in range(idx.R) for e in range(idx.E) for t in range(idx.H)
        )
        goodput_w = sum(req.weight for r, req in enumerate(problem.requests) if x_sol[idx.y(r)] > 0.5)
        denom = max(1e-9, (1.0 - problem.cvar_alpha) * idx.R)
        cvar_s = float(x_sol[idx.eta()]) + sum(float(x_sol[idx.xi(r)]) for r in range(idx.R)) / denom

        return R3Result(
            action=PlanAction(
                tier_decisions=tier_decisions,
                pause_ids=tuple(pause_ids),
                resume_ids=tuple(resume_ids),
                migrations=tuple(migrations),
            ),
            status=status,
            stage1_obj=float(-np.dot(c1, x_sol)),
            goodput_weight=goodput_w,
            cvar_s=cvar_s,
            migration_bytes=mig_bytes,
            node_tiers=node_tiers,
            node_temps_C=node_temps_C,
            goodput_req_ids=tuple(goodput_ids),
            wall_time_s=wall_time_s,
            mip_gap=mip_gap,
            fallback_reason=fallback_reason,
            stage1_status=stage1_diag[0],
            stage1_wall_time_s=stage1_diag[1],
            stage1_mip_gap=stage1_diag[2],
            stage2_status=stage2_diag[0],
            stage2_wall_time_s=stage2_diag[1],
            stage2_mip_gap=stage2_diag[2],
        )
