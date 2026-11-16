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
class R3NodeSpec:
    node_id: int
    init_temp_C: float
    kv_cap_bytes: float
    is_eclipse: tuple[bool, ...]
    init_dod_remaining_j: float | None = None
    tier_rates_tok_s: tuple[float, ...] | None = None


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
    n_mig: int = 0


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
    max_inflight: int | None = None
    max_mig_per_req: int | None = None
    hop_latency_s: float = 0.0
    routes: dict[tuple[int, int, int], tuple[int, ...]] | None = None
    batch_tier_rates: tuple[tuple[float, ...], ...] | None = None
    cvar_alpha: float = 0.95
    cvar_weight: float = 1e-3
    stage2_eps: float = 1e-5
    time_limit_s: float = 5.0


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


class _VarLayout:
    """Contiguous variable index layout for the R3 MILP."""

    def __init__(self, problem: R3Problem) -> None:
        self.H = problem.horizon_slots
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
            ("off_tx_complete", self.R * self.H),
            ("off_comp", self.R * self.H),
            ("off_y", self.R),
            ("off_w", self.E * self.H * self.K_max),
            ("off_b_ge", self.J * self.H * self.B_max),
            ("off_g", self.R * self.J * self.H),
            ("off_unlock", self.R * self.J * self.H),
            ("off_tx_just_done", self.R * self.H),
            ("off_b", self.R * self.H),
            ("off_tx_frac", self.R * self.H),
            ("off_prop_rem", self.R * self.H),
            ("off_d", self.R * self.J * self.H),
            ("off_arc", self.R * self.J * self.J * self.H),
            ("off_b_edge", self.R * self.E * self.H),
            ("off_eta", 1),
            ("off_xi", self.R),
        )
        off = 0
        for name, count in sizes:
            setattr(self, name, off)
            off += count
        self.total = off

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

    def tx_complete(self, r: int, t: int) -> int:
        return self.off_tx_complete + r * self.H + t

    def tx_just_done(self, r: int, t: int) -> int:
        return self.off_tx_just_done + r * self.H + t

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

    def d(self, r: int, j: int, t: int) -> int:
        return self.off_d + (r * self.J + j) * self.H + t

    def arc(self, r: int, s: int, d: int, t: int) -> int:
        return self.off_arc + (((r * self.J + s) * self.J + d) * self.H) + t

    def b_edge(self, r: int, e: int, t: int) -> int:
        return self.off_b_edge + (r * self.E + e) * self.H + t

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

        if not problem.requests:
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
        for r in range(idx.R):
            for e in range(idx.E):
                for t in range(idx.H):
                    c2[idx.b_edge(r, e, t)] = 1.0
            for j in range(idx.J):
                for t in range(idx.H):
                    # Break exact ties against holding destination KV / migrating earlier than needed
                    c2[idx.m(r, j, t)] = 1e-4 * (idx.H - t)

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

    def _validate(self, problem: R3Problem) -> None:
        if problem.horizon_slots <= 0 or problem.dt_s <= 0.0:
            raise ValueError("horizon_slots and dt_s must be positive")
        if not problem.nodes or not problem.tiers:
            raise ValueError("nodes and tiers must be non-empty")
        node_ids = {n.node_id for n in problem.nodes}
        for n in problem.nodes:
            if len(n.is_eclipse) != problem.horizon_slots:
                raise ValueError(f"node {n.node_id} is_eclipse length must match horizon_slots")
            if n.tier_rates_tok_s is not None and len(n.tier_rates_tok_s) != len(problem.tiers):
                raise ValueError(f"node {n.node_id} tier_rates_tok_s length must match tiers")
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

        tau = problem.r_th * problem.c_th
        alpha = math.exp(-problem.dt_s / tau) if tau > 0.0 else 0.0
        for j, node in enumerate(problem.nodes):
            t_cool_0 = alpha * node.init_temp_C + (1.0 - alpha) * problem.t_amb_C
            for t in range(idx.H):
                col = idx.theta(j, t)
                lb[col] = -math.inf
                ub[col] = max(problem.t_max_C, t_cool_0) if t == 0 else problem.t_max_C

        for r, req in enumerate(problem.requests):
            self._apply_request_bounds(problem, idx, node_pos, routes, lb, ub, r, req)

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
        elif problem.max_mig_per_req is not None and req.n_mig >= problem.max_mig_per_req:
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
            ub[idx.b(r, 0)] = min(req.kv_bytes, max(0.0, snap.chunk_rem_bytes))
        elif snap.phase == "propagating_chunk":
            block_slots = max(1, math.ceil(snap.prop_rem_s / problem.dt_s - 1e-9))
            for t in range(min(idx.H, block_slots)):
                ub[idx.tx(r, t)] = 0.0
        elif snap.phase == "propagating_unlock":
            ub[idx.tx(r, 0)] = 0.0
            lb[idx.tx_complete(r, 0)] = 1.0
            ub[idx.tx_just_done(r, 0)] = 0.0
            lb[idx.prop_rem(r, 0)] = max(0.0, snap.prop_rem_s - problem.dt_s)

    def _add_power_thermal_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        tau = problem.r_th * problem.c_th
        alpha = math.exp(-problem.dt_s / tau) if tau > 0.0 else 0.0
        one_minus_alpha = 1.0 - alpha
        zero_tiers = [l_i for l_i, t in enumerate(problem.tiers) if t.power_w <= 0.0 and t.rate_tok_s <= 0.0]

        for j, node in enumerate(problem.nodes):
            for t in range(idx.H):
                cb.add([(idx.z(j, l_i, t), 1.0) for l_i in range(idx.L)], 1.0, 1.0)
                terms: list[tuple[int, float]] = [(idx.theta(j, t), 1.0)]
                if t > 0:
                    terms.append((idx.theta(j, t - 1), -alpha))
                for l_i, tier in enumerate(problem.tiers):
                    terms.append((idx.z(j, l_i, t), -one_minus_alpha * problem.r_th * tier.power_w))
                rhs = one_minus_alpha * problem.t_amb_C + (alpha * node.init_temp_C if t == 0 else 0.0)
                cb.add(terms, rhs, rhs)

                for z0 in zero_tiers:
                    for r in range(idx.R):
                        cb.add([(idx.x(r, j, t), 1.0), (idx.z(j, z0, t), 1.0)], -math.inf, 1.0)

                if zero_tiers and idx.R > 0:
                    for l_i, tier in enumerate(problem.tiers):
                        if tier.power_w > 0.0:
                            cb.add(
                                [(idx.z(j, l_i, t), 1.0)] + [(idx.x(r, j, t), -1.0) for r in range(idx.R)],
                                -math.inf,
                                0.0,
                            )

            if problem.eclipse_dod_budget_j is not None:
                for w_start, w_end in _eclipse_windows(node.is_eclipse):
                    budget = (
                        node.init_dod_remaining_j
                        if (w_start == 0 and node.init_dod_remaining_j is not None)
                        else problem.eclipse_dod_budget_j
                    )
                    terms = [
                        (idx.z(j, l_i, t), problem.tiers[l_i].power_w * problem.dt_s)
                        for t in range(w_start, w_end)
                        for l_i in range(idx.L)
                    ]
                    cb.add(terms, -math.inf, budget)

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
                kv_terms = []
                for r, req in enumerate(problem.requests):
                    res_bytes = req.kv_bytes if req.reserved_kv_bytes is None else req.reserved_kv_bytes
                    kv_terms.append((idx.u(r, j, t), res_bytes))
                    kv_terms.append((idx.m(r, j, t), res_bytes))
                cb.add(kv_terms, -math.inf, problem.nodes[j].kv_cap_bytes)

                for r, req in enumerate(problem.requests):
                    cap_terms = [(idx.g(r, j, t), 1.0)]
                    cap_terms.extend((idx.z(j, l_i, t), -rates_j[l_i] * problem.dt_s) for l_i in range(idx.L))
                    cb.add(cap_terms, -math.inf, 0.0)
                    cb.add(
                        [(idx.g(r, j, t), 1.0), (idx.x(r, j, t), -min(req.remaining_tokens, max_rate_j * problem.dt_s))],
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
                        m_big = max(max_rate_j, max(rates_k, default=0.0)) * problem.dt_s
                        for r in range(idx.R):
                            b_terms = [(idx.g(r, j, t), 1.0), (idx.b_ge(j, t, b_idx), m_big)]
                            b_terms.extend((idx.z(j, l_i, t), -rates_k[l_i] * problem.dt_s) for l_i in range(idx.L))
                            cb.add(b_terms, -math.inf, m_big)

        for r, req in enumerate(problem.requests):
            cb.add([(idx.comp(r, t), 1.0) for t in range(idx.H)], -math.inf, 1.0)
            valid_slots = [t for t in range(idx.H) if (t + 1) * problem.dt_s <= req.deadline_s + 1e-9]
            if valid_slots:
                cb.add([(idx.y(r), 1.0)] + [(idx.comp(r, t), -1.0) for t in valid_slots], -math.inf, 0.0)
            else:
                cb.add([(idx.y(r), 1.0)], 0.0, 0.0)

            for t in range(idx.H):
                tok_terms = [(idx.comp(r, tau), req.remaining_tokens) for tau in range(t + 1)]
                tok_terms.extend((idx.g(r, j, tau), -1.0) for tau in range(t + 1) for j in range(idx.J))
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
        m_prop = max(problem.dt_s, max_hops * problem.hop_latency_s) + problem.dt_s
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
            prop_terms = [(idx.prop_rem(r, t), 1.0), (idx.tx_frac(r, t), -problem.dt_s), (idx.tx_just_done(r, t), -m_prop)]
            for (s_j, d_j, t_r), eids in routes.items():
                if t_r == t:
                    prop_terms.append((idx.arc(r, s_j, d_j, t), -problem.hop_latency_s * len(eids)))
            cb.add(prop_terms, -problem.dt_s - m_prop, math.inf)
            if t >= 1:
                cb.add(
                    [
                        (idx.prop_rem(r, t), 1.0),
                        (idx.prop_rem(r, t - 1), -1.0),
                        (idx.tx_complete(r, t), -m_prop),
                        (idx.tx_just_done(r, t), m_prop),
                    ],
                    -problem.dt_s - m_prop,
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
            init_recv = req.init_mig_received_bytes if j == init_mig_j else 0.0
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

        for r, req in enumerate(problem.requests):
            self._add_request_net_constraints(idx, cb, r, req)
        for e_i, edge in enumerate(problem.edges):
            self._add_edge_fair_constraints(problem, idx, edge_routes_by_t, cb, e_i, edge)

    def _add_request_net_constraints(self, idx: _VarLayout, cb: _ConstraintBuilder, r: int, req: R3RequestSpec) -> None:
        for t in range(idx.H):
            cb.add([(idx.tx(r, t), 1.0)] + [(idx.m(r, j, t), -1.0) for j in range(idx.J)], -math.inf, 0.0)
            cb.add([(idx.b(r, t), 1.0), (idx.tx(r, t), -req.kv_bytes)], -math.inf, 0.0)
            cb.add([(idx.d(r, j, t), 1.0) for j in range(idx.J)] + [(idx.b(r, t), -1.0)], 0.0, 0.0)
            for j in range(idx.J):
                cb.add([(idx.d(r, j, t), 1.0), (idx.m(r, j, t), -req.kv_bytes)], -math.inf, 0.0)

            arc_all = [(idx.arc(r, s, d, t), 1.0) for s in range(idx.J) for d in range(idx.J) if s != d]
            cb.add(arc_all + [(idx.tx(r, t), -1.0)], 0.0, 0.0)
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

    def _add_edge_fair_constraints(  # noqa: PLR0913
        self,
        problem: R3Problem,
        idx: _VarLayout,
        edge_routes_by_t: list[list[list[tuple[int, int]]]],
        cb: _ConstraintBuilder,
        e_i: int,
        edge: R3EdgeSpec,
    ) -> None:
        for t in range(idx.H):
            bg = 0.0 if edge.bg_bps is None else edge.bg_bps[t]
            cap_bytes = (problem.dt_s / 8.0) * max(0.0, edge.tot_bps[t] - bg)
            cb.add([(idx.b_edge(r, e_i, t), 1.0) for r in range(idx.R)], -math.inf, cap_bytes)

            pairs = edge_routes_by_t[t][e_i]
            if idx.K_max >= 2:
                cb.add([(idx.w(e_i, t, k), 1.0) for k in range(1, idx.K_max + 1)], -math.inf, 1.0)
                count_terms = [(idx.w(e_i, t, k), float(k)) for k in range(1, idx.K_max + 1)]
                for r in range(idx.R):
                    for s_j, d_j in pairs:
                        count_terms.append((idx.arc(r, s_j, d_j, t), -1.0))
                cb.add(count_terms, 0.0, 0.0)

            for r, req in enumerate(problem.requests):
                use_terms = [(idx.arc(r, s_j, d_j, t), 1.0) for s_j, d_j in pairs]
                cb.add([(idx.b_edge(r, e_i, t), 1.0)] + [(c, -req.kv_bytes) for c, _ in use_terms], -math.inf, 0.0)
                cb.add(
                    [(idx.b_edge(r, e_i, t), 1.0), (idx.b(r, t), -1.0)] + [(c, -req.kv_bytes) for c, _ in use_terms],
                    -req.kv_bytes,
                    math.inf,
                )
                if idx.K_max >= 2:
                    fair_terms = [(idx.b_edge(r, e_i, t), 1.0)]
                    fair_terms.extend((idx.w(e_i, t, k), -(cap_bytes / k)) for k in range(1, idx.K_max + 1))
                    fair_terms.extend((c, req.kv_bytes) for c, _ in use_terms)
                    cb.add(fair_terms, -math.inf, req.kv_bytes)
                if cap_bytes > 0.0 and pairs:
                    self._add_tx_frac_bound(idx, cb, e_i, t, r, cap_bytes, use_terms)

    @staticmethod
    def _add_tx_frac_bound(  # noqa: PLR0913
        idx: _VarLayout,
        cb: _ConstraintBuilder,
        e_i: int,
        t: int,
        r: int,
        cap_bytes: float,
        use_terms: list[tuple[int, float]],
    ) -> None:
        if idx.K_max >= 2:
            for k in range(1, idx.K_max + 1):
                terms = [(idx.tx_frac(r, t), 1.0), (idx.b_edge(r, e_i, t), -(k / cap_bytes))]
                terms.append((idx.w(e_i, t, k), -float(k)))
                terms.extend((c, -float(k)) for c, _ in use_terms)
                cb.add(terms, -2.0 * float(k), math.inf)
        else:
            terms = [(idx.tx_frac(r, t), 1.0), (idx.b_edge(r, e_i, t), -(1.0 / cap_bytes))]
            terms.extend((c, -1.0) for c, _ in use_terms)
            cb.add(terms, -1.0, math.inf)

    def _add_cvar_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        miss_lat = (problem.horizon_slots + 1) * problem.dt_s
        for r, req in enumerate(problem.requests):
            terms: list[tuple[int, float]] = [(idx.xi(r), 1.0), (idx.eta(), 1.0)]
            for t in range(idx.H):
                coeff = (t + 1) * problem.dt_s - miss_lat
                terms.append((idx.comp(r, t), -coeff))
            cb.add(terms, req.elapsed_s + miss_lat, math.inf)

    def _stage1_cost(self, problem: R3Problem, idx: _VarLayout) -> np.ndarray:
        c = np.zeros(idx.total, dtype=float)
        for r, req in enumerate(problem.requests):
            c[idx.y(r)] = -req.weight
        if problem.cvar_weight > 0.0:
            denom = max(1e-9, (1.0 - problem.cvar_alpha) * idx.R)
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
                    init_recv = (
                        snap.received_bytes
                        if (snap is not None and problem.nodes[dst_j].node_id == snap.dst)
                        else 0.0
                    )
                    rem_bytes = max(0.0, req.kv_bytes - init_recv)
                    sent_0 = float(x_sol[idx.b(r, 0)])
                    nbytes = None if sent_0 >= rem_bytes - 1e-6 else sent_0
                    migrations.append(
                        MigrationOrder(req_id=req.req_id, dst=problem.nodes[dst_j].node_id, nbytes=nbytes)
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
