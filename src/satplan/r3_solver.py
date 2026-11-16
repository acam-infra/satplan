"""Pure-data R3 time-expanded Two-Stage Goodput-CVaR MILP solver (zero SimPy/CoWorld dependency)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from satplan.actions import MigrationOrder, PlanAction


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


class _VarLayout:
    """Contiguous variable index layout for the R3 MILP."""

    def __init__(self, problem: R3Problem) -> None:
        self.H = problem.horizon_slots
        self.J = len(problem.nodes)
        self.L = len(problem.tiers)
        self.E = len(problem.edges)
        self.R = len(problem.requests)

        n_z = self.J * self.L * self.H
        n_theta = self.J * self.H
        n_x = self.R * self.J * self.H
        n_h = self.R * self.J * self.H
        n_u = self.R * self.J * self.H
        n_g = self.R * self.J * self.H
        n_comp = self.R * self.H
        n_y = self.R
        n_s = self.R * self.J * self.H
        n_d = self.R * self.J * self.H
        n_f = self.R * self.E * self.H
        n_xi = self.R

        off = 0
        self.off_z = off
        off += n_z
        self.off_theta = off
        off += n_theta
        self.off_x = off
        off += n_x
        self.off_h = off
        off += n_h
        self.off_u = off
        off += n_u
        self.off_g = off
        off += n_g
        self.off_comp = off
        off += n_comp
        self.off_y = off
        off += n_y
        self.off_s = off
        off += n_s
        self.off_d = off
        off += n_d
        self.off_f = off
        off += n_f
        self.off_eta = off
        off += 1
        self.off_xi = off
        off += n_xi
        self.total = off

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

    def g(self, r: int, j: int, t: int) -> int:
        return self.off_g + (r * self.J + j) * self.H + t

    def comp(self, r: int, t: int) -> int:
        return self.off_comp + r * self.H + t

    def y(self, r: int) -> int:
        return self.off_y + r

    def s(self, r: int, j: int, t: int) -> int:
        return self.off_s + (r * self.J + j) * self.H + t

    def d(self, r: int, j: int, t: int) -> int:
        return self.off_d + (r * self.J + j) * self.H + t

    def f(self, r: int, e: int, t: int) -> int:
        return self.off_f + (r * self.E + e) * self.H + t

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


class R3Solver:
    """Two-stage Goodput-CVaR MILP solver using scipy.optimize.milp (HiGHS)."""

    def solve(self, problem: R3Problem) -> R3Result:
        self._validate(problem)
        if not problem.requests:
            idle_idx = min(problem.tiers, key=lambda t: t.power_w).index
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
            )

        idx = _VarLayout(problem)
        node_pos = {n.node_id: j for j, n in enumerate(problem.nodes)}
        integrality, lb, ub = self._build_bounds(problem, idx, node_pos)
        cb = _ConstraintBuilder()

        self._add_power_thermal_constraints(problem, idx, cb)
        self._add_request_compute_constraints(problem, idx, node_pos, cb)
        self._add_flow_constraints(problem, idx, node_pos, cb)
        self._add_cvar_constraints(problem, idx, cb)

        c1 = self._stage1_cost(problem, idx)
        constraints1 = cb.build(idx.total)
        res1 = milp(
            c=c1,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints1,
            options={"time_limit": problem.time_limit_s},
        )
        if not res1.success or res1.x is None:
            return self._infeasible_result(problem)

        v1_star = float(-np.dot(c1, res1.x))
        # Stage 2: minimize total migration bytes subject to Stage 1 obj >= v1_star - eps
        # Since c1 minimizes (-obj1), c1 @ x <= -v1_star + eps
        cb.add([(col, float(c1[col])) for col in np.nonzero(c1)[0]], -math.inf, -v1_star + problem.stage2_eps)
        c2 = np.zeros(idx.total, dtype=float)
        for r in range(idx.R):
            for e in range(idx.E):
                for t in range(idx.H):
                    c2[idx.f(r, e, t)] = 1.0
        constraints2 = cb.build(idx.total)
        res2 = milp(
            c=c2,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints2,
            options={"time_limit": problem.time_limit_s},
        )
        x_sol = res2.x if (res2.success and res2.x is not None) else res1.x
        status = "optimal" if (res1.status == 0 and (res2.success and res2.status == 0)) else "feasible"
        return self._extract_result(problem, idx, node_pos, x_sol, c1, status)

    def _validate(self, problem: R3Problem) -> None:
        if problem.horizon_slots <= 0 or problem.dt_s <= 0.0:
            raise ValueError("horizon_slots and dt_s must be positive")
        if not problem.nodes or not problem.tiers:
            raise ValueError("nodes and tiers must be non-empty")
        for n in problem.nodes:
            if len(n.is_eclipse) != problem.horizon_slots:
                raise ValueError(f"node {n.node_id} is_eclipse length must match horizon_slots")
        for e in problem.edges:
            if len(e.tot_bps) != problem.horizon_slots:
                raise ValueError(f"edge ({e.src}, {e.dst}) tot_bps length must match horizon_slots")
            if e.bg_bps is not None and len(e.bg_bps) != problem.horizon_slots:
                raise ValueError(f"edge ({e.src}, {e.dst}) bg_bps length must match horizon_slots")

    def _build_bounds(
        self, problem: R3Problem, idx: _VarLayout, node_pos: dict[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        integrality = np.zeros(idx.total, dtype=int)
        lb = np.zeros(idx.total, dtype=float)
        ub = np.full(idx.total, math.inf, dtype=float)

        # Binary variables: z, x, h, u, comp, y
        for start, end in (
            (idx.off_z, idx.off_theta),
            (idx.off_x, idx.off_g),
            (idx.off_comp, idx.off_s),
        ):
            integrality[start:end] = 1
            ub[start:end] = 1.0

        # Temperature theta_{j, t+1} <= T_max (guard t=0 if init_temp > T_max)
        tau = problem.r_th * problem.c_th
        alpha = math.exp(-problem.dt_s / tau) if tau > 0.0 else 0.0
        for j, node in enumerate(problem.nodes):
            t_cool_0 = alpha * node.init_temp_C + (1.0 - alpha) * problem.t_amb_C
            for t in range(idx.H):
                col = idx.theta(j, t)
                lb[col] = -math.inf
                ub[col] = max(problem.t_max_C, t_cool_0) if t == 0 else problem.t_max_C

        # Initial u_{r, j, 0}
        for r, req in enumerate(problem.requests):
            j0 = node_pos[req.init_node]
            for j in range(idx.J):
                col = idx.u(r, j, 0)
                if j == j0:
                    lb[col] = 1.0
                    ub[col] = 1.0
                else:
                    lb[col] = 0.0
                    ub[col] = 0.0

        # CVaR eta is free in R
        lb[idx.eta()] = -math.inf
        return integrality, lb, ub

    def _add_power_thermal_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        tau = problem.r_th * problem.c_th
        alpha = math.exp(-problem.dt_s / tau) if tau > 0.0 else 0.0
        one_minus_alpha = 1.0 - alpha
        zero_tiers = [l_i for l_i, t in enumerate(problem.tiers) if t.power_w <= 0.0 and t.rate_tok_s <= 0.0]

        for j, node in enumerate(problem.nodes):
            for t in range(idx.H):
                # Exactly one discrete tier per node-slot
                cb.add([(idx.z(j, l_i, t), 1.0) for l_i in range(idx.L)], 1.0, 1.0)

                # RC thermal recurrence: theta_{j,t+1} - alpha * theta_{j,t} - (1-alpha)*R_th*P_{j,t} = (1-alpha)*T_amb
                terms: list[tuple[int, float]] = [(idx.theta(j, t), 1.0)]
                if t > 0:
                    terms.append((idx.theta(j, t - 1), -alpha))
                for l_i, tier in enumerate(problem.tiers):
                    terms.append((idx.z(j, l_i, t), -one_minus_alpha * problem.r_th * tier.power_w))
                rhs = one_minus_alpha * problem.t_amb_C + (alpha * node.init_temp_C if t == 0 else 0.0)
                cb.add(terms, rhs, rhs)

                # If node is in 0W idle tier, requests cannot compute (must suspend)
                for z0 in zero_tiers:
                    for r in range(idx.R):
                        cb.add([(idx.x(r, j, t), 1.0), (idx.z(j, z0, t), 1.0)], -math.inf, 1.0)

            # Eclipse DoD budget per contiguous eclipse window
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

    def _add_request_compute_constraints(
        self, problem: R3Problem, idx: _VarLayout, node_pos: dict[int, int], cb: _ConstraintBuilder
    ) -> None:
        max_tok_slot = max(t.rate_tok_s for t in problem.tiers) * problem.dt_s
        for j, node in enumerate(problem.nodes):
            for t in range(idx.H):
                # KV memory reservation: sum_r S_r^res * u_{rjt} <= M_j
                kv_terms = [
                    (
                        idx.u(r, j, t),
                        req.kv_bytes if req.reserved_kv_bytes is None else req.reserved_kv_bytes,
                    )
                    for r, req in enumerate(problem.requests)
                ]
                cb.add(kv_terms, -math.inf, node.kv_cap_bytes)

                # Node-slot token capacity: sum_r g_{rjt} <= sum_l (mu_l * dt) * z_{jlt}
                cap_terms = [(idx.g(r, j, t), 1.0) for r in range(idx.R)]
                cap_terms.extend(
                    (idx.z(j, l_i, t), -problem.tiers[l_i].rate_tok_s * problem.dt_s) for l_i in range(idx.L)
                )
                cb.add(cap_terms, -math.inf, 0.0)

        for r, req in enumerate(problem.requests):
            # At most one completion slot
            cb.add([(idx.comp(r, t), 1.0) for t in range(idx.H)], -math.inf, 1.0)

            # Goodput y_r <= sum_{t: (t+1)*dt <= D_r^sh} comp_{r, t}
            valid_slots = [t for t in range(idx.H) if (t + 1) * problem.dt_s <= req.deadline_s + 1e-9]
            if valid_slots:
                cb.add([(idx.y(r), 1.0)] + [(idx.comp(r, t), -1.0) for t in valid_slots], -math.inf, 0.0)
            else:
                cb.add([(idx.y(r), 1.0)], 0.0, 0.0)

            for t in range(idx.H):
                # Store-and-run: while unfinished, sum_j (x_rjt + h_rjt) = 1 - sum_{tau < t} comp_{r,tau}
                act_terms: list[tuple[int, float]] = []
                for j in range(idx.J):
                    cb.add([(idx.x(r, j, t), 1.0), (idx.h(r, j, t), 1.0), (idx.u(r, j, t), -1.0)], -math.inf, 0.0)
                    cb.add([(idx.g(r, j, t), 1.0), (idx.x(r, j, t), -min(req.remaining_tokens, max_tok_slot))], -math.inf, 0.0)
                    act_terms.append((idx.x(r, j, t), 1.0))
                    act_terms.append((idx.h(r, j, t), 1.0))
                for tau in range(t):
                    act_terms.append((idx.comp(r, tau), 1.0))
                cb.add(act_terms, 1.0, 1.0)

                # Cumulative token completion: K_r * sum_{tau<=t} comp_{r,tau} <= sum_{tau<=t, j} g_{rj,tau}
                tok_terms = [(idx.comp(r, tau), req.remaining_tokens) for tau in range(t + 1)]
                tok_terms.extend((idx.g(r, j, tau), -1.0) for tau in range(t + 1) for j in range(idx.J))
                cb.add(tok_terms, -math.inf, 0.0)

            # Unused helper var check avoided; node_pos used in _add_flow_constraints
            _ = node_pos

    def _add_flow_constraints(
        self, problem: R3Problem, idx: _VarLayout, node_pos: dict[int, int], cb: _ConstraintBuilder
    ) -> None:
        in_edges: list[list[int]] = [[] for _ in range(idx.J)]
        out_edges: list[list[int]] = [[] for _ in range(idx.J)]
        for e_i, edge in enumerate(problem.edges):
            if edge.src in node_pos and edge.dst in node_pos:
                out_edges[node_pos[edge.src]].append(e_i)
                in_edges[node_pos[edge.dst]].append(e_i)

        # Link capacity: sum_r f_{r,e,t} <= (dt/8) * (b_tot - b_bg)
        for e_i, edge in enumerate(problem.edges):
            for t in range(idx.H):
                bg = 0.0 if edge.bg_bps is None else edge.bg_bps[t]
                cap_bytes = (problem.dt_s / 8.0) * max(0.0, edge.tot_bps[t] - bg)
                cb.add([(idx.f(r, e_i, t), 1.0) for r in range(idx.R)], -math.inf, cap_bytes)

        for r, req in enumerate(problem.requests):
            init_mig_j = node_pos.get(req.init_mig_dst)
            for j in range(idx.J):
                for t in range(idx.H):
                    # Flow conservation: s_{rjt} + sum_{in} f - d_{rjt} - sum_{out} f = 0
                    terms: list[tuple[int, float]] = [(idx.s(r, j, t), 1.0), (idx.d(r, j, t), -1.0)]
                    terms.extend((idx.f(r, e_i, t), 1.0) for e_i in in_edges[j])
                    terms.extend((idx.f(r, e_i, t), -1.0) for e_i in out_edges[j])
                    cb.add(terms, 0.0, 0.0)

                    # Source bytes only from unlocked node: s_{rjt} <= S_r^kv * u_{rjt}
                    cb.add([(idx.s(r, j, t), 1.0), (idx.u(r, j, t), -req.kv_bytes)], -math.inf, 0.0)

                    # Cumulative Unlock (t >= 1): S_r^kv * (u_{r,j,t} - u_{r,j,t-1}) <= init + sum_{tau<t} d_{rj,tau}
                    if t >= 1:
                        init_recv = req.init_mig_received_bytes if j == init_mig_j else 0.0
                        unlock_terms = [
                            (idx.u(r, j, t), req.kv_bytes),
                            (idx.u(r, j, t - 1), -req.kv_bytes),
                        ]
                        unlock_terms.extend((idx.d(r, j, tau), -1.0) for tau in range(t))
                        cb.add(unlock_terms, -math.inf, init_recv)

    def _add_cvar_constraints(self, problem: R3Problem, idx: _VarLayout, cb: _ConstraintBuilder) -> None:
        # L_r = elapsed_s + (H+1)*dt + sum_t ((t+1)*dt - (H+1)*dt) * comp_{r,t}
        # xi_r >= L_r - eta  =>  xi_r + eta - sum_t ((t+1)*dt - (H+1)*dt) * comp_{r,t} >= elapsed_s + (H+1)*dt
        miss_lat = (problem.horizon_slots + 1) * problem.dt_s
        for r, req in enumerate(problem.requests):
            terms: list[tuple[int, float]] = [(idx.xi(r), 1.0), (idx.eta(), 1.0)]
            for t in range(idx.H):
                coeff = (t + 1) * problem.dt_s - miss_lat
                terms.append((idx.comp(r, t), -coeff))
            cb.add(terms, req.elapsed_s + miss_lat, math.inf)

    def _stage1_cost(self, problem: R3Problem, idx: _VarLayout) -> np.ndarray:
        # Minimize -(sum_r w_r y_r - cvar_weight * (eta + 1/((1-alpha)*R) * sum_r xi_r))
        c = np.zeros(idx.total, dtype=float)
        for r, req in enumerate(problem.requests):
            c[idx.y(r)] = -req.weight
        if problem.cvar_weight > 0.0:
            denom = max(1e-9, (1.0 - problem.cvar_alpha) * idx.R)
            c[idx.eta()] = problem.cvar_weight
            for r in range(idx.R):
                c[idx.xi(r)] = problem.cvar_weight / denom
        return c

    def _infeasible_result(self, problem: R3Problem) -> R3Result:
        return R3Result(
            action=PlanAction(),
            status="infeasible",
            stage1_obj=-math.inf,
            goodput_weight=0.0,
            cvar_s=math.inf,
            migration_bytes=0.0,
            node_tiers={n.node_id: () for n in problem.nodes},
            node_temps_C={n.node_id: () for n in problem.nodes},
            goodput_req_ids=(),
        )

    def _extract_result(
        self,
        problem: R3Problem,
        idx: _VarLayout,
        node_pos: dict[int, int],
        x_sol: np.ndarray,
        c1: np.ndarray,
        status: str,
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
            if x_sol[idx.h(r, j0, 0)] > 0.5 and not req.currently_paused:
                pause_ids.append(req.req_id)
            elif x_sol[idx.x(r, j0, 0)] > 0.5 and req.currently_paused:
                resume_ids.append(req.req_id)

            sent_0 = float(x_sol[idx.s(r, j0, 0)])
            if sent_0 > 1e-6:
                dst_j = max(
                    (j for j in range(idx.J) if j != j0),
                    key=lambda j: sum(float(x_sol[idx.d(r, j, tau)]) for tau in range(idx.H)),
                    default=None,
                )
                if dst_j is not None:
                    nbytes = None if sent_0 >= req.kv_bytes - 1e-6 else sent_0
                    migrations.append(
                        MigrationOrder(req_id=req.req_id, dst=problem.nodes[dst_j].node_id, nbytes=nbytes)
                    )

        mig_bytes = sum(
            float(x_sol[idx.f(r, e, t)]) for r in range(idx.R) for e in range(idx.E) for t in range(idx.H)
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
        )
