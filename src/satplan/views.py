"""Observation seams: narrow LocalStateView (current state only) vs ForecastView (H-slot lookahead)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from satplan.trace import INF

if TYPE_CHECKING:
    from satplan.actions import BackgroundFlowSnapshot, MigrationSessionSnapshot
    from satplan.power_thermal import PhysicsParams
    from satplan.world import CoWorld, Request


class LocalStateView:
    """Read-only observation seam exposing ONLY current simulation state.

    Never exposes ``trace``, ``power_segments``, ``next_conn``, or future ``eclipse`` intervals.
    """

    def __init__(self, world: CoWorld) -> None:
        self._w = world

    @property
    def now(self) -> float:
        return self._w.env.now

    @property
    def compute_nodes(self) -> tuple[int, ...]:
        return tuple(self._w.trace.compute)

    @property
    def bpt(self) -> int:
        return self._w.bpt

    @property
    def kv_cap(self) -> float:
        return self._w.kv_cap

    @property
    def hop_latency_s(self) -> float:
        return self._w.hop_lat

    @property
    def control_period_s(self) -> float:
        return self._w.pt.period

    @property
    def t_max_C(self) -> float:
        return self._w.pt.params.t_max_C

    @property
    def t_amb_C(self) -> float:
        return self._w.pt.params.t_amb_C

    @property
    def stream(self) -> bool:
        return self._w.stream

    @property
    def rs_downlink(self) -> bool:
        return self._w.rs_downlink

    @property
    def inflight_migrations(self) -> int:
        return self._w.inflight_migrations()

    def node_load(self, j: int) -> int:
        return self._w.node_load(j)

    def active_load(self, j: int) -> int:
        return self._w._active_count(j)

    def node_requests(self, j: int) -> tuple[Request, ...]:
        return tuple(self._w.nodes[j])

    def kv_used(self, j: int) -> float:
        return self._w.kv_used(j)

    def kv_headroom(self, j: int) -> float:
        return max(0.0, self._w.kv_cap - self._w.kv_used(j))

    def fits(self, j: int, req: Request) -> bool:
        return self._w.fits(j, req)

    def temp_C(self, j: int) -> float:
        return self._w.pt.temp_C[j]

    def dod_remaining_j(self, j: int) -> float:
        return self._w.pt.dod_remaining_j[j]

    def is_eclipse(self, j: int) -> bool:
        return self._w.trace.is_eclipse(j, self._w.env.now)

    def edge_rates(self) -> dict[int, float]:
        return dict(self._w.net_edge_rates())

    def upload_edge_rates(self) -> dict[int, float]:
        return self._w.upload_edge_rates()

    def migration_snapshot(self, req_id: int) -> MigrationSessionSnapshot | None:
        return self._w.migration_snapshot(req_id)

    def edge_utilization(self) -> dict[int, float]:
        rates = self.edge_rates()
        return {e: rate / self._w.net.cap[e] for e, rate in rates.items() if self._w.net.cap[e] > 0}

    def compute_by_hops(self, origin: int) -> tuple[tuple[int, int], ...]:
        return self._w.trace.compute_by_hops(origin, self._w.env.now)

    def hop_count(self, s: int, d: int) -> int | None:
        return self._w.trace.hop_count(s, d, self._w.env.now)

    def connected(self, o: int, j: int) -> bool:
        return self.hop_count(o, j) is not None

    def path(self, s: int, d: int) -> tuple[list[int], list[int]] | None:
        return self._w.trace.path(s, d, self._w.env.now)

    def iter_of(
        self,
        j: int,
        extra: Request | None = None,
        *,
        extra_load: int = 0,
        extra_kv: float = 0.0,
    ) -> float:
        rs = self._w._active_reqs(j)
        nd = len(rs) + extra_load + (1 if extra is not None else 0)
        cf = self._w.pt.candidate_clock_f(j, nd, is_eclipse=self.is_eclipse(j))
        if cf <= 0.0:
            return INF
        ctx = sum(r.kv for r in rs) + extra_kv + (extra.kv if extra is not None else 0.0)
        return self._w.pt.params.iter_time(max(nd, 1), ctx, cf)

    def transfer_estimate(self, s: int, d: int, nbytes: float) -> float | None:
        p = self.path(s, d)
        if p is None:
            return None
        edges = p[1]
        if not edges:
            return 0.0
        users: dict[int, int] = {}
        for f in self._w.net.flows:
            for _, _, e in f.hops:
                users[e] = users.get(e, 0) + 1
        bw = min(self._w.net.cap[e] / (users.get(e, 0) + 1) for e in edges)
        if bw <= 0:
            return INF
        return nbytes / bw + self._w.hop_lat * len(edges)


class ForecastView(LocalStateView):
    """Extended observation seam exposing H-slot future eclipse/power, link, and connectivity lookahead."""

    @property
    def horizon_s(self) -> float:
        return self._w.trace.horizon

    @property
    def f_eclipse(self) -> float:
        return self._w.pt.params.f_eclipse

    @property
    def stall_weight(self) -> float:
        return self._w.stall_weight

    @property
    def physics_params(self) -> PhysicsParams:
        return self._w.pt.params

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._w.trace.edges)

    def f_of(self, j: int, t: float | None = None) -> float:
        return self._w.f_of(j, t)

    def is_eclipse_at(self, j: int, t: float) -> bool:
        return self._w.trace.is_eclipse(j, t)

    def power_segments(self, j: int, t0: float, t1: float | None = None) -> list[tuple[float, float, bool]]:
        end = self._w.trace.horizon if t1 is None else min(t1, self._w.trace.horizon)
        if t0 >= end:
            return []
        segs: list[tuple[float, float, bool]] = []
        t = t0
        for a, b in self._w.trace.eclipse[j]:
            if b <= t:
                continue
            if a >= end:
                break
            if a > t:
                segs.append((t, a, False))
            segs.append((max(t, a), min(b, end), True))
            t = min(b, end)
        if t < end:
            segs.append((t, end, False))
        return segs

    def edge_cap_Bps(self, e: int) -> float:
        return self._w.net.cap[e]

    def edge_up_at(self, e: int, t: float) -> bool:
        return self._w.trace.edge_up(e, t)

    def edge_up_until(self, e: int, t: float) -> float:
        return self._w.trace.edge_up_until(e, t)

    def next_edge_up(self, e: int, t: float) -> float:
        return self._w.trace.next_edge_up(e, t)

    def background_flow_snapshots(self) -> tuple[BackgroundFlowSnapshot, ...]:
        return self._w.background_flow_snapshots()

    def path_at(self, s: int, d: int, t: float) -> tuple[list[int], list[int]] | None:
        return self._w.trace.path(s, d, t)

    def next_conn(self, o: int, j: int, t: float) -> float:
        return self._w.trace.next_conn(o, j, t)

    def conn_until(self, o: int, j: int, t: float) -> float:
        return self._w.trace.conn_until(o, j, t)

    def disconnected_time(self, o: int, j: int, t0: float, t1: float) -> float:
        return self._w.trace.disconnected_time(o, j, t0, t1)

    def iter_of_at(self, j: int, f: float, extra: Request | None = None) -> float:
        if f <= 0.0:
            return INF
        rs = self._w._active_reqs(j)
        nd = len(rs) + (1 if extra is not None else 0)
        ctx = sum(r.kv for r in rs) + (extra.kv if extra is not None else 0.0)
        return self._w.pt.params.iter_time(max(nd, 1), ctx, f)

    def predict_decode_end(self, j: int, req: Request, t0: float, *, joining: bool = True) -> float:
        extra = req if joining else None
        it_lit = self.iter_of_at(j, 1.0, extra)
        it_ecl = self.iter_of_at(j, self.f_eclipse, extra)
        left = req.remaining
        if left <= 0:
            return t0
        for a, b, ecl in self.power_segments(j, t0):
            it = it_ecl if ecl else it_lit
            if it == INF:
                continue
            span = b - a
            if left * it <= span:
                return a + left * it
            left -= span / it
        return INF
