"""Initial placement rules (Placer): select j_0(r) on request arrival."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from satplan.trace import INF

if TYPE_CHECKING:
    from satplan.views import ForecastView, LocalStateView
    from satplan.world import Request


def nearest(view: LocalStateView, req: Request) -> int | None:
    """Pick the reachable compute satellite with KV room and fewest hops from origin."""
    best_j: int | None = None
    best_hops = 1 << 30
    best_load = 1 << 30
    for hops, j in view.compute_by_hops(req.origin):
        if hops > best_hops:
            break
        if not view.fits(j, req):
            continue
        load = view.node_load(j)
        if hops < best_hops or load < best_load:
            best_hops = hops
            best_load = load
            best_j = j
    return best_j


def least_loaded(view: LocalStateView, req: Request) -> int | None:
    """Pick the reachable compute satellite with KV room and smallest queue load (tie-break hops, id)."""
    best_j: int | None = None
    best_key: tuple[int, int, int] | None = None
    for hops, j in view.compute_by_hops(req.origin):
        if not view.fits(j, req):
            continue
        key = (view.node_load(j), hops, j)
        if best_key is None or key < best_key:
            best_key = key
            best_j = j
    return best_j


def orbit_thermal(view: ForecastView, req: Request) -> int | None:
    """Pick the reachable compute satellite minimizing predicted delivery across sunlit/eclipse & thermal state."""
    best_j: int | None = None
    best_key: tuple[int, float, int, int] | None = None
    now = view.now
    t_warn = view.t_max_C - 10.0
    for hops, j in view.compute_by_hops(req.origin):
        if not view.fits(j, req):
            continue
        up_dur = view.transfer_estimate(req.origin, j, req.upload_bytes)
        if up_dur is None or up_dur == INF:
            continue
        t_start = now + up_dur
        t_dec = view.predict_decode_end(j, req, t_start, joining=True)
        if not view.rs_downlink:
            t_del = t_dec
            stall_cost = 0.0
        else:
            t_del = INF if t_dec == INF else view.next_conn(req.origin, j, t_dec)
            stall_cost = (
                view.stall_weight * view.disconnected_time(req.origin, j, t_start, t_dec)
                if t_del < INF and view.stream
                else 0.0
            )
        thermal_pen = max(0.0, view.temp_C(j) - t_warn) * 5.0
        primary = (t_del + stall_cost if t_del < INF else t_dec) + thermal_pen
        key = (0 if t_del < INF else 1, primary, hops, j)
        if best_key is None or key < best_key:
            best_key = key
            best_j = j
    return best_j


orbit_thermal.requires_forecast = True  # type: ignore[attr-defined]


PLACERS: dict[str, Callable[[Any, Request], int | None]] = {
    "nearest": nearest,
    "least_loaded": least_loaded,
    "orbit_thermal": orbit_thermal,
}


def get_placer(name: str) -> Callable[[Any, Request], int | None]:
    if name not in PLACERS:
        raise ValueError(f"unknown placer {name!r}; available: {sorted(PLACERS)}")
    return PLACERS[name]
