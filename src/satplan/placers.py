"""Initial placement rules (Placer): select j_0(r) on request arrival."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from satplan.world import CoWorld, Request


def nearest(world: CoWorld, req: Request) -> int | None:
    """Pick the reachable compute satellite with KV room and fewest hops from origin."""
    now = world.env.now
    best_j: int | None = None
    best_hops = 1 << 30
    best_load = 1 << 30
    for hops, j in world.trace.compute_by_hops(req.origin, now):
        if hops > best_hops:
            break
        if not world.fits(j, req):
            continue
        load = world.node_load(j)
        if hops < best_hops or load < best_load:
            best_hops = hops
            best_load = load
            best_j = j
    return best_j


PLACERS: dict[str, Callable[[CoWorld, Request], int | None]] = {
    "nearest": nearest,
}


def get_placer(name: str) -> Callable[[CoWorld, Request], int | None]:
    if name not in PLACERS:
        raise ValueError(f"unknown placer {name!r}; available: {sorted(PLACERS)}")
    return PLACERS[name]
