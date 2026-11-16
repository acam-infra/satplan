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
    best_key: tuple[int, int, int] | None = None
    for j in world.trace.compute:
        if not world.fits(j, req):
            continue
        hops = world.trace.hop_count(req.origin, j, now)
        if hops is None:
            continue
        key = (hops, world.node_load(j), j)
        if best_key is None or key < best_key:
            best_key = key
            best_j = j
    return best_j


PLACERS: dict[str, Callable[[CoWorld, Request], int | None]] = {
    "nearest": nearest,
}


def get_placer(name: str) -> Callable[[CoWorld, Request], int | None]:
    if name not in PLACERS:
        raise ValueError(f"unknown placer {name!r}; available: {sorted(PLACERS)}")
    return PLACERS[name]
