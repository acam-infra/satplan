"""Periodic runtime reallocation planners (RuntimePlanner)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from satplan.world import CoWorld


class RuntimePlanner(Protocol):
    def rebalance(self, world: CoWorld) -> None: ...


class NonePlanner:
    """No-op runtime planner (baseline without migration or tier override)."""

    def rebalance(self, world: CoWorld) -> None:
        return None


PLANNERS: dict[str, type[RuntimePlanner]] = {
    "none": NonePlanner,
}


def get_planner(name: str) -> RuntimePlanner:
    if name not in PLANNERS:
        raise ValueError(f"unknown planner {name!r}; available: {sorted(PLANNERS)}")
    return PLANNERS[name]()
