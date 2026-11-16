"""Pure-data runtime actuation orders shared between planners and CoWorld."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MigrationOrder:
    req_id: int
    dst: int
    nbytes: float | None = None


@dataclass(frozen=True)
class PlanAction:
    tier_decisions: dict[int, int | None] = field(default_factory=dict)
    pause_ids: tuple[int, ...] = ()
    resume_ids: tuple[int, ...] = ()
    migrations: tuple[MigrationOrder, ...] = ()
