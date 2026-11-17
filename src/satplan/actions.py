"""Pure-data runtime actuation orders shared between planners and CoWorld."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MigrationOrder:
    req_id: int
    dst: int
    nbytes: float | None = None


@dataclass(frozen=True)
class MigrationSessionSnapshot:
    phase: str  # idle | sending | propagating_chunk | propagating_unlock | resumable
    dst: int
    target_bytes: float
    received_bytes: float
    chunk_sent_bytes: float = 0.0
    chunk_rem_bytes: float = 0.0
    prop_rem_s: float = 0.0
    path_edges: tuple[int, ...] = ()
    path_nodes: tuple[int, ...] = ()


@dataclass(frozen=True)
class BackgroundFlowSnapshot:
    req_id: int | None
    hops: tuple[tuple[int, int, int], ...]  # (from_node, to_node, edge_idx)
    remaining_bytes: float
    total_bytes: float
    origin: int = 0


@dataclass(frozen=True)
class PlanAction:
    tier_decisions: dict[int, int | None] = field(default_factory=dict)
    pause_ids: tuple[int, ...] = ()
    resume_ids: tuple[int, ...] = ()
    migrations: tuple[MigrationOrder, ...] = ()
