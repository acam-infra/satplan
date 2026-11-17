"""Shared utilities for LAB-2046 mechanism and W53 validation experiments."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from satplan.world import CoWorld

if TYPE_CHECKING:
    from satplan.actions import MigrationOrder
    from satplan.views import LocalStateView
    from satplan.world import Request


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def git_head_sha(repo_dir: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()


class ReplayPlacer:
    """Deterministic placer replaying frozen initial targets recorded from baseline B."""

    def __init__(self, initial_targets: dict[int, int]) -> None:
        self.initial_targets = dict(initial_targets)

    def __call__(self, view: LocalStateView, req: Request) -> int | None:
        dst = self.initial_targets.get(req.i)
        if dst is None:
            return None
        return dst if view.fits(dst, req) else None


class ZeroCostAtomicWorld(CoWorld):
    """Idealized oracle world: migration itself has 0 transfer bytes, 0 hop latency, 0 ISL bandwidth.

    Resident KV is tracked solely via _detach_node/_attach_node (_node_reserved_tokens).
    Upload/in-flight reservations (self.reserved) are NOT touched during atomic migration.
    """

    def check_kv_invariants(self, *, at_end: bool = False) -> dict[str, Any]:
        """Verify per-node non-negativity, capacity, unique residency, and end zeroing."""
        seen: dict[int, int] = {}
        max_kv_used = 0.0
        for j in self.nodes:
            res_b = float(self.reserved[j])
            node_b = float(self._node_reserved_tokens[j] * self.bpt)
            used_b = float(self.kv_used(j))
            assert res_b >= -1e-6, f"node {j} negative upload/mig reserved: {res_b}"
            assert node_b >= -1e-6, f"node {j} negative resident KV: {node_b}"
            assert used_b >= -1e-6, f"node {j} negative kv_used: {used_b}"
            assert used_b <= self.kv_cap + 1e-6, f"node {j} kv_used {used_b} > kv_cap {self.kv_cap}"
            max_kv_used = max(max_kv_used, used_b)
            for r in self.nodes[j]:
                assert r.i not in seen, f"req {r.i} duplicated on nodes {seen[r.i]} and {j}"
                assert r.node == j, f"req {r.i} node={r.node} != attached node {j}"
                assert r.state == "decode", f"req {r.i} attached in state={r.state}"
                seen[r.i] = j
        decode_ids = {r.i for r in self.reqs if r.state == "decode"}
        assert set(seen) == decode_ids, f"resident set {set(seen)} != decode set {decode_ids}"
        if at_end:
            undel = sum(1 for r in self.reqs if r.t_delivered is None)
            if undel == 0:
                for j in self.nodes:
                    assert abs(self.reserved[j]) < 1e-6, f"node {j} leftover reserved={self.reserved[j]}"
                    assert self._node_reserved_tokens[j] == 0, f"node {j} leftover resident tokens"
        return {
            "max_kv_used_bytes": max_kv_used,
            "final_reserved_sum": float(sum(self.reserved.values())),
            "final_resident_sum": float(sum(self._node_reserved_tokens.values()) * self.bpt),
        }

    def _start_migration(self, order: MigrationOrder) -> None:
        r = self._req_by_id.get(order.req_id)
        if r is None or r.state != "decode" or r.node == order.dst or order.dst not in self.nodes:
            return
        if self.trace.path(r.node, order.dst, self.env.now, via_rs=False) is None:
            return
        if not self.fits(order.dst, r):
            return
        src = r.node
        gen_before = r.gen
        self.check_kv_invariants()
        self._detach_node(src, r)
        r.node = order.dst
        self._attach_node(order.dst, r)
        r.trail.append(order.dst)
        r.n_mig += 1
        assert r.gen == gen_before, "decode progress altered during atomic migration"
        self.check_kv_invariants()
        self.log("mig_atomic", req=r.i, src=src, dst=order.dst)
        self.wake_node(src)
        self.wake_node(order.dst)
