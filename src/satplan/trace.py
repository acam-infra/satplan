"""orbitsim contact trace -> time-indexed topology, power, and instance-owned routing caches."""
from __future__ import annotations

import bisect
import heapq
import json
import math
from pathlib import Path

SCHEMA = "orbitsim.contact_trace/v1"
INF = math.inf


def _components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        a, b = find(u), find(v)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(n)]


def _find_run(runs: list[tuple[float, float]], starts: list[float], t: float) -> int:
    k = bisect.bisect_right(starts, t) - 1
    if k >= 0 and runs[k][0] <= t < runs[k][1]:
        return k
    return -1


class Trace:
    def __init__(self, data: dict):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not an {SCHEMA} file")
        self.name = str(data["name"])
        self.horizon = float(data["horizon_s"])
        self.nodes = data["nodes"]
        self.n = len(self.nodes)
        self.role = [str(x["role"]) for x in self.nodes]
        self.rs = [i for i, r in enumerate(self.role) if r == "rs"]
        self.compute = [i for i, r in enumerate(self.role) if r != "rs"]
        self.edges = [(int(e["u"]), int(e["v"])) for e in data["edges"]]
        self.kind = [str(e["kind"]) for e in data["edges"]]
        self.up = [[(float(a), float(b)) for a, b in e["up"]] for e in data["edges"]]
        self._up_starts = [[a for a, _ in iv] for iv in self.up]
        self.eclipse = [[(float(a), float(b)) for a, b in d] for d in data["dark"]]
        self._eclipse_starts = [[a for a, _ in d] for d in self.eclipse]
        if len(self.eclipse) != self.n:
            raise ValueError("eclipse intervals must cover every node")

        self._adj_cache: dict[int, list[list[tuple[int, int]]]] = {}
        self._path_cache: dict[tuple[int, int, int, bool], tuple[list[int], list[int]] | None] = {}
        self._hop_cache: dict[tuple[int, int, int, bool], int | None] = {}
        self._kpaths_cache: dict[tuple[int, int, int, int, bool], list[tuple[list[int], list[int]]]] = {}
        self._hops_sorted_cache: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        self._build_epochs()
        self._build_conn()

    def _build_epochs(self) -> None:
        times = {0.0}
        for iv in self.up:
            for a, b in iv:
                times.update((a, b))
        self.epochs = sorted(t for t in times if 0.0 <= t < self.horizon)
        self.epoch_edges: list[list[int]] = []
        self.epoch_comp: list[list[int]] = []
        for t0 in self.epochs:
            ids = [i for i in range(len(self.edges)) if self.edge_up(i, t0)]
            self.epoch_edges.append(ids)
            self.epoch_comp.append(_components(self.n, [self.edges[i] for i in ids]))

    def _build_conn(self) -> None:
        self._conn: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self._conn_starts: dict[tuple[int, int], list[float]] = {}
        for o in self.rs:
            for j in range(self.n):
                runs: list[tuple[float, float]] = []
                for k, t0 in enumerate(self.epochs):
                    if self.epoch_comp[k][o] != self.epoch_comp[k][j]:
                        continue
                    t1 = self.epochs[k + 1] if k + 1 < len(self.epochs) else self.horizon
                    if runs and runs[-1][1] == t0:
                        runs[-1] = (runs[-1][0], t1)
                    else:
                        runs.append((t0, t1))
                self._conn[(o, j)] = runs
                self._conn_starts[(o, j)] = [a for a, _ in runs]

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def change_times(self) -> list[float]:
        ts = set(self.epochs)
        for iv in self.eclipse:
            for a, b in iv:
                ts.update((a, b))
        return sorted(t for t in ts if 0.0 < t < self.horizon)

    def epoch_index(self, t: float) -> int:
        return max(0, bisect.bisect_right(self.epochs, t) - 1)

    def edge_up(self, e: int, t: float) -> bool:
        return _find_run(self.up[e], self._up_starts[e], t) >= 0

    def edge_up_until(self, e: int, t: float) -> float:
        k = _find_run(self.up[e], self._up_starts[e], t)
        if k < 0:
            return t
        end = self.up[e][k][1]
        return INF if end >= self.horizon else end

    def next_edge_up(self, e: int, t: float) -> float:
        if self.edge_up(e, t):
            return t
        k = bisect.bisect_right(self._up_starts[e], t)
        return self.up[e][k][0] if k < len(self.up[e]) else INF

    def adjacency(self, t: float) -> list[list[tuple[int, int]]]:
        k = self.epoch_index(t)
        adj = self._adj_cache.get(k)
        if adj is None:
            adj = [[] for _ in range(self.n)]
            for e in self.epoch_edges[k]:
                u, v = self.edges[e]
                adj[u].append((v, e))
                adj[v].append((u, e))
            for row in adj:
                row.sort()
            self._adj_cache[k] = adj
        return adj

    def path(self, s: int, d: int, t: float, *, via_rs: bool = False) -> tuple[list[int], list[int]] | None:
        key = (self.epoch_index(t), s, d, via_rs)
        if key not in self._path_cache:
            res = self._bfs_path(self.adjacency(t), s, d, set(), set(), via_rs)
            self._path_cache[key] = res
            self._hop_cache[key] = len(res[1]) if res is not None else None
        return self._path_cache[key]

    def hop_count(self, s: int, d: int, t: float, *, via_rs: bool = False) -> int | None:
        key = (self.epoch_index(t), s, d, via_rs)
        if key not in self._hop_cache:
            self.path(s, d, t, via_rs=via_rs)
        return self._hop_cache[key]

    def compute_by_hops(self, origin: int, t: float) -> tuple[tuple[int, int], ...]:
        """Return ``((hops, compute_node), ...)`` sorted by fewest hops then node id for ``origin`` at ``t``."""
        ep = self.epoch_index(t)
        key = (ep, origin)
        cached = self._hops_sorted_cache.get(key)
        if cached is not None:
            return cached
        adj = self.adjacency(t)
        dists = [-1] * self.n
        dists[origin] = 0
        q = [origin]
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            if u != origin and self.role[u] == "rs":
                continue
            du = dists[u] + 1
            for v, _ in adj[u]:
                if dists[v] == -1:
                    dists[v] = du
                    q.append(v)
        out = tuple(sorted((dists[j], j) for j in self.compute if dists[j] >= 0))
        self._hops_sorted_cache[key] = out
        return out

    def k_shortest_paths(
        self, s: int, d: int, t: float, k: int = 3, *, via_rs: bool = False
    ) -> list[tuple[list[int], list[int]]]:
        if k <= 0:
            return []
        key = (self.epoch_index(t), s, d, k, via_rs)
        if key in self._kpaths_cache:
            return self._kpaths_cache[key]
        first = self.path(s, d, t, via_rs=via_rs)
        if first is None or s == d:
            out = [first] if first is not None else []
            self._kpaths_cache[key] = out
            return out
        adj = self.adjacency(t)
        found: list[tuple[list[int], list[int]]] = [first]
        cands: list[tuple[int, list[int], list[int]]] = []
        seen = {tuple(first[0])}
        for _ in range(1, k):
            self._expand_yen(adj, found, cands, seen, d, via_rs)
            if not cands:
                break
            _, nodes, eids = heapq.heappop(cands)
            found.append((nodes, eids))
        self._kpaths_cache[key] = found
        return found

    def _expand_yen(
        self,
        adj: list[list[tuple[int, int]]],
        found: list[tuple[list[int], list[int]]],
        cands: list[tuple[int, list[int], list[int]]],
        seen: set[tuple[int, ...]],
        d: int,
        via_rs: bool,
    ) -> None:
        prev_nodes, prev_eids = found[-1]
        for idx in range(len(prev_nodes) - 1):
            spur = prev_nodes[idx]
            root_n = prev_nodes[: idx + 1]
            root_e = prev_eids[:idx]
            blocked_e = {e for pn, pe in found if pn[: idx + 1] == root_n and len(pe) > idx for e in (pe[idx],)}
            spur_res = self._bfs_path(adj, spur, d, set(root_n[:-1]), blocked_e, via_rs)
            if spur_res is None:
                continue
            cand_n = root_n[:-1] + spur_res[0]
            cand_e = root_e + spur_res[1]
            if tuple(cand_n) not in seen:
                seen.add(tuple(cand_n))
                heapq.heappush(cands, (len(cand_e), cand_n, cand_e))

    def _bfs_path(
        self,
        adj: list[list[tuple[int, int]]],
        s: int,
        d: int,
        blocked_nodes: set[int],
        blocked_edges: set[int],
        via_rs: bool,
    ) -> tuple[list[int], list[int]] | None:
        if s == d:
            return [s], []
        prev: dict[int, tuple[int, int] | None] = {s: None}
        q = [s]
        i = 0
        while i < len(q) and d not in prev:
            u = q[i]
            i += 1
            if u != s and not via_rs and self.role[u] == "rs":
                continue
            for v, e in adj[u]:
                if v in blocked_nodes or e in blocked_edges or v in prev:
                    continue
                prev[v] = (u, e)
                q.append(v)
        if d not in prev:
            return None
        nodes, eids = [d], []
        x = d
        while prev[x] is not None:
            u, e = prev[x]  # type: ignore[misc]
            nodes.append(u)
            eids.append(e)
            x = u
        return nodes[::-1], eids[::-1]

    def is_eclipse(self, j: int, t: float) -> bool:
        return _find_run(self.eclipse[j], self._eclipse_starts[j], t) >= 0

    def connected(self, o: int, j: int, t: float) -> bool:
        return _find_run(self._conn[(o, j)], self._conn_starts[(o, j)], t) >= 0

    def next_conn(self, o: int, j: int, t: float) -> float:
        runs, starts = self._conn[(o, j)], self._conn_starts[(o, j)]
        if _find_run(runs, starts, t) >= 0:
            return t
        k = bisect.bisect_right(starts, t)
        return runs[k][0] if k < len(runs) else INF

    def conn_until(self, o: int, j: int, t: float) -> float:
        runs, starts = self._conn[(o, j)], self._conn_starts[(o, j)]
        k = _find_run(runs, starts, t)
        if k < 0:
            return t
        return INF if runs[k][1] >= self.horizon else runs[k][1]

    def disconnected_time(self, o: int, j: int, t0: float, t1: float) -> float:
        t1c = min(t1, self.horizon)
        if t1c <= t0:
            return 0.0
        runs, starts = self._conn[(o, j)], self._conn_starts[(o, j)]
        k = max(0, bisect.bisect_right(starts, t0) - 1)
        up = 0.0
        while k < len(runs) and runs[k][0] < t1c:
            a, b = max(runs[k][0], t0), min(runs[k][1], t1c)
            if b > a:
                up += b - a
            k += 1
        return (t1c - t0) - up
