"""Fluid network over the time-varying trace: max-min fair rates per directed link."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import simpy


def allocate_max_min_rates(
    active_hops: Sequence[tuple[int, Sequence[tuple[int, int, int]]]],
    edge_cap_fn: Callable[[int], float],
) -> dict[int, float]:
    """Progressive-filling max-min fair rate allocation over directed hops."""
    rates = {fid: 0.0 for fid, _ in active_hops}
    rem: dict[tuple[int, int], float] = {}
    for _, hops in active_hops:
        for a, b, e in hops:
            rem[(a, b)] = edge_cap_fn(e)
    live = list(active_hops)
    while live:
        users: dict[tuple[int, int], int] = defaultdict(int)
        for _, hops in live:
            for a, b, _ in hops:
                users[(a, b)] += 1
        share = min(rem[k] / c for k, c in users.items())
        for fid, hops in live:
            rates[fid] += share
            for a, b, _ in hops:
                rem[(a, b)] -= share
        full = {k for k, v in rem.items() if v <= 1e-6 * max(1.0, share)}
        live = [(fid, hops) for fid, hops in live if not any((a, b) in full for a, b, _ in hops)]
    return rates


@dataclass(eq=False)
class Flow:
    tag: str
    hops: list[tuple[int, int, int]]  # (from, to, edge_id)
    remaining: float
    done: simpy.Event
    rate: float = 0.0
    sent: float = 0.0
    failed: bool = False
    meta: dict = field(default_factory=dict)


class FluidNet:
    def __init__(self, env: simpy.Environment, edge_cap_Bps: list[float]):
        self.env = env
        self.cap = edge_cap_Bps
        self.flows: list[Flow] = []
        self.t = 0.0
        self.wake = env.event()
        self.bytes_hops = 0.0
        env.process(self._run())

    def add(self, tag: str, nodes: list[int], eids: list[int], nbytes: float, **meta) -> Flow:
        self.advance(self.env.now)
        hops = [(nodes[k], nodes[k + 1], eids[k]) for k in range(len(eids))]
        f = Flow(tag, hops, max(float(nbytes), 0.0), self.env.event(), meta=meta)
        if not hops or f.remaining <= 0:
            f.done.succeed()
            return f
        self.flows.append(f)
        self._kick()
        return f

    def cancel(self, f: Flow) -> None:
        self.advance(self.env.now)
        if f in self.flows:
            self.flows.remove(f)
            f.rate = 0.0
            self._kick()

    def abort_dead(self, alive) -> list[Flow]:
        self.advance(self.env.now)
        dead = [f for f in self.flows if any(not alive(e) for _, _, e in f.hops)]
        for f in dead:
            self.flows.remove(f)
            f.rate = 0.0
            f.failed = True
            if not f.done.triggered:
                f.done.succeed()
        if dead:
            self._kick()
        return dead

    def edge_rates(self) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for f in self.flows:
            for _, _, e in f.hops:
                out[e] += f.rate
        return out

    def _kick(self) -> None:
        if not self.wake.triggered:
            self.wake.succeed()

    def advance(self, now: float) -> None:
        dt = now - self.t
        self.t = now
        if dt <= 0:
            return
        done = []
        for f in self.flows:
            if f.rate <= 0:
                continue
            step = min(f.remaining, f.rate * dt)
            f.remaining -= step
            f.sent += step
            self.bytes_hops += step * len(f.hops)
            if f.remaining <= 1.0:
                f.remaining = 0.0
                done.append(f)
        for f in done:
            self.flows.remove(f)
            f.rate = 0.0
            if not f.done.triggered:
                f.done.succeed()

    def _allocate(self) -> None:
        for f in self.flows:
            f.rate = 0.0
        if not self.flows:
            return
        rates = allocate_max_min_rates(
            [(i, f.hops) for i, f in enumerate(self.flows)],
            lambda e: self.cap[e],
        )
        for i, f in enumerate(self.flows):
            f.rate = rates[i]

    def _run(self):
        while True:
            self.advance(self.env.now)
            self._allocate()
            if self.wake.triggered:
                self.wake = self.env.event()
            dts = [f.remaining / f.rate for f in self.flows if f.rate > 1e-9]
            if dts:
                yield self.env.timeout(max(min(dts), 1e-9)) | self.wake
            else:
                yield self.wake
