"""Seam tests for Trace: orbitsim.contact_trace/v1 loading and instance-owned routing caches."""
from __future__ import annotations

from satplan.trace import SCHEMA, Trace


def _diamond_trace_data(*, extra_link: bool = True) -> dict:
    edges = [
        {"u": 0, "v": 1, "kind": "rs", "up": [[0.0, 100.0]]},
        {"u": 1, "v": 2, "kind": "inter", "up": [[0.0, 100.0]]},
        {"u": 1, "v": 3, "kind": "inter", "up": [[0.0, 100.0]]},
        {"u": 2, "v": 4, "kind": "inter", "up": [[0.0, 100.0]]},
    ]
    if extra_link:
        edges.append({"u": 3, "v": 4, "kind": "inter", "up": [[0.0, 100.0]]})
    return {
        "schema": SCHEMA,
        "name": "synthetic-diamond",
        "horizon_s": 100.0,
        "nodes": [
            {"id": 0, "name": "RS-0", "role": "rs"},
            {"id": 1, "name": "C-1", "role": "compute"},
            {"id": 2, "name": "C-2", "role": "compute"},
            {"id": 3, "name": "C-3", "role": "compute"},
            {"id": 4, "name": "C-4", "role": "compute"},
        ],
        "edges": edges,
        "dark": [[], [[10.0, 30.0]], [], [], []],
    }


def test_trace_instance_caches_are_isolated() -> None:
    t1 = Trace(_diamond_trace_data(extra_link=True))
    t2 = Trace(_diamond_trace_data(extra_link=False))

    assert t1.hop_count(0, 4, 5.0) == 3
    assert t2.hop_count(0, 4, 5.0) == 3

    paths1 = t1.k_shortest_paths(0, 4, 5.0, k=3)
    paths2 = t2.k_shortest_paths(0, 4, 5.0, k=3)
    assert len(paths1) == 2
    assert len(paths2) == 1
    assert paths1[0][0] == [0, 1, 2, 4]
    assert paths1[1][0] == [0, 1, 3, 4]


def test_trace_eclipse_and_rs_relay_exclusion() -> None:
    data = _diamond_trace_data(extra_link=False)
    data["edges"][1]["up"] = []  # disable 1-2 so only RS-5 connects 4 and 1
    data["nodes"].append({"id": 5, "name": "RS-5", "role": "rs"})
    data["edges"].append({"u": 4, "v": 5, "kind": "rs", "up": [[0.0, 100.0]]})
    data["edges"].append({"u": 5, "v": 1, "kind": "rs", "up": [[0.0, 100.0]]})
    data["dark"].append([])

    tr = Trace(data)
    assert tr.is_eclipse(1, 15.0)
    assert not tr.is_eclipse(1, 5.0)
    assert tr.path(4, 1, 5.0, via_rs=False) is None
    assert tr.path(4, 1, 5.0, via_rs=True) == ([4, 5, 1], [4, 5])
