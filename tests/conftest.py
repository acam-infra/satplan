"""Shared fixture entry points and explicit default test tiers."""
from __future__ import annotations

from collections.abc import Callable

import pytest
from factories import _base_cfg, _two_cs_trace
from satplan.trace import Trace

_UNIT_MODULES = frozenset({"test_trace.py", "test_power_thermal.py", "test_r3_solver.py"})
_TIERS = ("unit", "regression", "e2e")


@pytest.fixture
def base_cfg_factory() -> Callable[..., dict]:
    return _base_cfg


@pytest.fixture
def two_cs_trace_factory() -> Callable[..., Trace]:
    return _two_cs_trace


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not any(item.get_closest_marker(tier) for tier in _TIERS):
            tier = "unit" if item.path.name in _UNIT_MODULES else "regression"
            item.add_marker(getattr(pytest.mark, tier))
