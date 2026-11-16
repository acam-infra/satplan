"""Clean r3 time-expanded runtime reallocation engine."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from satplan.metrics import RunResult
    from satplan.runner import load_config, run_simulation
    from satplan.suite import SuiteResult, run_suite
    from satplan.trace import Trace

__all__ = ["RunResult", "SuiteResult", "Trace", "load_config", "run_simulation", "run_suite"]


def __getattr__(name: str) -> Any:
    if name == "RunResult":
        from satplan.metrics import RunResult

        return RunResult
    if name == "Trace":
        from satplan.trace import Trace

        return Trace
    if name in ("SuiteResult", "run_suite"):
        from satplan import suite

        return getattr(suite, name)
    if name in ("load_config", "run_simulation"):
        from satplan import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
