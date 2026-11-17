# Test budget and execution tiers

Tests protect invariants; growing a full simulation for every review round is not a goal. Prefer a small `R3Problem` or module test when it can expose the defect. Add a bounded `CoWorld → ProblemBuilder → R3Solver → PlanAction` case when that seam is the behavior under review. Do not copy production formulas into the oracle or add tests for reversible prose changes.

## Admission and size

- Before adding a case, identify the invariant, the smallest failing seam and the existing case it extends. Put equivalent positive/negative inputs in a parameter table rather than copying a whole simulation. Retain a useful positive case when a conservative fix could reject all work.
- Around 800 lines in one test file triggers a split by invariant domain. Do not hide growth in formatting or delete distinct coverage to meet a number. A split must preserve collected test cases and independent assertions.
- Shared builders belong in `tests/factories.py`; their fixture entry points are centralized in `tests/conftest.py` (`base_cfg_factory`, `two_cs_trace_factory`). Do not import helpers from another test module. Preserve different physical defaults explicitly instead of merging unlike configurations silently.
- E2E suite tests use one seed and a short horizon to verify composition, accounting and artifact contracts. Multi-seed performance matrices belong in declarative `experiments/<id>/` packages on `cnb-compute/2`, with fixed requests, configuration and raw evidence.

## Execution

```bash
pytest -m "not e2e"       # daily unit + bounded regression feedback
pytest -m unit            # pure-model/module checks
pytest -m regression      # public-seam invariants
pytest -m e2e             # bounded runner/suite wiring
pytest                    # full PR gate; nothing is skipped by default
pytest --durations=10      # identify the current cost concentration
```

The registered markers are strict. Pure trace/thermal/solver modules default to `unit`; unmarked invariant cases default to `regression`; E2E modules/functions opt in explicitly. Choose the marker by scope, not by whether the test currently happens to be fast. Record current timings when changing the suite; historical timings from another commit or machine are not a service-level promise.

The first cleanup keeps the collected cases while splitting P0–P2 into:

| Domain | Module |
| --- | --- |
| model, batch, budget and fallback contracts | `test_p0_p2_model.py` |
| migration session/path/propagation lifecycle | `test_p0_p2_migration.py` |
| power, thermal and DoD admission | `test_p0_p2_power.py` |
| real actuation cadence and runtime validation | `test_p0_p2_cadence.py` |
| background flow occupancy and retries | `test_p0_p2_background.py` |

`test_p0_p2_consistency.py` is a small compatibility facade for archived probes, not a second test suite. P3/P4 tests use the shared builders directly. When another module is touched, fold genuinely identical `_cfg`/`_trace` builders into these fixtures; distinct model defaults must remain named and explicit. Do not combine test reorganization with an unrelated numerical-model change.

## Archived independent probes

Once an invariant is captured in pytest and passes the PR gate, its historical review script leaves the routine gate. Keep the original script, output and reviewed SHA as evidence; do not rerun every old round on every PR. A review runs the relevant changed-point probe, if it still provides an independent oracle, plus pytest and root lint. Revisit an archived probe only when that mechanism changes or pytest lacks the needed coverage.

## Experiments and slow references

The all-fine-grid MILP is a separate bounded experiment reference with an explicit timeout, incumbent/fallback status and measured solver cost. Compare it against the mixed-grid method with the same horizon, data, physics and information; label any wider-timeout run separately. It is not a mathematical upper bound by default. Do not expand pytest into a multi-seed solver benchmark or claim that the mixed-grid planner is universally faster from a tiny seam timing.

Run `python3 cosim.py lint satplan/src satplan/tests` from the umbrella root before merge. P5 scientific runs continue independently; their request/config/source hashes must stay fixed while this test-only cleanup is integrated.
