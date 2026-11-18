"""Regression and invariant tests for LAB-2084 demand validation."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "experiments" / "LAB-2084"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("lab2084_common", LAB_DIR / "common.py")
assert _spec and _spec.loader
_c84 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c84)

compute_frozen_initial_targets = _c84.compute_frozen_initial_targets
digest = _c84.digest
make_base_cfg = _c84.make_base_cfg
make_dataset = _c84.make_dataset
make_trace = _c84.make_trace
run_single_arm = _c84.run_single_arm

SPEC = json.loads((LAB_DIR / "spec.json").read_text())


def test_paired_input_and_resource_budget_rescue() -> None:
    cfg = make_base_cfg(SPEC)
    tr = make_trace(6, staggered=True, horizon_s=SPEC["horizon_s"])
    reqs, pert = make_dataset(101, 24, 12, length_ratio=2.0, dl_long_off_s=255.0, post_start_k=0)
    assert pert["length_ratio"] == 2.0
    assert len(pert["underestimated_req_ids"]) == 36

    targets = compute_frozen_initial_targets(cfg, tr, reqs)
    st = run_single_arm("Static", cfg, tr, reqs, targets, 101)
    cf = run_single_arm("Dyn-CF", cfg, tr, reqs, targets, 101)

    assert st["metrics"]["actual_admitted"] == cf["metrics"]["actual_admitted"] == 36
    assert st["metrics"]["thermal_violations"] == 0
    assert st["metrics"]["dod_violations"] == 0
    assert cf["metrics"]["thermal_violations"] == 0
    assert cf["metrics"]["dod_violations"] == 0
    assert cf["metrics"]["good_long_frac"] > st["metrics"]["good_long_frac"]
    assert cf["metrics"]["n_migrations"] > 0


def test_negative_control_no_contact_disables_rescue() -> None:
    cfg = make_base_cfg(SPEC)
    tr = make_trace(6, staggered=True, horizon_s=SPEC["horizon_s"], cut_inter_group_at_s=80.0)
    reqs, _ = make_dataset(101, 24, 12, length_ratio=2.0, dl_long_off_s=255.0, post_start_k=0)
    targets = compute_frozen_initial_targets(cfg, tr, reqs)

    st = run_single_arm("Static", cfg, tr, reqs, targets, 101)
    cf = run_single_arm("Dyn-CF", cfg, tr, reqs, targets, 101)

    assert cf["metrics"]["good_long_frac"] == st["metrics"]["good_long_frac"]
    assert cf["reject_totals"].get("no_contact", 0) > 0


def test_post_start_perturbation_preserves_total_requests_and_true_pout() -> None:
    reqs_base, _ = make_dataset(101, 24, 12, length_ratio=1.0, dl_long_off_s=255.0, post_start_k=0)
    reqs_post, pert = make_dataset(101, 24, 12, length_ratio=1.0, dl_long_off_s=255.0, post_start_k=4)

    assert len(reqs_base) == len(reqs_post) == 36
    assert len(pert["post_start_req_ids"]) == 4
    assert digest([(r.i, r.pout, r.pin) for r in reqs_base]) == digest([(r.i, r.pout, r.pin) for r in reqs_post])
    for rid in pert["post_start_req_ids"]:
        assert reqs_post[rid].t >= 11.0
