# LAB-2082 · K1/K2/K3 Parameter Scan & Counterexamples

Controlled parameter scan and deterministic L0 counterexamples on `acam-infra/satplan` (base `bb0ff76003eb53f31d1d6654c26147208426dc46`, branch `exp/lab2082-knob-scan`).

## Key Code Changes
- **Causal Length Prediction (`src/satplan/world.py`, `src/satplan/views.py`)**:
  - Added `Request.predicted_pout: int | None = None` and `Request.predicted_remaining`.
  - Updated `ForecastView.predict_decode_end()` and `CoWorld.observed_arrivals()` to read `predicted_remaining` / `predicted_pout` instead of true `pout`, ensuring true `pout` only drives simulation progress and final SLO/Goodput settlement.
- **Arms (`experiments/LAB-2082/common.py`)**:
  - `Static`, `Queue-Reroute` (reroutes only requests still in `upload` state, preserving upload & compute cost), `Dyn-Zero`, `Dyn-CF`, `Dyn-Sh-Naive`, and `Dyn-Sh-Aware` (selects target minimizing `estimate_shared_mig_time + remaining_decode_time`).

## Reproduce

```bash
# 1. Run all unit, causal, and L0 counterexample tests
PYTHONPATH=. uv run pytest -q

# 2. Run full L1 (dev seeds 101-103) + Oracle + L2 (val seeds 142-146, 6-sat & 24-sat)
PYTHONPATH=.:src uv run python experiments/LAB-2082/run_scan.py --phase all --workers 16

# 3. Generate summary tables and plots from raw JSON
PYTHONPATH=.:src uv run --with matplotlib==3.10.9 python experiments/LAB-2082/analyze_and_plot.py
```

Outputs are written to `experiments/LAB-2082/out/`:
- `l1_k1k2.json`, `l1_k3.json`, `l1_oracle.json`, `l2_val_6sat.json`, `l2_val_24sat.json`, `manifest.json`
- `summary_tables.md`, `gap_waterfall.png`, `cdf_val.png`
