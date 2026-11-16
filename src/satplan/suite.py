"""3x3 cross-matrix SuiteRunner (Placer x RuntimePlanner) under uniform RebalanceInvariants."""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from satplan.planners import RebalanceInvariants
from satplan.runner import run_simulation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from satplan.trace import Trace

PLACERS_3X3: tuple[str, ...] = ("nearest", "least_loaded", "orbit_thermal")
PLANNERS_3X3: tuple[str, ...] = ("none", "reactive", "r3_milp")

# (metric_key, header_with_unit_and_arrow, scale, higher_is_better, decimals)
_SUMMARY_COLS: tuple[tuple[str, str, float, bool, int], ...] = (
    ("good_frac", "SLO 达标率(%) ↑", 100.0, True, 2),
    ("good_long_frac", "长请求达标(%) ↑", 100.0, True, 2),
    ("goodput_tok_s", "Goodput(tok/s) ↑", 1.0, True, 2),
    ("badput_tok_s", "Badput(tok/s) ↓", 1.0, False, 2),
    ("e2e_p95_all_s", "E2E P95(s) ↓", 1.0, False, 1),
    ("peak_temp_C", "峰值温度(°C) ↓", 1.0, False, 1),
    ("link_GB_hops", "链路流量(GB·hop) ↓", 1.0, False, 3),
)

_CSV_FIELDS: tuple[str, ...] = (
    "placer",
    "planner",
    "seed",
    "rs_downlink",
    "interval_s",
    "max_inflight",
    "max_mig_per_req",
    "n_req",
    "n_long",
    "delivered",
    "good",
    "good_long",
    "good_frac",
    "good_long_frac",
    "goodput_tok_s",
    "badput_tok_s",
    "waste_ratio",
    "undelivered_frac",
    "e2e_p95_all_s",
    "long_p95_all_s",
    "peak_temp_C",
    "energy_kj",
    "zero_power_s",
    "dod_exhausted_s",
    "upload_fail",
    "link_GB_hops",
)


@dataclass(frozen=True)
class SuiteResult:
    rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    csv_path: Path
    markdown_path: Path
    markdown: str


def _mean_metrics(cell_rows: list[dict[str, Any]], placer: str, planner: str) -> dict[str, Any]:
    out: dict[str, Any] = {"placer": placer, "planner": planner}
    for key, _, _, _, _ in _SUMMARY_COLS:
        vals = [float(r[key]) for r in cell_rows if r.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else 0.0
    return out


def _fmt_cell(val: float, base: float | None, *, scale: float, decimals: int, is_best: bool) -> str:
    scaled = val * scale
    num = f"{scaled:.{decimals}f}"
    if is_best:
        num = f"**{num}**"
    if base is None:
        return num
    base_scaled = base * scale
    if abs(base_scaled) <= 1e-12:
        pct = "N/A" if abs(scaled) > 1e-12 else "+0.0%"
    else:
        delta = (scaled - base_scaled) / abs(base_scaled) * 100.0
        pct = f"{delta:+.1f}%"
    return f"{num} ({pct})"


def _render_markdown(
    cfg: dict,
    seeds: tuple[int, ...],
    invariants: RebalanceInvariants,
    placers: Sequence[str],
    planners: Sequence[str],
    summary_rows: list[dict[str, Any]],
) -> str:
    by_cell = {(r["placer"], r["planner"]): r for r in summary_rows}
    lines = [
        f"# {cfg.get('name', 'satplan')} 3×3 Cross-Matrix Summary",
        "",
        f"- **seeds**: `{list(seeds)}` | **rs_downlink**: `{bool(cfg.get('rs_downlink', False))}`",
        (
            f"- **rebalance_invariants**: `interval_s={invariants.interval_s}`, "
            f"`max_inflight={invariants.max_inflight}`, `max_mig_per_req={invariants.max_mig_per_req}`"
        ),
        "- **基准说明**: 每个 `Placer` 行组内以 `none` 为 100% 基准，括号内为相对同行 `none` 的百分比变化。",
        "",
    ]

    headers = ["Placer", "Planner", *(col[1] for col in _SUMMARY_COLS)]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for placer in placers:
        group = [by_cell[(placer, p)] for p in planners if (placer, p) in by_cell]
        if not group:
            continue
        best_val: dict[str, float] = {}
        for key, _, _, higher_better, _ in _SUMMARY_COLS:
            vals = [float(r[key]) for r in group]
            best_val[key] = max(vals) if higher_better else min(vals)

        base_row = by_cell.get((placer, "none"))
        for planner in planners:
            row = by_cell.get((placer, planner))
            if row is None:
                continue
            cells = [placer, planner]
            for key, _, scale, _, decimals in _SUMMARY_COLS:
                val = float(row[key])
                base = None if (planner == "none" or base_row is None) else float(base_row[key])
                is_best = abs(val - best_val[key]) <= 1e-12
                cells.append(_fmt_cell(val, base, scale=scale, decimals=decimals, is_best=is_best))
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def run_suite(
    trace: Trace,
    cfg: dict,
    out_dir: Path,
    *,
    seeds: Sequence[int] | None = None,
    matrix: tuple[Sequence[str], Sequence[str]] = (PLACERS_3X3, PLANNERS_3X3),
    invariants: RebalanceInvariants | None = None,
    rs_downlink: bool | None = None,
) -> SuiteResult:
    """Execute a Placer x RuntimePlanner x multi-seed suite under uniform invariants."""
    placers, planners = matrix
    suite_cfg = copy.deepcopy(cfg)
    inv = invariants or RebalanceInvariants.from_cfg(suite_cfg)
    suite_cfg["rebalance_invariants"] = {
        "interval_s": inv.interval_s,
        "max_inflight": inv.max_inflight,
        "max_mig_per_req": inv.max_mig_per_req,
    }
    if rs_downlink is not None:
        suite_cfg["rs_downlink"] = bool(rs_downlink)
    else:
        suite_cfg["rs_downlink"] = bool(suite_cfg.get("rs_downlink", False))

    seed_tuple = tuple(int(s) for s in (seeds if seeds is not None else suite_cfg.get("seeds", [1])))
    if not seed_tuple:
        raise ValueError("run_suite requires at least one seed")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for placer in placers:
        for planner in planners:
            cell_rows: list[dict[str, Any]] = []
            for seed in seed_tuple:
                res = run_simulation(trace, copy.deepcopy(suite_cfg), placer, planner, seed, record=False)
                row: dict[str, Any] = {
                    "placer": placer,
                    "planner": planner,
                    "seed": seed,
                    "rs_downlink": suite_cfg["rs_downlink"],
                    "interval_s": inv.interval_s,
                    "max_inflight": inv.max_inflight,
                    "max_mig_per_req": inv.max_mig_per_req,
                    **res.metrics,
                }
                rows.append(row)
                cell_rows.append(row)
            summary_rows.append(_mean_metrics(cell_rows, placer, planner))

    csv_path = out_path / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown = _render_markdown(suite_cfg, seed_tuple, inv, placers, planners, summary_rows)
    md_path = out_path / "summary.md"
    md_path.write_text(markdown, encoding="utf-8")

    return SuiteResult(
        rows=tuple(rows),
        summary_rows=tuple(summary_rows),
        csv_path=csv_path,
        markdown_path=md_path,
        markdown=markdown,
    )
