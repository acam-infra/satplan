"""Aggregate P5 raw JSON results into compact, unit-annotated summary.md."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

COLS: tuple[tuple[str, str, float, bool, int], ...] = (
    ("good_frac", "SLO 达标率(%) ↑", 100.0, True, 2),
    ("good_long_frac", "长任务达标(%) ↑", 100.0, True, 2),
    ("goodput_tok_s", "Goodput(tok/s) ↑", 1.0, True, 2),
    ("badput_tok_s", "Badput(tok/s) ↓", 1.0, False, 2),
    ("undelivered_frac", "未完成率(%) ↓", 100.0, False, 2),
    ("e2e_p95_all_s", "E2E P95(s) ↓", 1.0, False, 1),
    ("peak_temp_C", "峰值温度(°C) ↓", 1.0, False, 1),
    ("zero_power_frac", "0W 空闲(%) ↑", 100.0, True, 2),
    ("energy_kj", "总能耗(kJ) ↓", 1.0, False, 1),
    ("link_GB_hops", "链路流量(GB·hop) ↓", 1.0, False, 3),
)


def _fmt(val: float, base: float | None, *, scale: float, decimals: int, is_best: bool) -> str:
    scaled = val * scale
    s = f"{scaled:.{decimals}f}"
    if is_best:
        s = f"**{s}**"
    if base is None:
        return s
    b_scaled = base * scale
    if abs(b_scaled) <= 1e-12:
        pct = "N/A" if abs(scaled) > 1e-12 else "+0.0%"
    else:
        pct = f"{(scaled - b_scaled) / abs(b_scaled) * 100.0:+.1f}%"
    return f"{s} ({pct})"


def _mean_cell(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, _, _, _, _ in COLS:
        vals = [float(r["metrics"][k]) for r in rows if r["metrics"].get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def _render_table(
    title: str,
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    placers: tuple[str, ...],
    planners: tuple[str, ...],
    base_key: tuple[str, str],
) -> list[str]:
    lines = [f"## {title}", ""]
    means = {k: _mean_cell(v) for k, v in rows_by_key.items() if v}
    base_row = means.get(base_key)
    best: dict[str, float] = {}
    active = [means[(pl, pn)] for pl in placers for pn in planners if (pl, pn) in means]
    for k, _, _, hi, _ in COLS:
        vals = [r[k] for r in active]
        if vals:
            best[k] = max(vals) if hi else min(vals)

    headers = ["Placer", "Planner", *(c[1] for c in COLS)]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for pl in placers:
        for pn in planners:
            r = means.get((pl, pn))
            if r is None:
                continue
            cells = [pl, pn]
            for k, _, sc, _, dec in COLS:
                val = r[k]
                b = None if ((pl, pn) == base_key or base_row is None) else base_row[k]
                is_b = k in best and abs(val - best[k]) <= 1e-12
                cells.append(_fmt(val, b, scale=sc, decimals=dec, is_best=is_b))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def generate_report(raw_dir: Path, out_md: Path) -> None:
    files = sorted(raw_dir.glob("*.json"))
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    lines = [
        "# LAB-2025 P5 Bounded W53 Fair Experiment Summary",
        "",
        f"- **Raw unit files**: `{len(records)}`",
        "- **基准说明**: 括号内为相对同预算组 `nearest+none`（静态 `place_once`）的百分比变化；每列仅加粗最优值。",
        "",
    ]

    placers = ("nearest", "least_loaded", "orbit_thermal")
    for mig_tag, mig_label in (("mig4", "max_mig_per_req = 4"), ("mignull", "max_mig_per_req = null")):
        by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in records:
            if r.get("group") in ("main", "fc") and (
                (mig_tag == "mig4" and r.get("max_mig_per_req") == 4)
                or (mig_tag == "mignull" and r.get("max_mig_per_req") is None)
            ):
                by_key.setdefault((r["placer"], r["planner"]), []).append(r)
        lines.extend(
            _render_table(
                f"主矩阵 + 同信息预测对照 ({mig_label})",
                by_key,
                placers,
                ("none", "reactive", "r3_milp", "forecast_aware", "r3_milp_fc"),
                ("nearest", "none"),
            )
        )

    # Minimal ablation
    abl_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("group") == "ablation":
            abl_by_key.setdefault((r["placer"], r["planner"]), []).append(r)
    if abl_by_key:
        lines.extend(
            _render_table(
                "最小消融 (nearest, max_mig_per_req = 4)",
                abl_by_key,
                ("nearest",),
                ("tail_off_fc_off", "tail_on_fc_off", "tail_on_fc_on"),
                ("nearest", "tail_off_fc_off"),
            )
        )

    # Solver diagnostics
    r3_rows = [r for r in records if r.get("metrics", {}).get("solver_calls", 0) > 0]
    if r3_rows:
        lines.extend(["## R3 求解器诊断汇总", ""])
        lines.append("| 组别 | Calls | Optimal | Feasible | Fallback率(%) ↓ | Mean Wall(s) ↓ | P95 Wall(s) ↓ | Max Wall(s) ↓ |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        by_grp: dict[str, list[dict[str, Any]]] = {}
        for r in r3_rows:
            grp = f"{r.get('group')}:{r.get('planner')}:mig={r.get('max_mig_per_req')}"
            by_grp.setdefault(grp, []).append(r)
        for grp, rs in sorted(by_grp.items()):
            calls = sum(r["metrics"]["solver_calls"] for r in rs)
            opt = sum(r["metrics"]["solver_optimal"] for r in rs)
            feas = sum(r["metrics"]["solver_feasible"] for r in rs)
            fb = sum(r["metrics"]["solver_fallback"] for r in rs)
            mw = sum(r["metrics"]["solver_wall_mean_s"] for r in rs) / len(rs)
            p95 = max(r["metrics"]["solver_wall_p95_s"] for r in rs)
            mx = max(r["metrics"]["solver_wall_max_s"] for r in rs)
            lines.append(
                f"| {grp} | {calls} | {opt} | {feas} | {fb / max(1, calls) * 100:.2f} | {mw:.3f} | {p95:.3f} | {mx:.3f} |"
            )
        lines.append("")

    # Diagnostic subset
    diag = next((r for r in records if r.get("unit") == "diag__subset"), None)
    if diag is not None:
        lines.extend(["## 固定诊断子集与有界 Oracle 参照", ""])
        lines.append("| 场景 | 策略 | SLO 达标率(%) ↑ | Goodput(tok/s) ↑ | 迁移/交付 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for sc in diag["scenarios"]:
            m = sc["metrics"]
            lines.append(
                f"| {sc['scenario']} | {sc['policy']} | {m['good_frac']*100:.1f} | {m['goodput_tok_s']:.2f} | {m['delivered']}/{m['n_req']} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    exp_dir = Path(__file__).resolve().parent
    raw = Path(os.environ.get("RAW_DIR", str(exp_dir / "out")))
    out = Path(os.environ.get("SUMMARY_MD", str(raw.parent / "summary.md")))
    generate_report(raw, out)
