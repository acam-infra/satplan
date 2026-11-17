"""Aggregate bounded all-fine-grid MILP + timeout reference into separate-timeout tables."""
from __future__ import annotations

import json
import os
from pathlib import Path


def generate_report(raw_dir: Path, out_md: Path) -> None:
    records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(raw_dir.glob("*.json"))]
    lines = [
        "# LAB-2025 P5 Bounded All-Fine-Grid MILP + Timeout Reference",
        "",
        "- **说明**：全细网格仅作同一物理/信息/总规划窗口（156s）与同线程配额下的**有界参考**（非数学全局上界）；不同 `time_limit_s` 分表呈现，超时有可行解/无解/回退全量保留。",
        "",
    ]
    for case_key, title in (
        ("small_120s_tl5", "小型代表案例（W53 120s 窗口，time_limit_s = 5.0s）"),
        ("w53_600s_tl0p5", "较大并发案例（W53 600s 窗口，time_limit_s = 0.5s）"),
    ):
        rows = [r for r in records if r.get("case") == case_key]
        lines.extend([f"## {title}", ""])
        lines.append(
            "| 网格 | 槽划分(总窗156s) | Budget(s) | Calls | Optimal | Feasible(超时有解) | Timeout无解 | Fallback率(%) ↓ | Mean Solve(s) ↓ | P95 Solve(s) ↓ | Total Wall(s) ↓ | SLO(%) ↑ | Goodput(tok/s) ↑ | 峰值温度(°C) ↓ |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            m = r["metrics"]
            lines.append(
                f"| {r['grid']} | {r['slot_dt_desc']} ({r['total_slots']}) | {r['time_limit_s']:.1f} | "
                f"{m['solver_calls']} | {m['solver_optimal']} | {m['solver_feasible']} | {m['solver_timeout_no_inc']} | "
                f"{m['solver_fallback_rate']*100:.2f} | {m['solver_wall_mean_s']:.3f} | {m['solver_wall_p95_s']:.3f} | "
                f"{r['wall_time_s']:.2f} | {m['good_frac']*100:.2f} | {m['goodput_tok_s']:.2f} | {m['peak_temp_C']:.1f} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    exp_dir = Path(__file__).resolve().parent
    raw = Path(os.environ.get("RAW_DIR", str(exp_dir / "out")))
    out = Path(os.environ.get("SUMMARY_MD", str(raw.parent / "summary.md")))
    generate_report(raw, out)
