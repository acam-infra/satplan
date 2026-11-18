"""Analyze LAB-2082 raw scan outputs and generate summary tables + matplotlib plots."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def _fmt_pct_delta(val: float, base: float, *, pp: bool = False) -> str:
    if pp:
        d = (val - base) * 100.0
        return f"{val * 100:.1f}% ({d:+.1f} pp)"
    if abs(base) < 1e-12:
        return f"{val:.1f}"
    d = (val - base) / base * 100.0
    return f"{val:.1f} ({d:+.1f}%)"


def summarize_group(cells: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        ms = [c["arms"][arm]["metrics"] for c in cells if arm in c["arms"]]
        if not ms:
            continue
        rows.append({
            "arm": arm,
            "good_long_frac": mean(m["good_long_frac"] for m in ms),
            "good_short_frac": mean(m["good_short_frac"] for m in ms),
            "goodput_tok_s": mean(m["goodput_tok_s"] for m in ms),
            "badput_tok_s": mean(m["badput_tok_s"] for m in ms),
            "throughput_req_s": mean(m["throughput_req_s"] for m in ms),
            "e2e_p95_all_s": mean(m["e2e_p95_all_s"] for m in ms),
            "link_GB_hops": mean(m["link_GB_hops"] for m in ms),
            "energy_kj": mean(m["energy_kj"] for m in ms),
            "n_migrations": mean(m["n_migrations"] for m in ms),
            "goodput_sd": stdev([m["goodput_tok_s"] for m in ms]) if len(ms) > 1 else 0.0,
        })
    return rows


def main() -> None:
    out = HERE / "out"
    l1_k1k2 = json.loads((out / "l1_k1k2.json").read_text())
    l1_k3 = json.loads((out / "l1_k3.json").read_text())
    l1_ora = json.loads((out / "l1_oracle.json").read_text())
    l2_6 = json.loads((out / "l2_val_6sat.json").read_text())
    l2_24 = json.loads((out / "l2_val_24sat.json").read_text())

    lines: list[str] = ["# LAB-2082 Summary Tables (from raw JSON)", ""]

    # 1. L1 K1xK2 Table
    lines.append("## L1 · K1 × K2 (6-sat, dev seeds 101–103)")
    lines.append("| Ratio | Spare | PostK | Arm | Long SLO(%) ↑ | Short SLO(%) ↑ | Goodput(tok/s) ↑ | Badput(tok/s) ↓ | Link(GB·hop) ↓ | Energy(kJ) ↓ | Migs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    arms_k1 = ["Static", "Queue-Reroute", "Dyn-Zero", "Dyn-CF", "Dyn-Sh-Naive"]
    groups = sorted({(c["ratio"], c["spare"], c["post_k"]) for c in l1_k1k2})
    for ratio, spare, post_k in groups:
        sub = [c for c in l1_k1k2 if (c["ratio"], c["spare"], c["post_k"]) == (ratio, spare, post_k)]
        srows = summarize_group(sub, arms_k1)
        base = srows[0]
        for r in srows:
            is_b = r["arm"] == "Static"
            ls = f"{r['good_long_frac']*100:.1f}%" if is_b else _fmt_pct_delta(r["good_long_frac"], base["good_long_frac"], pp=True)
            ss = f"{r['good_short_frac']*100:.1f}%" if is_b else _fmt_pct_delta(r["good_short_frac"], base["good_short_frac"], pp=True)
            gp = f"{r['goodput_tok_s']:.1f}" if is_b else _fmt_pct_delta(r["goodput_tok_s"], base["goodput_tok_s"])
            bp = f"{r['badput_tok_s']:.1f}" if is_b else _fmt_pct_delta(r["badput_tok_s"], base["badput_tok_s"])
            lines.append(
                f"| {ratio}x | {spare} | {post_k} | {r['arm']} | {ls} | {ss} | {gp} | {bp} | {r['link_GB_hops']:.1f} | {r['energy_kj']:.1f} | {r['n_migrations']:.1f} |"
            )

    # 2. L1 K3 Table
    lines.append("")
    lines.append(f"## L1 · K3 (6-sat, ratio={l1_k3['chosen_ratio']}x, dev seeds 101–103)")
    lines.append("| BG Frac | Pattern | KV Scale | MaxInflight | Arm | Long SLO(%) ↑ | Goodput(tok/s) ↑ | Link(GB·hop) ↓ | Migs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    k3_cells = l1_k3["cells"]
    k3_groups = sorted({(c["bg_frac"], c["pattern"], c["kv_scale"], c["max_inflight"]) for c in k3_cells})
    for bg_f, pat, kv_s, mi in k3_groups:
        sub = [c for c in k3_cells if (c["bg_frac"], c["pattern"], c["kv_scale"], c["max_inflight"]) == (bg_f, pat, kv_s, mi)]
        srows = summarize_group(sub, ["Dyn-CF", "Dyn-Sh-Naive", "Dyn-Sh-Aware"])
        base = next((x for x in srows if x["arm"] == "Dyn-Sh-Naive"), srows[0])
        for r in srows:
            ls = f"{r['good_long_frac']*100:.1f}%" if r["arm"] == "Dyn-Sh-Naive" else _fmt_pct_delta(r["good_long_frac"], base["good_long_frac"], pp=True)
            gp = f"{r['goodput_tok_s']:.1f}" if r["arm"] == "Dyn-Sh-Naive" else _fmt_pct_delta(r["goodput_tok_s"], base["goodput_tok_s"])
            lines.append(f"| {bg_f} | {pat} | {kv_s}x | {mi} | {r['arm']} | {ls} | {gp} | {r['link_GB_hops']:.1f} | {r['n_migrations']:.1f} |")

    # 3. L2 Validation Tables
    for label, dataset in [("L2 · 6-sat Paired Validation (seeds 142–146, 2x)", l2_6), ("L2 · 24-sat Trend Check (seeds 142–146, 2x)", l2_24)]:
        lines.append("")
        lines.append(f"## {label}")
        lines.append("| Arm | Long SLO(%) ↑ | Short SLO(%) ↑ | Goodput(tok/s) ↑ | Badput(tok/s) ↓ | Req/s ↑ | P95 E2E(s) ↓ | Link(GB·hop) ↓ | Energy(kJ) ↓ |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        arms_v = ["Static", "Queue-Reroute", "Dyn-Zero", "Dyn-CF", "Dyn-Sh-Naive", "Dyn-Sh-Aware"]
        srows = summarize_group(dataset, arms_v)
        base = srows[0]
        for r in srows:
            is_b = r["arm"] == "Static"
            ls = f"{r['good_long_frac']*100:.1f}%" if is_b else _fmt_pct_delta(r["good_long_frac"], base["good_long_frac"], pp=True)
            ss = f"{r['good_short_frac']*100:.1f}%" if is_b else _fmt_pct_delta(r["good_short_frac"], base["good_short_frac"], pp=True)
            gp = f"{r['goodput_tok_s']:.1f}" if is_b else _fmt_pct_delta(r["goodput_tok_s"], base["goodput_tok_s"])
            bp = f"{r['badput_tok_s']:.1f}" if is_b else _fmt_pct_delta(r["badput_tok_s"], base["badput_tok_s"])
            lines.append(
                f"| {r['arm']} | {ls} | {ss} | {gp} | {bp} | {r['throughput_req_s']:.2f} | {r['e2e_p95_all_s']:.1f} | {r['link_GB_hops']:.1f} | {r['energy_kj']:.1f} |"
            )

    (out / "summary_tables.md").write_text("\n".join(lines) + "\n")

    # Plot 1: Gap Waterfall (6-sat & 24-sat L2 Goodput & Long SLO)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    arms_v = ["Static", "Queue-Reroute", "Dyn-Zero", "Dyn-CF", "Dyn-Sh-Naive", "Dyn-Sh-Aware"]
    for ax, dataset, title in [(axes[0], l2_6, "6-Sat Validation (2x, seeds 142-146)"), (axes[1], l2_24, "24-Sat Validation (2x, seeds 142-146)")]:
        srows = summarize_group(dataset, arms_v)
        vals = [r["goodput_tok_s"] for r in srows]
        colors = ["#6c757d", "#adb5bd", "#2a9d8f", "#457b9d", "#e76f51", "#1d3557"]
        bars = ax.bar(range(len(arms_v)), vals, color=colors)
        ax.set_xticks(range(len(arms_v)))
        ax.set_xticklabels(arms_v, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Goodput (tok/s) ↑")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "gap_waterfall.png", dpi=160)
    plt.close()

    # Plot 2: CDF (6-sat and 24-sat, all-arrival denominator)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dataset, title in [(axes[0], l2_6, "6-Sat Long Req CDF (All Arrivals)"), (axes[1], l2_24, "24-Sat Long Req CDF (All Arrivals)")]:
        for arm, ls in [("Static", "--"), ("Dyn-Zero", ":"), ("Dyn-CF", "-."), ("Dyn-Sh-Naive", "-"), ("Dyn-Sh-Aware", "-")]:
            lats = []
            total = 0
            for c in dataset:
                for r in c["requests_by_arm"][arm]:
                    if r["long"]:
                        total += 1
                        if r["good"] and r["t_delivered"] is not None:
                            lats.append(r["t_delivered"] - r["t"])
            lats.sort()
            xs = [0.0] + lats + [320.0]
            ys = [0.0] + [i / max(1, total) for i in range(1, len(lats) + 1)] + [len(lats) / max(1, total)]
            ax.step(xs, ys, where="post", linestyle=ls, label=arm)
        ax.set_xlabel("E2E Latency (s) ↓")
        ax.set_ylabel("On-Time Completion Fraction ↑")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "cdf_val.png", dpi=160)
    plt.close()
    print("Plots and summary_tables.md generated.", flush=True)


if __name__ == "__main__":
    main()
