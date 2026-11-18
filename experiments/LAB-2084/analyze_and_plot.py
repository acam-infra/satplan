"""Analyze LAB-2084 raw JSON outputs and generate summary tables + matplotlib plots."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SPEC = json.loads((HERE / "spec.json").read_text())


def _mean_std_ci(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(n)
    return m, sd, ci


def _agg_cells(cells: list[dict[str, Any]], arm: str) -> dict[str, tuple[float, float, float]]:
    keys = [
        "good_long_frac",
        "good_short_frac",
        "goodput_tok_s",
        "badput_tok_s",
        "throughput_req_s",
        "throughput_tok_s_horizon",
        "link_GB_hops",
        "energy_kj",
        "n_migrations",
        "thermal_violations",
        "dod_violations",
        "peak_temp_C",
        "dod_exhausted_s",
        "actual_admitted",
    ]
    out: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        vals = [float(c["arms"][arm]["metrics"][k]) for c in cells]
        out[k] = _mean_std_ci(vals)
    return out


def _paired_deltas(cells: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for k in ("good_long_frac", "good_short_frac", "goodput_tok_s", "badput_tok_s", "throughput_req_s", "throughput_tok_s_horizon", "link_GB_hops", "energy_kj"):
        diffs = [
            float(c["arms"]["Dyn-CF"]["metrics"][k]) - float(c["arms"]["Static"]["metrics"][k])
            for c in cells
        ]
        out[k] = _mean_std_ci(diffs)
    return out


def plot_cdfs(val_6sat: list[dict[str, Any]], out_path: Path) -> None:
    panels = [
        ("mod_1.0x", "Moderate 1.0x (dl=+255s)", 255.0),
        ("mod_2.0x", "Moderate 2.0x Underest (dl=+255s)", 255.0),
        ("hp_2.0x", "High-Press 2.0x (dl=+195s)", 195.0),
        ("mod_2.0x_no_contact", "NegCtrl: No Contact (dl=+255s)", 255.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True, sharey=True)
    ts = np.linspace(0.0, 320.0, 321)

    for ax, (sc_id, title, dl_long) in zip(axes.flat, panels, strict=True):
        cells = [c for c in val_6sat if c["scenario_id"] == sc_id]
        for arm, color, ls in [("Static", "#374151", "-"), ("Dyn-CF", "#0284c7", "-")]:
            for is_long, lbl_suffix, alpha, lw in [(True, "Long", 1.0, 2.2), (False, "Short", 0.55, 1.5)]:
                reqs = [r for c in cells for r in c["requests_by_arm"][arm] if r["long"] == is_long]
                n_tot = max(1, len(reqs))
                # Completion time relative to arrival (includes late completions <= horizon; unfinished stay < 100%)
                lats = [r["t_delivered"] - r["t"] for r in reqs if r["t_delivered"] is not None]
                ys = [sum(1 for x in lats if x <= t) / n_tot * 100.0 for t in ts]
                style = ls if is_long else ":"
                ax.plot(ts, ys, linestyle=style, color=color, alpha=alpha, linewidth=lw, label=f"{arm} {lbl_suffix}")
        ax.axvline(45.0, color="#9ca3af", linestyle="--", linewidth=1.0, label="Short DL (+45s)")
        ax.axvline(dl_long, color="#dc2626", linestyle="--", linewidth=1.1, label=f"Long DL (+{int(dl_long)}s)")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylim(-2, 105)
        ax.set_xlim(0, 320)
        ax.grid(alpha=0.25)

    axes[1, 0].set_xlabel("Time Since Arrival (s) ↑ earlier is better")
    axes[1, 1].set_xlabel("Time Since Arrival (s) ↑ earlier is better")
    axes[0, 0].set_ylabel("Cumulative Completed (%) ↑")
    axes[1, 0].set_ylabel("Cumulative Completed (%) ↑")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=True)
    fig.suptitle(
        "LAB-2084 · Long & Short Request Completion Curves (Val Seeds 142–146, 6-sat)\n"
        "(Includes late completions up to 320s horizon; unfinished requests remain below 100%)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_paired_deltas(val_6sat: list[dict[str, Any]], out_path: Path) -> None:
    sc_ids = [sc["id"] for sc in SPEC["budget_grid_6sat"]]
    labels = [
        "mod_1.0x",
        "mod_1.5x",
        "mod_2.0x",
        "mod_1.0x_post4",
        "mod_2.0x_no_contact",
        "mod_2.0x_no_spare",
        "hp_1.0x",
        "hp_2.0x",
        "hp_2.0x_no_contact",
    ]
    long_means, long_cis = [], []
    gp_means, gp_cis = [], []
    for sc_id in sc_ids:
        cells = [c for c in val_6sat if c["scenario_id"] == sc_id]
        dl = [ (c["arms"]["Dyn-CF"]["metrics"]["good_long_frac"] - c["arms"]["Static"]["metrics"]["good_long_frac"]) * 100.0 for c in cells ]
        dg = [
            ((c["arms"]["Dyn-CF"]["metrics"]["goodput_tok_s"] - c["arms"]["Static"]["metrics"]["goodput_tok_s"]) / max(1e-9, c["arms"]["Static"]["metrics"]["goodput_tok_s"])) * 100.0
            for c in cells
        ]
        m_l, _, ci_l = _mean_std_ci(dl)
        m_g, _, ci_g = _mean_std_ci(dg)
        long_means.append(m_l)
        long_cis.append(ci_l)
        gp_means.append(m_g)
        gp_cis.append(ci_g)

    y = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    colors = ["#dc2626" if "no_" in s else "#0284c7" for s in labels]
    ax1.barh(y, long_means, xerr=long_cis, color=colors, alpha=0.88, capsize=3)
    ax1.axvline(10.0, color="#16a34a", linestyle="--", linewidth=1.2, label="Ref +10 pp")
    ax1.axvline(0.0, color="#374151", linewidth=0.8)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Paired Δ Long SLO (pp) ↑ (mean ± 95% CI)")
    ax1.set_title("Long SLO Gain (Dyn-CF − Static)")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="x", alpha=0.25)

    ax2.barh(y, gp_means, xerr=gp_cis, color=colors, alpha=0.88, capsize=3)
    ax2.axvline(20.0, color="#16a34a", linestyle="--", linewidth=1.2, label="Ref +20%")
    ax2.axvline(0.0, color="#374151", linewidth=0.8)
    ax2.set_xlabel("Paired Δ Goodput (%) ↑ (mean ± 95% CI)")
    ax2.set_title("Goodput Relative Gain (Dyn-CF vs Static)")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("LAB-2084 · Paired Seed Differences Across Budget, Perturbation & Negative Controls (n=5)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    anchor = json.loads((OUT / "anchor_repro.json").read_text())
    dev = json.loads((OUT / "dev_6sat.json").read_text())
    val_6 = json.loads((OUT / "val_6sat.json").read_text())
    val_24 = json.loads((OUT / "val_24sat.json").read_text())

    plot_cdfs(val_6, OUT / "cdf_long_short_val.png")
    plot_paired_deltas(val_6, OUT / "paired_delta_grid.png")

    lines: list[str] = ["# LAB-2084 Summary Tables (from raw JSON)\n"]

    # 1. Anchor reproduction
    lines.append("## 1. Anchor Reproduction (LAB-2082 Parity Check)\n")
    lines.append("| Anchor | Arm | Long SLO(%) ↑ | Short SLO(%) ↑ | Goodput(tok/s) ↑ | Badput(tok/s) ↓ | Link(GB·hop) ↓ | Energy(kJ) ↓ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    s6 = anchor["sat6"]
    for arm in SPEC["arms"]:
        m = s6["arms"][arm]["metrics"]
        lines.append(
            f"| 6-sat s142 (2.0x) | {arm} | {m['good_long_frac']*100:.1f}% | {m['good_short_frac']*100:.1f}% | "
            f"{m['goodput_tok_s']:.1f} | {m['badput_tok_s']:.1f} | {m['link_GB_hops']:.1f} | {m['energy_kj']:.1f} |"
        )
    for arm in SPEC["arms"]:
        ag = _agg_cells(anchor["sat24"], arm)
        lines.append(
            f"| 24-sat s142–146 (2.0x) | {arm} | {ag['good_long_frac'][0]*100:.1f}% | {ag['good_short_frac'][0]*100:.1f}% | "
            f"{ag['goodput_tok_s'][0]:.1f} | {ag['badput_tok_s'][0]:.1f} | {ag['link_GB_hops'][0]:.1f} | {ag['energy_kj'][0]:.1f} |"
        )

    # 2. Held-out 6-sat Validation (seeds 142-146)
    lines.append("\n## 2. Held-Out 6-sat Paired Validation (Val Seeds 142–146, n=5)\n")
    lines.append(
        "| Scenario | Arm | Long SLO(%) ↑ | Short SLO(%) ↑ | Goodput(tok/s) ↑ | Tput(req/s,25s) ↑ | Tput(tok/s,320s) ↑ | "
        "Badput(tok/s) ↓ | Link(GB·hop) ↓ | Energy(kJ) ↓ | Therm/DoD Viol ↓ | Δ Long(pp,95%CI) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in SPEC["budget_grid_6sat"]:
        sc_id = sc["id"]
        cells = [c for c in val_6 if c["scenario_id"] == sc_id]
        st = _agg_cells(cells, "Static")
        cf = _agg_cells(cells, "Dyn-CF")
        pd = _paired_deltas(cells)
        dl_m, dl_sd, dl_ci = pd["good_long_frac"]
        gp_rel = (cf["goodput_tok_s"][0] - st["goodput_tok_s"][0]) / max(1e-9, st["goodput_tok_s"][0]) * 100.0
        for arm, ag in [("Static", st), ("Dyn-CF", cf)]:
            viol = ag["thermal_violations"][0] + ag["dod_violations"][0]
            if arm == "Static":
                lines.append(
                    f"| {sc_id} | {arm} | {ag['good_long_frac'][0]*100:.1f}% | {ag['good_short_frac'][0]*100:.1f}% | "
                    f"{ag['goodput_tok_s'][0]:.1f} | {ag['throughput_req_s'][0]:.2f} | {ag['throughput_tok_s_horizon'][0]:.1f} | "
                    f"{ag['badput_tok_s'][0]:.1f} | {ag['link_GB_hops'][0]:.1f} | {ag['energy_kj'][0]:.1f} | {viol:.0f} | — |"
                )
            else:
                lines.append(
                    f"| {sc_id} | **{arm}** | **{ag['good_long_frac'][0]*100:.1f}%** | {ag['good_short_frac'][0]*100:.1f}% | "
                    f"**{ag['goodput_tok_s'][0]:.1f}** ({gp_rel:+.1f}%) | {ag['throughput_req_s'][0]:.2f} | {ag['throughput_tok_s_horizon'][0]:.1f} | "
                    f"{ag['badput_tok_s'][0]:.1f} | {ag['link_GB_hops'][0]:.1f} | {ag['energy_kj'][0]:.1f} | {viol:.0f} | "
                    f"{dl_m*100:+.1f} ± {dl_ci*100:.1f} |"
                )

    # 3. 24-sat Direction Check
    lines.append("\n## 3. 24-sat Direction Check (Val Seeds 142–146, n=5)\n")
    lines.append("| Scenario | Arm | Long SLO(%) ↑ | Short SLO(%) ↑ | Goodput(tok/s) ↑ | Tput(req/s,25s) ↑ | Badput(tok/s) ↓ | Link(GB·hop) ↓ | Energy(kJ) ↓ | Δ Long(pp,95%CI) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for sc in SPEC["direction_24sat"]:
        sc_id = sc["id"]
        cells = [c for c in val_24 if c["scenario_id"] == sc_id]
        st = _agg_cells(cells, "Static")
        cf = _agg_cells(cells, "Dyn-CF")
        pd = _paired_deltas(cells)
        dl_m, _, dl_ci = pd["good_long_frac"]
        gp_rel = (cf["goodput_tok_s"][0] - st["goodput_tok_s"][0]) / max(1e-9, st["goodput_tok_s"][0]) * 100.0
        for arm, ag in [("Static", st), ("Dyn-CF", cf)]:
            if arm == "Static":
                lines.append(
                    f"| {sc_id} | {arm} | {ag['good_long_frac'][0]*100:.1f}% | {ag['good_short_frac'][0]*100:.1f}% | "
                    f"{ag['goodput_tok_s'][0]:.1f} | {ag['throughput_req_s'][0]:.2f} | {ag['badput_tok_s'][0]:.1f} | "
                    f"{ag['link_GB_hops'][0]:.1f} | {ag['energy_kj'][0]:.1f} | — |"
                )
            else:
                lines.append(
                    f"| {sc_id} | **{arm}** | **{ag['good_long_frac'][0]*100:.1f}%** | {ag['good_short_frac'][0]*100:.1f}% | "
                    f"**{ag['goodput_tok_s'][0]:.1f}** ({gp_rel:+.1f}%) | {ag['throughput_req_s'][0]:.2f} | {ag['badput_tok_s'][0]:.1f} | "
                    f"{ag['link_GB_hops'][0]:.1f} | {ag['energy_kj'][0]:.1f} | {dl_m*100:+.1f} ± {dl_ci*100:.1f} |"
                )

    (OUT / "summary_tables.md").write_text("\n".join(lines) + "\n")
    print("Wrote summary_tables.md and plots.")


if __name__ == "__main__":
    main()
