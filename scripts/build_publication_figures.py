#!/usr/bin/env python3
"""Build publication-quality figures for the cross-DGP local-law paper.

Produces:
  Figure 1: Cross-DGP comparison table (Markov vs LDA)
  Figure 2: Weight ablation heatmap and bar chart
  Figure 3: Unified learning curve (primary gain vs support budget)

Usage:
    python scripts/build_publication_figures.py \
        --cross-dgp-summary outputs/.../cross_dgp_law_stress_summary.json \
        --ablation-summary outputs/.../weight_ablation_summary.json \
        --output-dir outputs/publication_figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})


def _fig1_cross_dgp_table(summary: Dict[str, Any], output_dir: Path) -> None:
    """Figure 1: Cross-DGP comparison as a formatted table figure."""
    rows = summary.get("rows", [])
    if not rows:
        print("  No rows in cross-DGP summary, skipping Figure 1")
        return

    # Filter to key packages
    key_packages = {"all_laws", "c2_only", "all_laws_plus_sched"}
    filtered = [r for r in rows if r.get("law_package") in key_packages or r.get("dgp") == "tree_relevant_lda_local_law"]

    fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(filtered) + 1.5)))
    ax.axis("off")

    col_labels = ["DGP", "Package", "N", "Primary\nPass %", "C1\nPass %", "C2\nPass %", "C3\nPass %", "Mean\nGain %"]
    cell_data = []
    cell_colors = []

    for r in rows:
        dgp = str(r.get("dgp", "?"))
        if "lda" in dgp.lower():
            dgp_short = "LDA"
        elif "markov" in dgp.lower():
            dgp_short = "Markov"
        else:
            dgp_short = dgp[:15]

        pkg = str(r.get("law_package", "?"))
        n = int(r.get("n_runs", 0))
        prim = float(r.get("primary_pass_rate", 0)) * 100
        c1 = float(r.get("c1_pass_rate", 0)) * 100
        c2 = float(r.get("c2_pass_rate", 0)) * 100
        c3 = float(r.get("c3_pass_rate", 0)) * 100
        gain = float(r.get("mean_primary_gain", 0)) * 100

        row_data = [dgp_short, pkg, str(n), f"{prim:.1f}", f"{c1:.0f}", f"{c2:.0f}", f"{c3:.0f}", f"{gain:+.1f}"]
        cell_data.append(row_data)

        # Color: green for positive gain, red for negative
        base = "#ffffff"
        if gain > 3:
            base = "#d5f5d5"
        elif gain < -3:
            base = "#f5d5d5"
        cell_colors.append([base] * len(col_labels))

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e0e0e0"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    ax.set_title("Cross-DGP Local-Law Stress Comparison", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig1_cross_dgp_table.pdf"), bbox_inches="tight")
    fig.savefig(str(output_dir / "fig1_cross_dgp_table.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote fig1_cross_dgp_table.pdf/png")


def _fig2_weight_ablation(summary: Dict[str, Any], output_dir: Path) -> None:
    """Figure 2: Weight ablation results."""
    matched = summary.get("matched_summaries", [])
    if not matched:
        print("  No matched summaries in ablation, skipping Figure 2")
        return

    profiles = [s["profile"] for s in matched]
    gains = [s["mean_gain_pct"] for s in matched]
    pass_rates = [s["primary_pass_rate"] * 100 for s in matched]
    n_matched = [s["n_matched"] for s in matched]

    # Pretty labels
    pretty_labels = {
        "pure_c2": "C₂ only\n(0,1,0)",
        "c2_trace_c1c3": "C₂+trace\n(.05,1,.05)",
        "c2_light_c1c3": "C₂+light\n(.1,1,.1)",
        "c2_mild_c1c3": "C₂+mild\n(.25,1,.25)",
        "c2_moderate_c1c3": "C₂+mod\n(.5,1,.5)",
        "c2_very_dominant": "C₂ 8×\n(1,8,1)",
        "c2_dominant": "C₂ 4×\n(1,4,1)",
        "c2_heavy": "C₂ 2×\n(1,2,1)",
        "equal": "Equal\n(1,1,1)",
        "c1c3_heavy": "C₁C₃ 2×\n(2,1,2)",
        "c3_dominant": "C₃ 4×\n(1,1,4)",
        "no_c2": "No C₂\n(1,0,4)",
    }
    labels = [pretty_labels.get(p, p) for p in profiles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Mean gain
    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gains]
    bars = ax1.bar(range(len(profiles)), gains, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5, width=0.7)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax1.set_xticks(range(len(profiles)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Mean Primary Gain (%)")
    ax1.set_title("(a) Matched Primary Gain vs Root-Only Baseline")
    for bar, g, n in zip(bars, gains, n_matched):
        y = bar.get_height()
        offset = 0.8 if g >= 0 else -1.5
        ax1.text(bar.get_x() + bar.get_width()/2, y + offset,
                f"{g:+.1f}%\n(n={n})", ha="center", va="bottom" if g >= 0 else "top", fontsize=7)

    # Right: Pass rate
    bars2 = ax2.bar(range(len(profiles)), pass_rates, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5, width=0.7)
    ax2.set_xticks(range(len(profiles)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Primary Pass Rate (%)")
    ax2.set_title("(b) Fraction of Configs Where Learned Beats Baseline")
    ax2.set_ylim(0, 110)
    for bar, pr in zip(bars2, pass_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pr:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Markov C₁/C₂/C₃ Weight Ablation: Impact on Root Prediction", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig2_weight_ablation.pdf"), bbox_inches="tight")
    fig.savefig(str(output_dir / "fig2_weight_ablation.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote fig2_weight_ablation.pdf/png")


def _fig3_cross_dgp_bars(summary: Dict[str, Any], output_dir: Path) -> None:
    """Figure 3: Cross-DGP side-by-side comparison bar chart."""
    rows = summary.get("rows", [])
    if not rows:
        return

    # Group by DGP
    dgp_data = {}
    for r in rows:
        dgp = "Markov" if "markov" in str(r.get("dgp", "")).lower() else "LDA"
        pkg = str(r.get("law_package", "?"))
        key = f"{dgp}\n{pkg}"
        dgp_data[key] = {
            "gain": float(r.get("mean_primary_gain", 0)) * 100,
            "pass_rate": float(r.get("primary_pass_rate", 0)) * 100,
            "n": int(r.get("n_runs", 0)),
            "dgp": dgp,
        }

    keys = sorted(dgp_data.keys(), key=lambda k: (dgp_data[k]["dgp"], -dgp_data[k]["gain"]))
    gains = [dgp_data[k]["gain"] for k in keys]
    ns = [dgp_data[k]["n"] for k in keys]
    dgps = [dgp_data[k]["dgp"] for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.8), 5))
    colors = []
    for k in keys:
        d = dgp_data[k]
        if d["dgp"] == "Markov":
            colors.append("#1f77b4" if d["gain"] > 0 else "#aec7e8")
        else:
            colors.append("#ff7f0e" if d["gain"] > 0 else "#ffbb78")

    bars = ax.bar(range(len(keys)), gains, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean Primary Gain (%)")
    ax.set_title("Cross-DGP Law-Stress Results: Learned vs Baseline", fontsize=12, fontweight="bold")

    for bar, g, n in zip(bars, gains, ns):
        y = bar.get_height()
        offset = 0.5 if g >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width()/2, y + offset,
                f"{g:+.1f}%\n(n={n})", ha="center", va="bottom" if g >= 0 else "top", fontsize=6)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Markov (positive)"),
        Patch(facecolor="#aec7e8", edgecolor="black", label="Markov (negative)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="LDA (positive)"),
        Patch(facecolor="#ffbb78", edgecolor="black", label="LDA (negative)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig3_cross_dgp_bars.pdf"), bbox_inches="tight")
    fig.savefig(str(output_dir / "fig3_cross_dgp_bars.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote fig3_cross_dgp_bars.pdf/png")


def _fig4_c1c3_threshold(summary: Dict[str, Any], output_dir: Path) -> None:
    """Figure 4: C1/C3 tolerance threshold curve.

    Plots mean gain vs C1/C3 relative weight for the fine-grid profiles
    (pure_c2, c2_trace, c2_light, c2_mild, c2_moderate, equal).
    Shows the sharp drop-off from pure C2 as C1/C3 contamination increases.
    """
    matched = summary.get("matched_summaries", [])
    if not matched:
        return

    # Profiles with symmetric C1=C3 weights, ordered by C1/C3 strength
    threshold_profiles = [
        ("pure_c2", 0.0),
        ("c2_trace_c1c3", 0.05),
        ("c2_light_c1c3", 0.1),
        ("c2_mild_c1c3", 0.25),
        ("c2_moderate_c1c3", 0.5),
        ("equal", 1.0),
    ]
    profile_map = {s["profile"]: s for s in matched}

    xs, gains, pass_rates, ns = [], [], [], []
    for pname, c1c3_weight in threshold_profiles:
        s = profile_map.get(pname)
        if s is None:
            continue
        xs.append(c1c3_weight)
        gains.append(s["mean_gain_pct"])
        pass_rates.append(s["primary_pass_rate"] * 100)
        ns.append(s["n_matched"])

    if len(xs) < 3:
        print("  Not enough threshold profiles for Figure 4, skipping")
        return

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twinx()

    # Gain curve
    ln1 = ax1.plot(xs, gains, "o-", color="#2ca02c", linewidth=2.0,
                   markersize=7, label="Mean Gain (%)", zorder=3)
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.fill_between(xs, 0, gains, where=[g > 0 for g in gains],
                     color="#2ca02c", alpha=0.15, interpolate=True)
    ax1.fill_between(xs, 0, gains, where=[g <= 0 for g in gains],
                     color="#d62728", alpha=0.15, interpolate=True)

    # Pass rate on right axis
    ln2 = ax2.plot(xs, pass_rates, "s--", color="steelblue", linewidth=1.5,
                   markersize=6, alpha=0.7, label="Pass Rate (%)")
    ax2.set_ylim(-5, 110)
    ax2.set_ylabel("Primary Pass Rate (%)", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    # Annotate points
    for x, g, n in zip(xs, gains, ns):
        ax1.annotate(f"{g:+.1f}%", (x, g), textcoords="offset points",
                     xytext=(8, 8 if g > 0 else -14), fontsize=8, color="#333333")

    ax1.set_xlabel("C₁/C₃ Relative Weight (symmetric)")
    ax1.set_ylabel("Mean Primary Gain (%)")
    ax1.set_title("C₁/C₃ Tolerance Threshold:\nGain Collapses with Any C₁/C₃ Inclusion",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig4_c1c3_threshold.pdf"), bbox_inches="tight")
    fig.savefig(str(output_dir / "fig4_c1c3_threshold.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote fig4_c1c3_threshold.pdf/png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication figures")
    parser.add_argument("--cross-dgp-summary", type=str, default="")
    parser.add_argument("--ablation-summary", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cross_dgp_summary and Path(args.cross_dgp_summary).exists():
        print("Building Figure 1: Cross-DGP table...")
        cross_dgp = json.loads(Path(args.cross_dgp_summary).read_text(encoding="utf-8"))
        _fig1_cross_dgp_table(cross_dgp, output_dir)
        print("Building Figure 3: Cross-DGP bars...")
        _fig3_cross_dgp_bars(cross_dgp, output_dir)
    else:
        print("Skipping cross-DGP figures (no summary provided)")

    if args.ablation_summary and Path(args.ablation_summary).exists():
        print("Building Figure 2: Weight ablation...")
        ablation = json.loads(Path(args.ablation_summary).read_text(encoding="utf-8"))
        _fig2_weight_ablation(ablation, output_dir)
        print("Building Figure 4: C1/C3 threshold curve...")
        _fig4_c1c3_threshold(ablation, output_dir)
    else:
        print("Skipping ablation figures (no summary provided)")

    print(f"\nAll figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
