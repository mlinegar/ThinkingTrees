#!/usr/bin/env python3
"""Plot expanded IPW propensity/coverage diagnostics from stress-ladder summary rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot expanded IPW diagnostics from run_ipw_stress_ladder summary CSV.")
    p.add_argument(
        "--input-csv",
        type=str,
        default="outputs/ipw_stress_ladder_hard_large_20260302_183753/summary_rows.csv",
    )
    p.add_argument("--metric", choices=["violation", "preference"], default="violation")
    p.add_argument("--target-coverage", type=float, default=0.90)
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/ipw_propensity_diagnostics.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/ipw_propensity_diagnostics_report.json",
    )
    return p.parse_args()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def _f(row: Dict[str, str], k: str) -> float:
    return float(row[k])


def _i(row: Dict[str, str], k: str) -> int:
    return int(float(row[k]))


def main() -> int:
    args = _parse_args()
    metric = str(args.metric)
    in_csv = Path(args.input_csv)
    rows = _load_rows(in_csv)
    if not rows:
        raise ValueError(f"No rows in input CSV: {in_csv}")

    cov_key = f"{metric}_coverage_mean"
    bias_key = f"ipw_{metric}_bias_mean"
    width_key = f"{metric}_mean_width_mean"
    naive_width_key = f"naive_{metric}_mean_width_mean"

    cases = sorted({str(r["case"]) for r in rows})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(2, len(cases))))
    color_map = {c: colors[idx % len(colors)] for idx, c in enumerate(cases)}

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.6), constrained_layout=True)
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]

    # Panel A: coverage gap vs max weight.
    for c in cases:
        grp = [r for r in rows if str(r["case"]) == c]
        x = np.asarray([_f(r, "max_joint_weight_mean") for r in grp], dtype=np.float64)
        y = np.asarray([_f(r, cov_key) - float(args.target_coverage) for r in grp], dtype=np.float64)
        s = np.asarray([max(20.0, _f(r, "high_signal_low_propensity_overlap_mean") * 240.0) for r in grp], dtype=np.float64)
        ax0.scatter(x, y, s=s, alpha=0.75, color=color_map[c], label=c)
    ax0.axhline(0.0, color="#222222", linestyle=":", linewidth=1.2)
    ax0.set_xlabel("max_joint_weight_mean")
    ax0.set_ylabel(f"{metric}_coverage_mean - target")
    ax0.set_title("Coverage Deviation vs Weight Extremity")
    ax0.grid(alpha=0.25)

    # Panel B: IPW bias vs high-signal overlap.
    for c in cases:
        grp = [r for r in rows if str(r["case"]) == c]
        x = np.asarray([_f(r, "high_signal_low_propensity_overlap_mean") for r in grp], dtype=np.float64)
        y = np.asarray([_f(r, bias_key) for r in grp], dtype=np.float64)
        ax1.scatter(x, y, s=42, alpha=0.8, color=color_map[c], label=c)
    ax1.axhline(0.0, color="#222222", linestyle=":", linewidth=1.2)
    ax1.set_xlabel("high_signal_low_propensity_overlap_mean")
    ax1.set_ylabel(f"ipw_{metric}_bias_mean")
    ax1.set_title("IPW Bias vs Adversarial Overlap")
    ax1.grid(alpha=0.25)

    # Panel C: effective sample size vs n_docs.
    for c in cases:
        grp = sorted((r for r in rows if str(r["case"]) == c), key=lambda rr: _i(rr, "n_docs"))
        xs = [_i(r, "n_docs") for r in grp]
        ys = [_f(r, "mean_effective_sample_size_mean") for r in grp]
        ax2.plot(xs, ys, marker="o", linewidth=1.8, color=color_map[c], label=c)
    ax2.set_xscale("log")
    ax2.set_xlabel("n_docs")
    ax2.set_ylabel("mean_effective_sample_size_mean")
    ax2.set_title("Effective Sample Size Scaling")
    ax2.grid(alpha=0.25)

    # Panel D: width ratio (IPW / Naive) vs n_docs.
    for c in cases:
        grp = sorted((r for r in rows if str(r["case"]) == c), key=lambda rr: _i(rr, "n_docs"))
        xs = np.asarray([_i(r, "n_docs") for r in grp], dtype=np.float64)
        ipw_w = np.asarray([_f(r, width_key) for r in grp], dtype=np.float64)
        naive_w = np.asarray([_f(r, naive_width_key) for r in grp], dtype=np.float64)
        ratio = np.where(np.isfinite(naive_w) & (naive_w > 0), ipw_w / naive_w, np.nan)
        ax3.plot(xs, ratio, marker="o", linewidth=1.8, color=color_map[c], label=c)
    ax3.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2)
    ax3.set_xscale("log")
    ax3.set_xlabel("n_docs")
    ax3.set_ylabel("IPW width / naive width")
    ax3.set_title("Interval Width Efficiency")
    ax3.grid(alpha=0.25)

    # Global legend
    handles: List[object] = []
    labels: List[str] = []
    for ax in (ax0, ax1, ax2, ax3):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    dedup: Dict[str, object] = {}
    for h, l in zip(handles, labels):
        if l not in dedup:
            dedup[l] = h
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=3, fontsize=8, frameon=False)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    # Compact diagnostics for report tables.
    summary_rows: List[Dict[str, float | str | int]] = []
    for c in cases:
        grp = sorted((r for r in rows if str(r["case"]) == c), key=lambda rr: _i(rr, "n_docs"))
        if not grp:
            continue
        first, last = grp[0], grp[-1]
        summary_rows.append(
            {
                "case": c,
                "n_docs_min": _i(first, "n_docs"),
                "n_docs_max": _i(last, "n_docs"),
                "coverage_min_docs": _f(first, cov_key),
                "coverage_max_docs": _f(last, cov_key),
                "bias_max_docs": _f(last, bias_key),
                "max_weight_max_docs": _f(last, "max_joint_weight_mean"),
                "overlap_max_docs": _f(last, "high_signal_low_propensity_overlap_mean"),
                "neff_max_docs": _f(last, "mean_effective_sample_size_mean"),
            }
        )

    report = {
        "input_csv": str(in_csv),
        "metric": metric,
        "target_coverage": float(args.target_coverage),
        "n_rows": int(len(rows)),
        "cases": cases,
        "summary_rows": summary_rows,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

