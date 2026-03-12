#!/usr/bin/env python3
"""Plot calibration instability diagnostics for segmented-LDA C-TreePO sweeps.

This is intended to make it obvious when affine calibration is underdetermined:
with too few calibration samples, the calibrated proxy can be much worse than
the uncalibrated proxy (even before any guidance).

Inputs are per-run JSON outputs from:
  `scripts/run_segmented_lda_ctreepo_simulation.py`
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot calibration regression vs calibration sample size.")
    p.add_argument("--input-glob", type=str, default="outputs/segmented_lda_ctreepo/**/*.json")
    p.add_argument("--topic-phi-estimator", type=str, default="", help="Optional exact filter.")
    p.add_argument("--train-docs", type=int, default=-1, help="Optional exact filter.")
    p.add_argument(
        "--min-calibration-samples",
        type=int,
        default=1,
        help="Only include runs with calibration_samples >= this (default: 1).",
    )
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segmented_lda_ctreepo/calibration_regression.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/segmented_lda_ctreepo/calibration_regression_report.json",
    )
    return p.parse_args()


def _extract(payload: dict, path: Path) -> dict:
    cfg = payload.get("config", {}) or {}
    m = payload.get("metrics", {}) or {}
    cal = m.get("estimated_calibrated", {}) or {}
    unc = m.get("estimated_uncalibrated", {}) or {}
    return {
        "path": str(path),
        "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
        "train_docs": int(cfg.get("n_books_train", -1)),
        "n_topics": int(cfg.get("n_topics", -1)),
        "calibration_leaf_query_rate": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
        "calibration_policy": str(cfg.get("calibration_policy", "")),
        "calibration_samples": int(payload.get("calibration_samples", 0) or 0),
        "root_l1_uncalibrated": float(unc.get("root_l1_mean", float("nan"))),
        "root_l1_calibrated": float(cal.get("root_l1_mean", float("nan"))),
    }


def main() -> int:
    args = _parse_args()
    paths = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not paths:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = [_extract(json.loads(p.read_text(encoding="utf-8")), p) for p in paths]
    if str(args.topic_phi_estimator):
        rows = [r for r in rows if str(r["topic_phi_estimator"]) == str(args.topic_phi_estimator)]
    if int(args.train_docs) >= 0:
        rows = [r for r in rows if int(r["train_docs"]) == int(args.train_docs)]
    rows = [r for r in rows if int(r["calibration_samples"]) >= int(args.min_calibration_samples)]

    # Only keep finite root L1 values.
    clean: List[dict] = []
    for r in rows:
        a = float(r["root_l1_uncalibrated"])
        b = float(r["root_l1_calibrated"])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        clean.append(r)
    rows = clean
    if not rows:
        raise ValueError("no rows after filters")

    # Build arrays.
    x = np.asarray([int(r["calibration_samples"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["root_l1_calibrated"]) - float(r["root_l1_uncalibrated"]) for r in rows], dtype=np.float64)
    cal_rates = np.asarray([float(r["calibration_leaf_query_rate"]) for r in rows], dtype=np.float64)

    # Color map by calibration rate (discrete-ish).
    unique_rates = sorted({float(v) for v in cal_rates.tolist() if np.isfinite(float(v))})
    rate_to_idx: Dict[float, int] = {r: i for i, r in enumerate(unique_rates)}
    c = np.asarray([rate_to_idx.get(float(v), -1) for v in cal_rates.tolist()], dtype=np.int64)

    # Heuristic threshold for underdetermined affine map: n <= (k+1).
    ks = [int(r["n_topics"]) for r in rows if int(r["n_topics"]) > 0]
    k = int(max(ks)) if ks else 0
    under_thresh = int(k + 1) if k > 0 else None

    fig, ax = plt.subplots(figsize=(11.8, 6.0), constrained_layout=True)
    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="tab10",
        s=22,
        alpha=0.65,
        linewidths=0.0,
    )
    ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.5)
    if under_thresh is not None:
        ax.axvline(float(under_thresh), color="#b22222", linestyle="--", linewidth=1.4, alpha=0.9, label="k+1")
    ax.set_xscale("log")
    ax.set_xlabel("calibration_samples (log)")
    ax.set_ylabel("Δ root L1 = calibrated - uncalibrated")
    title_bits = ["Calibration Regression"]
    if str(args.topic_phi_estimator):
        title_bits.append(f"phi={args.topic_phi_estimator}")
    if int(args.train_docs) >= 0:
        title_bits.append(f"train_docs={int(args.train_docs)}")
    ax.set_title(" | ".join(title_bits))
    ax.grid(alpha=0.22)

    # Legend for rates (compact).
    handles: List[Tuple[plt.Line2D, str]] = []
    for r in unique_rates[:10]:
        idx = rate_to_idx[r]
        handles.append((plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.get_cmap("tab10")(idx), markersize=7), f"cal={r:g}"))
    if handles:
        ax.legend([h for h, _ in handles], [lab for _, lab in handles], frameon=False, fontsize=9, loc="upper left")

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=190)
    plt.close(fig)

    # Summaries.
    worst = sorted(
        (
            (
                float(r["root_l1_calibrated"]) - float(r["root_l1_uncalibrated"]),
                int(r["calibration_samples"]),
                float(r["calibration_leaf_query_rate"]),
                str(r["path"]),
            )
            for r in rows
        ),
        key=lambda t: t[0],
        reverse=True,
    )[:15]
    frac_bad = float(np.mean((y > 0.25).astype(np.float64))) if y.size else float("nan")

    report = {
        "input_glob": str(args.input_glob),
        "filters": {
            "topic_phi_estimator": str(args.topic_phi_estimator),
            "train_docs": int(args.train_docs),
            "min_calibration_samples": int(args.min_calibration_samples),
        },
        "n_rows_after_filters": int(len(rows)),
        "n_topics_max": int(k),
        "underdetermined_threshold_k_plus_1": int(under_thresh) if under_thresh is not None else None,
        "calibration_rates": unique_rates,
        "fraction_delta_gt_0p25": float(frac_bad),
        "worst_regressions": [
            {"delta": float(d), "calibration_samples": int(n), "calibration_leaf_query_rate": float(cr), "path": path}
            for d, n, cr, path in worst
        ],
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

