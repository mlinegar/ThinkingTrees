#!/usr/bin/env python3
"""Plot Segment-LDA estimator stress under full audit."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot estimator stress diagnostics for Segment-LDA at full audit.")
    p.add_argument(
        "--input-glob",
        type=str,
        default="outputs/cpu_megasweep_20260302_megasweep_paper_v2/segment_lda_ops_weight_recovery/**/*seed_*.json",
    )
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segment_lda_estimator_stress.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/segment_lda_estimator_stress_report.json",
    )
    return p.parse_args()


def _reduce(vals: List[float], agg: str) -> float:
    clean = [float(x) for x in vals if np.isfinite(float(x))]
    if not clean:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(clean, dtype=np.float64)))
    return float(statistics.median(clean))


def _q(vals: List[float], p: float) -> float:
    clean = np.asarray([float(x) for x in vals if np.isfinite(float(x))], dtype=np.float64)
    if clean.size == 0:
        return float("nan")
    return float(np.percentile(clean, float(p)))


def main() -> int:
    args = _parse_args()
    agg = str(args.aggregate)

    # estimator -> train_docs -> ridge_root_mae list
    cube: Dict[str, Dict[int, List[float]]] = {}
    # estimator -> exact list
    exact_by_est: Dict[str, List[float]] = {}
    for fp in glob.glob(str(args.input_glob), recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        if abs(float(cfg.get("audit_fraction", float("nan"))) - 1.0) > 1e-12:
            continue
        est = str(cfg.get("topic_phi_estimator", ""))
        td = int(cfg.get("train_docs", -1))
        met = payload.get("metrics", {}) or {}
        ridge = float((met.get("ridge", {}) or {}).get("root_mae", float("nan")))
        exact = float((met.get("exact", {}) or {}).get("root_mae", float("nan")))
        cube.setdefault(est, {}).setdefault(td, []).append(ridge)
        exact_by_est.setdefault(est, []).append(exact)

    if not cube:
        raise ValueError("No full-audit rows found for Segment-LDA estimator stress.")

    estimators = sorted(cube.keys())
    train_docs = sorted({td for est in estimators for td in cube[est].keys()})

    heat = np.full((len(estimators), len(train_docs)), np.nan, dtype=np.float64)
    heat_p90 = np.full((len(estimators), len(train_docs)), np.nan, dtype=np.float64)
    for i, est in enumerate(estimators):
        for j, td in enumerate(train_docs):
            vals = cube[est].get(td, [])
            heat[i, j] = _reduce(vals, agg)
            heat_p90[i, j] = _q(vals, 90.0)

    exact_by_est_agg = {est: _reduce(exact_by_est.get(est, []), agg) for est in estimators}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)
    ax0, ax1 = axes

    im0 = ax0.imshow(heat, aspect="auto", cmap="viridis")
    ax0.set_xticks(np.arange(len(train_docs), dtype=np.float64))
    ax0.set_xticklabels([str(x) for x in train_docs])
    ax0.set_yticks(np.arange(len(estimators), dtype=np.float64))
    ax0.set_yticklabels(estimators)
    ax0.set_xlabel("train_docs")
    ax0.set_title(f"Ridge Root MAE ({agg}) @ audit_fraction=1")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cbar0.set_label("root_mae")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            if np.isfinite(v):
                ax0.text(j, i, f"{v:.2g}", ha="center", va="center", color="white", fontsize=8)

    x = np.arange(len(estimators), dtype=np.float64)
    gaps = np.asarray(
        [
            _reduce([heat[i, j] for j in range(len(train_docs)) if np.isfinite(heat[i, j])], agg) - exact_by_est_agg[est]
            for i, est in enumerate(estimators)
        ],
        dtype=np.float64,
    )
    p90_line = np.asarray(
        [_reduce([heat_p90[i, j] for j in range(len(train_docs)) if np.isfinite(heat_p90[i, j])], agg) for i in range(len(estimators))],
        dtype=np.float64,
    )
    ax1.bar(x, gaps, color="#ff7f0e", alpha=0.85, label="median gap to exact")
    ax1.plot(x, p90_line, marker="o", color="#2ca02c", linewidth=1.8, label="p90 ridge error")
    ax1.axhline(0.0, color="#222222", linestyle=":", linewidth=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(estimators, rotation=25, ha="right")
    ax1.set_ylabel("Error")
    ax1.set_title("Estimator Stress: Gap & Tail")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=9)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=190)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "aggregate": agg,
        "estimators": estimators,
        "train_docs": [int(x) for x in train_docs],
        "heat_median_or_mean": heat.tolist(),
        "heat_p90": heat_p90.tolist(),
        "exact_by_estimator": {k: float(v) for k, v in exact_by_est_agg.items()},
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

