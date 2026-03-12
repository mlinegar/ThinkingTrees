#!/usr/bin/env python3
"""Plot learning curves for the Segment-LDA OPS weight-recovery simulation.

This expects per-run JSON outputs from `run_segment_lda_ops_weight_recovery_simulation.py`.

Compared to the grid heatmap, this plot makes it easier to see whether:
  - prediction error shrinks as we spend more oracle labels/cost, and
  - the learned model approaches the relevant bias floor (e.g. undersupported sketch).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Segment-LDA OPS weight-recovery learning curves.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery/**/*seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--ridge-key",
        type=str,
        default="ridge",
        help="Which metrics entry to plot (e.g. ridge, ridge_true_topics, ridge_infer_true_phi, ridge_infer_est_phi).",
    )
    parser.add_argument(
        "--audit-strategy",
        type=str,
        default="random",
        help="Filter to this audit_strategy (e.g. random, active_small, profile).",
    )
    parser.add_argument(
        "--topic-phi-estimator",
        type=str,
        default=None,
        help="Optional filter on cfg_topic_phi_estimator (e.g. true, noisy_theory, neural_hybrid).",
    )
    parser.add_argument(
        "--topic-phi-docs",
        type=int,
        default=None,
        help="Optional filter on cfg_topic_phi_docs (<=0 means 'use train_docs' in the sim).",
    )
    parser.add_argument(
        "--oracle-noise-std",
        type=float,
        default=None,
        help="Optional filter on cfg_oracle_noise_std (defaults to include only 0 if present).",
    )
    parser.add_argument(
        "--topic-source",
        action="append",
        default=[],
        help="Filter to a specific topic_source (repeatable). Default: include all.",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=["train_docs", "total_labels_total", "oracle_queries_total", "oracle_cost_total", "oracle_cost_ratio"],
        default="oracle_queries_total",
        help="X axis for the learning curves.",
    )
    parser.add_argument(
        "--budget-axis",
        type=str,
        choices=["audit_fraction", "internal_per_leaf"],
        default="audit_fraction",
        help="Which budget axis to stratify lines by. 'audit_fraction' matches the sweep knob; "
        "'internal_per_leaf' is the realized label rate after rounding.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "root_mae",
            "merge_mae",
            "leaf_mae",
            "leaf_accuracy_train",
            "leaf_accuracy_test",
            "theta_cosine",
            "bigram_cosine",
            "lambda_abs_error",
            "rank_over_d",
            "log10_a_condition",
            "train_rmse",
        ],
        default="root_mae",
        help="Y axis metric to plot (from the learned ridge model unless noted).",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["median", "mean"],
        default="median",
        help="How to aggregate across seeds for each point.",
    )
    parser.add_argument(
        "--band",
        type=str,
        choices=["none", "p10_p90", "p25_p75"],
        default="p10_p90",
        help="Optional quantile band to shade across seeds.",
    )
    parser.add_argument("--log-x", action="store_true", help="Use a log scale for the x-axis.")
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery_lines.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery_lines_report.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def _reduce(vals: List[float], *, agg: str) -> float:
    if not vals:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(vals, dtype=np.float64)))
    if agg == "median":
        return float(statistics.median(vals))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _band_quantiles(kind: str) -> Optional[Tuple[float, float]]:
    if kind == "none":
        return None
    if kind == "p10_p90":
        return (10.0, 90.0)
    if kind == "p25_p75":
        return (25.0, 75.0)
    raise ValueError(f"unsupported band: {kind!r}")


def _collect_rows(
    files: Iterable[Path],
    *,
    ridge_key: str,
    audit_strategy: str,
    topic_phi_estimator: str | None,
    topic_phi_docs: int | None,
) -> List[dict]:
    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        geom = payload.get("training_geometry", {})
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        ridge = metrics.get(str(ridge_key), {})
        if not isinstance(ridge, dict):
            continue

        if audit_strategy and str(cfg.get("audit_strategy", "")) != str(audit_strategy):
            continue
        if topic_phi_estimator is not None:
            if str(cfg.get("topic_phi_estimator", "")) != str(topic_phi_estimator):
                continue
        if topic_phi_docs is not None:
            if int(cfg.get("topic_phi_docs", 0)) != int(topic_phi_docs):
                continue

        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        lam = float(cfg.get("lambda_multiplier", float("nan")))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
        topic_source = str(cfg.get("topic_source", ""))
        oracle_noise_std = float(cfg.get("oracle_noise_std", 0.0))

        mean_leaves = float(geom.get("mean_leaves", float("nan")))
        mean_internal = float(geom.get("mean_internal_labels", float("nan")))
        internal_per_leaf = (
            float(mean_internal) / float(mean_leaves)
            if np.isfinite(mean_internal) and np.isfinite(mean_leaves) and mean_leaves > 0
            else float("nan")
        )

        rows.append(
            {
                "path": str(path),
                "train_docs": int(train_docs),
                "seed": int(seed),
                "lambda_multiplier": float(lam),
                "audit_fraction": float(audit_fraction),
                "internal_per_leaf": float(internal_per_leaf),
                "topic_source": str(topic_source),
                "oracle_noise_std": float(oracle_noise_std),
                "total_labels_total": int(geom.get("total_labels_total", -1)),
                "oracle_queries_total": float(ridge.get("oracle_queries_total", float("nan"))),
                "oracle_cost_total": float(ridge.get("oracle_cost_total", float("nan"))),
                "oracle_cost_ratio": float(ridge.get("oracle_cost_ratio", float("nan"))),
                "ridge": {
                    "root_mae": float(ridge.get("root_mae", float("nan"))),
                    "merge_mae": float(ridge.get("merge_mae", float("nan"))),
                    "leaf_mae": float(ridge.get("leaf_mae", float("nan"))),
                    "leaf_accuracy_train": float(ridge.get("leaf_accuracy_train", float("nan"))),
                    "leaf_accuracy_test": float(ridge.get("leaf_accuracy_test", float("nan"))),
                    "theta_cosine": float(ridge.get("theta_cosine", float("nan"))),
                    "bigram_cosine": float(ridge.get("bigram_cosine", float("nan"))),
                    "lambda_abs_error": float(ridge.get("lambda_abs_error", float("nan"))),
                    "rank_over_d": (
                        float(ridge.get("rank", float("nan"))) / float(ridge.get("d", float("nan")))
                        if float(ridge.get("d", float("nan"))) > 0
                        else float("nan")
                    ),
                    "log10_a_condition": (
                        float(math.log10(float(ridge.get("a_condition"))))
                        if np.isfinite(float(ridge.get("a_condition", float("nan"))))
                        and float(ridge.get("a_condition", float("nan"))) > 0
                        else float("nan")
                    ),
                    "train_rmse": float(ridge.get("train_rmse", float("nan"))),
                },
                "baselines": {
                    "exact": metrics.get("exact", {}),
                    "undersupported": metrics.get("undersupported", {}),
                },
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(
        files,
        ridge_key=str(args.ridge_key),
        audit_strategy=str(args.audit_strategy),
        topic_phi_estimator=(str(args.topic_phi_estimator) if args.topic_phi_estimator is not None else None),
        topic_phi_docs=(int(args.topic_phi_docs) if args.topic_phi_docs is not None else None),
    )
    if not rows:
        raise ValueError("no usable ridge rows found (check audit_strategy filter and input_glob)")

    # Noise filtering (avoid mixing by default).
    noise_values = sorted({float(r["oracle_noise_std"]) for r in rows if np.isfinite(float(r["oracle_noise_std"]))})
    if args.oracle_noise_std is not None:
        target = float(args.oracle_noise_std)
        rows = [r for r in rows if float(r["oracle_noise_std"]) == target]
        if not rows:
            raise ValueError(f"no rows matched oracle_noise_std={target:g}")
        noise_values = [target]
    elif len(noise_values) > 1:
        raise ValueError(
            f"multiple oracle_noise_std values present ({noise_values}); pass --oracle-noise-std to filter"
        )

    if args.topic_source:
        keep = {str(x) for x in args.topic_source}
        rows = [r for r in rows if str(r["topic_source"]) in keep]
        if not rows:
            raise ValueError("no rows remaining after topic_source filtering")

    x_axis = str(args.x_axis)
    budget_axis = str(args.budget_axis)
    metric = str(args.metric)
    agg = str(args.aggregate)
    band_q = _band_quantiles(str(args.band))

    lambdas = sorted({float(r["lambda_multiplier"]) for r in rows if np.isfinite(float(r["lambda_multiplier"]))})
    topic_sources = sorted({str(r["topic_source"]) for r in rows})
    budgets = sorted({float(r[budget_axis]) for r in rows if np.isfinite(float(r[budget_axis]))})
    if not lambdas or not topic_sources or not budgets:
        raise ValueError("insufficient variation in inputs to plot curves")

    cmap = plt.get_cmap("viridis")
    colors = {b: cmap(i / max(1, len(budgets) - 1)) for i, b in enumerate(budgets)}

    def _x_label() -> str:
        if x_axis == "train_docs":
            return "train docs"
        if x_axis == "total_labels_total":
            return "total labeled spans (train)"
        if x_axis == "oracle_queries_total":
            return "oracle queries (train)"
        if x_axis == "oracle_cost_total":
            return "oracle cost (train)"
        if x_axis == "oracle_cost_ratio":
            return "oracle cost / full-doc cost"
        return x_axis

    def _y_label() -> str:
        down = " (↓)"
        up = " (↑)"
        if metric in {"root_mae", "merge_mae", "leaf_mae", "lambda_abs_error", "log10_a_condition", "train_rmse"}:
            return f"{metric}{down}"
        if metric in {"theta_cosine", "bigram_cosine", "rank_over_d"}:
            return f"{metric}{up}"
        return metric

    def _budget_label(b: float) -> str:
        if budget_axis == "audit_fraction":
            return f"{b:g}"
        return f"{b:.3f}".rstrip("0").rstrip(".") + "/leaf"

    nrows = len(lambdas)
    ncols = len(topic_sources)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.6 * ncols + 1.2, 4.1 * nrows),
        constrained_layout=True,
        sharex=False,
        sharey=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.asarray([[axes]])
    elif nrows == 1:
        axes = np.asarray([axes])
    elif ncols == 1:
        axes = np.asarray([[ax] for ax in axes])

    report: Dict[str, object] = {
        "input_files": len(files),
        "rows": len(rows),
        "ridge_key": str(args.ridge_key),
        "audit_strategy": str(args.audit_strategy),
        "topic_phi_estimator": (str(args.topic_phi_estimator) if args.topic_phi_estimator is not None else None),
        "topic_phi_docs": (int(args.topic_phi_docs) if args.topic_phi_docs is not None else None),
        "oracle_noise_std": (float(noise_values[0]) if noise_values else None),
        "x_axis": x_axis,
        "budget_axis": budget_axis,
        "metric": metric,
        "aggregate": agg,
        "band": str(args.band),
        "lambda_values": lambdas,
        "topic_sources": topic_sources,
        "budgets": budgets,
        "series": {},
    }

    for r_i, lam in enumerate(lambdas):
        for c_i, ts in enumerate(topic_sources):
            ax = axes[r_i, c_i]
            sub = [r for r in rows if float(r["lambda_multiplier"]) == float(lam) and str(r["topic_source"]) == ts]
            if not sub:
                continue

            x_values = sorted({float(r[x_axis]) for r in sub if np.isfinite(float(r[x_axis]))})
            if not x_values:
                continue

            # Baseline lines (only meaningful for MAE-like sketch metrics).
            if metric in {"root_mae", "merge_mae", "leaf_mae"}:
                # One baseline value per seed (dedupe across train_docs/budgets).
                exact_by_seed: Dict[int, float] = {}
                under_by_seed: Dict[int, float] = {}
                for r in sub:
                    seed = int(r["seed"])
                    exact = r.get("baselines", {}).get("exact", {})
                    under = r.get("baselines", {}).get("undersupported", {})
                    if seed not in exact_by_seed and isinstance(exact, dict):
                        exact_by_seed[seed] = float(exact.get(metric, float("nan")))
                    if seed not in under_by_seed and isinstance(under, dict):
                        under_by_seed[seed] = float(under.get(metric, float("nan")))

                exact_vals = [v for v in exact_by_seed.values() if np.isfinite(v)]
                under_vals = [v for v in under_by_seed.values() if np.isfinite(v)]
                if exact_vals:
                    ax.axhline(
                        _reduce(exact_vals, agg=agg),
                        color="black",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.6,
                        label="exact",
                    )
                if under_vals:
                    ax.axhline(
                        _reduce(under_vals, agg=agg),
                        color="gray",
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.9,
                        label="undersupported",
                    )

            # Learned ridge curves.
            for b in budgets:
                xs: List[float] = []
                ys: List[float] = []
                lo: List[float] = []
                hi: List[float] = []
                for x in x_values:
                    vals = [
                        float(r["ridge"][metric])
                        for r in sub
                        if np.isfinite(float(r[x_axis]))
                        and float(r[x_axis]) == float(x)
                        and np.isfinite(float(r[budget_axis]))
                        and float(r[budget_axis]) == float(b)
                        and np.isfinite(float(r["ridge"].get(metric, float("nan"))))
                    ]
                    if not vals:
                        continue
                    xs.append(float(x))
                    ys.append(_reduce(vals, agg=agg))
                    if band_q is not None:
                        lo.append(_percentile(vals, band_q[0]))
                        hi.append(_percentile(vals, band_q[1]))
                if not xs:
                    continue
                color = colors[b]
                ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=f"{_budget_label(b)}")
                if band_q is not None and len(lo) == len(xs) and len(hi) == len(xs):
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0.0)

            ax.set_title(f"λ={lam:g} | topic_source={ts}", fontsize=10)
            ax.set_xlabel(_x_label(), fontsize=9)
            ax.set_ylabel(_y_label(), fontsize=9)
            ax.grid(True, alpha=0.25)
            if bool(args.log_x):
                ax.set_xscale("log")

            # Keep legends short: only put them on the top-right panel.
            if r_i == 0 and c_i == (ncols - 1):
                ax.legend(fontsize=8, loc="best", frameon=True)

    fig.suptitle(
        "Segment-LDA OPS weight recovery learning curves"
        + (f" | audit_strategy={args.audit_strategy}" if args.audit_strategy else "")
        + (f" | oracle_noise_std={noise_values[0]:g}" if noise_values else ""),
        fontsize=12,
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
