#!/usr/bin/env python3
"""Plot learning curves for the bigram-score oracle-guidance simulation.

This expects per-run JSON outputs from `run_bigram_score_guidance_simulation.py`.

Compared to the grid heatmap, this plot makes it easier to see:
  - whether error decreases as we spend more oracle labels/cost, and
  - whether leaf-only guidance exhibits a bias floor when boundary correction is needed.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bigram-score guidance learning curves.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/bigram_score_guidance/train_*_seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/bigram_score_guidance_lines.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/bigram_score_guidance_lines_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=["train_docs", "oracle_queries_total", "oracle_cost_total", "oracle_cost_ratio"],
        default="train_docs",
        help="X axis for the learning curves.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["rmse", "weight_cosine", "weight_rmse"],
        default="rmse",
        help="Y axis metric to plot.",
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
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log scale for the x-axis.",
    )
    return parser.parse_args()


def _reduce(xs: List[float], *, agg: str) -> float:
    if not xs:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(xs, dtype=np.float64)))
    if agg == "median":
        return float(statistics.median(xs))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _band_quantiles(kind: str) -> Optional[Tuple[float, float]]:
    if kind == "none":
        return None
    if kind == "p10_p90":
        return (10.0, 90.0)
    if kind == "p25_p75":
        return (25.0, 75.0)
    raise ValueError(f"unsupported band: {kind!r}")


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        metrics = payload.get("metrics", {})
        full_cost = float(payload.get("train_full_doc_cost_total", float("nan")))

        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        cross_mult = float(cfg.get("cross_topic_weight_multiplier", float("nan")))
        boundary_frac = float(payload.get("test_boundary_term_fraction", float("nan")))
        if train_docs <= 0 or seed < 0 or not np.isfinite(cross_mult):
            continue

        for _policy, m in metrics.items():
            if not isinstance(m, dict):
                continue
            strat = str(m.get("guidance_strategy", ""))
            level = float(m.get("guidance_per_leaf", float("nan")))
            oracle_queries_total = float(m.get("oracle_queries_total", float("nan")))
            oracle_cost_total = float(m.get("oracle_cost_total", float("nan")))
            cost_ratio = (
                float(oracle_cost_total / full_cost)
                if np.isfinite(oracle_cost_total) and np.isfinite(full_cost) and full_cost > 0
                else float("nan")
            )
            rows.append(
                {
                    "train_docs": int(train_docs),
                    "seed": int(seed),
                    "strategy": str(strat),
                    "guidance_per_leaf": float(level),
                    "cross_mult": float(cross_mult),
                    "boundary_frac": float(boundary_frac),
                    "oracle_queries_total": float(oracle_queries_total),
                    "oracle_cost_total": float(oracle_cost_total),
                    "oracle_cost_ratio": float(cost_ratio),
                    "rmse": float(m.get("rmse", float("nan"))),
                    "weight_rmse": float(m.get("weight_rmse", float("nan"))),
                    "weight_cosine": float(m.get("weight_cosine", float("nan"))),
                }
            )

    if not rows:
        raise ValueError("no usable rows found in inputs")

    x_axis = str(args.x_axis)
    metric = str(args.metric)
    agg = str(args.aggregate)
    band_q = _band_quantiles(str(args.band))

    strategies = sorted({str(r["strategy"]) for r in rows})
    multipliers = sorted({float(r["cross_mult"]) for r in rows if np.isfinite(float(r["cross_mult"]))})
    guidance_levels = sorted(
        {float(r["guidance_per_leaf"]) for r in rows if np.isfinite(float(r["guidance_per_leaf"]))}
    )

    # Layout: rows = multipliers, cols = strategies.
    nrows = int(max(1, len(multipliers)))
    ncols = int(max(1, len(strategies)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols + 1.2, 3.9 * nrows),
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

    cmap = plt.get_cmap("viridis")
    colors = {g: cmap(i / max(1, len(guidance_levels) - 1)) for i, g in enumerate(guidance_levels)}

    def _x_label() -> str:
        if x_axis == "train_docs":
            return "train docs"
        if x_axis == "oracle_queries_total":
            return "oracle queries (train)"
        if x_axis == "oracle_cost_total":
            return "oracle cost (train)"
        if x_axis == "oracle_cost_ratio":
            return "oracle cost / full-doc cost"
        return x_axis

    def _y_label() -> str:
        if metric == "rmse":
            return "RMSE (↓)"
        if metric == "weight_rmse":
            return "weight RMSE (↓)"
        if metric == "weight_cosine":
            return "cosine(w_hat, w_true) (↑)"
        return metric

    report: Dict[str, object] = {
        "x_axis": x_axis,
        "metric": metric,
        "aggregate": agg,
        "band": str(args.band),
        "strategies": strategies,
        "multipliers": multipliers,
        "guidance_levels": guidance_levels,
        "series": {},
    }

    for r_i, mult in enumerate(multipliers):
        # Provide a hint about realized boundary fraction (averaged across all runs at this multiplier).
        fracs = [
            float(r["boundary_frac"])
            for r in rows
            if float(r["cross_mult"]) == float(mult) and np.isfinite(float(r["boundary_frac"]))
        ]
        boundary_hint = f" (b≈{float(np.mean(np.asarray(fracs, dtype=np.float64))):.2f})" if fracs else ""
        for c_i, strat in enumerate(strategies):
            ax = axes[r_i, c_i]
            sub = [r for r in rows if float(r["cross_mult"]) == float(mult) and str(r["strategy"]) == str(strat)]
            if not sub:
                continue
            x_values = sorted({float(r[x_axis]) for r in sub if np.isfinite(float(r[x_axis]))})
            for g in guidance_levels:
                xs: List[float] = []
                ys: List[float] = []
                lo: List[float] = []
                hi: List[float] = []
                for x in x_values:
                    vals = [
                        float(r[metric])
                        for r in sub
                        if float(r["guidance_per_leaf"]) == float(g)
                        and float(r[x_axis]) == float(x)
                        and np.isfinite(float(r[metric]))
                    ]
                    if not vals:
                        continue
                    xs.append(float(x))
                    ys.append(float(_reduce(vals, agg=agg)))
                    if band_q is not None:
                        lo.append(float(_percentile(vals, band_q[0])))
                        hi.append(float(_percentile(vals, band_q[1])))
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[g], label=f"{g:g}/leaf")
                if band_q is not None and len(lo) == len(xs) and len(hi) == len(xs):
                    ax.fill_between(xs, lo, hi, color=colors[g], alpha=0.20, linewidth=0)

                # Save to report.
                key = f"mult={mult:g}|strat={strat}|g={g:g}"
                report["series"][key] = [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]

            ax.set_title(f"{strat} | mult={mult:g}{boundary_hint}")
            ax.set_xlabel(_x_label())
            if c_i == 0:
                ax.set_ylabel(_y_label())
            ax.grid(True, alpha=0.25)
            if args.log_x:
                ax.set_xscale("log")

    # Global legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(6, len(by_label)))

    fig.suptitle(
        f"Bigram score guidance learning curves | x={x_axis} | y={metric} | agg={agg}",
        fontsize=12,
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

