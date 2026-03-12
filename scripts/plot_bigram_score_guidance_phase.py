#!/usr/bin/env python3
"""Plot a "need for correction" × "correction budget" phase diagram for bigram-score guidance sims.

This complements `plot_bigram_score_guidance_grid.py` by collapsing out `train_docs` and instead
showing how performance changes as:
  - the oracle becomes more boundary-dependent (need), and
  - we spend more internal-node labels (correction budget).

It expects per-run JSON outputs from `run_bigram_score_guidance_simulation.py`.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean
import statistics
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bigram-score guidance phase diagram.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/bigram_score_guidance/train_*_seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/bigram_score_guidance_phase.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/bigram_score_guidance_phase_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--train-docs",
        type=int,
        default=0,
        help="Which train_docs slice to plot. <=0 selects the maximum train_docs available.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["mean", "median", "p10", "p90"],
        default="median",
        help="How to aggregate across seeds for each cell.",
    )
    parser.add_argument(
        "--y-axis",
        type=str,
        choices=["cross_topic_weight_multiplier", "boundary_term_fraction"],
        default="cross_topic_weight_multiplier",
        help="Which 'need' axis to use.",
    )
    return parser.parse_args()


def _reduce(xs: List[float], *, agg: str) -> float:
    if not xs:
        return float("nan")
    if agg == "mean":
        return float(fmean(xs))
    if agg == "median":
        return float(statistics.median(xs))
    if agg == "p10":
        return float(np.percentile(np.asarray(xs, dtype=np.float64), 10))
    if agg == "p90":
        return float(np.percentile(np.asarray(xs, dtype=np.float64), 90))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    xlabels: List[str],
    ylabels: List[str],
    title: str,
    cmap: str,
) -> None:
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(list(range(len(xlabels))))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(list(range(len(ylabels))))
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


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

        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        cross_mult = float(cfg.get("cross_topic_weight_multiplier", float("nan")))
        boundary_frac = float(payload.get("test_boundary_term_fraction", float("nan")))
        full_cost = float(payload.get("train_full_doc_cost_total", float("nan")))
        if train_docs <= 0 or seed < 0 or not np.isfinite(cross_mult):
            continue

        for policy, m in metrics.items():
            if not isinstance(m, dict):
                continue
            strat = str(m.get("guidance_strategy", ""))
            level = float(m.get("guidance_per_leaf", float("nan")))
            rmse = float(m.get("rmse", float("nan")))
            weight_cos = float(m.get("weight_cosine", float("nan")))
            oracle_cost_total = float(m.get("oracle_cost_total", float("nan")))
            cost_ratio = (
                float(oracle_cost_total / full_cost)
                if np.isfinite(oracle_cost_total) and np.isfinite(full_cost) and full_cost > 0
                else float("nan")
            )
            rows.append(
                {
                    "path": str(path),
                    "train_docs": int(train_docs),
                    "seed": int(seed),
                    "policy": str(policy),
                    "strategy": str(strat),
                    "guidance_per_leaf": float(level),
                    "cross_mult": float(cross_mult),
                    "boundary_frac": float(boundary_frac),
                    "rmse": float(rmse),
                    "cost_ratio": float(cost_ratio),
                    "weight_cosine": float(weight_cos),
                }
            )

    if not rows:
        raise ValueError("no usable rows found in inputs")

    train_docs_values = sorted({int(r["train_docs"]) for r in rows})
    td = int(args.train_docs)
    if td <= 0:
        td = int(max(train_docs_values))
    rows = [r for r in rows if int(r["train_docs"]) == int(td)]
    if not rows:
        raise ValueError(f"no rows for train_docs={td}")

    strategies = sorted({str(r["strategy"]) for r in rows})
    levels = sorted({float(r["guidance_per_leaf"]) for r in rows if np.isfinite(float(r["guidance_per_leaf"]))})
    if not levels:
        raise ValueError("no finite guidance_per_leaf values found")

    y_axis = str(args.y_axis)
    if y_axis == "cross_topic_weight_multiplier":
        needs = sorted({float(r["cross_mult"]) for r in rows if np.isfinite(float(r["cross_mult"]))})
    else:
        needs = sorted({float(r["boundary_frac"]) for r in rows if np.isfinite(float(r["boundary_frac"]))})
    if not needs:
        raise ValueError("no finite need axis values found")

    def _need_label(x: float) -> str:
        if y_axis == "cross_topic_weight_multiplier":
            # Add a hint about realized boundary fraction (averaged across seeds).
            fracs = [
                float(r["boundary_frac"])
                for r in rows
                if float(r["cross_mult"]) == float(x) and np.isfinite(float(r["boundary_frac"]))
            ]
            if fracs:
                return f"{x:g} (b={_reduce(fracs, agg='mean'):.2f})"
            return f"{x:g}"
        return f"{x:.2f}"

    xlabels = [f"{x:g}/leaf" for x in levels]
    ylabels = [_need_label(x) for x in needs]

    def _grid(metric: str, *, strat: str) -> np.ndarray:
        mat = np.full((len(needs), len(levels)), np.nan, dtype=np.float64)
        for yi, need in enumerate(needs):
            for xi, lvl in enumerate(levels):
                vals: List[float] = []
                for r in rows:
                    if str(r["strategy"]) != str(strat):
                        continue
                    if float(r["guidance_per_leaf"]) != float(lvl):
                        continue
                    if y_axis == "cross_topic_weight_multiplier":
                        if float(r["cross_mult"]) != float(need):
                            continue
                    else:
                        if float(r["boundary_frac"]) != float(need):
                            continue
                    v = float(r.get(metric, float("nan")))
                    if np.isfinite(v):
                        vals.append(v)
                if vals:
                    mat[yi, xi] = _reduce(vals, agg=str(args.aggregate))
        return mat

    metrics_to_plot: Tuple[Tuple[str, str, str], ...] = (
        ("rmse", "RMSE to oracle score (↓)", "viridis_r"),
        ("cost_ratio", "Oracle cost / full-doc cost (↓)", "viridis_r"),
        ("weight_cosine", "Cosine(w_hat, w_true) (↑)", "viridis"),
    )

    fig, axs = plt.subplots(
        len(metrics_to_plot),
        len(strategies),
        figsize=(5.4 * len(strategies) + 2.0, 3.6 * len(metrics_to_plot)),
        constrained_layout=True,
    )
    if len(metrics_to_plot) == 1:
        axs = np.asarray([axs])
    if len(strategies) == 1:
        axs = axs.reshape(len(metrics_to_plot), 1)

    for r_i, (metric, title, cmap) in enumerate(metrics_to_plot):
        for c_i, strat in enumerate(strategies):
            mat = _grid(metric, strat=strat)
            _heatmap(
                axs[r_i, c_i],
                mat,
                xlabels=xlabels,
                ylabels=ylabels,
                title=f"{strat} | {title}",
                cmap=cmap,
            )
            axs[r_i, c_i].set_xlabel("extra internal oracle queries per leaf")
            axs[r_i, c_i].set_ylabel("need for correction" if y_axis == "boundary_term_fraction" else "cross-topic weight multiplier")

    fig.suptitle(
        f"Bigram score guidance phase diagram | train_docs={td} | y={y_axis} | agg={args.aggregate}",
        fontsize=12,
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(files)),
        "train_docs": int(td),
        "aggregate": str(args.aggregate),
        "y_axis": y_axis,
        "strategies": strategies,
        "guidance_levels": levels,
        "needs": needs,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

