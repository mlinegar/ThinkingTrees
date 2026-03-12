#!/usr/bin/env python3
"""Plot a guidance grid for the mergeable bigram-score oracle-guidance simulation.

This expects per-run JSON outputs from `run_bigram_score_guidance_simulation.py`.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a guidance grid for bigram-score oracle guidance.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/bigram_score_guidance/train_*_seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/bigram_score_guidance_grid.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/bigram_score_guidance_grid_report.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def _stats(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {"n": int(len(xs)), "mean": float(fmean(xs)), "std": float(pstdev(xs))}


def _format_level(x: float) -> str:
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    if abs(x) < 1.0:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.2f}".rstrip("0").rstrip(".")


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if len(files) == 0:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        if train_docs < 0:
            raise ValueError(f"missing config.train_docs in {path}")
        full_cost = float(payload.get("train_full_doc_cost_total", float("nan")))

        metrics = payload.get("metrics", {})
        for policy, m in metrics.items():
            strat = str(m.get("guidance_strategy", ""))
            level = float(m.get("guidance_per_leaf", float("nan")))
            level_key = _format_level(level)
            oracle_cost_total = float(m.get("oracle_cost_total", float("nan")))
            cost_ratio = oracle_cost_total / full_cost if np.isfinite(full_cost) and full_cost > 0 else float("nan")
            rows.append(
                {
                    "train_docs": int(train_docs),
                    "seed": int(seed),
                    "policy": str(policy),
                    "strategy": strat,
                    "guidance_per_leaf": float(level),
                    "guidance_key": str(level_key),
                    "rmse": float(m.get("rmse", float("nan"))),
                    "mean_abs_error": float(m.get("mean_abs_error", float("nan"))),
                    "weight_rmse": float(m.get("weight_rmse", float("nan"))),
                    "weight_cosine": float(m.get("weight_cosine", float("nan"))),
                    "oracle_cost_total": float(oracle_cost_total),
                    "oracle_cost_ratio": float(cost_ratio),
                }
            )

    if len(rows) == 0:
        raise ValueError("no policy rows found in inputs")

    train_docs_values = sorted({int(r["train_docs"]) for r in rows})
    strategies = sorted({str(r["strategy"]) for r in rows})
    level_values: Dict[str, float] = {}
    for r in rows:
        if np.isfinite(float(r["guidance_per_leaf"])):
            level_values[str(r["guidance_key"])] = float(r["guidance_per_leaf"])
    levels = sorted(level_values.keys(), key=lambda k: float(level_values[k]))
    if len(levels) == 0:
        raise ValueError("no finite guidance_per_leaf values found")

    metrics_to_plot = (
        ("rmse", "RMSE to oracle score (↓)", "viridis_r"),
        ("oracle_cost_ratio", "Oracle cost / full-doc cost (↓)", "viridis_r"),
        ("weight_cosine", "Cosine(w_hat, w_true) (↑)", "viridis"),
    )

    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for metric, _ylabel, _cmap in metrics_to_plot:
        aggregated[metric] = {}
        for strat in strategies:
            aggregated[metric][strat] = {}
            for td in train_docs_values:
                aggregated[metric][strat][str(td)] = {}
                for key in levels:
                    vals = [
                        float(r[metric])
                        for r in rows
                        if int(r["train_docs"]) == int(td)
                        and str(r["strategy"]) == str(strat)
                        and str(r["guidance_key"]) == str(key)
                        and np.isfinite(float(r[metric]))
                    ]
                    aggregated[metric][strat][str(td)][key] = _stats(vals)

    fig, axs = plt.subplots(
        len(metrics_to_plot),
        len(strategies),
        figsize=(5.2 * len(strategies) + 2.0, 3.8 * len(metrics_to_plot)),
        constrained_layout=True,
    )
    if len(metrics_to_plot) == 1:
        axs = np.asarray([axs])
    if len(strategies) == 1:
        axs = axs.reshape(len(metrics_to_plot), 1)

    ytick_labels = [f"{k}/leaf" for k in levels]

    for row_i, (metric, ylabel, cmap_name) in enumerate(metrics_to_plot):
        for col_i, strat in enumerate(strategies):
            arr = np.full((len(levels), len(train_docs_values)), np.nan, dtype=np.float64)
            for i, lvl_key in enumerate(levels):
                for j, td in enumerate(train_docs_values):
                    stat = aggregated[metric][strat][str(td)][lvl_key]
                    arr[i, j] = float(stat["mean"])

            ax = axs[row_i, col_i]
            masked = np.ma.masked_invalid(arr)
            im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap_name)

            ax.set_title(f"{strat} | {ylabel}")
            ax.set_xticks(list(range(len(train_docs_values))))
            ax.set_xticklabels([str(x) for x in train_docs_values], rotation=45, ha="right")
            ax.set_yticks(list(range(len(levels))))
            ax.set_yticklabels(ytick_labels)
            ax.set_xlabel("train_docs")
            ax.set_ylabel("extra internal oracle queries per leaf")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Bigram Score: Oracle Guidance Grid (Split-Invariant Target)", fontsize=12)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(files)),
        "n_rows": int(len(rows)),
        "train_docs_values": train_docs_values,
        "strategies": strategies,
        "guidance_per_leaf_levels": [{"key": k, "value": float(level_values[k])} for k in levels],
        "metrics": {m: {"ylabel": y} for m, y, _c in metrics_to_plot},
        "aggregated": aggregated,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_fig}")
    print(f"wrote_json | {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
