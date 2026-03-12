#!/usr/bin/env python3
"""Plot Markov changepoint honesty metrics vs training data size."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fixed", "chunker_honest", "chunker_leaky")
POLICY_LABEL = {
    "fixed": "fixed",
    "chunker_honest": "honest",
    "chunker_leaky": "leaky",
}
POLICY_COLOR = {
    "fixed": "#555555",
    "chunker_honest": "#1f77b4",
    "chunker_leaky": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate/plot changepoint honesty scaling vs train_docs."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_scaling/train_*_seed_*.json",
        help="Glob for per-run JSON outputs (must contain varying train_docs).",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_scaling_summary.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_scaling_report.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--baseline-f1",
        type=float,
        default=None,
        help="Optional dotted baseline for boundary F1.",
    )
    parser.add_argument(
        "--baseline-cost",
        type=float,
        default=None,
        help="Optional dotted baseline for boundary cost.",
    )
    parser.add_argument(
        "--baseline-l1",
        type=float,
        default=None,
        help="Optional dotted baseline for L1 distortion.",
    )
    return parser.parse_args()


def _load_rows(files: List[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        if train_docs < 0:
            raise ValueError(f"missing config.train_docs in {path}")

        for policy in POLICIES:
            m = payload["metrics"][policy]
            rows.append(
                {
                    "train_docs": float(train_docs),
                    "seed": float(seed),
                    "policy": float(POLICIES.index(policy)),
                    "policy_name": policy,
                    "boundary_f1": float(m["boundary_f1"]),
                    "mean_boundary_cost": float(m["mean_boundary_cost"]),
                    "mean_l1": float(m["mean_l1"]),
                    "predicted_to_true_ratio": float(m["predicted_to_true_ratio"]),
                }
            )
    rows.sort(key=lambda r: (int(r["train_docs"]), str(r["policy_name"]), int(r["seed"])))
    return rows


def _stats(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(len(xs)),
        "mean": float(fmean(xs)),
        "std": float(pstdev(xs)),
    }


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    metrics = ("boundary_f1", "mean_boundary_cost", "mean_l1", "predicted_to_true_ratio")
    train_docs_values = sorted({int(r["train_docs"]) for r in rows})

    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for metric in metrics:
        out[metric] = {}
        for policy in POLICIES:
            out[metric][policy] = {}
            for td in train_docs_values:
                vals = [
                    float(r[metric])
                    for r in rows
                    if int(r["train_docs"]) == int(td) and r["policy_name"] == policy
                ]
                out[metric][policy][str(td)] = _stats(vals)
    return out


def _plot_series(
    ax: plt.Axes,
    *,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    ylabel: str,
    title: str,
    baseline: float | None,
) -> None:
    for policy in POLICIES:
        ks = sorted(int(k) for k in summary[policy].keys())
        means = np.asarray([float(summary[policy][str(k)]["mean"]) for k in ks], dtype=np.float64)
        stds = np.asarray([float(summary[policy][str(k)]["std"]) for k in ks], dtype=np.float64)

        ax.plot(ks, means, marker="o", linewidth=1.6, color=POLICY_COLOR[policy], label=POLICY_LABEL[policy])
        ax.fill_between(ks, means - stds, means + stds, alpha=0.12, color=POLICY_COLOR[policy])

    if baseline is not None:
        ax.axhline(float(baseline), linestyle=":", linewidth=1.9, color="#333333", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("train_docs (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if len(files) == 0:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _load_rows(files)
    agg = _aggregate(rows)

    fig, axs = plt.subplots(2, 2, figsize=(13.2, 8.1), constrained_layout=True)

    _plot_series(
        axs[0, 0],
        summary=agg["boundary_f1"],
        ylabel="Boundary F1",
        title="Boundary F1 vs Train Docs",
        baseline=args.baseline_f1,
    )
    axs[0, 0].legend(frameon=False, fontsize=9)

    _plot_series(
        axs[0, 1],
        summary=agg["mean_boundary_cost"],
        ylabel="Mean boundary cost",
        title="Boundary Cost vs Train Docs",
        baseline=args.baseline_cost,
    )

    _plot_series(
        axs[1, 0],
        summary=agg["mean_l1"],
        ylabel="Mean posterior L1",
        title="Posterior Distortion (L1) vs Train Docs",
        baseline=args.baseline_l1,
    )

    _plot_series(
        axs[1, 1],
        summary=agg["predicted_to_true_ratio"],
        ylabel="Predicted/true boundary ratio",
        title="Boundary Count Ratio vs Train Docs",
        baseline=1.0,
    )

    fig.suptitle("Markov Changepoint Honesty Scaling", fontsize=12)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "n_files": int(len(files)),
        "n_rows": int(len(rows)),
        "input_glob": str(args.input_glob),
        "aggregated": agg,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_fig}")
    print(f"wrote_json | {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
