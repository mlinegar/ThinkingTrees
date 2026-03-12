#!/usr/bin/env python3
"""Plot Markov changepoint preference-gap scaling vs train_docs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fixed", "chunker_honest", "chunker_leaky", "oracle_cut")
POLICY_LABEL = {
    "fixed": "fixed",
    "chunker_honest": "honest",
    "chunker_leaky": "leaky",
    "oracle_cut": "oracle_cut",
}
POLICY_COLOR = {
    "fixed": "#555555",
    "chunker_honest": "#1f77b4",
    "chunker_leaky": "#d62728",
    "oracle_cut": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate/plot changepoint preference-gap scaling vs train_docs."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_preference/train_*_seed_*.json",
        help="Glob for per-run JSON outputs (must contain varying train_docs).",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_preference_scaling.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_preference_scaling_report.json",
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def _stats(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(len(xs)),
        "mean": float(fmean(xs)),
        "std": float(pstdev(xs)),
    }


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

        metrics = payload.get("metrics", {})
        for policy in POLICIES:
            if policy not in metrics:
                continue
            m = metrics[policy]
            rows.append(
                {
                    "train_docs": int(train_docs),
                    "seed": int(seed),
                    "policy": str(policy),
                    "boundary_f1": float(m["boundary_f1"]),
                    "predicted_to_true_ratio": float(m["predicted_to_true_ratio"]),
                    "mean_abs_count_error": float(m["mean_abs_count_error"]),
                    "mean_dpo_loss_gap_to_opt": float(m["mean_dpo_loss_gap_to_opt"]),
                }
            )

    rows.sort(key=lambda r: (r["train_docs"], r["policy"], r["seed"]))
    train_docs_values = sorted({int(r["train_docs"]) for r in rows})

    metrics_to_plot = (
        ("boundary_f1", "Boundary F1", None),
        ("predicted_to_true_ratio", "Predicted/true boundary ratio", 1.0),
        ("mean_abs_count_error", "Mean |count error|", 0.0),
        ("mean_dpo_loss_gap_to_opt", "DPO gap to oracle optimum", 0.0),
    )

    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for metric, _ylabel, _baseline in metrics_to_plot:
        aggregated[metric] = {}
        for policy in POLICIES:
            aggregated[metric][policy] = {}
            for td in train_docs_values:
                vals = [
                    float(r[metric])
                    for r in rows
                    if int(r["train_docs"]) == int(td) and r["policy"] == policy
                ]
                aggregated[metric][policy][str(td)] = _stats(vals)

    fig, axs = plt.subplots(2, 2, figsize=(13.3, 8.4), constrained_layout=True)
    axs = axs.reshape(-1)

    for ax, (metric, ylabel, baseline) in zip(axs.tolist(), metrics_to_plot):
        for policy in POLICIES:
            ks = sorted(int(k) for k in aggregated[metric][policy].keys())
            if not ks:
                continue
            means = np.asarray(
                [float(aggregated[metric][policy][str(k)]["mean"]) for k in ks], dtype=np.float64
            )
            stds = np.asarray(
                [float(aggregated[metric][policy][str(k)]["std"]) for k in ks], dtype=np.float64
            )
            ax.plot(
                ks,
                means,
                marker="o",
                linewidth=1.6,
                color=POLICY_COLOR[policy],
                label=POLICY_LABEL[policy],
            )
            ax.fill_between(ks, means - stds, means + stds, alpha=0.12, color=POLICY_COLOR[policy])

        if baseline is not None:
            ax.axhline(float(baseline), linestyle=":", linewidth=1.9, color="#333333", alpha=0.9)

        ax.set_xscale("log")
        ax.set_xlabel("train_docs (log scale)")
        ax.set_ylabel(str(ylabel))
        ax.set_title(str(ylabel))
        ax.grid(alpha=0.2)

    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Markov Changepoint: Preference Gap Scaling", fontsize=12)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(files)),
        "n_rows": int(len(rows)),
        "policies": list(POLICIES),
        "metrics": {m: {"ylabel": y, "baseline": b} for m, y, b in metrics_to_plot},
        "train_docs_values": train_docs_values,
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

