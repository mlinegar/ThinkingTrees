#!/usr/bin/env python3
"""Plot cut-budgeted changepoint scaling vs train_docs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fixed", "dp_honest", "chunker_honest", "oracle_opt")
POLICY_LABEL = {
    "fixed": "fixed",
    "dp_honest": "honest_dp",
    "chunker_honest": "honest_greedy",
    "oracle_opt": "oracle_opt",
}
POLICY_COLOR = {
    "fixed": "#555555",
    "dp_honest": "#1f77b4",
    "chunker_honest": "#ff7f0e",
    "oracle_opt": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate/plot cut-budgeted changepoint scaling vs train_docs."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_cut_budget/train_*_seed_*.json",
        help="Glob for per-run JSON outputs (must contain varying train_docs).",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_cut_budget_scaling.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_cut_budget_scaling_report.json",
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
        mean_fixed_cut_budget = float(payload.get("mean_fixed_cut_budget", float("nan")))
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
                    "mean_hamming_loss": float(m["mean_hamming_loss"]),
                    "mean_hamming_gap_to_oracle": float(m["mean_hamming_gap_to_oracle"]),
                    "mean_predicted_boundary_count": float(m["mean_predicted_boundary_count"]),
                    "mean_true_boundary_count": float(m["mean_true_boundary_count"]),
                    "mean_fixed_cut_budget": float(mean_fixed_cut_budget),
                }
            )

    fixed_by_run = {
        (int(r["train_docs"]), int(r["seed"])): r for r in rows if str(r["policy"]) == "fixed"
    }
    for r in rows:
        fixed = fixed_by_run.get((int(r["train_docs"]), int(r["seed"])))
        if fixed is None:
            r["cuts_saved_vs_fixed"] = float("nan")
            r["hamming_improvement_vs_fixed"] = float("nan")
            continue
        r["cuts_saved_vs_fixed"] = float(
            float(fixed["mean_predicted_boundary_count"]) - float(r["mean_predicted_boundary_count"])
        )
        r["hamming_improvement_vs_fixed"] = float(
            float(fixed["mean_hamming_loss"]) - float(r["mean_hamming_loss"])
        )

    rows.sort(key=lambda r: (r["train_docs"], r["policy"], r["seed"]))
    train_docs_values = sorted({int(r["train_docs"]) for r in rows})

    metrics_to_plot = (
        ("mean_hamming_loss", "Mean Hamming loss (fp+fn)", None),
        ("hamming_improvement_vs_fixed", "Hamming improvement vs fixed (↑)", 0.0),
        ("mean_hamming_gap_to_oracle", "Mean Hamming gap to oracle (↓)", 0.0),
        ("mean_predicted_boundary_count", "Mean cuts used (↓)", None),
        ("cuts_saved_vs_fixed", "Cuts saved vs fixed (↑)", 0.0),
        ("predicted_to_true_ratio", "Predicted/true boundary ratio", 1.0),
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

    fig, axs = plt.subplots(2, 3, figsize=(18.8, 8.4), constrained_layout=True)
    axs = axs.reshape(-1)

    for ax, (metric, ylabel, baseline) in zip(axs.tolist(), metrics_to_plot):
        for policy in POLICIES:
            ks = [
                int(k)
                for k in sorted(int(k) for k in aggregated[metric][policy].keys())
                if int(aggregated[metric][policy][str(k)]["n"]) > 0
            ]
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
    fig.suptitle("Markov Changepoint: Cut-Budgeted DP Scaling", fontsize=12)

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
