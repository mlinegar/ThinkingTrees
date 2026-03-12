#!/usr/bin/env python3
"""Bias/variance view for changepoint honesty scaling outputs."""

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
COMPARE_POLICIES = ("fixed", "chunker_honest")
POLICY_LABEL = {
    "fixed": "fixed",
    "chunker_honest": "honest",
    "chunker_leaky": "leaky",
}
POLICY_COLOR = {
    "fixed": "#555555",
    "chunker_honest": "#1f77b4",
}
METRIC_SPECS = {
    # Regret is defined relative to leaky using the same seed.
    # Positive regret means "worse than leaky".
    "boundary_f1": {
        "label": "Boundary F1 Regret (leaky - policy)",
        "higher_is_better": True,
    },
    "mean_boundary_cost": {
        "label": "Boundary Cost Regret (policy - leaky)",
        "higher_is_better": False,
    },
    "mean_l1": {
        "label": "Posterior L1 Regret (policy - leaky)",
        "higher_is_better": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-metric bias and variance for fixed/honest policies "
            "relative to same-seed leaky baseline."
        )
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_scaling/train_*_seed_*.json",
        help="Glob for per-run changepoint scaling JSON files.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_bias_variance.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_bias_variance_report.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def _load_runs(paths: List[Path]) -> Dict[int, Dict[int, Dict[str, Dict[str, float]]]]:
    data: Dict[int, Dict[int, Dict[str, Dict[str, float]]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        if train_docs < 0 or seed < 0:
            raise ValueError(f"missing config.train_docs/seed in {path}")

        by_policy: Dict[str, Dict[str, float]] = {}
        for policy in POLICIES:
            m = payload["metrics"][policy]
            by_policy[policy] = {
                "boundary_f1": float(m["boundary_f1"]),
                "mean_boundary_cost": float(m["mean_boundary_cost"]),
                "mean_l1": float(m["mean_l1"]),
                "predicted_to_true_ratio": float(m["predicted_to_true_ratio"]),
            }

        data.setdefault(train_docs, {})[seed] = by_policy
    return data


def _regret(policy_value: float, leaky_value: float, *, higher_is_better: bool) -> float:
    if higher_is_better:
        return float(leaky_value - policy_value)
    return float(policy_value - leaky_value)


def _moments(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {
            "n": 0,
            "bias": float("nan"),
            "std": float("nan"),
            "variance": float("nan"),
            "bias_sq": float("nan"),
            "mse": float("nan"),
            "mse_minus_bias2_minus_var": float("nan"),
        }

    bias = float(fmean(values))
    std = float(pstdev(values))
    variance = float(std * std)
    bias_sq = float(bias * bias)
    mse = float(fmean([v * v for v in values]))
    return {
        "n": int(len(values)),
        "bias": bias,
        "std": std,
        "variance": variance,
        "bias_sq": bias_sq,
        "mse": mse,
        "mse_minus_bias2_minus_var": float(mse - bias_sq - variance),
    }


def _aggregate(
    runs: Dict[int, Dict[int, Dict[str, Dict[str, float]]]]
) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], Dict[str, Dict[str, Dict[str, float]]]]:
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    perf: Dict[str, Dict[str, Dict[str, float]]] = {}

    train_docs_values = sorted(runs.keys())
    for metric, spec in METRIC_SPECS.items():
        summary[metric] = {}
        perf[metric] = {}
        for policy in POLICIES:
            perf[metric][policy] = {}
            for td in train_docs_values:
                vals = [runs[td][seed][policy][metric] for seed in sorted(runs[td].keys())]
                perf[metric][policy][str(td)] = {
                    "n": int(len(vals)),
                    "mean": float(fmean(vals)),
                    "std": float(pstdev(vals)) if len(vals) > 0 else float("nan"),
                }

        for policy in COMPARE_POLICIES:
            summary[metric][policy] = {}
            for td in train_docs_values:
                errors: List[float] = []
                for seed in sorted(runs[td].keys()):
                    seed_row = runs[td][seed]
                    p = float(seed_row[policy][metric])
                    l = float(seed_row["chunker_leaky"][metric])
                    errors.append(_regret(p, l, higher_is_better=bool(spec["higher_is_better"])))
                summary[metric][policy][str(td)] = _moments(errors)
    return summary, perf


def _plot(summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], output_path: Path) -> None:
    metrics = list(METRIC_SPECS.keys())
    fig, axs = plt.subplots(len(metrics), 2, figsize=(13.0, 10.2), constrained_layout=True)

    for r, metric in enumerate(metrics):
        metric_label = str(METRIC_SPECS[metric]["label"])
        for c, y_key in enumerate(("bias", "variance")):
            ax = axs[r, c]
            for policy in COMPARE_POLICIES:
                by_td = summary[metric][policy]
                xs = np.asarray(sorted(int(k) for k in by_td.keys()), dtype=np.float64)
                ys = np.asarray([float(by_td[str(int(x))][y_key]) for x in xs], dtype=np.float64)
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    color=POLICY_COLOR[policy],
                    label=POLICY_LABEL[policy],
                )

            if y_key == "bias":
                ax.axhline(0.0, linestyle=":", linewidth=1.6, color="#333333", alpha=0.9)
            ax.set_xscale("log")
            ax.set_xlabel("train_docs (log scale)")
            ax.set_ylabel("bias" if y_key == "bias" else "variance")
            ax.set_title(f"{metric_label} | {y_key}")
            ax.grid(alpha=0.2)

    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Markov Changepoint: Bias/Variance vs Train Docs (relative to leaky)", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if len(files) == 0:
        raise ValueError(f"no files matched: {args.input_glob}")

    runs = _load_runs(files)
    summary, perf = _aggregate(runs)

    out_fig = Path(args.output_figure)
    _plot(summary, output_path=out_fig)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_files": int(len(files)),
        "train_docs_values": sorted(int(x) for x in runs.keys()),
        "metrics": METRIC_SPECS,
        "performance_summary": perf,
        "bias_variance_summary": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_fig}")
    print(f"wrote_json | {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
