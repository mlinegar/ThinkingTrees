#!/usr/bin/env python3
"""Policy-centric plots for Markov changepoint honesty sweep outputs."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List, Optional

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
        description="Create policy-centric changepoint honesty visualizations."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_overnight/seed_*.json",
        help="Glob for per-seed JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_policy_view.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_policy_view_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--baseline-f1",
        type=float,
        default=None,
        help="Optional theoretical/reference baseline for boundary F1 (dotted line).",
    )
    parser.add_argument(
        "--baseline-cost",
        type=float,
        default=None,
        help="Optional theoretical/reference baseline for boundary cost (dotted line).",
    )
    parser.add_argument(
        "--baseline-l1",
        type=float,
        default=None,
        help="Optional theoretical/reference baseline for L1 distortion (dotted line).",
    )
    return parser.parse_args()


def _seed_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _collect_rows(files: List[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = _seed_from_path(path)
        for policy in POLICIES:
            m = payload["metrics"][policy]
            rows.append(
                {
                    "seed": float(seed),
                    "policy": str(policy),
                    "boundary_f1": float(m["boundary_f1"]),
                    "mean_boundary_cost": float(m["mean_boundary_cost"]),
                    "mean_l1": float(m["mean_l1"]),
                    "predicted_to_true_ratio": float(m["predicted_to_true_ratio"]),
                }
            )
    rows.sort(key=lambda r: (int(r["seed"]), str(r["policy"])))
    return rows


def _mean_std_ci(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    n = len(values)
    mean = float(fmean(values))
    std = float(pstdev(values))
    ci95 = 1.96 * std / math.sqrt(max(1, n))
    return {"n": int(n), "mean": mean, "std": std, "ci95": float(ci95)}


def _group(rows: List[Dict[str, float]], metric: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for p in POLICIES:
        out[p] = [float(r[metric]) for r in rows if r["policy"] == p]
    return out


def _seed_value(rows: List[Dict[str, float]], seed: int, policy: str, metric: str) -> float:
    for r in rows:
        if int(r["seed"]) == int(seed) and r["policy"] == policy:
            return float(r[metric])
    raise KeyError((seed, policy, metric))


def _build_report(rows: List[Dict[str, float]]) -> Dict[str, object]:
    metrics = ("boundary_f1", "mean_boundary_cost", "mean_l1", "predicted_to_true_ratio")
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for metric in metrics:
        grouped = _group(rows, metric)
        summary[metric] = {p: _mean_std_ci(vals) for p, vals in grouped.items()}

    seeds = sorted({int(r["seed"]) for r in rows})
    deltas = {
        "boundary_f1": {
            "honest_minus_fixed": [
                _seed_value(rows, s, "chunker_honest", "boundary_f1")
                - _seed_value(rows, s, "fixed", "boundary_f1")
                for s in seeds
            ],
            "leaky_minus_honest": [
                _seed_value(rows, s, "chunker_leaky", "boundary_f1")
                - _seed_value(rows, s, "chunker_honest", "boundary_f1")
                for s in seeds
            ],
        },
        "mean_boundary_cost": {
            "fixed_minus_honest": [
                _seed_value(rows, s, "fixed", "mean_boundary_cost")
                - _seed_value(rows, s, "chunker_honest", "mean_boundary_cost")
                for s in seeds
            ],
            "honest_minus_leaky": [
                _seed_value(rows, s, "chunker_honest", "mean_boundary_cost")
                - _seed_value(rows, s, "chunker_leaky", "mean_boundary_cost")
                for s in seeds
            ],
        },
    }

    delta_summary = {
        metric: {name: _mean_std_ci(vals) for name, vals in by_name.items()}
        for metric, by_name in deltas.items()
    }

    ordering = {
        "f1_leaky_ge_honest": int(
            sum(
                1
                for s in seeds
                if _seed_value(rows, s, "chunker_leaky", "boundary_f1")
                >= _seed_value(rows, s, "chunker_honest", "boundary_f1")
            )
        ),
        "cost_honest_le_leaky": int(
            sum(
                1
                for s in seeds
                if _seed_value(rows, s, "chunker_honest", "mean_boundary_cost")
                <= _seed_value(rows, s, "chunker_leaky", "mean_boundary_cost")
            )
        ),
        "n_seeds": int(len(seeds)),
    }

    return {
        "n_rows": int(len(rows)),
        "n_seeds": int(len(seeds)),
        "metrics_summary": summary,
        "delta_summary": delta_summary,
        "ordering": ordering,
    }


def _plot_metric_panel(
    ax: plt.Axes,
    rows: List[Dict[str, float]],
    *,
    metric: str,
    ylabel: str,
    baseline: float,
    baseline_label: str,
) -> None:
    x_pos = np.arange(len(POLICIES), dtype=np.float64)

    for i, policy in enumerate(POLICIES):
        vals = np.asarray([float(r[metric]) for r in rows if r["policy"] == policy], dtype=np.float64)
        jitter = np.linspace(-0.12, 0.12, num=len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full_like(vals, x_pos[i]) + jitter,
            vals,
            s=14,
            alpha=0.35,
            color=POLICY_COLOR[policy],
            edgecolors="none",
            zorder=2,
        )
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ci95 = 1.96 * std / math.sqrt(max(1, len(vals)))
        ax.errorbar(
            [x_pos[i]],
            [mean],
            yerr=[ci95],
            fmt="o",
            markersize=7,
            color=POLICY_COLOR[policy],
            capsize=4,
            linewidth=1.6,
            zorder=3,
        )

    ax.axhline(float(baseline), linestyle=":", linewidth=1.9, color="#333333", alpha=0.9)
    ax.text(
        0.01,
        0.98,
        baseline_label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#333333",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([POLICY_LABEL[p] for p in POLICIES])
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, axis="y")


def _plot_delta_panel(ax: plt.Axes, rows: List[Dict[str, float]]) -> None:
    seeds = sorted({int(r["seed"]) for r in rows})
    d_f1 = np.asarray(
        [
            _seed_value(rows, s, "chunker_honest", "boundary_f1")
            - _seed_value(rows, s, "fixed", "boundary_f1")
            for s in seeds
        ],
        dtype=np.float64,
    )
    d_cost = np.asarray(
        [
            _seed_value(rows, s, "fixed", "mean_boundary_cost")
            - _seed_value(rows, s, "chunker_honest", "mean_boundary_cost")
            for s in seeds
        ],
        dtype=np.float64,
    )

    # Normalize to z-scores so both deltas share one y-axis.
    d_f1_z = (d_f1 - float(np.mean(d_f1))) / (float(np.std(d_f1)) + 1e-12)
    d_cost_z = (d_cost - float(np.mean(d_cost))) / (float(np.std(d_cost)) + 1e-12)

    ax.scatter(d_f1, d_cost, s=18, alpha=0.55, color="#2ca02c", edgecolors="none")
    ax.axvline(0.0, linestyle=":", linewidth=1.8, color="#444444", alpha=0.9)
    ax.axhline(0.0, linestyle=":", linewidth=1.8, color="#444444", alpha=0.9)
    ax.set_xlabel("honest - fixed (F1)")
    ax.set_ylabel("fixed - honest (cost gain)")
    ax.set_title("Honest vs Fixed Delta Scatter")
    ax.grid(alpha=0.2)


def _plot_tradeoff_panel(ax: plt.Axes, rows: List[Dict[str, float]], baseline_f1: float, baseline_cost: float) -> None:
    for p in POLICIES:
        xs = np.asarray([float(r["boundary_f1"]) for r in rows if r["policy"] == p], dtype=np.float64)
        ys = np.asarray([float(r["mean_boundary_cost"]) for r in rows if r["policy"] == p], dtype=np.float64)
        ax.scatter(xs, ys, s=18, alpha=0.35, color=POLICY_COLOR[p], edgecolors="none", label=POLICY_LABEL[p])
        ax.scatter([float(np.mean(xs))], [float(np.mean(ys))], s=120, marker="*", color=POLICY_COLOR[p], edgecolors="black", linewidths=0.4)

    ax.axvline(float(baseline_f1), linestyle=":", linewidth=1.9, color="#333333", alpha=0.9)
    ax.axhline(float(baseline_cost), linestyle=":", linewidth=1.9, color="#333333", alpha=0.9)
    ax.set_xlabel("Boundary F1")
    ax.set_ylabel("Mean boundary cost")
    ax.set_title("Policy Tradeoff (F1 vs Cost)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if len(files) == 0:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(files)
    report = _build_report(rows)

    fixed_f1_mean = report["metrics_summary"]["boundary_f1"]["fixed"]["mean"]
    fixed_cost_mean = report["metrics_summary"]["mean_boundary_cost"]["fixed"]["mean"]
    fixed_l1_mean = report["metrics_summary"]["mean_l1"]["fixed"]["mean"]

    baseline_f1 = float(args.baseline_f1) if args.baseline_f1 is not None else float(fixed_f1_mean)
    baseline_cost = float(args.baseline_cost) if args.baseline_cost is not None else float(fixed_cost_mean)
    baseline_l1 = float(args.baseline_l1) if args.baseline_l1 is not None else float(fixed_l1_mean)

    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)

    _plot_metric_panel(
        axs[0, 0],
        rows,
        metric="boundary_f1",
        ylabel="Boundary F1",
        baseline=baseline_f1,
        baseline_label="dotted: baseline F1",
    )
    axs[0, 0].set_title("Boundary F1 by Policy (seed distribution + mean/CI)")

    _plot_metric_panel(
        axs[0, 1],
        rows,
        metric="mean_boundary_cost",
        ylabel="Mean boundary cost",
        baseline=baseline_cost,
        baseline_label="dotted: baseline cost",
    )
    axs[0, 1].set_title("Boundary Cost by Policy (seed distribution + mean/CI)")

    _plot_tradeoff_panel(axs[1, 0], rows, baseline_f1=baseline_f1, baseline_cost=baseline_cost)
    _plot_delta_panel(axs[1, 1], rows)

    fig.suptitle("Markov Changepoint Honesty: Policy-Centric View", fontsize=12)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_fig}")
    print(f"wrote_json | {out_json}")
    print(
        "means | f1 fixed={:.6f} honest={:.6f} leaky={:.6f} | "
        "cost fixed={:.6f} honest={:.6f} leaky={:.6f}".format(
            report["metrics_summary"]["boundary_f1"]["fixed"]["mean"],
            report["metrics_summary"]["boundary_f1"]["chunker_honest"]["mean"],
            report["metrics_summary"]["boundary_f1"]["chunker_leaky"]["mean"],
            report["metrics_summary"]["mean_boundary_cost"]["fixed"]["mean"],
            report["metrics_summary"]["mean_boundary_cost"]["chunker_honest"]["mean"],
            report["metrics_summary"]["mean_boundary_cost"]["chunker_leaky"]["mean"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
