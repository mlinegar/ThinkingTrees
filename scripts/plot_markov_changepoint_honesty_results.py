#!/usr/bin/env python3
"""Aggregate and plot Markov changepoint honesty sweep results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fixed", "chunker_honest", "chunker_leaky")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate/plot Markov changepoint honesty sweep results."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_overnight/seed_*.json",
        help="Glob for per-seed JSON outputs from run_markov_changepoint_honesty_simulation.py.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_overnight_summary.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/markov_changepoint_overnight_rows.csv",
        help="Output CSV with one row per seed/policy.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_overnight_report.json",
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def _seed_from_path(path: Path) -> int:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"cannot parse seed from filename: {path}")
    return int(stem.split("_")[-1])


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _collect_rows(files: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = _seed_from_path(path)
        for policy in POLICIES:
            m = payload["metrics"][policy]
            rows.append(
                {
                    "seed": int(seed),
                    "policy": str(policy),
                    "boundary_f1": float(m["boundary_f1"]),
                    "boundary_precision": float(m["boundary_precision"]),
                    "boundary_recall": float(m["boundary_recall"]),
                    "mean_boundary_cost": float(m["mean_boundary_cost"]),
                    "mean_l1": float(m["mean_l1"]),
                    "mean_kl": float(m["mean_kl"]),
                    "mean_loglik_drop": float(m["mean_loglik_drop"]),
                    "mean_num_boundaries": float(m["mean_num_boundaries"]),
                    "predicted_to_true_ratio": float(m["predicted_to_true_ratio"]),
                }
            )
    rows.sort(key=lambda r: (int(r["seed"]), str(r["policy"])))
    return rows


def _build_report(rows: List[Dict[str, object]]) -> Dict[str, object]:
    seeds = sorted({int(r["seed"]) for r in rows})

    by_policy: Dict[str, Dict[str, List[float]]] = {}
    for p in POLICIES:
        p_rows = [r for r in rows if r["policy"] == p]
        by_policy[p] = {
            "boundary_f1": [float(r["boundary_f1"]) for r in p_rows],
            "mean_boundary_cost": [float(r["mean_boundary_cost"]) for r in p_rows],
            "mean_l1": [float(r["mean_l1"]) for r in p_rows],
            "mean_num_boundaries": [float(r["mean_num_boundaries"]) for r in p_rows],
        }

    policy_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for p in POLICIES:
        policy_summary[p] = {}
        for metric, values in by_policy[p].items():
            policy_summary[p][metric] = _summary_stats(values)

    per_seed = {}
    for s in seeds:
        r_fixed = next(r for r in rows if int(r["seed"]) == s and r["policy"] == "fixed")
        r_hon = next(r for r in rows if int(r["seed"]) == s and r["policy"] == "chunker_honest")
        r_leak = next(r for r in rows if int(r["seed"]) == s and r["policy"] == "chunker_leaky")
        per_seed[str(s)] = {
            "fixed_f1": float(r_fixed["boundary_f1"]),
            "honest_f1": float(r_hon["boundary_f1"]),
            "leaky_f1": float(r_leak["boundary_f1"]),
            "fixed_cost": float(r_fixed["mean_boundary_cost"]),
            "honest_cost": float(r_hon["mean_boundary_cost"]),
            "leaky_cost": float(r_leak["mean_boundary_cost"]),
            "delta_f1_honest_minus_fixed": float(r_hon["boundary_f1"]) - float(r_fixed["boundary_f1"]),
            "delta_f1_leaky_minus_honest": float(r_leak["boundary_f1"]) - float(r_hon["boundary_f1"]),
            "delta_cost_fixed_minus_honest": float(r_fixed["mean_boundary_cost"]) - float(r_hon["mean_boundary_cost"]),
            "delta_cost_honest_minus_leaky": float(r_hon["mean_boundary_cost"]) - float(r_leak["mean_boundary_cost"]),
        }

    f1_order_theory = sum(
        1
        for s in seeds
        if per_seed[str(s)]["leaky_f1"] >= per_seed[str(s)]["honest_f1"] >= per_seed[str(s)]["fixed_f1"]
    )
    cost_order_theory = sum(
        1
        for s in seeds
        if per_seed[str(s)]["leaky_cost"] <= per_seed[str(s)]["honest_cost"] <= per_seed[str(s)]["fixed_cost"]
    )

    return {
        "n_seeds": int(len(seeds)),
        "seeds": [int(s) for s in seeds],
        "policy_summary": policy_summary,
        "ordering_checks": {
            "f1_leaky_ge_honest_ge_fixed": {
                "passed": int(f1_order_theory),
                "total": int(len(seeds)),
            },
            "cost_leaky_le_honest_le_fixed": {
                "passed": int(cost_order_theory),
                "total": int(len(seeds)),
            },
        },
        "per_seed": per_seed,
    }


def _plot(rows: List[Dict[str, object]], out_path: Path) -> None:
    seeds = sorted({int(r["seed"]) for r in rows})
    x = np.asarray(seeds, dtype=np.int64)

    def yvals(policy: str, key: str) -> np.ndarray:
        arr = [float(r[key]) for r in rows if r["policy"] == policy]
        return np.asarray(arr, dtype=np.float64)

    yf = yvals("fixed", "boundary_f1")
    yh = yvals("chunker_honest", "boundary_f1")
    yl = yvals("chunker_leaky", "boundary_f1")

    cf = yvals("fixed", "mean_boundary_cost")
    ch = yvals("chunker_honest", "mean_boundary_cost")
    cl = yvals("chunker_leaky", "mean_boundary_cost")

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(x, yf, marker="o", linewidth=1.3, label="fixed", color="#555555")
    ax.plot(x, yh, marker="o", linewidth=1.3, label="chunker_honest", color="#1f77b4")
    ax.plot(x, yl, marker="o", linewidth=1.3, label="chunker_leaky", color="#d62728")
    # Theoretical baseline: fixed-policy mean.
    ax.axhline(float(np.mean(yf)), linestyle=":", linewidth=1.8, color="#555555", alpha=0.85)
    ax.set_title("Boundary F1 by Seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("F1")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[0, 1]
    ax.plot(x, cf, marker="o", linewidth=1.3, label="fixed", color="#555555")
    ax.plot(x, ch, marker="o", linewidth=1.3, label="chunker_honest", color="#1f77b4")
    ax.plot(x, cl, marker="o", linewidth=1.3, label="chunker_leaky", color="#d62728")
    # Theoretical baseline: fixed-policy mean.
    ax.axhline(float(np.mean(cf)), linestyle=":", linewidth=1.8, color="#555555", alpha=0.85)
    ax.set_title("Boundary Cost by Seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean boundary cost")
    ax.grid(alpha=0.2)

    ax = axs[1, 0]
    d_hf = yh - yf
    d_lh = yl - yh
    ax.plot(x, d_hf, marker="o", linewidth=1.3, label="honest - fixed (F1)", color="#1f77b4")
    ax.plot(x, d_lh, marker="o", linewidth=1.3, label="leaky - honest (F1)", color="#d62728")
    # Theoretical no-change baseline.
    ax.axhline(0.0, linestyle=":", linewidth=1.8, color="#444444", alpha=0.9)
    ax.set_title("F1 Deltas")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Delta")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[1, 1]
    gain_hf = cf - ch
    gain_hl = ch - cl
    ax.plot(x, gain_hf, marker="o", linewidth=1.3, label="fixed - honest (cost gain)", color="#1f77b4")
    ax.plot(x, gain_hl, marker="o", linewidth=1.3, label="honest - leaky (cost gain)", color="#d62728")
    # Theoretical no-change baseline.
    ax.axhline(0.0, linestyle=":", linewidth=1.8, color="#444444", alpha=0.9)
    ax.set_title("Cost Deltas (positive is better)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Delta")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Markov Changepoint Honesty Sweep Summary", fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if len(files) == 0:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(files)
    report = _build_report(rows)

    csv_path = Path(args.output_csv)
    _write_rows_csv(csv_path, rows)

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    fig_path = Path(args.output_figure)
    _plot(rows, fig_path)

    print(f"wrote_csv | {csv_path}")
    print(f"wrote_json | {json_path}")
    print(f"wrote_figure | {fig_path}")
    print(
        "ordering_f1_leaky_ge_honest_ge_fixed | "
        f"{report['ordering_checks']['f1_leaky_ge_honest_ge_fixed']['passed']}"
        f"/{report['ordering_checks']['f1_leaky_ge_honest_ge_fixed']['total']}"
    )
    print(
        "ordering_cost_leaky_le_honest_le_fixed | "
        f"{report['ordering_checks']['cost_leaky_le_honest_le_fixed']['passed']}"
        f"/{report['ordering_checks']['cost_leaky_le_honest_le_fixed']['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
