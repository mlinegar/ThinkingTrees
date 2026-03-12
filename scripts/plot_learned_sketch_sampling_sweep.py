#!/usr/bin/env python3
"""Plot multi-seed learned-sketch sampling sweep summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregated multi-seed learned-sketch sampling sweeps from "
            "run_learned_sketch_sampling_sweep.py JSON output."
        )
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/learned_sketch_sampling_sweep_summary.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/learned_sketch_sampling_sweep.png",
    )
    return parser.parse_args()


def _group_by_state(rows: List[dict]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for row in rows:
        state_dim = int(row["state_dim"])
        out.setdefault(state_dim, []).append(row)
    for state_dim in out:
        out[state_dim] = sorted(out[state_dim], key=lambda r: int(r["train_size"]))
    return out


def _best_rows_by_train_size(rows: List[dict]) -> List[dict]:
    train_sizes = sorted({int(r["train_size"]) for r in rows})
    out: List[dict] = []
    for train_size in train_sizes:
        cands = [r for r in rows if int(r["train_size"]) == train_size]
        if len(cands) == 0:
            continue
        best = min(
            cands,
            key=lambda r: (
                max(0.0, float(r["distance_to_hll_floor_rel_rmse_mean"])),
                abs(float(r["distance_to_hll_floor_rel_rmse_mean"])),
                float(r["learned_relative_rmse_mean"]),
            ),
        )
        out.append(best)
    return out


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.json_summary).read_text(encoding="utf-8"))
    rows = payload.get("aggregated_rows", [])
    if len(rows) == 0:
        raise ValueError("No aggregated_rows found in JSON summary")

    by_state = _group_by_state(rows)
    best_rows = _best_rows_by_train_size(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    # Panel 1: learned relative RMSE vs train-size with std band.
    ax = axes[0]
    for state_dim, srows in sorted(by_state.items()):
        xs = np.array([int(r["train_size"]) for r in srows], dtype=np.float64)
        ys = np.array([float(r["learned_relative_rmse_mean"]) for r in srows], dtype=np.float64)
        ystd = np.array([float(r["learned_relative_rmse_std"]) for r in srows], dtype=np.float64)
        ax.plot(xs, ys, marker="o", label=f"learned d={state_dim}")
        ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.15)

    for state_dim, srows in sorted(by_state.items()):
        if len(srows) == 0:
            continue
        y = float(srows[0]["hll_relative_rmse_mean"])
        y_floor = float(srows[0]["hll_rse_theory_mean"])
        x_min = min(int(r["train_size"]) for r in srows)
        x_max = max(int(r["train_size"]) for r in srows)
        bits = int(srows[0]["hll_memory_bits"])
        ax.hlines(y, xmin=x_min, xmax=x_max, colors="gray", linestyles="--", alpha=0.35)
        ax.hlines(y_floor, xmin=x_min, xmax=x_max, colors="gray", linestyles=":", alpha=0.55)
        ax.text(x_max, y, f" HLL({bits}b)", fontsize=8, va="center", ha="left", color="gray")
        ax.text(x_max, y_floor, " floor", fontsize=8, va="center", ha="left", color="gray")

    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Relative RMSE")
    ax.set_title("Relative RMSE (mean±std)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: primary metric (distance to theory floor) vs train-size.
    ax = axes[1]
    for state_dim, srows in sorted(by_state.items()):
        xs = np.array([int(r["train_size"]) for r in srows], dtype=np.float64)
        ys = np.array([float(r["distance_to_hll_floor_rel_rmse_mean"]) for r in srows], dtype=np.float64)
        ystd = np.array([float(r["distance_to_hll_floor_rel_rmse_std"]) for r in srows], dtype=np.float64)
        ax.plot(xs, ys, marker="o", label=f"d={state_dim}")
        ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.15)
    ax.axhline(0.0, color="gray", linewidth=1, alpha=0.5)
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Distance to HLL Theory Floor")
    ax.set_title("Primary Metric: Excess Relative RMSE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Panel 3: Audit geometry (aggregated).
    ax = axes[2]
    ref_state = sorted(by_state.keys())[0]
    geom_rows = by_state[ref_state]
    xs = np.array([int(r["train_size"]) for r in geom_rows], dtype=np.float64)
    mean_internal = np.array(
        [float(r["train_mean_internal_nodes_mean"]) for r in geom_rows],
        dtype=np.float64,
    )
    mean_audit = np.array(
        [float(r["train_audit_nodes_mean_mean"]) for r in geom_rows],
        dtype=np.float64,
    )
    coverage = np.array(
        [float(r["train_audit_coverage_mean_mean"]) for r in geom_rows],
        dtype=np.float64,
    )
    ax.plot(xs, mean_internal, marker="o", label="mean internal nodes/doc")
    ax.plot(xs, mean_audit, marker="s", label="mean audited nodes/doc")
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Nodes / Document")
    ax.set_title("Audit Geometry (mean over seeds)")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(xs, coverage, marker="^", linestyle="--", color="tab:green", label="audit coverage")
    ax2.set_ylabel("Audit Coverage")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, fontsize=8, loc="lower right")

    cfg = payload.get("config", {})
    runtime = float(payload.get("runtime_seconds_total", 0.0))
    subtitle = (
        f"state_dims={cfg.get('state_dims', [])} | train_sizes={cfg.get('train_sizes', [])} | "
        f"epochs={cfg.get('n_epochs', 'n/a')} | seeds={payload.get('seeds', [])} | "
        f"runtime={runtime:.1f}s"
    )
    fig.suptitle(f"Aggressive Learned Sketch Sampling Sweep | {subtitle}", fontsize=11)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote_figure | {out}")

    if len(best_rows) > 0:
        gaps = [float(r["distance_to_hll_floor_rel_rmse_mean"]) for r in best_rows]
        print(
            "best_frontier_floor_summary | mean_dist | median_dist | min_dist | max_dist"
        )
        print(
            f"{statistics.mean(gaps):+.6f} | {statistics.median(gaps):+.6f} | "
            f"{min(gaps):+.6f} | {max(gaps):+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
