#!/usr/bin/env python3
"""Plot learned-sketch simulation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot learned sketch vs HLL results from run_learned_sketch_simulation.py JSON output."
        )
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/learned_sketch_simulation_summary.json",
        help="JSON summary input path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/learned_sketch_simulation.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def _group_by_state(rows: List[dict]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for row in rows:
        sd = int(row["state_dim"])
        out.setdefault(sd, []).append(row)
    for sd in out:
        out[sd] = sorted(out[sd], key=lambda r: int(r["train_size"]))
    return out


def _best_rows_by_train_size(rows: List[dict]) -> List[dict]:
    train_sizes = sorted({int(r["train_size"]) for r in rows})
    best: List[dict] = []
    for n in train_sizes:
        candidates = [r for r in rows if int(r["train_size"]) == n]
        if len(candidates) == 0:
            continue
        candidates = sorted(
            candidates,
            key=lambda r: (
                max(0.0, float(r["distance_to_hll_floor_rel_rmse"])),
                abs(float(r["distance_to_hll_floor_rel_rmse"])),
                float(r["learned_relative_rmse"]),
            ),
        )
        best.append(candidates[0])
    return best


def main() -> int:
    args = parse_args()
    path = Path(args.json_summary)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if len(rows) == 0:
        raise ValueError("no rows in summary")

    by_state = _group_by_state(rows)
    best_rows = _best_rows_by_train_size(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    # Panel 1: Relative RMSE vs train size with HLL empirical + theory floors.
    ax = axes[0]
    for color_idx, (sd, srows) in enumerate(sorted(by_state.items())):
        xs = [int(r["train_size"]) for r in srows]
        ys = [float(r["learned_relative_rmse"]) for r in srows]
        line = ax.plot(xs, ys, marker="o", label=f"learned d={sd}")[0]
        color = line.get_color()
        y_hll_emp = float(srows[0]["hll_relative_rmse"])
        y_hll_theory = float(srows[0]["hll_rse_theory"])
        xmin = min(xs)
        xmax = max(xs)
        ax.hlines(
            y_hll_emp,
            xmin=xmin,
            xmax=xmax,
            colors=color,
            linestyles="--",
            alpha=0.35,
            label=f"HLL emp d={sd}" if color_idx == 0 else None,
        )
        ax.hlines(
            y_hll_theory,
            xmin=xmin,
            xmax=xmax,
            colors=color,
            linestyles=":",
            alpha=0.65,
            label=f"HLL theory d={sd}" if color_idx == 0 else None,
        )
        ax.text(
            xmax,
            y_hll_theory,
            f" floor({int(srows[0]['hll_memory_bits'])}b)",
            fontsize=8,
            va="center",
            ha="left",
            color=color,
            alpha=0.8,
        )
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Relative RMSE")
    ax.set_title("Relative RMSE vs HLL Floors")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: Distance to theoretical floor (primary metric).
    ax = axes[1]
    for sd, srows in sorted(by_state.items()):
        xs = [int(r["train_size"]) for r in srows]
        ys = [float(r["distance_to_hll_floor_rel_rmse"]) for r in srows]
        ax.plot(xs, ys, marker="o", label=f"d={sd}")
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Distance to HLL Theory Floor")
    ax.set_title("Primary Metric: Excess Relative RMSE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # Panel 3: Audit geometry (sampled nodes vs internal nodes + coverage).
    ax = axes[2]
    ref_state = sorted(by_state.keys())[0]
    geom_rows = by_state[ref_state]
    xs = np.array([int(r["train_size"]) for r in geom_rows], dtype=np.float64)
    mean_internal = np.array(
        [float(r["train_mean_internal_nodes"]) for r in geom_rows],
        dtype=np.float64,
    )
    mean_audit = np.array(
        [float(r["train_audit_nodes_mean"]) for r in geom_rows],
        dtype=np.float64,
    )
    coverage = np.array(
        [float(r["train_audit_coverage_mean"]) for r in geom_rows],
        dtype=np.float64,
    )
    ax.plot(xs, mean_internal, marker="o", label="mean internal nodes/doc")
    ax.plot(xs, mean_audit, marker="s", label="mean audited nodes/doc")
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Nodes / Document")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(xs, coverage, marker="^", linestyle="--", color="tab:green", label="audit coverage")
    ax2.set_ylabel("Audit Coverage")
    ax.set_title("Audit Geometry")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, fontsize=8, loc="lower right")

    runtime = payload.get("runtime_config", {})
    subtitle = (
        f"device={runtime.get('device_used', 'unknown')} | "
        f"leaf_size={runtime.get('leaf_size', 'n/a')} | "
        f"epochs={runtime.get('n_epochs', 'n/a')}"
    )
    fig.suptitle(f"Learned Mergeable Sketch Simulation | {subtitle}", fontsize=11)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote_figure | {out}")

    # Compact textual diagnostics.
    if len(best_rows) > 0:
        best_gaps = [float(r["distance_to_hll_floor_rel_rmse"]) for r in best_rows]
        print(
            "best_by_train | mean_dist_to_floor | median_dist_to_floor | "
            "max_dist_to_floor"
        )
        print(
            f"{statistics.mean(best_gaps):+.6f} | "
            f"{statistics.median(best_gaps):+.6f} | {max(best_gaps):+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
