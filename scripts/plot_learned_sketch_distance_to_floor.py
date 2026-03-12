#!/usr/bin/env python3
"""Plot distance-to-theoretical-floor for the learned sketch simulation.

Produces a 2-panel figure:
  Left:  Excess RMSE over theoretical floor (D = E_model - E_floor).
  Right: Ratio to theoretical floor (R = E_model / E_floor).

Supports both single-run JSON (from run_learned_sketch_simulation.py)
and aggregated multi-seed JSON (from run_learned_sketch_sampling_sweep.py),
detected by the presence of an ``aggregated_rows`` key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distance-to-floor for the learned sketch simulation."
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
        default="outputs/learned_sketch_distance_to_floor.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["rmse", "rel_rmse"],
        default="rmse",
        help="Domain for the floor comparison: absolute RMSE or relative RMSE.",
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


def _suffix(key: str, agg: bool) -> str:
    return f"{key}_mean" if agg else key


def _std_key(key: str) -> str:
    return f"{key}_std"


def plot_floor(rows: List[dict], out_path: Path, metric: str) -> None:
    """Two-panel figure: excess-to-floor and ratio-to-floor."""
    agg = "excess_rmse_mean" in rows[0] if rows else False
    by_state = _group_by_state(rows)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    if metric == "rmse":
        excess_key = "excess_rmse"
        ratio_key = "ratio_to_floor_rmse"
        hll_excess_key = "hll_empirical_excess_rmse"
        hll_ratio_source = ("hll_rmse", "theoretical_floor_rmse")
        ylabel_excess = "Excess RMSE over theoretical floor"
        ylabel_ratio = "RMSE / theoretical floor"
    else:
        excess_key = "distance_to_hll_floor_rel_rmse"
        ratio_key = "ratio_to_floor_rel_rmse"
        hll_excess_key = "hll_empirical_excess_rel_rmse"
        hll_ratio_source = ("hll_relative_rmse", "hll_rse_theory")
        ylabel_excess = "Excess relative RMSE over theoretical floor"
        ylabel_ratio = "Relative RMSE / theoretical floor"

    # ---- Left panel: excess-to-floor ----
    for sd, srows in sorted(by_state.items()):
        xs = np.array([int(r["train_size"]) for r in srows], dtype=np.float64)
        ys = np.array(
            [float(r[_suffix(excess_key, agg)]) for r in srows],
            dtype=np.float64,
        )
        line = ax1.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                        label=f"learned d={sd}")[0]
        color = line.get_color()
        if agg and _std_key(excess_key) in srows[0]:
            ystd = np.array(
                [float(r[_std_key(excess_key)]) for r in srows],
                dtype=np.float64,
            )
            ax1.fill_between(xs, ys - ystd, ys + ystd, alpha=0.12, color=color)

        # HLL empirical excess (dashed, color-matched).
        hll_val = float(srows[0][_suffix(hll_excess_key, agg)])
        ax1.hlines(
            hll_val,
            xmin=float(xs.min()), xmax=float(xs.max()),
            colors=color, linestyles="--", alpha=0.4,
        )

    # Theoretical floor at y=0.
    ax1.axhline(0.0, color="#444444", linewidth=1.2, alpha=0.6)
    ax1.text(
        float(xs.max()), 0.0, "  theoretical floor",
        fontsize=8, va="bottom", ha="right", color="#444444", alpha=0.7,
    )
    # Annotate the rightmost point of the last state_dim as "full sampling".
    last_sd = max(by_state.keys())
    last_row = by_state[last_sd][-1]
    ax1.annotate(
        "full sampling",
        xy=(int(last_row["train_size"]),
            float(last_row[_suffix(excess_key, agg)])),
        xytext=(10, 12), textcoords="offset points",
        fontsize=8, alpha=0.7,
        arrowprops=dict(arrowstyle="->", color="#444444", alpha=0.5),
    )

    ax1.set_xlabel("Train documents (oracle queries)")
    ax1.set_ylabel(ylabel_excess)
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=9)

    # ---- Right panel: ratio-to-floor ----
    for sd, srows in sorted(by_state.items()):
        xs = np.array([int(r["train_size"]) for r in srows], dtype=np.float64)
        ys = np.array(
            [float(r[_suffix(ratio_key, agg)]) for r in srows],
            dtype=np.float64,
        )
        line = ax2.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                        label=f"learned d={sd}")[0]
        color = line.get_color()
        if agg and _std_key(ratio_key) in srows[0]:
            ystd = np.array(
                [float(r[_std_key(ratio_key)]) for r in srows],
                dtype=np.float64,
            )
            ax2.fill_between(xs, ys - ystd, ys + ystd, alpha=0.12, color=color)

        # HLL empirical ratio.
        num_key, den_key = hll_ratio_source
        hll_num = float(srows[0][_suffix(num_key, agg)])
        hll_den = float(srows[0].get(
            _suffix(den_key, agg),
            srows[0].get(den_key, 1.0),
        ))
        hll_ratio = hll_num / max(1e-12, hll_den)
        ax2.hlines(
            hll_ratio,
            xmin=float(xs.min()), xmax=float(xs.max()),
            colors=color, linestyles="--", alpha=0.4,
        )

    # Theoretical floor at ratio=1.
    ax2.axhline(1.0, color="#444444", linewidth=1.2, alpha=0.6)
    ax2.text(
        float(xs.max()), 1.0, "  theoretical floor",
        fontsize=8, va="bottom", ha="right", color="#444444", alpha=0.7,
    )

    ax2.set_xlabel("Train documents (oracle queries)")
    ax2.set_ylabel(ylabel_ratio)
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, fontsize=9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote_figure | {out_path}")


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.json_summary).read_text(encoding="utf-8"))

    # Auto-detect aggregated vs single-run format.
    if "aggregated_rows" in payload:
        rows = payload["aggregated_rows"]
    else:
        rows = payload["rows"]

    if len(rows) == 0:
        raise ValueError("no rows in summary")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_floor(rows, out_path, metric=str(args.metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
