#!/usr/bin/env python3
"""Plot approach-to-floor and schedule invariance vs query budget.

Reads JSON summaries from scripts/run_hll_merge_learning_sweep.py.
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
        description="Plot approach-to-floor curves for HLL merge learning."
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/hll_merge_learning_summary.json",
        help="Sweep JSON summary input path.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=10,
        help="HLL precision p to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/hll_merge_learning_learning_curves.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def _group_by_audit(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(str(r["audit_policy"]), []).append(r)
    for k in out:
        out[k] = sorted(out[k], key=lambda rr: int(rr["train_docs"]))
    return out


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.json_summary).read_text(encoding="utf-8"))
    rows = payload.get("aggregated_rows") or payload.get("rows") or payload.get("raw_rows") or []
    if len(rows) == 0:
        raise ValueError("no rows found in JSON summary")

    agg = "distance_to_hll_floor_rel_rmse_mean" in rows[0]
    if not agg:
        raise ValueError("expected aggregated_rows with *_mean keys (run the sweep script)")

    p = int(args.precision)
    filt = [r for r in rows if int(r["precision"]) == p]
    if len(filt) == 0:
        raise ValueError(f"no rows for precision={p}")

    by_audit = _group_by_audit(filt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    # Panel 1: distance to theory floor vs query budget.
    for audit, srows in sorted(by_audit.items()):
        xs = np.array([float(r["train_total_queries_estimate_mean"]) for r in srows], dtype=np.float64)
        ys = np.array([float(r["distance_to_hll_floor_rel_rmse_mean"]) for r in srows], dtype=np.float64)
        ystd = np.array([float(r["distance_to_hll_floor_rel_rmse_std"]) for r in srows], dtype=np.float64)
        ax1.plot(xs, ys, marker="o", linewidth=1.6, label=audit)
        ax1.fill_between(xs, ys - ystd, ys + ystd, alpha=0.12)
    ax1.axhline(0.0, color="#444444", linewidth=1.1, alpha=0.6)
    ax1.set_xlabel("Training audit queries (estimated total)")
    ax1.set_ylabel("Distance to HLL theory floor (relative RMSE)")
    ax1.set_title("Approach to Theory Floor")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=9)

    # Panel 2: schedule spread vs query budget.
    for audit, srows in sorted(by_audit.items()):
        xs = np.array([float(r["train_total_queries_estimate_mean"]) for r in srows], dtype=np.float64)
        ys = np.array([float(r["learned_schedule_spread_mean_mean"]) for r in srows], dtype=np.float64)
        ystd = np.array([float(r["learned_schedule_spread_mean_std"]) for r in srows], dtype=np.float64)
        ax2.plot(xs, ys, marker="o", linewidth=1.6, label=audit)
        ax2.fill_between(xs, ys - ystd, ys + ystd, alpha=0.12)
    ax2.set_xlabel("Training audit queries (estimated total)")
    ax2.set_ylabel("Schedule spread mean (estimate units)")
    ax2.set_title("Schedule Invariance Emergence")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle(f"HLL Merge Learning Curves | precision p={p}", fontsize=11)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote_figure | {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

