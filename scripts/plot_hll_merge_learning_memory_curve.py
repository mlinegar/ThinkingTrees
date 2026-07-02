#!/usr/bin/env python3
"""Plot the theory-linked memory curve for HLL merge-learning sweeps.

Inputs are JSON summaries from scripts/OLD_run_hll_merge_learning_sweep.py (archived).
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
        description="Plot learned merge vs HLL baseline + theory floor as a function of memory."
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/hll_merge_learning_summary.json",
        help="JSON summary input path (sweep output).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/hll_merge_learning_memory_curve.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--train-docs",
        type=int,
        default=None,
        help="Train-doc count to plot (default: max available in summary).",
    )
    parser.add_argument(
        "--audit-policy",
        type=str,
        default="all",
        help="Audit policy to plot (default: all).",
    )
    return parser.parse_args()


def _group_by_precision(rows: List[dict]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for r in rows:
        out.setdefault(int(r["precision"]), []).append(r)
    for p in out:
        out[p] = sorted(out[p], key=lambda rr: int(rr["train_docs"]))
    return out


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.json_summary).read_text(encoding="utf-8"))
    rows = payload.get("aggregated_rows") or payload.get("raw_rows") or payload.get("rows") or []
    if len(rows) == 0:
        raise ValueError("no rows found in JSON summary")

    agg = "learned_relative_rmse_mean" in rows[0]
    rel_key = "learned_relative_rmse_mean" if agg else "learned_relative_rmse"
    rel_std_key = "learned_relative_rmse_std" if agg else None
    hll_key = "hll_relative_rmse_mean" if agg else "hll_relative_rmse"
    hll_std_key = "hll_relative_rmse_std" if agg else None

    train_docs_values = sorted({int(r["train_docs"]) for r in rows})
    train_docs = int(args.train_docs) if args.train_docs is not None else int(max(train_docs_values))
    audit_policy = str(args.audit_policy)

    filt = [
        r
        for r in rows
        if int(r["train_docs"]) == train_docs and str(r["audit_policy"]) == audit_policy
    ]
    if len(filt) == 0:
        raise ValueError(
            f"no rows for train_docs={train_docs} and audit_policy={audit_policy!r}"
        )

    by_p = _group_by_precision(filt)
    ps = sorted(by_p.keys())
    xs = np.array([float(by_p[p][0]["memory_bytes"]) for p in ps], dtype=np.float64)

    y_learned = np.array([float(by_p[p][0][rel_key]) for p in ps], dtype=np.float64)
    y_floor = np.array([float(by_p[p][0]["hll_rse_theory"]) for p in ps], dtype=np.float64)
    y_hll = np.array([float(by_p[p][0][hll_key]) for p in ps], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)

    # Theory floor.
    ax.plot(xs, y_floor, linestyle=":", color="#444444", linewidth=1.7, label="HLL theory floor")

    # HLL baseline (empirical).
    ax.plot(xs, y_hll, linestyle="--", color="#888888", linewidth=1.4, label="HLL empirical")
    if hll_std_key is not None and hll_std_key in by_p[ps[0]][0]:
        ystd = np.array([float(by_p[p][0][hll_std_key]) for p in ps], dtype=np.float64)
        ax.fill_between(xs, y_hll - ystd, y_hll + ystd, color="#888888", alpha=0.10)

    # Learned merger.
    ax.plot(xs, y_learned, marker="o", linewidth=1.8, label="learned merge")
    if rel_std_key is not None and rel_std_key in by_p[ps[0]][0]:
        ystd = np.array([float(by_p[p][0][rel_std_key]) for p in ps], dtype=np.float64)
        ax.fill_between(xs, y_learned - ystd, y_learned + ystd, alpha=0.15)

    for p, x, y in zip(ps, xs, y_learned):
        ax.text(x, y, f" p={p}", fontsize=8, alpha=0.75, ha="left", va="center")

    ax.set_xlabel("HLL memory (bytes)")
    ax.set_ylabel("Relative RMSE")
    ax.set_title(f"HLL Merge Learning | train_docs={train_docs}, audit={audit_policy}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(f"wrote_figure | {out_path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

