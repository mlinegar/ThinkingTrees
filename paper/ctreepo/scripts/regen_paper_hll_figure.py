#!/usr/bin/env python3
"""Regenerate the HLL merge-learning memory figure for the minimal paper.

Replicates the median-plot logic from
`treepo/src/treepo/bench/reports/cardinality.py::_plot_hll_merge_memory_median`
but applies the shared paperplot rcparams and single-column figsize, then
saves both PDF (for \\includegraphics) and PNG (for previews) under
`paper/ctreepo/assets/hll/figures/hll_merge_learning_memory_median.{pdf,png}`.

Usage
-----
    python paper/ctreepo/scripts/regen_paper_hll_figure.py \\
        [--json-summary outputs/.../hll_merge_learning/summary.json] \\
        [--output-stem paper/ctreepo/assets/hll/figures/hll_merge_learning_memory_median]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import paperplot  # noqa: E402


DEFAULT_JSON = Path(
    "outputs/treepo_fullrun_20260306_194657/hll_merge_learning/summary.json"
)
DEFAULT_OUTPUT_STEM = Path(
    "paper/ctreepo/assets/hll/figures/hll_merge_learning_memory_median"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-summary", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--train-docs",
        type=int,
        default=None,
        help="Train-doc count to plot (default: max available).",
    )
    parser.add_argument("--audit-policy", type=str, default="all")
    return parser.parse_args()


def _metric(row: dict, base: str) -> float:
    if f"{base}_mean" in row and row[f"{base}_mean"] not in ("", None):
        return float(row[f"{base}_mean"])
    return float(row[base])


def _load_rows(path: Path) -> list:
    """Prefer per-seed raw rows so medians are actually across seeds."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("raw_rows") or payload.get("aggregated_rows") or payload.get("rows") or []
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def main() -> int:
    args = parse_args()
    rows = _load_rows(args.json_summary)

    train_docs_values = sorted({int(r["train_docs"]) for r in rows})
    train_docs = int(args.train_docs) if args.train_docs is not None else int(max(train_docs_values))
    audit_policy = str(args.audit_policy)

    filt = [
        r
        for r in rows
        if int(r["train_docs"]) == train_docs and str(r["audit_policy"]) == audit_policy
    ]
    if not filt:
        raise ValueError(
            f"no rows for train_docs={train_docs} and audit_policy={audit_policy!r}"
        )

    grouped: dict[int, list[dict]] = {}
    for row in filt:
        grouped.setdefault(int(row["precision"]), []).append(row)

    plot_rows = []
    for precision, grows in sorted(grouped.items()):
        learned_vals = np.asarray([_metric(r, "learned_relative_rmse") for r in grows], dtype=np.float64)
        hll_vals = np.asarray([_metric(r, "hll_relative_rmse") for r in grows], dtype=np.float64)
        plot_rows.append(
            {
                "precision": int(precision),
                "memory_bytes": float(grows[0]["memory_bits"]) / 8.0,
                "learned_median": float(np.median(learned_vals)),
                "hll_median": float(np.median(hll_vals)),
                "hll_rse_theory": float(np.mean([float(r["hll_rse_theory"]) for r in grows])),
                "n_points": int(len(grows)),
            }
        )

    xs = np.asarray([r["precision"] for r in plot_rows], dtype=np.float64)
    learned = np.asarray([r["learned_median"] for r in plot_rows], dtype=np.float64)
    hll = np.asarray([r["hll_median"] for r in plot_rows], dtype=np.float64)
    floor = np.asarray([r["hll_rse_theory"] for r in plot_rows], dtype=np.float64)

    paperplot.rcparams()
    fig, ax = plt.subplots(figsize=paperplot.FIGSIZE_ONE_COL_TALL, constrained_layout=True)

    ax.plot(
        xs,
        floor,
        linestyle=":",
        linewidth=1.1,
        color=paperplot.ANCHOR_COLORS["theory"],
        label=r"Theory floor ($1.04/\sqrt{m}$)",
    )
    ax.plot(
        xs,
        hll,
        linestyle="--",
        linewidth=1.1,
        color=paperplot.ANCHOR_COLORS["baseline"],
        label="Exact HLL",
    )
    ax.plot(
        xs,
        learned,
        marker="o",
        linewidth=1.4,
        markersize=4.0,
        color=paperplot.ANCHOR_COLORS["hll"],
        label="Learned merge (median)",
    )

    ax.set_xlabel(r"Precision $p$  ($m=2^p$ registers)")
    ax.set_ylabel("Relative RMSE")
    ax.set_xticks([int(r["precision"]) for r in plot_rows])
    ax.legend(loc="upper right")

    written = paperplot.save(fig, args.output_stem)
    plt.close(fig)

    for path in written:
        print(f"wrote_figure | {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
