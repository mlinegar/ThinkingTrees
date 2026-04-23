#!/usr/bin/env python3
"""Plot Markov local-coverage frontiers from summarized sweep outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_float


GOOD_GREEN = "#2E8B57"
ADDITIVE_GREEN = "#1B6E3C"
RED = "#C0392B"
LEAF_COLORS = {
    8: "#1f77b4",
    12: "#17becf",
    16: "#2ca02c",
    24: "#ff7f0e",
    32: "#d62728",
    48: "#9467bd",
    64: "#8c564b",
}


_safe_float = safe_float


def _load_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend([row for row in payload.get("rows", []) if isinstance(row, dict)])
    return rows


def _median(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(np.median(arr))


def _group_rows(rows: Iterable[dict], metric_name: str) -> Dict[Tuple[object, ...], List[dict]]:
    groups: Dict[Tuple[object, ...], List[dict]] = {}
    for row in rows:
        key = (
            row.get("model_family"),
            row.get("fixed_leaf_tokens"),
            row.get("train_docs"),
            row.get("access_pattern"),
            round(_safe_float(row.get("root_query_rate")), 6),
            round(_safe_float(row.get("local_oracle_coverage")), 6),
        )
        groups.setdefault(key, []).append(row)
    return groups


def _aggregate_rows(rows: List[dict], metric_name: str) -> List[dict]:
    grouped = _group_rows(rows, metric_name)
    out: List[dict] = []
    for key, group in sorted(grouped.items()):
        out.append(
            {
                "model_family": key[0],
                "fixed_leaf_tokens": key[1],
                "train_docs": key[2],
                "access_pattern": key[3],
                "root_query_rate": key[4],
                "local_oracle_coverage": key[5],
                metric_name: _median([_safe_float(row.get(metric_name)) for row in group]),
                "n_rows": len(group),
            }
        )
    return out


def _filter_values(rows: List[dict], *, root_query_rate: float, access_patterns: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        if abs(_safe_float(row.get("root_query_rate")) - float(root_query_rate)) > 1e-9:
            continue
        if access_patterns and str(row.get("access_pattern")) not in set(access_patterns):
            continue
        out.append(row)
    return out


def _sorted_unique(rows: Iterable[dict], key: str) -> List[object]:
    vals = []
    seen = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        vals.append(value)
    try:
        return sorted(vals)
    except Exception:
        return vals


def _plot_metric_panels(
    rows: List[dict],
    *,
    metric_name: str,
    root_query_rate: float,
    access_patterns: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    filtered = _filter_values(rows, root_query_rate=root_query_rate, access_patterns=access_patterns)
    if not filtered:
        raise ValueError("no rows for requested filter")
    agg = _aggregate_rows(filtered, metric_name)
    train_docs_values = [int(x) for x in _sorted_unique(agg, "train_docs")]
    access_vals = [str(x) for x in _sorted_unique(agg, "access_pattern")]
    if access_patterns:
        access_vals = [x for x in access_patterns if x in set(access_vals)]
    fig, axes = plt.subplots(
        len(access_vals),
        len(train_docs_values),
        figsize=(4.6 * len(train_docs_values), 3.7 * len(access_vals)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    model_styles = {"neural": "-", "additive": "--"}
    model_labels = {"neural": "neural", "additive": "additive ceiling"}

    for r, access_pattern in enumerate(access_vals):
        for c, train_docs in enumerate(train_docs_values):
            ax = axes[r][c]
            panel_rows = [
                row
                for row in agg
                if str(row.get("access_pattern")) == access_pattern and int(row.get("train_docs")) == int(train_docs)
            ]
            for model_family in ["neural", "additive"]:
                family_rows = [row for row in panel_rows if str(row.get("model_family")) == model_family]
                if not family_rows:
                    continue
                leaf_vals = [int(x) for x in _sorted_unique(family_rows, "fixed_leaf_tokens")]
                for leaf_tokens in leaf_vals:
                    leaf_rows = [row for row in family_rows if int(row.get("fixed_leaf_tokens")) == int(leaf_tokens)]
                    leaf_rows.sort(key=lambda row: float(row.get("local_oracle_coverage")))
                    xs = [float(row.get("local_oracle_coverage")) for row in leaf_rows]
                    ys = [float(row.get(metric_name)) for row in leaf_rows]
                    color = LEAF_COLORS.get(int(leaf_tokens), GOOD_GREEN if model_family == "neural" else ADDITIVE_GREEN)
                    label = f"{model_labels.get(model_family, model_family)} | leaf={leaf_tokens}"
                    ax.plot(
                        xs,
                        ys,
                        linestyle=model_styles.get(model_family, "-"),
                        marker="o",
                        linewidth=1.9,
                        markersize=4.5,
                        color=color,
                        alpha=0.95 if model_family == "neural" else 0.8,
                        label=label,
                    )
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(f"train_docs={train_docs}")
            if c == 0:
                ax.set_ylabel(f"{access_pattern}\n{metric_name}")
            if r == len(access_vals) - 1:
                ax.set_xlabel("local oracle coverage")

    handles, labels = axes[0][0].get_legend_handles_labels()
    dedup: Dict[str, object] = {}
    for h, l in zip(handles, labels):
        dedup.setdefault(l, h)
    if dedup:
        fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Markov local-coverage frontiers.")
    parser.add_argument("--summary-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-query-rate", type=float, default=0.0)
    parser.add_argument(
        "--access-patterns",
        type=str,
        default="merge_only,leaf_only,local_mixed",
        help="Comma-separated access patterns to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.summary_json)
    patterns = [item.strip() for item in str(args.access_patterns).split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_panels(
        rows,
        metric_name="learned_root_mae",
        root_query_rate=float(args.root_query_rate),
        access_patterns=patterns,
        output_path=args.output_dir / "markov_root_mae_vs_local_coverage.png",
        title=f"Markov root MAE vs local oracle coverage (root_query_rate={args.root_query_rate:g})",
    )
    _plot_metric_panels(
        rows,
        metric_name="learned_merge_mae",
        root_query_rate=float(args.root_query_rate),
        access_patterns=patterns,
        output_path=args.output_dir / "markov_merge_mae_vs_local_coverage.png",
        title=f"Markov merge MAE vs local oracle coverage (root_query_rate={args.root_query_rate:g})",
    )
    print(
        json.dumps(
            {
                "summary_json": str(args.summary_json),
                "output_dir": str(args.output_dir),
                "root_query_rate": float(args.root_query_rate),
                "access_patterns": patterns,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
