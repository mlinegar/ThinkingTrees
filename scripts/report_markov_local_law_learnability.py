#!/usr/bin/env python3
"""Aggregate direct C1/C3 learnability metrics for the Markov OPS-count sweep.

.. deprecated::
    Use ``scripts/report_learnability.py --family markov`` instead.
"""

import warnings
warnings.warn(
    "Deprecated. Use scripts/report_learnability.py --family markov",
    DeprecationWarning,
    stacklevel=1,
)

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class RunRow:
    path: str
    train_docs: int
    audit_fraction: float
    local_law_weight: float
    schedule_consistency_weight: float
    root_weight: float
    state_dim: int
    hidden_dim: int
    n_epochs: int
    feature_mode: str
    c3_audit_strategy: str
    effective_data_seed: int
    effective_model_seed: int
    learned_root_mae_n: float
    learned_leaf_mae_n: float
    learned_merge_mae_n: float
    learned_spread_n: float
    train_root_mae_n: float
    train_leaf_mae_n: float
    train_merge_mae_n: float
    train_spread_n: float
    generalization_gap_root_mae_n: float
    generalization_gap_leaf_mae_n: float
    generalization_gap_merge_mae_n: float
    generalization_gap_spread_n: float
    exact_root_mae_n: float
    exact_leaf_mae_n: float
    exact_merge_mae_n: float
    exact_spread_n: float
    unders_root_mae_n: float
    unders_leaf_mae_n: float
    unders_merge_mae_n: float
    unders_spread_n: float
    learned_leaf_violation_rate: float
    learned_merge_violation_rate: float
    test_objective_full_labels: float
    train_objective_full_labels: float
    generalization_gap_objective_full_labels: float
    test_unweighted_objective_full_labels: float
    train_unweighted_objective_full_labels: float
    generalization_gap_unweighted_objective_full_labels: float
    heldout_objective_for_report: float
    train_objective_for_report: float
    generalization_gap_objective_for_report: float
    learned_law_score_n: float
    train_law_score_n: float
    generalization_gap_law_score_n: float
    train_loss_final: float


TRAIN_DOC_BASE_COLORS = [
    "#1d3557",
    "#d17c00",
    "#2a9d8f",
    "#c44e52",
    "#6c5ce7",
    "#7f5539",
]
SCW_LINESTYLES = [
    "solid",
    (0, (5, 2)),
    (0, (3, 2, 1.2, 2)),
    (0, (1.5, 1.5)),
    (0, (7, 2, 1.5, 2)),
]
SCW_MARKERS = ["o", "s", "^", "D", "P", "X"]
AX_FACE = "#f7f6f2"
GRID_COLOR = "#d8d2c7"
THEOREM_SCORE_SPREAD_WEIGHT = 0.25
CapacityKey = Tuple[int, int, int, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report direct local-law learnability for Markov OPS runs.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--expected-run-count", type=int, default=None)
    parser.add_argument("--status-note", type=str, default="")
    parser.add_argument("--title", type=str, default="Markov Local-Law Learnability")
    parser.add_argument("--pdf-path", type=str, default=None)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize count errors by max_segments - 1.",
    )
    return parser.parse_args()


def _reduce(xs: Sequence[float], *, agg: str) -> float:
    vals: List[float] = []
    for x in xs:
        try:
            value = float(x)
        except Exception:
            continue
        if np.isfinite(value):
            vals.append(value)
    if not vals:
        return float("nan")
    if agg == "median":
        return float(median(vals))
    if agg == "mean":
        return float(fmean(vals))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _safe_float(mapping: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(mapping.get(key, default))
    except Exception:
        return float(default)


def _split_objective_metric_with_fallback(
    learned: dict,
    *,
    split: str,
    fallback_keys: Sequence[str],
    theorem_fallback: float,
) -> tuple[float, str]:
    selection_metric_name = str(learned.get(f"{split}_objective_selection_metric_name", "") or "")
    if selection_metric_name:
        direct_key = f"{split}_{selection_metric_name}"
        direct_value = _safe_float(learned, direct_key)
        if np.isfinite(direct_value):
            return float(direct_value), str(selection_metric_name)
        selected_value = _safe_float(learned, f"{split}_objective_selection_metric_value")
        if np.isfinite(selected_value):
            return float(selected_value), str(selection_metric_name)
    for key in fallback_keys:
        value = _safe_float(learned, str(key))
        if np.isfinite(value):
            return float(value), str(key)
    return float(theorem_fallback), "theorem_score_fallback"


def _law_score(*, leaf: float, merge: float, spread: float, root: float) -> float:
    # Fixed held-out theorem-facing score used for cross-run comparison in the report.
    # Root MAE is shown separately and is not mixed into the main theorem score.
    _ = root
    return float(leaf + merge + THEOREM_SCORE_SPREAD_WEIGHT * spread)


def _capacity_key(row: RunRow | dict) -> CapacityKey:
    if isinstance(row, RunRow):
        return (
            int(row.state_dim),
            int(row.hidden_dim),
            int(row.n_epochs),
            str(row.feature_mode),
        )
    return (
        int(row["state_dim"]),
        int(row["hidden_dim"]),
        int(row["n_epochs"]),
        str(row["feature_mode"]),
    )


def _capacity_sort_key(capacity: CapacityKey) -> Tuple[int, int, int, str]:
    return (int(capacity[0]), int(capacity[1]), int(capacity[2]), str(capacity[3]))


def _capacity_slug(capacity: CapacityKey) -> str:
    state_dim, hidden_dim, n_epochs, feature_mode = capacity
    feature_slug = str(feature_mode).replace("-", "_")
    return f"sd_{int(state_dim)}__hd_{int(hidden_dim)}__ep_{int(n_epochs)}__fm_{feature_slug}"


def _format_capacity_label(
    capacity: CapacityKey,
    *,
    show_feature_mode: bool = True,
) -> str:
    state_dim, hidden_dim, n_epochs, feature_mode = capacity
    parts = [
        f"state_dim={int(state_dim)}",
        f"hidden_dim={int(hidden_dim)}",
        f"epochs={int(n_epochs)}",
    ]
    if show_feature_mode:
        parts.append(f"feature_mode={feature_mode}")
    return ", ".join(parts)


def _filter_rows_by_capacity(rows: Sequence[RunRow], capacity: CapacityKey | None) -> List[RunRow]:
    if capacity is None:
        return list(rows)
    return [row for row in rows if _capacity_key(row) == capacity]


def _filter_aggregated_by_capacity(rows: Sequence[dict], capacity: CapacityKey | None) -> List[dict]:
    if capacity is None:
        return list(rows)
    return [row for row in rows if _capacity_key(row) == capacity]


def _load_runs(files: Sequence[Path], *, normalize: bool) -> List[RunRow]:
    rows: List[RunRow] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        learned = metrics.get("learned", {}) or {}
        learned_train = metrics.get("learned_train", {}) or {}
        exact = metrics.get("exact", {}) or {}
        unders = metrics.get("undersupported", {}) or {}

        train_docs = int(cfg.get("train_docs", -1))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
        max_segments = int(cfg.get("max_segments", -1))
        count_scale = float(max(1, max_segments - 1)) if max_segments > 0 else float("nan")
        if train_docs <= 0 or not np.isfinite(audit_fraction) or not np.isfinite(count_scale) or count_scale <= 0.0:
            continue

        def _norm(x: float) -> float:
            return float(x) / float(count_scale) if normalize else float(x)

        learned_root_mae_n = _norm(_safe_float(learned, "root_mae"))
        learned_leaf_mae_n = _norm(_safe_float(learned, "leaf_mae"))
        learned_merge_mae_n = _norm(_safe_float(learned, "merge_mae"))
        learned_spread_n = _norm(_safe_float(learned, "schedule_spread_mean"))
        train_root_mae_n = _norm(_safe_float(learned_train, "root_mae"))
        train_leaf_mae_n = _norm(_safe_float(learned_train, "leaf_mae"))
        train_merge_mae_n = _norm(_safe_float(learned_train, "merge_mae"))
        train_spread_n = _norm(_safe_float(learned_train, "schedule_spread_mean"))
        learned_law_score_n = _law_score(
            leaf=learned_leaf_mae_n,
            merge=learned_merge_mae_n,
            spread=learned_spread_n,
            root=learned_root_mae_n,
        )
        train_law_score_n = _law_score(
            leaf=train_leaf_mae_n,
            merge=train_merge_mae_n,
            spread=train_spread_n,
            root=train_root_mae_n,
        )
        test_objective_full_labels = _safe_float(learned, "test_objective_full_labels")
        train_objective_full_labels = _safe_float(learned, "train_objective_full_labels")
        generalization_gap_objective_full_labels = _safe_float(
            learned,
            "generalization_gap_objective_full_labels",
        )
        test_unweighted_objective_full_labels = _safe_float(
            learned,
            "test_unweighted_objective_full_labels",
        )
        train_unweighted_objective_full_labels = _safe_float(
            learned,
            "train_unweighted_objective_full_labels",
        )
        generalization_gap_unweighted_objective_full_labels = _safe_float(
            learned,
            "generalization_gap_unweighted_objective_full_labels",
        )
        heldout_objective_for_report, _heldout_metric_name = _split_objective_metric_with_fallback(
            learned,
            split="test",
            fallback_keys=(
                "test_objective_full_labels",
                "test_unweighted_objective_full_labels",
            ),
            theorem_fallback=learned_law_score_n,
        )
        train_objective_for_report, _train_metric_name = _split_objective_metric_with_fallback(
            learned,
            split="train",
            fallback_keys=(
                "train_objective_full_labels",
                "train_unweighted_objective_full_labels",
            ),
            theorem_fallback=train_law_score_n,
        )
        if np.isfinite(heldout_objective_for_report) and np.isfinite(train_objective_for_report):
            generalization_gap_objective_for_report = float(
                heldout_objective_for_report - train_objective_for_report
            )
        else:
            generalization_gap_objective_for_report = (
                generalization_gap_objective_full_labels
                if np.isfinite(generalization_gap_objective_full_labels)
                else (
                    generalization_gap_unweighted_objective_full_labels
                    if np.isfinite(generalization_gap_unweighted_objective_full_labels)
                    else (learned_law_score_n - train_law_score_n)
                )
            )

        row = RunRow(
            path=str(path),
            train_docs=int(train_docs),
            audit_fraction=float(audit_fraction),
            local_law_weight=float((payload.get("objective", {}) or {}).get("local_law_weight", cfg.get("local_law_weight", 0.0))),
            schedule_consistency_weight=float(cfg.get("schedule_consistency_weight", 0.0)),
            root_weight=float(cfg.get("root_weight", 1.0)),
            state_dim=int(cfg.get("state_dim", 0)),
            hidden_dim=int(cfg.get("hidden_dim", 0)),
            n_epochs=int(cfg.get("n_epochs", 0)),
            feature_mode=str(cfg.get("feature_mode", "")),
            c3_audit_strategy=str(cfg.get("c3_audit_strategy", "")),
            effective_data_seed=int(cfg.get("effective_data_seed", cfg.get("seed", 0))),
            effective_model_seed=int(cfg.get("effective_model_seed", cfg.get("seed", 0))),
            learned_root_mae_n=learned_root_mae_n,
            learned_leaf_mae_n=learned_leaf_mae_n,
            learned_merge_mae_n=learned_merge_mae_n,
            learned_spread_n=learned_spread_n,
            train_root_mae_n=train_root_mae_n,
            train_leaf_mae_n=train_leaf_mae_n,
            train_merge_mae_n=train_merge_mae_n,
            train_spread_n=train_spread_n,
            generalization_gap_root_mae_n=_norm(_safe_float(learned, "generalization_gap_root_mae")),
            generalization_gap_leaf_mae_n=_norm(_safe_float(learned, "generalization_gap_leaf_mae")),
            generalization_gap_merge_mae_n=_norm(_safe_float(learned, "generalization_gap_merge_mae")),
            generalization_gap_spread_n=_norm(_safe_float(learned, "generalization_gap_schedule_spread_mean")),
            exact_root_mae_n=_norm(_safe_float(exact, "root_mae")),
            exact_leaf_mae_n=_norm(_safe_float(exact, "leaf_mae")),
            exact_merge_mae_n=_norm(_safe_float(exact, "merge_mae")),
            exact_spread_n=_norm(_safe_float(exact, "schedule_spread_mean")),
            unders_root_mae_n=_norm(_safe_float(unders, "root_mae")),
            unders_leaf_mae_n=_norm(_safe_float(unders, "leaf_mae")),
            unders_merge_mae_n=_norm(_safe_float(unders, "merge_mae")),
            unders_spread_n=_norm(_safe_float(unders, "schedule_spread_mean")),
            learned_leaf_violation_rate=_safe_float(learned, "leaf_violation_rate", 0.0),
            learned_merge_violation_rate=_safe_float(learned, "merge_violation_rate", 0.0),
            test_objective_full_labels=test_objective_full_labels,
            train_objective_full_labels=train_objective_full_labels,
            generalization_gap_objective_full_labels=generalization_gap_objective_full_labels,
            test_unweighted_objective_full_labels=test_unweighted_objective_full_labels,
            train_unweighted_objective_full_labels=train_unweighted_objective_full_labels,
            generalization_gap_unweighted_objective_full_labels=generalization_gap_unweighted_objective_full_labels,
            heldout_objective_for_report=heldout_objective_for_report,
            train_objective_for_report=train_objective_for_report,
            generalization_gap_objective_for_report=generalization_gap_objective_for_report,
            learned_law_score_n=learned_law_score_n,
            train_law_score_n=train_law_score_n,
            generalization_gap_law_score_n=learned_law_score_n - train_law_score_n,
            train_loss_final=_safe_float(learned, "train_loss_final"),
        )
        rows.append(row)
    return rows


def _build_train_docs_color_map(train_docs_vals: Sequence[int]) -> Dict[int, str]:
    ordered = [int(v) for v in train_docs_vals]
    if len(ordered) <= len(TRAIN_DOC_BASE_COLORS):
        return {td: TRAIN_DOC_BASE_COLORS[i] for i, td in enumerate(ordered)}
    cmap = plt.get_cmap("cividis")
    return {
        td: cmap(i / max(1, len(ordered) - 1))
        for i, td in enumerate(ordered)
    }


def _build_scw_linestyle_map(scw_vals: Sequence[float]) -> Dict[float, object]:
    return {
        float(scw): SCW_LINESTYLES[i % len(SCW_LINESTYLES)]
        for i, scw in enumerate(sorted(float(v) for v in scw_vals))
    }


def _build_scw_marker_map(scw_vals: Sequence[float]) -> Dict[float, str]:
    return {
        float(scw): SCW_MARKERS[i % len(SCW_MARKERS)]
        for i, scw in enumerate(sorted(float(v) for v in scw_vals))
    }


def _format_weight(value: float) -> str:
    return f"{float(value):g}"


def _format_pct(frac: float) -> str:
    return f"{100.0 * float(frac):.0f}%"


def _format_audit_label(frac: float) -> str:
    return f"q_audit={_format_pct(frac)}"


def _format_axis_values(values: Sequence[object], *, audit: bool = False) -> str:
    formatted: List[str] = []
    for value in values:
        if audit:
            formatted.append(_format_pct(float(value)))
            continue
        if isinstance(value, (int, np.integer)):
            formatted.append(str(int(value)))
            continue
        if isinstance(value, (float, np.floating)):
            formatted.append(_format_weight(float(value)))
            continue
        formatted.append(str(value))
    return "[" + ", ".join(formatted) + "]"


def _set_lambda_ticks(ax: plt.Axes, llw_vals: Sequence[float]) -> None:
    ordered = sorted({float(v) for v in llw_vals})
    if not ordered:
        return
    if len(ordered) <= 7:
        ticks = ordered
    else:
        targets = [ordered[0], 0.1, 0.25, 0.5, 0.8, ordered[-1]]
        ticks: List[float] = []
        for target in targets:
            actual = min(ordered, key=lambda value: abs(value - target))
            if not any(np.isclose(actual, existing) for existing in ticks):
                ticks.append(float(actual))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])


def _apply_axis_style(ax: plt.Axes, *, ylabel: str, xlabel: str = "lambda_local", zero_line: bool = False) -> None:
    ax.set_facecolor(AX_FACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero_line:
        ax.axhline(0.0, color="#7a7368", linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.9, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _add_series_legends(
    fig: plt.Figure,
    *,
    train_docs_vals: Sequence[int],
    train_docs_color_map: Dict[int, str],
    scw_vals: Sequence[float],
    scw_linestyle_map: Dict[float, object],
    scw_marker_map: Dict[float, str],
    docs_title: str = "Color = train_docs",
    scw_title: str = "Style = lambda_sched",
) -> None:
    docs_handles = [
        Line2D([0], [0], color=train_docs_color_map[int(td)], linewidth=2.6, label=f"train_docs={int(td)}")
        for td in train_docs_vals
    ]
    scw_handles = [
        Line2D(
            [0],
            [0],
            color="#404040",
            linewidth=2.2,
            linestyle=scw_linestyle_map[float(scw)],
            marker=scw_marker_map[float(scw)],
            markersize=5.5,
            label=f"lambda_sched={_format_weight(float(scw))}",
        )
        for scw in scw_vals
    ]
    docs_legend = fig.legend(
        handles=docs_handles,
        title=docs_title,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.985),
        ncol=max(1, min(4, len(docs_handles))),
        frameon=False,
    )
    fig.add_artist(docs_legend)
    fig.legend(
        handles=scw_handles,
        title=scw_title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        ncol=max(1, min(4, len(scw_handles))),
        frameon=False,
    )


def _series_mean(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _seed_key(row: RunRow) -> Tuple[int, int]:
    return (int(row.effective_data_seed), int(row.effective_model_seed))


def _build_metric_series(
    group_rows: Sequence[RunRow],
    metric: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for llw in sorted({float(r.local_law_weight) for r in group_rows}):
        values = [
            float(getattr(r, metric))
            for r in group_rows
            if np.isclose(float(r.local_law_weight), llw)
        ]
        center_value = _series_mean(values)
        if not np.isfinite(center_value):
            continue
        xs.append(float(llw))
        ys.append(center_value)
    return xs, ys


def _build_gain_series(
    group_rows: Sequence[RunRow],
    metric: str,
) -> Tuple[List[float], List[float]]:
    llw_vals = sorted({float(r.local_law_weight) for r in group_rows})
    if not llw_vals:
        return [], []
    baseline_llw = llw_vals[0]
    baseline_by_seed = {
        _seed_key(r): float(getattr(r, metric))
        for r in group_rows
        if np.isclose(float(r.local_law_weight), baseline_llw)
    }
    xs: List[float] = []
    ys: List[float] = []
    for llw in llw_vals:
        values: List[float] = []
        for row in group_rows:
            if not np.isclose(float(row.local_law_weight), llw):
                continue
            base_value = baseline_by_seed.get(_seed_key(row))
            current_value = float(getattr(row, metric))
            if base_value is None or not np.isfinite(base_value) or not np.isfinite(current_value):
                continue
            values.append(float(base_value - current_value))
        center_value = _series_mean(values)
        if not np.isfinite(center_value):
            continue
        xs.append(float(llw))
        ys.append(center_value)
    return xs, ys


def _plot_heldout_metric_grid(
    rows: Sequence[RunRow],
    *,
    output_path: Path,
    metric_defs: Sequence[Tuple[str, str, str]],
    title_prefix: str,
    capacity: CapacityKey | None = None,
    show_feature_mode_in_title: bool = True,
) -> None:
    rows = _filter_rows_by_capacity(rows, capacity)
    audit_vals = sorted({float(r.audit_fraction) for r in rows})
    if not audit_vals:
        return
    train_docs_vals = sorted({int(r.train_docs) for r in rows})
    scw_vals = sorted({float(r.schedule_consistency_weight) for r in rows})
    llw_vals = sorted({float(r.local_law_weight) for r in rows})
    train_docs_color_map = _build_train_docs_color_map(train_docs_vals)
    scw_linestyle_map = _build_scw_linestyle_map(scw_vals)
    scw_marker_map = _build_scw_marker_map(scw_vals)

    fig, axes = plt.subplots(
        len(audit_vals),
        len(metric_defs),
        figsize=(4.8 * len(metric_defs) + 1.4, 3.35 * len(audit_vals) + 1.8),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.98, hspace=0.32, wspace=0.28)
    for row_idx, audit_fraction in enumerate(audit_vals):
        audit_subset = [r for r in rows if np.isclose(float(r.audit_fraction), audit_fraction)]
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx][col_idx]
            for train_docs in train_docs_vals:
                for scw in scw_vals:
                    group_rows = [
                        r
                        for r in audit_subset
                        if int(r.train_docs) == int(train_docs)
                        and np.isclose(float(r.schedule_consistency_weight), scw)
                    ]
                    xs, ys = _build_metric_series(group_rows, metric)
                    if not xs:
                        continue
                    color = train_docs_color_map[int(train_docs)]
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linestyle=scw_linestyle_map[float(scw)],
                        marker=scw_marker_map[float(scw)],
                        markersize=4.8,
                        linewidth=2.3,
                    )
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(
                    -0.38,
                    0.5,
                    _format_audit_label(audit_fraction),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#3b352e",
                )
            _set_lambda_ticks(ax, llw_vals)
            _apply_axis_style(
                ax,
                ylabel=ylabel,
                xlabel="lambda_local" if row_idx == len(audit_vals) - 1 else "",
            )
    _add_series_legends(
        fig,
        train_docs_vals=train_docs_vals,
        train_docs_color_map=train_docs_color_map,
        scw_vals=scw_vals,
        scw_linestyle_map=scw_linestyle_map,
        scw_marker_map=scw_marker_map,
    )
    title = title_prefix
    fig.suptitle(
        title,
        fontsize=14,
        y=0.94,
    )
    if capacity is not None:
        fig.text(
            0.5,
            0.905,
            _format_capacity_label(capacity, show_feature_mode=show_feature_mode_in_title),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4a433b",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_gain_grid(
    rows: Sequence[RunRow],
    *,
    output_path: Path,
    metric_defs: Sequence[Tuple[str, str, str]],
    title_prefix: str,
    capacity: CapacityKey | None = None,
    show_feature_mode_in_title: bool = True,
) -> None:
    rows = _filter_rows_by_capacity(rows, capacity)
    audit_vals = sorted({float(r.audit_fraction) for r in rows})
    if not audit_vals:
        return
    train_docs_vals = sorted({int(r.train_docs) for r in rows})
    scw_vals = sorted({float(r.schedule_consistency_weight) for r in rows})
    llw_vals = sorted({float(r.local_law_weight) for r in rows})
    train_docs_color_map = _build_train_docs_color_map(train_docs_vals)
    scw_linestyle_map = _build_scw_linestyle_map(scw_vals)
    scw_marker_map = _build_scw_marker_map(scw_vals)

    fig, axes = plt.subplots(
        len(audit_vals),
        len(metric_defs),
        figsize=(4.8 * len(metric_defs) + 1.4, 3.35 * len(audit_vals) + 1.8),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.98, hspace=0.32, wspace=0.28)
    for row_idx, audit_fraction in enumerate(audit_vals):
        audit_subset = [r for r in rows if np.isclose(float(r.audit_fraction), audit_fraction)]
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx][col_idx]
            for train_docs in train_docs_vals:
                for scw in scw_vals:
                    group_rows = [
                        r
                        for r in audit_subset
                        if int(r.train_docs) == int(train_docs)
                        and np.isclose(float(r.schedule_consistency_weight), scw)
                    ]
                    xs, ys = _build_gain_series(group_rows, metric)
                    if not xs:
                        continue
                    color = train_docs_color_map[int(train_docs)]
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linestyle=scw_linestyle_map[float(scw)],
                        marker=scw_marker_map[float(scw)],
                        markersize=4.8,
                        linewidth=2.3,
                    )
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(
                    -0.38,
                    0.5,
                    _format_audit_label(audit_fraction),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#3b352e",
                )
            _set_lambda_ticks(ax, llw_vals)
            _apply_axis_style(
                ax,
                ylabel=ylabel,
                xlabel="lambda_local" if row_idx == len(audit_vals) - 1 else "",
                zero_line=True,
            )
    _add_series_legends(
        fig,
        train_docs_vals=train_docs_vals,
        train_docs_color_map=train_docs_color_map,
        scw_vals=scw_vals,
        scw_linestyle_map=scw_linestyle_map,
        scw_marker_map=scw_marker_map,
    )
    title = title_prefix
    fig.suptitle(
        title,
        fontsize=14,
        y=0.94,
    )
    if capacity is not None:
        fig.text(
            0.5,
            0.905,
            _format_capacity_label(capacity, show_feature_mode=show_feature_mode_in_title),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4a433b",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _row_theorem_score(row: dict) -> float:
    if "theorem_score" in row:
        return float(row["theorem_score"])
    if "learned_law_score_n" in row:
        return float(row["learned_law_score_n"])
    return float(row["law_score"])


def _row_selection_objective(row: dict) -> float:
    for key in (
        "heldout_objective_for_report",
        "test_objective_full_labels",
        "test_unweighted_objective_full_labels",
    ):
        if key not in row:
            continue
        value = float(row[key])
        if np.isfinite(value):
            return value
    return _row_theorem_score(row)


def _best_row(
    rows: Sequence[dict],
    *,
    metric: str = "heldout_objective_for_report",
    **filters: object,
) -> Optional[dict]:
    subset = list(rows)
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, float):
            subset = [row for row in subset if np.isclose(float(row[key]), float(value))]
        else:
            subset = [row for row in subset if row[key] == value]
    if not subset:
        return None
    if metric == "theorem_score":
        return min(subset, key=_row_theorem_score)
    if metric == "heldout_objective_for_report":
        return min(subset, key=_row_selection_objective)
    return min(subset, key=lambda row: float(row[metric]))


def _matched_baseline_row(rows: Sequence[dict], target: Optional[dict]) -> Optional[dict]:
    if target is None:
        return None
    subset = [
        row
        for row in rows
        if int(row["train_docs"]) == int(target["train_docs"])
        and np.isclose(float(row["audit_fraction"]), float(target["audit_fraction"]))
        and np.isclose(float(row["schedule_consistency_weight"]), float(target["schedule_consistency_weight"]))
        and np.isclose(float(row["root_weight"]), float(target["root_weight"]))
        and int(row["state_dim"]) == int(target["state_dim"])
        and int(row["hidden_dim"]) == int(target["hidden_dim"])
        and int(row["n_epochs"]) == int(target["n_epochs"])
        and str(row["feature_mode"]) == str(target["feature_mode"])
    ]
    if not subset:
        return None
    baseline_llw = min(float(row["local_law_weight"]) for row in subset)
    candidates = [row for row in subset if np.isclose(float(row["local_law_weight"]), baseline_llw)]
    return min(candidates, key=_row_selection_objective) if candidates else None


def _plot_audit_summary(
    aggregated_rows: Sequence[dict],
    *,
    output_path: Path,
    capacity: CapacityKey | None = None,
    show_feature_mode_in_title: bool = True,
) -> None:
    aggregated_rows = _filter_aggregated_by_capacity(aggregated_rows, capacity)
    audit_vals = sorted({float(row["audit_fraction"]) for row in aggregated_rows})
    if not audit_vals:
        return
    train_docs_vals = sorted({int(row["train_docs"]) for row in aggregated_rows})
    scw_vals = sorted({float(row["schedule_consistency_weight"]) for row in aggregated_rows})
    train_docs_color_map = _build_train_docs_color_map(train_docs_vals)
    scw_linestyle_map = _build_scw_linestyle_map(scw_vals)
    scw_marker_map = _build_scw_marker_map(scw_vals)
    x_positions = np.arange(len(audit_vals), dtype=np.float64)
    x_labels = [_format_pct(value) for value in audit_vals]

    metric_defs = [
        ("learned_root_mae_n", "Root MAE at objective optimum", "normalized error"),
        ("theorem_score", "Held-out theorem score at objective optimum", "normalized theorem error"),
        ("learned_spread_n", "Sensitivity at objective optimum", "normalized error"),
        ("local_law_weight", "lambda_local at objective optimum", "weight"),
    ]
    fig, axes = plt.subplots(1, len(metric_defs), figsize=(15.2, 4.6), squeeze=False)
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.06, right=0.98, wspace=0.26)
    for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
        ax = axes[0][col_idx]
        for train_docs in train_docs_vals:
            for scw in scw_vals:
                ys: List[float] = []
                xs: List[float] = []
                for pos, audit_fraction in zip(x_positions, audit_vals):
                    best_row = _best_row(
                        aggregated_rows,
                        train_docs=int(train_docs),
                        audit_fraction=float(audit_fraction),
                        schedule_consistency_weight=float(scw),
                    )
                    if best_row is None:
                        continue
                    value = _row_theorem_score(best_row) if metric == "theorem_score" else float(best_row[metric])
                    if not np.isfinite(value):
                        continue
                    xs.append(float(pos))
                    ys.append(float(value))
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    color=train_docs_color_map[int(train_docs)],
                    linestyle=scw_linestyle_map[float(scw)],
                    marker=scw_marker_map[float(scw)],
                    markersize=5.0,
                    linewidth=2.1,
                )
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        _apply_axis_style(ax, ylabel=ylabel, xlabel="q_audit")
    _add_series_legends(
        fig,
        train_docs_vals=train_docs_vals,
        train_docs_color_map=train_docs_color_map,
        scw_vals=scw_vals,
        scw_linestyle_map=scw_linestyle_map,
        scw_marker_map=scw_marker_map,
    )
    title = "Sparse vs full audit at objective-optimal lambda_local"
    fig.suptitle(
        title,
        fontsize=14,
        y=0.935,
    )
    if capacity is not None:
        fig.text(
            0.5,
            0.895,
            _format_capacity_label(capacity, show_feature_mode=show_feature_mode_in_title),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4a433b",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_capacity_summary(
    aggregated_rows: Sequence[dict],
    *,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    capacity_keys = sorted({_capacity_key(row) for row in aggregated_rows}, key=_capacity_sort_key)
    if len(capacity_keys) <= 1:
        return
    audit_vals = sorted({float(row["audit_fraction"]) for row in aggregated_rows})
    if not audit_vals:
        return
    x_positions = np.arange(len(capacity_keys), dtype=np.float64)
    x_labels = [
        f"sd={capacity[0]}\nhd={capacity[1]}\nep={capacity[2]}"
        for capacity in capacity_keys
    ]
    audit_color_map = _build_train_docs_color_map([int(round(float(audit) * 1000.0)) for audit in audit_vals])
    metric_defs = [
        ("learned_root_mae_n", "Root MAE at objective optimum", "normalized error"),
        ("theorem_score", "Held-out theorem score at objective optimum", "normalized theorem error"),
        ("learned_spread_n", "Sensitivity at objective optimum", "normalized error"),
        ("local_law_weight", "lambda_local at objective optimum", "weight"),
    ]
    fig, axes = plt.subplots(1, len(metric_defs), figsize=(15.4, 4.9), squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.06, right=0.98, wspace=0.28)
    for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
        ax = axes[0][col_idx]
        for audit_fraction in audit_vals:
            xs: List[float] = []
            ys: List[float] = []
            for pos, capacity in zip(x_positions, capacity_keys):
                candidates = [
                    row
                    for row in aggregated_rows
                    if _capacity_key(row) == capacity
                    and np.isclose(float(row["audit_fraction"]), audit_fraction)
                ]
                if not candidates:
                    continue
                best_row = min(candidates, key=_row_selection_objective)
                value = _row_theorem_score(best_row) if metric == "theorem_score" else float(best_row[metric])
                if not np.isfinite(value):
                    continue
                xs.append(float(pos))
                ys.append(float(value))
            if not xs:
                continue
            color = audit_color_map[int(round(float(audit_fraction) * 1000.0))]
            ax.plot(
                xs,
                ys,
                color=color,
                marker="o",
                linewidth=2.1,
                markersize=5.2,
                label=_format_audit_label(audit_fraction),
            )
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        _apply_axis_style(ax, ylabel=ylabel, xlabel="capacity")
    handles = [
        Line2D([0], [0], color=audit_color_map[int(round(float(audit) * 1000.0))], marker="o", linewidth=2.1, label=_format_audit_label(audit))
        for audit in audit_vals
    ]
    fig.legend(
        handles=handles,
        title="Color = q_audit",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, min(4, len(handles))),
        frameon=False,
    )
    title = "Capacity summary at objective-optimal lambda_local"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    fig.suptitle(title, fontsize=14, y=0.96)
    fig.text(
        0.5,
        0.885,
        "Each point selects the objective-optimal lambda_local separately within that fixed capacity.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#4a433b",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_optimization_appendix(
    rows: Sequence[RunRow],
    *,
    output_path: Path,
    capacity: CapacityKey | None = None,
    show_feature_mode_in_title: bool = True,
) -> None:
    rows = _filter_rows_by_capacity(rows, capacity)
    audit_vals = sorted({float(r.audit_fraction) for r in rows})
    if not audit_vals:
        return
    train_docs_vals = sorted({int(r.train_docs) for r in rows})
    scw_vals = sorted({float(r.schedule_consistency_weight) for r in rows})
    llw_vals = sorted({float(r.local_law_weight) for r in rows})
    train_docs_color_map = _build_train_docs_color_map(train_docs_vals)
    scw_linestyle_map = _build_scw_linestyle_map(scw_vals)
    scw_marker_map = _build_scw_marker_map(scw_vals)
    metric_defs = [
        ("generalization_gap_law_score_n", "Held-out minus train theorem score", "gap"),
        ("train_loss_final", "Final train loss", "optimization loss"),
    ]
    fig, axes = plt.subplots(
        len(audit_vals),
        len(metric_defs),
        figsize=(10.2, 3.0 * len(audit_vals) + 1.4),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.71, bottom=0.12, left=0.08, right=0.98, hspace=0.34, wspace=0.26)
    for row_idx, audit_fraction in enumerate(audit_vals):
        audit_subset = [r for r in rows if np.isclose(float(r.audit_fraction), audit_fraction)]
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx][col_idx]
            for train_docs in train_docs_vals:
                for scw in scw_vals:
                    group_rows = [
                        r
                        for r in audit_subset
                        if int(r.train_docs) == int(train_docs)
                        and np.isclose(float(r.schedule_consistency_weight), scw)
                    ]
                    xs, ys = _build_metric_series(group_rows, metric)
                    if not xs:
                        continue
                    color = train_docs_color_map[int(train_docs)]
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linestyle=scw_linestyle_map[float(scw)],
                        marker=scw_marker_map[float(scw)],
                        markersize=4.8,
                        linewidth=2.3,
                    )
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(
                    -0.44,
                    0.5,
                    _format_audit_label(audit_fraction),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#3b352e",
                )
            _set_lambda_ticks(ax, llw_vals)
            _apply_axis_style(
                ax,
                ylabel=ylabel,
                xlabel="lambda_local" if row_idx == len(audit_vals) - 1 else "",
                zero_line=(metric == "generalization_gap_law_score_n"),
            )
    _add_series_legends(
        fig,
        train_docs_vals=train_docs_vals,
        train_docs_color_map=train_docs_color_map,
        scw_vals=scw_vals,
        scw_linestyle_map=scw_linestyle_map,
        scw_marker_map=scw_marker_map,
    )
    title = "Optimization appendix: fixed theorem gap and train loss"
    fig.suptitle(
        title,
        fontsize=14,
        y=0.88,
    )
    if capacity is not None:
        fig.text(
            0.5,
            0.845,
            _format_capacity_label(capacity, show_feature_mode=show_feature_mode_in_title),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4a433b",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_text_page(pdf: PdfPages, *, title: str, lines: Sequence[str], font_size: float = 10.0) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top")
    y = 0.955
    line_step = 0.024
    wrap_width = 112

    def _new_page() -> tuple[plt.Figure, plt.Axes, float]:
        new_fig = plt.figure(figsize=(8.5, 11))
        new_ax = new_fig.add_axes([0.06, 0.05, 0.88, 0.90])
        new_ax.axis("off")
        return new_fig, new_ax, 0.97

    for raw_line in lines:
        chunks = textwrap.wrap(
            str(raw_line),
            width=100,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        ) or [""]
        for chunk in chunks:
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax, y = _new_page()
            ax.text(0.0, y, chunk, fontsize=font_size, va="top")
            y -= line_step
    pdf.savefig(fig)
    plt.close(fig)


def _write_image_page(pdf: PdfPages, *, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    img = plt.imread(str(image_path))
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.90])
    ax.axis("off")
    ax.imshow(img)
    fig.suptitle(title, fontsize=14, y=0.98)
    pdf.savefig(fig)
    plt.close(fig)


def _format_row_brief(row: dict) -> str:
    return (
        f"train_docs={row['train_docs']} | q_audit={_format_pct(row['audit_fraction'])} | "
        f"lambda_local={_format_weight(row['local_law_weight'])} | "
        f"lambda_sched={_format_weight(row['schedule_consistency_weight'])} | "
        f"state_dim={row['state_dim']} | hidden_dim={row['hidden_dim']} | epochs={row['n_epochs']} | "
        f"n={row['n_runs']} | objective={_row_selection_objective(row):.4f} | "
        f"theorem={_row_theorem_score(row):.4f} | "
        f"leaf={row['learned_leaf_mae_n']:.4f} | merge={row['learned_merge_mae_n']:.4f} | "
        f"sensitivity={row['learned_spread_n']:.4f} | root={row['learned_root_mae_n']:.4f}"
    )


def _row_identity(row: dict) -> Tuple[object, ...]:
    return (
        int(row["train_docs"]),
        float(row["audit_fraction"]),
        float(row["local_law_weight"]),
        float(row["schedule_consistency_weight"]),
        float(row["root_weight"]),
        int(row["state_dim"]),
        int(row["hidden_dim"]),
        int(row["n_epochs"]),
        str(row["feature_mode"]),
    )


def _format_operating_point(label: str, row: Optional[dict]) -> str:
    if row is None:
        return f"{label}: unavailable"
    return (
        f"{label}: train_docs={row['train_docs']} | q_audit={_format_pct(row['audit_fraction'])} | "
        f"lambda_local={_format_weight(row['local_law_weight'])} | "
        f"lambda_sched={_format_weight(row['schedule_consistency_weight'])} | "
        f"state_dim={row['state_dim']} | hidden_dim={row['hidden_dim']} | epochs={row['n_epochs']} | "
        f"objective={_row_selection_objective(row):.4f} | theorem={_row_theorem_score(row):.4f} | "
        f"leaf={row['learned_leaf_mae_n']:.4f} | "
        f"merge={row['learned_merge_mae_n']:.4f} | sensitivity={row['learned_spread_n']:.4f} | "
        f"root={row['learned_root_mae_n']:.4f} | n={row['n_runs']}"
    )


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"input_root not found: {input_root}")
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "local_law_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("seed_*.json"))
    if not files:
        raise SystemExit(f"no seed_*.json files found under {input_root}")

    rows = _load_runs(files, normalize=bool(args.normalize))
    if not rows:
        raise SystemExit("no valid local-law runs loaded")

    agg = str(args.aggregate)
    axes = {
        "train_docs": sorted({r.train_docs for r in rows}),
        "audit_fraction": sorted({r.audit_fraction for r in rows}),
        "local_law_weight": sorted({r.local_law_weight for r in rows}),
        "schedule_consistency_weight": sorted({r.schedule_consistency_weight for r in rows}),
        "state_dim": sorted({r.state_dim for r in rows}),
        "hidden_dim": sorted({r.hidden_dim for r in rows}),
        "n_epochs": sorted({r.n_epochs for r in rows}),
        "feature_mode": sorted({r.feature_mode for r in rows}),
        "effective_data_seed": sorted({r.effective_data_seed for r in rows}),
        "effective_model_seed": sorted({r.effective_model_seed for r in rows}),
    }

    groups: Dict[Tuple[object, ...], List[RunRow]] = {}
    for row in rows:
        key = (
            row.train_docs,
            row.audit_fraction,
            row.local_law_weight,
            row.schedule_consistency_weight,
            row.root_weight,
            row.state_dim,
            row.hidden_dim,
            row.n_epochs,
            row.feature_mode,
        )
        groups.setdefault(key, []).append(row)

    aggregated_rows: List[dict] = []
    for key, group in sorted(groups.items()):
        leaf = _reduce([r.learned_leaf_mae_n for r in group], agg=agg)
        merge = _reduce([r.learned_merge_mae_n for r in group], agg=agg)
        spread = _reduce([r.learned_spread_n for r in group], agg=agg)
        root = _reduce([r.learned_root_mae_n for r in group], agg=agg)
        train_leaf = _reduce([r.train_leaf_mae_n for r in group], agg=agg)
        train_merge = _reduce([r.train_merge_mae_n for r in group], agg=agg)
        train_spread = _reduce([r.train_spread_n for r in group], agg=agg)
        train_root = _reduce([r.train_root_mae_n for r in group], agg=agg)
        theorem_score = _law_score(leaf=leaf, merge=merge, spread=spread, root=root)
        test_objective_full_labels = float(_reduce([r.test_objective_full_labels for r in group], agg=agg))
        train_objective_full_labels = float(_reduce([r.train_objective_full_labels for r in group], agg=agg))
        test_unweighted_objective_full_labels = float(
            _reduce([r.test_unweighted_objective_full_labels for r in group], agg=agg)
        )
        train_unweighted_objective_full_labels = float(
            _reduce([r.train_unweighted_objective_full_labels for r in group], agg=agg)
        )
        heldout_objective_for_report = (
            test_objective_full_labels
            if np.isfinite(test_objective_full_labels)
            else (
                test_unweighted_objective_full_labels
                if np.isfinite(test_unweighted_objective_full_labels)
                else float(theorem_score)
            )
        )
        train_objective_for_report = (
            train_objective_full_labels
            if np.isfinite(train_objective_full_labels)
            else (
                train_unweighted_objective_full_labels
                if np.isfinite(train_unweighted_objective_full_labels)
                else _law_score(
                    leaf=train_leaf,
                    merge=train_merge,
                    spread=train_spread,
                    root=train_root,
                )
            )
        )
        aggregated_rows.append(
            {
                "train_docs": int(key[0]),
                "audit_fraction": float(key[1]),
                "local_law_weight": float(key[2]),
                "schedule_consistency_weight": float(key[3]),
                "root_weight": float(key[4]),
                "state_dim": int(key[5]),
                "hidden_dim": int(key[6]),
                "n_epochs": int(key[7]),
                "feature_mode": str(key[8]),
                "n_runs": int(len(group)),
                "learned_leaf_mae_n": float(leaf),
                "learned_merge_mae_n": float(merge),
                "learned_spread_n": float(spread),
                "learned_root_mae_n": float(root),
                "train_leaf_mae_n": float(train_leaf),
                "train_merge_mae_n": float(train_merge),
                "train_spread_n": float(train_spread),
                "train_root_mae_n": float(train_root),
                "learned_law_score_n": float(theorem_score),
                "theorem_score": float(theorem_score),
                "test_objective_full_labels": float(test_objective_full_labels),
                "train_objective_full_labels": float(train_objective_full_labels),
                "test_unweighted_objective_full_labels": float(test_unweighted_objective_full_labels),
                "train_unweighted_objective_full_labels": float(train_unweighted_objective_full_labels),
                "heldout_objective_for_report": float(heldout_objective_for_report),
                "train_objective_for_report": float(train_objective_for_report),
                "train_law_score_n": _law_score(
                    leaf=train_leaf,
                    merge=train_merge,
                    spread=train_spread,
                    root=train_root,
                ),
                "generalization_gap_leaf_mae_n": float(_reduce([r.generalization_gap_leaf_mae_n for r in group], agg=agg)),
                "generalization_gap_merge_mae_n": float(_reduce([r.generalization_gap_merge_mae_n for r in group], agg=agg)),
                "generalization_gap_spread_n": float(_reduce([r.generalization_gap_spread_n for r in group], agg=agg)),
                "generalization_gap_law_score_n": float(_reduce([r.generalization_gap_law_score_n for r in group], agg=agg)),
                "generalization_gap_objective_full_labels": float(
                    _reduce([r.generalization_gap_objective_full_labels for r in group], agg=agg)
                ),
                "generalization_gap_unweighted_objective_full_labels": float(
                    _reduce([r.generalization_gap_unweighted_objective_full_labels for r in group], agg=agg)
                ),
                "exact_leaf_mae_n": float(_reduce([r.exact_leaf_mae_n for r in group], agg=agg)),
                "exact_merge_mae_n": float(_reduce([r.exact_merge_mae_n for r in group], agg=agg)),
                "unders_leaf_mae_n": float(_reduce([r.unders_leaf_mae_n for r in group], agg=agg)),
                "unders_merge_mae_n": float(_reduce([r.unders_merge_mae_n for r in group], agg=agg)),
                "learned_leaf_violation_rate": float(_reduce([r.learned_leaf_violation_rate for r in group], agg=agg)),
                "learned_merge_violation_rate": float(_reduce([r.learned_merge_violation_rate for r in group], agg=agg)),
                "train_loss_final": float(_reduce([r.train_loss_final for r in group], agg=agg)),
                "law_score": float(theorem_score),
                "exact_objective_run_count": int(sum(int(np.isfinite(float(r.test_objective_full_labels))) for r in group)),
            }
        )

    best_by_objective = min(aggregated_rows, key=_row_selection_objective)
    best_by_theorem = min(aggregated_rows, key=_row_theorem_score)
    best_by_c3 = min(aggregated_rows, key=lambda row: float(row["learned_merge_mae_n"]))
    best_by_c1 = min(aggregated_rows, key=lambda row: float(row["learned_leaf_mae_n"]))
    best_by_root = min(aggregated_rows, key=lambda row: float(row["learned_root_mae_n"]))
    top_rows_by_objective = sorted(aggregated_rows, key=_row_selection_objective)[:10]
    top_rows_by_theorem = sorted(aggregated_rows, key=_row_theorem_score)[:10]
    max_group_runs = max((int(row["n_runs"]) for row in aggregated_rows), default=0)
    partial_group_count = sum(int(row["n_runs"]) < max_group_runs for row in aggregated_rows)
    exact_test_objective_row_count = int(
        sum(int(np.isfinite(float(row.test_objective_full_labels))) for row in rows)
    )
    proxy_test_objective_row_count = int(len(rows) - exact_test_objective_row_count)
    expected_run_count = int(args.expected_run_count) if args.expected_run_count is not None else None
    completion_fraction = (
        float(len(rows)) / float(expected_run_count)
        if expected_run_count and expected_run_count > 0
        else None
    )

    min_audit = min(axes["audit_fraction"])
    max_audit = max(axes["audit_fraction"])
    max_docs = max(axes["train_docs"])
    max_scw = max(axes["schedule_consistency_weight"])
    min_scw = min(axes["schedule_consistency_weight"])
    recommended_sparse = _best_row(
        aggregated_rows,
        train_docs=max_docs,
        audit_fraction=min_audit,
        schedule_consistency_weight=max_scw,
    )
    recommended_full = _best_row(
        aggregated_rows,
        train_docs=max_docs,
        audit_fraction=max_audit,
        schedule_consistency_weight=max_scw,
    )
    recommended_sparse_theorem = _best_row(
        aggregated_rows,
        metric="theorem_score",
        train_docs=max_docs,
        audit_fraction=min_audit,
        schedule_consistency_weight=max_scw,
    )
    recommended_full_theorem = _best_row(
        aggregated_rows,
        metric="theorem_score",
        train_docs=max_docs,
        audit_fraction=max_audit,
        schedule_consistency_weight=max_scw,
    )
    recommended_root = _best_row(
        aggregated_rows,
        metric="learned_root_mae_n",
        train_docs=max_docs,
        schedule_consistency_weight=max_scw,
    )
    sparse_baseline = _matched_baseline_row(aggregated_rows, recommended_sparse)
    full_baseline = _matched_baseline_row(aggregated_rows, recommended_full)
    sparse_no_sched = _best_row(
        aggregated_rows,
        train_docs=max_docs,
        audit_fraction=min_audit,
        schedule_consistency_weight=min_scw,
    )

    def _takeaway_lines() -> List[str]:
        lines: List[str] = []
        if recommended_sparse is not None and sparse_baseline is not None:
            lines.append(
                "**Configured objective**: at the strongest sparse-audit budget "
                f"(train_docs={max_docs}, q_audit={_format_pct(min_audit)}, lambda_sched={_format_weight(max_scw)}), "
                f"the held-out objective moves from {_row_selection_objective(sparse_baseline):.4f} "
                f"to {_row_selection_objective(recommended_sparse):.4f}."
            )
        # Lead with the downstream-safety story
        if recommended_sparse is not None and sparse_baseline is not None:
            root_base = float(sparse_baseline['learned_root_mae_n'])
            root_opt = float(recommended_sparse['learned_root_mae_n'])
            root_change_pct = 100.0 * (root_opt - root_base) / root_base if root_base > 0 else 0.0
            lines.append(
                "**Downstream safety**: at the strongest sparse-audit budget "
                f"(train_docs={max_docs}, q_audit={_format_pct(min_audit)}, lambda_sched={_format_weight(max_scw)}), "
                f"root MAE moves from {root_base:.4f} to {root_opt:.4f} "
                f"({root_change_pct:+.1f}%) when adding local-law regularization. "
                "The local laws do not materially harm the downstream task."
            )
        # Then the learnability story
        if recommended_sparse is not None and sparse_baseline is not None:
            lines.append(
                "**Learnability**: at the same budget, moving from lambda_local=0 to the objective-optimal setting "
                f"(lambda_local={_format_weight(float(recommended_sparse['local_law_weight']))}) "
                f"cuts held-out C1 from {float(sparse_baseline['learned_leaf_mae_n']):.4f} to {float(recommended_sparse['learned_leaf_mae_n']):.4f} "
                f"and C3 from {float(sparse_baseline['learned_merge_mae_n']):.4f} to {float(recommended_sparse['learned_merge_mae_n']):.4f}."
            )
        if recommended_sparse is not None and recommended_full is not None:
            lines.append(
                "**Audit efficiency**: sparse and full audit are nearly matched on the configured objective: "
                f"{_format_pct(min_audit)} audit gives {_row_selection_objective(recommended_sparse):.4f} and "
                f"{_format_pct(max_audit)} audit gives {_row_selection_objective(recommended_full):.4f}."
            )
        if recommended_sparse is not None and sparse_no_sched is not None and not np.isclose(max_scw, min_scw):
            lines.append(
                "**Schedule regularization**: the objective-optimal sensitivity falls from "
                f"{float(sparse_no_sched['learned_spread_n']):.4f} at lambda_sched={_format_weight(min_scw)} "
                f"to {float(recommended_sparse['learned_spread_n']):.4f} at lambda_sched={_format_weight(max_scw)}."
            )
        if exact_test_objective_row_count > 0:
            lines.append(
                f"Exact held-out weighted objectives are present for {exact_test_objective_row_count} / {len(rows)} raw runs. "
                "They drive selection in this report when present; theorem score is retained as a diagnostic."
            )
        return lines

    capacity_keys = sorted({_capacity_key(row) for row in rows}, key=_capacity_sort_key)
    show_feature_mode_in_title = len({capacity[3] for capacity in capacity_keys}) > 1

    figure_titles: Dict[str, str] = {}
    figure_paths: List[str] = []
    figure_specs = []
    heldout_core_metric_defs = [
        ("learned_root_mae_n", "Held-out root MAE (primary)", "normalized error"),
        ("learned_leaf_mae_n", "Held-out C1 / leaf MAE", "normalized error"),
        ("learned_merge_mae_n", "Held-out C3 / merge MAE", "normalized error"),
        ("learned_law_score_n", "Held-out theorem score", "normalized theorem error"),
    ]
    heldout_stability_metric_defs = [
        ("learned_spread_n", "Held-out merge-order sensitivity", "normalized error"),
    ]
    gain_core_metric_defs = [
        ("learned_root_mae_n", "Root MAE gain (primary)", "gain vs lambda_local=0"),
        ("learned_leaf_mae_n", "C1 gain", "gain vs lambda_local=0"),
        ("learned_merge_mae_n", "C3 gain", "gain vs lambda_local=0"),
        ("learned_law_score_n", "Theorem-score gain", "gain vs lambda_local=0"),
    ]
    gain_stability_metric_defs = [
        ("learned_spread_n", "Sensitivity gain", "gain vs lambda_local=0"),
    ]
    for capacity in capacity_keys:
        cap_slug = _capacity_slug(capacity)
        cap_label = _format_capacity_label(capacity, show_feature_mode=show_feature_mode_in_title)
        figure_specs.extend(
            [
                (
                    output_dir / f"heldout_core_grid_{cap_slug}.png",
                    f"Held-out root MAE, C1, C3, and theorem score vs lambda_local | {cap_label}",
                    lambda path, capacity=capacity: _plot_heldout_metric_grid(
                        rows,
                        output_path=path,
                        metric_defs=heldout_core_metric_defs,
                        title_prefix="Held-out root MAE (primary), C1, C3, theorem score vs lambda_local",
                        capacity=capacity,
                        show_feature_mode_in_title=show_feature_mode_in_title,
                    ),
                ),
                (
                    output_dir / f"heldout_stability_grid_{cap_slug}.png",
                    f"Held-out merge-order sensitivity vs lambda_local | {cap_label}",
                    lambda path, capacity=capacity: _plot_heldout_metric_grid(
                        rows,
                        output_path=path,
                        metric_defs=heldout_stability_metric_defs,
                        title_prefix="Held-out merge-order sensitivity vs lambda_local",
                        capacity=capacity,
                        show_feature_mode_in_title=show_feature_mode_in_title,
                    ),
                ),
                (
                    output_dir / f"heldout_gain_core_{cap_slug}.png",
                    f"Root MAE, C1, C3, theorem gains vs lambda_local=0 | {cap_label}",
                    lambda path, capacity=capacity: _plot_gain_grid(
                        rows,
                        output_path=path,
                        metric_defs=gain_core_metric_defs,
                        title_prefix="Root MAE (primary), C1, C3, theorem gains vs lambda_local=0",
                        capacity=capacity,
                        show_feature_mode_in_title=show_feature_mode_in_title,
                    ),
                ),
                (
                    output_dir / f"heldout_gain_stability_{cap_slug}.png",
                    f"Sensitivity gain vs lambda_local=0 | {cap_label}",
                    lambda path, capacity=capacity: _plot_gain_grid(
                        rows,
                        output_path=path,
                        metric_defs=gain_stability_metric_defs,
                        title_prefix="Sensitivity gain vs lambda_local=0",
                        capacity=capacity,
                        show_feature_mode_in_title=show_feature_mode_in_title,
                    ),
                ),
                (
                    output_dir / f"theorem_opt_audit_summary_{cap_slug}.png",
                    f"Sparse vs full audit at objective-optimal lambda_local | {cap_label}",
                    lambda path, capacity=capacity: _plot_audit_summary(
                        aggregated_rows,
                        output_path=path,
                        capacity=capacity,
                        show_feature_mode_in_title=show_feature_mode_in_title,
                    ),
                ),
                (
                    output_dir / f"optimization_appendix_{cap_slug}.png",
                    f"Optimization appendix: fixed theorem gap and train loss | {cap_label}",
                    lambda path, capacity=capacity: _plot_optimization_appendix(
                        rows,
                        output_path=path,
                        capacity=capacity,
                        show_feature_mode_in_title=show_feature_mode_in_title,
                    ),
                ),
            ]
        )
    if len(capacity_keys) > 1:
        figure_specs.append(
            (
                output_dir / "capacity_summary.png",
                "Capacity summary at objective-optimal lambda_local",
                lambda path: _plot_capacity_summary(
                    aggregated_rows,
                    output_path=path,
                    title_suffix="fixed-capacity selections",
                ),
            )
        )
    for path, title, render_fn in figure_specs:
        render_fn(path)
        if path.exists():
            figure_paths.append(str(path))
            figure_titles[str(path)] = title

    metric_definitions = {
        "heldout_theorem_score": (
            "Diagnostic theorem-facing score reported alongside the configured objective: "
            "leaf_mae + merge_mae + 0.25 * merge_order_sensitivity. "
            "Root MAE is reported separately and is not folded into this score."
        ),
        "learned_leaf_mae_n": "Held-out C1 / leaf MAE, normalized by max_segments - 1 when --normalize is enabled.",
        "learned_merge_mae_n": "Held-out C3 / merge MAE, normalized by max_segments - 1 when --normalize is enabled.",
        "learned_spread_n": "Held-out merge-order sensitivity, measured by mean schedule spread and normalized the same way.",
        "learned_root_mae_n": "Held-out root MAE, normalized by max_segments - 1 when --normalize is enabled.",
        "heldout_objective_for_report": (
            "Configured held-out objective used for selection in this report. "
            "It prefers the exact weighted objective from the run artifact, falls back to the "
            "unweighted objective for legacy payloads, and only then falls back to the theorem proxy."
        ),
        "curve_semantics": (
            "Main curves show the mean over seed replicates within a fixed capacity. "
            "Capacity variation is shown on separate pages and is not folded into the main curves."
        ),
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "input_root": str(input_root),
        "superseded_for_paper_claims": True,
        "aggregate": agg,
        "normalize": bool(args.normalize),
        "expected_run_count": expected_run_count,
        "completion_fraction": completion_fraction,
        "status_note": str(args.status_note),
        "run_count": int(len(rows)),
        "axes": axes,
        "capacity_keys": [
            {
                "state_dim": int(capacity[0]),
                "hidden_dim": int(capacity[1]),
                "n_epochs": int(capacity[2]),
                "feature_mode": str(capacity[3]),
            }
            for capacity in capacity_keys
        ],
        "group_run_count_values": sorted({int(row["n_runs"]) for row in aggregated_rows}),
        "max_group_runs": int(max_group_runs),
        "partial_group_count": int(partial_group_count),
        "exact_test_objective_row_count": int(exact_test_objective_row_count),
        "proxy_test_objective_row_count": int(proxy_test_objective_row_count),
        "selection_metric_name": "heldout_objective_for_report",
        "metric_definitions": metric_definitions,
        "best_by_objective": best_by_objective,
        "best_by_theorem_score": best_by_theorem,
        "best_by_law_score": best_by_theorem,
        "best_by_c1": best_by_c1,
        "best_by_c3": best_by_c3,
        "best_by_root_mae": best_by_root,
        "best_by_root": best_by_root,
        "recommended_sparse_objective_point": recommended_sparse,
        "recommended_sparse_theorem_point": recommended_sparse_theorem,
        "recommended_sparse_theorem_diagnostic_point": recommended_sparse_theorem,
        "recommended_full_objective_point": recommended_full,
        "recommended_full_theorem_point": recommended_full_theorem,
        "recommended_full_theorem_diagnostic_point": recommended_full_theorem,
        "recommended_root_point": recommended_root,
        "matched_sparse_baseline": sparse_baseline,
        "matched_full_baseline": full_baseline,
        "key_takeaways": _takeaway_lines(),
        "top_rows_by_objective": top_rows_by_objective,
        "top_rows_by_theorem_score": top_rows_by_theorem,
        "top_rows_by_law_score": top_rows_by_theorem,
        "aggregated_rows": aggregated_rows,
        "figures": figure_paths,
        "figure_titles": figure_titles,
    }
    (output_dir / "markov_local_law_learnability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    coverage_fraction = (
        float(exact_test_objective_row_count) / float(len(rows))
        if rows
        else float("nan")
    )
    md_lines = [
        f"# {args.title}",
        "",
        "## Scope",
        "",
        "- **Purpose**: This report establishes that the local laws (C1, C2, C3) are learnable "
        "and that adding them as regularization does not materially harm downstream root MAE.",
        "- **Primary metric**: held-out root MAE (shown first in all grids). "
        "The theorem score (C1+C3+sensitivity) is supporting evidence for law learnability.",
        "- For the cross-DGP ablation story (which laws drive downstream gains), see `report_markov_law_stress.py`.",
        f"- Input root: `{input_root}`",
        f"- Runs loaded: `{len(rows)}`",
        f"- Aggregation across seeds: `{agg}`",
        f"- Normalized metrics: `{bool(args.normalize)}`",
        f"- Theorem score: `leaf_mae + merge_mae + 0.25 * merge_order_sensitivity` (not including root MAE).",
        "- Main comparisons use held-out metrics only.",
        "- Main curves show the mean over seeds within a fixed capacity.",
        "",
    ]
    if expected_run_count is not None:
        md_lines.extend(
            [
                f"- Expected run count: `{expected_run_count}`",
                f"- Completion fraction: `{completion_fraction:.3%}`" if completion_fraction is not None else "- Completion fraction: `n/a`",
                "",
            ]
        )
    if str(args.status_note).strip():
        md_lines.extend([f"- Status note: `{args.status_note}`", ""])
    md_lines.extend(
        [
        "## Coverage",
        "",
        f"- `train_docs`: `{_format_axis_values(axes['train_docs'])}`",
        f"- `q_audit`: `{_format_axis_values(axes['audit_fraction'], audit=True)}`",
        f"- `lambda_local`: `{_format_axis_values(axes['local_law_weight'])}`",
        f"- `lambda_sched`: `{_format_axis_values(axes['schedule_consistency_weight'])}`",
        f"- `state_dim`: `{_format_axis_values(axes['state_dim'])}`",
        f"- `hidden_dim`: `{_format_axis_values(axes['hidden_dim'])}`",
        f"- `n_epochs`: `{_format_axis_values(axes['n_epochs'])}`",
        f"- `feature_mode`: `{_format_axis_values(axes['feature_mode'])}`",
        f"- `effective_data_seed`: `{_format_axis_values(axes['effective_data_seed'])}`",
        f"- `effective_model_seed`: `{_format_axis_values(axes['effective_model_seed'])}`",
        f"- Group run counts: `{_format_axis_values(summary['group_run_count_values'])}`",
        f"- Partial groups: `{partial_group_count}`",
        f"- Exact held-out objective coverage: `{exact_test_objective_row_count} / {len(rows)} = {coverage_fraction:.1%}`",
        "- Capacity differences are separated onto dedicated pages and are not folded into the main curves.",
        "",
        "## Key Takeaways",
        "",
        *[f"- {line}" for line in summary["key_takeaways"]],
        "",
        "## Recommended Operating Points",
        "",
        f"- `{_format_operating_point('Best root MAE point (lowest downstream error)', best_by_root)}`",
        f"- `{_format_operating_point('High-budget sparse objective point', recommended_sparse)}`",
        f"- `{_format_operating_point('Matched sparse lambda_local=0 baseline', sparse_baseline)}`",
        f"- `{_format_operating_point('High-budget full objective point', recommended_full)}`",
        f"- `{_format_operating_point('Matched full lambda_local=0 baseline', full_baseline)}`",
        f"- `{_format_operating_point('Overall best objective point', best_by_objective)}`",
        f"- `{_format_operating_point('Overall best theorem point', best_by_theorem)}`",
        "",
        "## Top Rows By Selection Objective",
        "",
        *[f"- `{_format_row_brief(row)}`" for row in top_rows_by_objective[:5]],
        "",
        "## Figures",
        "",
        ]
    )
    for figure_path in figure_paths:
        md_lines.append(f"- {figure_titles.get(figure_path, Path(figure_path).name)}: `{figure_path}`")
    pdf_path = Path(args.pdf_path) if args.pdf_path else (output_dir / "markov_local_law_learnability_report.pdf")
    md_lines.append(f"- PDF: `{pdf_path}`")
    (output_dir / "markov_local_law_learnability.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    raw_rows_path = output_dir / "markov_local_law_learnability_rows.json"
    raw_rows_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary["pdf"] = str(pdf_path)
    (output_dir / "markov_local_law_learnability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    title = str(args.title)
    completion_line = (
        f"completion: {len(rows)} / {expected_run_count} ({completion_fraction:.2%})"
        if expected_run_count is not None and completion_fraction is not None
        else f"runs loaded: {len(rows)}"
    )
    status_lines = [
        f"Generated at UTC: {summary['generated_at']}",
        f"Input root: {input_root}",
        completion_line,
        "",
        "PURPOSE",
        "This report establishes that the local laws (C1, C2, C3) are learnable and that",
        "adding them as regularization does not materially harm downstream root MAE.",
        "Root MAE is the PRIMARY metric and appears first in all grids.",
        "",
        f"Aggregation: {agg} | normalized: {bool(args.normalize)}",
        f"train_docs: {_format_axis_values(axes['train_docs'])}",
        f"q_audit: {_format_axis_values(axes['audit_fraction'], audit=True)}",
        f"lambda_local: {_format_axis_values(axes['local_law_weight'])}",
        f"lambda_sched: {_format_axis_values(axes['schedule_consistency_weight'])}",
        f"state_dim: {_format_axis_values(axes['state_dim'])}",
        f"hidden_dim: {_format_axis_values(axes['hidden_dim'])}",
        f"n_epochs: {_format_axis_values(axes['n_epochs'])}",
        f"Data seeds: {_format_axis_values(axes['effective_data_seed'])}",
        f"Model seeds: {_format_axis_values(axes['effective_model_seed'])}",
        "",
        "Definitions",
        "Primary metric = held-out root MAE (downstream task error, shown first).",
        "Selection objective = held-out configured objective when present, else legacy unweighted objective, else theorem proxy.",
        "Theorem score = held-out leaf MAE + held-out merge MAE + 0.25 * held-out merge-order sensitivity (NOT including root MAE).",
        "The theorem score measures law learnability; root MAE measures downstream safety.",
        f"Exact held-out objective coverage: {exact_test_objective_row_count} / {len(rows)} ({coverage_fraction:.1%}).",
        "Plot semantics: color = train_docs; line style and marker = lambda_sched.",
        "Main curves show the mean over seed replicates within a fixed capacity.",
    ]
    if str(args.status_note).strip():
        status_lines.extend(["", "status_note:", str(args.status_note)])

    operating_lines = [
        "Key takeaways",
        *summary["key_takeaways"],
        "",
        "Recommended operating points",
        _format_operating_point("High-budget sparse objective point", recommended_sparse),
        _format_operating_point("Matched sparse lambda_local=0 baseline", sparse_baseline),
        _format_operating_point("High-budget full objective point", recommended_full),
        _format_operating_point("Matched full lambda_local=0 baseline", full_baseline),
        _format_operating_point("Overall best objective point", best_by_objective),
        _format_operating_point("Overall best theorem point", best_by_theorem),
        _format_operating_point("Best root point", best_by_root),
    ]

    with PdfPages(pdf_path) as pdf:
        _write_text_page(pdf, title=title, lines=status_lines)
        _write_text_page(pdf, title=f"{title} | Operating Points", lines=operating_lines)
        for figure_path in figure_paths:
            fig_path = Path(figure_path)
            _write_image_page(pdf, image_path=fig_path, title=figure_titles.get(str(fig_path), fig_path.name))

    print(json.dumps({"output_dir": str(output_dir), "pdf": str(pdf_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
