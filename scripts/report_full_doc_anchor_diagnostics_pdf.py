#!/usr/bin/env python3
"""Render mode-specific PDF reports for Markov full-document diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASELINE_ORDER = (
    "official_fno",
    "official_fno_sumlen",
    "cnn1d",
    "palette_block_exact",
    "raw_token_ngram_ridge",
    "tree_ridge_leaf",
    "tree_doc_ridge",
    "tree_neural_c2",
    "tree_neural_c2c3",
    "tree_neural",
)
BASELINE_COLORS = {
    "official_fno": "#1d4ed8",
    "official_fno_sumlen": "#60a5fa",
    "cnn1d": "#dc2626",
    "palette_block_exact": "#111827",
    "raw_token_ngram_ridge": "#6b7280",
    "tree_ridge_leaf": "#8b5cf6",
    "tree_doc_ridge": "#92400e",
    "tree_neural_c2": "#166534",
    "tree_neural_c2c3": "#16a34a",
    "tree_neural": "#0f766e",
}
BASELINE_MARKERS = {
    "official_fno": "o",
    "official_fno_sumlen": "s",
    "cnn1d": "^",
    "palette_block_exact": "D",
    "raw_token_ngram_ridge": "x",
    "tree_ridge_leaf": "P",
    "tree_doc_ridge": "H",
    "tree_neural_c2": "p",
    "tree_neural_c2c3": "v",
    "tree_neural": "d",
}
RECOVERABLE_COMPARISON_FAMILIES = (
    "official_fno",
    "official_fno_sumlen",
    "tree_neural_c2",
    "tree_neural_c2c3",
    "tree_neural",
    "tree_doc_ridge",
    "tree_ridge_leaf",
    "raw_token_ngram_ridge",
    "cnn1d",
    "palette_block_exact",
)
LEARNED_COMPARISON_FAMILIES = frozenset(
    {
        "official_fno",
        "official_fno_sumlen",
        "tree_neural_c2",
        "tree_neural_c2c3",
        "tree_neural",
    }
)
SEGMENT_BAND_ORDER = ("low", "mid", "high")
DEFAULT_ROOT_MAE_FLOOR = 1e-4
DEFAULT_ROOT_MAE_CEILING = 2e1
FOCUS_FAMILIES = (
    "official_fno",
    "official_fno_sumlen",
    "cnn1d",
    "palette_block_exact",
)
REPORTED_SPLITS = ("train", "val", "test")

# Readable column name mapping for tables
_COL_RENAME = {
    "baseline_family": "Family",
    "config_label": "Config",
    "train_doc_count": "Train Docs",
    "cell_id": "Cell",
    "n_regimes": "Regimes",
    "segment_density_band": "Seg. Density",
    "train_root_mae_mean": "Train MAE",
    "val_root_mae_mean": "Val MAE",
    "test_root_mae_mean": "MAE (mean)",
    "test_root_mae_std": "MAE (std)",
    "test_root_mae_min": "MAE (min)",
    "test_root_mae_max": "MAE (max)",
    "test_root_mae_median": "MAE (med)",
    "train_exact_match_rate_mean": "Train Exact Match",
    "val_exact_match_rate_mean": "Val Exact Match",
    "test_exact_match_rate_mean": "Exact Match",
    "test_exact_match_rate_std": "EM (std)",
    "fixed_leaf_tokens": "Leaf Tokens",
    "val_unweighted_full_law_objective_mean": "Val Full-Law Obj",
    "val_unweighted_active_objective_mean": "Val Active Obj",
    "test_unweighted_full_law_objective_mean": "Full-Law Obj",
    "test_unweighted_active_objective_mean": "Active Obj",
    "test_leaf_mae_mean": "Leaf MAE",
    "test_merge_mae_mean": "Merge MAE",
    "test_schedule_spread_mean_mean": "Merge-Order Spread",
    "elapsed_s_mean": "Elapsed (s)",
    "n_runs": "Seeds",
    "parameterization": "Param",
    "optimization_root_weight": "Root Wt",
    "local_law_c1_weight": "C1 Wt",
    "local_law_c2_weight": "C2 Wt",
    "local_law_c3_weight": "C3 Wt",
    "tree_root_supervision_kind": "Root Supervision",
    "tree_leaf_fno_width": "Leaf FNO Width",
    "tree_leaf_fno_n_modes": "Leaf FNO Modes",
    "tree_leaf_fno_n_layers": "Leaf FNO Layers",
    "tree_aux_doc_sequence_fraction": "Aux Seq Frac",
    "tuning_stage": "Tune Stage",
    "task_objective_weight_source": "Task Wt Src",
    "c2_metric_kind": "C2 Metric",
    "comparison_semantics_label": "Semantics",
    "best_full_doc_fno_family": "Best FNO",
    "best_full_doc_fno_test_root_mae_mean": "Best FNO MAE",
    "tree_neural_test_root_mae_mean": "Tree Neural MAE",
    "tree_neural_c2_test_root_mae_mean": "Tree C2 MAE",
    "tree_neural_c2c3_test_root_mae_mean": "Tree C2+C3 MAE",
    "best_parity_tree_family": "Best Tree",
    "best_parity_tree_test_root_mae_mean": "Best Tree MAE",
    "tree_neural_gap_pct_vs_best_fno": "Tree Gap %",
    "best_parity_tree_gap_pct_vs_best_fno": "Best Tree Gap %",
    "best_upper_bound_tree_family": "Best Aux Tree",
    "best_upper_bound_tree_test_root_mae_mean": "Best Aux Tree MAE",
    "best_upper_bound_tree_gap_pct_vs_best_fno": "Best Aux Gap %",
    "budget_total_calls_per_doc": "Calls / Doc",
    "full_doc_budget_share": "Full-Doc Share",
    "doc_consumption_mode": "Doc Mode",
    "local_split_mode": "Local Split",
    "effective_full_doc_mass_per_doc_mean": "Eff Full-Doc Mass / Doc",
    "best_tree_family": "Best Tree",
    "best_tree_test_root_mae_mean": "Best Tree MAE",
    "best_reference_family": "Best Reference",
    "best_reference_test_root_mae_mean": "Best Reference MAE",
    "primary_success_within_10pct": "Tree<=10%",
    "secondary_success_within_10pct": "Any Tree<=10%",
}

# Alternating row colors for tables
_ROW_EVEN = "#f0f0f0"
_ROW_ODD = "#ffffff"
_HEADER_COLOR = "#404040"
_HEADER_TEXT = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a PDF report from a full-doc anchor diagnostic summary.json."
    )
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-pdf", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--root-mae-floor", type=float, default=DEFAULT_ROOT_MAE_FLOOR)
    parser.add_argument("--root-mae-ceiling", type=float, default=DEFAULT_ROOT_MAE_CEILING)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    runs = list(payload.get("runs") or [])
    if runs:
        from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
            _payload_from_saved_runs,
        )

        payload = {**payload, **_payload_from_saved_runs(runs=runs)}
    return payload


def _infer_report_mode(payload: Mapping[str, Any]) -> str:
    hardness_grid = str(payload.get("hardness_grid", "")).strip()
    if not hardness_grid:
        return "recoverable_scale"
    train_doc_counts = payload.get("train_doc_counts")
    values: set[int] = set()
    if isinstance(train_doc_counts, Mapping):
        for seq in train_doc_counts.values():
            values.update(int(item) for item in list(seq or []))
    else:
        values.update(int(item) for item in list(train_doc_counts or []))
    if len(values) > 1:
        return "structural_grid"
    if len(list(payload.get("seeds") or [])) > 1:
        return "structural_stability"
    return "structural_grid"


def _ordered_baselines(rows: pd.DataFrame, *, focus_only: bool = False) -> list[str]:
    observed = [str(x) for x in rows["baseline_family"].dropna().unique().tolist()]
    base = [family for family in BASELINE_ORDER if family in observed]
    ordered = base + sorted([family for family in observed if family not in base])
    if focus_only:
        focused = [family for family in FOCUS_FAMILIES if family in ordered]
        return focused or ordered
    return ordered


def _recoverable_plot_families(rows: pd.DataFrame) -> list[str]:
    observed = [str(x) for x in rows["baseline_family"].dropna().unique().tolist()]
    base = [family for family in RECOVERABLE_COMPARISON_FAMILIES if family in observed]
    extras = sorted(
        family for family in observed if family not in RECOVERABLE_COMPARISON_FAMILIES
    )
    return base + extras


def _ordered_regimes(rows: pd.DataFrame) -> list[int]:
    if "n_regimes" not in rows.columns:
        return []
    return sorted(int(x) for x in rows["n_regimes"].dropna().unique().tolist())


def _ordered_train_doc_counts(rows: pd.DataFrame) -> list[int]:
    if "train_doc_count" not in rows.columns:
        return []
    return sorted(int(x) for x in rows["train_doc_count"].dropna().unique().tolist())


def _recoverable_axis_spec(rows: pd.DataFrame) -> tuple[str, str, list[int], str]:
    train_doc_counts = _ordered_train_doc_counts(rows)
    leaf_tokens = []
    if "fixed_leaf_tokens" in rows.columns:
        leaf_tokens = sorted(
            int(x)
            for x in rows["fixed_leaf_tokens"].dropna().unique().tolist()
            if int(x) > 0
        )
    if len(train_doc_counts) <= 1 and len(leaf_tokens) > 1:
        return ("fixed_leaf_tokens", "leaf tokens", leaf_tokens, "Leaf Geometry")
    return ("train_doc_count", "train docs", train_doc_counts, "Train Documents")


def _pretty_family_name(family: str) -> str:
    mapping = {
        "official_fno": "Official FNO",
        "official_fno_sumlen": "FNO + Sum/Len",
        "cnn1d": "CNN1D",
        "palette_block_exact": "Palette Exact",
        "raw_token_ngram_ridge": "Token Ngram Ridge",
        "tree_ridge_leaf": "Tree Ridge (Leaf)",
        "tree_doc_ridge": "Doc-Span Ridge",
        "tree_neural_c2": "Tree Neural (C2)",
        "tree_neural_c2c3": "Tree Neural (C2+C3)",
        "tree_neural": "Tree Neural (All Laws)",
    }
    return mapping.get(str(family), str(family).replace("_", " "))


def _tree_neural_semantics_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = pd.DataFrame(list(payload.get("aggregate_rows") or []))
    if rows.empty or "baseline_family" not in rows.columns:
        return pd.DataFrame()
    subset = rows.loc[
        rows["baseline_family"].isin(["tree_neural_c2", "tree_neural_c2c3", "tree_neural"])
    ].copy()
    if subset.empty:
        return subset
    subset["baseline_family"] = subset["baseline_family"].map(_pretty_family_name)
    return subset


def _fair_parity_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    summary = dict(payload.get("tree_fno_fair_parity_summary") or {})
    comparisons = list(summary.get("comparisons") or [])
    if not comparisons:
        return pd.DataFrame()
    rows = pd.DataFrame(comparisons)
    if rows.empty:
        return rows
    for family_col in ("best_full_doc_fno_family", "best_parity_tree_family"):
        if family_col in rows.columns:
            rows[family_col] = rows[family_col].map(_pretty_family_name)
    if "tree_neural_gap_ratio_vs_best_fno" in rows.columns:
        rows["tree_neural_gap_pct_vs_best_fno"] = (
            rows["tree_neural_gap_ratio_vs_best_fno"].astype(float) * 100.0
        )
    if "best_parity_tree_gap_ratio_vs_best_fno" in rows.columns:
        rows["best_parity_tree_gap_pct_vs_best_fno"] = (
            rows["best_parity_tree_gap_ratio_vs_best_fno"].astype(float) * 100.0
        )
    return rows


def _upper_bound_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    summary = dict(payload.get("tree_fno_upper_bound_summary") or {})
    comparisons = list(summary.get("comparisons") or [])
    if not comparisons:
        return pd.DataFrame()
    rows = pd.DataFrame(comparisons)
    if rows.empty:
        return rows
    for family_col in ("best_full_doc_fno_family", "best_upper_bound_tree_family"):
        if family_col in rows.columns:
            rows[family_col] = rows[family_col].map(_pretty_family_name)
    if "best_upper_bound_tree_gap_ratio_vs_best_fno" in rows.columns:
        rows["best_upper_bound_tree_gap_pct_vs_best_fno"] = (
            rows["best_upper_bound_tree_gap_ratio_vs_best_fno"].astype(float) * 100.0
        )
    return rows


def _budget_frontier_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    summary = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
    tree_rows = {
        float(row.get("budget_total_calls_per_doc", 0.0)): dict(row)
        for row in list(summary.get("best_tree_by_budget") or [])
    }
    ref_rows = {
        float(row.get("budget_total_calls_per_doc", 0.0)): dict(row)
        for row in list(summary.get("best_reference_by_budget") or [])
    }
    budgets = sorted(set(tree_rows.keys()) | set(ref_rows.keys()))
    if not budgets:
        return pd.DataFrame()
    rows = []
    for budget in budgets:
        tree_row = dict(tree_rows.get(float(budget), {}))
        ref_row = dict(ref_rows.get(float(budget), {}))
        rows.append(
            {
                "budget_total_calls_per_doc": float(budget),
                "best_tree_family": _pretty_family_name(
                    str(tree_row.get("baseline_family", ""))
                ),
                "best_tree_test_root_mae_mean": float(
                    tree_row.get("test_root_mae_mean", float("nan"))
                ),
                "full_doc_budget_share": float(
                    tree_row.get("full_doc_budget_share", float("nan"))
                ),
                "doc_consumption_mode": str(tree_row.get("doc_consumption_mode", "")),
                "local_split_mode": str(tree_row.get("local_split_mode", "")),
                "effective_full_doc_mass_per_doc_mean": float(
                    tree_row.get("effective_full_doc_mass_per_doc_mean", float("nan"))
                ),
                "best_reference_family": _pretty_family_name(
                    str(ref_row.get("baseline_family", ""))
                ),
                "best_reference_test_root_mae_mean": float(
                    ref_row.get("test_root_mae_mean", float("nan"))
                ),
            }
        )
    return pd.DataFrame(rows)


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8.5, color="#555555",
             wrap=True)


def _add_legend_below(fig: plt.Figure, axes, *, ncol: int = 4) -> None:
    """Add a shared legend below the subplots, avoiding title overlap."""
    # Find first axis with legend handles
    handles, labels = [], []
    for ax in np.atleast_1d(axes).flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if not handles:
        return
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(ncol, len(labels)),
        frameon=False,
        fontsize=9,
    )


def _root_mae_floor_clip(values: Sequence[float], *, floor: float, ceiling: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, float(floor), float(ceiling))


def _best_row(rows: pd.DataFrame, metric_key: str, *, ascending: bool = True) -> pd.Series | None:
    if rows.empty:
        return None
    ordered = rows.sort_values(metric_key, ascending=ascending)
    if ordered.empty:
        return None
    return ordered.iloc[0]


def _make_metric_matrix(
    rows: pd.DataFrame,
    *,
    baseline_family: str,
    metric_key: str,
    regimes: Sequence[int],
) -> np.ndarray:
    subset = rows.loc[rows["baseline_family"] == baseline_family].copy()
    matrix = np.full((len(regimes), len(SEGMENT_BAND_ORDER)), np.nan, dtype=np.float64)
    for i, regime in enumerate(regimes):
        for j, band in enumerate(SEGMENT_BAND_ORDER):
            band_rows = subset.loc[
                (subset["n_regimes"] == int(regime))
                & (subset["segment_density_band"] == str(band))
            ]
            if band_rows.empty:
                continue
            matrix[i, j] = float(band_rows.iloc[0][metric_key])
    return matrix


def _control_exactness_line(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("grid_diagnostic_summary") or {})
    control_exactness = dict(summary.get("control_exactness") or {})
    stats = dict(control_exactness.get("palette_block_exact") or {})
    if stats:
        return (
            "palette_block_exact exactness: "
            f"exact_like={bool(stats.get('remains_exact_like', False))}, "
            f"max_root_mae={float(stats.get('max_root_mae_mean', float('nan'))):.4g}"
        )
    return "palette_block_exact exactness: see aggregate rows"


def _fixed_eval_line(payload: Mapping[str, Any]) -> str:
    runs = list(payload.get("runs") or [])
    signatures: dict[str, set[tuple[str, str]]] = {}
    for run in runs:
        key = str(run.get("cell_id") or run.get("benchmark") or "")
        signatures.setdefault(key, set()).add(
            (
                str(run.get("val_corpus_signature", "")),
                str(run.get("test_corpus_signature", "")),
            )
        )
    fixed = bool(signatures) and all(len(items) == 1 for items in signatures.values())
    return f"fixed val/test reuse within comparison units: {fixed}"


def _style_table(table: plt.table, n_data_rows: int) -> None:
    """Apply consistent styling: header color, alternating rows, auto width."""
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        if row == 0:
            cell.set_facecolor(_HEADER_COLOR)
            cell.set_text_props(color=_HEADER_TEXT, weight="bold")
        else:
            cell.set_facecolor(_ROW_EVEN if row % 2 == 0 else _ROW_ODD)
        cell.set_edgecolor("#cccccc")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    if cells:
        max_col = max(c for (_, c) in cells.keys()) + 1
        table.auto_set_column_width(list(range(max_col)))
    table.scale(1.0, 1.35)


def _draw_text_summary_page(
    pdf: PdfPages,
    *,
    title: str,
    subtitle: str,
    sections: Sequence[tuple[str, Sequence[str]]],
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.95
    ax.text(0.03, y, title, fontsize=18, weight="bold", va="top")
    y -= 0.05
    if subtitle:
        ax.text(0.03, y, subtitle, fontsize=11, color="#444444", va="top")
        y -= 0.05
    for heading, bullets in sections:
        ax.text(0.03, y, heading, fontsize=13, weight="bold", va="top")
        y -= 0.035
        for bullet in bullets:
            wrapped = bullet[:130] + ("..." if len(bullet) > 130 else "")
            ax.text(0.05, y, f"- {wrapped}", fontsize=10, va="top")
            y -= 0.028
        y -= 0.015
    pdf.savefig(fig)
    plt.close(fig)


def _line_label(family: str, *, config_label: str = "") -> str:
    label = _pretty_family_name(family)
    config_label = str(config_label).strip()
    if config_label:
        return f"{label} [{config_label}]"
    return label


def _draw_recoverable_primary_ranking_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
    rows: pd.DataFrame,
    *,
    root_mae_floor: float,
    root_mae_ceiling: float,
) -> None:
    baselines = _recoverable_plot_families(rows)
    x_col, x_label, x_values, x_title = _recoverable_axis_spec(rows)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for family in baselines:
        fam = rows.loc[rows["baseline_family"] == family].sort_values(x_col)
        if fam.empty:
            continue
        x = fam[x_col].astype(int).to_numpy()
        y_root = _root_mae_floor_clip(
            fam["test_root_mae_mean"].astype(float).to_numpy(),
            floor=root_mae_floor, ceiling=root_mae_ceiling,
        )
        color = BASELINE_COLORS.get(family)
        marker = BASELINE_MARKERS.get(family, "o")
        label = _pretty_family_name(family)

        ax.plot(x, y_root, color=color, marker=marker, linewidth=2.2, label=label)
        if "test_root_mae_std" in fam.columns:
            std = fam["test_root_mae_std"].astype(float).to_numpy()
            lo = _root_mae_floor_clip(y_root - std, floor=root_mae_floor, ceiling=root_mae_ceiling)
            hi = _root_mae_floor_clip(y_root + std, floor=root_mae_floor, ceiling=root_mae_ceiling)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)

    ax.set_title(f"Primary Ranking: Test Root MAE vs {x_title}", fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("test root MAE")
    ax.set_yscale("log")
    ax.set_ylim(float(root_mae_floor), float(root_mae_ceiling))
    ax.set_xticks(x_values)
    ax.grid(True, which="both", alpha=0.25)

    learned = rows.loc[rows["baseline_family"].isin(LEARNED_COMPARISON_FAMILIES)].copy()
    best = _best_row(learned, "test_root_mae_mean", ascending=True)
    if best is not None:
        bx = int(best[x_col])
        by = float(np.clip(best["test_root_mae_mean"], root_mae_floor, root_mae_ceiling))
        ax.annotate(
            f"best: {_pretty_family_name(str(best['baseline_family']))}\nMAE={float(best['test_root_mae_mean']):.4g}",
            xy=(bx, by), xytext=(10, -25), textcoords="offset points",
            fontsize=8, arrowprops={"arrowstyle": "->", "color": "#444444"},
        )
    exact_rows = rows.loc[rows["baseline_family"] == "palette_block_exact"]
    if not exact_rows.empty:
        ax.text(
            0.03, 0.05, "palette_block_exact pinned at floor\n(true error = 0)",
            transform=ax.transAxes, fontsize=8,
            color=BASELINE_COLORS["palette_block_exact"], va="bottom",
        )

    fig.suptitle("Recoverable Scale Primary Ranking", fontsize=15, y=0.99)
    _add_legend_below(fig, [ax])
    _caption(
        fig,
        "Paper-facing ranking uses only mean test root-count MAE over seeds. "
        "Exact-match, train/val, objective, and law curves are diagnostic-only.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_recoverable_perseed_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
    rows: pd.DataFrame,
    *,
    root_mae_floor: float,
    root_mae_ceiling: float,
) -> None:
    """Per-seed scatter: show individual seed dots behind the mean lines."""
    runs = list(payload.get("runs") or [])
    if not runs:
        return
    runs_df = pd.DataFrame(runs)
    if "test_root_mae" not in runs_df.columns or "baseline_family" not in runs_df.columns:
        return

    baselines = _recoverable_plot_families(rows)
    x_col, x_label, x_values, _x_title = _recoverable_axis_spec(rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for family in baselines:
        fam_runs = runs_df.loc[runs_df["baseline_family"] == family].copy()
        if fam_runs.empty:
            continue
        color = BASELINE_COLORS.get(family)
        marker = BASELINE_MARKERS.get(family, "o")
        label = _pretty_family_name(family)

        x_seeds = fam_runs[x_col].astype(int).to_numpy()
        y_seeds = _root_mae_floor_clip(
            fam_runs["test_root_mae"].astype(float).to_numpy(),
            floor=root_mae_floor, ceiling=root_mae_ceiling,
        )
        jitter = np.random.default_rng(42).uniform(-0.02, 0.02, size=len(x_seeds))
        x_jittered = x_seeds * (1.0 + jitter)
        ax.scatter(x_jittered, y_seeds, color=color, alpha=0.35, s=30, marker=marker,
                   zorder=2)

        fam_agg = rows.loc[rows["baseline_family"] == family].sort_values(x_col)
        if not fam_agg.empty:
            x_mean = fam_agg[x_col].astype(int).to_numpy()
            y_mean = _root_mae_floor_clip(
                fam_agg["test_root_mae_mean"].astype(float).to_numpy(),
                floor=root_mae_floor, ceiling=root_mae_ceiling,
            )
            ax.plot(x_mean, y_mean, color=color, marker=marker, linewidth=2.5,
                    label=label, zorder=3)

    ax.set_title("Per-Seed Root MAE (dots) with Mean Lines", fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("root MAE")
    ax.set_yscale("log")
    ax.set_ylim(float(root_mae_floor), float(root_mae_ceiling))
    ax.set_xticks(x_values)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("Recoverable Scale: Per-Seed Detail", fontsize=15, y=0.99)
    _caption(fig, "Each transparent dot is one seed run. Spread indicates seed variability at each train scale.")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_recoverable_split_metric_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
    *,
    metric_base: str,
    title: str,
    ylabel: str,
    caption: str,
    yscale: str = "linear",
    y_limits: tuple[float, float] | None = None,
) -> None:
    baselines = _recoverable_plot_families(rows)
    x_col, x_label, x_values, _x_title = _recoverable_axis_spec(rows)
    fig, axes = plt.subplots(1, len(REPORTED_SPLITS), figsize=(11, 4.8), sharex=True)
    for ax, split in zip(np.atleast_1d(axes), REPORTED_SPLITS):
        metric_key = f"{split}_{metric_base}_mean"
        metric_std_key = f"{split}_{metric_base}_std"
        for family in baselines:
            fam = rows.loc[rows["baseline_family"] == family].sort_values(x_col)
            if fam.empty or metric_key not in fam.columns:
                continue
            x = fam[x_col].astype(int).to_numpy()
            y = fam[metric_key].astype(float).to_numpy()
            color = BASELINE_COLORS.get(family)
            marker = BASELINE_MARKERS.get(family, "o")
            ax.plot(x, y, color=color, marker=marker, linewidth=2.0, label=_pretty_family_name(family))
            if metric_std_key in fam.columns:
                std = fam[metric_std_key].astype(float).to_numpy()
                lo = y - std
                hi = y + std
                if y_limits is not None:
                    lo = np.clip(lo, y_limits[0], y_limits[1])
                    hi = np.clip(hi, y_limits[0], y_limits[1])
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_title(split.capitalize(), fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_xticks(x_values)
        ax.grid(True, which="both", alpha=0.25)
        if yscale == "log":
            ax.set_yscale("log")
        if y_limits is not None:
            ax.set_ylim(*y_limits)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=15, y=0.99)
    _add_legend_below(fig, axes)
    _caption(fig, caption)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_objective_curve_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
) -> None:
    runs = list(payload.get("runs") or [])
    curve_runs = []
    rows = pd.DataFrame(list(payload.get("aggregate_rows") or []))
    x_col, x_label, facet_values, facet_title = _recoverable_axis_spec(rows)
    for run in runs:
        fit_diag = dict(run.get("fit_diagnostics") or {})
        curve = list(fit_diag.get("train_loss_curve") or [])
        if len(curve) <= 1:
            continue
        curve_runs.append(
            {
                "baseline_family": str(run.get("baseline_family", "")),
                "facet_value": int(run.get(x_col, 0)),
                "config_label": str(run.get("config_label", "")),
                "fixed_leaf_tokens": int(run.get("fixed_leaf_tokens", 0)),
                "curve": np.asarray([float(value) for value in curve], dtype=np.float64),
            }
        )
    if not curve_runs:
        return

    ncols = min(2, len(facet_values))
    nrows = int(np.ceil(len(facet_values) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * max(1, nrows)))
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, facet_value in zip(axes_flat, facet_values):
        subset = [item for item in curve_runs if int(item["facet_value"]) == int(facet_value)]
        grouped: dict[tuple[str, str], list[np.ndarray]] = {}
        for item in subset:
            config_label = str(item["config_label"])
            if x_col != "train_doc_count":
                leaf_suffix = f"leaf{int(item['fixed_leaf_tokens'])}"
                config_label = f"{config_label}:{leaf_suffix}" if config_label else leaf_suffix
            grouped.setdefault(
                (
                    str(item["baseline_family"]),
                    config_label,
                ),
                [],
            ).append(np.asarray(item["curve"], dtype=np.float64))
        for (family, config_label), curves in sorted(grouped.items()):
            min_len = min(len(curve) for curve in curves)
            if min_len <= 1:
                continue
            aligned = np.vstack([curve[:min_len] for curve in curves])
            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)
            x = np.arange(1, min_len + 1, dtype=np.int64)
            color = BASELINE_COLORS.get(family)
            marker = BASELINE_MARKERS.get(family, "o")
            label = _line_label(family, config_label=config_label)
            ax.plot(x, mean_curve, color=color, marker=marker, markevery=max(1, min_len // 8), linewidth=2.0, label=label)
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
        ax.set_title(f"{x_label}={int(facet_value)}", fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("weighted training objective")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(facet_values):]:
        ax.axis("off")
    fig.suptitle(
        f"Optimization Diagnostics: Weighted Training Objective Curves by {facet_title}",
        fontsize=15,
        y=0.99,
    )
    _add_legend_below(fig, axes)
    _caption(
        fig,
        "These curves track the weighted optimization objective used during fitting. "
        "They are diagnostic-only and are not used for paper-facing ranking.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_unweighted_test_objective_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
) -> None:
    metric_specs = [
        ("val_unweighted_full_law_objective_mean", "Common Full-Law Val Objective"),
        ("val_unweighted_active_objective_mean", "Active-Term Val Objective"),
        ("test_unweighted_full_law_objective_mean", "Common Full-Law Test Objective"),
        ("test_unweighted_active_objective_mean", "Active-Term Test Objective"),
    ]
    if not any(metric in rows.columns for metric, _ in metric_specs):
        return
    baselines = _recoverable_plot_families(rows)
    x_col, x_label, x_values, _x_title = _recoverable_axis_spec(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), sharex=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, (metric_key, metric_label) in zip(axes_flat, metric_specs):
        std_key = metric_key.replace("_mean", "_std")
        any_finite = False
        for family in baselines:
            fam = rows.loc[rows["baseline_family"] == family].sort_values(x_col)
            if fam.empty:
                continue
            y = fam[metric_key].astype(float).to_numpy()
            if not np.isfinite(y).any():
                continue
            any_finite = True
            x = fam[x_col].astype(int).to_numpy()
            color = BASELINE_COLORS.get(family)
            marker = BASELINE_MARKERS.get(family, "o")
            ax.plot(x, y, color=color, marker=marker, linewidth=2.0, label=_pretty_family_name(family))
            if std_key in fam.columns:
                std = fam[std_key].astype(float).to_numpy()
                ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)
        if not any_finite:
            ax.axis("off")
            continue
        ax.set_title(metric_label, fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_xticks(x_values)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(metric_specs):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("unweighted objective")
    axes[1, 0].set_ylabel("unweighted objective")
    fig.suptitle("Diagnostic-Only: Unweighted Validation/Test Objectives", fontsize=15, y=0.99)
    _add_legend_below(fig, axes)
    _caption(
        fig,
        "Each panel is diagnostic-only. Full-law = root MAE + leaf MAE + C2 count-drift + merge MAE. "
        "Active-term = root MAE plus only the law terms active for that family. "
        "Validation is shown for model-selection context; test remains the paper-facing split.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_law_diagnostics_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
) -> None:
    c2_metric_key = (
        "test_c2_count_drift_r1_mae_mean"
        if "test_c2_count_drift_r1_mae_mean" in rows.columns
        else "test_c2_idempotence_mae_mean"
    )
    metric_specs = [
        ("test_leaf_mae_mean", "C1 / Leaf MAE"),
        (c2_metric_key, "C2 / Count Drift MAE"),
        ("test_merge_mae_mean", "C3 / Merge MAE"),
        ("test_schedule_spread_mean_mean", "Merge-Order Spread (schedule spread)"),
    ]
    if not any(metric in rows.columns for metric, _ in metric_specs):
        return
    baselines = _recoverable_plot_families(rows)
    x_col, x_label, x_values, _x_title = _recoverable_axis_spec(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for ax, (metric_key, metric_label) in zip(list(np.atleast_1d(axes).flat), metric_specs):
        if metric_key not in rows.columns:
            ax.axis("off")
            continue
        std_key = metric_key.replace("_mean", "_std")
        any_finite = False
        for family in baselines:
            fam = rows.loc[rows["baseline_family"] == family].sort_values(x_col)
            if fam.empty:
                continue
            y = fam[metric_key].astype(float).to_numpy()
            if not np.isfinite(y).any():
                continue
            any_finite = True
            x = fam[x_col].astype(int).to_numpy()
            color = BASELINE_COLORS.get(family)
            marker = BASELINE_MARKERS.get(family, "o")
            ax.plot(x, y, color=color, marker=marker, linewidth=2.0, label=_pretty_family_name(family))
            if std_key in fam.columns:
                std = fam[std_key].astype(float).to_numpy()
                ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)
        if not any_finite:
            ax.axis("off")
            continue
        ax.set_title(metric_label, fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_xticks(x_values)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Law Diagnostics Appendix", fontsize=15, y=0.99)
    _add_legend_below(fig, axes)
    _caption(
        fig,
        "Local-law and merge-order diagnostics are supporting measurements only. "
        "Merge-order spread is max(pred_root)-min(pred_root) across balanced, left-to-right, "
        "and right-to-left merge schedules. None of these metrics replace test root-count MAE for ranking.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_recoverable_contract_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
    rows: pd.DataFrame,
) -> None:
    """Contract checks page — dense two-column + bottom config."""
    fig = plt.figure(figsize=(11, 8.5))

    # Top section: two columns, positioned tighter
    ax_left = fig.add_axes([0.04, 0.55, 0.44, 0.38])
    ax_right = fig.add_axes([0.52, 0.55, 0.44, 0.38])
    ax_left.axis("off")
    ax_right.axis("off")

    sample_run = dict(list(payload.get("runs") or [])[0] if list(payload.get("runs") or []) else {})
    target_hist = dict(dict(sample_run.get("target_support") or {}).get("test") or {}).get("histogram", {})
    distinct_hist = dict(dict(sample_run.get("distinct_regime_support") or {}).get("test") or {}).get("histogram", {})

    ax_left.text(0.0, 1.0, "Support / Recoverability", fontsize=12, weight="bold", va="top")
    support_lines = [
        f"Test root-count support: {target_hist}",
        f"Test distinct-regime support: {distinct_hist}",
        _fixed_eval_line(payload),
        (
            "Primary score: "
            f"{payload.get('primary_report_metric', 'test_root_mae_mean')} "
            f"({payload.get('primary_report_weighting', 'unweighted_mae')})"
        ),
        f"Dev selection metric: {payload.get('dev_selection_metric', 'val_root_mae_mean')}",
        "Same full token-sequence inputs across baselines",
        "palette_block_exact = recoverability witness",
    ]
    ax_left.text(0.0, 0.88, "\n".join(f"  {s}" for s in support_lines), va="top",
                 fontsize=9, family="monospace", linespacing=1.5)

    readout = dict(payload.get("diagnostic_readout") or {})
    ax_right.text(0.0, 1.0, "Headline Readout", fontsize=12, weight="bold", va="top")
    headline_lines = [
        f"Status: {readout.get('status', 'NA')}",
        f"Best FNO root MAE: {float(readout.get('fno_best_root_mae_mean', float('nan'))):.4g}",
        f"Best FNO train docs: {int(readout.get('fno_best_train_doc_count', 0)) if 'fno_best_train_doc_count' in readout else 'NA'}",
        f"FNO seed std at best: {float(readout.get('fno_seed_std_at_best', float('nan'))):.4g}",
        f"Best control: {readout.get('best_control_family', 'NA')} (MAE={float(readout.get('best_control_root_mae_mean', float('nan'))):.4g})",
        f"Gap to best control: {float(readout.get('gap_to_best_control', float('nan'))):.4g}",
        f"FNO data-scale gain: {float(readout.get('fno_data_scale_gain', float('nan'))):.4g}",
        "Train/val/error curves and law metrics are diagnostic-only",
    ]
    ax_right.text(0.0, 0.88, "\n".join(f"  {s}" for s in headline_lines), va="top",
                  fontsize=9, family="monospace", linespacing=1.5)

    # Bottom section: config — starts higher to close the gap
    ax_bottom = fig.add_axes([0.04, 0.04, 0.92, 0.48])
    ax_bottom.axis("off")

    bundle = dict(payload.get("bundle_manifest") or {})
    seeds = list(payload.get("seeds") or [])
    train_counts = payload.get("train_doc_counts")
    families = list(payload.get("baseline_families") or [])

    ax_bottom.text(0.0, 1.0, "Experiment Configuration", fontsize=12, weight="bold", va="top")
    config_lines = [
        f"Seeds: {seeds}",
        f"Train doc counts: {train_counts}",
        f"Baseline families: {', '.join(families)}",
        f"Bundles: {list(bundle.keys())}",
    ]

    for bname, binfo in bundle.items():
        if isinstance(binfo, dict):
            config_lines.append(f"  {bname}: train={binfo.get('train_docs', '?')}, "
                                f"val={binfo.get('val_docs', '?')}, test={binfo.get('test_docs', '?')}")

    fno_config_found = False
    for run in list(payload.get("runs") or []):
        if run.get("baseline_family") in ("official_fno", "official_fno_sumlen"):
            fit = dict(run.get("fit_diagnostics") or {})
            actual = dict(fit.get("baseline_fno_actual_config") or {})
            cfg = dict(run.get("config") or {})
            if actual:
                config_lines.append("")
                config_lines.append("FNO Architecture (actual):")
                for k, v in actual.items():
                    config_lines.append(f"  {k}: {v}")
                fno_config_found = True
            elif cfg:
                config_lines.append("")
                config_lines.append("FNO Config (from run):")
                for k in ("hidden_dim", "n_epochs", "lr", "batch_size",
                          "doc_sequence_fno_pooling", "doc_sequence_fno_concat_length_feature"):
                    if k in cfg:
                        config_lines.append(f"  {k}: {cfg[k]}")
                fno_config_found = True
            break

    if not fno_config_found:
        config_lines.append("")
        config_lines.append("FNO Architecture: n_layers=4, n_modes=min(16,L//2), width=max(64,state_dim)")
        config_lines.append("  (hardcoded in markov_neural_operator_baselines.py)")

    ax_bottom.text(0.0, 0.88, "\n".join(config_lines), va="top",
                   fontsize=9, family="monospace", linespacing=1.4)

    fig.suptitle("Recoverable Contract Checks", fontsize=15, y=0.98)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_heatmap_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
    *,
    title: str,
    metric_key: str,
    metric_label: str,
    cmap: str,
    norm: Normalize,
    annotate_formatter: str,
    midpoint: float,
    caption: str,
) -> None:
    baselines = _ordered_baselines(rows, focus_only=True)
    regimes = _ordered_regimes(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes_flat = list(axes.flat)
    for idx, (ax, family) in enumerate(zip(axes_flat, baselines)):
        matrix = _make_metric_matrix(rows, baseline_family=family, metric_key=metric_key, regimes=regimes)
        display = np.asarray(matrix, dtype=np.float64).copy()
        if isinstance(norm, LogNorm):
            display = np.clip(display, float(norm.vmin), float(norm.vmax))
        im = ax.imshow(display, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(_pretty_family_name(family), color=BASELINE_COLORS.get(family, "black"),
                      fontsize=11, pad=6)
        ax.set_xticks(range(len(SEGMENT_BAND_ORDER)))
        ax.set_xticklabels(list(SEGMENT_BAND_ORDER))
        ax.set_yticks(range(len(regimes)))
        ax.set_yticklabels([str(int(x)) for x in regimes])
        ax.set_xlabel("segment density")
        ax.set_ylabel("regimes")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                label = "NA" if not np.isfinite(value) else format(float(value), annotate_formatter)
                color = "white" if np.isfinite(display[i, j]) and float(display[i, j]) >= float(midpoint) else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color,
                        weight="bold")
        if family == "official_fno_sumlen":
            flat = []
            for i, regime in enumerate(regimes):
                for j, band in enumerate(SEGMENT_BAND_ORDER):
                    if np.isfinite(matrix[i, j]):
                        flat.append((float(matrix[i, j]), int(regime), str(band)))
            if flat:
                easiest = min(flat, key=lambda item: item[0])
                hardest = max(flat, key=lambda item: item[0])
                ax.text(
                    0.02, 0.02,
                    f"Easiest: r{easiest[1]}/{easiest[2]}\nHardest: r{hardest[1]}/{hardest[2]}",
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric_label)
    for ax in axes_flat[len(baselines):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, y=0.99)
    _caption(fig, caption)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_structural_line_pages(
    pdf: PdfPages,
    rows: pd.DataFrame,
    *,
    train_doc_count: int,
    root_mae_floor: float,
    root_mae_ceiling: float,
) -> None:
    baselines = _ordered_baselines(rows, focus_only=True)
    regimes = _ordered_regimes(rows)

    # Page 1: MAE vs regimes, faceted by segment density
    fig, axes = plt.subplots(1, len(SEGMENT_BAND_ORDER), figsize=(11, 4.8), sharey=True)
    for ax, band in zip(np.atleast_1d(axes), SEGMENT_BAND_ORDER):
        subset = rows.loc[rows["segment_density_band"] == str(band)].copy()
        for family in baselines:
            fam = subset.loc[subset["baseline_family"] == family].sort_values("n_regimes")
            if fam.empty:
                continue
            x = fam["n_regimes"].astype(int).to_numpy()
            y = _root_mae_floor_clip(
                fam["test_root_mae_mean"].astype(float).to_numpy(),
                floor=root_mae_floor, ceiling=root_mae_ceiling,
            )
            color = BASELINE_COLORS.get(family)
            ax.plot(x, y, color=color, marker=BASELINE_MARKERS.get(family, "o"),
                    linewidth=2.0, label=_pretty_family_name(family))
            if "test_root_mae_std" in fam.columns:
                std = fam["test_root_mae_std"].astype(float).to_numpy()
                lo = _root_mae_floor_clip(y - std, floor=root_mae_floor, ceiling=root_mae_ceiling)
                hi = _root_mae_floor_clip(y + std, floor=root_mae_floor, ceiling=root_mae_ceiling)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_title(f"density: {band}", fontsize=11)
        ax.set_xlabel("regimes")
        ax.set_yscale("log")
        ax.set_ylim(float(root_mae_floor), float(root_mae_ceiling))
        ax.grid(True, which="both", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("root MAE")

    fig.suptitle(f"Root MAE vs Regimes | train_docs={int(train_doc_count)}", fontsize=14, y=0.99)
    _add_legend_below(fig, axes)
    _caption(fig, "Each panel fixes segment density. Shading = +/- 1 std across seeds.")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: MAE vs segment density, faceted by regime count
    fig, axes = plt.subplots(1, len(regimes), figsize=(11, 4.8), sharey=True)
    x_positions = np.arange(len(SEGMENT_BAND_ORDER), dtype=np.float64)
    for ax, regime in zip(np.atleast_1d(axes), regimes):
        subset = rows.loc[rows["n_regimes"] == int(regime)].copy()
        for family in baselines:
            fam = subset.loc[subset["baseline_family"] == family].copy()
            if fam.empty:
                continue
            fam["segment_density_band"] = pd.Categorical(
                fam["segment_density_band"], categories=list(SEGMENT_BAND_ORDER), ordered=True,
            )
            fam = fam.sort_values("segment_density_band")
            x = x_positions[: len(fam)]
            y = _root_mae_floor_clip(
                fam["test_root_mae_mean"].astype(float).to_numpy(),
                floor=root_mae_floor, ceiling=root_mae_ceiling,
            )
            color = BASELINE_COLORS.get(family)
            ax.plot(x, y, color=color, marker=BASELINE_MARKERS.get(family, "o"),
                    linewidth=2.0, label=_pretty_family_name(family))
            if "test_root_mae_std" in fam.columns:
                std = fam["test_root_mae_std"].astype(float).to_numpy()
                lo = _root_mae_floor_clip(y - std, floor=root_mae_floor, ceiling=root_mae_ceiling)
                hi = _root_mae_floor_clip(y + std, floor=root_mae_floor, ceiling=root_mae_ceiling)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_title(f"regimes: {int(regime)}", fontsize=11)
        ax.set_xlabel("segment density")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(SEGMENT_BAND_ORDER))
        ax.set_yscale("log")
        ax.set_ylim(float(root_mae_floor), float(root_mae_ceiling))
        ax.grid(True, which="both", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("root MAE")

    fig.suptitle(f"Root MAE vs Segment Density | train_docs={int(train_doc_count)}", fontsize=14, y=0.99)
    _add_legend_below(fig, axes)
    _caption(fig, "Each panel fixes regime count. Shading = +/- 1 std across seeds.")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_structural_improvement_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
    *,
    base_train_doc_count: int,
    final_train_doc_count: int,
    root_mae_floor: float,
    root_mae_ceiling: float,
) -> None:
    """Line/dot plot showing MAE improvement (1x -> 10x) per grid cell, faceted by density."""
    baselines = _ordered_baselines(rows, focus_only=True)
    regimes = _ordered_regimes(rows)

    fig, axes = plt.subplots(1, len(SEGMENT_BAND_ORDER), figsize=(11, 5.0), sharey=True)
    for ax, band in zip(np.atleast_1d(axes), SEGMENT_BAND_ORDER):
        base_slice = rows.loc[
            (rows["train_doc_count"] == int(base_train_doc_count))
            & (rows["segment_density_band"] == str(band))
        ]
        final_slice = rows.loc[
            (rows["train_doc_count"] == int(final_train_doc_count))
            & (rows["segment_density_band"] == str(band))
        ]
        for family in baselines:
            base_fam = base_slice.loc[base_slice["baseline_family"] == family].sort_values("n_regimes")
            final_fam = final_slice.loc[final_slice["baseline_family"] == family].sort_values("n_regimes")
            if base_fam.empty or final_fam.empty:
                continue
            # Align on regimes
            merged = base_fam.set_index("n_regimes")[["test_root_mae_mean"]].rename(
                columns={"test_root_mae_mean": "base_mae"}
            ).join(
                final_fam.set_index("n_regimes")[["test_root_mae_mean"]].rename(
                    columns={"test_root_mae_mean": "final_mae"}
                ),
                how="inner",
            )
            if merged.empty:
                continue
            x = merged.index.astype(int).to_numpy()
            improvement = merged["base_mae"].to_numpy() - merged["final_mae"].to_numpy()
            color = BASELINE_COLORS.get(family)
            marker = BASELINE_MARKERS.get(family, "o")
            ax.plot(x, improvement, color=color, marker=marker, linewidth=2.0,
                    markersize=8, label=_pretty_family_name(family))

        ax.axhline(0, color="#888888", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"density: {band}", fontsize=11)
        ax.set_xlabel("regimes")
        ax.set_xticks(regimes)
        ax.grid(True, alpha=0.25)
        # Color the background: green above 0, red below 0
        ylims = ax.get_ylim()
        ax.axhspan(0, max(ylims[1], 0.01), color="#d4edda", alpha=0.15, zorder=0)
        ax.axhspan(min(ylims[0], -0.01), 0, color="#f8d7da", alpha=0.15, zorder=0)

    np.atleast_1d(axes)[0].set_ylabel("MAE improvement (higher = better)")

    fig.suptitle(
        f"MAE Improvement: {int(base_train_doc_count)} -> {int(final_train_doc_count)} docs",
        fontsize=14, y=0.99,
    )
    _add_legend_below(fig, axes)
    _caption(fig, "Above zero (green zone) = 10x data helped. Below zero (red zone) = regression. Exact witness should be at zero.")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_structural_stability_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
    rows: pd.DataFrame,
) -> None:
    baselines = _ordered_baselines(rows, focus_only=True)
    cells = sorted(
        rows["cell_id"].dropna().astype(str).unique().tolist(),
        key=lambda cell: (int(rows.loc[rows["cell_id"] == cell, "n_regimes"].iloc[0]), cell),
    )
    n_cells = len(cells)
    ncols = min(n_cells, 2)
    nrows = (n_cells + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.0 * nrows + 0.5), sharey=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    x_positions = np.arange(len(baselines), dtype=np.float64)
    y_max = float(
        np.nanmax(np.asarray(rows["test_root_mae_max"].astype(float).to_list(), dtype=np.float64))
    )
    for ax, cell in zip(axes_flat, cells):
        subset = rows.loc[rows["cell_id"] == cell].copy()
        for idx, family in enumerate(baselines):
            fam = subset.loc[subset["baseline_family"] == family]
            if fam.empty:
                continue
            row = fam.iloc[0]
            ax.errorbar(
                [x_positions[idx]], [float(row["test_root_mae_mean"])],
                yerr=[float(row["test_root_mae_std"])],
                fmt=BASELINE_MARKERS.get(family, "o"),
                color=BASELINE_COLORS.get(family), capsize=5, markersize=8, linewidth=1.5,
            )
        ax.set_title(str(cell), fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([_pretty_family_name(f) for f in baselines], rotation=25, ha="right",
                           fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        worst = subset.sort_values("test_root_mae_mean", ascending=False).iloc[0]
        ax.text(
            0.02, 0.96,
            f"worst: {_pretty_family_name(str(worst['baseline_family']))}\nMAE={float(worst['test_root_mae_mean']):.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for ax in axes_flat[len(cells):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("root MAE mean +/- std")
    for ax in axes_flat[: len(cells)]:
        ax.set_ylim(0.0, max(1e-6, y_max * 1.15))
    fig.suptitle("Structural 10x Stability Anchors", fontsize=15, y=0.99)
    _caption(fig, "Small error bars vs large means = systematic bias, not seed noise.")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    pdf.savefig(fig)
    plt.close(fig)

    # Stability table page
    _draw_styled_table(
        pdf, rows,
        columns=[
            "cell_id", "baseline_family",
            "test_root_mae_mean", "test_root_mae_std",
            "test_root_mae_min", "test_root_mae_max",
            "test_exact_match_rate_mean",
        ],
        title="Stability Mean/Std Table",
        subtitle=_control_exactness_line(payload),
        sort_by=["cell_id", "baseline_family"],
        focus_only=True,
    )


def _draw_styled_table(
    pdf: PdfPages,
    rows: pd.DataFrame,
    *,
    columns: list[str],
    title: str,
    subtitle: str = "",
    sort_by: list[str] | None = None,
    focus_only: bool = False,
) -> None:
    """Draw a publication-quality table page with proper formatting."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    display = rows.copy()
    if focus_only:
        baselines = _ordered_baselines(rows, focus_only=True)
        display = display.loc[display["baseline_family"].isin(baselines)]

    keep = [c for c in columns if c in display.columns]
    display = display[keep]
    if sort_by:
        sort_cols = [c for c in sort_by if c in display.columns]
        if sort_cols:
            display = display.sort_values(sort_cols)

    # Drop columns where all values are empty/blank or all identical
    for col in list(display.columns):
        vals = display[col].astype(str).str.strip()
        if vals.eq("").all() or vals.eq("nan").all():
            display = display.drop(columns=[col])
        elif len(vals.unique()) == 1 and col not in (
            "test_root_mae_mean",
            "test_root_mae_std",
            "test_exact_match_rate_mean",
            "parameterization",
            "optimization_root_weight",
            "local_law_c1_weight",
            "local_law_c2_weight",
            "local_law_c3_weight",
            "tree_root_supervision_kind",
            "tree_leaf_fno_width",
            "tree_leaf_fno_n_modes",
            "tree_leaf_fno_n_layers",
            "tree_aux_doc_sequence_fraction",
            "task_objective_weight_source",
            "c2_metric_kind",
            "comparison_semantics_label",
        ):
            display = display.drop(columns=[col])

    if display.empty or display.shape[1] == 0:
        ax.text(
            0.5,
            0.56,
            title,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            transform=ax.transAxes,
        )
        message = "No non-degenerate tabular columns were available for this view."
        if subtitle:
            message = f"{message}\n{subtitle}"
        ax.text(
            0.5,
            0.45,
            message,
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
            wrap=True,
        )
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Format numeric columns
    for col in display.columns:
        if (
            col.startswith("test_root_mae")
            or col.startswith("test_exact_match")
            or col.endswith("_weight")
            or col == "optimization_root_weight"
            or col.endswith("_gap_pct_vs_best_fno")
        ):
            display[col] = display[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")

    col_labels = [_COL_RENAME.get(c, c) for c in display.columns]

    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    _style_table(table, len(display))

    ax.set_title(title, fontsize=15, pad=20)
    if subtitle:
        _caption(fig, subtitle)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_tree_neural_semantics_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
) -> None:
    rows = _tree_neural_semantics_rows(payload)
    if rows.empty:
        return
    validation = dict(payload.get("tree_neural_validation_summary") or {})
    subtitle_lines = []
    if validation:
        subtitle_lines.append(
            "Aligned recoverable comparison: "
            f"all_laws_worse_than_c2_only="
            f"{bool(validation.get('all_laws_worse_than_c2_only_still_holds', False))}"
        )
    legacy_rows = rows.loc[rows.get("comparison_semantics", pd.Series(dtype=str)) == "legacy"]
    if not legacy_rows.empty:
        subtitle_lines.append("Legacy rows detected and labeled separately")
    _draw_styled_table(
        pdf,
        rows,
        columns=[
            "baseline_family",
            "config_label",
            "tuning_stage",
            "train_doc_count",
            "tree_root_supervision_kind",
            "tree_leaf_fno_width",
            "tree_leaf_fno_n_modes",
            "tree_leaf_fno_n_layers",
            "tree_aux_doc_sequence_fraction",
            "parameterization",
            "optimization_root_weight",
            "local_law_c1_weight",
            "local_law_c2_weight",
            "local_law_c3_weight",
            "task_objective_weight_source",
            "c2_metric_kind",
            "comparison_semantics_label",
        ],
        title="Tree-Neural Objective Semantics",
        subtitle=" | ".join(subtitle_lines),
        sort_by=["train_doc_count", "baseline_family"],
        focus_only=False,
    )


def _draw_fair_parity_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
) -> None:
    rows = _fair_parity_rows(payload)
    if rows.empty:
        return
    summary = dict(payload.get("tree_fno_fair_parity_summary") or {})
    subtitle = (
        f"cfg={summary.get('parity_config_label', '')} | "
        f"root={summary.get('tree_root_supervision_kind', '')} | "
        f"leaf_fno=({summary.get('tree_leaf_fno_width')},"
        f"{summary.get('tree_leaf_fno_n_modes')},"
        f"{summary.get('tree_leaf_fno_n_layers')}) | "
        f"aux_seq_frac={float(summary.get('tree_aux_doc_sequence_fraction', float('nan'))):.4g} | "
        f"primary={bool(summary.get('primary_success_met', False))} | "
        f"secondary={bool(summary.get('secondary_success_met', False))}"
    )
    _draw_styled_table(
        pdf,
        rows,
        columns=[
            "train_doc_count",
            "best_full_doc_fno_family",
            "best_full_doc_fno_test_root_mae_mean",
            "tree_neural_test_root_mae_mean",
            "tree_neural_c2_test_root_mae_mean",
            "tree_neural_c2c3_test_root_mae_mean",
            "best_parity_tree_family",
            "best_parity_tree_test_root_mae_mean",
            "tree_neural_gap_pct_vs_best_fno",
            "best_parity_tree_gap_pct_vs_best_fno",
            "primary_success_within_10pct",
            "secondary_success_within_10pct",
        ],
        title="FNO vs Tree Fair-Parity",
        subtitle=subtitle,
        sort_by=["train_doc_count"],
        focus_only=False,
    )


def _draw_upper_bound_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
) -> None:
    rows = _upper_bound_rows(payload)
    if rows.empty:
        return
    summary = dict(payload.get("tree_fno_upper_bound_summary") or {})
    subtitle = (
        "Appendix-only upper bound | "
        f"aux_fracs={list(summary.get('aux_fractions') or [])} | "
        f"best_gate_aux={float(summary.get('best_gate_aux_fraction', float('nan'))):.4g} | "
        f"best_gate_tree={_pretty_family_name(str(summary.get('best_gate_upper_bound_family', '')))}"
    )
    _draw_styled_table(
        pdf,
        rows,
        columns=[
            "train_doc_count",
            "tree_aux_doc_sequence_fraction",
            "best_full_doc_fno_family",
            "best_full_doc_fno_test_root_mae_mean",
            "tree_neural_test_root_mae_mean",
            "tree_neural_c2_test_root_mae_mean",
            "tree_neural_c2c3_test_root_mae_mean",
            "best_upper_bound_tree_family",
            "best_upper_bound_tree_test_root_mae_mean",
            "best_upper_bound_tree_gap_pct_vs_best_fno",
        ],
        title="Tree+Aux Upper Bound",
        subtitle=subtitle,
        sort_by=["train_doc_count", "tree_aux_doc_sequence_fraction"],
        focus_only=False,
    )


def _recoverable_sections(payload: Mapping[str, Any], rows: pd.DataFrame) -> list[tuple[str, Sequence[str]]]:
    readout = dict(payload.get("diagnostic_readout") or {})
    efficiency = dict(payload.get("learning_efficiency_summary") or {})
    budget_frontier = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
    parity = dict(payload.get("tree_fno_fair_parity_summary") or {})
    cheapest = dict(efficiency.get("cheapest_within_10pct") or {})
    learned = rows.loc[rows["baseline_family"].isin(LEARNED_COMPARISON_FAMILIES)].copy()
    best_learned = _best_row(learned, "test_root_mae_mean", ascending=True)
    bullets_headline = []
    if best_learned is not None:
        bullets_headline.append(
            f"Best learned: {_pretty_family_name(str(best_learned['baseline_family']))} "
            f"at {int(best_learned['train_doc_count'])} docs, MAE={float(best_learned['test_root_mae_mean']):.4g}"
        )
    if cheapest:
        bullets_headline.append(
            "Cheapest near-best point: "
            f"{_pretty_family_name(str(cheapest.get('baseline_family', '')))} "
            f"by {int(cheapest.get('first_within_10pct_train_doc_count', 0))} docs "
            f"(within 10% of that family's best MAE)"
        )
    bullets_headline.extend([
        f"Best control: {readout.get('best_control_family', 'NA')} MAE={float(readout.get('best_control_root_mae_mean', float('nan'))):.4g}",
        f"FNO data-scale gain: {float(readout.get('fno_data_scale_gain', float('nan'))):.4g}",
        f"FNO seed std at best: {float(readout.get('fno_seed_std_at_best', float('nan'))):.4g}",
        _fixed_eval_line(payload),
    ])
    if parity:
        bullets_headline.append(
            "Fair parity gate: "
            f"primary={bool(parity.get('primary_success_met', False))}, "
            f"secondary={bool(parity.get('secondary_success_met', False))}, "
            f"best FNO={_pretty_family_name(str(parity.get('best_full_doc_fno_family_at_gate', '')))}, "
            f"best tree={_pretty_family_name(str(parity.get('best_parity_tree_family_at_gate', '')))}"
        )
    if dict(payload.get("tree_fno_upper_bound_summary") or {}):
        bullets_headline.append(
            "Tree+aux upper-bound appendix is shown separately; width/layers/modes sweeps and runtime frontiers belong in the dedicated tree-FNO tuning PDF."
        )
    if budget_frontier:
        best_tree_rows = list(budget_frontier.get("best_tree_by_budget") or [])
        if best_tree_rows:
            cheapest_tree = min(
                best_tree_rows,
                key=lambda row: (
                    float(row.get("budget_total_calls_per_doc", float("inf"))),
                    float(row.get("test_root_mae_mean", float("inf"))),
                ),
            )
            bullets_headline.append(
                "Oracle-attention budget frontier: cheapest best-tree point in the current sweep is "
                f"{_pretty_family_name(str(cheapest_tree.get('baseline_family', '')))} "
                f"at {float(cheapest_tree.get('budget_total_calls_per_doc', float('nan'))):.4g} calls/doc "
                f"with full-doc share={float(cheapest_tree.get('full_doc_budget_share', float('nan'))):.4g}."
            )
    return [
        (
            "What this report shows",
            [
                "The paper-facing ranking is mean test root-count MAE over seeds on the fixed test split.",
                "Train/val metrics, exact match, objective curves, and law metrics remain visible but are diagnostic-only.",
            ],
        ),
        (
            "DGP: piecewise_disjoint_palette",
            [
                "Each hidden regime owns a disjoint token palette.",
                "Target = document-level changepoint count; adjacent transitions carry direct evidence.",
                "palette_block_exact proves recoverability (counts palette-block transitions from tokens).",
            ],
        ),
        ("Headline findings", bullets_headline),
    ]


def _recoverable_page_titles(payload: Mapping[str, Any], rows: pd.DataFrame) -> list[str]:
    titles = [
        "Recoverable Scale Summary",
        "Recoverable Scale Primary Ranking",
    ]
    if not _budget_frontier_rows(payload).empty:
        titles.append("Oracle Attention Budget Share")
    if dict(payload.get("learning_efficiency_summary") or {}).get("families"):
        titles.append("Learning Efficiency Frontier")
    titles.extend(
        [
            "Split Diagnostics: Final Root MAE by Split",
            "Split Diagnostics: Final Exact-Match by Split",
            "Optimization Diagnostics: Weighted Training Objective Curves",
            "Diagnostic-Only: Unweighted Validation/Test Objectives",
            "Law Diagnostics Appendix",
            "Recoverable Scale: Per-Seed Detail",
            "Recoverable Contract Checks",
        ]
    )
    if not _tree_neural_semantics_rows(payload).empty:
        titles.append("Tree-Neural Objective Semantics")
    if not _fair_parity_rows(payload).empty:
        titles.append("FNO vs Tree Fair-Parity")
    if not _upper_bound_rows(payload).empty:
        titles.append("Tree+Aux Upper Bound")
    titles.append("Recoverable Aggregate Rows")
    return titles


def _draw_budget_frontier_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
) -> None:
    rows = _budget_frontier_rows(payload)
    if rows.empty:
        return
    _draw_styled_table(
        pdf,
        rows,
        columns=[
            "budget_total_calls_per_doc",
            "best_tree_family",
            "best_tree_test_root_mae_mean",
            "full_doc_budget_share",
            "doc_consumption_mode",
            "local_split_mode",
            "effective_full_doc_mass_per_doc_mean",
            "best_reference_family",
            "best_reference_test_root_mae_mean",
        ],
        title="Oracle Attention Budget Share",
        subtitle=(
            "Best tree policy and best document-only reference at each raw-call budget. "
            "Raw budget is enforced in oracle calls; effective full-doc mass is reported only as a companion interpretation."
        ),
        sort_by=["budget_total_calls_per_doc"],
        focus_only=False,
    )


def _draw_learning_efficiency_page(
    pdf: PdfPages,
    payload: Mapping[str, Any],
) -> None:
    efficiency = dict(payload.get("learning_efficiency_summary") or {})
    rows = list(efficiency.get("families") or [])
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if "baseline_family" in df.columns:
        df["baseline_family"] = df["baseline_family"].map(_pretty_family_name)
    df = df.rename(
        columns={
            "baseline_family": "family",
            "best_train_doc_count": "best docs",
            "best_test_root_mae_mean": "best MAE",
            "first_within_10pct_train_doc_count": "within 10%",
            "first_within_25pct_train_doc_count": "within 25%",
        }
    )
    _draw_styled_table(
        pdf,
        df,
        columns=["family", "best docs", "best MAE", "within 10%", "within 25%"],
        title="Learning Efficiency Frontier",
        subtitle=(
            "Earliest train size that is already near each family's own best test-root-MAE point."
        ),
        sort_by=["within 10%", "best MAE"],
        focus_only=False,
    )


def _structural_grid_sections(payload: Mapping[str, Any], rows: pd.DataFrame) -> list[tuple[str, Sequence[str]]]:
    summary = dict(payload.get("grid_diagnostic_summary") or {})
    train_doc_counts = _ordered_train_doc_counts(rows)
    target_family = str(summary.get("target_family", "official_fno_sumlen"))
    bullets = [
        "Grid hardens recoverable DGP along two axes: more regimes + denser segment schedules.",
        "palette_block_exact staying exact = grid remains recoverable even when learned models fail.",
        f"Low-scale dominant failure axis: {summary.get('main_failure_axis', 'unknown')}.",
    ]
    if len(train_doc_counts) >= 2:
        bullets.append(
            f"Main comparison: {int(train_doc_counts[0])} vs {int(train_doc_counts[-1])} train docs."
        )
    target_rows = rows.loc[rows["baseline_family"] == target_family].copy()
    if not target_rows.empty and len(train_doc_counts) >= 2:
        base = target_rows.loc[target_rows["train_doc_count"] == int(train_doc_counts[0])]
        final = target_rows.loc[target_rows["train_doc_count"] == int(train_doc_counts[-1])]
        if not base.empty and not final.empty:
            bullets.append(
                f"{_pretty_family_name(target_family)} mean MAE: "
                f"{float(base['test_root_mae_mean'].mean()):.4g} -> {float(final['test_root_mae_mean'].mean()):.4g}"
            )
    return [
        (
            "What this report shows",
            [
                "Whether the structural grid remains recoverable in principle.",
                "Whether 10x train data rescues learned models across the grid, not just on easy cells, under the same test-root-MAE ranking contract.",
            ],
        ),
        (
            "DGP: piecewise_disjoint_palette (hardened)",
            [
                "Same generator; hardness from more regimes + more boundaries.",
                "Exact witness isolates learned-model failure from DGP non-recoverability.",
            ],
        ),
        ("Headline findings", bullets),
    ]


def _structural_stability_sections(payload: Mapping[str, Any], rows: pd.DataFrame) -> list[tuple[str, Sequence[str]]]:
    summary = dict(payload.get("grid_diagnostic_summary") or {})
    target_family = str(summary.get("target_family", "official_fno_sumlen"))
    target_rows = rows.loc[rows["baseline_family"] == target_family].copy()
    mean_std = float(target_rows["test_root_mae_std"].mean()) if not target_rows.empty else float("nan")
    mean_mae = float(target_rows["test_root_mae_mean"].mean()) if not target_rows.empty else float("nan")
    return [
        (
            "What this report shows",
            [
                "Whether remaining error at 10x scale is seed instability or systematic bias.",
                "Whether the same learned family wins consistently across anchor cells under the same test-root-MAE ranking contract.",
            ],
        ),
        (
            "DGP: structural_core_v1 anchors",
            [
                "Subset of structural grid. palette_block_exact = recoverability witness.",
            ],
        ),
        (
            "Headline findings",
            [
                f"Target family: {_pretty_family_name(target_family)}.",
                f"Mean seed std = {mean_std:.4g} vs mean MAE = {mean_mae:.4g}.",
                "Small std/mean ratio = systematic bias, not seed noise.",
            ],
        ),
    ]


def _draw_generic_aggregate_table(pdf: PdfPages, rows: pd.DataFrame, *, title: str) -> None:
    columns = [col for col in (
        "baseline_family", "config_label", "tuning_stage", "train_doc_count", "fixed_leaf_tokens", "cell_id",
        "n_regimes", "segment_density_band",
        "test_root_mae_mean", "test_root_mae_std",
        "test_exact_match_rate_mean", "n_runs",
    ) if col in rows.columns]
    _draw_styled_table(
        pdf, rows,
        columns=columns,
        title=title,
        sort_by=columns[:3],
        focus_only=False,
    )


def main() -> int:
    from scripts._markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_full_doc_anchor_diagnostics_pdf.py",
        replacements=(
            "python3 scripts/report_markov_optimization_tradeoffs.py --summary-json <tradeoff_pipeline/tradeoff_report/summary.json>",
            "python3 scripts/run_markov_publication_bundle.py --config <...> --plan-only",
        ),
        note=(
            "The legacy full-doc anchor PDF is archived. Use the canonical v3 "
            "tradeoff/publication report path instead."
        ),
    )

    args = parse_args()
    summary_json = Path(str(args.summary_json))
    payload = _load_payload(summary_json)
    output_pdf = Path(str(args.output_pdf)) if str(args.output_pdf).strip() else summary_json.with_suffix(".pdf")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(list(payload.get("aggregate_rows") or []))
    if rows.empty:
        raise ValueError(f"no aggregate_rows found in {summary_json}")

    mode = _infer_report_mode(payload)
    root_mae_floor = float(max(1e-12, args.root_mae_floor))
    root_mae_ceiling = float(max(root_mae_floor * 10.0, args.root_mae_ceiling))
    default_title = {
        "recoverable_scale": "Markov Recoverable Scale Report",
        "structural_grid": "Markov Structural Grid Report",
        "structural_stability": "Markov Structural Stability Report",
    }[mode]
    report_title = str(args.title).strip() or default_title

    with PdfPages(output_pdf) as pdf:
        if mode == "recoverable_scale":
            _draw_text_summary_page(
                pdf, title=report_title,
                subtitle=f"source benchmark: {payload.get('benchmark')}",
                sections=_recoverable_sections(payload, rows),
            )
            _draw_recoverable_primary_ranking_page(
                pdf, payload, rows,
                root_mae_floor=root_mae_floor, root_mae_ceiling=root_mae_ceiling,
            )
            _draw_budget_frontier_page(pdf, payload)
            _draw_learning_efficiency_page(pdf, payload)
            _draw_recoverable_split_metric_page(
                pdf,
                rows,
                metric_base="root_mae",
                title="Split Diagnostics: Final Root MAE by Split",
                ylabel="root MAE",
                caption=(
                    "Train/val/test root-count MAE are shown separately for diagnosis. "
                    "Only the test panel feeds the paper-facing ranking."
                ),
                yscale="log",
                y_limits=(float(root_mae_floor), float(root_mae_ceiling)),
            )
            _draw_recoverable_split_metric_page(
                pdf,
                rows,
                metric_base="exact_match_rate",
                title="Split Diagnostics: Final Exact-Match by Split",
                ylabel="exact-match rate",
                caption=(
                    "Exact-match rates are secondary diagnostics. They help interpret whether "
                    "MAE improvements reflect nearby count errors or exact recovery."
                ),
                y_limits=(-0.02, 1.02),
            )
            _draw_objective_curve_page(pdf, payload)
            _draw_unweighted_test_objective_page(pdf, rows)
            _draw_law_diagnostics_page(pdf, rows)
            _draw_recoverable_perseed_page(
                pdf, payload, rows,
                root_mae_floor=root_mae_floor, root_mae_ceiling=root_mae_ceiling,
            )
            _draw_recoverable_contract_page(pdf, payload, rows)
            _draw_tree_neural_semantics_page(pdf, payload)
            _draw_fair_parity_page(pdf, payload)
            _draw_upper_bound_page(pdf, payload)
            _draw_generic_aggregate_table(pdf, rows, title="Recoverable Aggregate Rows")

        elif mode == "structural_grid":
            _draw_text_summary_page(
                pdf, title=report_title,
                subtitle=f"hardness grid: {payload.get('hardness_grid')}",
                sections=_structural_grid_sections(payload, rows),
            )
            train_doc_counts = _ordered_train_doc_counts(rows)
            for train_doc_count in train_doc_counts:
                slice_df = rows.loc[rows["train_doc_count"] == int(train_doc_count)].copy()
                _draw_heatmap_page(
                    pdf, slice_df,
                    title=f"Root MAE Heatmaps | train_docs={int(train_doc_count)}",
                    metric_key="test_root_mae_mean",
                    metric_label=f"root MAE (log [{root_mae_floor:.0e}, {root_mae_ceiling:.0e}])",
                    cmap="RdYlGn_r",  # red=high MAE (bad), green=low MAE (good)
                    norm=LogNorm(vmin=root_mae_floor, vmax=root_mae_ceiling),
                    annotate_formatter=".3f",
                    midpoint=float(np.sqrt(root_mae_floor * root_mae_ceiling)),
                    caption="Red = high error (bad). Green = low error (good). Exact witness should be green.",
                )
            if len(train_doc_counts) >= 2:
                _draw_structural_improvement_page(
                    pdf, rows,
                    base_train_doc_count=int(train_doc_counts[0]),
                    final_train_doc_count=int(train_doc_counts[-1]),
                    root_mae_floor=root_mae_floor,
                    root_mae_ceiling=root_mae_ceiling,
                )
            for train_doc_count in train_doc_counts:
                _draw_structural_line_pages(
                    pdf,
                    rows.loc[rows["train_doc_count"] == int(train_doc_count)].copy(),
                    train_doc_count=int(train_doc_count),
                    root_mae_floor=root_mae_floor, root_mae_ceiling=root_mae_ceiling,
                )
            _draw_tree_neural_semantics_page(pdf, payload)
            _draw_generic_aggregate_table(pdf, rows, title="Structural Grid Aggregate Rows")

        else:
            _draw_text_summary_page(
                pdf, title=report_title,
                subtitle=f"hardness grid: {payload.get('hardness_grid')} | train_docs={_ordered_train_doc_counts(rows)}",
                sections=_structural_stability_sections(payload, rows),
            )
            _draw_structural_stability_page(pdf, payload, rows)
            _draw_tree_neural_semantics_page(pdf, payload)
            _draw_generic_aggregate_table(pdf, rows, title="Stability Aggregate Rows")

    print(str(output_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
