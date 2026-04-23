#!/usr/bin/env python3
"""Render a dedicated tree-FNO capacity/parity tuning PDF."""

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
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    load_markov_full_doc_anchor_diagnostics_from_output_dir,
)


BASELINE_COLORS = {
    "official_fno": "#1d4ed8",
    "official_fno_sumlen": "#60a5fa",
    "tree_neural_c2": "#166534",
    "tree_neural_c2c3": "#16a34a",
    "tree_neural": "#0f766e",
}
TREE_FAMILIES = ("tree_neural_c2", "tree_neural_c2c3", "tree_neural")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the dedicated tree-FNO tuning PDF from capacity/parity roots."
    )
    parser.add_argument("--capacity-root", type=str, required=True)
    parser.add_argument("--parity-root", type=str, required=True)
    parser.add_argument("--output-pdf", type=str, default="")
    return parser.parse_args()


def _pretty_family_name(family: str) -> str:
    mapping = {
        "official_fno": "Official FNO",
        "official_fno_sumlen": "FNO + Sum/Len",
        "tree_neural_c2": "Tree Neural (C2)",
        "tree_neural_c2c3": "Tree Neural (C2+C3)",
        "tree_neural": "Tree Neural (All Laws)",
    }
    return mapping.get(str(family), str(family).replace("_", " "))


def _screen_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = pd.DataFrame(list(payload.get("aggregate_rows") or []))
    if rows.empty:
        return rows
    return rows.loc[
        (rows["baseline_family"] == "tree_neural")
        & (rows["tuning_stage"] == "capacity_screen")
    ].copy()


def _locked_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = pd.DataFrame(list(payload.get("aggregate_rows") or []))
    if rows.empty:
        return rows
    return rows.loc[
        (rows["baseline_family"] == "tree_neural")
        & (rows["tuning_stage"] == "capacity_locked")
    ].copy()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _heatmap_matrix(
    rows: pd.DataFrame,
    *,
    metric_key: str,
    row_key: str,
    col_key: str,
    facet_key: str,
    facet_value: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    subset = rows.loc[rows[facet_key] == int(facet_value)].copy()
    row_values = sorted(int(x) for x in subset[row_key].dropna().unique().tolist())
    col_values = sorted(int(x) for x in subset[col_key].dropna().unique().tolist())
    matrix = np.full((len(row_values), len(col_values)), np.nan, dtype=np.float64)
    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            match = subset.loc[
                (subset[row_key] == int(row_value))
                & (subset[col_key] == int(col_value))
            ]
            if match.empty:
                continue
            matrix[i, j] = float(match.iloc[0][metric_key])
    return matrix, row_values, col_values


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
        wrap=True,
    )


def _draw_text_page(
    pdf: PdfPages,
    *,
    title: str,
    lines: Sequence[str],
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, title, fontsize=18, weight="bold", va="top")
    y = 0.9
    for line in lines:
        ax.text(0.04, y, line, fontsize=10, va="top", family="monospace")
        y -= 0.04
    pdf.savefig(fig)
    plt.close(fig)


def _draw_heatmap_facet_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
    *,
    metric_key: str,
    title: str,
    subtitle: str,
    row_key: str,
    col_key: str,
    facet_key: str,
) -> None:
    facet_values = sorted(int(x) for x in rows[facet_key].dropna().unique().tolist())
    if not facet_values:
        return
    ncols = min(3, len(facet_values))
    nrows = int(np.ceil(len(facet_values) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * max(nrows, 1)))
    axes_flat = list(np.atleast_1d(axes).flat)
    metric_values = rows[metric_key].astype(float).to_numpy()
    finite = metric_values[np.isfinite(metric_values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    for ax, facet_value in zip(axes_flat, facet_values):
        matrix, row_values, col_values = _heatmap_matrix(
            rows,
            metric_key=metric_key,
            row_key=row_key,
            col_key=col_key,
            facet_key=facet_key,
            facet_value=int(facet_value),
        )
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"{facet_key}={int(facet_value)}", fontsize=11)
        ax.set_xticks(range(len(col_values)))
        ax.set_xticklabels([str(int(value)) for value in col_values])
        ax.set_yticks(range(len(row_values)))
        ax.set_yticklabels([str(int(value)) for value in row_values])
        ax.set_xlabel(col_key.replace("tree_leaf_fno_", "").replace("_", " "))
        ax.set_ylabel(row_key.replace("tree_leaf_fno_", "").replace("_", " "))
        for i, row_value in enumerate(row_values):
            for j, col_value in enumerate(col_values):
                value = matrix[i, j]
                label = "NA" if not np.isfinite(value) else f"{float(value):.4g}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, weight="bold")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[len(facet_values):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=0.99)
    _caption(fig, subtitle)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_axis_marginals_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    axis_specs = [
        ("tree_leaf_fno_width", "width"),
        ("tree_leaf_fno_n_layers", "layers"),
        ("tree_leaf_fno_n_modes", "modes"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8))
    for ax, (axis_key, axis_label) in zip(np.atleast_1d(axes), axis_specs):
        grouped = rows.groupby(axis_key)["val_root_mae_mean"]
        x = np.asarray(sorted(int(key) for key in grouped.groups.keys()), dtype=np.int64)
        mean_vals = np.asarray(
            [float(grouped.get_group(int(key)).mean()) for key in x],
            dtype=np.float64,
        )
        best_vals = np.asarray(
            [
                float(
                    rows.loc[rows[axis_key] == int(key), "val_root_mae_mean"]
                    .astype(float)
                    .min()
                )
                for key in x
            ],
            dtype=np.float64,
        )
        ax.plot(x, mean_vals, color="#1d4ed8", marker="o", linewidth=2.0, label="mean over other knobs")
        ax.plot(x, best_vals, color="#0f766e", marker="s", linewidth=2.0, label="best over other knobs")
        ax.set_title(axis_label.capitalize(), fontsize=11)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("val root MAE")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Axis Marginals", fontsize=15, y=0.99)
    _caption(fig, "Validation-only marginal summaries used to understand capacity sensitivity without promoting test-set tuning.")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_efficiency_frontier_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    points = rows[
        [
            "config_label",
            "tree_leaf_fno_width",
            "tree_leaf_fno_n_modes",
            "tree_leaf_fno_n_layers",
            "elapsed_s_mean",
            "val_root_mae_mean",
        ]
    ].copy()
    points["elapsed_s_mean"] = points["elapsed_s_mean"].astype(float)
    points["val_root_mae_mean"] = points["val_root_mae_mean"].astype(float)
    ax.scatter(
        points["elapsed_s_mean"],
        points["val_root_mae_mean"],
        c=points["tree_leaf_fno_n_modes"].astype(float),
        cmap="viridis",
        s=80,
        alpha=0.9,
    )
    pareto = []
    running_best = float("inf")
    for row in points.sort_values(["elapsed_s_mean", "val_root_mae_mean"]).itertuples():
        if float(row.val_root_mae_mean) < running_best - 1e-12:
            pareto.append(row)
            running_best = float(row.val_root_mae_mean)
    if pareto:
        ax.plot(
            [float(item.elapsed_s_mean) for item in pareto],
            [float(item.val_root_mae_mean) for item in pareto],
            color="#111827",
            linewidth=2.0,
            linestyle="--",
            label="Pareto frontier",
        )
        for item in pareto:
            ax.annotate(
                str(item.config_label),
                xy=(float(item.elapsed_s_mean), float(item.val_root_mae_mean)),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("mean elapsed seconds per seed")
    ax.set_ylabel("val root MAE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Efficiency Frontier", fontsize=15, y=0.99)
    _caption(fig, "Pareto configs are those not dominated on both validation MAE and per-seed runtime.")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_law_diagnostics_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    top = rows.sort_values("val_root_mae_mean").head(10).copy()
    c2_col = (
        "test_c2_count_drift_r1_mae_mean"
        if "test_c2_count_drift_r1_mae_mean" in top.columns
        else "test_c2_idempotence_mae_mean"
    )
    display_cols = [
        "config_label",
        "tree_leaf_fno_width",
        "tree_leaf_fno_n_modes",
        "tree_leaf_fno_n_layers",
        "val_root_mae_mean",
        "test_root_mae_mean",
        "elapsed_s_mean",
        c2_col,
        "test_merge_mae_mean",
        "test_schedule_spread_mean_mean",
    ]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    table = ax.table(
        cellText=top[display_cols].round(4).astype(str).values.tolist(),
        colLabels=[
            "config",
            "width",
            "modes",
            "layers",
            "val MAE",
            "test MAE",
            "elapsed(s)",
            "C2 Count Drift",
            "merge",
            "schedule spread",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)
    ax.set_title("Law Diagnostics (Top Validation Configs)", fontsize=15, pad=16)
    _caption(
        fig,
        "These are supporting diagnostics only. Merge-order spread remains a stability proxy, not a ranking objective.",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_aux_upper_bound_page(pdf: PdfPages, parity_payload: Mapping[str, Any]) -> None:
    summary = dict(parity_payload.get("tree_fno_upper_bound_summary") or {})
    comparisons = list(summary.get("comparisons") or [])
    if not comparisons:
        return
    gate_count = int(summary.get("gate_train_doc_count", 10240))
    rows = [
        dict(row)
        for row in comparisons
        if int(row.get("train_doc_count", 0)) == int(gate_count)
    ]
    if not rows:
        return
    best_fno_mae = float(rows[0].get("best_full_doc_fno_test_root_mae_mean", float("nan")))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.asarray([float(row.get("tree_aux_doc_sequence_fraction", 0.0)) for row in rows], dtype=np.float64)
    for family in TREE_FAMILIES:
        y = np.asarray(
            [float(row.get(f"{family}_test_root_mae_mean", float("nan"))) for row in rows],
            dtype=np.float64,
        )
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            color=BASELINE_COLORS.get(family),
            label=_pretty_family_name(family),
        )
    ax.axhline(best_fno_mae, color="#111827", linestyle="--", linewidth=2.0, label="Best FNO")
    ax.set_xlabel("doc_sequence_train_fraction")
    ax.set_ylabel("test root MAE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Aux Upper Bound", fontsize=15, y=0.99)
    _caption(fig, "Appendix-only tree+aux comparison. This does not redefine the main pure-tree parity claim.")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_scale_backfill_page(pdf: PdfPages, parity_payload: Mapping[str, Any]) -> None:
    summary = dict(parity_payload.get("tree_fno_fair_parity_summary") or {})
    comparisons = list(summary.get("comparisons") or [])
    if len(comparisons) < 2:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.asarray([int(row.get("train_doc_count", 0)) for row in comparisons], dtype=np.int64)
    fno = np.asarray(
        [float(row.get("best_full_doc_fno_test_root_mae_mean", float("nan"))) for row in comparisons],
        dtype=np.float64,
    )
    best_tree = np.asarray(
        [float(row.get("best_parity_tree_test_root_mae_mean", float("nan"))) for row in comparisons],
        dtype=np.float64,
    )
    tree_neural = np.asarray(
        [float(row.get("tree_neural_test_root_mae_mean", float("nan"))) for row in comparisons],
        dtype=np.float64,
    )
    ax.plot(x, fno, color=BASELINE_COLORS["official_fno"], marker="o", linewidth=2.0, label="Best FNO")
    ax.plot(x, best_tree, color=BASELINE_COLORS["tree_neural_c2"], marker="s", linewidth=2.0, label="Best parity tree")
    ax.plot(x, tree_neural, color=BASELINE_COLORS["tree_neural"], marker="d", linewidth=2.0, label="Tree Neural")
    ax.set_xlabel("train docs")
    ax.set_ylabel("test root MAE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Scale Backfill", fontsize=15, y=0.99)
    _caption(fig, "Shown only after parity has been run across the scale curve.")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _tuning_report_page_titles(
    capacity_screen_payload: Mapping[str, Any],
    capacity_locked_payload: Mapping[str, Any],
    parity_payload: Mapping[str, Any],
) -> list[str]:
    titles = [
        "Summary",
        "Validation Selection Heatmaps: Width x Layers | facet Modes",
        "Validation Selection Heatmaps: Width x Modes | facet Layers",
        "Validation Selection Heatmaps: Layers x Modes | facet Width",
        "Axis Marginals",
        "Post-Hoc Test Diagnostics: Width x Layers | facet Modes",
        "Post-Hoc Test Diagnostics: Width x Modes | facet Layers",
        "Post-Hoc Test Diagnostics: Layers x Modes | facet Width",
        "Efficiency Frontier",
        "Law Diagnostics",
    ]
    if dict(parity_payload.get("tree_fno_upper_bound_summary") or {}):
        titles.append("Aux Upper Bound")
    parity = dict(parity_payload.get("tree_fno_fair_parity_summary") or {})
    if len(list(parity.get("comparisons") or [])) >= 2:
        titles.append("Scale Backfill")
    return titles


def main() -> int:
    from scripts._markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_tree_fno_tuning_pdf.py",
        replacements=(
            "python3 scripts/report_markov_optimization_tradeoffs.py --summary-json <tradeoff_pipeline/tradeoff_report/summary.json>",
            "python3 scripts/run_markov_publication_bundle.py --config <...> --plan-only",
        ),
        note=(
            "The dedicated tree/FNO tuning PDF is archived. The supported v3 surface "
            "is the canonical tradeoff/publication report."
        ),
    )

    args = parse_args()
    capacity_root = Path(str(args.capacity_root))
    parity_root = Path(str(args.parity_root))
    output_pdf = (
        Path(str(args.output_pdf))
        if str(args.output_pdf).strip()
        else parity_root / "tree_fno_tuning_report.pdf"
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    capacity_screen_payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(
        capacity_root / "screen"
    )
    capacity_locked_payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(
        capacity_root / "locked"
    )
    parity_payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(parity_root)
    capacity_locked_summary = _load_summary(
        capacity_root / "tree_fno_capacity_locked_summary.json"
    )

    screen_rows = _screen_rows(capacity_screen_payload)
    locked_rows = _locked_rows(capacity_locked_payload)
    if screen_rows.empty or locked_rows.empty:
        raise ValueError("capacity screen/locked payloads must be non-empty")

    parity = dict(parity_payload.get("tree_fno_fair_parity_summary") or {})
    upper = dict(parity_payload.get("tree_fno_upper_bound_summary") or {})
    winning = dict(capacity_locked_summary.get("winning_config") or {})
    summary_lines = [
        f"best pure-tree locked config: {winning.get('config_label', '')}",
        (
            "leaf-FNO width/modes/layers: "
            f"{int(winning.get('tree_leaf_fno_width', 0))}/"
            f"{int(winning.get('tree_leaf_fno_n_modes', 0))}/"
            f"{int(winning.get('tree_leaf_fno_n_layers', 0))}"
        ),
        f"best FNO at gate: {parity.get('best_full_doc_fno_family_at_gate', '')}",
        (
            "pure-tree gap vs best FNO at gate: "
            f"{100.0 * float(parity.get('tree_neural_gap_ratio_vs_best_fno_at_gate', float('nan'))):.3g}%"
        ),
        (
            "best aux upper-bound gap vs best FNO at gate: "
            f"{100.0 * float(upper.get('best_gate_upper_bound_gap_ratio_vs_best_fno', float('nan'))):.3g}%"
        ),
        f"primary success met: {bool(parity.get('primary_success_met', False))}",
        f"secondary success met: {bool(parity.get('secondary_success_met', False))}",
    ]

    with PdfPages(output_pdf) as pdf:
        _draw_text_page(pdf, title="Tree-FNO Tuning Report", lines=summary_lines)
        _draw_heatmap_facet_page(
            pdf,
            screen_rows,
            metric_key="val_root_mae_mean",
            title="Validation Selection Heatmaps: Width x Layers | facet Modes",
            subtitle="Validation metric only. Used for capacity selection.",
            row_key="tree_leaf_fno_width",
            col_key="tree_leaf_fno_n_layers",
            facet_key="tree_leaf_fno_n_modes",
        )
        _draw_heatmap_facet_page(
            pdf,
            screen_rows,
            metric_key="val_root_mae_mean",
            title="Validation Selection Heatmaps: Width x Modes | facet Layers",
            subtitle="Validation metric only. Used for capacity selection.",
            row_key="tree_leaf_fno_width",
            col_key="tree_leaf_fno_n_modes",
            facet_key="tree_leaf_fno_n_layers",
        )
        _draw_heatmap_facet_page(
            pdf,
            screen_rows,
            metric_key="val_root_mae_mean",
            title="Validation Selection Heatmaps: Layers x Modes | facet Width",
            subtitle="Validation metric only. Used for capacity selection.",
            row_key="tree_leaf_fno_n_layers",
            col_key="tree_leaf_fno_n_modes",
            facet_key="tree_leaf_fno_width",
        )
        _draw_axis_marginals_page(pdf, screen_rows)
        _draw_heatmap_facet_page(
            pdf,
            screen_rows,
            metric_key="test_root_mae_mean",
            title="Post-Hoc Test Diagnostics: Width x Layers | facet Modes",
            subtitle="Diagnostic only. Not used for capacity selection.",
            row_key="tree_leaf_fno_width",
            col_key="tree_leaf_fno_n_layers",
            facet_key="tree_leaf_fno_n_modes",
        )
        _draw_heatmap_facet_page(
            pdf,
            screen_rows,
            metric_key="test_root_mae_mean",
            title="Post-Hoc Test Diagnostics: Width x Modes | facet Layers",
            subtitle="Diagnostic only. Not used for capacity selection.",
            row_key="tree_leaf_fno_width",
            col_key="tree_leaf_fno_n_modes",
            facet_key="tree_leaf_fno_n_layers",
        )
        _draw_heatmap_facet_page(
            pdf,
            screen_rows,
            metric_key="test_root_mae_mean",
            title="Post-Hoc Test Diagnostics: Layers x Modes | facet Width",
            subtitle="Diagnostic only. Not used for capacity selection.",
            row_key="tree_leaf_fno_n_layers",
            col_key="tree_leaf_fno_n_modes",
            facet_key="tree_leaf_fno_width",
        )
        _draw_efficiency_frontier_page(pdf, screen_rows)
        _draw_law_diagnostics_page(pdf, screen_rows)
        _draw_aux_upper_bound_page(pdf, parity_payload)
        if len(list(parity.get("comparisons") or [])) >= 2:
            _draw_scale_backfill_page(pdf, parity_payload)

    print(str(output_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
