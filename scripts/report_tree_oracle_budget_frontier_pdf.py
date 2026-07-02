#!/usr/bin/env python3
"""Render the dedicated oracle-budget-share frontier PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


BASELINE_COLORS = {
    "official_fno": "#1d4ed8",
    "official_fno_sumlen": "#60a5fa",
    "tree_doc_ridge": "#6b7280",
    "tree_neural_c2": "#166534",
    "tree_neural_c2c3": "#16a34a",
    "tree_neural": "#0f766e",
}
MODE_STYLES = {"root_only": "-", "doc_sequence": "--"}
LOCAL_SPLIT_STYLES = {
    "balanced": "-",
    "leaf_heavy": "--",
    "internal_heavy": ":",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the dedicated tree oracle budget-share frontier PDF."
    )
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-pdf", type=str, default="")
    return parser.parse_args()


def _pretty_family_name(family: str) -> str:
    mapping = {
        "official_fno": "Official FNO",
        "official_fno_sumlen": "FNO + Sum/Len",
        "tree_doc_ridge": "Doc-Span Ridge",
        "tree_neural_c2": "Tree Neural (C2)",
        "tree_neural_c2c3": "Tree Neural (C2+C3)",
        "tree_neural": "Tree Neural (All Laws)",
    }
    return mapping.get(str(family), str(family).replace("_", " "))


def _budget_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = pd.DataFrame(list(payload.get("aggregate_rows") or []))
    if rows.empty:
        return rows
    if "budget_total_calls" not in rows.columns:
        rows["budget_total_calls"] = 0
    if "budget_total_calls_per_doc" not in rows.columns:
        rows["budget_total_calls_per_doc"] = 0.0
    if "study_name" not in rows.columns:
        rows["study_name"] = ""
    rows = rows.loc[
        (rows["budget_total_calls"].fillna(0).astype(int) > 0)
        | (rows["budget_total_calls_per_doc"].fillna(0.0).astype(float) > 0.0)
        | (rows["study_name"].fillna("").astype(str) == "oracle_budget_share_frontier")
    ].copy()
    for key in (
        "budget_total_calls_per_doc",
        "full_doc_budget_share",
        "effective_full_doc_mass_per_doc_mean",
        "test_root_mae_mean",
    ):
        if key in rows.columns:
            rows[key] = rows[key].astype(float)
    for key in ("doc_consumption_mode", "local_split_mode", "baseline_family"):
        if key in rows.columns:
            rows[key] = rows[key].astype(str)
    return rows


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


def _draw_text_page(pdf: PdfPages, *, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, title, fontsize=18, weight="bold", va="top")
    y = 0.9
    for line in lines:
        ax.text(0.04, y, line, fontsize=10, va="top")
        y -= 0.04
    pdf.savefig(fig)
    plt.close(fig)


def _draw_mae_vs_budget_by_share(pdf: PdfPages, rows: pd.DataFrame) -> None:
    tree_rows = rows.loc[rows["baseline_family"].str.startswith("tree_")].copy()
    share_values = sorted(tree_rows["full_doc_budget_share"].dropna().unique().tolist())
    if not share_values:
        return
    ncols = min(3, len(share_values))
    nrows = int(np.ceil(len(share_values) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * max(nrows, 1)), sharey=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, share in zip(axes_flat, share_values):
        subset = rows.loc[np.isclose(rows["full_doc_budget_share"], float(share))].copy()
        for family in sorted(subset["baseline_family"].unique().tolist()):
            fam = subset.loc[subset["baseline_family"] == family].sort_values(
                "budget_total_calls_per_doc"
            )
            ax.plot(
                fam["budget_total_calls_per_doc"].to_numpy(),
                fam["test_root_mae_mean"].to_numpy(),
                marker="o",
                linewidth=2.0,
                color=BASELINE_COLORS.get(str(family)),
                label=_pretty_family_name(str(family)),
            )
        ax.set_title(f"full-doc share={float(share):.2f}", fontsize=11)
        ax.set_xlabel("raw oracle calls / doc")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(share_values):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("test root MAE")
    fig.suptitle("Root MAE vs Raw Oracle Calls | faceted by full-doc budget share", fontsize=15, y=0.99)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    _caption(
        fig,
        "Non-tree references appear only at full-doc share = 1.0. Canonical metric remains mean test root-count MAE.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_mae_vs_share_by_budget(pdf: PdfPages, rows: pd.DataFrame) -> None:
    budget_values = sorted(rows["budget_total_calls_per_doc"].dropna().unique().tolist())
    if not budget_values:
        return
    ncols = min(3, len(budget_values))
    nrows = int(np.ceil(len(budget_values) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * max(nrows, 1)), sharey=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, budget in zip(axes_flat, budget_values):
        subset = rows.loc[
            np.isclose(rows["budget_total_calls_per_doc"], float(budget))
        ].copy()
        for family in sorted(subset["baseline_family"].unique().tolist()):
            fam = subset.loc[subset["baseline_family"] == family].sort_values(
                "full_doc_budget_share"
            )
            ax.plot(
                fam["full_doc_budget_share"].to_numpy(),
                fam["test_root_mae_mean"].to_numpy(),
                marker="o",
                linewidth=2.0,
                color=BASELINE_COLORS.get(str(family)),
                label=_pretty_family_name(str(family)),
            )
        ax.set_title(f"calls/doc={float(budget):.2f}", fontsize=11)
        ax.set_xlabel("full-doc budget share")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(budget_values):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("test root MAE")
    fig.suptitle("Root MAE vs Full-Doc Budget Share | faceted by raw-call budget", fontsize=15, y=0.99)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    _caption(
        fig,
        "Tree families sweep alpha_full_doc < 1.0; non-tree references stay at alpha_full_doc = 1.0 by construction.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_doc_mode_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    subset = rows.loc[
        rows["baseline_family"].isin(["tree_neural", "tree_neural_c2", "tree_neural_c2c3"])
        & (rows["local_split_mode"] == "balanced")
    ].copy()
    if subset.empty:
        return
    budget_values = sorted(subset["budget_total_calls_per_doc"].dropna().unique().tolist())
    ncols = min(3, len(budget_values))
    nrows = int(np.ceil(len(budget_values) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * max(nrows, 1)), sharey=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, budget in zip(axes_flat, budget_values):
        budget_rows = subset.loc[
            np.isclose(subset["budget_total_calls_per_doc"], float(budget))
        ].copy()
        for family in sorted(budget_rows["baseline_family"].unique().tolist()):
            for doc_mode in ("root_only", "doc_sequence"):
                fam = budget_rows.loc[
                    (budget_rows["baseline_family"] == family)
                    & (budget_rows["doc_consumption_mode"] == doc_mode)
                ].sort_values("full_doc_budget_share")
                if fam.empty:
                    continue
                ax.plot(
                    fam["full_doc_budget_share"].to_numpy(),
                    fam["test_root_mae_mean"].to_numpy(),
                    marker="o",
                    linewidth=2.0,
                    linestyle=MODE_STYLES.get(doc_mode, "-"),
                    color=BASELINE_COLORS.get(str(family)),
                    label=f"{_pretty_family_name(str(family))} | {doc_mode}",
                )
        ax.set_title(f"calls/doc={float(budget):.2f}", fontsize=11)
        ax.set_xlabel("full-doc budget share")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(budget_values):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("test root MAE")
    fig.suptitle("Document Mode Comparison: root_only vs doc_sequence", fontsize=15, y=0.99)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    _caption(
        fig,
        "Document labels are paid once either way; this page compares consuming the same document-label budget through tree-root supervision versus doc-sequence supervision.",
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_local_split_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    subset = rows.loc[
        rows["baseline_family"].isin(["tree_neural", "tree_neural_c2", "tree_neural_c2c3"])
        & (rows["doc_consumption_mode"] == "root_only")
        & (rows["full_doc_budget_share"] < 1.0)
    ].copy()
    if subset.empty:
        return
    budget_values = sorted(subset["budget_total_calls_per_doc"].dropna().unique().tolist())
    ncols = min(3, len(budget_values))
    nrows = int(np.ceil(len(budget_values) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * max(nrows, 1)), sharey=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, budget in zip(axes_flat, budget_values):
        budget_rows = subset.loc[
            np.isclose(subset["budget_total_calls_per_doc"], float(budget))
        ].copy()
        for family in sorted(budget_rows["baseline_family"].unique().tolist()):
            for local_split in ("balanced", "leaf_heavy", "internal_heavy"):
                fam = budget_rows.loc[
                    (budget_rows["baseline_family"] == family)
                    & (budget_rows["local_split_mode"] == local_split)
                ].sort_values("full_doc_budget_share")
                if fam.empty:
                    continue
                ax.plot(
                    fam["full_doc_budget_share"].to_numpy(),
                    fam["test_root_mae_mean"].to_numpy(),
                    marker="o",
                    linewidth=2.0,
                    linestyle=LOCAL_SPLIT_STYLES.get(local_split, "-"),
                    color=BASELINE_COLORS.get(str(family)),
                    label=f"{_pretty_family_name(str(family))} | {local_split}",
                )
        ax.set_title(f"calls/doc={float(budget):.2f}", fontsize=11)
        ax.set_xlabel("full-doc budget share")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(budget_values):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("test root MAE")
    fig.suptitle("Local Remainder Allocation: balanced vs leaf-heavy vs internal-heavy", fontsize=15, y=0.99)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    _caption(
        fig,
        "These curves vary only the local-remainder split under the same raw-call budget and full-doc budget share. The main curve remains the balanced local split.",
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_effective_mass_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for family in sorted(rows["baseline_family"].unique().tolist()):
        fam = rows.loc[rows["baseline_family"] == family].copy()
        fam = fam.sort_values("effective_full_doc_mass_per_doc_mean")
        ax.plot(
            fam["effective_full_doc_mass_per_doc_mean"].to_numpy(),
            fam["test_root_mae_mean"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=BASELINE_COLORS.get(str(family)),
            label=_pretty_family_name(str(family)),
        )
    ax.set_xlabel("effective full-doc label mass / doc")
    ax.set_ylabel("test root MAE")
    ax.grid(True, alpha=0.25)
    fig.suptitle("Effective Full-Doc Label Mass Companion View", fontsize=15, y=0.99)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    _caption(
        fig,
        "The enforced budget is raw oracle calls. Effective full-doc mass is a secondary interpretation layer: a 16/64 leaf contributes 0.25, a 32/64 internal span contributes 0.5, and a document label contributes 1.0.",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _budget_report_page_titles(payload: Mapping[str, Any]) -> list[str]:
    rows = _budget_rows(payload)
    titles = [
        "Oracle Attention Budget Share Summary",
        "Root MAE vs Raw Oracle Calls",
        "Root MAE vs Full-Doc Budget Share",
        "Document Mode Comparison",
        "Local Remainder Allocation",
        "Effective Full-Doc Label Mass",
    ]
    if rows.empty:
        return titles[:1]
    return titles


def _coverage_summary(summary_json: Path, rows: pd.DataFrame) -> dict[str, Any]:
    output_root = summary_json.parent
    manifest_path = output_root / "mig_job_manifest.json"
    controller_results_path = output_root / "controller_results.json"
    expected_cells: set[tuple[str, float, float, str, str]] = set()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        for job in list(dict(manifest or {}).get("jobs") or []):
            family = str(job.get("family", "")).strip()
            if not family:
                continue
            expected_cells.add(
                (
                    family,
                    float(job.get("budget_total_calls_per_doc", 0.0) or 0.0),
                    float(job.get("full_doc_budget_share", 1.0) if job.get("full_doc_budget_share", 1.0) not in {"", None} else 1.0),
                    str(job.get("doc_consumption_mode", "")),
                    str(job.get("local_split_mode", "")),
                )
            )
    completed_cells: set[tuple[str, float, float, str, str]] = set()
    if not rows.empty:
        for row in rows.to_dict(orient="records"):
            completed_cells.add(
                (
                    str(row.get("baseline_family", "")),
                    float(row.get("budget_total_calls_per_doc", 0.0) or 0.0),
                    float(row.get("full_doc_budget_share", 1.0) if row.get("full_doc_budget_share", 1.0) not in {"", None} else 1.0),
                    str(row.get("doc_consumption_mode", "")),
                    str(row.get("local_split_mode", "")),
                )
            )
    missing_cells = sorted(expected_cells - completed_cells)
    failed_jobs = None
    if controller_results_path.exists():
        try:
            controller_results = json.loads(controller_results_path.read_text(encoding="utf-8"))
        except Exception:
            controller_results = {}
        if isinstance(controller_results, dict):
            failed_jobs = list(controller_results.get("failed_jobs") or [])
    return {
        "expected_cells": len(expected_cells),
        "completed_cells": len(completed_cells),
        "missing_cells": missing_cells,
        "failed_jobs": failed_jobs,
    }


def main() -> int:
    args = parse_args()
    summary_json = Path(str(args.summary_json))
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid payload in {summary_json}")
    output_pdf = (
        Path(str(args.output_pdf))
        if str(args.output_pdf).strip()
        else summary_json.parent / "tree_oracle_budget_frontier_report.pdf"
    )
    rows = _budget_rows(payload)
    if rows.empty:
        raise ValueError(f"no budget frontier rows found in {summary_json}")

    summary = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
    best_tree = list(summary.get("best_tree_by_budget") or [])
    best_ref = list(summary.get("best_reference_by_budget") or [])
    coverage = _coverage_summary(summary_json, rows)
    missing_cells = list(coverage.get("missing_cells") or [])
    failed_jobs = coverage.get("failed_jobs")
    lines = [
        f"benchmark: {payload.get('benchmark', '')}",
        f"train-doc budgets: {list(summary.get('budget_levels_per_doc') or [])}",
        f"full-doc shares: {list(summary.get('full_doc_budget_shares') or [])}",
        f"tree families: {sorted(rows.loc[rows['baseline_family'].str.startswith('tree_'), 'baseline_family'].unique().tolist())}",
        f"reference families: {sorted(rows.loc[~rows['baseline_family'].str.startswith('tree_'), 'baseline_family'].unique().tolist())}",
        f"coverage: {int(coverage.get('completed_cells', 0))} / {int(coverage.get('expected_cells', 0))} aggregate cells complete",
        (
            f"missing aggregate cells: {len(missing_cells)}"
            if missing_cells
            else "missing aggregate cells: 0"
        ),
        (
            f"failed jobs recorded by controller: {len(failed_jobs)}"
            if isinstance(failed_jobs, list)
            else "failed jobs recorded by controller: unavailable"
        ),
        "",
        "best tree by budget:",
        *[
            f"  calls/doc={float(row.get('budget_total_calls_per_doc', float('nan'))):.4g} -> "
            f"{_pretty_family_name(str(row.get('baseline_family', '')))} "
            f"(mae={float(row.get('test_root_mae_mean', float('nan'))):.4g}, "
            f"alpha={float(row.get('full_doc_budget_share', float('nan'))):.4g}, "
            f"doc_mode={str(row.get('doc_consumption_mode', ''))}, "
            f"local_split={str(row.get('local_split_mode', ''))})"
            for row in best_tree
        ],
        "",
        "best document-only reference by budget:",
        *[
            f"  calls/doc={float(row.get('budget_total_calls_per_doc', float('nan'))):.4g} -> "
            f"{_pretty_family_name(str(row.get('baseline_family', '')))} "
            f"(mae={float(row.get('test_root_mae_mean', float('nan'))):.4g})"
            for row in best_ref
        ],
    ]
    if missing_cells:
        lines.extend(
            [
                "",
                "missing cells (shown as absent elsewhere in this partial report):",
                *[
                    f"  family={family}, calls/doc={budget:.4g}, alpha={share:.4g}, doc_mode={doc_mode}, local_split={local_split}"
                    for family, budget, share, doc_mode, local_split in missing_cells[:12]
                ],
            ]
        )
        if len(missing_cells) > 12:
            lines.append(f"  ... plus {len(missing_cells) - 12} more")
    

    with PdfPages(output_pdf) as pdf:
        _draw_text_page(
            pdf,
            title="Oracle Attention Budget Share Frontier",
            lines=lines,
        )
        _draw_mae_vs_budget_by_share(pdf, rows)
        _draw_mae_vs_share_by_budget(pdf, rows)
        _draw_doc_mode_page(pdf, rows)
        _draw_local_split_page(pdf, rows)
        _draw_effective_mass_page(pdf, rows)

    print(str(output_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
