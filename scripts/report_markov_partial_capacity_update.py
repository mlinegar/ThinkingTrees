#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.report.pdf_utils import write_image_page, write_text_page


DEFAULT_R20_PACKAGE_ORDER: Sequence[str] = (
    "full20",
    "full20_leaf_count10_internal_count10",
    "full20_leaf_count20_internal_count20",
    "full20_leaf_count50_internal_count50",
    "full20_leaf_count100_internal_count100",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a source-directory-safe partial Markov capacity update report."
    )
    parser.add_argument("--current-output-root", type=Path, required=True)
    parser.add_argument("--reference-report-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs")
        / f"markov_partial_capacity_update_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_sort_key(package_name: str) -> tuple[int, str]:
    try:
        return (list(DEFAULT_R20_PACKAGE_ORDER).index(str(package_name)), str(package_name))
    except ValueError:
        return (len(DEFAULT_R20_PACKAGE_ORDER), str(package_name))


def _summary_row(scope: str, package_name: str, summary_path: Path) -> Dict[str, Any]:
    payload = dict(_load_json(summary_path))
    winning = dict(payload.get("winning_config") or {})
    return {
        "scope": str(scope),
        "package_name": str(package_name),
        "summary_path": str(summary_path),
        "train_doc_count": int(payload.get("train_doc_count") or 0),
        "benchmark": str(payload.get("resolved_benchmark_name") or payload.get("benchmark") or ""),
        "winning_config_label": str(payload.get("winning_config_label") or ""),
        "test_root_mae_mean": float(winning.get("test_root_mae_mean") or float("nan")),
        "val_root_mae_mean": float(winning.get("val_root_mae_mean") or float("nan")),
    }


def _completed_rows(current_output_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"recoverable": [], "structural": []}
    package_root = current_output_root / "package_capacity"
    for scope in ("recoverable", "structural"):
        scope_root = package_root / scope
        if not scope_root.exists():
            continue
        for package_dir in sorted(scope_root.iterdir()):
            if not package_dir.is_dir():
                continue
            summary_path = package_dir / "tree_fno_capacity_locked_summary.json"
            if summary_path.exists():
                out[scope].append(_summary_row(scope, package_dir.name, summary_path))
        out[scope].sort(key=lambda row: _package_sort_key(str(row["package_name"])))
    return out


def _plot_scope(rows: Sequence[Mapping[str, Any]], *, title: str, note: str, output_path: Path) -> str:
    if not rows:
        return ""
    labels = [str(row["package_name"]) for row in rows]
    values = [float(row["test_root_mae_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(labels)))
    bars = ax.bar(xs, values, color=["#0f766e", "#b91c1c", "#e66a2c", "#1d4ed8", "#64748b"][: len(xs)])
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("test_root_mae")
    ax.set_title(title)
    ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def _reference_lines(reference_report_dir: Path | None) -> List[str]:
    if reference_report_dir is None:
        return []
    summary_path = reference_report_dir / "summary.json"
    pdf_path = reference_report_dir / "report.pdf"
    lines = [
        "Reference context is kept separate from the current-cohort plots.",
        f"- `reference report dir`: `{reference_report_dir}`",
    ]
    if pdf_path.exists():
        lines.append(f"- `reference pdf`: `{pdf_path}`")
    if summary_path.exists():
        lines.append(f"- `reference summary`: `{summary_path}`")
    return lines


def main() -> int:
    from scripts._markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_markov_partial_capacity_update.py",
        replacements=(
            "python3 scripts/report_markov_optimization_tradeoffs.py --summary-json <tradeoff_pipeline/tradeoff_report/summary.json>",
        ),
        note=(
            "The partial-capacity update report is archived. Use the canonical v3 "
            "tradeoff report package-ladder sections instead."
        ),
    )

    args = _parse_args()
    current_output_root = Path(args.current_output_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    completed = _completed_rows(current_output_root)
    recoverable_rows = list(completed["recoverable"])
    structural_rows = list(completed["structural"])

    recoverable_fig = _plot_scope(
        recoverable_rows,
        title="Current Recoverable Capacity Winners",
        note="Same current output root only. No older external reference points are drawn here.",
        output_path=figure_dir / "recoverable_current_cohort.png",
    )
    structural_fig = _plot_scope(
        structural_rows,
        title="Current Structural Capacity Winners",
        note="Same current output root only. No older external reference points are drawn here.",
        output_path=figure_dir / "structural_current_cohort.png",
    )

    md_lines = [
        "# Markov Partial Capacity Update",
        "",
        "This report is source-directory-safe: plots only include completed roots from the same `current-output-root`.",
        "",
        "**Current Cohort**",
        "",
        f"- `current-output-root`: `{current_output_root}`",
        f"- `completed recoverable roots`: `{len(recoverable_rows)}`",
        f"- `completed structural roots`: `{len(structural_rows)}`",
    ]
    ref_lines = _reference_lines(args.reference_report_dir)
    if ref_lines:
        md_lines.extend(["", "**Reference Context**", ""])
        md_lines.extend(ref_lines)
    md_lines.extend(
        [
            "",
            "**Completed Recoverable Roots**",
            "",
            "| Package | Winner | test_root_mae_mean | val_root_mae_mean |",
            "|---|---|---:|---:|",
        ]
    )
    for row in recoverable_rows:
        md_lines.append(
            f"| `{row['package_name']}` | `{row['winning_config_label']}` | {row['test_root_mae_mean']:.6f} | {row['val_root_mae_mean']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "**Completed Structural Roots**",
            "",
            "| Package | Winner | test_root_mae_mean | val_root_mae_mean |",
            "|---|---|---:|---:|",
        ]
    )
    for row in structural_rows:
        md_lines.append(
            f"| `{row['package_name']}` | `{row['winning_config_label']}` | {row['test_root_mae_mean']:.6f} | {row['val_root_mae_mean']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "**Artifacts**",
            "",
            f"- `recoverable figure`: `{recoverable_fig}`" if recoverable_fig else "- `recoverable figure`: unavailable",
            f"- `structural figure`: `{structural_fig}`" if structural_fig else "- `structural figure`: unavailable",
        ]
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_kind": "markov_partial_capacity_update",
        "current_output_root": str(current_output_root),
        "reference_report_dir": str(args.reference_report_dir.expanduser().resolve()) if args.reference_report_dir else "",
        "source_policy": {
            "plots_use_single_current_output_root": True,
            "external_reference_is_context_only": bool(args.reference_report_dir),
        },
        "completed": completed,
        "figures": {
            "recoverable_current_cohort": recoverable_fig,
            "structural_current_cohort": structural_fig,
        },
    }

    report_md = output_dir / "report.md"
    report_pdf = output_dir / "report.pdf"
    summary_json = output_dir / "summary.json"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with PdfPages(report_pdf) as pdf:
        write_text_page(pdf, title="Markov Partial Capacity Update", lines=md_lines[2:14])
        if ref_lines:
            write_text_page(pdf, title="Reference Context", lines=ref_lines)
        write_text_page(pdf, title="Recoverable Roots", lines=md_lines[md_lines.index("**Completed Recoverable Roots**") + 2 : md_lines.index("**Completed Structural Roots**")])
        write_text_page(pdf, title="Structural Roots", lines=md_lines[md_lines.index("**Completed Structural Roots**") + 2 : md_lines.index("**Artifacts**")])
        if recoverable_fig:
            write_image_page(pdf, image_path=Path(recoverable_fig), title="Current Recoverable Capacity Winners")
        if structural_fig:
            write_image_page(pdf, image_path=Path(structural_fig), title="Current Structural Capacity Winners")

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
