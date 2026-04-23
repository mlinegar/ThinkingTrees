#!/usr/bin/env python3
"""
Summarize optimizer audit manifests and optional Markov recoverable diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.optimization.performance import summarize_optimizer_runs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optimizer_rows_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in list(manifest.get("entries") or []):
        run_dir = Path(str(entry.get("run_dir", "")))
        final_stats_path = run_dir / "final_stats.json"
        if not final_stats_path.exists():
            continue
        final_stats = _load_json(final_stats_path)
        diag = dict(final_stats.get("optimizer_diagnostics") or {})
        for row in list(diag.get("runs") or []):
            rows.append(dict(row))
        for row in list(diag.get("comparison_control_runs") or []):
            rows.append(dict(row))
    return rows


def _render_optimizer_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Optimizer | Component | Dataset Regime | Budget | Class | Success | Gain Rate | Median Held-Out Gain |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("optimizer_requested", "")),
                    str(row.get("component", "")),
                    str(row.get("dataset_regime", "")),
                    str(row.get("budget_mode", "")),
                    str(row.get("classification", "")),
                    f"{float(row.get('operational_success_rate', float('nan'))):.3f}",
                    f"{float(row.get('gain_rate', float('nan'))):.3f}",
                    f"{float(row.get('median_heldout_gain', float('nan'))):.6g}",
                ]
            )
            + " |"
        )
    return lines


def _render_markov_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Family | Train Docs | Test Root MAE | Gap To Ridge | Gap To Exact | Cause | Objective |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("baseline_family", "")),
                    str(int(row.get("train_doc_count", 0))),
                    f"{float(row.get('test_root_mae_mean', float('nan'))):.6g}",
                    f"{float(row.get('gap_to_ridge_control', float('nan'))):.6g}",
                    f"{float(row.get('gap_to_exact_witness', float('nan'))):.6g}",
                    str(row.get("cause_code", "")),
                    str(row.get("objective_variant", "")),
                ]
            )
            + " |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Report optimizer audit results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--markov-summary", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    optimizer_runs = _optimizer_rows_from_manifest(manifest)
    cell_summaries = summarize_optimizer_runs(optimizer_runs)
    ranking_rows = [
        row
        for row in cell_summaries
        if not bool(row.get("comparison_control_flag", False))
        and str(row.get("classification", "")) != "forced_control"
    ]
    forced_control_rows = [
        row
        for row in cell_summaries
        if bool(row.get("comparison_control_flag", False))
        or str(row.get("classification", "")) == "forced_control"
    ]

    payload: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "optimizer_run_count": len(optimizer_runs),
        "optimizer_cell_summaries": cell_summaries,
        "optimizer_ranking_rows": ranking_rows,
        "forced_control_rows": forced_control_rows,
    }

    markov_rows: List[Dict[str, Any]] = []
    if args.markov_summary:
        markov_payload = _load_json(Path(args.markov_summary))
        markov_rows = list(markov_payload.get("witness_gap_table") or [])
        payload["markov_summary"] = str(args.markov_summary)
        payload["markov_witness_gap_table"] = markov_rows

    output_root = manifest_path.parent
    summary_json = output_root / "optimizer_performance_summary.json"
    summary_md = output_root / "optimizer_performance_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Optimizer Performance Audit",
        "",
        "## DSPy Optimizer Matrix",
        "",
        *_render_optimizer_table(ranking_rows),
    ]
    if forced_control_rows:
        md_lines.extend(
            [
                "",
                "## Fixed Controls",
                "",
                "These rows are reported for visibility but excluded from the first-pass optimizer ranking.",
                "",
                *_render_optimizer_table(forced_control_rows),
            ]
        )
    if markov_rows:
        md_lines.extend([
            "",
            "## Markov Recoverable Witness-Gap Table",
            "",
            *_render_markov_table(markov_rows),
        ])
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
