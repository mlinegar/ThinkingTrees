#!/usr/bin/env python3
"""Common Markov route output schema and lightweight report helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


COMMON_METRIC_FIELDS: tuple[str, ...] = (
    "cell_id",
    "status",
    "route",
    "objective",
    "backend",
    "input_encoding",
    "summary_family",
    "leaf_tokens",
    "doc_tokens",
    "train_docs",
    "eval_docs",
    "n_iter",
    "epochs",
    "batch_size",
    "hidden_dim",
    "channels",
    "n_modes",
    "seed",
    "root_count_mae",
    "theta_mae",
    "theta_first_regime_accuracy",
    "theta_last_regime_accuracy",
    "eps_leaf",
    "eps_merge",
    "eps_idemp",
    "contextual_mae",
    "pred_truth_corr",
    "pred_std",
    "leaf_first_acc",
    "leaf_last_acc",
    "merge_first_acc",
    "merge_last_acc",
    "full_exact_rate",
    "boundary_f1",
    "elapsed_sec",
    "output_root",
)


def _nested(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _clean_number(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _count_mae(section: Mapping[str, Any]) -> Any:
    diag = section.get("count_diagnostics")
    if isinstance(diag, Mapping):
        return diag.get("root_mae")
    return None


def normalize_jax_summary(
    summary: Mapping[str, Any],
    *,
    cell_id: str,
    objective: str,
    output_root: str,
    status: str = "completed",
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    """Normalize a contextual-sbijax summary into the route schema."""

    test = dict(_nested(summary, ("diagnostics", "test")) or {})
    args = dict(summary.get("args") or {})
    provenance = dict(summary.get("provenance") or {})
    return {
        "cell_id": cell_id,
        "status": status,
        "route": "jax",
        "objective": objective,
        "backend": "jax",
        "input_encoding": summary.get("input_encoding") or provenance.get("input_encoding"),
        "summary_family": provenance.get("local_law_summary_family"),
        "leaf_tokens": provenance.get("fragment_len") or args.get("fragment_len"),
        "doc_tokens": _nested(summary, ("data_source_metadata", "doc_tokens")),
        "train_docs": args.get("train_docs"),
        "eval_docs": args.get("test_docs"),
        "n_iter": args.get("n_iter"),
        "epochs": None,
        "batch_size": args.get("batch_size"),
        "hidden_dim": args.get("hidden_dim") or provenance.get("hidden_dim"),
        "channels": None,
        "n_modes": provenance.get("local_law_summary_fno_n_modes"),
        "seed": args.get("seed") or provenance.get("seed"),
        "root_count_mae": test.get("theta_count_raw_mae"),
        "theta_mae": test.get("theta_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
        "eps_leaf": test.get("eps_leaf"),
        "eps_merge": test.get("eps_merge"),
        "eps_idemp": test.get("eps_idemp"),
        "contextual_mae": test.get("contextual_mae"),
        "pred_truth_corr": test.get("pred_truth_corr"),
        "pred_std": test.get("pred_std"),
        "leaf_first_acc": test.get("theta_first_regime_accuracy"),
        "leaf_last_acc": test.get("theta_last_regime_accuracy"),
        "merge_first_acc": None,
        "merge_last_acc": None,
        "full_exact_rate": None,
        "boundary_f1": None,
        "elapsed_sec": elapsed_sec,
        "output_root": output_root,
    }


def normalize_pytorch_summary(
    summary: Mapping[str, Any],
    *,
    cell_id: str,
    objective: str,
    output_root: str,
    status: str = "completed",
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    """Normalize a CleanUnifiedNO summary into the route schema."""

    args = dict(summary.get("args") or {})
    learned_test = dict(
        _nested(summary, ("learned_prediction_diagnostics", "test")) or {}
    )
    witness = dict(summary.get("markov_node_witness_diagnostics") or {})
    witness_test = dict(_nested(witness, ("splits", "test")) or {})
    local_law = dict(summary.get("markov_local_law_fno_diagnostics") or {})
    law_test = dict(_nested(local_law, ("splits", "test")) or {})
    leaf = dict(_first_present(law_test.get("leaf"), witness_test.get("leaf"), {}) or {})
    merge = dict(_first_present(law_test.get("merge"), witness_test.get("merge"), {}) or {})
    root = dict(_first_present(law_test.get("root"), witness_test.get("root"), {}) or {})
    boundary = dict(summary.get("boundary_supervision_ablation") or {})
    boundary_diag = dict(_nested(boundary, ("boundary_diagnostics", "test")) or {})
    return {
        "cell_id": cell_id,
        "status": status,
        "route": "pytorch",
        "objective": objective,
        "backend": "pytorch",
        "input_encoding": "token_ids",
        "summary_family": "CleanUnifiedNO",
        "leaf_tokens": args.get("leaf_tokens") or summary.get("leaf_tokens"),
        "doc_tokens": args.get("doc_tokens") or _nested(summary, ("generation", "doc_tokens")),
        "train_docs": args.get("train_docs"),
        "eval_docs": args.get("eval_docs"),
        "n_iter": None,
        "epochs": args.get("epochs"),
        "batch_size": args.get("batch_size"),
        "hidden_dim": None,
        "channels": args.get("channels"),
        "n_modes": args.get("g_n_modes"),
        "seed": args.get("seed"),
        "root_count_mae": _first_present(
            _count_mae(root),
            summary.get("test_root_mae"),
            learned_test.get("root_mae"),
        ),
        "theta_mae": root.get("theta_mae"),
        "theta_first_regime_accuracy": root.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": root.get("theta_last_regime_accuracy"),
        "eps_leaf": leaf.get("theta_mae"),
        "eps_merge": merge.get("theta_mae"),
        "eps_idemp": _first_present(root.get("eps_idemp_range"), root.get("eps_idemp_to_exact")),
        "contextual_mae": None,
        "pred_truth_corr": learned_test.get("pred_truth_corr"),
        "pred_std": learned_test.get("pred_std"),
        "leaf_first_acc": leaf.get("theta_first_regime_accuracy"),
        "leaf_last_acc": leaf.get("theta_last_regime_accuracy"),
        "merge_first_acc": merge.get("theta_first_regime_accuracy"),
        "merge_last_acc": merge.get("theta_last_regime_accuracy"),
        "full_exact_rate": _first_present(
            root.get("full_witness_exact_rate"),
            leaf.get("full_witness_exact_rate"),
        ),
        "boundary_f1": _first_present(boundary_diag.get("f1"), boundary_diag.get("accuracy")),
        "elapsed_sec": elapsed_sec,
        "output_root": output_root,
    }


def route_fieldnames(rows: list[Mapping[str, Any]]) -> list[str]:
    fields = [field for field in COMMON_METRIC_FIELDS if any(field in row for row in rows)]
    seen = set(fields)
    fields.extend(
        sorted({field for row in rows for field in row.keys() if field not in seen})
    )
    return fields


def write_route_outputs(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        with (output_root / "manifest.json").open("w") as fh:
            json.dump(manifest, fh, indent=2)
    summary = {
        "title": title,
        "n_rows": len(rows),
        "n_completed": sum(1 for row in rows if row.get("status") == "completed"),
        "rows": rows,
    }
    with (output_root / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    fields = route_fieldnames(rows)
    with (output_root / "grid_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean_number(row.get(key)) for key in fields})
    with (output_root / "grid_report.md").open("w") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"Rows: {len(rows)}\n\n")
        if not rows:
            fh.write("No rows.\n")
            return
        report_fields = [
            "cell_id",
            "route",
            "objective",
            "summary_family",
            "leaf_tokens",
            "root_count_mae",
            "theta_mae",
            "theta_first_regime_accuracy",
            "theta_last_regime_accuracy",
            "eps_leaf",
            "eps_merge",
            "pred_std",
        ]
        report_fields = [field for field in report_fields if field in fields]
        fh.write("| " + " | ".join(report_fields) + " |\n")
        fh.write("| " + " | ".join("---" for _ in report_fields) + " |\n")
        for row in rows:
            values = [str(_clean_number(row.get(field, ""))) for field in report_fields]
            fh.write("| " + " | ".join(values) + " |\n")
