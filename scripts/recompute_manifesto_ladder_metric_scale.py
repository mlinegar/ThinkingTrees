#!/usr/bin/env python3
"""Recompute Manifesto ladder external metrics on an explicit expert scale.

This is a bookkeeping pass for saved prediction records. It does not rerun
any model calls and it does not modify the source ladder artifacts. Instead,
it writes a mirrored companion tree under
``RUN_ROOT/scale_corrected/<target-scale>/`` with corrected external metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.ladder_reporting import (  # noqa: E402
    summarize_ladder_grid,
    write_corrected_scale_markdown_summary,
)
from src.experiments.metrics import pearson as _pearson
from src.experiments.script_io import (  # noqa: E402
    now_iso as _utc_now,
    read_json_object as _read_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
)
from src.experiments.script_parse import safe_float as _safe_float, safe_int as _safe_int
from src.tasks.manifesto.result_rows import row_manifesto_id as _manifesto_id
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_CHOICES,
    EXPERT_SCALE_NORMALIZED_1_7,
    EXPERT_SCALE_RAW,
    expert_scale_metadata,
    resolve_benoit_expert_target,
    scorer_1_7_to_expert_target,
)


METRIC_FIELDS = (
    "internal_f_pearson",
    "external_expert_pearson",
    "f_star_gap",
    "internal_f_mae",
    "external_expert_mae",
    "mean_prediction",
    "mean_teacher",
    "mean_expert",
    "internal_f_mae_1_7",
    "external_expert_mae_1_7",
    "mean_prediction_1_7",
    "mean_teacher_1_7",
    "mean_expert_1_7",
)

EXTERNAL_FIELDS = (
    "external_expert_pearson",
    "external_expert_mae",
    "external_expert_mae_1_7",
    "mean_prediction",
    "mean_prediction_1_7",
    "mean_expert",
    "mean_expert_1_7",
)



def _prediction_value(
    row: Mapping[str, Any],
    *,
    dimension: str,
    target_expert_scale: str,
) -> Optional[float]:
    row_scale = str(row.get("metrics_scale") or row.get("prediction_scale") or "")
    if target_expert_scale == EXPERT_SCALE_RAW:
        for key in ("prediction_native", "pred_score_native", "prediction"):
            if key in row and (key != "prediction" or row_scale in {"", EXPERT_SCALE_RAW}):
                value = _safe_float(row.get(key))
                if value is not None:
                    return value
        for key in ("prediction_1_7", "pred_score_1_7", "score_1_7"):
            if key in row:
                value = scorer_1_7_to_expert_target(
                    row.get(key),
                    dimension=dimension,
                    scale=target_expert_scale,
                )
                if value is not None:
                    return value
        return None

    for key in ("prediction_1_7", "pred_score_1_7", "score_1_7"):
        if key in row:
            value = _safe_float(row.get(key))
            if value is not None:
                return value
    if row_scale in {"", target_expert_scale}:
        value = _safe_float(row.get("prediction"))
        if value is not None:
            return value
    return None



def _history_paths(run_root: Path) -> list[Path]:
    base = run_root / "ladder" if (run_root / "ladder").exists() else run_root
    paths = sorted(base.glob("*/leaf*/iteration_history.json"))
    return [path for path in paths if path.is_file()]


def _source_rows_by_id(source_results: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(source_results):
        doc_id = _manifesto_id(row)
        if doc_id:
            out[doc_id] = row
    return out


def _records_for_split(records: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    if split == "all":
        return list(records)
    return [record for record in records if str(record.get("split") or "") == split]


def _compute_external_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    dimension: str,
    target_expert_scale: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    preds: list[float] = []
    experts: list[float] = []
    counts = {
        "records_seen": len(records),
        "pairs_used": 0,
        "missing_doc_id": 0,
        "missing_source_row": 0,
        "missing_prediction": 0,
        "missing_expert": 0,
    }
    for record in records:
        doc_id = _manifesto_id(record)
        if not doc_id:
            counts["missing_doc_id"] += 1
            continue
        source_row = rows_by_id.get(doc_id)
        if source_row is None:
            counts["missing_source_row"] += 1
            continue
        pred = _prediction_value(
            record,
            dimension=dimension,
            target_expert_scale=target_expert_scale,
        )
        if pred is None:
            counts["missing_prediction"] += 1
            continue
        expert = resolve_benoit_expert_target(
            source_row,
            dimension=dimension,
            scale=target_expert_scale,
        )
        if expert is None:
            counts["missing_expert"] += 1
            continue
        preds.append(float(pred))
        experts.append(float(expert))

    counts["pairs_used"] = len(preds)
    if not preds:
        return {
            "n": 0,
            "external_expert_pearson": None,
            "external_expert_mae": None,
            "external_expert_mae_1_7": None,
            "mean_prediction": None,
            "mean_prediction_1_7": None,
            "mean_expert": None,
            "mean_expert_1_7": None,
        }, counts

    mae = sum(abs(pred - expert) for pred, expert in zip(preds, experts)) / len(preds)
    values_are_one_to_seven = target_expert_scale == EXPERT_SCALE_NORMALIZED_1_7
    return {
        "n": len(preds),
        "external_expert_pearson": _pearson(preds, experts),
        "external_expert_mae": mae,
        "external_expert_mae_1_7": mae if values_are_one_to_seven else None,
        "mean_prediction": sum(preds) / len(preds),
        "mean_prediction_1_7": sum(preds) / len(preds) if values_are_one_to_seven else None,
        "mean_expert": sum(experts) / len(experts),
        "mean_expert_1_7": sum(experts) / len(experts) if values_are_one_to_seven else None,
    }, counts


def _merge_metric_block(
    original: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    target_expert_scale: str,
) -> dict[str, Any]:
    out = dict(original)
    out["n"] = recomputed.get("n", 0)
    for field in EXTERNAL_FIELDS:
        out[field] = recomputed.get(field)
    internal = _safe_float(out.get("internal_f_pearson"))
    external = _safe_float(out.get("external_expert_pearson"))
    out["f_star_gap"] = internal - external if internal is not None and external is not None else None
    out["metrics_scale"] = target_expert_scale
    out["external_expert_scale"] = target_expert_scale
    out["prediction_scale"] = target_expert_scale
    return out


def _add_counts(total: dict[str, int], counts: Mapping[str, int]) -> None:
    for key, value in counts.items():
        total[key] = int(total.get(key, 0)) + int(value or 0)


def _prediction_records_path(history_path: Path, iteration: int) -> Path:
    return history_path.parent / "prediction_records" / f"iter_{iteration:02d}_post_eval.jsonl"


def _correct_history(
    history_path: Path,
    *,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    dimension: str,
    source_expert_scale: str,
    target_expert_scale: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    original = _read_json(history_path)
    corrected = copy.deepcopy(original)
    aggregate = {
        "histories": 1,
        "iterations": 0,
        "prediction_record_files": 0,
        "missing_prediction_record_files": 0,
        "records_seen": 0,
        "pairs_used": 0,
        "missing_doc_id": 0,
        "missing_source_row": 0,
        "missing_prediction": 0,
        "missing_expert": 0,
    }
    scale_info = {
        "source_iteration_history": str(history_path),
        "source_expert_scale": source_expert_scale,
        "target_expert_scale": target_expert_scale,
        **expert_scale_metadata(dimension=dimension, scale=target_expert_scale),
    }
    corrected["metrics_scale"] = target_expert_scale
    corrected["scale_correction"] = scale_info

    for iteration in corrected.get("iterations") or []:
        if not isinstance(iteration, dict):
            continue
        aggregate["iterations"] += 1
        iteration_index = _safe_int(iteration.get("iteration"))
        if iteration_index is None:
            continue
        records_path = _prediction_records_path(history_path, iteration_index)
        iteration["metrics_scale"] = target_expert_scale
        iteration_scale_info = dict(scale_info)
        iteration_scale_info["source_prediction_records"] = str(records_path)
        if not records_path.exists():
            aggregate["missing_prediction_record_files"] += 1
            iteration_scale_info["status"] = "missing_prediction_records"
            iteration["scale_correction"] = iteration_scale_info
            continue

        aggregate["prediction_record_files"] += 1
        records = _read_jsonl(records_path)
        split_metrics = dict(iteration.get("split_metrics") or {})
        split_names = sorted(set(split_metrics) | {"all"} | {str(r.get("split")) for r in records if r.get("split")})
        split_counts: dict[str, dict[str, int]] = {}
        corrected_split_metrics: dict[str, dict[str, Any]] = {}
        for split in split_names:
            split_records = _records_for_split(records, split)
            recomputed, counts = _compute_external_metrics(
                split_records,
                rows_by_id=rows_by_id,
                dimension=dimension,
                target_expert_scale=target_expert_scale,
            )
            if split == "all":
                _add_counts(aggregate, counts)
            split_counts[split] = dict(counts)
            original_metrics = split_metrics.get(split) if isinstance(split_metrics.get(split), dict) else {}
            corrected_split_metrics[split] = _merge_metric_block(
                original_metrics,
                recomputed,
                target_expert_scale=target_expert_scale,
            )
        iteration["split_metrics"] = corrected_split_metrics
        iteration_scale_info["status"] = "corrected"
        iteration_scale_info["splits"] = split_counts
        iteration["scale_correction"] = iteration_scale_info

    corrected["scale_correction"] = {**scale_info, "counts": aggregate}
    return corrected, aggregate



def recompute_run(
    *,
    run_root: Path,
    source_results: Path,
    output_root: Optional[Path],
    dimension: str,
    source_expert_scale: str = EXPERT_SCALE_RAW,
    target_expert_scale: str = EXPERT_SCALE_RAW,
    eval_split: str = "test",
) -> dict[str, Any]:
    run_root = Path(run_root)
    source_results = Path(source_results)
    output_root = Path(output_root) if output_root is not None else run_root / "scale_corrected" / target_expert_scale
    rows_by_id = _source_rows_by_id(source_results)
    histories: list[dict[str, Any]] = []
    aggregate = {
        "histories": 0,
        "iterations": 0,
        "prediction_record_files": 0,
        "missing_prediction_record_files": 0,
        "records_seen": 0,
        "pairs_used": 0,
        "missing_doc_id": 0,
        "missing_source_row": 0,
        "missing_prediction": 0,
        "missing_expert": 0,
    }
    history_paths = _history_paths(run_root)
    if not history_paths:
        raise FileNotFoundError(f"no iteration_history.json files found under {run_root}")

    for history_path in history_paths:
        corrected, counts = _correct_history(
            history_path,
            rows_by_id=rows_by_id,
            dimension=dimension,
            source_expert_scale=source_expert_scale,
            target_expert_scale=target_expert_scale,
        )
        _add_counts(aggregate, counts)
        histories.append(corrected)
        rel = history_path.relative_to(run_root)
        _write_json(output_root / rel, corrected)

    rows = summarize_ladder_grid(
        histories,
        eval_split=eval_split,
        row_fields=(
            "family",
            "axis_kind",
            "axis_value",
            "leaf_count",
            "leaf_size_tokens",
            "metrics_scale",
        ),
        metric_fields=METRIC_FIELDS,
    )
    created_at = _utc_now()
    scale_info = {
        "created_at": created_at,
        "source_run_root": str(run_root),
        "source_results": str(source_results),
        "dimension": str(dimension),
        "source_expert_scale": str(source_expert_scale),
        "target_expert_scale": str(target_expert_scale),
        "metrics_scale": str(target_expert_scale),
        "eval_split": str(eval_split),
        "scale_correction": {
            "source_expert_scale": str(source_expert_scale),
            "target_expert_scale": str(target_expert_scale),
            "metrics_scale": str(target_expert_scale),
            **expert_scale_metadata(dimension=dimension, scale=target_expert_scale),
            "prediction_transform": (
                "scorer_1_7_to_raw_benoit"
                if target_expert_scale == EXPERT_SCALE_RAW
                else "identity_or_native_prediction"
            ),
            "counts": aggregate,
        },
        "rows": rows,
    }
    _write_json(output_root / "grid_summary.json", scale_info)
    _write_json(output_root / "ladder" / "grid_summary.json", scale_info)
    write_corrected_scale_markdown_summary(rows, output_root / "grid_summary.md", eval_split=eval_split)
    write_corrected_scale_markdown_summary(rows, output_root / "ladder" / "grid_summary.md", eval_split=eval_split)
    manifest = {
        "created_at": created_at,
        "mode": "manifesto_ladder_metric_scale_recompute",
        "run_root": str(run_root),
        "source_results": str(source_results),
        "output_root": str(output_root),
        "dimension": str(dimension),
        "source_expert_scale": str(source_expert_scale),
        "target_expert_scale": str(target_expert_scale),
        "eval_split": str(eval_split),
        "counts": aggregate,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--source-expert-scale", choices=EXPERT_SCALE_CHOICES, default=EXPERT_SCALE_RAW)
    parser.add_argument(
        "--target-expert-scale",
        choices=EXPERT_SCALE_CHOICES,
        default=EXPERT_SCALE_RAW,
    )
    parser.add_argument("--eval-split", default="test")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = recompute_run(
        run_root=Path(args.run_root),
        source_results=Path(args.source_results),
        output_root=args.output_root,
        dimension=str(args.dimension),
        source_expert_scale=str(args.source_expert_scale),
        target_expert_scale=str(args.target_expert_scale),
        eval_split=str(args.eval_split),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
