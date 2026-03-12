#!/usr/bin/env python3
"""Summarize dense Markov local-support sweeps into flat rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List


def _safe_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _utility_recovery_fraction(observed: object, exact: object, undersupported: object) -> float:
    obs = _safe_float(observed)
    good = _safe_float(exact)
    bad = _safe_float(undersupported)
    if not all(math.isfinite(v) for v in [obs, good, bad]):
        return math.nan
    if bad <= good + 1e-12:
        return 1.0 if obs <= good + 1e-12 else 0.0
    raw = (bad - obs) / (bad - good)
    return float(min(1.0, max(0.0, raw)))


def _access_pattern(
    *,
    leaf_label_coverage: float,
    internal_label_coverage: float,
    root_query_rate: float,
) -> str:
    has_leaf = math.isfinite(leaf_label_coverage) and leaf_label_coverage > 1e-12
    has_internal = math.isfinite(internal_label_coverage) and internal_label_coverage > 1e-12
    has_root = math.isfinite(root_query_rate) and root_query_rate > 1e-12
    if has_root:
        if has_leaf and has_internal:
            return "root_plus_local_mixed"
        if has_internal:
            return "root_plus_merge"
        if has_leaf:
            return "root_plus_leaf"
        return "root_only"
    if has_leaf and has_internal:
        return "local_mixed"
    if has_internal:
        return "merge_only"
    if has_leaf:
        return "leaf_only"
    return "none"


def _iter_rows(input_root: Path) -> Iterable[dict]:
    for path in sorted(input_root.rglob("*.json")):
        if path.name.endswith(".summary.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        geometry = payload.get("training_geometry", {}) or {}

        mean_leaves = _safe_float(geometry.get("mean_leaves"))
        mean_internal_nodes = _safe_float(geometry.get("mean_internal_nodes"))
        mean_leaf_labels = _safe_float(geometry.get("mean_leaf_labels"))
        mean_internal_labels = _safe_float(geometry.get("mean_internal_labels"))
        total_local_nodes = mean_leaves + mean_internal_nodes
        if all(math.isfinite(v) for v in [mean_leaf_labels, mean_internal_labels, total_local_nodes]) and total_local_nodes > 0.0:
            local_oracle_coverage = (mean_leaf_labels + mean_internal_labels) / total_local_nodes
        else:
            local_oracle_coverage = math.nan
        leaf_label_coverage = (
            mean_leaf_labels / mean_leaves
            if all(math.isfinite(v) for v in [mean_leaf_labels, mean_leaves]) and mean_leaves > 0.0
            else math.nan
        )
        internal_label_coverage = (
            mean_internal_labels / mean_internal_nodes
            if all(math.isfinite(v) for v in [mean_internal_labels, mean_internal_nodes]) and mean_internal_nodes > 0.0
            else math.nan
        )
        train_docs = _safe_float(config.get("train_docs"))
        root_queries_total = _safe_float(geometry.get("root_queries_total"))
        root_query_rate = (
            root_queries_total / train_docs
            if all(math.isfinite(v) for v in [root_queries_total, train_docs]) and train_docs > 0.0
            else math.nan
        )
        info_signature = "|".join(
            [
                f"leafcov={leaf_label_coverage:.3f}" if math.isfinite(leaf_label_coverage) else "leafcov=na",
                f"intcov={internal_label_coverage:.3f}" if math.isfinite(internal_label_coverage) else "intcov=na",
                f"rootq={root_query_rate:.3f}" if math.isfinite(root_query_rate) else "rootq=na",
            ]
        )
        access_pattern = _access_pattern(
            leaf_label_coverage=leaf_label_coverage,
            internal_label_coverage=internal_label_coverage,
            root_query_rate=root_query_rate,
        )

        learned = metrics.get("learned", {}) or {}
        exact = metrics.get("exact", {}) or {}
        undersupported = metrics.get("undersupported", {}) or {}

        yield {
            "source_path": str(path),
            "fixed_leaf_tokens": config.get("fixed_leaf_tokens"),
            "train_docs": config.get("train_docs"),
            "test_docs": config.get("test_docs"),
            "model_family": config.get("model_family"),
            "audit_fraction": config.get("audit_fraction"),
            "leaf_query_rate": config.get("leaf_query_rate"),
            "include_root_query": config.get("include_root_query"),
            "seed": config.get("seed"),
            "mean_leaves": mean_leaves,
            "mean_internal_nodes": mean_internal_nodes,
            "mean_leaf_labels": mean_leaf_labels,
            "mean_internal_labels": mean_internal_labels,
            "mean_queries_per_doc": _safe_float(geometry.get("mean_queries_per_doc")),
            "total_queries_estimate": _safe_float(geometry.get("total_queries_estimate")),
            "root_queries_total": geometry.get("root_queries_total"),
            "leaf_label_coverage": leaf_label_coverage,
            "internal_label_coverage": internal_label_coverage,
            "root_query_rate": root_query_rate,
            "local_oracle_coverage": local_oracle_coverage,
            "local_undersupport": (1.0 - local_oracle_coverage) if math.isfinite(local_oracle_coverage) else math.nan,
            "info_signature": info_signature,
            "access_pattern": access_pattern,
            "learned_root_mae": _safe_float(learned.get("root_mae")),
            "learned_merge_mae": _safe_float(learned.get("merge_mae")),
            "learned_schedule_spread_mean": _safe_float(learned.get("schedule_spread_mean")),
            "exact_root_mae": _safe_float(exact.get("root_mae")),
            "exact_merge_mae": _safe_float(exact.get("merge_mae")),
            "undersupported_root_mae": _safe_float(undersupported.get("root_mae")),
            "undersupported_merge_mae": _safe_float(undersupported.get("merge_mae")),
            "root_utility_recovery": _utility_recovery_fraction(
                learned.get("root_mae"),
                exact.get("root_mae"),
                undersupported.get("root_mae"),
            ),
        }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Markov local-support grid outputs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = list(_iter_rows(args.input_root))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    _write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
