#!/usr/bin/env python3
"""Build deterministic train/val/test split views from teacher-trace records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    build_benchmark_docs,
    build_summary_pair_rows,
    load_teacher_trace_records_jsonl,
    write_jsonl,
)


VALID_SPLITS = ("train", "val", "test")


def _normalize_split(value: str) -> str:
    rendered = str(value or "").strip().lower()
    if rendered == "validation":
        return "val"
    return rendered


def _assign_missing_splits(
    records: Sequence[TeacherTraceRecord],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[str]:
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total <= 0.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be positive")

    n = len(records)
    if n == 0:
        return []

    ratios = [
        float(train_ratio) / total,
        float(val_ratio) / total,
        float(test_ratio) / total,
    ]
    counts = [int(round(ratio * n)) for ratio in ratios]
    delta = n - sum(counts)
    order = sorted(range(3), key=lambda idx: ratios[idx], reverse=True)
    for idx in order:
        if delta == 0:
            break
        counts[idx] += 1 if delta > 0 else -1
        delta += -1 if delta > 0 else 1
    counts = [max(0, count) for count in counts]
    if sum(counts) != n:
        counts[0] += n - sum(counts)

    labels = (
        ["train"] * counts[0]
        + ["val"] * counts[1]
        + ["test"] * counts[2]
    )
    rng = random.Random(int(seed))
    rng.shuffle(labels)
    return labels


def _ensure_source_disjoint(records: Sequence[TeacherTraceRecord]) -> None:
    source_to_split: Dict[str, str] = {}
    conflicts: List[str] = []
    for row in records:
        source_id = str(row.source_manifesto_id or "")
        split = _normalize_split(row.split)
        if not source_id or split not in VALID_SPLITS:
            continue
        previous = source_to_split.get(source_id)
        if previous is None:
            source_to_split[source_id] = split
            continue
        if previous != split:
            conflicts.append(f"{source_id}: {previous} vs {split}")
    if conflicts:
        preview = ", ".join(conflicts[:5])
        raise ValueError(f"Source-manifesto split leakage detected: {preview}")


def _partition_records(
    records: Sequence[TeacherTraceRecord],
    *,
    assign_missing: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[TeacherTraceRecord]]:
    buckets: Dict[str, List[TeacherTraceRecord]] = {split: [] for split in VALID_SPLITS}

    missing_indices: List[int] = []
    normalized: List[str] = []
    for idx, row in enumerate(records):
        split = _normalize_split(row.split)
        normalized.append(split)
        if split not in VALID_SPLITS:
            missing_indices.append(idx)

    if missing_indices and not assign_missing:
        raise ValueError(
            "Records contain missing/invalid split labels. "
            "Use --assign-missing-splits to fill deterministically."
        )

    if missing_indices:
        fallback_labels = _assign_missing_splits(
            [records[idx] for idx in missing_indices],
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
        )
        for idx, split in zip(missing_indices, fallback_labels):
            normalized[idx] = split

    for row, split in zip(records, normalized):
        if split not in VALID_SPLITS:
            continue
        # Keep row object untouched; this utility only emits split views.
        buckets[split].append(row)

    return buckets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split views from teacher-trace records")
    parser.add_argument("--records", type=Path, required=True, help="Path to teacher_trace_records.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--assign-missing-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enforce-source-disjoint", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    records = load_teacher_trace_records_jsonl(args.records)
    if not records:
        raise ValueError(f"No teacher-trace records loaded from {args.records}")

    buckets = _partition_records(
        records,
        assign_missing=bool(args.assign_missing_splits),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    if bool(args.enforce_source_disjoint):
        _ensure_source_disjoint(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ids: Dict[str, List[str]] = {}
    split_source_ids: Dict[str, List[str]] = {}
    pair_counts: Dict[str, int] = {}

    for split in VALID_SPLITS:
        rows = buckets.get(split) or []
        docs = build_benchmark_docs(rows)
        pairs = build_summary_pair_rows(rows)

        docs_path = output_dir / f"benchmark_docs_{split}.jsonl"
        pairs_path = output_dir / f"summary_pairs_{split}.jsonl"

        write_jsonl(docs_path, docs)
        write_jsonl(pairs_path, pairs)

        split_ids[split] = [str(row.example_id) for row in rows]
        split_source_ids[split] = sorted({str(row.source_manifesto_id) for row in rows})
        pair_counts[split] = len(pairs)

    split_manifest = {
        "records_path": str(args.records),
        "counts": {
            "docs": {split: len(split_ids.get(split) or []) for split in VALID_SPLITS},
            "summary_pairs": pair_counts,
        },
        "split_ids": split_ids,
        "source_manifesto_ids": split_source_ids,
        "artifacts": {
            split: {
                "benchmark_docs": str(output_dir / f"benchmark_docs_{split}.jsonl"),
                "summary_pairs": str(output_dir / f"summary_pairs_{split}.jsonl"),
            }
            for split in VALID_SPLITS
        },
    }

    (output_dir / "split_ids.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
