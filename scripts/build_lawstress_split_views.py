#!/usr/bin/env python3
"""Build deterministic train/val/test split views from LawStress records.

Outputs mirror the teacher-trace split-view builder so downstream SFT tooling
can consume either source consistently:

- benchmark_docs_{split}.jsonl
- summary_pairs_{split}.jsonl
- split_ids.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.lawstress_generator import (
    LawStressRecord,
    load_lawstress_records_jsonl,
    write_jsonl,
)


VALID_SPLITS = ("train", "val", "test")


def _normalize_split(value: str) -> str:
    rendered = str(value or "").strip().lower()
    if rendered == "validation":
        return "val"
    return rendered


def _pair_rows_for_record(
    record: LawStressRecord,
    *,
    include_idempotence_pairs: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reference = str(record.reference_summary or "").strip()
    if not reference:
        return rows

    source_rile_raw = float(record.teacher_score_doc)

    rows.append(
        {
            "id": f"{record.example_id}_hop1",
            "example_id": record.example_id,
            "split": record.split,
            "hop": 1,
            "input_text": record.text,
            "target_summary": reference,
            "source_rile_raw": source_rile_raw,
            "target_score_raw": source_rile_raw,
        }
    )
    if include_idempotence_pairs:
        rows.append(
            {
                "id": f"{record.example_id}_hop2",
                "example_id": record.example_id,
                "split": record.split,
                "hop": 2,
                "input_text": reference,
                "target_summary": reference,
                "source_rile_raw": source_rile_raw,
                "target_score_raw": source_rile_raw,
            }
        )
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split views from LawStress records")
    parser.add_argument("--records", type=Path, required=True, help="Path to lawstress_records.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-idempotence-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include trivial hop-2 identity pairs on reference summaries (stability hint).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    records = load_lawstress_records_jsonl(args.records)
    if not records:
        raise ValueError(f"No LawStress records loaded from {args.records}")

    buckets: Dict[str, List[LawStressRecord]] = {split: [] for split in VALID_SPLITS}
    for record in records:
        split = _normalize_split(record.split)
        if split not in VALID_SPLITS:
            continue
        buckets[split].append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ids: Dict[str, List[str]] = {}
    pair_counts: Dict[str, int] = {}

    for split in VALID_SPLITS:
        rows = sorted(buckets.get(split) or [], key=lambda rec: str(rec.example_id))
        docs = [rec.to_benchmark_doc() for rec in rows]
        pairs: List[Dict[str, Any]] = []
        for rec in rows:
            pairs.extend(
                _pair_rows_for_record(
                    rec,
                    include_idempotence_pairs=bool(args.include_idempotence_pairs),
                )
            )

        docs_path = output_dir / f"benchmark_docs_{split}.jsonl"
        pairs_path = output_dir / f"summary_pairs_{split}.jsonl"
        write_jsonl(docs_path, docs)
        write_jsonl(pairs_path, pairs)

        split_ids[split] = [str(rec.example_id) for rec in rows]
        pair_counts[split] = len(pairs)

    split_manifest = {
        "records_path": str(args.records),
        "counts": {
            "docs": {split: len(split_ids.get(split) or []) for split in VALID_SPLITS},
            "summary_pairs": pair_counts,
        },
        "split_ids": split_ids,
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

