from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.tasks.manifesto.lawstress_generator import (
    LawStressRecord,
    normalize_rile,
    write_lawstress_records_jsonl,
)


def _make_record(example_id: str, split: str) -> LawStressRecord:
    return LawStressRecord(
        example_id=example_id,
        split=split,
        bin_name="center",
        law_target="c1_sufficiency",
        family="polarity_cancellation",
        difficulty="control",
        anchor_source="synthetic",
        text=f"DOC::{example_id}",
        segment_a=f"A::{example_id}",
        segment_b=f"B::{example_id}",
        policy_atoms=[],
        target_raw=10.0,
        y_raw=10.0,
        y_norm=normalize_rile(10.0),
        yA_raw=8.0,
        yB_raw=12.0,
        y_merge_expected_raw=10.0,
        teacher_score_doc=9.0,
        teacher_score_segment_a=7.0,
        teacher_score_segment_b=11.0,
        naive_summary=f"NAIVE::{example_id}",
        naive_score_raw=5.0,
        naive_drift_norm=0.02,
        reference_summary=f"REF::{example_id}",
        attempts_used=1,
    )


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def test_build_lawstress_split_views_writes_docs_and_pairs(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.build_lawstress_split_views")

    records = [
        _make_record("ex1", "train"),
        _make_record("ex2", "train"),
        _make_record("ex3", "train"),
        _make_record("ex4", "val"),
        _make_record("ex5", "test"),
        _make_record("ex6", "test"),
    ]
    records_path = tmp_path / "lawstress_records.jsonl"
    write_lawstress_records_jsonl(records_path, records)

    out_dir = tmp_path / "views"
    rc = cli.main(["--records", str(records_path), "--output-dir", str(out_dir)])
    assert rc == 0

    assert _jsonl_count(out_dir / "benchmark_docs_train.jsonl") == 3
    assert _jsonl_count(out_dir / "benchmark_docs_val.jsonl") == 1
    assert _jsonl_count(out_dir / "benchmark_docs_test.jsonl") == 2

    # Default includes hop1 + hop2 per record.
    assert _jsonl_count(out_dir / "summary_pairs_train.jsonl") == 6
    assert _jsonl_count(out_dir / "summary_pairs_val.jsonl") == 2
    assert _jsonl_count(out_dir / "summary_pairs_test.jsonl") == 4

    manifest = json.loads((out_dir / "split_ids.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["docs"]["train"] == 3
    assert manifest["counts"]["docs"]["val"] == 1
    assert manifest["counts"]["docs"]["test"] == 2

