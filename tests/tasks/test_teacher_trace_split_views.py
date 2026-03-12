from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    write_teacher_trace_records_jsonl,
)


def _make_record(example_id: str, split: str, source_manifesto_id: str) -> TeacherTraceRecord:
    return TeacherTraceRecord(
        example_id=example_id,
        split=split,
        source_manifesto_id=source_manifesto_id,
        source_party_abbrev="P",
        source_country_name="Country",
        source_year=2000,
        source_rile_raw=10.0,
        source_bin_name="center_right",
        source_text=f"source::{example_id}",
        expanded_text=f"expanded::{example_id}",
        expanded_score_raw=12.0,
        expanded_delta_raw=2.0,
        summary1=f"summary1::{example_id}",
        summary1_score_raw=11.0,
        summary1_delta_raw=1.0,
        summary2=f"summary2::{example_id}",
        summary2_score_raw=10.5,
        summary2_delta_raw=0.5,
        summary2_vs_summary1_delta_raw=-0.5,
        same_side_summary1=True,
        same_side_summary2=True,
        trace_critical_points=["cp"],
        trace_entities=["entity"],
        trace_qualifiers=["qual"],
        trace_invariants=["inv"],
        trace_notes="ok",
        attempts_used=1,
    )


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def test_split_views_partitions_records_and_pairs(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.build_teacher_trace_split_views")

    records = [
        _make_record("ex1", "train", "s1"),
        _make_record("ex2", "train", "s2"),
        _make_record("ex3", "train", "s3"),
        _make_record("ex4", "val", "s4"),
        _make_record("ex5", "test", "s5"),
        _make_record("ex6", "test", "s6"),
    ]

    records_path = tmp_path / "teacher_trace_records.jsonl"
    write_teacher_trace_records_jsonl(records_path, records)

    output_dir = tmp_path / "split_views"
    rc = cli.main([
        "--records",
        str(records_path),
        "--output-dir",
        str(output_dir),
    ])
    assert rc == 0

    assert _jsonl_count(output_dir / "benchmark_docs_train.jsonl") == 3
    assert _jsonl_count(output_dir / "benchmark_docs_val.jsonl") == 1
    assert _jsonl_count(output_dir / "benchmark_docs_test.jsonl") == 2

    assert _jsonl_count(output_dir / "summary_pairs_train.jsonl") == 6
    assert _jsonl_count(output_dir / "summary_pairs_val.jsonl") == 2
    assert _jsonl_count(output_dir / "summary_pairs_test.jsonl") == 4

    split_ids = json.loads((output_dir / "split_ids.json").read_text(encoding="utf-8"))
    assert split_ids["counts"]["docs"]["train"] == 3
    assert split_ids["counts"]["docs"]["val"] == 1
    assert split_ids["counts"]["docs"]["test"] == 2


def test_split_views_detects_source_leakage(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.build_teacher_trace_split_views")

    records = [
        _make_record("ex1", "train", "shared_source"),
        _make_record("ex2", "test", "shared_source"),
    ]

    records_path = tmp_path / "teacher_trace_records.jsonl"
    write_teacher_trace_records_jsonl(records_path, records)

    with pytest.raises(ValueError):
        cli.main([
            "--records",
            str(records_path),
            "--output-dir",
            str(tmp_path / "split_views"),
            "--enforce-source-disjoint",
        ])
