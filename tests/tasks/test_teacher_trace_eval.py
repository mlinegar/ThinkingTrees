from __future__ import annotations

from src.tasks.manifesto.teacher_trace_eval import (
    TeacherTraceEvalConfig,
    TeacherTracePrediction,
    build_eval_metrics,
    score_predictions,
    split_segments,
)
from src.tasks.manifesto.teacher_trace_generator import TeacherTraceRecord


def _record(example_id: str, split: str, source_id: str, source_rile_raw: float, expanded_text: str) -> TeacherTraceRecord:
    return TeacherTraceRecord(
        example_id=example_id,
        split=split,
        source_manifesto_id=source_id,
        source_party_abbrev="P",
        source_country_name="Country",
        source_year=2000,
        source_rile_raw=source_rile_raw,
        source_bin_name="center",
        source_text=f"source::{example_id}",
        expanded_text=expanded_text,
        expanded_score_raw=source_rile_raw,
        expanded_delta_raw=0.0,
        summary1=f"summary1::{example_id}",
        summary1_score_raw=source_rile_raw,
        summary1_delta_raw=0.0,
        summary2=f"summary2::{example_id}",
        summary2_score_raw=source_rile_raw,
        summary2_delta_raw=0.0,
        summary2_vs_summary1_delta_raw=0.0,
        same_side_summary1=True,
        same_side_summary2=True,
        trace_critical_points=[],
        trace_entities=[],
        trace_qualifiers=[],
        trace_invariants=[],
        trace_notes="",
        attempts_used=1,
    )


def test_score_predictions_computes_c1_c2_c3() -> None:
    pass_text = "A" * 500
    fail_text = "B" * 500

    records = [
        _record("doc_pass", "test", "m1", 25.0, pass_text),
        _record("doc_fail", "test", "m2", -20.0, fail_text),
    ]

    seg_a_pass, seg_b_pass = split_segments(pass_text)
    seg_a_fail, seg_b_fail = split_segments(fail_text)

    predictions = [
        TeacherTracePrediction(
            example_id="doc_pass",
            split="test",
            source_manifesto_id="m1",
            source_bin_name="center",
            source_rile_raw=25.0,
            summary1="s1_pass",
            summary2="s2_pass",
            segment_a=seg_a_pass,
            segment_b=seg_b_pass,
            summary_a="sa_pass",
            summary_b="sb_pass",
            merged_summary="sm_pass",
        ),
        TeacherTracePrediction(
            example_id="doc_fail",
            split="test",
            source_manifesto_id="m2",
            source_bin_name="center",
            source_rile_raw=-20.0,
            summary1="s1_fail",
            summary2="s2_fail",
            segment_a=seg_a_fail,
            segment_b=seg_b_fail,
            summary_a="sa_fail",
            summary_b="sb_fail",
            merged_summary="sm_fail",
        ),
    ]

    score_map = {
        "s1_pass": 24.0,
        "s2_pass": 23.5,
        "sm_pass": 25.5,
        "s1_fail": 5.0,
        "s2_fail": -30.0,
        "sm_fail": 0.0,
        seg_a_pass: 24.0,
        seg_b_pass: 26.0,
        seg_a_fail: -22.0,
        seg_b_fail: -18.0,
    }

    def score_fn(text: str) -> float:
        return score_map[text]

    results = score_predictions(
        records,
        predictions,
        score_fn=score_fn,
        config=TeacherTraceEvalConfig(),
        num_workers=2,
    )

    assert len(results) == 2
    by_id = {row.example_id: row for row in results}

    assert by_id["doc_pass"].c1_pass is True
    assert by_id["doc_pass"].c2_pass is True
    assert by_id["doc_pass"].c3_pass is True

    assert by_id["doc_fail"].c1_pass is False
    assert by_id["doc_fail"].c2_pass is False
    assert by_id["doc_fail"].c3_pass is False

    metrics, groups = build_eval_metrics(results)
    assert metrics["overall"]["n"] == 2
    assert metrics["overall"]["c1_pass_rate"] == 50.0
    assert metrics["overall"]["c2_pass_rate"] == 50.0
    assert metrics["overall"]["c3_pass_rate"] == 50.0
    assert groups["split"]["test"]["n"] == 2
