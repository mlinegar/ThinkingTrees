from __future__ import annotations

from src.tasks.manifesto.lawstress_eval import (
    LawStressEvalConfig,
    LawStressPrediction,
    build_eval_metrics,
    compute_metric_row,
    score_and_judge_predictions,
    strict_same_side,
)
from src.tasks.manifesto.lawstress_generator import LawStressRecord


def _make_record(example_id: str, *, split: str, difficulty: str, y_raw: float, y_merge_expected_raw: float) -> LawStressRecord:
    return LawStressRecord(
        example_id=example_id,
        split=split,
        bin_name="center_right",
        law_target="c3_merge",
        family="merge_order_asymmetry",
        difficulty=difficulty,
        anchor_source="synthetic",
        text=f"text::{example_id}",
        segment_a=f"segA::{example_id}",
        segment_b=f"segB::{example_id}",
        policy_atoms=[],
        target_raw=y_raw,
        y_raw=y_raw,
        y_norm=(y_raw + 100.0) / 200.0,
        yA_raw=y_raw - 5.0,
        yB_raw=y_raw + 5.0,
        y_merge_expected_raw=y_merge_expected_raw,
        teacher_score_doc=y_raw,
        teacher_score_segment_a=y_raw - 5.0,
        teacher_score_segment_b=y_raw + 5.0,
        naive_summary="naive",
        naive_score_raw=y_raw,
        naive_drift_norm=0.02,
        reference_summary=f"ref::{example_id}",
        attempts_used=1,
    )


def test_strict_same_side_is_strict_about_neutral_prediction() -> None:
    assert strict_same_side(0.6, 0.55, neutral_norm=0.5)
    assert not strict_same_side(0.5, 0.55, neutral_norm=0.5)
    assert not strict_same_side(0.4, 0.6, neutral_norm=0.5)


def test_score_and_judge_predictions_computes_c1_c2_c3_and_genrm() -> None:
    records = [
        _make_record("doc_pass", split="train", difficulty="hard", y_raw=20.0, y_merge_expected_raw=18.0),
        _make_record("doc_fail", split="test", difficulty="control", y_raw=-20.0, y_merge_expected_raw=-15.0),
    ]
    predictions = [
        LawStressPrediction(
            example_id="doc_pass",
            split="train",
            difficulty="hard",
            law_target="c3_merge",
            family="merge_order_asymmetry",
            bin_name="center_right",
            summary1="s1_pass",
            summary2="s2_pass",
            summary_a="sa_pass",
            summary_b="sb_pass",
            merged_summary="sm_pass",
            reference_summary="ref::doc_pass",
        ),
        LawStressPrediction(
            example_id="doc_fail",
            split="test",
            difficulty="control",
            law_target="c3_merge",
            family="merge_order_asymmetry",
            bin_name="center_right",
            summary1="s1_fail",
            summary2="s2_fail",
            summary_a="sa_fail",
            summary_b="sb_fail",
            merged_summary="sm_fail",
            reference_summary="ref::doc_fail",
        ),
    ]

    score_map = {
        "s1_pass": 22.0,
        "s2_pass": 21.0,
        "sm_pass": 17.0,
        "s1_fail": 5.0,
        "s2_fail": 0.0,
        "sm_fail": 25.0,
    }

    def score_fn(text: str) -> float:
        return score_map[text]

    def judge_fn(context: str, original_text: str, summary_a: str, summary_b: str, law_type: str):
        if "doc_pass" in summary_b:
            return {"preferred": "A", "confidence": 0.9}
        return {"preferred": "B", "confidence": 0.7}

    config = LawStressEvalConfig(c1_threshold_norm=0.10, c2_threshold_norm=0.06, c3_threshold_norm=0.08)
    results = score_and_judge_predictions(
        records,
        predictions,
        score_fn=score_fn,
        judge_fn=judge_fn,
        config=config,
        num_workers=2,
    )

    assert len(results) == 2

    by_id = {row.example_id: row for row in results}
    assert by_id["doc_pass"].c1_pass
    assert by_id["doc_pass"].c2_pass
    assert by_id["doc_pass"].c3_pass
    assert by_id["doc_pass"].genrm_tie_or_win is True

    assert by_id["doc_fail"].c1_pass is False
    assert by_id["doc_fail"].c2_pass is False
    assert by_id["doc_fail"].c3_pass is False
    assert by_id["doc_fail"].genrm_tie_or_win is False


def test_metric_and_success_bundle_includes_absolute_and_relative() -> None:
    records = [
        _make_record("doc1", split="train", difficulty="hard", y_raw=30.0, y_merge_expected_raw=28.0),
        _make_record("doc2", split="train", difficulty="control", y_raw=-30.0, y_merge_expected_raw=-28.0),
    ]
    predictions = [
        LawStressPrediction(
            example_id="doc1",
            split="train",
            difficulty="hard",
            law_target="c1_sufficiency",
            family="polarity_cancellation",
            bin_name="right",
            summary1="s1_doc1",
            summary2="s2_doc1",
            summary_a="sa_doc1",
            summary_b="sb_doc1",
            merged_summary="sm_doc1",
            reference_summary="ref::doc1",
        ),
        LawStressPrediction(
            example_id="doc2",
            split="train",
            difficulty="control",
            law_target="c1_sufficiency",
            family="polarity_cancellation",
            bin_name="left",
            summary1="s1_doc2",
            summary2="s2_doc2",
            summary_a="sa_doc2",
            summary_b="sb_doc2",
            merged_summary="sm_doc2",
            reference_summary="ref::doc2",
        ),
    ]

    score_map = {
        "s1_doc1": 30.0,
        "s2_doc1": 30.0,
        "sm_doc1": 28.0,
        "s1_doc2": -30.0,
        "s2_doc2": -30.0,
        "sm_doc2": -28.0,
    }

    def score_fn(text: str) -> float:
        return score_map[text]

    results = score_and_judge_predictions(
        records,
        predictions,
        score_fn=score_fn,
        judge_fn=None,
        config=LawStressEvalConfig(),
        num_workers=2,
    )

    overall = compute_metric_row(results)
    assert overall["mae"] == 0.0
    assert overall["c1_pass_rate"] == 100.0
    assert overall["genrm_tie_or_win_rate"] is None

    baseline = {
        "mae": 0.04,
        "same_side_of_neutral_pct": 40.0,
        "c1_pass_rate": 50.0,
        "c2_pass_rate": 50.0,
        "c3_pass_rate": 50.0,
    }

    metrics, groups = build_eval_metrics(
        results,
        config=LawStressEvalConfig(
            abs_hard_same_side_min=50.0,
            abs_hard_c1_min=50.0,
            abs_hard_c2_min=50.0,
            abs_hard_c3_min=50.0,
            abs_control_same_side_min=50.0,
            abs_control_c1_min=50.0,
            abs_control_c2_min=50.0,
            abs_control_c3_min=50.0,
        ),
        baseline_overall=baseline,
    )

    assert metrics["success"]["absolute"]["pass"] is True
    assert metrics["success"]["relative"]["enabled"] is True
    assert metrics["success"]["overall_pass"] is True
    assert groups["difficulty"]["hard"]["n"] == 1
    assert groups["difficulty"]["control"]["n"] == 1
