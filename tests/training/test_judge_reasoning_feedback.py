"""Tests for judge-optimization reasoning feedback plumbing."""

from types import SimpleNamespace

import pytest


def test_create_judge_trainset_includes_cleaned_oracle_and_judge_reasoning():
    from src.training.judge_optimization import create_judge_trainset
    from src.training.preference.types import PreferencePair

    pair = PreferencePair(
        pair_id="p1",
        source_example_id="ex1",
        original_text="Original text",
        rubric="Rubric",
        reference_score=0.0,
        summary_a="Summary A",
        summary_b="Summary B",
        preferred="A",
        reasoning="<think>hidden</think> A is more complete.",
        confidence=0.9,
        oracle_error_a=0.1,
        oracle_error_b=0.5,
    )

    examples, skipped = create_judge_trainset([pair], tie_margin=0.05, use_oracle_as_ground_truth=True)
    assert skipped.total == 0
    assert len(examples) == 1
    example = examples[0]

    assert example.ground_truth_preference == "A"
    assert "Oracle label: A" in example.ground_truth_reasoning
    assert "Judge said A" in example.ground_truth_reasoning
    assert "hidden" not in example.ground_truth_reasoning  # think block stripped


def test_judge_accuracy_metric_returns_scorewithfeedback_with_rationales():
    gepa_utils = pytest.importorskip("dspy.teleprompt.gepa.gepa_utils")
    ScoreWithFeedback = getattr(gepa_utils, "ScoreWithFeedback", None)
    if ScoreWithFeedback is None:
        pytest.skip("ScoreWithFeedback not available in this DSPy version")

    from src.training.judge_optimization import create_judge_trainset, judge_accuracy_metric
    from src.training.preference.types import PreferencePair

    pair = PreferencePair(
        pair_id="p2",
        source_example_id="ex2",
        original_text="Original text",
        rubric="Rubric",
        reference_score=0.0,
        summary_a="Summary A",
        summary_b="Summary B",
        preferred="A",
        reasoning="A is better.",
        confidence=0.9,
        oracle_error_a=0.1,
        oracle_error_b=0.5,
    )

    examples, _ = create_judge_trainset([pair], tie_margin=0.05, use_oracle_as_ground_truth=True)
    example = examples[0]

    pred = SimpleNamespace(preference="B", reasoning="B feels more complete.")
    result = judge_accuracy_metric(example, pred)
    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.0
    assert "Your rationale:" in result.feedback
    assert "Gold rationale:" in result.feedback

