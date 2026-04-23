from __future__ import annotations

from types import SimpleNamespace

import pytest


def _score_with_feedback_type():
    gepa_utils = pytest.importorskip("dspy.teleprompt.gepa.gepa_utils")
    score_with_feedback = getattr(gepa_utils, "ScoreWithFeedback", None)
    if score_with_feedback is None:
        pytest.skip("ScoreWithFeedback not available in this DSPy version")
    return score_with_feedback


def test_rich_gepa_feedback_handles_parse_failure():
    ScoreWithFeedback = _score_with_feedback_type()
    from scripts.phase1_optimize_scorer import _make_gepa_metric
    from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension

    spec = BENOIT_DIMENSIONS[PolicyDimension.ECONOMIC]
    metric = _make_gepa_metric(spec, mode="mae", feedback_mode="rich")
    result = metric(
        SimpleNamespace(expert_mean=5.0),
        {"score": "NA", "reasoning": "No direct evidence."},
    )

    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.0
    assert "failed to parse" in result.feedback
    assert "Target score: 5.000" in result.feedback
    assert spec.anchor_low in result.feedback
    assert spec.anchor_high in result.feedback


def test_rich_gepa_feedback_reports_mae_direction_and_reasoning():
    ScoreWithFeedback = _score_with_feedback_type()
    from scripts.phase1_optimize_scorer import _make_gepa_metric
    from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension

    spec = BENOIT_DIMENSIONS[PolicyDimension.ENVIRONMENT]
    metric = _make_gepa_metric(spec, mode="mae", feedback_mode="rich")
    result = metric(
        SimpleNamespace(expert_mean=6.0),
        SimpleNamespace(score="3", reasoning="The text prioritizes growth."),
    )

    assert isinstance(result, ScoreWithFeedback)
    assert result.score == pytest.approx(0.5)
    assert "Predicted 3.000 vs target 6.000" in result.feedback
    assert "score higher" in result.feedback
    assert "The text prioritizes growth." in result.feedback


def test_rich_gepa_feedback_reports_rank_side_mismatch():
    ScoreWithFeedback = _score_with_feedback_type()
    from scripts.phase1_optimize_scorer import _make_gepa_metric
    from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension

    spec = BENOIT_DIMENSIONS[PolicyDimension.SOCIAL]
    metric = _make_gepa_metric(
        spec,
        mode="rank",
        label_center=4.0,
        feedback_mode="rich",
    )
    result = metric(
        SimpleNamespace(expert_mean=5.0),
        SimpleNamespace(score="3.0", reasoning="Mixed evidence."),
    )

    assert isinstance(result, ScoreWithFeedback)
    assert result.score == pytest.approx(0.15 * (1.0 - 2.0 / 6.0))
    assert "Rank-side mismatch" in result.feedback
    assert "center 4.000" in result.feedback


def test_scalar_gepa_feedback_mode_returns_float():
    from scripts.phase1_optimize_scorer import _make_gepa_metric
    from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension

    spec = BENOIT_DIMENSIONS[PolicyDimension.ECONOMIC]
    metric = _make_gepa_metric(spec, mode="mae", feedback_mode="scalar")
    result = metric(
        SimpleNamespace(expert_mean=4.0),
        SimpleNamespace(score="5.0", reasoning=""),
    )

    assert isinstance(result, float)
    assert result == pytest.approx(1.0 - 1.0 / 6.0)
