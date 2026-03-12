"""Tests for GEPA metric feedback wrapping."""

import pytest


def test_gepa_wrap_metric_converts_feedback_dict_to_scorewithfeedback():
    gepa_utils = pytest.importorskip("dspy.teleprompt.gepa.gepa_utils")
    ScoreWithFeedback = getattr(gepa_utils, "ScoreWithFeedback", None)
    if ScoreWithFeedback is None:
        pytest.skip("ScoreWithFeedback not available in this DSPy version")

    from src.training.optimization.gepa import GEPAOptimizer

    opt = GEPAOptimizer()

    def metric(*_args, **_kwargs):
        return {"score": 0.25, "feedback": "Non-numeric output"}

    wrapped = opt._wrap_metric_gepa(metric)
    result = wrapped(None, None, None, None, None)
    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.25
    assert result.feedback == "Non-numeric output"


def test_gepa_wrap_metric_passes_through_float():
    from src.training.optimization.gepa import GEPAOptimizer

    opt = GEPAOptimizer()

    def metric(*_args, **_kwargs):
        return 0.75

    wrapped = opt._wrap_metric_gepa(metric)
    assert wrapped(None, None, None, None, None) == 0.75


def test_gepa_wrap_metric_extracts_score_from_dict_without_feedback():
    from src.training.optimization.gepa import GEPAOptimizer

    opt = GEPAOptimizer()

    def metric(*_args, **_kwargs):
        return {"score": "0.5"}

    wrapped = opt._wrap_metric_gepa(metric)
    assert wrapped(None, None, None, None, None) == 0.5

