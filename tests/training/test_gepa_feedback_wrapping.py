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


def test_gepa_wrap_metric_records_immediate_timing_metadata():
    from src.training.optimization.gepa import GEPAOptimizer

    opt = GEPAOptimizer()

    def metric(*_args, **_kwargs):
        return 0.5

    wrapped = opt._wrap_metric_gepa(metric)
    timing = wrapped.supervision_timing

    assert timing["acquisition_policy"] == "synchronous_optimizer_metric"
    assert timing["activation_barrier"] == "immediate"
    assert timing["consumer"] == "gepa_optimizer"
    assert timing["blocking"] is True


def test_gepa_compile_audit_records_supervision_timing(monkeypatch):
    from src.training.optimization.gepa import GEPAOptimizer

    class FakeGEPA:
        def __init__(self, **_kwargs):
            pass

        def compile(self, student=None, trainset=None, valset=None):
            return student

    monkeypatch.setattr("src.training.optimization.gepa.dspy.GEPA", FakeGEPA)
    opt = GEPAOptimizer()
    student = {"module": "student"}

    compiled = opt.compile(
        student=student,
        trainset=[],
        valset=[],
        metric=lambda *_args, **_kwargs: 1.0,
    )

    assert compiled is student
    timing = opt.last_compile_audit["supervision_timing"]
    assert timing["acquisition_policy"] == "synchronous_optimizer_metric"
    assert timing["activation_barrier"] == "immediate"
    assert timing["blocking"] is True
