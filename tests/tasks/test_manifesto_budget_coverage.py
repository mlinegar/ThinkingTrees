"""Regression test: every dspy predictor in the manifesto surface area
must pass an explicit ``max_tokens`` override per-call.

If a new summarizer/merger/scorer/comparator module lands without wiring
the unified two-tier output budget from
``src/tasks/manifesto/pipeline_config.py``, this test fails with a clear
"LM call made without max_tokens override" message.

Why this matters: the LM default on our vLLM profile is 4200 tokens, and
``prompt_tokens + max_tokens`` must fit inside the 12K context window.
When a long summary is scored or compared, the default blows past the
cap and the request fails with a ContextWindowExceededError at runtime.
Pinning per-call caps at module-construction time is the invariant.
"""

from __future__ import annotations

import os

# Force env so pipeline.compute_output_budget knows the vLLM cap.
os.environ.setdefault("MANIFESTO_CONTEXT_WINDOW", "12000")

from typing import Any

import dspy
import pytest


class _RecordingLM(dspy.BaseLM):
    """A no-network LM that returns a canned completion and records every
    ``max_tokens`` passed per-call so we can assert budgets are set."""

    def __init__(self, response_text: str = "1"):
        super().__init__(model="recording/stub")
        self.model = "recording/stub"
        self.kwargs = {"temperature": 0.0, "max_tokens": 4200}
        self.calls: list[dict] = []
        self._response_text = response_text

    def __call__(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs).get("choices", [])

    def forward(self, prompt=None, messages=None, **kwargs):
        merged = {**self.kwargs, **kwargs}
        self.calls.append(merged)
        return {"choices": [{"message": {"content": self._response_text},
                             "finish_reason": "stop"}]}


@pytest.fixture(autouse=True)
def _configure_recording_lm(monkeypatch):
    lm = _RecordingLM()
    # DSPy uses dspy.settings.lm for the active LM.
    monkeypatch.setattr(dspy.settings, "lm", lm, raising=False)
    yield lm


def _assert_all_calls_have_explicit_budget(lm: _RecordingLM):
    """Every captured LM call must set max_tokens to something other than
    the baseline default 4200. Callers that set config={"max_tokens": N}
    surface N here via the merge in dspy.LM.forward."""
    assert lm.calls, "expected at least one LM call"
    unbudgeted = [c for c in lm.calls if c.get("max_tokens") == 4200]
    assert not unbudgeted, (
        f"{len(unbudgeted)} / {len(lm.calls)} LM calls left max_tokens at "
        f"the LM default 4200 — these must pass an explicit per-call override "
        f"via config={{\"max_tokens\": ...}}."
    )


def test_manifesto_summarizer_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.pipeline import ManifestoSummarizer
    summ = ManifestoSummarizer()
    try:
        summ(text="party advocates climate action. " * 40,
             rubric="preserve positions")
    except Exception:
        # DSPy may not parse our stub response; all we need is that the LM
        # call got made with the right kwargs.
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_unified_manifesto_g_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.pipeline import UnifiedManifestoG
    g = UnifiedManifestoG()
    try:
        g(content="party advocates climate action. " * 40,
          rubric="preserve positions")
    except Exception:
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_manifesto_merger_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.pipeline import ManifestoMerger
    m = ManifestoMerger()
    try:
        m(summary1="A" * 500, summary2="B" * 500, rubric="preserve")
    except Exception:
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_manifesto_scorer_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.pipeline import ManifestoScorer
    s = ManifestoScorer()
    try:
        s(summary="summary text", task_context="rile task context")
    except Exception:
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_dimension_scorer_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.dimension_scorer import DimensionScorer
    from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension
    s = DimensionScorer(BENOIT_DIMENSIONS[PolicyDimension.ECONOMIC])
    try:
        s(summary="summary text")
    except Exception:
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_joint_dimension_scorer_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.joint_scorer import JointDimensionScorer
    from src.tasks.manifesto.dimensions import PolicyDimension
    s = JointDimensionScorer()
    try:
        s(summary="summary text", dimension=PolicyDimension.ECONOMIC)
    except Exception:
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_rile_comparator_sets_budget(_configure_recording_lm):
    from src.tasks.manifesto.dspy_signatures import RILEComparator
    c = RILEComparator()
    try:
        c(original_text="A"*200, summary_text="B"*200,
          task_context="score -100..+100")
    except Exception:
        pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_leaf_and_merge_summarizers_set_budget(_configure_recording_lm):
    from src.tasks.manifesto.summarizer import LeafSummarizer, MergeSummarizer
    leaf = LeafSummarizer()
    merge = MergeSummarizer()
    for attempt in (
        lambda: leaf(content="text " * 100, rubric="preserve"),
        lambda: merge(left_summary="L"*200, right_summary="R"*200, rubric="preserve"),
    ):
        try:
            attempt()
        except Exception:
            pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)


def test_strategy_compatible_wrappers_inherit_budget(_configure_recording_lm):
    from src.tasks.manifesto.pipeline import (
        StrategyCompatibleSummarizer, StrategyCompatibleMerger,
    )
    summ = StrategyCompatibleSummarizer()
    mrg = StrategyCompatibleMerger()
    for attempt in (
        lambda: summ(content="text " * 100, rubric="preserve"),
        lambda: mrg(left_summary="L"*200, right_summary="R"*200, rubric="preserve"),
    ):
        try:
            attempt()
        except Exception:
            pass
    _assert_all_calls_have_explicit_budget(_configure_recording_lm)
