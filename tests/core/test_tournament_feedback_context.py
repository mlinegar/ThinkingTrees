"""Tests for tournament feedback request context plumbing."""

import threading
from types import SimpleNamespace

import pytest


@pytest.mark.anyio
async def test_tournament_strategy_enriched_feedback_includes_judge_context():
    from src.core.strategy import CallableStrategy, TournamentConfig, TournamentStrategy
    from src.feedback.types import FeedbackResponse

    counter_lock = threading.Lock()
    counter = {"n": 0}

    def summarizer(*, content: str, rubric: str):
        with counter_lock:
            counter["n"] += 1
            idx = counter["n"]
        return f"candidate_{idx}: {content[:10]}"

    class FakeJudge:
        use_dspy_predictor = True
        use_dspy_prompt = False

        def forward(self, *, context: str, original_text: str, summary_a: str, summary_b: str, law_type: str):
            return SimpleNamespace(
                preference="A",
                reasoning="<think>internal</think> A is better.",
                confidence=0.9,
                helpfulness_a=2.0,
                helpfulness_b=1.0,
                ranking_score=2,
            )

    class RecordingCollector:
        def __init__(self):
            self.requests = []

        def collect(self, request, **_kwargs):
            self.requests.append(request)
            return FeedbackResponse(
                request_id=request.request_id,
                preferred="A",
                reasoning="ok",
                source="human",
            )

    collector = RecordingCollector()
    strategy = TournamentStrategy(
        base=CallableStrategy(summarizer=summarizer),
        judge=FakeJudge(),
        config=TournamentConfig(k=2),
        feedback_collector=collector,
    )

    winner = await strategy.summarize("Some document text", "Rubric")
    assert winner
    assert collector.requests

    req = collector.requests[0]
    assert req.text_b is not None
    assert {d.kind for d in req.dimensions} >= {"pairwise", "critique"}
    assert req.context.get("judge_preferred") == "A"
    assert req.context.get("judge_model") == "dspy-optimizable"
    assert req.context.get("judge_score_estimate_a") == 2.0
    assert req.context.get("judge_score_estimate_b") == 1.0
    assert "internal" not in (req.context.get("judge_reasoning") or "")
