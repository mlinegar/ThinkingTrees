from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


@pytest.mark.anyio
async def test_tournament_strategy_uses_judge_backend_metadata() -> None:
    from src.core.strategy import CallableStrategy, TournamentConfig, TournamentStrategy

    class LargeJudge:
        judge_backend = "large_qwen"

        def compare(
            self,
            *,
            context: str,
            original_text: str,
            summary_a: str,
            summary_b: str,
            law_type: str = "sufficiency",
        ):
            return SimpleNamespace(
                preferred="A",
                confidence=0.8,
                reasoning="A retains more constraints.",
                score_estimate_a=4.0,
                score_estimate_b=2.0,
            )

    def summarizer(*, content: str, rubric: str) -> str:
        return f"{content[:24]} | {rubric[:18]}"

    strategy = TournamentStrategy(
        base=CallableStrategy(summarizer=summarizer),
        judge=LargeJudge(),
        config=TournamentConfig(k=2),
    )
    _ = await strategy.summarize("Document text", "Rubric text")
    prefs = strategy.get_preferences()
    assert prefs, "Tournament should collect at least one preference pair"
    assert prefs[0].judge_model == "large_qwen"
    assert prefs[0].preference_supervision.response_signal_name == "response_score"
    assert prefs[0].score_estimate_a == 4.0


@pytest.mark.anyio
async def test_tournament_strategy_collects_listwise_comparative_judgment() -> None:
    from src.core.strategy import CallableStrategy, TournamentConfig, TournamentStrategy

    counter_lock = threading.Lock()
    counter = {"n": 0}

    def summarizer(*, content: str, rubric: str) -> str:
        with counter_lock:
            counter["n"] += 1
            idx = counter["n"]
        return f"candidate_{idx}: {content[:12]} | {rubric[:10]}"

    class ListwiseJudge:
        judge_backend = "large_qwen_listwise"

        def __init__(self) -> None:
            self.calls = 0

        def rank_candidates(
            self,
            *,
            context: str,
            original_text: str,
            candidate_summaries,
            law_type: str = "sufficiency",
        ):
            self.calls += 1
            assert len(candidate_summaries) == 4
            return {
                "ordered_candidate_ids": ["C3", "C1", "C4", "C2"],
                "candidate_scores": {
                    "C1": 0.82,
                    "C2": 0.10,
                    "C3": 0.95,
                    "C4": 0.41,
                },
                "reasoning": "C3 preserves the most information overall.",
                "confidence": 0.88,
                "response_signal_name": "listwise_candidate_score",
            }

    judge = ListwiseJudge()
    strategy = TournamentStrategy(
        base=CallableStrategy(summarizer=summarizer),
        judge=judge,
        config=TournamentConfig(k=4),
    )

    winner = await strategy.summarize("Document text", "Rubric text")

    assert winner.startswith("candidate_3:")
    assert judge.calls == 1

    prefs = strategy.get_preferences()
    assert len(prefs) == 1
    assert prefs[0].pair_id.endswith("listwise_top2")
    assert prefs[0].judge_model == "large_qwen_listwise"
    assert prefs[0].summary_a == winner
    assert prefs[0].preferred == "A"
    assert prefs[0].preference_supervision.metadata["collection_mode"] == "listwise_projection"

    records = strategy.get_comparative_judgments()
    assert len(records) == 1
    assert records[0].judge_model == "large_qwen_listwise"
    assert records[0].preference_supervision.preference_family == "groupwise"
    assert [candidate.candidate_id for candidate in records[0].candidates] == [
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    assert [candidate.rank for candidate in records[0].candidates] == [2, 4, 1, 3]
