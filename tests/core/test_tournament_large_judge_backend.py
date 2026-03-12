from __future__ import annotations

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
