from __future__ import annotations

from types import SimpleNamespace

from src.training.preference.collector import PreferenceCollector
from src.training.preference.large_judge_dspy import LargeJudgeListwiseModule
from src.training.preference.types import GenerationConfig


class _SequentialSummarizer:
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self._index = 0

    def __call__(self, *, content: str, rubric: str) -> SimpleNamespace:
        del content, rubric
        if self._index >= len(self._outputs):
            raise IndexError("No more prepared outputs")
        output = self._outputs[self._index]
        self._index += 1
        return SimpleNamespace(summary=output)


class _FakeListwiseJudge:
    def rank_candidates(
        self,
        *,
        context: str,
        original_text: str,
        candidate_summaries: list[str],
        law_type: str = "sufficiency",
    ) -> dict[str, object]:
        del context, original_text, law_type
        assert len(candidate_summaries) == 3
        return {
            "ordered_candidate_ids": ["C2", "C1", "C3"],
            "candidate_scores": {"C1": 3.5, "C2": 4.9, "C3": 1.2},
            "reasoning": "C2 best, then C1, then C3.",
            "confidence": 0.85,
            "response_signal_name": "judge_score",
        }


def test_large_judge_listwise_module_parses_order_and_scores() -> None:
    module = LargeJudgeListwiseModule(use_cot=False)
    module.compare = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        ordered_candidates="C3 > C1 > C2",
        candidate_scores_json='{"C1": 4.0, "C2": 2.0, "C3": 4.5}',
        reasoning="C3 strongest.",
        confidence="0.9",
    )

    result = module.rank_candidates(
        context="Preserve key facts.",
        original_text="Original",
        candidate_summaries=["A", "B", "C"],
        law_type="sufficiency",
    )

    assert result["ordered_candidate_ids"] == ["C3", "C1", "C2"]
    assert result["candidate_scores"] == {"C1": 4.0, "C2": 2.0, "C3": 4.5}
    assert result["confidence"] == 0.9


def test_preference_collector_collects_single_comparative_record_listwise() -> None:
    collector = PreferenceCollector(
        summarizer=_SequentialSummarizer(["summary one", "summary two", "summary three"]),
        judge=_FakeListwiseJudge(),
        strategy="judge",
        k=3,
        generation_configs=[
            GenerationConfig(temperature=0.2, prompt_variant="a"),
            GenerationConfig(temperature=0.5, prompt_variant="b"),
            GenerationConfig(temperature=0.8, prompt_variant="c"),
        ],
        comparison_mode="listwise",
    )

    record = collector.collect_comparative_for_example(
        example_id="doc1",
        original_text="Original text",
        rubric="Preserve key facts.",
        reference_score=0.0,
        law_type="sufficiency",
    )

    assert record.source_example_id == "doc1"
    assert len(record.candidates) == 3
    assert [candidate.response for candidate in record.candidates] == [
        "summary one",
        "summary two",
        "summary three",
    ]
    assert [candidate.rank for candidate in record.candidates] == [2, 1, 3]
    assert record.preference_supervision.preference_family == "groupwise"
    assert record.preference_supervision.response_signal_name == "judge_score"
    assert collector.get_comparative_dataset().records[0].record_id == record.record_id
    stats = collector.get_statistics()
    assert stats["comparative"]["records_collected"] == 1
    assert stats["comparative"]["candidates_total"] == 3
