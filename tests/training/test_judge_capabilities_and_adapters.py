from __future__ import annotations

from typing import Dict

import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.preference_supervision import preference_supervision_metadata
from src.training.judge_optimization import JudgeOptimizationConfig, JudgeOptimizer
from src.training.preference.judge_capabilities import invoke_comparative_judgment_sync
from src.training.preference.optimizer_adapters import (
    build_dpo_training_records,
    build_group_grpo_training_records,
    build_reward_model_training_records,
    prepare_binary_optimizer_dataset,
)
from src.training.preference.types import (
    ComparativeCandidate,
    ComparativeDataset,
    ComparativeJudgmentRecord,
)


def _make_comparative_record(
    *,
    record_id: str = "cmp1",
    example_id: str = "doc1",
    num_candidates: int = 3,
) -> ComparativeJudgmentRecord:
    candidates = []
    for index in range(1, num_candidates + 1):
        score = float(num_candidates - index + 1)
        candidates.append(
            ComparativeCandidate(
                candidate_id=f"C{index}",
                response=f"summary {index}",
                rank=index,
                response_signal_value=score,
                metadata={"generation_config": {"temperature": 0.1 * index}},
            )
        )
    return ComparativeJudgmentRecord(
        record_id=record_id,
        source_example_id=example_id,
        original_text=f"original {example_id}",
        rubric="rubric",
        reference_score=0.75,
        law_type="sufficiency",
        candidates=candidates,
        sampling=SamplingMetadata(
            joint_propensity=0.25,
            unit_kind=ObservationUnitKind.PAIR,
        ),
        preference_supervision=preference_supervision_metadata(
            application_name="test_collection",
            law_type="sufficiency",
            response_signal_name="judge_score",
            response_signal_min=1.0,
            response_signal_max=float(num_candidates),
        ).with_updates(preference_family="groupwise"),
        aggregate_sample_weight=4.0,
        metadata={"confidence": 0.9, "reasoning": "ordered by utility"},
    )


def test_invoke_comparative_judgment_sync_falls_back_to_pairwise() -> None:
    score_map: Dict[str, float] = {
        "summary alpha": 3.0,
        "summary beta": 2.0,
        "summary gamma": 1.0,
    }

    class PairwiseOnlyJudge:
        def compare(
            self,
            *,
            context: str,
            original_text: str,
            summary_a: str,
            summary_b: str,
            law_type: str = "sufficiency",
        ) -> dict[str, object]:
            del context, original_text, law_type
            score_a = score_map[summary_a]
            score_b = score_map[summary_b]
            return {
                "preferred": "A" if score_a >= score_b else "B",
                "confidence": 0.9,
                "score_estimate_a": score_a,
                "score_estimate_b": score_b,
                "response_signal_name": "judge_score",
            }

    result = invoke_comparative_judgment_sync(
        PairwiseOnlyJudge(),
        context="rubric",
        original_text="original",
        candidate_summaries=["summary alpha", "summary beta", "summary gamma"],
        law_type="sufficiency",
    )

    assert result.ordered_candidate_ids == ["C1", "C2", "C3"]
    assert result.candidate_scores["C1"] == pytest.approx(3.0)
    assert result.candidate_scores["C2"] == pytest.approx(2.0)
    assert result.raw_payload["pairwise_fallback"] is True


def test_optimizer_adapters_handle_comparative_records_directly() -> None:
    dataset = ComparativeDataset([_make_comparative_record()])

    binary_dataset = prepare_binary_optimizer_dataset(
        dataset,
        projection="adjacent",
        keep_existing=False,
    )
    assert len(binary_dataset.pairs) == 2
    assert [pair.preferred for pair in binary_dataset.pairs] == ["A", "A"]

    dpo_rows = build_dpo_training_records(dataset, projection="adjacent")
    assert len(dpo_rows) == 2
    assert dpo_rows[0]["chosen"] == "summary 1"
    assert dpo_rows[0]["rejected"] == "summary 2"

    reward_rows = build_reward_model_training_records(dataset, projection="adjacent")
    assert len(reward_rows) == 2
    assert reward_rows[0]["chosen"] == "summary 1"
    assert reward_rows[0]["chosen_score"] == pytest.approx(3.0)
    assert reward_rows[0]["rejected_score"] == pytest.approx(2.0)

    grpo_rows = build_group_grpo_training_records(dataset)
    assert len(grpo_rows) == 1
    assert grpo_rows[0]["responses"] == ["summary 1", "summary 2", "summary 3"]
    assert grpo_rows[0]["ranks"] == [1, 2, 3]
    assert grpo_rows[0]["reference_score"] == pytest.approx(0.75)
    assert grpo_rows[0]["original_text"] == "original doc1"


def test_judge_optimizer_accepts_comparative_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    comparative_dataset = ComparativeDataset(
        [_make_comparative_record(record_id="cmp_many", example_id="doc_many", num_candidates=11)]
    )

    class FakeJudge:
        use_dspy_predictor = True

        def forward(
            self,
            *,
            context: str,
            original_text: str,
            summary_a: str,
            summary_b: str,
            law_type: str = "sufficiency",
        ) -> dict[str, object]:
            del context, original_text, summary_a, summary_b, law_type
            return {"preferred": "A", "confidence": 0.9}

    class FakeGEPA:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def compile(self, module, trainset):
            assert len(trainset) >= 8
            return module

    import src.training.judge_optimization as judge_optimization_module

    monkeypatch.setattr(judge_optimization_module.dspy, "GEPA", FakeGEPA)

    optimizer = JudgeOptimizer(
        config=JudgeOptimizationConfig(
            budget="light",
            num_threads=1,
            test_split=0.2,
            use_propensity_weighting=False,
        )
    )

    optimized_judge, results = optimizer.optimize(
        comparative_dataset,
        use_oracle_as_ground_truth=False,
        initial_judge=FakeJudge(),
    )

    assert isinstance(optimized_judge, FakeJudge)
    assert results["total_comparative_input"] == 1
    assert results["total_pairs_input"] == 10
    assert results["baseline"]["accuracy"] == pytest.approx(1.0)
