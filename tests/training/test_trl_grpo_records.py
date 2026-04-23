from __future__ import annotations

import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.preference_supervision import preference_supervision_metadata
from src.training.preference.types import (
    ComparativeCandidate,
    ComparativeDataset,
    ComparativeJudgmentRecord,
    PreferenceDataset,
    PreferencePair,
)
from src.training.supervision import SupervisionDataset
from src.training.trl_training import (
    TRLPropensityWeightingConfig,
    TRLTrainingConfig,
    _build_grpo_train_records,
)


def _trl_config(**propensity_kwargs) -> TRLTrainingConfig:
    return TRLTrainingConfig(
        propensity_weighting=TRLPropensityWeightingConfig(**propensity_kwargs)
    )


def _make_pair(
    pair_id: str,
    *,
    joint_propensity: float = 1.0,
    reference_score: float = 0.5,
    law_type: str = "sufficiency",
) -> PreferencePair:
    return PreferencePair(
        pair_id=pair_id,
        source_example_id=f"doc_{pair_id}",
        original_text=f"original text {pair_id}",
        rubric="rubric",
        reference_score=float(reference_score),
        summary_a="summary a",
        summary_b="summary b",
        preferred="A",
        reasoning="reason",
        confidence=0.9,
        law_type=law_type,
        sampling=SamplingMetadata(
            joint_propensity=float(joint_propensity),
            unit_kind=ObservationUnitKind.PAIR,
        ),
        comparison_signal_value=1.0,
        preference_supervision=preference_supervision_metadata(
            law_type=law_type,
            comparison_signal_name="genrm_ranking_score",
            comparison_signal_min=1.0,
            comparison_signal_max=6.0,
            response_signal_name="genrm_helpfulness",
            response_signal_min=1.0,
            response_signal_max=5.0,
        ),
    )


def test_build_grpo_train_records_preserves_reward_context_without_weighting() -> None:
    dataset = PreferenceDataset([_make_pair("p1", reference_score=0.25)])
    config = _trl_config(use_propensity_weighting=False)

    rows = _build_grpo_train_records(
        dataset,
        config=config,
        law_type=None,
        prompt_builder=None,
    )

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row.get("prompt"), str) and row["prompt"]
    assert row["reference_score"] == pytest.approx(0.25)
    assert row["original_text"] == "original text p1"
    assert row["sample_weight"] == pytest.approx(1.0)
    assert row["preference_supervision"]["law_type"] == "sufficiency"
    assert row["comparative_signal"]["comparison_signal_name"] == "genrm_ranking_score"
    assert row["metadata"]["comparative_signal"]["response_signal_name"] == "genrm_helpfulness"
    assert row["metadata"]["treepo"]["rl_role"] == "grpo_prompt"
    assert row["metadata"]["treepo"]["sample_weight_source"] == "effective_weight"


def test_build_grpo_train_records_preserves_reward_context_with_weighting() -> None:
    dataset = PreferenceDataset([_make_pair("p2", joint_propensity=0.25, reference_score=-0.4)])
    config = _trl_config(
        use_propensity_weighting=True,
        propensity_resample=False,
        propensity_native_loss_weighting=True,
    )

    rows = _build_grpo_train_records(
        dataset,
        config=config,
        law_type=None,
        prompt_builder=None,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["reference_score"] == pytest.approx(-0.4)
    assert row["original_text"] == "original text p2"
    assert row["sample_weight"] == pytest.approx(4.0)
    assert row["preference_supervision"]["supervision_signal_name"] == "pairwise_preference"
    assert row["comparative_signal"]["comparison_signal_value"] == pytest.approx(1.0)
    assert row["metadata"]["treepo"]["ipw_weight"] == pytest.approx(4.0)
    assert row["metadata"]["treepo"]["effective_weight"] == pytest.approx(4.0)


def test_grouped_grpo_uses_single_comparative_record_for_multiple_attempts() -> None:
    p_ab = PreferencePair(
        pair_id="p_ab",
        source_example_id="doc_multi",
        original_text="original text multi",
        rubric="rubric",
        reference_score=0.8,
        summary_a="summary a",
        summary_b="summary b",
        preferred="A",
        reasoning="a beats b",
        confidence=0.9,
        score_estimate_a=0.9,
        score_estimate_b=0.4,
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
    )
    p_ac = PreferencePair(
        pair_id="p_ac",
        source_example_id="doc_multi",
        original_text="original text multi",
        rubric="rubric",
        reference_score=0.8,
        summary_a="summary a",
        summary_b="summary c",
        preferred="A",
        reasoning="a beats c",
        confidence=0.9,
        score_estimate_a=0.9,
        score_estimate_b=0.2,
        sampling=SamplingMetadata(joint_propensity=0.25, unit_kind=ObservationUnitKind.PAIR),
    )
    p_bc = PreferencePair(
        pair_id="p_bc",
        source_example_id="doc_multi",
        original_text="original text multi",
        rubric="rubric",
        reference_score=0.8,
        summary_a="summary b",
        summary_b="summary c",
        preferred="A",
        reasoning="b beats c",
        confidence=0.8,
        score_estimate_a=0.4,
        score_estimate_b=0.2,
        sampling=SamplingMetadata(joint_propensity=0.25, unit_kind=ObservationUnitKind.PAIR),
    )
    dataset = PreferenceDataset([p_ab, p_ac, p_bc])

    comparative_dataset = dataset.to_comparative_dataset()
    assert len(comparative_dataset) == 1
    record = comparative_dataset.records[0]
    assert len(record.candidates) == 3
    assert [candidate.response for candidate in record.candidates] == [
        "summary a",
        "summary b",
        "summary c",
    ]
    assert [candidate.rank for candidate in record.candidates] == [1, 2, 3]
    assert record.aggregate_sample_weight == pytest.approx(10.0)

    grouped = dataset.to_grouped_grpo_format()
    assert len(grouped) == 1
    assert grouped[0]["responses"] == ["summary a", "summary b", "summary c"]
    assert grouped[0]["ranks"] == [1, 2, 3]

    rows = _build_grpo_train_records(
        dataset,
        config=_trl_config(use_propensity_weighting=True, propensity_native_loss_weighting=True),
        law_type=None,
        prompt_builder=None,
    )
    assert len(rows) == 1
    assert rows[0]["sample_weight"] == pytest.approx(10.0)
    assert rows[0]["metadata"]["num_candidates"] == 3
    assert rows[0]["metadata"]["treepo"]["ipw_weight"] == pytest.approx(10.0)
    assert rows[0]["metadata"]["treepo"]["joint_propensity"] == pytest.approx(0.1)
    assert rows[0]["metadata"]["treepo"]["joint_propensity_source"] == "aggregate_sample_weight"


def test_grouped_grpo_applies_discounted_tree_objective_weight() -> None:
    direct_record = ComparativeJudgmentRecord(
        record_id="cmp_discounted",
        source_example_id="doc_discounted",
        original_text="original text discounted",
        rubric="rubric",
        reference_score=0.6,
        law_type="sufficiency",
        candidates=[
            ComparativeCandidate(candidate_id="C1", response="summary a", rank=1, response_signal_value=0.9),
            ComparativeCandidate(candidate_id="C2", response="summary b", rank=2, response_signal_value=0.4),
        ],
        sampling=SamplingMetadata(
            joint_propensity=0.5,
            unit_kind=ObservationUnitKind.PAIR,
            metadata={"depth": 2, "node_id": "node_discounted"},
        ),
        aggregate_sample_weight=2.0,
    )

    rows = _build_grpo_train_records(
        ComparativeDataset([direct_record]),
        config=_trl_config(
            use_propensity_weighting=True,
            propensity_native_loss_weighting=True,
            tree_objective_weighting_mode="discounted_tree",
            discount_gamma=0.5,
        ),
        law_type=None,
        prompt_builder=None,
    )

    assert len(rows) == 1
    treepo = rows[0]["metadata"]["treepo"]
    assert treepo["node_id"] == "node_discounted"
    assert treepo["depth"] == 2
    assert treepo["objective_weight"] == pytest.approx(0.25)
    assert treepo["ipw_weight"] == pytest.approx(2.0)
    assert treepo["effective_weight"] == pytest.approx(0.5)
    assert rows[0]["sample_weight"] == pytest.approx(0.5)


def test_preference_dataset_preserves_direct_comparative_records(tmp_path) -> None:
    direct_record = ComparativeJudgmentRecord(
        record_id="cmp_direct",
        source_example_id="doc_direct",
        original_text="original text direct",
        rubric="rubric",
        reference_score=0.7,
        law_type="sufficiency",
        candidates=[
            ComparativeCandidate(candidate_id="C1", response="summary a", rank=2, response_signal_value=0.6),
            ComparativeCandidate(candidate_id="C2", response="summary b", rank=1, response_signal_value=0.9),
            ComparativeCandidate(candidate_id="C3", response="summary c", rank=3, response_signal_value=0.2),
        ],
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
        preference_supervision=preference_supervision_metadata(
            law_type="sufficiency",
            response_signal_name="listwise_candidate_score",
        ).with_updates(preference_family="groupwise"),
    )
    projected_pair = PreferencePair(
        pair_id="cmp_direct:top2",
        source_example_id="doc_direct",
        original_text="original text direct",
        rubric="rubric",
        reference_score=0.7,
        summary_a="summary b",
        summary_b="summary a",
        preferred="A",
        reasoning="b beats a",
        confidence=0.8,
        law_type="sufficiency",
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
    )
    dataset = PreferenceDataset([projected_pair], comparative_records=[direct_record])

    comparative_dataset = dataset.to_comparative_dataset()
    assert len(comparative_dataset) == 1
    assert comparative_dataset.records[0].record_id == "cmp_direct"
    assert comparative_dataset.records[0].candidates[1].rank == 1

    path = tmp_path / "preferences.json"
    dataset.save(path)
    loaded = PreferenceDataset.load(path)
    assert len(loaded.pairs) == 1
    assert len(loaded.comparative_records) == 1
    assert loaded.to_grouped_grpo_format()[0]["responses"] == [
        "summary b",
        "summary a",
        "summary c",
    ]


def test_build_grpo_train_records_accepts_comparative_dataset_directly() -> None:
    direct_record = ComparativeJudgmentRecord(
        record_id="cmp_direct_only",
        source_example_id="doc_direct_only",
        original_text="original text direct only",
        rubric="rubric",
        reference_score=0.6,
        law_type="sufficiency",
        candidates=[
            ComparativeCandidate(candidate_id="C1", response="summary a", rank=1, response_signal_value=0.9),
            ComparativeCandidate(candidate_id="C2", response="summary b", rank=2, response_signal_value=0.4),
            ComparativeCandidate(candidate_id="C3", response="summary c", rank=3, response_signal_value=0.1),
        ],
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
        preference_supervision=preference_supervision_metadata(
            law_type="sufficiency",
            response_signal_name="listwise_candidate_score",
        ).with_updates(preference_family="groupwise"),
        aggregate_sample_weight=2.0,
    )

    rows = _build_grpo_train_records(
        ComparativeDataset([direct_record]),
        config=_trl_config(use_propensity_weighting=False),
        law_type=None,
        prompt_builder=None,
    )

    assert len(rows) == 1
    assert rows[0]["reference_score"] == pytest.approx(0.6)
    assert rows[0]["original_text"] == "original text direct only"
    assert rows[0]["preference_supervision"]["preference_family"] == "groupwise"


def test_build_grpo_train_records_accepts_supervision_dataset_directly() -> None:
    direct_record = ComparativeJudgmentRecord(
        record_id="cmp_supervision",
        source_example_id="doc_supervision",
        original_text="original text supervision",
        rubric="rubric",
        reference_score=0.9,
        law_type="sufficiency",
        candidates=[
            ComparativeCandidate(candidate_id="C1", response="summary a", rank=1, response_signal_value=0.95),
            ComparativeCandidate(candidate_id="C2", response="summary b", rank=2, response_signal_value=0.3),
        ],
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
        aggregate_sample_weight=2.0,
    )

    rows = _build_grpo_train_records(
        SupervisionDataset(comparative_judgments=[direct_record]),
        config=_trl_config(use_propensity_weighting=False),
        law_type=None,
        prompt_builder=None,
    )

    assert len(rows) == 1
    assert rows[0]["reference_score"] == pytest.approx(0.9)
    assert rows[0]["responses"] == ["summary a", "summary b"]
