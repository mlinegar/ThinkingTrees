from __future__ import annotations

import pytest

from src.training.preference.types import PreferenceDataset, PreferencePair
from src.training.trl_training import TRLTrainingConfig, _build_grpo_train_records


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
        joint_propensity=float(joint_propensity),
    )


def test_build_grpo_train_records_preserves_reward_context_without_weighting() -> None:
    dataset = PreferenceDataset([_make_pair("p1", reference_score=0.25)])
    config = TRLTrainingConfig(use_propensity_weighting=False)

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


def test_build_grpo_train_records_preserves_reward_context_with_weighting() -> None:
    dataset = PreferenceDataset([_make_pair("p2", joint_propensity=0.25, reference_score=-0.4)])
    config = TRLTrainingConfig(
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
