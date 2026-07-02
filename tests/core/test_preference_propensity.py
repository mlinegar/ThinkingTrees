"""Tests for preference propensity defaults and IPW utilities."""

import inspect
import math
from types import SimpleNamespace

import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.preference_supervision import preference_supervision_metadata
from src.core.supervision_metadata import judgment_supervision_metadata
from src.training.preference.types import PreferenceDataset, PreferencePair
from src.training.supervision import ResponseJudgment, SupervisionDataset
from src.training.supervision.optimizer_metadata import (
    build_treepo_optimizer_export_metadata,
    resolve_treepo_objective_weight,
)
from src.training.trl_training import (
    TRLPropensityWeightingConfig,
    TRLTrainingConfig,
    _build_processing_class_kwargs,
    _extract_sample_weight,
    _build_weighted_dpo_trainer,
    _build_weighted_grpo_trainer,
    _build_weighted_reward_trainer,
    _build_weighted_scalar_reward_trainer,
    _resample_records_by_weight,
)
from src.stats.sampling import (
    largest_remainder_allocation,
    normalize_positive_weights,
    pps_inclusion_probabilities,
    systematic_pps_sample_indices,
)


def _make_pair(pair_id: str, preferred: str = "A", **kwargs) -> PreferencePair:
    sampling = kwargs.pop("sampling", None)
    joint_propensity = kwargs.pop("joint_propensity", None)
    if sampling is None:
        sampling = SamplingMetadata(
            joint_propensity=joint_propensity,
            unit_kind=ObservationUnitKind.PAIR,
        )
    return PreferencePair(
        pair_id=pair_id,
        source_example_id=f"doc_{pair_id}",
        original_text="original",
        rubric="rubric",
        reference_score=0.5,
        summary_a="summary a",
        summary_b="summary b",
        preferred=preferred,
        reasoning="reason",
        confidence=0.8,
        sampling=sampling,
        **kwargs,
    )


def test_preference_pair_defaults_to_uniform_propensity():
    pair = _make_pair("p1")
    assert pair.sampling.document_propensity == 1.0
    assert pair.sampling.unit_propensity == 1.0
    assert pair.sampling.label_propensity == 1.0
    assert pair.effective_joint_propensity() == 1.0
    assert pair.ipw_weight() == 1.0


def test_preference_pair_uses_joint_propensity_for_weight():
    pair = _make_pair("p2", joint_propensity=0.2)
    assert pair.effective_joint_propensity() == 0.2
    assert pair.ipw_weight() == 5.0


def test_old_dict_load_gets_uniform_defaults():
    legacy = {
        "pair_id": "legacy",
        "source_example_id": "doc_legacy",
        "original_text": "original",
        "rubric": "rubric",
        "reference_score": 0.5,
        "summary_a": "a",
        "summary_b": "b",
        "preferred": "A",
        "reasoning": "r",
        "confidence": 0.7,
        "law_type": "sufficiency",
        "judge_model": "judge",
        "timestamp": "2026-01-01T00:00:00",
    }
    pair = PreferencePair.from_dict(legacy)
    assert pair.sampling.document_propensity == 1.0
    assert pair.sampling.unit_propensity == 1.0
    assert pair.sampling.label_propensity == 1.0
    assert pair.effective_joint_propensity() == 1.0
    assert pair.ipw_weight() == 1.0
    assert pair.preference_supervision.law_type == "sufficiency"
    assert pair.truth_label_source == "unknown"
    assert pair.source_doc_id is None


def test_preference_pair_roundtrip_preserves_provenance_fields():
    pair = _make_pair(
        "p_prov",
        truth_label_source="human",
        source_doc_id="doc_shared",
        three_layer_roles={"chunk": "train", "summarizer": "train", "oracle": "eval"},
        oracle_view="eval",
        oracle_proxy_source="embed_linear_v1",
    )
    as_dict = pair.to_dict()
    restored = PreferencePair.from_dict(as_dict)
    assert restored.truth_label_source == "human"
    assert restored.source_doc_id == "doc_shared"
    assert restored.three_layer_roles["oracle"] == "eval"
    assert restored.oracle_view == "eval"
    assert restored.oracle_proxy_source == "embed_linear_v1"


def test_dpo_export_contains_sample_weight():
    pair = _make_pair("p3", joint_propensity=0.25)
    dataset = PreferenceDataset([pair])
    exported = dataset.to_preference_format("dpo")
    assert len(exported) == 1
    assert exported[0]["sample_weight"] == 4.0
    assert exported[0]["metadata"]["preference_supervision"]["law_type"] == "sufficiency"
    assert exported[0]["metadata"]["treepo"]["sample_weight_source"] == "effective_weight"


def test_discounted_dpo_export_includes_treepo_rl_metadata():
    pair = _make_pair(
        "p_discounted",
        joint_propensity=0.25,
        sampling=SamplingMetadata(
            joint_propensity=0.25,
            unit_kind=ObservationUnitKind.PAIR,
            metadata={"depth": 2, "node_id": "leaf_7"},
        ),
    )
    dataset = PreferenceDataset([pair])

    exported = dataset.to_preference_format(
        "dpo",
        tree_objective_weighting_mode="discounted_tree",
        discount_gamma=0.5,
    )

    assert len(exported) == 1
    row = exported[0]
    treepo = row["metadata"]["treepo"]
    assert treepo["document_id"] == "doc_p_discounted"
    assert treepo["node_id"] == "leaf_7"
    assert treepo["depth"] == 2
    assert treepo["channel"] == "c1"
    assert treepo["objective_weight"] == pytest.approx(0.25)
    assert treepo["ipw_weight"] == pytest.approx(4.0)
    assert treepo["effective_weight"] == pytest.approx(1.0)
    assert treepo["sample_weight"] == pytest.approx(1.0)
    assert treepo["rl_role"] == "dpo_pair"
    assert row["sample_weight"] == pytest.approx(1.0)


def test_optimizer_metadata_preserves_local_law_adjustment_payload():
    metadata = build_treepo_optimizer_export_metadata(
        fallback_node_id="node_1",
        source_example_id="doc_1",
        source_doc_id="doc_1",
        source_observation_ids=(),
        sampling=SamplingMetadata(joint_propensity=0.5),
        law_type="sufficiency",
        supervision_channel_name=None,
        supervision_signal_name=None,
        local_law_adjustment={
            "corrected_mean": 0.2,
            "proxy_oracle_gap_bound": 0.1,
            "local_law_weight": 0.3,
        },
    )

    payload = metadata.to_dict()
    assert payload["sample_weight"] == pytest.approx(payload["effective_weight"])
    assert payload["local_law_adjustment"]["corrected_mean"] == pytest.approx(0.2)
    assert payload["local_law_adjustment"]["proxy_oracle_gap_bound"] == pytest.approx(0.1)


def test_extract_sample_weight_uses_zero_effective_weight():
    record = {
        "metadata": {
            "treepo": {
                "objective_weight": 0.0,
                "ipw_weight": 4.0,
                "effective_weight": 0.0,
            }
        }
    }
    assert _extract_sample_weight(record) == pytest.approx(0.0)


def test_resolve_treepo_objective_weight_discounted_root_vs_leaf():
    assert resolve_treepo_objective_weight(
        channel="root",
        depth=0,
        weighting_mode="discounted_tree",
        discount_gamma=0.0,
    ) == pytest.approx(1.0)
    assert resolve_treepo_objective_weight(
        channel="c1",
        depth=3,
        weighting_mode="discounted_tree",
        discount_gamma=0.0,
    ) == pytest.approx(0.0)


def test_exports_preserve_comparative_signal_payload():
    pair = _make_pair(
        "p_genrm",
        comparison_signal_value=1.0,
        score_estimate_a=5.0,
        score_estimate_b=2.0,
        preference_supervision=preference_supervision_metadata(
            law_type="sufficiency",
            comparison_signal_name="genrm_ranking_score",
            comparison_signal_min=1.0,
            comparison_signal_max=6.0,
            response_signal_name="genrm_helpfulness",
            response_signal_min=1.0,
            response_signal_max=5.0,
        ),
    )
    dataset = PreferenceDataset([pair])

    dpo_export = dataset.to_preference_format("dpo")[0]
    assert dpo_export["metadata"]["comparative_signal"]["comparison_signal_name"] == (
        "genrm_ranking_score"
    )
    assert dpo_export["metadata"]["comparative_signal"]["comparison_signal_value"] == 1.0
    assert dpo_export["metadata"]["comparative_signal"]["response_signal_a"] == 5.0

    reward_export = dataset.to_reward_model_format()[0]
    assert reward_export["metadata"]["comparative_signal"]["response_signal_name"] == (
        "genrm_helpfulness"
    )


def test_scalar_reward_export_uses_effective_treepo_weight():
    judgment = ResponseJudgment(
        judgment_id="j_scalar",
        source_example_id="doc_scalar",
        original_text="original",
        rubric="rubric",
        response="response",
        response_id="A",
        reference_score=0.5,
        law_type="document_level_target",
        source_doc_id="doc_scalar",
        sampling=SamplingMetadata(
            joint_propensity=0.5,
            unit_kind=ObservationUnitKind.PAIR,
            metadata={"depth": 1, "node_id": "root_1"},
        ),
        supervision_metadata=judgment_supervision_metadata(
            law_type="document_level_target",
            supervision_signal_name="document_level_target",
        ),
        response_signal_value=0.7,
    )
    dataset = SupervisionDataset(response_judgments=[judgment])

    rows = dataset.to_scalar_reward_records(
        tree_objective_weighting_mode="discounted_tree",
        discount_gamma=0.5,
    )

    assert len(rows) == 1
    treepo = rows[0]["metadata"]["treepo"]
    assert treepo["channel"] == "root"
    assert treepo["depth"] == 1
    assert treepo["objective_weight"] == pytest.approx(0.5)
    assert treepo["ipw_weight"] == pytest.approx(2.0)
    assert treepo["effective_weight"] == pytest.approx(1.0)
    assert treepo["rl_role"] == "scalar_reward"
    assert rows[0]["sample_weight"] == pytest.approx(1.0)


def test_propensity_resampling_biases_toward_high_ipw_weight():
    low_weight_pair = _make_pair("low", joint_propensity=1.0)
    high_weight_pair = _make_pair("high", joint_propensity=0.05)
    dataset = PreferenceDataset([low_weight_pair, high_weight_pair])

    resampled = dataset.resample_by_propensity(target_size=200, seed=7)
    ids = [pair.pair_id for pair in resampled.pairs]
    assert ids.count("high") > ids.count("low")


def test_normalize_positive_weights_degenerate_falls_back_to_uniform():
    normalized = normalize_positive_weights([0.0, -1.0, 0.0])
    assert normalized == [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def test_largest_remainder_allocation_matches_total():
    allocation = largest_remainder_allocation(7, [2.1, 1.4, 3.5])
    assert sum(allocation) == 7
    assert allocation == [2, 1, 4]


def test_pps_inclusion_probabilities_are_bounded_and_sum_to_sample_size():
    inclusion = pps_inclusion_probabilities([1.0, 2.0, 3.0, 4.0], sample_size=2)
    assert len(inclusion) == 4
    assert all(0.0 <= prob <= 1.0 for prob in inclusion)
    assert math.isclose(sum(inclusion), 2.0, rel_tol=1e-8, abs_tol=1e-8)


def test_systematic_pps_indices_have_expected_cardinality():
    inclusion = [0.2, 0.8, 0.4, 0.6]
    indices = systematic_pps_sample_indices(inclusion, sample_size=2)
    assert len(indices) == 2
    assert len(set(indices)) == 2
    assert all(0 <= idx < len(inclusion) for idx in indices)


def test_trl_resample_uses_propensity_stratify_key():
    records = [
        {"prompt": "a", "sample_weight": 1.0, "group": "x"},
        {"prompt": "b", "sample_weight": 2.0, "group": "y"},
        {"prompt": "c", "sample_weight": 3.0, "group": "x"},
        {"prompt": "d", "sample_weight": 4.0, "group": "y"},
    ]
    config = TRLTrainingConfig(
        propensity_weighting=TRLPropensityWeightingConfig(
            propensity_sampling_strategy="stratified_multinomial",
            propensity_stratify_key="group",
            propensity_random_seed=11,
        ),
    )

    sampled = _resample_records_by_weight(records, config)
    assert len(sampled) == len(records)
    assert all("prompt" in record for record in sampled)


def test_trl_training_config_validates_discount_gamma():
    with pytest.raises(ValueError):
        TRLTrainingConfig(
            propensity_weighting=TRLPropensityWeightingConfig(
                tree_objective_weighting_mode="discounted_tree",
                discount_gamma=1.5,
            ),
        )


def test_weighted_grpo_trainer_scales_advantages_by_sample_weight():
    import torch

    class DummyGRPOTrainer:
        def _generate_and_score_completions(self, inputs):
            return {"advantages": torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)}

    WeightedGRPOTrainer = _build_weighted_grpo_trainer(DummyGRPOTrainer)
    trainer = WeightedGRPOTrainer()
    inputs = [
        {"sample_weight": 1.0},
        {"sample_weight": 2.0},
        {"sample_weight": 1.0},
    ]

    batch = trainer._generate_and_score_completions(inputs)
    assert "sample_weight" in batch
    torch.testing.assert_close(
        batch["sample_weight"],
        torch.tensor([0.75, 1.5, 0.75], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch["advantages"],
        torch.tensor([0.75, -3.0, 2.25], dtype=torch.float32),
    )


def test_weighted_dpo_trainer_supports_dict_style_trl_outputs():
    import torch

    class DummyDPOTrainer:
        loss_type = ["sigmoid"]
        loss_weights = None
        args = SimpleNamespace(rpo_alpha=None)
        use_weighting = False
        aux_loss_enabled = False
        aux_loss_coef = 0.0

        def concatenated_forward(self, model, batch):
            return {
                "chosen_logps": torch.tensor([2.0, 4.0], dtype=torch.float32),
                "rejected_logps": torch.tensor([1.0, 1.5], dtype=torch.float32),
                "mean_chosen_logits": torch.tensor([0.2, 0.6], dtype=torch.float32),
                "mean_rejected_logits": torch.tensor([0.1, 0.3], dtype=torch.float32),
            }

        def compute_ref_log_probs(self, batch):
            return (
                torch.tensor([0.0, 0.0], dtype=torch.float32),
                torch.tensor([0.0, 0.0], dtype=torch.float32),
            )

        def dpo_loss(
            self,
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_type="sigmoid",
            model_output=None,
        ):
            losses = chosen_logps - rejected_logps
            return losses, chosen_logps, rejected_logps

    WeightedDPOTrainer = _build_weighted_dpo_trainer(DummyDPOTrainer)
    trainer = WeightedDPOTrainer()
    loss, metrics = trainer.get_batch_loss_metrics(
        model=None,
        batch={"sample_weight": [1.0, 3.0]},
        train_eval="train",
    )

    expected_loss = ((2.0 - 1.0) * 1.0 + (4.0 - 1.5) * 3.0) / (1.0 + 3.0)
    assert float(loss.item()) == pytest.approx(expected_loss)
    assert "rewards/chosen" in metrics
    assert "logps/chosen" in metrics


def test_build_processing_class_kwargs_prefers_processing_class():
    class NewStyleTrainer:
        def __init__(self, processing_class=None):
            self.processing_class = processing_class

    kwargs = _build_processing_class_kwargs(NewStyleTrainer, processing_class="tok")
    assert kwargs == {"processing_class": "tok"}


def test_build_processing_class_kwargs_falls_back_to_tokenizer():
    class OldStyleTrainer:
        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

    kwargs = _build_processing_class_kwargs(OldStyleTrainer, processing_class="tok")
    assert kwargs == {"tokenizer": "tok"}


def test_weighted_reward_trainer_matches_trl_compute_loss_signature_shape():
    class DummyRewardTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return 0.0

    WeightedRewardTrainer = _build_weighted_reward_trainer(DummyRewardTrainer)
    signature = inspect.signature(WeightedRewardTrainer.compute_loss)
    assert "num_items_in_batch" in signature.parameters


def test_weighted_scalar_reward_trainer_uses_ipw_weighted_regression_loss():
    import torch

    class DummyScalarTrainer:
        pass

    class DummyModel:
        def __call__(self, **kwargs):
            del kwargs
            return SimpleNamespace(logits=torch.tensor([[2.0], [5.0]], dtype=torch.float32))

    WeightedScalarTrainer = _build_weighted_scalar_reward_trainer(
        DummyScalarTrainer,
        loss_name="mse",
    )
    trainer = WeightedScalarTrainer()
    loss = trainer.compute_loss(
        DummyModel(),
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "labels": torch.tensor([1.0, 3.0], dtype=torch.float32),
            "sample_weight": [1.0, 3.0],
        },
    )

    expected = ((2.0 - 1.0) ** 2 * 1.0 + (5.0 - 3.0) ** 2 * 3.0) / 4.0
    assert float(loss.item()) == pytest.approx(expected)


def test_weighted_scalar_reward_trainer_matches_trainer_compute_loss_signature_shape():
    class DummyScalarTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return 0.0

    WeightedScalarTrainer = _build_weighted_scalar_reward_trainer(DummyScalarTrainer)
    signature = inspect.signature(WeightedScalarTrainer.compute_loss)
    assert "num_items_in_batch" in signature.parameters


def test_build_processing_class_kwargs_matches_current_trl_api():
    try:
        trl = __import__("trl")
    except Exception as exc:
        pytest.skip(f"trl import unavailable in test environment: {exc}")
    kwargs_dpo = _build_processing_class_kwargs(trl.DPOTrainer, processing_class="tok")
    kwargs_reward = _build_processing_class_kwargs(trl.RewardTrainer, processing_class="tok")
    assert kwargs_dpo == {"processing_class": "tok"}
    assert kwargs_reward == {"processing_class": "tok"}
