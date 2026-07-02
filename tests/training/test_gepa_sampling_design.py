"""Tests for GEPA sampling designs and weighting behavior."""

import argparse
from types import SimpleNamespace

import pytest

from src.training.gepa_sampling import (
    sample_srswor_examples,
    sample_two_stage_pps_bernoulli,
)
from src.training.run_pipeline import (
    _apply_gepa_corrected_local_law_score,
    _apply_gepa_sampling_weight,
    _dispatch_gepa_sampling_examples,
    resolve_gepa_sampling_design,
)


def _build_examples(doc_sizes: list[int]) -> list[SimpleNamespace]:
    rows: list[SimpleNamespace] = []
    counter = 0
    for doc_idx, size in enumerate(doc_sizes):
        for _ in range(int(size)):
            rows.append(
                SimpleNamespace(
                    doc_id=f"doc_{doc_idx}",
                    row_id=f"row_{counter}",
                    rubric="rubric",
                    content=f"content_{counter}",
                    reference_score=0.5,
                )
            )
            counter += 1
    return rows


def _row_ids(examples: list[SimpleNamespace]) -> list[str]:
    return [str(getattr(ex, "row_id", "")) for ex in examples]


def test_two_stage_sampling_is_deterministic_for_seed():
    examples = _build_examples([6, 6, 6, 6, 6, 6])
    sampled_a, meta_a = sample_two_stage_pps_bernoulli(
        examples,
        component_id="leaf",
        split="train",
        seed=123,
        target_size=10,
        min_required=4,
    )
    sampled_b, meta_b = sample_two_stage_pps_bernoulli(
        examples,
        component_id="leaf",
        split="train",
        seed=123,
        target_size=10,
        min_required=4,
    )

    assert _row_ids(sampled_a) == _row_ids(sampled_b)
    assert meta_a["design"] == "two_stage_pps_bernoulli"
    assert meta_a["sample_size"] == meta_b["sample_size"]
    assert meta_a["seed"] == 123


def test_two_stage_sampling_logs_valid_propensities():
    examples = _build_examples([8, 8, 8, 8, 8])
    sampled, meta = sample_two_stage_pps_bernoulli(
        examples,
        component_id="merge",
        split="val",
        seed=17,
        target_size=10,
        min_required=1,
    )

    assert meta["enabled"] is True
    assert meta["design"] == "two_stage_pps_bernoulli"
    assert meta["doc_population_size"] == 5
    assert meta["doc_sample_size"] >= 1
    assert meta["joint_propensity_min"] > 0.0
    assert meta["joint_propensity_max"] <= 1.0

    for ex in sampled:
        assert 0.0 < ex.sampling_doc_inclusion_prob <= 1.0
        assert 0.0 < ex.sampling_node_given_doc_prob <= 1.0
        assert 0.0 < ex.sampling_joint_inclusion_prob <= 1.0
        assert ex.sampling_ipw_weight > 0.0
        assert ex.sampling_hajek_weight > 0.0
        assert ex.sampling_population_size == len(examples)
        assert ex.sampling_realized_sample_size == len(sampled)
        assert ex.sampling_joint_inclusion_prob == pytest.approx(
            ex.sampling_doc_inclusion_prob * ex.sampling_node_given_doc_prob
        )


def test_sampling_does_not_mutate_original_examples():
    examples = _build_examples([5, 5, 5, 5])
    sampled, _ = sample_two_stage_pps_bernoulli(
        examples,
        component_id="leaf",
        split="train",
        seed=22,
        target_size=8,
        min_required=1,
    )
    assert sampled
    assert not any(hasattr(ex, "sampling_design") for ex in examples)
    assert all(hasattr(ex, "sampling_design") for ex in sampled)


def test_two_stage_sampling_realized_size_varies_around_target():
    examples = _build_examples([12, 12, 12, 12])
    sizes = []
    for seed in range(40, 100):
        sampled, _ = sample_two_stage_pps_bernoulli(
            examples,
            component_id="leaf",
            split="train",
            seed=seed,
            target_size=8,
            min_required=1,
        )
        sizes.append(len(sampled))

    assert len(set(sizes)) > 1
    assert abs(sum(sizes) / len(sizes) - 8.0) < 2.0


def test_srswor_sampling_remains_deterministic_and_uniform():
    examples = _build_examples([1] * 20)
    sampled_a, meta_a = sample_srswor_examples(
        examples,
        component_id="scorer",
        split="train",
        seed=999,
        target_size=7,
        min_required=4,
    )
    sampled_b, meta_b = sample_srswor_examples(
        examples,
        component_id="scorer",
        split="train",
        seed=999,
        target_size=7,
        min_required=4,
    )

    assert _row_ids(sampled_a) == _row_ids(sampled_b)
    assert len(sampled_a) == 7
    assert meta_a["design"] == "srswor"
    assert meta_a["inclusion_prob"] == pytest.approx(7.0 / 20.0)
    assert meta_a["sample_size"] == meta_b["sample_size"] == 7
    for ex in sampled_a:
        assert ex.sampling_design == "srswor"
        assert ex.sampling_hajek_weight == pytest.approx(1.0)


def test_component_design_selection_defaults():
    args = argparse.Namespace(gepa_leaf_merge_sampling_design="two_stage_pps_bernoulli")
    assert resolve_gepa_sampling_design(args, "scorer") == "srswor"
    assert resolve_gepa_sampling_design(args, "leaf") == "two_stage_pps_bernoulli"
    assert resolve_gepa_sampling_design(args, "merge") == "two_stage_pps_bernoulli"

    args.gepa_leaf_merge_sampling_design = "srswor"
    assert resolve_gepa_sampling_design(args, "leaf") == "srswor"
    assert resolve_gepa_sampling_design(args, "merge") == "srswor"

    args.gepa_leaf_merge_sampling_design = "invalid_value"
    assert resolve_gepa_sampling_design(args, "leaf") == "two_stage_pps_bernoulli"


def test_sampling_dispatch_component_routing_smoke():
    examples = _build_examples([4, 4, 4, 4, 4])
    args = argparse.Namespace(
        gepa_leaf_merge_sampling_design="two_stage_pps_bernoulli",
        gepa_ipw_min_propensity=1e-6,
    )

    _, scorer_meta = _dispatch_gepa_sampling_examples(
        examples,
        args=args,
        component_id="scorer",
        split="train",
        seed=111,
        target_size=8,
        min_required=1,
    )
    _, leaf_meta = _dispatch_gepa_sampling_examples(
        examples,
        args=args,
        component_id="leaf",
        split="train",
        seed=111,
        target_size=8,
        min_required=1,
    )
    _, merge_meta = _dispatch_gepa_sampling_examples(
        examples,
        args=args,
        component_id="merge",
        split="train",
        seed=111,
        target_size=8,
        min_required=1,
    )
    assert scorer_meta["design"] == "srswor"
    assert leaf_meta["design"] == "two_stage_pps_bernoulli"
    assert merge_meta["design"] == "two_stage_pps_bernoulli"

    args.gepa_leaf_merge_sampling_design = "srswor"
    _, leaf_meta_srs = _dispatch_gepa_sampling_examples(
        examples,
        args=args,
        component_id="leaf",
        split="train",
        seed=111,
        target_size=8,
        min_required=1,
    )
    _, merge_meta_srs = _dispatch_gepa_sampling_examples(
        examples,
        args=args,
        component_id="merge",
        split="train",
        seed=111,
        target_size=8,
        min_required=1,
    )
    assert leaf_meta_srs["design"] == "srswor"
    assert merge_meta_srs["design"] == "srswor"


def test_apply_gepa_sampling_weight_handles_hajek_and_ht():
    example = SimpleNamespace(
        sampling_design="two_stage_pps_bernoulli",
        sampling_hajek_weight=1.5,
        sampling_ht_weight=0.4,
        sampling_ipw_weight=4.0,
        sampling_joint_inclusion_prob=0.25,
        sampling_population_size=100,
        sampling_realized_sample_size=10,
    )
    assert _apply_gepa_sampling_weight(example, 0.8, estimator="hajek") == pytest.approx(1.2)
    assert _apply_gepa_sampling_weight(example, 0.8, estimator="horvitz_thompson") == pytest.approx(0.32)

    delattr(example, "sampling_ht_weight")
    # fallback path computes HT-style scaling from ipw * (n/N)
    assert _apply_gepa_sampling_weight(example, 0.8, estimator="horvitz_thompson") == pytest.approx(0.32)


def test_gepa_corrected_local_law_score_uses_loss_level_adjustment():
    example = SimpleNamespace(
        sampling_design="two_stage_pps_bernoulli",
        sampling_hajek_weight=10.0,
        sampling_joint_inclusion_prob=0.5,
        local_law_adjustment={
            "enabled": True,
            "proxy_loss": 0.4,
            "oracle_loss": 0.1,
            "observed": True,
            "propensity": 0.5,
        },
    )

    # corrected loss = .4 + (.1-.4)/.5 = -.2, converted back to score and
    # not multiplied by the legacy Hajek score weight.
    assert _apply_gepa_corrected_local_law_score(example, 0.6) == pytest.approx(1.2)


def test_gepa_corrected_local_law_score_missing_payload_keeps_legacy_fallback():
    example = SimpleNamespace(sampling_hajek_weight=2.0)

    assert _apply_gepa_corrected_local_law_score(example, 0.6) is None
    assert _apply_gepa_sampling_weight(example, 0.6, estimator="hajek") == pytest.approx(0.6)


def test_ipw_estimators_track_population_mean_in_aggregate():
    examples = _build_examples([6, 8, 10, 12, 14, 16, 18, 20])
    for idx, ex in enumerate(examples):
        ex.y = float((idx % 17) / 17.0)

    true_mean = sum(ex.y for ex in examples) / float(len(examples))
    hajek_estimates: list[float] = []
    ht_estimates: list[float] = []
    for seed in range(300, 420):
        sampled, _ = sample_two_stage_pps_bernoulli(
            examples,
            component_id="leaf",
            split="train",
            seed=seed,
            target_size=10,
            min_required=1,
        )
        if not sampled:
            continue
        n = float(len(sampled))
        hajek_est = sum(float(ex.y) * float(ex.sampling_hajek_weight) for ex in sampled) / n
        ht_est = sum(float(ex.y) * float(ex.sampling_ht_weight) for ex in sampled) / n
        hajek_estimates.append(hajek_est)
        ht_estimates.append(ht_est)

    assert hajek_estimates
    assert ht_estimates
    avg_hajek = sum(hajek_estimates) / float(len(hajek_estimates))
    avg_ht = sum(ht_estimates) / float(len(ht_estimates))
    assert abs(avg_hajek - true_mean) < 0.06
    assert abs(avg_ht - true_mean) < 0.06
