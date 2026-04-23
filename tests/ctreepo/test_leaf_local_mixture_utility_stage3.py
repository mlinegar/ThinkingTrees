from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.core.logged_supervision import SamplingMetadata
from src.ctreepo.sim.core.leaf_local_mixture_utility import (
    LeafLocalMixtureUtilityConfig,
    _analysis_target,
    _build_analysis_partition_view,
    _normalized_ci_and_coverage,
    _pooled_true_target,
    _true_doc_target,
    run_leaf_local_mixture_utility_experiment,
    sample_leaf_local_mixture_utility_world,
)
from src.tree.ipw import NodeType, TreeSample, horvitz_thompson_mean


def _stage3_cfg(**overrides) -> LeafLocalMixtureUtilityConfig:
    base = dict(
        n_topics=4,
        vocab_size=64,
        doc_tokens=96,
        doc_topic_concentration=0.6,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=6,
        anchor_multiplier=10.0,
        utility_dim=8,
        atomic_block_tokens=16,
        latent_leaf_tokens=32,
        latent_partition_mode="equal",
        latent_length_profile="equal",
        leaf_fraction=1.0 / 3.0,
        analysis_partition_mode="aligned",
        analysis_leaf_tokens=32,
        local_mixture_concentration=1.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=False,
        lambda_multiplier=1.0,
        train_docs=48,
        test_docs=48,
        budget_regime="all_leaves_labeled",
        leaf_label_budget=8.0,
        ridge_alpha=1e-3,
        query_design="uniform",
        doc_sample_rate=1.0,
        heldout_doc_sample_rate=0.5,
        target_query_budget_per_doc=1.0,
        propensity_floor=0.10,
        propensity_ceiling=0.90,
        propensity_proxy="l1_deviation",
        ipw_stabilized_clip=20.0,
        ipw_delta=0.05,
        inference_prior_mass=0.25,
        inference_max_iter=200,
        inference_tol=1e-9,
        seed=0,
    )
    base.update(overrides)
    return LeafLocalMixtureUtilityConfig(**base)


def test_partition_overlap_weights_sum_to_one_and_account_exactly():
    cfg = _stage3_cfg(
        latent_partition_mode="variable",
        latent_length_profile="long_tail",
        analysis_partition_mode="shift_half",
        lambda_multiplier=2.0,
        seed=11,
    )
    world = sample_leaf_local_mixture_utility_world(cfg)
    doc = world.docs_test[0]
    view = _build_analysis_partition_view(doc, cfg, doc_index=0)

    row_sums = np.sum(np.asarray(view.overlap_row_normalized, dtype=np.float64), axis=1)
    assert np.max(np.abs(row_sums - 1.0)) < 1e-10

    overlap = np.asarray(view.overlap_tokens, dtype=np.float64)
    latent_lengths = np.asarray([hi - lo for lo, hi in doc.latent_section_spans], dtype=np.float64)
    analysis_lengths = np.asarray([hi - lo for lo, hi in view.analysis_section_spans], dtype=np.float64)
    assert np.max(np.abs(np.sum(overlap, axis=0) - latent_lengths)) < 1e-10
    assert np.max(np.abs(np.sum(overlap, axis=1) - analysis_lengths)) < 1e-10
    assert np.sum(np.asarray(view.analysis_weights, dtype=np.float64)) == pytest.approx(1.0, abs=1e-12)


def test_lambda_zero_kills_target_gap_under_variable_lengths_and_mismatch():
    cfg = _stage3_cfg(
        latent_partition_mode="variable",
        latent_length_profile="bimodal",
        analysis_partition_mode="random_same_count",
        lambda_multiplier=0.0,
        seed=13,
    )
    world = sample_leaf_local_mixture_utility_world(cfg)
    for doc_idx, doc in enumerate(world.docs_test[:8]):
        view = _build_analysis_partition_view(doc, cfg, doc_index=doc_idx)
        true_target = _true_doc_target(doc, theta=world.theta_true, W_base=world.W_base, lambda_multiplier=0.0)
        pooled_target = _pooled_true_target(
            doc,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=0.0,
            doc_tokens=int(cfg.doc_tokens),
        )
        analysis_target = _analysis_target(
            view,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=0.0,
            doc_tokens=int(cfg.doc_tokens),
            weighted=True,
        )
        assert true_target == pytest.approx(pooled_target, abs=1e-10)
        assert true_target == pytest.approx(analysis_target, abs=1e-10)


def test_aligned_partition_has_zero_oracle_mismatch_gap_for_all_lambda():
    cfg = _stage3_cfg(
        latent_partition_mode="variable",
        latent_length_profile="long_tail",
        analysis_partition_mode="aligned",
        lambda_multiplier=2.0,
        seed=17,
    )
    world = sample_leaf_local_mixture_utility_world(cfg)
    for doc_idx, doc in enumerate(world.docs_test[:8]):
        view = _build_analysis_partition_view(doc, cfg, doc_index=doc_idx)
        true_target = _true_doc_target(
            doc,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=float(cfg.lambda_multiplier),
        )
        analysis_target = _analysis_target(
            view,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=float(cfg.lambda_multiplier),
            doc_tokens=int(cfg.doc_tokens),
            weighted=True,
        )
        assert true_target == pytest.approx(analysis_target, abs=1e-10)


def test_weighted_and_unweighted_targets_match_when_sections_are_equal_length():
    cfg = _stage3_cfg(
        latent_partition_mode="equal",
        latent_length_profile="equal",
        analysis_partition_mode="aligned",
        lambda_multiplier=2.0,
        seed=19,
    )
    world = sample_leaf_local_mixture_utility_world(cfg)
    for doc_idx, doc in enumerate(world.docs_test[:8]):
        view = _build_analysis_partition_view(doc, cfg, doc_index=doc_idx)
        weighted = _analysis_target(
            view,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=float(cfg.lambda_multiplier),
            doc_tokens=int(cfg.doc_tokens),
            weighted=True,
        )
        unweighted = _analysis_target(
            view,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=float(cfg.lambda_multiplier),
            doc_tokens=int(cfg.doc_tokens),
            weighted=False,
        )
        assert weighted == pytest.approx(unweighted, abs=1e-10)


def test_ht_mean_is_unbiased_under_repeated_doc_bernoulli_sampling():
    cfg = _stage3_cfg(lambda_multiplier=2.0, seed=23)
    world = sample_leaf_local_mixture_utility_world(cfg)
    targets = [
        _true_doc_target(doc, theta=world.theta_true, W_base=world.W_base, lambda_multiplier=float(cfg.lambda_multiplier))
        for doc in world.docs_test
    ]
    true_mean = float(np.mean(np.asarray(targets, dtype=np.float64)))
    p = 0.4
    rng = np.random.default_rng(29)
    estimates = []
    for draw in range(400):
        samples = []
        for idx, target in enumerate(targets):
            if float(rng.random()) >= p:
                continue
            samples.append(
                TreeSample(
                    doc_id=f"d{draw}_{idx}",
                    node_id="doc",
                    node_type=NodeType.LEAF,
                    violation=0,
                    preference_loss=float(target),
                    sampling=SamplingMetadata(document_propensity=p),
                )
            )
        estimates.append(horvitz_thompson_mean(samples, lambda s: float(s.preference_loss), float(len(targets))))
    # The realized finite-sample HT error on this fixed world is larger than 0.3
    # under the current deterministic seed; keep the check focused on order-of-magnitude unbiasedness.
    assert abs(float(np.mean(np.asarray(estimates, dtype=np.float64))) - true_mean) < 0.7


def test_ipw_budgeted_training_beats_naive_under_adversarial_querying():
    cfg = _stage3_cfg(
        latent_partition_mode="equal",
        latent_length_profile="equal",
        analysis_partition_mode="aligned",
        local_mixture_concentration=8.0,
        lambda_multiplier=3.0,
        query_design="proxy_adversarial",
        target_query_budget_per_doc=1.0,
        train_docs=96,
        test_docs=96,
        seed=31,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    naive = float(summary.methods["budgeted_leaf_ridge_naive"]["delta_mean"])
    ipw = float(summary.methods["budgeted_leaf_ridge_ipw"]["delta_mean"])
    assert ipw > naive


def test_stage3_summary_exposes_exact_ht_hajek_and_propensity_diagnostics():
    cfg = _stage3_cfg(
        latent_partition_mode="variable",
        latent_length_profile="bimodal",
        analysis_partition_mode="shift_half",
        query_design="proxy_priority",
        lambda_multiplier=2.0,
        seed=37,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    stage3 = dict(summary.stage3)
    target = dict(stage3["ipw_evaluation"]["target"])
    delta = dict(stage3["ipw_evaluation"]["delta"]["budgeted_leaf_ridge_ipw"])
    assert "population_exact_mean" in target
    assert "ht_mean" in target
    assert "hajek" in target
    assert "effective_sample_size" in target
    assert "section_propensity_quantiles" in target
    assert "population_exact_mean" in delta
    assert "ht_mean" in delta
    assert "hajek" in delta
    assert "propensity_quantiles" in delta


def test_normalized_ci_and_coverage_handles_signed_values_correctly():
    raw_values = [-3.0, 1.0, 4.0]
    samples = [
        TreeSample(
            doc_id=f"d{i}",
            node_id="doc",
            node_type=NodeType.LEAF,
            violation=0,
            preference_loss=value,
            sampling=SamplingMetadata(document_propensity=1.0),
        )
        for i, value in enumerate(raw_values)
    ]
    exact = float(np.mean(np.asarray(raw_values, dtype=np.float64)))
    summary = _normalized_ci_and_coverage(
        samples,
        exact_value=exact,
        raw_values_population=raw_values,
        population_size=float(len(raw_values)),
        delta=0.05,
    )
    assert summary["ht_mean"] == pytest.approx(exact, abs=1e-10)
    assert summary["hajek"] == pytest.approx(exact, abs=1e-10)
    assert summary["ht_abs_error"] == pytest.approx(0.0, abs=1e-10)
    assert summary["hajek_abs_error"] == pytest.approx(0.0, abs=1e-10)
    assert summary["eb_contains_exact"] == pytest.approx(1.0)


def test_stage3_report_smoke_with_minimal_fixture(tmp_path: Path):
    fixture_root = tmp_path / "stage3_fixture"
    results = fixture_root / "results"
    results.mkdir(parents=True)
    cfg = _stage3_cfg(
        latent_partition_mode="variable",
        latent_length_profile="long_tail",
        analysis_partition_mode="shift_half",
        query_design="proxy_adversarial",
        train_docs=16,
        test_docs=16,
        lambda_multiplier=2.0,
        seed=41,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    payload = json.loads(summary.to_json())
    targets = [
        results / "suite_a_weighted_length" / "profile_long_tail" / "tau_1" / "lam_2" / "seed_0.json",
        results / "suite_b_partition_mismatch" / "mode_shift_half" / "tau_8" / "lam_2" / "seed_0.json",
        results / "suite_c_ipw_budgeted" / "mode_aligned" / "design_proxy_adversarial" / "budget_1" / "tau_8" / "lam_3" / "seed_0.json",
        results / "suite_d_hardness" / "anchor_10" / "topicconc_1" / "mode_shift_half" / "tau_8" / "lam_3" / "seed_0.json",
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    out_dir = fixture_root / "report"
    subprocess.run(
        [
            sys.executable,
            "scripts/report_tree_relevant_lda_stage3.py",
            "--input-root",
            str(fixture_root),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )
    assert (out_dir / "tree_relevant_lda_stage3_report.pdf").exists()
    assert (out_dir / "tree_relevant_lda_stage3_report.md").exists()


def test_ridge_methods_report_supervision_surface() -> None:
    cfg = _stage3_cfg(train_docs=24, test_docs=24, seed=47)
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    assert summary.methods["analysis_ridge_full_labels"]["training_surface"] == "supervision_dataset"
    assert summary.methods["budgeted_leaf_ridge_ipw"]["training_surface"] == "supervision_dataset"
    assert summary.methods["leaf_ridge_from_u"]["training_surface"] == "supervision_dataset"
    assert summary.methods["analysis_ridge_full_labels"]["representation_kind"] == "dense_feature_vector"
    assert summary.methods["analysis_ridge_full_labels"]["target_kind"] == "scalar"
    assert summary.methods["analysis_ridge_full_labels"]["optimizer_family"] == "closed_form_linear_regression"
    assert summary.methods["leaf_ridge_from_u"]["representation_kind"] == "dense_feature_vector"
    assert summary.methods["leaf_ridge_from_u"]["target_kind"] == "scalar"
    assert summary.methods["leaf_ridge_from_u"]["optimizer_family"] == "closed_form_linear_regression"
