import numpy as np
import pytest

from src.ctreepo.sim.core.leaf_local_mixture_utility import (
    LeafLocalMixtureUtilityConfig,
    _base_leaf_utilities,
    _leaf_additive_utility,
    _sample_leaf_local_mixture_docs,
    run_leaf_local_mixture_utility_experiment,
    run_leaf_local_mixture_utility_experiment_from_world,
    sample_leaf_local_mixture_utility_world,
)
from src.ctreepo.sim.core.segment_lda_ops_weight_recovery import sample_topic_distributions


def _small_cfg(**overrides) -> LeafLocalMixtureUtilityConfig:
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
        latent_leaf_tokens=16,
        leaf_fraction=1.0,
        local_mixture_concentration=1.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=False,
        lambda_multiplier=1.0,
        train_docs=24,
        test_docs=24,
        budget_regime="all_leaves_labeled",
        leaf_label_budget=8.0,
        ridge_alpha=1e-3,
        seed=0,
    )
    base.update(overrides)
    return LeafLocalMixtureUtilityConfig(**base)


def _true_leaf_minus_pooled_gap(doc, *, theta, W_base, lambda_multiplier: float, latent_leaf_tokens: int, doc_tokens: int) -> float:
    oracle_true = float(
        np.sum(
            _base_leaf_utilities(
                doc,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=lambda_multiplier,
                latent_leaf_tokens=latent_leaf_tokens,
            )
        )
    )
    mean_local = np.mean(np.asarray(doc.local_topic_weights, dtype=np.float64), axis=0)
    pooled_true = float(doc_tokens) * _leaf_additive_utility(
        mean_local,
        theta=theta,
        W_base=W_base,
        lambda_multiplier=lambda_multiplier,
    )
    return float(oracle_true - pooled_true)


def test_low_tau_increases_local_mixture_dispersion():
    topics_phi, _meta = sample_topic_distributions(
        vocab_size=64,
        n_topics=4,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=6,
        anchor_multiplier=10.0,
        seed=11,
    )

    _docs_low, stats_low = _sample_leaf_local_mixture_docs(
        128,
        topics_phi=topics_phi,
        doc_tokens=96,
        latent_leaf_tokens=16,
        doc_topic_concentration=0.6,
        local_mixture_concentration=0.25,
        seed=17,
    )
    _docs_high, stats_high = _sample_leaf_local_mixture_docs(
        128,
        topics_phi=topics_phi,
        doc_tokens=96,
        latent_leaf_tokens=16,
        doc_topic_concentration=0.6,
        local_mixture_concentration=64.0,
        seed=17,
    )

    assert float(stats_low["mean_local_mixture_dispersion"]) > float(stats_high["mean_local_mixture_dispersion"])


def test_true_gap_is_zero_when_lambda_is_zero():
    cfg = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=0.0, seed=23)
    world = sample_leaf_local_mixture_utility_world(cfg)

    gaps = [
        _true_leaf_minus_pooled_gap(
            doc,
            theta=np.asarray(world.theta_true, dtype=np.float64),
            W_base=np.asarray(world.W_base, dtype=np.float64),
            lambda_multiplier=0.0,
            latent_leaf_tokens=int(cfg.latent_leaf_tokens),
            doc_tokens=int(cfg.doc_tokens),
        )
        for doc in world.docs_test
    ]

    assert max(abs(float(x)) for x in gaps) < 1e-10


def test_true_gap_scales_linearly_with_lambda_on_fixed_world():
    cfg_world = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=1.0, seed=29)
    world = sample_leaf_local_mixture_utility_world(cfg_world)
    theta = np.asarray(world.theta_true, dtype=np.float64)
    W_base = np.asarray(world.W_base, dtype=np.float64)

    gaps_lam1 = np.asarray(
        [
            _true_leaf_minus_pooled_gap(
                doc,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=1.0,
                latent_leaf_tokens=int(cfg_world.latent_leaf_tokens),
                doc_tokens=int(cfg_world.doc_tokens),
            )
            for doc in world.docs_test
        ],
        dtype=np.float64,
    )
    gaps_lam2 = np.asarray(
        [
            _true_leaf_minus_pooled_gap(
                doc,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=2.0,
                latent_leaf_tokens=int(cfg_world.latent_leaf_tokens),
                doc_tokens=int(cfg_world.doc_tokens),
            )
            for doc in world.docs_test
        ],
        dtype=np.float64,
    )

    assert np.mean(np.abs(gaps_lam1)) > 1e-6
    assert np.max(np.abs(gaps_lam2 - 2.0 * gaps_lam1)) < 1e-10


def test_summary_reports_zero_gap_signal_at_lambda_zero():
    cfg = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=0.0, seed=31)
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    assert float(summary.heterogeneity["mean_test_gap_signal"]) == pytest.approx(0.0, abs=1e-10)


def test_summary_gap_signal_scales_linearly_with_lambda_on_fixed_world():
    cfg_world = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=1.0, seed=37)
    world = sample_leaf_local_mixture_utility_world(cfg_world)

    cfg_lam1 = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=1.0, seed=37)
    cfg_lam2 = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=2.0, seed=37)

    summary_lam1 = run_leaf_local_mixture_utility_experiment_from_world(cfg_lam1, world)
    summary_lam2 = run_leaf_local_mixture_utility_experiment_from_world(cfg_lam2, world)

    gap1 = float(summary_lam1.heterogeneity["mean_test_gap_signal"])
    gap2 = float(summary_lam2.heterogeneity["mean_test_gap_signal"])

    assert abs(gap1) > 1e-6
    assert gap2 == pytest.approx(2.0 * gap1, abs=1e-10)


def test_summary_serializes_top_level_objective_contract():
    cfg = _small_cfg(local_mixture_concentration=0.25, lambda_multiplier=2.0, seed=41)
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    objective = dict(summary.objective)
    assert objective["name"] == "leaf_local_mixture_utility_target"
    assert objective["optimized_against"] == "document_level_local_mixture_utility"
    assert objective["weighting_scheme"] == "linear_plus_lambda_local_quadratic_utility"
    assert objective["component_weights"] == {
        "topic_mixture_linear_term": 1.0,
        "local_topic_mixture_quadratic_term": pytest.approx(2.0),
    }
