from __future__ import annotations

from src.ctreepo.sim.objective_backfill import safe_objective_backfill


def test_leaf_local_mixture_payload_backfills_objective_contract():
    payload = {
        "family": "leaf_local_mixture_utility",
        "target_kind": "local_nonlinear_leaf_sum",
        "config": {
            "lambda_multiplier": 2.0,
            "latent_partition_mode": "variable",
            "analysis_partition_mode": "shift_half",
        },
    }
    objective = safe_objective_backfill(payload)
    assert objective is not None
    assert objective["name"] == "leaf_local_mixture_utility_target"
    assert objective["optimized_against"] == "document_level_local_mixture_utility"
    assert objective["component_weights"] == {
        "topic_mixture_linear_term": 1.0,
        "local_topic_mixture_quadratic_term": 2.0,
    }


def test_segment_lda_payload_backfills_objective_contract():
    payload = {
        "family": "segment_lda_ops_weight_recovery",
        "config": {
            "lambda_multiplier": 1.5,
            "topic_process": "markov",
            "topic_phi_estimator": "true",
            "feature_inference": "hard",
        },
    }
    objective = safe_objective_backfill(payload)
    assert objective is not None
    assert objective["name"] == "segment_lda_oracle_target"
    assert objective["optimized_against"] == "ridge_regression_on_oracle_span_labels"
    assert objective["component_weights"] == {
        "latent_topic_counts": 1.0,
        "latent_topic_bigrams": 1.5,
    }
