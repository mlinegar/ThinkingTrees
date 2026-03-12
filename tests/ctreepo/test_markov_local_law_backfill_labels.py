from __future__ import annotations

from src.ctreepo.sim.local_law_backfill import (
    collect_law_stress_assessments,
    load_or_backfill_local_law_payload,
)


def _payload(
    *,
    law_package: str,
    study_role: str,
    c1_rel: float,
    c2_rel: float,
    c3_rel: float,
    root_error: float,
    c1: float,
    c2: float,
    c3: float,
) -> dict:
    policies = {}
    if study_role == "baseline_g":
        policies["root_only"] = {
            "name": "root_only",
            "role": "baseline_g",
            "split_metrics": {
                "test": {
                    "local_law_metrics": {
                        "c1": c1,
                        "c2": c2,
                        "c3": c3,
                        "root_error": root_error,
                    },
                    "downstream_metrics": {"root_error": root_error},
                }
            },
        }
    else:
        policies["learned_g"] = {
            "name": "learned_g",
            "role": "learned_g",
            "split_metrics": {
                "test": {
                    "local_law_metrics": {
                        "c1": c1,
                        "c2": c2,
                        "c3": c3,
                        "root_error": root_error,
                    },
                    "downstream_metrics": {"root_error": root_error},
                }
            },
        }

    return {
        "config": {
            "law_package": law_package,
            "c1_relative_weight": c1_rel,
            "c2_relative_weight": c2_rel,
            "c3_relative_weight": c3_rel,
            "train_docs": 128,
            "val_docs": 32,
            "test_docs": 64,
            "audit_fraction": 0.1,
            "root_weight": 1.0,
            "state_dim": 64,
            "hidden_dim": 256,
            "n_epochs": 4,
            "feature_mode": "full",
            "model_family": "neural",
            "effective_data_seed": 0,
            "effective_model_seed": 0,
            "effective_val_seed": 5000,
            "effective_test_seed": 10000,
        },
        "local_law_learnability": {
            "family": "markov_ops_count",
            "dgp": "markov_changepoint_ops_count",
            "oracle_name": "changepoint_count_exact_summary",
            "study_role": study_role,
            "split_ids": {
                "train": "markov:train:seed=0:docs=128",
                "val": "markov:val:seed=5000:docs=32",
                "test": "markov:test:seed=10000:docs=64",
            },
            "support_budget": {
                "train_docs": 128,
                "val_docs": 32,
                "test_docs": 64,
                "metadata": {"audit_fraction": 0.1},
            },
            "selection": {"selected_candidate": "learned_g"},
            "policies": policies,
            "counterexamples": [],
            "thresholds": {},
            "metadata": {},
        },
    }


def test_collect_law_stress_assessments_infers_weight_profile_for_blank_markov_package() -> None:
    baseline_payload = _payload(
        law_package="root_only",
        study_role="baseline_g",
        c1_rel=1.0,
        c2_rel=0.0,
        c3_rel=4.0,
        root_error=0.20,
        c1=0.30,
        c2=0.20,
        c3=0.40,
    )
    treatment_payload = _payload(
        law_package="",
        study_role="learned_g",
        c1_rel=0.0,
        c2_rel=1.0,
        c3_rel=0.0,
        root_error=0.10,
        c1=0.10,
        c2=0.10,
        c3=0.10,
    )

    baseline_summary, baseline_augmented = load_or_backfill_local_law_payload(
        baseline_payload,
        source_path="baseline.json",
    ) or (None, None)
    treatment_summary, treatment_augmented = load_or_backfill_local_law_payload(
        treatment_payload,
        source_path="treatment.json",
    ) or (None, None)

    assert baseline_summary is not None
    assert treatment_summary is not None

    assessments = collect_law_stress_assessments(
        [
            ("baseline.json", baseline_summary, baseline_augmented),
            ("treatment.json", treatment_summary, treatment_augmented),
        ]
    )

    assert len(assessments) == 1
    assert assessments[0]["law_package"] == "pure_c2"
