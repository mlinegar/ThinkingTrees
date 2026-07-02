from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.ctreepo.sim.expectations import build_local_law_expectation_report
from src.ctreepo.sim.local_law_backfill import (
    collect_law_stress_assessments,
    compute_law_stress_for_summary,
    load_or_backfill_local_law_payload,
)
from src.ctreepo.sim.local_law_learnability import (
    DownstreamMetrics,
    GArtifact,
    LocalLawCounterexampleEvaluation,
    LocalLawMetrics,
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    SupportBudgetSummary,
)
from src.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl, write_manifest_jsonl
from src.ctreepo.sim.runner import run_commands


def _split_metrics(
    *,
    local_key: str,
    downstream_key: str,
    combined: float,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    delta: float = 0.0,
    abs_error: float = 0.0,
) -> dict:
    return {
        local_key: LocalLawMetrics(
            c1=float(c1),
            c2=float(c2),
            c3=float(c3),
            combined=float(combined),
        ).to_dict(),
        downstream_key: DownstreamMetrics(
            oracle_target_abs_error=float(abs_error),
            oracle_target_delta=float(delta),
        ).to_dict(),
    }


def _write_summary(path: Path, summary: LocalLawRunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"local_law_learnability": summary.to_dict()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _legacy_lda_payload() -> dict:
    return {
        "config": {
            "train_docs": 24,
            "val_docs": 8,
            "test_docs": 16,
            "seed": 7,
            "local_law_mode": "diagnostics_and_learned",
            "law_leaf_query_rate": 0.1,
            "law_internal_query_rate": 0.2,
            "analysis_partition_mode": "aligned",
            "lambda_multiplier": 1.5,
            "law_leaf_query_design": "uniform",
            "law_internal_query_design": "uniform",
            "law_c1_weight": 1.0,
            "law_c2_proxy_weight": 0.25,
            "law_c3_weight": 1.0,
        },
        "local_law": {
            "config": {
                "law_c1_threshold": 0.2,
                "law_c2_threshold": 0.2,
                "law_c3_threshold": 0.2,
            },
            "training": {
                "leaf_label_count": 12,
                "internal_label_count": 6,
            },
            "selection": {
                "selected_candidate": "law_calibrated_ipw_stabilized",
                "selection_split": "val",
                "selection_metric": "combined_law_score",
            },
            "policy_metrics": {
                "infer_identity": {
                    "mean_c1": 0.4,
                    "mean_c2_proxy": 0.3,
                    "mean_c3": 0.5,
                    "combined_law_score": 1.2,
                    "mean_aux_oracle_target_abs_error": 0.25,
                },
                "oracle_true_summary": {
                    "mean_c1": 0.0,
                    "mean_c2_proxy": 0.0,
                    "mean_c3": 0.0,
                    "combined_law_score": 0.0,
                    "mean_aux_oracle_target_abs_error": 0.0,
                },
                "law_calibrated_naive": {
                    "mean_c1": 0.25,
                    "mean_c2_proxy": 0.2,
                    "mean_c3": 0.3,
                    "combined_law_score": 0.75,
                    "mean_aux_oracle_target_abs_error": 0.15,
                },
                "law_calibrated_ipw": {
                    "mean_c1": 0.2,
                    "mean_c2_proxy": 0.15,
                    "mean_c3": 0.25,
                    "combined_law_score": 0.6,
                    "mean_aux_oracle_target_abs_error": 0.12,
                },
                "law_calibrated_ipw_stabilized": {
                    "mean_c1": 0.1,
                    "mean_c2_proxy": 0.08,
                    "mean_c3": 0.12,
                    "combined_law_score": 0.3,
                    "mean_aux_oracle_target_abs_error": 0.05,
                },
            },
            "ipw_evaluation": {
                "law_calibrated_ipw_stabilized": {
                    "c1": {
                        "population_exact_mean": 0.1,
                        "ht_mean": 0.11,
                        "hajek": 0.12,
                        "eb_lo": 0.09,
                        "eb_hi": 0.15,
                    },
                    "c2_proxy": {
                        "population_exact_mean": 0.08,
                        "ht_mean": 0.09,
                        "hajek": 0.10,
                        "eb_lo": 0.07,
                        "eb_hi": 0.12,
                    },
                    "c3": {
                        "population_exact_mean": 0.12,
                        "ht_mean": 0.13,
                        "hajek": 0.14,
                        "eb_lo": 0.11,
                        "eb_hi": 0.16,
                    },
                }
            },
        },
        "methods": {
            "analysis_infer_law_calibrated_oracle_target": {
                "diagnostics": {"calibration_variant": "law_calibrated_ipw_stabilized"}
            }
        },
    }


def _legacy_markov_payload(*, law_package: str = "root_only") -> dict:
    return {
        "config": {
            "train_docs": 16,
            "val_docs": 4,
            "test_docs": 8,
            "max_segments": 5,
            "violation_tau": 1.0,
            "leaf_query_rate": 0.1,
            "include_root_query": True,
            "fixed_leaf_tokens": 16,
            "feature_mode": "full",
            "model_family": "mlp",
            "n_regimes": 3,
            "effective_data_seed": 11,
            "effective_val_seed": 12,
            "effective_test_seed": 13,
            "law_package": law_package,
        },
        "objective": {"law_package": law_package},
        "training_geometry": {
            "mean_leaf_labels": 2.0,
            "mean_internal_labels": 1.0,
            "mean_internal_nodes": 2.0,
            "mean_queries_per_doc": 3.0,
            "total_queries_estimate": 48.0,
        },
        "metrics": {
            "exact": {
                "c1_leaf_mae_n": 0.0,
                "c2_idempotence_mae_n": 0.0,
                "c3_merge_mae_n": 0.0,
                "root_mae_n": 0.0,
            },
            "learned": {
                "val_objective_full_labels": 0.18,
                "test_objective_full_labels": 0.16,
                "val_theorem_bundle_score_n": 0.2,
            },
            "learned_train": {
                "c1_leaf_mae_n": 0.2,
                "c2_idempotence_mae_n": 0.15,
                "c3_merge_mae_n": 0.25,
                "root_mae_n": 0.3,
            },
            "learned_val": {
                "c1_leaf_mae_n": 0.18,
                "c2_idempotence_mae_n": 0.14,
                "c3_merge_mae_n": 0.22,
                "root_mae_n": 0.28,
            },
            "learned_test": {
                "c1_leaf_mae_n": 0.16,
                "c2_idempotence_mae_n": 0.12,
                "c3_merge_mae_n": 0.2,
                "root_mae_n": 0.24,
            },
            "leaf_bucket": {
                "c1_leaf_mae_n": 0.5,
                "c2_idempotence_mae_n": 0.05,
                "c3_merge_mae_n": 0.1,
            },
            "undersupported": {
                "c1_leaf_mae_n": 0.1,
                "c2_idempotence_mae_n": 0.05,
                "c3_merge_mae_n": 0.45,
            },
            "flip_R2": {
                "c1_leaf_mae_n": 0.05,
                "c2_idempotence_mae_n": 0.4,
                "c3_merge_mae_n": 0.08,
            },
        },
    }


def _stage3_only_payload() -> dict:
    return {
        "family": "leaf_local_mixture_utility",
        "target_kind": "local_nonlinear_leaf_sum",
        "config": {
            "train_docs": 48,
            "test_docs": 48,
            "seed": 11,
            "analysis_partition_mode": "aligned",
            "lambda_multiplier": 2.0,
        },
        "methods": {
            "budgeted_leaf_ridge_ipw": {
                "delta_mean": 0.1,
            }
        },
        "stage3": {
            "ipw_evaluation": {
                "target": {
                    "population_exact_mean": 1.0,
                }
            }
        },
    }


def test_local_law_schema_roundtrip():
    artifact = GArtifact(
        artifact_id="learned_g",
        name="law_calibrated_ipw_stabilized",
        role=PolicyRole.LEARNED_G,
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        fmt="json",
        manifest_path="/tmp/learned_g.json",
        metadata={"suite_role": "support_scaling"},
    )
    summary = LocalLawRunSummary(
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        oracle_name="oracle_true_summary",
        study_role="diagnostics_and_learned",
        split_ids={"train": "train", "val": "val", "test": "test"},
        support_budget=SupportBudgetSummary(
            train_docs=32,
            val_docs=8,
            test_docs=16,
            total_queries_estimate=64.0,
        ),
        selection={
            "selection_split": "val",
            "selection_metric": "combined_law_score",
            "selected_candidate": "law_calibrated_ipw_stabilized",
            "test_metrics_used_for_selection": False,
        },
        policies={
            "oracle_true_summary": LocalLawPolicyEvaluation(
                name="oracle_true_summary",
                role=PolicyRole.ORACLE_G,
                artifact_id="oracle_g",
                split_metrics={"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.0)},
            ),
            "learned_g": LocalLawPolicyEvaluation(
                name="law_calibrated_ipw_stabilized",
                role=PolicyRole.LEARNED_G,
                artifact_id=artifact.artifact_id,
                split_metrics={"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.1)},
            ),
        },
        counterexamples=[
            LocalLawCounterexampleEvaluation(
                name="shift_half",
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=["C3"],
                metrics={"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.3, c3=0.3)},
            )
        ],
        thresholds={"c1": 0.2, "c2": 0.2, "c3": 0.2},
        suite_role="support_scaling",
        compositional_learning_problem={
            "name": "schema_roundtrip_test",
            "uses_full_document_labels": True,
        },
        metadata={"analysis_partition_mode": "aligned", "lambda_multiplier": 1.5},
    )

    assert GArtifact.from_dict(artifact.to_dict()).to_dict() == artifact.to_dict()
    assert LocalLawRunSummary.from_dict(summary.to_dict()).to_dict() == summary.to_dict()


def test_direct_local_law_payload_backfills_compositional_learning_problem():
    summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role=PolicyRole.BASELINE_G.value,
        split_ids={"train": "train", "val": "val", "test": "test"},
        support_budget=SupportBudgetSummary(
            train_docs=16,
            val_docs=4,
            test_docs=8,
            leaf_query_rate=0.1,
            internal_query_rate=0.25,
            root_query_rate=1.0,
            mean_leaf_labels_per_doc=2.0,
            mean_internal_labels_per_doc=1.0,
            mean_queries_per_doc=3.0,
            total_queries_estimate=48.0,
        ),
        selection={
            "selection_split": "val",
            "selection_metric": "configured_objective_hajek",
            "selected_candidate": "root_only",
            "uses_test_metrics": False,
        },
        policies={
            "root_only": LocalLawPolicyEvaluation(
                name="root_only",
                role=PolicyRole.BASELINE_G,
                selection_metric_value=0.18,
                split_metrics={
                    "test": {
                        "local_law_metrics": LocalLawMetrics(
                            c1=0.2,
                            c2=0.1,
                            c3=0.25,
                            combined=0.55,
                            root_error=0.3,
                        ).to_dict(),
                        "downstream_metrics": DownstreamMetrics(
                            root_error=0.3,
                        ).to_dict(),
                        "objective_metrics": {
                            "task_weight": 1.0,
                            "available_estimators": ["exact", "hajek"],
                            "selection_estimator": "hajek",
                            "local_law_weights": {"c1": 0.0, "c2": 0.0, "c3": 0.0},
                            "proxy_weights": {},
                        },
                    }
                },
            )
        },
        counterexamples=[],
        thresholds={"c1_tau": 0.2, "c2_tau": 0.2, "c3_tau": 0.2},
        suite_role="support_scaling",
        metadata={"law_package": "root_only"},
    )

    direct_payload = {"local_law_learnability": summary.to_dict()}
    direct_payload["local_law_learnability"].pop("compositional_learning_problem", None)

    loaded = load_or_backfill_local_law_payload(
        direct_payload,
        source_path="outputs/markov_ops_count/direct/seed_0.json",
    )
    assert loaded is not None
    loaded_summary, augmented = loaded
    problem = dict(loaded_summary.compositional_learning_problem)
    assert problem["name"] == "markov_ops_count_local_law_learning"
    assert problem["uses_full_document_labels"] is True
    assert problem["uses_sampled_substructure_labels"] is True
    assert problem["requires_propensity_logging"] is True
    sampled_channel = problem["supervision_channels"][1]
    assert sampled_channel["delivery_mode"] == "online_oracle_query"
    assert sampled_channel["supports_unbiased_risk"] is True
    assert sampled_channel["query_policy"]["logs_realized_propensities"] is True
    assert augmented["local_law_learnability"]["compositional_learning_problem"]["name"] == (
        "markov_ops_count_local_law_learning"
    )


def test_legacy_backfill_maps_lda_and_markov_payloads():
    lda_loaded = load_or_backfill_local_law_payload(
        _legacy_lda_payload(),
        source_path="outputs/tree_relevant_lda_local_law_legacy/results/suite_b_local_law_learnability/train_24/seed_7.json",
    )
    assert lda_loaded is not None
    lda_summary, lda_augmented = lda_loaded
    assert lda_summary.family == "tree_relevant_lda_local_law"
    assert lda_summary.suite_role == "support_scaling"
    assert lda_summary.selection["selection_split"] == "val"
    assert lda_summary.selection["selected_candidate"] == "law_calibrated_ipw_stabilized"
    assert lda_summary.policies["infer_identity"].role == PolicyRole.BASELINE_G
    assert lda_summary.policies["oracle_true_summary"].role == PolicyRole.ORACLE_G
    assert lda_summary.policies["learned_g"].name == "law_calibrated_ipw_stabilized"
    lda_objective = lda_summary.policies["learned_g"].split_metrics["test"]["objective"]
    assert lda_objective["weighting_scheme"] == "legacy_local_law_only_weighted_sum"
    assert float(lda_objective["root_share"]) == pytest.approx(0.0)
    assert float(lda_objective["local_law_weight_total"]) == pytest.approx(1.0)
    assert float(lda_objective["configured_local_law_objective"]) == pytest.approx(0.24 / 2.25)
    assert float(lda_objective["configured_local_law_objective_hajek"]) == pytest.approx(
        0.285 / 2.25
    )
    assert lda_summary.metadata["resolved_local_law_weights"] == {
        "c1": pytest.approx(1.0),
        "c2_proxy": pytest.approx(0.25),
        "c3": pytest.approx(1.0),
    }
    lda_problem = dict(lda_summary.compositional_learning_problem)
    assert lda_problem["name"] == "leaf_local_mixture_utility_local_law_learning"
    assert lda_problem["uses_full_document_labels"] is False
    assert lda_problem["uses_sampled_substructure_labels"] is True
    assert lda_problem["uses_online_oracle_queries"] is True
    assert lda_problem["requires_propensity_logging"] is True
    lda_sampled_channel = lda_problem["supervision_channels"][1]
    assert lda_sampled_channel["delivery_mode"] == "online_oracle_query"
    assert lda_sampled_channel["supports_unbiased_risk"] is True
    assert "leaf=uniform" in lda_sampled_channel["query_policy"]["selection_strategy"]
    assert "local_law_learnability" in lda_augmented
    assert lda_augmented["_local_law_backfill"]["mode"] == "legacy_lda"

    markov_loaded = load_or_backfill_local_law_payload(
        _legacy_markov_payload(law_package="root_only"),
        source_path="outputs/markov_law_stress_legacy/transition_map_suite/markov_changepoint_ops_count/learned/seed_0.json",
    )
    assert markov_loaded is not None
    markov_summary, markov_augmented = markov_loaded
    assert markov_summary.family == "markov_ops_count"
    assert markov_summary.suite_role == "support_scaling"
    assert markov_summary.policies["root_only"].role == PolicyRole.BASELINE_G
    assert {counterexample.name for counterexample in markov_summary.counterexamples} == {
        "leaf_bucket",
        "count_only",
        "flip_R2",
    }
    assert markov_summary.selection["selection_split"] == "val"
    assert markov_summary.selection["selection_metric"] == "val_objective_full_labels"
    assert markov_summary.policies["root_only"].selection_metric_value == pytest.approx(0.18)
    markov_problem = dict(markov_summary.compositional_learning_problem)
    assert markov_problem["name"] == "markov_ops_count_local_law_learning"
    assert markov_problem["uses_full_document_labels"] is True
    assert markov_problem["uses_sampled_substructure_labels"] is True
    assert markov_problem["uses_online_oracle_queries"] is True
    assert markov_problem["requires_propensity_logging"] is True
    markov_sampled_channel = markov_problem["supervision_channels"][1]
    assert markov_sampled_channel["delivery_mode"] == "online_oracle_query"
    assert markov_sampled_channel["supports_unbiased_risk"] is False
    assert markov_sampled_channel["query_policy"]["logs_realized_propensities"] is False
    assert "local_law_learnability" in markov_augmented
    assert markov_augmented["_local_law_backfill"]["mode"] == "legacy_markov"

    exact_markov_loaded = load_or_backfill_local_law_payload(
        {
            **_legacy_markov_payload(law_package="root_only"),
            "config": {
                **_legacy_markov_payload(law_package="root_only")["config"],
                "exact_family": "count_only",
            },
            "metrics": {
                "exact": _legacy_markov_payload(law_package="root_only")["metrics"]["exact"],
                "stress_family": _legacy_markov_payload(law_package="root_only")["metrics"]["undersupported"],
                "leaf_bucket": _legacy_markov_payload(law_package="root_only")["metrics"]["leaf_bucket"],
                "undersupported": _legacy_markov_payload(law_package="root_only")["metrics"]["undersupported"],
                "flip_R2": _legacy_markov_payload(law_package="root_only")["metrics"]["flip_R2"],
            },
        },
        source_path="outputs/markov_law_stress_legacy/sanity_suite/markov_changepoint_ops_count/exact/exact_count_only/seed_0.json",
    )
    assert exact_markov_loaded is not None
    exact_markov_summary, _ = exact_markov_loaded
    assert exact_markov_summary.suite_role == "failure_modes"


def test_compute_law_stress_prefers_selected_learned_policy_and_downstream_metric():
    summary = LocalLawRunSummary(
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        oracle_name="oracle_true_summary",
        study_role="diagnostics_and_learned",
        split_ids={"train": "train", "val": "val", "test": "test"},
        support_budget=SupportBudgetSummary(train_docs=32, val_docs=8, test_docs=16, total_queries_estimate=32.0),
        selection={
            "selection_split": "val",
            "selection_metric": "combined_law_score",
            "selected_candidate": "law_calibrated_ipw",
            "test_metrics_used_for_selection": False,
        },
        policies={
            "infer_identity": LocalLawPolicyEvaluation(
                name="infer_identity",
                role=PolicyRole.BASELINE_G,
                split_metrics={
                    "test": {
                        "local_law": LocalLawMetrics(c1=0.4, c2=0.4, c3=0.4, combined=1.2, root_error=0.1).to_dict(),
                        "downstream": DownstreamMetrics(oracle_target_abs_error=1.0, root_error=0.1).to_dict(),
                    }
                },
            ),
            "law_calibrated_naive": LocalLawPolicyEvaluation(
                name="law_calibrated_naive",
                role=PolicyRole.CANDIDATE_G,
                split_metrics={
                    "test": {
                        "local_law": LocalLawMetrics(c1=0.05, c2=0.05, c3=0.05, combined=0.15, root_error=0.01).to_dict(),
                        "downstream": DownstreamMetrics(oracle_target_abs_error=0.98, root_error=0.01).to_dict(),
                    }
                },
            ),
            "learned_g": LocalLawPolicyEvaluation(
                name="law_calibrated_ipw",
                role=PolicyRole.LEARNED_G,
                split_metrics={
                    "test": {
                        "local_law": LocalLawMetrics(c1=0.2, c2=0.2, c3=0.2, combined=0.6, root_error=0.5).to_dict(),
                        "downstream": DownstreamMetrics(oracle_target_abs_error=0.5, root_error=0.5).to_dict(),
                    }
                },
            ),
        },
        counterexamples=[],
        thresholds={"c1": 0.2, "c2": 0.2, "c3": 0.2},
        suite_role="support_scaling",
        metadata={"law_package": "all_laws"},
    )
    raw_payload = {
        "local_law": {
            "selection": {"selected_candidate": "law_calibrated_ipw"},
            "law_stress": {
                "law_calibrated_naive": {
                    "bundle_status": "failure",
                    "bundle_full_success": False,
                    "primary_pass": False,
                    "primary_gain_frac": 0.02,
                    "c1_pass": True,
                    "c2_pass": True,
                    "c3_pass": True,
                    "laws_improved": 3,
                },
                "law_calibrated_ipw": {
                    "bundle_status": "full_success",
                    "bundle_full_success": True,
                    "primary_pass": True,
                    "primary_gain_frac": 0.5,
                    "c1_pass": True,
                    "c2_pass": True,
                    "c3_pass": True,
                    "laws_improved": 3,
                },
            },
        }
    }

    from_raw = compute_law_stress_for_summary(summary, raw_payload=raw_payload)
    assert from_raw is not None
    assert from_raw["primary_pass"] is True
    assert from_raw["primary_gain_frac"] == 0.5

    from_summary = compute_law_stress_for_summary(summary, raw_payload=None)
    assert from_summary is not None
    assert from_summary["primary_pass"] is True
    assert from_summary["c1_pass"] is True
    assert abs(float(from_summary["primary_gain_frac"]) - 0.5) < 1e-9


def test_collect_law_stress_assessments_pairs_markov_with_root_only_baseline():
    baseline_summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role=PolicyRole.BASELINE_G.value,
        split_ids={"train": "train", "val": "val", "test": "test"},
        support_budget=SupportBudgetSummary(train_docs=64, val_docs=32, test_docs=32, total_queries_estimate=128.0, metadata={"audit_fraction": 0.1}),
        selection={},
        policies={
            "root_only": LocalLawPolicyEvaluation(
                name="root_only",
                role=PolicyRole.BASELINE_G,
                split_metrics={
                    "test": {
                        "local_law_metrics": LocalLawMetrics(c1=0.5, c2=0.4, c3=0.5, combined=1.4, root_error=1.2, schedule_spread=0.3).to_dict(),
                        "downstream_metrics": DownstreamMetrics(root_error=1.2, schedule_spread=0.3).to_dict(),
                    }
                },
            )
        },
        counterexamples=[],
        thresholds={"c1_tau": 0.0, "c2_tau": 0.0, "c3_tau": 0.0},
        suite_role="support_scaling",
        metadata={"law_package": "root_only", "feature_mode": "full", "model_family": "neural", "n_regimes": 4, "fixed_leaf_tokens": 16},
    )
    learned_summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role=PolicyRole.LEARNED_G.value,
        split_ids={"train": "train", "val": "val", "test": "test"},
        support_budget=SupportBudgetSummary(train_docs=64, val_docs=32, test_docs=32, total_queries_estimate=128.0, metadata={"audit_fraction": 0.1}),
        selection={"selected_candidate": "all_laws", "selection_split": "val"},
        policies={
            "learned_g": LocalLawPolicyEvaluation(
                name="all_laws",
                role=PolicyRole.LEARNED_G,
                split_metrics={
                    "test": {
                        "local_law_metrics": LocalLawMetrics(c1=0.3, c2=0.2, c3=0.25, combined=0.75, root_error=0.8, schedule_spread=0.2).to_dict(),
                        "downstream_metrics": DownstreamMetrics(root_error=0.8, schedule_spread=0.2).to_dict(),
                    }
                },
            )
        },
        counterexamples=[],
        thresholds={"c1_tau": 0.0, "c2_tau": 0.0, "c3_tau": 0.0},
        suite_role="support_scaling",
        metadata={"law_package": "all_laws", "feature_mode": "full", "model_family": "neural", "n_regimes": 4, "fixed_leaf_tokens": 16},
    )
    baseline_payload = {
        "config": {
            "law_package": "root_only",
            "n_regimes": 4,
            "fixed_leaf_tokens": 16,
            "audit_fraction": 0.1,
            "root_weight": 1.0,
            "state_dim": 64,
            "hidden_dim": 256,
            "n_epochs": 10,
            "feature_mode": "full",
            "model_family": "neural",
            "effective_data_seed": 1,
            "effective_model_seed": 2,
            "effective_val_seed": 3,
            "effective_test_seed": 4,
        }
    }
    learned_payload = {
        "config": {
            **baseline_payload["config"],
            "law_package": "all_laws",
        }
    }

    assessments = list(
        collect_law_stress_assessments(
            [
                ("baseline.json", baseline_summary, baseline_payload),
                ("learned.json", learned_summary, learned_payload),
            ]
        )
    )
    assert len(assessments) == 1
    record = assessments[0]
    assert record["family"] == "markov_ops_count"
    assert record["law_package"] == "all_laws"
    assert record["baseline_source_path"] == "baseline.json"
    assert record["assessment"]["primary_pass"] is True
    assert record["assessment"]["c1_pass"] is True
    assert record["assessment"]["c2_pass"] is True
    assert record["assessment"]["c3_pass"] is True


def test_structured_local_law_expectations_pass_on_standardized_payloads(tmp_path: Path):
    markov_summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role="baseline_g",
        split_ids={"train": "m-train", "val": "m-val", "test": "m-test"},
        support_budget=SupportBudgetSummary(train_docs=16, val_docs=4, test_docs=8, total_queries_estimate=32.0),
        selection={"selection_split": "val", "selection_metric": "val_theorem_bundle_score_n", "selected_candidate": "root_only", "uses_test_metrics": False},
        policies={
            "oracle_g": LocalLawPolicyEvaluation(
                name="oracle_g",
                role=PolicyRole.ORACLE_G,
                split_metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.0)},
            ),
            "root_only": LocalLawPolicyEvaluation(
                name="root_only",
                role=PolicyRole.BASELINE_G,
                split_metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.2, c1=0.1, c3=0.1)},
            ),
        },
        counterexamples=[
            LocalLawCounterexampleEvaluation(
                name="leaf_bucket",
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=["C1"],
                metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.3, c1=0.3)},
            ),
            LocalLawCounterexampleEvaluation(
                name="count_only",
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=["C3"],
                metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.4, c3=0.4)},
            ),
            LocalLawCounterexampleEvaluation(
                name="flip_R2",
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=["C2"],
                metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.5, c2=0.5)},
            ),
        ],
        thresholds={"c1_tau": 0.0, "c2_tau": 0.0, "c3_tau": 0.0},
        suite_role="positive_controls",
        metadata={"fixed_leaf_tokens": 16, "feature_mode": "full", "model_family": "neural"},
    )
    lda_low = LocalLawRunSummary(
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        oracle_name="oracle_true_summary",
        study_role="diagnostics_and_learned",
        split_ids={"train": "l-train-1", "val": "l-val", "test": "l-test"},
        support_budget=SupportBudgetSummary(train_docs=16, val_docs=8, test_docs=8, total_queries_estimate=16.0),
        selection={"selection_split": "val", "selection_metric": "combined_law_score", "selected_candidate": "law_calibrated_ipw", "test_metrics_used_for_selection": False},
        policies={
            "oracle_true_summary": LocalLawPolicyEvaluation(
                name="oracle_true_summary",
                role=PolicyRole.ORACLE_G,
                split_metrics={"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.0)},
            ),
            "infer_identity": LocalLawPolicyEvaluation(
                name="infer_identity",
                role=PolicyRole.BASELINE_G,
                split_metrics={"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.6, delta=0.2)},
            ),
            "learned_g": LocalLawPolicyEvaluation(
                name="law_calibrated_ipw",
                role=PolicyRole.LEARNED_G,
                split_metrics={
                    "val": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.4, abs_error=0.1),
                    "test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.4, delta=0.18),
                },
            ),
        },
        counterexamples=[],
        thresholds={"c1": 0.2, "c2": 0.2, "c3": 0.2},
        suite_role="support_scaling",
        metadata={"analysis_partition_mode": "aligned", "lambda_multiplier": 1.5},
    )
    lda_high = LocalLawRunSummary.from_dict(
        {
            **lda_low.to_dict(),
            "split_ids": {"train": "l-train-2", "val": "l-val", "test": "l-test"},
            "support_budget": {
                **lda_low.support_budget.to_dict(),
                "train_docs": 64,
                "total_queries_estimate": 64.0,
            },
            "policies": {
                **lda_low.to_dict()["policies"],
                "infer_identity": {
                    **lda_low.to_dict()["policies"]["infer_identity"],
                    "split_metrics": {"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.6, delta=0.2)},
                },
                "learned_g": {
                    **lda_low.to_dict()["policies"]["learned_g"],
                    "split_metrics": {
                        "val": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.2, abs_error=0.05),
                        "test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.2, delta=0.15),
                    },
                },
            },
        }
    )
    lda_lambda_zero = LocalLawRunSummary.from_dict(
        {
            **lda_high.to_dict(),
            "selection": {
                "selection_split": "val",
                "selection_metric": "combined_law_score",
                "selected_candidate": "law_calibrated_ipw",
                "test_metrics_used_for_selection": False,
            },
            "metadata": {"analysis_partition_mode": "aligned", "lambda_multiplier": 0.0},
            "policies": {
                **lda_high.to_dict()["policies"],
                "infer_identity": {
                    **lda_high.to_dict()["policies"]["infer_identity"],
                    "split_metrics": {"test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.6, delta=0.01)},
                },
                "learned_g": {
                    **lda_high.to_dict()["policies"]["learned_g"],
                    "split_metrics": {
                        "val": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.2, abs_error=0.02),
                        "test": _split_metrics(local_key="local_law", downstream_key="downstream", combined=0.2, delta=0.02),
                    },
                },
            },
        }
    )

    _write_summary(tmp_path / "markov_positive.json", markov_summary)
    _write_summary(tmp_path / "lda_low.json", lda_low)
    _write_summary(tmp_path / "lda_high.json", lda_high)
    _write_summary(tmp_path / "lda_lambda_zero.json", lda_lambda_zero)

    report = build_local_law_expectation_report(output_root=tmp_path)
    findings = {(finding.kind, finding.title): finding.status for finding in report.expectations}
    assert any(kind == "local_law_oracle_ceiling" and status == "pass" for (kind, _), status in findings.items())
    assert any(kind == "counterexample_breaks_target" and status == "pass" for (kind, _), status in findings.items())
    assert any(kind == "support_scaling_improves_gap" and status == "pass" for (kind, _), status in findings.items())
    assert any(kind == "validation_only_selection" and status == "pass" for (kind, _), status in findings.items())
    assert any(kind == "lambda_zero_null_control" and status == "pass" for (kind, _), status in findings.items())


def test_organize_existing_local_law_runs_inventories_legacy_roots(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)

    outputs_root = tmp_path / "outputs"
    primary_root = outputs_root / "markov_law_stress_fixture"
    exploratory_root = outputs_root / "markov_local_law_learnability_fixture"

    primary_json = primary_root / "transition_map_suite" / "markov_changepoint_ops_count" / "learned" / "seed_0.json"
    primary_json.parent.mkdir(parents=True, exist_ok=True)
    primary_json.write_text(json.dumps(_legacy_markov_payload(law_package="root_only"), indent=2), encoding="utf-8")

    exploratory_json = exploratory_root / "transition_map_suite" / "markov_changepoint_ops_count" / "learned" / "seed_1.json"
    exploratory_json.parent.mkdir(parents=True, exist_ok=True)
    exploratory_json.write_text(json.dumps(_legacy_markov_payload(law_package="all_laws"), indent=2), encoding="utf-8")

    default_out = tmp_path / "inventory_default"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/organize_existing_local_law_runs.py",
            "--outputs-root",
            str(outputs_root),
            "--output-dir",
            str(default_out),
        ],
        cwd=repo_root,
        env=env,
    )
    default_summary = json.loads(
        (default_out / "existing_local_law_inventory_summary.json").read_text(encoding="utf-8")
    )
    assert default_summary["totals"]["manifest_runs"] == 1
    assert default_summary["included_roots"][0]["root"].endswith("markov_law_stress_fixture")
    assert default_summary["excluded_roots"][0]["root"].endswith("markov_local_law_learnability_fixture")

    explicit_out = tmp_path / "inventory_explicit"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/organize_existing_local_law_runs.py",
            "--outputs-root",
            str(outputs_root),
            "--output-dir",
            str(explicit_out),
            "--include-root",
            str(exploratory_root),
        ],
        cwd=repo_root,
        env=env,
    )
    explicit_summary = json.loads(
        (explicit_out / "existing_local_law_inventory_summary.json").read_text(encoding="utf-8")
    )
    assert explicit_summary["totals"]["manifest_runs"] == 1
    assert explicit_summary["included_roots"][0]["root"].endswith("markov_local_law_learnability_fixture")
    manifest_runs = read_manifest_jsonl(explicit_out / "existing_local_law_manifest.jsonl")
    assert len(manifest_runs) == 1
    assert manifest_runs[0].config["backfill_mode"] == "legacy_markov"


def test_organize_existing_local_law_runs_skips_stage3_payloads_as_non_local_law(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)

    outputs_root = tmp_path / "outputs"
    stage3_root = outputs_root / "tree_relevant_lda_stage3_fixture"
    stage3_json = stage3_root / "results" / "suite_c" / "seed_0.json"
    stage3_json.parent.mkdir(parents=True, exist_ok=True)
    stage3_json.write_text(json.dumps(_stage3_only_payload(), indent=2), encoding="utf-8")

    out_dir = tmp_path / "inventory_stage3"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/organize_existing_local_law_runs.py",
            "--outputs-root",
            str(outputs_root),
            "--output-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        env=env,
    )
    summary = json.loads((out_dir / "existing_local_law_inventory_summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["manifest_runs"] == 0
    assert summary["totals"]["skipped_non_local_law"] == 1
    assert summary["excluded_roots"][0]["root"].endswith("tree_relevant_lda_stage3_fixture")
    assert summary["excluded_roots"][0]["skipped_non_local_law"] == 1
    assert summary["excluded_roots"][0]["unsupported"] == 0


def test_direct_counterexample_selection_is_not_treated_as_validation_violation(tmp_path: Path):
    summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role="counterexample_g",
        split_ids={"train": "m-train", "val": "m-val", "test": "m-test"},
        support_budget=SupportBudgetSummary(train_docs=8, val_docs=4, test_docs=4, total_queries_estimate=16.0),
        selection={
            "selection_split": "config",
            "selection_metric": "configured_exact_family",
            "selected_candidate": "count_only",
            "uses_test_metrics": False,
        },
        policies={
            "count_only": LocalLawPolicyEvaluation(
                name="count_only",
                role=PolicyRole.COUNTEREXAMPLE_G,
                split_metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.4, c3=0.4)},
            )
        },
        counterexamples=[
            LocalLawCounterexampleEvaluation(
                name="count_only",
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=["C3"],
                metrics={"test": _split_metrics(local_key="local_law_metrics", downstream_key="downstream_metrics", combined=0.4, c3=0.4)},
            )
        ],
        thresholds={"c1_tau": 0.0, "c2_tau": 0.0, "c3_tau": 0.0},
        suite_role="",
        metadata={"fixed_leaf_tokens": 8, "feature_mode": "full", "model_family": "neural"},
    )
    path = tmp_path / "markov_law_stress_fixture" / "sanity_suite" / "markov_changepoint_ops_count" / "exact" / "exact_count_only" / "seed_0.json"
    _write_summary(path, summary)

    report = build_local_law_expectation_report(output_root=tmp_path)
    selection_finding = next(
        finding for finding in report.expectations if finding.kind == "validation_only_selection"
    )
    assert selection_finding.status == "pass"


def test_combined_local_law_meta_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["MPLBACKEND"] = "Agg"

    markov_root = tmp_path / "markov" / "sanity_suite" / "markov_changepoint_ops_count"
    lda_root = tmp_path / "lda"
    lda_results_root = lda_root / "results"

    markov_learned_json = markov_root / "learned" / "seed_0.json"
    markov_exact_json = markov_root / "exact" / "seed_0.json"
    lda_json = (
        lda_results_root
        / "suite_b_local_law_learnability"
        / "train_32"
        / "leafrate_0.1"
        / "internalrate_0.1"
        / "tau_8"
        / "lam_1.5"
        / "seed_0.json"
    )

    markov_manifest = tmp_path / "markov_manifest.jsonl"
    lda_manifest = tmp_path / "lda_manifest.jsonl"
    write_manifest_jsonl(
        markov_manifest,
        [
            RunSpec.create(
                family="markov_ops_count",
                config={"suite_role": "positive_controls"},
                outputs={
                    "json_summary": str(markov_learned_json),
                    "csv_summary": str(markov_learned_json.with_suffix(".csv")),
                    "artifact_dir": str(markov_learned_json.parent / "seed_0_artifacts"),
                },
                command=(
                    f"{sys.executable} src/ctreepo/sim/cli/run_markov_changepoint_ops_count.py "
                    f"--train-docs 4 --val-docs 2 --test-docs 4 --state-dim 8 --hidden-dim 32 "
                    f"--n-epochs 1 --batch-size 2 --lr 1e-3 --weight-decay 0.0 "
                    f"--law-package root_only --device cpu --torch-threads 1 --suite-role positive_controls "
                    f"--artifact-dir {markov_learned_json.parent / 'seed_0_artifacts'} "
                    f"--json-summary {markov_learned_json} --csv-summary {markov_learned_json.with_suffix('.csv')}"
                ),
            ),
            RunSpec.create(
                family="markov_ops_count",
                config={"suite_role": "failure_modes"},
                outputs={
                    "json_summary": str(markov_exact_json),
                    "csv_summary": str(markov_exact_json.with_suffix(".csv")),
                    "artifact_dir": str(markov_exact_json.parent / "seed_0_artifacts"),
                },
                command=(
                    f"{sys.executable} src/ctreepo/sim/cli/run_markov_changepoint_ops_count.py "
                    f"--train-docs 4 --val-docs 2 --test-docs 4 --state-dim 8 --hidden-dim 32 "
                    f"--n-epochs 1 --batch-size 2 --lr 1e-3 --weight-decay 0.0 "
                    f"--exact-family exact --device cpu --torch-threads 1 --suite-role failure_modes "
                    f"--artifact-dir {markov_exact_json.parent / 'seed_0_artifacts'} "
                    f"--json-summary {markov_exact_json} --csv-summary {markov_exact_json.with_suffix('.csv')}"
                ),
            ),
        ],
    )
    write_manifest_jsonl(
        lda_manifest,
        [
            RunSpec.create(
                family="tree_relevant_lda_local_law",
                config={"suite_role": "support_scaling"},
                outputs={
                    "json_summary": str(lda_json),
                    "csv_summary": str(lda_json.with_suffix(".csv")),
                    "artifact_dir": str(lda_json.parent / "seed_0_artifacts"),
                },
                command=(
                    f"{sys.executable} scripts/run_leaf_local_mixture_utility_simulation.py "
                    f"--train-docs 24 --val-docs 8 --test-docs 16 --doc-tokens 96 "
                    f"--latent-leaf-tokens 32 --analysis-leaf-tokens 32 --analysis-partition-mode aligned "
                    f"--local-mixture-concentration 8.0 --quadratic-utility-weight 1.5 "
                    f"--local-law-mode diagnostics_and_learned --law-leaf-query-rate 0.1 --law-internal-query-rate 0.1 "
                    f"--suite-role support_scaling --artifact-dir {lda_json.parent / 'seed_0_artifacts'} "
                    f"--json-summary {lda_json} --csv-summary {lda_json.with_suffix('.csv')}"
                ),
            )
        ],
    )

    merged_manifest = tmp_path / "merged_manifest.jsonl"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_local_law_meta_manifest.py",
            "--manifest",
            str(markov_manifest),
            "--manifest",
            str(lda_manifest),
            "--output-manifest",
            str(merged_manifest),
            "--cmd-file",
            str(tmp_path / "merged_cmds.txt"),
        ],
        cwd=repo_root,
        env=env,
    )

    results = run_commands(
        [run.command for run in read_manifest_jsonl(merged_manifest)],
        jobs=2,
        log_dir=tmp_path / "logs",
        fail_fast=True,
        env=env,
    )
    assert results
    assert all(int(result.returncode) == 0 for result in results)

    lda_payload = json.loads(lda_json.read_text(encoding="utf-8"))
    extra_targets = [
        lda_results_root / "suite_a_exact_controls" / "mode_aligned" / "tau_8" / "lam_1.5" / "seed_0.json",
        lda_results_root / "suite_c_mismatch_mediation" / "mode_aligned" / "tau_8" / "lam_1.5" / "seed_0.json",
        lda_results_root / "suite_d_ipw_sparse_labels" / "mode_aligned" / "leafdesign_uniform" / "internaldesign_uniform" / "leafrate_0.1" / "internalrate_0.1" / "tau_8" / "lam_1.5" / "seed_0.json",
        lda_results_root / "suite_e_hardness" / "mode_aligned" / "anchor_25" / "topicconc_0.2" / "tau_8" / "lam_1.5" / "seed_0.json",
    ]
    for path in extra_targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lda_payload, indent=2, sort_keys=True), encoding="utf-8")

    markov_report_dir = tmp_path / "markov_report"
    lda_report_dir = tmp_path / "lda_report"
    meta_report_dir = tmp_path / "meta_report"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.ctreepo.sim.cli.report.law_stress",
            "--family",
            "markov",
            "--input-root",
            str(markov_root),
            "--output-dir",
            str(markov_report_dir),
        ],
        cwd=repo_root,
        env=env,
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_tree_relevant_lda_local_law.py",
            "--input-root",
            str(lda_root),
            "--output-dir",
            str(lda_report_dir),
        ],
        cwd=repo_root,
        env=env,
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_local_law_meta.py",
            "--manifest",
            str(merged_manifest),
            "--output-dir",
            str(meta_report_dir),
        ],
        cwd=repo_root,
        env=env,
    )

    assert (markov_report_dir / "law_stress_report.pdf").exists()
    assert (lda_report_dir / "tree_relevant_lda_local_law_report.pdf").exists()
    assert (meta_report_dir / "local_law_meta_report.md").exists()
    assert (meta_report_dir / "local_law_meta_report_summary.json").exists()
