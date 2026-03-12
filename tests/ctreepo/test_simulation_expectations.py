from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.ctreepo.sim.expectations import (
    BudgetTrendExpectation,
    ExpectationConfig,
    ExpectationReport,
    MarkovOPSAdapter,
    MergeableAblationAdapter,
    SegmentLDAOPSAdapter,
    SegmentedLDACtreePOAdapter,
    VALID_FAMILIES,
    assess_trend,
    build_expectation_report,
)
from src.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _trend_rows(values: list[tuple[float, float]], *, test_identity: str | None = "shared") -> list:
    from src.ctreepo.sim.expectations import NormalizedRow

    return [
        NormalizedRow(
            family="trend",
            scenario="demo",
            seed=i,
            method="demo",
            x_axis_name="train_docs",
            x_axis_value=float(x),
            secondary_axis_name=None,
            secondary_axis_value=None,
            metric_name="root_mae",
            metric_value=float(y),
            doc_scale_tokens=128.0,
            leaf_tokens=16.0,
            leaves_per_doc=8.0,
            oracle_budget_fraction=1.0,
            train_docs=float(x),
            evidence_status="APPROX_AUDITED",
            source_path="/tmp/demo.json",
            test_identity=test_identity,
            metadata={},
        )
        for i, (x, y) in enumerate(values)
    ]


def _markov_payload(*, train_docs: int, audit_fraction: float, leaf_tokens: int, seed: int, learned_root: float, learned_merge: float, unders_root: float, unders_merge: float) -> dict:
    return {
        "config": {
            "train_docs": int(train_docs),
            "test_docs": 64,
            "audit_fraction": float(audit_fraction),
            "fixed_leaf_tokens": int(leaf_tokens),
            "seed": int(seed),
            "model_family": "additive",
            "feature_mode": "full",
            "leaf_query_rate": 1.0,
            "include_root_query": True,
            "local_law_weight": 0.4,
            "transition_log_std": 1.25,
            "min_segments": 12,
            "max_segments": 24,
            "min_seg_len": 8,
            "max_seg_len": 32,
            "schedule_consistency_weight": 0.0,
            "test_identity": "markov-test-fixed",
        },
        "training_geometry": {
            "mean_tokens": 320.0,
            "mean_leaves": float(320 / leaf_tokens),
        },
        "objective": {
            "name": "configured_objective",
            "kind": "local_law_weighted_objective",
            "optimized_against": "weighted_task_plus_local_laws",
            "weighting_scheme": "normalized_lambda_tradeoff",
            "parameterization": "formal_local_law_weight",
            "local_law_active": True,
            "local_law_weight": 0.4,
            "selection_metric_name": "configured_objective_hajek",
            "interprets_lambda_as": "optimization_tradeoff",
            "component_weights": {
                "task": 0.4,
                "c1": 0.2,
                "c2_proxy": 0.2,
                "c3": 0.2,
            },
            "metadata": {"law_package": "all_laws"},
        },
        "metrics": {
            "exact": {
                "root_mae": 0.0,
                "merge_mae": 0.0,
                "schedule_spread_mean": 0.0,
            },
            "undersupported": {
                "root_mae": float(unders_root),
                "merge_mae": float(unders_merge),
                "schedule_spread_mean": 0.0,
            },
            "learned": {
                "root_mae": float(learned_root),
                "merge_mae": float(learned_merge),
                "schedule_spread_mean": 0.05 + 0.01 * seed,
            },
        },
        "estimator_diagnostics": {},
    }


def _segment_payload(
    *,
    train_docs: int,
    audit_fraction: float,
    seed: int,
    topic_process: str,
    lambda_multiplier: float,
    ridge_root: float,
    ridge_merge: float,
    unders_root: float,
) -> dict:
    return {
        "config": {
            "train_docs": int(train_docs),
            "test_docs": 64,
            "audit_fraction": float(audit_fraction),
            "leaf_tokens": 16,
            "seed": int(seed),
            "topic_process": str(topic_process),
            "lambda_multiplier": float(lambda_multiplier),
            "topic_phi_estimator": "true",
            "feature_inference": "hard",
            "test_identity": f"segment-test-{topic_process}",
        },
        "training_geometry": {
            "mean_tokens": 256.0,
            "mean_leaves": 16.0,
        },
        "objective": {
            "name": "segment_lda_oracle_target",
            "kind": "latent_oracle_target",
            "optimized_against": "ridge_regression_on_oracle_span_labels",
            "weighting_scheme": "linear_plus_lambda_interaction",
            "selection_metric_name": "",
            "interprets_lambda_as": "dgp_term_multiplier",
            "component_weights": {
                "latent_topic_counts": 1.0,
                "latent_topic_bigrams": float(lambda_multiplier),
            },
            "metadata": {"family": "segment_lda_ops_weight_recovery"},
        },
        "weight_truth": {},
        "metrics": {
            "exact": {
                "root_mae": 0.0,
                "merge_mae": 0.0,
                "schedule_spread_mean": 0.0,
            },
            "undersupported": {
                "root_mae": float(unders_root),
                "merge_mae": float(unders_root * 0.9),
                "schedule_spread_mean": 0.0,
            },
            "flip_R1": {
                "root_mae": 0.01,
                "merge_mae": 0.02,
                "schedule_spread_mean": 0.18,
            },
            "ridge": {
                "root_mae": float(ridge_root),
                "merge_mae": float(ridge_merge),
                "schedule_spread_mean": 0.02,
                "theta_rmse": 0.12,
                "lambda_abs_error": 0.09,
                "leaf_accuracy_test": 0.88,
            },
            "ridge_true_topics": {
                "root_mae": float(ridge_root * 0.8),
                "merge_mae": float(ridge_merge * 0.8),
                "schedule_spread_mean": 0.01,
                "theta_rmse": 0.08,
                "lambda_abs_error": 0.05,
                "leaf_accuracy_test": 0.98,
            },
        },
    }


def _ctree_payload(
    *,
    train_docs: int,
    cal_rate: float,
    eval_leaf: float,
    eval_internal: float,
    leaf_tokens: int,
    seed: int,
) -> dict:
    base = 0.42
    train_gain = 0.08 if train_docs >= 256 else 0.0
    cal_gain = 0.05 * cal_rate
    leaf_gain = 0.08 * eval_leaf
    internal_gain = 0.10 * eval_internal
    granularity_penalty = 0.06 if leaf_tokens <= 8 and (cal_rate + eval_leaf + eval_internal) <= 1.1 else 0.0
    unc = base - train_gain - 0.04 * cal_rate + 0.01 * seed + granularity_penalty
    cal = unc - 0.02 * cal_rate
    budgeted = base - train_gain - cal_gain - leaf_gain - internal_gain + 0.01 * seed + granularity_penalty
    total = max(0.0, budgeted + 0.01)
    return {
        "config": {
            "topic_process": "segments",
            "n_books_train": int(train_docs),
            "n_books_test": 64,
            "min_segments": 8,
            "max_segments": 12,
            "min_seg_tokens": 16,
            "max_seg_tokens": 24,
            "fixed_leaf_tokens": int(leaf_tokens),
            "calibration_leaf_query_rate": float(cal_rate),
            "eval_leaf_query_rate": float(eval_leaf),
            "eval_internal_query_rate": float(eval_internal),
            "leaf_theta_estimator": "lstsq",
            "topic_phi_estimator": "spectral_numpy",
            "seed": int(seed),
        },
        "objective": {
            "name": "segmented_lda_ctreepo_benchmark",
            "kind": "discrepancy_benchmark",
            "optimized_against": "ridge_calibration_on_queried_leaves",
            "weighting_scheme": "benchmark_metrics_only",
            "selection_metric_name": "",
            "interprets_lambda_as": "not_applicable",
            "component_weights": {},
            "metadata": {"benchmark_metric_name": "root_l1_mean"},
        },
        "topic_meta": {
            "corpus_signature_test": "ctree-test-fixed",
        },
        "calibration_samples": 64,
        "metrics": {
            "oracle_tree": {
                "root_l1_mean": 0.0,
                "c1_violation_rate": 0.0,
                "c3_violation_rate": 0.0,
                "mean_leaf_queries": 0.0,
                "mean_internal_queries": 0.0,
            },
            "estimated_uncalibrated": {
                "root_l1_mean": float(unc),
                "c1_violation_rate": 0.22,
                "c3_violation_rate": 0.24,
                "mean_leaf_queries": 0.0,
                "mean_internal_queries": 0.0,
            },
            "estimated_calibrated": {
                "root_l1_mean": float(cal),
                "c1_violation_rate": 0.20,
                "c3_violation_rate": 0.22,
                "mean_leaf_queries": 0.0,
                "mean_internal_queries": 0.0,
            },
            "estimated_calibrated_budgeted": {
                "root_l1_mean": float(budgeted),
                "c1_violation_rate": max(0.01, float(budgeted * 0.5)),
                "c3_violation_rate": max(0.01, float(budgeted * 0.55)),
                "mean_leaf_queries": float(32 * eval_leaf),
                "mean_internal_queries": float(16 * eval_internal),
            },
        },
        "decomposition": {
            "total_root_l1_mean": float(total),
            "upper_bound_mean": float(total + 0.03),
            "slack_mean": 0.03,
        },
    }


def _mergeable_chunk_quality_payload() -> dict:
    rows = [
        {
            "method_name": "one_pass_reference",
            "chunk_budget": 6,
            "fixed_chunk_size": 32,
            "mean_abs_bias": 0.04,
            "supports_target": True,
        },
        {
            "method_name": "perfect_token_leaves_all",
            "chunk_budget": 32,
            "fixed_chunk_size": 1,
            "mean_abs_bias": 0.0,
            "supports_target": True,
        },
        {
            "method_name": "grid_fixed_s1_b1",
            "chunk_budget": 1,
            "fixed_chunk_size": 1,
            "mean_abs_bias": 0.22,
            "supports_target": True,
        },
        {
            "method_name": "grid_fixed_s2_b1",
            "chunk_budget": 1,
            "fixed_chunk_size": 2,
            "mean_abs_bias": 0.12,
            "supports_target": True,
        },
        {
            "method_name": "grid_fixed_s1_b6",
            "chunk_budget": 6,
            "fixed_chunk_size": 1,
            "mean_abs_bias": 0.06,
            "supports_target": True,
        },
        {
            "method_name": "grid_fixed_s2_b6",
            "chunk_budget": 6,
            "fixed_chunk_size": 2,
            "mean_abs_bias": 0.045,
            "supports_target": True,
        },
    ]
    return {
        "distribution": {"n_tokens": 32},
        "objective": {
            "name": "generic_k_recovery_target",
            "kind": "mergeable_target",
            "optimized_against": "probability_of_at_least_k_spikes",
            "weighting_scheme": "direct_target_supervision",
            "selection_metric_name": "",
            "interprets_lambda_as": "not_applicable",
            "component_weights": {"indicator_ge_5_spikes": 1.0},
            "metadata": {"target_kind": "probability_of_at_least_k_spikes"},
        },
        "target_k": 5,
        "sketch_order": 5,
        "chunker": "fixed",
        "selector": "top-proxy",
        "chunk_sizes": [1, 2],
        "chunk_budgets": [1, 6],
        "rows": rows,
        "reference_rows": {
            "one_pass_reference": rows[0],
            "perfect_token_leaves_all": rows[1],
        },
    }


def _mergeable_k_phase_payload() -> dict:
    return {
        "distribution": {"n_tokens": 32},
        "objective": {
            "name": "generic_k_recovery_target_family",
            "kind": "mergeable_target",
            "optimized_against": "probability_of_at_least_k_spikes",
            "weighting_scheme": "direct_target_supervision",
            "selection_metric_name": "",
            "interprets_lambda_as": "not_applicable",
            "component_weights": {
                "indicator_ge_2_spikes": 1.0,
                "indicator_ge_3_spikes": 1.0,
            },
            "metadata": {"target_kind": "probability_of_at_least_k_spikes"},
        },
        "target_ks": [2, 3],
        "sketch_orders": [1, 2, 3, 4],
        "rows": [
            {"method_name": "full_model_m1", "target_k": 2, "sketch_order": 1, "mean_abs_bias": 0.22, "supports_target": False},
            {"method_name": "full_model_m2", "target_k": 2, "sketch_order": 2, "mean_abs_bias": 0.05, "supports_target": True},
            {"method_name": "full_model_m3", "target_k": 2, "sketch_order": 3, "mean_abs_bias": 0.06, "supports_target": True},
            {"method_name": "full_model_m2", "target_k": 3, "sketch_order": 2, "mean_abs_bias": 0.25, "supports_target": False},
            {"method_name": "full_model_m3", "target_k": 3, "sketch_order": 3, "mean_abs_bias": 0.06, "supports_target": True},
            {"method_name": "full_model_m4", "target_k": 3, "sketch_order": 4, "mean_abs_bias": 0.07, "supports_target": True},
            {"method_name": "naive_majority", "target_k": 2, "sketch_order": 1, "mean_abs_bias": 0.40, "supports_target": False},
            {"method_name": "naive_mean_of_means", "target_k": 2, "sketch_order": 1, "mean_abs_bias": 0.55, "supports_target": False},
        ],
        "budget_values": [1, 6],
        "budget_target_k": 3,
        "budget_rows": [
            {"method_name": "budget_one_pass_reference", "target_k": 3, "chunk_budget": 6, "mean_abs_bias": 0.04},
            {"method_name": "budget_full_model_b1", "target_k": 3, "chunk_budget": 1, "mean_abs_bias": 0.20},
            {"method_name": "budget_full_model_b6", "target_k": 3, "chunk_budget": 6, "mean_abs_bias": 0.05},
            {"method_name": "budget_wrong_chunker_b1", "target_k": 3, "chunk_budget": 1, "mean_abs_bias": 0.32},
            {"method_name": "budget_wrong_chunker_b6", "target_k": 3, "chunk_budget": 6, "mean_abs_bias": 0.24},
        ],
    }


def _mergeable_complexity_payload() -> dict:
    return {
        "config": {"n_tokens": 32},
        "objective": {
            "name": "mergeable_complexity_ladder_target_family",
            "kind": "mergeable_target",
            "optimized_against": "stagewise_parameter_vector_recovery",
            "weighting_scheme": "direct_target_supervision",
            "selection_metric_name": "",
            "interprets_lambda_as": "not_applicable",
            "component_weights": {
                "p_spike": 1.0,
                "p_two_given_spike": 1.0,
                "p_boundary_given_spike": 1.0,
                "p_three_given_spike": 1.0,
            },
            "metadata": {"target_kind": "parameter_vector"},
        },
        "stage_order": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        "stage_metrics": {
            "one_pass_oracle": {"stage1": 0.03, "stage2": 0.04, "stage3": 0.04, "stage4": 0.04, "stage5": 0.05},
            "full_model_aligned": {"stage1": 0.035, "stage2": 0.045, "stage3": 0.046, "stage4": 0.05, "stage5": 0.055},
            "right_rule_wrong_chunker": {"stage1": 0.31, "stage2": 0.34, "stage3": 0.37, "stage4": 0.39, "stage5": 0.41},
            "naive_majority_same_chunker": {"stage1": 0.45, "stage2": 0.49, "stage3": 0.53, "stage4": 0.59, "stage5": 0.62},
            "full_model_missing_boundary_stat": {"stage4": 0.21},
            "full_model_missing_three_stat": {"stage4": 0.16},
        },
        "stage_rows": {
            "stage4": {
                "full_model_aligned": {
                    "mean_abs_bias_p_spike": 0.03,
                    "mean_abs_bias_p_two_given_spike": 0.04,
                    "mean_abs_bias_p_three_given_spike": 0.05,
                    "mean_abs_bias_p_boundary_given_spike": 0.04,
                },
                "full_model_missing_boundary_stat": {
                    "mean_abs_bias_p_spike": 0.04,
                    "mean_abs_bias_p_two_given_spike": 0.05,
                    "mean_abs_bias_p_three_given_spike": 0.05,
                    "mean_abs_bias_p_boundary_given_spike": 0.45,
                },
                "full_model_missing_three_stat": {
                    "mean_abs_bias_p_spike": 0.03,
                    "mean_abs_bias_p_two_given_spike": 0.04,
                    "mean_abs_bias_p_three_given_spike": 0.30,
                    "mean_abs_bias_p_boundary_given_spike": 0.05,
                },
            }
        },
    }


def _build_fixture_tree(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    for seed in (0, 1):
        for leaf_tokens, learned_low, learned_high in ((8, 0.38, 0.20), (16, 0.28, 0.16)):
            for train_docs in (100, 200):
                for audit_fraction, budget_gain in ((0.25, 0.12), (1.0, 0.0)):
                    root_val = learned_low if audit_fraction < 1.0 else learned_high
                    root_val -= 0.08 if train_docs >= 200 else 0.0
                    merge_val = root_val + 0.08
                    payload = _markov_payload(
                        train_docs=train_docs,
                        audit_fraction=audit_fraction,
                        leaf_tokens=leaf_tokens,
                        seed=seed,
                        learned_root=root_val + 0.01 * seed,
                        learned_merge=merge_val + 0.01 * seed,
                        unders_root=0.40 + 0.005 * budget_gain,
                        unders_merge=0.36 + 0.005 * budget_gain,
                    )
                    _write_json(
                        root / "markov" / f"markov_leaf{leaf_tokens}_td{train_docs}_a{audit_fraction}_s{seed}.json",
                        payload,
                    )

    for seed in (0, 1):
        for train_docs in (100, 200):
            for audit_fraction in (0.25, 1.0):
                _write_json(
                    root / "segment" / f"segment_sensitive_td{train_docs}_a{audit_fraction}_s{seed}.json",
                    _segment_payload(
                        train_docs=train_docs,
                        audit_fraction=audit_fraction,
                        seed=seed,
                        topic_process="segments",
                        lambda_multiplier=1.0,
                        ridge_root=0.38 - 0.12 * audit_fraction - (0.10 if train_docs >= 200 else 0.0) + 0.01 * seed,
                        ridge_merge=0.42 - 0.12 * audit_fraction - (0.11 if train_docs >= 200 else 0.0) + 0.01 * seed,
                        unders_root=0.44,
                    ),
                )
                _write_json(
                    root / "segment" / f"segment_control_td{train_docs}_a{audit_fraction}_s{seed}.json",
                    _segment_payload(
                        train_docs=train_docs,
                        audit_fraction=audit_fraction,
                        seed=seed,
                        topic_process="bag_of_words",
                        lambda_multiplier=0.0,
                        ridge_root=0.22 - 0.03 * audit_fraction - (0.04 if train_docs >= 200 else 0.0) + 0.005 * seed,
                        ridge_merge=0.24 - 0.03 * audit_fraction - (0.04 if train_docs >= 200 else 0.0) + 0.005 * seed,
                        unders_root=0.01,
                    ),
                )

    for seed in (0, 1):
        for leaf_tokens in (8, 16):
            for train_docs in (128, 256):
                for cal_rate in (0.1, 0.5):
                    for eval_leaf in (0.0, 1.0):
                        for eval_internal in (0.0, 1.0):
                            _write_json(
                                root / "ctree" / f"ctree_leaf{leaf_tokens}_td{train_docs}_c{cal_rate}_l{eval_leaf}_i{eval_internal}_s{seed}.json",
                                _ctree_payload(
                                    train_docs=train_docs,
                                    cal_rate=cal_rate,
                                    eval_leaf=eval_leaf,
                                    eval_internal=eval_internal,
                                    leaf_tokens=leaf_tokens,
                                    seed=seed,
                                ),
                            )

    _write_json(root / "mergeable" / "mergeable_chunk_quality.json", _mergeable_chunk_quality_payload())
    _write_json(root / "mergeable" / "mergeable_k_phase.json", _mergeable_k_phase_payload())
    _write_json(root / "mergeable" / "mergeable_complexity.json", _mergeable_complexity_payload())
    return root


def test_assess_trend_pass_warn_fail() -> None:
    cfg = ExpectationConfig()
    passing = assess_trend(
        _trend_rows([(100, 0.40), (200, 0.30), (300, 0.29), (400, 0.18)]),
        axis_name="train_docs",
        direction="decreasing",
        config=cfg,
    )
    assert passing.status == "pass"

    warning = assess_trend(
        _trend_rows([(100, 0.40), (200, 0.28), (300, 0.31), (400, 0.18)]),
        axis_name="train_docs",
        direction="decreasing",
        config=cfg,
    )
    assert warning.status == "warn"

    failing = assess_trend(
        _trend_rows([(100, 0.20), (200, 0.21), (300, 0.22)]),
        axis_name="train_docs",
        direction="decreasing",
        config=cfg,
    )
    assert failing.status == "fail"


def test_family_adapters_load_rows(tmp_path: Path) -> None:
    root = _build_fixture_tree(tmp_path)
    adapter_cases = [
        (MarkovOPSAdapter(), next((root / "markov").glob("*.json"))),
        (SegmentLDAOPSAdapter(), next((root / "segment").glob("*.json"))),
        (SegmentedLDACtreePOAdapter(), next((root / "ctree").glob("*.json"))),
        (MergeableAblationAdapter(), root / "mergeable" / "mergeable_chunk_quality.json"),
    ]
    for adapter, path in adapter_cases:
        assert adapter.can_load(path)
        rows = adapter.load_rows(path)
        assert rows
        assert all(r.family == adapter.family for r in rows)


def test_family_adapters_preserve_objective_semantics(tmp_path: Path) -> None:
    root = _build_fixture_tree(tmp_path)
    adapter_cases = [
        (
            MarkovOPSAdapter(),
            next((root / "markov").glob("*.json")),
            "local_law_weighted_objective",
            "optimization_tradeoff",
        ),
        (
            SegmentLDAOPSAdapter(),
            next((root / "segment").glob("*.json")),
            "latent_oracle_target",
            "dgp_term_multiplier",
        ),
        (
            SegmentedLDACtreePOAdapter(),
            next((root / "ctree").glob("*.json")),
            "discrepancy_benchmark",
            "not_applicable",
        ),
        (
            MergeableAblationAdapter(),
            root / "mergeable" / "mergeable_chunk_quality.json",
            "mergeable_target",
            "not_applicable",
        ),
    ]
    for adapter, path, expected_kind, expected_lambda_role in adapter_cases:
        rows = adapter.load_rows(path)
        assert rows
        assert all(row.metadata.get("objective_kind") == expected_kind for row in rows)
        assert all(
            row.metadata.get("objective_interprets_lambda_as") == expected_lambda_role
            for row in rows
        )


def test_markov_adapter_splits_feature_modes_into_distinct_scenarios(tmp_path: Path) -> None:
    root = tmp_path / "markov"
    payload_full = _markov_payload(
        train_docs=100,
        audit_fraction=0.25,
        leaf_tokens=16,
        seed=0,
        learned_root=0.3,
        learned_merge=0.35,
        unders_root=0.4,
        unders_merge=0.45,
    )
    payload_no_endpoints = json.loads(json.dumps(payload_full))
    payload_no_endpoints["config"]["feature_mode"] = "no_endpoints"
    _write_json(root / "full.json", payload_full)
    _write_json(root / "no_endpoints.json", payload_no_endpoints)

    adapter = MarkovOPSAdapter()
    rows = [*adapter.load_rows(root / "full.json"), *adapter.load_rows(root / "no_endpoints.json")]
    scenarios = {row.scenario for row in rows}
    assert len(scenarios) == 2


def test_markov_adapter_splits_local_law_weights_into_distinct_scenarios(tmp_path: Path) -> None:
    root = tmp_path / "markov"
    payload_low = _markov_payload(
        train_docs=100,
        audit_fraction=0.25,
        leaf_tokens=16,
        seed=0,
        learned_root=0.3,
        learned_merge=0.35,
        unders_root=0.4,
        unders_merge=0.45,
    )
    payload_high = json.loads(json.dumps(payload_low))
    payload_high["config"]["local_law_weight"] = 0.8
    payload_high["objective"]["local_law_weight"] = 0.8
    _write_json(root / "low.json", payload_low)
    _write_json(root / "high.json", payload_high)

    adapter = MarkovOPSAdapter()
    rows = [*adapter.load_rows(root / "low.json"), *adapter.load_rows(root / "high.json")]
    scenarios = {row.scenario for row in rows}

    assert len(scenarios) == 2


def test_budget_trend_warn_only_downgrades_failures() -> None:
    rows = _trend_rows([(100.0, 1.0), (200.0, 1.5)], test_identity="shared")
    finding = BudgetTrendExpectation(
        family="demo",
        scenario="demo",
        method="merge",
        metric="merge_mae",
        axis_name="train_docs",
        direction="decreasing",
        title="demo",
        warn_only=True,
    ).evaluate(rows, config=ExpectationConfig())
    assert finding.status == "warn"


def test_segment_adapter_boundary_sensitivity_follows_lambda_not_segment_process(tmp_path: Path) -> None:
    root = tmp_path / "segment"
    sensitive = _segment_payload(
        train_docs=100,
        audit_fraction=0.25,
        seed=0,
        topic_process="segments",
        lambda_multiplier=1.0,
        ridge_root=0.12,
        ridge_merge=0.15,
        unders_root=0.30,
    )
    control = _segment_payload(
        train_docs=100,
        audit_fraction=0.25,
        seed=0,
        topic_process="segments",
        lambda_multiplier=0.0,
        ridge_root=0.12,
        ridge_merge=0.15,
        unders_root=0.0,
    )
    sensitive_path = _write_json(root / "sensitive.json", sensitive)
    control_path = _write_json(root / "control.json", control)

    adapter = SegmentLDAOPSAdapter()
    sensitive_rows = adapter.load_rows(sensitive_path)
    control_rows = adapter.load_rows(control_path)

    assert sensitive_rows
    assert control_rows
    assert all(bool(row.metadata["boundary_sensitive"]) for row in sensitive_rows)
    assert all(not bool(row.metadata["boundary_sensitive"]) for row in control_rows)
    assert {str(row.metadata["control_group"]) for row in sensitive_rows} == {
        str(row.metadata["control_group"]) for row in control_rows
    }


def test_segment_adapter_derives_shared_test_identity_across_train_docs(tmp_path: Path) -> None:
    root = tmp_path / "segment"
    payload_small = _segment_payload(
        train_docs=16,
        audit_fraction=1.0,
        seed=0,
        topic_process="segments",
        lambda_multiplier=2.0,
        ridge_root=0.1,
        ridge_merge=0.12,
        unders_root=0.3,
    )
    payload_large = _segment_payload(
        train_docs=128,
        audit_fraction=1.0,
        seed=0,
        topic_process="segments",
        lambda_multiplier=2.0,
        ridge_root=0.05,
        ridge_merge=0.08,
        unders_root=0.3,
    )
    small_path = _write_json(root / "small.json", payload_small)
    large_path = _write_json(root / "large.json", payload_large)

    adapter = SegmentLDAOPSAdapter()
    small_rows = adapter.load_rows(small_path)
    large_rows = adapter.load_rows(large_path)

    assert small_rows
    assert large_rows
    assert {row.test_identity for row in small_rows} == {row.test_identity for row in large_rows}


def test_mergeable_k_phase_missing_unsupported_is_warn_not_fail(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    payload = _mergeable_k_phase_payload()
    payload["target_ks"] = [2, 3]
    payload["rows"] = [row for row in payload["rows"] if not (row["target_k"] == 2 and row["sketch_order"] < 2)]
    _write_json(root / "mergeable" / "mergeable_k_phase.json", payload)

    report = build_expectation_report(output_root=root, config=ExpectationConfig())
    finding = next(
        f
        for f in report.expectations
        if f.title == "Mergeable k-vs-m phase: insufficient sketch order (m<k) is materially worse than exact support"
    )
    assert finding.status == "warn"


def test_build_expectation_report_output_root_and_manifest(tmp_path: Path) -> None:
    root = _build_fixture_tree(tmp_path)
    report = build_expectation_report(output_root=root, config=ExpectationConfig())
    assert set(report.families_scanned) == set(VALID_FAMILIES)
    assert int(report.summary["n_fail"]) == 0
    assert report.rows_scanned > 0
    titles = {f.title: f.status for f in report.expectations}
    assert titles["Markov high-support anchor: learned root_mae beats undersupported"] == "pass"
    assert titles["Segment-LDA boundary-sensitive regime: undersupported stays separated from exact"] == "pass"
    assert titles["Segmented-LDA decomposition upper bound dominates total error"] == "pass"
    assert any("Mergeable complexity ladder: full model tracks the one-pass oracle" == f.title and f.status == "pass" for f in report.expectations)

    manifest_path = tmp_path / "suite_manifest.jsonl"
    runs = []
    for path in sorted(root.rglob("*.json")):
        runs.append(
            RunSpec.create(
                family="fixture",
                config={"path": str(path)},
                outputs={"json_summary": str(path)},
                command="true",
            )
        )
    write_manifest_jsonl(manifest_path, runs)
    report_from_manifest = build_expectation_report(manifest_path=manifest_path, config=ExpectationConfig())
    assert set(report_from_manifest.families_scanned) == set(VALID_FAMILIES)
    assert int(report_from_manifest.summary["n_fail"]) == 0


def test_cli_strict_and_report_render(tmp_path: Path) -> None:
    root = _build_fixture_tree(tmp_path)
    check_script = Path("/home/mlinegar/ThinkingTrees/scripts/check_simulation_expectations.py")
    report_script = Path("/home/mlinegar/ThinkingTrees/scripts/report_simulation_expectations.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(check_script),
            "--output-root",
            str(root),
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    json_path = root / "simulation_expectations.json"
    md_path = root / "simulation_expectations.md"
    assert json_path.exists()
    assert md_path.exists()

    proc_report = subprocess.run(
        [
            sys.executable,
            str(report_script),
            "--input-json",
            str(json_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc_report.returncode == 0, proc_report.stdout + proc_report.stderr
    loaded = ExpectationReport.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
    assert int(loaded.summary["n_fail"]) == 0

    broken = _markov_payload(
        train_docs=100,
        audit_fraction=1.0,
        leaf_tokens=16,
        seed=0,
        learned_root=0.2,
        learned_merge=0.25,
        unders_root=0.4,
        unders_merge=0.35,
    )
    broken["metrics"]["exact"]["root_mae"] = 0.1
    broken_root = tmp_path / "broken"
    _write_json(broken_root / "markov_bad.json", broken)
    proc_fail = subprocess.run(
        [
            sys.executable,
            str(check_script),
            "--output-root",
            str(broken_root),
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc_fail.returncode == 1, proc_fail.stdout + proc_fail.stderr
