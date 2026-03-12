from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from src.ctreepo.sim.core.leaf_local_mixture_utility import (
    LeafLocalMixtureUtilityConfig,
    _local_law_objective_spec,
    _select_local_law_candidate,
    run_leaf_local_mixture_utility_experiment,
    run_leaf_local_mixture_utility_experiment_from_world,
    sample_leaf_local_mixture_utility_world,
)


def _law_cfg(**overrides) -> LeafLocalMixtureUtilityConfig:
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
        local_mixture_concentration=8.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=False,
        lambda_multiplier=1.5,
        train_docs=48,
        test_docs=48,
        budget_regime="all_leaves_labeled",
        leaf_label_budget=8.0,
        ridge_alpha=1e-3,
        query_design="uniform",
        doc_sample_rate=1.0,
        heldout_doc_sample_rate=1.0,
        target_query_budget_per_doc=1.0,
        propensity_floor=0.10,
        propensity_ceiling=0.90,
        propensity_proxy="l1_deviation",
        ipw_stabilized_clip=20.0,
        ipw_delta=0.05,
        local_law_mode="diagnostics_and_learned",
        law_leaf_query_rate=0.10,
        law_internal_query_rate=0.10,
        law_leaf_query_design="uniform",
        law_internal_query_design="uniform",
        law_task_objective_weight=1.0,
        law_c1_weight=1.0 / 3.0,
        law_c3_weight=1.0 / 3.0,
        law_c2_proxy_weight=1.0 / 3.0,
        law_calibration_ridge=1e-3,
        law_eval_leaf_sample_rate=1.0,
        law_eval_internal_sample_rate=1.0,
        law_c1_threshold=0.20,
        law_c3_threshold=0.20,
        law_c2_threshold=0.20,
        inference_prior_mass=0.25,
        inference_max_iter=200,
        inference_tol=1e-9,
        seed=0,
    )
    base.update(overrides)
    return LeafLocalMixtureUtilityConfig(**base)


def test_oracle_local_laws_are_zero_in_exact_controls():
    cfg = _law_cfg(local_law_mode="diagnostics", analysis_partition_mode="shift_half", seed=5)
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    oracle = dict(summary.local_law["policy_metrics"]["oracle_true_summary"])
    assert float(oracle["mean_c1"]) == pytest.approx(0.0, abs=1e-10)
    assert float(oracle["mean_c3"]) == pytest.approx(0.0, abs=1e-10)
    assert float(oracle["mean_c2_proxy"]) < 0.08


def test_local_law_metrics_are_lambda_free_on_a_fixed_world():
    cfg0 = _law_cfg(local_law_mode="diagnostics", lambda_multiplier=0.0, seed=7)
    cfg1 = _law_cfg(local_law_mode="diagnostics", lambda_multiplier=3.0, seed=7)
    world = sample_leaf_local_mixture_utility_world(cfg0)
    summary0 = run_leaf_local_mixture_utility_experiment_from_world(cfg0, world)
    summary1 = run_leaf_local_mixture_utility_experiment_from_world(cfg1, world)
    m0 = dict(summary0.local_law["policy_metrics"]["infer_identity"])
    m1 = dict(summary1.local_law["policy_metrics"]["infer_identity"])
    assert float(m0["mean_c1"]) == pytest.approx(float(m1["mean_c1"]), abs=1e-10)
    assert float(m0["mean_c3"]) == pytest.approx(float(m1["mean_c3"]), abs=1e-10)
    assert float(m0["mean_c2_proxy"]) == pytest.approx(float(m1["mean_c2_proxy"]), abs=1e-10)


def test_lambda_zero_produces_near_zero_delta_for_identity():
    """At lambda=0, the within-framework delta (baseline_aux_err -
    policy_aux_err) should be exactly zero for the identity baseline itself.
    The oracle_true_summary can still show a large delta (true topics always
    improve aux predictions). Learned calibrators may show non-trivial deltas
    even at lambda=0 since calibration can improve topic estimation quality
    for the linear utility term."""
    cfg = _law_cfg(
        lambda_multiplier=0.0,
        local_law_mode="diagnostics_and_learned",
        train_docs=64,
        test_docs=64,
        seed=42,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    identity_delta = float(
        summary.local_law["policy_metrics"]["infer_identity"]["mean_aux_oracle_target_delta"]
    )
    # Identity vs itself: delta must be exactly zero
    assert abs(identity_delta) < 1e-12, f"infer_identity delta={identity_delta}, expected 0"


def test_shift_and_random_boundaries_raise_c3_relative_to_aligned_on_fixed_world():
    base_cfg = _law_cfg(local_law_mode="diagnostics", seed=11)
    world = sample_leaf_local_mixture_utility_world(base_cfg)
    aligned = run_leaf_local_mixture_utility_experiment_from_world(
        _law_cfg(local_law_mode="diagnostics", analysis_partition_mode="aligned", seed=11),
        world,
    )
    shifted = run_leaf_local_mixture_utility_experiment_from_world(
        _law_cfg(local_law_mode="diagnostics", analysis_partition_mode="shift_half", seed=11),
        world,
    )
    random_same = run_leaf_local_mixture_utility_experiment_from_world(
        _law_cfg(
            local_law_mode="diagnostics", analysis_partition_mode="random_same_count", seed=11
        ),
        world,
    )
    aligned_c3 = float(aligned.local_law["policy_metrics"]["infer_identity"]["mean_c3"])
    shifted_c3 = float(shifted.local_law["policy_metrics"]["infer_identity"]["mean_c3"])
    random_c3 = float(random_same.local_law["policy_metrics"]["infer_identity"]["mean_c3"])
    assert shifted_c3 >= aligned_c3
    assert random_c3 >= aligned_c3


def test_full_sample_ht_and_hajek_recover_law_population_means():
    cfg = _law_cfg(
        heldout_doc_sample_rate=1.0,
        law_eval_leaf_sample_rate=1.0,
        law_eval_internal_sample_rate=1.0,
        seed=13,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    eval_identity = dict(summary.local_law["ipw_evaluation"]["infer_identity"])
    for key in ("c1", "c2_proxy", "c3", "combined"):
        item = dict(eval_identity[key])
        assert float(item["ht_abs_error"]) == pytest.approx(0.0, abs=1e-8)
        assert float(item["hajek_abs_error"]) == pytest.approx(0.0, abs=1e-8)


def test_stabilized_law_calibrator_improves_heldout_c1_on_nontrivial_world():
    cfg = _law_cfg(
        analysis_partition_mode="aligned",
        train_docs=128,
        test_docs=128,
        law_leaf_query_rate=0.20,
        law_internal_query_rate=0.20,
        law_leaf_query_design="uniform",
        law_internal_query_design="uniform",
        seed=19,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    identity_c1 = float(summary.local_law["policy_metrics"]["infer_identity"]["mean_c1"])
    stabilized_c1 = float(
        summary.local_law["policy_metrics"]["law_calibrated_ipw_stabilized"]["mean_c1"]
    )
    assert stabilized_c1 < identity_c1


def test_local_law_json_fields_are_present_and_additive():
    cfg = _law_cfg(seed=19)
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    payload = json.loads(summary.to_json())
    assert "stage3" in payload
    assert "local_law" in payload
    assert "local_law_learnability" in payload
    local_law = dict(payload["local_law"])
    for key in (
        "config",
        "exact_metrics",
        "violation_rates",
        "policy_metrics",
        "ipw_evaluation",
        "mediation",
    ):
        assert key in local_law


def test_local_law_validation_selection_and_artifacts_are_serialized(tmp_path: Path):
    cfg = _law_cfg(
        train_docs=32,
        val_docs=16,
        test_docs=24,
        artifact_dir=str(tmp_path / "artifacts"),
        seed=29,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    payload = json.loads(summary.to_json())
    learnability = dict(payload["local_law_learnability"])
    selection = dict(learnability["selection"])
    assert selection["selection_split"] == "val"
    assert selection["test_metrics_used_for_selection"] is False
    assert selection["selection_metric"] == "configured_objective_hajek"
    assert str(selection["selected_candidate"]) in {
        "law_calibrated_naive",
        "law_calibrated_ipw",
        "law_calibrated_ipw_stabilized",
    }

    split_metrics = dict(payload["local_law"]["split_policy_metrics"])
    assert "val" in split_metrics
    assert "test" in split_metrics
    g_artifacts = dict(payload["g_artifacts"])
    expected_ids = {
        "oracle_g",
        "baseline_g",
        "candidate_law_calibrated_naive",
        "candidate_law_calibrated_ipw",
        "candidate_law_calibrated_ipw_stabilized",
        "learned_g",
    }
    assert expected_ids <= set(g_artifacts.keys())
    for artifact_id in expected_ids:
        manifest_path = Path(g_artifacts[artifact_id]["manifest_path"])
        assert manifest_path.exists()


def test_local_law_payload_emits_shared_configured_objective_wrapper() -> None:
    cfg = _law_cfg(
        train_docs=32,
        val_docs=16,
        test_docs=24,
        law_task_objective_weight=1.5,
        law_c1_weight=0.5,
        law_c3_weight=1.25,
        law_c2_proxy_weight=0.1,
        seed=31,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    payload = json.loads(summary.to_json())
    local_law = dict(payload["local_law"])
    objective = dict(local_law["objective"])
    learnability = dict(payload["local_law_learnability"])
    metadata = dict(learnability["metadata"])

    assert objective["selection_metric_name"] == "configured_objective_hajek"
    assert "hajek" in list(objective["available_estimators"])
    assert float(objective["task_weight"]) == pytest.approx(1.5)
    assert float(objective["local_law_weights"]["c1"]) == pytest.approx(0.5)
    assert float(objective["local_law_weights"]["c2_proxy"]) == pytest.approx(0.1)
    assert float(objective["local_law_weights"]["c3"]) == pytest.approx(1.25)
    assert float(objective["normalized_task_share"]) == pytest.approx(1.5 / 3.35)
    assert float(objective["normalized_local_law_share"]) == pytest.approx(1.85 / 3.35)
    assert metadata["law_package"] == "all_laws"
    assert float(metadata["resolved_local_law_weights"]["c1"]) == pytest.approx(0.5)
    assert float(metadata["resolved_local_law_weights"]["c2_proxy"]) == pytest.approx(0.1)
    assert float(metadata["resolved_local_law_weights"]["c3"]) == pytest.approx(1.25)

    infer_identity = dict(local_law["policy_metrics"]["infer_identity"])
    expected_total = (
        1.5 * float(infer_identity["mean_aux_oracle_target_abs_error"])
        + 0.5 * float(infer_identity["mean_c1"])
        + 0.1 * float(infer_identity["mean_c2_proxy"])
        + 1.25 * float(infer_identity["mean_c3"])
    )
    expected_combined = (
        0.5 * float(infer_identity["mean_c1"])
        + 0.1 * float(infer_identity["mean_c2_proxy"])
        + 1.25 * float(infer_identity["mean_c3"])
    )
    assert float(infer_identity["combined_law_score"]) == pytest.approx(expected_combined)
    assert float(infer_identity["configured_objective"]) == pytest.approx(expected_total)
    assert float(infer_identity["configured_objective_task_term"]) == pytest.approx(
        1.5 * float(infer_identity["mean_aux_oracle_target_abs_error"])
    )
    assert float(infer_identity["configured_objective_c1_term"]) == pytest.approx(
        0.5 * float(infer_identity["mean_c1"])
    )
    assert float(infer_identity["configured_objective_c2_proxy_term"]) == pytest.approx(
        0.1 * float(infer_identity["mean_c2_proxy"])
    )
    assert float(infer_identity["configured_objective_c3_term"]) == pytest.approx(
        1.25 * float(infer_identity["mean_c3"])
    )
    assert math.isfinite(float(infer_identity["configured_objective_hajek"]))


def test_local_law_candidate_selection_recomputes_configured_objective_from_raw_metrics() -> None:
    spec = _local_law_objective_spec(
        _law_cfg(law_task_objective_weight=2.0, law_c1_weight=1.0, law_c3_weight=1.0)
    )
    choice = _select_local_law_candidate(
        {
            "law_calibrated_ipw_stabilized": {
                "configured_objective": 0.05,
                "mean_aux_oracle_target_abs_error": 0.30,
                "mean_c1": 0.05,
                "mean_c2_proxy": 0.0,
                "mean_c3": 0.05,
            },
            "law_calibrated_ipw": {
                "configured_objective": 9.0,
                "mean_aux_oracle_target_abs_error": 0.10,
                "mean_c1": 0.30,
                "mean_c2_proxy": 0.0,
                "mean_c3": 0.05,
            },
            "law_calibrated_naive": {
                "configured_objective": 0.01,
                "mean_aux_oracle_target_abs_error": 0.12,
                "mean_c1": 0.20,
                "mean_c2_proxy": 0.0,
                "mean_c3": 0.30,
            },
        },
        objective_spec=spec,
    )
    assert choice == "law_calibrated_ipw"


def test_local_law_objective_wrapper_respects_law_package_mask() -> None:
    cfg = _law_cfg(
        train_docs=32,
        val_docs=16,
        test_docs=24,
        law_package="root_only",
        law_task_objective_weight=2.0,
        law_c1_weight=0.9,
        law_c3_weight=0.7,
        law_c2_proxy_weight=0.4,
        seed=37,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    payload = json.loads(summary.to_json())
    local_law = dict(payload["local_law"])
    local_law_config = dict(local_law["config"])
    objective = dict(local_law["objective"])
    infer_identity = dict(local_law["policy_metrics"]["infer_identity"])
    learnability = dict(payload["local_law_learnability"])
    metadata = dict(learnability["metadata"])

    assert local_law_config["law_package"] == "root_only"
    assert float(objective["local_law_weights"]["c1"]) == pytest.approx(0.0)
    assert float(objective["local_law_weights"]["c2_proxy"]) == pytest.approx(0.0)
    assert float(objective["local_law_weights"]["c3"]) == pytest.approx(0.0)
    assert float(objective["local_law_weight_total"]) == pytest.approx(0.0)
    assert metadata["law_package"] == "root_only"
    assert float(metadata["resolved_local_law_weights"]["c1"]) == pytest.approx(0.0)
    assert float(metadata["resolved_local_law_weights"]["c2_proxy"]) == pytest.approx(0.0)
    assert float(metadata["resolved_local_law_weights"]["c3"]) == pytest.approx(0.0)
    assert float(infer_identity["combined_law_score"]) == pytest.approx(0.0)
    assert float(infer_identity["configured_objective"]) == pytest.approx(
        2.0 * float(infer_identity["mean_aux_oracle_target_abs_error"])
    )


def test_local_law_report_smoke_with_minimal_fixture(tmp_path: Path):
    fixture_root = tmp_path / "local_law_fixture"
    results = fixture_root / "results"
    results.mkdir(parents=True)
    cfg = _law_cfg(
        analysis_partition_mode="shift_half",
        law_leaf_query_design="proxy_priority",
        law_internal_query_design="risk",
        law_c1_weight=0.5,
        law_c2_proxy_weight=0.1,
        law_c3_weight=1.25,
        train_docs=16,
        test_docs=16,
        seed=23,
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)
    payload = json.loads(summary.to_json())
    targets = [
        results
        / "suite_a_exact_controls"
        / "mode_shift_half"
        / "tau_8"
        / "lam_1.5"
        / "seed_0.json",
        results
        / "suite_b_local_law_learnability"
        / "train_128"
        / "leafrate_0.1"
        / "internalrate_0.1"
        / "tau_8"
        / "lam_1.5"
        / "seed_0.json",
        results
        / "suite_c_mismatch_mediation"
        / "mode_shift_half"
        / "tau_8"
        / "lam_1.5"
        / "seed_0.json",
        results
        / "suite_d_ipw_sparse_labels"
        / "mode_shift_half"
        / "leafdesign_proxy_priority"
        / "internaldesign_risk"
        / "leafrate_0.1"
        / "internalrate_0.1"
        / "tau_8"
        / "lam_1.5"
        / "seed_0.json",
        results
        / "suite_e_hardness"
        / "mode_shift_half"
        / "anchor_10"
        / "topicconc_1"
        / "tau_8"
        / "lam_1.5"
        / "seed_0.json",
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    out_dir = fixture_root / "report"
    subprocess.run(
        [
            sys.executable,
            "scripts/report_tree_relevant_lda_local_law.py",
            "--input-root",
            str(fixture_root),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )
    assert (out_dir / "tree_relevant_lda_local_law_report.pdf").exists()
    assert (out_dir / "tree_relevant_lda_local_law_report.md").exists()
    assert (out_dir / "tree_relevant_lda_local_law_report_summary.json").exists()
    report_summary = json.loads(
        (out_dir / "tree_relevant_lda_local_law_report_summary.json").read_text(encoding="utf-8")
    )
    report_markdown = (out_dir / "tree_relevant_lda_local_law_report.md").read_text(
        encoding="utf-8"
    )
    assert "unified_core" in report_summary
    assert (
        report_summary["law_score_label"]
        == "Configured local-law score (0.5*C1 + 0.1*C2-proxy + 1.25*C3)"
    )
    assert report_summary["law_score_is_uniform"] is True
    profiles = list(report_summary["objective_weight_profiles"])
    assert len(profiles) == 1
    assert profiles[0]["law_package"] == "all_laws"
    assert int(profiles[0]["n_runs"]) == 5
    assert float(profiles[0]["c1"]) == pytest.approx(0.5)
    assert float(profiles[0]["c2_proxy"]) == pytest.approx(0.1)
    assert float(profiles[0]["c3"]) == pytest.approx(1.25)
    assert "Configured local-law score (0.5*C1 + 0.1*C2-proxy + 1.25*C3)" in report_markdown
    assert "0.25*C2-proxy" not in report_markdown
