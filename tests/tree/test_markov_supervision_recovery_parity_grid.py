from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.run_markov_supervision_recovery_parity_grid as mod


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path / "parity",
        benchmark="recoverable_v4",
        structural_benchmark="structural_core_v1::r12_seg10to12",
        train_doc_counts="1024",
        seed=0,
        mig_uuids=" ".join(f"MIG-{idx}" for idx in range(20)),
        recipe_ids="",
        fixed_leaf_tokens="",
        include_structural=True,
        prepared_data_root="outputs/_prepared_data/example",
        corpus_root="",
        prepared_data_allow_create=False,
        tree_exact_eval_max_docs=64,
        gpu_runtime_data_mode="resident",
        gpu_runtime_bucket_mode="exact_then_bucketed",
        gpu_runtime_preload_splits=("train", "val", "test"),
        gpu_runtime_preload_targets=True,
        batch_size=512,
        epoch_cap=0,
        exact_metric_selection_doc_limit=0,
        exact_metric_selection_interval=1,
        torch_threads=1,
        use_cuda=False,
        resume=True,
        cleanup_stale_children=True,
        max_gpu_items_per_mig=1,
        scheduler_launch_stagger_seconds=0.0,
        scheduler_min_mem_available_gib=128.0,
        scheduler_min_swap_free_gib=2.0,
        skip_fno_baselines=True,
        include_supervision_sweep=False,
        lean_faithful_diagnostic_matrix=False,
        lean_faithful_weight_balance_sweep=False,
        exact_collapse_repair_diagnostic_matrix=False,
        full_local_laws_topology_diagnostic_4096=False,
        unified_g_topology_diagnostic_4096=False,
        topology_seeds=mod.UNIFIED_G_TOPOLOGY_DEFAULT_SEEDS,
        topology_leaf_tokens=mod.UNIFIED_G_TOPOLOGY_DEFAULT_LEAF_TOKENS,
        topology_stress_leaf_tokens=mod.UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_LEAF_TOKENS,
        topology_stress_seeds=mod.UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_SEEDS,
        topology_posttrain_train_doc_limit=0,
        topology_posttrain_diagnostics_mode="",
        topology_stress_posttrain_diagnostics_mode="",
        plan_only=True,
    )

def test_build_plan_writes_claim_separated_parity_grid_files(tmp_path: Path) -> None:
    args = _args(tmp_path)

    entries = mod.build_parity_grid_entries(args)
    assert len(entries) == 19
    assert sum(1 for entry in entries if entry.scope_label == "recoverable") == 18
    assert sum(1 for entry in entries if entry.scope_label == "structural") == 1
    assert sum(
        1 for entry in entries if entry.claim_level == mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY
    ) == 18
    assert sum(
        1 for entry in entries if entry.claim_level == mod.CLAIM_LEVEL_EXACT_COLLAPSE
    ) == 1
    assert [entry.recipe_id for entry in entries[:4]] == [
        "historical_replay",
        "optimization_fairness",
        "capacity_fairness",
        "matched_root",
    ]
    assert [entry.fixed_leaf_tokens for entry in entries[4:8]] == [64, 64, 64, 64]
    assert all(entry.config.batch_size == 512 for entry in entries)

    plan = mod.build_plan(args)
    assert len(plan["items"]) == 19
    mod._write_plan_files(plan)

    manifest = json.loads(
        (args.output_root / mod.PARITY_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    status = json.loads(
        (args.output_root / mod.PARITY_STATUS_NAME).read_text(encoding="utf-8")
    )
    summary = json.loads(
        (args.output_root / mod.PARITY_SUMMARY_NAME).read_text(encoding="utf-8")
    )

    assert len(manifest["jobs"]) == 19
    assert len(manifest["requested_devices"]) == 20
    assert manifest["canonical_train_ladder"] == [1024, 4096, 10240]
    assert manifest["one_leaf_target_fixed_leaf_tokens"] == 128
    assert manifest["assumed_doc_tokens"] == 128
    assert manifest["scheduler_plan"]["min_mem_available_kib"] == 128 * 1024 * 1024
    assert manifest["scheduler_plan"]["min_swap_free_kib"] == 2 * 1024 * 1024
    assert status["items_total"] == 19
    assert status["completed_items"] == 0
    assert status["pending_items"] == 19
    assert status["rows_by_claim_level"][mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY] == 18
    assert status["rows_by_claim_level"][mod.CLAIM_LEVEL_EXACT_COLLAPSE] == 1
    assert summary["items_total"] == 19
    assert summary["canonical_train_ladder"] == [1024, 4096, 10240]
    assert summary["evidence_status"] == "partial"
    assert len(summary["rows"]) == 19
    assert all(row["state"] == "planned" for row in summary["rows"])
    representative_tree = next(
        item for item in plan["items"] if item.metadata.get("model_family") == "tree_neural"
    )
    assert representative_tree.metadata["scope"] == "recoverable_v4"
    assert representative_tree.metadata["package"] == "full100"
    assert representative_tree.metadata["worker_kind"] == "full_doc_diagnostics"
    assert representative_tree.metadata["train_docs"] in {1024, 4096, 10240}
    assert {
        row["claim_level"] for row in summary["rows"]
    } == {
        mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
        mod.CLAIM_LEVEL_EXACT_COLLAPSE,
    }
    assert (
        args.output_root / "results.jsonl"
    ).exists(), "plan materialization should still create the canonical results file"


def test_epoch_cap_caps_total_epochs_and_preserves_exact_collapse_surface(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.epoch_cap = 10
    args.exact_metric_selection_interval = 5

    entries = mod.build_parity_grid_entries(args)

    historical = next(
        entry
        for entry in entries
        if entry.recipe_id == "historical_replay" and entry.fixed_leaf_tokens == 16
    )
    assert historical.config.comparison_mode == "comparable"
    assert historical.config.n_epochs == 10
    assert historical.config.tree_stage1_epochs == 2
    assert historical.config.tree_stage2_epochs == 8
    assert historical.config.exact_metric_selection_interval == 5

    matched_root = next(
        entry
        for entry in entries
        if entry.recipe_id == "matched_root" and entry.fixed_leaf_tokens == 16
    )
    assert matched_root.config.n_epochs == 10
    assert matched_root.config.tree_stage1_epochs == 0
    assert matched_root.config.tree_stage2_epochs == 0
    assert matched_root.config.tree_local_law_weight == pytest.approx(0.8)
    assert matched_root.config.tree_task_objective_weight == pytest.approx(1.0)
    assert matched_root.config.tree_c1_relative_weight == pytest.approx(1.0)
    assert matched_root.config.tree_c2_relative_weight == pytest.approx(1.0)
    assert matched_root.config.tree_c3_relative_weight == pytest.approx(1.0)

    exact_collapse = next(
        entry
        for entry in entries
        if entry.recipe_id == mod.EXACT_COLLAPSE_RECIPE_ID
        and entry.scope_label == "recoverable"
    )
    exact_mapping = mod._config_mapping_for_run_config(exact_collapse.config)
    assert exact_collapse.config.comparison_mode == "exact_collapse"
    assert exact_mapping["n_epochs"] == 10
    assert exact_mapping["comparison_mode"] == "exact_collapse"
    assert (
        mod._config_diff_vs_official_fno(
            config_mapping=exact_mapping,
            reference_surface=exact_collapse.official_fno_reference_surface,
        )
        == {}
    )
    assert exact_mapping["tree_root_supervision_kind"] == "mse"
    assert exact_mapping["fixed_leaf_tokens"] == 128
    assert exact_mapping["tree_leaf_fno_width"] == 128
    assert exact_mapping["tree_leaf_fno_n_modes"] == 8
    assert exact_mapping["local_law_weight"] == pytest.approx(0.0)
    assert exact_mapping["task_objective_weight"] == pytest.approx(1.0)
    assert exact_mapping["c1_relative_weight"] == pytest.approx(0.0)
    assert exact_mapping["c2_relative_weight"] == pytest.approx(0.0)
    assert exact_mapping["c3_relative_weight"] == pytest.approx(0.0)


def test_build_entries_use_benchmark_specific_corpus_payloads(tmp_path: Path) -> None:
    args = _args(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "recoverable_v4",
                "benchmarks": {
                    "recoverable_v4": {
                        "bundle_paths": {"1024": "/tmp/recoverable_bundle.pkl"},
                        "prepared_data_root": "/tmp/recoverable_prepared",
                    },
                    "structural_core_v1::r12_seg10to12": {
                        "bundle_paths": {"1024": "/tmp/structural_bundle.pkl"},
                        "prepared_data_root": "/tmp/structural_prepared",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    args.corpus_root = str(corpus_root)

    entries = mod.build_parity_grid_entries(args)

    recoverable_entry = next(
        entry for entry in entries if entry.benchmark == "recoverable_v4"
    )
    structural_entry = next(
        entry
        for entry in entries
        if entry.benchmark == "structural_core_v1::r12_seg10to12"
    )
    assert recoverable_entry.config.base_bundle_path == "/tmp/recoverable_bundle.pkl"
    assert recoverable_entry.config.prepared_data_root == "/tmp/recoverable_prepared"
    assert structural_entry.config.base_bundle_path == "/tmp/structural_bundle.pkl"
    assert structural_entry.config.prepared_data_root == "/tmp/structural_prepared"


def test_build_entries_fail_fast_when_structural_corpus_is_missing(tmp_path: Path) -> None:
    args = _args(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "recoverable_v4",
                "bundle_paths": {"1024": "/tmp/recoverable_bundle.pkl"},
                "prepared_data_root": "/tmp/recoverable_prepared",
            }
        ),
        encoding="utf-8",
    )
    args.corpus_root = str(corpus_root)

    with pytest.raises(FileNotFoundError, match="requires 'structural_core_v1::r12_seg10to12'"):
        mod.build_parity_grid_entries(args)


def test_supervision_sweep_uses_count_only_labels_and_budget_metadata(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.include_supervision_sweep = True

    entries = mod.build_parity_grid_entries(args)

    r10_local_20 = next(
        entry for entry in entries if entry.recipe_id == "r10_local_20"
    )
    r10_local_0 = next(
        entry for entry in entries if entry.recipe_id == "r10_local_0"
    )
    r20_local_50 = next(
        entry for entry in entries if entry.recipe_id == "r20_local_50"
    )

    assert r10_local_20.config.leaf_supervision_kind == "count_only"
    assert r10_local_20.config.leaf_exact_supervision is False
    assert r10_local_20.config.leaf_label_rate == pytest.approx(0.2)
    assert r10_local_20.config.internal_supervision_kind == "count_only"
    assert r10_local_20.config.internal_label_rate == pytest.approx(0.2)
    assert r10_local_20.job.budget_total_calls_per_doc == pytest.approx(0.1)
    assert r10_local_20.job.full_doc_budget_share == pytest.approx(1.0)
    assert r10_local_20.job.doc_consumption_mode == "root_only"
    assert r10_local_20.job.local_split_mode == "balanced"
    assert r10_local_20.config.package_semantics == "superset"

    assert r10_local_0.config.leaf_supervision_kind == "count_only"
    assert r10_local_0.config.leaf_exact_supervision is False
    assert r10_local_0.config.internal_supervision_kind == "none"
    assert r10_local_0.job.budget_total_calls_per_doc == pytest.approx(0.1)

    assert r20_local_50.config.leaf_supervision_kind == "count_only"
    assert r20_local_50.config.leaf_exact_supervision is False
    assert r20_local_50.config.leaf_label_rate == pytest.approx(0.5)
    assert r20_local_50.job.budget_total_calls_per_doc == pytest.approx(0.2)
    assert r20_local_50.config.package_semantics == "superset"


def test_entry_filters_support_small_smoke_subsets(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.include_supervision_sweep = True
    args.skip_fno_baselines = False
    args.include_structural = False
    args.recipe_ids = "matched_root fairfno_matched_root r10_local_20 r20_local_50 fno_baseline"
    args.fixed_leaf_tokens = "16 128"

    entries = mod.build_parity_grid_entries(args)

    assert len(entries) == 8
    assert {entry.scope_label for entry in entries} == {"recoverable"}
    assert {entry.recipe_id for entry in entries} == {
        "matched_root",
        "fairfno_matched_root",
        "r10_local_20",
        "r20_local_50",
        mod.FNO_RECIPE_ID,
    }
    assert {entry.fixed_leaf_tokens for entry in entries} == {16, 128}


def test_lean_faithful_diagnostic_matrix_builds_expected_12_job_cohort(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "1024 4096 10240"
    args.skip_fno_baselines = False
    args.include_structural = True
    args.lean_faithful_diagnostic_matrix = True

    entries = mod.build_parity_grid_entries(args)

    assert len(entries) == 12
    assert {entry.scope_label for entry in entries} == {"recoverable"}
    assert {entry.benchmark for entry in entries} == {"recoverable_v4"}
    assert {entry.job.train_doc_count for entry in entries} == {10240}
    assert {entry.fixed_leaf_tokens for entry in entries} == {16}
    assert sum(1 for entry in entries if entry.job.family == "tree_neural") == 10
    assert sum(1 for entry in entries if entry.job.family == "official_fno") == 1
    assert sum(1 for entry in entries if entry.job.family == "official_fno_sumlen") == 1
    sumlen_entry = next(
        entry for entry in entries if entry.job.family == "official_fno_sumlen"
    )
    assert sumlen_entry.config.baseline_family == "official_fno_sumlen"
    assert sumlen_entry.config.fixed_leaf_tokens == 128
    assert sumlen_entry.config.official_fno_preserve_requested_leaf_tokens is False

    local_entries = [
        entry
        for entry in entries
        if entry.recipe_id in {"r10_local_20", "r20_local_50"}
    ]
    assert len(local_entries) == 8
    assert {
        (
            entry.recipe_id,
            entry.config.leaf_supervision_kind,
            entry.config.tree_local_weighting_mode,
        )
        for entry in local_entries
    } == {
        ("r10_local_20", "count_only", "subset_mean"),
        ("r10_local_20", "count_only", "fixed_k_hajek"),
        ("r10_local_20", "bounded_full_sketch", "subset_mean"),
        ("r10_local_20", "bounded_full_sketch", "fixed_k_hajek"),
        ("r20_local_50", "count_only", "subset_mean"),
        ("r20_local_50", "count_only", "fixed_k_hajek"),
        ("r20_local_50", "bounded_full_sketch", "subset_mean"),
        ("r20_local_50", "bounded_full_sketch", "fixed_k_hajek"),
    }
    assert all(entry.config.internal_supervision_kind == entry.config.leaf_supervision_kind for entry in local_entries)
    assert all(entry.job.budget_total_calls_per_doc == pytest.approx(0.0) for entry in local_entries)
    assert all(entry.job.doc_consumption_mode == "" for entry in local_entries)
    assert all(entry.job.local_split_mode == "" for entry in local_entries)
    assert all(entry.config.package_semantics == "local_only" for entry in local_entries)
    assert all(
        entry.config.tree_local_law_weight == pytest.approx(0.8)
        for entry in local_entries
    )
    assert all(
        entry.config.tree_task_objective_weight == pytest.approx(1.0)
        for entry in local_entries
    )
    assert {
        entry.nominal_recipe_metadata["nominal_recipe_budget_total_calls_per_doc"]
        for entry in local_entries
    } == {0.1, 0.2}
    assert len({entry.job.tuning_stage for entry in local_entries}) == 8


def test_exact_collapse_repair_diagnostic_matrix_builds_expected_4_job_cohort(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "1024 4096 10240"
    args.skip_fno_baselines = False
    args.include_structural = True
    args.exact_collapse_repair_diagnostic_matrix = True

    entries = mod.build_parity_grid_entries(args)

    assert len(entries) == 4
    assert {entry.scope_label for entry in entries} == {"recoverable"}
    assert {entry.benchmark for entry in entries} == {"recoverable_v4"}
    assert {entry.job.train_doc_count for entry in entries} == {10240}
    assert {entry.fixed_leaf_tokens for entry in entries} == {128}
    assert sum(1 for entry in entries if entry.job.family == "official_fno") == 1
    assert {
        entry.recipe_id for entry in entries if entry.job.family == "tree_neural"
    } == {
        mod.EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID,
        mod.EXACT_COLLAPSE_RECIPE_ID,
        mod.EXACT_COLLAPSE_RUNTIME_MATCH_RECIPE_ID,
    }
    config_matched = next(
        entry for entry in entries if entry.recipe_id == mod.EXACT_COLLAPSE_RECIPE_ID
    )
    runtime_matched = next(
        entry
        for entry in entries
        if entry.recipe_id == mod.EXACT_COLLAPSE_RUNTIME_MATCH_RECIPE_ID
    )
    legacy_control = next(
        entry
        for entry in entries
        if entry.recipe_id == mod.EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID
    )
    official_fno = next(entry for entry in entries if entry.job.family == "official_fno")
    exact_mapping = mod._config_mapping_for_run_config(config_matched.config)
    assert config_matched.claim_level == mod.CLAIM_LEVEL_EXACT_COLLAPSE
    assert exact_mapping["tree_root_supervision_kind"] == "mse"
    assert exact_mapping["fixed_leaf_tokens"] == 128
    assert exact_mapping["tree_leaf_fno_width"] == 128
    assert exact_mapping["tree_leaf_fno_n_modes"] == 8
    assert exact_mapping["local_law_weight"] == pytest.approx(0.0)
    assert exact_mapping["task_objective_weight"] == pytest.approx(1.0)
    assert exact_mapping["leaf_supervision_kind"] == "count_only"
    assert exact_mapping["leaf_label_rate"] == pytest.approx(0.0)
    assert exact_mapping["internal_supervision_kind"] == "none"
    assert exact_mapping["internal_label_rate"] == pytest.approx(0.0)
    assert config_matched.config.tree_exact_collapse_mode == (
        mod.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
    )
    assert mod._config_diff_vs_official_fno(
        config_mapping=exact_mapping,
        reference_surface=config_matched.official_fno_reference_surface,
    ) == {}
    assert runtime_matched.config.tree_exact_collapse_mode == (
        mod.EXACT_COLLAPSE_RUNTIME_IDENTITY_MODE
    )
    assert runtime_matched.claim_level == mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY
    assert legacy_control.claim_level == mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY
    assert official_fno.config.fixed_leaf_tokens == 128
    assert official_fno.config.official_fno_preserve_requested_leaf_tokens is True
    assert config_matched.config.official_fno_preserve_requested_leaf_tokens is True
    assert runtime_matched.config.official_fno_preserve_requested_leaf_tokens is True


def test_full_local_laws_topology_diagnostic_4096_builds_expected_8_job_cohort(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "1024 4096 10240"
    args.skip_fno_baselines = False
    args.include_structural = True
    args.full_local_laws_topology_diagnostic_4096 = True

    entries = mod.build_parity_grid_entries(args)

    assert len(entries) == 8
    assert {entry.scope_label for entry in entries} == {"recoverable"}
    assert {entry.benchmark for entry in entries} == {"recoverable_v4"}
    assert {entry.job.train_doc_count for entry in entries} == {4096}
    assert {entry.fixed_leaf_tokens for entry in entries} == {64, 128}
    assert {entry.seed for entry in entries} == {0, 1}
    assert {
        str(entry.job.study_axis) for entry in entries
    } == {mod.FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS}
    assert {entry.claim_level for entry in entries} == {mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY}

    fno_entries = [
        entry for entry in entries if str(entry.job.family) == "official_fno"
    ]
    tree_entries = [
        entry for entry in entries if str(entry.job.family) == mod.TREE_BASELINE_FAMILY
    ]
    assert len(fno_entries) == 4
    assert len(tree_entries) == 4
    assert {
        str(entry.job.axis_value) for entry in fno_entries
    } == {"official_fno_leaf64", "official_fno_leaf128"}
    assert all(
        bool(entry.config.official_fno_preserve_requested_leaf_tokens)
        for entry in fno_entries
    )

    assert {entry.recipe_id for entry in tree_entries} == {mod.FULL_LOCAL_LAWS_TREE_RECIPE_ID}
    assert {
        str(entry.job.axis_value) for entry in tree_entries
    } == {"tree_neural_leaf64", "tree_neural_leaf128"}
    assert all(bool(entry.config.preserve_requested_leaf_tokens) for entry in tree_entries)
    assert all(
        str(entry.job.locked_tree_neural_config_label)
        == mod.SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET
        for entry in tree_entries
    )
    for entry in tree_entries:
        mapping = mod._config_mapping_for_run_config(entry.config)
        assert mapping["n_epochs"] == 20
        assert mapping["batch_size"] == 64
        assert mapping["fixed_leaf_tokens"] in {64, 128}
        assert mapping["tree_theorem_surface_mode"] == "factorized_score_fiber"
        assert mapping["tree_summary_spec_root_mode"] == "factored_theorem_readout"
        assert mapping["summary_spec_name"] == "markov_count_sketch"
        assert mapping["tree_task_head_mode"] == "theorem_feature_scalar"
        assert mapping["local_law_weight"] == pytest.approx(0.8)
        assert mapping["task_objective_weight"] == pytest.approx(1.0)
        assert mapping["c1_relative_weight"] == pytest.approx(1.0)
        assert mapping["c2_relative_weight"] == pytest.approx(1.0)
        assert mapping["c3_relative_weight"] == pytest.approx(1.0)
        assert mapping["leaf_supervision_kind"] == "full_sketch"
        assert mapping["leaf_label_rate"] == pytest.approx(1.0)
        assert mapping["internal_supervision_kind"] == "full_sketch"
        assert mapping["internal_label_rate"] == pytest.approx(1.0)
        assert mapping["tree_join_bit_weight"] == pytest.approx(1.0)


def test_runtime_override_args_flow_into_topology_configs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "4096"
    args.skip_fno_baselines = False
    args.include_structural = False
    args.unified_g_topology_diagnostic_4096 = True
    args.gpu_runtime_preload_splits = ("train", "val")
    args.gpu_runtime_preload_targets = False

    entries = mod.build_parity_grid_entries(args)

    tree_entry = next(
        entry for entry in entries if str(entry.job.family) == mod.TREE_BASELINE_FAMILY
    )
    fno_entry = next(
        entry for entry in entries if str(entry.job.family) == "official_fno"
    )
    assert tree_entry.config.gpu_runtime_preload_splits == ("train", "val")
    assert tree_entry.config.gpu_runtime_preload_targets is False
    assert fno_entry.config.gpu_runtime_preload_splits == ("train", "val")
    assert fno_entry.config.gpu_runtime_preload_targets is False


def test_topology_posttrain_overrides_apply_to_tree_configs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "4096"
    args.skip_fno_baselines = True
    args.include_structural = False
    args.unified_g_topology_diagnostic_4096 = True
    args.topology_posttrain_train_doc_limit = 128
    args.topology_posttrain_diagnostics_mode = "full"
    args.topology_stress_leaf_tokens = (16,)
    args.topology_stress_posttrain_diagnostics_mode = "minimal"

    entries = mod.build_parity_grid_entries(args)

    leaf32_entry = next(
        entry
        for entry in entries
        if entry.fixed_leaf_tokens == 32 and str(entry.job.family) == mod.TREE_BASELINE_FAMILY
    )
    leaf16_entry = next(
        entry
        for entry in entries
        if entry.fixed_leaf_tokens == 16 and str(entry.job.family) == mod.TREE_BASELINE_FAMILY
    )
    assert leaf32_entry.config.tree_posttrain_train_doc_limit == 128
    assert leaf32_entry.config.posttrain_diagnostics_mode == "full"
    assert leaf16_entry.config.tree_posttrain_train_doc_limit == 128
    assert leaf16_entry.config.posttrain_diagnostics_mode == "minimal"


def test_lean_faithful_weight_balance_sweep_augments_diagnostic_matrix(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "10240"
    args.benchmark = "recoverable_v4"
    args.include_structural = False
    args.skip_fno_baselines = False
    args.lean_faithful_diagnostic_matrix = True
    args.lean_faithful_weight_balance_sweep = True
    args.recipe_ids = "matched_root fairfno_matched_root r10_local_20 r20_local_50 fno_baseline"
    args.fixed_leaf_tokens = "16"

    entries = mod.build_parity_grid_entries(args)

    assert len(entries) == 18
    weight_entries = [
        entry
        for entry in entries
        if entry.job.study_axis == "lean_weight_balance"
    ]
    assert len(weight_entries) == 6
    assert {
        (
            entry.config.tree_local_law_weight,
            entry.config.tree_task_objective_weight,
            entry.config.tree_c1_relative_weight,
            entry.config.tree_c3_relative_weight,
            entry.config.leaf_supervision_kind,
            entry.config.tree_local_weighting_mode,
        )
        for entry in weight_entries
    } == {
        (0.10, 0.90, 1.0, 1.0, "bounded_full_sketch", "fixed_k_hajek"),
        (0.10, 0.90, 2.0, 1.0, "bounded_full_sketch", "fixed_k_hajek"),
        (0.25, 0.75, 1.0, 1.0, "bounded_full_sketch", "fixed_k_hajek"),
        (0.25, 0.75, 2.0, 1.0, "bounded_full_sketch", "fixed_k_hajek"),
        (0.50, 0.50, 1.0, 1.0, "bounded_full_sketch", "fixed_k_hajek"),
        (0.50, 0.50, 2.0, 1.0, "bounded_full_sketch", "fixed_k_hajek"),
    }
    assert all(entry.config.package_semantics == "local_only" for entry in weight_entries)
    assert all(
        entry.nominal_recipe_metadata["nominal_recipe_budget_total_calls_per_doc"]
        == pytest.approx(0.2)
        for entry in weight_entries
    )


def test_explicit_prepared_data_allow_create_propagates_into_corpus_backed_entries(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "10240"
    args.benchmark = "recoverable_v4"
    args.include_structural = False
    args.skip_fno_baselines = False
    args.lean_faithful_diagnostic_matrix = True
    args.prepared_data_allow_create = True
    args.recipe_ids = "matched_root fairfno_matched_root r10_local_20 r20_local_50 fno_baseline"
    args.fixed_leaf_tokens = "16"

    entries = mod.build_parity_grid_entries(args)

    assert entries
    assert all(entry.config.prepared_data_allow_create is True for entry in entries)


def test_row_from_manifest_job_carries_local_objective_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    config = mod._config_for_entry(
        args,
        recipe_id="matched_root",
        benchmark="recoverable_v4",
        fixed_leaf_tokens=16,
    )
    job = mod._JobSpec(
        family="tree_neural",
        train_doc_count=10240,
        benchmark="recoverable_v4",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(0,),
        config=config,
    )
    entry = mod.ParityGridEntry(
        recipe_id="matched_root",
        benchmark="recoverable_v4",
        scope_key="recoverable_v4",
        scope_label="recoverable",
        claim_level=mod.CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
        fixed_leaf_tokens=16,
        seed=0,
        config=config,
        job=job,
    )
    manifest_job = entry.manifest_row(
        output_root=tmp_path,
        main_train_doc_count=10240,
        epoch_cap=10,
    )
    metrics = {
        "state": "completed",
        "tree_local_weighting_mode": "fixed_k_hajek",
        "local_loss_kind": "bounded_full_sketch",
        "local_sampling_design_name": "deterministic_fixed_k_uniform",
        "leaf_population_size": 8.0,
        "leaf_sample_size": 2.0,
        "leaf_effective_propensity": 0.25,
        "merge_population_size": 7.0,
        "merge_sample_size": 2.0,
        "merge_effective_propensity": 2.0 / 7.0,
        "local_objective_audit": {
            "weighting_mode": "fixed_k_hajek",
            "design_name": "deterministic_fixed_k_uniform",
        },
    }
    monkeypatch.setattr(mod, "_summary_metrics_for_job", lambda _job_output_dir: dict(metrics))

    row = mod._row_from_manifest_job(
        manifest_job,
        failed_job_names=set(),
    )

    assert row["tree_local_weighting_mode"] == "fixed_k_hajek"
    assert row["local_loss_kind"] == "bounded_full_sketch"
    assert row["local_sampling_design_name"] == "deterministic_fixed_k_uniform"
    assert row["leaf_population_size"] == pytest.approx(8.0)
    assert row["leaf_sample_size"] == pytest.approx(2.0)
    assert row["leaf_effective_propensity"] == pytest.approx(0.25)
    assert row["merge_population_size"] == pytest.approx(7.0)
    assert row["merge_sample_size"] == pytest.approx(2.0)
    assert row["merge_effective_propensity"] == pytest.approx(2.0 / 7.0)
    assert row["local_objective_audit"] == {
        "weighting_mode": "fixed_k_hajek",
        "design_name": "deterministic_fixed_k_uniform",
    }


def test_manifest_row_preserves_nominal_recipe_metadata_without_overwriting_config(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.train_doc_counts = "10240"
    args.include_structural = False
    args.skip_fno_baselines = False
    args.lean_faithful_diagnostic_matrix = True
    args.recipe_ids = "r10_local_20"
    args.fixed_leaf_tokens = "16"

    [entry] = [
        entry
        for entry in mod.build_parity_grid_entries(args)
        if entry.recipe_id == "r10_local_20"
    ][:1]

    manifest_job = entry.manifest_row(
        output_root=tmp_path,
        main_train_doc_count=10240,
        epoch_cap=10,
    )

    assert manifest_job["config"]["budget_total_calls_per_doc"] == pytest.approx(0.0)
    assert manifest_job["config"]["package_semantics"] == "local_only"
    assert manifest_job["nominal_recipe_budget_total_calls_per_doc"] == pytest.approx(0.1)
    assert manifest_job["nominal_recipe_metadata"]["nominal_recipe_id"] == "r10_local_20"


def test_row_from_manifest_job_requires_exact_collapse_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.include_structural = False
    config = mod._config_for_entry(
        args,
        recipe_id=mod.EXACT_COLLAPSE_RECIPE_ID,
        benchmark="recoverable_v4",
        fixed_leaf_tokens=128,
    )
    entry = mod.ParityGridEntry(
        recipe_id=mod.EXACT_COLLAPSE_RECIPE_ID,
        benchmark="recoverable_v4",
        scope_key="recoverable_v4",
        scope_label="recoverable",
        claim_level=mod.CLAIM_LEVEL_EXACT_COLLAPSE,
        fixed_leaf_tokens=128,
        seed=0,
        config=config,
        job=mod._JobSpec(
            family="tree_neural",
            train_doc_count=10240,
            benchmark="recoverable_v4",
            hardness_grid="",
            grid_cell_ids=(),
            seeds=(0,),
            config=config,
        ),
        official_fno_reference_surface=mod._official_fno_reference_surface(
            args,
            benchmark="recoverable_v4",
            fixed_leaf_tokens=128,
        ),
    )
    manifest_job = entry.manifest_row(
        output_root=tmp_path,
        main_train_doc_count=10240,
        epoch_cap=10,
    )
    prepared_root = tmp_path / "prepared"
    prepared_dir = prepared_root / "prepared_sig_ok"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "metadata.json").write_text(
        json.dumps(
            {
                "train_prefix_counts": [10240],
                "train_prefix_signatures": {"10240": "train-10240"},
                "train_corpus_signature": "train-full",
                "val_corpus_signature": "val-fixed",
                "test_corpus_signature": "test-fixed",
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "state": "completed",
        "test_root_mae_mean": 0.125,
        "train_corpus_signature": "train-10240",
        "val_corpus_signature": "val-fixed",
        "test_corpus_signature": "test-fixed",
        "optimization_root_weight": 1.0,
        "local_law_c1_weight": 0.0,
        "local_law_c2_weight": 0.0,
        "local_law_c3_weight": 0.0,
        "config": {
            **mod._config_mapping_for_run_config(config),
            "prepared_data_root": str(prepared_root),
            "prepared_data_signature": "sig_ok",
            "base_bundle_path": "/tmp/bundle_train10240.pkl",
        },
    }

    monkeypatch.setattr(
        mod,
        "_summary_metrics_for_job",
        lambda _job_output_dir: dict(
            metrics,
            bundle_source="",
            config={**dict(metrics["config"]), "base_bundle_path": ""},
        ),
    )
    row_missing = mod._row_from_manifest_job(
        manifest_job,
        failed_job_names=set(),
    )
    assert row_missing["config_diff_vs_official_fno"] == {}
    assert row_missing["strict_collapse_pass"] is False

    monkeypatch.setattr(
        mod,
        "_summary_metrics_for_job",
        lambda _job_output_dir: dict(
            metrics,
            bundle_source="/tmp/bundle_train10240.pkl",
        ),
    )
    row_present = mod._row_from_manifest_job(
        manifest_job,
        failed_job_names=set(),
    )
    assert row_present["config_diff_vs_official_fno"] == {}
    assert row_present["strict_collapse_pass"] is True


def test_summary_metrics_for_job_preserves_run_level_provenance_under_aggregates(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "summary.json").write_text(
        json.dumps(
            {
                "aggregate_rows": [
                    {
                        "test_root_mae_mean": 0.25,
                    }
                ],
                "runs": [
                    {
                        "bundle_source": "/tmp/bundle.pkl",
                        "train_corpus_signature": "train-fixed",
                        "val_corpus_signature": "val-fixed",
                        "test_corpus_signature": "test-fixed",
                        "collapse_runtime_delegate_family": "official_fno",
                        "collapse_runtime_mode": mod.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE,
                        "optimization_root_weight": 1.0,
                        "local_law_c1_weight": 0.0,
                        "local_law_c2_weight": 0.0,
                        "local_law_c3_weight": 0.0,
                        "config": {"base_bundle_path": "/tmp/bundle.pkl"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = mod._summary_metrics_for_job(job_dir)

    assert metrics["bundle_source"] == "/tmp/bundle.pkl"
    assert metrics["train_corpus_signature"] == "train-fixed"
    assert metrics["val_corpus_signature"] == "val-fixed"
    assert metrics["test_corpus_signature"] == "test-fixed"
    assert metrics["collapse_runtime_delegate_family"] == "official_fno"
    assert metrics["collapse_runtime_mode"] == mod.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
    assert metrics["optimization_root_weight"] == pytest.approx(1.0)
