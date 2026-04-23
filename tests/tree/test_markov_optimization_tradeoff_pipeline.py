from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest
import torch

import scripts.run_markov_optimization_tradeoff_pipeline as pipeline
from src.ctreepo.sim.core.tree_reference_presets import (
    COMPARISON_GRID_V3_PRESET,
    UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
    resolve_tree_reference_preset,
    resolve_tree_reference_preset_config,
    tree_reference_preset_names,
)
from src.ctreepo.sim.core.markov_study_names import (
    resolve_law_package_names,
    resolve_supervision_recovery_package_names,
    supervision_recovery_package_public_name,
)

from scripts.run_markov_optimization_tradeoff_pipeline import (
    _aggregate_batch_timing,
    _aggregate_docs_epochs,
    _aggregate_full_doc_upper_bound_from_payloads,
    _aggregate_large_batch_diagnosis,
    _aggregate_law_packages_from_payloads,
    _aggregate_medium_grid,
    _aggregate_supervision_recovery_from_payloads,
    _aggregate_supervision_sweep_from_payloads,
    _aggregate_weight_ablation_from_payloads,
    _build_law_package_phase,
    _supervision_recovery_result_rows_from_summary,
    _build_supervision_recovery_phase,
    _build_support_phase,
    _build_supervision_phase,
    _direct_task,
    _load_supervision_recovery_refresh_payloads,
    _load_report_version_manifest,
    _parse_args,
    _phase_config_fingerprint,
    _refresh_existing_tradeoff_outputs,
    _refresh_selected_source_statuses,
    _resolve_devices,
    _register_phase_source,
    _tradeoff_experiment_spec,
    _stage_report_sources,
    _write_report_version_manifest,
    build_run_plan,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _with_supervision_recovery_v3_payloads(
    payloads: list[dict[str, object]],
) -> list[dict[str, object]]:
    def _default_run_intent_hash(
        *,
        payload_index: int,
        record: dict[str, object],
        config: dict[str, object],
        fixed_leaf_tokens: int,
        comparison_semantics: str,
    ) -> str:
        baseline_family = str(record.get("baseline_family", "") or "")
        train_doc_count = int(
            record.get("train_doc_count", config.get("train_docs", 0)) or 0
        )
        scope_key = str(
            record.get(
                "cell_id",
                config.get(
                    "pipeline_supervision_recovery_scope",
                    config.get("pipeline_supervision_recovery_scope_label", ""),
                ),
            )
            or ""
        )
        package_name = str(
            config.get("pipeline_supervision_recovery_package", "") or ""
        )
        comparison_arm = str(
            config.get("pipeline_supervision_recovery_comparison_arm", "primary")
            or "primary"
        )
        depth_discount_gamma = float(
            record.get(
                "depth_discount_gamma",
                config.get("depth_discount_gamma", 1.0),
            )
            or 1.0
        )
        return (
            f"payload{payload_index}"
            f"::{scope_key}"
            f"::{package_name}"
            f"::{baseline_family}"
            f"::{train_doc_count}"
            f"::{fixed_leaf_tokens}"
            f"::{comparison_arm}"
            f"::g{depth_discount_gamma:0.6f}"
            f"::{comparison_semantics}"
        )

    updated_payloads: list[dict[str, object]] = []
    for payload_index, payload in enumerate(payloads):
        payload_map = dict(payload)
        config = dict(payload_map.get("config") or {})
        aggregate_rows = []
        for row_index, raw_row in enumerate(list(payload_map.get("aggregate_rows") or [])):
            row = dict(raw_row or {})
            fixed_leaf_tokens = int(
                row.get("fixed_leaf_tokens", config.get("fixed_leaf_tokens", 128)) or 128
            )
            comparison_semantics = str(
                row.get("comparison_semantics", "current") or "current"
            )
            aggregate_rows.append(
                {
                    **row,
                    "fixed_leaf_tokens": int(fixed_leaf_tokens),
                    "comparison_mode": str(
                        row.get(
                            "comparison_mode",
                            config.get("comparison_mode", "comparable"),
                        )
                        or "comparable"
                    ),
                    "comparison_semantics": comparison_semantics,
                    "comparison_semantics_label": str(
                        row.get(
                            "comparison_semantics_label",
                            "tree_neural_objective_v2"
                            if comparison_semantics == "current"
                            else comparison_semantics,
                        )
                        or ""
                    ),
                    "run_intent_hash": str(
                        row.get(
                            "run_intent_hash",
                            _default_run_intent_hash(
                                payload_index=payload_index,
                                record=row,
                                config=config,
                                fixed_leaf_tokens=fixed_leaf_tokens,
                                comparison_semantics=comparison_semantics,
                            ),
                        )
                        or ""
                    ),
                    "run_intent_validation_status": str(
                        row.get(
                            "run_intent_validation_status",
                            "validated"
                            if comparison_semantics != "locked_comparator"
                            else "locked_comparator",
                        )
                        or ""
                    ),
                    "requested_fixed_leaf_tokens": int(
                        row.get("requested_fixed_leaf_tokens", fixed_leaf_tokens)
                        or fixed_leaf_tokens
                    ),
                    "executed_fixed_leaf_tokens": int(
                        row.get(
                            "executed_fixed_leaf_tokens",
                            row.get("fixed_leaf_tokens", fixed_leaf_tokens),
                        )
                        or fixed_leaf_tokens
                    ),
                    "depth_discount_gamma": float(
                        row.get(
                            "depth_discount_gamma",
                            config.get("depth_discount_gamma", 1.0),
                        )
                        or 1.0
                    ),
                }
            )
        runs = []
        for run_index, raw_run in enumerate(list(payload_map.get("runs") or [])):
            run = dict(raw_run or {})
            run_config = dict(run.get("config") or {})
            fixed_leaf_tokens = int(
                run.get(
                    "fixed_leaf_tokens",
                    run_config.get("fixed_leaf_tokens", config.get("fixed_leaf_tokens", 128)),
                )
                or 128
            )
            comparison_semantics = str(
                run.get("comparison_semantics", "current") or "current"
            )
            runs.append(
                {
                    **run,
                    "fixed_leaf_tokens": int(fixed_leaf_tokens),
                    "comparison_mode": str(
                        run.get(
                            "comparison_mode",
                            run_config.get(
                                "comparison_mode",
                                config.get("comparison_mode", "comparable"),
                            ),
                        )
                        or "comparable"
                    ),
                    "comparison_semantics": comparison_semantics,
                    "comparison_semantics_label": str(
                        run.get(
                            "comparison_semantics_label",
                            "tree_neural_objective_v2"
                            if comparison_semantics == "current"
                            else comparison_semantics,
                        )
                        or ""
                    ),
                    "run_intent_hash": str(
                        run.get(
                            "run_intent_hash",
                            _default_run_intent_hash(
                                payload_index=payload_index,
                                record=run,
                                config=config,
                                fixed_leaf_tokens=fixed_leaf_tokens,
                                comparison_semantics=comparison_semantics,
                            ),
                        )
                        or ""
                    ),
                    "run_intent_validation_status": str(
                        run.get(
                            "run_intent_validation_status",
                            "validated"
                            if comparison_semantics != "locked_comparator"
                            else "locked_comparator",
                        )
                        or ""
                    ),
                    "requested_fixed_leaf_tokens": int(
                        run.get("requested_fixed_leaf_tokens", fixed_leaf_tokens)
                        or fixed_leaf_tokens
                    ),
                    "executed_fixed_leaf_tokens": int(
                        run.get(
                            "executed_fixed_leaf_tokens",
                            run.get("fixed_leaf_tokens", fixed_leaf_tokens),
                        )
                        or fixed_leaf_tokens
                    ),
                    "depth_discount_gamma": float(
                        run.get(
                            "depth_discount_gamma",
                            run_config.get(
                                "depth_discount_gamma",
                                config.get("depth_discount_gamma", 1.0),
                            ),
                        )
                        or 1.0
                    ),
                }
            )
        updated_payloads.append(
            {
                **payload_map,
                "aggregate_rows": aggregate_rows,
                "runs": runs,
            }
        )
    return updated_payloads


def test_root_only_parity_presets_are_registered() -> None:
    historical = dict(
        pipeline.TREE_REFERENCE_PRESET_CONFIGS[
            pipeline.ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET
        ]
    )
    optimization = dict(
        pipeline.TREE_REFERENCE_PRESET_CONFIGS[
            pipeline.ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET
        ]
    )
    capacity = dict(
        pipeline.TREE_REFERENCE_PRESET_CONFIGS[
            pipeline.ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET
        ]
    )
    matched = dict(
        pipeline.TREE_REFERENCE_PRESET_CONFIGS[
            pipeline.ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
        ]
    )
    structural = dict(
        pipeline.TREE_REFERENCE_PRESET_CONFIGS[
            pipeline.STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
        ]
    )

    assert historical["state_dim"] == 128
    assert historical["hidden_dim"] == 512
    assert historical["n_epochs"] == 52
    assert historical["leaf_supervision_kind"] == "count_only"
    assert historical["leaf_label_rate"] == 0.0
    assert historical["internal_supervision_kind"] == "none"
    assert historical["internal_label_rate"] == 0.0
    assert historical["tree_training_schedule"] == "two_stage"
    assert historical["tree_checkpoint_metric"] == "val_exact_sketch_direct"
    assert historical["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"

    assert optimization["state_dim"] == 128
    assert optimization["hidden_dim"] == 512
    assert optimization["n_epochs"] == 128
    assert optimization["leaf_supervision_kind"] == "count_only"
    assert optimization["internal_supervision_kind"] == "none"
    assert optimization["tree_training_schedule"] == "single_stage"
    assert optimization["tree_checkpoint_metric"] == "val_root_mae"
    assert optimization["tree_stage1_root_weight"] == 1.0

    assert capacity["state_dim"] == 256
    assert capacity["hidden_dim"] == 1024
    assert capacity["leaf_supervision_kind"] == "count_only"
    assert capacity["internal_supervision_kind"] == "none"
    assert capacity["tree_training_schedule"] == "two_stage"

    assert matched["state_dim"] == 256
    assert matched["hidden_dim"] == 1024
    assert matched["n_epochs"] == 128
    assert matched["leaf_supervision_kind"] == "count_only"
    assert matched["internal_supervision_kind"] == "none"
    assert matched["tree_training_schedule"] == "single_stage"
    assert matched["tree_checkpoint_metric"] == "val_root_mae"
    assert matched["tree_stage1_root_weight"] == 0.0
    assert matched["tree_root_supervision_kind"] == "count_ce"
    assert matched["local_law_weight"] == pytest.approx(0.0)
    assert matched["c1_relative_weight"] == pytest.approx(0.0)
    assert matched["c2_relative_weight"] == pytest.approx(0.0)
    assert matched["c3_relative_weight"] == pytest.approx(0.0)
    assert matched["tree_model_version"] == "unified_g"
    assert matched["tree_score_merge_mode"] == "exact_projected_sketch"
    assert matched["tree_theorem_surface_mode"] == "factorized_score_fiber"
    assert matched["tree_leaf_fno_width"] == 256
    assert matched["tree_leaf_fno_n_modes"] == 16

    assert structural["state_dim"] == 256
    assert structural["hidden_dim"] == 1024
    assert structural["n_epochs"] == 128
    assert structural["leaf_supervision_kind"] == "count_only"
    assert structural["internal_supervision_kind"] == "none"
    assert structural["tree_training_schedule"] == "single_stage"
    assert structural["tree_checkpoint_metric"] == "val_root_mae"
    assert structural["tree_stage1_root_weight"] == 0.0
    assert structural["tree_root_supervision_kind"] == "count_ce"
    assert structural["local_law_weight"] == pytest.approx(0.0)
    assert structural["c1_relative_weight"] == pytest.approx(0.0)
    assert structural["c2_relative_weight"] == pytest.approx(0.0)
    assert structural["c3_relative_weight"] == pytest.approx(0.0)


def test_public_tree_reference_preset_aliases_resolve_to_recipe() -> None:
    preset_record = resolve_tree_reference_preset(COMPARISON_GRID_V3_PRESET)

    assert preset_record["requested_name"] == COMPARISON_GRID_V3_PRESET
    assert preset_record["public_name"] == COMPARISON_GRID_V3_PRESET
    assert preset_record["recipe_name"] == UNIFIED_G_FULL_LOCAL_LAWS_PRESET
    assert preset_record["is_public_alias"] is True
    assert preset_record["config"] == resolve_tree_reference_preset_config(
        UNIFIED_G_FULL_LOCAL_LAWS_PRESET
    )
    assert COMPARISON_GRID_V3_PRESET in tree_reference_preset_names(public_only=True)


def test_supervision_recovery_package_aliases_resolve_public_names_and_groups() -> None:
    resolved = resolve_supervision_recovery_package_names(
        ["comparison_grid_v3", "root100_extra_leaf05_internal10"],
        valid_names=tuple(pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS.keys()),
    )

    assert resolved == [
        "full100",
        "r100_superset_local_eq_10p0",
        "r100_superset_local_eq_15p0",
        "r100_superset_local_eq_20p0",
        "r100_superset_leaf05_internal10p0",
    ]
    assert supervision_recovery_package_public_name("full100") == "root100"
    assert supervision_recovery_package_public_name(
        "r100_superset_local_eq_10p0"
    ) == "root100_extra_local10"


def test_supervision_recovery_package_aliases_resolve_redistribution_names_and_groups() -> None:
    resolved = resolve_supervision_recovery_package_names(
        ["redistribution_r100_coarse"],
        valid_names=tuple(pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS.keys()),
    )

    assert resolved == [
        "full100",
        "r100_node_mass_eq_20p0",
        "r100_node_mass_eq_50p0",
        "r100_node_mass_eq_80p0",
        "r100_node_mass_eq_100p0",
    ]
    assert supervision_recovery_package_public_name(
        "r100_node_mass_eq_50p0"
    ) == "root50_nodes50"


def test_supervision_recovery_package_aliases_resolve_depth_aware_redistribution_groups() -> None:
    leaf_only = resolve_supervision_recovery_package_names(
        ["mass_preserving_leaf_only_deciles"],
        valid_names=tuple(pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS.keys()),
    )
    levels_equal = resolve_supervision_recovery_package_names(
        ["mass_preserving_levels_equal_deciles"],
        valid_names=tuple(pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS.keys()),
    )
    root_ladder = resolve_supervision_recovery_package_names(
        ["root_ladder_deciles"],
        valid_names=tuple(pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS.keys()),
    )

    assert root_ladder == [
        "full100",
        "full90",
        "full80",
        "full70",
        "full60",
        "full50",
        "full40",
        "full30",
        "full20",
        "full10",
    ]
    assert leaf_only[0] == "full100"
    assert "r50_leaf_mass_eq_50p0" in leaf_only
    assert "r0_leaf_mass_eq_100p0" in leaf_only
    assert levels_equal[0] == "full100"
    assert "r50_depth_equal_mass_eq_50p0" in levels_equal
    assert "r0_depth_equal_mass_eq_100p0" in levels_equal
    assert supervision_recovery_package_public_name("full40") == "root40"
    assert (
        supervision_recovery_package_public_name("r50_leaf_mass_eq_50p0")
        == "root50_leaf50"
    )
    assert (
        supervision_recovery_package_public_name("r50_depth_equal_mass_eq_50p0")
        == "root50_levels_equal50"
    )


def test_law_package_aliases_resolve_to_canonical_names() -> None:
    resolved = resolve_law_package_names(
        ["c2_only", "all_laws"],
        valid_names=tuple(pipeline.LAW_PACKAGE_CONFIGS.keys()),
    )

    assert resolved == ["tree_c2_only", "tree_all_laws"]


def _write_capacity_locked_tree_reference(root: Path) -> Path:
    capacity_root = root / "capacity"
    winning_label = "fair_fno_v1_w128_m4_l4"
    locked_summary = {
        "runs": [
            {
                "baseline_family": "tree_neural",
                "config_label": winning_label,
                "config": {
                    "n_epochs": 32,
                    "batch_size": 64,
                    "lr": 5e-4,
                    "weight_decay": 0.0,
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "fixed_leaf_tokens": 16,
                    "tree_model_version": "v2",
                    "tree_batch_runtime_mode": "unified_v2",
                    "tree_batch_pack_mode": "fixed_fused",
                    "gpu_runtime_data_mode": "resident",
                    "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
                    "tree_leaf_fno_width": 128,
                    "tree_leaf_fno_n_modes": 4,
                    "tree_leaf_fno_n_layers": 4,
                    "tree_training_schedule": "two_stage",
                    "tree_task_head_mode": "theorem_feature_scalar",
                    "tree_theorem_surface_mode": "slotwise",
                    "tree_summary_spec_root_mode": "factored_theorem_readout",
                    "tree_root_supervision_kind": "mse",
                    "tree_checkpoint_metric": "val_exact_sketch_direct",
                    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                    "tree_stage1_epochs": 12,
                    "tree_stage2_epochs": 20,
                    "summary_spec_name": "markov_count_sketch",
                    "slot_count": 4,
                    "tree_aux_doc_sequence_fraction": 0.0,
                    "leaf_supervision_kind": "full_sketch",
                    "leaf_label_rate": 1.0,
                    "internal_supervision_kind": "none",
                    "internal_label_rate": 0.0,
                    "root_weight": 1.0,
                    "gpu_runtime_workers_per_mig": 4,
                },
            }
        ]
    }
    _write_json(capacity_root / "locked" / "summary.json", locked_summary)
    _write_json(
        capacity_root / "tree_fno_capacity_locked_summary.json",
        {
            "winning_config_label": winning_label,
            "locked_summary_json": str(capacity_root / "locked" / "summary.json"),
        },
    )
    return capacity_root


def _write_package_capacity_locked_tree_references(
    root: Path,
    *,
    package_configs: dict[str, dict[str, object]],
) -> Path:
    base_root = root / "package_capacity"
    for package_name, overrides in package_configs.items():
        package_root = base_root / str(package_name)
        winning_label = str(overrides.get("winning_label", f"{package_name}_cfg"))
        config = {
            "n_epochs": 32,
            "batch_size": 64,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "state_dim": 128,
            "hidden_dim": 512,
            "fixed_leaf_tokens": 16,
            "tree_model_version": "v2",
            "tree_batch_runtime_mode": "unified_v2",
            "tree_batch_pack_mode": "fixed_fused",
            "gpu_runtime_data_mode": "resident",
            "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
            "tree_leaf_fno_width": 128,
            "tree_leaf_fno_n_modes": 4,
            "tree_leaf_fno_n_layers": 4,
            "tree_training_schedule": "two_stage",
            "tree_task_head_mode": "theorem_feature_scalar",
            "tree_theorem_surface_mode": "slotwise",
            "tree_summary_spec_root_mode": "factored_theorem_readout",
            "tree_root_supervision_kind": "mse",
            "tree_checkpoint_metric": "val_exact_sketch_direct",
            "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "summary_spec_name": "markov_count_sketch",
            "slot_count": 4,
            "leaf_supervision_kind": "full_sketch",
            "leaf_label_rate": 1.0,
            "internal_supervision_kind": "full_sketch",
            "internal_label_rate": 1.0,
            "root_weight": 1.0,
            "schedule_consistency_weight": 0.0,
            "gpu_runtime_workers_per_mig": 1,
        }
        config.update({str(key): value for key, value in overrides.items() if key != "winning_label"})
        _write_json(
            package_root / "locked" / "summary.json",
            {
                "runs": [
                    {
                        "baseline_family": "tree_neural",
                        "config_label": winning_label,
                        "config": config,
                    }
                ]
            },
        )
        _write_json(
            package_root / "tree_fno_capacity_locked_summary.json",
            {
                "winning_config_label": winning_label,
                "locked_summary_json": str(package_root / "locked" / "summary.json"),
            },
        )
    return base_root


def _write_weak_capacity_locked_tree_reference(root: Path) -> Path:
    capacity_root = root / "weak_capacity"
    winning_label = "weak_tree_cfg"
    locked_summary = {
        "runs": [
            {
                "baseline_family": "tree_neural",
                "config_label": winning_label,
                "config": {
                    "state_dim": 32,
                    "hidden_dim": 64,
                    "fixed_leaf_tokens": 16,
                    "tree_model_version": "v2",
                    "tree_batch_runtime_mode": "unified_v2",
                    "tree_batch_pack_mode": "fixed_fused",
                    "gpu_runtime_data_mode": "resident",
                    "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
                    "tree_training_schedule": "two_stage",
                    "tree_stage1_epochs": 5,
                    "tree_stage2_epochs": 10,
                    "tree_task_head_mode": "full_state_scalar",
                    "tree_summary_spec_root_mode": "task_split_ablation",
                    "tree_root_supervision_kind": "count_ce",
                    "tree_checkpoint_metric": "val_root_mae",
                    "tree_stage1_checkpoint_metric": "val_root_mae",
                    "summary_spec_name": "",
                    "slot_count": 0,
                },
            }
        ]
    }
    _write_json(capacity_root / "locked" / "summary.json", locked_summary)
    _write_json(
        capacity_root / "tree_fno_capacity_locked_summary.json",
        {
            "winning_config_label": winning_label,
            "locked_summary_json": str(capacity_root / "locked" / "summary.json"),
        },
    )
    return capacity_root


def test_profile_aggregators_smoke(tmp_path: Path) -> None:
    batch_summary = {
        "config": {"train_docs": 1000},
        "runs": {
            "no_autotune": {
                "wall_clock_s": 2.0,
                "epochs_completed": 1,
                "best_val_mae": 0.1,
                "timing_breakdown": {
                    "train_loop_s": 1.5,
                    "train_forward_s": 0.6,
                    "train_backward_s": 0.7,
                    "screen_eval_s": 0.05,
                    "exact_metric_eval_s": 0.06,
                    "eval_total_s": 0.11,
                },
                "batching_metrics": {
                    "gpu_reserved_mem_peak_gb": 0.2,
                    "train_docs_per_batch_mean": 128.0,
                    "train_nodes_per_batch_mean": 512.0,
                },
                "runtime_efficiency": {
                    "runtime_data_mode": "resident",
                    "runtime_bucket_mode": "exact_then_bucketed",
                    "runtime_workers_per_mig": 1,
                    "resident_store_build_time_s": 0.02,
                    "steady_state_h2d_bytes": 128.0,
                    "steady_state_h2d_time_s": 0.001,
                    "resident_store_hits": 8,
                    "resident_store_misses": 0,
                    "cpu_fallback_reason_counts": {},
                },
                "train": {"root_mae": 0.2},
                "val": {"root_mae": 0.1, "exact_match": 0.9},
            }
        },
    }
    _write_json(tmp_path / "batch" / "bs0128" / "summary.json", batch_summary)
    batch_rows = _aggregate_batch_timing(
        [{"output_path": str(tmp_path / "batch" / "bs0128" / "summary.json")}]
    )
    assert batch_rows["summary"][0]["batch_size"] == 128
    assert batch_rows["summary"][0]["docs_per_s_wall"] == 500.0
    assert batch_rows["summary"][0]["exact_metric_eval_s"] == 0.06
    assert batch_rows["runtime_efficiency"]["runtime_data_mode"] == "resident"
    assert batch_rows["runtime_efficiency"]["resident_store_hits_total"] == 8

    medium_summary = {
        "config": {"train_docs": 10240},
        "runs": {
            "no_autotune": {
                "wall_clock_s": 5.0,
                "epochs_completed": 5,
                "best_val_mae": 0.01,
                "timing_breakdown": {
                    "exact_metric_eval_s": 0.04,
                    "eval_total_s": 0.09,
                },
                "batching_metrics": {"gpu_reserved_mem_peak_gb": 0.3},
                "runtime_efficiency": {
                    "runtime_data_mode": "resident",
                    "runtime_bucket_mode": "exact_then_bucketed",
                    "resident_store_build_time_s": 0.03,
                    "steady_state_h2d_bytes": 64.0,
                    "steady_state_h2d_time_s": 0.0005,
                },
                "val": {"root_mae": 0.011},
            }
        },
    }
    _write_json(tmp_path / "medium" / "bs0256_seed0" / "summary.json", medium_summary)
    medium = _aggregate_medium_grid(
        [{"output_path": str(tmp_path / "medium" / "bs0256_seed0" / "summary.json")}]
    )
    assert medium["by_batch_size"]["256"]["mean_best_val_mae"] == 0.01
    assert medium["by_batch_size"]["256"]["best_run"] == "bs0256_seed0"
    assert medium["by_batch_size"]["256"]["mean_resident_store_build_time_s"] == 0.03
    assert medium["train_docs"] == 10240
    assert medium["epochs"] == 5

    docs_summary = {
        "config": {"train_docs": 2048},
        "runs": {
            "no_autotune": {
                "wall_clock_s": 4.0,
                "epochs_completed": 2,
                "best_val_mae": 0.02,
                "timing_breakdown": {
                    "train_loop_s": 3.0,
                    "exact_metric_eval_s": 0.05,
                    "eval_total_s": 0.1,
                },
                "batching_metrics": {"gpu_reserved_mem_peak_gb": 0.4},
                "train": {"root_mae": 0.03},
                "val": {"root_mae": 0.02, "exact_match": 0.95},
            }
        },
    }
    _write_json(tmp_path / "docs" / "train02048_ep02" / "summary.json", docs_summary)
    docs = _aggregate_docs_epochs(
        [{"output_path": str(tmp_path / "docs" / "train02048_ep02" / "summary.json")}]
    )
    assert docs["rows"][0]["train_docs"] == 2048
    assert docs["by_train_docs"]["2048"]["best_val_epochs"] == 2


def test_ops_payload_aggregators_smoke() -> None:
    payloads = [
        {
            "config": {
                "train_docs": 2048,
                "data_seed": 0,
                "state_dim": 32,
                "hidden_dim": 64,
                "pipeline_law_package_name": "tree_all_laws",
            },
            "objective": {
                "local_law_weight": 1.0,
                "local_law_c1_share": 0.0,
                "local_law_c2_share": 1.0,
                "local_law_c3_share": 0.0,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 1.0,
                "local_law_c3_weight": 0.0,
            },
            "metrics": {
                "learned": {
                    "root_mae": 0.2,
                    "leaf_mae": 0.3,
                    "merge_mae": 0.4,
                    "c2_idempotence_mae": 0.05,
                    "train_root_mae": 0.18,
                    "train_leaf_mae": 0.28,
                    "train_merge_mae": 0.38,
                    "epochs_completed": 5,
                    "training_selection_best_epoch": 3,
                },
                "fno": {"root_mae": 0.12},
                "fno_train": {"root_mae": 0.1},
                "fno_training": {"best_epoch": 4},
            },
            "wall_clock_s": 12.0,
        },
        {
            "config": {
                "train_docs": 2048,
                "data_seed": 0,
                "state_dim": 32,
                "hidden_dim": 64,
            },
            "objective": {
                "local_law_weight": 0.0,
                "local_law_c1_share": 0.0,
                "local_law_c2_share": 0.0,
                "local_law_c3_share": 0.0,
            },
            "metrics": {"learned": {"root_mae": 0.5}},
        },
        {
            "config": {
                "train_docs": 2048,
                "data_seed": 0,
                "state_dim": 32,
                "hidden_dim": 64,
            },
            "objective": {
                "local_law_weight": 1.0,
                "local_law_c1_share": 0.0,
                "local_law_c2_share": 1.0,
                "local_law_c3_share": 0.0,
            },
            "metrics": {"learned": {"root_mae": 0.3}},
        },
    ]

    weight = _aggregate_weight_ablation_from_payloads(payloads)
    assert weight["n_total"] == 3
    assert weight["profile_summaries"][0]["profile"] == "root_only"
    assert weight["matched_summaries"][0]["profile"] == "pure_c2"

    laws = _aggregate_law_packages_from_payloads(payloads[:1])
    assert laws["tree_all_laws"]["test_root_mae"] == 0.2
    assert laws["tree_all_laws"]["doc_fno_test_root_mae"] == 0.12
    assert laws["tree_all_laws"]["doc_fno_best_epoch"] == 4
    assert laws["tree_all_laws"]["wall_seconds"] == 12.0


def test_full_doc_and_large_batch_aggregators_smoke(tmp_path: Path) -> None:
    upper = _aggregate_full_doc_upper_bound_from_payloads(
        [
            {
                "benchmark": "pipeline_current_markov",
                "template_benchmark": "recoverable_v4",
                "rows": [
                    {
                        "baseline_family": "official_fno",
                        "train_doc_count": 1024,
                        "test_root_mae": 0.11,
                        "train_root_mae": 0.10,
                        "val_root_mae": 0.12,
                        "family_wall_clock_s": 4.0,
                        "best_epoch": 8,
                    },
                    {
                        "baseline_family": "official_fno_sumlen",
                        "train_doc_count": 1024,
                        "test_root_mae": 0.09,
                        "train_root_mae": 0.08,
                        "val_root_mae": 0.10,
                        "family_wall_clock_s": 5.0,
                        "best_epoch": 9,
                    },
                ],
            }
        ]
    )
    assert upper["rows"][0]["best_full_doc_fno_family"] == "official_fno_sumlen"
    assert upper["rows"][0]["second_best_full_doc_fno_family"] == "official_fno"

    for name, batch_size, epochs, lr, best_val in (
        ("fixed_epoch__bs0256__ep05__lr0.0010", 256, 5, 0.001, 0.01),
        ("constant_steps__bs0512__ep10__lr0.0010", 512, 10, 0.001, 0.012),
        ("constant_steps__bs1024__ep20__lr0.0010", 1024, 20, 0.001, 0.013),
        ("retune_1024__bs1024__ep20__lr0.0020", 1024, 20, 0.002, 0.011),
    ):
        payload = {
            "config": {"train_docs": 10240, "batch_size": batch_size, "lr": lr},
            "runs": {
                "no_autotune": {
                    "wall_clock_s": 10.0,
                    "epochs_completed": epochs,
                    "best_epoch": epochs - 1,
                    "best_val_mae": best_val,
                    "timing_breakdown": {"train_loop_s": 8.0},
                    "batching_metrics": {"gpu_reserved_mem_peak_gb": 0.5},
                }
            },
        }
        _write_json(tmp_path / "diag" / name / "summary.json", payload)
    large = _aggregate_large_batch_diagnosis(
        [
            {"name": "fixed_epoch__bs0256__ep05__lr0.0010", "output_path": str(tmp_path / "diag" / "fixed_epoch__bs0256__ep05__lr0.0010" / "summary.json")},
            {"name": "constant_steps__bs0512__ep10__lr0.0010", "output_path": str(tmp_path / "diag" / "constant_steps__bs0512__ep10__lr0.0010" / "summary.json")},
            {"name": "constant_steps__bs1024__ep20__lr0.0010", "output_path": str(tmp_path / "diag" / "constant_steps__bs1024__ep20__lr0.0010" / "summary.json")},
            {"name": "retune_1024__bs1024__ep20__lr0.0020", "output_path": str(tmp_path / "diag" / "retune_1024__bs1024__ep20__lr0.0020" / "summary.json")},
        ]
    )
    by_run = {row["run"]: row for row in large["rows"]}
    assert by_run["fixed_epoch__bs0256__ep05__lr0.0010"]["total_optimizer_steps"] == 200
    assert by_run["constant_steps__bs0512__ep10__lr0.0010"]["total_optimizer_steps"] == 200
    assert by_run["constant_steps__bs1024__ep20__lr0.0010"]["total_optimizer_steps"] == 200
    assert large["classification"] == "update_budget_limited"


def test_law_package_phase_uses_current_standard_doc_scale(tmp_path: Path) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=10240,
        val_docs=1024,
        test_docs=1024,
        law_batch_size=256,
        law_epochs=10,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
    )
    tasks, _phase_root = _build_law_package_phase(args, tmp_path)
    assert tasks
    task_request = json.loads(
        (tasks[0].output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    config = dict(task_request.get("config") or {})
    assert config["train_docs"] == 10240
    assert config["val_docs"] == 1024
    assert config["test_docs"] == 1024
    assert config["include_fno_baseline"] is True


def test_pipeline_selection_config_applies_defaults(tmp_path: Path, monkeypatch) -> None:
    capacity_root = _write_capacity_locked_tree_reference(tmp_path)
    config_path = tmp_path / "selection.toml"
    config_path.write_text(
        "\n".join(
            [
                "[tradeoff_pipeline]",
                'preset = "smoke"',
                'phases = ["law_packages", "support_grid"]',
                "train_docs = 4096",
                'law_package_names = ["tree_c2_only"]',
                'support_modes = ["supported"]',
                "",
                "[tradeoff_pipeline.tree_reference]",
                'mode = "capacity_locked"',
                f'capacity_root = "{capacity_root}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_markov_optimization_tradeoff_pipeline.py",
            "--selection-config",
            str(config_path),
        ],
    )
    args = _parse_args()
    assert args.preset == "smoke"
    assert args.phases == "law_packages support_grid"
    assert args.train_docs == 4096
    assert args.law_package_names == "tree_c2_only"
    assert args.support_modes == "supported"
    assert args.tree_reference_mode == "capacity_locked"
    assert Path(args.tree_reference_capacity_root) == capacity_root


def test_pipeline_selection_config_applies_tree_reference_preset(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "selection.toml"
    config_path.write_text(
        "\n".join(
            [
                "[tradeoff_pipeline]",
                'preset = "standard"',
                'phases = ["supervision_recovery"]',
                "exact_metric_final_doc_limit = 128",
                "tree_posttrain_train_doc_limit = 96",
                "",
                "[tradeoff_pipeline.tree_reference]",
                'mode = "preset"',
                'preset = "recoverable_slotwise_dense_v1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_markov_optimization_tradeoff_pipeline.py",
            "--selection-config",
            str(config_path),
        ],
    )
    args = _parse_args()
    assert args.tree_reference_mode == "preset"
    assert args.tree_reference_preset == "recoverable_slotwise_dense_v1"
    assert args.exact_metric_final_doc_limit == 128
    assert args.tree_posttrain_train_doc_limit == 96


def test_phase_builders_respect_selected_packages_and_support_modes(tmp_path: Path) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=10240,
        val_docs=1024,
        test_docs=1024,
        law_batch_size=256,
        law_epochs=10,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        law_package_names="c2_only all_laws",
        support_leaf_tokens="8",
        support_seeds="0",
        support_modes="supported",
        support_batch_size=256,
        support_epochs=5,
    )
    law_tasks, _ = _build_law_package_phase(args, tmp_path / "law")
    assert [task.output_path.parent.name for task in law_tasks] == ["tree_c2_only", "tree_all_laws"]
    support_tasks, _ = _build_support_phase(args, tmp_path / "support")
    assert all("supported" in task.output_path.parent.name for task in support_tasks)
    assert all("unsupported" not in task.output_path.parent.name for task in support_tasks)


def test_capacity_locked_tree_reference_overrides_tree_study_configs(tmp_path: Path) -> None:
    capacity_root = _write_capacity_locked_tree_reference(tmp_path)
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=10240,
        val_docs=1024,
        test_docs=1024,
        law_batch_size=256,
        law_epochs=10,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        law_package_names="tree_all_laws",
        support_leaf_tokens="8",
        support_seeds="0",
        support_modes="supported",
        support_batch_size=256,
        support_epochs=5,
        tree_reference_mode="capacity_locked",
        tree_reference_capacity_root=capacity_root,
    )
    law_tasks, _ = _build_law_package_phase(args, tmp_path / "law")
    law_request = json.loads((law_tasks[0].output_path.parent / "task.request").read_text(encoding="utf-8"))
    law_config = dict(law_request.get("config") or {})
    assert law_config["tree_leaf_fno_width"] == 128
    assert law_config["tree_leaf_fno_n_modes"] == 4
    assert law_config["tree_leaf_fno_n_layers"] == 4
    assert law_config["fixed_leaf_tokens"] == 16
    assert "tree_aux_doc_sequence_fraction" not in law_config

    support_tasks, _ = _build_support_phase(args, tmp_path / "support")
    support_request = json.loads((support_tasks[0].output_path.parent / "task.request").read_text(encoding="utf-8"))
    support_config = dict(support_request.get("config") or {})
    assert support_config["tree_leaf_fno_n_modes"] == 4
    assert support_config["fixed_leaf_tokens"] == 8
    assert "tree_aux_doc_sequence_fraction" not in support_config


def test_capacity_locked_tree_reference_preserves_supervision_sweep_axes(tmp_path: Path) -> None:
    capacity_root = _write_capacity_locked_tree_reference(tmp_path)
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=10240,
        val_docs=1024,
        test_docs=1024,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="capacity_locked",
        tree_reference_capacity_root=capacity_root,
        supervision_train_docs="1024",
        supervision_leaf_profiles="none",
        supervision_internal_profiles="count_q50",
        supervision_seeds="0",
        supervision_batch_size=256,
        supervision_epochs=10,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        runtime_data_mode="resident",
        runtime_bucket_mode="exact_then_bucketed",
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )
    tasks, _ = _build_supervision_phase(args, tmp_path / "supervision")
    request = json.loads((tasks[0].output_path.parent / "task.request").read_text(encoding="utf-8"))
    config = dict(request.get("config") or {})
    assert config["tree_leaf_fno_width"] == 128
    assert config["fixed_leaf_tokens"] == 16
    assert config["leaf_supervision_kind"] == "count_only"
    assert config["leaf_label_rate"] == 0.0
    assert config["internal_supervision_kind"] == "count_only"
    assert config["internal_label_rate"] == 0.5
    assert config["gpu_runtime_workers_per_mig"] == 1
    assert config["min_tokens"] == 64
    assert config["max_tokens"] == 64
    assert config["min_segments"] == 2
    assert config["max_segments"] == 6
    assert config["fixed_leaf_tokens"] == 16


def test_supervision_aggregator_reports_best_rows() -> None:
    payloads = [
        {
            "config": {
                "train_docs": 1024,
                "data_seed": 0,
                "pipeline_supervision_leaf_profile": "none",
                "pipeline_supervision_internal_profile": "none",
                "leaf_supervision_kind": "count_only",
                "leaf_label_rate": 0.0,
                "internal_supervision_kind": "none",
                "internal_label_rate": 0.0,
            },
            "metrics": {"learned_test": {"test_root_mae": 0.20, "val_root_mae": 0.21}},
            "wall_clock_s": 10.0,
        },
        {
            "config": {
                "train_docs": 1024,
                "data_seed": 1,
                "pipeline_supervision_leaf_profile": "full_q100",
                "pipeline_supervision_internal_profile": "count_q50",
                "leaf_supervision_kind": "full_sketch",
                "leaf_label_rate": 1.0,
                "internal_supervision_kind": "count_only",
                "internal_label_rate": 0.5,
            },
            "metrics": {"learned_test": {"test_root_mae": 0.05, "val_root_mae": 0.06}},
            "wall_clock_s": 12.0,
        },
    ]
    summary = _aggregate_supervision_sweep_from_payloads(payloads)
    assert summary["best_overall"]["leaf_profile"] == "full_q100"
    assert summary["best_by_train_docs"]["1024"]["internal_profile"] == "count_q50"


def test_supervision_recovery_phase_respects_capacity_locked_tree_reference(tmp_path: Path) -> None:
    capacity_root = _write_capacity_locked_tree_reference(tmp_path)
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="capacity_locked",
        tree_reference_capacity_root=capacity_root,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="structural_factorized_sketch_v3",
        supervision_batch_size=256,
        supervision_epochs=10,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )
    tasks, _ = _build_supervision_recovery_phase(args, tmp_path / "focused")
    fno_task = next(
        task
        for task in tasks
        if task.name.startswith("recoverable_v4__")
        and "__fno__" in task.name
    )
    tree_task = next(
        task
        for task in tasks
        if task.name.startswith("recoverable_v4__")
        and "__tree_neural__" in task.name
    )
    fno_request = json.loads((fno_task.output_path.parent / "task.request").read_text(encoding="utf-8"))
    tree_request = json.loads((tree_task.output_path.parent / "task.request").read_text(encoding="utf-8"))
    fno_config = dict(fno_request.get("config") or {})
    tree_config = dict(tree_request.get("config") or {})

    assert fno_request["baseline_families"] == ["official_fno", "official_fno_sumlen"]
    assert tree_request["baseline_families"] == ["tree_neural"]
    assert fno_config["pipeline_supervision_recovery_package"] == "full100"
    assert tree_config["pipeline_supervision_recovery_package"] == "full100"
    assert tree_config["state_dim"] == 128
    assert tree_config["hidden_dim"] == 512
    assert tree_config["tree_leaf_fno_width"] == 128
    assert tree_config["tree_leaf_fno_n_modes"] == 4
    assert tree_config["tree_leaf_fno_n_layers"] == 4
    assert tree_config["tree_training_schedule"] == "two_stage"
    assert tree_config["tree_stage1_epochs"] == 12
    assert tree_config["tree_stage2_epochs"] == 20
    assert tree_config["tree_model_version"] == "v2"
    assert tree_config["tree_batch_runtime_mode"] == "unified_v2"
    assert tree_config["tree_batch_pack_mode"] == "fixed_fused"
    assert tree_config["tree_batch_autotune"] is False
    assert tree_config["gpu_runtime_bucket_mode"] == "leaf_count_auto_queue"
    assert tree_config["tree_batch_structural_pad_limit"] == 0.5
    assert tree_config["tree_batch_auto_queue_min_docs"] == 8
    assert tree_config["tree_batch_auto_queue_min_fill_ratio"] == 0.5
    assert tree_config["leaf_supervision_kind"] == "count_only"
    assert tree_config["leaf_label_rate"] == 0.0
    assert tree_config["internal_supervision_kind"] == "none"
    assert tree_config["internal_label_rate"] == 0.0
    assert tree_config["doc_consumption_mode"] == "root_only"
    assert tree_config["budget_total_calls_per_doc"] == 1.0


def test_supervision_recovery_phase_supports_package_capacity_locked_tree_reference(
    tmp_path: Path,
) -> None:
    capacity_root = _write_package_capacity_locked_tree_references(
        tmp_path,
        package_configs={
            "full10": {
                "winning_label": "full10_cfg",
                "state_dim": 128,
                "hidden_dim": 384,
                "tree_leaf_fno_width": 128,
                "tree_leaf_fno_n_modes": 2,
            },
            "full10_leaf_count20_internal_count20": {
                "winning_label": "full10_leaf20_cfg",
                "state_dim": 160,
                "hidden_dim": 640,
                "tree_leaf_fno_width": 160,
                "tree_leaf_fno_n_modes": 6,
            },
        },
    )
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="package_capacity_locked",
        tree_reference_capacity_root=capacity_root,
        structural_tree_reference_mode="package_capacity_locked",
        structural_tree_reference_capacity_root=capacity_root,
        structural_tree_reference_preset="",
        supervision_recovery_packages="full10 full10_leaf_count20_internal_count20",
        supervision_batch_size=256,
        supervision_epochs=10,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )
    tasks, _ = _build_supervision_recovery_phase(args, tmp_path / "package_locked")
    full10_task = next(
        task
        for task in tasks
        if task.name.startswith("recoverable_v4__train01024__full10__tree_neural__d0")
    )
    leaf20_task = next(
        task
        for task in tasks
        if task.name.startswith(
            "recoverable_v4__train01024__full10_leaf_count20_internal_count20__tree_neural__d0"
        )
    )
    structural_leaf20_task = next(
        task
        for task in tasks
        if task.name.startswith(
            "r12_seg10to12__train01024__full10_leaf_count20_internal_count20__tree_neural__d0"
        )
    )
    full10_request = json.loads(
        (full10_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    leaf20_request = json.loads(
        (leaf20_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_leaf20_request = json.loads(
        (structural_leaf20_task.output_path.parent / "task.request").read_text(
            encoding="utf-8"
        )
    )
    full10_config = dict(full10_request.get("config") or {})
    leaf20_config = dict(leaf20_request.get("config") or {})
    structural_leaf20_config = dict(structural_leaf20_request.get("config") or {})

    assert full10_config["pipeline_tree_reference_mode"] == "package_capacity_locked"
    assert leaf20_config["pipeline_tree_reference_mode"] == "package_capacity_locked"
    assert full10_config["state_dim"] == 128
    assert full10_config["tree_leaf_fno_width"] == 128
    assert leaf20_config["state_dim"] == 160
    assert leaf20_config["tree_leaf_fno_width"] == 160
    assert leaf20_config["leaf_supervision_kind"] == "count_only"
    assert leaf20_config["leaf_label_rate"] == 0.2
    assert leaf20_config["internal_supervision_kind"] == "count_only"
    assert leaf20_config["internal_label_rate"] == 0.2
    assert structural_leaf20_config["state_dim"] == 160
    assert (
        structural_leaf20_config["pipeline_tree_reference_label"]
        == "full10_leaf20_cfg"
    )


def test_supervision_recovery_phase_supports_strong_tree_reference_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "SUPERVISION_RECOVERY_PACKAGE_ORDER",
        pipeline.SUPERVISION_RECOVERY_PACKAGE_ORDER
        + ("full100_leaf_full100_internal_count100",),
    )
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="recoverable_slotwise_dense_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="structural_factorized_sketch_v3",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )
    tasks, _ = _build_supervision_recovery_phase(args, tmp_path / "preset")
    recoverable_task = next(
        task
        for task in tasks
        if task.name.startswith("recoverable_v4__train01024__full100__tree_neural__d0")
    )
    structural_task = next(
        task
        for task in tasks
        if task.name.startswith("r12_seg10to12__train01024__full100__tree_neural__d0")
    )
    structural_dense_tree_task = next(
        task
        for task in tasks
        if task.name.startswith(
            "r12_seg10to12__train01024__full100_leaf_full100_internal_count100__tree_neural__d0"
        )
    )
    structural_sparse_task = next(
        task
        for task in tasks
        if task.name.startswith(
            "r12_seg10to12__train01024__full10_leaf_full100_internal_count100__tree_neural__d0"
        )
    )
    recoverable_request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_dense_tree_request = json.loads(
        (structural_dense_tree_task.output_path.parent / "task.request").read_text(
            encoding="utf-8"
        )
    )
    structural_sparse_request = json.loads(
        (structural_sparse_task.output_path.parent / "task.request").read_text(
            encoding="utf-8"
        )
    )
    tree_config = dict(recoverable_request.get("config") or {})
    structural_config = dict(structural_request.get("config") or {})
    structural_dense_tree_config = dict(structural_dense_tree_request.get("config") or {})
    structural_sparse_config = dict(structural_sparse_request.get("config") or {})

    assert tree_config["state_dim"] == 128
    assert tree_config["hidden_dim"] == 512
    assert tree_config["fixed_leaf_tokens"] == 16
    assert tree_config["batch_size"] == 64
    assert tree_config["lr"] == pytest.approx(5e-4)
    assert tree_config["weight_decay"] == pytest.approx(0.0)
    assert tree_config["task_objective_weight"] == pytest.approx(1.0)
    assert tree_config["local_law_weight"] == pytest.approx(0.8)
    assert tree_config["tree_root_supervision_kind"] == "mse"
    assert tree_config["tree_task_head_mode"] == "theorem_feature_scalar"
    assert tree_config["tree_theorem_surface_mode"] == "slotwise"
    assert tree_config["tree_summary_spec_root_mode"] == "factored_theorem_readout"
    assert tree_config["summary_spec_name"] == "markov_count_sketch"
    assert tree_config["slot_count"] == 4
    assert tree_config["tree_document_loss_normalization_mode"] == "auto"
    assert tree_config["tree_theorem_feature_dim"] == 48
    assert tree_config["tree_theorem_feature_hidden_dim"] == 256
    assert tree_config["tree_leaf_fno_width"] == 128
    assert tree_config["tree_leaf_fno_n_modes"] == 8
    assert tree_config["tree_leaf_fno_n_layers"] == 4
    assert tree_config["tree_checkpoint_metric"] == "val_exact_sketch_direct"
    assert tree_config["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"
    assert tree_config["tree_stage1_epochs"] == 12
    assert tree_config["tree_stage2_epochs"] == 40
    assert tree_config["tree_join_bit_weight"] == pytest.approx(1.0)
    assert tree_config["tree_phi_compose_weight"] == pytest.approx(0.0)
    assert tree_config["tree_phi_contrastive_weight"] == pytest.approx(0.0)
    assert tree_config["exact_metric_final_doc_limit"] == 128
    assert tree_config["tree_posttrain_train_doc_limit"] == 96
    assert tree_config["tree_batch_pack_mode"] == "fixed_fused"
    assert tree_config["gpu_runtime_bucket_mode"] == "leaf_count_auto_queue"
    assert tree_config["leaf_supervision_kind"] == "count_only"
    assert tree_config["leaf_label_rate"] == 0.0
    assert tree_config["internal_supervision_kind"] == "none"
    assert tree_config["internal_label_rate"] == 0.0
    assert recoverable_task.metadata["n_epochs"] == 52
    assert structural_request["benchmark_name"] == "structural_core_v1::r12_seg10to12"
    assert structural_request["hardness_grid"] == "structural_core_v1"
    assert structural_request["grid_cell_ids"] == ["r12_seg10to12"]
    assert structural_config["pipeline_tree_reference_label"] == "structural_factorized_sketch_v3"
    assert structural_config["pipeline_tree_scope_kind"] == "structural"
    assert structural_config["tree_training_schedule"] == "two_stage"
    assert structural_config["tree_task_head_mode"] == "theorem_feature_scalar"
    assert structural_config["tree_theorem_surface_mode"] == "factorized_score_fiber"
    assert structural_config["tree_summary_spec_root_mode"] == "factored_theorem_readout"
    assert structural_config["tree_root_supervision_kind"] == "mse"
    assert structural_config["tree_checkpoint_metric"] == "val_root_mae"
    assert structural_config["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"
    assert structural_config["tree_theorem_feature_dim"] == 48
    assert structural_config["tree_theorem_feature_hidden_dim"] == 256
    assert structural_config["tree_theorem_score_dim"] == 1
    assert structural_config["tree_theorem_fiber_dim"] == 47
    assert structural_config["summary_spec_name"] == "markov_count_sketch"
    assert structural_config["slot_count"] == 4
    assert structural_config["tree_theorem_count_dim"] == 8
    assert structural_config["tree_theorem_first_dim"] == 8
    assert structural_config["tree_theorem_last_dim"] == 8
    assert structural_config["tree_stage1_epochs"] == 12
    assert structural_config["tree_stage2_epochs"] == 40
    assert structural_dense_tree_config["tree_checkpoint_metric"] == "val_root_mae"
    assert structural_sparse_config["tree_checkpoint_metric"] == "val_exact_sketch_direct"
    assert structural_task.metadata["n_epochs"] == 52


def test_supervision_recovery_phase_common_factorized_uses_root_checkpoint_for_all_ladder_points(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="common_factorized_sketch_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="common_factorized_sketch_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(args, tmp_path / "common")
    expected_names = [
        "recoverable_v4__train01024__full50__tree_neural__d0",
        "recoverable_v4__train01024__full50_leaf_full100_internal_count100__tree_neural__d0",
        "r12_seg10to12__train01024__full50__tree_neural__d0",
        "r12_seg10to12__train01024__full50_leaf_full100_internal_count100__tree_neural__d0",
    ]
    requests = []
    for name in expected_names:
        task = next(task for task in tasks if task.name == name)
        requests.append(
            json.loads((task.output_path.parent / "task.request").read_text(encoding="utf-8"))
        )

    for request in requests:
        config = dict(request.get("config") or {})
        assert config["pipeline_tree_reference_label"] == "common_factorized_sketch_v1"
        assert config["tree_checkpoint_metric"] == "val_root_mae"
        assert config["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"


def test_supervision_recovery_phase_supports_r10_local_law_rate_package(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="common_factorized_sketch_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="common_factorized_sketch_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "local_law_rate",
        package_order=["full10_leaf_count20_internal_count20"],
    )

    assert len(tasks) == 2
    assert all("__fno__" not in task.name for task in tasks)

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__full10_leaf_count20_internal_count20__tree_neural__d0"
    )
    request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    config = dict(request.get("config") or {})

    assert config["pipeline_supervision_recovery_package"] == "full10_leaf_count20_internal_count20"
    assert config["budget_total_calls_per_doc"] == pytest.approx(0.1)
    assert config["doc_consumption_mode"] == "root_only"
    assert config["leaf_supervision_kind"] == "count_only"
    assert config["leaf_label_rate"] == pytest.approx(0.2)
    assert config["internal_supervision_kind"] == "count_only"
    assert config["internal_label_rate"] == pytest.approx(0.2)
    assert config["max_internal_depth"] == 0
    assert config["pipeline_tree_reference_label"] == "common_factorized_sketch_v1"
    assert config["tree_checkpoint_metric"] == "val_root_mae"
    assert config["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"


def test_supervision_recovery_phase_supports_r20_local_law_rate_package(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="common_factorized_sketch_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="common_factorized_sketch_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "local_law_rate_r20",
        package_order=["full20_leaf_count50_internal_count50"],
    )

    assert len(tasks) == 2
    assert all("__fno__" not in task.name for task in tasks)

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__full20_leaf_count50_internal_count50__tree_neural__d0"
    )
    request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    config = dict(request.get("config") or {})

    assert config["pipeline_supervision_recovery_package"] == "full20_leaf_count50_internal_count50"
    assert config["budget_total_calls_per_doc"] == pytest.approx(0.2)
    assert config["doc_consumption_mode"] == "root_only"
    assert config["leaf_supervision_kind"] == "count_only"
    assert config["leaf_label_rate"] == pytest.approx(0.5)
    assert config["internal_supervision_kind"] == "count_only"
    assert config["internal_label_rate"] == pytest.approx(0.5)
    assert config["max_internal_depth"] == 0
    assert config["pipeline_tree_reference_label"] == "common_factorized_sketch_v1"
    assert config["tree_checkpoint_metric"] == "val_root_mae"
    assert config["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"


@pytest.mark.parametrize(
    ("package_name", "fixed_leaf_tokens", "expected_doc_review_mass"),
    [
        ("r10_mass_local_eq_0p5", 16, 0.08),
        ("r10_mass_local_eq_1p0", 16, 0.06),
        ("r10_mass_local_eq_1p5", 16, 0.04),
        ("r10_mass_local_eq_2p0", 16, 0.02),
        ("r20_mass_local_eq_1p0", 16, 0.16),
        ("r20_mass_local_eq_2p0", 16, 0.12),
        ("r20_mass_local_eq_3p0", 16, 0.08),
        ("r20_mass_local_eq_4p0", 16, 0.04),
        ("r80_mass_local_eq_5p0", 16, 0.60),
        ("r80_mass_local_eq_10p0", 16, 0.40),
        ("r80_mass_local_eq_15p0", 16, 0.20),
        ("r80_mass_local_eq_16p0", 16, 0.16),
        ("r90_mass_local_eq_5p0", 16, 0.70),
        ("r90_mass_local_eq_10p0", 16, 0.50),
        ("r90_mass_local_eq_15p0", 16, 0.30),
        ("r90_mass_local_eq_18p0", 16, 0.18),
        ("r100_mass_local_eq_5p0", 16, 0.80),
        ("r100_mass_local_eq_10p0", 16, 0.60),
        ("r100_mass_local_eq_15p0", 16, 0.40),
        ("r100_mass_local_eq_20p0", 16, 0.20),
        ("r10_mass_local_eq_0p5", 8, 0.075),
        ("r10_mass_local_eq_1p0", 8, 0.05),
        ("r10_mass_local_eq_1p5", 8, 0.025),
        ("r10_mass_local_eq_2p0", 8, 0.0),
        ("r20_mass_local_eq_1p0", 8, 0.15),
        ("r20_mass_local_eq_2p0", 8, 0.10),
        ("r20_mass_local_eq_3p0", 8, 0.05),
        ("r20_mass_local_eq_4p0", 8, 0.0),
        ("r80_mass_local_eq_5p0", 8, 0.55),
        ("r80_mass_local_eq_10p0", 8, 0.30),
        ("r80_mass_local_eq_15p0", 8, 0.05),
        ("r80_mass_local_eq_16p0", 8, 0.0),
        ("r90_mass_local_eq_5p0", 8, 0.65),
        ("r90_mass_local_eq_10p0", 8, 0.40),
        ("r90_mass_local_eq_15p0", 8, 0.15),
        ("r90_mass_local_eq_18p0", 8, 0.0),
        ("r100_mass_local_eq_5p0", 8, 0.75),
        ("r100_mass_local_eq_10p0", 8, 0.50),
        ("r100_mass_local_eq_15p0", 8, 0.25),
        ("r100_mass_local_eq_20p0", 8, 0.0),
        ("r100_mass_local_eq_10p0", 128, 0.90),
        ("r100_mass_local_eq_15p0", 128, 0.85),
        ("r100_mass_local_eq_20p0", 128, 0.80),
    ],
)
def test_mass_matched_supervision_recovery_package_resolves_residual_root_mass(
    package_name: str,
    fixed_leaf_tokens: int,
    expected_doc_review_mass: float,
) -> None:
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        package_name,
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name],
        min_tokens=128,
        max_tokens=128,
        fixed_leaf_tokens=fixed_leaf_tokens,
        scope_key="recoverable_v4" if fixed_leaf_tokens == 16 else "r12_seg10to12",
    )

    assert resolved_spec["doc_consumption_mode"] == "root_only"
    assert resolved_spec["full_doc_budget_share"] == pytest.approx(1.0)
    assert resolved_spec["budget_total_calls_per_doc"] == pytest.approx(
        expected_doc_review_mass
    )
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(
        expected_doc_review_mass
    )
    assert accounting["computed_total_mass_per_doc"] == pytest.approx(
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name]["mass_target_per_doc"]
    )


def test_mass_matched_supervision_recovery_package_uses_surface_geometry() -> None:
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        "r100_mass_local_eq_15p0",
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS["r100_mass_local_eq_15p0"],
        min_tokens=96,
        max_tokens=96,
        fixed_leaf_tokens=32,
        scope_key="r12_seg10to12",
    )

    assert accounting["assumed_doc_tokens"] == 96
    assert resolved_spec["budget_total_calls_per_doc"] == pytest.approx(0.6)
    assert accounting["computed_local_mass_per_doc"] == pytest.approx(0.4)
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("fixed_leaf_tokens", "expected_rate"),
    [
        (64, 0.25),
        (32, 1.0 / 6.0),
        (16, 0.125),
        (8, 0.1),
    ],
)
def test_node_mass_target_package_preserves_exact_split_across_geometries(
    fixed_leaf_tokens: int,
    expected_rate: float,
) -> None:
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        "r100_node_mass_eq_50p0",
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS["r100_node_mass_eq_50p0"],
        min_tokens=128,
        max_tokens=128,
        fixed_leaf_tokens=fixed_leaf_tokens,
        scope_key="recoverable_v4_t128",
    )

    assert resolved_spec["leaf_label_rate"] == pytest.approx(expected_rate)
    assert resolved_spec["internal_label_rate"] == pytest.approx(expected_rate)
    assert accounting["local_mass_target_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_local_mass_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_total_mass_per_doc"] == pytest.approx(1.0)


def test_node_mass_target_package_can_assign_all_mass_to_nodes() -> None:
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        "r100_node_mass_eq_100p0",
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS["r100_node_mass_eq_100p0"],
        min_tokens=128,
        max_tokens=128,
        fixed_leaf_tokens=64,
        scope_key="recoverable_v4_t128",
    )

    assert resolved_spec["leaf_label_rate"] == pytest.approx(0.5)
    assert resolved_spec["internal_label_rate"] == pytest.approx(0.5)
    assert accounting["computed_local_mass_per_doc"] == pytest.approx(1.0)
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(0.0)
    assert accounting["computed_total_mass_per_doc"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fixed_leaf_tokens", "expected_leaf_rate", "expected_internal_rate", "expected_depth"),
    [
        (64, 0.5, 0.0, 0),
        (32, 0.25, 0.25, 1),
        (16, 1.0 / 6.0, 1.0 / 6.0, 2),
        (8, 0.125, 0.125, 3),
    ],
)
def test_depth_equal_mass_preserving_packages_track_nonroot_depth_geometry(
    fixed_leaf_tokens: int,
    expected_leaf_rate: float,
    expected_internal_rate: float,
    expected_depth: int,
) -> None:
    package_name = "r50_depth_equal_mass_eq_50p0"
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        package_name,
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name],
        min_tokens=128,
        max_tokens=128,
        fixed_leaf_tokens=fixed_leaf_tokens,
        scope_key="recoverable_v4_t128",
    )

    assert resolved_spec["leaf_label_rate"] == pytest.approx(expected_leaf_rate)
    assert resolved_spec["internal_label_rate"] == pytest.approx(expected_internal_rate)
    assert resolved_spec["max_internal_depth"] == expected_depth
    assert accounting["computed_local_mass_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_total_mass_per_doc"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fixed_leaf_tokens", "expected_leaf_rate"),
    [
        (64, 0.5),
        (32, 0.5),
        (16, 0.5),
        (8, 0.5),
    ],
)
def test_leaf_only_mass_preserving_packages_assign_all_local_mass_to_leaves(
    fixed_leaf_tokens: int,
    expected_leaf_rate: float,
) -> None:
    package_name = "r50_leaf_mass_eq_50p0"
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        package_name,
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name],
        min_tokens=128,
        max_tokens=128,
        fixed_leaf_tokens=fixed_leaf_tokens,
        scope_key="recoverable_v4_t128",
    )

    assert resolved_spec["leaf_label_rate"] == pytest.approx(expected_leaf_rate)
    assert resolved_spec["internal_supervision_kind"] == "none"
    assert resolved_spec["internal_label_rate"] == pytest.approx(0.0)
    assert accounting["computed_leaf_mass_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_internal_mass_per_doc"] == pytest.approx(0.0)
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(0.5)
    assert accounting["computed_total_mass_per_doc"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("package_name", "fixed_leaf_tokens", "scope_key", "expected_local_mass"),
    [
        ("r100_superset_local_eq_10p0", 16, "recoverable_v4", 0.4),
        ("r100_superset_local_eq_15p0", 16, "recoverable_v4", 0.6),
        ("r100_superset_local_eq_20p0", 128, "recoverable_v4_t128", 0.2),
    ],
)
def test_superset_supervision_recovery_package_keeps_full_root_mass(
    package_name: str,
    fixed_leaf_tokens: int,
    scope_key: str,
    expected_local_mass: float,
) -> None:
    resolved_spec, accounting = pipeline._resolve_supervision_recovery_package_for_scope(
        package_name,
        pipeline.SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name],
        min_tokens=128,
        max_tokens=128,
        fixed_leaf_tokens=fixed_leaf_tokens,
        scope_key=scope_key,
    )

    assert resolved_spec["package_semantics"] == "superset"
    assert resolved_spec["doc_consumption_mode"] == "root_only"
    assert resolved_spec["full_doc_budget_share"] == pytest.approx(1.0)
    assert resolved_spec["budget_total_calls_per_doc"] == pytest.approx(1.0)
    assert accounting["package_semantics"] == "superset"
    assert accounting["computed_doc_review_mass_per_doc"] == pytest.approx(1.0)
    assert accounting["computed_local_mass_per_doc"] == pytest.approx(expected_local_mass)
    assert accounting["computed_total_mass_per_doc"] == pytest.approx(
        1.0 + expected_local_mass
    )
    assert accounting["mass_target_per_doc"] != accounting["mass_target_per_doc"]


def test_supervision_recovery_accounting_tokens_prefer_bundle_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ctreepo.sim.core.markov_changepoint_ops_count import (
        ChangepointMarkovDoc,
        MarkovOPSDataBundle,
    )

    doc = ChangepointMarkovDoc(
        tokens=tuple(range(96)),
        token_regimes=tuple(0 for _ in range(96)),
        transition_regimes=tuple(0 for _ in range(96)),
        true_boundaries=tuple(),
    )
    bundle = MarkovOPSDataBundle(
        train_docs=(doc,),
        val_docs=(doc,),
        test_docs=(doc,),
        train_corpus_signature="train",
        val_corpus_signature="val",
        test_corpus_signature="test",
    )
    bundle_path = tmp_path / "observed_token_bundle.json"
    bundle.save(bundle_path)

    class _Benchmark:
        canonical_bundle_path = str(bundle_path)
        expanded_bundle_path = ""

    pipeline._resolved_full_doc_bundle_token_geometry.cache_clear()
    monkeypatch.setattr(
        pipeline,
        "_resolved_full_doc_benchmark_spec",
        lambda benchmark_name, hardness_grid, grid_cell_ids: _Benchmark(),
    )

    min_tokens, max_tokens = pipeline._supervision_recovery_accounting_tokens(
        benchmark_name="recoverable_v4",
        hardness_grid="",
        grid_cell_ids=tuple(),
        surfaced_min_tokens=128,
        surfaced_max_tokens=128,
    )

    assert min_tokens == 96
    assert max_tokens == 96
    pipeline._resolved_full_doc_bundle_token_geometry.cache_clear()


def test_supervision_recovery_phase_supports_mass_matched_unified_g_packages(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "mass_matched_unified_g",
        package_order=[
            "full10",
            "r10_mass_local_eq_1p0",
            "full20",
            "r20_mass_local_eq_2p0",
        ],
    )

    assert len(tasks) == 12
    assert len([task for task in tasks if "__fno__" in task.name]) == 4
    assert all(
        not task.name.endswith("__r10_mass_local_eq_1p0__fno__d0")
        and not task.name.endswith("__r20_mass_local_eq_2p0__fno__d0")
        for task in tasks
    )

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__r10_mass_local_eq_1p0__tree_neural__d0"
    )
    recoverable_request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    recoverable_config = dict(recoverable_request.get("config") or {})

    assert recoverable_config["pipeline_supervision_recovery_package"] == "r10_mass_local_eq_1p0"
    assert recoverable_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert recoverable_config["tree_model_version"] == "unified_g"
    assert recoverable_config["tree_score_merge_mode"] == "exact_projected_sketch"
    assert recoverable_config["n_epochs"] == 40
    assert recoverable_config["tree_stage1_epochs"] == 10
    assert recoverable_config["tree_stage2_epochs"] == 30
    assert recoverable_config["fixed_leaf_tokens"] == 16
    assert recoverable_config["mass_target_per_doc"] == pytest.approx(0.1)
    assert recoverable_config["computed_doc_review_mass_per_doc"] == pytest.approx(19.0 / 300.0)
    assert recoverable_config["computed_local_mass_per_doc"] == pytest.approx(11.0 / 300.0)
    assert recoverable_config["computed_leaf_mass_per_doc"] == pytest.approx(0.01)
    assert recoverable_config["computed_internal_mass_per_doc"] == pytest.approx(2.0 / 75.0)
    assert recoverable_config["budget_total_calls_per_doc"] == pytest.approx(19.0 / 300.0)

    structural_task = next(
        task
        for task in tasks
        if task.name
        == "r12_seg10to12__train01024__r20_mass_local_eq_2p0__tree_neural__d0"
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_config = dict(structural_request.get("config") or {})

    assert structural_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert structural_config["tree_model_version"] == "unified_g"
    assert structural_config["fixed_leaf_tokens"] == 8
    assert structural_config["mass_target_per_doc"] == pytest.approx(0.2)
    assert structural_config["computed_doc_review_mass_per_doc"] == pytest.approx(8.0 / 75.0)
    assert structural_config["computed_local_mass_per_doc"] == pytest.approx(7.0 / 75.0)
    assert structural_config["computed_leaf_mass_per_doc"] == pytest.approx(0.02)
    assert structural_config["computed_internal_mass_per_doc"] == pytest.approx(11.0 / 150.0)
    assert structural_config["budget_total_calls_per_doc"] == pytest.approx(8.0 / 75.0)


def test_supervision_recovery_phase_supports_r100_mass_matched_unified_g_packages(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "mass_matched_unified_g_r100",
        package_order=[
            "full100",
            "r100_mass_local_eq_10p0",
        ],
    )

    assert len(tasks) == 6
    assert len([task for task in tasks if "__fno__" in task.name]) == 2
    assert all(
        not task.name.endswith("__r100_mass_local_eq_10p0__fno__d0")
        for task in tasks
    )

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__r100_mass_local_eq_10p0__tree_neural__d0"
    )
    recoverable_request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    recoverable_config = dict(recoverable_request.get("config") or {})
    assert recoverable_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert recoverable_config["tree_model_version"] == "unified_g"
    assert recoverable_config["fixed_leaf_tokens"] == 16
    assert recoverable_config["mass_target_per_doc"] == pytest.approx(1.0)
    assert recoverable_config["computed_doc_review_mass_per_doc"] == pytest.approx(19.0 / 30.0)
    assert recoverable_config["computed_local_mass_per_doc"] == pytest.approx(11.0 / 30.0)
    assert recoverable_config["computed_leaf_mass_per_doc"] == pytest.approx(0.10)
    assert recoverable_config["computed_internal_mass_per_doc"] == pytest.approx(4.0 / 15.0)
    assert recoverable_config["budget_total_calls_per_doc"] == pytest.approx(19.0 / 30.0)

    structural_task = next(
        task
        for task in tasks
        if task.name
        == "r12_seg10to12__train01024__r100_mass_local_eq_10p0__tree_neural__d0"
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_config = dict(structural_request.get("config") or {})
    assert structural_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert structural_config["tree_model_version"] == "unified_g"
    assert structural_config["fixed_leaf_tokens"] == 8
    assert structural_config["mass_target_per_doc"] == pytest.approx(1.0)
    assert structural_config["computed_doc_review_mass_per_doc"] == pytest.approx(8.0 / 15.0)
    assert structural_config["computed_local_mass_per_doc"] == pytest.approx(7.0 / 15.0)
    assert structural_config["computed_leaf_mass_per_doc"] == pytest.approx(0.10)
    assert structural_config["computed_internal_mass_per_doc"] == pytest.approx(11.0 / 30.0)
    assert structural_config["budget_total_calls_per_doc"] == pytest.approx(8.0 / 15.0)


def test_supervision_recovery_phase_supports_r100_superset_unified_g_packages(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "superset_unified_g_r100",
        package_order=[
            "full100",
            "r100_superset_local_eq_10p0",
        ],
    )

    assert len(tasks) == 6
    assert len([task for task in tasks if "__fno__" in task.name]) == 2
    assert all(
        not task.name.endswith("__r100_superset_local_eq_10p0__fno__d0")
        for task in tasks
    )

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__r100_superset_local_eq_10p0__tree_neural__d0"
    )
    recoverable_request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    recoverable_config = dict(recoverable_request.get("config") or {})

    assert recoverable_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert recoverable_config["tree_model_version"] == "unified_g"
    assert recoverable_config["fixed_leaf_tokens"] == 16
    assert recoverable_config["package_semantics"] == "superset"
    assert recoverable_config["budget_total_calls_per_doc"] == pytest.approx(1.0)
    assert recoverable_config["computed_doc_review_mass_per_doc"] == pytest.approx(1.0)
    assert recoverable_config["computed_local_mass_per_doc"] == pytest.approx(11.0 / 30.0)
    assert recoverable_config["computed_total_mass_per_doc"] == pytest.approx(41.0 / 30.0)
    assert recoverable_config["mass_target_per_doc"] != recoverable_config["mass_target_per_doc"]

    structural_task = next(
        task
        for task in tasks
        if task.name
        == "r12_seg10to12__train01024__r100_superset_local_eq_10p0__tree_neural__d0"
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_config = dict(structural_request.get("config") or {})

    assert structural_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert structural_config["tree_model_version"] == "unified_g"
    assert structural_config["fixed_leaf_tokens"] == 8
    assert structural_config["package_semantics"] == "superset"
    assert structural_config["budget_total_calls_per_doc"] == pytest.approx(1.0)
    assert structural_config["computed_doc_review_mass_per_doc"] == pytest.approx(1.0)
    assert structural_config["computed_local_mass_per_doc"] == pytest.approx(7.0 / 15.0)
    assert structural_config["computed_total_mass_per_doc"] == pytest.approx(22.0 / 15.0)
    assert structural_config["mass_target_per_doc"] != structural_config["mass_target_per_doc"]


def test_supervision_recovery_phase_threads_gamma_axis_into_task_names_and_configs(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=96,
        supervision_max_tokens=96,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=32,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_depth_discount_gammas="1.0,0.9",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "gamma_axis",
        package_order=["full100", "r100_mass_local_eq_15p0"],
    )

    assert len(tasks) == 12
    assert any("__g1p00__" in task.name for task in tasks)
    assert any("__g0p90__" in task.name for task in tasks)
    gamma_task = next(
        task for task in tasks if "__g0p90__" in task.name and "__tree_neural__" in task.name
    )
    gamma_request = json.loads(
        (gamma_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    gamma_config = dict(gamma_request.get("config") or {})
    assert gamma_config["depth_discount_gamma"] == pytest.approx(0.9)
    assert gamma_config["pipeline_supervision_recovery_depth_discount_gamma"] == pytest.approx(0.9)


def test_supervision_recovery_aggregation_keeps_gamma_rows_distinct() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "data_seed": 0,
                "train_docs": 10240,
                "depth_discount_gamma": 1.0,
            },
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "test_root_mae_mean": 0.03,
                    "depth_discount_gamma": 1.0,
                    "tree_supervision_source": "manifest",
                    "local_estimand_mode": "span_mass_ipw_sum",
                    "c2_pair_weighting_mode": "pair_ipw_geomean",
                }
            ],
            "runs": [],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "data_seed": 1,
                "train_docs": 10240,
                "depth_discount_gamma": 0.9,
            },
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "test_root_mae_mean": 0.02,
                    "depth_discount_gamma": 0.9,
                    "tree_supervision_source": "manifest",
                    "local_estimand_mode": "span_mass_ipw_sum",
                    "c2_pair_weighting_mode": "pair_ipw_geomean",
                }
            ],
            "runs": [],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full100"],
    )

    rows = [
        row
        for row in list(summary.get("family_rows") or [])
        if str(row.get("baseline_family", "")) == "tree_neural"
    ]
    assert len(rows) == 2
    assert sorted(float(row["depth_discount_gamma"]) for row in rows) == pytest.approx(
        [0.9, 1.0]
    )
    gamma_map = {float(row["depth_discount_gamma"]): row for row in rows}
    assert gamma_map[1.0]["report_row_key"] == "full100__g1p00"
    assert gamma_map[0.9]["report_row_key"] == "full100__g0p90"
    assert gamma_map[1.0]["tree_supervision_source"] == "manifest"
    assert gamma_map[1.0]["local_estimand_mode"] == "span_mass_ipw_sum"
    assert gamma_map[1.0]["c2_pair_weighting_mode"] == "pair_ipw_geomean"
    assert bool(gamma_map[1.0]["is_authoritative_parity_row"]) is True
    assert bool(gamma_map[0.9]["is_authoritative_parity_row"]) is True


def test_supervision_recovery_aggregation_keeps_leaf_geometries_distinct() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "pipeline_supervision_recovery_leaf_tokens": 8,
                "data_seed": 0,
                "train_docs": 10240,
                "depth_discount_gamma": 1.0,
                "fixed_leaf_tokens": 8,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 16,
            },
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "test_root_mae_mean": 0.041,
                    "fixed_leaf_tokens": 8,
                    "executed_fixed_leaf_tokens": 8,
                    "test_mean_leaves_per_doc": 16.0,
                    "depth_discount_gamma": 1.0,
                    "tree_supervision_source": "manifest",
                    "local_estimand_mode": "span_mass_ipw_sum",
                    "c2_pair_weighting_mode": "pair_ipw_geomean",
                }
            ],
            "runs": [],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "pipeline_supervision_recovery_leaf_tokens": 16,
                "data_seed": 1,
                "train_docs": 10240,
                "depth_discount_gamma": 1.0,
                "fixed_leaf_tokens": 16,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 8,
            },
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "test_root_mae_mean": 0.029,
                    "fixed_leaf_tokens": 16,
                    "executed_fixed_leaf_tokens": 16,
                    "test_mean_leaves_per_doc": 8.0,
                    "depth_discount_gamma": 1.0,
                    "tree_supervision_source": "manifest",
                    "local_estimand_mode": "span_mass_ipw_sum",
                    "c2_pair_weighting_mode": "pair_ipw_geomean",
                }
            ],
            "runs": [],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full100"],
    )

    rows = [
        row
        for row in list(summary.get("family_rows") or [])
        if str(row.get("baseline_family", "")) == "tree_neural"
        and str(row.get("scope_key", "")) == "recoverable_v4"
        and str(row.get("package_name", "")) == "full100"
    ]

    assert len(rows) == 2
    assert {row["supervision_recovery_geometry_label"] for row in rows} == {
        "leaf008",
        "leaf016",
    }
    assert {row["supervision_recovery_geometry_key"] for row in rows} == {
        "leaf008__req8__exec8__n16",
        "leaf016__req16__exec16__n8",
    }
    assert {row["report_row_key"] for row in rows} == {
        "full100__leaf008__g1p00",
        "full100__leaf016__g1p00",
    }
    assert {
        int(row["pipeline_supervision_recovery_leaf_tokens"])
        for row in rows
    } == {8, 16}

    scope_rows = summary["scopes"]["recoverable_v4"]["rows_by_train_docs"][0]["rows"]
    assert len(scope_rows) == 2
    assert {row["supervision_recovery_geometry_label"] for row in scope_rows} == {
        "leaf008",
        "leaf016",
    }


def test_supervision_recovery_phase_supports_r80_r90_mass_matched_unified_g_packages(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "mass_matched_unified_g_r80_r90",
        package_order=[
            "full80",
            "r80_mass_local_eq_16p0",
            "full90",
            "r90_mass_local_eq_18p0",
        ],
    )

    assert len(tasks) == 12
    assert len([task for task in tasks if "__fno__" in task.name]) == 4
    assert all(
        "__r80_mass_local_eq_16p0__fno__d0" not in task.name
        and "__r90_mass_local_eq_18p0__fno__d0" not in task.name
        for task in tasks
    )

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__r80_mass_local_eq_16p0__tree_neural__d0"
    )
    recoverable_request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    recoverable_config = dict(recoverable_request.get("config") or {})
    assert recoverable_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert recoverable_config["tree_model_version"] == "unified_g"
    assert recoverable_config["fixed_leaf_tokens"] == 16
    assert recoverable_config["mass_target_per_doc"] == pytest.approx(0.8)
    assert recoverable_config["computed_doc_review_mass_per_doc"] == pytest.approx(16.0 / 75.0)
    assert recoverable_config["computed_local_mass_per_doc"] == pytest.approx(44.0 / 75.0)
    assert recoverable_config["computed_leaf_mass_per_doc"] == pytest.approx(0.16)
    assert recoverable_config["computed_internal_mass_per_doc"] == pytest.approx(32.0 / 75.0)
    assert recoverable_config["budget_total_calls_per_doc"] == pytest.approx(16.0 / 75.0)

    structural_task = next(
        task
        for task in tasks
        if task.name
        == "r12_seg10to12__train01024__r90_mass_local_eq_18p0__tree_neural__d0"
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_config = dict(structural_request.get("config") or {})
    assert structural_config["pipeline_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert structural_config["tree_model_version"] == "unified_g"
    assert structural_config["fixed_leaf_tokens"] == 8
    assert structural_config["mass_target_per_doc"] == pytest.approx(0.9)
    assert structural_config["computed_doc_review_mass_per_doc"] == pytest.approx(0.06)
    assert structural_config["computed_local_mass_per_doc"] == pytest.approx(0.84)
    assert structural_config["computed_leaf_mass_per_doc"] == pytest.approx(0.18)
    assert structural_config["computed_internal_mass_per_doc"] == pytest.approx(0.66)
    assert structural_config["budget_total_calls_per_doc"] == pytest.approx(0.06)


def test_supervision_recovery_leafgrid_preserves_explicit_leaf_tokens_for_unified_g(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=16,
        supervision_recovery_leaf_token_ladder="128 64 32 16 8",
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "leafgrid_unified_g",
        package_order=["full10"],
    )

    assert len(tasks) == 12
    assert len([task for task in tasks if "__fno__" in task.name]) == 2
    assert {
        int(
            json.loads(
                (task.output_path.parent / "task.request").read_text(encoding="utf-8")
            )["config"]["fixed_leaf_tokens"]
        )
        for task in tasks
        if task.name.endswith("__tree_neural__d0")
        and task.name.startswith("recoverable_v4__")
    } == {8, 16, 32, 64, 128}
    assert {
        int(
            json.loads(
                (task.output_path.parent / "task.request").read_text(encoding="utf-8")
            )["config"]["fixed_leaf_tokens"]
        )
        for task in tasks
        if task.name.endswith("__tree_neural__d0")
        and task.name.startswith("r12_seg10to12__")
    } == {8, 16, 32, 64, 128}

    recoverable_task = next(
        task
        for task in tasks
        if task.name
        == "recoverable_v4__train01024__full10__leaf128__tree_neural__d0"
    )
    structural_task = next(
        task
        for task in tasks
        if task.name
        == "r12_seg10to12__train01024__full10__leaf128__tree_neural__d0"
    )
    recoverable_request = json.loads(
        (recoverable_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )

    assert recoverable_request["config"]["fixed_leaf_tokens"] == 128
    assert recoverable_request["config"]["pipeline_supervision_recovery_leaf_tokens"] == 128
    assert recoverable_request["config"]["preserve_requested_leaf_tokens"] is True
    assert (
        recoverable_request["config"]["official_fno_preserve_requested_leaf_tokens"]
        is True
    )
    assert structural_request["config"]["fixed_leaf_tokens"] == 128
    assert structural_request["config"]["pipeline_supervision_recovery_leaf_tokens"] == 128
    assert structural_request["config"]["preserve_requested_leaf_tokens"] is True
    assert (
        structural_request["config"]["official_fno_preserve_requested_leaf_tokens"]
        is True
    )

    assert {
        int(
            json.loads(
                (task.output_path.parent / "task.request").read_text(encoding="utf-8")
            )["config"]["fixed_leaf_tokens"]
        )
        for task in tasks
        if task.name.endswith("__fno__d0")
        and task.name.startswith("recoverable_v4__")
    } == {128}
    assert all(
        bool(
            json.loads(
                (task.output_path.parent / "task.request").read_text(encoding="utf-8")
            )["config"].get("official_fno_preserve_requested_leaf_tokens")
        )
        for task in tasks
        if task.name.endswith("__fno__d0")
    )
    assert all(
        bool(
            json.loads(
                (task.output_path.parent / "task.request").read_text(encoding="utf-8")
            )["config"].get("preserve_requested_leaf_tokens")
        )
        for task in tasks
        if task.name.endswith("__fno__d0")
    )


def test_resolved_full_doc_task_config_preserves_leafgrid_request_for_official_fno() -> None:
    config = pipeline._resolved_full_doc_task_config(
        worker_kind="full_doc_diagnostics",
        config={
            "fixed_leaf_tokens": 128,
            "pipeline_supervision_recovery_leafgrid_active": True,
            "pipeline_supervision_recovery_leaf_tokens": 128,
            "pipeline_supervision_recovery_package": "full100",
            "pipeline_supervision_recovery_scope": "recoverable_v4",
        },
        task_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["official_fno"],
            "train_doc_counts": [1024],
            "seeds": [0],
        },
    )

    assert config["fixed_leaf_tokens"] == 128
    assert config["preserve_requested_leaf_tokens"] is True
    assert config["official_fno_preserve_requested_leaf_tokens"] is True


def test_resolved_full_doc_task_config_defaults_mixed_family_requests_to_comparable() -> None:
    config = pipeline._resolved_full_doc_task_config(
        worker_kind="full_doc_diagnostics",
        config={
            "fixed_leaf_tokens": 128,
            "tree_leaf_fno_width": 96,
            "tree_leaf_fno_n_modes": 12,
            "tree_leaf_fno_n_layers": 5,
        },
        task_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["official_fno", "tree_neural"],
            "train_doc_counts": [1024],
            "seeds": [0],
        },
    )

    assert config["comparison_mode"] == "comparable"
    assert config["fixed_leaf_tokens"] == 128
    assert config["preserve_requested_leaf_tokens"] is True
    assert config["official_fno_preserve_requested_leaf_tokens"] is True
    assert config["tree_leaf_fno_width"] == 96
    assert config["fno_width"] == 96


def test_direct_task_serializes_comparable_surface_metadata(tmp_path: Path) -> None:
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__official_fno__comparable",
        config={
            "train_docs": 1024,
            "comparison_mode": "comparable",
            "fixed_leaf_tokens": 128,
            "tree_leaf_fno_width": 96,
            "tree_leaf_fno_n_modes": 12,
            "tree_leaf_fno_n_layers": 5,
            "tree_root_supervision_kind": "count_ce",
            "n_epochs": 10,
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["official_fno"],
            "train_doc_counts": [1024],
            "seeds": [0],
        },
    )

    payload = json.loads((task.output_path.parent / "task.request").read_text(encoding="utf-8"))
    assert payload["comparison_mode"] == "comparable"
    assert payload["config"]["comparison_mode"] == "comparable"
    assert payload["config"]["comparison_surface_diff"] == {}
    assert payload["comparison_surface_snapshot"]["fixed_leaf_tokens"] == 128
    assert payload["comparison_surface_snapshot"]["tree_leaf_fno_width"] == 96
    assert task.metadata["comparison_mode"] == "comparable"


def test_direct_task_canonicalizes_full_doc_config_aliases(tmp_path: Path) -> None:
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__tree_neural__canonicalized",
        config={
            "train_docs": 1024,
            "n_epochs": 10,
            "tree_local_law_weight": 0.8,
            "tree_task_objective_weight": 0.2,
            "tree_c1_relative_weight": 0.0,
            "tree_c2_relative_weight": 1.0,
            "tree_c3_relative_weight": 0.0,
            "tree_document_loss_normalization_mode": "supervised_docs",
            "tree_supervision_source": "manifest",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["tree_neural"],
            "train_doc_counts": [1024],
            "seeds": [0],
        },
    )

    payload = json.loads((task.output_path.parent / "task.request").read_text(encoding="utf-8"))
    assert payload["config"]["local_law_weight"] == pytest.approx(0.8)
    assert payload["config"]["task_objective_weight"] == pytest.approx(0.2)
    assert payload["config"]["c1_relative_weight"] == pytest.approx(0.0)
    assert payload["config"]["c2_relative_weight"] == pytest.approx(1.0)
    assert payload["config"]["c3_relative_weight"] == pytest.approx(0.0)
    assert payload["config"]["tree_document_loss_normalization_mode"] == "supervised_docs"
    assert payload["config"]["tree_supervision_source"] == "manifest"
    assert "tree_local_law_weight" not in payload["config"]
    assert "tree_task_objective_weight" not in payload["config"]


def test_supervision_recovery_leafgrid_promotes_one_leaf_full100_tree_rows_to_exact_full_doc_parity(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        one_leaf_tree_reference_mode="preset",
        one_leaf_tree_reference_preset="unified_g_fno_parity_canary_v1",
        one_leaf_tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=16,
        supervision_recovery_leaf_token_ladder="128",
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "leafgrid_exact_collapse",
        package_order=["full100"],
    )

    assert len(tasks) == 4
    assert not [task for task in tasks if "__tree_neural_exactcollapse__" in task.name]
    parity_tree_tasks = [
        task for task in tasks if task.name.endswith("__tree_neural__d0")
    ]
    assert {
        task.name for task in parity_tree_tasks
    } == {
        "recoverable_v4__train01024__full100__leaf128__tree_neural__d0",
        "r12_seg10to12__train01024__full100__leaf128__tree_neural__d0",
    }
    for task in parity_tree_tasks:
        payload = json.loads(
            (task.output_path.parent / "task.request").read_text(encoding="utf-8")
        )
        config = dict(payload.get("config") or {})
        assert payload["comparison_mode"] == "exact_collapse"
        assert config["fixed_leaf_tokens"] == 128
        assert config["comparison_mode"] == "exact_collapse"
        assert config["comparison_surface_diff"] == {}
        assert config["preserve_requested_leaf_tokens"] is True
        assert config["official_fno_preserve_requested_leaf_tokens"] is True
        assert (
            config["tree_exact_collapse_mode"]
            == pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
        )
        assert (
            config["pipeline_supervision_recovery_parity_mode"] == "exact_full_doc"
        )
        assert (
            config["pipeline_supervision_recovery_is_exact_full_doc_parity_row"]
            is True
        )


def test_supervision_recovery_leafgrid_promotes_canary_alias_to_exact_full_doc_parity(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="fno_parity_canary",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="fno_parity_canary",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_recovery_recoverable_benchmark="recoverable_v4_t128",
        supervision_recovery_structural_grid="structural_core_v1_t128",
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=16,
        supervision_recovery_leaf_token_ladder="128",
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "leafgrid_canary_alias_exact_collapse",
        package_order=["full100"],
    )

    parity_tree_tasks = [
        task for task in tasks if task.name.endswith("__tree_neural__d0")
    ]
    assert len(parity_tree_tasks) == 2
    for task in parity_tree_tasks:
        payload = json.loads(
            (task.output_path.parent / "task.request").read_text(encoding="utf-8")
        )
        config = dict(payload.get("config") or {})
        assert payload["comparison_mode"] == "exact_collapse"
        assert config["comparison_mode"] == "exact_collapse"
        assert (
            config["tree_exact_collapse_mode"]
            == pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
        )
        assert (
            config["pipeline_supervision_recovery_exact_full_doc_parity_requested"]
            is True
        )
        assert (
            config["pipeline_supervision_recovery_is_exact_full_doc_parity_row"]
            is True
        )


def test_supervision_recovery_leafgrid_does_not_auto_promote_one_leaf_full100_without_explicit_parity_request(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=16,
        supervision_recovery_leaf_token_ladder="128",
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "leafgrid_standard_one_leaf",
        package_order=["full100"],
    )

    parity_tree_tasks = [
        task for task in tasks if task.name.endswith("__tree_neural__d0")
    ]
    assert len(parity_tree_tasks) == 2
    for task in parity_tree_tasks:
        payload = json.loads(
            (task.output_path.parent / "task.request").read_text(encoding="utf-8")
        )
        config = dict(payload.get("config") or {})
        assert payload["comparison_mode"] == "comparable"
        assert config["comparison_mode"] == "comparable"
        assert config.get("tree_exact_collapse_mode", "") == ""
        assert (
            config.get("pipeline_supervision_recovery_exact_full_doc_parity_requested")
            is False
        )
        assert (
            config.get("pipeline_supervision_recovery_is_exact_full_doc_parity_row")
            is None
        )


def test_supervision_recovery_leafgrid_routes_one_leaf_root_only_rows_to_exact_full_doc_parity(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="unified_g_full_local_laws_v1",
        tree_reference_capacity_root=None,
        one_leaf_tree_reference_mode="preset",
        one_leaf_tree_reference_preset="unified_g_fno_parity_canary_v1",
        one_leaf_tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="unified_g_full_local_laws_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=128,
        supervision_max_tokens=128,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=16,
        supervision_recovery_leaf_token_ladder="128",
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "leafgrid_one_leaf_root_only_exact_parity",
        package_order=["full90"],
    )

    tree_tasks = [task for task in tasks if task.name.endswith("__tree_neural__d0")]
    assert {
        task.name for task in tree_tasks
    } == {
        "recoverable_v4__train01024__full90__leaf128__tree_neural__d0",
        "r12_seg10to12__train01024__full90__leaf128__tree_neural__d0",
    }
    for task in tree_tasks:
        payload = json.loads(
            (task.output_path.parent / "task.request").read_text(encoding="utf-8")
        )
        config = dict(payload.get("config") or {})
        assert payload["comparison_mode"] == "exact_collapse"
        assert config["comparison_mode"] == "exact_collapse"
        assert (
            config.get("pipeline_supervision_recovery_exact_full_doc_parity_requested")
            is True
        )
        assert (
            config["tree_exact_collapse_mode"]
            == pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
        )
        assert (
            config["pipeline_supervision_recovery_parity_mode"] == "exact_full_doc"
        )
        assert (
            config["pipeline_supervision_recovery_is_exact_full_doc_parity_row"]
            is True
        )
        assert config["fixed_leaf_tokens"] == 128
        assert config["preserve_requested_leaf_tokens"] is True
        assert config["official_fno_preserve_requested_leaf_tokens"] is True


def test_supervision_recovery_phase_uses_structural_factorized_tree_reference_when_explicitly_requested(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="preset",
        tree_reference_preset="recoverable_slotwise_dense_v1",
        tree_reference_capacity_root=None,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="structural_factorized_fiber_v1",
        supervision_batch_size=256,
        supervision_epochs=10,
        exact_metric_final_doc_limit=128,
        tree_posttrain_train_doc_limit=96,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(args, tmp_path / "structural")
    structural_task = next(
        task
        for task in tasks
        if task.name.startswith("r12_seg10to12__train01024__full100__tree_neural__d0")
    )
    structural_request = json.loads(
        (structural_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    structural_config = dict(structural_request.get("config") or {})

    assert structural_config["tree_training_schedule"] == "single_stage"
    assert structural_config["tree_theorem_surface_mode"] == "factorized_score_fiber"
    assert structural_config["tree_summary_spec_root_mode"] == "factored_theorem_readout"
    assert structural_config["tree_checkpoint_metric"] == "val_root_mae"
    assert structural_config["fixed_leaf_tokens"] == 8
    assert structural_config["state_dim"] == 32
    assert structural_config["hidden_dim"] == 64
    assert structural_config["tree_theorem_feature_dim"] == 16
    assert structural_config["tree_theorem_feature_hidden_dim"] == 32
    assert structural_config["tree_theorem_score_dim"] == 1
    assert structural_config["tree_theorem_fiber_dim"] == 15
    assert structural_config["tree_batch_pack_mode"] == "fixed_fused"
    assert structural_config["gpu_runtime_bucket_mode"] == "leaf_count_auto_queue"
    assert structural_config["pipeline_tree_reference_label"] == "structural_factorized_fiber_v1"
    assert structural_task.metadata["n_epochs"] == 10


def test_supervision_recovery_phase_rejects_weak_tree_reference(tmp_path: Path) -> None:
    capacity_root = _write_weak_capacity_locked_tree_reference(tmp_path)
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="capacity_locked",
        tree_reference_capacity_root=capacity_root,
        supervision_batch_size=256,
        supervision_epochs=10,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    with pytest.raises(ValueError, match="not learning-capable"):
        _build_supervision_recovery_phase(args, tmp_path / "weak")


def test_build_run_plan_rejects_smoke_or_weak_supervision_recovery_tree_setup(
    tmp_path: Path,
) -> None:
    capacity_root = _write_weak_capacity_locked_tree_reference(tmp_path)
    config_path = tmp_path / "weak_supervision_recovery.toml"
    config_path.write_text(
        "\n".join(
            [
                "[tradeoff_pipeline]",
                'preset = "standard"',
                'phases = ["supervision_recovery", "report"]',
                "train_docs = 4096",
                "val_docs = 512",
                "test_docs = 512",
                'supervision_recovery_train_docs = [1024]',
                'supervision_recovery_seeds = [0]',
                "",
                "[tradeoff_pipeline.tree_reference]",
                'mode = "capacity_locked"',
                f'capacity_root = "{capacity_root}"',
                "",
                "[tradeoff_pipeline.runtime]",
                'data_mode = "resident"',
                'bucket_mode = "leaf_count_auto_queue"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not learning-capable"):
        build_run_plan(
            _parse_args(["--config", str(config_path), "--plan-only"]),
            devices=["MIG-a"],
        )


def test_supervision_recovery_phase_allows_explicit_tree_schedule_override(
    tmp_path: Path,
) -> None:
    capacity_root = _write_capacity_locked_tree_reference(tmp_path)
    args = argparse.Namespace(
        preset="standard",
        seed=42,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        device_mode="cpu",
        tree_reference_mode="capacity_locked",
        tree_reference_capacity_root=capacity_root,
        tree_training_schedule="two_stage",
        tree_stage1_epochs=12,
        tree_stage2_epochs=20,
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="structural_factorized_sketch_v3",
        supervision_batch_size=256,
        supervision_epochs=32,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        tree_batch_autotune=False,
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_tree_batch_structural_pad_limit=0.5,
        runtime_tree_batch_auto_queue_min_docs=8,
        runtime_tree_batch_auto_queue_min_fill_ratio=0.5,
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
    )

    tasks, _ = _build_supervision_recovery_phase(args, tmp_path / "override")
    tree_task = next(task for task in tasks if "__tree_neural__" in task.name)
    tree_request = json.loads(
        (tree_task.output_path.parent / "task.request").read_text(encoding="utf-8")
    )
    tree_config = dict(tree_request.get("config") or {})

    assert tree_config["tree_training_schedule"] == "two_stage"
    assert tree_config["tree_stage1_epochs"] == 12
    assert tree_config["tree_stage2_epochs"] == 20
    assert tree_task.metadata["n_epochs"] == 32


def test_supervision_recovery_aggregator_reuses_full10_fno_baseline() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.12,
                },
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.10,
                },
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.20,
                },
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.18,
                },
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.14,
                    "test_leaf_mae_mean": 0.04,
                    "test_merge_mae_mean": 0.03,
                }
            ],
        },
    ]
    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
    )
    scope = summary["scopes"]["recoverable_v4"]
    rows_by_train_docs = {
        str(item["train_doc_count"]): item
        for item in scope["rows_by_train_docs"]
    }
    row = next(
        row
        for row in rows_by_train_docs[str(1024)]["rows"]
        if row["package_name"] == "full10_leaf_full100_internal_count100"
    )

    assert row["fno_reference_package"] == "full10"
    assert row["fno_reference_family"] == "official_fno_sumlen"
    assert row["fno_reference_test_root_mae"] == 0.18
    assert row["matched_fno_family_rows"]["official_fno"]["test_root_mae"] == 0.20
    assert row["matched_fno_family_rows"]["official_fno_sumlen"]["test_root_mae"] == 0.18
    assert row["full100_fno_test_root_mae"] == 0.10
    assert row["full100_fno_family_rows"]["official_fno"]["test_root_mae"] == 0.12
    assert row["full100_fno_family_rows"]["official_fno_sumlen"]["test_root_mae"] == 0.10
    assert row["best_full100_fno_family"] == "official_fno_sumlen"
    assert row["delta_vs_best_full100_fno"] == 0.14 - 0.10
    assert row["delta_vs_full10_fno"] == 0.14 - 0.18
    assert summary["best_tree_summary"][0]["package_name"] == "full10_leaf_full100_internal_count100"


def test_supervision_recovery_aggregator_uses_matching_full20_fno_baseline() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.10,
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full20",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.16,
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.18,
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full20_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.12,
                    "test_leaf_mae_mean": 0.03,
                    "test_merge_mae_mean": 0.02,
                }
            ],
        },
    ]
    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
    )
    scope = summary["scopes"]["recoverable_v4"]
    rows_by_train_docs = {
        str(item["train_doc_count"]): item
        for item in scope["rows_by_train_docs"]
    }
    row = next(
        row
        for row in rows_by_train_docs[str(1024)]["rows"]
        if row["package_name"] == "full20_leaf_full100_internal_count100"
    )

    assert row["fno_reference_package"] == "full20"
    assert row["fno_reference_family"] == "official_fno_sumlen"
    assert row["fno_reference_test_root_mae"] == 0.16
    assert row["full100_fno_test_root_mae"] == 0.10
    assert row["delta_vs_full10_fno"] == 0.12 - 0.16


def test_supervision_recovery_aggregator_uses_matching_full100_fno_baseline() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.10,
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.11,
                    "test_leaf_mae_mean": 0.02,
                    "test_merge_mae_mean": 0.01,
                }
            ],
        },
    ]
    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full100", "full100_leaf_full100_internal_count100"],
    )
    scope = summary["scopes"]["recoverable_v4"]
    rows_by_train_docs = {
        str(item["train_doc_count"]): item
        for item in scope["rows_by_train_docs"]
    }
    row = next(
        row
        for row in rows_by_train_docs[str(1024)]["rows"]
        if row["package_name"] == "full100_leaf_full100_internal_count100"
    )

    assert summary["package_order"] == ["full100", "full100_leaf_full100_internal_count100"]
    assert row["fno_reference_package"] == "full100"
    assert row["fno_reference_family"] == "official_fno_sumlen"
    assert row["fno_reference_test_root_mae"] == 0.10
    assert row["full100_fno_test_root_mae"] == 0.10
    assert row["delta_vs_full10_fno"] == 0.11 - 0.10


def test_supervision_recovery_aggregator_carries_mass_matched_unified_g_metadata() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.09,
                },
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.095,
                    "effective_full_doc_mass_per_doc_mean": 0.10,
                },
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "seed": 42,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "tree_reference_mode": "preset",
                    "tree_training_schedule": "two_stage",
                    "tree_checkpoint_metric": "val_root_mae",
                    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "r10_mass_local_eq_1p0",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "unified_g_full_local_laws_v1",
                "tree_training_schedule": "two_stage",
                "tree_model_version": "unified_g",
                "tree_score_merge_mode": "exact_projected_sketch",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "summary_spec_name": "markov_count_sketch",
                "slot_count": 4,
                "state_dim": 128,
                "hidden_dim": 512,
                "fixed_leaf_tokens": 16,
                "mass_target_per_doc": 0.1,
                "computed_doc_review_mass_per_doc": 0.06,
                "computed_local_mass_per_doc": 0.04,
                "computed_leaf_mass_per_doc": 0.01,
                "computed_internal_mass_per_doc": 0.03,
                "computed_total_mass_per_doc": 0.10,
                "computed_leaf_mass_full_per_doc": 1.0,
                "computed_internal_mass_full_per_doc": 3.0,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 8,
                "computed_assumed_internal_nodes": 7,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.07,
                    "effective_full_doc_mass_per_doc_mean": 0.101,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "seed": 42,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "tree_reference_mode": "preset",
                    "tree_training_schedule": "two_stage",
                    "tree_checkpoint_metric": "val_root_mae",
                    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                    "requested_root_mass_per_doc": 0.06,
                    "root_supervision_docs_total": 61,
                }
            ],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full10", "r10_mass_local_eq_1p0"],
    )

    scope = summary["scopes"]["recoverable_v4"]
    rows_by_train_docs = {
        str(item["train_doc_count"]): item for item in scope["rows_by_train_docs"]
    }
    row = next(
        row
        for row in rows_by_train_docs["1024"]["rows"]
        if row["package_name"] == "r10_mass_local_eq_1p0"
    )

    assert summary["common_tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert summary["canonical_tree_selection_metric"] == "val_root_mae"
    assert summary["package_doc_equivalent"]["r10_mass_local_eq_1p0"][
        "mass_target_per_doc"
    ] == pytest.approx(0.1)
    assert row["doc_equiv_mass_target_per_doc"] == pytest.approx(0.1)
    assert row["doc_equiv_doc_review_mass_per_doc"] == pytest.approx(0.06)
    assert row["doc_equiv_local_mass_per_doc"] == pytest.approx(0.04)
    assert row["tree_effective_full_doc_mass_per_doc"] == pytest.approx(0.101)
    assert row["tree_mass_target_per_doc"] == pytest.approx(0.1)
    assert row["tree_computed_doc_review_mass_per_doc"] == pytest.approx(0.06)
    assert row["tree_reference_label"] == "unified_g_full_local_laws_v1"
    assert row["fixed_leaf_tokens"] == 16
    assert row["leaves_per_doc"] == 8
    assert row["internal_nodes_per_doc"] == 7
    assert row["is_fno_equivalent_geometry"] is False

    family_row = next(
        item
        for item in summary["family_rows"]
        if item["scope_key"] == "recoverable_v4"
        and item["package_name"] == "r10_mass_local_eq_1p0"
        and item["baseline_family"] == "tree_neural"
    )
    assert family_row["fixed_leaf_tokens"] == 16
    assert family_row["leaves_per_doc"] == 8
    assert family_row["internal_nodes_per_doc"] == 7
    assert family_row["is_fno_equivalent_geometry"] is False
    assert family_row["requested_root_mass_per_doc_mean"] == pytest.approx(0.06)
    assert family_row["root_supervision_docs_total_mean"] == pytest.approx(61.0)


def test_supervision_recovery_aggregator_preserves_superset_package_semantics() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4_t128",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4_t128",
                "train_docs": 10240,
                "data_seed": 0,
                "package_semantics": "full_doc_only",
                "fixed_leaf_tokens": 128,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 1,
                "computed_assumed_internal_nodes": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4_t128",
                    "baseline_family": "official_fno",
                    "train_doc_count": 10240,
                    "test_root_mae_mean": 0.011,
                    "fixed_leaf_tokens": 128,
                    "test_mean_leaves_per_doc": 1.0,
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "r100_superset_local_eq_10p0",
                "pipeline_supervision_recovery_scope": "recoverable_v4_t128",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4_t128",
                "train_docs": 10240,
                "data_seed": 0,
                "package_semantics": "superset",
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "unified_g_full_local_laws_v1",
                "tree_training_schedule": "two_stage",
                "tree_model_version": "unified_g",
                "tree_score_merge_mode": "exact_projected_sketch",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "fixed_leaf_tokens": 128,
                "budget_total_calls_per_doc": 1.0,
                "mass_target_per_doc": float("nan"),
                "computed_doc_review_mass_per_doc": 1.0,
                "computed_local_mass_per_doc": 0.1,
                "computed_leaf_mass_per_doc": 0.1,
                "computed_internal_mass_per_doc": 0.0,
                "computed_total_mass_per_doc": 1.1,
                "computed_leaf_mass_full_per_doc": 1.0,
                "computed_internal_mass_full_per_doc": 0.0,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 1,
                "computed_assumed_internal_nodes": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4_t128",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "package_semantics": "superset",
                    "test_root_mae_mean": 0.010,
                    "effective_full_doc_mass_per_doc_mean": 1.1,
                    "fixed_leaf_tokens": 128,
                    "test_mean_leaves_per_doc": 1.0,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4_t128",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "seed": 0,
                    "package_semantics": "superset",
                    "requested_root_mass_per_doc": 1.0,
                    "root_supervision_docs_total": 10240,
                    "effective_full_doc_mass_per_doc": 1.1,
                    "executed_fixed_leaf_tokens": 128,
                    "executed_leaves_per_doc": 1,
                    "fixed_leaf_tokens": 128,
                }
            ],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full100", "r100_superset_local_eq_10p0"],
        recoverable_benchmark="recoverable_v4_t128",
    )

    assert summary["package_definitions"]["r100_superset_local_eq_10p0"]["package_semantics"] == "superset"
    assert summary["package_doc_equivalent"]["r100_superset_local_eq_10p0"]["package_semantics"] == "superset"

    scope = summary["scopes"]["recoverable_v4_t128"]
    rows_by_train_docs = {
        str(item["train_doc_count"]): item for item in scope["rows_by_train_docs"]
    }
    row = next(
        row
        for row in rows_by_train_docs["10240"]["rows"]
        if row["package_name"] == "r100_superset_local_eq_10p0"
    )

    assert row["package_semantics"] == "superset"
    assert row["package_label"] == "R100 superset + 10.0% leaf/internal count"
    assert row["tree_effective_full_doc_mass_per_doc"] == pytest.approx(1.1)
    assert row["doc_equiv_total_mass_per_doc"] == pytest.approx(1.1)
    assert row["tree_computed_doc_review_mass_per_doc"] == pytest.approx(1.0)
    assert row["tree_computed_local_mass_per_doc"] == pytest.approx(0.1)
    assert row["tree_mass_target_per_doc"] != row["tree_mass_target_per_doc"]

    family_row = next(
        item
        for item in summary["family_rows"]
        if item["scope_key"] == "recoverable_v4_t128"
        and item["package_name"] == "r100_superset_local_eq_10p0"
        and item["baseline_family"] == "tree_neural"
    )
    assert family_row["package_semantics"] == "superset"
    assert family_row["requested_root_mass_per_doc_mean"] == pytest.approx(1.0)
    assert family_row["root_supervision_docs_total_mean"] == pytest.approx(10240.0)


def test_supervision_recovery_aggregator_prefers_executed_geometry_and_surfaces_exact_collapse_rows() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 4096,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "unified_g_full_local_laws_v1",
                "tree_training_schedule": "single_stage",
                "tree_checkpoint_metric": "val_root_mae",
                "fixed_leaf_tokens": 128,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 1,
                "computed_assumed_internal_nodes": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno",
                    "train_doc_count": 4096,
                    "test_root_mae_mean": 0.0117,
                    "fixed_leaf_tokens": 128,
                    "test_mean_leaves_per_doc": 1.0,
                },
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 4096,
                    "test_root_mae_mean": 0.2032,
                    "fixed_leaf_tokens": 16,
                    "test_mean_leaves_per_doc": 6.0,
                },
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 4096,
                    "seed": 42,
                    "config": {
                        "fixed_leaf_tokens": 16,
                    },
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "tree_reference_mode": "preset",
                    "tree_training_schedule": "single_stage",
                    "tree_checkpoint_metric": "val_root_mae",
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "pipeline_supervision_recovery_comparison_arm": (
                    pipeline.SUPERVISION_RECOVERY_EXACT_COLLAPSE_ONE_TREE_COMPARISON_ARM
                ),
                "train_docs": 4096,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "unified_g_full_local_laws_v1",
                "tree_training_schedule": "single_stage",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_exact_collapse_mode": pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE,
                "fixed_leaf_tokens": 128,
                "computed_assumed_doc_tokens": 128,
                "computed_assumed_leaves": 1,
                "computed_assumed_internal_nodes": 0,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 4096,
                    "test_root_mae_mean": 0.0117,
                    "fixed_leaf_tokens": 128,
                    "test_mean_leaves_per_doc": 1.0,
                    "tree_exact_collapse_mode": pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 4096,
                    "seed": 42,
                    "config": {
                        "fixed_leaf_tokens": 128,
                        "tree_exact_collapse_mode": pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE,
                        "pipeline_supervision_recovery_comparison_arm": (
                            pipeline.SUPERVISION_RECOVERY_EXACT_COLLAPSE_ONE_TREE_COMPARISON_ARM
                        ),
                    },
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "tree_reference_mode": "preset",
                    "tree_training_schedule": "single_stage",
                    "tree_checkpoint_metric": "val_root_mae",
                    "tree_exact_collapse_mode": pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE,
                }
            ],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full100"],
    )

    family_row = next(
        item
        for item in summary["family_rows"]
        if item["scope_key"] == "recoverable_v4"
        and item["package_name"] == "full100"
        and item["baseline_family"] == "tree_neural"
    )
    assert family_row["computed_assumed_leaves"] == 1
    assert family_row["requested_fixed_leaf_tokens"] == 16
    assert family_row["executed_fixed_leaf_tokens"] == 16
    assert family_row["fixed_leaf_tokens"] == 16
    assert family_row["test_mean_leaves_per_doc"] == pytest.approx(6.0)
    assert family_row["executed_leaves_per_doc"] == 6
    assert family_row["leaves_per_doc"] == 6
    assert family_row["internal_nodes_per_doc"] == 5
    assert family_row["parity_mode"] == ""
    assert family_row["is_exact_full_doc_parity_row"] is False
    assert family_row["is_fno_equivalent_geometry"] is False

    exact_row = summary["exact_collapse_rows"][0]
    assert (
        exact_row["comparison_arm"]
        == pipeline.SUPERVISION_RECOVERY_EXACT_COLLAPSE_ONE_TREE_COMPARISON_ARM
    )
    assert (
        exact_row["tree_exact_collapse_mode"]
        == pipeline.EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
    )
    assert exact_row["requested_fixed_leaf_tokens"] == 128
    assert exact_row["executed_fixed_leaf_tokens"] == 128
    assert exact_row["fixed_leaf_tokens"] == 128
    assert exact_row["executed_leaves_per_doc"] == 1
    assert exact_row["leaves_per_doc"] == 1
    assert exact_row["parity_mode"] == "exact_full_doc"
    assert exact_row["is_exact_full_doc_parity_row"] is True
    assert exact_row["is_fno_equivalent_geometry"] is True
    assert exact_row["official_fno_family"] == "official_fno"
    assert exact_row["official_fno_test_root_mae"] == pytest.approx(0.0117)
    assert exact_row["ordinary_tree_test_root_mae"] == pytest.approx(0.2032)
    assert exact_row["delta_vs_official_fno"] == pytest.approx(0.0)


def test_supervision_recovery_aggregator_rolls_up_runtime_telemetry() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
                "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "recoverable_slotwise_dense_v1",
                "tree_training_schedule": "two_stage",
                "tree_checkpoint_metric": "val_exact_sketch_direct",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "summary_spec_name": "markov_count_sketch",
                "slot_count": 4,
                "state_dim": 128,
                "hidden_dim": 512,
                "fixed_leaf_tokens": 16,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.14,
                    "test_leaf_mae_mean": 0.04,
                    "test_merge_mae_mean": 0.03,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "runtime_efficiency": {
                        "runtime_bucket_mode": "leaf_count_auto_queue",
                        "steady_state_h2d_bytes": 0.0,
                        "steady_state_h2d_events": 0.0,
                        "resident_store_hits": 32.0,
                        "resident_store_misses": 1.0,
                        "auto_queue_family_count": 3.0,
                        "structural_padding_waste_ratio": 0.125,
                        "auto_queue_fused_batches": 17.0,
                        "auto_queue_generic_fallback_batches": 2.0,
                        "fixed_shape_dense_bucket_store_hits": 9.0,
                    },
                    "autotuned_batch_budgets": {
                        "auto_queue_target_leaf_counts": [64, 128, 256],
                    },
                }
            ],
        }
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(payloads)
    family_row = summary["family_rows"][0]

    assert family_row["runtime_bucket_mode"] == "leaf_count_auto_queue"
    assert family_row["steady_state_h2d_bytes"] == 0.0
    assert family_row["steady_state_h2d_events"] == 0.0
    assert family_row["resident_store_hits"] == 32.0
    assert family_row["resident_store_misses"] == 1.0
    assert family_row["auto_queue_family_count"] == 3.0
    assert family_row["auto_queue_target_leaf_counts"] == [64, 128, 256]
    assert family_row["structural_padding_waste_ratio"] == 0.125
    assert family_row["auto_queue_fused_batches"] == 17.0
    assert family_row["auto_queue_generic_fallback_batches"] == 2.0
    assert family_row["fixed_shape_dense_bucket_store_hits"] == 9.0
    assert family_row["tree_reference_label"] == "recoverable_slotwise_dense_v1"
    assert family_row["tree_training_schedule"] == "two_stage"
    assert family_row["summary_spec_name"] == "markov_count_sketch"
    assert family_row["slot_count"] == 4
    assert family_row["state_dim"] == 128
    assert family_row["hidden_dim"] == 512
    assert family_row["fixed_leaf_tokens"] == 16

    scope_reference = summary["scope_tree_references"]["recoverable_v4"]
    assert scope_reference["tree_reference_label"] == "recoverable_slotwise_dense_v1"
    assert scope_reference["tree_training_schedule"] == "two_stage"
    assert scope_reference["tree_checkpoint_metric"] == "val_exact_sketch_direct"
    assert scope_reference["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"
    assert scope_reference["summary_spec_name"] == "markov_count_sketch"
    assert scope_reference["slot_count"] == 4
    assert scope_reference["state_dim"] == 128
    assert scope_reference["hidden_dim"] == 512
    assert scope_reference["fixed_leaf_tokens"] == 16


def test_checked_in_leafgrid_canary_config_builds_plan() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_canary.toml"
    )
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [128]
    assert set(plan["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 40
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"]["leaf_token_ladder"]
        == [128]
    )
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"][
            "exact_full_doc_parity_leaf_tokens"
        ]
        == [128]
    )


def test_t128_benchmark_geometry_resolves_from_surface_when_bundle_missing() -> None:
    pipeline._resolved_full_doc_bundle_token_geometry.cache_clear()
    assert pipeline._resolved_full_doc_bundle_token_geometry(
        "recoverable_v4_t128",
        "",
        tuple(),
    ) == (128, 128)
    assert pipeline._resolved_full_doc_bundle_token_geometry(
        "structural_core_v1_t128::r12_seg10to12",
        "structural_core_v1_t128",
        ("r12_seg10to12",),
    ) == (128, 128)
    pipeline._resolved_full_doc_bundle_token_geometry.cache_clear()


def test_checked_in_t128_canary_config_builds_plan() -> None:
    config_path = Path("config/markov/tradeoff_pipeline.fno_parity_canary_test_t128.toml")
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_recoverable_benchmark"] == "recoverable_v4_t128"
    assert plan["resolved_selection"]["supervision_recovery_structural_grid"] == "structural_core_v1_t128"
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [128]
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 12
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"][
            "exact_full_doc_parity_leaf_tokens"
        ]
        == [128]
    )
    assert plan["phase_task_counts"]["supervision_recovery"]["details"]["benchmarks"] == [
        "recoverable_v4_t128",
        "structural_core_v1_t128::r12_seg10to12",
    ]


def test_checked_in_t128_gamma_config_builds_plan() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_gamma_t128.toml"
    )
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_recoverable_benchmark"] == "recoverable_v4_t128"
    assert plan["resolved_selection"]["supervision_recovery_structural_grid"] == "structural_core_v1_t128"
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0]
    assert plan["resolved_selection"]["supervision_recovery_depth_discount_gammas"] == pytest.approx([1.0, 0.9, 0.75])
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [32, 16, 8]
    assert plan["resolved_selection"]["supervision_recovery_packages"] == [
        "full100",
        "r100_mass_local_eq_10p0",
        "r100_mass_local_eq_15p0",
        "r100_mass_local_eq_20p0",
    ]
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 78
    assert plan["phase_task_counts"]["supervision_recovery"]["details"]["benchmarks"] == [
        "recoverable_v4_t128",
        "structural_core_v1_t128::r12_seg10to12",
    ]


def test_checked_in_t128_superset_gamma_config_builds_plan() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_gamma_t128.toml"
    )
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_recoverable_benchmark"] == "recoverable_v4_t128"
    assert plan["resolved_selection"]["supervision_recovery_structural_grid"] == "structural_core_v1_t128"
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0]
    assert plan["resolved_selection"]["supervision_recovery_depth_discount_gammas"] == pytest.approx([1.0, 0.9, 0.75])
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [32, 16, 8]
    assert plan["resolved_selection"]["supervision_recovery_packages"] == [
        "full100",
        "r100_superset_local_eq_10p0",
        "r100_superset_local_eq_15p0",
        "r100_superset_local_eq_20p0",
    ]
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"][
            "exact_full_doc_parity_leaf_tokens"
        ]
        == []
    )


def test_checked_in_leaf128_superset_debug_config_builds_plan() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_leaf128_r100_debug.toml"
    )
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_recoverable_benchmark"] == "recoverable_v4_t128"
    assert plan["resolved_selection"]["supervision_recovery_structural_grid"] == "structural_core_v1_t128"
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 4096, 10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0]
    assert plan["resolved_selection"]["supervision_recovery_depth_discount_gammas"] == pytest.approx([1.0])
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [128]
    assert plan["resolved_selection"]["supervision_recovery_packages"] == [
        "full100",
        "r100_superset_local_eq_10p0",
        "r100_superset_local_eq_15p0",
        "r100_superset_local_eq_20p0",
    ]


def test_checked_in_leaf32_superset_c1half_config_builds_plan() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_leaf32_gamma09_c1half.toml"
    )
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0]
    assert plan["resolved_selection"]["supervision_recovery_depth_discount_gammas"] == pytest.approx([0.9])
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [32]
    assert plan["resolved_selection"]["supervision_recovery_packages"] == [
        "full100",
        "r100_superset_local_eq_10p0",
    ]
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"][
            "exact_full_doc_parity_leaf_tokens"
        ]
        == []
    )


def test_checked_in_leaf32_superset_leafratehalf_config_builds_plan() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_leaf32_gamma09_leafratehalf.toml"
    )
    assert config_path.exists()

    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0]
    assert plan["resolved_selection"]["supervision_recovery_depth_discount_gammas"] == pytest.approx([0.9])
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [32]
    assert plan["resolved_selection"]["supervision_recovery_packages"] == [
        "full100",
        "r100_superset_local_eq_10p0",
        "r100_superset_leaf05_internal10p0",
    ]
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"][
            "exact_full_doc_parity_leaf_tokens"
        ]
        == []
    )
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"][
            "exact_full_doc_parity_leaf_tokens"
        ]
        == []
    )


def test_checked_in_leafgrid_r100_config_resolves_single_leaf_override() -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100.toml"
    )
    assert config_path.exists()

    args = _parse_args(
        [
            "--config",
            str(config_path),
            "--plan-only",
            "--supervision-recovery-leaf-token-ladder",
            "64",
        ]
    )
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [64]
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 72
    assert (
        plan["phase_task_counts"]["supervision_recovery"]["details"]["leaf_token_ladder"]
        == [64]
    )


def test_multileaf_full100_leaf64_allows_canonical_fno_reference_surface(
    tmp_path: Path,
) -> None:
    config_path = Path(
        "config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml"
    )
    assert config_path.exists()

    args = _parse_args(
        [
            "--config",
            str(config_path),
            "--tree-reference-preset",
            "multileaf_root_only",
            "--structural-tree-reference-preset",
            "multileaf_root_only",
            "--supervision-recovery-train-docs",
            "1024",
            "--supervision-recovery-seeds",
            "0",
            "--supervision-recovery-leaf-token-ladder",
            "64",
        ]
    )

    tasks, _ = _build_supervision_recovery_phase(
        args,
        tmp_path / "leaf64_full100_canonical_fno_reference",
        package_order=["full100"],
    )

    tree_tasks = [task for task in tasks if task.name.endswith("__tree_neural__d0")]
    fno_tasks = [task for task in tasks if task.name.endswith("__fno__d0")]
    assert len(tree_tasks) == 2
    assert len(fno_tasks) == 2
    assert all("__leaf064__" in task.name for task in tree_tasks)
    assert all("__leaf128__" in task.name for task in fno_tasks)
    for task in tree_tasks:
        payload = json.loads(
            (task.output_path.parent / "task.request").read_text(encoding="utf-8")
        )
        assert int(payload["config"]["fixed_leaf_tokens"]) == 64
    for task in fno_tasks:
        payload = json.loads(
            (task.output_path.parent / "task.request").read_text(encoding="utf-8")
        )
        assert int(payload["config"]["fixed_leaf_tokens"]) == 128


def test_supervision_recovery_aggregator_emits_runtime_diagnosis_summary() -> None:
    payloads = [
        {
            "wall_clock_s": 45.0,
            "config": {
                "pipeline_supervision_recovery_package": "full10_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
                "gpu_runtime_data_mode": "resident",
                "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
                "tree_batch_pack_mode": "fixed_fused",
                "pipeline_tree_reference_label": "recoverable_slotwise_dense_v1",
                "tree_training_schedule": "two_stage",
                "tree_stage1_epochs": 5,
                "tree_stage2_epochs": 10,
                "tree_document_loss_normalization_mode": "auto",
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.14,
                    "test_leaf_mae_mean": 0.04,
                    "test_merge_mae_mean": 0.03,
                    "elapsed_s_mean": 45.0,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "seed": 42,
                    "fit_diagnostics": {"epochs_completed": 15},
                    "timing_breakdown": {
                        "train_loop_s": 30.0,
                        "stage1_train_loop_s": 12.0,
                        "stage2_train_loop_s": 18.0,
                        "exact_metric_eval_s": 3.0,
                    },
                    "runtime_efficiency": {
                        "runtime_data_mode": "resident",
                        "runtime_bucket_mode": "leaf_count_auto_queue",
                        "tree_document_loss_normalization_mode": "auto",
                        "effective_tree_document_loss_normalization_mode": "supervised_docs",
                        "steady_state_h2d_bytes": 0.0,
                        "steady_state_h2d_events": 0.0,
                        "resident_store_hits": 32.0,
                        "resident_store_misses": 0.0,
                        "auto_queue_family_count": 3.0,
                        "auto_queue_fused_batches": 17.0,
                        "fixed_shape_dense_bucket_store_hits": 9.0,
                        "document_supervision_docs_total": 128,
                        "root_supervision_docs_total": 96,
                        "doc_sequence_supervision_docs_total": 32,
                        "document_supervision_coverage_rate": 0.25,
                        "document_loss_mean_batch_scale": 4.0,
                        "normalized_root_contribution_final": 0.18,
                    },
                }
            ],
        }
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(payloads)
    runtime = dict(summary["runtime_diagnosis"])

    assert runtime["status"] == "ready"
    assert runtime["tree_fast_path_confirmed_runs"] == 1
    assert runtime["tree_partial_or_fallback_runs"] == 0
    assert runtime["tree_fast_path_completion_rate"] == 1.0
    assert runtime["tree_zero_h2d_rate"] == 1.0
    assert runtime["tree_median_train_loop_s_per_epoch"] == 2.0
    assert runtime["tree_median_train_loop_s_per_epoch_per_1k_docs"] == pytest.approx(
        30.0 / 15.0 / 1.024
    )
    assert runtime["tree_median_document_loss_batch_scale"] == pytest.approx(4.0)
    assert runtime["grouped_rows"][0]["fast_path_classification"] == "fast_path_confirmed"
    assert runtime["grouped_rows"][0]["tree_reference_label"] == "recoverable_slotwise_dense_v1"
    assert runtime["grouped_rows"][0]["tree_training_schedule"] == "two_stage"
    assert (
        runtime["grouped_rows"][0]["effective_tree_document_loss_normalization_mode"]
        == "supervised_docs"
    )
    assert runtime["grouped_rows"][0]["document_loss_mean_batch_scale_median"] == pytest.approx(
        4.0
    )


def test_supervision_recovery_aggregator_marks_mixed_package_tree_references() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "package_capacity_locked",
                "pipeline_tree_reference_label": "full10_cfg",
                "tree_checkpoint_metric": "val_exact_sketch_direct",
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.14,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "tree_reference_label": "full10_cfg",
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full10_leaf_count20_internal_count20",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "package_capacity_locked",
                "pipeline_tree_reference_label": "full10_leaf20_cfg",
                "tree_checkpoint_metric": "val_exact_sketch_direct",
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.11,
                }
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "tree_reference_label": "full10_leaf20_cfg",
                }
            ],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        package_order=["full10", "full10_leaf_count20_internal_count20"],
    )

    assert summary["common_tree_reference_label"] == ""
    assert summary["comparator_alignment_status"] == "mixed"
    assert summary["tree_reference_labels"] == ["full10_cfg", "full10_leaf20_cfg"]
    assert "full10_cfg" in summary["comparator_alignment_warning"]
    assert "full10_leaf20_cfg" in summary["comparator_alignment_warning"]


def test_supervision_recovery_aggregator_emits_canonical_root_comparison_metadata() -> None:
    payloads = [
        {
            "config": {
                "pipeline_supervision_recovery_package": "full50_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                "train_docs": 1024,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "common_factorized_sketch_v1",
                "tree_training_schedule": "two_stage",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "summary_spec_name": "markov_count_sketch",
                "slot_count": 4,
                "state_dim": 128,
                "hidden_dim": 512,
                "fixed_leaf_tokens": 16,
            },
            "aggregate_rows": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.12,
                    "train_root_mae_mean": 0.10,
                    "val_root_mae_mean": 0.11,
                    "test_leaf_mae_mean": 0.03,
                    "test_merge_mae_mean": 0.02,
                    "test_unweighted_full_law_objective_mean": 0.31,
                    "val_unweighted_full_law_objective_mean": 0.29,
                    "test_unweighted_active_objective_mean": 0.23,
                    "val_unweighted_active_objective_mean": 0.21,
                    "best_epoch_mean": 47.0,
                    "selection_metric": "val_root_mae",
                    "selection_metric_value_mean": 0.11,
                    "n_runs": 1,
                },
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.09,
                    "n_runs": 1,
                },
            ],
            "runs": [
                {
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "seed": 42,
                    "best_epoch": 47,
                    "selection_metric": "val_root_mae",
                    "selection_metric_value": 0.11,
                    "tree_checkpoint_metric": "val_root_mae",
                    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                }
            ],
        },
        {
            "config": {
                "pipeline_supervision_recovery_package": "full50_leaf_full100_internal_count100",
                "pipeline_supervision_recovery_scope": "r12_seg10to12",
                "pipeline_supervision_recovery_scope_label": "structural_core_v1::r12_seg10to12",
                "train_docs": 1024,
                "data_seed": 0,
                "pipeline_tree_reference_mode": "preset",
                "pipeline_tree_reference_label": "common_factorized_sketch_v1",
                "tree_training_schedule": "two_stage",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "summary_spec_name": "markov_count_sketch",
                "slot_count": 4,
                "state_dim": 128,
                "hidden_dim": 512,
                "fixed_leaf_tokens": 16,
            },
            "aggregate_rows": [
                {
                    "cell_id": "r12_seg10to12",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.22,
                    "train_root_mae_mean": 0.19,
                    "val_root_mae_mean": 0.20,
                    "test_leaf_mae_mean": 0.08,
                    "test_merge_mae_mean": 0.11,
                    "test_unweighted_full_law_objective_mean": 0.48,
                    "val_unweighted_full_law_objective_mean": 0.44,
                    "test_unweighted_active_objective_mean": 0.36,
                    "val_unweighted_active_objective_mean": 0.33,
                    "best_epoch_mean": 43.0,
                    "selection_metric": "val_root_mae",
                    "selection_metric_value_mean": 0.20,
                    "n_runs": 1,
                },
                {
                    "cell_id": "r12_seg10to12",
                    "baseline_family": "official_fno_sumlen",
                    "train_doc_count": 1024,
                    "test_root_mae_mean": 0.27,
                    "n_runs": 1,
                },
            ],
            "runs": [
                {
                    "cell_id": "r12_seg10to12",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "seed": 42,
                    "best_epoch": 43,
                    "selection_metric": "val_root_mae",
                    "selection_metric_value": 0.20,
                    "tree_checkpoint_metric": "val_root_mae",
                    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                }
            ],
        },
    ]

    payloads = _with_supervision_recovery_v3_payloads(payloads)
    summary = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        structural_cell="r12_seg10to12",
        package_order=["full50_leaf_full100_internal_count100"],
    )

    assert summary["common_tree_reference_label"] == "common_factorized_sketch_v1"
    assert summary["comparator_selection_status"] == "root_comparable"
    assert summary["canonical_tree_selection_metric"] == "val_root_mae"
    assert (
        summary["canonical_comparison_rule"]
        == "all tree ladder points selected on val_root_mae; local metrics are diagnostics"
    )
    assert summary["tree_checkpoint_metrics"] == ["val_root_mae"]

    recoverable_scope = summary["scope_tree_references"]["recoverable_v4"]
    structural_scope = summary["scope_tree_references"]["r12_seg10to12"]
    assert recoverable_scope["tree_reference_label"] == "common_factorized_sketch_v1"
    assert structural_scope["tree_reference_label"] == "common_factorized_sketch_v1"
    assert recoverable_scope["tree_checkpoint_metric"] == "val_root_mae"
    assert structural_scope["tree_checkpoint_metric"] == "val_root_mae"

    recoverable_row = next(
        row
        for row in summary["best_tree_summary"]
        if row["scope_key"] == "recoverable_v4"
    )
    assert recoverable_row["tree_checkpoint_metric"] == "val_root_mae"
    assert recoverable_row["tree_stage1_checkpoint_metric"] == "val_theorem_bootstrap_direct"
    assert recoverable_row["tree_val_root_mae"] == pytest.approx(0.11)
    assert recoverable_row["tree_best_epoch"] == pytest.approx(47.0)
    assert recoverable_row["tree_test_full_law_objective"] == pytest.approx(0.31)


def test_tradeoff_run_plan_reports_resolved_task_counts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "tradeoff",
        selection_config=None,
        preset="standard",
        phases="law_packages support_grid report",
        device_mode="cpu",
        max_workers=0,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        medium_epochs=5,
        medium_val_docs=1024,
        medium_exact_doc_limit=128,
        docs_epochs_batch_size=256,
        law_batch_size=256,
        law_epochs=10,
        support_batch_size=256,
        support_epochs=5,
        seed=42,
        data_seeds="0",
        batch_sizes=None,
        medium_batch_sizes=None,
        medium_seeds=None,
        docs_epochs_train_docs=None,
        docs_epochs_epochs=None,
        learnability_train_docs=None,
        learnability_weights=None,
        learnability_profiles=None,
        weight_ablation_train_docs=None,
        weight_ablation_profiles=None,
        law_package_names="tree_c2_only tree_all_laws",
        support_leaf_tokens="8 16",
        support_seeds="0",
        support_modes="supported",
        full_doc_anchor_train_docs=None,
        full_doc_anchor_seeds=None,
        full_doc_anchor_families="official_fno official_fno_sumlen",
        efficiency_anchor_mode="both",
        efficiency_train_docs="2048 4096",
        efficiency_anchor_train_docs_dense="256 512 1024",
        efficiency_anchor_seeds="0",
        efficiency_hardness_grid="structural_core_v1",
        efficiency_structural_cells="r4_seg4to6",
        oracle_budget_train_docs=4096,
        oracle_budget_seeds="0",
        oracle_budget_tree_families="tree_neural",
        oracle_budget_reference_families="official_fno official_fno_sumlen",
        oracle_budget_calls_per_doc="1.0",
        oracle_budget_full_doc_shares="0.5 1.0",
        oracle_budget_doc_consumption_modes="root_only doc_sequence",
        oracle_budget_local_split_modes="balanced",
        oracle_budget_tree_config_mode="parity",
        oracle_budget_capacity_root=None,
        tree_reference_mode="preset",
        tree_reference_capacity_root=None,
        tree_reference_preset="recoverable_slotwise_dense_v1",
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="structural_factorized_sketch_v3",
        large_batch_batch_sizes=None,
        large_batch_fixed_epochs=5,
        large_batch_target_steps=200,
        large_batch_lrs="0.001 0.002 0.004",
        migs="",
        write_selection_template=None,
        write_run_plan=None,
        plan_only=False,
        worker_task=None,
    )
    plan = build_run_plan(args, devices=[""])
    assert plan["phase_task_counts"]["law_packages"]["worker_tasks"] == 2
    assert plan["phase_task_counts"]["support_grid"]["worker_tasks"] == 2
    assert plan["phase_task_counts"]["report"]["worker_tasks"] == 0
    assert "oracle_budget_frontier" not in plan["phase_task_counts"]
    assert "efficiency_suite" not in plan["phase_task_counts"]


def test_tradeoff_run_plan_reports_supervision_recovery_counts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "tradeoff",
        selection_config=None,
        preset="standard",
        phases="supervision_recovery report",
        device_mode="cpu",
        max_workers=0,
        train_docs=4096,
        val_docs=512,
        test_docs=512,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        medium_epochs=5,
        medium_val_docs=1024,
        medium_exact_doc_limit=128,
        docs_epochs_batch_size=256,
        law_batch_size=256,
        law_epochs=10,
        support_batch_size=256,
        support_epochs=5,
        seed=42,
        data_seeds="0",
        batch_sizes=None,
        medium_batch_sizes=None,
        medium_seeds=None,
        docs_epochs_train_docs=None,
        docs_epochs_epochs=None,
        learnability_train_docs=None,
        learnability_weights=None,
        learnability_profiles=None,
        weight_ablation_train_docs=None,
        weight_ablation_profiles=None,
        law_package_names=None,
        support_leaf_tokens=None,
        support_seeds=None,
        support_modes=None,
        full_doc_anchor_train_docs=None,
        full_doc_anchor_seeds=None,
        full_doc_anchor_families="official_fno official_fno_sumlen",
        efficiency_anchor_mode="both",
        efficiency_train_docs="2048 4096",
        efficiency_anchor_train_docs_dense="256 512 1024",
        efficiency_anchor_seeds="0",
        efficiency_hardness_grid="structural_core_v1",
        efficiency_structural_cells="r4_seg4to6",
        oracle_budget_train_docs=4096,
        oracle_budget_seeds="0",
        oracle_budget_tree_families="tree_neural",
        oracle_budget_reference_families="official_fno official_fno_sumlen",
        oracle_budget_calls_per_doc="1.0",
        oracle_budget_full_doc_shares="0.5 1.0",
        oracle_budget_doc_consumption_modes="root_only doc_sequence",
        oracle_budget_local_split_modes="balanced",
        oracle_budget_tree_config_mode="parity",
        oracle_budget_capacity_root=None,
        tree_reference_mode="preset",
        tree_reference_capacity_root=None,
        tree_reference_preset="recoverable_slotwise_dense_v1",
        structural_tree_reference_mode="preset",
        structural_tree_reference_capacity_root=None,
        structural_tree_reference_preset="structural_factorized_sketch_v3",
        large_batch_batch_sizes=None,
        large_batch_fixed_epochs=5,
        large_batch_target_steps=200,
        large_batch_lrs="0.001 0.002 0.004",
        supervision_train_docs=None,
        supervision_leaf_profiles=None,
        supervision_internal_profiles=None,
        supervision_seeds=None,
        supervision_batch_size=128,
        supervision_epochs=5,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        supervision_recovery_train_docs="1024",
        supervision_recovery_seeds="0",
        supervision_recovery_tree_family="tree_neural",
        supervision_recovery_structural_cell="r12_seg10to12",
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
        migs="",
        write_selection_template=None,
        write_run_plan=None,
        plan_only=False,
        worker_task=None,
        scheduler_mode="global_per_run",
        default_job_granularity="family_train_seed",
        cleanup_stale_children=True,
        max_gpu_items_per_mig=1,
        report_sources=None,
    )
    plan = build_run_plan(args, devices=[""])
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 36
    assert plan["phase_task_counts"]["supervision_recovery"]["details"]["tree_family"] == "tree_neural"
    assert plan["phase_task_counts"]["supervision_recovery"]["details"]["structural_cell"] == "r12_seg10to12"


def test_checked_in_tradeoff_config_builds_plan() -> None:
    config_path = Path("config/markov/tradeoff_pipeline.standard.toml")
    assert config_path.exists()
    args = _parse_args(["--config", str(config_path), "--plan-only"])
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])
    assert plan["resolved_selection"]["train_docs"] == 10240
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 4096, 10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0, 1]
    assert plan["resolved_selection"]["supervision_recovery_tree_family"] == "tree_neural"
    assert plan["resolved_selection"]["supervision_recovery_structural_cell"] == "r12_seg10to12"
    assert plan["resolved_selection"]["runtime"]["data_mode"] == "resident"
    assert plan["resolved_selection"]["runtime"]["bucket_mode"] == "leaf_count_auto_queue"
    assert plan["resolved_selection"]["runtime"]["capacity_workers_per_mig"] == 2
    assert plan["resolved_selection"]["tree_reference"]["mode"] == "preset"
    assert plan["resolved_selection"]["tree_reference"]["preset"] == "common_factorized_sketch_v1"
    assert plan["resolved_selection"]["structural_tree_reference"]["preset"] == "common_factorized_sketch_v1"
    assert set(plan["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert plan["phase_task_counts"]["supervision_recovery"]["details"]["train_docs"] == [1024, 4096, 10240]


def test_checked_in_autoqueue_supervision_recovery_config_builds_plan(tmp_path: Path) -> None:
    config_path = Path("config/markov/tradeoff_pipeline.supervision_recovery_publication_10240_autoqueue.toml")
    assert config_path.exists()

    args = _parse_args(
        [
            "--config",
            str(config_path),
            "--plan-only",
        ]
    )
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert plan["resolved_selection"]["train_docs"] == 10240
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 2048, 4096, 10240]
    assert plan["resolved_selection"]["runtime"]["bucket_mode"] == "leaf_count_auto_queue"
    assert plan["resolved_selection"]["runtime"]["tree_batch_structural_pad_limit"] == 0.5
    assert plan["resolved_selection"]["runtime"]["tree_batch_auto_queue_min_docs"] == 8
    assert plan["resolved_selection"]["runtime"]["tree_batch_auto_queue_min_fill_ratio"] == 0.5
    assert plan["resolved_selection"]["tree_reference"]["mode"] == "preset"
    assert plan["resolved_selection"]["tree_reference"]["preset"] == "common_factorized_sketch_v1"
    assert plan["resolved_selection"]["structural_tree_reference"]["mode"] == "preset"
    assert (
        plan["resolved_selection"]["structural_tree_reference"]["preset"]
        == "common_factorized_sketch_v1"
    )
    assert set(plan["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 288


def test_checked_in_strong_tree_autoqueue_config_builds_extended_grid_plan() -> None:
    config_path = Path("config/markov/tradeoff_pipeline.supervision_recovery_strong_tree_autoqueue.toml")
    assert config_path.exists()

    args = _parse_args(
        [
            "--config",
            str(config_path),
            "--plan-only",
        ]
    )
    plan = build_run_plan(args, devices=["MIG-a", "MIG-b"])

    assert (
        plan["resolved_selection"]["supervision_recovery_packages"][0]
        == "full10"
    )
    assert (
        "full0_leaf_full100_internal_count100"
        in plan["resolved_selection"]["supervision_recovery_packages"]
    )
    assert (
        plan["resolved_selection"]["supervision_recovery_packages"][-1]
        == "full100_leaf_full100_internal_count100"
    )
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 384


def test_direct_task_materializes_effective_official_fno_config(tmp_path: Path) -> None:
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__full10__fno__d0",
        config={
            "train_docs": 1024,
            "state_dim": 32,
            "hidden_dim": 64,
            "n_epochs": 10,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 0.25,
            "fixed_leaf_tokens": 8,
            "pipeline_supervision_recovery_scope": "recoverable_v4",
            "pipeline_supervision_recovery_package": "full10",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["official_fno", "official_fno_sumlen"],
            "train_doc_counts": [1024],
            "seeds": [42],
        },
    )

    payload = json.loads((tmp_path / task.name / "task.request").read_text(encoding="utf-8"))
    assert payload["config"]["state_dim"] == 32
    assert payload["config"]["hidden_dim"] == 64
    assert payload["config"]["n_epochs"] == 10
    assert payload["config"]["batch_size"] == 256
    assert payload["config"]["lr"] == 1e-3
    assert payload["config"]["weight_decay"] == 0.25
    assert payload["config"]["fixed_leaf_tokens"] == 128
    assert payload["config"]["preserve_requested_leaf_tokens"] is False
    assert payload["config"]["official_fno_preserve_requested_leaf_tokens"] is False
    assert task.metadata["n_epochs"] == 10


def test_direct_task_materializes_structural_official_fno_geometry(tmp_path: Path) -> None:
    task = _direct_task(
        root=tmp_path,
        name="r12_seg10to12__train01024__full100__fno__d0",
        config={
            "train_docs": 1024,
            "state_dim": 32,
            "hidden_dim": 64,
            "n_epochs": 10,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 0.25,
            "fixed_leaf_tokens": 8,
            "pipeline_supervision_recovery_scope": "r12_seg10to12",
            "pipeline_supervision_recovery_scope_label": "structural_core_v1::r12_seg10to12",
            "pipeline_supervision_recovery_package": "full100",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": pipeline._structural_supervision_recovery_benchmark_name(
                "r12_seg10to12"
            ),
            "hardness_grid": "structural_core_v1",
            "grid_cell_ids": ["r12_seg10to12"],
            "baseline_families": ["official_fno", "official_fno_sumlen"],
            "train_doc_counts": [1024],
            "seeds": [42],
        },
    )

    payload = json.loads((tmp_path / task.name / "task.request").read_text(encoding="utf-8"))
    assert payload["config"]["n_regimes"] == 12
    assert payload["config"]["vocab_size"] == 48
    assert payload["config"]["generator_profile"] == "piecewise_disjoint_palette"
    assert payload["config"]["min_segments"] == 10
    assert payload["config"]["max_segments"] == 12
    assert payload["config"]["fixed_leaf_tokens"] == 128
    assert payload["config"]["state_dim"] == 32
    assert payload["config"]["hidden_dim"] == 64
    assert payload["config"]["n_epochs"] == 10


def test_direct_task_materializes_effective_official_fno_sumlen_config(tmp_path: Path) -> None:
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__full10__fno_sumlen__d0",
        config={
            "train_docs": 1024,
            "state_dim": 32,
            "hidden_dim": 64,
            "n_epochs": 10,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 0.25,
            "fixed_leaf_tokens": 8,
            "pipeline_supervision_recovery_scope": "recoverable_v4",
            "pipeline_supervision_recovery_package": "full10",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["official_fno_sumlen"],
            "train_doc_counts": [1024],
            "seeds": [42],
        },
    )

    payload = json.loads(
        (tmp_path / task.name / "task.request").read_text(encoding="utf-8")
    )
    assert payload["config"]["baseline_family"] == "official_fno_sumlen"
    assert payload["config"]["fixed_leaf_tokens"] == 128
    assert payload["config"]["preserve_requested_leaf_tokens"] is False
    assert payload["config"]["official_fno_preserve_requested_leaf_tokens"] is False


def test_direct_task_uses_two_stage_total_epochs_in_metadata(tmp_path: Path) -> None:
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__full10__tree_neural__d0",
        config={
            "train_docs": 1024,
            "n_epochs": 10,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "pipeline_supervision_recovery_scope": "recoverable_v4",
            "pipeline_supervision_recovery_package": "full10",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["tree_neural"],
            "train_doc_counts": [1024],
            "seeds": [42],
        },
    )

    payload = json.loads((tmp_path / task.name / "task.request").read_text(encoding="utf-8"))
    assert payload["config"]["n_epochs"] == 10
    assert task.metadata["n_epochs"] == 32


def test_direct_task_reuses_existing_stage1_artifact_for_tree_diagnostics(tmp_path: Path) -> None:
    from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
        _effective_train_config_for_full_doc_run,
        _tree_stage1_expected_layout_metadata,
        resolve_full_doc_diagnostic_benchmark,
    )
    from src.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig
    from src.ctreepo.sim.core.theorem_feature_route import (
        write_theorem_feature_stage1_artifact,
    )

    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    artifact_root = tmp_path / "stage1_cache"
    effective = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=1024,
        config=OPSCountConfig(
            train_docs=1024,
            n_epochs=10,
            tree_training_schedule="two_stage",
            tree_stage1_epochs=12,
            tree_stage2_epochs=20,
            tree_stage1_artifact_root=str(artifact_root),
            tree_stage1_resume_if_available=True,
        ),
    )
    write_theorem_feature_stage1_artifact(
        effective.tree_stage1_artifact_dir,
        model_state={"weight": torch.tensor([1.0])},
        metadata={
            "selection_metric_name": "val_root_mae",
            "selection_metric_value": 0.1,
            "best_epoch": 0,
            "epochs_completed": 12,
            "training_schedule": "two_stage",
            "artifact_source": "trained",
            "n_regimes": int(effective.n_regimes),
            "vocab_size": int(effective.vocab_size),
            "generator_profile": str(effective.generator_profile),
            "fixed_leaf_tokens": int(effective.fixed_leaf_tokens),
            **_tree_stage1_expected_layout_metadata(effective),
        },
    )

    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__full10__tree_neural__d0",
        config={
            "train_docs": 1024,
            "n_epochs": 10,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_stage1_artifact_root": str(artifact_root),
            "tree_stage1_resume_if_available": True,
            "pipeline_supervision_recovery_scope": "recoverable_v4",
            "pipeline_supervision_recovery_package": "full10",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["tree_neural"],
            "train_doc_counts": [1024],
            "seeds": [42],
        },
    )

    payload = json.loads((tmp_path / task.name / "task.request").read_text(encoding="utf-8"))
    assert payload["config"]["tree_stage1_artifact_dir"] == str(
        Path(effective.tree_stage1_artifact_dir).expanduser()
    )
    assert payload["config"]["tree_stage1_epochs"] == 0
    assert task.metadata["n_epochs"] == 20


def test_direct_task_preserves_prepared_data_and_exact_eval_settings(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__full10__tree_neural__prepared",
        config={
            "train_docs": 1024,
            "n_epochs": 10,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_exact_eval_max_docs": 64,
            "prepared_data_root": str(prepared_root),
            "prepared_data_allow_create": False,
            "pipeline_supervision_recovery_scope": "recoverable_v4",
            "pipeline_supervision_recovery_package": "full10",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["tree_neural"],
            "train_doc_counts": [1024],
            "seeds": [42],
        },
    )

    payload = json.loads((tmp_path / task.name / "task.request").read_text(encoding="utf-8"))
    assert payload["config"]["tree_exact_eval_max_docs"] == 64
    assert payload["config"]["prepared_data_root"] == str(prepared_root)
    assert payload["config"]["prepared_data_allow_create"] is False


def test_run_worker_full_doc_diagnostics_uses_central_config_serializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _direct_task(
        root=tmp_path,
        name="recoverable_v4__train01024__tree_neural__worker_codec",
        config={
            "train_docs": 1024,
            "n_epochs": 1,
            "tree_local_law_weight": 0.8,
            "tree_task_objective_weight": 0.2,
            "tree_c1_relative_weight": 0.0,
            "tree_c2_relative_weight": 1.0,
            "tree_c3_relative_weight": 0.0,
            "tree_document_loss_normalization_mode": "supervised_docs",
            "tree_supervision_source": "manifest",
            "pipeline_supervision_leaf_profile": "full100",
        },
        worker_kind="full_doc_diagnostics",
        extra_payload={
            "benchmark_name": "recoverable_v4",
            "baseline_families": ["tree_neural"],
            "train_doc_counts": [1024],
            "seeds": [0],
        },
    )
    task_request = task.output_path.parent / "task.request"
    output_json = task.output_path

    import src.ctreepo.sim.core.full_doc_anchor_diagnostics as diagnostics

    captured: dict[str, object] = {}

    def _fake_run_markov_full_doc_anchor_diagnostics(**kwargs):
        captured.update(kwargs)
        return {
            "benchmark": kwargs.get("benchmark_name", ""),
            "runs": [],
            "aggregate_rows": [],
        }

    monkeypatch.setattr(
        diagnostics,
        "run_markov_full_doc_anchor_diagnostics",
        _fake_run_markov_full_doc_anchor_diagnostics,
    )

    assert pipeline._run_worker(task_request) == 0
    assert dict(captured["config_overrides"])["local_law_weight"] == pytest.approx(0.8)
    assert dict(captured["config_overrides"])["task_objective_weight"] == pytest.approx(0.2)
    assert dict(captured["config_overrides"])["c2_relative_weight"] == pytest.approx(1.0)
    assert (
        dict(captured["config_overrides"])["tree_document_loss_normalization_mode"]
        == "supervised_docs"
    )
    assert dict(captured["config_overrides"])["tree_supervision_source"] == "manifest"
    assert "tree_local_law_weight" not in dict(captured["config_overrides"])

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["config"]["local_law_weight"] == pytest.approx(0.8)
    assert payload["config"]["task_objective_weight"] == pytest.approx(0.2)
    assert payload["config"]["c2_relative_weight"] == pytest.approx(1.0)
    assert payload["config"]["tree_document_loss_normalization_mode"] == "supervised_docs"
    assert payload["config"]["tree_supervision_source"] == "manifest"
    assert payload["config"]["pipeline_supervision_leaf_profile"] == "full100"
    assert "tree_local_law_weight" not in payload["config"]


def test_resolve_devices_prefers_free_migs(monkeypatch) -> None:
    listing = "\n".join(
        [
            "GPU 0: GPU0 (UUID: GPU-0)",
            "  MIG 1g.24gb     Device  0: (UUID: MIG-free-a)",
            "  MIG 1g.24gb     Device  1: (UUID: MIG-busy-a)",
            "GPU 1: GPU1 (UUID: GPU-1)",
            "  MIG 1g.24gb     Device  0: (UUID: MIG-free-b)",
        ]
    )
    gpu0_xml = """
    <nvidia_smi_log>
      <gpu>
        <mig_devices>
          <mig_device><index>0</index><fb_memory_usage><total>24192 MiB</total><used>67 MiB</used><free>24126 MiB</free></fb_memory_usage></mig_device>
          <mig_device><index>1</index><fb_memory_usage><total>24192 MiB</total><used>23942 MiB</used><free>251 MiB</free></fb_memory_usage></mig_device>
        </mig_devices>
      </gpu>
    </nvidia_smi_log>
    """
    gpu1_xml = """
    <nvidia_smi_log>
      <gpu>
        <mig_devices>
          <mig_device><index>0</index><fb_memory_usage><total>24192 MiB</total><used>67 MiB</used><free>24126 MiB</free></fb_memory_usage></mig_device>
        </mig_devices>
      </gpu>
    </nvidia_smi_log>
    """

    def _fake_run(cmd, **kwargs):
        if cmd == ["nvidia-smi", "-L"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")
        if cmd == ["nvidia-smi", "-i", "0", "-q", "-x"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=gpu0_xml, stderr="")
        if cmd == ["nvidia-smi", "-i", "1", "-q", "-x"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=gpu1_xml, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(pipeline.subprocess, "run", _fake_run)

    args = argparse.Namespace(device_mode="auto", migs="", max_workers=0)
    assert _resolve_devices(args) == ["MIG-free-a", "MIG-free-b"]
    args.migs = "MIG-busy-a"
    assert _resolve_devices(args) == ["MIG-busy-a"]


def test_checked_in_iteration_and_publication_tradeoff_configs_build_plans() -> None:
    iteration = Path("config/markov/tradeoff_pipeline.iteration.toml")
    publication = Path("config/markov/tradeoff_pipeline.publication.toml")
    no10240 = Path("config/markov/tradeoff_pipeline.no10240.toml")
    assert iteration.exists()
    assert publication.exists()
    assert no10240.exists()

    iteration_plan = build_run_plan(
        _parse_args(["--config", str(iteration), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )
    publication_plan = build_run_plan(
        _parse_args(["--config", str(publication), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )
    no10240_plan = build_run_plan(
        _parse_args(["--config", str(no10240), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )

    assert iteration_plan["resolved_selection"]["train_docs"] == 10240
    assert publication_plan["resolved_selection"]["train_docs"] == 10240
    assert no10240_plan["resolved_selection"]["train_docs"] == 4096
    assert iteration_plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 2048, 4096, 10240]
    assert publication_plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 4096, 10240]
    assert no10240_plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 2048, 4096]
    assert set(iteration_plan["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert set(publication_plan["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert set(no10240_plan["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert (
        publication_plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"]
        == no10240_plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"]
    )
    assert (
        iteration_plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"]
        > publication_plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"]
    )


def test_checked_in_v3_tradeoff_config_builds_plan() -> None:
    config_path = Path("config/markov/tradeoff_pipeline.v3.toml")
    assert config_path.exists()

    plan = build_run_plan(
        _parse_args(["--config", str(config_path), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )

    assert plan["resolved_selection"]["preset"] == "v3"
    assert plan["resolved_selection"]["supervision_recovery_recoverable_benchmark"] == "recoverable_v4_t128"
    assert plan["resolved_selection"]["supervision_recovery_structural_grid"] == "structural_core_v1_t128"
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024, 4096, 10240]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0, 1]
    assert plan["resolved_selection"]["supervision_recovery_leaf_token_ladder"] == [32, 16, 8]
    assert plan["resolved_selection"]["supervision_recovery_depth_discount_gammas"] == pytest.approx([1.0, 0.9])
    assert plan["resolved_selection"]["supervision_recovery_packages"] == [
        "full100",
        "r100_superset_local_eq_10p0",
        "r100_superset_local_eq_15p0",
        "r100_superset_local_eq_20p0",
    ]
    assert plan["resolved_selection"]["tree_reference"]["preset"] == "comparison_grid_v3"
    assert (
        plan["resolved_selection"]["tree_reference"]["preset_recipe"]
        == "unified_g_full_local_laws_v1"
    )
    assert plan["resolved_selection"]["structural_tree_reference"]["preset"] == "comparison_grid_v3"
    assert (
        plan["resolved_selection"]["structural_tree_reference"]["preset_recipe"]
        == "unified_g_full_local_laws_v1"
    )
    assert set(plan["phase_task_counts"]) == {"supervision_recovery", "report"}


def test_checked_in_long_v4_supervision_configs_build_plans() -> None:
    long_v4 = Path("config/markov/tradeoff_pipeline.long_v4.toml")
    long_v4_incremental = Path("config/markov/tradeoff_pipeline.long_v4_incremental.toml")
    supervision_followup = Path("config/markov/tradeoff_pipeline.long_v4_supervision_followup.toml")
    assert long_v4.exists()
    assert long_v4_incremental.exists()
    assert supervision_followup.exists()

    long_plan = build_run_plan(
        _parse_args(["--config", str(long_v4), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )
    incremental_plan = build_run_plan(
        _parse_args(["--config", str(long_v4_incremental), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )
    followup_plan = build_run_plan(
        _parse_args(["--config", str(supervision_followup), "--plan-only"]),
        devices=["MIG-a", "MIG-b"],
    )

    assert "supervision_recovery" in long_plan["phase_task_counts"]
    assert "supervision_recovery" in incremental_plan["phase_task_counts"]
    assert "supervision_recovery" in followup_plan["phase_task_counts"]
    assert long_plan["phase_task_counts"]["supervision_recovery"]["details"]["train_docs"] == [1024, 2048, 4096]
    assert followup_plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] > 0
    assert "support_grid" not in followup_plan["phase_task_counts"]


def test_stage_report_sources_copies_into_version_root(tmp_path: Path) -> None:
    output_root = tmp_path / "version"
    manifest = _load_report_version_manifest(output_root)
    source_path = tmp_path / "external.json"
    _write_json(source_path, {"value": 1})

    _stage_report_sources(
        output_root=output_root,
        manifest=manifest,
        overrides={"law_comparison_json": source_path},
    )
    selected = manifest["selected_sources"]["law_comparison_json"]
    staged_path = output_root / selected["relpath"]
    source_path.unlink()

    assert selected["origin"] == "staged_copy"
    assert staged_path.exists()
    assert json.loads(staged_path.read_text(encoding="utf-8"))["value"] == 1


def test_refresh_selected_source_statuses_marks_modified_staged_copy_stale(tmp_path: Path) -> None:
    output_root = tmp_path / "version"
    manifest = _load_report_version_manifest(output_root)
    source_path = tmp_path / "external.json"
    _write_json(source_path, {"value": 1})
    _stage_report_sources(
        output_root=output_root,
        manifest=manifest,
        overrides={"support_summary": source_path},
    )
    selected = manifest["selected_sources"]["support_summary"]
    staged_path = output_root / selected["relpath"]
    _write_json(staged_path, {"value": 2})

    args = argparse.Namespace(
        preset="smoke",
        phases="report",
        selection_config=None,
        device_mode="cpu",
        max_workers=0,
        train_docs=1024,
        val_docs=64,
        test_docs=64,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        fixed_leaf_tokens=8,
        state_dim=32,
        hidden_dim=64,
        fno_width=16,
        fno_n_modes=8,
        fno_n_layers=2,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        medium_epochs=5,
        medium_val_docs=64,
        medium_exact_doc_limit=64,
        docs_epochs_batch_size=128,
        law_batch_size=128,
        law_epochs=5,
        support_batch_size=128,
        support_epochs=5,
        seed=42,
        data_seeds="0",
        batch_sizes=None,
        medium_batch_sizes=None,
        medium_seeds=None,
        docs_epochs_train_docs=None,
        docs_epochs_epochs=None,
        learnability_train_docs=None,
        learnability_weights=None,
        learnability_profiles=None,
        weight_ablation_train_docs=None,
        weight_ablation_profiles=None,
        law_package_names=None,
        support_leaf_tokens=None,
        support_seeds=None,
        support_modes=None,
        full_doc_anchor_train_docs=None,
        full_doc_anchor_seeds=None,
        full_doc_anchor_families="official_fno official_fno_sumlen",
        efficiency_anchor_mode="both",
        efficiency_train_docs=None,
        efficiency_anchor_train_docs_dense=None,
        efficiency_anchor_seeds=None,
        efficiency_hardness_grid="structural_core_v1",
        efficiency_structural_cells=None,
        oracle_budget_train_docs=None,
        oracle_budget_seeds=None,
        oracle_budget_tree_families=None,
        oracle_budget_reference_families="official_fno official_fno_sumlen",
        oracle_budget_calls_per_doc=None,
        oracle_budget_full_doc_shares=None,
        oracle_budget_doc_consumption_modes=None,
        oracle_budget_local_split_modes=None,
        oracle_budget_tree_config_mode="parity",
        oracle_budget_capacity_root=None,
        tree_reference_mode="default",
        tree_reference_capacity_root=None,
        large_batch_batch_sizes=None,
        large_batch_fixed_epochs=5,
        large_batch_target_steps=200,
        large_batch_lrs="0.001 0.002 0.004",
        supervision_train_docs=None,
        supervision_leaf_profiles=None,
        supervision_internal_profiles=None,
        supervision_seeds=None,
        supervision_batch_size=128,
        supervision_epochs=5,
        supervision_min_tokens=64,
        supervision_max_tokens=64,
        supervision_min_segments=2,
        supervision_max_segments=6,
        supervision_fixed_leaf_tokens=8,
        runtime_data_mode="resident",
        runtime_bucket_mode="exact_then_bucketed",
        runtime_preload_splits="train val test",
        runtime_preload_targets=True,
        runtime_workers_per_mig=1,
        runtime_allow_multi_worker_screen=True,
        runtime_capacity_workers_per_mig=2,
        scheduler_mode="global_per_run",
        default_job_granularity="family_train_seed",
        cleanup_stale_children=True,
        max_gpu_items_per_mig=1,
        report_sources=None,
        output_root=output_root,
    )
    _refresh_selected_source_statuses(manifest, output_root=output_root, args=args)

    assert manifest["selected_sources"]["support_summary"]["status"] == "stale"


def test_phase_attempt_retention_keeps_multiple_attempts(tmp_path: Path) -> None:
    output_root = tmp_path / "version"
    manifest = _load_report_version_manifest(output_root)
    args = _parse_args(["--preset", "smoke", "--phases", "batch_timing", "--output-root", str(output_root)])
    first_path = output_root / "batch_timing" / "attempts" / "a1" / "summary.json"
    second_path = output_root / "batch_timing" / "attempts" / "a2" / "summary.json"
    _write_json(first_path, {"value": 1})
    _write_json(second_path, {"value": 2})
    fingerprint = _phase_config_fingerprint(args, "batch_timing")

    _register_phase_source(
        manifest,
        output_root=output_root,
        phase="batch_timing",
        source_key="batch_timing_summary",
        attempt_id="a1",
        config_fingerprint=fingerprint,
        artifact_path=first_path,
    )
    _register_phase_source(
        manifest,
        output_root=output_root,
        phase="batch_timing",
        source_key="batch_timing_summary",
        attempt_id="a2",
        config_fingerprint=fingerprint,
        artifact_path=second_path,
    )
    manifest_path = _write_report_version_manifest(output_root, manifest)
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    attempts = reloaded["phase_attempts"]["batch_timing"]["attempts"]
    assert set(attempts) == {"a1", "a2"}
    assert reloaded["selected_sources"]["batch_timing_summary"]["selected_attempt_id"] == "a2"


def test_refresh_existing_output_root_rebuilds_supervision_recovery_and_report(tmp_path: Path) -> None:
    output_root = tmp_path / "refreshable"
    raw_root = (
        output_root
        / "supervision_recovery"
        / "attempts"
        / "20260412_000000_000000"
        / "raw"
    )
    payloads = _with_supervision_recovery_v3_payloads(
        [
            {
                "config": {
                    "pipeline_supervision_recovery_package": "full100",
                    "pipeline_supervision_recovery_scope": "recoverable_v4",
                    "pipeline_supervision_recovery_scope_label": "recoverable_v4",
                    "data_seed": 0,
                    "train_docs": 10240,
                    "fixed_leaf_tokens": 64,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                },
                "aggregate_rows": [
                    {
                        "baseline_family": "tree_neural",
                        "train_doc_count": 10240,
                        "test_root_mae_mean": 0.03,
                        "tree_supervision_source": "manifest",
                        "local_estimand_mode": "span_mass_ipw_sum",
                        "c2_pair_weighting_mode": "pair_ipw_geomean",
                        "fixed_leaf_tokens": 64,
                        "requested_fixed_leaf_tokens": 64,
                        "executed_fixed_leaf_tokens": 64,
                        "computed_assumed_doc_tokens": 128,
                        "leaves_per_doc": 2,
                        "tree_reference_label": "unified_g_full_local_laws_v1",
                    },
                    {
                        "baseline_family": "official_fno",
                        "train_doc_count": 10240,
                        "test_root_mae_mean": 0.02,
                        "fixed_leaf_tokens": 128,
                        "requested_fixed_leaf_tokens": 128,
                        "executed_fixed_leaf_tokens": 128,
                    },
                ],
                "runs": [],
            },
            {
                "config": {
                    "pipeline_supervision_recovery_package": "full100",
                    "pipeline_supervision_recovery_scope": "r12_seg10to12",
                    "pipeline_supervision_recovery_scope_label": "structural_core_v1::r12_seg10to12",
                    "data_seed": 0,
                    "train_docs": 10240,
                    "fixed_leaf_tokens": 64,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                },
                "aggregate_rows": [
                    {
                        "baseline_family": "tree_neural",
                        "train_doc_count": 10240,
                        "test_root_mae_mean": 0.11,
                        "tree_supervision_source": "manifest",
                        "local_estimand_mode": "span_mass_ipw_sum",
                        "c2_pair_weighting_mode": "pair_ipw_geomean",
                        "fixed_leaf_tokens": 64,
                        "requested_fixed_leaf_tokens": 64,
                        "executed_fixed_leaf_tokens": 64,
                        "computed_assumed_doc_tokens": 128,
                        "leaves_per_doc": 2,
                        "tree_reference_label": "unified_g_full_local_laws_v1",
                    },
                    {
                        "baseline_family": "official_fno",
                        "train_doc_count": 10240,
                        "test_root_mae_mean": 0.09,
                        "fixed_leaf_tokens": 128,
                        "requested_fixed_leaf_tokens": 128,
                        "executed_fixed_leaf_tokens": 128,
                    },
                ],
                "runs": [],
            },
        ]
    )
    for index, payload in enumerate(payloads):
        _write_json(raw_root / f"payload_{index}" / "summary.json", payload)

    _write_json(
        output_root / "experiment_status.json",
        {
            "experiment_id": "refresh-test",
            "state": "completed",
            "active_phase": "",
            "items_total": 3,
            "completed_items": 2,
            "failed_items": 1,
            "active_items": 0,
            "pending_items": 0,
            "percent_complete": 100.0,
            "artifact_targets": [],
            "live_child_status_path": "",
            "metadata": {"adapter": "markov_tree"},
        },
    )
    _write_json(
        output_root / "scheduler_status.json",
        {
            "generated_at": "2026-04-12T00:00:00+00:00",
            "state": "failed",
            "status_kind": "experiment_progress",
            "experiment_id": "refresh-test",
            "experiment_adapter": "markov_tree",
            "experiment_title": "markov_tradeoff_pipeline",
            "active_phase": "supervision_recovery",
            "items_total": 3,
            "initial_items_total": 3,
            "dynamic_items_added": 0,
            "completed_items": 2,
            "failed_items": 1,
            "active_items": 0,
            "pending_items": 0,
            "percent_complete": 100.0,
            "progress_bar": "####################",
            "artifact_targets": [],
            "phase_progress": {
                "supervision_recovery": {
                    "total": 2,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 1,
                    "percent_complete": 100.0,
                    "epochs_completed": 20,
                    "epochs_total": 40,
                    "epoch_percent": 50.0,
                },
                "report": {
                    "total": 1,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 0,
                    "percent_complete": 100.0,
                },
            },
            "by_scope": {
                "recoverable_v4": {
                    "total": 2,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 1,
                    "percent_complete": 100.0,
                }
            },
            "by_train_docs": {
                "10240": {
                    "total": 2,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 1,
                    "percent_complete": 100.0,
                }
            },
            "by_model_family": {
                "tree_neural": {
                    "total": 1,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 0,
                    "percent_complete": 100.0,
                },
                "official_fno": {
                    "total": 1,
                    "completed": 0,
                    "active": 0,
                    "pending": 0,
                    "failed": 1,
                    "percent_complete": 100.0,
                },
            },
            "by_package": {
                "full100": {
                    "total": 2,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 1,
                    "percent_complete": 100.0,
                }
            },
            "by_worker_kind": {
                "gpu_command": {
                    "total": 2,
                    "completed": 1,
                    "active": 0,
                    "pending": 0,
                    "failed": 1,
                    "percent_complete": 100.0,
                }
            },
            "active_item_details": [],
            "first_failed_item": {"item_id": "old-failure"},
        },
    )

    args = _parse_args(
        [
            "--output-root",
            str(output_root),
            "--phases",
            "supervision_recovery,report",
            "--device-mode",
            "cpu",
            "--refresh-existing-output-root",
        ]
    )
    result = _refresh_existing_tradeoff_outputs(args, output_root=output_root)

    assert (output_root / "supervision_recovery" / "summary.json").exists()
    assert (output_root / "tradeoff_report" / "summary.json").exists()
    assert (output_root / "report_version_manifest.json").exists()
    assert (output_root / "pipeline_summary.json").exists()
    assert result["refreshed"]["supervision_recovery_summary"].endswith(
        "supervision_recovery/summary.json"
    )
    version_manifest = _load_report_version_manifest(output_root)
    assert version_manifest["selected_sources"]["supervision_recovery_summary"]["status"] == "ready"
    experiment_status = json.loads(
        (output_root / "experiment_status.json").read_text(encoding="utf-8")
    )
    assert experiment_status["state"] == "completed"
    assert experiment_status["failed_items"] == 0
    assert experiment_status["completed_items"] == 3
    scheduler_status = json.loads(
        (output_root / "scheduler_status.json").read_text(encoding="utf-8")
    )
    assert scheduler_status["state"] == "completed"
    assert scheduler_status["failed_items"] == 0
    assert scheduler_status["completed_items"] == 3
    assert scheduler_status["phase_progress"]["supervision_recovery"]["completed"] == 2
    assert scheduler_status["phase_progress"]["supervision_recovery"]["failed"] == 0
    assert scheduler_status["phase_progress"]["supervision_recovery"]["epochs_completed"] == 40
    assert "first_failed_item" not in scheduler_status


def test_load_supervision_recovery_refresh_payloads_prefers_run_payloads(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_json(raw_root / "payload_0" / "summary.json", {})
    run_payload = {
        "config": {
            "pipeline_supervision_recovery_package": "full90",
            "pipeline_supervision_recovery_scope": "r12_seg10to12",
            "pipeline_supervision_recovery_scope_label": "structural_core_v1::r12_seg10to12",
            "data_seed": 42,
            "train_docs": 10240,
            "fixed_leaf_tokens": 128,
        },
        "baseline_family": "tree_neural",
        "benchmark": "structural_core_v1_t128::r12_seg10to12",
        "cell_id": "r12_seg10to12",
        "comparison_mode": "exact_collapse",
        "comparison_semantics": "locked_comparator",
        "comparison_semantics_label": "exact_full_doc_parity",
        "run_intent_hash": "intent-123",
        "run_intent_validation_status": "validated",
        "test_root_mae": 0.375,
        "val_root_mae": 0.3125,
        "train_root_mae": 0.28125,
        "train_doc_count": 10240,
        "requested_fixed_leaf_tokens": 128,
        "executed_fixed_leaf_tokens": 128,
        "executed_leaves_per_doc": 1,
        "hardness_grid": "r12_seg10to12",
    }
    run_path = (
        raw_root
        / "payload_0"
        / "summary_artifacts"
        / "runs"
        / "r12_seg10to12__tree_neural__train_10240__seed_42.json"
    )
    _write_json(run_path, run_payload)

    payloads = _load_supervision_recovery_refresh_payloads(raw_root)

    assert len(payloads) == 1
    payload = dict(payloads[0])
    assert payload["source_summary_json"] == str(run_path)
    assert payload["benchmark"] == "structural_core_v1_t128::r12_seg10to12"
    assert payload["config"]["pipeline_supervision_recovery_package"] == "full90"
    assert len(payload["aggregate_rows"]) == 1
    assert payload["aggregate_rows"][0]["baseline_family"] == "tree_neural"
    assert payload["aggregate_rows"][0]["test_root_mae"] == pytest.approx(0.375)


def test_tradeoff_experiment_spec_uses_canonical_surface(tmp_path: Path) -> None:
    args = _parse_args(
        [
            "--output-root",
            str(tmp_path / "tradeoff"),
            "--phases",
            "supervision_recovery report",
            "--migs",
            "MIG-a MIG-b",
        ]
    )
    plan = {
        "resolved_selection": {
            "supervision_recovery_tree_family": "tree_neural",
            "supervision_recovery_structural_cell": "r12_seg10to12",
        },
        "phase_task_counts": {
            "supervision_recovery": {"worker_tasks": 10},
            "report": {"worker_tasks": 0},
        },
        "devices": ["MIG-a", "MIG-b"],
    }
    spec = _tradeoff_experiment_spec(
        args=args,
        output_root=Path(args.output_root),
        run_plan=plan,
    )
    assert spec.adapter_id == "markov_tree"
    assert spec.title == "markov_tradeoff_pipeline"
    assert "supervision_recovery" in {phase.phase_id for phase in spec.phases}


def test_supervision_recovery_summary_emits_canonical_result_rows(tmp_path: Path) -> None:
    args = _parse_args(
        [
            "--output-root",
            str(tmp_path / "tradeoff"),
            "--phases",
            "supervision_recovery",
            "--migs",
            "MIG-a",
        ]
    )
    spec = _tradeoff_experiment_spec(
        args=args,
        output_root=tmp_path / "tradeoff",
        run_plan={
            "resolved_selection": {
                "supervision_recovery_tree_family": "tree_neural",
                "supervision_recovery_structural_cell": "r12_seg10to12",
            },
            "phase_task_counts": {"supervision_recovery": {"worker_tasks": 1}},
            "devices": ["MIG-a"],
        },
    )
    summary = {
        "tree_family": "tree_neural",
        "scopes": {
            "recoverable_v4": {
                "scope_label": "recoverable_v4",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            {
                                "package_name": "full10_leaf_count50_internal_count50",
                                "tree_test_root_mae": 0.11,
                                "tree_val_root_mae": 0.09,
                                "tree_test_leaf_mae": 0.04,
                                "tree_test_merge_mae": 0.06,
                                "tree_test_full_law_objective": 0.22,
                                "tree_val_full_law_objective": 0.18,
                                "tree_test_active_objective": 0.14,
                                "tree_val_active_objective": 0.12,
                                "tree_best_epoch": 32,
                                "leaf_supervision_kind": "count_only",
                                "leaf_label_rate": 0.5,
                                "internal_supervision_kind": "count_only",
                                "internal_label_rate": 0.5,
                                "fno_family_rows": {
                                    "official_fno": {"test_root_mae": 0.08, "val_root_mae": 0.07},
                                },
                            }
                        ],
                    }
                ],
            }
        },
    }
    rows = _supervision_recovery_result_rows_from_summary(spec=spec, summary=summary)
    assert rows
    tree_rows = [row for row in rows if row.method_ref.family == "tree_neural"]
    assert any(row.metric_name == "test_root_mae" for row in tree_rows)
    assert tree_rows[0].supervision_ref is not None
    assert tree_rows[0].supervision_ref.coverage_label == "full10_leaf_count50_internal_count50"
    fno_rows = [row for row in rows if row.method_ref.family == "official_fno"]
    assert any(row.metric_name == "test_root_mae" for row in fno_rows)
    assert fno_rows[0].supervision_ref is not None
    assert fno_rows[0].supervision_ref.root_rate == pytest.approx(0.1)
    assert fno_rows[0].supervision_ref.leaf_rate == pytest.approx(0.0)
    assert fno_rows[0].supervision_ref.internal_rate == pytest.approx(0.0)
    assert fno_rows[0].method_ref.control_ref is None
