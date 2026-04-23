from __future__ import annotations

import json
from pathlib import Path

from scripts.launch_markov_sticky_simple_fixed10240_quick import (
    SURFACE_ALLOCATION_POLICY_GRID,
    SURFACE_FULL_GRID,
    SURFACE_REPAIR_LEAF128_COUNTONLY,
    _balanced_node_mass_package,
    _build_pipeline_config,
    _depth_equal_mass_package,
    _leaf_mass_package,
    _missing_task_names_for_surface,
    _package_leaf_overrides_for_missing_tasks,
    _required_task_names_for_surface,
)
from scripts.run_markov_optimization_tradeoff_pipeline import (
    SUPERVISION_RECOVERY_PACKAGE_SPECS,
    _one_leaf_package_has_local_supervision,
)
from src.ctreepo.sim.core.markov_study_names import (
    resolve_supervision_recovery_package_name,
)
from scripts.render_markov_sticky_simple_fixed10240_current import (
    _build_current_supervision_recovery_summary,
    _build_coverage_summary,
)


def _write_raw_summary(
    root: Path,
    *,
    job_key: str,
    task_name: str,
    package_name: str,
    scope_key: str,
    baseline_family: str,
    leaf_tokens: int,
    root_mae: float,
) -> None:
    path = (
        root
        / job_key
        / "supervision_recovery"
        / "attempts"
        / "20260415_000000_000000"
        / "raw"
        / task_name
        / "summary.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "pipeline_supervision_recovery_package": package_name,
            "pipeline_supervision_recovery_scope": scope_key,
            "pipeline_supervision_recovery_scope_label": scope_key,
            "train_docs": 10240,
            "data_seed": 0,
            "pipeline_tree_reference_mode": "preset",
            "pipeline_tree_reference_label": "unified_g_full_local_laws_v1",
            "tree_training_schedule": "two_stage",
            "tree_checkpoint_metric": "val_root_mae",
            "fixed_leaf_tokens": int(leaf_tokens),
            "comparison_mode": "comparable",
            "depth_discount_gamma": 1.0,
            "computed_assumed_doc_tokens": 128,
            "computed_assumed_leaves": max(1, int(round(128 / int(leaf_tokens)))),
            "computed_assumed_internal_nodes": max(
                0,
                int(round(128 / int(leaf_tokens))) - 1,
            ),
        },
        "aggregate_rows": [
            {
                "cell_id": scope_key,
                "baseline_family": baseline_family,
                "train_doc_count": 10240,
                "test_root_mae_mean": float(root_mae),
                "fixed_leaf_tokens": int(leaf_tokens),
                "test_mean_leaves_per_doc": float(128 / int(leaf_tokens)),
                "comparison_mode": "comparable",
                "comparison_semantics": "current",
                "comparison_semantics_label": "current",
                "run_intent_hash": f"{task_name}_intent",
                "run_intent_validation_status": "current",
                "requested_fixed_leaf_tokens": int(leaf_tokens),
                "executed_fixed_leaf_tokens": int(leaf_tokens),
                "depth_discount_gamma": 1.0,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _materialize_visible_job(root: Path, job_key: str) -> None:
    status_path = root / job_key / "experiment_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "state": "completed",
                "completed_items": 1,
                "active_items": 0,
                "pending_items": 0,
                "failed_items": 0,
                "items_total": 1,
            }
        ),
        encoding="utf-8",
    )


def test_full_grid_required_rows_include_deciles_and_leaf128_mass_fill_markers() -> None:
    required = _required_task_names_for_surface(SURFACE_FULL_GRID)

    assert "recoverable_v5_t128__train10240__full60__leaf128__tree_neural__d0" in required
    assert "recoverable_v5_t128__train10240__full40__leaf128__tree_neural__d0" in required
    assert "r12_p079__train10240__full30__leaf128__fno__d0" in required
    assert (
        "recoverable_v5_t128__train10240__full60_leaf_full100_internal_count100__leaf128__tree_neural__d0"
        in required
    )
    assert (
        "r12_p079__train10240__r60_leaf_mass_eq_40p0__leaf128__tree_neural__d0"
        in required
    )
    assert (
        "recoverable_v5_t128__train10240__r90_leaf_mass_eq_10p0__leaf064__tree_neural__d0"
        in required
    )
    assert (
        "recoverable_v5_t128__train10240__r100_leaf_mass_eq_0p0__leaf128__tree_neural__d0"
        not in required
    )


def test_missing_rows_respect_landed_one_leaf_rows_on_existing_root(tmp_path: Path) -> None:
    output_root = tmp_path / "sticky_simple_root"
    _write_raw_summary(
        output_root,
        job_key="oneleaf_root_budget_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0",
        package_name="full90",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.4,
    )
    _write_raw_summary(
        output_root,
        job_key="oneleaf_root_budget_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full90__leaf128__fno__d0",
        package_name="full90",
        scope_key="recoverable_v5_t128",
        baseline_family="fno",
        leaf_tokens=128,
        root_mae=0.4,
    )
    _write_raw_summary(
        output_root,
        job_key="oneleaf_local_law_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full90_leaf_full100_internal_count100__leaf128__tree_neural__d0",
        package_name="full90_leaf_full100_internal_count100",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.3,
    )

    missing = _missing_task_names_for_surface(output_root, SURFACE_FULL_GRID)

    assert "recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0" not in missing
    assert "recoverable_v5_t128__train10240__full90__leaf128__fno__d0" not in missing
    assert (
        "recoverable_v5_t128__train10240__full90_leaf_full100_internal_count100__leaf128__tree_neural__d0"
        not in missing
    )
    assert "recoverable_v5_t128__train10240__full60__leaf128__tree_neural__d0" in missing


def test_leaf_mass_fill_overrides_include_new_leaf128_markers() -> None:
    missing = {
        "recoverable_v5_t128__train10240__r60_leaf_mass_eq_40p0__leaf128__tree_neural__d0",
        "recoverable_v5_t128__train10240__r40_leaf_mass_eq_60p0__leaf128__tree_neural__d0",
        "recoverable_v5_t128__train10240__r30_leaf_mass_eq_70p0__leaf128__tree_neural__d0",
    }

    package_names, overrides = _package_leaf_overrides_for_missing_tasks(
        missing,
        surface=SURFACE_FULL_GRID,
    )

    assert package_names == [
        _leaf_mass_package(60),
        _leaf_mass_package(40),
        _leaf_mass_package(30),
    ]
    assert overrides[_leaf_mass_package(60)] == [128]
    assert overrides[_leaf_mass_package(40)] == [128]
    assert overrides[_leaf_mass_package(30)] == [128]


def test_repair_surface_targets_only_recoverable_leaf128_countonly_rows() -> None:
    required = _required_task_names_for_surface(SURFACE_REPAIR_LEAF128_COUNTONLY)

    assert len(required) == 9
    assert "recoverable_v5_t128__train10240__r90_leaf_mass_eq_10p0__leaf128__tree_neural__d0" in required
    assert "recoverable_v5_t128__train10240__r30_leaf_mass_eq_70p0__leaf128__tree_neural__d0" in required
    assert "r12_p079__train10240__r90_leaf_mass_eq_10p0__leaf128__tree_neural__d0" not in required
    assert "recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0" not in required


def test_allocation_surface_required_rows_cover_internal_policies_only_where_distinct() -> None:
    required = _required_task_names_for_surface(SURFACE_ALLOCATION_POLICY_GRID)

    assert (
        "recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0"
        in required
    )
    assert (
        "recoverable_v5_t128__train10240__r90_leaf_mass_eq_10p0__leaf128__tree_neural__d0"
        in required
    )
    assert (
        "recoverable_v5_t128__train10240__r90_depth_equal_mass_eq_10p0__leaf032__tree_neural__d0"
        in required
    )
    assert (
        "recoverable_v5_t128__train10240__r100_node_mass_eq_10p0__leaf016__tree_neural__d0"
        in required
    )
    assert (
        "r12_p079__train10240__r0_depth_equal_mass_eq_100p0__leaf008__tree_neural__d0"
        in required
    )
    assert (
        "recoverable_v5_t128__train10240__r90_depth_equal_mass_eq_10p0__leaf064__tree_neural__d0"
        not in required
    )
    assert (
        "recoverable_v5_t128__train10240__r100_node_mass_eq_10p0__leaf128__tree_neural__d0"
        not in required
    )


def test_allocation_surface_missing_rows_preserve_existing_root_and_leaf_rows(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "sticky_allocation_root"
    _materialize_visible_job(output_root, "combined_scheduler_run")
    _write_raw_summary(
        output_root,
        job_key="combined_scheduler_run",
        task_name="recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0",
        package_name="full90",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.45,
    )
    _write_raw_summary(
        output_root,
        job_key="combined_scheduler_run",
        task_name="recoverable_v5_t128__train10240__r90_leaf_mass_eq_10p0__leaf128__tree_neural__d0",
        package_name="r90_leaf_mass_eq_10p0",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.40,
    )

    missing = _missing_task_names_for_surface(output_root, SURFACE_ALLOCATION_POLICY_GRID)

    assert "recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0" not in missing
    assert (
        "recoverable_v5_t128__train10240__r90_leaf_mass_eq_10p0__leaf128__tree_neural__d0"
        not in missing
    )
    assert (
        "recoverable_v5_t128__train10240__r90_depth_equal_mass_eq_10p0__leaf032__tree_neural__d0"
        in missing
    )
    assert (
        "recoverable_v5_t128__train10240__r100_node_mass_eq_10p0__leaf032__tree_neural__d0"
        in missing
    )


def test_allocation_surface_leaf_overrides_match_policy_distinctions() -> None:
    missing = {
        "recoverable_v5_t128__train10240__r90_depth_equal_mass_eq_10p0__leaf032__tree_neural__d0",
        "recoverable_v5_t128__train10240__r90_depth_equal_mass_eq_10p0__leaf016__tree_neural__d0",
        "recoverable_v5_t128__train10240__r100_node_mass_eq_10p0__leaf032__tree_neural__d0",
        "recoverable_v5_t128__train10240__r100_node_mass_eq_10p0__leaf008__tree_neural__d0",
    }

    package_names, overrides = _package_leaf_overrides_for_missing_tasks(
        missing,
        surface=SURFACE_ALLOCATION_POLICY_GRID,
    )

    assert package_names == [
        _depth_equal_mass_package(90),
        _balanced_node_mass_package(90),
    ]
    assert overrides[_depth_equal_mass_package(90)] == [32, 16]
    assert overrides[_balanced_node_mass_package(90)] == [32, 8]


def test_repair_surface_pipeline_config_filters_to_recoverable_scope(tmp_path: Path) -> None:
    config_text = _build_pipeline_config(
        package_names=["r90_leaf_mass_eq_10p0"],
        package_leaf_token_overrides={"r90_leaf_mass_eq_10p0": [128]},
        surface=SURFACE_REPAIR_LEAF128_COUNTONLY,
        output_root=tmp_path / "repair_root",
    )

    assert 'supervision_recovery_scope_keys = "recoverable_v5_t128"' in config_text
    assert "supervision_recovery_leaf_token_ladder = [128]" in config_text


def test_allocation_surface_pipeline_config_uses_full_geometry_ladder(tmp_path: Path) -> None:
    config_text = _build_pipeline_config(
        package_names=[
            _depth_equal_mass_package(90),
            _balanced_node_mass_package(90),
        ],
        package_leaf_token_overrides={
            _depth_equal_mass_package(90): [32, 16, 8],
            _balanced_node_mass_package(90): [32, 16, 8],
        },
        surface=SURFACE_ALLOCATION_POLICY_GRID,
        output_root=tmp_path / "allocation_root",
    )

    assert (
        'supervision_recovery_scope_keys = "recoverable_v5_t128 r12_p079"'
        in config_text
    )
    assert "supervision_recovery_leaf_token_ladder = [128, 64, 32, 16, 8]" in config_text
    assert "_stage1_artifacts/combined_scheduler_allocation_policy_grid" in config_text


def test_leaf_mass_eq_one_leaf_packages_are_not_treated_as_root_only() -> None:
    assert _one_leaf_package_has_local_supervision(
        SUPERVISION_RECOVERY_PACKAGE_SPECS["r90_leaf_mass_eq_10p0"]
    )
    assert not _one_leaf_package_has_local_supervision(
        SUPERVISION_RECOVERY_PACKAGE_SPECS["full90"]
    )


def test_root40_richer_duplicate_local_alias_resolves() -> None:
    assert (
        resolve_supervision_recovery_package_name(
            "root40_extra_leaffull100_internalcount100",
            valid_names=(
                "full40",
                "full40_leaf_full100_internal_count100",
            ),
        )
        == "full40_leaf_full100_internal_count100"
    )


def test_current_renderer_preserves_landed_decile_rows_from_raw_summaries(tmp_path: Path) -> None:
    output_root = tmp_path / "sticky_simple_root"
    _materialize_visible_job(output_root, "combined_scheduler_run")
    _write_raw_summary(
        output_root,
        job_key="combined_scheduler_run",
        task_name="recoverable_v5_t128__train10240__full90__leaf064__tree_neural__d0",
        package_name="full90",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=64,
        root_mae=0.5,
    )
    _write_raw_summary(
        output_root,
        job_key="oneleaf_root_budget_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full90__leaf128__tree_neural__d0",
        package_name="full90",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.4,
    )
    _write_raw_summary(
        output_root,
        job_key="oneleaf_root_budget_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full80__leaf128__tree_neural__d0",
        package_name="full80",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.45,
    )
    _write_raw_summary(
        output_root,
        job_key="oneleaf_root_budget_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full70__leaf128__fno__d0",
        package_name="full70",
        scope_key="recoverable_v5_t128",
        baseline_family="fno",
        leaf_tokens=128,
        root_mae=0.6,
    )
    _write_raw_summary(
        output_root,
        job_key="oneleaf_local_law_fixed10240_simple",
        task_name="recoverable_v5_t128__train10240__full90_leaf_full100_internal_count100__leaf128__tree_neural__d0",
        package_name="full90_leaf_full100_internal_count100",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.35,
    )

    merged = _build_current_supervision_recovery_summary(output_root)
    recovery = merged["supervision_recovery"]
    scope_rows = recovery["scopes"]["recoverable_v5_t128"]["rows_by_train_docs"]
    rows = scope_rows["10240"]["rows"]
    packages = {row["package_name"] for row in rows}

    assert "full90" in packages
    assert "full80" in packages
    assert "full70" in packages
    assert "full90_leaf_full100_internal_count100" in packages

    coverage = _build_coverage_summary(
        merged,
        train_doc_count=10240,
        root_shares=[100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    )
    recoverable_coverage = coverage["scopes"]["recoverable_v5_t128"]["root_shares"]
    assert recoverable_coverage["90"]["root_only_tree_leaf_tokens"] == [128, 64]
    assert recoverable_coverage["90"]["duplicate_local_leaf_tokens"] == [128]
    assert recoverable_coverage["80"]["root_only_tree_leaf_tokens"] == [128]
    assert recoverable_coverage["70"]["root_only_fno_leaf128_present"] is True


def test_current_renderer_prefers_overlay_root_for_duplicate_task_names(tmp_path: Path) -> None:
    base_root = tmp_path / "base_root"
    repair_root = tmp_path / "repair_root"
    _materialize_visible_job(base_root, "combined_scheduler_run")
    _materialize_visible_job(repair_root, "combined_scheduler_repair_leaf128_countonly")
    task_name = "recoverable_v5_t128__train10240__r90_leaf_mass_eq_10p0__leaf128__tree_neural__d0"
    _write_raw_summary(
        base_root,
        job_key="combined_scheduler_run",
        task_name=task_name,
        package_name="r90_leaf_mass_eq_10p0",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=1.79,
    )
    _write_raw_summary(
        repair_root,
        job_key="combined_scheduler_repair_leaf128_countonly",
        task_name=task_name,
        package_name="r90_leaf_mass_eq_10p0",
        scope_key="recoverable_v5_t128",
        baseline_family="tree_neural",
        leaf_tokens=128,
        root_mae=0.42,
    )

    merged = _build_current_supervision_recovery_summary(
        base_root,
        overlay_output_roots=[repair_root],
    )
    rows = merged["supervision_recovery"]["scopes"]["recoverable_v5_t128"]["rows_by_train_docs"]["10240"]["rows"]
    matching = [
        row for row in rows
        if row["package_name"] == "r90_leaf_mass_eq_10p0"
        and row["baseline_family"] == "tree_neural"
        and int(row["fixed_leaf_tokens"]) == 128
    ]

    assert len(matching) == 1
    assert matching[0]["test_root_mae_mean"] == 0.42
    assert "repair_root" in str(matching[0]["source_summary_json"])
