from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.launch_markov_v3_initial_grid import (
    build_launch_plan,
    launch_plan_jobs,
    main,
    materialize_job_config_payload,
    selected_job_specs,
)


def test_initial_grid_plan_contains_expected_curated_jobs() -> None:
    plan = build_launch_plan(
        group_names=["initial_grid"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_initial_grid_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert "v3_main_grid" in keys
    assert "superset_gamma_t128" in keys
    assert "mass_matched_gamma_t128" in keys
    assert "preset_ablation_canary" in keys
    assert "preset_ablation_full_laws" in keys
    assert "multileaf_root_only" in keys
    assert "multileaf_full_laws" in keys
    assert "mass_matched_full_coverage" not in keys
    axis_by_key = {str(item["key"]): item for item in list(plan["axis_coverage"])}
    assert axis_by_key["superset_semantics"]["status"] == "ready"
    assert axis_by_key["mass_matched_semantics"]["status"] == "ready"
    assert axis_by_key["geometry_endpoints"]["status"] == "ready"
    assert axis_by_key["one_leaf_protocol"]["status"] == "ready"
    assert axis_by_key["mass_full_coverage"]["status"] == "available_optional"


def test_initial_grid_is_check_basics_plus_after_basics() -> None:
    basics = build_launch_plan(
        group_names=["check_basics"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_check_basics_merge_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    after = build_launch_plan(
        group_names=["after_basics"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_after_basics_merge_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    initial = build_launch_plan(
        group_names=["initial_grid"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_initial_grid_merge_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    expected_keys = [str(job["key"]) for job in list(basics["jobs"])] + [
        str(job["key"]) for job in list(after["jobs"])
    ]
    actual_keys = [str(job["key"]) for job in list(initial["jobs"])]
    assert actual_keys == expected_keys


def test_after_basics_is_scientific_plus_protocol_followups() -> None:
    scientific = build_launch_plan(
        group_names=["scientific_followups"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_scientific_followups_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    protocol = build_launch_plan(
        group_names=["protocol_followups"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_protocol_followups_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    after = build_launch_plan(
        group_names=["after_basics"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_after_basics_composition_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    expected_keys = [str(job["key"]) for job in list(scientific["jobs"])] + [
        str(job["key"]) for job in list(protocol["jobs"])
    ]
    actual_keys = [str(job["key"]) for job in list(after["jobs"])]
    assert actual_keys == expected_keys


def test_check_basics_group_targets_bringup_surfaces() -> None:
    plan = build_launch_plan(
        group_names=["check_basics"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_check_basics_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert keys == [
        "superset_gamma_t128",
        "mass_matched_gamma_t128",
        "preset_ablation_canary",
        "preset_ablation_full_laws",
    ]
    axis_by_key = {str(item["key"]): item for item in list(plan["axis_coverage"])}
    assert axis_by_key["superset_semantics"]["status"] == "ready"
    assert axis_by_key["mass_matched_semantics"]["status"] == "ready"
    assert axis_by_key["official_fno_base"]["status"] == "ready"
    assert axis_by_key["one_leaf_canary_vs_standard"]["status"] == "ready"
    assert axis_by_key["gamma_sweep"]["status"] == "ready"
    assert axis_by_key["geometry_endpoints"]["status"] == "ready"
    assert axis_by_key["one_leaf_protocol"]["status"] == "partial"


def test_small_train_local_law_group_targets_valid_law_surfaces() -> None:
    plan = build_launch_plan(
        group_names=["small_train_local_law"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_small_train_local_law_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert keys == [
        "preset_ablation_canary",
        "one_leaf_duplicate_local_full_laws",
        "small_train_multileaf_root_only",
        "small_train_multileaf_full_laws",
        "small_train_r100_superset_local10",
    ]
    axis_by_key = {str(item["key"]): item for item in list(plan["axis_coverage"])}
    assert axis_by_key["official_fno_base"]["status"] == "ready"
    assert axis_by_key["one_leaf_duplicate_local_no_harm"]["status"] == "ready"
    assert axis_by_key["local_law_validity"]["status"] == "ready"
    assert axis_by_key["geometry_endpoints"]["status"] == "ready"


def test_local_law_quickcheck_group_targets_two_leaf_bringup() -> None:
    plan = build_launch_plan(
        group_names=["local_law_quickcheck"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_local_law_quickcheck_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert keys == [
        "preset_ablation_canary",
        "one_leaf_duplicate_local_full_laws",
        "quick_two_leaf_root_only",
        "quick_two_leaf_full100_local_full_laws",
        "quick_two_leaf_r100_superset_local10",
    ]
    axis_by_key = {str(item["key"]): item for item in list(plan["axis_coverage"])}
    assert axis_by_key["official_fno_base"]["status"] == "ready"
    assert axis_by_key["one_leaf_duplicate_local_no_harm"]["status"] == "ready"
    assert axis_by_key["local_law_validity"]["status"] == "ready"


def test_redistribution_groups_target_exact_root_node_split_surfaces() -> None:
    quick_plan = build_launch_plan(
        group_names=["redistribution_quickcheck"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_redistribution_quickcheck_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    quick_keys = [str(job["key"]) for job in list(quick_plan["jobs"])]
    assert quick_keys == ["redistribution_quickcheck"]
    quick_payload = materialize_job_config_payload(selected_job_specs(
        group_names=["redistribution_quickcheck"],
        explicit_job_keys=[],
    )[0])
    tradeoff = dict(quick_payload["tradeoff_pipeline"])
    assert tradeoff["supervision_recovery_packages"] == ["redistribution_r100_coarse"]
    assert tradeoff["supervision_recovery_leaf_token_ladder"] == [64]

    small_plan = build_launch_plan(
        group_names=["redistribution_small_train"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_redistribution_small_train_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    small_keys = [str(job["key"]) for job in list(small_plan["jobs"])]
    assert small_keys == ["redistribution_small_train"]
    axis_by_key = {str(item["key"]): item for item in list(small_plan["axis_coverage"])}
    assert axis_by_key["root_node_redistribution"]["status"] == "selected_optional"
    assert axis_by_key["official_fno_base"]["status"] == "ready"
    small_payload = materialize_job_config_payload(selected_job_specs(
        group_names=["redistribution_small_train"],
        explicit_job_keys=[],
    )[0])
    tradeoff = dict(small_payload["tradeoff_pipeline"])
    assert tradeoff["supervision_recovery_packages"] == ["redistribution_r100"]
    assert tradeoff["supervision_recovery_leaf_token_ladder"] == [64, 32, 16, 8]
    assert tradeoff["supervision_recovery_train_docs"] == [1024, 4096]
    assert tradeoff["supervision_recovery_depth_discount_gammas"] == [1.0]


def test_depth_redistribution_groups_target_root_and_depth_aware_surfaces() -> None:
    root_plan = build_launch_plan(
        group_names=["depth_redistribution_root_ladder"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_root_ladder_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(root_plan["jobs"])] == [
        "root_budget_ladder_small_train"
    ]
    root_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_root_ladder"],
            explicit_job_keys=[],
        )[0]
    )
    root_tradeoff = dict(root_payload["tradeoff_pipeline"])
    assert root_tradeoff["supervision_recovery_packages"] == ["root_ladder_deciles"]
    assert root_tradeoff["supervision_recovery_leaf_token_ladder"] == [128, 64, 32, 16, 8]
    assert root_tradeoff["supervision_recovery_recoverable_benchmark"] == "recoverable_v5_t128"
    assert root_tradeoff["supervision_recovery_structural_grid"] == "structural_core_v2_t128"
    assert root_tradeoff["supervision_recovery_structural_cell"] == "r12_p079"

    leaf_plan = build_launch_plan(
        group_names=["depth_redistribution_leaf_only"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_leaf_only_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(leaf_plan["jobs"])] == [
        "mass_preserving_leaf_only_small_train"
    ]
    leaf_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_leaf_only"],
            explicit_job_keys=[],
        )[0]
    )
    leaf_tradeoff = dict(leaf_payload["tradeoff_pipeline"])
    assert leaf_tradeoff["supervision_recovery_packages"] == [
        "mass_preserving_leaf_only_deciles"
    ]
    assert leaf_tradeoff["supervision_recovery_leaf_token_ladder"] == [64, 32, 16, 8]

    equal_plan = build_launch_plan(
        group_names=["depth_redistribution_levels_equal"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_equal_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(equal_plan["jobs"])] == [
        "mass_preserving_depth_equal_small_train"
    ]
    equal_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_levels_equal"],
            explicit_job_keys=[],
        )[0]
    )
    equal_tradeoff = dict(equal_payload["tradeoff_pipeline"])
    assert equal_tradeoff["supervision_recovery_packages"] == [
        "mass_preserving_levels_equal_deciles"
    ]
    assert equal_tradeoff["supervision_recovery_leaf_token_ladder"] == [32, 16, 8]

    full_plan = build_launch_plan(
        group_names=["depth_redistribution"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_full_grid_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    axis_by_key = {str(item["key"]): item for item in list(full_plan["axis_coverage"])}
    assert axis_by_key["root_budget_ladder"]["status"] == "selected_optional"
    assert axis_by_key["mass_preserving_leaf_only"]["status"] == "selected_optional"
    assert axis_by_key["mass_preserving_depth_equal"]["status"] == "selected_optional"


def test_depth_redistribution_large_train_groups_target_10240_followup() -> None:
    stable_plan = build_launch_plan(
        group_names=["depth_redistribution_large_train_stable"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_large_train_stable_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(stable_plan["jobs"])] == [
        "root_budget_ladder_large_train",
        "mass_preserving_leaf_only_large_train",
    ]
    stable_axis_by_key = {
        str(item["key"]): item for item in list(stable_plan["axis_coverage"])
    }
    assert stable_axis_by_key["root_budget_ladder"]["status"] == "selected_optional"
    assert stable_axis_by_key["mass_preserving_leaf_only"]["status"] == "selected_optional"
    assert (
        stable_axis_by_key["mass_preserving_depth_equal"]["status"]
        == "available_optional"
    )
    assert stable_axis_by_key["geometry_endpoints"]["status"] == "ready"
    assert stable_axis_by_key["official_fno_base"]["status"] == "ready"

    large_plan = build_launch_plan(
        group_names=["depth_redistribution_large_train"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_large_train_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(large_plan["jobs"])] == [
        "root_budget_ladder_large_train",
        "mass_preserving_leaf_only_large_train",
        "mass_preserving_depth_equal_large_train",
    ]
    large_axis_by_key = {
        str(item["key"]): item for item in list(large_plan["axis_coverage"])
    }
    assert large_axis_by_key["root_budget_ladder"]["status"] == "selected_optional"
    assert (
        large_axis_by_key["mass_preserving_leaf_only"]["status"]
        == "selected_optional"
    )
    assert (
        large_axis_by_key["mass_preserving_depth_equal"]["status"]
        == "selected_optional"
    )

    root_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_large_root_ladder"],
            explicit_job_keys=[],
        )[0]
    )
    root_tradeoff = dict(root_payload["tradeoff_pipeline"])
    assert root_tradeoff["supervision_recovery_train_docs"] == [10240]
    assert root_tradeoff["supervision_recovery_leaf_token_ladder"] == [
        128,
        64,
        32,
        16,
        8,
    ]

    leaf_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_large_leaf_only"],
            explicit_job_keys=[],
        )[0]
    )
    leaf_tradeoff = dict(leaf_payload["tradeoff_pipeline"])
    assert leaf_tradeoff["supervision_recovery_train_docs"] == [10240]
    assert leaf_tradeoff["supervision_recovery_leaf_token_ladder"] == [64, 32, 16, 8]

    equal_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_large_levels_equal"],
            explicit_job_keys=[],
        )[0]
    )
    equal_tradeoff = dict(equal_payload["tradeoff_pipeline"])
    assert equal_tradeoff["supervision_recovery_train_docs"] == [10240]
    assert equal_tradeoff["supervision_recovery_leaf_token_ladder"] == [32, 16, 8]


def test_depth_redistribution_large_train_tuning_targets_regression_cells() -> None:
    tuning_plan = build_launch_plan(
        group_names=["depth_redistribution_large_train_tuning"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_depth_large_train_tuning_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(tuning_plan["jobs"])] == [
        "root_budget_ladder_large_train_longschedule",
        "mass_preserving_leaf_only_large_train_longschedule",
    ]

    root_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["depth_redistribution_large_train_tuning"],
            explicit_job_keys=[],
        )[0]
    )
    root_tradeoff = dict(root_payload["tradeoff_pipeline"])
    assert root_tradeoff["supervision_recovery_packages"] == [
        "full10",
        "full20",
        "full30",
        "full40",
        "full50",
        "full70",
    ]
    assert root_tradeoff["supervision_epochs"] == 52
    assert root_tradeoff["tree_stage1_epochs"] == 12
    assert root_tradeoff["tree_stage2_epochs"] == 40

    leaf_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["mass_preserving_leaf_only_large_train_longschedule"],
        )[0]
    )
    leaf_tradeoff = dict(leaf_payload["tradeoff_pipeline"])
    assert leaf_tradeoff["supervision_recovery_packages"] == [
        "r0_leaf_mass_eq_100p0",
        "r10_leaf_mass_eq_90p0",
        "r30_leaf_mass_eq_70p0",
        "r70_leaf_mass_eq_30p0",
        "r80_leaf_mass_eq_20p0",
        "r90_leaf_mass_eq_10p0",
    ]
    assert leaf_tradeoff["supervision_recovery_leaf_token_ladder"] == [64, 32, 16, 8]
    assert leaf_tradeoff["supervision_epochs"] == 52


def test_publication_fullval_group_includes_clean_local_law_reruns() -> None:
    plan = build_launch_plan(
        group_names=["publication_fullval"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_publication_fullval_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert keys == [
        "oneleaf_root_budget_publication_fullval",
        "root_budget_publication_multileaf_fullval",
        "leaf_only_publication_focus_fullval",
        "depth_equal_publication_focus_fullval",
        "local_law_publication_fullval",
        "r100_superset_local10_publication_fullval",
    ]

    local_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=["publication_local_law"],
            explicit_job_keys=[],
        )[0]
    )
    local_tradeoff = dict(local_payload["tradeoff_pipeline"])
    assert local_tradeoff["supervision_recovery_packages"] == [
        "full100",
        "root100_extra_leaffull100_internalcount100",
    ]
    assert local_tradeoff["supervision_recovery_recoverable_benchmark"] == "recoverable_v5_t128"
    assert local_tradeoff["supervision_recovery_structural_grid"] == "structural_core_v2_t128"
    assert local_tradeoff["supervision_recovery_structural_cell"] == "r12_p079"
    assert local_tradeoff["supervision_recovery_train_docs"] == [4096, 10240]
    assert local_tradeoff["supervision_recovery_leaf_token_ladder"] == [64, 32, 16, 8]
    assert local_tradeoff["tree_stage1_screen_doc_limit"] == 0
    assert local_tradeoff["tree_stage1_final_exact_doc_limit"] == 0
    assert local_tradeoff["exact_metric_final_doc_limit"] == 0

    superset_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["r100_superset_local10_publication_fullval"],
        )[0]
    )
    superset_tradeoff = dict(superset_payload["tradeoff_pipeline"])
    assert superset_tradeoff["supervision_recovery_packages"] == [
        "root100",
        "root100_extra_local10",
    ]
    assert superset_tradeoff["supervision_recovery_recoverable_benchmark"] == "recoverable_v5_t128"
    assert superset_tradeoff["supervision_recovery_structural_grid"] == "structural_core_v2_t128"
    assert superset_tradeoff["supervision_recovery_structural_cell"] == "r12_p079"
    assert superset_tradeoff["supervision_recovery_train_docs"] == [4096, 10240]
    assert superset_tradeoff["supervision_recovery_leaf_token_ladder"] == [64, 32, 16, 8]


def test_overnight_xlarge_group_targets_20480_extensions() -> None:
    plan = build_launch_plan(
        group_names=["overnight_xlarge"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_overnight_xlarge_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert keys == [
        "root_budget_ladder_xlarge_train",
        "mass_preserving_leaf_only_xlarge_train",
        "mass_preserving_depth_equal_xlarge_train",
        "oneleaf_root_budget_publication_xlarge",
        "root_budget_publication_multileaf_xlarge",
        "leaf_only_publication_focus_xlarge",
        "depth_equal_publication_focus_xlarge",
        "local_law_publication_xlarge",
        "r100_superset_local10_publication_xlarge",
    ]

    root_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["root_budget_ladder_xlarge_train"],
        )[0]
    )
    root_tradeoff = dict(root_payload["tradeoff_pipeline"])
    assert root_tradeoff["supervision_recovery_train_docs"] == [20480]
    assert root_tradeoff["supervision_recovery_seeds"] == [0]
    assert root_tradeoff["supervision_recovery_recoverable_benchmark"] == "recoverable_v5_t128"
    assert root_tradeoff["supervision_recovery_structural_grid"] == "structural_core_v2_t128"

    oneleaf_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["oneleaf_root_budget_publication_xlarge"],
        )[0]
    )
    oneleaf_tradeoff = dict(oneleaf_payload["tradeoff_pipeline"])
    assert oneleaf_tradeoff["supervision_recovery_train_docs"] == [20480]
    assert oneleaf_tradeoff["supervision_recovery_seeds"] == [0]
    assert oneleaf_tradeoff["supervision_recovery_leaf_token_ladder"] == [128]
    assert oneleaf_tradeoff["tree_stage1_screen_doc_limit"] == 0


def test_publication_plot_fillers_target_missing_leaf128_columns() -> None:
    plan = build_launch_plan(
        group_names=["publication_plot_fillers"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_publication_plot_fillers_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(plan["jobs"])] == [
        "oneleaf_root_budget_longschedule_fill_fullval",
        "oneleaf_local_law_root_sweep_fullval",
        "oneleaf_root_budget_longschedule_fill_xlarge",
        "oneleaf_local_law_root_sweep_xlarge",
    ]

    longschedule_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["oneleaf_root_budget_longschedule_fill_fullval"],
        )[0]
    )
    longschedule_tradeoff = dict(longschedule_payload["tradeoff_pipeline"])
    assert longschedule_tradeoff["supervision_recovery_packages"] == [
        "root100",
        "root90",
        "root80",
        "root70",
        "root50",
        "root20",
        "root10",
    ]
    assert longschedule_tradeoff["supervision_recovery_train_docs"] == [10240]
    assert longschedule_tradeoff["supervision_recovery_leaf_token_ladder"] == [128]
    assert longschedule_tradeoff["tree_training_schedule"] == "two_stage"
    assert longschedule_tradeoff["tree_stage1_epochs"] == 12
    assert longschedule_tradeoff["tree_stage2_epochs"] == 40

    local_law_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["oneleaf_local_law_root_sweep_fullval"],
        )[0]
    )
    local_law_tradeoff = dict(local_law_payload["tradeoff_pipeline"])
    assert local_law_tradeoff["supervision_recovery_packages"] == [
        "root100_extra_leaffull100_internalcount100",
        "root90_extra_leaffull100_internalcount100",
        "root80_extra_leaffull100_internalcount100",
        "root70_extra_leaffull100_internalcount100",
        "root50_extra_leaffull100_internalcount100",
        "root20_extra_leaffull100_internalcount100",
        "root10_extra_leaffull100_internalcount100",
    ]
    assert local_law_tradeoff["supervision_recovery_train_docs"] == [10240]
    assert local_law_tradeoff["supervision_recovery_leaf_token_ladder"] == [128]
    assert local_law_tradeoff["tree_reference"]["preset"] == "full_laws"
    assert local_law_tradeoff["structural_tree_reference"]["preset"] == "full_laws"


def test_structural_oneleaf_rescue_group_targets_debug_matrix() -> None:
    plan = build_launch_plan(
        group_names=["structural_oneleaf_rescue"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_structural_oneleaf_rescue_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    assert [str(job["key"]) for job in list(plan["jobs"])] == [
        "structural_oneleaf_matched_root_v2_rescue",
        "structural_oneleaf_matched_root_v3_rescue",
        "structural_oneleaf_recoverable_recipe_v3_rescue",
        "structural_oneleaf_canary_anchor_rescue",
    ]

    v2_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["structural_oneleaf_matched_root_v2_rescue"],
        )[0]
    )
    v2_tradeoff = dict(v2_payload["tradeoff_pipeline"])
    assert v2_tradeoff["supervision_recovery_packages"] == [
        "full90",
        "full80",
        "full50",
        "full10",
    ]
    assert v2_tradeoff["supervision_recovery_train_docs"] == [10240, 20480]
    assert v2_tradeoff["supervision_recovery_leaf_token_ladder"] == [128]
    assert v2_tradeoff["tree_reference"]["preset"] == "root_only_matched_v2"
    assert v2_tradeoff["structural_tree_reference"]["preset"] == "structural_root_only_matched_v2"

    recoverable_recipe_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["structural_oneleaf_recoverable_recipe_v3_rescue"],
        )[0]
    )
    recoverable_recipe_tradeoff = dict(recoverable_recipe_payload["tradeoff_pipeline"])
    assert recoverable_recipe_tradeoff["tree_reference"]["preset"] == "root_only_matched"
    assert recoverable_recipe_tradeoff["structural_tree_reference"]["preset"] == "root_only_matched"

    canary_payload = materialize_job_config_payload(
        selected_job_specs(
            group_names=[],
            explicit_job_keys=["structural_oneleaf_canary_anchor_rescue"],
        )[0]
    )
    canary_tradeoff = dict(canary_payload["tradeoff_pipeline"])
    assert canary_tradeoff["supervision_recovery_packages"] == ["full100"]
    assert canary_tradeoff["tree_reference"]["preset"] == "fno_parity_canary"
    assert canary_tradeoff["structural_tree_reference"]["preset"] == "fno_parity_canary"
    assert canary_tradeoff["one_leaf_tree_reference"]["preset"] == "fno_parity_canary"


def test_after_basics_group_avoids_rerunning_bringup_jobs() -> None:
    plan = build_launch_plan(
        group_names=["after_basics"],
        explicit_job_keys=[],
        output_root_base=Path("/tmp/markov_v3_after_basics_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    keys = [str(job["key"]) for job in list(plan["jobs"])]
    assert keys == [
        "v3_main_grid",
        "full100_leaf_ladder_standard",
        "full100_leaf_ladder_half_c1",
        "superset_leaf32_c1half",
        "superset_leaf32_leafratehalf",
        "preset_ablation_mse_only",
        "preset_ablation_two_stage_no_laws",
        "multileaf_root_only",
        "multileaf_full_laws",
    ]
    assert "superset_gamma_t128" not in keys
    assert "mass_matched_gamma_t128" not in keys
    assert "preset_ablation_canary" not in keys
    assert "preset_ablation_full_laws" not in keys


def test_main_defaults_to_check_basics(capsys: object) -> None:
    exit_code = main(["--output-root-base", "/tmp/markov_v3_default_group_test"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Groups: check_basics" in captured.out


def test_build_launch_plan_reports_existing_completed_job(tmp_path: Path) -> None:
    base_root = tmp_path / "markov_restartable_base"
    job_root = base_root / "_launchers" / "preset_ablation_canary"
    output_root = base_root / "preset_ablation_canary"
    job_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scheduler_status.json").write_text(
        json.dumps({"state": "completed"}),
        encoding="utf-8",
    )
    (job_root / "manifest.json").write_text(
        json.dumps(
            {
                "name": "markov_v3_initial_grid__preset_ablation_canary",
                "job_root": str(job_root),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-10T00:00:00+00:00",
                "log_path": str(job_root / "job.log"),
                "command": [
                    sys.executable,
                    "scripts/run_markov_optimization_tradeoff_pipeline.py",
                    "--output-root",
                    str(output_root),
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_launch_plan(
        group_names=[],
        explicit_job_keys=["preset_ablation_canary"],
        output_root_base=base_root,
        python_bin=Path(sys.executable),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
        inspect_existing_launchers=True,
    )
    job = list(plan["jobs"])[0]
    existing_state = dict(job["existing_state"])
    assert existing_state["state"] == "completed"
    assert existing_state["scheduler_state"] == "completed"


def test_long_job_command_uses_boolean_optional_flag_form() -> None:
    plan = build_launch_plan(
        group_names=[],
        explicit_job_keys=["preset_ablation_canary"],
        output_root_base=Path("/tmp/markov_v3_flag_form_test"),
        python_bin=Path("/tmp/fake_python"),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    launcher_command = list(plan["jobs"])[0]["launcher_command"]
    assert "--no-replace-existing" in launcher_command
    assert "--replace-existing" not in launcher_command
    assert "false" not in launcher_command


def test_launch_plan_jobs_skips_completed_and_continues_on_failure(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    plan = build_launch_plan(
        group_names=[],
        explicit_job_keys=["preset_ablation_canary", "preset_ablation_full_laws"],
        output_root_base=tmp_path / "batch_base",
        python_bin=Path(sys.executable),
        launch_backend="auto",
        replace_existing=False,
        env_assignments=[],
    )
    jobs = list(plan["jobs"])
    jobs[0]["existing_state"] = {"state": "completed"}
    jobs[1]["existing_state"] = {"state": "not_launched"}
    plan["jobs"] = jobs

    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: str,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check, cwd
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="synthetic launch failure",
        )

    monkeypatch.setattr("scripts.launch_markov_v3_initial_grid.subprocess.run", _fake_run)
    summary = launch_plan_jobs(
        plan,
        python_bin=Path(sys.executable),
        skip_running=True,
        skip_completed=True,
        fail_fast=False,
    )

    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["launched_count"] == 0
    assert summary["skipped_jobs"][0]["reason"] == "completed"
    assert summary["failed_jobs"][0]["key"] == "preset_ablation_full_laws"
    assert len(calls) == 1


def test_template_jobs_inject_expected_presets() -> None:
    ablation_spec = next(
        spec for spec in selected_job_specs(group_names=[], explicit_job_keys=["preset_ablation_canary"])
    )
    ablation_payload = materialize_job_config_payload(ablation_spec)
    assert (
        ablation_payload["tradeoff_pipeline"]["tree_reference"]["preset"]
        == "fno_parity_canary"
    )
    assert (
        ablation_payload["tradeoff_pipeline"]["structural_tree_reference"]["preset"]
        == "fno_parity_canary"
    )
    assert (
        ablation_payload["tradeoff_pipeline"]["supervision_recovery_recoverable_benchmark"]
        == "recoverable_v5_t128"
    )
    assert (
        ablation_payload["tradeoff_pipeline"]["supervision_recovery_structural_grid"]
        == "structural_core_v2_t128"
    )

    multileaf_spec = next(
        spec for spec in selected_job_specs(group_names=[], explicit_job_keys=["multileaf_root_only"])
    )
    multileaf_payload = materialize_job_config_payload(multileaf_spec)
    assert (
        multileaf_payload["tradeoff_pipeline"]["tree_reference"]["preset"]
        == "multileaf_root_only"
    )
    assert (
        multileaf_payload["tradeoff_pipeline"]["structural_tree_reference"]["preset"]
        == "multileaf_root_only"
    )

    one_leaf_duplicate_spec = next(
        spec
        for spec in selected_job_specs(
            group_names=[], explicit_job_keys=["one_leaf_duplicate_local_full_laws"]
        )
    )
    one_leaf_duplicate_payload = materialize_job_config_payload(
        one_leaf_duplicate_spec
    )
    assert (
        one_leaf_duplicate_payload["tradeoff_pipeline"]["tree_reference"]["preset"]
        == "full_laws"
    )
    assert (
        one_leaf_duplicate_payload["tradeoff_pipeline"][
            "supervision_recovery_packages"
        ]
        == ["root100_extra_leaffull100_internalcount100"]
    )
    assert (
        one_leaf_duplicate_payload["tradeoff_pipeline"][
            "supervision_recovery_train_docs"
        ]
        == [1024, 4096]
    )

    small_multileaf_spec = next(
        spec
        for spec in selected_job_specs(
            group_names=[], explicit_job_keys=["small_train_multileaf_full_laws"]
        )
    )
    small_multileaf_payload = materialize_job_config_payload(small_multileaf_spec)
    assert (
        small_multileaf_payload["tradeoff_pipeline"][
            "supervision_recovery_train_docs"
        ]
        == [1024, 4096]
    )
    assert small_multileaf_payload["tradeoff_pipeline"][
        "supervision_recovery_leaf_token_ladder"
    ] == [64, 32, 16, 8]
    assert (
        small_multileaf_payload["tradeoff_pipeline"][
            "supervision_recovery_packages"
        ]
        == ["root100_extra_leaffull100_internalcount100"]
    )

    quick_two_leaf_payload = materialize_job_config_payload(
        next(
            spec
            for spec in selected_job_specs(
                group_names=[],
                explicit_job_keys=["quick_two_leaf_full100_local_full_laws"],
            )
        )
    )
    assert quick_two_leaf_payload["tradeoff_pipeline"][
        "supervision_recovery_leaf_token_ladder"
    ] == [64]
    assert (
        quick_two_leaf_payload["tradeoff_pipeline"][
            "supervision_recovery_packages"
        ]
        == ["root100_extra_leaffull100_internalcount100"]
    )

    quick_r100_payload = materialize_job_config_payload(
        next(
            spec
            for spec in selected_job_specs(
                group_names=[],
                explicit_job_keys=["quick_two_leaf_r100_superset_local10"],
            )
        )
    )
    assert quick_r100_payload["tradeoff_pipeline"][
        "supervision_recovery_leaf_token_ladder"
    ] == [64]
    assert (
        quick_r100_payload["tradeoff_pipeline"][
            "supervision_recovery_packages"
        ]
        == ["root100", "root100_extra_local10"]
    )


def test_core_grid_jobs_materialize_endpoint_coverage_and_one_leaf_parity() -> None:
    v3_spec = next(
        spec for spec in selected_job_specs(group_names=[], explicit_job_keys=["v3_main_grid"])
    )
    v3_payload = materialize_job_config_payload(v3_spec)
    assert v3_payload["tradeoff_pipeline"]["supervision_recovery_leaf_token_ladder"] == [
        64,
        32,
        16,
        8,
    ]
    assert "one_leaf_tree_reference" not in v3_payload["tradeoff_pipeline"]

    superset_gamma_spec = next(
        spec
        for spec in selected_job_specs(
            group_names=[], explicit_job_keys=["superset_gamma_t128"]
        )
    )
    superset_gamma_payload = materialize_job_config_payload(superset_gamma_spec)
    assert superset_gamma_payload["tradeoff_pipeline"][
        "supervision_recovery_leaf_token_ladder"
    ] == [64, 32, 16, 8]
    assert "one_leaf_tree_reference" not in superset_gamma_payload["tradeoff_pipeline"]

    full100_ladder_spec = next(
        spec
        for spec in selected_job_specs(
            group_names=[], explicit_job_keys=["full100_leaf_ladder_standard"]
        )
    )
    full100_ladder_payload = materialize_job_config_payload(full100_ladder_spec)
    assert full100_ladder_payload["tradeoff_pipeline"][
        "supervision_recovery_leaf_token_ladder"
    ] == [128, 64, 32, 16, 8]
    assert (
        full100_ladder_payload["tradeoff_pipeline"]["one_leaf_tree_reference"]["preset"]
        == "fno_parity_canary"
    )
