from __future__ import annotations

from scripts import run_tree_neural_fixed_bundle_memory_bisect as mod


def test_case_specs_cover_expected_four_way_bisect() -> None:
    cases = mod._case_specs()
    assert [case.name for case in cases] == [
        "slotwise_control_legacy_exact",
        "shared_feature_adapters_exact_selection_legacy",
        "shared_feature_adapters_cheap_selection_legacy",
        "shared_feature_adapters_cheap_selection_streaming",
    ]


def test_build_case_config_uses_expected_stage1_selection_modes() -> None:
    args = mod._parser().parse_args(
        [
            "run",
            "--benchmark",
            "smoke",
            "--train-docs",
            "8",
            "--no-use-cuda",
        ]
    )
    cases = {case.name: case for case in mod._case_specs()}

    control_cfg = mod._build_case_config(
        args,
        cases["slotwise_control_legacy_exact"],
    )
    cheap_cfg = mod._build_case_config(
        args,
        cases["shared_feature_adapters_cheap_selection_streaming"],
    )

    assert control_cfg.tree_training_schedule == "single_stage"
    assert control_cfg.tree_checkpoint_metric == "val_exact_sketch_direct"
    assert cheap_cfg.tree_training_schedule == "two_stage"
    assert cheap_cfg.tree_stage1_checkpoint_metric == "val_root_mae"
