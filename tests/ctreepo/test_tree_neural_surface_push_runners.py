from __future__ import annotations

from scripts import run_tree_neural_learned_surface_push as surface_push
from scripts import run_tree_neural_slotwise_scaling_push as scaling_push


def test_shared_feature_phase1_configs_include_search_variants_and_controls() -> None:
    args = surface_push._parser().parse_args([])

    configs = surface_push._phase1_configs(args)
    by_label = {config.label: config for _, config in configs}

    assert set(by_label) == {
        "shared_feature_phi128_decodeheavy",
        "shared_feature_phi128_balanced",
        "shared_feature_phi192_decodeheavy",
        "shared_feature_phi192_balanced",
        "shared_feature_adapters_phi128",
        "shared_feature_adapters_phi192",
        "fiber_primary_phi128",
        "fiber_primary_phi192",
        "shared_bottleneck48_control",
        "slotwise_control",
    }
    assert by_label["shared_feature_phi128_decodeheavy"].tree_theorem_surface_mode == "shared_feature"
    assert by_label["shared_feature_phi128_decodeheavy"].tree_theorem_feature_dim == 128
    assert by_label["shared_feature_phi128_decodeheavy"].tree_phi_compose_weight == 0.25
    assert by_label["shared_feature_phi128_decodeheavy"].tree_phi_contrastive_weight == 0.0
    assert by_label["shared_feature_adapters_phi192"].tree_theorem_surface_mode == (
        "shared_feature_adapters"
    )
    assert by_label["shared_feature_adapters_phi192"].tree_theorem_feature_dim == 192
    assert by_label["fiber_primary_phi128"].tree_c2_mode == "fiber"
    assert by_label["fiber_primary_phi128"].tree_phi_contrastive_weight == 2.0
    assert by_label["slotwise_control"].tree_theorem_surface_mode == "slotwise"
    assert by_label["shared_bottleneck48_control"].tree_theorem_surface_mode == (
        "shared_bottleneck"
    )


def test_shared_feature_phase1_promotions_filter_controls_and_apply_gate() -> None:
    runs = [
        {
            "tuning_stage": "phase1",
            "config_label": "slotwise_control",
            "root_direct_count_mae": 0.30,
            "leaf_direct_exact_match": 0.95,
            "merge_direct_exact_match": 0.90,
            "phi_merge_alignment": 0.99,
        },
        {
            "tuning_stage": "phase1",
            "config_label": "shared_feature_phi128_decodeheavy",
            "root_direct_count_mae": 0.52,
            "leaf_direct_exact_match": 0.74,
            "merge_direct_exact_match": 0.61,
            "phi_merge_alignment": 0.94,
        },
        {
            "tuning_stage": "phase1",
            "config_label": "shared_feature_phi192_decodeheavy",
            "root_direct_count_mae": 0.49,
            "leaf_direct_exact_match": 0.79,
            "merge_direct_exact_match": 0.66,
            "phi_merge_alignment": 0.95,
        },
        {
            "tuning_stage": "phase1",
            "config_label": "shared_feature_phi128_balanced",
            "root_direct_count_mae": 0.74,
            "leaf_direct_exact_match": 0.75,
            "merge_direct_exact_match": 0.60,
            "phi_merge_alignment": 0.95,
        },
    ]

    promoted = surface_push._select_phase1_promotions(runs)

    assert promoted == [
        "shared_feature_phi192_decodeheavy",
        "shared_feature_phi128_decodeheavy",
    ]


def test_shared_feature_phase2_winner_requires_slotwise_margin() -> None:
    runs = [
        {
            "tuning_stage": "phase2",
            "config_label": "slotwise_control__internal_full_dense_256",
            "train_doc_count": 256,
            "root_direct_count_mae": 0.44,
            "leaf_direct_exact_match": 0.90,
            "merge_direct_exact_match": 0.80,
            "phi_direct_probe_merge_gap": 0.04,
            "phi_merge_alignment": 0.92,
        },
        {
            "tuning_stage": "phase2",
            "config_label": "shared_feature_phi128_decodeheavy__internal_full_dense_256",
            "train_doc_count": 256,
            "root_direct_count_mae": 0.39,
            "leaf_direct_exact_match": 0.89,
            "merge_direct_exact_match": 0.79,
            "phi_direct_probe_merge_gap": 0.03,
            "phi_merge_alignment": 0.93,
        },
        {
            "tuning_stage": "phase2",
            "config_label": "shared_feature_phi192_decodeheavy__internal_full_dense_256",
            "train_doc_count": 256,
            "root_direct_count_mae": 0.37,
            "leaf_direct_exact_match": 0.83,
            "merge_direct_exact_match": 0.70,
            "phi_direct_probe_merge_gap": 0.02,
            "phi_merge_alignment": 0.94,
        },
    ]

    winner = surface_push._select_phase2_winner(runs)

    assert winner == "shared_feature_phi128_decodeheavy"


def test_slotwise_scaling_best_5120_label_prefers_lower_root_with_strong_exactness() -> None:
    runs = [
        {
            "tuning_stage": "phase2",
            "train_doc_count": 5120,
            "config_label": "slotwise_scaling_internal_full_dense_5120",
            "root_direct_count_mae": 0.28,
            "leaf_direct_exact_match": 0.90,
            "merge_direct_exact_match": 0.86,
        },
        {
            "tuning_stage": "phase2",
            "train_doc_count": 5120,
            "config_label": "slotwise_scaling_internal_full_r0p25_5120",
            "root_direct_count_mae": 0.31,
            "leaf_direct_exact_match": 0.90,
            "merge_direct_exact_match": 0.83,
        },
    ]

    best = scaling_push._best_5120_label(runs)

    assert best == "slotwise_scaling_internal_full_dense_5120"
