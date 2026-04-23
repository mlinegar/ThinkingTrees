from __future__ import annotations

from scripts.render_markov_sticky_allocation_policy_grid import (
    BALANCED_NODE_FAMILY,
    DEPTH_EQUAL_FAMILY,
    LEAF_ONLY_FAMILY,
    ROOT_ONLY_FAMILY,
    _build_allocation_coverage_summary,
    _build_pure_allocation_view,
    _build_replacement_view,
    _classify_allocation_package,
    _pure_allocation_caption_text,
    _replacement_caption_text,
)


def _merged_summary_fixture() -> dict:
    rows = [
        {
            "package_name": "full100",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.20,
        },
        {
            "package_name": "full90",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 128,
            "test_root_mae_mean": 0.30,
        },
        {
            "package_name": "full90",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.31,
        },
        {
            "package_name": "full90",
            "baseline_family": "fno",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 128,
            "test_root_mae_mean": 0.25,
        },
        {
            "package_name": "r90_leaf_mass_eq_10p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 128,
            "test_root_mae_mean": 0.28,
        },
        {
            "package_name": "r90_leaf_mass_eq_10p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.24,
        },
        {
            "package_name": "r90_depth_equal_mass_eq_10p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.22,
        },
        {
            "package_name": "r100_node_mass_eq_10p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.23,
        },
        {
            "package_name": "r0_leaf_mass_eq_100p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.85,
        },
        {
            "package_name": "r0_depth_equal_mass_eq_100p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.72,
        },
        {
            "package_name": "r100_node_mass_eq_100p0",
            "baseline_family": "tree_neural",
            "train_doc_count": 10240,
            "fixed_leaf_tokens": 32,
            "test_root_mae_mean": 0.66,
        },
    ]
    return {
        "supervision_recovery": {
            "scopes": {
                "recoverable_v5_t128": {
                    "scope_key": "recoverable_v5_t128",
                    "scope_label": "recoverable_v5_t128",
                    "rows_by_train_docs": {
                        "10240": {
                            "train_doc_count": 10240,
                            "rows": rows,
                        }
                    },
                }
            }
        }
    }


def test_classify_allocation_package_normalizes_node_mass_to_residual_root_share() -> None:
    payload = _classify_allocation_package("r100_node_mass_eq_60p0")

    assert payload["family"] == BALANCED_NODE_FAMILY
    assert payload["root_share"] == 40
    assert payload["local_mass_percent"] == 60.0


def test_replacement_and_pure_views_use_normalized_allocation_families() -> None:
    merged_summary = _merged_summary_fixture()

    replacement_view = _build_replacement_view(
        merged_summary,
        scope_key="recoverable_v5_t128",
        train_doc_count=10240,
    )
    pure_view = _build_pure_allocation_view(
        merged_summary,
        scope_key="recoverable_v5_t128",
        train_doc_count=10240,
    )

    panel90 = next(panel for panel in replacement_view["panels"] if panel["root_share"] == 90)
    assert panel90["fno_root_mae"] == 0.25
    assert [point["leaf_tokens"] for point in panel90["series"][ROOT_ONLY_FAMILY]] == [128, 32]
    assert [point["leaf_tokens"] for point in panel90["series"][LEAF_ONLY_FAMILY]] == [128, 32]
    assert [point["leaf_tokens"] for point in panel90["series"][DEPTH_EQUAL_FAMILY]] == [32]
    assert [point["leaf_tokens"] for point in panel90["series"][BALANCED_NODE_FAMILY]] == [32]

    leaf32 = next(panel for panel in pure_view["panels"] if panel["leaf_tokens"] == 32)
    assert leaf32["root_only_reference"]["root_share"] == 100
    assert [point["root_share"] for point in leaf32["series"][LEAF_ONLY_FAMILY]] == [90, 0]
    assert [point["root_share"] for point in leaf32["series"][DEPTH_EQUAL_FAMILY]] == [90, 0]
    assert [point["root_share"] for point in leaf32["series"][BALANCED_NODE_FAMILY]] == [90, 0]


def test_allocation_coverage_summary_tracks_family_specific_support() -> None:
    coverage = _build_allocation_coverage_summary(
        _merged_summary_fixture(),
        train_doc_count=10240,
    )
    recoverable = coverage["scopes"]["recoverable_v5_t128"]

    assert recoverable["replacement_root_shares"]["90"]["root_only_leaf_tokens"] == [128, 32]
    assert recoverable["replacement_root_shares"]["90"]["leaf_only_leaf_tokens"] == [128, 32]
    assert recoverable["replacement_root_shares"]["90"]["depth_equal_leaf_tokens"] == [32]
    assert recoverable["replacement_root_shares"]["90"]["balanced_node_leaf_tokens"] == [32]
    assert recoverable["pure_allocation_leaf_tokens"]["32"]["leaf_only_root_shares"] == [90, 0]
    assert recoverable["pure_allocation_leaf_tokens"]["32"]["depth_equal_root_shares"] == [90, 0]
    assert recoverable["pure_allocation_leaf_tokens"]["32"]["balanced_node_root_shares"] == [90, 0]


def test_allocation_captions_state_budget_semantics_explicitly() -> None:
    replacement = _replacement_caption_text(train_doc_count=10240)
    pure_allocation = _pure_allocation_caption_text()

    assert "same root-label budget" in replacement
    assert "total full-document-equivalent mass stays fixed at `1.0`" in replacement
    assert "root-only ladder is intentionally excluded" in pure_allocation
    assert "total full-document-equivalent supervision mass fixed at `1.0`" in pure_allocation
