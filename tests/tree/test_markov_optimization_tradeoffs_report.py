from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.report_markov_optimization_tradeoffs import (
    BEST_FULL_ROOT_CEILING_COLOR,
    FNO_OFFICIAL_COLOR,
    FNO_SUMLEN_COLOR,
    NEUTRAL_COLOR,
    TREE_LOCAL_COLOR,
    TREE_PRIMARY_COLOR,
    _best_tree_summary_rows,
    _best_full_root_root_mae_by_train_docs,
    _effective_fixed_leaf_tokens,
    _effective_leaves_per_doc,
    _focused_scope_lines,
    _is_exact_full_doc_parity_row,
    _leaf_geometry_warning_lines,
    _merge_supervision_recovery_payloads,
    _ordered_family_payloads,
    _package_semantics,
    _package_tick_label,
    _row_intent_discriminator,
    _summarize_law_packages,
    _summarize_medium_grid,
    _summarize_supervision_recovery,
    _summarize_runtime_efficiency,
    _summarize_support,
    _summarize_supervision_sweep,
    _tree_root_mae_from_family_row,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _with_v3_row_contract(
    row: dict[str, object],
    *,
    comparison_mode: str = "comparable",
    comparison_semantics: str = "current",
    run_intent_hash: str = "intent_hash",
    requested_fixed_leaf_tokens: int | None = None,
    executed_fixed_leaf_tokens: int | None = None,
    depth_discount_gamma: float = 1.0,
) -> dict[str, object]:
    updated = dict(row)
    fixed_leaf_tokens = int(updated.get("fixed_leaf_tokens", 128) or 128)
    updated.setdefault("comparison_mode", comparison_mode)
    updated.setdefault("comparison_semantics", comparison_semantics)
    updated.setdefault(
        "comparison_semantics_label",
        "tree_neural_objective_v2" if comparison_semantics == "current" else comparison_semantics,
    )
    updated.setdefault("run_intent_hash", run_intent_hash)
    updated.setdefault(
        "run_intent_validation_status",
        "validated" if comparison_semantics != "locked_comparator" else "locked_comparator",
    )
    updated.setdefault(
        "requested_fixed_leaf_tokens",
        fixed_leaf_tokens if requested_fixed_leaf_tokens is None else requested_fixed_leaf_tokens,
    )
    updated.setdefault(
        "executed_fixed_leaf_tokens",
        fixed_leaf_tokens if executed_fixed_leaf_tokens is None else executed_fixed_leaf_tokens,
    )
    updated.setdefault("depth_discount_gamma", depth_discount_gamma)
    return updated


def test_report_helpers_distinguish_r100_superset_from_mass_matched_lane() -> None:
    recovery = {
        "status": "ready",
        "package_order": [
            "full100",
            "r100_mass_local_eq_10p0",
            "r100_superset_local_eq_10p0",
        ],
        "package_definitions": {
            "full100": {
                "label": "100% full-doc only",
                "package_semantics": "full_doc_only",
            },
            "r100_mass_local_eq_10p0": {
                "label": "R100 mass-matched + 10.0% leaf/internal count",
                "package_semantics": "mass_matched",
            },
            "r100_superset_local_eq_10p0": {
                "label": "R100 superset + 10.0% leaf/internal count",
                "package_semantics": "superset",
            },
        },
        "train_doc_counts": [10240],
        "scopes": {
            "recoverable_v4_t128": {
                "scope_label": "recoverable_v4_t128",
            }
        },
    }
    summary = {
        "supervision_recovery": recovery,
        "source_records": {
            "supervision_recovery_summary": {
                "status": "ready",
                "phase": "supervision_recovery",
            }
        },
    }

    lines = _focused_scope_lines(
        summary,
        scope_key="recoverable_v4_t128",
        title_kind="recoverable",
    )

    assert _package_semantics(recovery, "full100") == "full_doc_only"
    assert _package_semantics(recovery, "r100_mass_local_eq_10p0") == "mass_matched"
    assert _package_semantics(recovery, "r100_superset_local_eq_10p0") == "superset"
    assert _package_tick_label("r100_superset_local_eq_10p0") == "R100sup+10.0"
    assert any("superset" in line for line in lines)
    assert any("mass-matched" in line for line in lines)


def test_row_intent_discriminator_empty_fallback_is_stable() -> None:
    assert _row_intent_discriminator({}) == ""


def _focused_supervision_recovery_payload(
    *,
    train_docs: list[int] | None = None,
    package_order: list[str] | None = None,
    include_r10_local_law_rates: bool = False,
    include_r20_local_law_rates: bool = False,
    include_r10_mass_matched_rates: bool = False,
    include_r20_mass_matched_rates: bool = False,
    include_r100_mass_matched_rates: bool = False,
) -> dict[str, object]:
    package_order = package_order or [
        "full100",
        "full50",
        "full30",
        "full20",
        "full10",
        "full10_leaf_count100",
        "full10_leaf_full100",
        "full10_leaf_full100_internal_depth1_count100",
        "full10_leaf_full100_internal_depth2_count100",
        "full10_leaf_full100_internal_count100",
        "full20_leaf_full100_internal_count100",
        "full30_leaf_full100_internal_count100",
        "full50_leaf_full100_internal_count100",
    ]
    if include_r10_local_law_rates:
        package_order = list(package_order) + [
            "full10_leaf_count10_internal_count10",
            "full10_leaf_count20_internal_count20",
            "full10_leaf_count50_internal_count50",
            "full10_leaf_count100_internal_count100",
        ]
    if include_r20_local_law_rates:
        package_order = list(package_order) + [
            "full20_leaf_count10_internal_count10",
            "full20_leaf_count20_internal_count20",
            "full20_leaf_count50_internal_count50",
            "full20_leaf_count100_internal_count100",
        ]
    if include_r10_mass_matched_rates:
        package_order = list(package_order) + [
            "r10_mass_local_eq_0p5",
            "r10_mass_local_eq_1p0",
            "r10_mass_local_eq_1p5",
            "r10_mass_local_eq_2p0",
        ]
    if include_r20_mass_matched_rates:
        package_order = list(package_order) + [
            "r20_mass_local_eq_1p0",
            "r20_mass_local_eq_2p0",
            "r20_mass_local_eq_3p0",
            "r20_mass_local_eq_4p0",
        ]
    if include_r100_mass_matched_rates:
        package_order = list(package_order) + [
            "r100_mass_local_eq_5p0",
            "r100_mass_local_eq_10p0",
            "r100_mass_local_eq_15p0",
            "r100_mass_local_eq_20p0",
        ]
    train_doc_counts = train_docs or [1024, 2048, 4096]
    package_definitions = {
        "full100": {"label": "100% full-doc only", "fno_reference_package": "full100"},
        "full50": {"label": "50% full-doc only", "fno_reference_package": "full50"},
        "full30": {"label": "30% full-doc only", "fno_reference_package": "full30"},
        "full20": {"label": "20% full-doc only", "fno_reference_package": "full20"},
        "full10": {"label": "10% full-doc only", "fno_reference_package": "full10"},
        "full10_leaf_count100": {"label": "10% full-doc + leaf count", "fno_reference_package": "full10"},
        "full10_leaf_count10_internal_count10": {
            "label": "10% full-doc + 10% leaf/internal count",
            "fno_reference_package": "full10",
        },
        "full10_leaf_count20_internal_count20": {
            "label": "10% full-doc + 20% leaf/internal count",
            "fno_reference_package": "full10",
        },
        "full10_leaf_count50_internal_count50": {
            "label": "10% full-doc + 50% leaf/internal count",
            "fno_reference_package": "full10",
        },
        "full10_leaf_count100_internal_count100": {
            "label": "10% full-doc + 100% leaf/internal count",
            "fno_reference_package": "full10",
        },
        "full20_leaf_count10_internal_count10": {
            "label": "20% full-doc + 10% leaf/internal count",
            "fno_reference_package": "full20",
        },
        "full20_leaf_count20_internal_count20": {
            "label": "20% full-doc + 20% leaf/internal count",
            "fno_reference_package": "full20",
        },
        "full20_leaf_count50_internal_count50": {
            "label": "20% full-doc + 50% leaf/internal count",
            "fno_reference_package": "full20",
        },
        "full20_leaf_count100_internal_count100": {
            "label": "20% full-doc + 100% leaf/internal count",
            "fno_reference_package": "full20",
        },
        "r10_mass_local_eq_0p5": {
            "label": "R10 mass-matched + 0.5% leaf/internal count",
            "fno_reference_package": "full10",
            "mass_target_per_doc": 0.1,
            "leaf_label_rate": 0.005,
        },
        "r10_mass_local_eq_1p0": {
            "label": "R10 mass-matched + 1.0% leaf/internal count",
            "fno_reference_package": "full10",
            "mass_target_per_doc": 0.1,
            "leaf_label_rate": 0.01,
        },
        "r10_mass_local_eq_1p5": {
            "label": "R10 mass-matched + 1.5% leaf/internal count",
            "fno_reference_package": "full10",
            "mass_target_per_doc": 0.1,
            "leaf_label_rate": 0.015,
        },
        "r10_mass_local_eq_2p0": {
            "label": "R10 mass-matched + 2.0% leaf/internal count",
            "fno_reference_package": "full10",
            "mass_target_per_doc": 0.1,
            "leaf_label_rate": 0.02,
        },
        "r20_mass_local_eq_1p0": {
            "label": "R20 mass-matched + 1.0% leaf/internal count",
            "fno_reference_package": "full20",
            "mass_target_per_doc": 0.2,
            "leaf_label_rate": 0.01,
        },
        "r20_mass_local_eq_2p0": {
            "label": "R20 mass-matched + 2.0% leaf/internal count",
            "fno_reference_package": "full20",
            "mass_target_per_doc": 0.2,
            "leaf_label_rate": 0.02,
        },
        "r20_mass_local_eq_3p0": {
            "label": "R20 mass-matched + 3.0% leaf/internal count",
            "fno_reference_package": "full20",
            "mass_target_per_doc": 0.2,
            "leaf_label_rate": 0.03,
        },
        "r20_mass_local_eq_4p0": {
            "label": "R20 mass-matched + 4.0% leaf/internal count",
            "fno_reference_package": "full20",
            "mass_target_per_doc": 0.2,
            "leaf_label_rate": 0.04,
        },
        "r100_mass_local_eq_5p0": {
            "label": "R100 mass-matched + 5.0% leaf/internal count",
            "fno_reference_package": "full100",
            "mass_target_per_doc": 1.0,
            "leaf_label_rate": 0.05,
        },
        "r100_mass_local_eq_10p0": {
            "label": "R100 mass-matched + 10.0% leaf/internal count",
            "fno_reference_package": "full100",
            "mass_target_per_doc": 1.0,
            "leaf_label_rate": 0.10,
        },
        "r100_mass_local_eq_15p0": {
            "label": "R100 mass-matched + 15.0% leaf/internal count",
            "fno_reference_package": "full100",
            "mass_target_per_doc": 1.0,
            "leaf_label_rate": 0.15,
        },
        "r100_mass_local_eq_20p0": {
            "label": "R100 mass-matched + 20.0% leaf/internal count",
            "fno_reference_package": "full100",
            "mass_target_per_doc": 1.0,
            "leaf_label_rate": 0.20,
        },
        "full10_leaf_full100": {"label": "10% full-doc + leaf full", "fno_reference_package": "full10"},
        "full10_leaf_full100_internal_depth1_count100": {
            "label": "10% full-doc + leaf full + depth-1 internal count",
            "fno_reference_package": "full10",
        },
        "full10_leaf_full100_internal_depth2_count100": {
            "label": "10% full-doc + leaf full + depth-1+2 internal count",
            "fno_reference_package": "full10",
        },
        "full10_leaf_full100_internal_count100": {
            "label": "10% full-doc + leaf full + all internal count",
            "fno_reference_package": "full10",
        },
        "full20_leaf_full100_internal_count100": {
            "label": "20% full-doc + leaf full + all internal count",
            "fno_reference_package": "full20",
        },
        "full30_leaf_full100_internal_count100": {
            "label": "30% full-doc + leaf full + all internal count",
            "fno_reference_package": "full30",
        },
        "full50_leaf_full100_internal_count100": {
            "label": "50% full-doc + leaf full + all internal count",
            "fno_reference_package": "full50",
        },
    }

    def _scope_rows(
        scope_label: str,
        *,
        structural: bool = False,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
        rows_by_train_docs = []
        dense_anchor_rows = []
        best_tree_by_train_docs: dict[str, dict[str, object]] = {}
        per_doc_values = {
            1024: (0.10 if not structural else 0.22, 0.18 if not structural else 0.30, 0.14 if not structural else 0.24),
            2048: (0.06 if not structural else 0.13, 0.11 if not structural else 0.19, 0.09 if not structural else 0.16),
            4096: (0.04 if not structural else 0.08, 0.07 if not structural else 0.12, 0.06 if not structural else 0.10),
        }
        local_mass_multiplier = 5.0 if structural else 4.0
        for train_doc_count in train_doc_counts:
            fno_full100, fno_full10, best_tree = per_doc_values[train_doc_count]
            rows = [
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full100",
                    "tree_test_root_mae": fno_full100 + 0.03,
                    "fno_reference_package": "full100",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full100,
                    "delta_vs_full10_fno": fno_full100 + 0.03 - fno_full10,
                    "delta_vs_full100_fno_ceiling": 0.03,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full50",
                    "tree_test_root_mae": best_tree + 0.02,
                    "fno_reference_package": "full50",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10 - 0.08,
                    "delta_vs_full10_fno": best_tree + 0.02 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.02 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full30",
                    "tree_test_root_mae": best_tree + 0.018,
                    "fno_reference_package": "full30",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10 - 0.05,
                    "delta_vs_full10_fno": best_tree + 0.018 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.018 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full20",
                    "tree_test_root_mae": best_tree + 0.015,
                    "fno_reference_package": "full20",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10 - 0.02,
                    "delta_vs_full10_fno": best_tree + 0.015 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.015 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full10",
                    "tree_test_root_mae": fno_full10 + 0.02,
                    "fno_reference_package": "full10",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10,
                    "delta_vs_full10_fno": 0.02,
                    "delta_vs_full100_fno_ceiling": fno_full10 + 0.02 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full10_leaf_count100",
                    "tree_test_root_mae": best_tree + 0.03,
                    "fno_reference_package": "full10",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10,
                    "delta_vs_full10_fno": best_tree + 0.03 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.03 - fno_full100,
                },
                *(
                    [
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full10_leaf_count10_internal_count10",
                            "tree_test_root_mae": best_tree + 0.028,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.028 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.028 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full10_leaf_count20_internal_count20",
                            "tree_test_root_mae": best_tree + 0.020,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.020 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.020 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full10_leaf_count50_internal_count50",
                            "tree_test_root_mae": best_tree + 0.012,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.012 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.012 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full10_leaf_count100_internal_count100",
                            "tree_test_root_mae": best_tree + 0.008,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.008 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.008 - fno_full100,
                        },
                    ]
                    if include_r10_local_law_rates
                    else []
                ),
                *(
                    [
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full20_leaf_count10_internal_count10",
                            "tree_test_root_mae": best_tree + 0.024,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.024 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.024 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full20_leaf_count20_internal_count20",
                            "tree_test_root_mae": best_tree + 0.018,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.018 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.018 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full20_leaf_count50_internal_count50",
                            "tree_test_root_mae": best_tree + 0.010,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.010 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.010 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "full20_leaf_count100_internal_count100",
                            "tree_test_root_mae": best_tree + 0.006,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.006 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.006 - fno_full100,
                        },
                    ]
                    if include_r20_local_law_rates
                    else []
                ),
                *(
                    [
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r10_mass_local_eq_0p5",
                            "tree_test_root_mae": best_tree + 0.018,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.018 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.018 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r10_mass_local_eq_1p0",
                            "tree_test_root_mae": best_tree + 0.012,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.012 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.012 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r10_mass_local_eq_1p5",
                            "tree_test_root_mae": best_tree + 0.008,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.008 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.008 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r10_mass_local_eq_2p0",
                            "tree_test_root_mae": best_tree + 0.005,
                            "fno_reference_package": "full10",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10,
                            "delta_vs_full10_fno": best_tree + 0.005 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.005 - fno_full100,
                        },
                    ]
                    if include_r10_mass_matched_rates
                    else []
                ),
                *(
                    [
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r20_mass_local_eq_1p0",
                            "tree_test_root_mae": best_tree + 0.017,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.017 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.017 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r20_mass_local_eq_2p0",
                            "tree_test_root_mae": best_tree + 0.010,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.010 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.010 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r20_mass_local_eq_3p0",
                            "tree_test_root_mae": best_tree + 0.006,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.006 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.006 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r20_mass_local_eq_4p0",
                            "tree_test_root_mae": best_tree + 0.004,
                            "fno_reference_package": "full20",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full10 - 0.02,
                            "delta_vs_full10_fno": best_tree + 0.004 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.004 - fno_full100,
                        },
                    ]
                    if include_r20_mass_matched_rates
                    else []
                ),
                *(
                    [
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r100_mass_local_eq_5p0",
                            "tree_test_root_mae": best_tree + 0.020,
                            "fno_reference_package": "full100",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full100,
                            "delta_vs_full10_fno": best_tree + 0.020 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.020 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r100_mass_local_eq_10p0",
                            "tree_test_root_mae": best_tree + 0.014,
                            "fno_reference_package": "full100",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full100,
                            "delta_vs_full10_fno": best_tree + 0.014 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.014 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r100_mass_local_eq_15p0",
                            "tree_test_root_mae": best_tree + 0.009,
                            "fno_reference_package": "full100",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full100,
                            "delta_vs_full10_fno": best_tree + 0.009 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.009 - fno_full100,
                        },
                        {
                            "scope_label": scope_label,
                            "train_doc_count": train_doc_count,
                            "package_name": "r100_mass_local_eq_20p0",
                            "tree_test_root_mae": best_tree + 0.006,
                            "fno_reference_package": "full100",
                            "fno_reference_family": "official_fno_sumlen",
                            "fno_reference_test_root_mae": fno_full100,
                            "delta_vs_full10_fno": best_tree + 0.006 - fno_full10,
                            "delta_vs_full100_fno_ceiling": best_tree + 0.006 - fno_full100,
                        },
                    ]
                    if include_r100_mass_matched_rates
                    else []
                ),
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full10_leaf_full100",
                    "tree_test_root_mae": best_tree + 0.01,
                    "fno_reference_package": "full10",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10,
                    "delta_vs_full10_fno": best_tree + 0.01 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.01 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full10_leaf_full100_internal_depth1_count100",
                    "tree_test_root_mae": best_tree + 0.006,
                    "fno_reference_package": "full10",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10,
                    "delta_vs_full10_fno": best_tree + 0.006 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.006 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full10_leaf_full100_internal_depth2_count100",
                    "tree_test_root_mae": best_tree + 0.002,
                    "fno_reference_package": "full10",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10,
                    "delta_vs_full10_fno": best_tree + 0.002 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.002 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full10_leaf_full100_internal_count100",
                    "tree_test_root_mae": best_tree,
                    "fno_reference_package": "full10",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10,
                    "delta_vs_full10_fno": best_tree - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full20_leaf_full100_internal_count100",
                    "tree_test_root_mae": best_tree + 0.01,
                    "fno_reference_package": "full20",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10 - 0.02,
                    "delta_vs_full10_fno": best_tree + 0.01 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.01 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full30_leaf_full100_internal_count100",
                    "tree_test_root_mae": best_tree + 0.008,
                    "fno_reference_package": "full30",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10 - 0.05,
                    "delta_vs_full10_fno": best_tree + 0.008 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.008 - fno_full100,
                },
                {
                    "scope_label": scope_label,
                    "train_doc_count": train_doc_count,
                    "package_name": "full50_leaf_full100_internal_count100",
                    "tree_test_root_mae": best_tree + 0.006,
                    "fno_reference_package": "full50",
                    "fno_reference_family": "official_fno_sumlen",
                    "fno_reference_test_root_mae": fno_full10 - 0.08,
                    "delta_vs_full10_fno": best_tree + 0.006 - fno_full10,
                    "delta_vs_full100_fno_ceiling": best_tree + 0.006 - fno_full100,
                },
            ]
            for index, row in enumerate(rows):
                root_mae = float(row["tree_test_root_mae"])
                row["tree_val_root_mae"] = max(root_mae - 0.005, 0.0)
                row["tree_test_leaf_mae"] = max(root_mae * 0.45, 0.002)
                row["tree_test_merge_mae"] = max(root_mae * 0.75, 0.003)
                row["tree_test_full_law_objective"] = root_mae + 0.20
                row["tree_val_full_law_objective"] = max(root_mae + 0.16, 0.0)
                row["tree_test_active_objective"] = root_mae + 0.12
                row["tree_val_active_objective"] = max(root_mae + 0.10, 0.0)
                row["tree_best_epoch"] = 40 + index
                row["tree_selection_metric_name"] = "val_root_mae"
                row["tree_selection_metric_value"] = row["tree_val_root_mae"]
                row["tree_checkpoint_metric"] = "val_root_mae"
                row["tree_stage1_checkpoint_metric"] = "val_theorem_bootstrap_direct"
                row["tree_reference_label"] = "common_factorized_sketch_v1"
                if row["package_name"].startswith(("r10_mass_", "r20_mass_")):
                    spec = dict(package_definitions[row["package_name"]])
                    rate = float(spec["leaf_label_rate"])
                    target_mass = float(spec["mass_target_per_doc"])
                    computed_local_mass = rate * local_mass_multiplier
                    row["tree_mass_target_per_doc"] = target_mass
                    row["tree_computed_doc_review_mass_per_doc"] = max(
                        target_mass - computed_local_mass,
                        0.0,
                    )
                    row["tree_computed_local_mass_per_doc"] = computed_local_mass
                    row["tree_computed_leaf_mass_per_doc"] = rate
                    row["tree_computed_internal_mass_per_doc"] = rate * (
                        local_mass_multiplier - 1.0
                    )
                    row["tree_effective_full_doc_mass_per_doc"] = target_mass + 0.001
                if row["package_name"].startswith("r100_mass_"):
                    spec = dict(package_definitions[row["package_name"]])
                    rate = float(spec["leaf_label_rate"])
                    target_mass = float(spec["mass_target_per_doc"])
                    computed_local_mass = rate * local_mass_multiplier
                    row["tree_mass_target_per_doc"] = target_mass
                    row["tree_computed_doc_review_mass_per_doc"] = max(
                        target_mass - computed_local_mass,
                        0.0,
                    )
                    row["tree_computed_local_mass_per_doc"] = computed_local_mass
                    row["tree_computed_leaf_mass_per_doc"] = rate
                    row["tree_computed_internal_mass_per_doc"] = rate * (
                        local_mass_multiplier - 1.0
                    )
                    row["tree_effective_full_doc_mass_per_doc"] = target_mass + 0.001
            rows_by_train_docs.append({"train_doc_count": train_doc_count, "rows": rows})
            dense_anchor_rows.append(
                {
                    "train_doc_count": train_doc_count,
                    "package_name": "full100",
                    "tree_test_root_mae": fno_full100 + 0.03,
                    "fno_reference_test_root_mae": fno_full100,
                    "delta_vs_full100_fno_ceiling": 0.03,
                }
            )
            best_tree_by_train_docs[str(train_doc_count)] = dict(
                min(rows, key=lambda row: float(row["tree_test_root_mae"]))
            )
        return rows_by_train_docs, dense_anchor_rows, best_tree_by_train_docs

    recoverable_rows, recoverable_dense, recoverable_best = _scope_rows("recoverable_v4")
    structural_rows, structural_dense, structural_best = _scope_rows(
        "structural_core_v1::r12_seg10to12",
        structural=True,
    )
    return {
        "status": "ready",
        "tree_family": "tree_neural",
        "canonical_fno_families": ["official_fno", "official_fno_sumlen"],
        "train_doc_counts": train_doc_counts,
        "seeds": [0, 1],
        "seed_count": 2,
        "package_order": package_order,
        "package_definitions": package_definitions,
        "structural_scope_key": "r12_seg10to12",
        "scope_tree_references": {
            "recoverable_v4": {
                "scope_key": "recoverable_v4",
                "scope_label": "recoverable_v4",
                "tree_reference_mode": "preset",
                "tree_reference_label": "common_factorized_sketch_v1",
                "tree_training_schedule": "two_stage",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "summary_spec_name": "markov_count_sketch",
                "slot_count": 4,
                "state_dim": 128,
                "hidden_dim": 512,
                "fixed_leaf_tokens": 16,
            },
            "r12_seg10to12": {
                "scope_key": "r12_seg10to12",
                "scope_label": "structural_core_v1::r12_seg10to12",
                "tree_reference_mode": "preset",
                "tree_reference_label": "common_factorized_sketch_v1",
                "tree_training_schedule": "two_stage",
                "tree_checkpoint_metric": "val_root_mae",
                "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                "summary_spec_name": "markov_count_sketch",
                "slot_count": 4,
                "state_dim": 128,
                "hidden_dim": 512,
                "fixed_leaf_tokens": 16,
            },
        },
        "scopes": {
            "recoverable_v4": {
                "scope_label": "recoverable_v4",
                "rows_by_train_docs": recoverable_rows,
                "dense_anchor_rows": recoverable_dense,
                "best_tree_by_train_docs": recoverable_best,
            },
            "r12_seg10to12": {
                "scope_label": "structural_core_v1::r12_seg10to12",
                "rows_by_train_docs": structural_rows,
                "dense_anchor_rows": structural_dense,
                "best_tree_by_train_docs": structural_best,
            },
        },
        "best_tree_summary": [
            {"scope_key": "recoverable_v4", "scope_label": "recoverable_v4", "train_doc_count": int(key), **value}
            for key, value in sorted(recoverable_best.items())
        ]
        + [
            {
                "scope_key": "r12_seg10to12",
                "scope_label": "structural_core_v1::r12_seg10to12",
                "train_doc_count": int(key),
                **value,
            }
            for key, value in sorted(structural_best.items())
        ],
        "runtime_diagnosis": {
            "status": "ready",
            "tree_fast_path_confirmed_runs": 6,
            "tree_partial_or_fallback_runs": 0,
            "tree_fast_path_completion_rate": 1.0,
            "tree_zero_h2d_rate": 1.0,
            "tree_median_train_loop_s_per_epoch": 0.25,
            "tree_median_train_loop_s_per_epoch_per_1k_docs": 0.12,
            "tree_median_resident_store_hits": 48.0,
            "tree_median_dense_bucket_hits": 48.0,
            "tree_median_auto_queue_fused_batches": 64.0,
            "tree_median_document_loss_batch_scale": 4.0,
            "current_evidence_status": "fast_path_engaged_and_likely_materially_helping",
            "grouped_rows": [
                {
                    "scope_label": "recoverable_v4",
                    "train_doc_count": 1024,
                    "package_name": "full10_leaf_full100_internal_count100",
                    "n_seeds_completed": 2,
                    "fast_path_classification": "fast_path_confirmed",
                    "tree_reference_label": "recoverable_slotwise_dense_v1",
                    "tree_training_schedule": "two_stage",
                    "effective_tree_document_loss_normalization_mode": "supervised_docs",
                    "zero_h2d_rate": 1.0,
                    "train_loop_s_per_epoch_median": 0.25,
                    "train_loop_s_per_epoch_per_1k_docs_median": 0.12,
                    "document_loss_mean_batch_scale_median": 4.0,
                }
            ],
        },
        "common_tree_reference_label": "common_factorized_sketch_v1",
        "comparator_alignment_status": "aligned",
        "comparator_alignment_warning": "",
        "tree_checkpoint_metrics": ["val_root_mae"],
        "comparator_selection_status": "root_comparable",
        "comparator_selection_warning": "",
        "canonical_tree_selection_metric": "val_root_mae",
        "canonical_tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
        "canonical_comparison_rule": "all tree ladder points selected on val_root_mae; local metrics are diagnostics",
    }


def _retarget_supervision_recovery_payload_to_t128(
    payload: dict[str, object],
    *,
    recoverable_scope_key: str = "recoverable_v4_t128",
    structural_grid: str = "structural_core_v1_t128",
    structural_cell: str = "r12_seg10to12",
) -> dict[str, object]:
    updated = json.loads(json.dumps(payload))
    recoverable_label = str(recoverable_scope_key)
    structural_label = f"{structural_grid}::{structural_cell}"

    def _rewrite(value: object) -> object:
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        if isinstance(value, dict):
            rewritten = {key: _rewrite(item) for key, item in value.items()}
            if rewritten.get("scope_key") == "recoverable_v4":
                rewritten["scope_key"] = recoverable_label
            if rewritten.get("scope_label") == "recoverable_v4":
                rewritten["scope_label"] = recoverable_label
            if rewritten.get("scope_label") == "structural_core_v1::r12_seg10to12":
                rewritten["scope_label"] = structural_label
            return rewritten
        return value

    updated = _rewrite(updated)
    assert isinstance(updated, dict)
    updated["recoverable_scope_key"] = recoverable_label
    updated["recoverable_scope_label"] = recoverable_label
    updated["structural_hardness_grid"] = structural_grid
    updated["structural_scope_key"] = structural_cell
    updated["structural_scope_label"] = structural_label

    scope_tree_references = dict(updated.get("scope_tree_references") or {})
    recoverable_ref = dict(scope_tree_references.pop("recoverable_v4"))
    recoverable_ref["scope_key"] = recoverable_label
    recoverable_ref["scope_label"] = recoverable_label
    scope_tree_references[recoverable_label] = recoverable_ref
    structural_ref = dict(scope_tree_references.get(structural_cell) or {})
    structural_ref["scope_label"] = structural_label
    scope_tree_references[structural_cell] = structural_ref
    updated["scope_tree_references"] = scope_tree_references

    scopes = dict(updated.get("scopes") or {})
    recoverable_scope = dict(scopes.pop("recoverable_v4"))
    recoverable_scope["scope_label"] = recoverable_label
    scopes[recoverable_label] = recoverable_scope
    structural_scope = dict(scopes.get(structural_cell) or {})
    structural_scope["scope_label"] = structural_label
    scopes[structural_cell] = structural_scope
    updated["scopes"] = scopes
    return updated


def _exact_full_doc_canary_payload() -> dict[str, object]:
    train_doc_counts = [1024, 2048, 4096]
    recoverable_scope_key = "recoverable_v4_t128"
    structural_scope_key = "r12_seg10to12"
    structural_label = "structural_core_v1_t128::r12_seg10to12"
    family_rows: list[dict[str, object]] = []

    def _scope_payload(
        scope_key: str,
        scope_label: str,
        values: dict[int, tuple[float, float, float]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
        rows_by_train_docs: list[dict[str, object]] = []
        dense_anchor_rows: list[dict[str, object]] = []
        best_tree_by_train_docs: dict[str, dict[str, object]] = {}
        for train_doc_count in train_doc_counts:
            tree_mae, official_fno, official_fno_sumlen = values[train_doc_count]
            tree_row = {
                "scope_key": scope_key,
                "scope_label": scope_label,
                "train_doc_count": train_doc_count,
                "package_name": "full100",
                "tree_test_root_mae": tree_mae,
                "fno_reference_package": "full100",
                "fno_reference_family": "official_fno",
                "fno_reference_test_root_mae": official_fno,
                "requested_fixed_leaf_tokens": 128,
                "fixed_leaf_tokens": 128,
                "executed_fixed_leaf_tokens": 128,
                "computed_assumed_doc_tokens": 128,
                "leaves_per_doc": 1,
                "executed_leaves_per_doc": 1,
                "parity_mode": "exact_full_doc",
                "is_exact_full_doc_parity_row": True,
                "is_authoritative_parity_row": True,
                "is_fno_equivalent_geometry": True,
                "tree_supervision_source": "manifest",
                "local_estimand_mode": "span_mass_ipw_sum",
                "tree_family": "tree_neural",
                "tree_reference_label": "unified_g_fno_parity_canary_v1",
            }
            rows_by_train_docs.append(
                {
                    "train_doc_count": train_doc_count,
                    "rows": [dict(tree_row)],
                }
            )
            dense_anchor_rows.append(dict(tree_row))
            best_tree_by_train_docs[str(train_doc_count)] = dict(tree_row)
            family_rows.extend(
                [
                    {
                        "scope_key": scope_key,
                        "scope_label": scope_label,
                        "train_doc_count": train_doc_count,
                        "package_name": "full100",
                        "baseline_family": "tree_neural",
                        "test_root_mae_mean": tree_mae,
                        "fixed_leaf_tokens": 128,
                        "executed_fixed_leaf_tokens": 128,
                        "computed_assumed_doc_tokens": 128,
                        "leaves_per_doc": 1,
                        "executed_leaves_per_doc": 1,
                        "parity_mode": "exact_full_doc",
                        "is_exact_full_doc_parity_row": True,
                        "is_authoritative_parity_row": True,
                        "is_fno_equivalent_geometry": True,
                        "tree_supervision_source": "manifest",
                        "local_estimand_mode": "span_mass_ipw_sum",
                        "n_runs": 1,
                    },
                    {
                        "scope_key": scope_key,
                        "scope_label": scope_label,
                        "train_doc_count": train_doc_count,
                        "package_name": "full100",
                        "baseline_family": "official_fno",
                        "test_root_mae_mean": official_fno,
                        "n_runs": 1,
                    },
                    {
                        "scope_key": scope_key,
                        "scope_label": scope_label,
                        "train_doc_count": train_doc_count,
                        "package_name": "full100",
                        "baseline_family": "official_fno_sumlen",
                        "test_root_mae_mean": official_fno_sumlen,
                        "n_runs": 1,
                    },
                ]
            )
        return rows_by_train_docs, dense_anchor_rows, best_tree_by_train_docs

    recoverable_rows, recoverable_dense, recoverable_best = _scope_payload(
        recoverable_scope_key,
        recoverable_scope_key,
        {
            1024: (0.00, 0.00, 0.00),
            2048: (0.00, 0.00, 0.00),
            4096: (0.00, 0.00, 0.00),
        },
    )
    structural_rows, structural_dense, structural_best = _scope_payload(
        structural_scope_key,
        structural_label,
        {
            1024: (0.73828125, 0.73828125, 0.6953125),
            2048: (0.734375, 0.734375, 0.75),
            4096: (0.56640625, 0.56640625, 0.4453125),
        },
    )
    return {
        "status": "ready",
        "tree_family": "tree_neural",
        "canonical_fno_families": ["official_fno", "official_fno_sumlen"],
        "train_doc_counts": train_doc_counts,
        "seeds": [42],
        "seed_count": 1,
        "package_order": ["full100"],
        "package_definitions": {
            "full100": {"label": "100% full-doc only", "fno_reference_package": "full100"}
        },
        "recoverable_scope_key": recoverable_scope_key,
        "recoverable_scope_label": recoverable_scope_key,
        "structural_hardness_grid": "structural_core_v1_t128",
        "structural_scope_key": structural_scope_key,
        "structural_scope_label": structural_label,
        "scope_tree_references": {
            recoverable_scope_key: {
                "scope_key": recoverable_scope_key,
                "scope_label": recoverable_scope_key,
                "tree_reference_label": "unified_g_fno_parity_canary_v1",
                "tree_training_schedule": "single_stage",
                "fixed_leaf_tokens": 128,
                "state_dim": 128,
                "hidden_dim": 512,
            },
            structural_scope_key: {
                "scope_key": structural_scope_key,
                "scope_label": structural_label,
                "tree_reference_label": "unified_g_fno_parity_canary_v1",
                "tree_training_schedule": "single_stage",
                "fixed_leaf_tokens": 128,
                "state_dim": 128,
                "hidden_dim": 512,
            },
        },
        "scopes": {
            recoverable_scope_key: {
                "scope_label": recoverable_scope_key,
                "rows_by_train_docs": recoverable_rows,
                "dense_anchor_rows": recoverable_dense,
                "best_tree_by_train_docs": recoverable_best,
            },
            structural_scope_key: {
                "scope_label": structural_label,
                "rows_by_train_docs": structural_rows,
                "dense_anchor_rows": structural_dense,
                "best_tree_by_train_docs": structural_best,
            },
        },
        "family_rows": [
            _with_v3_row_contract(
                dict(row),
                run_intent_hash=(
                    f"{row.get('scope_key')}::{row.get('train_doc_count')}::"
                    f"{row.get('package_name')}::{row.get('baseline_family')}"
                ),
            )
            for row in family_rows
        ],
        "best_tree_summary": [
            {
                "scope_key": recoverable_scope_key,
                "scope_label": recoverable_scope_key,
                "train_doc_count": int(key),
                **value,
            }
            for key, value in sorted(recoverable_best.items())
        ]
        + [
            {
                "scope_key": structural_scope_key,
                "scope_label": structural_label,
                "train_doc_count": int(key),
                **value,
            }
            for key, value in sorted(structural_best.items())
        ],
        "runtime_diagnosis": {
            "status": "ready",
            "tree_fast_path_confirmed_runs": 6,
            "tree_partial_or_fallback_runs": 0,
            "tree_fast_path_completion_rate": 1.0,
            "tree_zero_h2d_rate": 1.0,
            "tree_median_train_loop_s_per_epoch": 0.25,
            "tree_median_train_loop_s_per_epoch_per_1k_docs": 0.12,
            "tree_median_resident_store_hits": 48.0,
            "tree_median_dense_bucket_hits": 48.0,
            "tree_median_auto_queue_fused_batches": 64.0,
            "tree_median_document_loss_batch_scale": 1.0,
            "current_evidence_status": "fast_path_engaged_and_likely_materially_helping",
        },
    }


def test_markov_optimization_tradeoffs_report_smoke(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _focused_supervision_recovery_payload(),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    pdf_path = output_dir / "report.pdf"

    assert summary["report_focus"] == "supervision_recovery_v1"
    assert summary["protocol"]["tree_family"] == "tree_neural"
    assert summary["protocol"]["train_doc_counts"] == [1024, 2048, 4096]
    assert summary["protocol"]["package_definitions"]["full10_leaf_full100"]["fno_reference_package"] == "full10"
    assert summary["recoverable_ladder"]["scope_label"] == "recoverable_v4"
    assert summary["structural_ladder"]["scope_label"] == "structural_core_v1::r12_seg10to12"
    assert summary["best_tree_summary"][0]["package_name"] == "full10_leaf_full100_internal_count100"
    assert summary["pdf"] == str(pdf_path)
    assert "Dense Full-Doc Anchor" in summary["figures"]
    assert "Recoverable Package Ladder" in summary["figures"]
    assert "Structural Package Ladder" in summary["figures"]
    assert "Recoverable Ordered Families" in summary["figures"]
    assert "Structural Ordered Families" in summary["figures"]
    assert "Recoverable Dense-Local Root Ladder" in summary["figures"]
    assert "Structural Dense-Local Root Ladder" in summary["figures"]
    assert "Recoverable R10 Local Ablations" in summary["figures"]
    assert "Structural R10 Local Ablations" in summary["figures"]
    assert "Recoverable Tree Diagnostics" in summary["figures"]
    assert "Structural Tree Diagnostics" in summary["figures"]
    assert pdf_path.exists()

    assert "Markov Supervision-Recovery Report" in markdown
    assert "## Key Concepts" in markdown
    assert "## Setup" in markdown
    assert "## Full-Supervision Baseline" in markdown
    assert "## Recoverable All Supervision Settings" in markdown
    assert "## Structural All Supervision Settings" in markdown
    assert "## Recoverable Root-Supervision Sweep" in markdown
    assert "## Structural Root-Supervision Sweep" in markdown
    assert "## Recoverable Full Local Supervision + Root Sweep" in markdown
    assert "## Structural Full Local Supervision + Root Sweep" in markdown
    assert "## Recoverable What Extra Tree Labels Help at R10?" in markdown
    assert "## Structural What Extra Tree Labels Help at R10?" in markdown
    assert "## Recoverable Tree-Only Diagnostics" in markdown
    assert "## Structural Tree-Only Diagnostics" in markdown
    assert "## Best Tree Summary" in markdown
    assert "## Stability Warnings" in markdown
    assert "## Runtime Notes" in markdown
    assert "## Appendix" in markdown
    assert "Canonical tree checkpoint metric: `val_root_mae`." in markdown
    assert "`official_fno` is the canonical parity comparator." in markdown
    assert "matched `full10` baselines:" in markdown or "matched `full100` baselines:" in markdown
    assert "full100 baselines:" in markdown
    assert "Tree recipe (`recoverable_v4`): reference=common_factorized_sketch_v1, schedule=two_stage, state/hidden=128/512, leaf_tokens=16." in markdown
    assert "Tree recipe (`structural_core_v1::r12_seg10to12`): reference=common_factorized_sketch_v1, schedule=two_stage, state/hidden=128/512, leaf_tokens=16." in markdown
    assert "training-doc equivalents" in markdown
    assert "root-labeled docs" in markdown
    assert "![Recoverable Root-Supervision Sweep](figures/recoverable_ordered_families.png)" in markdown
    assert "![Structural Root-Supervision Sweep](figures/structural_ordered_families.png)" in markdown
    assert "| package | description | FNO reference |" not in markdown
    assert "| scope | family | root share |" not in markdown
    assert "| scope | train_docs | package |" not in markdown
    assert "## Dense Full-Doc Anchor" not in markdown
    assert "## Recoverable Package Ladder" not in markdown
    assert "Batch Throughput" not in markdown
    assert "Large-Batch Diagnosis" not in markdown
    assert "Law Packages" not in markdown
    assert "tree_neural_c2" not in markdown
    assert "fast_path_engaged_and_likely_materially_helping" in markdown


def test_markov_optimization_tradeoffs_report_accepts_t128_scope_metadata(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _retarget_supervision_recovery_payload_to_t128(
            _focused_supervision_recovery_payload()
        ),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert summary["protocol"]["benchmarks"] == [
        "recoverable_v4_t128",
        "structural_core_v1_t128::r12_seg10to12",
    ]
    assert summary["recoverable_ladder"]["scope_label"] == "recoverable_v4_t128"
    assert summary["structural_ladder"]["scope_label"] == "structural_core_v1_t128::r12_seg10to12"
    assert "Tree recipe (`recoverable_v4_t128`): reference=common_factorized_sketch_v1, schedule=two_stage, state/hidden=128/512, leaf_tokens=16." in markdown
    assert "Tree recipe (`structural_core_v1_t128::r12_seg10to12`): reference=common_factorized_sketch_v1, schedule=two_stage, state/hidden=128/512, leaf_tokens=16." in markdown


def test_markov_optimization_tradeoffs_report_r10_local_law_coverage_grid(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _focused_supervision_recovery_payload(include_r10_local_law_rates=True),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert "Recoverable R10 Local-Law Coverage" in summary["figures"]
    assert "Structural R10 Local-Law Coverage" in summary["figures"]
    assert "## Recoverable Extra Count Labels at R10" in markdown
    assert "## Structural Extra Count Labels at R10" in markdown


def test_markov_optimization_tradeoffs_report_r20_local_law_coverage_grid(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _focused_supervision_recovery_payload(include_r20_local_law_rates=True),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert "Recoverable R20 Local-Law Coverage" in summary["figures"]
    assert "Structural R20 Local-Law Coverage" in summary["figures"]
    assert "## Recoverable Extra Count Labels at R20" in markdown
    assert "## Structural Extra Count Labels at R20" in markdown


def test_markov_optimization_tradeoffs_report_mass_matched_coverage_grids(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _focused_supervision_recovery_payload(
            include_r10_mass_matched_rates=True,
            include_r20_mass_matched_rates=True,
            include_r100_mass_matched_rates=True,
        ),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert "Recoverable R10 Mass-Matched Coverage" in summary["figures"]
    assert "Recoverable R20 Mass-Matched Coverage" in summary["figures"]
    assert "Recoverable R100 Mass-Matched Coverage" in summary["figures"]
    assert "Recoverable Mass-Matched Overlay" in summary["figures"]
    assert "Structural R10 Mass-Matched Coverage" in summary["figures"]
    assert "Structural R20 Mass-Matched Coverage" in summary["figures"]
    assert "Structural R100 Mass-Matched Coverage" in summary["figures"]
    assert "Structural Mass-Matched Overlay" in summary["figures"]
    assert "## Recoverable Fixed-Budget Tree vs FNO at R10" in markdown
    assert "## Recoverable Fixed-Budget Tree vs FNO at R20" in markdown
    assert "## Recoverable Fixed-Budget Tree vs FNO at R100" in markdown
    assert "## Recoverable Fixed-Budget Comparison Across Budgets" in markdown
    assert "## Structural Fixed-Budget Tree vs FNO at R10" in markdown
    assert "## Structural Fixed-Budget Tree vs FNO at R20" in markdown
    assert "## Structural Fixed-Budget Tree vs FNO at R100" in markdown
    assert "## Structural Fixed-Budget Comparison Across Budgets" in markdown
    assert "training-doc-equivalent budget" in markdown
    assert "`0%` is the root-only `full10` anchor" in markdown
    assert "flat comparison baselines" in markdown
    assert "literal local-rate meaning" in markdown


def test_supervision_recovery_merge_preserves_distinct_leaf_geometries() -> None:
    payload_leaf128 = {
        "status": "ready",
        "tree_family": "tree_neural",
        "package_order": ["full10", "r10_mass_local_eq_2p0"],
        "train_doc_counts": [1024],
        "family_rows": [
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "official_fno",
                "test_root_mae_mean": 0.08,
            },
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "official_fno_sumlen",
                "test_root_mae_mean": 0.09,
            },
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "tree_neural",
                "test_root_mae_mean": 0.082,
                "fixed_leaf_tokens": 128,
                "computed_assumed_doc_tokens": 128,
                "leaves_per_doc": 1,
                "internal_nodes_per_doc": 0,
                "is_fno_equivalent_geometry": True,
            },
        ],
        "scopes": {
            "recoverable_v4": {
                "scope_label": "recoverable_v4",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 1024,
                        "rows": [
                            {
                                "package_name": "full10",
                                "baseline_family": "tree_neural",
                                "tree_test_root_mae": 0.082,
                                "fixed_leaf_tokens": 128,
                                "computed_assumed_doc_tokens": 128,
                                "leaves_per_doc": 1,
                                "internal_nodes_per_doc": 0,
                                "is_fno_equivalent_geometry": True,
                            }
                        ],
                    }
                ],
                "dense_anchor_rows": [],
                "best_tree_by_train_docs": {},
            }
        },
    }
    payload_leaf16 = {
        "status": "ready",
        "tree_family": "tree_neural",
        "package_order": ["full10", "r10_mass_local_eq_2p0"],
        "train_doc_counts": [1024],
        "family_rows": [
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "official_fno",
                "test_root_mae_mean": 0.08,
            },
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "official_fno_sumlen",
                "test_root_mae_mean": 0.09,
            },
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "r10_mass_local_eq_2p0",
                "baseline_family": "tree_neural",
                "test_root_mae_mean": 0.05,
                "fixed_leaf_tokens": 16,
                "computed_assumed_doc_tokens": 128,
                "leaves_per_doc": 8,
                "internal_nodes_per_doc": 7,
                "is_fno_equivalent_geometry": False,
            },
        ],
        "scopes": {
            "recoverable_v4": {
                "scope_label": "recoverable_v4",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 1024,
                        "rows": [
                            {
                                "package_name": "r10_mass_local_eq_2p0",
                                "baseline_family": "tree_neural",
                                "tree_test_root_mae": 0.05,
                                "fixed_leaf_tokens": 16,
                                "computed_assumed_doc_tokens": 128,
                                "leaves_per_doc": 8,
                                "internal_nodes_per_doc": 7,
                                "is_fno_equivalent_geometry": False,
                            }
                        ],
                    }
                ],
                "dense_anchor_rows": [],
                "best_tree_by_train_docs": {},
            }
        },
    }
    payload_leaf128["family_rows"] = [
        _with_v3_row_contract(
            dict(row),
            run_intent_hash=(
                f"{row.get('scope_key')}::{row.get('package_name')}::{row.get('baseline_family')}"
                if str(row.get("baseline_family")) in {"official_fno", "official_fno_sumlen"}
                else f"leaf128::{index}"
            ),
        )
        for index, row in enumerate(payload_leaf128["family_rows"])
    ]
    payload_leaf16["family_rows"] = [
        _with_v3_row_contract(
            dict(row),
            run_intent_hash=(
                f"{row.get('scope_key')}::{row.get('package_name')}::{row.get('baseline_family')}"
                if str(row.get("baseline_family")) in {"official_fno", "official_fno_sumlen"}
                else f"leaf016::{index}"
            ),
        )
        for index, row in enumerate(payload_leaf16["family_rows"])
    ]

    merged = _merge_supervision_recovery_payloads([payload_leaf128, payload_leaf16])

    tree_rows = [
        row
        for row in merged["family_rows"]
        if row["scope_key"] == "recoverable_v4"
        and row["baseline_family"] == "tree_neural"
    ]
    assert {row["fixed_leaf_tokens"] for row in tree_rows} == {16, 128}
    assert sum(
        1
        for row in merged["family_rows"]
        if row["scope_key"] == "recoverable_v4"
        and row["package_name"] == "full10"
        and row["baseline_family"] == "official_fno"
    ) == 1


def test_markov_optimization_tradeoffs_report_accepts_repeated_supervision_recovery_summaries(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    payload_leaf128 = _focused_supervision_recovery_payload(
        train_docs=[1024],
        include_r10_mass_matched_rates=True,
    )
    payload_leaf16 = _focused_supervision_recovery_payload(
        train_docs=[1024],
        include_r10_mass_matched_rates=True,
    )
    payload_leaf128["family_rows"] = [
        _with_v3_row_contract(
            {
            "scope_key": "recoverable_v4",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno",
            "test_root_mae_mean": 0.08,
            },
            run_intent_hash="recoverable_fno_leaf128",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "recoverable_v4",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno_sumlen",
            "test_root_mae_mean": 0.09,
            },
            run_intent_hash="recoverable_fno_sumlen_leaf128",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "recoverable_v4",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "tree_neural",
            "test_root_mae_mean": 0.082,
            "fixed_leaf_tokens": 128,
            "computed_assumed_doc_tokens": 128,
            "leaves_per_doc": 1,
            "internal_nodes_per_doc": 0,
            "is_fno_equivalent_geometry": True,
            },
            run_intent_hash="recoverable_tree_leaf128",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "r12_seg10to12",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno",
            "test_root_mae_mean": 0.16,
            },
            run_intent_hash="structural_fno_leaf128",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "r12_seg10to12",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno_sumlen",
            "test_root_mae_mean": 0.18,
            },
            run_intent_hash="structural_fno_sumlen_leaf128",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "r12_seg10to12",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "tree_neural",
            "test_root_mae_mean": 0.165,
            "fixed_leaf_tokens": 128,
            "computed_assumed_doc_tokens": 128,
            "leaves_per_doc": 1,
            "internal_nodes_per_doc": 0,
            "is_fno_equivalent_geometry": True,
            },
            run_intent_hash="structural_tree_leaf128",
        ),
    ]
    payload_leaf16["family_rows"] = [
        _with_v3_row_contract(
            {
            "scope_key": "recoverable_v4",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno",
            "test_root_mae_mean": 0.08,
            },
            run_intent_hash="recoverable_fno_leaf016",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "recoverable_v4",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno_sumlen",
            "test_root_mae_mean": 0.09,
            },
            run_intent_hash="recoverable_fno_sumlen_leaf016",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "recoverable_v4",
            "train_doc_count": 1024,
            "package_name": "r10_mass_local_eq_2p0",
            "baseline_family": "tree_neural",
            "test_root_mae_mean": 0.05,
            "fixed_leaf_tokens": 16,
            "computed_assumed_doc_tokens": 128,
            "leaves_per_doc": 8,
            "internal_nodes_per_doc": 7,
            "is_fno_equivalent_geometry": False,
            },
            run_intent_hash="recoverable_tree_leaf016",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "r12_seg10to12",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno",
            "test_root_mae_mean": 0.16,
            },
            run_intent_hash="structural_fno_leaf016",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "r12_seg10to12",
            "train_doc_count": 1024,
            "package_name": "full10",
            "baseline_family": "official_fno_sumlen",
            "test_root_mae_mean": 0.18,
            },
            run_intent_hash="structural_fno_sumlen_leaf016",
        ),
        _with_v3_row_contract(
            {
            "scope_key": "r12_seg10to12",
            "train_doc_count": 1024,
            "package_name": "r10_mass_local_eq_2p0",
            "baseline_family": "tree_neural",
            "test_root_mae_mean": 0.12,
            "fixed_leaf_tokens": 16,
            "computed_assumed_doc_tokens": 128,
            "leaves_per_doc": 8,
            "internal_nodes_per_doc": 7,
            "is_fno_equivalent_geometry": False,
            },
            run_intent_hash="structural_tree_leaf016",
        ),
    ]

    _write_json(inputs / "sr_leaf128.json", payload_leaf128)
    _write_json(inputs / "sr_leaf16.json", payload_leaf16)

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "sr_leaf128.json"),
            "--supervision-recovery-summary",
            str(inputs / "sr_leaf16.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert "Recoverable R10 Leaf Geometry" in summary["figures"]
    assert "Structural R10 Leaf Geometry" in summary["figures"]
    assert "## Recoverable Leaves/Doc at R10" in markdown
    assert "## Structural Leaves/Doc at R10" in markdown
    assert (
        "requested `leaf128` alone is not enough."
    ) in markdown


def test_supervision_recovery_merge_keeps_rerun_lineages_side_by_side() -> None:
    publication = {
        "source_summary_json": (
            "/tmp/outputs/markov_v3_publication_fullval_20260412_0142/"
            "root_budget_publication_multileaf_fullval/supervision_recovery/summary.json"
        ),
        "package_order": ["full100"],
        "train_doc_counts": [10240],
        "family_rows": [
            _with_v3_row_contract(
                {
                    "scope_key": "r12_seg10to12",
                    "train_doc_count": 10240,
                    "package_name": "full100",
                    "baseline_family": "tree_neural",
                    "tree_test_root_mae": 0.0187,
                    "test_root_mae_mean": 0.0187,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "fixed_leaf_tokens": 64,
                    "computed_assumed_doc_tokens": 128,
                    "leaves_per_doc": 2,
                },
                run_intent_hash="structural_full100_leaf064",
                requested_fixed_leaf_tokens=64,
                executed_fixed_leaf_tokens=64,
            )
        ],
        "scopes": {
            "r12_seg10to12": {
                "scope_label": "structural_core_v1::r12_seg10to12",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            _with_v3_row_contract(
                                {
                                    "package_name": "full100",
                                    "baseline_family": "tree_neural",
                                    "tree_test_root_mae": 0.0187,
                                    "fno_family_rows": {},
                                    "tree_reference_label": "unified_g_full_local_laws_v1",
                                    "requested_fixed_leaf_tokens": 64,
                                    "executed_fixed_leaf_tokens": 64,
                                    "fixed_leaf_tokens": 64,
                                    "computed_assumed_doc_tokens": 128,
                                    "leaves_per_doc": 2,
                                },
                                run_intent_hash="structural_full100_leaf064",
                                requested_fixed_leaf_tokens=64,
                                executed_fixed_leaf_tokens=64,
                            )
                        ],
                    }
                ],
            }
        },
    }
    exploratory = {
        "source_summary_json": (
            "/tmp/outputs/markov_v3_overnight_fill_20260412_0454/"
            "multileaf_root_only/supervision_recovery/summary.json"
        ),
        "package_order": ["full100"],
        "train_doc_counts": [10240],
        "family_rows": [
            _with_v3_row_contract(
                {
                    "scope_key": "r12_seg10to12",
                    "train_doc_count": 10240,
                    "package_name": "full100",
                    "baseline_family": "tree_neural",
                    "tree_test_root_mae": 0.6877,
                    "test_root_mae_mean": 0.6877,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "fixed_leaf_tokens": 64,
                    "computed_assumed_doc_tokens": 128,
                    "leaves_per_doc": 2,
                },
                run_intent_hash="structural_full100_leaf064",
                requested_fixed_leaf_tokens=64,
                executed_fixed_leaf_tokens=64,
            )
        ],
        "scopes": {
            "r12_seg10to12": {
                "scope_label": "structural_core_v1::r12_seg10to12",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            _with_v3_row_contract(
                                {
                                    "package_name": "full100",
                                    "baseline_family": "tree_neural",
                                    "tree_test_root_mae": 0.6877,
                                    "fno_family_rows": {},
                                    "tree_reference_label": "unified_g_full_local_laws_v1",
                                    "requested_fixed_leaf_tokens": 64,
                                    "executed_fixed_leaf_tokens": 64,
                                    "fixed_leaf_tokens": 64,
                                    "computed_assumed_doc_tokens": 128,
                                    "leaves_per_doc": 2,
                                },
                                run_intent_hash="structural_full100_leaf064",
                                requested_fixed_leaf_tokens=64,
                                executed_fixed_leaf_tokens=64,
                            )
                        ],
                    }
                ],
            }
        },
    }

    merged = _merge_supervision_recovery_payloads([exploratory, publication])
    summary = _summarize_supervision_recovery(
        merged,
        expected_train_doc_counts=[10240],
        expected_package_order=["full100"],
        expected_structural_cell="r12_seg10to12",
    )

    rows = summary["scopes"]["r12_seg10to12"]["rows_by_train_docs"]["10240"]["rows"]
    assert len(rows) == 2
    assert sorted(row["tree_test_root_mae"] for row in rows) == pytest.approx(
        [0.0187, 0.6877]
    )
    assert summary["duplicate_resolution"] == []
    ordered_payloads = _ordered_family_payloads(summary, scope_key="r12_seg10to12")
    assert len(ordered_payloads[0]["tree_root_only_series"]) == 2
    assert sorted(
        list(series["tree_test_root_mae"])[-1]
        for series in ordered_payloads[0]["tree_root_only_series"]
    ) == pytest.approx([0.0187, 0.6877])


def test_ordered_family_payloads_collapse_identical_duplicate_lineages() -> None:
    def _root_only_payload(source_summary_json: str, full100: float, full90: float) -> dict[str, object]:
        return {
            "source_summary_json": source_summary_json,
            "package_order": ["full100", "full90"],
            "train_doc_counts": [10240],
            "family_rows": [],
            "scopes": {
                "recoverable_v4": {
                    "scope_label": "recoverable_v4",
                    "rows_by_train_docs": [
                        {
                            "train_doc_count": 10240,
                            "rows": [
                                _with_v3_row_contract(
                                    {
                                        "package_name": "full100",
                                        "baseline_family": "tree_neural",
                                        "tree_test_root_mae": full100,
                                        "fno_family_rows": {},
                                        "tree_reference_label": "unified_g_full_local_laws_v1",
                                        "requested_fixed_leaf_tokens": 64,
                                        "executed_fixed_leaf_tokens": 64,
                                        "fixed_leaf_tokens": 64,
                                        "computed_assumed_doc_tokens": 128,
                                        "leaves_per_doc": 2,
                                    },
                                    run_intent_hash="recoverable_full100_leaf064",
                                    requested_fixed_leaf_tokens=64,
                                    executed_fixed_leaf_tokens=64,
                                ),
                                _with_v3_row_contract(
                                    {
                                        "package_name": "full90",
                                        "baseline_family": "tree_neural",
                                        "tree_test_root_mae": full90,
                                        "fno_family_rows": {},
                                        "tree_reference_label": "unified_g_full_local_laws_v1",
                                        "requested_fixed_leaf_tokens": 64,
                                        "executed_fixed_leaf_tokens": 64,
                                        "fixed_leaf_tokens": 64,
                                        "computed_assumed_doc_tokens": 128,
                                        "leaves_per_doc": 2,
                                    },
                                    run_intent_hash="recoverable_full90_leaf064",
                                    requested_fixed_leaf_tokens=64,
                                    executed_fixed_leaf_tokens=64,
                                ),
                            ],
                        }
                    ],
                }
            },
        }

    merged = _merge_supervision_recovery_payloads(
        [
            _root_only_payload(
                "/tmp/outputs/markov_v3_depth_redistribution_large_train_stable_20260411_084653/"
                "root_budget_ladder_large_train/supervision_recovery/summary.json",
                0.0204,
                0.0305,
            ),
            _root_only_payload(
                "/tmp/outputs/markov_v3_depth_redistribution_large_train_stable_20260411_084653/"
                "mass_preserving_leaf_only_large_train/supervision_recovery/summary.json",
                0.0204,
                0.0305,
            ),
            _root_only_payload(
                "/tmp/outputs/markov_v3_publication_fullval_20260412_0142/"
                "root_budget_publication_multileaf_fullval/supervision_recovery/summary.json",
                0.0217,
                0.0321,
            ),
        ]
    )
    summary = _summarize_supervision_recovery(
        merged,
        expected_train_doc_counts=[10240],
        expected_package_order=["full100", "full90"],
        expected_structural_cell="r12_seg10to12",
    )
    ordered_payloads = _ordered_family_payloads(summary, scope_key="recoverable_v4")
    assert len(ordered_payloads) == 1
    series = ordered_payloads[0]["tree_root_only_series"]
    assert len(series) == 2
    assert any("(+1 matching bundles)" in str(item.get("lineage_label", "")) for item in series)


def test_ordered_family_payloads_use_canonical_train_doc_counts_only() -> None:
    merged = _merge_supervision_recovery_payloads(
        [
            _focused_supervision_recovery_payload(train_docs=[1024, 2048, 4096]),
        ]
    )
    summary = _summarize_supervision_recovery(
        merged,
        expected_train_doc_counts=[1024, 4096],
        expected_package_order=list(merged["package_order"]),
        expected_structural_cell="r12_seg10to12",
    )
    ordered_payloads = _ordered_family_payloads(summary, scope_key="recoverable_v4")
    assert [payload["train_doc_count"] for payload in ordered_payloads] == [1024, 4096]


def test_supervision_recovery_merge_dedupes_exact_duplicate_summary_ingestion() -> None:
    publication = {
        "source_summary_json": (
            "/tmp/outputs/markov_v3_publication_fullval_20260412_0142/"
            "root_budget_publication_multileaf_fullval/supervision_recovery/summary.json"
        ),
        "package_order": ["full100"],
        "train_doc_counts": [10240],
        "family_rows": [
            _with_v3_row_contract(
                {
                    "scope_key": "recoverable_v4",
                    "train_doc_count": 10240,
                    "package_name": "full100",
                    "baseline_family": "tree_neural",
                    "tree_test_root_mae": 0.0187,
                    "test_root_mae_mean": 0.0187,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "fixed_leaf_tokens": 64,
                    "computed_assumed_doc_tokens": 128,
                    "leaves_per_doc": 2,
                },
                run_intent_hash="recoverable_full100_leaf064",
                requested_fixed_leaf_tokens=64,
                executed_fixed_leaf_tokens=64,
            )
        ],
        "scopes": {
            "recoverable_v4": {
                "scope_label": "recoverable_v4",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            _with_v3_row_contract(
                                {
                                    "package_name": "full100",
                                    "baseline_family": "tree_neural",
                                    "tree_test_root_mae": 0.0187,
                                    "fno_family_rows": {},
                                    "tree_reference_label": "unified_g_full_local_laws_v1",
                                    "requested_fixed_leaf_tokens": 64,
                                    "executed_fixed_leaf_tokens": 64,
                                    "fixed_leaf_tokens": 64,
                                    "computed_assumed_doc_tokens": 128,
                                    "leaves_per_doc": 2,
                                },
                                run_intent_hash="recoverable_full100_leaf064",
                                requested_fixed_leaf_tokens=64,
                                executed_fixed_leaf_tokens=64,
                            )
                        ],
                    }
                ],
            }
        },
    }

    merged = _merge_supervision_recovery_payloads([publication, publication])
    summary = _summarize_supervision_recovery(
        merged,
        expected_train_doc_counts=[10240],
        expected_package_order=["full100"],
    )

    rows = summary["scopes"]["recoverable_v4"]["rows_by_train_docs"]["10240"]["rows"]
    assert len(rows) == 1
    assert rows[0]["tree_test_root_mae"] == pytest.approx(0.0187)
    assert len(summary["duplicate_resolution"]) == 2


def test_supervision_recovery_summary_excludes_structural_one_leaf_partial_root_rescue_rows() -> None:
    payload = {
        "status": "ready",
        "tree_family": "tree_neural",
        "package_order": ["full90"],
        "family_rows": [
            _with_v3_row_contract(
                {
                    "scope_key": "r12_seg10to12",
                    "train_doc_count": 10240,
                    "package_name": "full90",
                    "baseline_family": "tree_neural",
                    "tree_test_root_mae": 0.677,
                    "test_root_mae_mean": 0.677,
                    "tree_reference_label": "structural_root_only_parity_matched_root_v3",
                    "fixed_leaf_tokens": 128,
                    "computed_assumed_doc_tokens": 128,
                    "leaves_per_doc": 1,
                },
                run_intent_hash="structural_oneleaf_partial_root",
            )
        ],
        "scopes": {
            "r12_seg10to12": {
                "scope_label": "structural_core_v1::r12_seg10to12",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            _with_v3_row_contract(
                                {
                                    "package_name": "full90",
                                    "scope_key": "r12_seg10to12",
                                    "baseline_family": "tree_neural",
                                    "tree_test_root_mae": 0.677,
                                    "fno_family_rows": {},
                                    "tree_reference_label": "structural_root_only_parity_matched_root_v3",
                                    "requested_fixed_leaf_tokens": 128,
                                    "executed_fixed_leaf_tokens": 128,
                                    "fixed_leaf_tokens": 128,
                                    "computed_assumed_doc_tokens": 128,
                                    "leaves_per_doc": 1,
                                },
                                run_intent_hash="structural_oneleaf_partial_root",
                            )
                        ],
                    }
                ],
            }
        },
    }

    summary = _summarize_supervision_recovery(
        payload,
        expected_train_doc_counts=[10240],
        expected_package_order=["full90"],
        expected_structural_cell="r12_seg10to12",
    )

    assert summary["scopes"]["r12_seg10to12"]["rows_by_train_docs"]["10240"]["rows"] == []
    assert summary["quarantined_scope_rows"]
    assert summary["hidden_invalid_row_count"] >= 1
    assert "structural_one_leaf_partial_root_rescue_pending" in summary["hidden_invalid_reasons"]
    assert (
        summary["quarantined_scope_rows"][0]["contract_diagnostic_reasons"]
        == ["structural_one_leaf_partial_root_rescue_pending"]
    )


def test_supervision_recovery_summary_accepts_mapping_rows_by_train_docs() -> None:
    payload = {
        "status": "ready",
        "tree_family": "tree_neural",
        "package_order": ["full100"],
        "family_rows": [
            _with_v3_row_contract(
                {
                    "scope_key": "r12_seg10to12",
                    "train_doc_count": 10240,
                    "package_name": "full100",
                    "baseline_family": "tree_neural",
                    "tree_test_root_mae": 0.0187,
                    "test_root_mae_mean": 0.0187,
                    "tree_reference_label": "unified_g_full_local_laws_v1",
                    "fixed_leaf_tokens": 64,
                    "computed_assumed_doc_tokens": 128,
                    "leaves_per_doc": 2,
                },
                run_intent_hash="mapping_rows_by_train_docs",
                requested_fixed_leaf_tokens=64,
                executed_fixed_leaf_tokens=64,
            )
        ],
        "scopes": {
            "r12_seg10to12": {
                "scope_label": "structural_core_v1::r12_seg10to12",
                "rows_by_train_docs": {
                    "10240": {
                        "train_doc_count": 10240,
                        "rows": [
                            _with_v3_row_contract(
                                {
                                    "package_name": "full100",
                                    "scope_key": "r12_seg10to12",
                                    "baseline_family": "tree_neural",
                                    "tree_test_root_mae": 0.0187,
                                    "fno_family_rows": {},
                                    "tree_reference_label": "unified_g_full_local_laws_v1",
                                    "requested_fixed_leaf_tokens": 64,
                                    "executed_fixed_leaf_tokens": 64,
                                    "fixed_leaf_tokens": 64,
                                    "computed_assumed_doc_tokens": 128,
                                    "leaves_per_doc": 2,
                                },
                                run_intent_hash="mapping_rows_by_train_docs",
                                requested_fixed_leaf_tokens=64,
                                executed_fixed_leaf_tokens=64,
                            )
                        ],
                    }
                },
            }
        },
    }

    summary = _summarize_supervision_recovery(
        payload,
        expected_train_doc_counts=[10240],
        expected_package_order=["full100"],
        expected_structural_cell="r12_seg10to12",
    )

    rows = summary["scopes"]["r12_seg10to12"]["rows_by_train_docs"]["10240"]["rows"]
    assert len(rows) == 1
    assert rows[0]["tree_test_root_mae"] == pytest.approx(0.0187)


def test_leaf_geometry_helpers_recover_geometry_from_canary_style_rows() -> None:
    row = {
        "baseline_family": "tree_neural",
        "test_root_mae_mean": 0.133,
        "fixed_leaf_tokens": 16,
        "computed_assumed_doc_tokens": 128,
        "leaves_per_doc": 1,
        "is_fno_equivalent_geometry": True,
    }

    assert _effective_leaves_per_doc(row) == 1
    assert _effective_fixed_leaf_tokens(row) == 16
    assert _tree_root_mae_from_family_row(row) == 0.133
    assert _is_exact_full_doc_parity_row(row) is False


def test_leaf_geometry_warning_lines_surface_requested_vs_executed_mismatch() -> None:
    summary = {
        "supervision_recovery": {
            "status": "ready",
            "tree_family": "tree_neural",
            "family_rows": [
                {
                    "scope_key": "recoverable_v4",
                    "train_doc_count": 4096,
                    "package_name": "full100",
                    "baseline_family": "tree_neural",
                    "requested_fixed_leaf_tokens": 128,
                    "fixed_leaf_tokens": 16,
                    "computed_assumed_doc_tokens": 128,
                    "leaves_per_doc": 6,
                }
            ],
        }
    }

    lines = _leaf_geometry_warning_lines(summary)

    assert len(lines) == 1
    assert "requested `fixed_leaf_tokens=128`" in lines[0]
    assert "`6 leaves/doc`" in lines[0]


def test_markov_optimization_tradeoffs_report_auto_splits_multi_geometry_default_profile(
    tmp_path: Path,
) -> None:
    def _retarget_tree_geometry(
        payload: dict[str, object],
        *,
        leaf_tokens: int,
        leaves_per_doc: int,
    ) -> dict[str, object]:
        updated = json.loads(json.dumps(payload))
        geometry_label = f"leaf{leaf_tokens:03d}"

        def _apply_tree_geometry(row: dict[str, object]) -> None:
            row["requested_fixed_leaf_tokens"] = leaf_tokens
            row["fixed_leaf_tokens"] = leaf_tokens
            row["executed_fixed_leaf_tokens"] = leaf_tokens
            row["computed_assumed_doc_tokens"] = 128
            row["leaves_per_doc"] = leaves_per_doc
            row["executed_leaves_per_doc"] = leaves_per_doc
            row["supervision_recovery_geometry_key"] = geometry_label
            row["supervision_recovery_geometry_label"] = geometry_label

        for reference in dict(updated.get("scope_tree_references") or {}).values():
            if isinstance(reference, dict):
                reference["fixed_leaf_tokens"] = leaf_tokens
                reference["requested_fixed_leaf_tokens"] = leaf_tokens
                reference["executed_fixed_leaf_tokens"] = leaf_tokens
                reference["executed_leaves_per_doc"] = leaves_per_doc
                reference["supervision_recovery_geometry_key"] = geometry_label
                reference["supervision_recovery_geometry_label"] = geometry_label
        for scope in dict(updated.get("scopes") or {}).values():
            if not isinstance(scope, dict):
                continue
            for row_group in list(scope.get("rows_by_train_docs") or []):
                if not isinstance(row_group, dict):
                    continue
                for row in list(row_group.get("rows") or []):
                    if isinstance(row, dict):
                        _apply_tree_geometry(row)
            for row in list(scope.get("dense_anchor_rows") or []):
                if isinstance(row, dict):
                    _apply_tree_geometry(row)
            for row in dict(scope.get("best_tree_by_train_docs") or {}).values():
                if isinstance(row, dict):
                    _apply_tree_geometry(row)
        for row in list(updated.get("best_tree_summary") or []):
            if isinstance(row, dict):
                _apply_tree_geometry(row)
        family_rows: list[dict[str, object]] = []
        for scope_key, scope in dict(updated.get("scopes") or {}).items():
            if not isinstance(scope, dict):
                continue
            scope_label = str(scope.get("scope_label", scope_key) or scope_key)
            for row_group in list(scope.get("rows_by_train_docs") or []):
                if not isinstance(row_group, dict):
                    continue
                train_doc_count = int(row_group.get("train_doc_count", 0))
                for row in list(row_group.get("rows") or []):
                    if not isinstance(row, dict):
                        continue
                    tree_row = {
                        "scope_key": str(scope_key),
                        "scope_label": scope_label,
                        "train_doc_count": train_doc_count,
                        "package_name": str(row.get("package_name", "") or ""),
                        "baseline_family": "tree_neural",
                        "test_root_mae_mean": float(row.get("tree_test_root_mae", 0.0)),
                        "tree_supervision_source": "manifest",
                        "local_estimand_mode": "span_mass_ipw_sum",
                        "c2_pair_weighting_mode": "pair_ipw_geomean",
                        "n_runs": 1,
                    }
                    _apply_tree_geometry(tree_row)
                    family_rows.append(
                        _with_v3_row_contract(
                            tree_row,
                            run_intent_hash=(
                                f"{scope_key}::{train_doc_count}::"
                                f"{row.get('package_name', '')}::tree_neural::"
                                f"{geometry_label}"
                            ),
                        )
                    )
                    for family in ("official_fno", "official_fno_sumlen"):
                        family_rows.append(
                            _with_v3_row_contract(
                                {
                                    "scope_key": str(scope_key),
                                    "scope_label": scope_label,
                                    "train_doc_count": train_doc_count,
                                    "package_name": str(row.get("package_name", "") or ""),
                                    "baseline_family": family,
                                    "test_root_mae_mean": float(
                                        row.get("fno_reference_test_root_mae", 0.0)
                                    ),
                                    "n_runs": 1,
                                },
                                run_intent_hash=(
                                    f"{scope_key}::{train_doc_count}::"
                                    f"{row.get('package_name', '')}::{family}"
                                ),
                            )
                        )
        updated["family_rows"] = family_rows
        return updated

    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    geometry_a = _retarget_tree_geometry(
        _focused_supervision_recovery_payload(),
        leaf_tokens=16,
        leaves_per_doc=8,
    )
    geometry_b = _retarget_tree_geometry(
        _focused_supervision_recovery_payload(),
        leaf_tokens=32,
        leaves_per_doc=4,
    )
    merged_payload = _merge_supervision_recovery_payloads([geometry_a, geometry_b])
    _write_json(inputs / "supervision_recovery.json", merged_payload)

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    geometry_labels = [
        group["geometry_label"]
        for group in summary["supervision_recovery"]["geometry_groups"]
    ]
    assert geometry_labels == ["leaf032", "leaf016"]
    assert "Recoverable Ordered Families (leaf016, 8 leaves/doc)" in summary["figures"]
    assert "Recoverable Ordered Families (leaf032, 4 leaves/doc)" in summary["figures"]
    assert "Structural Ordered Families (leaf016, 8 leaves/doc)" in summary["figures"]
    assert "Structural Ordered Families (leaf032, 4 leaves/doc)" in summary["figures"]
    assert "## Geometry Group (leaf016, 8 leaves/doc)" in markdown
    assert "## Geometry Group (leaf032, 4 leaves/doc)" in markdown
    assert "### Recoverable Root-Supervision Sweep (leaf016, 8 leaves/doc)" in markdown
    assert "### Recoverable Root-Supervision Sweep (leaf032, 4 leaves/doc)" in markdown
    assert "### Structural Root-Supervision Sweep (leaf016, 8 leaves/doc)" in markdown
    assert "### Structural Root-Supervision Sweep (leaf032, 4 leaves/doc)" in markdown
    assert (
        summary["figures"]["Recoverable Ordered Families (leaf016, 8 leaves/doc)"]
        != summary["figures"]["Recoverable Ordered Families (leaf032, 4 leaves/doc)"]
    )


def test_markov_optimization_tradeoffs_report_r10_coverage_focused_profile(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _focused_supervision_recovery_payload(include_r10_local_law_rates=True),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--report-profile",
            "r10_coverage_focused",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert summary["report_focus"] == "r10_coverage_focused"
    assert "Recoverable Ordered Families" in summary["figures"]
    assert "Structural Ordered Families" in summary["figures"]
    assert "Recoverable R10 Local-Law Coverage" in summary["figures"]
    assert "Structural R10 Local-Law Coverage" in summary["figures"]
    assert "Recoverable R10 Local Ablations" in summary["figures"]
    assert "Structural R10 Local Ablations" in summary["figures"]
    assert "Recoverable R20 Local-Law Coverage" not in summary["figures"]
    assert "Structural R20 Local-Law Coverage" not in summary["figures"]
    assert "Dense Full-Doc Anchor" not in summary["figures"]
    assert "Recoverable Dense-Local Root Ladder" not in summary["figures"]
    assert "Structural Dense-Local Root Ladder" not in summary["figures"]
    assert "Recoverable Package Ladder" not in summary["figures"]
    assert "Structural Package Ladder" not in summary["figures"]
    assert "Recoverable Tree Diagnostics" not in summary["figures"]
    assert "Structural Tree Diagnostics" not in summary["figures"]

    assert "## Recoverable Extra Count Labels at R10" in markdown
    assert "## Structural Extra Count Labels at R10" in markdown
    assert "## Recoverable Full-Supervision Reference" in markdown
    assert "## Structural Full-Supervision Reference" in markdown
    assert "## Recoverable R10 Endpoints" in markdown
    assert "## Structural R10 Endpoints" in markdown
    assert "\n### Recoverable What Extra Tree Labels Help at R10?" in markdown
    assert "\n### Structural What Extra Tree Labels Help at R10?" in markdown
    assert "\n## Recoverable What Extra Tree Labels Help at R10?" not in markdown
    assert "\n## Structural What Extra Tree Labels Help at R10?" not in markdown
    assert "Recoverable R20 Local-Law Coverage" not in markdown
    assert "Structural R20 Local-Law Coverage" not in markdown
    assert "R0+Lf+Ia" not in markdown
    assert "`0%` appears only" not in markdown
    assert "same `10%` root supervision budget" in markdown
    assert "`R100` means full root/doc supervision" in markdown
    assert "root-labeled docs" in markdown
    assert "dotted benchmark line" in markdown
    assert "training-doc equivalents" in markdown


def test_markov_optimization_tradeoffs_report_auto_compacts_exact_parity_canary(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    _write_json(
        inputs / "supervision_recovery.json",
        _exact_full_doc_canary_payload(),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    pdf_path = output_dir / "report.pdf"

    assert summary["report_focus"] == "exact_parity_canary"
    assert set(summary["figures"]) == {
        "Recoverable Exact Full-Doc Canary",
        "Structural Exact Full-Doc Canary",
    }
    assert pdf_path.exists()

    assert "# Markov Exact Full-Doc Canary Report" in markdown
    assert "## What This Covers" in markdown
    assert "## Setup" in markdown
    assert "## Recoverable Exact Full-Doc Canary" in markdown
    assert "## Structural Exact Full-Doc Canary" in markdown
    assert "## Parity Summary" in markdown
    assert "## Runtime Notes" in markdown
    assert "Package set: `full100` only." in markdown
    assert "exact full-doc parity at `1 leaf/doc`" in markdown
    assert "## Recoverable Root-Supervision Sweep" not in markdown
    assert "## Structural Root-Supervision Sweep" not in markdown
    assert "## Recoverable All Supervision Settings" not in markdown
    assert "## Key Concepts" not in markdown


def test_supervision_recovery_report_uses_official_palette() -> None:
    assert TREE_PRIMARY_COLOR == "#16a34a"
    assert TREE_LOCAL_COLOR == "#16a34a"
    assert FNO_OFFICIAL_COLOR == "#dc2626"
    assert FNO_SUMLEN_COLOR == "#f59e0b"
    assert BEST_FULL_ROOT_CEILING_COLOR == NEUTRAL_COLOR


def test_supervision_recovery_full_root_ceiling_uses_only_full_root_rows() -> None:
    payload = _focused_supervision_recovery_payload(include_r10_local_law_rates=True)
    recovery = _summarize_supervision_recovery(
        payload,
        expected_train_doc_counts=[1024, 2048, 4096],
        expected_structural_cell="r12_seg10to12",
    )
    ceiling = _best_full_root_root_mae_by_train_docs(recovery, scope_key="recoverable_v4")

    assert ceiling[1024] == 0.10
    assert ceiling[2048] == 0.06
    assert ceiling[4096] == 0.04


def test_supervision_recovery_full_root_ceiling_falls_back_to_ceiling_recovery() -> None:
    primary_payload = _focused_supervision_recovery_payload(include_r10_local_law_rates=True)
    primary_recovery = _summarize_supervision_recovery(
        primary_payload,
        expected_train_doc_counts=[1024, 2048, 4096],
        expected_structural_cell="r12_seg10to12",
    )
    ceiling_payload = _focused_supervision_recovery_payload()
    ceiling_recovery = _summarize_supervision_recovery(
        ceiling_payload,
        expected_train_doc_counts=[1024, 2048, 4096],
        expected_structural_cell="r12_seg10to12",
    )
    combined_recovery = {**primary_recovery, "ceiling_recovery": ceiling_recovery}

    ceiling = _best_full_root_root_mae_by_train_docs(combined_recovery, scope_key="recoverable_v4")

    assert ceiling[1024] == 0.10
    assert ceiling[2048] == 0.06
    assert ceiling[4096] == 0.04


def test_markov_supervision_recovery_paper_audit_smoke(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    report_dir = tmp_path / "report"
    audit_dir = tmp_path / "audit"
    _write_json(
        inputs / "supervision_recovery.json",
        _focused_supervision_recovery_payload(),
    )

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--output-dir",
            str(report_dir),
        ],
        cwd=repo_root,
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_supervision_recovery_paper_audit.py",
            "--supervision-recovery-summary",
            str(inputs / "supervision_recovery.json"),
            "--report-summary",
            str(report_dir / "summary.json"),
            "--output-dir",
            str(audit_dir),
        ],
        cwd=repo_root,
    )

    audit = json.loads((audit_dir / "paper_audit.json").read_text(encoding="utf-8"))
    markdown = (audit_dir / "paper_audit.md").read_text(encoding="utf-8")
    pdf_path = audit_dir / "paper_audit.pdf"

    assert audit["status"] == "ready"
    assert audit["canonical_status"]["common_tree_reference_label"] == "common_factorized_sketch_v1"
    assert audit["canonical_status"]["canonical_tree_selection_metric"] == "val_root_mae"
    assert audit["publication_readiness"]["status"] == "close_for_paper_draft"
    assert audit["missing_grid_points"]["required_now"] == []
    assert any(
        row["title"] == "Recoverable Ordered Families"
        for row in audit["main_text_figures"]
    )
    assert any(
        row["title"] == "Recoverable Package Ladder"
        for row in audit["appendix_figures"]
    )
    assert "## Figure Plan" in markdown
    assert "## Publication Readiness" in markdown
    assert "## Bridge To LLMs And Preferences" in markdown
    assert pdf_path.exists()


def test_supervision_report_tie_break_uses_profile_order() -> None:
    payload = {
        "rows": [
            {
                "train_doc_count": 1024,
                "leaf_profile": "count_q100",
                "internal_profile": "count_q100",
                "leaf_supervision_kind": "count_only",
                "leaf_label_rate": 1.0,
                "internal_supervision_kind": "count_only",
                "internal_label_rate": 1.0,
                "mean_test_root_mae": 0.0,
            },
            {
                "train_doc_count": 1024,
                "leaf_profile": "none",
                "internal_profile": "none",
                "leaf_supervision_kind": "count_only",
                "leaf_label_rate": 0.0,
                "internal_supervision_kind": "none",
                "internal_label_rate": 0.0,
                "mean_test_root_mae": 0.0,
            },
        ],
        "by_train_docs": {
            "1024": {
                "rows": [
                    {
                        "train_doc_count": 1024,
                        "leaf_profile": "count_q100",
                        "internal_profile": "count_q100",
                        "leaf_supervision_kind": "count_only",
                        "leaf_label_rate": 1.0,
                        "internal_supervision_kind": "count_only",
                        "internal_label_rate": 1.0,
                        "mean_test_root_mae": 0.0,
                    },
                    {
                        "train_doc_count": 1024,
                        "leaf_profile": "none",
                        "internal_profile": "none",
                        "leaf_supervision_kind": "count_only",
                        "leaf_label_rate": 0.0,
                        "internal_supervision_kind": "none",
                        "internal_label_rate": 0.0,
                        "mean_test_root_mae": 0.0,
                    },
                ]
            }
        },
        "leaf_profiles": ["none", "count_q100"],
        "internal_profiles": ["none", "count_q100"],
    }
    summary = _summarize_supervision_sweep(payload)
    assert summary["best_overall"]["leaf_profile"] == "none"
    assert summary["best_by_train_docs"]["1024"]["internal_profile"] == "none"


def test_supervision_recovery_summary_accepts_reordered_package_contract_as_payload_order() -> None:
    payload = _focused_supervision_recovery_payload(
        package_order=[
            "full50",
            "full30",
            "full20",
            "full10",
            "full100",
            "full10_leaf_count100",
            "full10_leaf_full100",
            "full10_leaf_full100_internal_depth1_count100",
            "full10_leaf_full100_internal_depth2_count100",
            "full10_leaf_full100_internal_count100",
            "full20_leaf_full100_internal_count100",
            "full30_leaf_full100_internal_count100",
            "full50_leaf_full100_internal_count100",
        ]
    )
    summary = _summarize_supervision_recovery(
        payload,
        expected_package_order=[
            "full100",
            "full50",
            "full30",
            "full20",
            "full10",
            "full10_leaf_count100",
            "full10_leaf_full100",
            "full10_leaf_full100_internal_depth1_count100",
            "full10_leaf_full100_internal_depth2_count100",
            "full10_leaf_full100_internal_count100",
            "full20_leaf_full100_internal_count100",
            "full30_leaf_full100_internal_count100",
            "full50_leaf_full100_internal_count100",
        ],
    )
    assert summary["status"] == "ready"
    assert summary["package_order"][0] == "full50"
    assert any("payload package order differs" in notice for notice in summary["notices"])


def test_supervision_recovery_summary_tracks_missing_train_docs() -> None:
    payload = _focused_supervision_recovery_payload(train_docs=[1024, 4096])
    summary = _summarize_supervision_recovery(
        payload,
        expected_train_doc_counts=[1024, 2048, 4096],
    )
    assert summary["status"] == "ready"
    assert summary["scopes"]["recoverable_v4"]["missing_train_docs"] == [2048]
    assert any("missing train-doc counts [2048]" in notice for notice in summary["notices"])


def test_supervision_recovery_summary_accepts_custom_package_order_from_payload() -> None:
    payload = {
        "status": "ready",
        "tree_family": "tree_neural",
        "package_order": ["full100", "full100_leaf_full100_internal_count100"],
        "scopes": {
            "r12_seg10to12": {
                "scope_label": "structural_core_v1::r12_seg10to12",
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            {
                                "package_name": "full100",
                                "tree_test_root_mae": 0.10,
                                "fno_reference_package": "full100",
                                "fno_reference_family": "official_fno_sumlen",
                                "fno_reference_test_root_mae": 0.05,
                            },
                            {
                                "package_name": "full100_leaf_full100_internal_count100",
                                "tree_test_root_mae": 0.06,
                                "fno_reference_package": "full100",
                                "fno_reference_family": "official_fno_sumlen",
                                "fno_reference_test_root_mae": 0.05,
                            },
                        ],
                    }
                ],
            }
        },
    }
    summary = _summarize_supervision_recovery(
        payload,
        expected_train_doc_counts=[10240],
        expected_structural_cell="r12_seg10to12",
    )
    assert summary["status"] == "ready"
    assert summary["package_order"] == ["full100", "full100_leaf_full100_internal_count100"]
    assert summary["expected_package_order"] == ["full100", "full100_leaf_full100_internal_count100"]


def test_supervision_recovery_summary_preserves_separate_fno_families() -> None:
    payload = {
        "status": "ready",
        "tree_family": "tree_neural",
        "package_order": ["full100"],
        "family_rows": [
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 10240,
                "package_name": "full100",
                "baseline_family": "official_fno",
                "test_root_mae_mean": 0.01,
                "n_runs": 2,
            },
            {
                "scope_key": "recoverable_v4",
                "train_doc_count": 10240,
                "package_name": "full100",
                "baseline_family": "official_fno_sumlen",
                "test_root_mae_mean": 0.02,
                "n_runs": 2,
            },
        ],
        "scopes": {
            "recoverable_v4": {
                "scope_label": "recoverable_v4",
                "dense_anchor_rows": [
                    {
                        "train_doc_count": 10240,
                        "package_name": "full100",
                        "tree_test_root_mae": 0.03,
                        "fno_reference_family": "official_fno",
                        "fno_reference_test_root_mae": 0.01,
                    }
                ],
                "rows_by_train_docs": [
                    {
                        "train_doc_count": 10240,
                        "rows": [
                            {
                                "package_name": "full100",
                                "tree_test_root_mae": 0.03,
                                "fno_reference_family": "official_fno",
                                "fno_reference_test_root_mae": 0.01,
                            }
                        ],
                    }
                ],
            }
        },
    }
    summary = _summarize_supervision_recovery(
        payload,
        expected_train_doc_counts=[10240],
    )
    row = summary["scopes"]["recoverable_v4"]["rows_by_train_docs"]["10240"]["rows"][0]
    assert row["fno_family_rows"]["official_fno"]["test_root_mae"] == 0.01
    assert row["fno_family_rows"]["official_fno_sumlen"]["test_root_mae"] == 0.02
    dense_row = summary["scopes"]["recoverable_v4"]["dense_anchor_rows"][0]
    assert dense_row["fno_family_rows"]["official_fno"]["test_root_mae"] == 0.01
    assert dense_row["fno_family_rows"]["official_fno_sumlen"]["test_root_mae"] == 0.02


def test_best_tree_summary_rows_keep_fno_families_separate() -> None:
    recovery = {
        "family_rows": [
            {
                "scope_key": "recoverable_v4_t128",
                "train_doc_count": 1024,
                "package_name": "full100",
                "baseline_family": "official_fno",
                "test_root_mae_mean": 0.30,
                "n_runs": 1,
            },
            {
                "scope_key": "recoverable_v4_t128",
                "train_doc_count": 1024,
                "package_name": "full100",
                "baseline_family": "official_fno_sumlen",
                "test_root_mae_mean": 0.20,
                "n_runs": 1,
            },
            {
                "scope_key": "recoverable_v4_t128",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "official_fno",
                "test_root_mae_mean": 0.40,
                "n_runs": 1,
            },
            {
                "scope_key": "recoverable_v4_t128",
                "train_doc_count": 1024,
                "package_name": "full10",
                "baseline_family": "official_fno_sumlen",
                "test_root_mae_mean": 0.35,
                "n_runs": 1,
            },
        ],
        "best_tree_summary": [
            {
                "scope_key": "recoverable_v4_t128",
                "scope_label": "recoverable_v4_t128",
                "train_doc_count": 1024,
                "package_name": "full10_leaf_full100_internal_count100",
                "fno_reference_package": "full10",
                "tree_test_root_mae": 0.33,
            }
        ],
    }
    rows = _best_tree_summary_rows(recovery)
    assert len(rows) == 1
    row = rows[0]
    assert row["matched_fno_family_rows"]["official_fno"]["test_root_mae"] == 0.40
    assert row["matched_fno_family_rows"]["official_fno"]["delta_vs_tree"] == 0.33 - 0.40
    assert row["matched_fno_family_rows"]["official_fno_sumlen"]["test_root_mae"] == 0.35
    assert row["matched_fno_family_rows"]["official_fno_sumlen"]["delta_vs_tree"] == 0.33 - 0.35
    assert row["full100_fno_family_rows"]["official_fno"]["test_root_mae"] == 0.30
    assert row["full100_fno_family_rows"]["official_fno_sumlen"]["test_root_mae"] == 0.20
    assert row["best_full100_fno_family"] == "official_fno_sumlen"
    assert row["best_full100_fno_test_root_mae"] == 0.20
    assert row["delta_vs_best_full100_fno"] == 0.33 - 0.20


def test_supervision_recovery_summary_accepts_superset_package_order_when_core_contract_is_preserved() -> None:
    payload = _focused_supervision_recovery_payload(
        package_order=[
            "full100",
            "full50",
            "full30",
            "full20",
            "full10",
            "full10_leaf_count100",
            "full10_leaf_full100",
            "full10_leaf_full100_internal_depth1_count100",
            "full10_leaf_full100_internal_depth2_count100",
            "full10_leaf_full100_internal_count100",
            "full20_leaf_full100_internal_count100",
            "full30_leaf_full100_internal_count100",
            "full50_leaf_full100_internal_count100",
            "full100_leaf_full100_internal_count100",
        ]
    )
    summary = _summarize_supervision_recovery(
        payload,
        expected_package_order=[
            "full100",
            "full50",
            "full30",
            "full20",
            "full10",
            "full10_leaf_count100",
            "full10_leaf_full100",
            "full10_leaf_full100_internal_depth1_count100",
            "full10_leaf_full100_internal_depth2_count100",
            "full10_leaf_full100_internal_count100",
            "full20_leaf_full100_internal_count100",
            "full30_leaf_full100_internal_count100",
            "full50_leaf_full100_internal_count100",
        ],
    )
    assert summary["status"] == "ready"
    assert summary["package_order"][-1] == "full100_leaf_full100_internal_count100"
    assert summary["expected_package_order"][-1] == "full100_leaf_full100_internal_count100"


def test_markov_optimization_tradeoffs_report_partial_manifest_renders_focus_placeholders(
    tmp_path: Path,
) -> None:
    version_root = tmp_path / "version"
    output_dir = tmp_path / "report"
    repo_root = Path(__file__).resolve().parents[2]
    support_path = version_root / "report_sources" / "support_summary" / "seed" / "support.json"
    _write_json(
        support_path,
        {
            "rows": [
                {"fixed_leaf_tokens": 8, "model_family": "tree_neural", "train_docs": 1024}
            ]
        },
    )
    manifest_path = version_root / "report_version_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-03-26T00:00:00+00:00",
                "version_root": str(version_root),
                "selected_sources": {
                    "support_summary": {
                        "relpath": str(support_path.relative_to(version_root)),
                        "origin": "staged_copy",
                        "phase": "support_grid",
                        "sha256": "",
                        "config_fingerprint": "",
                        "status": "ready",
                        "reason": "",
                        "selected_attempt_id": "",
                    }
                },
                "phase_attempts": {},
                "report_outputs": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--manifest",
            str(manifest_path),
            "--version-root",
            str(version_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert summary["source_records"]["supervision_recovery_summary"]["status"] == "missing"
    assert summary["source_records"]["support_summary"]["status"] == "ready"
    assert "Source key: supervision_recovery_summary" in markdown
    assert "Status: missing" in markdown
    assert "Markov Supervision-Recovery Report" in markdown
    assert "Runtime Efficiency" not in markdown
    assert "Large-Batch Diagnosis" not in markdown
    assert "Tree Geometry And Support" not in markdown


def test_markov_medium_grid_infers_metadata_from_runs() -> None:
    payload = {
        "by_batch_size": {
            "128": {"mean_best_val_mae": 0.02, "mean_docs_per_s_wall_effective": 1000.0}
        },
        "runs": {
            "bs0128_seed0": {"config": {"train_docs": 10240, "n_epochs": 5}},
        },
    }
    summary = _summarize_medium_grid(payload)
    assert summary["status"] == "ready"
    assert summary["train_docs"] == 10240
    assert summary["epochs"] == 5


def test_markov_runtime_efficiency_marks_missing_runtime_fields_unavailable() -> None:
    summary = _summarize_runtime_efficiency(batch_timing={}, batch_quality={}, docs_epochs={})
    assert summary["status"] == "unavailable"
    assert "runtime fields" in summary["reason"]


def test_markov_law_packages_external_reference_suppresses_same_run_wording() -> None:
    summary = _summarize_law_packages(
        {
            "tree_all_laws": {
                "test_root_mae": 0.2,
                "doc_fno_test_root_mae": 0.1,
            }
        },
        same_run_doc_fno=False,
    )
    assert summary["same_run_doc_fno"] is False
    assert summary["doc_fno_label"] == "staged doc FNO reference"
    assert summary["rows"][0]["tree_vs_doc_fno_root_mae_gap"] == 0.1


def test_markov_support_summary_rejects_mixed_model_families() -> None:
    summary = _summarize_support(
        {
            "rows": [
                {"fixed_leaf_tokens": 8, "model_family": "additive", "train_docs": 1024},
                {"fixed_leaf_tokens": 8, "model_family": "tree_neural", "train_docs": 1024},
            ]
        }
    )
    assert summary["status"] == "incompatible"
    assert "mixes model families" in summary["reason"]


def test_markov_supervision_summary_flags_invariant_profiles() -> None:
    summary = _summarize_supervision_sweep(
        {
            "rows": [
                {
                    "train_doc_count": 2048,
                    "leaf_profile": "none",
                    "internal_profile": "none",
                    "leaf_supervision_kind": "count_only",
                    "leaf_label_rate": 0.0,
                    "internal_supervision_kind": "none",
                    "internal_label_rate": 0.0,
                    "mean_test_root_mae": 0.0,
                },
                {
                    "train_doc_count": 2048,
                    "leaf_profile": "full_q100",
                    "internal_profile": "full_q100",
                    "leaf_supervision_kind": "full_sketch",
                    "leaf_label_rate": 1.0,
                    "internal_supervision_kind": "full_sketch",
                    "internal_label_rate": 1.0,
                    "mean_test_root_mae": 0.0,
                },
                {
                    "train_doc_count": 2048,
                    "leaf_profile": "count_q100",
                    "internal_profile": "count_q100",
                    "leaf_supervision_kind": "count_only",
                    "leaf_label_rate": 1.0,
                    "internal_supervision_kind": "count_only",
                    "internal_label_rate": 1.0,
                    "mean_test_root_mae": 0.0,
                },
            ],
            "by_train_docs": {},
            "leaf_profiles": ["none", "count_q100", "full_q100"],
            "internal_profiles": ["none", "count_q100", "full_q100"],
        },
        expected_train_doc_counts=[2048],
    )
    assert summary["status"] == "suspicious"
    assert "invariant" in summary["reason"]


def test_report_surfaces_canonical_reporting_views_from_results_jsonl(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "canonical_run"
    output_dir = tmp_path / "report"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "artifacts.json", {"artifacts": {"summary_json": {"path": str(output_root / "summary.json")}}})
    rows = [
        {
            "experiment_id": "exp_markov",
            "phase": "eval",
            "benchmark_ref": {
                "benchmark_id": "b1",
                "family": "markov_full_doc",
                "scope": "recoverable_v4",
                "name": "recoverable_v4",
            },
            "method_ref": {
                "method_id": "m1",
                "family": "tree_neural",
                "variant": "markov",
                "adapter": "markov_tree",
            },
            "split": "test",
            "seed": 0,
            "train_docs": 10240,
            "supervision_ref": {
                "root_rate": 0.1,
                "leaf_rate": 0.5,
                "internal_rate": 0.5,
                "topology_scope": "tree",
                "unit_selector": "root+leaf+internal",
                "supervision_kind": "scalar",
                "label_source": "dataset_labels",
                "labeler_kind": "precomputed",
                "coverage_label": "R10+LcIa50",
            },
            "control_ref": {
                "control_family": "tree_local_law",
                "law_ids": ["L1", "L2"],
                "applies_to": "tree_nodes",
                "enabled": True,
                "source_kind": "verifier",
            },
            "metric_name": "root_mae",
            "metric_value": 0.12,
            "artifact_refs": [],
        },
        {
            "experiment_id": "exp_treepo",
            "phase": "eval",
            "benchmark_ref": {
                "benchmark_id": "b2",
                "family": "treepo_task",
                "scope": "manifesto_rile",
                "name": "manifesto_rile",
            },
            "method_ref": {
                "method_id": "m2",
                "family": "ctreepo",
                "variant": "local_law_training",
                "adapter": "treepo_training",
            },
            "split": "test",
            "seed": 0,
            "train_docs": 10240,
            "supervision_ref": {
                "topology_scope": "tree",
                "unit_selector": "internal",
                "supervision_kind": "comparative",
                "label_source": "label_now",
                "labeler_kind": "oracle_score",
                "coverage_label": "internal_only",
            },
            "control_ref": {
                "control_family": "ctreepo_local_law",
                "law_ids": ["L1", "L2"],
                "applies_to": "leaf+internal",
                "enabled": True,
                "source_kind": "oracle_callback",
            },
            "metric_name": "root_mae",
            "metric_value": 0.09,
            "artifact_refs": [],
        },
    ]
    (output_root / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_optimization_tradeoffs.py",
            "--output-root",
            str(output_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report_views = summary["canonical_reporting"]["report_views"]
    assert report_views["method_families"] == ["ctreepo", "tree_neural"]
    assert "supervised_doc_regression" in report_views["comparison_domains"]
    assert report_views["supervision_labels"] == ["R10+LcIa50", "internal_only"]
    assert "ctreepo_local_law:L1+L2" in report_views["control_labels"]
    assert report_views["comparable_metrics"]["root_mae"]["method_families"] == ["ctreepo", "tree_neural"]
    assert report_views["reference_summaries"]["observed_frontiers"]
