from __future__ import annotations

from src.ctreepo.sim.core.markov_v3_row_contract import (
    annotate_downstream_v3_row,
)


def _base_row() -> dict[str, object]:
    return {
        "baseline_family": "tree_neural",
        "scope_key": "recoverable_v4",
        "comparison_mode": "comparable",
        "comparison_semantics": "current",
        "comparison_semantics_label": "tree_neural_objective_v4",
        "run_intent_hash": "intent_hash",
        "run_intent_validation_status": "validated",
        "requested_fixed_leaf_tokens": 128,
        "executed_fixed_leaf_tokens": 128,
        "depth_discount_gamma": 1.0,
        "package_name": "full90",
        "tree_reference_label": "unified_g_full_local_laws_v1",
        "computed_leaf_mass_per_doc": 0.0,
        "computed_internal_mass_per_doc": 0.0,
    }


def test_known_invalid_one_leaf_root_only_recipe_is_quarantined() -> None:
    row = annotate_downstream_v3_row(
        _base_row(),
        canonical_fno_families=("official_fno", "official_fno_sumlen"),
        canonical_fno_fixed_leaf_tokens=128,
    )

    assert row["contract_status"] == "legacy_quarantined"
    assert row["contract_headline_eligible"] is False
    assert "known_invalid_one_leaf_root_only_recipe" in row["contract_failures"]


def test_one_leaf_full100_canary_row_is_not_quarantined_by_recipe_guard() -> None:
    payload = _base_row()
    payload.update(
        {
            "package_name": "full100",
            "tree_reference_label": "unified_g_fno_parity_canary_v1",
        }
    )
    row = annotate_downstream_v3_row(
        payload,
        canonical_fno_families=("official_fno", "official_fno_sumlen"),
        canonical_fno_fixed_leaf_tokens=128,
    )

    assert row["contract_status"] == "current"
    assert "known_invalid_one_leaf_root_only_recipe" not in row["contract_failures"]


def test_multileaf_root_only_row_is_not_quarantined_by_one_leaf_guard() -> None:
    payload = _base_row()
    payload.update(
        {
            "requested_fixed_leaf_tokens": 64,
            "executed_fixed_leaf_tokens": 64,
        }
    )
    row = annotate_downstream_v3_row(
        payload,
        canonical_fno_families=("official_fno", "official_fno_sumlen"),
        canonical_fno_fixed_leaf_tokens=128,
    )

    assert row["contract_status"] == "current"
    assert "known_invalid_one_leaf_root_only_recipe" not in row["contract_failures"]


def test_known_invalid_one_leaf_matched_root_v1_recipe_is_quarantined() -> None:
    payload = _base_row()
    payload.update(
        {
            "tree_reference_label": "recoverable_root_only_parity_matched_root_v1",
            "computed_leaf_mass_per_doc": 0.0,
            "computed_internal_mass_per_doc": 0.0,
        }
    )
    row = annotate_downstream_v3_row(
        payload,
        canonical_fno_families=("official_fno", "official_fno_sumlen"),
        canonical_fno_fixed_leaf_tokens=128,
    )

    assert row["contract_status"] == "legacy_quarantined"
    assert row["contract_headline_eligible"] is False
    assert "known_invalid_one_leaf_matched_root_v1_recipe" in row["contract_failures"]


def test_structural_one_leaf_partial_root_rows_are_quarantined_pending_rescue() -> None:
    payload = _base_row()
    payload.update(
        {
            "scope_key": "r12_seg10to12",
            "tree_reference_label": "structural_root_only_parity_matched_root_v3",
        }
    )
    row = annotate_downstream_v3_row(
        payload,
        canonical_fno_families=("official_fno", "official_fno_sumlen"),
        canonical_fno_fixed_leaf_tokens=128,
    )

    assert row["contract_status"] == "diagnostic_only"
    assert row["contract_headline_eligible"] is False
    assert row["contract_failures"] == []
    assert (
        "structural_one_leaf_partial_root_rescue_pending"
        in row["contract_diagnostic_reasons"]
    )
