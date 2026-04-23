from __future__ import annotations

from src.ctreepo.sim.core.full_doc_family_contract import (
    full_doc_family_api_report,
    full_doc_law_contract_report,
)


def test_official_fno_and_tree_neural_share_unified_family_api_group() -> None:
    fno_api = full_doc_family_api_report("official_fno")
    tree_api = full_doc_family_api_report("tree_neural")

    assert fno_api["family_api_group"] == "markov_full_doc_neuraloperator"
    assert tree_api["family_api_group"] == "markov_full_doc_neuraloperator"
    assert fno_api["shared_framework_group"] == "shared_markov_fno_encoder"
    assert tree_api["shared_framework_group"] == "shared_markov_fno_encoder"
    assert fno_api["family_runner_kind"] == "official_fno_doc_sequence"
    assert tree_api["family_runner_kind"] == "tree_fno_count_sketch"


def test_tree_neural_law_contract_flags_trivial_c2_and_single_leaf_geometry() -> None:
    contract = full_doc_law_contract_report(
        "tree_neural",
        config_like={
            "objective_weights_active": True,
            "tree_c2_mode": "reconstruction",
            "summary_spec_name": "markov_count_sketch",
            "local_law_c1_weight": 0.1,
            "local_law_c2_weight": 0.1,
            "local_law_c3_weight": 0.1,
        },
        objective_weights_active=True,
        mean_leaves_per_doc=1.0,
    )

    assert contract["law_alignment_status"] == "approximate_with_gaps"
    assert contract["c2_nontriviality_status"] == "decoded_summary_replay"
    assert "single_leaf_geometry_collapses_tree_local_laws" in contract["law_contract_gaps"]
    assert "c2_replay_proxy_not_exact_paper_idempotence" in contract["law_contract_limitations"]
    assert "c2_lacks_external_fiber_contrast" in contract["law_contract_limitations"]
    assert contract["law_contract"]["c2"]["train_semantics"] == "decode_encode_replay"


def test_tree_neural_law_contract_recognizes_nontrivial_decoded_summary_replay() -> None:
    contract = full_doc_law_contract_report(
        "tree_neural_c2",
        config_like={
            "objective_weights_active": True,
            "tree_c2_mode": "reconstruction",
            "summary_spec_name": "markov_count_sketch",
            "local_law_c2_weight": 0.2,
        },
        objective_weights_active=True,
        mean_leaves_per_doc=8.0,
    )

    assert contract["law_alignment_status"] == "approximate_audited"
    assert contract["c2_nontriviality_status"] == "decoded_summary_replay"
    assert contract["law_contract_gap_count"] == 0
    assert contract["law_contract_limitation_count"] == 2
    assert "c2_replay_proxy_not_exact_paper_idempotence" in contract["law_contract_limitations"]
    assert contract["law_contract"]["c2"]["objective_enforced"] is True
