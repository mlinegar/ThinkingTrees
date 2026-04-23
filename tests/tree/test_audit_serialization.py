from __future__ import annotations

from src.tree.audit_serialization import audit_problem_manifest, audit_report_to_dict
from src.tree.auditor import AlwaysPassScorer, AuditConfig, Auditor, SamplingStrategy
from src.tree.compositional_operator import make_deterministic_summary_operator
from src.tree.theorem_backing import broadest_exact_theorem_assumptions


def test_audit_report_to_dict_preserves_problem_metadata(simple_tree) -> None:
    operator = make_deterministic_summary_operator(
        lambda text, rubric: text.upper(),
        name="exactish_summary",
        assumptions=broadest_exact_theorem_assumptions().operator_assumptions,
    )
    auditor = Auditor(
        AlwaysPassScorer(),
        config=AuditConfig(sample_budget=2, sampling_strategy=SamplingStrategy.RANDOM),
        theorem_operator=operator,
    )

    report = auditor.audit_tree(simple_tree)
    payload = audit_report_to_dict(report)

    assert payload["operator_capabilities"]["operator_name"] == "exactish_summary"
    assert payload["compositional_learning_problem"]["name"] == "tree_audit_verification"
    assert payload["compositional_learning_problem"]["operator_assumptions"] is not None
    assert payload["compositional_learning_problem"]["requires_propensity_logging"] is True
    assert payload["violation_rates"]["sufficiency"]["samples"] == payload["sufficiency_samples"]
    assert payload["inclusion_probability_map"]


def test_audit_problem_manifest_surfaces_channel_and_propensity_state() -> None:
    report_payload = {
        "sampling_strategy": "random",
        "sampling_probability": 1.0,
        "inclusion_probability_map": {"leaf_1": 0.5},
        "operator_capabilities": {"operator_name": "demo_operator"},
        "compositional_learning_problem": {
            "name": "tree_audit_verification",
            "uses_full_document_labels": False,
            "uses_sampled_substructure_labels": True,
            "uses_online_oracle_queries": True,
            "requires_propensity_logging": True,
            "supports_theorem_backing": True,
                "supervision_channels": [
                {
                    "name": "sampled_substructure_supervision",
                    "kind": "sampled_substructure",
                    "delivery_mode": "online_oracle_query",
                    "requires_propensity_logging": True,
                }
            ],
            "operator_assumptions": {"evidence_status": "theorem_backed"},
        },
    }

    manifest = audit_problem_manifest(report_payload)

    assert manifest["problem_name"] == "tree_audit_verification"
    assert manifest["uses_sampled_substructure_labels"] is True
    assert manifest["uses_online_oracle_queries"] is True
    assert manifest["requires_propensity_logging"] is True
    assert manifest["logged_inclusion_probabilities"] is True
    assert manifest["operator_assumptions"] == {"evidence_status": "theorem_backed"}
    assert manifest["operator_capabilities"] == {"operator_name": "demo_operator"}
