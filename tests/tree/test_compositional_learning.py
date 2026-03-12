from src.core.ops_checks import LawKind
from src.tree.compositional_learning import (
    CompositionalLearningProblemSpec,
    FullDocumentLabelObservation,
    OracleQueryPolicySpec,
    SampledSubstructureLabelObservation,
    SupervisionDeliveryMode,
    SupervisionChannelKind,
    full_document_supervision_channel,
    oracle_query_policy,
    sampled_substructure_supervision_channel,
)
from src.tree.theorem_backing import broadest_exact_theorem_assumptions


def test_problem_spec_tracks_both_supervision_channels() -> None:
    problem = CompositionalLearningProblemSpec(
        name="llm_summary_learning",
        document_type_name="documents",
        theorem_domain_name="summary_objects",
        operator_name="text_codec",
        theorem_assumptions=broadest_exact_theorem_assumptions(),
        supervision_channels=(
            full_document_supervision_channel(
                target_name="document_score",
                notes=("Whole-document downstream objective.",),
            ),
            sampled_substructure_supervision_channel(
                target_name="node_score",
                targeted_laws=(LawKind.L1_LEAF, LawKind.L2_MERGE, LawKind.L3_IDEMPOTENCE),
                notes=("Randomly sampled leaves and internal nodes.",),
            ),
        ),
    )

    assert problem.uses_full_document_labels is True
    assert problem.uses_sampled_substructure_labels is True
    assert problem.uses_online_oracle_queries is False
    assert problem.requires_propensity_logging is True
    assert problem.supports_theorem_backing is True
    assert problem.supports_nested_summaries is True
    assert problem.capability_report() is not None
    assert "full-document labels + sampled substructure labels" in problem.problem_statement()


def test_full_document_channel_defaults_to_complete_labels() -> None:
    channel = full_document_supervision_channel(target_name="document_target")

    assert channel.kind == SupervisionChannelKind.FULL_DOCUMENT
    assert channel.requires_propensity_logging is False
    assert channel.supports_unbiased_risk is True


def test_sampled_substructure_channel_records_targeted_laws() -> None:
    channel = sampled_substructure_supervision_channel(
        target_name="node_target",
        targeted_laws=(LawKind.L1_LEAF, LawKind.L2_MERGE),
    )

    assert channel.kind == SupervisionChannelKind.SAMPLED_SUBSTRUCTURE
    assert channel.requires_propensity_logging is True
    assert channel.targeted_laws == (LawKind.L1_LEAF, LawKind.L2_MERGE)


def test_label_observations_normalize_truth_sources() -> None:
    full = FullDocumentLabelObservation(
        document_id="doc-1",
        label={"score": 1.0},
        truth_label_source="judge",
    )
    sampled = SampledSubstructureLabelObservation(
        document_id="doc-1",
        unit={"node_id": "n-1"},
        label=0.25,
        propensity=0.1,
        truth_label_source="manual",
    )

    assert full.to_dict()["truth_label_source"] == "oracle"
    assert sampled.to_dict()["truth_label_source"] == "human"
    assert sampled.is_propensity_annotated is True


def test_online_query_policy_serializes_on_sampled_channel() -> None:
    policy = oracle_query_policy(
        name="ipw_node_queries",
        query_unit_name="tree_nodes",
        selection_strategy="content_weighted",
        adaptive=True,
        budget={"sample_budget": 12},
        propensity_field_name="joint_propensity",
        logs_realized_propensities=True,
        supports_ipw_estimation=True,
        notes=("Oracle calls are made on sampled nodes only.",),
    )
    channel = sampled_substructure_supervision_channel(
        target_name="node_target",
        delivery_mode=SupervisionDeliveryMode.ONLINE_ORACLE_QUERY,
        query_policy=policy,
        targeted_laws=(LawKind.L1_LEAF, LawKind.L2_MERGE),
    )

    assert isinstance(policy, OracleQueryPolicySpec)
    assert channel.delivery_mode == SupervisionDeliveryMode.ONLINE_ORACLE_QUERY
    assert channel.query_policy is not None
    assert channel.to_dict()["query_policy"]["selection_strategy"] == "content_weighted"
    assert channel.to_dict()["query_policy"]["logs_realized_propensities"] is True
