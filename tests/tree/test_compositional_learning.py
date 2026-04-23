from src.core.ops_checks import LawKind
from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
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
    shared_full_document_supervision_channel,
    shared_logged_document_observation,
    shared_logged_substructure_observation,
    shared_protocol_problem_notes,
    shared_sampled_substructure_query_policy,
    shared_sampled_substructure_supervision_channel,
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
    assert "runtime_capabilities" not in problem.to_dict()


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
        sampling=SamplingMetadata(
            joint_propensity=0.1,
            unit_kind=ObservationUnitKind.LEAF,
        ),
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


def test_shared_protocol_helpers_emit_canonical_names() -> None:
    policy = shared_sampled_substructure_query_policy(
        selection_strategy="uniform_random",
        budget={"sample_budget": 5},
        logs_realized_propensities=True,
        supports_ipw_estimation=True,
    )
    full_channel = shared_full_document_supervision_channel(active=True)
    sampled_channel = shared_sampled_substructure_supervision_channel(
        active=True,
        delivery_mode=SupervisionDeliveryMode.ONLINE_ORACLE_QUERY,
        query_policy=policy,
        targeted_laws=(LawKind.L1_LEAF,),
    )
    notes = shared_protocol_problem_notes(application_name="demo_application")

    assert isinstance(policy, OracleQueryPolicySpec)
    assert full_channel.name == "full_document_supervision"
    assert sampled_channel.name == "sampled_substructure_supervision"
    assert sampled_channel.target_name == "substructure_level_target"
    assert sampled_channel.query_policy is not None
    assert sampled_channel.query_policy.name == "sampled_substructure_query_policy"
    assert notes[0] == "application=demo_application"


def test_shared_logged_observation_helpers_use_canonical_targets() -> None:
    sampling = SamplingMetadata(
        document_propensity=1.0,
        unit_propensity=0.5,
        label_propensity=1.0,
        unit_kind=ObservationUnitKind.LEAF,
    )
    substructure = shared_logged_substructure_observation(
        document_id="doc-1",
        unit_id="leaf_0",
        unit_kind=ObservationUnitKind.LEAF,
        label=0.3,
        sampling=sampling,
        application_name="demo_application",
        supervision_signal_name="c1",
        law_kind=LawKind.L1_LEAF,
        context={"raw_score": 0.3},
    )
    document = shared_logged_document_observation(
        document_id="doc-1",
        label=1.0,
        sampling=SamplingMetadata(unit_kind=ObservationUnitKind.DOCUMENT),
        application_name="demo_application",
        supervision_signal_name="root_score",
    )

    assert substructure.target_name == "substructure_level_target"
    assert substructure.context["application_name"] == "demo_application"
    assert substructure.context["supervision_signal_name"] == "c1"
    assert substructure.context["law_kind"] == LawKind.L1_LEAF.value
    assert document.target_name == "document_level_target"
    assert document.context["supervision_channel_name"] == "full_document_supervision"
