from __future__ import annotations

from src.core.logged_supervision import (
    LoggedLabelObservation,
    ObservationUnitKind,
    SamplingMetadata,
    read_logged_observations_jsonl,
    summarize_logged_observations,
    write_logged_observations_jsonl,
)
from src.tree.ipw import NodeType, TreeSample


def test_sampling_metadata_uses_product_when_joint_missing() -> None:
    sampling = SamplingMetadata(
        document_propensity=0.5,
        unit_propensity=0.25,
        label_propensity=1.0,
        unit_kind=ObservationUnitKind.LEAF,
    )

    assert sampling.effective_joint_propensity() == 0.125
    assert sampling.ipw_weight() == 8.0


def test_logged_observations_jsonl_roundtrip(tmp_path) -> None:
    rows = [
        LoggedLabelObservation(
            observation_id="doc-1:c1:leaf_0",
            document_id="doc-1",
            unit_id="leaf_0",
            unit_kind=ObservationUnitKind.LEAF,
            target_name="c1",
            label=0.2,
            sampling=SamplingMetadata(
                document_propensity=1.0,
                unit_propensity=0.5,
                label_propensity=1.0,
                unit_kind=ObservationUnitKind.LEAF,
            ),
        )
    ]

    artifact = write_logged_observations_jsonl(
        tmp_path / "logged.jsonl",
        rows,
        channel_name="sampled_substructure_supervision",
    )
    loaded = read_logged_observations_jsonl(tmp_path / "logged.jsonl")
    summary = summarize_logged_observations(loaded)

    assert artifact.count == 1
    assert artifact.channel_name == "sampled_substructure_supervision"
    assert loaded[0].to_dict() == rows[0].to_dict()
    assert summary["supports_ipw_estimation"] is True
    assert summary["joint_propensity_min"] == 0.5


def test_tree_sample_from_logged_observation() -> None:
    observation = LoggedLabelObservation(
        observation_id="doc-1:sufficiency:leaf_0",
        document_id="doc-1",
        unit_id="leaf_0",
        unit_kind=ObservationUnitKind.LEAF,
        target_name="sufficiency",
        label=1,
        sampling=SamplingMetadata(
            document_propensity=1.0,
            unit_propensity=0.5,
            label_propensity=1.0,
            unit_kind=ObservationUnitKind.LEAF,
        ),
        context={"discrepancy_score": 0.8},
    )

    sample = TreeSample.from_logged_observation(
        observation,
        violation=int(observation.label),
        preference_loss=float(observation.context["discrepancy_score"]),
    )

    assert sample.node_type == NodeType.LEAF
    assert sample.joint_propensity == 0.5
    assert sample.weight == 2.0
