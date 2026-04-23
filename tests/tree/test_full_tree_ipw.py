from __future__ import annotations

import pytest

from src.tree.full_tree_ipw import (
    classify_layered_sampling_regime,
    DocumentLevelPredictionRecord,
    FullTreeIPWSummaryAccumulator,
    FullTreeNodeRecord,
    exact_full_node_mean_loss,
    layered_propensity_policy,
    project_node_records_to_tree_samples,
    run_full_tree_estimator_monte_carlo,
    sampled_ht_node_mean_loss,
    summarize_full_tree_ipw,
)
from src.tree.ipw import NodeType


def _node(
    *,
    doc_id: str,
    node_id: str,
    node_type: NodeType,
    depth: int,
    target: float,
    prediction: float,
    sampled: bool,
    propensity: float,
    is_root: bool = False,
) -> FullTreeNodeRecord:
    return FullTreeNodeRecord(
        doc_id=doc_id,
        node_id=node_id,
        depth=depth,
        node_type=node_type,
        is_root=is_root,
        prediction=prediction,
        target=target,
        sampled=sampled,
        propensity=propensity,
    )


def test_full_node_exact_mean_matches_ht_under_full_sample() -> None:
    records = [
        _node(
            doc_id="d1",
            node_id="leaf_0",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=1.0,
            sampled=True,
            propensity=1.0,
        ),
        _node(
            doc_id="d1",
            node_id="leaf_1",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=0.0,
            sampled=True,
            propensity=1.0,
        ),
        _node(
            doc_id="d1",
            node_id="root",
            node_type=NodeType.MERGE,
            depth=0,
            target=0.5,
            prediction=0.0,
            sampled=True,
            propensity=1.0,
            is_root=True,
        ),
    ]
    exact = exact_full_node_mean_loss(records)
    ht = sampled_ht_node_mean_loss(records)
    assert exact == pytest.approx((1.0 + 0.0 + 0.25) / 3.0)
    assert ht == pytest.approx(exact)


def test_naive_is_biased_under_skewed_propensities_but_ht_is_nearly_unbiased() -> None:
    records = [
        _node(
            doc_id="d1",
            node_id="hard_0",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=1.0,
            sampled=False,
            propensity=1.0,
        ),
        _node(
            doc_id="d1",
            node_id="hard_1",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=1.0,
            sampled=False,
            propensity=1.0,
        ),
        _node(
            doc_id="d1",
            node_id="easy_0",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=0.0,
            sampled=False,
            propensity=1.0,
        ),
        _node(
            doc_id="d1",
            node_id="easy_1",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=0.0,
            sampled=False,
            propensity=1.0,
        ),
    ]
    summary = run_full_tree_estimator_monte_carlo(
        records,
        [],
        propensity_fn=lambda record: 0.2 if "hard" in record.node_id else 0.8,
        n_trials=2000,
        seed=17,
        policy_name="skewed",
    )
    assert summary.true_full_node_mean == pytest.approx(0.5)
    assert summary.naive.bias < -0.15
    assert abs(summary.ht.bias) < 0.05


def test_document_top_loss_is_separate_from_node_estimand() -> None:
    node_records = [
        _node(
            doc_id="d1",
            node_id="leaf_0",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.2,
            prediction=0.2,
            sampled=True,
            propensity=1.0,
        ),
        _node(
            doc_id="d1",
            node_id="root",
            node_type=NodeType.MERGE,
            depth=0,
            target=0.4,
            prediction=0.4,
            sampled=True,
            propensity=1.0,
            is_root=True,
        ),
    ]
    document_records = [
        DocumentLevelPredictionRecord(
            doc_id="d1",
            prediction=0.9,
            target=0.1,
        )
    ]
    summary = summarize_full_tree_ipw(node_records, document_records)
    assert summary["full_node_exact_mean_loss"] == pytest.approx(0.0)
    assert summary["document_top_loss"] == pytest.approx(0.64)


def test_projection_preserves_root_metadata_and_propensity() -> None:
    records = [
        _node(
            doc_id="d1",
            node_id="root",
            node_type=NodeType.MERGE,
            depth=0,
            target=0.0,
            prediction=1.0,
            sampled=True,
            propensity=0.25,
            is_root=True,
        )
    ]
    samples = project_node_records_to_tree_samples(records)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.node_type == NodeType.MERGE
    assert sample.joint_propensity == pytest.approx(0.25)
    assert sample.metadata["is_root"] is True
    assert sample.metadata["depth"] == 0


def test_layered_policy_uses_internal_rate_for_root_and_leaf_rate_for_leaves() -> None:
    policy = layered_propensity_policy(leaf_rate=0.2, internal_rate=0.7)
    leaf = _node(
        doc_id="d1",
        node_id="leaf_0",
        node_type=NodeType.LEAF,
        depth=1,
        target=0.0,
        prediction=0.0,
        sampled=False,
        propensity=1.0,
    )
    root = _node(
        doc_id="d1",
        node_id="root",
        node_type=NodeType.MERGE,
        depth=0,
        target=0.0,
        prediction=0.0,
        sampled=False,
        propensity=1.0,
        is_root=True,
    )
    assert policy(leaf) == pytest.approx(0.2)
    assert policy(root) == pytest.approx(0.7)
    assert (
        classify_layered_sampling_regime(leaf_rate=0.2, internal_rate=0.7)
        == "internal_heavy"
    )


def test_summary_reports_document_vs_root_node_gap() -> None:
    node_records = [
        FullTreeNodeRecord(
            doc_id="d1",
            node_id="root",
            node_type=NodeType.MERGE,
            depth=0,
            target=0.2,
            prediction=0.8,
            objective_prediction=0.3,
            sampled=True,
            propensity=1.0,
            is_root=True,
        )
    ]
    document_records = [
        DocumentLevelPredictionRecord(
            doc_id="d1",
            prediction=0.8,
            target=0.9,
        )
    ]
    summary = summarize_full_tree_ipw(node_records, document_records)
    assert summary["document_top_mae"] == pytest.approx(0.1)
    assert summary["document_vs_root_node_target_gap_mae"] == pytest.approx(0.7)
    assert summary["document_vs_root_node_prediction_gap_mae"] == pytest.approx(0.5)


def test_streaming_summary_matches_batch_summary() -> None:
    node_records = [
        _node(
            doc_id="d1",
            node_id="leaf_0",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=1.0,
            sampled=True,
            propensity=0.5,
        ),
        _node(
            doc_id="d1",
            node_id="root",
            node_type=NodeType.MERGE,
            depth=0,
            target=0.25,
            prediction=0.0,
            sampled=True,
            propensity=1.0,
            is_root=True,
        ),
        _node(
            doc_id="d2",
            node_id="leaf_0",
            node_type=NodeType.LEAF,
            depth=1,
            target=0.0,
            prediction=0.0,
            sampled=False,
            propensity=0.5,
        ),
    ]
    document_records = [
        DocumentLevelPredictionRecord(doc_id="d1", prediction=0.75, target=0.5),
        DocumentLevelPredictionRecord(doc_id="d2", prediction=0.0, target=0.0),
    ]
    batch_summary = summarize_full_tree_ipw(node_records, document_records)
    accumulator = FullTreeIPWSummaryAccumulator()
    for record in node_records:
        accumulator.update_node_record(record)
    for record in document_records:
        accumulator.update_document_record(record)
    streaming_summary = accumulator.finalize()
    for key in (
        "population_size",
        "sampled_nodes",
        "sampled_fraction",
        "full_node_exact_mean_loss",
        "sampled_node_naive_mean_loss",
        "sampled_node_ht_mean_loss",
        "sampled_node_hajek_mean_loss",
        "document_top_loss",
        "document_top_mae",
        "document_vs_root_node_target_gap_mae",
        "document_vs_root_node_prediction_gap_mae",
        "document_vs_root_node_pair_count",
        "effective_sample_size",
        "max_weight",
        "weight_sum",
    ):
        assert streaming_summary[key] == pytest.approx(batch_summary[key], nan_ok=True)
    for section in ("type_breakdown", "depth_breakdown"):
        assert set(streaming_summary[section].keys()) == set(batch_summary[section].keys())
        for bucket, bucket_summary in streaming_summary[section].items():
            for key, value in bucket_summary.items():
                assert value == pytest.approx(
                    batch_summary[section][bucket][key],
                    nan_ok=True,
                )
