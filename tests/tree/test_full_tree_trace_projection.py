from __future__ import annotations

import json

import pytest

from treepo.training.local_law import (
    local_law_training_objective_mean as local_law_objective_mean,
)
from src.tree.full_tree_ipw import (
    document_record_from_state_tree,
    full_tree_node_records_from_state_tree,
    local_law_observations_from_state_tree,
)
from src.tree.labeled import LabeledNode, LabeledTree
from src.tree.state_tree import (
    local_law_trace_metadata,
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)


def _tiny_labeled_tree() -> LabeledTree:
    tree = LabeledTree(doc_id="doc-1", document_text="alpha beta", document_score=0.7)
    left = LabeledNode(
        node_id="leaf_0",
        doc_id="doc-1",
        level=0,
        text="alpha",
        score=0.2,
    )
    right = LabeledNode(
        node_id="leaf_1",
        doc_id="doc-1",
        level=0,
        text="beta",
        score=0.4,
    )
    root = LabeledNode(
        node_id="root",
        doc_id="doc-1",
        level=1,
        text="alpha beta",
        score=0.7,
        left_child_id="leaf_0",
        right_child_id="leaf_1",
    )
    for node in (left, right, root):
        tree.add_node(node)
    tree.metadata["split"] = "test"
    return tree


def test_state_tree_skeleton_preserves_stable_topology() -> None:
    trace = state_tree_skeleton_from_labeled_tree(
        _tiny_labeled_tree(),
        method_family="unit",
        state_kind="summary_text",
    )

    assert trace.root.id == "root"
    assert trace.root.left_child is not None
    assert trace.root.right_child is not None
    assert trace.root.left_child.id == "leaf_0"
    assert trace.root.right_child.id == "leaf_1"
    assert trace.root.metadata["depth"] == 0
    assert trace.root.left_child.metadata["depth"] == 1
    assert trace.metadata["split"] == "test"


def test_updating_state_tree_node_does_not_change_topology() -> None:
    trace = state_tree_skeleton_from_labeled_tree(_tiny_labeled_tree())
    before = (trace.root.left_child.id, trace.root.right_child.id)  # type: ignore[union-attr]
    update_state_tree_node(
        trace,
        "leaf_0",
        rendered="generated summary",
        metadata={
            "prediction": 0.25,
            "target": 0.2,
            "proxy_loss": 0.0025,
            "oracle_loss": 0.0025,
            "observed": True,
            "propensity": 1.0,
        },
    )

    after = (trace.root.left_child.id, trace.root.right_child.id)  # type: ignore[union-attr]
    assert before == after
    assert trace.root.left_child.rendered == "generated summary"  # type: ignore[union-attr]


def test_state_tree_projects_to_estimator_records_and_loss_rows() -> None:
    trace = state_tree_skeleton_from_labeled_tree(_tiny_labeled_tree())
    for node in trace.traverse_preorder():
        target = float(node.metadata["target"])
        prediction = target + 0.1
        update_state_tree_node(
            trace,
            node.id,
            metadata={
                "prediction": prediction,
                "target": target,
                "proxy_loss": (prediction - target) ** 2,
                "oracle_loss": (prediction - target) ** 2,
                "observed": True,
                "propensity": 1.0,
            },
        )

    records = full_tree_node_records_from_state_tree(trace)
    observations = local_law_observations_from_state_tree(trace)
    document = document_record_from_state_tree(trace)

    assert len(records) == 3
    assert len(observations) == 3
    assert document is not None
    assert document.doc_id == "doc-1"
    assert all(record.sampled for record in records)
    assert observations[0].corrected_loss() == pytest.approx(observations[0].oracle_loss)


def test_state_tree_local_law_projection_keeps_proxy_only_rows() -> None:
    trace = state_tree_skeleton_from_labeled_tree(_tiny_labeled_tree())
    for node in trace.traverse_preorder():
        update_state_tree_node(
            trace,
            node.id,
            metadata={
                "prediction": 0.5,
                "proxy_loss": 0.25,
                "observed": False,
                "propensity": 0.0,
            },
        )

    observations = local_law_observations_from_state_tree(trace)
    metrics = state_tree_trace_metrics([trace])

    assert len(observations) == 3
    assert all(observation.oracle_loss is None for observation in observations)
    assert all(not observation.observed for observation in observations)
    assert local_law_objective_mean(observations) == pytest.approx(0.25)
    assert local_law_objective_mean(observations, objective_mode="sampled_ipw") == pytest.approx(0.0)
    assert metrics["count_proxy_rows"] == 3
    assert metrics["count_oracle_rows"] == 0


def test_state_tree_root_only_trace_has_sparse_oracle_rows() -> None:
    trace = state_tree_skeleton_from_labeled_tree(_tiny_labeled_tree())
    for node in trace.traverse_preorder():
        target = float(node.metadata["target"])
        prediction = target + 0.1
        is_root = bool(node.metadata["is_root"])
        update_state_tree_node(
            trace,
            node.id,
            metadata={
                "target": target,
                **local_law_trace_metadata(
                    prediction=prediction,
                    proxy_target=target,
                    oracle_target=target if is_root else None,
                    observed=is_root,
                    sampled=is_root,
                    propensity=1.0 if is_root else None,
                    law_channel="root" if is_root else "node",
                    state_kind="unit_state",
                ),
            },
        )

    observations = local_law_observations_from_state_tree(trace)
    metrics = state_tree_trace_metrics([trace])

    assert len(observations) == 3
    assert sum(1 for row in observations if row.observed) == 1
    assert metrics["count_proxy_rows"] == 3
    assert metrics["count_oracle_rows"] == 1
    assert metrics["count_observed_nodes"] == 1
    assert all(row.propensity == pytest.approx(0.0) for row in observations if not row.observed)


def test_write_state_trees_jsonl_emits_json_safe_trace(tmp_path) -> None:
    trace = state_tree_skeleton_from_labeled_tree(_tiny_labeled_tree())
    update_state_tree_node(trace, "root", metadata={"prediction": 0.8, "target": 0.7})
    output = write_state_trees_jsonl([trace], tmp_path / "full_tree_traces_test.jsonl")
    payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert payload["root_id"] == "root"
    assert payload["nodes"]["root"]["metadata"]["prediction"] == pytest.approx(0.8)
    assert state_tree_trace_metrics([trace])["count_nodes"] == 3
