from __future__ import annotations

import pytest

from src.core.data_models import Node, Tree
from src.ctreepo.local_law_rows import (
    LAW_KIND_LEAF,
    LAW_KIND_MERGE,
    NODE_ROLE_INTERNAL,
    NODE_ROLE_LEAF,
    NODE_ROLE_ROOT,
    SAMPLING_BERNOULLI,
    SAMPLING_FIXED_SIZE_UNIFORM,
    SAMPLING_FULL_OBS,
    SAMPLING_PERSISTENT_MASK,
    build_local_law_rows,
    classify_node_role,
    full_binary_tree_population_size,
)


def _node(node_id: str, *, proxy_loss: float = 0.1, oracle_loss: float = 0.2) -> Node:
    return Node(
        id=node_id,
        summary=node_id,
        metadata={"proxy_loss": proxy_loss, "oracle_loss": oracle_loss},
    )


def _full_binary_tree() -> Tree:
    leaves = [_node(f"leaf_{idx}") for idx in range(4)]
    left = _node("internal_left")
    right = _node("internal_right")
    root = _node("root")
    left.left_child, left.right_child = leaves[0], leaves[1]
    right.left_child, right.right_child = leaves[2], leaves[3]
    root.left_child, root.right_child = left, right
    for child in (left, right):
        child.parent = root
        child.level = 1
    for child in leaves[:2]:
        child.parent = left
    for child in leaves[2:]:
        child.parent = right
    root.level = 2
    return Tree(root=root, metadata={"doc_id": "doc_1"})


def test_full_binary_tree_rows_have_expected_population_and_roles() -> None:
    tree = _full_binary_tree()
    result = build_local_law_rows(
        tree,
        doc_id="doc_1",
        law_kind=LAW_KIND_MERGE,
        sampling_policy=SAMPLING_FULL_OBS,
    )

    assert len(result.rows) == full_binary_tree_population_size(4) == 7
    assert len(result.by_role(NODE_ROLE_LEAF)) == 4
    assert len(result.by_role(NODE_ROLE_INTERNAL)) == 2
    assert len(result.by_role(NODE_ROLE_ROOT)) == 1
    assert result.by_role(NODE_ROLE_ROOT)[0].depth == 0
    assert result.by_role(NODE_ROLE_ROOT)[0].metadata["no_double_count"] is True
    assert all(row.observed for row in result.rows)


def test_classify_node_role_is_stable_for_root_leaf_and_internal() -> None:
    tree = _full_binary_tree()
    root = tree.root
    internal = tree.root.left_child
    leaf = tree.root.left_child.left_child

    assert classify_node_role(root, root_id="root") == NODE_ROLE_ROOT
    assert classify_node_role(internal, root_id="root") == NODE_ROLE_INTERNAL
    assert classify_node_role(leaf, root_id="root") == NODE_ROLE_LEAF


def test_fixed_size_uniform_sampling_logs_q_over_n_propensity() -> None:
    nodes = [
        {"node_id": f"n{idx}", "proxy_loss": 0.1, "oracle_loss": 0.2}
        for idx in range(5)
    ]
    result = build_local_law_rows(
        {"root_id": "n0", "nodes": nodes},
        law_kind=LAW_KIND_LEAF,
        sampling_policy=SAMPLING_FIXED_SIZE_UNIFORM,
        sample_size=2,
        seed=11,
    )

    assert len(result.rows) == 5
    assert sum(1 for row in result.rows if row.observed) == 2
    assert all(row.propensity == pytest.approx(2.0 / 5.0) for row in result.rows)


def test_bernoulli_and_persistent_masks_use_design_propensity() -> None:
    nodes = [
        {"node_id": f"n{idx}", "proxy_loss": 0.1, "oracle_loss": 0.2}
        for idx in range(4)
    ]
    bernoulli = build_local_law_rows(
        {"root_id": "n0", "nodes": nodes},
        sampling_policy=SAMPLING_BERNOULLI,
        sample_rate=0.25,
        seed=3,
    )
    assert all(row.propensity == pytest.approx(0.25) for row in bernoulli.rows)

    first = build_local_law_rows(
        {"root_id": "n0", "nodes": nodes},
        sampling_policy=SAMPLING_PERSISTENT_MASK,
        sample_rate=0.25,
        persistent_mask=[False, True, False, True],
        seed=1,
    )
    second = build_local_law_rows(
        {"root_id": "n0", "nodes": nodes},
        sampling_policy=SAMPLING_PERSISTENT_MASK,
        sample_rate=0.25,
        persistent_mask=[False, True, False, True],
        seed=999,
    )

    assert [row.observed for row in first.rows] == [row.observed for row in second.rows]
    assert [row.observed for row in first.rows] == [False, True, False, True]
    assert all(row.propensity == pytest.approx(0.25) for row in first.rows)


def test_missing_oracle_payload_becomes_proxy_only_training_row() -> None:
    result = build_local_law_rows(
        [{"node_id": "n0", "proxy_loss": 0.5}],
        sampling_policy=SAMPLING_FULL_OBS,
    )

    assert len(result.rows) == 1
    assert result.rows[0].proxy_loss == pytest.approx(0.5)
    assert result.rows[0].oracle_loss is None
    assert result.rows[0].observed is False
    assert result.rows[0].propensity == pytest.approx(0.0)


def test_duplicate_cumulative_root_row_is_dropped() -> None:
    rows = [
        {
            "node_id": "root",
            "proxy_loss": 0.1,
            "oracle_loss": 0.2,
            "metadata": {"is_root": True},
        },
        {
            "node_id": "root",
            "proxy_loss": 0.3,
            "oracle_loss": 0.4,
            "metadata": {"is_root": True, "row_kind": "cumulative_merge"},
        },
    ]

    result = build_local_law_rows(
        {"root_id": "root", "nodes": rows},
        sampling_policy=SAMPLING_FULL_OBS,
    )

    assert len(result.rows) == 1
    assert result.rows[0].node_id == "root"
    assert result.rows[0].proxy_loss == pytest.approx(0.1)
