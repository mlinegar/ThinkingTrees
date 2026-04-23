from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel
from src.tree.embedding_tree import build_embedding_tree
from src.tree.packed_execution import (
    apply_runtime_mode_to_packed_trees,
    build_packed_tree_batch_from_bucket_store,
    build_packed_tree_bucket_stores,
    build_packed_embedding_tree,
    build_packed_tree_batch,
    forward_packed_tree_batch,
)


def _make_tree(*, text: str, n_leaves: int, embedding_dim: int = 4):
    if n_leaves <= 0:
        raise ValueError("n_leaves must be positive")
    windows = []
    chunk = max(1, len(text) // n_leaves)
    start = 0
    for leaf_idx in range(n_leaves):
        end = len(text) if leaf_idx == n_leaves - 1 else min(len(text), start + chunk)
        windows.append((start, end))
        start = end
    embeddings = [
        [float((leaf_idx + 1) * (dim_idx + 1)) / 10.0 for dim_idx in range(embedding_dim)]
        for leaf_idx in range(n_leaves)
    ]
    return build_embedding_tree(text, embeddings, windows)


def _naive_forward(model: CTreePOModel, nodes):
    device = next(model.parameters()).device
    states = []
    with torch.no_grad():
        for node in nodes:
            if node.is_leaf:
                assert isinstance(node.embedding, torch.Tensor)
                states.append(model.encode_leaf(node.embedding.to(device=device)))
                continue
            left_idx, right_idx = node.children
            states.append(model.merge(states[left_idx], states[right_idx]))
    return states


def test_packed_forward_matches_naive_tree_walk_for_mixed_trees() -> None:
    model = CTreePOModel(CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16)).eval()
    trees = [
        _make_tree(text="abcdefghij", n_leaves=3),
        _make_tree(text="qrstuvwxyzab", n_leaves=4),
    ]
    packed_trees = [build_packed_embedding_tree(nodes) for nodes in trees]
    apply_runtime_mode_to_packed_trees(packed_trees, device=torch.device("cpu"))
    packed_batch = build_packed_tree_batch(packed_trees, device=torch.device("cpu"))

    result = forward_packed_tree_batch(model, packed_batch, materialize_nodes=True)

    assert result.runtime_stats["packed_executor_mode"] == "generic_packed"
    assert result.runtime_stats["materialized_node_sketch_count"] == sum(len(nodes) for nodes in trees)
    for doc_index, nodes in enumerate(trees):
        expected_states = _naive_forward(model, nodes)
        for node_index, expected_state in enumerate(expected_states):
            actual_state = result.state_batch[result.global_index(doc_index, node_index)]
            assert torch.allclose(actual_state, expected_state, atol=1e-6, rtol=1e-6)
            assert torch.allclose(nodes[node_index].sketch, expected_state, atol=1e-6, rtol=1e-6)


def test_fixed_fused_matches_generic_packed_for_balanced_batches() -> None:
    model = CTreePOModel(CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16)).eval()
    trees = [
        _make_tree(text="abcdefghijklmnop", n_leaves=4),
        _make_tree(text="qrstuvwxyzabcdef", n_leaves=4),
    ]
    packed_trees = [build_packed_embedding_tree(nodes) for nodes in trees]
    packed_batch = build_packed_tree_batch(packed_trees, device=torch.device("cpu"))
    assert packed_batch.fixed_fused_eligible is True

    fused_result = forward_packed_tree_batch(model, packed_batch)
    generic_batch = replace(
        packed_batch,
        fixed_fused_eligible=False,
        runtime_stats={**packed_batch.runtime_stats, "packed_executor_mode": "generic_packed"},
    )
    generic_result = forward_packed_tree_batch(model, generic_batch)

    assert fused_result.runtime_stats["packed_executor_mode"] == "fixed_fused"
    assert generic_result.runtime_stats["packed_executor_mode"] == "generic_packed"
    assert torch.allclose(fused_result.state_batch, generic_result.state_batch, atol=1e-6, rtol=1e-6)
    assert torch.equal(fused_result.root_indices, generic_result.root_indices)


def test_bucket_store_subbatch_matches_direct_batch_for_fixed_shape_trees() -> None:
    model = CTreePOModel(CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16)).eval()
    trees = [
        _make_tree(text="abcdefghijklmnop", n_leaves=4),
        _make_tree(text="qrstuvwxyzabcdef", n_leaves=4),
        _make_tree(text="ghijklmnopqrstuv", n_leaves=4),
    ]
    packed_trees = [build_packed_embedding_tree(nodes) for nodes in trees]
    apply_runtime_mode_to_packed_trees(packed_trees, device=torch.device("cpu"))
    bucket_stores = build_packed_tree_bucket_stores(packed_trees, device=torch.device("cpu"))

    assert len(bucket_stores) == 1
    selected = [packed_trees[0], packed_trees[2]]
    from_store = build_packed_tree_batch_from_bucket_store(
        bucket_stores[0],
        selected,
        device=torch.device("cpu"),
    )
    direct = build_packed_tree_batch(selected, device=torch.device("cpu"))

    assert from_store.runtime_stats["packed_bucket_store_hit"] is True
    assert from_store.runtime_stats["packed_bucket_store_mode"] == "staged_rows"
    assert torch.equal(from_store.root_indices, direct.root_indices)
    assert torch.equal(from_store.leaf_node_indices, direct.leaf_node_indices)
    assert torch.allclose(from_store.leaf_embeddings, direct.leaf_embeddings, atol=0.0, rtol=0.0)
    store_result = forward_packed_tree_batch(model, from_store)
    direct_result = forward_packed_tree_batch(model, direct)
    assert torch.allclose(store_result.state_batch, direct_result.state_batch, atol=1e-6, rtol=1e-6)


def test_cpu_runtime_mode_stays_staged() -> None:
    tree = build_packed_embedding_tree(_make_tree(text="abcdefgh", n_leaves=2))
    summary = apply_runtime_mode_to_packed_trees([tree], device=torch.device("cpu"))
    packed_batch = build_packed_tree_batch([tree], device=torch.device("cpu"))

    assert summary["runtime_data_mode"] == "staged"
    assert tree.runtime_data_mode == "staged"
    assert tree.leaf_embeddings_resident is None
    assert packed_batch.runtime_stats["runtime_data_mode"] == "staged"
    assert packed_batch.runtime_stats["host_to_device_bytes"] == 0
    assert packed_batch.runtime_stats["host_to_device_events"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_staged_cuda_runtime_uses_pinned_staging_and_counts_h2d() -> None:
    device = torch.device("cuda")
    tree = build_packed_embedding_tree(_make_tree(text="abcdefghijkl", n_leaves=3))
    summary = apply_runtime_mode_to_packed_trees(
        [tree],
        device=device,
        resident_fraction=0.0,
    )
    packed_batch = build_packed_tree_batch([tree], device=device)

    assert summary["runtime_data_mode"] == "staged"
    assert tree.runtime_data_mode == "staged"
    assert tree.leaf_embeddings_staged is not None and tree.leaf_embeddings_staged.is_pinned()
    assert tree.merge_edge_indices_staged is not None and tree.merge_edge_indices_staged.is_pinned()
    assert tree.merge_left_weights_staged is not None and tree.merge_left_weights_staged.is_pinned()
    assert packed_batch.runtime_stats["runtime_data_mode"] == "staged"
    assert packed_batch.runtime_stats["host_to_device_bytes"] > 0
    assert packed_batch.runtime_stats["host_to_device_events"] > 0
    assert packed_batch.runtime_stats["packed_executor_mode"] in {"generic_packed", "fixed_fused"}
    assert packed_batch.leaf_embeddings.device.type == "cuda"
    if packed_batch.levels:
        assert packed_batch.levels[0].parent_indices.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_resident_cuda_runtime_has_zero_h2d_after_preload() -> None:
    device = torch.device("cuda")
    trees = [
        build_packed_embedding_tree(_make_tree(text="abcdefghijklmnop", n_leaves=4)),
        build_packed_embedding_tree(_make_tree(text="qrstuvwxyzabcdef", n_leaves=4)),
    ]
    summary = apply_runtime_mode_to_packed_trees(trees, device=device)
    if summary["runtime_data_mode"] != "resident":
        pytest.skip("resident preload unavailable for current free VRAM")

    packed_batch = build_packed_tree_batch(trees, device=device)
    bucket_stores = build_packed_tree_bucket_stores(trees, device=device)
    model = CTreePOModel(CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16)).to(device).eval()
    result = forward_packed_tree_batch(model, packed_batch)

    assert packed_batch.runtime_stats["runtime_data_mode"] == "resident"
    assert packed_batch.runtime_stats["host_to_device_bytes"] == 0
    assert packed_batch.runtime_stats["host_to_device_events"] == 0
    assert packed_batch.runtime_stats["resident_store_hits"] == len(trees)
    assert packed_batch.runtime_stats["resident_store_misses"] == 0
    assert packed_batch.runtime_stats["packed_executor_mode"] == "fixed_fused"
    assert result.runtime_stats["host_to_device_events"] == 0
    assert result.runtime_stats["host_to_device_bytes"] == 0
    assert len(bucket_stores) == 1
    assert bucket_stores[0].leaf_dense_resident is not None
    assert bucket_stores[0].bucket_store_mode == "dense_resident"
