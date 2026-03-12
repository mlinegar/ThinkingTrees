import hashlib

import numpy as np
import pytest

from src.core.data_models import Tree, leaf, node
from src.training.embedding_sketch import EmbeddingSketchConfig, MergeableEmbeddingSketch
from src.tree.unified_artifacts import (
    attach_chunk_support,
    build_mergeable_phi_state,
    build_mergeable_sum_state_from_embeddings,
    embed_leaf_texts,
    get_root_state,
    predict_root_from_phi_state,
)


class FakeEmbeddingClient:
    """Deterministic embedding client for tests (no server needed)."""

    def __init__(self, dim: int = 8):
        self.dim = int(dim)

    def embed_texts(self, texts):
        out = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8", "surrogatepass")).hexdigest()
            seed = int(digest[:8], 16)
            rng = np.random.RandomState(seed)
            out.append(rng.randn(self.dim).astype(np.float32).tolist())
        return out


def _build_tree_with_boundaries() -> tuple[Tree, list, list]:
    leaves = [
        leaf("alpha", node_id="l0"),
        leaf("bravo", node_id="l1"),
        leaf("charlie", node_id="l2"),
        leaf("delta", node_id="l3"),
    ]
    left = node(leaves[0], leaves[1], summary="left", node_id="i0")
    right = node(leaves[2], leaves[3], summary="right", node_id="i1")
    root = node(left, right, summary="root", node_id="root")

    boundaries = []
    cursor = 0
    for idx, leaf_node in enumerate(leaves):
        start = cursor
        end = start + len(leaf_node.raw_text_span or "")
        boundaries.append(
            {
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
                "token_count": 0,
            }
        )
        cursor = end

    tree = Tree(root=root, rubric="test", metadata={"chunk_boundaries": boundaries})
    return tree, leaves, boundaries


def test_attach_chunk_support_sets_leaf_and_internal_spans():
    tree, leaves, boundaries = _build_tree_with_boundaries()
    attach_chunk_support(tree)

    for leaf_node, boundary in zip(leaves, boundaries):
        assert leaf_node.metadata["chunk_index"] == boundary["chunk_index"]
        assert leaf_node.metadata["char_start"] == boundary["start_char"]
        assert leaf_node.metadata["char_end"] == boundary["end_char"]

    root = tree.root
    assert root is not None
    assert root.metadata["char_start"] == 0
    assert root.metadata["char_end"] == boundaries[-1]["end_char"]


def test_mergeable_sum_state_matches_sum_of_leaf_embeddings():
    tree, leaves, _boundaries = _build_tree_with_boundaries()
    client = FakeEmbeddingClient(dim=8)
    embedded = embed_leaf_texts(tree, embedding_client=client, overwrite=True)
    assert embedded == len(leaves)

    build_mergeable_sum_state_from_embeddings(tree, overwrite=True)
    state = get_root_state(tree)
    assert state is not None
    assert state.count == len(leaves)

    expected = np.zeros((8,), dtype=np.float32)
    for leaf_node in tree.leaves:
        expected += np.asarray(leaf_node.metadata["leaf_embedding"], dtype=np.float32)
    assert np.allclose(state.sum_vec, expected, atol=1e-6)


def test_phi_state_prediction_matches_direct_forward_pass():
    tree, _leaves, _boundaries = _build_tree_with_boundaries()
    client = FakeEmbeddingClient(dim=8)
    embed_leaf_texts(tree, embedding_client=client, overwrite=True)

    import torch

    torch.manual_seed(0)
    cfg = EmbeddingSketchConfig(
        embedding_dim=8,
        state_dim=6,
        phi_hidden_dim=12,
        readout_hidden_dim=8,
        dropout=0.0,
        include_meta=False,
        use_count_feature=True,
    )
    model = MergeableEmbeddingSketch(cfg).eval()

    build_mergeable_phi_state(tree, model=model, overwrite=True, device="cpu")
    pred_from_state = predict_root_from_phi_state(tree, model=model, device="cpu")
    assert pred_from_state is not None

    mat = np.stack(
        [np.asarray(leaf_node.metadata["leaf_embedding"], dtype=np.float32) for leaf_node in tree.leaves],
        axis=0,
    )
    xb = torch.from_numpy(mat).to(dtype=torch.float32).unsqueeze(0)
    cb = torch.tensor([int(mat.shape[0])], dtype=torch.int64)
    with torch.no_grad():
        direct = float(model(xb, counts=cb).detach().cpu().numpy().reshape(-1)[0])

    assert float(pred_from_state) == pytest.approx(direct, abs=1e-5)

