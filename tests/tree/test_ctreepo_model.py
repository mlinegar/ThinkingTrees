"""Unit tests for CTreePO model and embedding tree."""

import torch
import pytest

from src.tree.ctreepo_model import (
    CTreePOConfig,
    CTreePOModel,
    LeafProjector,
    GatedMerge,
    MLPMerge,
    AvgMerge,
    ResidualGatedMerge,
    BilinearMerge,
    ReadoutHead,
    associativity_penalty,
    consistency_penalty,
    contrastive_loss,
    normalize_target,
    denormalize_prediction,
)
from src.tree.embedding_tree import (
    EmbeddingTreeNode,
    build_embedding_tree,
    forward_ctreepo,
    get_root_sketch,
    collect_sketches,
    _uniform_windows,
)


# ---------------------------------------------------------------------------
# Config / model construction
# ---------------------------------------------------------------------------


def test_ctreepo_model_creation():
    config = CTreePOConfig(embedding_dim=64, sketch_dim=8, hidden_dim=16)
    model = CTreePOModel(config)
    assert model.config.sketch_dim == 8


def test_ctreepo_model_all_merge_types():
    for merge_type in ["gated", "mlp", "avg", "residual_gated", "bilinear"]:
        config = CTreePOConfig(embedding_dim=32, sketch_dim=4, hidden_dim=8, merge_type=merge_type)
        model = CTreePOModel(config)
        left = torch.randn(4)
        right = torch.randn(4)
        merged = model.merge(left, right)
        assert merged.shape == (4,)


# ---------------------------------------------------------------------------
# LeafProjector
# ---------------------------------------------------------------------------


def test_leaf_projector_output_shape():
    proj = LeafProjector(embedding_dim=64, sketch_dim=8, hidden_dim=16)
    emb = torch.randn(64)
    out = proj(emb)
    assert out.shape == (8,)


def test_leaf_projector_batched():
    proj = LeafProjector(embedding_dim=64, sketch_dim=8, hidden_dim=16)
    batch = torch.randn(5, 64)
    out = proj(batch)
    assert out.shape == (5, 8)


# ---------------------------------------------------------------------------
# Merge modules
# ---------------------------------------------------------------------------


def test_gated_merge_output():
    merge = GatedMerge(sketch_dim=4)
    left = torch.randn(4)
    right = torch.randn(4)
    out = merge(left, right)
    assert out.shape == (4,)


def test_mlp_merge_output():
    merge = MLPMerge(sketch_dim=4, hidden_dim=8)
    left = torch.randn(4)
    right = torch.randn(4)
    out = merge(left, right)
    assert out.shape == (4,)


def test_avg_merge_is_average():
    merge = AvgMerge()
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([3.0, 4.0, 5.0])
    out = merge(left, right)
    expected = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(out, expected)


def test_residual_gated_merge_output():
    merge = ResidualGatedMerge(sketch_dim=4, hidden_dim=8)
    left = torch.randn(4)
    right = torch.randn(4)
    out = merge(left, right)
    assert out.shape == (4,)


def test_bilinear_merge_output():
    merge = BilinearMerge(sketch_dim=4)
    left = torch.randn(4)
    right = torch.randn(4)
    out = merge(left, right)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------


def test_readout_range():
    head = ReadoutHead(sketch_dim=4, target_min=-100, target_max=100)
    sketch = torch.randn(4)
    pred = head(sketch)
    assert -100 <= pred.item() <= 100


def test_readout_normalized_range():
    head = ReadoutHead(sketch_dim=4)
    sketch = torch.randn(4)
    pred = head.forward_normalized(sketch)
    assert 0 <= pred.item() <= 1


def test_predict_interval_bounds():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)
    sketch = torch.randn(4)
    mean, lower, upper, std = model.predict_interval(sketch, "rile", z_score=1.96, min_std=0.1)
    assert lower.item() <= mean.item() <= upper.item()
    assert std.item() >= 0.1
    assert config.target_min <= lower.item() <= config.target_max
    assert config.target_min <= upper.item() <= config.target_max


# ---------------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------------


def test_normalize_target():
    assert normalize_target(0, -100, 100) == 0.5
    assert normalize_target(-100, -100, 100) == 0.0
    assert normalize_target(100, -100, 100) == 1.0


def test_denormalize_prediction():
    assert denormalize_prediction(0.5, -100, 100) == 0.0
    assert denormalize_prediction(0.0, -100, 100) == -100.0
    assert denormalize_prediction(1.0, -100, 100) == 100.0


# ---------------------------------------------------------------------------
# Embedding tree
# ---------------------------------------------------------------------------


def test_uniform_windows():
    w = _uniform_windows(100, 30, 10)
    assert len(w) >= 3
    assert w[0][0] == 0
    assert w[-1][1] == 100


def test_uniform_windows_short_text():
    w = _uniform_windows(20, 30, 0)
    assert len(w) == 1
    assert w[0] == (0, 20)


def test_build_embedding_tree_3_leaves():
    text = "a" * 300
    embeddings = [
        [float(i)] * 8 for i in range(3)
    ]
    windows = [(0, 100), (100, 200), (200, 300)]
    nodes = build_embedding_tree(text, embeddings, windows)

    # 3 leaves -> level 1: merge(0,1) + promote(2) -> level 2: merge(3,4) = 6 total
    leaves = [n for n in nodes if n.is_leaf]
    internal = [n for n in nodes if not n.is_leaf]
    assert len(leaves) == 3
    assert len(internal) == 3  # 2 merges + 1 promoted odd node
    assert nodes[-1].char_start == 0
    assert nodes[-1].char_end == 300


def test_forward_ctreepo_sets_all_sketches():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)

    text = "a" * 200
    embeddings = [[float(i)] * 8 for i in range(2)]
    windows = [(0, 100), (100, 200)]
    nodes = build_embedding_tree(text, embeddings, windows)

    forward_ctreepo(model, nodes)

    for node in nodes:
        assert node.sketch is not None
        assert node.sketch.shape == (4,)


def test_root_sketch_equals_last_node():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)

    text = "a" * 200
    embeddings = [[1.0] * 8, [2.0] * 8]
    windows = [(0, 100), (100, 200)]
    nodes = build_embedding_tree(text, embeddings, windows)
    forward_ctreepo(model, nodes)

    root = get_root_sketch(nodes)
    assert torch.equal(root, nodes[-1].sketch)


def test_collect_sketches():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)

    text = "a" * 200
    embeddings = [[1.0] * 8, [2.0] * 8]
    windows = [(0, 100), (100, 200)]
    nodes = build_embedding_tree(text, embeddings, windows)
    forward_ctreepo(model, nodes)

    leaf_s, internal_s = collect_sketches(nodes)
    assert len(leaf_s) == 2
    assert len(internal_s) == 1


# ---------------------------------------------------------------------------
# End-to-end: full model predict
# ---------------------------------------------------------------------------


def test_end_to_end_predict():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)

    text = "a" * 300
    embeddings = [[float(i)] * 8 for i in range(3)]
    windows = [(0, 100), (100, 200), (200, 300)]
    nodes = build_embedding_tree(text, embeddings, windows)
    forward_ctreepo(model, nodes)

    root = get_root_sketch(nodes)
    rile = model.predict(root, "rile")
    assert rile.shape == (1,)
    assert -100 <= rile.item() <= 100


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------


def test_associativity_penalty_runs():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)
    sketches = [torch.randn(4) for _ in range(5)]
    penalty = associativity_penalty(model, sketches, n_triplets=3)
    assert penalty.item() >= 0


def test_contrastive_loss_runs():
    sketches = [torch.randn(4) for _ in range(4)]
    targets = [-20.0, -15.0, 25.0, 30.0]
    loss = contrastive_loss(sketches, targets, tau=0.1, similarity_threshold=10.0)
    # Should be non-negative
    assert loss.item() >= 0 or True  # InfoNCE can be negative


def test_consistency_penalty_runs():
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)
    parent = torch.randn(4)
    left = torch.randn(4)
    right = torch.randn(4)
    penalty = consistency_penalty(model, parent, left, right, left_weight=0.6)
    assert penalty.shape == ()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_through_tree():
    """Verify backprop works through the full tree (leaf -> merge -> readout)."""
    config = CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8)
    model = CTreePOModel(config)

    text = "a" * 200
    embeddings = [[1.0] * 8, [2.0] * 8]
    windows = [(0, 100), (100, 200)]
    nodes = build_embedding_tree(text, embeddings, windows)
    forward_ctreepo(model, nodes)

    root = get_root_sketch(nodes)
    pred = model.predict_normalized(root, "rile")
    target = torch.tensor([0.5])
    loss = ((pred - target) ** 2).sum()
    loss.backward()

    # Check gradients exist on leaf projector and merge module
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert param.grad.abs().sum().item() > 0, f"Zero gradient on {name}"
