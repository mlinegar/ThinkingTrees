from __future__ import annotations

import torch

from src.training.embedding_sketch import (
    EmbeddingSketchConfig,
    MergeableEmbeddingSketch,
    merge_prediction_consistency,
)


def test_encode_windows_merge_matches_full_encoding():
    torch.manual_seed(0)
    cfg = EmbeddingSketchConfig(
        embedding_dim=4,
        state_dim=8,
        phi_hidden_dim=16,
        readout_hidden_dim=8,
        dropout=0.0,
        include_meta=False,
        use_count_feature=True,
    )
    model = MergeableEmbeddingSketch(cfg)

    windows = torch.randn(2, 6, 4)
    full = model.encode_windows(windows, counts=torch.tensor([6, 6]))

    left = model.encode_windows(windows[:, :3, :], counts=torch.tensor([3, 3]))
    right = model.encode_windows(windows[:, 3:, :], counts=torch.tensor([3, 3]))
    merged = left.merge(right)

    assert torch.allclose(merged.sum_phi, full.sum_phi, atol=1e-6)
    assert torch.allclose(merged.count, full.count, atol=1e-6)

    pred_full = model.predict_from_state(full)
    pred_merged = model.predict_from_state(merged)
    assert torch.allclose(pred_full, pred_merged, atol=1e-6)


def test_encode_windows_respects_counts_mask():
    torch.manual_seed(0)
    cfg = EmbeddingSketchConfig(
        embedding_dim=4,
        state_dim=8,
        phi_hidden_dim=16,
        readout_hidden_dim=8,
        dropout=0.0,
        include_meta=False,
        use_count_feature=True,
    )
    model = MergeableEmbeddingSketch(cfg)

    valid = torch.randn(1, 3, 4)
    padded = torch.cat([valid, torch.randn(1, 2, 4)], dim=1)  # [1,5,4], but only first 3 are valid

    state_padded = model.encode_windows(padded, counts=torch.tensor([3]))
    state_valid = model.encode_windows(valid, counts=torch.tensor([3]))

    assert torch.allclose(state_padded.sum_phi, state_valid.sum_phi, atol=1e-6)
    assert torch.allclose(state_padded.count, state_valid.count, atol=1e-6)


def test_forward_with_retrieval_features_and_delta_head():
    torch.manual_seed(0)
    cfg = EmbeddingSketchConfig(
        embedding_dim=4,
        state_dim=8,
        phi_hidden_dim=16,
        readout_hidden_dim=8,
        dropout=0.0,
        include_meta=True,
        use_count_feature=True,
        include_retrieval_features=True,
        retrieval_feature_dim=6,
        include_delta_head=True,
    )
    model = MergeableEmbeddingSketch(cfg)

    windows = torch.randn(2, 5, 4)
    counts = torch.tensor([5, 4], dtype=torch.int64)
    meta = torch.randn(2, 4)
    retrieval = torch.randn(2, 6)
    out = model(
        windows,
        counts=counts,
        meta_embeddings=meta,
        retrieval_features=retrieval,
        return_dict=True,
    )

    assert isinstance(out, dict)
    assert out.get("rile") is not None
    assert out.get("delta") is not None
    assert out["rile"].shape == (2,)
    assert out["delta"].shape == (2,)
    assert torch.all(out["rile"] >= 0.0)
    assert torch.all(out["rile"] <= 1.0)
    assert torch.all(out["delta"] >= -1.0)
    assert torch.all(out["delta"] <= 1.0)


def test_local_law_capabilities_are_explicit():
    cfg = EmbeddingSketchConfig(
        embedding_dim=4,
        state_dim=8,
        phi_hidden_dim=16,
        readout_hidden_dim=8,
        dropout=0.0,
        include_meta=False,
        use_count_feature=True,
    )
    model = MergeableEmbeddingSketch(cfg)

    caps = model.local_law_capabilities()

    assert caps["latent_mergeability_enforced"] is True
    assert caps["theorem_domain_decode_available"] is False
    assert caps["theorem_domain_reencode_available"] is False
    assert caps["supports_resummary_idempotence"] is False
    assert caps["objective_enforces_leaf_preservation"] is False
    assert caps["objective_enforces_merge_preservation_against_span_oracle"] is False
    assert caps["objective_enforces_idempotence"] is False
    assert caps["laws"]["l3_idempotence"]["available"] is False


def test_merge_prediction_consistency_reports_exact_mergeability():
    torch.manual_seed(0)
    cfg = EmbeddingSketchConfig(
        embedding_dim=4,
        state_dim=8,
        phi_hidden_dim=16,
        readout_hidden_dim=8,
        dropout=0.0,
        include_meta=False,
        use_count_feature=True,
    )
    model = MergeableEmbeddingSketch(cfg)

    windows = torch.randn(2, 6, 4)
    counts = torch.tensor([6, 5], dtype=torch.int64)
    stats = merge_prediction_consistency(model, windows, counts=counts)

    assert abs(float(stats["prediction_mae"])) < 1e-6
    assert abs(float(stats["prediction_max_abs"])) < 1e-6
