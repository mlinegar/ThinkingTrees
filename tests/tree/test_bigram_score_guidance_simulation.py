import numpy as np

from src.tree.bigram_score_guidance_simulation import (
    bigram_counts_sparse,
    fit_ridge_from_span_queries,
    oracle_bigram_score,
    oracle_merge_score,
)


def test_oracle_merge_score_matches_full_score():
    vocab_size = 5
    d = vocab_size * vocab_size
    w_true = np.zeros((d,), dtype=np.float64)
    w_true[2 * vocab_size + 3] = 1.25
    w_true[4 * vocab_size + 0] = -0.5

    left = (2, 3, 4)
    right = (0, 2)
    merged = left + right

    score_full = oracle_bigram_score(merged, w_true=w_true, vocab_size=vocab_size)
    score_merge = oracle_merge_score(left, right, w_true=w_true, vocab_size=vocab_size)
    assert score_merge == score_full


def test_leaf_only_queries_cannot_identify_cross_leaf_boundary_weight():
    # Two leaves of length 2: [0,1] and [2,0].
    # The only nonzero weight is for the boundary bigram (1,2), which occurs only across leaves.
    vocab_size = 3
    d = vocab_size * vocab_size
    w_true = np.zeros((d,), dtype=np.float64)
    boundary_idx = 1 * vocab_size + 2
    w_true[boundary_idx] = 3.0

    leaf1 = (0, 1)
    leaf2 = (2, 0)
    doc = leaf1 + leaf2

    y_leaf1 = oracle_bigram_score(leaf1, w_true=w_true, vocab_size=vocab_size)
    y_leaf2 = oracle_bigram_score(leaf2, w_true=w_true, vocab_size=vocab_size)
    y_doc = oracle_bigram_score(doc, w_true=w_true, vocab_size=vocab_size)
    assert y_leaf1 == 0.0
    assert y_leaf2 == 0.0
    assert y_doc == 3.0

    # Fit using only leaf queries (base supervision): boundary weight is unobserved.
    w_hat_leaf = fit_ridge_from_span_queries(
        [(leaf1, y_leaf1), (leaf2, y_leaf2)],
        vocab_size=vocab_size,
        ridge_lambda=1e-6,
    )
    assert abs(float(w_hat_leaf[boundary_idx])) < 1e-8

    idx, vals = bigram_counts_sparse(doc, vocab_size=vocab_size)
    y_pred_leaf = float(np.dot(w_hat_leaf[idx], vals)) if idx.size else 0.0
    assert y_pred_leaf == 0.0

    # Add one internal-node/root query spanning both leaves: boundary becomes identifiable.
    w_hat_root = fit_ridge_from_span_queries(
        [(leaf1, y_leaf1), (leaf2, y_leaf2), (doc, y_doc)],
        vocab_size=vocab_size,
        ridge_lambda=1e-6,
    )
    y_pred_root = float(np.dot(w_hat_root[idx], vals)) if idx.size else 0.0
    assert abs(y_pred_root - y_doc) < 1e-3

