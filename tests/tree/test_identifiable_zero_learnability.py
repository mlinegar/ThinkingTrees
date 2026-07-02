from __future__ import annotations

import math

import numpy as np
import pytest

from src.tree.segmented_lda_ctreepo_simulation import (
    SegmentedLDACtreePOConfig,
    run_segmented_lda_ctreepo_simulation,
)
from src.tree.markov_changepoint_ops_count_simulation import (
    OPSCountConfig,
    run_markov_changepoint_ops_count_experiment,
)
from src.tree.segment_lda_ops_weight_recovery_simulation import estimate_topic_distributions


def test_ctree_test_set_signature_stable_across_train_docs() -> None:
    seed = 123
    cfg_small = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=64,
        n_books_train=32,
        n_books_test=64,
        min_segments=4,
        max_segments=4,
        min_seg_tokens=12,
        max_seg_tokens=12,
        fixed_leaf_tokens=8,
        topic_phi_estimator="true",
        topic_phi_docs=0,
        topic_phi_permute=False,
        calibration_leaf_query_rate=0.10,
        eval_leaf_query_rate=0.0,
        eval_internal_query_rate=0.0,
        selection_audit_trials=0,
        seed=seed,
    )
    cfg_large = SegmentedLDACtreePOConfig(**{**cfg_small.__dict__, "n_books_train": 64})

    s_small = run_segmented_lda_ctreepo_simulation(cfg_small)
    s_large = run_segmented_lda_ctreepo_simulation(cfg_large)

    sig_small = str((s_small.topic_meta or {}).get("corpus_signature_test") or "")
    sig_large = str((s_large.topic_meta or {}).get("corpus_signature_test") or "")
    assert sig_small, "missing corpus_signature_test"
    assert sig_large, "missing corpus_signature_test"
    assert sig_small == sig_large


def test_ctree_bag_of_words_test_set_signature_stable_across_train_docs() -> None:
    seed = 123
    cfg_small = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=64,
        topic_process="bag_of_words",
        n_books_train=32,
        n_books_test=64,
        min_segments=4,
        max_segments=4,
        min_seg_tokens=12,
        max_seg_tokens=12,
        fixed_leaf_tokens=8,
        topic_phi_estimator="true",
        topic_phi_docs=0,
        topic_phi_permute=False,
        calibration_leaf_query_rate=0.10,
        eval_leaf_query_rate=0.0,
        eval_internal_query_rate=0.0,
        selection_audit_trials=0,
        seed=seed,
    )
    cfg_large = SegmentedLDACtreePOConfig(**{**cfg_small.__dict__, "n_books_train": 64})

    s_small = run_segmented_lda_ctreepo_simulation(cfg_small)
    s_large = run_segmented_lda_ctreepo_simulation(cfg_large)

    sig_small = str((s_small.topic_meta or {}).get("corpus_signature_test") or "")
    sig_large = str((s_large.topic_meta or {}).get("corpus_signature_test") or "")
    assert sig_small, "missing corpus_signature_test"
    assert sig_large, "missing corpus_signature_test"
    assert sig_small == sig_large


def test_markov_ops_count_smoke_runs() -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=32,
        min_tokens=96,
        max_tokens=96,
        min_segments=6,
        max_segments=8,
        min_seg_len=4,
        max_seg_len=24,
        fixed_leaf_tokens=8,
        train_docs=30,
        test_docs=40,
        model_family="neural",
        feature_mode="full",
        state_dim=16,
        hidden_dim=32,
        n_epochs=1,
        batch_size=8,
        audit_policy="fraction",
        audit_fraction=0.10,
        leaf_query_rate=1.0,
        include_root_query=True,
        use_cuda=False,
        torch_threads=1,
        seed=0,
    )
    summary = run_markov_changepoint_ops_count_experiment(cfg)
    learned = ((summary.metrics or {}).get("learned") or {}) if isinstance(summary.metrics, dict) else {}
    assert isinstance(learned, dict)
    root_mae = float(learned.get("root_mae", float("nan")))
    assert math.isfinite(root_mae)
    assert int(learned.get("n_docs", 0)) == 40


def test_topic_phi_estimator_sklearn_lda_runs_small() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    k = 4
    v = 48
    d_docs = 80
    doc_len = 64

    topics_true = [rng.dirichlet(np.full((v,), 0.20, dtype=np.float64)).astype(np.float64) for _ in range(k)]
    docs_tokens = []
    alpha = np.full((k,), 0.20, dtype=np.float64)
    for _ in range(d_docs):
        theta = rng.dirichlet(alpha).astype(np.float64)
        z = rng.choice(np.arange(k), size=doc_len, p=theta).astype(np.int64)
        w = [int(rng.choice(np.arange(v), p=np.asarray(topics_true[int(t)], dtype=np.float64))) for t in z.tolist()]
        docs_tokens.append(w)

    topics_est, meta, perm = estimate_topic_distributions(
        topics_true,
        estimator="sklearn_lda",
        n_docs=d_docs,
        doc_topic_concentration=0.20,
        tlda_delta=0.10,
        tlda_rate_constant=1.0,
        sigmaK_floor=1e-6,
        permute=False,
        seed=0,
        topic_word_concentration=0.20,
        docs_tokens=docs_tokens,
    )
    assert len(topics_est) == k
    assert tuple(int(np.asarray(t).size) for t in topics_est) == (v,) * k
    assert str(meta.get("topic_phi_estimator")) == "sklearn_lda"
    assert len(perm) == k
