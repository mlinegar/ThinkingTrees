import numpy as np

from src.tree.tensor_lda_book_weight_benchmark import (
    TensorLDABookBenchmarkConfig,
    _run_selection_bias_audit,
    generate_synthetic_books,
    run_tensor_lda_book_weight_benchmark,
    sample_topic_word_matrix,
)


def test_tensor_lda_dgp_shapes_and_simplex_constraints():
    cfg = TensorLDABookBenchmarkConfig(
        n_topics=4,
        vocab_size=80,
        chapters_per_book=6,
        tokens_per_chapter=32,
        n_books_train=8,
        n_books_test=8,
        seed=11,
    )
    rng = np.random.default_rng(cfg.seed)
    topic_word = sample_topic_word_matrix(cfg, rng=rng)
    data = generate_synthetic_books(cfg, topic_word=topic_word, n_books=7, rng=rng)

    assert data.topic_word.shape == (cfg.n_topics, cfg.vocab_size)
    assert data.book_topic_weights.shape == (7, cfg.n_topics)
    assert data.chapter_topic_weights.shape == (7, cfg.chapters_per_book, cfg.n_topics)
    assert data.chapter_word_counts.shape == (7, cfg.chapters_per_book, cfg.vocab_size)

    assert np.allclose(np.sum(data.topic_word, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.sum(data.book_topic_weights, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.sum(data.chapter_topic_weights, axis=2), 1.0, atol=1e-8)
    assert np.all(np.sum(data.chapter_word_counts, axis=2) == cfg.tokens_per_chapter)


def test_calibration_improves_proxy_tree_root_error():
    cfg = TensorLDABookBenchmarkConfig(
        n_topics=4,
        vocab_size=120,
        chapters_per_book=8,
        tokens_per_chapter=64,
        n_books_train=128,
        n_books_test=96,
        anchor_words_per_topic=10,
        calibration_leaf_query_rate=0.50,
        calibration_policy="entropy",
        eval_internal_query_rate=0.0,
        eval_internal_query_design="none",
        selection_audit_trials=0,
        seed=3,
    )
    out = run_tensor_lda_book_weight_benchmark(cfg)

    proxy_err = out.metrics["ctree_proxy"].root_l1_mean
    cal_err = out.metrics["ctree_calibrated"].root_l1_mean
    assert cal_err < proxy_err


def test_full_internal_guidance_recovers_exact_tree_root():
    cfg = TensorLDABookBenchmarkConfig(
        n_topics=4,
        vocab_size=100,
        chapters_per_book=8,
        tokens_per_chapter=64,
        n_books_train=64,
        n_books_test=64,
        calibration_leaf_query_rate=0.25,
        eval_internal_query_rate=1.0,
        eval_internal_query_design="risk",
        selection_audit_trials=0,
        seed=7,
    )
    out = run_tensor_lda_book_weight_benchmark(cfg)

    base = out.metrics["ctree_calibrated"].root_l1_mean
    guided = out.metrics["ctree_calibrated_budgeted"].root_l1_mean
    assert guided <= base + 1e-9
    assert guided < 1e-9


def test_selection_audit_ipw_unbiased_naive_biased():
    n = 2000
    y = np.zeros((n,), dtype=np.float64)
    y[: n // 2] = 1.0
    scores = 0.01 + y
    viol = y.copy()

    out = _run_selection_bias_audit(
        discrepancies=y,
        violations=viol,
        scores=scores,
        threshold=0.5,
        trials=250,
        sample_rate=0.05,
        pi_min=0.01,
        seed=0,
    )

    assert out.naive_mean_discrepancy.bias > 0.05
    assert abs(out.ipw_mean_discrepancy.bias) < 0.03
    assert abs(out.dsl0_mean_discrepancy.bias) < 0.03
    assert out.dsl_oracle_mean_discrepancy.variance < out.ipw_mean_discrepancy.variance

