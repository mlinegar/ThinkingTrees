import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_changepoint_honesty_simulation import (  # noqa: E402
    MarkovChangepointConfig,
    run_markov_changepoint_honesty_experiment,
)


def test_markov_changepoint_honesty_ordering_and_downstream_gain():
    cfg = MarkovChangepointConfig(
        n_regimes=4,
        vocab_size=96,
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=5,
        min_seg_len=8,
        max_seg_len=24,
        min_leaf_tokens=1,
        max_leaf_tokens=8,
        fixed_leaf_tokens=2,
        token_char_width=300,
        boundary_tolerance_tokens=2,
        train_docs=120,
        test_docs=60,
        sinkhorn_iters=30,
        transition_log_std=1.35,
        window_size=2,
        boundary_emb_dim=24,
        boundary_hidden_dim=48,
        boundary_batch_size=256,
        boundary_max_train_samples=60000,
        balance_training=True,
        n_epochs=6,
        lr=1e-3,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_honesty_experiment(cfg)
    fixed = summary.metrics["fixed"]
    honest = summary.metrics["chunker_honest"]
    leaky = summary.metrics["chunker_leaky"]

    assert fixed.total_true_boundaries > 0

    # Oracle/evaluation-role leakage should upper-bound honest boundary detection.
    assert leaky.boundary_f1 >= honest.boundary_f1 >= fixed.boundary_f1

    # Honest chunking should improve at least one downstream distortion surrogate.
    assert (honest.mean_boundary_cost < fixed.mean_boundary_cost) or (
        honest.mean_l1 < fixed.mean_l1
    )
