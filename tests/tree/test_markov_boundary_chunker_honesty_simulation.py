import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_boundary_chunker_honesty_simulation import (  # noqa: E402
    run_markov_chunker_honesty_experiment,
)
from src.tree.markov_boundary_honesty_simulation import MarkovBoundaryConfig  # noqa: E402


def test_markov_chunker_honesty_bridge_leakage_improves():
    cfg = MarkovBoundaryConfig(
        n_classes=4,
        vocab_size=96,
        min_tokens=96,
        max_tokens=96,
        min_leaf_tokens=8,
        max_leaf_tokens=32,
        fixed_leaf_tokens=16,
        train_docs=120,
        test_docs=60,
        sinkhorn_iters=30,
        transition_log_std=1.35,
        window_size=1,
        boundary_emb_dim=24,
        boundary_hidden_dim=48,
        boundary_batch_size=256,
        boundary_max_train_samples=60000,
        n_epochs=6,
        lr=1e-3,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_chunker_honesty_experiment(cfg, token_char_width=300)
    fixed = summary.metrics["fixed"]
    honest = summary.metrics["chunker_honest"]
    leaky = summary.metrics["chunker_leaky"]

    # Adaptive chunking with predicted signals should improve over fixed chunking.
    assert honest.mean_boundary_cost < fixed.mean_boundary_cost

    # Using evaluation-role oracle signals for chunking is a leakage path.
    assert leaky.mean_boundary_cost < honest.mean_boundary_cost

    # Sanity check: boundary cost should be non-degenerate.
    assert fixed.mean_boundary_cost > 0.1

