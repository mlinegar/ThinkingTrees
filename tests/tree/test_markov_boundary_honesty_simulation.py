import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_boundary_honesty_simulation import (  # noqa: E402
    MarkovBoundaryConfig,
    run_markov_boundary_experiment,
)


def test_markov_boundary_oracle_beats_worst_and_learned_beats_fixed():
    cfg = MarkovBoundaryConfig(
        n_classes=4,
        vocab_size=96,
        min_tokens=96,
        max_tokens=96,
        min_leaf_tokens=8,
        max_leaf_tokens=16,
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

    summary = run_markov_boundary_experiment(cfg)
    fixed = summary.metrics["fixed"]
    learned = summary.metrics["learned"]
    oracle = summary.metrics["oracle"]
    worst = summary.metrics["worst"]

    assert oracle.mean_l1 < worst.mean_l1
    assert oracle.mean_boundary_cost < worst.mean_boundary_cost

    # Learned boundaries should reduce the (true) boundary cost surrogate vs fixed-length chunking.
    assert learned.mean_boundary_cost < fixed.mean_boundary_cost

    # Sanity check: the boundary cost task should be non-degenerate.
    assert fixed.mean_boundary_cost > 0.1
