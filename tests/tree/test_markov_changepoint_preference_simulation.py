import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_changepoint_preference_simulation import (  # noqa: E402
    MarkovChangepointPreferenceConfig,
    run_markov_changepoint_preference_experiment,
)


def test_markov_changepoint_preference_oracle_cut_is_optimal_when_feasible():
    cfg = MarkovChangepointPreferenceConfig(
        n_regimes=3,
        vocab_size=32,
        min_tokens=32,
        max_tokens=32,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=8,
        min_leaf_tokens=8,
        max_leaf_tokens=8,
        fixed_leaf_tokens=8,
        token_char_width=120,
        boundary_tolerance_tokens=0,
        train_docs=40,
        test_docs=20,
        sinkhorn_iters=10,
        transition_log_std=1.0,
        window_size=2,
        boundary_emb_dim=16,
        boundary_hidden_dim=32,
        boundary_batch_size=128,
        boundary_max_train_samples=5000,
        balance_training=True,
        n_epochs=3,
        lr=2e-3,
        dpo_beta=1.0,
        dpo_policy_sharpness=2.5,
        dpo_negatives_per_doc=6,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_preference_experiment(cfg)
    oracle = summary.metrics["oracle_cut"]

    assert oracle.total_true_boundaries > 0
    assert oracle.boundary_f1 == pytest.approx(1.0, abs=1e-12)
    assert oracle.predicted_to_true_ratio == pytest.approx(1.0, abs=1e-12)
    assert oracle.mean_abs_count_error == pytest.approx(0.0, abs=1e-12)
    assert oracle.mean_dpo_loss_gap_to_opt == pytest.approx(0.0, abs=1e-8)

