import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_changepoint_cut_budget_simulation import (  # noqa: E402
    MarkovChangepointCutBudgetConfig,
    run_markov_changepoint_cut_budget_experiment,
)


def test_markov_changepoint_cut_budget_oracle_opt_hits_zero_hamming_when_feasible():
    cfg = MarkovChangepointCutBudgetConfig(
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
        max_cuts=None,  # use fixed cut budget (should equal true boundary count here)
        calibrate_prior=True,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_cut_budget_experiment(cfg)
    oracle = summary.metrics["oracle_opt"]

    assert oracle.total_true_boundaries > 0
    assert oracle.boundary_f1 == pytest.approx(1.0, abs=1e-12)
    assert oracle.predicted_to_true_ratio == pytest.approx(1.0, abs=1e-12)
    assert oracle.mean_hamming_loss == pytest.approx(0.0, abs=1e-12)
    assert oracle.mean_hamming_gap_to_oracle == pytest.approx(0.0, abs=1e-12)
    assert oracle.mean_theory_gap_upper_bound == pytest.approx(0.0, abs=1e-12)


def test_markov_changepoint_cut_budget_full_oracle_guidance_matches_oracle_opt():
    cfg = MarkovChangepointCutBudgetConfig(
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
        n_epochs=2,
        lr=2e-3,
        max_cuts=None,
        guidance_multipliers=(20.0,),
        guidance_strategies=("random", "active"),
        guidance_rounds=3,
        calibrate_prior=True,
        calibrate_pos_weight=True,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_cut_budget_experiment(cfg)
    oracle = summary.metrics["oracle_opt"]

    guided_random = summary.metrics["dp_guided_random_q2000"]
    guided_active = summary.metrics["dp_guided_active_q2000"]
    for guided in (guided_random, guided_active):
        assert guided.boundary_f1 == pytest.approx(1.0, abs=1e-12)
        assert guided.predicted_to_true_ratio == pytest.approx(1.0, abs=1e-12)
        assert guided.mean_hamming_loss == pytest.approx(oracle.mean_hamming_loss, abs=1e-12)
        assert guided.mean_hamming_gap_to_oracle == pytest.approx(0.0, abs=1e-12)
        assert guided.mean_theory_gap_upper_bound == pytest.approx(0.0, abs=1e-12)
        assert guided.mean_oracle_queries_used == pytest.approx(31.0, abs=1e-12)


def test_markov_changepoint_cut_budget_full_oracle_guidance_per_leaf_matches_oracle_opt():
    cfg = MarkovChangepointCutBudgetConfig(
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
        n_epochs=2,
        lr=2e-3,
        max_cuts=None,
        guidance_per_leaf=(8.0,),
        guidance_strategies=("random", "active"),
        guidance_rounds=3,
        calibrate_prior=True,
        calibrate_pos_weight=True,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_cut_budget_experiment(cfg)
    oracle = summary.metrics["oracle_opt"]

    guided_random = summary.metrics["dp_guided_random_l800"]
    guided_active = summary.metrics["dp_guided_active_l800"]
    for guided in (guided_random, guided_active):
        assert guided.boundary_f1 == pytest.approx(1.0, abs=1e-12)
        assert guided.predicted_to_true_ratio == pytest.approx(1.0, abs=1e-12)
        assert guided.mean_hamming_loss == pytest.approx(oracle.mean_hamming_loss, abs=1e-12)
        assert guided.mean_hamming_gap_to_oracle == pytest.approx(0.0, abs=1e-12)
        assert guided.mean_theory_gap_upper_bound == pytest.approx(0.0, abs=1e-12)
        assert guided.mean_oracle_queries_used == pytest.approx(31.0, abs=1e-12)


def test_markov_changepoint_cut_budget_full_tree_guidance_matches_oracle_opt():
    cfg = MarkovChangepointCutBudgetConfig(
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
        n_epochs=2,
        lr=2e-3,
        max_cuts=None,
        guidance_per_leaf=(2.0,),
        guidance_strategies=("random", "active"),
        guidance_interface="tree",
        guidance_rounds=3,
        calibrate_prior=True,
        calibrate_pos_weight=True,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_cut_budget_experiment(cfg)
    oracle = summary.metrics["oracle_opt"]

    guided_random = summary.metrics["dp_guided_random_l200"]
    guided_active = summary.metrics["dp_guided_active_l200"]
    for guided in (guided_random, guided_active):
        assert guided.boundary_f1 == pytest.approx(1.0, abs=1e-12)
        assert guided.predicted_to_true_ratio == pytest.approx(1.0, abs=1e-12)
        assert guided.mean_hamming_loss == pytest.approx(oracle.mean_hamming_loss, abs=1e-12)
        assert guided.mean_hamming_gap_to_oracle == pytest.approx(0.0, abs=1e-12)
        assert guided.mean_theory_gap_upper_bound == pytest.approx(0.0, abs=1e-12)
        assert guided.mean_oracle_queries_used == pytest.approx(7.0, abs=1e-12)


def test_markov_changepoint_cut_budget_theory_bound_upper_bounds_gap():
    cfg = MarkovChangepointCutBudgetConfig(
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
        n_epochs=2,
        lr=2e-3,
        max_cuts=None,
        guidance_multipliers=(0.5, 1.0),
        guidance_strategies=("random", "active"),
        guidance_rounds=2,
        calibrate_prior=True,
        calibrate_pos_weight=True,
        seed=0,
        use_cuda=False,
        torch_threads=0,
    )

    summary = run_markov_changepoint_cut_budget_experiment(cfg)
    for policy in (
        "dp_honest",
        "dp_guided_random_q50",
        "dp_guided_random_q100",
        "dp_guided_active_q50",
        "dp_guided_active_q100",
        "oracle_opt",
    ):
        m = summary.metrics[policy]
        assert m.mean_hamming_gap_to_oracle <= m.mean_theory_gap_upper_bound + 1e-9
