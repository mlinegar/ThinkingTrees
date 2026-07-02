# OLD_: archived 2026-07-02; tests OLD_hll_merge_learning_simulation.py (treepo.bench.hll_merge_learning removed upstream). Kept for reference; do not run.
import math

import pytest

torch = pytest.importorskip("torch")

from src.tree.hll_merge_learning_simulation import (  # noqa: E402
    ExactMaxMerger,
    HLLMergeLearningConfig,
    MeanMerger,
    evaluate_hll_baseline,
    generate_token_stream_docs,
    merge_leaf_states,
    run_hll_merge_learning_experiment,
)


def test_exact_max_merger_is_schedule_invariant_on_registers():
    merger = ExactMaxMerger()
    leaves = [
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([5.0, 1.0], dtype=torch.float32),
        torch.tensor([3.0, 9.0], dtype=torch.float32),
    ]
    expected = torch.tensor([5.0, 9.0], dtype=torch.float32)

    roots = [
        merge_leaf_states(merger, leaves, schedule=sched)
        for sched in ("balanced", "left_to_right", "right_to_left")
    ]
    for r in roots:
        assert torch.allclose(r, expected)


def test_mean_merger_is_schedule_sensitive():
    merger = MeanMerger()
    leaves = [
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([5.0], dtype=torch.float32),
        torch.tensor([9.0], dtype=torch.float32),
    ]
    left = merge_leaf_states(merger, leaves, schedule="left_to_right")
    right = merge_leaf_states(merger, leaves, schedule="right_to_left")
    assert float(left.item()) == pytest.approx(6.0)
    assert float(right.item()) == pytest.approx(4.0)
    assert not torch.allclose(left, right)


def test_hll_baseline_has_zero_schedule_spread_and_matches_theory_formula():
    docs = generate_token_stream_docs(
        64,
        universe_size=2048,
        min_tokens=256,
        max_tokens=256,
        leaf_size=64,
        zipf_alphas=(0.8,),
        seed=7,
    )
    precision = 6
    baseline = evaluate_hll_baseline(docs, precision=precision, hash_bits=64)
    assert baseline.metrics.schedule_spread_mean == pytest.approx(0.0)

    expected_rse = 1.04 / math.sqrt(float(2 ** precision))
    assert baseline.rse_theory == pytest.approx(expected_rse)


def test_hll_baseline_weighting_views_align_when_lengths_are_constant():
    docs = generate_token_stream_docs(
        40,
        universe_size=1024,
        min_tokens=256,
        max_tokens=256,
        leaf_size=64,
        zipf_alphas=(1.0,),
        seed=11,
    )
    baseline = evaluate_hll_baseline(docs, precision=6, hash_bits=64)
    assert baseline.weighting_views is not None
    assert baseline.legacy_weighting_mode == "doc"
    doc_rmse = float(baseline.weighting_views["doc"]["relative_rmse"]["mean_hat"])
    leaf_rmse = float(baseline.weighting_views["leaf"]["relative_rmse"]["mean_hat"])
    token_rmse = float(baseline.weighting_views["token"]["relative_rmse"]["mean_hat"])
    assert doc_rmse == pytest.approx(baseline.metrics.relative_rmse, rel=1e-6, abs=1e-9)
    assert doc_rmse == pytest.approx(leaf_rmse, rel=1e-6, abs=1e-9)
    assert doc_rmse == pytest.approx(token_rmse, rel=1e-6, abs=1e-9)


def test_hll_merge_learning_data_seed_holds_baseline_fixed_across_optimization_seeds():
    base = dict(
        precisions=(6,),
        train_docs_grid=(8,),
        audit_policies=("all",),
        n_test=8,
        n_epochs=1,
        batch_docs=4,
        hidden_dim=8,
        use_cuda=False,
        data_seed=123,
    )
    run0 = run_hll_merge_learning_experiment(HLLMergeLearningConfig(seed=0, **base))[0]
    run1 = run_hll_merge_learning_experiment(HLLMergeLearningConfig(seed=1, **base))[0]
    assert run0.hll_baseline.metrics.relative_rmse == pytest.approx(
        run1.hll_baseline.metrics.relative_rmse,
        rel=0.0,
        abs=1e-12,
    )
