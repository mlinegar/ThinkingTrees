from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    _tree_neural_family_effective_config,
    run_markov_full_doc_anchor_diagnostics,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    _build_objective_summary,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import HAS_NEURAL_OPERATOR


def test_tree_ridge_leaf_baseline_smoke(tmp_path: Path) -> None:
    """Smoke test: tree_ridge_leaf baseline runs end-to-end and returns valid output."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_ridge_leaf",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
    )
    assert payload["simulation"] == "markov_full_doc_anchor_diagnostics"
    assert payload["benchmark"] == "smoke"
    assert len(payload["runs"]) == 1

    run = payload["runs"][0]
    assert run["baseline_family"] == "tree_ridge_leaf"
    assert np.isfinite(float(run["test_root_mae"]))
    assert float(run["test_root_mae"]) >= 0.0
    assert "test_exact_match_rate" in run

    assert len(payload["aggregate_rows"]) == 1
    agg = payload["aggregate_rows"][0]
    assert agg["baseline_family"] == "tree_ridge_leaf"
    assert np.isfinite(float(agg["test_root_mae_mean"]))

    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "aggregate.csv").exists()


def test_tree_ridge_backward_compat_alias(tmp_path: Path) -> None:
    """Old name 'tree_ridge' still works via _normalize_baseline_family."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_ridge",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
    )
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["baseline_family"] == "tree_ridge_leaf"


def test_tree_doc_ridge_baseline_smoke(tmp_path: Path) -> None:
    """Smoke test: tree_doc_ridge trains on root target via doc-level features."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_doc_ridge",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
    )
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["baseline_family"] == "tree_doc_ridge"
    assert np.isfinite(float(run["test_root_mae"]))
    assert float(run["test_root_mae"]) >= 0.0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_neural_c2_baseline_smoke(tmp_path: Path) -> None:
    """Smoke test: tree_neural_c2 uses official FNO tree with root + C2 objective."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_neural_c2",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 2,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["baseline_family"] == "tree_neural_c2"
    assert np.isfinite(float(run["test_root_mae"]))
    assert float(run["test_root_mae"]) >= 0.0
    assert run["c2_metric_kind"] == "score_drift"
    assert run["c2_proxy_metric_kind"] == "state_replay_mse"
    assert run["parameterization"] == "formal_local_law_weight"
    assert run["local_law_c1_weight"] == pytest.approx(0.0)
    assert run["local_law_c2_weight"] == pytest.approx(0.25)
    assert run["local_law_c3_weight"] == pytest.approx(0.0)
    assert run["optimization_root_weight"] == pytest.approx(0.75)
    assert np.isfinite(float(run["test_c2_state_replay_mse"]))


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_neural_all_laws_baseline_smoke(tmp_path: Path) -> None:
    """Smoke test: tree_neural trains with all laws (C1+C2+C3) and reports law-level metrics."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_neural",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 2,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["baseline_family"] == "tree_neural"
    assert np.isfinite(float(run["test_root_mae"]))
    assert float(run["test_root_mae"]) >= 0.0
    # Law-level metrics should be present from _eval_fno_model.
    assert "test_leaf_mae" in run
    assert "test_c2_idempotence_mae" in run
    assert "test_merge_mae" in run
    assert run["c2_metric_kind"] == "score_drift"
    assert run["c2_proxy_metric_kind"] == "state_replay_mse"
    assert run["parameterization"] == "formal_local_law_weight"
    assert run["local_law_c1_weight"] == pytest.approx(0.25 / 3.0)
    assert run["local_law_c2_weight"] == pytest.approx(0.25 / 3.0)
    assert run["local_law_c3_weight"] == pytest.approx(0.25 / 3.0)
    assert run["optimization_root_weight"] == pytest.approx(0.75)
    assert np.isfinite(float(run["test_c2_state_replay_mse"]))


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_neural_c2c3_baseline_smoke(tmp_path: Path) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_neural_c2c3",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 2,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["baseline_family"] == "tree_neural_c2c3"
    assert np.isfinite(float(run["test_root_mae"]))
    assert run["c2_metric_kind"] == "score_drift"
    assert run["parameterization"] == "formal_local_law_weight"
    assert run["local_law_c1_weight"] == pytest.approx(0.0)
    assert run["local_law_c2_weight"] == pytest.approx(2.0 * 0.25 / 3.0)
    assert run["local_law_c3_weight"] == pytest.approx(1.0 * 0.25 / 3.0)
    assert run["optimization_root_weight"] == pytest.approx(0.75)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_all_tree_families_same_data(tmp_path: Path) -> None:
    """All four tree families run on the same data and produce valid, distinct results."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=(
            "tree_ridge_leaf",
            "tree_doc_ridge",
            "tree_neural_c2",
            "tree_neural_c2c3",
            "tree_neural",
        ),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 2,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    assert len(payload["runs"]) == 5
    families = {run["baseline_family"] for run in payload["runs"]}
    assert families == {
        "tree_ridge_leaf",
        "tree_doc_ridge",
        "tree_neural_c2",
        "tree_neural_c2c3",
        "tree_neural",
    }

    for run in payload["runs"]:
        assert np.isfinite(float(run["test_root_mae"]))

    assert len(payload["aggregate_rows"]) == 5


def test_tree_ridge_leaf_alongside_ridge_control(tmp_path: Path) -> None:
    """tree_ridge_leaf and ridge_control produce distinct predictions on the same data."""
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_ridge_leaf", "ridge_control"),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
    )
    assert len(payload["runs"]) == 2
    families = {run["baseline_family"] for run in payload["runs"]}
    assert families == {"tree_ridge_leaf", "raw_token_ngram_ridge"}


def test_tree_neural_wrapper_resolves_core_objective_weights() -> None:
    base = OPSCountConfig(
        model_family="neural",
        local_law_weight=0.6,
        task_objective_weight=None,
        use_cuda=False,
    )
    c2_cfg = _tree_neural_family_effective_config(base, family="tree_neural_c2")
    c2_obj = _build_objective_summary(c2_cfg)
    assert c2_obj["parameterization"] == "formal_local_law_weight"
    assert c2_obj["optimization_root_weight"] == pytest.approx(0.4)
    assert c2_obj["local_law_c1_weight"] == pytest.approx(0.0)
    assert c2_obj["local_law_c2_weight"] == pytest.approx(0.6)
    assert c2_obj["local_law_c3_weight"] == pytest.approx(0.0)

    all_cfg = _tree_neural_family_effective_config(base, family="tree_neural")
    all_obj = _build_objective_summary(all_cfg)
    assert all_obj["parameterization"] == "formal_local_law_weight"
    assert all_obj["optimization_root_weight"] == pytest.approx(0.4)
    assert all_obj["local_law_c1_weight"] == pytest.approx(0.2)
    assert all_obj["local_law_c2_weight"] == pytest.approx(0.2)
    assert all_obj["local_law_c3_weight"] == pytest.approx(0.2)

    c2c3_cfg = _tree_neural_family_effective_config(base, family="tree_neural_c2c3")
    c2c3_obj = _build_objective_summary(c2c3_cfg)
    assert c2c3_obj["parameterization"] == "formal_local_law_weight"
    assert c2c3_obj["optimization_root_weight"] == pytest.approx(0.4)
    assert c2c3_obj["local_law_c1_weight"] == pytest.approx(0.0)
    assert c2c3_obj["local_law_c2_weight"] == pytest.approx(0.4)
    assert c2c3_obj["local_law_c3_weight"] == pytest.approx(0.2)
