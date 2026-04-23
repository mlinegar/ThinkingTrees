from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    FNOCountSketch,
    HAS_NEURAL_OPERATOR,
    _markov_merge_objective_terms_batched,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_markov_exact_progression.py"


def _load_progression_module():
    spec = importlib.util.spec_from_file_location(
        "test_markov_exact_progression_module",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="requires neuraloperator")
def test_markov_merge_objective_teacher_parent_count_ignores_endpoint_labels():
    device = torch.device("cpu")
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=4,
        state_dim=32,
        hidden_dim=64,
        target_scale=5.0,
        n_regimes=3,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="opaque_carrier_exact_sketch",
        score_merge_mode="exact_projected_sketch",
        theorem_count_head_mode="scalar_mse",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        theorem_score_dim=1,
        theorem_fiber_dim=15,
        theorem_aux_dim=0,
    ).to(device)

    summary = torch.tensor(
        [[0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    left_state = model.encode_summary(summary)
    right_state = model.encode_summary(summary)
    parent_state = model.encode_summary(summary)
    truth_counts = torch.tensor([1.0], dtype=torch.float32, device=device)

    correct = _markov_merge_objective_terms_batched(
        model,
        left_state,
        right_state,
        parent_state,
        truth_counts=truth_counts,
        truth_first=torch.tensor([0], dtype=torch.long, device=device),
        truth_last=torch.tensor([1], dtype=torch.long, device=device),
        objective_mode="teacher_parent_count",
    )
    wrong = _markov_merge_objective_terms_batched(
        model,
        left_state,
        right_state,
        parent_state,
        truth_counts=truth_counts,
        truth_first=torch.tensor([2], dtype=torch.long, device=device),
        truth_last=torch.tensor([2], dtype=torch.long, device=device),
        objective_mode="teacher_parent_count",
    )

    assert torch.allclose(correct["total_loss"], wrong["total_loss"])
    assert torch.allclose(correct["first_loss"], wrong["first_loss"])
    assert torch.allclose(correct["last_loss"], wrong["last_loss"])


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="requires neuraloperator")
def test_markov_merge_objective_teacher_parent_full_sketch_uses_endpoint_labels():
    device = torch.device("cpu")
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=4,
        state_dim=32,
        hidden_dim=64,
        target_scale=5.0,
        n_regimes=3,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="opaque_carrier_exact_sketch",
        score_merge_mode="exact_projected_sketch",
        theorem_count_head_mode="scalar_mse",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        theorem_score_dim=1,
        theorem_fiber_dim=15,
        theorem_aux_dim=0,
    ).to(device)

    left_summary = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    right_summary = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    parent_summary = torch.tensor(
        [[0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    left_state = model.encode_summary(left_summary)
    right_state = model.encode_summary(right_summary)
    parent_state = model.encode_summary(parent_summary)

    correct = _markov_merge_objective_terms_batched(
        model,
        left_state,
        right_state,
        parent_state,
        truth_counts=torch.tensor([1.0], dtype=torch.float32, device=device),
        truth_first=torch.tensor([0], dtype=torch.long, device=device),
        truth_last=torch.tensor([1], dtype=torch.long, device=device),
        objective_mode="teacher_parent_full_sketch",
    )
    wrong = _markov_merge_objective_terms_batched(
        model,
        left_state,
        right_state,
        parent_state,
        truth_counts=torch.tensor([1.0], dtype=torch.float32, device=device),
        truth_first=torch.tensor([2], dtype=torch.long, device=device),
        truth_last=torch.tensor([2], dtype=torch.long, device=device),
        objective_mode="teacher_parent_full_sketch",
    )

    assert float(correct["mean_first_loss"].item()) <= float(wrong["mean_first_loss"].item())
    assert float(correct["mean_last_loss"].item()) <= float(wrong["mean_last_loss"].item())


def test_progression_default_spec_matrix_has_expected_lanes():
    module = _load_progression_module()
    args = argparse.Namespace(
        benchmark="recoverable_v4",
        train_doc_counts=[256],
        seeds=[0],
        count_head_mode="scalar_mse",
        state_dim=128,
        hidden_dim=256,
        theorem_feature_dim=128,
        theorem_feature_hidden_dim=256,
        batch_size=64,
        n_epochs=10,
        lr=1e-3,
        weight_decay=0.0,
    )
    specs = module._default_specs(args)
    assert len(specs) == 10
    labels = {spec.merge_objective for spec in specs}
    assert labels == {
        "strict_c3",
        "teacher_parent_count",
        "teacher_parent_full_sketch",
    }


def test_progression_aggregate_zero_losses_match_across_weightings():
    module = _load_progression_module()
    zero = torch.zeros(())
    flat = module._aggregate_level_means(
        level_means=[zero, zero],
        level_counts=[3, 1],
        weighting_mode="flat_mean",
    )
    depth = module._aggregate_level_means(
        level_means=[zero, zero],
        level_counts=[3, 1],
        weighting_mode="depth_balanced",
    )
    assert float(flat.item()) == pytest.approx(0.0)
    assert float(depth.item()) == pytest.approx(0.0)


def test_progression_lean_worked_example_still_passes():
    module = _load_progression_module()
    module.test_lean_worked_example()


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="requires neuraloperator")
def test_progression_exact_leaf_smoke_run(tmp_path):
    module = _load_progression_module()
    args = argparse.Namespace(
        benchmark="recoverable_v4",
        train_doc_counts=[16],
        seeds=[0],
        count_head_mode="scalar_mse",
        state_dim=32,
        hidden_dim=64,
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        batch_size=8,
        n_epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        use_cuda=False,
        max_runs=1,
        output_root=str(tmp_path / "progression_smoke"),
    )
    summary = module._run_study(args)
    assert int(summary["n_runs"]) == 1
    assert summary["aggregate"]
    first = summary["aggregate"][0]
    assert "merger_grad_norm_root_mean" in first
    assert "step1_count_only_root_mae_mean" in first
    assert (Path(args.output_root) / "runs").exists()
