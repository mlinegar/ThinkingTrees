from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.ctreepo.sim.core.full_tree_ipw_grid import run_markov_full_tree_ipw_grid
from src.ctreepo.sim.core.full_tree_ipw_grid import (
    build_markov_full_tree_ipw_tradeoff_summary,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig
from src.ctreepo.sim.core.markov_neural_operator_baselines import HAS_NEURAL_OPERATOR
from src.ctreepo.sim.suite.markov_observed_token_policy import (
    resolve_markov_observed_token_policy,
)


REPO_ROOT = Path("/home/mlinegar/ThinkingTrees")


def test_run_full_tree_ipw_simulation_writes_primary_grid_and_secondary_sweeps(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "full_tree_ipw_summary.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_full_tree_ipw_simulation.py",
            "--n-docs",
            "8",
            "--trials",
            "8",
            "--grid-rates",
            "0,1",
            "--secondary-target-rates",
            "0,1",
            "--json-summary",
            str(summary_path),
        ],
        cwd=REPO_ROOT,
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    primary_grid = payload["primary_grid"]
    assert primary_grid["rate_axis"] == [0.0, 1.0]
    assert len(primary_grid["cells"]) == 4
    assert primary_grid["anchors"]["doc_only"]["regime"] == "doc_only"
    assert primary_grid["anchors"]["full_tree"]["regime"] == "full_tree"
    assert "uniform" in payload["secondary_policy_sweeps"]
    assert len(payload["secondary_policy_sweeps"]["uniform"]["points"]) == 2


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_cli_writes_tradeoff_outputs(tmp_path: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_markov_full_tree_ipw_grid.py",
            "--observed-token-profile",
            "demo_v1",
            "--grid-rates",
            "0",
            "--root-only-fractions",
            "0",
            "--doc-sequence-train-fractions",
            "0",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--torch-threads",
            "1",
        ],
        cwd=REPO_ROOT,
    )
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "tradeoff_summary.json").exists()
    assert (tmp_path / "tradeoff_summary.md").exists()


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_helper_runs_tiny_2x2_sweep(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        model_family="neural",
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=12,
        fixed_leaf_tokens=8,
        train_docs=6,
        val_docs=2,
        test_docs=4,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        local_law_objective_mode="sampled_ipw",
        local_law_weight=0.5,
        use_residual_decomposition=True,
        use_cuda=False,
        torch_threads=1,
        seed=7,
    )
    payload = run_markov_full_tree_ipw_grid(
        base_config=cfg,
        rate_axis=[0.0, 1.0],
        output_dir=tmp_path,
    )
    assert payload["rate_axis"] == [0.0, 1.0]
    assert len(payload["cells"]) == 4
    assert payload["anchors"]["doc_only"]["regime"] == "doc_only"
    assert payload["anchors"]["full_tree"]["regime"] == "full_tree"
    assert len(payload["matrices"]["test_root_mae"]) == 2
    first_cell_path = Path(payload["cells"][0]["summary_json"])
    assert first_cell_path.exists()
    full_tree_test = payload["anchors"]["full_tree"]["test_metrics"]
    assert full_tree_test["population_size"] > 0
    assert full_tree_test["sampled_node_ht_abs_error"] >= 0.0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_supports_root_only_fraction_planes(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        model_family="neural",
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=12,
        fixed_leaf_tokens=8,
        train_docs=6,
        val_docs=2,
        test_docs=4,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        local_law_objective_mode="sampled_ipw",
        local_law_weight=0.5,
        use_residual_decomposition=True,
        use_cuda=False,
        torch_threads=1,
        seed=11,
    )
    payload = run_markov_full_tree_ipw_grid(
        base_config=cfg,
        rate_axis=[0.0, 1.0],
        root_only_fraction_axis=[0.0, 1.0],
        output_dir=tmp_path,
    )
    assert payload["root_only_fraction_axis"] == [0.0, 1.0]
    assert len(payload["planes"]) == 2
    second_plane = payload["planes"][1]
    assert second_plane["root_only_train_fraction"] == pytest.approx(1.0)
    assert len(second_plane["cells"]) == 4
    assert "test_root_only_view_root_mae" in second_plane["matrices"]


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_supports_doc_sequence_fraction_planes(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        model_family="neural",
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=12,
        fixed_leaf_tokens=8,
        train_docs=6,
        val_docs=2,
        test_docs=4,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        local_law_objective_mode="sampled_ipw",
        local_law_weight=0.5,
        use_residual_decomposition=True,
        use_cuda=False,
        torch_threads=1,
        seed=23,
    )
    payload = run_markov_full_tree_ipw_grid(
        base_config=cfg,
        rate_axis=[0.0],
        doc_sequence_train_fraction_axis=[0.0, 1.0],
        output_dir=tmp_path,
    )
    assert payload["doc_sequence_train_fraction_axis"] == [0.0, 1.0]
    assert len(payload["planes"]) == 2
    second_plane = payload["planes"][1]
    assert second_plane["doc_sequence_train_fraction"] == pytest.approx(1.0)
    assert "test_doc_sequence_view_root_mae" in second_plane["matrices"]
    assert len(second_plane["cells"]) == 1


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_cli_accepts_doc_sequence_fraction_sweeps(
    tmp_path: Path,
) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_markov_full_tree_ipw_grid.py",
            "--observed-token-profile",
            "custom",
            "--model-family",
            "neural",
            "--n-regimes",
            "2",
            "--vocab-size",
            "8",
            "--min-tokens",
            "32",
            "--max-tokens",
            "32",
            "--min-segments",
            "2",
            "--max-segments",
            "4",
            "--min-seg-len",
            "4",
            "--max-seg-len",
            "12",
            "--fixed-leaf-tokens",
            "8",
            "--train-docs",
            "6",
            "--val-docs",
            "2",
            "--test-docs",
            "4",
            "--state-dim",
            "8",
            "--hidden-dim",
            "16",
            "--n-epochs",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "1e-3",
            "--weight-decay",
            "0.0",
            "--fno-width",
            "8",
            "--fno-n-modes",
            "4",
            "--fno-n-layers",
            "1",
            "--grid-rates",
            "0",
            "--root-only-fractions",
            "0",
            "--doc-sequence-train-fractions",
            "0,1",
            "--skip-full-doc-anchors",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--torch-threads",
            "1",
        ],
        cwd=REPO_ROOT,
    )
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["doc_sequence_train_fraction_axis"] == [0.0, 1.0]


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_supports_rectangular_row_shards_and_resume(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        model_family="neural",
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=12,
        fixed_leaf_tokens=8,
        train_docs=6,
        val_docs=2,
        test_docs=4,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        local_law_objective_mode="sampled_ipw",
        local_law_weight=0.5,
        use_residual_decomposition=True,
        use_cuda=False,
        torch_threads=1,
        seed=17,
    )
    payload = run_markov_full_tree_ipw_grid(
        base_config=cfg,
        internal_rate_axis=[0.0],
        leaf_rate_axis=[0.0, 1.0],
        output_dir=tmp_path,
    )
    assert payload["internal_rate_axis"] == [0.0]
    assert payload["leaf_rate_axis"] == [0.0, 1.0]
    assert len(payload["cells"]) == 2
    assert len(payload["matrices"]["test_root_mae"]) == 1
    assert len(payload["matrices"]["test_root_mae"][0]) == 2

    resumed = run_markov_full_tree_ipw_grid(
        base_config=cfg,
        internal_rate_axis=[0.0],
        leaf_rate_axis=[0.0, 1.0],
        output_dir=tmp_path,
        skip_existing=True,
    )
    assert len(resumed["cells"]) == 2
    assert sorted(cell["summary_json"] for cell in resumed["cells"]) == sorted(
        cell["summary_json"] for cell in payload["cells"]
    )


def test_observed_token_demo_v1_profile_matches_saved_demo_defaults() -> None:
    profile = resolve_markov_observed_token_policy(profile_name="demo_v1")
    assert profile.profile == "demo_v1"
    assert profile.generator_profile == "piecewise_markov"
    assert profile.min_segments == 6
    assert profile.max_segments == 6
    assert profile.train_docs == 256
    assert profile.val_docs == 32
    assert profile.test_docs == 64
    assert profile.state_dim == 16
    assert profile.hidden_dim == 64
    assert profile.n_epochs == 6


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_can_attach_full_doc_anchors(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        model_family="neural",
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=12,
        fixed_leaf_tokens=8,
        train_docs=6,
        val_docs=2,
        test_docs=4,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        local_law_objective_mode="sampled_ipw",
        local_law_weight=0.5,
        use_residual_decomposition=True,
        use_cuda=False,
        torch_threads=1,
        seed=13,
    )
    payload = run_markov_full_tree_ipw_grid(
        base_config=cfg,
        rate_axis=[0.0],
        include_full_doc_anchors=True,
        output_dir=tmp_path,
    )
    assert "full_doc_anchors" in payload
    assert "doc_sequence" in payload["full_doc_anchors"]
    assert "doc_level" in payload["full_doc_anchors"]
    assert "doc_level_ridge" in payload["full_doc_anchors"]
    assert "rf_root" in payload["full_doc_anchors"]
    tradeoff = build_markov_full_tree_ipw_tradeoff_summary(payload)
    assert "planes" in tradeoff
    assert tradeoff["planes"][0]["pareto_frontier_root_mae_vs_sample_fraction"]


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_full_tree_ipw_grid_cli_can_aggregate_saved_row_shards(
    tmp_path: Path,
) -> None:
    common = [
        sys.executable,
        "scripts/run_markov_full_tree_ipw_grid.py",
        "--observed-token-profile",
        "custom",
        "--model-family",
        "neural",
        "--n-regimes",
        "2",
        "--vocab-size",
        "8",
        "--min-tokens",
        "32",
        "--max-tokens",
        "32",
        "--min-segments",
        "2",
        "--max-segments",
        "4",
        "--min-seg-len",
        "4",
        "--max-seg-len",
        "12",
        "--fixed-leaf-tokens",
        "8",
        "--train-docs",
        "6",
        "--val-docs",
        "2",
        "--test-docs",
        "4",
        "--state-dim",
        "8",
        "--hidden-dim",
        "16",
        "--n-epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "1e-3",
        "--weight-decay",
        "0.0",
        "--fno-width",
        "8",
        "--fno-n-modes",
        "4",
        "--fno-n-layers",
        "1",
        "--seed",
        "19",
        "--device",
        "cpu",
        "--torch-threads",
        "1",
        "--doc-sequence-train-fractions",
        "0",
        "--skip-full-doc-anchors",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.check_call(
        common
        + [
            "--internal-rates",
            "0",
            "--leaf-rates",
            "0,1",
            "--no-write-aggregate",
        ],
        cwd=REPO_ROOT,
    )
    subprocess.check_call(
        common
        + [
            "--internal-rates",
            "1",
            "--leaf-rates",
            "0,1",
            "--no-write-aggregate",
        ],
        cwd=REPO_ROOT,
    )
    subprocess.check_call(
        common
        + [
            "--internal-rates",
            "0,1",
            "--leaf-rates",
            "0,1",
            "--aggregate-only",
        ],
        cwd=REPO_ROOT,
    )
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["internal_rate_axis"] == [0.0, 1.0]
    assert payload["leaf_rate_axis"] == [0.0, 1.0]
    assert len(payload["cells"]) == 4
    tradeoff = json.loads((tmp_path / "tradeoff_summary.json").read_text(encoding="utf-8"))
    assert tradeoff["planes"][0]["doc_only_root_mae"] >= 0.0
