from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

import src.ctreepo.sim.core.full_doc_anchor_diagnostics as full_doc_diag
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    FAIR_FNO_PARITY_CONFIG_LABEL,
    DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES,
    _backfill_loaded_run_fields,
    _base_config_for_benchmark,
    _requires_explicit_budget_manifest_for_run,
    _effective_train_config_for_full_doc_run,
    _palette_block_exact_predictions,
    _resolve_device,
    _resolved_objective_metadata_for_run,
    _run_family_with_predictions,
    _payload_from_saved_runs,
    default_baseline_families_for_mode,
    load_markov_full_doc_anchor_diagnostics_from_output_dir,
    prepare_markov_full_doc_anchor_diagnostics_data,
    render_full_doc_anchor_diagnostic_markdown,
    resolve_full_doc_diagnostic_benchmark,
    resolve_full_doc_diagnostic_grid,
    run_markov_full_doc_anchor_diagnostics,
)
from src.ctreepo.sim.core.run_intent import materialize_tree_run_intent
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    TrainFitDiagnostics,
    build_budgeted_train_supervision_manifest,
    build_markov_changepoint_ops_count_data_bundle,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import HAS_NEURAL_OPERATOR


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_run_markov_full_doc_anchor_diagnostics_smoke_with_confusion(
    tmp_path: Path,
) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0, 1),
        train_doc_counts=(8,),
        baseline_families=("official_fno", "cnn1d", "mlp_bigram", "ridge_control"),
        emit_confusion=True,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 1,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
            "doc_sequence_fno_pooling": "sum",
            "doc_sequence_fno_concat_length_feature": True,
            "doc_sequence_fno_include_transition_channel": True,
        },
    )
    assert payload["simulation"] == "markov_full_doc_anchor_diagnostics"
    assert payload["benchmark"] == "smoke"
    assert len(payload["runs"]) == 8
    assert len(payload["aggregate_rows"]) == 4
    first_run = payload["runs"][0]
    assert "confusion" in first_run
    assert "prediction_histograms" in first_run
    assert first_run["backend_name"]
    assert first_run["operator_class"]
    assert first_run["objective_weights_active"] is False
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "aggregate.csv").exists()
    assert (
        tmp_path / "runs" / "smoke__official_fno__train_8__seed_0.json"
    ).exists()
    markdown = render_full_doc_anchor_diagnostic_markdown(payload)
    assert "# Markov Recoverable Scale Report" in markdown
    assert "## What This Report Is Trying To Show" in markdown
    assert "## Experimental Contract" in markdown
    assert "## Checks" in markdown
    assert "## Figure-First Interpretation" in markdown
    assert "## Compact Key Tables" in markdown
    assert "## Full Aggregates / Diagnostics Appendix" in markdown
    assert "official_fno" in markdown


def test_resolved_markov_target_scale_uses_process_support_not_sample_max() -> None:
    config = OPSCountConfig(max_segments=12)

    assert full_doc_diag._resolved_markov_target_scale(
        config,
        observed_targets=np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
    ) == pytest.approx(11.0)


def test_budgeted_manifest_supports_leaf_only_local_split_mode() -> None:
    config = OPSCountConfig(
        train_docs=4,
        val_docs=0,
        test_docs=0,
        generator_profile="hazard_topic",
        n_regimes=4,
        vocab_size=16,
        min_tokens=16,
        max_tokens=16,
        hazard_switch_prob=0.2,
        fixed_leaf_tokens=16,
        budget_total_calls_per_doc=1.0,
        full_doc_budget_share=0.0,
        doc_consumption_mode="root_only",
        local_split_mode="leaf_only",
        local_allocation_policy="breadth_first",
        seed=0,
        use_cuda=False,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    manifest = build_budgeted_train_supervision_manifest(
        docs=bundle.train_docs,
        config=config,
        baseline_family="tree_neural",
        seed=0,
    )

    assert manifest is not None
    assert manifest.local_split_mode == "leaf_only"
    assert all(not plan.internal_indices for plan in manifest.doc_plans)
    assert any(plan.leaf_indices for plan in manifest.doc_plans)


def test_official_fno_family_uses_benchmark_locked_config(monkeypatch) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    observed: dict[str, OPSCountConfig] = {}

    def _fake_fit_fno_baseline_with_predictions(**kwargs):
        observed["config"] = kwargs["config"]
        return {"baseline_family": "official_fno"}

    monkeypatch.setattr(full_doc_diag, "HAS_NEURAL_OPERATOR", True)
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_fno_baseline_with_predictions",
        _fake_fit_fno_baseline_with_predictions,
    )

    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=8,
        tree_batch_structural_pad_limit=0.5,
        tree_batch_auto_queue_min_docs=8,
        tree_batch_auto_queue_min_fill_ratio=0.5,
    )
    policy = full_doc_diag.resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )

    fit = _run_family_with_predictions(
        baseline_family="official_fno",
        config=config,
        benchmark=benchmark,
        seeds={"effective_model_seed": 13},
        device=torch.device("cpu"),
        train_docs=tuple(),
        val_docs=tuple(),
        test_docs=tuple(),
    )

    locked = observed["config"]
    assert fit["effective_config"] == locked
    assert locked.state_dim == 8
    assert locked.hidden_dim == 16
    assert locked.n_epochs == 1
    assert locked.batch_size == 4
    assert locked.lr == pytest.approx(1e-3)
    assert locked.weight_decay == pytest.approx(0.25)
    assert (
        locked.fixed_leaf_tokens
        == full_doc_diag.FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
    )
    assert locked.use_cuda is False
    assert locked.seed == 7
    assert locked.data_seed == 11
    assert locked.model_seed == 13
    assert locked.tree_batch_structural_pad_limit == pytest.approx(0.5)
    assert locked.tree_batch_auto_queue_min_docs == 8
    assert locked.tree_batch_auto_queue_min_fill_ratio == pytest.approx(0.5)


def test_official_fno_sumlen_family_uses_benchmark_locked_config(monkeypatch) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    observed: dict[str, OPSCountConfig] = {}

    def _fake_fit_fno_baseline_with_predictions(**kwargs):
        observed["config"] = kwargs["config"]
        return {"baseline_family": "official_fno_sumlen"}

    monkeypatch.setattr(full_doc_diag, "HAS_NEURAL_OPERATOR", True)
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_fno_baseline_with_predictions",
        _fake_fit_fno_baseline_with_predictions,
    )

    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=8,
    )

    fit = _run_family_with_predictions(
        baseline_family="official_fno_sumlen",
        config=config,
        benchmark=benchmark,
        seeds={"effective_model_seed": 13},
        device=torch.device("cpu"),
        train_docs=tuple(),
        val_docs=tuple(),
        test_docs=tuple(),
    )

    locked = observed["config"]
    assert fit["effective_config"] == locked
    assert (
        locked.fixed_leaf_tokens
        == full_doc_diag.FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
    )
    assert locked.doc_sequence_fno_pooling == "sum"
    assert locked.doc_sequence_fno_concat_length_feature is True
    assert locked.doc_sequence_fno_include_transition_channel is False


def test_official_fno_family_uses_structural_cell_geometry() -> None:
    benchmark = next(
        cell
        for cell in resolve_full_doc_diagnostic_grid("structural_core_v1")
        if str(cell.cell_id) == "r12_seg10to12"
    )
    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=8,
    )

    locked = full_doc_diag._official_fno_locked_config_for_benchmark(
        benchmark=benchmark,
        config=config,
    )

    assert locked.n_regimes == 12
    assert locked.vocab_size == 48
    assert locked.generator_profile == "piecewise_disjoint_palette"
    assert locked.min_segments == 10
    assert locked.max_segments == 12
    assert (
        locked.fixed_leaf_tokens
        == full_doc_diag.FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
    )
    assert locked.state_dim == 8
    assert locked.hidden_dim == 16
    assert locked.n_epochs == 1
    assert locked.preserve_requested_leaf_tokens is False
    assert locked.official_fno_preserve_requested_leaf_tokens is False


def test_official_fno_lock_uses_full_doc_geometry_by_default() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    config = OPSCountConfig(
        use_cuda=False,
        fixed_leaf_tokens=128,
    )

    locked = full_doc_diag._official_fno_locked_config_for_benchmark(
        benchmark=benchmark,
        config=config,
    )

    assert locked.fixed_leaf_tokens == 128
    assert locked.official_fno_preserve_requested_leaf_tokens is False
    assert locked.preserve_requested_leaf_tokens is False


def test_mixed_family_base_config_defaults_to_comparable_surface() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")

    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=1024,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        baseline_families=("official_fno", "tree_neural"),
        config_overrides={
            "fixed_leaf_tokens": 128,
            "tree_leaf_fno_width": 96,
            "tree_leaf_fno_n_modes": 12,
            "tree_leaf_fno_n_layers": 5,
            "tree_root_supervision_kind": "count_ce",
        },
    )

    assert config.comparison_mode == "comparable"
    assert config.fixed_leaf_tokens == 128
    assert config.tree_leaf_fno_width == 96
    assert config.tree_leaf_fno_n_modes == 12
    assert config.tree_leaf_fno_n_layers == 5
    assert config.fno_width == 96
    assert config.fno_n_modes == 12
    assert config.fno_n_layers == 5
    assert config.tree_root_supervision_kind == "count_ce"


def test_official_fno_lock_uses_comparable_surface_when_requested() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    config = OPSCountConfig(
        use_cuda=False,
        comparison_mode="comparable",
        fixed_leaf_tokens=128,
        tree_leaf_fno_width=96,
        tree_leaf_fno_n_modes=12,
        tree_leaf_fno_n_layers=5,
        tree_root_supervision_kind="count_ce",
    )

    locked = full_doc_diag._official_fno_locked_config_for_benchmark(
        benchmark=benchmark,
        config=config,
    )

    assert locked.comparison_mode == "comparable"
    assert locked.fixed_leaf_tokens == 128
    assert locked.tree_leaf_fno_width == 96
    assert locked.tree_leaf_fno_n_modes == 12
    assert locked.tree_leaf_fno_n_layers == 5
    assert locked.fno_width == 96
    assert locked.fno_n_modes == 12
    assert locked.fno_n_layers == 5
    assert locked.tree_root_supervision_kind == "count_ce"


def test_base_config_can_preserve_requested_leaf_tokens_for_tree_runs() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")

    default_config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=1024,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={"fixed_leaf_tokens": 64},
    )
    preserved_config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=1024,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "fixed_leaf_tokens": 64,
            "preserve_requested_leaf_tokens": True,
        },
    )

    assert default_config.fixed_leaf_tokens == 64
    assert default_config.preserve_requested_leaf_tokens is True
    assert default_config.official_fno_preserve_requested_leaf_tokens is True
    assert preserved_config.fixed_leaf_tokens == 64
    assert preserved_config.preserve_requested_leaf_tokens is True


def test_base_config_leafgrid_runs_preserve_requested_leaf_tokens_by_default() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")

    cfg = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=1024,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "fixed_leaf_tokens": 128,
            "pipeline_supervision_recovery_leafgrid_active": True,
            "pipeline_supervision_recovery_leaf_tokens": 128,
        },
    )

    assert cfg.fixed_leaf_tokens == 128
    assert cfg.preserve_requested_leaf_tokens is True
    assert cfg.official_fno_preserve_requested_leaf_tokens is True


def test_base_config_leafgrid_runs_preserve_requested_leaf_geometry_when_both_flags_are_set() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")

    cfg = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=1024,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "fixed_leaf_tokens": 128,
            "preserve_requested_leaf_tokens": True,
            "official_fno_preserve_requested_leaf_tokens": True,
            "pipeline_supervision_recovery_leafgrid_active": True,
            "pipeline_supervision_recovery_leaf_tokens": 128,
        },
    )

    assert cfg.fixed_leaf_tokens == 128
    assert cfg.preserve_requested_leaf_tokens is True
    assert cfg.official_fno_preserve_requested_leaf_tokens is True


def test_tree_neural_runtime_identity_mode_delegates_to_official_fno(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    observed: dict[str, OPSCountConfig] = {}

    def _fake_fit_fno_baseline_with_predictions(**kwargs):
        observed["config"] = kwargs["config"]
        return {
            "baseline_family": "official_fno",
            "test_preds": [],
            "test_truths": [],
            "test_metrics": {"root_mae": 0.0},
        }

    def _unexpected_tree_fit(**kwargs):
        raise AssertionError("tree_neural fitter should not be used in runtime identity mode")

    monkeypatch.setattr(full_doc_diag, "HAS_NEURAL_OPERATOR", True)
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_fno_baseline_with_predictions",
        _fake_fit_fno_baseline_with_predictions,
    )
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_tree_neural_baseline_with_predictions",
        _unexpected_tree_fit,
    )

    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=128,
        tree_root_supervision_kind="mse",
        tree_leaf_fno_width=128,
        tree_leaf_fno_n_modes=8,
        tree_leaf_fno_n_layers=4,
        tree_exact_collapse_mode="official_fno_runtime_identity",
        official_fno_preserve_requested_leaf_tokens=True,
    )

    fit = _run_family_with_predictions(
        baseline_family="tree_neural",
        config=config,
        benchmark=benchmark,
        seeds={"effective_model_seed": 13},
        device=torch.device("cpu"),
        train_docs=tuple(),
        val_docs=tuple(),
        test_docs=tuple(),
    )

    locked = observed["config"]
    assert fit["effective_config"] == locked
    assert fit["collapse_runtime_delegate_family"] == "official_fno"
    assert fit["collapse_runtime_mode"] == "official_fno_runtime_identity"
    assert locked.tree_exact_collapse_mode == "official_fno_runtime_identity"
    assert locked.tree_root_supervision_kind == "mse"
    assert locked.fixed_leaf_tokens == 128
    assert locked.tree_leaf_fno_width == 128
    assert locked.tree_leaf_fno_n_modes == 8
    assert locked.local_law_weight == pytest.approx(0.0)
    assert locked.task_objective_weight == pytest.approx(1.0)
    assert locked.c1_relative_weight == pytest.approx(0.0)
    assert locked.c2_relative_weight == pytest.approx(0.0)
    assert locked.c3_relative_weight == pytest.approx(0.0)


def test_tree_neural_one_tree_identity_mode_uses_official_fno_runtime_but_reports_tree_config(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    observed: dict[str, OPSCountConfig] = {}

    def _fake_fit_fno_baseline_with_predictions(**kwargs):
        observed["config"] = kwargs["config"]
        return {
            "baseline_family": "official_fno",
            "test_preds": [],
            "test_truths": [],
            "test_metrics": {"root_mae": 0.0},
        }

    monkeypatch.setattr(full_doc_diag, "HAS_NEURAL_OPERATOR", True)
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_fno_baseline_with_predictions",
        _fake_fit_fno_baseline_with_predictions,
    )

    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=128,
        tree_root_supervision_kind="mse",
        tree_leaf_fno_width=128,
        tree_leaf_fno_n_modes=8,
        tree_leaf_fno_n_layers=4,
        tree_exact_collapse_mode="official_fno_one_tree_identity",
        official_fno_preserve_requested_leaf_tokens=True,
        local_law_weight=0.25,
        c1_relative_weight=1.0,
        c2_relative_weight=1.0,
        c3_relative_weight=1.0,
        leaf_supervision_kind="full_sketch",
        leaf_label_rate=1.0,
    )

    fit = _run_family_with_predictions(
        baseline_family="tree_neural",
        config=config,
        benchmark=benchmark,
        seeds={"effective_model_seed": 13},
        device=torch.device("cpu"),
        train_docs=tuple(),
        val_docs=tuple(),
        test_docs=tuple(),
    )

    delegated = observed["config"]
    effective = fit["effective_config"]
    assert fit["collapse_runtime_delegate_family"] == "official_fno"
    assert fit["collapse_runtime_mode"] == "official_fno_one_tree_identity"
    assert fit["c2_metric_kind"] == full_doc_diag.FNO_TREE_C2_METRIC_KIND
    assert fit["c2_proxy_metric_kind"] == full_doc_diag.FNO_TREE_C2_PROXY_METRIC_KIND
    assert fit["c2_exact_witness_kind"] == full_doc_diag.FNO_TREE_C2_EXACT_WITNESS_KIND
    assert delegated.fixed_leaf_tokens == 128
    assert delegated.local_law_weight == pytest.approx(0.0)
    assert delegated.task_objective_weight == pytest.approx(1.0)
    assert effective.tree_exact_collapse_mode == "official_fno_one_tree_identity"
    assert effective.fixed_leaf_tokens == 128
    assert effective.local_law_weight == pytest.approx(0.0)
    assert effective.task_objective_weight == pytest.approx(1.0)
    assert effective.c1_relative_weight == pytest.approx(0.0)
    assert effective.c2_relative_weight == pytest.approx(0.0)
    assert effective.c3_relative_weight == pytest.approx(0.0)
    assert effective.leaf_supervision_kind == "count_only"
    assert effective.leaf_label_rate == pytest.approx(0.0)


def test_tree_neural_exact_collapse_uses_budget_subset_like_official_fno(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    observed: dict[str, Any] = {}

    def _fake_fit_fno_baseline_with_predictions(**kwargs):
        observed["train_docs"] = tuple(kwargs["train_docs"])
        observed["config"] = kwargs["config"]
        return {
            "baseline_family": "official_fno",
            "test_preds": [],
            "test_truths": [],
            "test_metrics": {"root_mae": 0.0},
        }

    monkeypatch.setattr(full_doc_diag, "HAS_NEURAL_OPERATOR", True)
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_fno_baseline_with_predictions",
        _fake_fit_fno_baseline_with_predictions,
    )

    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=128,
        tree_root_supervision_kind="count_ce",
        tree_leaf_fno_width=128,
        tree_leaf_fno_n_modes=8,
        tree_leaf_fno_n_layers=4,
        tree_exact_collapse_mode="official_fno_one_tree_identity",
        official_fno_preserve_requested_leaf_tokens=True,
        preserve_requested_leaf_tokens=True,
        full_doc_budget_share=0.5,
        doc_consumption_mode="root_only",
        package_semantics="full_doc_only",
        leaf_label_rate=0.0,
        internal_label_rate=0.0,
    )
    docs = ("doc0", "doc1", "doc2", "doc3")
    budget_manifest = full_doc_diag.BudgetedTrainSupervisionManifest(
        budget_total_calls=2,
        budget_total_calls_per_doc=0.5,
        budget_total_calls_used=2,
        budget_utilization=1.0,
        full_doc_budget_share=0.5,
        full_doc_calls_requested=2,
        full_doc_calls_total=2,
        local_calls_requested=0,
        local_calls_total=0,
        doc_consumption_mode="root_only",
        local_split_mode="",
        local_allocation_policy="",
        sampling_scheme="seeded_random_without_replacement",
        doc_touch_rate=0.5,
        mean_labels_per_touched_doc=1.0,
        touched_docs_total=2,
        effective_full_doc_mass_total=2.0,
        effective_full_doc_mass_per_doc=0.5,
        document_mass_share=1.0,
        leaf_mass_share=0.0,
        internal_mass_share=0.0,
        document_call_share=1.0,
        leaf_call_share=0.0,
        internal_call_share=0.0,
        doc_plans=(
            full_doc_diag.BudgetedTrainSupervisionDocPlan(
                doc_index=1,
                doc_tokens=128,
                document_mode="root_only",
                raw_call_cost=1,
                document_mass=1.0,
                effective_full_doc_mass=1.0,
            ),
            full_doc_diag.BudgetedTrainSupervisionDocPlan(
                doc_index=3,
                doc_tokens=128,
                document_mode="root_only",
                raw_call_cost=1,
                document_mass=1.0,
                effective_full_doc_mass=1.0,
            ),
        ),
    )

    fit = _run_family_with_predictions(
        baseline_family="tree_neural",
        config=config,
        benchmark=benchmark,
        seeds={"effective_model_seed": 13},
        device=torch.device("cpu"),
        train_docs=docs,
        val_docs=tuple(),
        test_docs=tuple(),
        budget_manifest=budget_manifest,
    )

    assert fit["collapse_runtime_delegate_family"] == "official_fno"
    assert observed["train_docs"] == ("doc1", "doc3")


def test_tree_neural_exact_collapse_reconstructs_budget_subset_when_doc_plans_missing(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    observed: dict[str, Any] = {}

    def _fake_fit_fno_baseline_with_predictions(**kwargs):
        observed["train_docs"] = tuple(kwargs["train_docs"])
        return {
            "baseline_family": "official_fno",
            "test_preds": [],
            "test_truths": [],
            "test_metrics": {"root_mae": 0.0},
        }

    monkeypatch.setattr(full_doc_diag, "HAS_NEURAL_OPERATOR", True)
    monkeypatch.setattr(
        full_doc_diag,
        "_fit_fno_baseline_with_predictions",
        _fake_fit_fno_baseline_with_predictions,
    )

    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        data_seed=11,
        model_seed=13,
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.25,
        fixed_leaf_tokens=128,
        tree_root_supervision_kind="count_ce",
        tree_leaf_fno_width=128,
        tree_leaf_fno_n_modes=8,
        tree_leaf_fno_n_layers=4,
        tree_exact_collapse_mode="official_fno_one_tree_identity",
        official_fno_preserve_requested_leaf_tokens=True,
        preserve_requested_leaf_tokens=True,
        full_doc_budget_share=1.0,
        budget_total_calls_per_doc=0.5,
        doc_consumption_mode="root_only",
        package_semantics="full_doc_only",
        leaf_label_rate=0.0,
        internal_label_rate=0.0,
    )
    docs = tuple(f"doc{idx}" for idx in range(4))
    budget_manifest = full_doc_diag.BudgetedTrainSupervisionManifest(
        budget_total_calls=2,
        budget_total_calls_per_doc=0.5,
        budget_total_calls_used=2,
        budget_utilization=1.0,
        full_doc_budget_share=1.0,
        full_doc_calls_requested=2,
        full_doc_calls_total=2,
        local_calls_requested=0,
        local_calls_total=0,
        doc_consumption_mode="root_only",
        local_split_mode="",
        local_allocation_policy="",
        sampling_scheme="seeded_random_without_replacement",
        requested_root_mass_per_doc=0.5,
        realized_root_mass_per_doc=0.5,
        doc_touch_rate=0.5,
        mean_labels_per_touched_doc=1.0,
        touched_docs_total=2,
        effective_full_doc_mass_total=2.0,
        effective_full_doc_mass_per_doc=0.5,
        document_mass_share=1.0,
        leaf_mass_share=0.0,
        internal_mass_share=0.0,
        document_call_share=1.0,
        leaf_call_share=0.0,
        internal_call_share=0.0,
        doc_plans=tuple(),
    )

    fit = _run_family_with_predictions(
        baseline_family="tree_neural",
        config=config,
        benchmark=benchmark,
        seeds={"effective_model_seed": 13},
        device=torch.device("cpu"),
        train_docs=docs,
        val_docs=tuple(),
        test_docs=tuple(),
        budget_manifest=budget_manifest,
    )

    assert fit["collapse_runtime_delegate_family"] == "official_fno"
    assert observed["train_docs"] == ("doc0", "doc2")


def test_exact_collapse_tree_manifest_rows_require_explicit_budget_manifest() -> None:
    config = OPSCountConfig(
        budget_total_calls_per_doc=0.9,
        doc_consumption_mode="root_only",
        tree_supervision_source="manifest",
        tree_exact_collapse_mode="official_fno_one_tree_identity",
    )

    assert _requires_explicit_budget_manifest_for_run(
        baseline_family="tree_neural",
        config=config,
    )


def test_noncollapse_tree_manifest_rows_require_explicit_budget_manifest() -> None:
    config = OPSCountConfig(
        budget_total_calls_per_doc=0.9,
        doc_consumption_mode="root_only",
        tree_supervision_source="manifest",
        tree_exact_collapse_mode="",
    )

    assert _requires_explicit_budget_manifest_for_run(
        baseline_family="tree_neural",
        config=config,
    )


def test_resolved_tree_supervision_manifest_preserves_explicit_manifest_for_manifest_runs() -> None:
    token_block = tuple(range(128))
    docs = tuple(
        full_doc_diag._FNOCountDoc(
            n_tokens=128,
            leaf_token_ids=(token_block,),
            leaf_counts=(0.0,),
            leaf_first_regimes=(0,),
            leaf_last_regimes=(0,),
            leaf_token_lengths=(128,),
            merge_counts_balanced=(),
            merge_sizes_balanced=(),
            merge_token_lengths=(),
            root_count=0.0,
        )
        for _ in range(4)
    )
    config = OPSCountConfig(
        fixed_leaf_tokens=128,
        tree_supervision_source="manifest",
        leaf_supervision_kind="count_only",
        leaf_label_rate=0.1,
        internal_supervision_kind="none",
        internal_label_rate=0.0,
    )
    explicit_manifest = full_doc_diag.BudgetedTrainSupervisionManifest(
        budget_total_calls=4,
        budget_total_calls_per_doc=1.0,
        budget_total_calls_used=4,
        budget_utilization=1.0,
        full_doc_budget_share=0.9,
        full_doc_calls_requested=3,
        full_doc_calls_total=3,
        local_calls_requested=1,
        local_calls_total=1,
        doc_consumption_mode="root_only",
        local_split_mode="balanced",
        local_allocation_policy="breadth_first",
        sampling_scheme="seeded_random_without_replacement",
        doc_touch_rate=1.0,
        mean_labels_per_touched_doc=1.0,
        touched_docs_total=4,
        effective_full_doc_mass_total=4.0,
        effective_full_doc_mass_per_doc=1.0,
        document_mass_share=0.75,
        leaf_mass_share=0.25,
        internal_mass_share=0.0,
        document_call_share=0.75,
        leaf_call_share=0.25,
        internal_call_share=0.0,
        doc_plans=(
            full_doc_diag.BudgetedTrainSupervisionDocPlan(
                doc_index=0,
                doc_tokens=128,
                document_mode="root_only",
                leaf_indices=(0,),
                raw_call_cost=2,
                document_mass=1.0,
                leaf_mass=1.0,
                effective_full_doc_mass=2.0,
            ),
        ),
    )

    resolved = full_doc_diag._resolved_tree_supervision_manifest(
        docs=docs,
        config=config,
        budget_manifest=explicit_manifest,
    )

    assert resolved is explicit_manifest


def test_resolved_tree_supervision_manifest_rebuilds_missing_local_mass_manifest() -> None:
    token_block = tuple(range(128))
    docs = tuple(
        full_doc_diag._FNOCountDoc(
            n_tokens=128,
            leaf_token_ids=(token_block,),
            leaf_counts=(0.0,),
            leaf_first_regimes=(0,),
            leaf_last_regimes=(0,),
            leaf_token_lengths=(128,),
            merge_counts_balanced=(),
            merge_sizes_balanced=(),
            merge_token_lengths=(),
            root_count=0.0,
        )
        for _ in range(4)
    )
    config = OPSCountConfig(
        fixed_leaf_tokens=128,
        tree_supervision_source="manifest",
        package_semantics="mass_matched",
        doc_consumption_mode="root_only",
        local_split_mode="leaf_only",
        budget_total_calls_per_doc=0.0,
        full_doc_budget_share=1.0,
        mass_target_per_doc=1.0,
        leaf_supervision_kind="count_only",
        leaf_label_rate=1.0,
        internal_supervision_kind="none",
        internal_label_rate=0.0,
        seed=0,
    )
    incomplete_manifest = full_doc_diag.BudgetedTrainSupervisionManifest(
        budget_total_calls=0,
        budget_total_calls_per_doc=0.0,
        budget_total_calls_used=0,
        budget_utilization=0.0,
        full_doc_budget_share=1.0,
        full_doc_calls_requested=0,
        full_doc_calls_total=0,
        local_calls_requested=0,
        local_calls_total=0,
        doc_consumption_mode="root_only",
        local_split_mode="leaf_only",
        local_allocation_policy="breadth_first",
        sampling_scheme="seeded_random_without_replacement",
        doc_touch_rate=0.0,
        mean_labels_per_touched_doc=0.0,
        touched_docs_total=0,
        effective_full_doc_mass_total=0.0,
        effective_full_doc_mass_per_doc=0.0,
        document_mass_share=0.0,
        leaf_mass_share=0.0,
        internal_mass_share=0.0,
        document_call_share=0.0,
        leaf_call_share=0.0,
        internal_call_share=0.0,
        doc_plans=tuple(
            full_doc_diag.BudgetedTrainSupervisionDocPlan(
                doc_index=i,
                doc_tokens=128,
                document_mode="",
                leaf_indices=tuple(),
                raw_call_cost=0,
                document_mass=0.0,
                leaf_mass=0.0,
                effective_full_doc_mass=0.0,
            )
            for i in range(4)
        ),
    )

    resolved = full_doc_diag._resolved_tree_supervision_manifest(
        docs=docs,
        config=config,
        budget_manifest=incomplete_manifest,
    )

    assert resolved is not incomplete_manifest
    assert resolved is not None
    assert resolved.local_calls_total > 0
    assert resolved.effective_full_doc_mass_per_doc >= 1.0 - 1e-9


def test_tree_manifest_planner_uses_actual_doc_geometry_and_contract_checks() -> None:
    token_block = tuple(range(32))
    docs = tuple(
        full_doc_diag._FNOCountDoc(
            n_tokens=96,
            leaf_token_ids=(token_block, token_block, token_block),
            leaf_counts=(0.0, 0.0, 0.0),
            leaf_first_regimes=(0, 0, 0),
            leaf_last_regimes=(0, 0, 0),
            leaf_token_lengths=(32, 32, 32),
            merge_counts_balanced=(0.0, 0.0),
            merge_sizes_balanced=(2, 3),
            merge_token_lengths=(64, 96),
            root_count=0.0,
        )
        for _ in range(4)
    )
    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        fixed_leaf_tokens=32,
        doc_consumption_mode="root_only",
        full_doc_budget_share=1.0,
        budget_total_calls_per_doc=0.5,
        mass_target_per_doc=0.5,
        leaf_supervision_kind="count_only",
        leaf_label_rate=(1.0 / 3.0),
        internal_supervision_kind="none",
        internal_label_rate=0.0,
        tree_supervision_source="manifest",
        tree_local_weighting_mode="span_mass_ipw_sum",
    )

    manifest = full_doc_diag._build_tree_rate_driven_supervision_manifest(
        docs=docs,
        config=config,
        leaf_sample_ordering_by_doc={idx: (0, 1, 2) for idx in range(len(docs))},
    )

    assert manifest is not None
    assert manifest.actual_doc_tokens_unique == (96,)
    assert manifest.realized_leaf_mass_per_doc == pytest.approx(32.0 / 96.0)
    assert manifest.mass_target_per_doc == pytest.approx(0.5)
    assert manifest.sampling_scheme == "seeded_random_without_replacement"
    assert manifest.requested_root_mass_per_doc == pytest.approx(1.0 / 6.0)
    assert manifest.realized_root_mass_per_doc == pytest.approx(0.25)
    assert manifest.effective_full_doc_mass_per_doc == pytest.approx(
        (32.0 / 96.0) + 0.25
    )
    assert manifest.local_calls_total == 4
    assert manifest.full_doc_calls_total == 1

    config_view = type(
        "ManifestConfigView",
        (),
        {
            "tree_supervision_source": "manifest",
            "computed_assumed_doc_tokens": 96,
            "leaf_label_rate": config.leaf_label_rate,
            "internal_supervision_kind": config.internal_supervision_kind,
            "internal_label_rate": config.internal_label_rate,
            "mass_target_per_doc": config.mass_target_per_doc,
        },
    )()
    contract = full_doc_diag._tree_supervision_contract_summary(
        config=config_view,
        budget_metadata=full_doc_diag._budget_manifest_metadata(manifest),
    )
    assert contract["required"] is True
    assert contract["passed"] is True
    assert contract["checks"]["geometry_match"] is True
    assert contract["checks"]["local_manifest_match"] is True
    assert contract["checks"]["mass_target_match"] is True
    assert contract["requested_root_mass_per_doc"] == pytest.approx(1.0 / 6.0)
    assert contract["realized_effective_full_doc_mass_per_doc"] == pytest.approx(
        (32.0 / 96.0) + 0.25
    )


def test_explicit_local_indices_from_rate_is_seeded_without_replacement() -> None:
    sampled = full_doc_diag._explicit_local_indices_from_rate(
        n_items=10,
        rate=0.35,
        ordering=None,
        seed=17,
    )

    assert sampled == (0, 3, 7)
    assert sampled == full_doc_diag._explicit_local_indices_from_rate(
        n_items=10,
        rate=0.35,
        ordering=None,
        seed=17,
    )
    assert len(sampled) == len(set(sampled))
    assert all(0 <= int(index) < 10 for index in sampled)


def test_explicit_local_indices_from_rate_returns_full_population_at_full_rate() -> None:
    sampled = full_doc_diag._explicit_local_indices_from_rate(
        n_items=6,
        rate=1.0,
        ordering=(5, 4, 3, 2, 1, 0),
        seed=99,
    )

    assert sampled == (0, 1, 2, 3, 4, 5)


def test_tree_manifest_planner_preserves_residual_root_mass_for_one_leaf_geometry() -> None:
    token_block = tuple(range(128))
    docs = tuple(
        full_doc_diag._FNOCountDoc(
            n_tokens=128,
            leaf_token_ids=(token_block,),
            leaf_counts=(0.0,),
            leaf_first_regimes=(0,),
            leaf_last_regimes=(0,),
            leaf_token_lengths=(128,),
            merge_counts_balanced=(),
            merge_sizes_balanced=(),
            merge_token_lengths=(),
            root_count=0.0,
        )
        for _ in range(20)
    )
    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        fixed_leaf_tokens=128,
        doc_consumption_mode="root_only",
        full_doc_budget_share=1.0,
        budget_total_calls_per_doc=0.9,
        mass_target_per_doc=1.0,
        leaf_supervision_kind="count_only",
        leaf_label_rate=0.1,
        internal_supervision_kind="none",
        internal_label_rate=0.0,
        tree_supervision_source="manifest",
        tree_local_weighting_mode="span_mass_ipw_sum",
    )

    manifest = full_doc_diag._build_tree_rate_driven_supervision_manifest(
        docs=docs,
        config=config,
    )

    assert manifest is not None
    assert manifest.actual_doc_tokens_unique == (128,)
    assert manifest.local_calls_total == 1
    assert manifest.full_doc_calls_total == 19
    assert manifest.requested_root_mass_per_doc == pytest.approx(0.95)
    assert manifest.realized_leaf_mass_per_doc == pytest.approx(0.05)
    assert manifest.realized_root_mass_per_doc == pytest.approx(0.95)
    assert manifest.effective_full_doc_mass_per_doc == pytest.approx(1.0)
    assert manifest.leaf_propensity_mean == pytest.approx(0.1)

    config_view = type(
        "ManifestConfigView",
        (),
        {
            "tree_supervision_source": "manifest",
            "computed_assumed_doc_tokens": 128,
            "leaf_label_rate": config.leaf_label_rate,
            "internal_supervision_kind": config.internal_supervision_kind,
            "internal_label_rate": config.internal_label_rate,
            "mass_target_per_doc": config.mass_target_per_doc,
        },
    )()
    contract = full_doc_diag._tree_supervision_contract_summary(
        config=config_view,
        budget_metadata=full_doc_diag._budget_manifest_metadata(manifest),
    )
    assert contract["required"] is True
    assert contract["passed"] is True
    assert contract["checks"]["mass_target_match"] is True
    assert contract["requested_root_mass_per_doc"] == pytest.approx(0.95)
    assert contract["realized_effective_full_doc_mass_per_doc"] == pytest.approx(1.0)


def test_tree_manifest_planner_keeps_full_root_coverage_for_superset_packages() -> None:
    token_block = tuple(range(128))
    docs = tuple(
        full_doc_diag._FNOCountDoc(
            n_tokens=128,
            leaf_token_ids=(token_block,),
            leaf_counts=(0.0,),
            leaf_first_regimes=(0,),
            leaf_last_regimes=(0,),
            leaf_token_lengths=(128,),
            merge_counts_balanced=(),
            merge_sizes_balanced=(),
            merge_token_lengths=(),
            root_count=0.0,
        )
        for _ in range(20)
    )
    config = OPSCountConfig(
        use_cuda=False,
        seed=7,
        fixed_leaf_tokens=128,
        doc_consumption_mode="root_only",
        package_semantics="superset",
        full_doc_budget_share=1.0,
        budget_total_calls_per_doc=1.0,
        leaf_supervision_kind="count_only",
        leaf_label_rate=0.1,
        internal_supervision_kind="none",
        internal_label_rate=0.0,
        tree_supervision_source="manifest",
        tree_local_weighting_mode="span_mass_ipw_sum",
    )

    manifest = full_doc_diag._build_tree_rate_driven_supervision_manifest(
        docs=docs,
        config=config,
    )

    assert manifest is not None
    assert manifest.package_semantics == "superset"
    assert manifest.actual_doc_tokens_unique == (128,)
    assert manifest.local_calls_total == 1
    assert manifest.full_doc_calls_total == 20
    assert manifest.requested_root_mass_per_doc == pytest.approx(1.0)
    assert manifest.realized_root_mass_per_doc == pytest.approx(1.0)
    assert manifest.realized_leaf_mass_per_doc == pytest.approx(0.05)
    assert manifest.effective_full_doc_mass_per_doc == pytest.approx(1.05)
    assert manifest.mass_target_per_doc != manifest.mass_target_per_doc

    config_view = type(
        "ManifestConfigView",
        (),
        {
            "tree_supervision_source": "manifest",
            "computed_assumed_doc_tokens": 128,
            "leaf_label_rate": config.leaf_label_rate,
            "internal_supervision_kind": config.internal_supervision_kind,
            "internal_label_rate": config.internal_label_rate,
            "mass_target_per_doc": config.mass_target_per_doc,
            "package_semantics": config.package_semantics,
        },
    )()
    contract = full_doc_diag._tree_supervision_contract_summary(
        config=config_view,
        budget_metadata=full_doc_diag._budget_manifest_metadata(manifest),
    )
    assert contract["required"] is True
    assert contract["passed"] is True
    assert contract["package_semantics"] == "superset"
    assert contract["checks"]["mass_target_match"] is True
    assert contract["requested_root_mass_per_doc"] == pytest.approx(1.0)
    assert contract["realized_effective_full_doc_mass_per_doc"] == pytest.approx(1.05)


def test_resolved_objective_metadata_for_one_tree_identity_never_reactivates_local_laws() -> None:
    metadata = _resolved_objective_metadata_for_run(
        OPSCountConfig(
            local_law_weight=0.25,
            c1_relative_weight=1.0,
            c2_relative_weight=1.0,
            c3_relative_weight=1.0,
            leaf_weight=1.0,
            c2_weight=1.0,
            c3_weight=1.0,
            task_objective_weight=None,
            tree_exact_collapse_mode="official_fno_one_tree_identity",
        ),
        baseline_family="tree_neural",
    )

    assert metadata["parameterization"] == "exact_collapse_root_only_identity"
    assert metadata["optimization_root_weight"] == pytest.approx(1.0)
    assert metadata["local_law_c1_weight"] == pytest.approx(0.0)
    assert metadata["local_law_c2_weight"] == pytest.approx(0.0)
    assert metadata["local_law_c3_weight"] == pytest.approx(0.0)


def test_progress_callback_emits_materializing_bundle_before_run_start(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []

    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("raw_token_ngram_ridge",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={"n_epochs": 1},
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert payload["runs"]
    stages = [str(event.get("stage", "")) for event in events]
    assert stages
    assert stages[0] == "materializing_bundle"
    assert "run_start" in stages
    assert stages.index("materializing_bundle") < stages.index("run_start")


def test_progress_callback_uses_effective_two_stage_epochs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[dict[str, object]] = []

    def _fake_run_payload(**kwargs):
        return {
            "benchmark": str(kwargs["benchmark"].name),
            "baseline_family": str(kwargs["baseline_family"]),
            "seed": int(kwargs["seeds"]["effective_model_seed"]),
            "train_doc_count": int(kwargs["train_doc_count"]),
        }

    monkeypatch.setattr(full_doc_diag, "_run_payload", _fake_run_payload)

    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="recoverable_v4",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_neural",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 10,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
        },
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert payload["runs"]
    run_start = next(event for event in events if str(event.get("stage", "")) == "run_start")
    assert int(run_start["epochs_total"]) == 32


def test_run_markov_full_doc_anchor_diagnostics_passes_run_metadata_to_run_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: list[dict[str, object]] = []

    def _fake_run_payload(**kwargs):
        observed.append(dict(kwargs.get("run_metadata") or {}))
        return {
            "benchmark": str(kwargs["benchmark"].name),
            "baseline_family": str(kwargs["baseline_family"]),
            "seed": int(kwargs["seeds"]["effective_model_seed"]),
            "train_doc_count": int(kwargs["train_doc_count"]),
        }

    monkeypatch.setattr(full_doc_diag, "_run_payload", _fake_run_payload)

    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("raw_token_ngram_ridge",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={"n_epochs": 1},
        run_metadata={"tuning_stage": "capacity_screen", "study_name": "r10_canary"},
    )

    assert payload["runs"]
    assert observed == [
        {"tuning_stage": "capacity_screen", "study_name": "r10_canary"}
    ]


def test_run_payload_uses_minimal_posttrain_diagnostics_for_capacity_screen(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("smoke")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={"n_epochs": 1},
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    observed: dict[str, object] = {}

    def _fake_run_family_with_predictions(**kwargs):
        observed["posttrain_diagnostics_mode"] = kwargs["posttrain_diagnostics_mode"]
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": kwargs["config"],
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="raw_token_ngram_ridge",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
        run_metadata={"tuning_stage": "capacity_screen"},
    )

    assert observed["posttrain_diagnostics_mode"] == "minimal"
    assert payload["train_doc_count"] == 8


def test_run_payload_uses_minimal_posttrain_diagnostics_for_capacity_locked(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("smoke")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={"n_epochs": 1},
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    observed: dict[str, object] = {}

    def _fake_run_family_with_predictions(**kwargs):
        observed["posttrain_diagnostics_mode"] = kwargs["posttrain_diagnostics_mode"]
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": kwargs["config"],
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="raw_token_ngram_ridge",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
        run_metadata={"tuning_stage": "capacity_locked"},
    )

    assert observed["posttrain_diagnostics_mode"] == "minimal"
    assert payload["train_doc_count"] == 8


def test_run_payload_uses_full_posttrain_diagnostics_outside_screen(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("smoke")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={"n_epochs": 1},
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    observed: dict[str, object] = {}

    def _fake_run_family_with_predictions(**kwargs):
        observed["posttrain_diagnostics_mode"] = kwargs["posttrain_diagnostics_mode"]
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": kwargs["config"],
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="raw_token_ngram_ridge",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
        run_metadata={"tuning_stage": "comparison"},
    )

    assert observed["posttrain_diagnostics_mode"] == "full"
    assert payload["train_doc_count"] == 8


def test_run_payload_honors_config_override_for_posttrain_diagnostics_mode(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("smoke")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={"n_epochs": 1, "posttrain_diagnostics_mode": "minimal"},
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    observed: dict[str, object] = {}

    def _fake_run_family_with_predictions(**kwargs):
        observed["posttrain_diagnostics_mode"] = kwargs["posttrain_diagnostics_mode"]
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": kwargs["config"],
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="raw_token_ngram_ridge",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
        run_metadata={"tuning_stage": "comparison"},
    )

    assert observed["posttrain_diagnostics_mode"] == "minimal"
    assert payload["train_doc_count"] == 8


def test_prepare_markov_full_doc_anchor_diagnostics_data_creates_nested_cache(
    tmp_path: Path,
) -> None:
    payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name="smoke",
        seeds=(0, 1),
        train_doc_counts=(4, 8),
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "fixed_leaf_tokens": 8,
            "prepared_data_root": str(tmp_path / "prepared"),
            "prepared_data_allow_create": True,
        },
    )

    assert payload["prepared"]
    prepared = payload["prepared"][0]
    root = Path(prepared["prepared_data_root"])
    assert root.exists()
    assert Path(prepared["metadata_json"]).exists()
    assert Path(prepared["train_fno_docs_json"]).exists()
    assert Path(prepared["leaf_orderings_json"]).exists()
    assert prepared["train_prefix_counts"] == [4, 8]
    assert set(prepared["train_prefix_signatures"]) == {"4", "8"}

    second = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name="smoke",
        seeds=(0, 1),
        train_doc_counts=(4, 8),
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "fixed_leaf_tokens": 8,
            "prepared_data_root": str(tmp_path / "prepared"),
            "prepared_data_allow_create": True,
        },
    )
    assert (
        second["prepared"][0]["prepared_data_signature"]
        == prepared["prepared_data_signature"]
    )
    assert second["prepared"][0]["prepared_data_root"] == prepared["prepared_data_root"]


def test_prepare_markov_full_doc_anchor_diagnostics_data_reuses_seed_and_prefix_superset_cache(
    tmp_path: Path,
) -> None:
    initial = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name="smoke",
        seeds=(0, 1),
        train_doc_counts=(4, 8),
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "fixed_leaf_tokens": 8,
            "prepared_data_root": str(tmp_path / "prepared"),
            "prepared_data_allow_create": True,
        },
    )

    reused = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "fixed_leaf_tokens": 8,
            "prepared_data_root": str(tmp_path / "prepared"),
            "prepared_data_allow_create": False,
        },
    )

    assert (
        reused["prepared"][0]["prepared_data_signature"]
        == initial["prepared"][0]["prepared_data_signature"]
    )
    assert reused["prepared"][0]["prepared_data_root"] == initial["prepared"][0]["prepared_data_root"]


def test_resolve_full_doc_diagnostic_t128_variants_surface_geometry() -> None:
    recoverable = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    assert recoverable.cell_id == "recoverable_v4_t128"
    assert recoverable.canonical_bundle_path.endswith(
        "markov_observed_token_recoverable_v4_t128/markov_data/observed_token_bundle.json"
    )
    assert recoverable.expanded_bundle_path.endswith(
        "markov_observed_token_recoverable_20x_v1_t128/markov_data/observed_token_bundle.json"
    )
    assert recoverable.expanded_train_docs_capacity == 20480

    recoverable_xlarge = resolve_full_doc_diagnostic_benchmark("recoverable_20x_v1_t128")
    assert recoverable_xlarge.canonical_bundle_path.endswith(
        "markov_observed_token_recoverable_20x_v1_t128/markov_data/observed_token_bundle.json"
    )
    assert recoverable_xlarge.canonical_train_docs_capacity == 20480

    structural_cells = resolve_full_doc_diagnostic_grid("structural_core_v1_t128")
    structural = next(
        cell for cell in structural_cells if cell.cell_id == "r12_seg10to12"
    )
    assert structural.grid_name == "structural_core_v1_t128"
    assert structural.name == "structural_core_v1_t128::r12_seg10to12"
    assert structural.canonical_bundle_path.endswith(
        "markov_observed_token_structural_core_v1_t128__r12_seg10to12/markov_data/observed_token_bundle.json"
    )
    assert structural.expanded_bundle_path.endswith(
        "markov_observed_token_structural_core_v1_t128_20x__r12_seg10to12/markov_data/observed_token_bundle.json"
    )
    assert structural.expanded_train_docs_capacity == 20480
    assert structural.config_overrides["min_tokens"] == 128
    assert structural.config_overrides["max_tokens"] == 128

    recoverable_sticky = resolve_full_doc_diagnostic_benchmark("recoverable_v5_t128")
    assert recoverable_sticky.cell_id == "recoverable_v5_t128"
    assert recoverable_sticky.config_overrides["generator_profile"] == "hazard_topic"
    assert abs(float(recoverable_sticky.config_overrides["hazard_switch_prob"]) - (5.0 / 127.0)) < 1e-12
    assert recoverable_sticky.canonical_bundle_path.endswith(
        "markov_observed_token_recoverable_v5_t128/markov_data/observed_token_bundle.json"
    )
    assert recoverable_sticky.expanded_bundle_path.endswith(
        "markov_observed_token_recoverable_20x_v2_t128/markov_data/observed_token_bundle.json"
    )

    sticky_structural_cells = resolve_full_doc_diagnostic_grid("structural_core_v2_t128")
    sticky_structural = next(
        cell for cell in sticky_structural_cells if cell.cell_id == "r12_p079"
    )
    assert sticky_structural.grid_name == "structural_core_v2_t128"
    assert sticky_structural.name == "structural_core_v2_t128::r12_p079"
    assert sticky_structural.config_overrides["generator_profile"] == "hazard_topic"
    assert float(sticky_structural.config_overrides["hazard_switch_prob"]) > 0.0
    assert sticky_structural.canonical_bundle_path.endswith(
        "markov_observed_token_structural_core_v2_t128__r12_p079/markov_data/observed_token_bundle.json"
    )
    assert sticky_structural.expanded_bundle_path.endswith(
        "markov_observed_token_structural_core_v2_t128_20x__r12_p079/markov_data/observed_token_bundle.json"
    )


def test_materialize_base_bundle_prefers_stable_t128_bundle_targets(
    tmp_path: Path,
) -> None:
    benchmark = full_doc_diag.FullDocDiagnosticBenchmarkSpec(
        name="recoverable_v4_t128",
        description="test benchmark",
        observed_token_profile="recoverable",
        canonical_bundle_path=str(tmp_path / "canonical" / "observed_token_bundle.json"),
        expanded_bundle_path=str(tmp_path / "expanded" / "observed_token_bundle.json"),
        canonical_train_docs_capacity=4,
        expanded_train_docs_capacity=16,
        cell_id="recoverable_v4_t128",
    )

    small_bundle, small_source = full_doc_diag._materialize_base_bundle(
        benchmark=benchmark,
        required_train_docs=4,
        output_dir=None,
    )
    assert len(small_bundle.train_docs) == 4
    assert Path(small_source) == Path(benchmark.canonical_bundle_path)
    assert Path(small_source).exists()
    assert {len(doc.tokens) for doc in small_bundle.train_docs} == {128}

    large_bundle, large_source = full_doc_diag._materialize_base_bundle(
        benchmark=benchmark,
        required_train_docs=12,
        output_dir=None,
    )
    assert len(large_bundle.train_docs) == 16
    assert Path(large_source) == Path(benchmark.expanded_bundle_path)
    assert Path(large_source).exists()
    assert {len(doc.tokens) for doc in large_bundle.train_docs} == {128}


def test_stage1_artifact_cache_key_includes_prepared_data_signature() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    base = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
        },
    )

    first = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=replace(base, prepared_data_signature="sig_a"),
    )
    second = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=replace(base, prepared_data_signature="sig_b"),
    )

    assert first.tree_stage1_artifact_dir != second.tree_stage1_artifact_dir


def test_stage1_artifact_cache_key_ignores_superset_budget_fields_under_same_standard_reference() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    base = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 10,
            "tree_stage2_epochs": 30,
            "fixed_leaf_tokens": 128,
            "comparison_mode": "comparable",
            "tree_exact_collapse_mode": "",
            "prepared_data_signature": "shared_sig",
        },
    )

    full100_like = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=replace(
            base,
            leaf_supervision_kind="count_only",
            leaf_label_rate=0.0,
            internal_supervision_kind="none",
            internal_label_rate=0.0,
            budget_total_calls_per_doc=1.0,
            mass_target_per_doc=float("nan"),
            package_semantics="full_doc_only",
        ),
    )
    superset_like = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=replace(
            base,
            local_law_weight=0.8,
            c1_relative_weight=0.4,
            c2_relative_weight=1.0,
            c3_relative_weight=1.0,
            leaf_supervision_kind="count_only",
            leaf_label_rate=0.1,
            internal_supervision_kind="count_only",
            internal_label_rate=0.1,
            budget_total_calls_per_doc=1.0,
            mass_target_per_doc=float("nan"),
            package_semantics="superset",
        ),
    )

    assert full100_like.tree_stage1_artifact_dir == superset_like.tree_stage1_artifact_dir


def test_tree_neural_family_effective_config_preserves_explicit_half_leaf_semantics() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    base = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "local_law_weight": 2.0 / 3.0,
            "c1_relative_weight": 0.5,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.1,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.1,
        },
    )

    effective = full_doc_diag._tree_neural_family_effective_config(
        base,
        family="tree_neural",
    )

    assert effective.local_law_weight == pytest.approx(2.0 / 3.0)
    assert effective.c1_relative_weight == pytest.approx(0.5)
    assert effective.c2_relative_weight == pytest.approx(1.0)
    assert effective.c3_relative_weight == pytest.approx(1.0)
    assert effective.leaf_supervision_kind == "count_only"
    assert effective.leaf_label_rate == pytest.approx(0.1)
    assert effective.internal_supervision_kind == "count_only"
    assert effective.internal_label_rate == pytest.approx(0.1)

    objective = _resolved_objective_metadata_for_run(
        effective,
        baseline_family="tree_neural",
    )
    assert objective["local_law_c1_weight"] == pytest.approx(2.0 / 15.0)
    assert objective["local_law_c2_weight"] == pytest.approx(4.0 / 15.0)
    assert objective["local_law_c3_weight"] == pytest.approx(4.0 / 15.0)


def test_tree_neural_family_effective_config_keeps_exact_collapse_locked_root_only() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    base = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "tree_exact_collapse_mode": "official_fno_one_tree_identity",
            "local_law_weight": 0.8,
            "c1_relative_weight": 0.5,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.1,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.1,
        },
    )

    effective = full_doc_diag._tree_neural_family_effective_config(
        base,
        family="tree_neural",
    )

    assert effective.local_law_weight == pytest.approx(0.0)
    assert effective.c1_relative_weight == pytest.approx(0.0)
    assert effective.c2_relative_weight == pytest.approx(0.0)
    assert effective.c3_relative_weight == pytest.approx(0.0)
    assert effective.leaf_label_rate == pytest.approx(0.0)
    assert effective.internal_label_rate == pytest.approx(0.0)
    assert effective.internal_supervision_kind == "none"


def test_tree_neural_family_effective_config_applies_legacy_c2_profile() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    base = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "local_law_weight": 2.0 / 3.0,
            "c1_relative_weight": 0.5,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 1.0,
        },
    )

    effective = full_doc_diag._tree_neural_family_effective_config(
        base,
        family="tree_neural_c2",
    )

    assert effective.local_law_weight == pytest.approx(2.0 / 3.0)
    assert effective.c1_relative_weight == pytest.approx(0.0)
    assert effective.c2_relative_weight == pytest.approx(1.0)
    assert effective.c3_relative_weight == pytest.approx(0.0)


def test_run_payload_records_tree_semantic_views_and_marks_vacuous_leaf_pressure(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 1,
            "local_law_weight": 0.8,
            "c1_relative_weight": 1.0,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.0,
            "internal_supervision_kind": "none",
            "internal_label_rate": 0.0,
        },
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_run_family_with_predictions(**kwargs):
        effective_config = full_doc_diag._tree_neural_family_effective_config(
            kwargs["config"],
            family="tree_neural",
        )
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": effective_config,
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
    )

    assert payload["semantic_config_views"]["requested"]["leaf_label_rate"] == pytest.approx(0.0)
    assert payload["semantic_config_views"]["effective_pre_family_normalization"]["leaf_label_rate"] == pytest.approx(0.0)
    assert payload["semantic_config_views"]["effective_post_family_normalization"]["leaf_label_rate"] == pytest.approx(0.0)
    assert payload["semantic_config_drift"]["requested_to_pre_family_normalization"] == {}
    assert payload["semantic_config_drift"]["pre_to_post_family_normalization"] == {}
    assert payload["semantic_config_drift"]["config_to_resolved_objective"] == {}
    assert payload["semantic_config_validation_status"] == "validated"
    assert payload["leaf_pressure_ablation_vacuous"] is True
    assert payload["local_pressure_ablation_vacuous"] is True


def test_run_payload_preserves_explicit_half_leaf_semantics_in_raw_summary(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 1,
            "local_law_weight": 2.0 / 3.0,
            "c1_relative_weight": 0.5,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.1,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.1,
        },
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_run_family_with_predictions(**kwargs):
        effective_config = full_doc_diag._tree_neural_family_effective_config(
            kwargs["config"],
            family="tree_neural",
        )
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": effective_config,
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
    )

    assert payload["semantic_config_views"]["requested"]["c1_relative_weight"] == pytest.approx(0.5)
    assert payload["semantic_config_views"]["effective_pre_family_normalization"]["c1_relative_weight"] == pytest.approx(0.5)
    assert payload["semantic_config_views"]["effective_post_family_normalization"]["c1_relative_weight"] == pytest.approx(0.5)
    assert payload["semantic_config_views"]["resolved_objective"]["local_law_c1_weight"] == pytest.approx(2.0 / 15.0)
    assert payload["semantic_config_views"]["resolved_objective"]["local_law_c2_weight"] == pytest.approx(4.0 / 15.0)
    assert payload["semantic_config_views"]["resolved_objective"]["local_law_c3_weight"] == pytest.approx(4.0 / 15.0)
    assert payload["semantic_config_drift"]["requested_to_pre_family_normalization"] == {}
    assert payload["semantic_config_drift"]["pre_to_post_family_normalization"] == {}
    assert payload["semantic_config_drift"]["config_to_resolved_objective"] == {}
    assert payload["leaf_pressure_ablation_vacuous"] is False
    assert payload["semantic_config_validation_status"] == "validated"
    assert payload["run_intent_validation_status"] == "validated"
    assert payload["requested_effective_run_intent_diff"] == {}
    assert payload["effective_reported_run_intent_diff"] == {}
    assert payload["comparison_semantics"] == "current"
    assert payload["requested_run_intent"]["baseline_family"] == "tree_neural"
    assert payload["requested_run_intent"]["c1_relative_weight"] == pytest.approx(0.5)
    assert payload["effective_run_intent"]["c1_relative_weight"] == pytest.approx(0.5)
    assert payload["reported_run_intent"]["c1_relative_weight"] == pytest.approx(0.5)
    assert payload["requested_run_intent"]["tree_c2_mode"] == "reconstruction"
    assert payload["run_intent_hash"]
    assert payload["family_api_group"] == "markov_full_doc_neuraloperator"
    assert payload["law_contract_version"]
    assert payload["law_alignment_status"] == "approximate_audited"
    assert payload["law_contract_gap_count"] == 0


def test_run_payload_raises_on_unexpected_tree_semantic_drift(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 1,
            "local_law_weight": 2.0 / 3.0,
            "c1_relative_weight": 0.5,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.1,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.1,
        },
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_run_family_with_predictions(**kwargs):
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": replace(
                kwargs["config"],
                c1_relative_weight=1.0,
                c2_relative_weight=1.0,
                c3_relative_weight=1.0,
            ),
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    with pytest.raises(ValueError, match="unexpected run intent drift"):
        full_doc_diag._run_payload(
            benchmark=benchmark,
            baseline_family="tree_neural",
            train_doc_count=8,
            config=config,
            seeds={"effective_model_seed": 0},
            device=torch.device("cpu"),
            bundle=bundle,
            bundle_source="synthetic",
            emit_confusion=False,
        )


def test_run_payload_allows_locked_comparator_run_intent_drift(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 1,
            "tree_exact_collapse_mode": "official_fno_runtime_identity",
            "tree_supervision_source": "manifest",
            "budget_total_calls_per_doc": 1.0,
            "full_doc_budget_share": 1.0,
            "doc_consumption_mode": "root_only",
            "local_split_mode": "balanced",
            "local_law_weight": None,
            "task_objective_weight": 1.0,
            "c1_relative_weight": 0.0,
            "c2_relative_weight": 0.0,
            "c3_relative_weight": 0.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.0,
            "internal_supervision_kind": "none",
            "internal_label_rate": 0.0,
        },
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_run_family_with_predictions(**kwargs):
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": replace(
                kwargs["config"],
                c1_relative_weight=1.0,
                c2_relative_weight=1.0,
                c3_relative_weight=1.0,
            ),
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
    )

    assert payload["run_intent_validation_status"] == "locked_comparator"
    assert payload["comparison_semantics"] == "locked_comparator"
    assert payload["requested_effective_run_intent_diff"]
    assert payload["effective_reported_run_intent_diff"] == {}
    assert payload["comparison_semantics_label"] == "locked_comparator"
    assert payload["budget_total_calls_per_doc"] == pytest.approx(1.0)
    assert payload["doc_consumption_mode"] == "root_only"
    assert payload["local_split_mode"] == "balanced"
    assert payload["requested_root_mass_per_doc"] == pytest.approx(1.0)
    assert payload["reported_run_intent"]["budget_total_calls_per_doc"] == pytest.approx(1.0)
    assert payload["reported_run_intent"]["doc_consumption_mode"] == "root_only"
    assert payload["reported_run_intent"]["local_split_mode"] == "balanced"


def test_run_payload_keeps_reported_run_intent_on_effective_config_for_fno(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        baseline_families=("official_fno",),
        config_overrides={
            "n_epochs": 1,
            "budget_total_calls_per_doc": 1.0,
            "full_doc_budget_share": 1.0,
            "doc_consumption_mode": "root_only",
            "local_split_mode": "balanced",
        },
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_run_family_with_predictions(**kwargs):
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": kwargs["config"],
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="official_fno",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
    )

    assert payload["local_split_mode"] == "inactive_for_family"
    assert payload["effective_reported_run_intent_diff"] == {}
    assert payload["reported_run_intent"]["baseline_family"] == "official_fno"
    assert payload["reported_run_intent"]["budget_total_calls_per_doc"] == pytest.approx(1.0)
    assert payload["reported_run_intent"]["doc_consumption_mode"] == "root_only"
    assert payload["reported_run_intent"]["local_split_mode"] == "balanced"


def test_run_payload_reports_max_internal_depth_in_run_intent(
    monkeypatch,
) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4_t128")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 1,
            "fixed_leaf_tokens": 8,
            "max_internal_depth": 3,
            "budget_total_calls_per_doc": 0.5,
            "mass_target_per_doc": 1.0,
            "full_doc_budget_share": 1.0,
            "doc_consumption_mode": "root_only",
            "package_semantics": "mass_matched",
            "local_split_mode": "balanced",
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.125,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.140625,
        },
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_run_family_with_predictions(**kwargs):
        metrics = full_doc_diag._eval_root_predictions(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            tau=float(config.violation_tau),
        )
        fit_diag = TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="val_root_mae",
            selection_split="val",
            selection_metric_name="root_mae",
            selection_metric_value=0.0,
            best_epoch=0,
            train_exact_match_rate=1.0,
            val_exact_match_rate=1.0,
            test_exact_match_rate=1.0,
        )
        return {
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "fit_diag": fit_diag,
            "train_preds": np.asarray([0.0], dtype=np.float64),
            "val_preds": np.asarray([0.0], dtype=np.float64),
            "test_preds": np.asarray([0.0], dtype=np.float64),
            "train_truths": np.asarray([0.0], dtype=np.float64),
            "val_truths": np.asarray([0.0], dtype=np.float64),
            "test_truths": np.asarray([0.0], dtype=np.float64),
            "train_docs_used": 1,
            "effective_config": kwargs["config"],
        }

    monkeypatch.setattr(
        full_doc_diag,
        "_run_family_with_predictions",
        _fake_run_family_with_predictions,
    )

    payload = full_doc_diag._run_payload(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        bundle=bundle,
        bundle_source="synthetic",
        emit_confusion=False,
    )

    assert payload["effective_reported_run_intent_diff"] == {}
    assert payload["config"]["max_internal_depth"] == 3
    assert payload["reported_run_intent"]["max_internal_depth"] == 3


def test_aggregate_runs_separates_distinct_run_intents() -> None:
    base_config = OPSCountConfig(
        comparison_mode="comparable",
        fixed_leaf_tokens=16,
        local_law_weight=0.5,
        task_objective_weight=0.5,
        c1_relative_weight=1.0,
        c2_relative_weight=1.0,
        c3_relative_weight=1.0,
        leaf_supervision_kind="count_only",
        leaf_label_rate=0.2,
        internal_supervision_kind="count_only",
        internal_label_rate=0.2,
        package_semantics="superset",
        depth_discount_gamma=1.0,
    )
    alt_config = replace(
        base_config,
        package_semantics="mass_matched",
        mass_target_per_doc=0.1,
        depth_discount_gamma=0.9,
    )
    runs = []
    for seed, cfg in enumerate((base_config, alt_config)):
        reported_run_intent = materialize_tree_run_intent(
            cfg,
            baseline_family_override="tree_neural",
        )
        run = {
            "benchmark": "recoverable_v4",
            "cell_id": "recoverable_v4",
            "baseline_family": "tree_neural",
            "seed": seed,
            "train_doc_count": 10240,
            "n_regimes": 4,
            "segment_density_band": "",
            "segment_min": 0,
            "segment_max": 0,
            "test_root_mae": 0.3,
            "test_exact_match_rate": 0.5,
            "test_c2_idempotence_mae": 0.0,
            "parameterization": "formal_local_law_weight",
            "weighting_scheme": "normalized_objective",
            "optimization_root_weight": 0.5,
            "local_law_c1_weight": 1.0 / 6.0,
            "local_law_c2_weight": 1.0 / 6.0,
            "local_law_c3_weight": 1.0 / 6.0,
            "task_objective_weight_source": "explicit_task_objective_weight",
            "c2_metric_kind": "score_drift",
            "semantics_version": full_doc_diag.CURRENT_TREE_NEURAL_SEMANTICS_VERSION,
            "requested_run_intent": dict(reported_run_intent),
            "effective_run_intent": dict(reported_run_intent),
            "reported_run_intent": dict(reported_run_intent),
            "run_intent_hash": full_doc_diag.intent_hash(reported_run_intent),
            "run_intent_validation_status": "validated",
            "package_semantics": str(cfg.package_semantics),
            "mass_target_per_doc": float(cfg.mass_target_per_doc),
            "depth_discount_gamma": float(cfg.depth_discount_gamma),
        }
        runs.append(run)

    payload = _payload_from_saved_runs(runs=runs)

    assert len(payload["aggregate_rows"]) == 2
    assert {
        str(row.get("package_semantics", ""))
        for row in payload["aggregate_rows"]
    } == {"superset", "mass_matched"}
    assert len(
        {
            str(row.get("run_intent_hash", ""))
            for row in payload["aggregate_rows"]
        }
    ) == 2


def test_run_intent_hash_separates_official_fno_families() -> None:
    cfg = OPSCountConfig(
        seed=0,
        state_dim=32,
        hidden_dim=64,
        n_epochs=8,
        batch_size=16,
        lr=1e-3,
        weight_decay=0.0,
        fixed_leaf_tokens=128,
    )

    official_intent = materialize_tree_run_intent(
        cfg,
        baseline_family_override="official_fno",
    )
    sumlen_intent = materialize_tree_run_intent(
        cfg,
        baseline_family_override="official_fno_sumlen",
    )

    assert official_intent["baseline_family"] == "official_fno"
    assert sumlen_intent["baseline_family"] == "official_fno_sumlen"
    assert full_doc_diag.intent_hash(official_intent) != full_doc_diag.intent_hash(
        sumlen_intent
    )


def test_loaded_run_backfill_preserves_tree_document_loss_normalization_mode() -> None:
    normalized = _backfill_loaded_run_fields(
        {
            "config": {
                "tree_document_loss_normalization_mode": "auto",
            }
        }
    )

    assert normalized["tree_document_loss_normalization_mode"] == "auto"
    assert normalized["effective_tree_document_loss_normalization_mode"] == "auto"


def test_progress_callback_uses_remaining_stage2_epochs_when_stage1_artifact_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.ctreepo.sim.core.theorem_feature_route import (
        write_theorem_feature_stage1_artifact,
    )

    events: list[dict[str, object]] = []

    def _fake_run_payload(**kwargs):
        return {
            "benchmark": str(kwargs["benchmark"].name),
            "baseline_family": str(kwargs["baseline_family"]),
            "seed": int(kwargs["seeds"]["effective_model_seed"]),
            "train_doc_count": int(kwargs["train_doc_count"]),
        }

    monkeypatch.setattr(full_doc_diag, "_run_payload", _fake_run_payload)

    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    base_config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 10,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_stage1_artifact_root": str(tmp_path / "stage1_cache"),
            "tree_stage1_resume_if_available": True,
        },
    )
    effective = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=base_config,
    )
    write_theorem_feature_stage1_artifact(
        effective.tree_stage1_artifact_dir,
        model_state={"weight": torch.tensor([1.0])},
        metadata={
            "selection_metric_name": "val_root_mae",
            "selection_metric_value": 0.1,
            "best_epoch": 0,
            "epochs_completed": 12,
            "training_schedule": "two_stage",
            "artifact_source": "trained",
            "n_regimes": int(effective.n_regimes),
            "vocab_size": int(effective.vocab_size),
            "generator_profile": str(effective.generator_profile),
            "fixed_leaf_tokens": int(effective.fixed_leaf_tokens),
            **full_doc_diag._tree_stage1_expected_layout_metadata(effective),
        },
    )

    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="recoverable_v4",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_neural",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 10,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_stage1_artifact_root": str(tmp_path / "stage1_cache"),
            "tree_stage1_resume_if_available": True,
        },
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert payload["runs"]
    run_start = next(event for event in events if str(event.get("stage", "")) == "run_start")
    assert int(run_start["epochs_total"]) == 20


def test_stage1_artifact_resume_requires_layout_compatible_metadata(
    tmp_path: Path,
) -> None:
    from src.ctreepo.sim.core.theorem_feature_route import (
        write_theorem_feature_stage1_artifact,
    )

    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")
    base_config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_stage1_artifact_root": str(tmp_path / "stage1_cache"),
            "tree_stage1_resume_if_available": True,
        },
    )
    effective = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=base_config,
    )
    write_theorem_feature_stage1_artifact(
        effective.tree_stage1_artifact_dir,
        model_state={"weight": torch.tensor([1.0])},
        metadata={
            "selection_metric_name": "val_root_mae",
            "selection_metric_value": 0.1,
            "best_epoch": 0,
            "epochs_completed": 12,
            "training_schedule": "two_stage",
            "artifact_source": "trained",
            "n_regimes": int(effective.n_regimes),
            "vocab_size": int(effective.vocab_size),
            "fixed_leaf_tokens": int(effective.fixed_leaf_tokens),
        },
    )

    resumed = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family="tree_neural",
        train_doc_count=8,
        config=base_config,
    )

    assert int(resumed.tree_stage1_epochs) == int(base_config.tree_stage1_epochs)


def test_inactive_objective_metadata_aggregates_without_nan_grouping_drift() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "smoke",
                "cell_id": "smoke",
                "baseline_family": "raw_token_ngram_ridge",
                "seed": 0,
                "train_doc_count": 8,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.1,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.0,
            },
            {
                "benchmark": "smoke",
                "cell_id": "smoke",
                "baseline_family": "raw_token_ngram_ridge",
                "seed": 1,
                "train_doc_count": 8,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.2,
                "test_exact_match_rate": 0.8,
                "test_c2_idempotence_mae": 0.0,
            },
        ]
    )
    assert len(payload["aggregate_rows"]) == 1
    row = payload["aggregate_rows"][0]
    assert row["parameterization"] == "inactive_for_family"
    assert row["optimization_root_weight"] == 0.0
    assert row["objective_weights_active"] is False
    assert row["backend_name"] == "ridge_control"
    assert row["operator_evidence_status"] == "PROXY_ONLY"


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_doc_sequence_budget_uses_root_class_support(monkeypatch) -> None:
    config = OPSCountConfig(
        train_docs=8,
        val_docs=2,
        test_docs=2,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        n_epochs=1,
        state_dim=8,
        hidden_dim=16,
        batch_size=4,
        lr=1e-3,
        use_cuda=False,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    captured: dict[str, object] = {}

    def _fake_train_fno_tree(*, model, train_docs, val_docs, **kwargs):
        captured["doc_sequence_class_values"] = tuple(
            int(v) for v in model.doc_sequence_class_values.detach().cpu().tolist()
        )
        captured["root_count_class_values"] = tuple(
            int(v) for v in model.root_count_class_values.detach().cpu().tolist()
        )
        captured["doc_sequence_class_index"] = dict(kwargs["doc_sequence_class_index"])
        return {
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=1,
                selection_metric_curve=(0.0,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name="root_mae",
                selection_metric_value=0.0,
                best_epoch=1,
            )
        }

    def _fake_eval_fno_model(_model, docs, *, device, tau):
        preds = [0.0 for _ in docs]
        truths = [float(doc.root_count) for doc in docs]
        return full_doc_diag._eval_root_predictions(preds, truths, tau=float(tau))

    monkeypatch.setattr(full_doc_diag, "train_fno_tree", _fake_train_fno_tree)
    monkeypatch.setattr(full_doc_diag, "_eval_fno_model", _fake_eval_fno_model)
    monkeypatch.setattr(
        full_doc_diag,
        "_fno_tree_root_predictions",
        lambda _model, docs, *, device: np.zeros((len(docs),), dtype=np.float64),
    )

    full_doc_diag._fit_tree_neural_baseline_with_predictions(
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        train_docs=bundle.train_docs,
        val_docs=bundle.val_docs,
        test_docs=bundle.test_docs,
        budget_manifest=None,
    )

    doc_sequence_class_values = tuple(captured["doc_sequence_class_values"])
    root_count_class_values = tuple(captured["root_count_class_values"])
    class_index = dict(captured["doc_sequence_class_index"])

    assert doc_sequence_class_values == root_count_class_values
    assert set(class_index.keys()).issubset(set(doc_sequence_class_values))
    assert doc_sequence_class_values == tuple(
        range(
            int(min(doc_sequence_class_values)),
            int(max(doc_sequence_class_values)) + 1,
        )
    )


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_neural_progress_snapshots_follow_artifact_dir(monkeypatch, tmp_path: Path) -> None:
    config = OPSCountConfig(
        train_docs=8,
        val_docs=2,
        test_docs=2,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        n_epochs=3,
        state_dim=8,
        hidden_dim=16,
        batch_size=4,
        lr=1e-3,
        use_cuda=False,
        artifact_dir=str(tmp_path / "artifacts"),
        tree_progress_snapshot_interval=7,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    captured: dict[str, object] = {}

    def _fake_train_fno_tree(*, model, train_docs, val_docs, **kwargs):
        captured["progress_snapshot_interval"] = int(kwargs["progress_snapshot_interval"])
        captured["progress_snapshot_dir"] = str(kwargs["progress_snapshot_dir"])
        return {
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=3,
                selection_metric_curve=(0.0,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name="root_mae",
                selection_metric_value=0.0,
                best_epoch=1,
            ),
            "progress_snapshot_interval": 7,
            "progress_snapshot_dir": str(
                Path(str(kwargs["progress_snapshot_dir"])).expanduser()
            ),
            "latest_progress_snapshot_path": str(
                Path(str(kwargs["progress_snapshot_dir"])).expanduser()
                / "single_stage_train__epoch_0003.json"
            ),
        }

    def _fake_eval_fno_model(_model, docs, *, device, tau):
        preds = [0.0 for _ in docs]
        truths = [float(doc.root_count) for doc in docs]
        return full_doc_diag._eval_root_predictions(preds, truths, tau=float(tau))

    monkeypatch.setattr(full_doc_diag, "train_fno_tree", _fake_train_fno_tree)
    monkeypatch.setattr(full_doc_diag, "_eval_fno_model", _fake_eval_fno_model)
    monkeypatch.setattr(
        full_doc_diag,
        "_fno_tree_root_predictions",
        lambda _model, docs, *, device: np.zeros((len(docs),), dtype=np.float64),
    )

    result = full_doc_diag._fit_tree_neural_baseline_with_predictions(
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        train_docs=bundle.train_docs,
        val_docs=bundle.val_docs,
        test_docs=bundle.test_docs,
        budget_manifest=None,
    )

    expected_snapshot_dir = str((tmp_path / "artifacts" / "training_progress").expanduser())
    assert captured["progress_snapshot_interval"] == 7
    assert captured["progress_snapshot_dir"] == expected_snapshot_dir
    assert result["training_progress_snapshot_interval"] == 7
    assert result["training_progress_snapshot_dir"] == expected_snapshot_dir
    assert result["latest_training_progress_snapshot_path"].endswith(
        "single_stage_train__epoch_0003.json"
    )


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_neural_exact_sketch_diagnostics_expose_exact_witness(
    monkeypatch,
) -> None:
    config = OPSCountConfig(
        train_docs=8,
        val_docs=2,
        test_docs=2,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        n_epochs=1,
        state_dim=8,
        hidden_dim=16,
        batch_size=4,
        lr=1e-3,
        tree_root_supervision_kind="count_ce",
        use_cuda=False,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_train_fno_tree(*, model, train_docs, val_docs, **kwargs):
        return {
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=1,
                selection_metric_curve=(0.0,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name="root_mae",
                selection_metric_value=0.0,
                best_epoch=1,
            )
        }

    def _fake_eval_fno_model(_model, docs, *, device, tau):
        preds = [0.0 for _ in docs]
        truths = [float(doc.root_count) for doc in docs]
        return full_doc_diag._eval_root_predictions(preds, truths, tau=float(tau))

    monkeypatch.setattr(full_doc_diag, "train_fno_tree", _fake_train_fno_tree)
    monkeypatch.setattr(full_doc_diag, "_eval_fno_model", _fake_eval_fno_model)
    monkeypatch.setattr(
        full_doc_diag,
        "_fno_tree_root_predictions",
        lambda _model, docs, *, device: np.zeros((len(docs),), dtype=np.float64),
    )

    result = full_doc_diag._fit_tree_neural_baseline_with_predictions(
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        train_docs=bundle.train_docs,
        val_docs=bundle.val_docs,
        test_docs=bundle.test_docs,
        budget_manifest=None,
    )

    diagnostics = dict(result.get("exact_sketch_diagnostics") or {})
    assert diagnostics["paper_to_lean_local_law_mapping"] == {
        "C1": "L1",
        "C2": "L3",
        "C3": "L2",
    }
    theorem_contract = dict(diagnostics.get("theorem_contract") or {})
    assert theorem_contract["summary_ref"] == "MarkovCountSketch"
    assert theorem_contract["codec_ref"] == "SketchCodecExactAssumptions"
    assert theorem_contract["bundle_ref"] == "approx_bundle_of_nodewise"
    observed_token_recoverability = dict(
        theorem_contract.get("observed_token_recoverability") or {}
    )
    assert observed_token_recoverability["generator_profile"] == str(
        config.generator_profile
    )
    assert observed_token_recoverability["lean_recoverable_in_principle"] is False
    assert observed_token_recoverability["lean_bayes_error_zero"] is False
    theorem_refs = dict(observed_token_recoverability.get("theorem_refs") or {})
    assert (
        theorem_refs["observed_token_path_ref"]
        == "piecewise_disjoint_palette_observed_tokens_recover_latent_path"
    )
    assert (
        theorem_refs["observed_token_exact_sketch_ref"]
        == "piecewise_disjoint_palette_observed_tokens_recover_exact_sketch"
    )
    assert theorem_refs["zero_bayes_error_ref"] == "piecewise_disjoint_palette_zero_bayes_error"
    assert diagnostics["schedule_proxy_status"] == "PROXY_ONLY"
    assert diagnostics["aligned_sketch_surface"] == ""
    assert diagnostics["internal_supervision_kind"] == "none"
    assert diagnostics["internal_label_rate"] == pytest.approx(0.0)
    assert diagnostics["leaf_exact_supervision"] is False
    assert diagnostics["leaf_supervision_kind"] == "full_sketch"
    assert diagnostics["summary_spec_name"] == ""
    assert diagnostics["slot_count"] == 0
    assert diagnostics["tree_theorem_count_head_mode"] == "scalar_mse"
    assert diagnostics["tree_theorem_count_dim"] == 0
    assert diagnostics["tree_theorem_first_dim"] == 0
    assert diagnostics["tree_theorem_last_dim"] == 0
    assert diagnostics["tree_training_schedule"] == "two_stage"
    witness_test = dict((diagnostics.get("exact_witness") or {}).get("test") or {})
    assert witness_test["law_metrics"]["root_mae"] == pytest.approx(0.0, abs=1e-9)
    assert witness_test["law_metrics"]["leaf_mae"] == pytest.approx(0.0, abs=1e-9)
    assert witness_test["law_metrics"]["c2_idempotence_mae"] == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert witness_test["law_metrics"]["merge_mae"] == pytest.approx(0.0, abs=1e-9)
    for level in ("leaf", "merge", "root"):
        assert witness_test[level]["direct"]["exact_summary_match_rate"] == pytest.approx(
            1.0,
            abs=1e-9,
        )
        assert witness_test[level]["probe_control"][
            "exact_summary_match_rate"
        ] == pytest.approx(1.0, abs=1e-9)
    tree_test = dict((diagnostics.get("tree_neural") or {}).get("test") or {})
    assert np.isfinite(tree_test["leaf"]["probe"]["count_mae"])
    assert np.isfinite(tree_test["merge"]["probe"]["count_mae"])
    assert np.isfinite(tree_test["root"]["probe"]["count_mae"])
    merge_decoded_consistency = dict(
        (tree_test.get("merge") or {}).get("decoded_consistency") or {}
    )
    assert "merge_join_bit_accuracy" in merge_decoded_consistency
    assert "merge_decoded_consistency_count_mae" in merge_decoded_consistency
    assert "merge_decoded_consistency_first_accuracy" in merge_decoded_consistency
    assert "merge_decoded_consistency_last_accuracy" in merge_decoded_consistency
    direct_selection = dict((diagnostics.get("direct_selection_metrics") or {}).get("test") or {})
    assert "task_root_mae" in direct_selection
    assert "task_root_mae_ablation" in direct_selection
    assert "c2_on_range_exact_match" in direct_selection
    assert "val_leaf_codec_direct" in direct_selection
    assert "val_theorem_bootstrap_direct" in direct_selection
    assert "phi_pair_same_accuracy" in direct_selection
    assert "phi_pair_diff_accuracy" in direct_selection
    assert "phi_pair_auc" in direct_selection
    assert "phi_replay_same_class_rate" in direct_selection
    assert "task_factorization_gap" in direct_selection
    assert "first_leaf_direct_accuracy" in direct_selection
    assert "last_leaf_direct_accuracy" in direct_selection
    assert "leaf_direct_probe_exact_gap" in direct_selection
    assert "merge_direct_probe_exact_gap" in direct_selection
    theorem_sections = dict(diagnostics.get("theorem_sections") or {})
    assert "C1_to_L1" in theorem_sections
    assert "C3_to_L2" in theorem_sections
    assert "C2_to_L3" in theorem_sections
    assert result["exact_sketch_failure_bucket"] in {
        "phi_not_sufficient",
        "phi_not_compositional",
        "leaf_boundary_encoding_gap",
        "theorem_count_decode_gap",
        "count_composition_gap",
        "subtree_label_value_gap",
        "internal_label_value_gap",
        "legacy_readout_gap",
        "insufficient_data",
    }


def test_markov_observed_token_recoverability_contract_tags_recoverable_cells() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")

    contract = full_doc_diag._markov_observed_token_recoverability_contract(
        benchmark=benchmark
    )

    assert contract["generator_profile"] == "piecewise_disjoint_palette"
    assert contract["lean_recoverable_in_principle"] is True
    assert contract["lean_bayes_error_zero"] is True


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_tree_neural_fit_exposes_teacher_first_decomposition(monkeypatch) -> None:
    config = OPSCountConfig(
        train_docs=8,
        val_docs=2,
        test_docs=2,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        n_epochs=1,
        state_dim=8,
        hidden_dim=16,
        batch_size=4,
        lr=1e-3,
        use_cuda=False,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    def _fake_train_fno_tree(*, model, train_docs, val_docs, **kwargs):
        return {
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=1,
                selection_metric_curve=(0.0,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name="root_mae",
                selection_metric_value=0.0,
                best_epoch=1,
            ),
            "stage1_best_model_state": {"dummy": torch.tensor([0.0])},
            "stage1_artifact": {"artifact_dir": "/tmp/stage1"},
        }

    def _fake_eval_fno_model(_model, docs, *, device, tau):
        preds = [0.0 for _ in docs]
        truths = [float(doc.root_count) for doc in docs]
        return full_doc_diag._eval_root_predictions(preds, truths, tau=float(tau))

    monkeypatch.setattr(full_doc_diag, "train_fno_tree", _fake_train_fno_tree)
    monkeypatch.setattr(full_doc_diag, "_eval_fno_model", _fake_eval_fno_model)
    monkeypatch.setattr(
        full_doc_diag,
        "_fno_tree_root_predictions",
        lambda _model, docs, *, device: np.zeros((len(docs),), dtype=np.float64),
    )
    monkeypatch.setattr(
        full_doc_diag,
        "_tree_exact_sketch_diagnostics",
        lambda **kwargs: {"failure_attribution": {}},
    )
    monkeypatch.setattr(
        full_doc_diag,
        "_eval_fno_teacher_first_decomposition_metrics",
        lambda *args, **kwargs: {
            "stage2_transport_budget": 0.3,
            "stage2_leaf_transport_mae": 0.1,
            "stage2_merge_transport_mae": 0.2,
            "stage2_fiber_error": 0.4,
            "stage2_fiber_pair_same_accuracy": 0.8,
            "stage2_fiber_pair_diff_accuracy": 0.7,
            "stage2_fiber_pair_auc": 0.75,
            "root_measurement_error": 0.2,
            "stage1_substitution_cost": 0.1,
            "teacher_first_total_bound": 1.0,
        },
    )

    result = full_doc_diag._fit_tree_neural_baseline_with_predictions(
        config=config,
        seeds={"effective_model_seed": 0},
        device=torch.device("cpu"),
        train_docs=bundle.train_docs,
        val_docs=bundle.val_docs,
        test_docs=bundle.test_docs,
        budget_manifest=None,
    )

    assert result["stage1_artifact"]["artifact_dir"] == "/tmp/stage1"
    assert result["teacher_first_decomposition"]["test"][
        "stage2_transport_budget"
    ] == pytest.approx(0.3)


def test_exact_sketch_raw_artifacts_write_to_disk(tmp_path: Path) -> None:
    config = OPSCountConfig(
        diagnostic_detail_mode="debug_raw",
        raw_diagnostic_artifact_dir=str(tmp_path / "raw_exact"),
    )
    records = {
        "leaf": {
            "state_features": np.ones((2, 3), dtype=np.float64),
            "phi_features": np.ones((2, 2), dtype=np.float64),
            "count_targets": np.asarray([0, 1], dtype=np.int64),
            "first_targets": np.asarray([0, 1], dtype=np.int64),
            "last_targets": np.asarray([1, 0], dtype=np.int64),
        },
        "merge": {
            "state_features": np.ones((1, 3), dtype=np.float64),
            "phi_features": np.ones((1, 2), dtype=np.float64),
            "count_targets": np.asarray([1], dtype=np.int64),
            "first_targets": np.asarray([0], dtype=np.int64),
            "last_targets": np.asarray([1], dtype=np.int64),
        },
        "root": {
            "state_features": np.ones((1, 3), dtype=np.float64),
            "phi_features": np.ones((1, 2), dtype=np.float64),
            "count_targets": np.asarray([1], dtype=np.int64),
            "first_targets": np.asarray([0], dtype=np.int64),
            "last_targets": np.asarray([1], dtype=np.int64),
        },
    }
    artifacts = full_doc_diag._write_tree_exact_split_raw_artifacts(
        config=config,
        split="test",
        records=records,
    )
    assert Path(artifacts["leaf"]).exists()
    assert Path(artifacts["merge"]).exists()
    assert Path(artifacts["root"]).exists()
    metadata = json.loads(Path(artifacts["metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["split"] == "test"
    assert metadata["levels"]["leaf"]["state_features"] == [2, 3]


def test_payload_exposes_score_contract_and_split_metrics() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "train_root_mae": 0.30,
                "val_root_mae": 0.20,
                "test_root_mae": 0.10,
                "train_exact_match_rate": 0.70,
                "val_exact_match_rate": 0.80,
                "test_exact_match_rate": 0.90,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 1,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "train_root_mae": 0.50,
                "val_root_mae": 0.25,
                "test_root_mae": 0.15,
                "train_exact_match_rate": 0.60,
                "val_exact_match_rate": 0.75,
                "test_exact_match_rate": 0.85,
                "test_c2_idempotence_mae": 0.02,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
        ]
    )
    assert payload["primary_report_metric"] == "test_root_mae_mean"
    assert payload["primary_report_split"] == "test"
    assert payload["primary_report_target"] == "root_count"
    assert payload["primary_report_weighting"] == "unweighted_mae"
    assert payload["dev_selection_metric"] == "val_root_mae_mean"
    run = payload["runs"][0]
    assert run["train_root_mae"] == pytest.approx(0.30)
    assert run["val_root_mae"] == pytest.approx(0.20)
    assert run["test_root_mae"] == pytest.approx(0.10)
    assert run["train_exact_match_rate"] == pytest.approx(0.70)
    assert run["val_exact_match_rate"] == pytest.approx(0.80)
    assert run["test_exact_match_rate"] == pytest.approx(0.90)
    aggregate_row = payload["aggregate_rows"][0]
    assert aggregate_row["train_root_mae_mean"] == pytest.approx(0.40)
    assert aggregate_row["val_root_mae_mean"] == pytest.approx(0.225)
    assert aggregate_row["test_root_mae_mean"] == pytest.approx(0.125)
    assert aggregate_row["train_exact_match_rate_mean"] == pytest.approx(0.65)
    assert aggregate_row["val_exact_match_rate_mean"] == pytest.approx(0.775)
    assert aggregate_row["test_exact_match_rate_mean"] == pytest.approx(0.875)


def test_payload_from_saved_runs_backfills_legacy_split_metrics_from_nested_fields() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_metrics": {
                    "root_mae": 0.10,
                    "leaf_mae": 0.20,
                    "c2_idempotence_mae": 0.30,
                    "merge_mae": 0.40,
                    "schedule_spread_mean": 9.0,
                },
                "train_metrics": {
                    "root_mae": 0.50,
                    "leaf_mae": 0.60,
                    "merge_mae": 0.70,
                    "schedule_spread_mean": 1.5,
                },
                "val_metrics": {
                    "root_mae": 0.25,
                    "leaf_mae": 0.35,
                    "merge_mae": 0.45,
                    "schedule_spread_mean": 1.0,
                },
                "fit_diagnostics": {
                    "train_exact_match_rate": 0.55,
                    "val_exact_match_rate": 0.65,
                    "test_exact_match_rate": 0.75,
                },
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "objective_weights_active": True,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "task_objective_weight_source": "configured_objective_builder",
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            }
        ]
    )
    run = payload["runs"][0]
    assert run["train_root_mae"] == pytest.approx(0.50)
    assert run["val_root_mae"] == pytest.approx(0.25)
    assert run["train_leaf_mae"] == pytest.approx(0.60)
    assert run["val_leaf_mae"] == pytest.approx(0.35)
    assert run["train_merge_mae"] == pytest.approx(0.70)
    assert run["val_merge_mae"] == pytest.approx(0.45)
    assert run["train_schedule_spread_mean"] == pytest.approx(1.5)
    assert run["val_schedule_spread_mean"] == pytest.approx(1.0)
    assert run["train_exact_match_rate"] == pytest.approx(0.55)
    assert run["val_exact_match_rate"] == pytest.approx(0.65)
    aggregate_row = payload["aggregate_rows"][0]
    assert aggregate_row["train_root_mae_mean"] == pytest.approx(0.50)
    assert aggregate_row["val_root_mae_mean"] == pytest.approx(0.25)


def test_payload_from_saved_runs_flattens_exact_sketch_direct_metrics() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 1024,
                "n_regimes": 4,
                "test_metrics": {"root_mae": 0.10},
                "train_metrics": {"root_mae": 0.20},
                "val_metrics": {"root_mae": 0.15},
                "fit_diagnostics": {
                    "train_exact_match_rate": 0.6,
                    "val_exact_match_rate": 0.7,
                    "test_exact_match_rate": 0.8,
                },
                "exact_sketch_diagnostics": {
                    "direct_selection_metrics": {
                        "test": {
                            "root_direct_count_mae": 0.12,
                            "leaf_direct_exact_match": 0.71,
                            "merge_direct_exact_match": 0.83,
                            "merge_join_bit_accuracy": 0.94,
                            "phi_merge_alignment": 0.91,
                            "phi_within_class_variance": 0.14,
                            "phi_between_class_margin": 1.42,
                            "phi_direct_probe_leaf_gap": 0.09,
                            "phi_direct_probe_merge_gap": 0.11,
                            "leaf_count_head_entropy_mean": 0.55,
                            "merge_count_head_entropy_mean": 0.66,
                            "leaf_count_head_margin_mean": 0.21,
                            "merge_count_head_margin_mean": 0.31,
                        }
                    },
                    "failure_attribution": {
                        "bucket": "theorem_count_decode_gap",
                        "theorem_count_decode_gap_score": 1.5,
                        "phi_not_sufficient_score": 0.4,
                        "phi_not_compositional_score": 0.2,
                    },
                },
            }
        ]
    )
    run = payload["runs"][0]
    assert run["test_root_direct_count_mae"] == pytest.approx(0.12)
    assert run["test_leaf_direct_exact_summary_match_rate"] == pytest.approx(0.71)
    assert run["test_merge_direct_exact_summary_match_rate"] == pytest.approx(0.83)
    assert run["test_merge_join_bit_accuracy"] == pytest.approx(0.94)
    assert run["phi_merge_alignment"] == pytest.approx(0.91)
    assert run["phi_within_class_variance"] == pytest.approx(0.14)
    assert run["phi_between_class_margin"] == pytest.approx(1.42)
    assert run["phi_direct_probe_leaf_gap"] == pytest.approx(0.09)
    assert run["phi_direct_probe_merge_gap"] == pytest.approx(0.11)
    assert run["leaf_count_head_entropy_mean"] == pytest.approx(0.55)
    assert run["merge_count_head_margin_mean"] == pytest.approx(0.31)
    assert run["exact_sketch_failure_bucket"] == "theorem_count_decode_gap"
    assert run["exact_sketch_theorem_count_decode_gap_score"] == pytest.approx(1.5)
    assert run["exact_sketch_phi_not_sufficient_score"] == pytest.approx(0.4)
    assert run["exact_sketch_phi_not_compositional_score"] == pytest.approx(0.2)


def test_payload_backfills_tree_parity_metadata_from_nested_config() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.05,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.01,
                "config_label": FAIR_FNO_PARITY_CONFIG_LABEL,
                "config": {
                    "tree_root_supervision_kind": "count_ce",
                    "tree_leaf_fno_width": 128,
                    "tree_leaf_fno_n_modes": 8,
                    "tree_leaf_fno_n_layers": 4,
                    "doc_sequence_train_fraction": 0.0,
                },
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            }
        ]
    )
    run = payload["runs"][0]
    aggregate_row = payload["aggregate_rows"][0]
    assert run["tree_root_supervision_kind"] == "count_ce"
    assert run["tree_leaf_fno_width"] == 128
    assert run["tree_leaf_fno_n_modes"] == 8
    assert run["tree_leaf_fno_n_layers"] == 4
    assert run["tree_aux_doc_sequence_fraction"] == pytest.approx(0.0)
    assert aggregate_row["tree_root_supervision_kind"] == "count_ce"
    assert aggregate_row["tree_leaf_fno_width"] == 128
    assert aggregate_row["tree_leaf_fno_n_modes"] == 8
    assert aggregate_row["tree_leaf_fno_n_layers"] == 4
    assert aggregate_row["tree_aux_doc_sequence_fraction"] == pytest.approx(0.0)


def test_payload_emits_tree_fno_fair_parity_summary() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "official_fno",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.040,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.0,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "official_fno_sumlen",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.038,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.0,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": FAIR_FNO_PARITY_CONFIG_LABEL,
                "tree_root_supervision_kind": "count_ce",
                "tree_leaf_fno_width": 128,
                "tree_leaf_fno_n_modes": 8,
                "tree_leaf_fno_n_layers": 4,
                "tree_aux_doc_sequence_fraction": 0.0,
                "test_root_mae": 0.039,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.3,
                "local_law_c3_weight": 0.0,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2c3",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": FAIR_FNO_PARITY_CONFIG_LABEL,
                "tree_root_supervision_kind": "count_ce",
                "tree_leaf_fno_width": 128,
                "tree_leaf_fno_n_modes": 8,
                "tree_leaf_fno_n_layers": 4,
                "tree_aux_doc_sequence_fraction": 0.0,
                "test_root_mae": 0.041,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.2,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": FAIR_FNO_PARITY_CONFIG_LABEL,
                "tree_root_supervision_kind": "count_ce",
                "tree_leaf_fno_width": 128,
                "tree_leaf_fno_n_modes": 8,
                "tree_leaf_fno_n_layers": 4,
                "tree_aux_doc_sequence_fraction": 0.0,
                "test_root_mae": 0.0415,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
        ]
    )
    summary = dict(payload.get("tree_fno_fair_parity_summary") or {})
    assert summary["parity_config_label"] == FAIR_FNO_PARITY_CONFIG_LABEL
    assert summary["tree_root_supervision_kind"] == "count_ce"
    assert summary["tree_leaf_fno_width"] == 128
    assert summary["tree_leaf_fno_n_modes"] == 8
    assert summary["tree_leaf_fno_n_layers"] == 4
    assert summary["tree_aux_doc_sequence_fraction"] == pytest.approx(0.0)
    assert summary["best_full_doc_fno_family_at_gate"] == "official_fno_sumlen"
    assert summary["best_parity_tree_family_at_gate"] == "tree_neural_c2"
    assert summary["primary_success_met"] is True
    assert summary["secondary_success_met"] is True


def test_payload_emits_tree_fno_upper_bound_summary() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "official_fno_sumlen",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.038,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.0,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": "fair_fno_v1_aux25",
                "tree_root_supervision_kind": "count_ce",
                "tree_leaf_fno_width": 128,
                "tree_leaf_fno_n_modes": 8,
                "tree_leaf_fno_n_layers": 4,
                "tree_aux_doc_sequence_fraction": 0.25,
                "test_root_mae": 0.037,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": "fair_fno_v1_aux100",
                "tree_root_supervision_kind": "count_ce",
                "tree_leaf_fno_width": 128,
                "tree_leaf_fno_n_modes": 8,
                "tree_leaf_fno_n_layers": 4,
                "tree_aux_doc_sequence_fraction": 1.0,
                "test_root_mae": 0.036,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.3,
                "local_law_c3_weight": 0.0,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
        ]
    )
    summary = dict(payload.get("tree_fno_upper_bound_summary") or {})
    assert summary["gate_train_doc_count"] == 10240
    assert summary["aux_fractions"] == [0.25, 1.0]
    assert summary["best_gate_upper_bound_family"] == "tree_neural_c2"
    assert summary["best_gate_aux_fraction"] == pytest.approx(1.0)


def test_payload_emits_tree_oracle_budget_frontier_summary() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.08,
                "test_exact_match_rate": 0.8,
                "test_c2_idempotence_mae": 0.02,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "budget_total_calls": 5120,
                "budget_total_calls_per_doc": 0.5,
                "full_doc_budget_share": 0.5,
                "full_doc_calls_total": 2560,
                "local_calls_total": 2560,
                "doc_consumption_mode": "root_only",
                "local_split_mode": "balanced",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 3840.0,
                "effective_full_doc_mass_per_doc": 0.375,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.07,
                "test_exact_match_rate": 0.82,
                "test_c2_idempotence_mae": 0.015,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.3,
                "local_law_c3_weight": 0.0,
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "budget_total_calls": 10240,
                "budget_total_calls_per_doc": 1.0,
                "full_doc_budget_share": 0.25,
                "full_doc_calls_total": 2560,
                "local_calls_total": 7680,
                "doc_consumption_mode": "doc_sequence",
                "local_split_mode": "leaf_heavy",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 5120.0,
                "effective_full_doc_mass_per_doc": 0.5,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "official_fno",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.05,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.0,
                "budget_total_calls": 10240,
                "budget_total_calls_per_doc": 1.0,
                "full_doc_budget_share": 1.0,
                "full_doc_calls_total": 10240,
                "local_calls_total": 0,
                "doc_consumption_mode": "full_doc_only",
                "local_split_mode": "inactive_for_family",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 10240.0,
                "effective_full_doc_mass_per_doc": 1.0,
            },
        ]
    )
    summary = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
    assert summary["study_name"] == "oracle_budget_share_frontier"
    assert summary["budget_levels_per_doc"] == [0.5, 1.0]
    assert summary["full_doc_budget_shares"] == [0.25, 0.5, 1.0]
    assert summary["best_tree_by_budget"][0]["baseline_family"] == "tree_neural"
    assert summary["best_reference_by_budget"][0]["baseline_family"] == "official_fno"


def test_payload_aggregates_elapsed_runtime_without_grouping_collision() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": "fair_fno_v1_w64_m2_l2",
                "tree_leaf_fno_width": 64,
                "tree_leaf_fno_n_modes": 2,
                "tree_leaf_fno_n_layers": 2,
                "tree_root_supervision_kind": "count_ce",
                "tree_aux_doc_sequence_fraction": 0.0,
                "elapsed_s": 12.0,
                "test_root_mae": 0.1,
                "val_root_mae": 0.2,
                "test_exact_match_rate": 0.8,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 1,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": "fair_fno_v1_w64_m2_l2",
                "tree_leaf_fno_width": 64,
                "tree_leaf_fno_n_modes": 2,
                "tree_leaf_fno_n_layers": 2,
                "tree_root_supervision_kind": "count_ce",
                "tree_aux_doc_sequence_fraction": 0.0,
                "elapsed_s": 16.0,
                "test_root_mae": 0.11,
                "val_root_mae": 0.21,
                "test_exact_match_rate": 0.81,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "config_label": "fair_fno_v1_aux25",
                "tree_leaf_fno_width": 64,
                "tree_leaf_fno_n_modes": 2,
                "tree_leaf_fno_n_layers": 2,
                "tree_root_supervision_kind": "count_ce",
                "tree_aux_doc_sequence_fraction": 0.25,
                "elapsed_s": 20.0,
                "test_root_mae": 0.09,
                "val_root_mae": 0.19,
                "test_exact_match_rate": 0.82,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            },
        ]
    )
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    assert len(aggregate_rows) == 2
    pure_row = next(row for row in aggregate_rows if row["config_label"] == "fair_fno_v1_w64_m2_l2")
    aux_row = next(row for row in aggregate_rows if row["config_label"] == "fair_fno_v1_aux25")
    assert pure_row["elapsed_s_mean"] == pytest.approx(14.0)
    assert aux_row["elapsed_s_mean"] == pytest.approx(20.0)


def test_unweighted_test_objectives_sum_active_terms_and_exclude_schedule_spread() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2c3",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "val_root_mae": 0.5,
                "val_leaf_mae": 1.0,
                "val_c2_idempotence_mae": 1.5,
                "val_merge_mae": 2.0,
                "test_root_mae": 1.0,
                "test_leaf_mae": 2.0,
                "test_c2_idempotence_mae": 3.0,
                "test_merge_mae": 4.0,
                "test_schedule_spread_mean": 100.0,
                "test_exact_match_rate": 0.9,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.2,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            }
        ]
    )
    run = payload["runs"][0]
    assert run["val_unweighted_full_law_objective"] == pytest.approx(5.0)
    assert run["val_unweighted_active_objective"] == pytest.approx(4.0)
    assert run["test_unweighted_full_law_objective"] == pytest.approx(10.0)
    assert run["test_unweighted_active_objective"] == pytest.approx(8.0)
    aggregate_row = payload["aggregate_rows"][0]
    assert aggregate_row["val_unweighted_full_law_objective_mean"] == pytest.approx(5.0)
    assert aggregate_row["val_unweighted_active_objective_mean"] == pytest.approx(4.0)
    assert aggregate_row["test_unweighted_full_law_objective_mean"] == pytest.approx(10.0)
    assert aggregate_row["test_unweighted_active_objective_mean"] == pytest.approx(8.0)


def test_root_mae_matches_plain_unweighted_confusion_mae(tmp_path: Path) -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("smoke")
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=8,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "n_epochs": 1,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    seeds, device = _resolve_device(config)
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    fit = _run_family_with_predictions(
        baseline_family="raw_token_ngram_ridge",
        config=config,
        seeds=seeds,
        device=device,
        train_docs=tuple(bundle.train_docs),
        val_docs=tuple(bundle.val_docs),
        test_docs=tuple(bundle.test_docs),
    )
    manual_mae = float(
        sum(
            abs(float(pred) - float(truth))
            for pred, truth in zip(fit["test_preds"], fit["test_truths"])
        )
        / max(1, len(list(fit["test_truths"])))
    )
    assert fit["test_metrics"].root_mae == pytest.approx(manual_mae)

    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("ridge_control",),
        emit_confusion=True,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 1,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    run = payload["runs"][0]
    assert run["test_root_mae"] == pytest.approx(manual_mae)
    assert run["test_metrics"]["root_mae"] == pytest.approx(manual_mae)


def test_load_output_dir_backfills_legacy_split_metrics_for_reporting(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    legacy_run = {
        "benchmark": "recoverable_v4",
        "cell_id": "recoverable_v4",
        "baseline_family": "tree_neural",
        "seed": 0,
        "train_doc_count": 10240,
        "n_regimes": 4,
        "segment_density_band": "",
        "segment_min": 0,
        "segment_max": 0,
        "test_metrics": {"root_mae": 0.11},
        "train_metrics": {"root_mae": 0.44},
        "val_metrics": {"root_mae": 0.22},
        "fit_diagnostics": {
            "train_exact_match_rate": 0.5,
            "val_exact_match_rate": 0.6,
            "test_exact_match_rate": 0.7,
        },
        "parameterization": "formal_local_law_weight",
        "optimization_root_weight": 0.7,
        "local_law_c1_weight": 0.1,
        "local_law_c2_weight": 0.1,
        "local_law_c3_weight": 0.1,
        "task_objective_weight_source": "configured_objective_builder",
        "objective_weights_active": True,
        "c2_metric_kind": "score_drift",
        "semantics_version": "tree_neural_objective_v2",
    }
    (runs_dir / "legacy.json").write_text(
        json.dumps(legacy_run, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(tmp_path)
    aggregate_row = payload["aggregate_rows"][0]
    assert aggregate_row["train_root_mae_mean"] == pytest.approx(0.44)
    assert aggregate_row["val_root_mae_mean"] == pytest.approx(0.22)


def test_markdown_report_surfaces_primary_and_diagnostic_sections() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "train_root_mae": 0.30,
                "val_root_mae": 0.20,
                "test_root_mae": 0.10,
                "train_exact_match_rate": 0.70,
                "val_exact_match_rate": 0.80,
                "test_exact_match_rate": 0.90,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "task_objective_weight_source": "configured_objective_builder",
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            }
        ]
    )
    markdown = render_full_doc_anchor_diagnostic_markdown(payload)
    assert "primary paper score" in markdown
    assert "dev/model-selection metric" in markdown
    assert "### Split Diagnostics" in markdown
    assert "### Diagnostic-Only Metric Roles" in markdown
    assert "Ranking below is by mean **test** root-count MAE" in markdown


def test_run_markov_full_doc_anchor_diagnostics_aggregate_multi_seed_multi_train(
    tmp_path: Path,
) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0, 1),
        train_doc_counts=(4, 8),
        baseline_families=("ridge_control",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 1,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    assert len(payload["runs"]) == 4
    assert len(payload["aggregate_rows"]) == 2
    train_doc_counts = sorted(
        int(row["train_doc_count"]) for row in payload["aggregate_rows"]
    )
    assert train_doc_counts == [4, 8]
    assert payload["diagnostic_readout"]["status"] == "insufficient_data"
    assert payload["runs"][0]["baseline_family"] == "raw_token_ngram_ridge"


def test_structural_core_grid_resolves_expected_cells() -> None:
    cells = resolve_full_doc_diagnostic_grid("structural_core_v1")
    assert len(cells) == 9
    assert {cell.cell_id for cell in cells} == {
        "r4_seg4to6",
        "r4_seg7to9",
        "r4_seg10to12",
        "r8_seg4to6",
        "r8_seg7to9",
        "r8_seg10to12",
        "r12_seg4to6",
        "r12_seg7to9",
        "r12_seg10to12",
    }


def test_sticky_structural_core_grid_resolves_explicit_regime_switch_cells() -> None:
    cells = resolve_full_doc_diagnostic_grid("structural_core_v2_t128")
    assert len(cells) == 4
    assert {cell.cell_id for cell in cells} == {
        "r4_p031",
        "r12_p031",
        "r4_p079",
        "r12_p079",
    }
    cell = next(cell for cell in cells if cell.cell_id == "r12_p079")
    assert cell.regime_count == 12
    assert cell.segment_density_band == "higher_switch"
    assert float(cell.hazard_switch_prob) > 0.07


def test_sticky_recoverable_broadens_root_support() -> None:
    benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v5_t128")
    cfg = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=1024,
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides=None,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    support = sorted(
        {
            int(full_doc_diag._oracle_count(doc, start=0, end=len(doc.tokens)))
            for doc in bundle.test_docs
        }
    )
    assert len(support) > 3
    assert set(support) != {3, 4, 5}


def test_structural_benchmark_aliases_resolve_same_cell() -> None:
    alias = resolve_full_doc_diagnostic_benchmark(
        "recoverable_structural_core_v1__r12_seg10to12"
    )
    canonical = resolve_full_doc_diagnostic_benchmark("structural_core_v1::r12_seg10to12")
    assert alias.cell_id == "r12_seg10to12"
    assert canonical.cell_id == "r12_seg10to12"
    assert alias.grid_name == "structural_core_v1"
    assert canonical.grid_name == "structural_core_v1"
    assert canonical.name == alias.name


def test_sticky_structural_aliases_resolve_same_explicit_cell() -> None:
    alias = resolve_full_doc_diagnostic_benchmark("structural_core_v2_t128::r12_seg10to12")
    canonical = resolve_full_doc_diagnostic_benchmark("structural_core_v2_t128::r12_p079")
    assert alias.cell_id == "r12_p079"
    assert canonical.cell_id == "r12_p079"
    assert alias.grid_name == "structural_core_v2_t128"
    assert canonical.grid_name == "structural_core_v2_t128"
    assert canonical.name == alias.name


def test_structural_generator_respects_distinct_regime_band() -> None:
    cfg = OPSCountConfig(
        n_regimes=6,
        vocab_size=24,
        generator_profile="piecewise_disjoint_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=6,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        min_distinct_regimes_per_doc=4,
        max_distinct_regimes_per_doc=4,
        train_docs=12,
        val_docs=4,
        test_docs=4,
        use_cuda=False,
        seed=0,
        data_seed=0,
        model_seed=0,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    for split_docs in (bundle.train_docs, bundle.val_docs, bundle.test_docs):
        distinct_values = {len(set(int(x) for x in doc.token_regimes)) for doc in split_docs}
        assert distinct_values == {4}


def test_palette_block_exact_is_exact_and_uses_tokens_not_latent_regimes() -> None:
    cfg = OPSCountConfig(
        n_regimes=8,
        vocab_size=32,
        generator_profile="piecewise_disjoint_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=7,
        max_segments=9,
        min_seg_len=8,
        max_seg_len=24,
        min_distinct_regimes_per_doc=7,
        max_distinct_regimes_per_doc=8,
        train_docs=8,
        val_docs=2,
        test_docs=2,
        use_cuda=False,
        seed=3,
        data_seed=3,
        model_seed=3,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    docs = tuple(bundle.train_docs[:4])
    preds = _palette_block_exact_predictions(
        docs,
        vocab_size=int(cfg.vocab_size),
        n_regimes=int(cfg.n_regimes),
    )
    truths = [float(len(doc.true_boundaries)) for doc in docs]
    assert preds.tolist() == truths

    corrupted_docs = tuple(
        replace(
            doc,
            token_regimes=tuple(0 for _ in doc.token_regimes),
            transition_regimes=tuple(0 for _ in doc.transition_regimes),
        )
        for doc in docs
    )
    corrupted_preds = _palette_block_exact_predictions(
        corrupted_docs,
        vocab_size=int(cfg.vocab_size),
        n_regimes=int(cfg.n_regimes),
    )
    assert corrupted_preds.tolist() == truths


def test_palette_block_exact_rejects_non_disjoint_generator(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="piecewise_disjoint_palette",
    ):
        run_markov_full_doc_anchor_diagnostics(
            benchmark_name="smoke",
            seeds=(0,),
            train_doc_counts=(8,),
            baseline_families=("palette_block_exact",),
            emit_confusion=False,
            output_dir=tmp_path,
            use_cuda=False,
            torch_threads=1,
        )


def test_run_markov_full_doc_anchor_diagnostics_structural_grid_smoke(
    tmp_path: Path,
) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        hardness_grid="structural_core_v1",
        grid_cell_ids=("r4_seg4to6", "r12_seg10to12"),
        seeds=(0, 1),
        train_doc_counts=(16,),
        baseline_families=("palette_block_exact",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={"n_epochs": 1},
    )
    assert payload["hardness_grid"] == "structural_core_v1"
    assert len(payload["grid_cells"]) == 2
    assert len(payload["runs"]) == 4
    assert len(payload["aggregate_rows"]) == 2
    assert len(payload["heatmap_rows"]) == 2
    assert payload["grid_diagnostic_summary"]["status"] == "ok"
    assert payload["grid_diagnostic_summary"]["control_exactness"]["palette_block_exact"][
        "remains_exact_like"
    ]
    cell_ids = {row["cell_id"] for row in payload["aggregate_rows"]}
    assert cell_ids == {"r4_seg4to6", "r12_seg10to12"}
    # Fixed per-cell eval splits should be reused across seeds.
    signatures = {}
    for run in payload["runs"]:
        signatures.setdefault(run["cell_id"], set()).add(
            (run["val_corpus_signature"], run["test_corpus_signature"])
        )
        distinct_support = run["distinct_regime_support"]["train"]["values"]
        assert all(2 <= int(value) <= int(run["n_regimes"]) for value in distinct_support)
    assert all(len(items) == 1 for items in signatures.values())
    assert (tmp_path / "heatmap.csv").exists()


def test_markdown_report_mode_structural_grid(tmp_path: Path) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        hardness_grid="structural_core_v1",
        grid_cell_ids=("r4_seg4to6", "r12_seg10to12"),
        seeds=(0,),
        train_doc_counts=(8, 16),
        baseline_families=("palette_block_exact",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={"n_epochs": 1},
    )
    markdown = render_full_doc_anchor_diagnostic_markdown(payload)
    assert "# Markov Structural Grid Report" in markdown
    assert "## Headline Findings" in markdown
    assert "## Figure-First Interpretation" in markdown
    assert "## What The Reader Should Conclude" in markdown
    assert "palette_block_exact" in markdown


def test_markdown_report_mode_structural_stability(tmp_path: Path) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        hardness_grid="structural_core_v1",
        grid_cell_ids=("r4_seg4to6", "r12_seg10to12"),
        seeds=(0, 1),
        train_doc_counts=(16,),
        baseline_families=("palette_block_exact",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={"n_epochs": 1},
    )
    markdown = render_full_doc_anchor_diagnostic_markdown(payload)
    assert "# Markov Structural Stability Report" in markdown
    assert "## Checks" in markdown
    assert "## What The Reader Should Conclude" in markdown
    assert "systematic bias" in markdown or "instability" in markdown


def test_cli_structural_grid_smoke(tmp_path: Path) -> None:
    script = Path("/home/mlinegar/ThinkingTrees/scripts/run_markov_full_doc_anchor_ladder.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "diagnostic",
            "--hardness-grid",
            "structural_core_v1",
            "--grid-cell-ids",
            "r4_seg4to6",
            "--seeds",
            "0",
            "--train-doc-counts",
            "16",
            "--baseline-families",
            "palette_block_exact",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ],
        cwd="/home/mlinegar/ThinkingTrees",
        check=True,
        capture_output=True,
        text=True,
    )
    assert "heatmap_csv" in proc.stdout
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "aggregate.csv").exists()


@pytest.mark.parametrize(
    ("mode_name", "payload_builder"),
    [
        (
            "recoverable",
            lambda tmp_path: run_markov_full_doc_anchor_diagnostics(
                benchmark_name="smoke",
                seeds=(0, 1),
                train_doc_counts=(4, 8),
                baseline_families=("ridge_control",),
                emit_confusion=False,
                output_dir=tmp_path,
                use_cuda=False,
                torch_threads=1,
                config_overrides={
                    "n_epochs": 1,
                    "state_dim": 8,
                    "hidden_dim": 16,
                    "batch_size": 4,
                    "lr": 1e-3,
                },
            ),
        ),
        (
            "structural_grid",
            lambda tmp_path: run_markov_full_doc_anchor_diagnostics(
                hardness_grid="structural_core_v1",
                grid_cell_ids=("r4_seg4to6", "r12_seg10to12"),
                seeds=(0,),
                train_doc_counts=(8, 16),
                baseline_families=("palette_block_exact",),
                emit_confusion=False,
                output_dir=tmp_path,
                use_cuda=False,
                torch_threads=1,
                config_overrides={"n_epochs": 1},
            ),
        ),
        (
            "structural_stability",
            lambda tmp_path: run_markov_full_doc_anchor_diagnostics(
                hardness_grid="structural_core_v1",
                grid_cell_ids=("r4_seg4to6", "r12_seg10to12"),
                seeds=(0, 1),
                train_doc_counts=(16,),
                baseline_families=("palette_block_exact",),
                emit_confusion=False,
                output_dir=tmp_path,
                use_cuda=False,
                torch_threads=1,
                config_overrides={"n_epochs": 1},
            ),
        ),
    ],
)
def test_pdf_report_smoke_by_mode(
    tmp_path: Path,
    mode_name: str,
    payload_builder,
) -> None:
    script = Path("/home/mlinegar/ThinkingTrees/scripts/report_full_doc_anchor_diagnostics_pdf.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd="/home/mlinegar/ThinkingTrees",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "archived" in result.stderr.lower()


def test_structural_grid_default_baseline_family_expansion() -> None:
    assert default_baseline_families_for_mode(
        hardness_grid="structural_core_v1"
    ) == DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES


def test_tree_stage1_expected_layout_metadata_opaque_carrier_exact_sketch() -> None:
    config = OPSCountConfig(
        train_docs=8,
        val_docs=2,
        test_docs=2,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        n_epochs=1,
        state_dim=128,
        hidden_dim=64,
        batch_size=4,
        lr=1e-3,
        use_cuda=False,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        tree_task_head_mode="theorem_feature_scalar",
        tree_summary_spec_root_mode="factored_theorem_readout",
        tree_theorem_surface_mode="opaque_carrier_exact_sketch",
        tree_theorem_feature_dim=128,
        tree_theorem_feature_hidden_dim=256,
        tree_merge_hidden_dim=256,
    )

    metadata = full_doc_diag._tree_stage1_expected_layout_metadata(config)

    assert metadata["state_dim"] == 128
    assert metadata["carrier_state_dim"] == 128
    assert metadata["merge_hidden_dim"] == 256
    assert metadata["count_theorem_dim"] == 1
    assert metadata["first_theorem_dim"] == 4
    assert metadata["last_theorem_dim"] == 4
    assert metadata["residual_dim"] == 128
    assert metadata["summary_state_merger_in_features"] == 0
    assert metadata["carrier_state_merger_in_features"] == 256


def test_summarize_tree_exact_split_emits_exact_projected_metrics(
    monkeypatch,
) -> None:
    config = OPSCountConfig(
        train_docs=2,
        val_docs=2,
        test_docs=2,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        use_cuda=False,
        seed=0,
        data_seed=0,
        model_seed=0,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    docs = tuple(bundle.train_docs[:2])
    leaf_states: list[np.ndarray] = []
    leaf_counts: list[int] = []
    leaf_firsts: list[int] = []
    leaf_lasts: list[int] = []
    merge_counts: list[int] = []
    merge_firsts: list[int] = []
    merge_lasts: list[int] = []
    root_counts: list[int] = []
    root_firsts: list[int] = []
    root_lasts: list[int] = []
    for doc in docs:
        tree = full_doc_diag._balanced_exact_state_tree(
            doc,
            leaf_tokens=int(config.fixed_leaf_tokens),
        )
        for state in tree["leaf"]:
            leaf_states.append(np.zeros((4,), dtype=np.float64))
            leaf_counts.append(int(state.count))
            leaf_firsts.append(int(state.first))
            leaf_lasts.append(int(state.last))
        for state in tree["merge"]:
            merge_counts.append(int(state.count))
            merge_firsts.append(int(state.first))
            merge_lasts.append(int(state.last))
        for state in tree["root"]:
            root_counts.append(int(state.count))
            root_firsts.append(int(state.first))
            root_lasts.append(int(state.last))

    split_records_payload = {
        "leaf": {
            "state_features": np.stack(leaf_states).astype(np.float64),
            "count_targets": np.asarray(leaf_counts, dtype=np.int64),
            "first_targets": np.asarray(leaf_firsts, dtype=np.int64),
            "last_targets": np.asarray(leaf_lasts, dtype=np.int64),
            "direct_count_preds": np.asarray(leaf_counts, dtype=np.float64),
            "direct_first_preds": np.asarray(leaf_firsts, dtype=np.int64),
            "direct_last_preds": np.asarray(leaf_lasts, dtype=np.int64),
            "direct_count_entropy": np.full((len(leaf_counts),), np.nan, dtype=np.float64),
            "direct_count_margin": np.full((len(leaf_counts),), np.nan, dtype=np.float64),
            "direct_first_entropy": np.zeros((len(leaf_counts),), dtype=np.float64),
            "direct_first_margin": np.ones((len(leaf_counts),), dtype=np.float64),
            "direct_last_entropy": np.zeros((len(leaf_counts),), dtype=np.float64),
            "direct_last_margin": np.ones((len(leaf_counts),), dtype=np.float64),
            "is_first_leaf": np.asarray(
                [idx % 4 == 0 for idx in range(len(leaf_counts))],
                dtype=bool,
            ),
            "is_last_leaf": np.asarray(
                [idx % 4 == 3 for idx in range(len(leaf_counts))],
                dtype=bool,
            ),
            "c2_on_range_exact_match": np.ones((len(leaf_counts),), dtype=np.float64),
            "phi_label_moments": {},
        },
        "merge": {
            "state_features": np.zeros((len(merge_counts), 4), dtype=np.float64),
            "count_targets": np.asarray(merge_counts, dtype=np.int64),
            "first_targets": np.asarray(merge_firsts, dtype=np.int64),
            "last_targets": np.asarray(merge_lasts, dtype=np.int64),
            "direct_count_preds": np.asarray(merge_counts, dtype=np.float64),
            "direct_first_preds": np.asarray(merge_firsts, dtype=np.int64),
            "direct_last_preds": np.asarray(merge_lasts, dtype=np.int64),
            "direct_count_entropy": np.full((len(merge_counts),), np.nan, dtype=np.float64),
            "direct_count_margin": np.full((len(merge_counts),), np.nan, dtype=np.float64),
            "direct_first_entropy": np.zeros((len(merge_counts),), dtype=np.float64),
            "direct_first_margin": np.ones((len(merge_counts),), dtype=np.float64),
            "direct_last_entropy": np.zeros((len(merge_counts),), dtype=np.float64),
            "direct_last_margin": np.ones((len(merge_counts),), dtype=np.float64),
            "merge_join_bit_correct": np.ones((len(merge_counts),), dtype=np.float64),
            "merge_consistency_count_abs": np.zeros((len(merge_counts),), dtype=np.float64),
            "merge_consistency_first_correct": np.ones((len(merge_counts),), dtype=np.float64),
            "merge_consistency_last_correct": np.ones((len(merge_counts),), dtype=np.float64),
            "phi_merge_alignment": np.full((len(merge_counts),), np.nan, dtype=np.float64),
            "c2_on_range_exact_match": np.ones((len(merge_counts),), dtype=np.float64),
            "phi_label_moments": {},
        },
        "root": {
            "state_features": np.zeros((len(root_counts), 4), dtype=np.float64),
            "count_targets": np.asarray(root_counts, dtype=np.int64),
            "task_targets": np.asarray(root_counts, dtype=np.float64),
            "first_targets": np.asarray(root_firsts, dtype=np.int64),
            "last_targets": np.asarray(root_lasts, dtype=np.int64),
            "direct_count_preds": np.asarray(root_counts, dtype=np.float64),
            "task_count_preds": np.asarray(root_counts, dtype=np.float64),
            "direct_first_preds": np.asarray(root_firsts, dtype=np.int64),
            "direct_last_preds": np.asarray(root_lasts, dtype=np.int64),
            "direct_count_entropy": np.full((len(root_counts),), np.nan, dtype=np.float64),
            "direct_count_margin": np.full((len(root_counts),), np.nan, dtype=np.float64),
            "direct_first_entropy": np.zeros((len(root_counts),), dtype=np.float64),
            "direct_first_margin": np.ones((len(root_counts),), dtype=np.float64),
            "direct_last_entropy": np.zeros((len(root_counts),), dtype=np.float64),
            "direct_last_margin": np.ones((len(root_counts),), dtype=np.float64),
            "c2_on_range_exact_match": np.ones((len(root_counts),), dtype=np.float64),
            "phi_label_moments": {},
        },
    }
    probe_models = {
        "leaf": {"count": None, "first": None, "last": None},
        "merge": {"count": None, "first": None, "last": None},
        "root": {"count": None, "first": None, "last": None},
    }

    class _ModelStub:
        use_markov_summary_spec = True

        @staticmethod
        def parameters():
            return iter(())

        @staticmethod
        def theorem_count_support_size() -> int:
            return 6

    monkeypatch.setattr(
        full_doc_diag,
        "_exact_projected_root_count_from_states",
        lambda model, states, schedule="balanced": float(len(states)),
    )

    _exact_split, _tree_split, direct_selection = full_doc_diag._summarize_tree_exact_split(
        split="test",
        docs=docs,
        split_records_payload=split_records_payload,
        probe_models=probe_models,
        model=_ModelStub(),
        config=config,
    )

    assert np.isfinite(float(direct_selection["exact_projected_root_mae"]))
    assert float(direct_selection["certified_projected_root_mae"]) == pytest.approx(
        float(direct_selection["exact_projected_root_mae"])
    )
    assert np.isfinite(
        float(direct_selection["root_mae_predicted_counts_predicted_endpoints"])
    )
    assert np.isfinite(
        float(direct_selection["root_mae_oracle_counts_predicted_endpoints"])
    )
    assert np.isfinite(
        float(direct_selection["root_mae_predicted_counts_oracle_endpoints"])
    )
    assert np.isfinite(float(direct_selection["learned_merger_gap"]))
    assert np.isfinite(float(direct_selection["leaf_first_accuracy"]))
    assert np.isfinite(float(direct_selection["leaf_last_accuracy"]))
    assert np.isfinite(float(direct_selection["merge_first_accuracy"]))
    assert np.isfinite(float(direct_selection["merge_last_accuracy"]))
    assert isinstance(direct_selection["leaf_count_off_by_k_histogram"], dict)
    assert isinstance(direct_selection["merge_exact_summary_match_rate_by_depth"], dict)


def test_tree_exact_sketch_diagnostics_markov_sufficiency_gap_is_finite_without_entropy(
    monkeypatch,
) -> None:
    config = OPSCountConfig(
        train_docs=1,
        val_docs=1,
        test_docs=1,
        fixed_leaf_tokens=16,
        n_regimes=4,
        use_cuda=False,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    docs = tuple(bundle.train_docs[:1])
    fno_docs = tuple(
        full_doc_diag._prepare_fno_count_docs(
            docs,
            leaf_tokens=int(config.fixed_leaf_tokens),
        )
    )

    dummy_records = {
        "leaf": {},
        "merge": {},
        "root": {},
    }

    def _fake_collect(**kwargs):
        return dummy_records

    def _fake_finalize(records):
        return dict(records)

    def _fake_probe_models(**kwargs):
        return {"leaf": {}, "merge": {}, "root": {}}

    def _fake_summarize(**kwargs):
        direct = {
            "root_direct_count_mae": 0.5,
            "root_mae_predicted_counts_predicted_endpoints": 0.3,
            "root_mae_oracle_counts_predicted_endpoints": 0.1,
            "root_mae_predicted_counts_oracle_endpoints": 0.25,
            "leaf_direct_exact_match": 0.6,
            "leaf_probe_exact_match": 0.8,
            "merge_direct_exact_match": 0.4,
            "merge_probe_exact_match": 0.7,
            "merge_join_bit_accuracy": 0.9,
            "c2_on_range_exact_match": 1.0,
            "leaf_first_accuracy": 0.95,
            "leaf_last_accuracy": 0.95,
            "merge_first_accuracy": 0.9,
            "merge_last_accuracy": 0.85,
            "leaf_count_off_by_k_histogram": {"0": 0.5, "1": 0.5},
            "merge_exact_summary_match_rate_by_depth": {"0": 0.4, "1": 0.6},
            "first_leaf_direct_accuracy": 0.95,
            "last_leaf_direct_accuracy": 0.95,
            "leaf_count_head_entropy_mean": float("nan"),
            "merge_count_head_entropy_mean": float("nan"),
            "phi_merge_alignment": float("nan"),
            "phi_within_class_variance": float("nan"),
            "phi_between_class_margin": float("nan"),
            "exact_projected_root_mae": 0.3,
            "certified_projected_root_mae": 0.3,
            "learned_merger_gap": 0.2,
        }
        tree = {
            "leaf": {
                "direct": {
                    "count_mae": 0.2,
                    "exact_summary_match_rate": 0.6,
                    "first_accuracy": 0.95,
                    "last_accuracy": 0.95,
                },
                "probe": {"exact_summary_match_rate": 0.8},
            },
            "merge": {
                "direct": {
                    "count_mae": 0.3,
                    "exact_summary_match_rate": 0.4,
                    "first_accuracy": 0.95,
                    "last_accuracy": 0.95,
                },
                "probe": {"exact_summary_match_rate": 0.7},
                "decoded_consistency": {
                    "merge_join_bit_accuracy": 0.9,
                },
            },
            "root": {
                "direct": {"count_mae": 0.5},
                "probe": {"count_mae": 0.1},
            },
        }
        return {}, tree, direct

    monkeypatch.setattr(full_doc_diag, "_collect_tree_exact_state_records", _fake_collect)
    monkeypatch.setattr(full_doc_diag, "_finalize_tree_exact_state_records", _fake_finalize)
    monkeypatch.setattr(full_doc_diag, "_fit_tree_exact_probe_models", _fake_probe_models)
    monkeypatch.setattr(full_doc_diag, "_summarize_tree_exact_split", _fake_summarize)

    payload = full_doc_diag._tree_exact_sketch_diagnostics(
        model=object(),
        config=config,
        device=torch.device("cpu"),
        train_docs=docs,
        val_docs=docs,
        test_docs=docs,
        train_fno_docs=fno_docs,
        val_fno_docs=fno_docs,
        test_fno_docs=fno_docs,
    )

    failure = dict(payload["failure_attribution"])
    assert float(failure["theorem_count_decode_gap_score"]) == pytest.approx(1.0)
    assert float(failure["markov_sufficiency_gap_score"]) == pytest.approx(1.0)


def test_cli_can_aggregate_recursive_shard_outputs(tmp_path: Path) -> None:
    script = Path("/home/mlinegar/ThinkingTrees/scripts/run_markov_full_doc_anchor_ladder.py")
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    common_prefix = [
        sys.executable,
        str(script),
        "--mode",
        "diagnostic",
        "--benchmark",
        "smoke",
        "--train-doc-counts",
        "8",
        "--baseline-families",
        "ridge_control",
        "--device",
        "cpu",
    ]
    subprocess.run(
        common_prefix + ["--seeds", "0", "--output-dir", str(shard_a)],
        cwd="/home/mlinegar/ThinkingTrees",
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        common_prefix + ["--seeds", "1", "--output-dir", str(shard_b)],
        cwd="/home/mlinegar/ThinkingTrees",
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "diagnostic",
            "--aggregate-only",
            "--output-dir",
            str(tmp_path),
        ],
        cwd="/home/mlinegar/ThinkingTrees",
        check=True,
        capture_output=True,
        text=True,
    )
    assert "aggregate_rows" in proc.stdout
    payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(tmp_path)
    assert len(payload["runs"]) == 2
    assert payload["baseline_families"] == ["raw_token_ngram_ridge"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "aggregate.csv").exists()


def test_payload_from_saved_runs_marks_legacy_tree_neural_semantics() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 8,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.3,
                "test_exact_match_rate": 0.5,
                "test_c2_idempotence_mae": 0.0,
            }
        ]
    )
    run = payload["runs"][0]
    assert run["comparison_semantics"] == "legacy_quarantined"
    assert run["legacy_semantics"] is True
    assert "missing_run_intent_metadata" in run["legacy_semantics_reason"]
    aggregate_row = payload["aggregate_rows"][0]
    assert aggregate_row["comparison_semantics"] == "legacy_quarantined"
    assert aggregate_row["legacy_semantics"] is True


def test_payload_from_saved_runs_backfills_tree_neural_objective_weights_active() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2",
                "seed": 0,
                "train_doc_count": 8,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "test_root_mae": 0.2,
                "test_exact_match_rate": 0.8,
                "test_c2_idempotence_mae": 0.01,
                "parameterization": "formal_local_law_weight",
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.3,
                "local_law_c3_weight": 0.0,
                "c2_metric_kind": "score_drift",
                "semantics_version": "tree_neural_objective_v2",
            }
        ]
    )
    run = payload["runs"][0]
    assert run["objective_weights_active"] is True
    aggregate_row = payload["aggregate_rows"][0]
    assert aggregate_row["objective_weights_active"] is True


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_recoverable_tree_neural_validation_summary_artifacts(tmp_path: Path) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="recoverable_v4",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("tree_neural_c2", "tree_neural"),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 1,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    assert len(payload["runs"]) == 2
    signatures = {
        (run["val_corpus_signature"], run["test_corpus_signature"])
        for run in payload["runs"]
    }
    assert len(signatures) == 1
    for run in payload["runs"]:
        assert run["parameterization"] == "formal_local_law_weight"
        assert run["c2_metric_kind"] == "count_drift"
        assert "test_c2_state_replay_mse" in run
    validation = dict(payload.get("tree_neural_validation_summary") or {})
    assert validation["benchmark"] == "recoverable_v4"
    assert validation["c2_metric_kind"] == "count_drift"
    assert len(list(validation.get("comparisons") or [])) == 1
    assert (tmp_path / "tree_neural_validation_summary.json").exists()
    assert (tmp_path / "tree_neural_validation_summary.md").exists()


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_official_fno_sumlen_is_separate_baseline_label(
    tmp_path: Path,
) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="smoke",
        seeds=(0,),
        train_doc_counts=(8,),
        baseline_families=("official_fno_sumlen",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "n_epochs": 1,
            "state_dim": 8,
            "hidden_dim": 16,
            "batch_size": 4,
            "lr": 1e-3,
        },
    )
    run = payload["runs"][0]
    assert run["baseline_family"] == "official_fno_sumlen"
    assert (
        run["config"]["fixed_leaf_tokens"]
        == full_doc_diag.FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
    )
    assert run["config"].get("preserve_requested_leaf_tokens", False) is False
    assert (
        run["config"].get("official_fno_preserve_requested_leaf_tokens", False)
        is False
    )
    assert run["config"]["doc_sequence_fno_pooling"] == "sum"
    assert run["config"]["doc_sequence_fno_concat_length_feature"] is True
    assert run["config"]["doc_sequence_fno_include_transition_channel"] is False
    assert run["backend_package"] == "neuraloperator"
    assert run["operator_class"] == "neuralop.models.FNO"
    assert run["operator_evidence_status"] == "PROXY_ONLY"


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_run_markov_full_doc_anchor_diagnostics_demo_v1_zero_sanity(
    tmp_path: Path,
) -> None:
    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name="demo_v1",
        seeds=(0,),
        train_doc_counts=(256,),
        baseline_families=("official_fno",),
        emit_confusion=False,
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        config_overrides={
            "state_dim": 16,
            "hidden_dim": 64,
            "n_epochs": 6,
            "batch_size": 8,
            "lr": 5e-4,
        },
    )
    assert payload["degenerate_benchmark"] is True
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["baseline_family"] == "official_fno"
    assert float(run["test_root_mae"]) == 0.0
    assert float(run["test_exact_match_rate"]) == 1.0
