from __future__ import annotations

"""Reusable tree-neural config recipes for full-doc Markov runners."""

import argparse
from dataclasses import replace
from typing import Any, Callable

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    FAIR_FNO_PARITY_CONFIG_LABEL,
    _base_config_for_benchmark,
    resolve_full_doc_diagnostic_benchmark,
)
from src.ctreepo.sim.core.tree_neural_facade import RunConfigSpec


def default_tree_batch_pack_mode(benchmark: str) -> str:
    return "fixed_fused" if str(benchmark).strip().lower() == "recoverable_v4" else "structure_bucket"


def resolved_tree_batch_pack_mode(*, benchmark: str, raw_value: str | None) -> str:
    raw = str(raw_value or "").strip()
    if raw:
        return raw
    return default_tree_batch_pack_mode(benchmark)


def resolve_benchmark_leaf_tokens(
    *,
    benchmark_name: str,
    train_doc_count: int,
    state_dim: int,
    hidden_dim: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> int:
    benchmark = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=int(train_doc_count),
        use_cuda=False,
        cuda_device=None,
        torch_threads=1,
        seed=0,
        config_overrides={
            "state_dim": int(state_dim),
            "hidden_dim": int(hidden_dim),
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
    )
    return int(config.fixed_leaf_tokens)


def fair_fno_tree_config_for_train_doc_count(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    label: str = FAIR_FNO_PARITY_CONFIG_LABEL,
    leaf_token_resolver: Callable[..., int] | None = None,
) -> RunConfigSpec:
    preload_splits = tuple(
        str(item)
        for item in list(getattr(args, "gpu_runtime_preload_splits", ("train", "val", "test")))
        if str(item).strip()
    )
    leaf_resolver = resolve_benchmark_leaf_tokens if leaf_token_resolver is None else leaf_token_resolver
    fixed_leaf_tokens = leaf_resolver(
        benchmark_name=str(args.benchmark),
        train_doc_count=int(train_doc_count),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    return RunConfigSpec(
        label=str(label),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        fixed_leaf_tokens=None,
        tree_local_law_weight=(
            None
            if args.tree_local_law_weight is None
            else float(args.tree_local_law_weight)
        ),
        tree_task_objective_weight=(
            None
            if args.tree_task_objective_weight is None
            else float(args.tree_task_objective_weight)
        ),
        tree_leaf_fno_width=max(64, int(args.state_dim)),
        tree_leaf_fno_n_modes=min(16, max(1, fixed_leaf_tokens // 2)),
        tree_leaf_fno_n_layers=4,
        tree_root_supervision_kind="count_ce",
        gpu_runtime_data_mode=str(
            getattr(args, "gpu_runtime_data_mode", "resident")
        ),
        gpu_runtime_bucket_mode=str(
            getattr(args, "gpu_runtime_bucket_mode", "exact_then_bucketed")
        ),
        gpu_runtime_preload_splits=preload_splits or ("train", "val", "test"),
        gpu_runtime_preload_targets=bool(
            getattr(args, "gpu_runtime_preload_targets", True)
        ),
        gpu_runtime_workers_per_mig=int(
            getattr(args, "gpu_runtime_workers_per_mig", 1)
        ),
        gpu_runtime_allow_multi_worker_screen=bool(
            getattr(args, "gpu_runtime_allow_multi_worker_screen", True)
        ),
        gpu_runtime_capacity_workers_per_mig=int(
            getattr(args, "gpu_runtime_capacity_workers_per_mig", 2)
        ),
        doc_sequence_train_fraction=0.0,
    )


def slot_exact_sanity_config(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    config_label: str,
    leaf_label_rate: float,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
    internal_label_rate: float,
    endpoint_loss_scale: float = 1.0,
    leaf_exact_supervision: bool = False,
    tree_summary_spec_root_mode: str | None = None,
    fair_config_func: Callable[..., RunConfigSpec] | None = None,
    batch_pack_resolver: Callable[..., str] | None = None,
) -> RunConfigSpec:
    fair_builder = fair_fno_tree_config_for_train_doc_count if fair_config_func is None else fair_config_func
    pack_resolver = resolved_tree_batch_pack_mode if batch_pack_resolver is None else batch_pack_resolver
    fair_base = fair_builder(
        args,
        train_doc_count=int(train_doc_count),
        label=str(config_label),
    )
    return replace(
        fair_base,
        label=str(config_label),
        tree_root_supervision_kind="mse",
        tree_checkpoint_metric=str(
            getattr(args, "tree_checkpoint_metric", "val_root_mae")
        ),
        tree_stage1_checkpoint_metric=str(
            getattr(args, "tree_stage1_checkpoint_metric", "val_root_mae")
        ),
        tree_stage1_eval_mode=str(
            getattr(args, "tree_stage1_eval_mode", "per_epoch")
        ),
        tree_stage1_screen_doc_limit=int(
            getattr(args, "tree_stage1_screen_doc_limit", 0)
        ),
        tree_stage1_final_exact_doc_limit=int(
            getattr(args, "tree_stage1_final_exact_doc_limit", 0)
        ),
        exact_metric_selection_doc_limit=int(
            getattr(args, "exact_metric_selection_doc_limit", 0)
        ),
        exact_metric_selection_interval=int(
            getattr(args, "exact_metric_selection_interval", 1)
        ),
        tree_batch_pack_mode=str(
            pack_resolver(
                benchmark=str(getattr(args, "benchmark", "")),
                raw_value=getattr(args, "tree_batch_pack_mode", ""),
            )
        ),
        tree_batch_token_budget=int(
            getattr(args, "tree_batch_token_budget", 0)
        ),
        tree_batch_node_budget=int(
            getattr(args, "tree_batch_node_budget", 0)
        ),
        tree_batch_autotune=bool(
            getattr(args, "tree_batch_autotune", True)
        ),
        tree_batch_structural_pad_limit=float(
            getattr(args, "tree_batch_structural_pad_limit", 0.5)
        ),
        tree_batch_auto_queue_min_docs=int(
            getattr(args, "tree_batch_auto_queue_min_docs", 8)
        ),
        tree_batch_auto_queue_min_fill_ratio=float(
            getattr(args, "tree_batch_auto_queue_min_fill_ratio", 0.5)
        ),
        tree_eval_workers_per_mig=int(
            getattr(args, "tree_eval_workers_per_mig", 0)
        ),
        tree_stage1_artifact_dir=str(
            getattr(args, "tree_stage1_artifact_dir", "")
        ),
        tree_stage1_root_weight=float(
            getattr(args, "tree_stage1_root_weight", 0.0)
        ),
        tree_join_bit_weight=float(
            getattr(args, "tree_join_bit_weight", 0.0)
        ),
        tree_training_schedule=str(
            getattr(args, "tree_training_schedule", "two_stage")
        ),
        tree_stage1_epochs=int(getattr(args, "tree_stage1_epochs", 12)),
        tree_stage2_epochs=int(getattr(args, "tree_stage2_epochs", 20)),
        tree_task_head_mode=str(
            getattr(args, "tree_task_head_mode", "theorem_feature_scalar")
        ),
        tree_theorem_surface_mode=str(
            getattr(args, "tree_theorem_surface_mode", "shared_bottleneck")
        ),
        tree_theorem_count_head_mode=str(
            getattr(args, "tree_theorem_count_head_mode", "scalar_mse")
        ),
        tree_theorem_feature_dim=int(
            getattr(args, "tree_theorem_feature_dim", 48)
        ),
        tree_theorem_feature_hidden_dim=int(
            getattr(args, "tree_theorem_feature_hidden_dim", 256)
        ),
        tree_merge_hidden_dim=int(
            getattr(args, "tree_merge_hidden_dim", 0)
        ),
        tree_theorem_score_dim=int(
            getattr(args, "tree_theorem_score_dim", 0)
        ),
        tree_theorem_fiber_dim=int(
            getattr(args, "tree_theorem_fiber_dim", 0)
        ),
        tree_theorem_aux_dim=int(
            getattr(args, "tree_theorem_aux_dim", 0)
        ),
        tree_score_merge_mode=str(
            getattr(args, "tree_score_merge_mode", "gated_affine")
        ),
        tree_phi_compose_weight=float(
            getattr(args, "tree_phi_compose_weight", 1.0)
        ),
        tree_phi_contrastive_weight=float(
            getattr(args, "tree_phi_contrastive_weight", 0.25)
        ),
        tree_phi_alignment_loss=str(
            getattr(args, "tree_phi_alignment_loss", "cosine_mse")
        ),
        tree_theorem_count_ordinal_weight=float(
            getattr(args, "tree_theorem_count_ordinal_weight", 1.0)
        ),
        tree_theorem_count_scalar_aux_weight=float(
            getattr(args, "tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        tree_theorem_count_threshold_balance=bool(
            getattr(args, "tree_theorem_count_threshold_balance", True)
        ),
        tree_summary_spec_root_mode=str(
            tree_summary_spec_root_mode
            if tree_summary_spec_root_mode is not None
            else getattr(args, "tree_summary_spec_root_mode", "factored_theorem_readout")
        ),
        doc_sequence_train_fraction=0.0,
        aligned_sketch_surface="",
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        tree_theorem_count_dim=int(getattr(args, "tree_theorem_count_dim", 8)),
        tree_theorem_first_dim=int(getattr(args, "tree_theorem_first_dim", 8)),
        tree_theorem_last_dim=int(getattr(args, "tree_theorem_last_dim", 8)),
        leaf_supervision_kind=str(leaf_supervision_kind),
        internal_supervision_kind=str(internal_supervision_kind),
        internal_label_rate=float(internal_label_rate),
        leaf_exact_supervision=bool(leaf_exact_supervision),
        leaf_label_rate=float(leaf_label_rate),
        endpoint_loss_scale=float(endpoint_loss_scale),
    )


__all__ = [
    "default_tree_batch_pack_mode",
    "fair_fno_tree_config_for_train_doc_count",
    "resolve_benchmark_leaf_tokens",
    "resolved_tree_batch_pack_mode",
    "slot_exact_sanity_config",
]
