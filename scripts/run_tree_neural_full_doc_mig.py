#!/usr/bin/env python3
"""Launch full-doc tree-baseline diagnostics across MIG slices.

This keeps the benchmark lane canonical by running the existing
``run_markov_full_doc_anchor_diagnostics`` entrypoint in separate shard
directories, one shard per launch job, then re-aggregating the saved run JSONs
from the shared output root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import numpy as np
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple, cast
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.markov_gpu_scheduler import (  # noqa: E402
    SchedulerConfig,
    SchedulerItem,
    SchedulerRunError,
    run_scheduler,
    summarize_scheduler_plan,
)
from src.ctreepo.sim.core.tree_reference_presets import (  # noqa: E402
    ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET,
    ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET,
    ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET,
    STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    resolve_tree_reference_preset_config,
    tree_reference_preset_names,
)
TREE_NEURAL_FAMILIES = frozenset(
    {"tree_neural_c2", "tree_neural_c2c3", "tree_neural"}
)
CLOSED_FORM_CONTROL_FAMILIES = frozenset({"tree_ridge_leaf", "tree_doc_ridge"})


def _default_tree_batch_pack_mode(benchmark: str) -> str:
    return "fixed_fused" if str(benchmark).strip().lower() == "recoverable_v4" else "structure_bucket"


def _resolved_tree_batch_pack_mode(*, benchmark: str, raw_value: str | None) -> str:
    raw = str(raw_value or "").strip()
    if raw:
        return raw
    return _default_tree_batch_pack_mode(benchmark)

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    FAIR_FNO_PARITY_CONFIG_LABEL,
    ORACLE_BUDGET_STUDY_NAME,
    VALID_BASELINE_FAMILIES,
    _base_config_for_benchmark,
    estimate_tree_worker_runtime_preflight,
    load_markov_full_doc_anchor_diagnostics_from_output_dir,
    prepare_markov_full_doc_anchor_diagnostics_data,
    render_full_doc_anchor_diagnostic_markdown,
    resolve_full_doc_diagnostic_benchmark,
    resolve_full_doc_diagnostic_grid,
    run_markov_full_doc_anchor_diagnostics,
)
from src.ctreepo.sim.core.full_doc_config_codec import (  # noqa: E402
    runtime_config_overrides_from_config_like,
    write_tree_run_config_json,
)
from src.ctreepo.sim.core.run_config import (  # noqa: E402
    run_config_from_mapping as _shared_run_config_from_mapping,
)
from src.ctreepo.sim.core.run_intent import (  # noqa: E402
    VALID_TOPOLOGIES,
    resolve_package_semantics,
)
from src.ctreepo.sim.suite.markov_observed_token_policy import (  # noqa: E402
    resolve_markov_observed_token_policy,
)

PARITY_TREE_FAMILIES = ("tree_neural_c2", "tree_neural_c2c3", "tree_neural")
PARITY_FNO_FAMILIES = ("official_fno", "official_fno_sumlen")
PARITY_COMPARISON_FAMILIES = (*PARITY_FNO_FAMILIES, *PARITY_TREE_FAMILIES)
PARITY_GATE_TRAIN_DOC_COUNT = 10240
PARITY_SCALE_CURVE_TRAIN_DOC_COUNTS = (1024, 2048, 3072, 4096, 5120, 8192, 10240)
CAPACITY_PRIORITY_FAMILY = "tree_neural"
CAPACITY_WIDTH_AXIS = (64, 128, 256)
CAPACITY_MODES_AXIS = (2, 4, 8)
CAPACITY_LAYERS_AXIS = (2, 4, 6)
ROOT_ONLY_CAPACITY_PROFILE_DEFAULT = "default"
ROOT_ONLY_CAPACITY_PROFILE_HISTORICAL_REPLAY = "root_only_parity_historical_replay"
ROOT_ONLY_CAPACITY_PROFILE_OPTIMIZATION_FAIRNESS = "root_only_parity_optimization_fairness"
ROOT_ONLY_CAPACITY_PROFILE_CAPACITY_FAIRNESS = "root_only_parity_capacity_fairness"
ROOT_ONLY_CAPACITY_PROFILE_MATCHED_ROOT = "root_only_parity_matched_root"
ROOT_ONLY_CAPACITY_PROFILE_STRUCTURAL_MATCHED_ROOT = "root_only_parity_structural_matched_root"
CAPACITY_PROFILE_CHOICES = (
    ROOT_ONLY_CAPACITY_PROFILE_DEFAULT,
    ROOT_ONLY_CAPACITY_PROFILE_HISTORICAL_REPLAY,
    ROOT_ONLY_CAPACITY_PROFILE_OPTIMIZATION_FAIRNESS,
    ROOT_ONLY_CAPACITY_PROFILE_CAPACITY_FAIRNESS,
    ROOT_ONLY_CAPACITY_PROFILE_MATCHED_ROOT,
    ROOT_ONLY_CAPACITY_PROFILE_STRUCTURAL_MATCHED_ROOT,
)
CAPACITY_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    ROOT_ONLY_CAPACITY_PROFILE_DEFAULT: {},
    ROOT_ONLY_CAPACITY_PROFILE_HISTORICAL_REPLAY: {
        "base_config_preset": ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET,
        "capacity_widths": (128,),
        "capacity_modes": (8,),
        "capacity_layers": (4,),
        "capacity_state_dims": (128,),
        "capacity_hidden_dims": (512,),
        "capacity_n_epochs": (52,),
        "capacity_tree_training_schedules": ("two_stage",),
        "capacity_tree_checkpoint_metrics": ("val_exact_sketch_direct",),
        "capacity_tree_stage1_checkpoint_metrics": ("val_theorem_bootstrap_direct",),
        "capacity_tree_stage1_root_weights": (0.0,),
        "capacity_slot_counts": (4,),
        "capacity_fixed_leaf_tokens": (16,),
    },
    ROOT_ONLY_CAPACITY_PROFILE_OPTIMIZATION_FAIRNESS: {
        "base_config_preset": ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET,
        "capacity_widths": (128,),
        "capacity_modes": (8,),
        "capacity_layers": (4,),
        "capacity_state_dims": (128,),
        "capacity_hidden_dims": (512,),
        "capacity_n_epochs": (128,),
        "capacity_tree_training_schedules": ("single_stage",),
        "capacity_tree_checkpoint_metrics": ("val_root_mae",),
        "capacity_tree_stage1_checkpoint_metrics": ("val_root_mae",),
        "capacity_tree_stage1_root_weights": (1.0,),
        "capacity_slot_counts": (4,),
        "capacity_fixed_leaf_tokens": (16,),
    },
    ROOT_ONLY_CAPACITY_PROFILE_CAPACITY_FAIRNESS: {
        "base_config_preset": ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET,
        "capacity_widths": (128,),
        "capacity_modes": (8,),
        "capacity_layers": (4,),
        "capacity_state_dims": (256,),
        "capacity_hidden_dims": (1024,),
        "capacity_n_epochs": (52,),
        "capacity_tree_training_schedules": ("two_stage",),
        "capacity_tree_checkpoint_metrics": ("val_exact_sketch_direct",),
        "capacity_tree_stage1_checkpoint_metrics": ("val_theorem_bootstrap_direct",),
        "capacity_tree_stage1_root_weights": (0.0,),
        "capacity_slot_counts": (4,),
        "capacity_fixed_leaf_tokens": (16,),
    },
    ROOT_ONLY_CAPACITY_PROFILE_MATCHED_ROOT: {
        "base_config_preset": ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
        "capacity_state_dims": (256,),
        "capacity_hidden_dims": (1024,),
        "capacity_n_epochs": (128,),
        "capacity_tree_training_schedules": ("single_stage",),
        "capacity_tree_checkpoint_metrics": ("val_root_mae",),
        "capacity_tree_stage1_checkpoint_metrics": ("val_root_mae",),
        "capacity_tree_stage1_root_weights": (1.0,),
        "capacity_slot_counts": (4,),
        "capacity_fixed_leaf_tokens": (16,),
    },
    ROOT_ONLY_CAPACITY_PROFILE_STRUCTURAL_MATCHED_ROOT: {
        "base_config_preset": STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
        "capacity_widths": (128,),
        "capacity_modes": (8,),
        "capacity_layers": (4,),
        "capacity_state_dims": (256,),
        "capacity_hidden_dims": (1024,),
        "capacity_n_epochs": (128,),
        "capacity_tree_training_schedules": ("single_stage",),
        "capacity_tree_checkpoint_metrics": ("val_root_mae",),
        "capacity_tree_stage1_checkpoint_metrics": ("val_root_mae",),
        "capacity_tree_stage1_root_weights": (1.0,),
        "capacity_slot_counts": (4,),
        "capacity_fixed_leaf_tokens": (16,),
    },
}
BUDGET_FRONTIER_TREE_FAMILIES = ("tree_neural", "tree_neural_c2", "tree_neural_c2c3")
BUDGET_FRONTIER_REFERENCE_FAMILIES = (
    "official_fno",
    "official_fno_sumlen",
    "tree_doc_ridge",
)
BUDGET_FRONTIER_BUDGETS_PER_DOC = (0.25, 0.5, 1.0, 2.0, 4.0)
BUDGET_FRONTIER_FULL_DOC_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
BUDGET_FRONTIER_DOC_CONSUMPTION_MODES = ("root_only", "doc_sequence")
BUDGET_FRONTIER_LOCAL_SPLIT_MODES = ("balanced", "leaf_heavy", "internal_heavy")
BUDGET_FRONTIER_ALLOCATION_POLICY = "breadth_first"
EXACT_SANITY_STUDY_NAME = "tree_neural_exact_sanity"
EXACT_SANITY_FAMILY = "tree_neural"
EXACT_SANITY_LEVELS = ("leaf", "merge", "root")
EXACT_SANITY_COMPONENT_METRICS = (
    "count_mae",
    "count_match_rate",
    "first_accuracy",
    "last_accuracy",
    "exact_summary_match_rate",
)
EXACT_SANITY_MERGE_CONSISTENCY_METRICS = (
    "merge_join_bit_accuracy",
    "merge_decoded_consistency_count_mae",
    "merge_decoded_consistency_first_accuracy",
    "merge_decoded_consistency_last_accuracy",
)
EXACT_SANITY_LAW_METRICS = (
    "root_mae",
    "leaf_mae",
    "c2_idempotence_mae",
    "merge_mae",
)
REPRESENTATION_SUFFICIENCY_STUDY_NAME = "tree_neural_representation_sufficiency"
REPRESENTATION_SUFFICIENCY_FAMILY = "tree_neural"
REPRESENTATION_SUFFICIENCY_SELECTION_METRIC = "val_exact_sketch_direct"
REPRESENTATION_SUFFICIENCY_SCREEN_STAGE = "representation_screen"
REPRESENTATION_SUFFICIENCY_LOCK_STAGE = "representation_lock"
REPRESENTATION_SUFFICIENCY_PROMOTION_STAGE = "representation_promotion"
REPRESENTATION_SUFFICIENCY_DEFAULT_SCREEN_DOC_COUNT = 512
REPRESENTATION_SUFFICIENCY_DEFAULT_LOCK_DOC_COUNT = 1024
REPRESENTATION_SUFFICIENCY_DEFAULT_PROMOTION_DOC_COUNT = 4096
REPRESENTATION_SUFFICIENCY_DEFAULT_TOP_K = 4
REPRESENTATION_SUFFICIENCY_DEFAULT_COUNT_HEAD_MODES = (
    "scalar_mse",
    "support_classifier",
    "hybrid_ordinal",
)
REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE = 0.02
REPRESENTATION_SUFFICIENCY_CONTROL_MIN_EXACT_MATCH = 0.95
REPRESENTATION_SUFFICIENCY_CONTROL_MAX_SUFFICIENCY_GAP = 0.05
REPRESENTATION_LEARNABILITY_STUDY_NAME = "tree_neural_representation_learnability"
REPRESENTATION_LEARNABILITY_WINNER_STAGE = "representation_learnability_winner"
REPRESENTATION_LEARNABILITY_SWEEP_STAGE = "representation_learnability_sweep"
REPRESENTATION_LEARNABILITY_DEFAULT_WINNER_DOC_COUNT = 1024
REPRESENTATION_LEARNABILITY_DEFAULT_SWEEP_DOC_COUNTS = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)
REPRESENTATION_LEARNABILITY_DEFAULT_BENCHMARK_CELLS = (
    "recoverable_v4",
    "r4_seg4to6",
    "r8_seg7to9",
    "r12_seg10to12",
)


@dataclass(frozen=True)
class _RunConfigSpec:
    label: str
    state_dim: int
    hidden_dim: int
    n_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    baseline_family: str = ""
    topology: str = ""
    fixed_leaf_tokens: int | None = None
    tree_local_law_weight: float | None = None
    tree_task_objective_weight: float | None = None
    tree_local_weighting_mode: str = "fixed_k_hajek"
    tree_exact_collapse_mode: str = ""
    tree_c1_relative_weight: float = 1.0
    tree_c2_relative_weight: float = 1.0
    tree_c3_relative_weight: float = 1.0
    official_fno_preserve_requested_leaf_tokens: bool = False
    preserve_requested_leaf_tokens: bool = False
    comparison_mode: str = "legacy"
    tree_leaf_fno_width: int | None = None
    tree_leaf_fno_n_modes: int | None = None
    tree_leaf_fno_n_layers: int | None = None
    tree_model_version: str = ""
    tree_batch_runtime_mode: str = ""
    tree_root_supervision_kind: str = "mse"
    tree_document_loss_normalization_mode: str = "auto"
    tree_supervision_source: str = "rate"
    tree_checkpoint_metric: str = "val_root_mae"
    tree_stage1_checkpoint_metric: str = "val_root_mae"
    tree_stage1_eval_mode: str = "per_epoch"
    tree_stage1_screen_doc_limit: int = 0
    tree_stage1_final_exact_doc_limit: int = 0
    exact_metric_selection_doc_limit: int = 0
    exact_metric_selection_interval: int = 1
    tree_exact_eval_max_docs: int = 0
    tree_posttrain_train_doc_limit: int = 0
    tree_batch_pack_mode: str = "structure_bucket"
    tree_batch_token_budget: int = 0
    tree_batch_node_budget: int = 0
    tree_batch_autotune: bool = True
    tree_batch_structural_pad_limit: float = 0.5
    tree_batch_auto_queue_min_docs: int = 8
    tree_batch_auto_queue_min_fill_ratio: float = 0.5
    tree_eval_workers_per_mig: int = 0
    gpu_runtime_data_mode: str = "resident"
    gpu_runtime_bucket_mode: str = "exact_then_bucketed"
    gpu_runtime_preload_splits: tuple[str, ...] = ("train", "val", "test")
    gpu_runtime_preload_targets: bool = True
    gpu_runtime_workers_per_mig: int = 1
    gpu_runtime_allow_multi_worker_screen: bool = True
    gpu_runtime_capacity_workers_per_mig: int = 2
    tree_stage1_artifact_dir: str = ""
    prepared_data_root: str = ""
    prepared_data_allow_create: bool = True
    base_bundle_path: str = ""
    diagnostic_detail_mode: str = "summary"
    posttrain_diagnostics_mode: str = ""
    raw_diagnostic_artifact_dir: str = ""
    tree_stage1_root_weight: float = 0.0
    tree_join_bit_weight: float = 0.0
    tree_training_schedule: str = "two_stage"
    tree_stage1_epochs: int = 12
    tree_stage2_epochs: int = 20
    tree_task_head_mode: str = "full_state_scalar"
    tree_theorem_surface_mode: str = "slotwise"
    tree_theorem_count_head_mode: str = "scalar_mse"
    tree_theorem_count_ordinal_weight: float = 1.0
    tree_theorem_count_scalar_aux_weight: float = 0.25
    tree_theorem_count_threshold_balance: bool = True
    tree_theorem_feature_dim: int = 48
    tree_theorem_feature_hidden_dim: int = 256
    tree_merge_hidden_dim: int = 0
    tree_phi_compose_weight: float = 1.0
    tree_phi_contrastive_weight: float = 0.25
    tree_phi_alignment_loss: str = "cosine_mse"
    tree_c2_mode: str = "reconstruction"
    oracle_metric_name: str = ""
    oracle_same_threshold: float = 0.0
    oracle_diff_threshold: float = 0.0
    theorem_feature_adapter: str = "markov_count_sketch"
    theorem_pair_same_threshold: float | None = None
    theorem_pair_diff_threshold: float | None = None
    tree_summary_spec_root_mode: str = "task_split_ablation"
    doc_sequence_train_fraction: float = 0.0
    aligned_sketch_surface: str = ""
    summary_spec_name: str = ""
    slot_count: int = 0
    tree_theorem_score_dim: int = 0
    tree_theorem_fiber_dim: int = 0
    tree_theorem_aux_dim: int = 0
    tree_score_merge_mode: str = "gated_affine"
    tree_theorem_count_dim: int = 0
    tree_theorem_first_dim: int = 0
    tree_theorem_last_dim: int = 0
    leaf_supervision_kind: str = "full_sketch"
    internal_supervision_kind: str = "none"
    internal_label_rate: float = 0.0
    max_internal_depth: int = 0
    leaf_exact_supervision: bool = False
    leaf_label_rate: float = 1.0
    root_weight: float = 1.0
    schedule_consistency_weight: float = 0.0
    endpoint_loss_scale: float = 1.0
    budget_total_calls: int = 0
    budget_total_calls_per_doc: float = 0.0
    mass_target_per_doc: float = float("nan")
    full_doc_budget_share: float = 1.0
    doc_consumption_mode: str = ""
    local_split_mode: str = ""
    local_allocation_policy: str = ""
    package_semantics: str = ""
    depth_discount_gamma: float = 1.0

    def __post_init__(self) -> None:
        topology = str(self.topology or "").strip()
        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {sorted(VALID_TOPOLOGIES)}, got {topology!r}"
            )


@dataclass(frozen=True)
class _JobSpec:
    family: str
    train_doc_count: int
    benchmark: str
    hardness_grid: str
    grid_cell_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    config: _RunConfigSpec
    tuning_stage: str = ""
    test_metrics_hidden_during_selection: bool = False
    study_name: str = ""
    study_axis: str = ""
    axis_value: str = ""
    locked_tree_neural_config_label: str = ""
    selection_metric: str = ""

    def __post_init__(self) -> None:
        family = str(self.family or "").strip()
        if not family:
            raise ValueError("_JobSpec.family must be non-empty")
        config_family = str(getattr(self.config, "baseline_family", "") or "").strip()
        if config_family and config_family != family:
            raise ValueError(
                f"_JobSpec family/config mismatch: job.family={family!r} "
                f"config.baseline_family={config_family!r}"
            )
        if not config_family:
            object.__setattr__(
                self,
                "config",
                replace(self.config, baseline_family=family),
            )

    @property
    def job_name(self) -> str:
        scope = self.hardness_grid or self.benchmark
        cell_suffix = ""
        if self.grid_cell_ids:
            cell_suffix = "__" + "_".join(str(cell) for cell in self.grid_cell_ids)
        leaf_suffix = ""
        if self.config.fixed_leaf_tokens is not None:
            leaf_suffix = f"__leaf_{int(self.config.fixed_leaf_tokens)}"
        seed_suffix = ""
        if len(self.seeds) == 1:
            seed_suffix = f"__seed_{int(self.seeds[0])}"
        stage_suffix = ""
        if str(self.tuning_stage).strip():
            stage_suffix = f"__stage_{str(self.tuning_stage)}"
        config_suffix = ""
        if str(self.config.label).strip():
            config_suffix = f"__cfg_{str(self.config.label)}"
        study_suffix = ""
        study_axis = _sanitize_label(str(self.study_axis))
        axis_value = _sanitize_label(str(self.axis_value))
        if study_axis and axis_value:
            study_suffix = f"__{study_axis}_{axis_value}"
        return (
            f"{scope}__{self.family}__train_{int(self.train_doc_count)}"
            f"{cell_suffix}{leaf_suffix}{stage_suffix}{config_suffix}{study_suffix}{seed_suffix}"
        )

    @property
    def budget_total_calls(self) -> int:
        return int(self.config.budget_total_calls)

    @property
    def budget_total_calls_per_doc(self) -> float:
        return float(self.config.budget_total_calls_per_doc)

    @property
    def mass_target_per_doc(self) -> float:
        return float(self.config.mass_target_per_doc)

    @property
    def full_doc_budget_share(self) -> float:
        return float(self.config.full_doc_budget_share)

    @property
    def doc_consumption_mode(self) -> str:
        return str(self.config.doc_consumption_mode)

    @property
    def local_split_mode(self) -> str:
        return str(self.config.local_split_mode)

    @property
    def local_allocation_policy(self) -> str:
        return str(self.config.local_allocation_policy)

    @property
    def package_semantics(self) -> str:
        return str(self.config.package_semantics)


def _with_run_intent_overrides(
    config: _RunConfigSpec,
    *,
    budget_total_calls: int | None = None,
    budget_total_calls_per_doc: float | None = None,
    mass_target_per_doc: float | None = None,
    full_doc_budget_share: float | None = None,
    doc_consumption_mode: str | None = None,
    local_split_mode: str | None = None,
    local_allocation_policy: str | None = None,
    package_semantics: str | None = None,
    depth_discount_gamma: float | None = None,
) -> _RunConfigSpec:
    updated = replace(
        config,
        budget_total_calls=(
            int(budget_total_calls)
            if budget_total_calls is not None
            else int(config.budget_total_calls)
        ),
        budget_total_calls_per_doc=(
            float(budget_total_calls_per_doc)
            if budget_total_calls_per_doc is not None
            else float(config.budget_total_calls_per_doc)
        ),
        mass_target_per_doc=(
            float(mass_target_per_doc)
            if mass_target_per_doc is not None
            else float(config.mass_target_per_doc)
        ),
        full_doc_budget_share=(
            float(full_doc_budget_share)
            if full_doc_budget_share is not None
            else float(config.full_doc_budget_share)
        ),
        doc_consumption_mode=(
            str(doc_consumption_mode)
            if doc_consumption_mode is not None
            else str(config.doc_consumption_mode)
        ),
        local_split_mode=(
            str(local_split_mode)
            if local_split_mode is not None
            else str(config.local_split_mode)
        ),
        local_allocation_policy=(
            str(local_allocation_policy)
            if local_allocation_policy is not None
            else str(config.local_allocation_policy)
        ),
        package_semantics=(
            str(package_semantics)
            if package_semantics is not None
            else str(config.package_semantics)
        ),
        depth_discount_gamma=(
            float(depth_discount_gamma)
            if depth_discount_gamma is not None
            else float(config.depth_discount_gamma)
        ),
    )
    recompute_package_semantics = package_semantics is None and any(
        value is not None
        for value in (
            budget_total_calls,
            budget_total_calls_per_doc,
            mass_target_per_doc,
            full_doc_budget_share,
            doc_consumption_mode,
            local_split_mode,
            local_allocation_policy,
        )
    )
    if recompute_package_semantics or not str(updated.package_semantics).strip():
        package_semantics_mapping = asdict(updated)
        if recompute_package_semantics:
            package_semantics_mapping["package_semantics"] = ""
        updated = replace(
            updated,
            package_semantics=str(resolve_package_semantics(package_semantics_mapping)),
        )
    return updated


def _config_mapping_for_run_config(config: _RunConfigSpec) -> Dict[str, Any]:
    return runtime_config_overrides_from_config_like(config)


def _write_run_config_spec(path: Path, config: _RunConfigSpec) -> None:
    write_tree_run_config_json(path, config)


def _tree_base_config_preset(args: argparse.Namespace) -> Dict[str, Any]:
    preset_name = str(getattr(args, "base_config_preset", "") or "").strip()
    if not preset_name and str(getattr(args, "mode", "") or "").strip() == "capacity":
        profile = CAPACITY_PROFILE_PRESETS.get(
            str(getattr(args, "capacity_profile", ROOT_ONLY_CAPACITY_PROFILE_DEFAULT) or ROOT_ONLY_CAPACITY_PROFILE_DEFAULT).strip(),
            {},
        )
        preset_name = str(profile.get("base_config_preset", "") or "").strip()
    if not preset_name:
        return {}
    try:
        preset = resolve_tree_reference_preset_config(preset_name)
    except ValueError as exc:
        raise ValueError(
            "unsupported base_config_preset "
            f"{preset_name!r}; expected one of {list(tree_reference_preset_names())}"
        ) from exc
    return dict(preset)


def _arg_or_preset(
    args: argparse.Namespace,
    preset: Mapping[str, Any],
    attr_name: str,
    default: Any,
    *,
    preset_key: str | None = None,
) -> Any:
    if hasattr(args, attr_name):
        value = getattr(args, attr_name)
        if value is not None:
            return value
    key = str(preset_key or attr_name)
    if key in preset:
        return preset.get(key)
    return default


def _parse_mig_uuids(value: str) -> List[str]:
    tokens = [
        token.strip()
        for token in str(value or "").replace(",", " ").split()
        if token.strip()
    ]
    return tokens


def _parse_name_list(value: Sequence[str] | str | None, default: Sequence[str]) -> List[str]:
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, str):
        tokens = [
            token.strip()
            for token in str(value).replace(",", " ").split()
            if token.strip()
        ]
    else:
        tokens = [str(item).strip() for item in value if str(item).strip()]
    return tokens or [str(item) for item in default]


def _parse_optional_name_tuple(value: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(_parse_name_list(value, ()))


def _discover_mig_uuids() -> List[str]:
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=True,
    )
    uuids: List[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "MIG" not in line or "UUID:" not in line:
            continue
        uuids.append(line.split("UUID: ", 1)[1].rstrip(")"))
    return uuids


def _parse_mig_layout_from_nvidia_smi_listing(listing: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    current_gpu_index: int | None = None
    current_gpu_uuid = ""
    for raw_line in str(listing or "").splitlines():
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("GPU ") and "UUID:" in line:
            try:
                prefix, uuid_part = line.split("UUID: ", 1)
                current_gpu_index = int(prefix.split("GPU ", 1)[1].split(":", 1)[0])
                current_gpu_uuid = uuid_part.rstrip(")")
            except Exception:
                current_gpu_index = None
                current_gpu_uuid = ""
            continue
        if "MIG" not in line or "UUID:" not in line:
            continue
        if current_gpu_index is None or not current_gpu_uuid:
            continue
        mig_uuid = line.split("UUID: ", 1)[1].rstrip(")")
        entries.append(
            {
                "gpu_index": int(current_gpu_index),
                "gpu_uuid": str(current_gpu_uuid),
                "mig_uuid": str(mig_uuid),
            }
        )
    return entries


def _discover_mig_layout() -> List[Dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=True,
    )
    return _parse_mig_layout_from_nvidia_smi_listing(result.stdout)


def _mig_layout_by_uuid(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        mig_uuid = str(entry.get("mig_uuid", "")).strip()
        if not mig_uuid:
            continue
        out[mig_uuid] = {
            "mig_uuid": mig_uuid,
            "gpu_index": int(entry.get("gpu_index", -1)),
            "gpu_uuid": str(entry.get("gpu_uuid", "")),
        }
    return out


def _interleave_devices_by_physical_gpu(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    grouped: Dict[str, List[str]] = {}
    unknown: List[str] = []
    for token in [str(value) for value in tokens]:
        info = layout_by_uuid.get(str(token))
        if info is None:
            unknown.append(str(token))
            continue
        gpu_key = str(info.get("gpu_uuid", "") or f"gpu_index_{int(info.get('gpu_index', -1))}")
        grouped.setdefault(gpu_key, []).append(str(token))
    interleaved: List[str] = []
    while any(grouped.values()):
        for gpu_key in list(grouped.keys()):
            if not grouped[gpu_key]:
                continue
            interleaved.append(str(grouped[gpu_key].pop(0)))
    interleaved.extend(unknown)
    return interleaved


def _apply_screen_device_order(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
    order_mode: str,
) -> List[str]:
    normalized = str(order_mode or "input").strip().lower() or "input"
    if normalized == "interleave_by_physical_gpu":
        return _interleave_devices_by_physical_gpu(tokens, layout_by_uuid=layout_by_uuid)
    return [str(token) for token in tokens]


def _limit_devices_per_physical_gpu(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
    max_per_physical_gpu: int,
) -> List[str]:
    limit = int(max_per_physical_gpu)
    if limit <= 0:
        return [str(token) for token in tokens]
    counts: Dict[str, int] = {}
    kept: List[str] = []
    for token in [str(value) for value in tokens]:
        info = layout_by_uuid.get(str(token))
        if info is None:
            kept.append(str(token))
            continue
        gpu_key = str(info.get("gpu_uuid", "") or f"gpu_index_{int(info.get('gpu_index', -1))}")
        current = int(counts.get(gpu_key, 0))
        if current >= limit:
            continue
        counts[gpu_key] = int(current + 1)
        kept.append(str(token))
    return kept


def _group_devices_by_physical_gpu(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for token in [str(value) for value in tokens]:
        info = dict(layout_by_uuid.get(str(token), {}))
        gpu_key = str(info.get("gpu_uuid", "") or f"unknown::{token}")
        entry = grouped.setdefault(
            gpu_key,
            {
                "gpu_uuid": str(info.get("gpu_uuid", "")),
                "gpu_index": int(info.get("gpu_index", -1)),
                "mig_uuids": [],
            },
        )
        entry["mig_uuids"].append(str(token))
    return list(grouped.values())


def _strong_capacity_screen_guard_enabled(
    configs: Sequence[_RunConfigSpec],
) -> bool:
    if not configs:
        return False
    config = configs[0]
    return (
        str(config.gpu_runtime_data_mode).strip() == "resident"
        and str(config.tree_batch_pack_mode).strip() == "fixed_fused"
        and bool(str(config.summary_spec_name).strip())
        and str(config.tree_task_head_mode).strip() == "theorem_feature_scalar"
        and config.fixed_leaf_tokens is not None
    )


def _worker_device_context() -> Dict[str, Any]:
    visible_devices = [
        token.strip()
        for token in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).replace(",", " ").split()
        if token.strip()
    ]
    primary_token = str(visible_devices[0]) if visible_devices else ""
    context: Dict[str, Any] = {
        "cuda_visible_devices": list(visible_devices),
        "primary_visible_device": primary_token,
    }
    if primary_token.startswith("MIG-"):
        try:
            layout_by_uuid = _mig_layout_by_uuid(_discover_mig_layout())
        except Exception as exc:
            context["mig_layout_error"] = str(exc)
            return context
        info = layout_by_uuid.get(primary_token)
        if info is not None:
            context["resolved_device"] = dict(info)
    return context


def _sanitize_label(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() else "_" for ch in str(value).strip()
    ).strip("_")
    return cleaned or "default"


def _format_float_label(value: float) -> str:
    text = f"{float(value):.6g}"
    return _sanitize_label(text.replace("-", "m").replace(".", "p"))


def _job_output_dir_name(
    job_name: str,
    *,
    max_component_length: int = 180,
) -> str:
    name = str(job_name).strip() or "job"
    if len(name) <= int(max_component_length):
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    suffix = f"__h_{digest}"
    prefix_budget = max(1, int(max_component_length) - len(suffix))
    prefix = name[:prefix_budget].rstrip("_") or name[:prefix_budget]
    return f"{prefix}{suffix}"


def _default_run_config(args: argparse.Namespace, *, label: str = "default") -> _RunConfigSpec:
    preset = _tree_base_config_preset(args)

    def _value(attr_name: str, default: Any, *, preset_key: str | None = None) -> Any:
        return _arg_or_preset(
            args,
            preset,
            attr_name,
            default,
            preset_key=preset_key,
        )

    raw_preload_splits = _value(
        "gpu_runtime_preload_splits",
        ("train", "val", "test"),
        preset_key="gpu_runtime_preload_splits",
    )
    if isinstance(raw_preload_splits, str):
        preload_splits = tuple(
            item for item in str(raw_preload_splits).replace(",", " ").split() if item
        )
    else:
        preload_splits = tuple(
            str(item) for item in list(raw_preload_splits or ("train", "val", "test")) if str(item).strip()
        )
    fixed_leaf_tokens = _value("fixed_leaf_tokens", None)
    tree_local_law_weight = _value(
        "tree_local_law_weight",
        None,
        preset_key="local_law_weight",
    )
    tree_task_objective_weight = _value(
        "tree_task_objective_weight",
        None,
        preset_key="task_objective_weight",
    )
    tree_local_weighting_mode = _value(
        "tree_local_weighting_mode",
        "fixed_k_hajek",
    )
    tree_exact_collapse_mode = _value(
        "tree_exact_collapse_mode",
        "",
    )
    preserve_by_default = bool(
        fixed_leaf_tokens is not None and int(fixed_leaf_tokens) > 0
    )
    official_fno_preserve_requested_leaf_tokens = bool(
        _value(
            "official_fno_preserve_requested_leaf_tokens",
            preserve_by_default,
        )
    )
    preserve_requested_leaf_tokens = bool(
        _value("preserve_requested_leaf_tokens", preserve_by_default)
    ) or bool(official_fno_preserve_requested_leaf_tokens)
    comparison_mode = str(_value("comparison_mode", "legacy") or "legacy")
    tree_leaf_fno_width = _value("tree_leaf_fno_width", None)
    tree_leaf_fno_n_modes = _value("tree_leaf_fno_n_modes", None)
    tree_leaf_fno_n_layers = _value("tree_leaf_fno_n_layers", None)
    raw_tree_batch_pack_mode = _value(
        "tree_batch_pack_mode",
        "",
        preset_key="tree_batch_pack_mode",
    )
    theorem_pair_same_threshold = _value("theorem_pair_same_threshold", None)
    theorem_pair_diff_threshold = _value("theorem_pair_diff_threshold", None)
    return _with_run_intent_overrides(
        _RunConfigSpec(
        label=_sanitize_label(label),
        state_dim=int(_value("state_dim", 128)),
        hidden_dim=int(_value("hidden_dim", 512)),
        n_epochs=int(_value("n_epochs", 32)),
        batch_size=int(_value("batch_size", 64)),
        lr=float(_value("lr", 5e-4)),
        weight_decay=float(_value("weight_decay", 0.0)),
        fixed_leaf_tokens=None if fixed_leaf_tokens is None else int(fixed_leaf_tokens),
        tree_local_law_weight=(
            None if tree_local_law_weight is None else float(tree_local_law_weight)
        ),
        tree_task_objective_weight=(
            None
            if tree_task_objective_weight is None
            else float(tree_task_objective_weight)
        ),
        tree_local_weighting_mode=str(tree_local_weighting_mode or "fixed_k_hajek"),
        tree_exact_collapse_mode=str(tree_exact_collapse_mode or ""),
        official_fno_preserve_requested_leaf_tokens=bool(
            official_fno_preserve_requested_leaf_tokens
        ),
        preserve_requested_leaf_tokens=bool(preserve_requested_leaf_tokens),
        comparison_mode=str(comparison_mode or "legacy"),
        tree_c1_relative_weight=float(
            _value("tree_c1_relative_weight", 1.0, preset_key="c1_relative_weight")
        ),
        tree_c2_relative_weight=float(
            _value("tree_c2_relative_weight", 1.0, preset_key="c2_relative_weight")
        ),
        tree_c3_relative_weight=float(
            _value("tree_c3_relative_weight", 1.0, preset_key="c3_relative_weight")
        ),
        tree_leaf_fno_width=(
            None if tree_leaf_fno_width is None else int(tree_leaf_fno_width)
        ),
        tree_leaf_fno_n_modes=(
            None if tree_leaf_fno_n_modes is None else int(tree_leaf_fno_n_modes)
        ),
        tree_leaf_fno_n_layers=(
            None if tree_leaf_fno_n_layers is None else int(tree_leaf_fno_n_layers)
        ),
        tree_model_version=str(_value("tree_model_version", "")),
        tree_batch_runtime_mode=str(_value("tree_batch_runtime_mode", "")),
        tree_root_supervision_kind=str(_value("tree_root_supervision_kind", "mse")),
        tree_document_loss_normalization_mode=str(
            _value("tree_document_loss_normalization_mode", "auto")
        ),
        tree_supervision_source=str(_value("tree_supervision_source", "rate")),
        tree_checkpoint_metric=str(_value("tree_checkpoint_metric", "val_root_mae")),
        tree_stage1_checkpoint_metric=str(
            _value("tree_stage1_checkpoint_metric", "val_root_mae")
        ),
        tree_stage1_eval_mode=str(_value("tree_stage1_eval_mode", "per_epoch")),
        tree_stage1_screen_doc_limit=int(_value("tree_stage1_screen_doc_limit", 0)),
        tree_stage1_final_exact_doc_limit=int(
            _value("tree_stage1_final_exact_doc_limit", 0)
        ),
        exact_metric_selection_doc_limit=int(
            _value("exact_metric_selection_doc_limit", 0)
        ),
        exact_metric_selection_interval=int(
            _value("exact_metric_selection_interval", 1)
        ),
        tree_exact_eval_max_docs=int(_value("tree_exact_eval_max_docs", 0)),
        tree_posttrain_train_doc_limit=int(
            _value("tree_posttrain_train_doc_limit", 0)
        ),
        tree_batch_pack_mode=str(
            _resolved_tree_batch_pack_mode(
                benchmark=str(getattr(args, "benchmark", "")),
                raw_value=raw_tree_batch_pack_mode,
            )
        ),
        tree_batch_token_budget=int(_value("tree_batch_token_budget", 0)),
        tree_batch_node_budget=int(_value("tree_batch_node_budget", 0)),
        tree_batch_autotune=bool(_value("tree_batch_autotune", True)),
        tree_batch_structural_pad_limit=float(
            _value("tree_batch_structural_pad_limit", 0.5)
        ),
        tree_batch_auto_queue_min_docs=int(
            _value("tree_batch_auto_queue_min_docs", 8)
        ),
        tree_batch_auto_queue_min_fill_ratio=float(
            _value("tree_batch_auto_queue_min_fill_ratio", 0.5)
        ),
        tree_eval_workers_per_mig=int(_value("tree_eval_workers_per_mig", 0)),
        gpu_runtime_data_mode=str(_value("gpu_runtime_data_mode", "resident")),
        gpu_runtime_bucket_mode=str(
            _value("gpu_runtime_bucket_mode", "exact_then_bucketed")
        ),
        gpu_runtime_preload_splits=preload_splits or ("train", "val", "test"),
        gpu_runtime_preload_targets=bool(_value("gpu_runtime_preload_targets", True)),
        gpu_runtime_workers_per_mig=int(_value("gpu_runtime_workers_per_mig", 1)),
        gpu_runtime_allow_multi_worker_screen=bool(
            _value("gpu_runtime_allow_multi_worker_screen", True)
        ),
        gpu_runtime_capacity_workers_per_mig=int(
            _value("gpu_runtime_capacity_workers_per_mig", 2)
        ),
        tree_stage1_artifact_dir=str(_value("tree_stage1_artifact_dir", "")),
        prepared_data_root=str(_value("prepared_data_root", "")),
        prepared_data_allow_create=bool(_value("prepared_data_allow_create", True)),
        diagnostic_detail_mode=str(_value("diagnostic_detail_mode", "summary")),
        posttrain_diagnostics_mode=str(
            _value("posttrain_diagnostics_mode", "")
        ),
        raw_diagnostic_artifact_dir=str(_value("raw_diagnostic_artifact_dir", "")),
        tree_stage1_root_weight=float(_value("tree_stage1_root_weight", 0.0)),
        tree_join_bit_weight=float(_value("tree_join_bit_weight", 0.0)),
        tree_training_schedule=str(_value("tree_training_schedule", "two_stage")),
        tree_stage1_epochs=int(_value("tree_stage1_epochs", 12)),
        tree_stage2_epochs=int(_value("tree_stage2_epochs", 20)),
        tree_task_head_mode=str(_value("tree_task_head_mode", "full_state_scalar")),
        tree_theorem_surface_mode=str(
            _value("tree_theorem_surface_mode", "slotwise")
        ),
        tree_theorem_count_head_mode=str(
            _value("tree_theorem_count_head_mode", "scalar_mse")
        ),
        tree_theorem_count_ordinal_weight=float(
            _value("tree_theorem_count_ordinal_weight", 1.0)
        ),
        tree_theorem_count_scalar_aux_weight=float(
            _value("tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        tree_theorem_count_threshold_balance=bool(
            _value("tree_theorem_count_threshold_balance", True)
        ),
        tree_theorem_feature_dim=int(_value("tree_theorem_feature_dim", 48)),
        tree_theorem_feature_hidden_dim=int(
            _value("tree_theorem_feature_hidden_dim", 256)
        ),
        tree_merge_hidden_dim=int(_value("tree_merge_hidden_dim", 0)),
        tree_theorem_score_dim=int(_value("tree_theorem_score_dim", 0)),
        tree_theorem_fiber_dim=int(_value("tree_theorem_fiber_dim", 0)),
        tree_theorem_aux_dim=int(_value("tree_theorem_aux_dim", 0)),
        tree_score_merge_mode=str(_value("tree_score_merge_mode", "gated_affine")),
        tree_phi_compose_weight=float(_value("tree_phi_compose_weight", 1.0)),
        tree_phi_contrastive_weight=float(
            _value("tree_phi_contrastive_weight", 0.25)
        ),
        tree_phi_alignment_loss=str(_value("tree_phi_alignment_loss", "cosine_mse")),
        tree_c2_mode=str(_value("tree_c2_mode", "reconstruction")),
        theorem_feature_adapter=str(
            _value("theorem_feature_adapter", "markov_count_sketch")
        ),
        theorem_pair_same_threshold=(
            None
            if theorem_pair_same_threshold is None
            else float(theorem_pair_same_threshold)
        ),
        theorem_pair_diff_threshold=(
            None
            if theorem_pair_diff_threshold is None
            else float(theorem_pair_diff_threshold)
        ),
        tree_summary_spec_root_mode=str(
            _value("tree_summary_spec_root_mode", "task_split_ablation")
        ),
        doc_sequence_train_fraction=float(_value("doc_sequence_train_fraction", 0.0)),
        aligned_sketch_surface=str(_value("aligned_sketch_surface", "")),
        summary_spec_name=str(_value("summary_spec_name", "")),
        slot_count=int(_value("slot_count", 0)),
        tree_theorem_count_dim=int(_value("tree_theorem_count_dim", 0)),
        tree_theorem_first_dim=int(_value("tree_theorem_first_dim", 0)),
        tree_theorem_last_dim=int(_value("tree_theorem_last_dim", 0)),
        leaf_supervision_kind=str(_value("leaf_supervision_kind", "full_sketch")),
        internal_supervision_kind=str(_value("internal_supervision_kind", "none")),
        internal_label_rate=float(_value("internal_label_rate", 0.0)),
        max_internal_depth=int(_value("max_internal_depth", 0)),
        leaf_exact_supervision=bool(_value("leaf_exact_supervision", False)),
        leaf_label_rate=float(_value("leaf_label_rate", 1.0)),
        root_weight=float(_value("root_weight", 1.0)),
        schedule_consistency_weight=float(
            _value("schedule_consistency_weight", 0.0)
        ),
        endpoint_loss_scale=float(_value("endpoint_loss_scale", 1.0)),
        ),
        budget_total_calls=int(_value("budget_total_calls", 0)),
        budget_total_calls_per_doc=float(_value("budget_total_calls_per_doc", 0.0)),
        mass_target_per_doc=float(_value("mass_target_per_doc", float("nan"))),
        full_doc_budget_share=float(_value("full_doc_budget_share", 1.0)),
        doc_consumption_mode=str(_value("doc_consumption_mode", "")),
        local_split_mode=str(_value("local_split_mode", "")),
        local_allocation_policy=str(_value("local_allocation_policy", "")),
        package_semantics=str(_value("package_semantics", "")),
        depth_discount_gamma=float(_value("depth_discount_gamma", 1.0)),
    )


def _tuning_grid(args: argparse.Namespace) -> List[_RunConfigSpec]:
    configs: List[_RunConfigSpec] = []
    base_config = _default_run_config(args, label="tuning_base")
    for n_epochs, lr, local_law_weight in product(
        [int(value) for value in args.screen_n_epochs],
        [float(value) for value in args.screen_lrs],
        [float(value) for value in args.screen_tree_local_law_weights],
    ):
        label = (
            f"ep{int(n_epochs)}_lr{_format_float_label(float(lr))}"
            f"_llw{_format_float_label(float(local_law_weight))}"
        )
        configs.append(
            _RunConfigSpec(
                **{
                    **asdict(base_config),
                    "label": _sanitize_label(label),
                    "n_epochs": int(n_epochs),
                    "lr": float(lr),
                    "fixed_leaf_tokens": None,
                    "tree_local_law_weight": float(local_law_weight),
                    "tree_root_supervision_kind": "mse",
                }
            )
        )
    return configs


def _resolve_benchmark_leaf_tokens(
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


def _resolved_benchmark_payload(benchmark_name: str) -> Dict[str, Any]:
    benchmark = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    return {
        "benchmark": str(benchmark_name),
        "resolved_benchmark_name": str(benchmark.name),
        "benchmark_cell_id": str(benchmark.cell_id or ""),
        "benchmark_grid_name": str(benchmark.grid_name or ""),
    }


def _fair_fno_tree_config_for_train_doc_count(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    label: str = FAIR_FNO_PARITY_CONFIG_LABEL,
) -> _RunConfigSpec:
    preload_splits = tuple(
        str(item)
        for item in list(getattr(args, "gpu_runtime_preload_splits", ("train", "val", "test")))
        if str(item).strip()
    )
    fixed_leaf_tokens = _resolve_benchmark_leaf_tokens(
        benchmark_name=str(args.benchmark),
        train_doc_count=int(train_doc_count),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    return _RunConfigSpec(
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


def _fair_fno_parity_tree_config(args: argparse.Namespace) -> _RunConfigSpec:
    return _fair_fno_tree_config_for_train_doc_count(
        args,
        train_doc_count=int(
            getattr(
                args,
                "gate_train_doc_count",
                getattr(args, "train_doc_count", PARITY_GATE_TRAIN_DOC_COUNT),
            )
        ),
        label=FAIR_FNO_PARITY_CONFIG_LABEL,
    )


def _slot_exact_sanity_config(
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
) -> _RunConfigSpec:
    fair_base = _fair_fno_tree_config_for_train_doc_count(
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
            _resolved_tree_batch_pack_mode(
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


def _exact_sanity_configs_for_train_doc_count(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
) -> List[_RunConfigSpec]:
    configs: List[_RunConfigSpec] = []
    if int(train_doc_count) <= 1024:
        configs.append(
            _fair_fno_tree_config_for_train_doc_count(
                args,
                train_doc_count=int(train_doc_count),
                label=FAIR_FNO_PARITY_CONFIG_LABEL,
            )
        )
    configs.extend(
        [
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_root_only",
                leaf_label_rate=0.0,
                leaf_supervision_kind="count_only",
                internal_supervision_kind="none",
                internal_label_rate=0.0,
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_leaf_sampled",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="none",
                internal_label_rate=0.0,
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_internal_count_r0p25",
                leaf_label_rate=0.25,
                leaf_supervision_kind="count_only",
                internal_supervision_kind="count_only",
                internal_label_rate=0.25,
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_internal_full_r0p25",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="full_sketch",
                internal_label_rate=0.25,
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="full_sketch",
                internal_label_rate=0.25,
                tree_summary_spec_root_mode="task_split_ablation",
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_leaf_dense",
                leaf_label_rate=1.0,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="none",
                internal_label_rate=0.0,
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_internal_count_dense",
                leaf_label_rate=1.0,
                leaf_supervision_kind="count_only",
                internal_supervision_kind="count_only",
                internal_label_rate=1.0,
            ),
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_internal_full_dense",
                leaf_label_rate=1.0,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="full_sketch",
                internal_label_rate=1.0,
            ),
            # Diagnostic: full_sketch with rebalanced endpoint loss (hypothesis: CE scale mismatch)
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_balanced_full_r0p25",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="full_sketch",
                internal_label_rate=0.25,
                endpoint_loss_scale=0.1,
            ),
            # Diagnostic: leaf endpoints supervised + internal count-only
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_leaf_ep_count_r0p25",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="count_only",
                internal_label_rate=0.25,
                leaf_exact_supervision=True,
            ),
            # --- unified_f readout lanes (gradient alignment fix) ---
            # A: unified_f root-only baseline
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_unified_f_root_only",
                leaf_label_rate=0.0,
                leaf_supervision_kind="count_only",
                internal_supervision_kind="none",
                internal_label_rate=0.0,
                tree_summary_spec_root_mode="unified_f",
            ),
            # B: unified_f + internal count @0.25
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_unified_f_count_r0p25",
                leaf_label_rate=0.25,
                leaf_supervision_kind="count_only",
                internal_supervision_kind="count_only",
                internal_label_rate=0.25,
                tree_summary_spec_root_mode="unified_f",
            ),
            # C: unified_f + internal full_sketch @0.25
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_unified_f_full_r0p25",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="full_sketch",
                internal_label_rate=0.25,
                tree_summary_spec_root_mode="unified_f",
            ),
            # D: unified_f + dense internal count
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_unified_f_count_dense",
                leaf_label_rate=1.0,
                leaf_supervision_kind="count_only",
                internal_supervision_kind="count_only",
                internal_label_rate=1.0,
                tree_summary_spec_root_mode="unified_f",
            ),
        ]
    )
    extra_rates = sorted(
        {
            float(rate)
            for rate in tuple(getattr(args, "extra_high_rates", tuple()) or tuple())
            if float(rate) > 0.0 and float(rate) < 1.0 and abs(float(rate) - 0.25) > 1e-9
        }
    )
    for rate in extra_rates:
        rate_label = _format_float_label(float(rate))
        configs.extend(
            [
                _slot_exact_sanity_config(
                    args,
                    train_doc_count=int(train_doc_count),
                    config_label=f"tree_neural_slot_align_v1_leaf_r{rate_label}",
                    leaf_label_rate=float(rate),
                    leaf_supervision_kind="full_sketch",
                    internal_supervision_kind="none",
                    internal_label_rate=0.0,
                ),
                _slot_exact_sanity_config(
                    args,
                    train_doc_count=int(train_doc_count),
                    config_label=f"tree_neural_slot_align_v1_internal_count_r{rate_label}",
                    leaf_label_rate=float(rate),
                    leaf_supervision_kind="count_only",
                    internal_supervision_kind="count_only",
                    internal_label_rate=float(rate),
                ),
                _slot_exact_sanity_config(
                    args,
                    train_doc_count=int(train_doc_count),
                    config_label=f"tree_neural_slot_align_v1_internal_full_r{rate_label}",
                    leaf_label_rate=float(rate),
                    leaf_supervision_kind="full_sketch",
                    internal_supervision_kind="full_sketch",
                    internal_label_rate=float(rate),
                ),
            ]
        )
    if int(train_doc_count) > 1024:
        configs.append(
            _slot_exact_sanity_config(
                args,
                train_doc_count=int(train_doc_count),
                config_label="tree_neural_slot_align_v1_internal_full_r0p5",
                leaf_label_rate=0.25,
                leaf_supervision_kind="full_sketch",
                internal_supervision_kind="full_sketch",
                internal_label_rate=0.5,
            )
        )
    return configs


def _representation_hidden_dim_for_state_dim(state_dim: int) -> int:
    return max(128, 4 * int(state_dim))


def _representation_default_merge_hidden_dim(state_dim: int) -> int:
    return max(32, 4 * int(state_dim))


def _representation_exact_count_head_modes(
    args: argparse.Namespace,
) -> List[str]:
    raw_modes = tuple(
        str(value).strip().lower()
        for value in tuple(
            getattr(
                args,
                "representation_count_head_modes",
                REPRESENTATION_SUFFICIENCY_DEFAULT_COUNT_HEAD_MODES,
            )
            or REPRESENTATION_SUFFICIENCY_DEFAULT_COUNT_HEAD_MODES
        )
        if str(value).strip()
    )
    modes: List[str] = []
    for mode in raw_modes:
        if mode not in {"scalar_mse", "support_classifier", "hybrid_ordinal"}:
            raise ValueError(f"unsupported representation count head mode: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("representation_count_head_modes must contain at least one mode")
    return modes


def _representation_merge_hidden_dims_for_state_dim(
    args: argparse.Namespace,
    state_dim: int,
) -> List[int]:
    requested = int(getattr(args, "tree_merge_hidden_dim", 0) or 0)
    if requested > 0:
        return [requested]
    if int(state_dim) == 128:
        return [256]
    if int(state_dim) == 256:
        return [512]
    return [_representation_default_merge_hidden_dim(int(state_dim))]


def _representation_stage1_artifact_dir(
    args: argparse.Namespace,
    *,
    label: str,
) -> str:
    requested = str(getattr(args, "tree_stage1_artifact_dir", "") or "").strip()
    if requested:
        return requested
    return str(Path(str(getattr(args, "output_root"))) / "_stage1_artifacts" / str(label))


def _representation_sufficiency_config_metadata(
    config: _RunConfigSpec,
    *,
    baseline_family: str = REPRESENTATION_SUFFICIENCY_FAMILY,
    promotion_stage: str = "",
) -> Dict[str, Any]:
    family = str(baseline_family or REPRESENTATION_SUFFICIENCY_FAMILY).strip()
    surface_mode = str(config.tree_theorem_surface_mode or "").strip() or "slotwise"
    c2_mode = str(config.tree_c2_mode or "reconstruction").strip() or "reconstruction"
    score_merge_mode = (
        str(config.tree_score_merge_mode or "gated_affine").strip()
        or "gated_affine"
    )
    theorem_feature_dim = int(
        config.tree_theorem_feature_dim or config.state_dim or 0
    )
    theorem_count_head_mode = str(
        config.tree_theorem_count_head_mode or "scalar_mse"
    ).strip() or "scalar_mse"
    merge_hidden_dim = int(
        config.tree_merge_hidden_dim
        or _representation_default_merge_hidden_dim(int(config.state_dim))
    )
    representation_family = surface_mode
    representation_variant = surface_mode
    promotion_eligible = False
    control_only = False
    reference_only = False
    if family == "official_fno":
        representation_family = "official_fno_reference"
        representation_variant = "official_fno_reference"
        promotion_eligible = False
        control_only = True
        reference_only = True
        representation_size = "reference"
    elif surface_mode == "slotwise":
        representation_family = "slotwise"
        representation_variant = "slotwise_control"
        promotion_eligible = False
        control_only = True
        representation_size = (
            f"state{int(config.state_dim)}_slot{int(config.slot_count or 0)}"
        )
    elif surface_mode == "opaque_carrier_exact_sketch":
        representation_family = "opaque_carrier_exact_sketch"
        representation_variant = "opaque_carrier_exact_sketch"
        promotion_eligible = True
        representation_size = (
            "state"
            f"{int(config.state_dim)}_phi{int(theorem_feature_dim)}"
            f"_merge{int(merge_hidden_dim)}_head{theorem_count_head_mode}"
        )
    elif surface_mode == "shared_feature" and c2_mode == "fiber":
        representation_family = "shared_feature"
        representation_variant = (
            "shared_feature__exact_projected_merge__c2_fiber"
            if score_merge_mode == "exact_projected_sketch"
            else "shared_feature__c2_fiber"
        )
        promotion_eligible = False
        representation_size = (
            f"state{int(config.state_dim)}_phi{int(theorem_feature_dim)}"
        )
    elif surface_mode == "shared_feature":
        representation_family = "shared_feature"
        representation_variant = (
            "shared_feature__exact_projected_merge"
            if score_merge_mode == "exact_projected_sketch"
            else "shared_feature"
        )
        promotion_eligible = False
        representation_size = (
            f"state{int(config.state_dim)}_phi{int(theorem_feature_dim)}"
        )
    elif surface_mode == "shared_feature_adapters":
        representation_family = "shared_feature_adapters"
        representation_variant = (
            "shared_feature_adapters__exact_projected_merge"
            if score_merge_mode == "exact_projected_sketch"
            else "shared_feature_adapters"
        )
        promotion_eligible = False
        representation_size = (
            f"state{int(config.state_dim)}_phi{int(theorem_feature_dim)}"
        )
    elif surface_mode == "factorized_score_fiber":
        representation_family = "factorized_score_fiber"
        representation_variant = "factorized_score_fiber"
        promotion_eligible = False
        representation_size = (
            f"state{int(config.state_dim)}_phi{int(theorem_feature_dim)}"
        )
    else:
        representation_size = (
            f"state{int(config.state_dim)}_phi{int(theorem_feature_dim)}"
        )
    return {
        "representation_family": str(representation_family),
        "representation_variant": str(representation_variant),
        "representation_size": str(representation_size),
        "promotion_eligible": bool(promotion_eligible),
        "control_only": bool(control_only),
        "reference_only": bool(reference_only),
        "promotion_stage": str(promotion_stage),
        "state_dim": int(config.state_dim),
        "hidden_dim": int(config.hidden_dim),
        "theorem_feature_dim": int(theorem_feature_dim),
        "theorem_feature_hidden_dim": int(config.tree_theorem_feature_hidden_dim),
        "merge_hidden_dim": int(merge_hidden_dim),
        "carrier_merge_input_dim": int(2 * int(config.state_dim)),
        "slot_count": int(config.slot_count or 0),
        "tree_theorem_surface_mode": str(surface_mode),
        "tree_theorem_count_head_mode": str(theorem_count_head_mode),
        "tree_c2_mode": str(c2_mode),
        "tree_score_merge_mode": str(score_merge_mode),
        "exact_lane": bool(
            surface_mode == "opaque_carrier_exact_sketch"
            and score_merge_mode == "exact_projected_sketch"
        ),
        "tree_theorem_score_dim": int(config.tree_theorem_score_dim or 0),
        "tree_theorem_fiber_dim": int(config.tree_theorem_fiber_dim or 0),
        "tree_theorem_aux_dim": int(config.tree_theorem_aux_dim or 0),
    }


def _representation_sufficiency_tree_config(
    args: argparse.Namespace,
    *,
    label: str,
    state_dim: int,
    theorem_surface_mode: str,
    theorem_feature_dim: int,
    merge_hidden_dim: int = 0,
    count_head_mode: str | None = None,
    c2_mode: str = "reconstruction",
    score_merge_mode: str = "gated_affine",
    score_dim: int = 0,
    fiber_dim: int = 0,
    aux_dim: int = 0,
) -> _RunConfigSpec:
    base = _default_run_config(args, label=str(label))
    hidden_dim = _representation_hidden_dim_for_state_dim(int(state_dim))
    theorem_feature_hidden_dim = max(32, 2 * int(theorem_feature_dim))
    effective_merge_hidden_dim = int(
        merge_hidden_dim or _representation_default_merge_hidden_dim(int(state_dim))
    )
    effective_count_head_mode = (
        str(count_head_mode or getattr(args, "tree_theorem_count_head_mode", "scalar_mse"))
        .strip()
        .lower()
        or "scalar_mse"
    )
    is_opaque_exact_lane = (
        str(theorem_surface_mode).strip() == "opaque_carrier_exact_sketch"
        and str(score_merge_mode).strip() == "exact_projected_sketch"
    )
    return replace(
        base,
        label=str(label),
        state_dim=int(state_dim),
        hidden_dim=int(hidden_dim),
        n_epochs=int(getattr(args, "n_epochs", 32)),
        batch_size=int(getattr(args, "batch_size", 64)),
        lr=float(getattr(args, "lr", 5e-4)),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
        fixed_leaf_tokens=None,
        tree_local_law_weight=(
            None
            if getattr(args, "tree_local_law_weight", 0.8) is None
            else float(getattr(args, "tree_local_law_weight", 0.8))
        ),
        tree_task_objective_weight=(
            None
            if getattr(args, "tree_task_objective_weight", None) is None
            else float(getattr(args, "tree_task_objective_weight", None))
        ),
        tree_c1_relative_weight=float(
            getattr(args, "tree_c1_relative_weight", 1.0)
        ),
        tree_c2_relative_weight=float(
            getattr(args, "tree_c2_relative_weight", 1.0)
        ),
        tree_c3_relative_weight=float(
            getattr(args, "tree_c3_relative_weight", 1.0)
        ),
        tree_root_supervision_kind="mse",
        tree_checkpoint_metric=REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
        tree_stage1_checkpoint_metric="val_theorem_bootstrap_direct",
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
        tree_exact_eval_max_docs=int(getattr(args, "tree_exact_eval_max_docs", 0)),
        tree_eval_workers_per_mig=int(getattr(args, "tree_eval_workers_per_mig", 0)),
        tree_stage1_artifact_dir=_representation_stage1_artifact_dir(
            args,
            label=str(label),
        ),
        tree_stage1_root_weight=0.0,
        tree_join_bit_weight=float(getattr(args, "tree_join_bit_weight", 1.0)),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=int(getattr(args, "tree_stage1_epochs", 12)),
        tree_stage2_epochs=int(getattr(args, "tree_stage2_epochs", 20)),
        tree_task_head_mode="theorem_feature_scalar",
        tree_theorem_surface_mode=str(theorem_surface_mode),
        tree_theorem_count_head_mode=str(effective_count_head_mode),
        tree_theorem_count_ordinal_weight=float(
            getattr(args, "tree_theorem_count_ordinal_weight", 1.0)
        ),
        tree_theorem_count_scalar_aux_weight=float(
            getattr(args, "tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        tree_theorem_count_threshold_balance=bool(
            getattr(args, "tree_theorem_count_threshold_balance", True)
        ),
        tree_theorem_feature_dim=int(theorem_feature_dim),
        tree_theorem_feature_hidden_dim=int(theorem_feature_hidden_dim),
        tree_merge_hidden_dim=int(effective_merge_hidden_dim),
        tree_phi_compose_weight=(
            0.0
            if is_opaque_exact_lane
            else float(getattr(args, "tree_phi_compose_weight", 1.0))
        ),
        tree_phi_contrastive_weight=(
            0.0
            if is_opaque_exact_lane
            else float(getattr(args, "tree_phi_contrastive_weight", 0.25))
        ),
        tree_phi_alignment_loss=str(
            getattr(args, "tree_phi_alignment_loss", "cosine_mse")
        ),
        tree_c2_mode=str(c2_mode),
        tree_score_merge_mode=str(score_merge_mode),
        theorem_feature_adapter="markov_count_sketch",
        tree_summary_spec_root_mode="factored_theorem_readout",
        doc_sequence_train_fraction=0.0,
        aligned_sketch_surface="",
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        tree_theorem_score_dim=int(score_dim),
        tree_theorem_fiber_dim=int(fiber_dim),
        tree_theorem_aux_dim=int(aux_dim),
        tree_theorem_count_dim=0,
        tree_theorem_first_dim=0,
        tree_theorem_last_dim=0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="full_sketch",
        internal_label_rate=1.0,
        max_internal_depth=int(getattr(args, "max_internal_depth", 0)),
        leaf_exact_supervision=False,
        leaf_label_rate=1.0,
    )


def _representation_sufficiency_official_fno_config(
    args: argparse.Namespace,
) -> _RunConfigSpec:
    base = _default_run_config(args, label="official_fno_reference")
    return replace(
        base,
        label="official_fno_reference",
        n_epochs=int(getattr(args, "n_epochs", 32)),
        batch_size=int(getattr(args, "batch_size", 64)),
        lr=float(getattr(args, "lr", 5e-4)),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
        fixed_leaf_tokens=None,
        doc_sequence_train_fraction=0.0,
    )


def _representation_sufficiency_screen_config_specs(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    state_dims = sorted(
        {
            int(value)
            for value in tuple(
                getattr(args, "representation_state_dims", (128,)) or (128,)
            )
            if int(value) > 0
        }
    )
    if not state_dims:
        raise ValueError("representation_state_dims must contain at least one value")
    count_head_modes = _representation_exact_count_head_modes(args)
    config_by_label: Dict[str, _RunConfigSpec] = {}
    slotwise_control_labels_by_state_dim: Dict[int, str] = {}
    for state_dim in state_dims:
        theorem_feature_dim = int(state_dim)
        slotwise_label = f"slotwise_control_s{int(state_dim)}"
        slotwise_control_labels_by_state_dim[int(state_dim)] = str(slotwise_label)
        config_by_label[str(slotwise_label)] = _representation_sufficiency_tree_config(
            args,
            label=str(slotwise_label),
            state_dim=int(state_dim),
            theorem_surface_mode="slotwise",
            theorem_feature_dim=int(theorem_feature_dim),
        )
        shared_label = f"shared_feature_s{int(state_dim)}_phi{int(theorem_feature_dim)}"
        config_by_label[str(shared_label)] = _representation_sufficiency_tree_config(
            args,
            label=str(shared_label),
            state_dim=int(state_dim),
            theorem_surface_mode="shared_feature",
            theorem_feature_dim=int(theorem_feature_dim),
        )
        for count_head_mode in count_head_modes:
            for merge_hidden_dim in _representation_merge_hidden_dims_for_state_dim(
                args,
                int(state_dim),
            ):
                opaque_label = (
                    "opaque_carrier_exact_sketch_"
                    f"s{int(state_dim)}_phi{int(theorem_feature_dim)}"
                    f"_m{int(merge_hidden_dim)}_head_{str(count_head_mode)}"
                )
                config_by_label[str(opaque_label)] = _representation_sufficiency_tree_config(
                    args,
                    label=str(opaque_label),
                    state_dim=int(state_dim),
                    theorem_surface_mode="opaque_carrier_exact_sketch",
                    theorem_feature_dim=int(theorem_feature_dim),
                    merge_hidden_dim=int(merge_hidden_dim),
                    count_head_mode=str(count_head_mode),
                    score_merge_mode="exact_projected_sketch",
                )
    official_fno_config = _representation_sufficiency_official_fno_config(args)
    config_by_label[str(official_fno_config.label)] = official_fno_config
    config_metadata_by_label = {
        str(label): _representation_sufficiency_config_metadata(
            config,
            baseline_family=(
                "official_fno"
                if str(label) == str(official_fno_config.label)
                else REPRESENTATION_SUFFICIENCY_FAMILY
            ),
            promotion_stage="screen",
        )
        for label, config in config_by_label.items()
    }
    return {
        "config_by_label": config_by_label,
        "config_metadata_by_label": config_metadata_by_label,
        "slotwise_control_labels_by_state_dim": {
            int(key): str(value)
            for key, value in slotwise_control_labels_by_state_dim.items()
        },
        "official_fno_label": str(official_fno_config.label),
    }


def _representation_learnability_benchmark_specs(
    args: argparse.Namespace,
) -> List[Any]:
    if bool(getattr(args, "full_structural_grid", False)):
        return [
            resolve_full_doc_diagnostic_benchmark("recoverable_v4"),
            *list(resolve_full_doc_diagnostic_grid("structural_core_v1")),
        ]
    raw_cells = tuple(
        str(value).strip()
        for value in tuple(
            getattr(
                args,
                "benchmark_cells",
                REPRESENTATION_LEARNABILITY_DEFAULT_BENCHMARK_CELLS,
            )
            or REPRESENTATION_LEARNABILITY_DEFAULT_BENCHMARK_CELLS
        )
        if str(value).strip()
    )
    if not raw_cells:
        raise ValueError("benchmark_cells must contain at least one benchmark cell")
    benchmarks: List[Any] = []
    seen: Set[str] = set()
    for token in raw_cells:
        normalized = str(token).strip()
        if (
            normalized != "recoverable_v4"
            and "::" not in normalized
            and not normalized.startswith("recoverable_structural_core_v1__")
        ):
            normalized = f"structural_core_v1::{normalized}"
        benchmark = resolve_full_doc_diagnostic_benchmark(normalized)
        key = str(benchmark.name)
        if key in seen:
            continue
        seen.add(key)
        benchmarks.append(benchmark)
    return benchmarks


def _representation_learnability_benchmark_metadata(
    benchmark_name: str,
) -> Dict[str, Any]:
    benchmark = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    policy = resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )
    generator_profile = str(
        (benchmark.config_overrides or {}).get("generator_profile", "")
        or getattr(policy, "generator_profile", "")
        or ""
    ).strip()
    lean_recoverable = (
        str(generator_profile).lower() == "piecewise_disjoint_palette"
    )
    regime_count = int(benchmark.regime_count or getattr(policy, "n_regimes", 0) or 0)
    segment_density_band = str(benchmark.segment_density_band or "canonical")
    return {
        "benchmark": str(benchmark.name),
        "benchmark_cell": str(benchmark.cell_id or benchmark.name),
        "benchmark_grid_name": str(benchmark.grid_name or ""),
        "benchmark_description": str(benchmark.description or ""),
        "generator_profile": str(generator_profile),
        "regime_count": int(regime_count),
        "segment_density_band": str(segment_density_band),
        "segment_min": int(benchmark.segment_min or getattr(policy, "min_segments", 0) or 0),
        "segment_max": int(benchmark.segment_max or getattr(policy, "max_segments", 0) or 0),
        "lean_recoverable_in_principle": bool(lean_recoverable),
        "lean_bayes_error_zero": bool(lean_recoverable),
        "lean_observed_token_recoverability_ref": (
            "piecewise_disjoint_palette_observed_tokens_recover_latent_path"
        ),
        "lean_exact_sketch_recoverability_ref": (
            "piecewise_disjoint_palette_observed_tokens_recover_exact_sketch"
        ),
        "lean_zero_bayes_error_ref": "piecewise_disjoint_palette_zero_bayes_error",
        "lean_representation_exact_pass_ref": (
            "markov_representation_exact_recovery_implies_query_sufficient"
        ),
        "lean_representation_zero_root_count_error_ref": (
            "markov_representation_exact_recovery_zero_root_count_error"
        ),
        "lean_representation_count_transport_ref": (
            "markov_count_error_le_exact_sketch_error"
        ),
    }


def _representation_metric_lookup(
    entry: Mapping[str, Any] | None,
    metric_key: str,
) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return dict((entry.get("metrics") or {}).get(metric_key) or {})


def _representation_metric_se(stats: Mapping[str, Any] | None) -> float:
    if not isinstance(stats, Mapping):
        return float("nan")
    mean = float(stats.get("mean", float("nan")))
    std = float(stats.get("std", float("nan")))
    n = int(stats.get("n", 0))
    if not np.isfinite(mean) or not np.isfinite(std) or n <= 0:
        return float("nan")
    return float(std / np.sqrt(float(n)))


def _budget_frontier_tree_config(
    args: argparse.Namespace,
) -> tuple[_RunConfigSpec, str]:
    config_mode = str(getattr(args, "budget_tree_config_mode", "parity")).strip().lower()
    if config_mode == "default":
        return (_default_run_config(args, label="budget_tree_default"), "")
    capacity_root_value = str(getattr(args, "capacity_root", "")).strip()
    if capacity_root_value:
        capacity_root = Path(capacity_root_value)
        locked_config, _capacity_summary = _locked_tree_neural_config_from_capacity_root(
            capacity_root
        )
        return (
            replace(
                locked_config,
                label=str(locked_config.label or FAIR_FNO_PARITY_CONFIG_LABEL),
            ),
            str(capacity_root),
        )
    return (
        _fair_fno_tree_config_for_train_doc_count(
            args,
            train_doc_count=int(args.train_doc_count),
            label=FAIR_FNO_PARITY_CONFIG_LABEL,
        ),
        "",
    )


def _capacity_config_label(
    *,
    width: int,
    n_modes: int,
    n_layers: int,
) -> str:
    return f"{FAIR_FNO_PARITY_CONFIG_LABEL}_w{int(width)}_m{int(n_modes)}_l{int(n_layers)}"


def _resolved_capacity_profile_name(args: argparse.Namespace) -> str:
    raw = str(
        getattr(args, "capacity_profile", ROOT_ONLY_CAPACITY_PROFILE_DEFAULT)
        or ROOT_ONLY_CAPACITY_PROFILE_DEFAULT
    ).strip()
    if raw not in CAPACITY_PROFILE_PRESETS:
        raise ValueError(
            f"unsupported capacity_profile {raw!r}; expected one of {sorted(CAPACITY_PROFILE_PRESETS)}"
        )
    return raw


def _capacity_profile_defaults(args: argparse.Namespace) -> Mapping[str, Any]:
    return dict(CAPACITY_PROFILE_PRESETS[_resolved_capacity_profile_name(args)])


def _resolved_capacity_width_values(args: argparse.Namespace) -> List[int]:
    return [int(value) for value in _capacity_axis_values(
        args,
        attr_name="capacity_widths",
        default_values=CAPACITY_WIDTH_AXIS,
        coerce=int,
    )]


def _resolved_capacity_mode_values(args: argparse.Namespace) -> List[int]:
    return [int(value) for value in _capacity_axis_values(
        args,
        attr_name="capacity_modes",
        default_values=CAPACITY_MODES_AXIS,
        coerce=int,
    )]


def _resolved_capacity_layer_values(args: argparse.Namespace) -> List[int]:
    return [int(value) for value in _capacity_axis_values(
        args,
        attr_name="capacity_layers",
        default_values=CAPACITY_LAYERS_AXIS,
        coerce=int,
    )]


def _resolved_capacity_axis_metadata(
    args: argparse.Namespace,
    *,
    base_config: _RunConfigSpec | None = None,
) -> Dict[str, Any]:
    effective_base = base_config or _default_run_config(args, label="capacity_axis_base")
    fixed_leaf_defaults = (
        (None,)
        if effective_base.fixed_leaf_tokens is None
        else (int(effective_base.fixed_leaf_tokens),)
    )
    return {
        "capacity_profile": _resolved_capacity_profile_name(args),
        "capacity_widths": _resolved_capacity_width_values(args),
        "capacity_modes": _resolved_capacity_mode_values(args),
        "capacity_layers": _resolved_capacity_layer_values(args),
        "capacity_state_dims": _capacity_axis_values(
            args,
            attr_name="capacity_state_dims",
            default_values=(int(effective_base.state_dim),),
            coerce=int,
        ),
        "capacity_hidden_dims": _capacity_axis_values(
            args,
            attr_name="capacity_hidden_dims",
            default_values=(int(effective_base.hidden_dim),),
            coerce=int,
        ),
        "capacity_n_epochs": _capacity_axis_values(
            args,
            attr_name="capacity_n_epochs",
            default_values=(int(effective_base.n_epochs),),
            coerce=int,
        ),
        "capacity_tree_training_schedules": _capacity_axis_values(
            args,
            attr_name="capacity_tree_training_schedules",
            default_values=(str(effective_base.tree_training_schedule),),
            coerce=str,
        ),
        "capacity_tree_checkpoint_metrics": _capacity_axis_values(
            args,
            attr_name="capacity_tree_checkpoint_metrics",
            default_values=(str(effective_base.tree_checkpoint_metric),),
            coerce=str,
        ),
        "capacity_tree_stage1_checkpoint_metrics": _capacity_axis_values(
            args,
            attr_name="capacity_tree_stage1_checkpoint_metrics",
            default_values=(str(effective_base.tree_stage1_checkpoint_metric),),
            coerce=str,
        ),
        "capacity_tree_stage1_root_weights": _capacity_axis_values(
            args,
            attr_name="capacity_tree_stage1_root_weights",
            default_values=(float(effective_base.tree_stage1_root_weight),),
            coerce=float,
        ),
        "capacity_slot_counts": _capacity_axis_values(
            args,
            attr_name="capacity_slot_counts",
            default_values=(int(effective_base.slot_count),),
            coerce=int,
        ),
        "capacity_fixed_leaf_tokens": _capacity_axis_values(
            args,
            attr_name="capacity_fixed_leaf_tokens",
            default_values=fixed_leaf_defaults,
            coerce=lambda value: None if value is None else int(value),
        ),
    }


def _capacity_axis_values(
    args: argparse.Namespace,
    *,
    attr_name: str,
    default_values: Sequence[Any],
    coerce,
) -> List[Any]:
    raw_values = getattr(args, attr_name, None)
    if raw_values is None or (isinstance(raw_values, Sequence) and not list(raw_values)):
        raw_values = _capacity_profile_defaults(args).get(attr_name, default_values)
    values: List[Any] = []
    for raw in list(raw_values or default_values):
        values.append(coerce(raw))
    deduped: List[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped or [coerce(item) for item in list(default_values)]


def _capacity_metric_slug(metric: str) -> str:
    aliases = {
        "val_root_mae": "root",
        "val_exact_sketch_direct": "exact",
        "val_theorem_bootstrap_direct": "theorem",
    }
    metric_text = str(metric or "").strip()
    if metric_text in aliases:
        return aliases[metric_text]
    return _sanitize_label(metric_text)


def _capacity_schedule_slug(schedule: str) -> str:
    mapping = {
        "single_stage": "single",
        "two_stage": "two",
    }
    return mapping.get(str(schedule or "").strip(), _sanitize_label(str(schedule or "")))


def _extended_capacity_config_label(
    *,
    args: argparse.Namespace,
    base_config: _RunConfigSpec,
    width: int,
    n_modes: int,
    n_layers: int,
    state_dim: int,
    hidden_dim: int,
    n_epochs: int,
    tree_training_schedule: str,
    tree_checkpoint_metric: str,
    tree_stage1_checkpoint_metric: str,
    tree_stage1_root_weight: float,
    slot_count: int,
    fixed_leaf_tokens: int | None,
    axis_sizes: Mapping[str, int],
) -> str:
    profile_name = _resolved_capacity_profile_name(args)
    no_extended_axes = all(
        int(axis_sizes.get(name, 1)) <= 1
        for name in (
            "capacity_state_dims",
            "capacity_hidden_dims",
            "capacity_n_epochs",
            "capacity_tree_training_schedules",
            "capacity_tree_checkpoint_metrics",
            "capacity_tree_stage1_checkpoint_metrics",
            "capacity_tree_stage1_root_weights",
            "capacity_slot_counts",
            "capacity_fixed_leaf_tokens",
        )
    )
    if profile_name == ROOT_ONLY_CAPACITY_PROFILE_DEFAULT and no_extended_axes:
        return _capacity_config_label(width=width, n_modes=n_modes, n_layers=n_layers)

    parts = [
        _sanitize_label(profile_name),
        f"w{int(width)}",
        f"m{int(n_modes)}",
        f"l{int(n_layers)}",
    ]
    if int(axis_sizes.get("capacity_state_dims", 1)) > 1 or int(state_dim) != int(base_config.state_dim):
        parts.append(f"sd{int(state_dim)}")
    if int(axis_sizes.get("capacity_hidden_dims", 1)) > 1 or int(hidden_dim) != int(base_config.hidden_dim):
        parts.append(f"hd{int(hidden_dim)}")
    if int(axis_sizes.get("capacity_n_epochs", 1)) > 1 or int(n_epochs) != int(base_config.n_epochs):
        parts.append(f"ep{int(n_epochs)}")
    if (
        int(axis_sizes.get("capacity_tree_training_schedules", 1)) > 1
        or str(tree_training_schedule) != str(base_config.tree_training_schedule)
    ):
        parts.append(f"sched{_capacity_schedule_slug(tree_training_schedule)}")
    if (
        int(axis_sizes.get("capacity_tree_checkpoint_metrics", 1)) > 1
        or str(tree_checkpoint_metric) != str(base_config.tree_checkpoint_metric)
    ):
        parts.append(f"ckpt{_capacity_metric_slug(tree_checkpoint_metric)}")
    if (
        int(axis_sizes.get("capacity_tree_stage1_checkpoint_metrics", 1)) > 1
        or str(tree_stage1_checkpoint_metric) != str(base_config.tree_stage1_checkpoint_metric)
    ):
        parts.append(f"s1{_capacity_metric_slug(tree_stage1_checkpoint_metric)}")
    base_stage1_root_weight = float(base_config.tree_stage1_root_weight)
    if (
        int(axis_sizes.get("capacity_tree_stage1_root_weights", 1)) > 1
        or abs(float(tree_stage1_root_weight) - base_stage1_root_weight) > 1e-9
    ):
        parts.append(f"s1rw{_format_float_label(float(tree_stage1_root_weight))}")
    if int(axis_sizes.get("capacity_slot_counts", 1)) > 1 or int(slot_count) != int(base_config.slot_count):
        parts.append(f"slot{int(slot_count)}")
    base_fixed_leaf_tokens = base_config.fixed_leaf_tokens
    if (
        int(axis_sizes.get("capacity_fixed_leaf_tokens", 1)) > 1
        or int(fixed_leaf_tokens or 0) != int(base_fixed_leaf_tokens or 0)
    ):
        parts.append(
            "leafnone" if fixed_leaf_tokens is None else f"leaf{int(fixed_leaf_tokens)}"
        )
    return "_".join(str(part) for part in parts if str(part).strip())


def _capacity_grid(args: argparse.Namespace) -> List[_RunConfigSpec]:
    configs: List[_RunConfigSpec] = []
    base_config = _default_run_config(args, label="capacity_base")
    use_preset_base = bool(_tree_base_config_preset(args))
    width_values = _resolved_capacity_width_values(args)
    mode_values = _resolved_capacity_mode_values(args)
    layer_values = _resolved_capacity_layer_values(args)
    state_dim_values = _capacity_axis_values(
        args,
        attr_name="capacity_state_dims",
        default_values=(int(base_config.state_dim),),
        coerce=int,
    )
    hidden_dim_values = _capacity_axis_values(
        args,
        attr_name="capacity_hidden_dims",
        default_values=(int(base_config.hidden_dim),),
        coerce=int,
    )
    n_epoch_values = _capacity_axis_values(
        args,
        attr_name="capacity_n_epochs",
        default_values=(int(base_config.n_epochs),),
        coerce=int,
    )
    training_schedule_values = _capacity_axis_values(
        args,
        attr_name="capacity_tree_training_schedules",
        default_values=(str(base_config.tree_training_schedule),),
        coerce=str,
    )
    checkpoint_metric_values = _capacity_axis_values(
        args,
        attr_name="capacity_tree_checkpoint_metrics",
        default_values=(str(base_config.tree_checkpoint_metric),),
        coerce=str,
    )
    stage1_checkpoint_metric_values = _capacity_axis_values(
        args,
        attr_name="capacity_tree_stage1_checkpoint_metrics",
        default_values=(str(base_config.tree_stage1_checkpoint_metric),),
        coerce=str,
    )
    stage1_root_weight_values = _capacity_axis_values(
        args,
        attr_name="capacity_tree_stage1_root_weights",
        default_values=(float(base_config.tree_stage1_root_weight),),
        coerce=float,
    )
    slot_count_values = _capacity_axis_values(
        args,
        attr_name="capacity_slot_counts",
        default_values=(int(base_config.slot_count),),
        coerce=int,
    )
    fixed_leaf_token_values = _capacity_axis_values(
        args,
        attr_name="capacity_fixed_leaf_tokens",
        default_values=((None,) if base_config.fixed_leaf_tokens is None else (int(base_config.fixed_leaf_tokens),)),
        coerce=lambda value: None if value is None else int(value),
    )
    axis_sizes = {
        "capacity_state_dims": len(state_dim_values),
        "capacity_hidden_dims": len(hidden_dim_values),
        "capacity_n_epochs": len(n_epoch_values),
        "capacity_tree_training_schedules": len(training_schedule_values),
        "capacity_tree_checkpoint_metrics": len(checkpoint_metric_values),
        "capacity_tree_stage1_checkpoint_metrics": len(stage1_checkpoint_metric_values),
        "capacity_tree_stage1_root_weights": len(stage1_root_weight_values),
        "capacity_slot_counts": len(slot_count_values),
        "capacity_fixed_leaf_tokens": len(fixed_leaf_token_values),
    }
    for (
        width,
        n_modes,
        n_layers,
        state_dim,
        hidden_dim,
        n_epochs,
        tree_training_schedule,
        tree_checkpoint_metric,
        tree_stage1_checkpoint_metric,
        tree_stage1_root_weight,
        slot_count,
        fixed_leaf_tokens,
    ) in product(
        width_values,
        mode_values,
        layer_values,
        state_dim_values,
        hidden_dim_values,
        n_epoch_values,
        training_schedule_values,
        checkpoint_metric_values,
        stage1_checkpoint_metric_values,
        stage1_root_weight_values,
        slot_count_values,
        fixed_leaf_token_values,
    ):
        overrides = {
            "label": _extended_capacity_config_label(
                args=args,
                base_config=base_config,
                width=int(width),
                n_modes=int(n_modes),
                n_layers=int(n_layers),
                state_dim=int(state_dim),
                hidden_dim=int(hidden_dim),
                n_epochs=int(n_epochs),
                tree_training_schedule=str(tree_training_schedule),
                tree_checkpoint_metric=str(tree_checkpoint_metric),
                tree_stage1_checkpoint_metric=str(tree_stage1_checkpoint_metric),
                tree_stage1_root_weight=float(tree_stage1_root_weight),
                slot_count=int(slot_count),
                fixed_leaf_tokens=(
                    None if fixed_leaf_tokens is None else int(fixed_leaf_tokens)
                ),
                axis_sizes=axis_sizes,
            ),
            "state_dim": int(state_dim),
            "hidden_dim": int(hidden_dim),
            "n_epochs": int(n_epochs),
            "tree_leaf_fno_width": int(width),
            "tree_leaf_fno_n_modes": int(n_modes),
            "tree_leaf_fno_n_layers": int(n_layers),
            "tree_training_schedule": str(tree_training_schedule),
            "tree_checkpoint_metric": str(tree_checkpoint_metric),
            "tree_stage1_checkpoint_metric": str(tree_stage1_checkpoint_metric),
            "tree_stage1_root_weight": float(tree_stage1_root_weight),
            "slot_count": int(slot_count),
            "fixed_leaf_tokens": (
                None if fixed_leaf_tokens is None else int(fixed_leaf_tokens)
            ),
            "doc_sequence_train_fraction": 0.0,
        }
        if not use_preset_base:
            overrides.update(
                {
                    "fixed_leaf_tokens": None,
                    "tree_root_supervision_kind": "count_ce",
                }
            )
        configs.append(
            _with_run_intent_overrides(
                _RunConfigSpec(**{**asdict(base_config), **overrides})
            )
        )
    return configs


def _capacity_screen_runtime_override_values(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    data_mode = str(getattr(args, "screen_gpu_runtime_data_mode", "") or "").strip()
    if data_mode:
        overrides["gpu_runtime_data_mode"] = data_mode
    bucket_mode = str(getattr(args, "screen_gpu_runtime_bucket_mode", "") or "").strip()
    if bucket_mode:
        overrides["gpu_runtime_bucket_mode"] = bucket_mode
    preload_splits = _parse_optional_name_tuple(
        getattr(args, "screen_gpu_runtime_preload_splits", None)
    )
    if preload_splits is not None:
        overrides["gpu_runtime_preload_splits"] = preload_splits
    preload_targets = getattr(args, "screen_gpu_runtime_preload_targets", None)
    if preload_targets is not None:
        overrides["gpu_runtime_preload_targets"] = bool(preload_targets)
    return overrides


def _apply_capacity_screen_runtime_overrides(
    config: _RunConfigSpec,
    *,
    args: argparse.Namespace,
) -> _RunConfigSpec:
    overrides = _capacity_screen_runtime_override_values(args)
    if not overrides:
        return config
    return replace(config, **overrides)


def _estimate_capacity_screen_worker_preflight(
    *,
    args: argparse.Namespace,
    config: _RunConfigSpec | None,
) -> Dict[str, Any]:
    if config is None:
        return {"available": False, "reason": "missing_config"}
    try:
        raw_seeds = getattr(args, "screen_seeds", None)
        if raw_seeds is None:
            raw_seeds = getattr(args, "seeds", None)
        seed_values = [int(seed) for seed in list(raw_seeds or ())]
        return estimate_tree_worker_runtime_preflight(
            benchmark_name=str(args.benchmark),
            hardness_grid="",
            grid_cell_ids=(),
            train_doc_count=int(args.train_doc_count),
            config_overrides=_config_mapping_for_run_config(config),
            use_cuda=bool(args.use_cuda),
            torch_threads=int(args.torch_threads),
            seed=int(seed_values[0] if seed_values else 0),
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": "estimation_failed",
            "error": str(exc),
        }


def _capacity_screen_effective_policy(
    *,
    args: argparse.Namespace,
    screen_configs: Sequence[_RunConfigSpec],
) -> Dict[str, Any]:
    strong_guard = _strong_capacity_screen_guard_enabled(screen_configs)
    requested_cap = int(
        getattr(args, "screen_max_concurrent_per_physical_gpu", 0) or 0
    )
    requested_order = (
        str(getattr(args, "screen_device_order", "input") or "input").strip().lower()
        or "input"
    )
    requested_allow_multi_worker_screen = bool(
        getattr(args, "gpu_runtime_allow_multi_worker_screen", True)
    )
    requested_workers_per_mig = max(
        1,
        int(getattr(args, "gpu_runtime_capacity_workers_per_mig", 2) or 1),
    )
    effective_cap = int(requested_cap)
    effective_order = str(requested_order)
    effective_allow_multi_worker_screen = bool(requested_allow_multi_worker_screen)
    effective_workers_per_mig = int(
        requested_workers_per_mig if requested_allow_multi_worker_screen else 1
    )
    auto_safe_applied = False

    if strong_guard and requested_order == "input":
        effective_order = "interleave_by_physical_gpu"

    return {
        "strong_guard_enabled": bool(strong_guard),
        "requested_screen_max_concurrent_per_physical_gpu": int(requested_cap),
        "requested_screen_device_order": str(requested_order),
        "requested_gpu_runtime_allow_multi_worker_screen": bool(
            requested_allow_multi_worker_screen
        ),
        "requested_gpu_runtime_capacity_workers_per_mig": int(
            requested_workers_per_mig
        ),
        "effective_screen_max_concurrent_per_physical_gpu": int(effective_cap),
        "effective_screen_device_order": str(effective_order),
        "effective_gpu_runtime_allow_multi_worker_screen": bool(
            effective_allow_multi_worker_screen
        ),
        "effective_gpu_runtime_capacity_workers_per_mig": int(
            effective_workers_per_mig
        ),
        "auto_safe_applied": bool(auto_safe_applied),
    }


def _capacity_screen_preflight(
    *,
    args: argparse.Namespace,
    screen_jobs: Sequence[_JobSpec],
    raw_screen_worker_slots: Sequence[str],
    ordered_screen_worker_slots: Sequence[str],
    active_screen_worker_slots: Sequence[str],
    screen_configs: Sequence[_RunConfigSpec],
    mig_layout: Sequence[Mapping[str, Any]],
    effective_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    layout_by_uuid = _mig_layout_by_uuid(mig_layout)
    representative_config = screen_configs[0] if screen_configs else None
    strong_guard = bool(effective_policy.get("strong_guard_enabled", False))
    worker_preflight = _estimate_capacity_screen_worker_preflight(
        args=args,
        config=representative_config,
    )
    active_groups = _group_devices_by_physical_gpu(
        active_screen_worker_slots,
        layout_by_uuid=layout_by_uuid,
    )
    raw_groups = _group_devices_by_physical_gpu(
        raw_screen_worker_slots,
        layout_by_uuid=layout_by_uuid,
    )
    first_wave_count = min(len(active_screen_worker_slots), len(screen_jobs))
    first_wave_rows: List[Dict[str, Any]] = []
    first_wave_by_gpu: Dict[str, Dict[str, Any]] = {}
    for token, job in zip(
        list(active_screen_worker_slots)[:first_wave_count],
        list(screen_jobs)[:first_wave_count],
    ):
        info = dict(layout_by_uuid.get(str(token), {}))
        gpu_key = str(info.get("gpu_uuid", "") or f"unknown::{token}")
        row = {
            "mig_uuid": str(token),
            "gpu_uuid": str(info.get("gpu_uuid", "")),
            "gpu_index": int(info.get("gpu_index", -1)),
            "job_name": str(job.job_name),
            "config_label": str(job.config.label),
        }
        first_wave_rows.append(dict(row))
        entry = first_wave_by_gpu.setdefault(
            gpu_key,
            {
                "gpu_uuid": str(info.get("gpu_uuid", "")),
                "gpu_index": int(info.get("gpu_index", -1)),
                "jobs": [],
            },
        )
        entry["jobs"].append(dict(row))
    max_active_per_gpu = max(
        (len(list(group.get("mig_uuids") or [])) for group in active_groups),
        default=0,
    )
    requested_cap = int(
        effective_policy.get("requested_screen_max_concurrent_per_physical_gpu", 0) or 0
    )
    effective_cap = int(
        effective_policy.get("effective_screen_max_concurrent_per_physical_gpu", 0) or 0
    )
    requested_order = str(
        effective_policy.get("requested_screen_device_order", "input") or "input"
    )
    effective_order = str(
        effective_policy.get("effective_screen_device_order", "input") or "input"
    )
    auto_safe_applied = bool(effective_policy.get("auto_safe_applied", False))
    worker_total_bytes = int(
        dict(worker_preflight).get("resident_store_bytes_total", 0) or 0
    )
    per_gpu_projected_bytes: List[Dict[str, Any]] = []
    for group in active_groups:
        slot_count = int(len(list(group.get("mig_uuids") or [])))
        per_gpu_projected_bytes.append(
            {
                "gpu_uuid": str(group.get("gpu_uuid", "")),
                "gpu_index": int(group.get("gpu_index", -1)),
                "active_screen_workers": int(slot_count),
                "projected_resident_store_bytes_total": int(
                    worker_total_bytes * int(slot_count)
                ),
            }
        )
    violations: List[Dict[str, Any]] = []
    recommended_flags: List[str] = []
    status = "ok"
    if strong_guard and requested_cap > 0 and max_active_per_gpu > 1:
        status = "unsafe_capacity_screen_layout"
        violations.append(
            {
                "code": "strong_screen_physical_gpu_concurrency",
                "message": (
                    "Strong resident capacity screen would run more than one active "
                    "screen worker on a physical GPU."
                ),
                "max_active_screen_workers_per_physical_gpu": int(max_active_per_gpu),
                "supported_safe_cap": 1,
                "requested_cap": int(requested_cap),
            }
        )
        recommended_flags = [
            "--screen-max-concurrent-per-physical-gpu 1",
            "--screen-device-order interleave_by_physical_gpu",
        ]
    return {
        "status": str(status),
        "strong_guard_enabled": bool(strong_guard),
        "requested_screen_max_concurrent_per_physical_gpu": int(requested_cap),
        "requested_screen_device_order": str(requested_order),
        "effective_screen_max_concurrent_per_physical_gpu": int(effective_cap),
        "effective_screen_device_order": str(effective_order),
        "auto_safe_applied": bool(auto_safe_applied),
        "raw_screen_worker_slots": [str(token) for token in raw_screen_worker_slots],
        "ordered_screen_worker_slots": [
            str(token) for token in ordered_screen_worker_slots
        ],
        "active_screen_worker_slots": [
            str(token) for token in active_screen_worker_slots
        ],
        "raw_screen_worker_slots_by_physical_gpu": list(raw_groups),
        "active_screen_worker_slots_by_physical_gpu": list(active_groups),
        "first_wave_jobs": list(first_wave_rows),
        "first_wave_jobs_by_physical_gpu": list(first_wave_by_gpu.values()),
        "worker_runtime_preflight": dict(worker_preflight),
        "projected_first_wave_bytes_by_physical_gpu": list(per_gpu_projected_bytes),
        "violations": list(violations),
        "recommended_safe_rerun_flags": list(recommended_flags),
    }


def _capacity_screen_job_order_key(job: _JobSpec) -> tuple[int, int, int, int, str]:
    config = job.config
    min_seed = int(min(job.seeds)) if job.seeds else 0
    n_layers = int(config.tree_leaf_fno_n_layers or 0)
    n_modes = int(config.tree_leaf_fno_n_modes or 0)
    width = int(config.tree_leaf_fno_width or 0)
    return (
        int(min_seed),
        -int(n_layers),
        -int(n_modes),
        -int(width),
        str(config.label),
    )


def _reorder_capacity_screen_jobs(
    jobs: Sequence[_JobSpec],
    *,
    strong_guard: bool,
) -> List[_JobSpec]:
    ordered = [job for job in jobs]
    if not strong_guard:
        return ordered
    return sorted(ordered, key=_capacity_screen_job_order_key)


def _cached_capacity_screen_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    cached = getattr(args, "_capacity_screen_job_bundle_cache", None)
    if isinstance(cached, dict):
        return cached
    bundle = build_capacity_screen_job_bundle(args)
    setattr(args, "_capacity_screen_job_bundle_cache", bundle)
    return bundle


def _parity_tree_config_from_base(
    base: _RunConfigSpec,
    *,
    config_label: str,
    doc_sequence_train_fraction: float,
) -> _RunConfigSpec:
    return _RunConfigSpec(
        **{
            **asdict(base),
            "label": str(config_label),
            "fixed_leaf_tokens": None,
            "tree_root_supervision_kind": "count_ce",
            "doc_sequence_train_fraction": float(doc_sequence_train_fraction),
        }
    )


def _write_worker_invocation_snapshot(snapshot: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(snapshot), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_proc_key_value_file(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return payload
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        value = raw_value.strip()
        if value.endswith(" kB"):
            try:
                payload[str(key)] = int(value[:-3].strip())
                continue
            except Exception:
                pass
        payload[str(key)] = value
    return payload


def _memory_probe_callback_from_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _callback(event: str, payload: Mapping[str, Any]) -> None:
        pid = int(os.getpid())
        proc_status = _read_proc_key_value_file(Path(f"/proc/{pid}/status"))
        proc_smaps_rollup = _read_proc_key_value_file(Path(f"/proc/{pid}/smaps_rollup"))
        row: Dict[str, Any] = {
            "event": str(event),
            "payload": {str(key): value for key, value in dict(payload).items()},
            "wall_time_s": float(time.time()),
            "monotonic_s": float(time.monotonic()),
            "pid": int(pid),
            "rss_kib": int(proc_status.get("VmRSS", 0) or 0),
            "swap_kib": int(proc_status.get("VmSwap", 0) or 0),
            "pss_kib": int(proc_smaps_rollup.get("Pss", 0) or 0),
            "private_dirty_kib": int(proc_smaps_rollup.get("Private_Dirty", 0) or 0),
            "anonymous_kib": int(proc_smaps_rollup.get("Anonymous", 0) or 0),
        }
        if torch.cuda.is_available():
            try:
                device_index = int(torch.cuda.current_device())
                row["cuda_device_index"] = int(device_index)
                row["memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device_index))
                row["memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device_index))
                row["max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(device_index)
                )
                row["max_memory_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(device_index)
                )
            except Exception:
                pass
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    return _callback


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return rows
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            rows.append({str(key): value for key, value in dict(payload).items()})
    return rows


def _summarize_memory_probe_file(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl_rows(path)
    job_dir = path.parent
    first_event = str(rows[0].get("event", "")) if rows else ""
    last_event = str(rows[-1].get("event", "")) if rows else ""
    max_private_dirty_kib = 0
    max_private_dirty_event = ""
    max_rss_kib = 0
    max_rss_event = ""
    max_swap_kib = 0
    max_swap_event = ""
    largest_private_dirty_delta_kib = 0
    largest_private_dirty_delta_from_event = ""
    largest_private_dirty_delta_to_event = ""
    largest_private_dirty_delta_from_kib = 0
    largest_private_dirty_delta_to_kib = 0
    reached_pre_exact_eval_batch = False
    reached_post_exact_eval_batch = False
    reached_post_exact_eval_batch_trim = False
    previous_row: Dict[str, Any] | None = None
    for row in rows:
        event = str(row.get("event", ""))
        private_dirty_kib = int(row.get("private_dirty_kib", 0) or 0)
        rss_kib = int(row.get("rss_kib", 0) or 0)
        swap_kib = int(row.get("swap_kib", 0) or 0)
        if private_dirty_kib >= max_private_dirty_kib:
            max_private_dirty_kib = int(private_dirty_kib)
            max_private_dirty_event = event
        if rss_kib >= max_rss_kib:
            max_rss_kib = int(rss_kib)
            max_rss_event = event
        if swap_kib >= max_swap_kib:
            max_swap_kib = int(swap_kib)
            max_swap_event = event
        if event == "pre_exact_eval_batch":
            reached_pre_exact_eval_batch = True
        elif event == "post_exact_eval_batch":
            reached_post_exact_eval_batch = True
        elif event == "post_exact_eval_batch_trim":
            reached_post_exact_eval_batch_trim = True
        if previous_row is not None:
            previous_private_dirty_kib = int(
                previous_row.get("private_dirty_kib", 0) or 0
            )
            delta_kib = int(private_dirty_kib - previous_private_dirty_kib)
            if delta_kib >= largest_private_dirty_delta_kib:
                largest_private_dirty_delta_kib = int(delta_kib)
                largest_private_dirty_delta_from_event = str(
                    previous_row.get("event", "")
                )
                largest_private_dirty_delta_to_event = event
                largest_private_dirty_delta_from_kib = int(previous_private_dirty_kib)
                largest_private_dirty_delta_to_kib = int(private_dirty_kib)
        previous_row = row
    return {
        "job_dir": str(job_dir),
        "job_dir_name": str(job_dir.name),
        "probe_jsonl": str(path),
        "n_rows": int(len(rows)),
        "first_event": first_event,
        "last_event": last_event,
        "reached_pre_exact_eval_batch": bool(reached_pre_exact_eval_batch),
        "reached_post_exact_eval_batch": bool(reached_post_exact_eval_batch),
        "reached_post_exact_eval_batch_trim": bool(reached_post_exact_eval_batch_trim),
        "max_private_dirty_kib": int(max_private_dirty_kib),
        "max_private_dirty_event": max_private_dirty_event,
        "max_rss_kib": int(max_rss_kib),
        "max_rss_event": max_rss_event,
        "max_swap_kib": int(max_swap_kib),
        "max_swap_event": max_swap_event,
        "largest_private_dirty_delta_kib": int(largest_private_dirty_delta_kib),
        "largest_private_dirty_delta_from_event": largest_private_dirty_delta_from_event,
        "largest_private_dirty_delta_to_event": largest_private_dirty_delta_to_event,
        "largest_private_dirty_delta_from_kib": int(
            largest_private_dirty_delta_from_kib
        ),
        "largest_private_dirty_delta_to_kib": int(
            largest_private_dirty_delta_to_kib
        ),
    }


def _write_memory_probe_summary(output_root: Path) -> Dict[str, Any]:
    probe_paths = sorted(output_root.rglob("memory_probe.jsonl"))
    worker_summaries = [
        _summarize_memory_probe_file(path)
        for path in probe_paths
    ]
    peak_private_dirty = sorted(
        worker_summaries,
        key=lambda row: int(row.get("max_private_dirty_kib", 0) or 0),
        reverse=True,
    )
    peak_private_dirty_deltas = sorted(
        worker_summaries,
        key=lambda row: int(row.get("largest_private_dirty_delta_kib", 0) or 0),
        reverse=True,
    )
    payload = {
        "output_root": str(output_root),
        "probe_files_found": int(len(probe_paths)),
        "jobs_with_rows": int(
            sum(1 for row in worker_summaries if int(row.get("n_rows", 0) or 0) > 0)
        ),
        "jobs_reaching_pre_exact_eval_batch": int(
            sum(
                1
                for row in worker_summaries
                if bool(row.get("reached_pre_exact_eval_batch", False))
            )
        ),
        "jobs_reaching_post_exact_eval_batch": int(
            sum(
                1
                for row in worker_summaries
                if bool(row.get("reached_post_exact_eval_batch", False))
            )
        ),
        "jobs_reaching_post_exact_eval_batch_trim": int(
            sum(
                1
                for row in worker_summaries
                if bool(row.get("reached_post_exact_eval_batch_trim", False))
            )
        ),
        "peak_private_dirty_jobs": list(peak_private_dirty[:8]),
        "largest_private_dirty_delta_jobs": list(peak_private_dirty_deltas[:8]),
        "workers": list(worker_summaries),
    }
    summary_path = output_root / "memory_probe_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_path)
    return payload


def _execute_worker_invocation(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(snapshot["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_overrides = dict(snapshot.get("config_overrides", {}))
    run_metadata = dict(snapshot.get("run_metadata", {}))
    base_bundle_path = str(
        snapshot.get("base_bundle_path", "")
        or run_metadata.get("base_bundle_path", "")
        or ""
    ).strip()
    if "gpu_runtime_preload_splits" in config_overrides:
        config_overrides["gpu_runtime_preload_splits"] = tuple(
            str(item) for item in list(config_overrides["gpu_runtime_preload_splits"])
        )
    memory_probe_jsonl = str(snapshot.get("memory_probe_jsonl", "") or "").strip()
    memory_probe = (
        _memory_probe_callback_from_jsonl(Path(memory_probe_jsonl))
        if memory_probe_jsonl
        else None
    )
    return run_markov_full_doc_anchor_diagnostics(
        benchmark_name=str(snapshot["benchmark_name"]),
        hardness_grid=str(snapshot.get("hardness_grid", "")),
        grid_cell_ids=tuple(
            str(cell) for cell in list(snapshot.get("grid_cell_ids", []))
        ),
        seeds=tuple(int(seed) for seed in list(snapshot.get("seeds", []))),
        train_doc_counts=tuple(
            int(count) for count in list(snapshot.get("train_doc_counts", []))
        ),
        baseline_families=tuple(
            str(family) for family in list(snapshot.get("baseline_families", []))
        ),
        emit_confusion=bool(snapshot.get("emit_confusion", False)),
        output_dir=output_dir,
        use_cuda=bool(snapshot.get("use_cuda", True)),
        cuda_device=(
            int(snapshot["cuda_device"])
            if snapshot.get("cuda_device", None) is not None
            else None
        ),
        torch_threads=int(snapshot.get("torch_threads", 1)),
        config_overrides=config_overrides,
        run_metadata=run_metadata,
        memory_probe=memory_probe,
        base_bundle_path=base_bundle_path,
    )


def _replay_worker_snapshot_payload(args: argparse.Namespace) -> Dict[str, Any]:
    snapshot_path = Path(str(args.snapshot_json))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    output_override = str(getattr(args, "output_dir", "") or "").strip()
    if output_override:
        snapshot["output_dir"] = output_override
    if getattr(args, "use_cuda", None) is not None:
        snapshot["use_cuda"] = bool(args.use_cuda)
        if not bool(args.use_cuda):
            snapshot["cuda_device"] = None
        elif snapshot.get("cuda_device", None) is None:
            snapshot["cuda_device"] = 0
    if getattr(args, "cuda_device", None) is not None:
        snapshot["cuda_device"] = int(args.cuda_device)
    if getattr(args, "torch_threads", None) is not None:
        snapshot["torch_threads"] = int(args.torch_threads)
    memory_probe_jsonl = str(getattr(args, "memory_probe_jsonl", "") or "").strip()
    if memory_probe_jsonl:
        snapshot["memory_probe_jsonl"] = memory_probe_jsonl
    output_dir = Path(str(snapshot["output_dir"]))
    replay_snapshot_path = output_dir / "replayed_worker_invocation_snapshot.json"
    _write_worker_invocation_snapshot(snapshot, replay_snapshot_path)
    t0 = time.monotonic()
    payload = _execute_worker_invocation(snapshot)
    payload["elapsed_seconds"] = float(time.monotonic() - t0)
    payload["job_name"] = str(snapshot.get("job_name", ""))
    payload["replayed_from_snapshot_json"] = str(snapshot_path)
    payload["replay_snapshot_json"] = str(replay_snapshot_path)
    return payload


def _worker_run_config_for_preflight(
    args: argparse.Namespace,
    *,
    config_overrides: Mapping[str, Any],
) -> _RunConfigSpec:
    label = str(args.config_label or "worker")
    merged = {
        **asdict(_default_run_config(args, label=label)),
        **dict(config_overrides),
        "label": label,
    }
    alias_pairs = (
        ("local_law_weight", "tree_local_law_weight"),
        ("task_objective_weight", "tree_task_objective_weight"),
        ("c1_relative_weight", "tree_c1_relative_weight"),
        ("c2_relative_weight", "tree_c2_relative_weight"),
        ("c3_relative_weight", "tree_c3_relative_weight"),
    )
    for source_key, target_key in alias_pairs:
        if source_key in merged and target_key not in dict(config_overrides):
            merged[target_key] = merged[source_key]
    for source_key, _target_key in alias_pairs:
        merged.pop(source_key, None)
    return _run_config_from_mapping(merged)


def _worker_payload(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_overrides: Dict[str, Any] = {
        "state_dim": int(args.state_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_epochs": int(args.n_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
    }
    if args.fixed_leaf_tokens is not None:
        config_overrides["fixed_leaf_tokens"] = int(args.fixed_leaf_tokens)
        config_overrides["preserve_requested_leaf_tokens"] = True
        config_overrides["official_fno_preserve_requested_leaf_tokens"] = True
    if args.tree_local_law_weight is not None:
        config_overrides["local_law_weight"] = float(args.tree_local_law_weight)
    if args.tree_task_objective_weight is not None:
        config_overrides["task_objective_weight"] = float(
            args.tree_task_objective_weight
        )
    config_overrides["tree_local_weighting_mode"] = str(
        getattr(args, "tree_local_weighting_mode", "fixed_k_hajek")
        or "fixed_k_hajek"
    )
    if str(getattr(args, "tree_exact_collapse_mode", "")).strip():
        config_overrides["tree_exact_collapse_mode"] = str(
            getattr(args, "tree_exact_collapse_mode", "")
        )
    if bool(getattr(args, "official_fno_preserve_requested_leaf_tokens", False)):
        config_overrides["official_fno_preserve_requested_leaf_tokens"] = True
    if bool(getattr(args, "preserve_requested_leaf_tokens", False)):
        config_overrides["preserve_requested_leaf_tokens"] = True
    config_overrides["comparison_mode"] = str(
        getattr(args, "comparison_mode", "legacy") or "legacy"
    )
    config_overrides["c1_relative_weight"] = float(
        getattr(args, "tree_c1_relative_weight", 1.0)
    )
    config_overrides["c2_relative_weight"] = float(
        getattr(args, "tree_c2_relative_weight", 1.0)
    )
    config_overrides["c3_relative_weight"] = float(
        getattr(args, "tree_c3_relative_weight", 1.0)
    )
    if args.tree_leaf_fno_width is not None:
        config_overrides["tree_leaf_fno_width"] = int(args.tree_leaf_fno_width)
    if args.tree_leaf_fno_n_modes is not None:
        config_overrides["tree_leaf_fno_n_modes"] = int(args.tree_leaf_fno_n_modes)
    if args.tree_leaf_fno_n_layers is not None:
        config_overrides["tree_leaf_fno_n_layers"] = int(args.tree_leaf_fno_n_layers)
    if str(getattr(args, "tree_model_version", "")).strip():
        config_overrides["tree_model_version"] = str(args.tree_model_version)
    if str(getattr(args, "tree_batch_runtime_mode", "")).strip():
        config_overrides["tree_batch_runtime_mode"] = str(args.tree_batch_runtime_mode)
    if str(args.tree_root_supervision_kind).strip():
        config_overrides["tree_root_supervision_kind"] = str(
            args.tree_root_supervision_kind
        )
    if str(getattr(args, "tree_document_loss_normalization_mode", "")).strip():
        config_overrides["tree_document_loss_normalization_mode"] = str(
            getattr(args, "tree_document_loss_normalization_mode", "auto")
        )
    if str(getattr(args, "tree_supervision_source", "")).strip():
        config_overrides["tree_supervision_source"] = str(
            getattr(args, "tree_supervision_source", "rate")
        )
    if str(getattr(args, "tree_checkpoint_metric", "")).strip():
        config_overrides["tree_checkpoint_metric"] = str(
            getattr(args, "tree_checkpoint_metric", "val_root_mae")
        )
    if str(getattr(args, "tree_stage1_checkpoint_metric", "")).strip():
        config_overrides["tree_stage1_checkpoint_metric"] = str(
            getattr(args, "tree_stage1_checkpoint_metric", "val_root_mae")
        )
    if str(getattr(args, "tree_stage1_eval_mode", "")).strip():
        config_overrides["tree_stage1_eval_mode"] = str(
            getattr(args, "tree_stage1_eval_mode", "per_epoch")
        )
    config_overrides["tree_stage1_screen_doc_limit"] = int(
        getattr(args, "tree_stage1_screen_doc_limit", 0)
    )
    config_overrides["tree_stage1_final_exact_doc_limit"] = int(
        getattr(args, "tree_stage1_final_exact_doc_limit", 0)
    )
    config_overrides["exact_metric_selection_doc_limit"] = int(
        getattr(args, "exact_metric_selection_doc_limit", 0)
    )
    config_overrides["exact_metric_selection_interval"] = int(
        getattr(args, "exact_metric_selection_interval", 1)
    )
    config_overrides["tree_exact_eval_max_docs"] = int(
        getattr(args, "tree_exact_eval_max_docs", 0)
    )
    config_overrides["tree_posttrain_train_doc_limit"] = int(
        getattr(args, "tree_posttrain_train_doc_limit", 0)
    )
    if str(getattr(args, "tree_batch_pack_mode", "")).strip():
        config_overrides["tree_batch_pack_mode"] = str(
            getattr(args, "tree_batch_pack_mode", "structure_bucket")
        )
    config_overrides["tree_batch_token_budget"] = int(
        getattr(args, "tree_batch_token_budget", 0)
    )
    config_overrides["tree_batch_node_budget"] = int(
        getattr(args, "tree_batch_node_budget", 0)
    )
    config_overrides["tree_batch_autotune"] = bool(
        getattr(args, "tree_batch_autotune", True)
    )
    config_overrides["tree_batch_structural_pad_limit"] = float(
        getattr(args, "tree_batch_structural_pad_limit", 0.5)
    )
    config_overrides["tree_batch_auto_queue_min_docs"] = int(
        getattr(args, "tree_batch_auto_queue_min_docs", 8)
    )
    config_overrides["tree_batch_auto_queue_min_fill_ratio"] = float(
        getattr(args, "tree_batch_auto_queue_min_fill_ratio", 0.5)
    )
    config_overrides["tree_eval_workers_per_mig"] = int(
        getattr(args, "tree_eval_workers_per_mig", 0)
    )
    config_overrides["gpu_runtime_data_mode"] = str(
        getattr(args, "gpu_runtime_data_mode", "resident")
    )
    config_overrides["gpu_runtime_bucket_mode"] = str(
        getattr(args, "gpu_runtime_bucket_mode", "exact_then_bucketed")
    )
    _raw_preload_splits = getattr(args, "gpu_runtime_preload_splits", ("train", "val", "test"))
    if isinstance(_raw_preload_splits, str):
        _normalized_preload_splits = tuple(
            item for item in str(_raw_preload_splits).replace(",", " ").split() if item
        )
    else:
        _normalized_preload_splits = tuple(
            str(item) for item in list(_raw_preload_splits) if str(item).strip()
        )
    config_overrides["gpu_runtime_preload_splits"] = (
        _normalized_preload_splits or ("train", "val", "test")
    )
    config_overrides["gpu_runtime_preload_targets"] = bool(
        getattr(args, "gpu_runtime_preload_targets", True)
    )
    config_overrides["gpu_runtime_workers_per_mig"] = int(
        getattr(args, "gpu_runtime_workers_per_mig", 1)
    )
    config_overrides["gpu_runtime_allow_multi_worker_screen"] = bool(
        getattr(args, "gpu_runtime_allow_multi_worker_screen", True)
    )
    config_overrides["gpu_runtime_capacity_workers_per_mig"] = int(
        getattr(args, "gpu_runtime_capacity_workers_per_mig", 2)
    )
    if str(getattr(args, "tree_stage1_artifact_dir", "")).strip():
        config_overrides["tree_stage1_artifact_dir"] = str(
            getattr(args, "tree_stage1_artifact_dir", "")
        )
    if str(getattr(args, "prepared_data_root", "")).strip():
        config_overrides["prepared_data_root"] = str(
            getattr(args, "prepared_data_root", "")
        )
    config_overrides["prepared_data_allow_create"] = bool(
        getattr(args, "prepared_data_allow_create", True)
    )
    config_overrides["diagnostic_detail_mode"] = str(
        getattr(args, "diagnostic_detail_mode", "summary")
    )
    if str(getattr(args, "posttrain_diagnostics_mode", "")).strip():
        config_overrides["posttrain_diagnostics_mode"] = str(
            getattr(args, "posttrain_diagnostics_mode", "")
        )
    if str(getattr(args, "raw_diagnostic_artifact_dir", "")).strip():
        config_overrides["raw_diagnostic_artifact_dir"] = str(
            getattr(args, "raw_diagnostic_artifact_dir", "")
        )
    config_overrides["tree_stage1_root_weight"] = float(
        getattr(args, "tree_stage1_root_weight", 0.0)
    )
    config_overrides["tree_join_bit_weight"] = float(
        getattr(args, "tree_join_bit_weight", 0.0)
    )
    if str(getattr(args, "tree_training_schedule", "")).strip():
        config_overrides["tree_training_schedule"] = str(
            getattr(args, "tree_training_schedule", "two_stage")
        )
    config_overrides["tree_stage1_epochs"] = int(getattr(args, "tree_stage1_epochs", 0))
    config_overrides["tree_stage2_epochs"] = int(getattr(args, "tree_stage2_epochs", 0))
    if str(getattr(args, "tree_task_head_mode", "")).strip():
        config_overrides["tree_task_head_mode"] = str(
            getattr(args, "tree_task_head_mode", "full_state_scalar")
        )
    if str(getattr(args, "tree_theorem_surface_mode", "")).strip():
        config_overrides["tree_theorem_surface_mode"] = str(
            getattr(args, "tree_theorem_surface_mode", "slotwise")
        )
    if str(getattr(args, "tree_theorem_count_head_mode", "")).strip():
        config_overrides["tree_theorem_count_head_mode"] = str(
            getattr(args, "tree_theorem_count_head_mode", "scalar_mse")
        )
    config_overrides["tree_theorem_count_ordinal_weight"] = float(
        getattr(args, "tree_theorem_count_ordinal_weight", 1.0)
    )
    config_overrides["tree_theorem_count_scalar_aux_weight"] = float(
        getattr(args, "tree_theorem_count_scalar_aux_weight", 0.25)
    )
    config_overrides["tree_theorem_count_threshold_balance"] = bool(
        getattr(args, "tree_theorem_count_threshold_balance", True)
    )
    config_overrides["tree_theorem_feature_dim"] = int(
        getattr(args, "tree_theorem_feature_dim", 48)
    )
    config_overrides["tree_theorem_feature_hidden_dim"] = int(
        getattr(args, "tree_theorem_feature_hidden_dim", 256)
    )
    config_overrides["tree_merge_hidden_dim"] = int(
        getattr(args, "tree_merge_hidden_dim", 0)
    )
    config_overrides["tree_theorem_score_dim"] = int(
        getattr(args, "tree_theorem_score_dim", 0)
    )
    config_overrides["tree_theorem_fiber_dim"] = int(
        getattr(args, "tree_theorem_fiber_dim", 0)
    )
    config_overrides["tree_theorem_aux_dim"] = int(
        getattr(args, "tree_theorem_aux_dim", 0)
    )
    config_overrides["tree_score_merge_mode"] = str(
        getattr(args, "tree_score_merge_mode", "gated_affine")
    )
    config_overrides["tree_phi_compose_weight"] = float(
        getattr(args, "tree_phi_compose_weight", 1.0)
    )
    config_overrides["tree_phi_contrastive_weight"] = float(
        getattr(args, "tree_phi_contrastive_weight", 0.25)
    )
    config_overrides["tree_phi_alignment_loss"] = str(
        getattr(args, "tree_phi_alignment_loss", "cosine_mse")
    )
    config_overrides["tree_c2_mode"] = str(
        getattr(args, "tree_c2_mode", "reconstruction")
    )
    _om_name = str(getattr(args, "oracle_metric_name", "")).strip()
    if _om_name:
        config_overrides["oracle_metric_name"] = _om_name
    config_overrides["oracle_same_threshold"] = float(
        getattr(args, "oracle_same_threshold", 0.0)
    )
    config_overrides["oracle_diff_threshold"] = float(
        getattr(args, "oracle_diff_threshold", 0.0)
    )
    config_overrides["theorem_feature_adapter"] = str(
        getattr(args, "theorem_feature_adapter", "markov_count_sketch")
    )
    if getattr(args, "theorem_pair_same_threshold", None) is not None:
        config_overrides["theorem_pair_same_threshold"] = float(
            args.theorem_pair_same_threshold
        )
    if getattr(args, "theorem_pair_diff_threshold", None) is not None:
        config_overrides["theorem_pair_diff_threshold"] = float(
            args.theorem_pair_diff_threshold
        )
    if str(getattr(args, "tree_summary_spec_root_mode", "")).strip():
        config_overrides["tree_summary_spec_root_mode"] = str(
            getattr(args, "tree_summary_spec_root_mode", "task_split_ablation")
        )
    if str(getattr(args, "aligned_sketch_surface", "")).strip():
        config_overrides["aligned_sketch_surface"] = str(args.aligned_sketch_surface)
    if str(getattr(args, "summary_spec_name", "")).strip():
        config_overrides["summary_spec_name"] = str(args.summary_spec_name)
        config_overrides["slot_count"] = int(getattr(args, "slot_count", 0))
        config_overrides["tree_theorem_count_dim"] = int(
            getattr(args, "tree_theorem_count_dim", 0)
        )
        config_overrides["tree_theorem_first_dim"] = int(
            getattr(args, "tree_theorem_first_dim", 0)
        )
        config_overrides["tree_theorem_last_dim"] = int(
            getattr(args, "tree_theorem_last_dim", 0)
        )
        config_overrides["leaf_label_rate"] = float(
            getattr(args, "leaf_label_rate", 1.0)
        )
    if str(getattr(args, "leaf_supervision_kind", "")).strip():
        config_overrides["leaf_supervision_kind"] = str(
            getattr(args, "leaf_supervision_kind", "full_sketch")
        )
    config_overrides["internal_supervision_kind"] = str(
        getattr(args, "internal_supervision_kind", "none")
    )
    config_overrides["internal_label_rate"] = float(
        getattr(args, "internal_label_rate", 0.0)
    )
    config_overrides["max_internal_depth"] = int(
        getattr(args, "max_internal_depth", 0)
    )
    config_overrides["leaf_exact_supervision"] = bool(
        getattr(args, "leaf_exact_supervision", False)
    )
    config_overrides["root_weight"] = float(getattr(args, "root_weight", 1.0))
    config_overrides["schedule_consistency_weight"] = float(
        getattr(args, "schedule_consistency_weight", 0.0)
    )
    config_overrides["endpoint_loss_scale"] = float(
        getattr(args, "endpoint_loss_scale", 1.0)
    )
    config_overrides["doc_sequence_train_fraction"] = float(
        args.doc_sequence_train_fraction
    )
    if int(getattr(args, "budget_total_calls", 0)) > 0:
        config_overrides["budget_total_calls"] = int(args.budget_total_calls)
    if float(getattr(args, "budget_total_calls_per_doc", 0.0)) > 0.0:
        config_overrides["budget_total_calls_per_doc"] = float(
            args.budget_total_calls_per_doc
        )
    if math.isfinite(float(getattr(args, "mass_target_per_doc", float("nan")))):
        config_overrides["mass_target_per_doc"] = float(args.mass_target_per_doc)
    config_overrides["full_doc_budget_share"] = float(
        getattr(args, "full_doc_budget_share", 1.0)
    )
    if str(getattr(args, "doc_consumption_mode", "")).strip():
        config_overrides["doc_consumption_mode"] = str(args.doc_consumption_mode)
    if str(getattr(args, "local_split_mode", "")).strip():
        config_overrides["local_split_mode"] = str(args.local_split_mode)
    if str(getattr(args, "local_allocation_policy", "")).strip():
        config_overrides["local_allocation_policy"] = str(args.local_allocation_policy)
    if str(getattr(args, "package_semantics", "")).strip():
        config_overrides["package_semantics"] = str(args.package_semantics)
    config_overrides["depth_discount_gamma"] = float(
        getattr(args, "depth_discount_gamma", 1.0)
    )
    config_spec_json_path = str(
        getattr(args, "config_spec_json_path", "") or ""
    ).strip()
    if config_spec_json_path:
        authoritative_run_config = _run_config_from_mapping(
            json.loads(Path(config_spec_json_path).read_text(encoding="utf-8"))
        )
    else:
        authoritative_run_config = _worker_run_config_for_preflight(
            args,
            config_overrides=config_overrides,
        )
    config_overrides = _config_mapping_for_run_config(authoritative_run_config)

    run_metadata = {
        "config_label": str(args.config_label),
        "tuning_stage": str(args.tuning_stage),
        "test_metrics_hidden_during_selection": bool(
            args.test_metrics_hidden_during_selection
        ),
        "study_name": str(args.study_name),
        "study_axis": str(args.study_axis),
        "axis_value": str(args.axis_value),
        "locked_tree_neural_config_label": str(args.locked_tree_neural_config_label),
        "selection_metric": str(args.selection_metric),
        "base_bundle_path": str(getattr(args, "base_bundle_path", "")),
    }
    device_context = _worker_device_context()
    runtime_preflight: Dict[str, Any] = {}
    if str(args.tuning_stage).strip() == "capacity_screen":
        runtime_preflight = _estimate_capacity_screen_worker_preflight(
            args=args,
            config=authoritative_run_config,
        )
    snapshot = {
        "job_name": str(args.job_name),
        "output_dir": str(output_dir),
        "benchmark_name": str(args.benchmark),
        "hardness_grid": str(args.hardness_grid),
        "grid_cell_ids": [str(cell) for cell in list(args.grid_cell_ids or ())],
        "seeds": [int(seed) for seed in list(args.seeds or ())],
        "train_doc_counts": [int(args.train_doc_count)],
        "baseline_families": [str(args.family)],
        "emit_confusion": False,
        "use_cuda": bool(args.use_cuda),
        "cuda_device": 0 if bool(args.use_cuda) else None,
        "torch_threads": int(args.torch_threads),
        "base_bundle_path": str(getattr(args, "base_bundle_path", "") or "").strip(),
        "config_overrides": config_overrides,
        "requested_run_config": asdict(authoritative_run_config),
        "config_spec_json_path": config_spec_json_path,
        "run_metadata": run_metadata,
        "device_context": dict(device_context),
        "runtime_preflight": dict(runtime_preflight),
        "memory_probe_jsonl": str(getattr(args, "memory_probe_jsonl", "") or "").strip(),
        "environment": {
            "cwd": str(REPO_ROOT),
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        },
    }
    snapshot_path = output_dir / "worker_invocation_snapshot.json"
    _write_worker_invocation_snapshot(snapshot, snapshot_path)
    debug_snapshot_json = str(getattr(args, "debug_snapshot_json", "") or "").strip()
    if debug_snapshot_json:
        _write_worker_invocation_snapshot(snapshot, Path(debug_snapshot_json))
    if bool(getattr(args, "debug_stop_after_snapshot", False)):
        return {
            "job_name": str(args.job_name),
            "output_dir": str(output_dir),
            "status": "snapshot_only",
            "worker_invocation_snapshot_json": str(snapshot_path),
            "debug_snapshot_json": debug_snapshot_json,
        }

    t0 = time.monotonic()
    payload = _execute_worker_invocation(snapshot)
    elapsed_s = time.monotonic() - t0
    run_dir = output_dir / "runs"
    run_paths = sorted(run_dir.glob("*.json"))
    elapsed_s_per_run = (
        float(elapsed_s / float(max(len(run_paths), 1)))
        if run_paths
        else float(elapsed_s)
    )
    for path in run_paths:
        try:
            run_payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_payload["elapsed_s"] = float(elapsed_s_per_run)
        run_payload["elapsed_s_job_total"] = float(elapsed_s)
        run_payload["job_seed_count"] = int(len(tuple(int(seed) for seed in args.seeds)))
        path.write_text(
            json.dumps(run_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    runs = list(payload.get("runs") or [])
    run = dict(runs[0] if runs else {})
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    aggregate = dict(aggregate_rows[0] if aggregate_rows else {})
    job_seeds = tuple(int(seed) for seed in args.seeds)
    return {
        "job_name": str(args.job_name),
        "family": str(args.family),
        "objective_weights_active": bool(str(args.family) in TREE_NEURAL_FAMILIES),
        "train_doc_count": int(args.train_doc_count),
        "benchmark": str(args.benchmark),
        "hardness_grid": str(args.hardness_grid),
        "grid_cell_ids": list(args.grid_cell_ids or []),
        "elapsed_s": float(elapsed_s),
        "config_label": str(args.config_label),
        "fixed_leaf_tokens": (
            None
            if args.fixed_leaf_tokens is None
            else int(args.fixed_leaf_tokens)
        ),
        "tree_leaf_fno_width": int(run.get("tree_leaf_fno_width", 0)),
        "tree_leaf_fno_n_modes": int(run.get("tree_leaf_fno_n_modes", 0)),
        "tree_leaf_fno_n_layers": int(run.get("tree_leaf_fno_n_layers", 0)),
        "tree_root_supervision_kind": str(run.get("tree_root_supervision_kind", "")),
        "tree_checkpoint_metric": str(run.get("tree_checkpoint_metric", "")),
        "tree_stage1_checkpoint_metric": str(
            run.get("tree_stage1_checkpoint_metric", "")
        ),
        "tree_stage1_eval_mode": str(run.get("tree_stage1_eval_mode", "")),
        "tree_stage1_screen_doc_limit": int(run.get("tree_stage1_screen_doc_limit", 0)),
        "tree_stage1_final_exact_doc_limit": int(
            run.get("tree_stage1_final_exact_doc_limit", 0)
        ),
        "tree_stage1_artifact_dir": str(run.get("tree_stage1_artifact_dir", "")),
        "tree_stage1_root_weight": float(run.get("tree_stage1_root_weight", 0.0)),
        "tree_training_schedule": str(run.get("tree_training_schedule", "")),
        "tree_stage1_epochs": int(run.get("tree_stage1_epochs", 0)),
        "tree_stage2_epochs": int(run.get("tree_stage2_epochs", 0)),
        "tree_task_head_mode": str(run.get("tree_task_head_mode", "")),
        "tree_theorem_surface_mode": str(run.get("tree_theorem_surface_mode", "")),
        "tree_c2_mode": str(run.get("tree_c2_mode", "")),
        "tree_summary_spec_root_mode": str(
            run.get("tree_summary_spec_root_mode", "")
        ),
        "tree_join_bit_weight": float(run.get("tree_join_bit_weight", 0.0)),
        "aligned_sketch_surface": str(run.get("aligned_sketch_surface", "")),
        "summary_spec_name": str(run.get("summary_spec_name", "")),
        "slot_count": int(run.get("slot_count", 0)),
        "internal_supervision_kind": str(
            run.get("internal_supervision_kind", "none")
        ),
        "internal_label_rate": float(run.get("internal_label_rate", 0.0)),
        "leaf_exact_supervision": bool(run.get("leaf_exact_supervision", False)),
        "leaf_supervision_kind": str(run.get("leaf_supervision_kind", "")),
        "leaf_label_rate": float(run.get("leaf_label_rate", 1.0)),
        "tree_aux_doc_sequence_fraction": float(
            run.get("tree_aux_doc_sequence_fraction", 0.0)
        ),
        "doc_sequence_train_fraction": float(args.doc_sequence_train_fraction),
        "tuning_stage": str(args.tuning_stage),
        "test_metrics_hidden_during_selection": bool(
            args.test_metrics_hidden_during_selection
        ),
        "study_name": str(args.study_name),
        "study_axis": str(args.study_axis),
        "axis_value": str(args.axis_value),
        "locked_tree_neural_config_label": str(args.locked_tree_neural_config_label),
        "selection_metric": str(args.selection_metric),
        "budget_total_calls": int(run.get("budget_total_calls", 0)),
        "budget_total_calls_per_doc": float(
            run.get("budget_total_calls_per_doc", 0.0)
        ),
        "budget_total_calls_used": int(run.get("budget_total_calls_used", 0)),
        "budget_utilization": float(run.get("budget_utilization", float("nan"))),
        "full_doc_budget_share": float(run.get("full_doc_budget_share", 1.0)),
        "full_doc_calls_total": int(run.get("full_doc_calls_total", 0)),
        "local_calls_total": int(run.get("local_calls_total", 0)),
        "doc_consumption_mode": str(run.get("doc_consumption_mode", "")),
        "local_split_mode": str(run.get("local_split_mode", "")),
        "local_allocation_policy": str(run.get("local_allocation_policy", "")),
        "effective_full_doc_mass_total": float(
            run.get("effective_full_doc_mass_total", 0.0)
        ),
        "effective_full_doc_mass_per_doc": float(
            run.get("effective_full_doc_mass_per_doc", 0.0)
        ),
        "train_root_mae": float(run.get("train_root_mae", float("nan"))),
        "val_root_mae": float(run.get("val_root_mae", float("nan"))),
        "test_root_mae": float(run.get("test_root_mae", float("nan"))),
        "train_exact_match_rate": float(
            run.get("train_exact_match_rate", float("nan"))
        ),
        "val_exact_match_rate": float(run.get("val_exact_match_rate", float("nan"))),
        "test_exact_match_rate": float(
            run.get("test_exact_match_rate", float("nan"))
        ),
        "selection_metric_name": str(
            dict(run.get("fit_diagnostics") or {}).get("selection_metric_name", "")
        ),
        "selection_metric_value": float(
            dict(run.get("fit_diagnostics") or {}).get(
                "selection_metric_value", float("nan")
            )
        ),
        "parameterization": str(run.get("parameterization", "")),
        "optimization_root_weight": float(
            run.get("optimization_root_weight", float("nan"))
        ),
        "local_law_c1_weight": float(
            run.get("local_law_c1_weight", float("nan"))
        ),
        "local_law_c2_weight": float(
            run.get("local_law_c2_weight", float("nan"))
        ),
        "local_law_c3_weight": float(
            run.get("local_law_c3_weight", float("nan"))
        ),
        "tree_local_weighting_mode": str(
            run.get("tree_local_weighting_mode", "fixed_k_hajek")
        ),
        "tree_exact_collapse_mode": str(run.get("tree_exact_collapse_mode", "")),
        "local_loss_kind": str(run.get("local_loss_kind", "")),
        "local_sampling_design_name": str(
            run.get("local_sampling_design_name", "")
        ),
        "leaf_population_size": float(run.get("leaf_population_size", float("nan"))),
        "leaf_sample_size": float(run.get("leaf_sample_size", float("nan"))),
        "leaf_effective_propensity": float(
            run.get("leaf_effective_propensity", float("nan"))
        ),
        "merge_population_size": float(
            run.get("merge_population_size", float("nan"))
        ),
        "merge_sample_size": float(run.get("merge_sample_size", float("nan"))),
        "merge_effective_propensity": float(
            run.get("merge_effective_propensity", float("nan"))
        ),
        "local_objective_audit": dict(run.get("local_objective_audit", {}) or {}),
        "c2_metric_kind": str(run.get("c2_metric_kind", "")),
        "comparison_semantics": str(run.get("comparison_semantics", "")),
        "run_intent_hash": str(run.get("run_intent_hash", "")),
        "run_intent_validation_status": str(
            run.get("run_intent_validation_status", "")
        ),
        "exact_sketch_diagnostics": (
            dict(run.get("exact_sketch_diagnostics") or {})
            if isinstance(run.get("exact_sketch_diagnostics"), Mapping)
            else {}
        ),
        "exact_sketch_markov_sufficiency_gap_score": float(
            run.get("exact_sketch_markov_sufficiency_gap_score", float("nan"))
        ),
        "exact_projected_root_mae": float(
            run.get("exact_projected_root_mae", float("nan"))
        ),
        "test_exact_projected_root_mae": float(
            run.get("test_exact_projected_root_mae", float("nan"))
        ),
        "certified_projected_root_mae": float(
            run.get("certified_projected_root_mae", float("nan"))
        ),
        "test_certified_projected_root_mae": float(
            run.get("test_certified_projected_root_mae", float("nan"))
        ),
        "root_mae_predicted_counts_predicted_endpoints": float(
            run.get("root_mae_predicted_counts_predicted_endpoints", float("nan"))
        ),
        "test_root_mae_predicted_counts_predicted_endpoints": float(
            run.get("test_root_mae_predicted_counts_predicted_endpoints", float("nan"))
        ),
        "root_mae_oracle_counts_predicted_endpoints": float(
            run.get("root_mae_oracle_counts_predicted_endpoints", float("nan"))
        ),
        "test_root_mae_oracle_counts_predicted_endpoints": float(
            run.get("test_root_mae_oracle_counts_predicted_endpoints", float("nan"))
        ),
        "root_mae_predicted_counts_oracle_endpoints": float(
            run.get("root_mae_predicted_counts_oracle_endpoints", float("nan"))
        ),
        "test_root_mae_predicted_counts_oracle_endpoints": float(
            run.get("test_root_mae_predicted_counts_oracle_endpoints", float("nan"))
        ),
        "learned_merger_gap": float(
            run.get("learned_merger_gap", float("nan"))
        ),
        "test_learned_merger_gap": float(
            run.get("test_learned_merger_gap", float("nan"))
        ),
        "leaf_first_accuracy": float(
            run.get("leaf_first_accuracy", float("nan"))
        ),
        "test_leaf_first_accuracy": float(
            run.get("test_leaf_first_accuracy", float("nan"))
        ),
        "leaf_last_accuracy": float(
            run.get("leaf_last_accuracy", float("nan"))
        ),
        "test_leaf_last_accuracy": float(
            run.get("test_leaf_last_accuracy", float("nan"))
        ),
        "merge_first_accuracy": float(
            run.get("merge_first_accuracy", float("nan"))
        ),
        "test_merge_first_accuracy": float(
            run.get("test_merge_first_accuracy", float("nan"))
        ),
        "merge_last_accuracy": float(
            run.get("merge_last_accuracy", float("nan"))
        ),
        "test_merge_last_accuracy": float(
            run.get("test_merge_last_accuracy", float("nan"))
        ),
        "leaf_count_off_by_k_histogram": dict(
            run.get("leaf_count_off_by_k_histogram", {}) or {}
        ),
        "merge_exact_summary_match_rate_by_depth": dict(
            run.get("merge_exact_summary_match_rate_by_depth", {}) or {}
        ),
        "aggregate_row": aggregate,
        "job_seeds": [int(seed) for seed in job_seeds],
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "summary.json"),
        "worker_invocation_snapshot_json": str(snapshot_path),
        "debug_snapshot_json": debug_snapshot_json,
        "memory_probe_jsonl": str(snapshot.get("memory_probe_jsonl", "") or ""),
    }


def _job_seeds_for_family(args: argparse.Namespace, family: str) -> List[int]:
    if bool(args.repeat_closed_form_controls) or str(family) not in CLOSED_FORM_CONTROL_FAMILIES:
        return [int(seed) for seed in args.seeds]
    return [int(args.seeds[0])]


def _job_priority(job: _JobSpec, *, family_order: Mapping[str, int]) -> tuple[int, int, int, str, int]:
    is_control = 1 if str(job.family) in CLOSED_FORM_CONTROL_FAMILIES else 0
    min_seed = int(min(job.seeds)) if job.seeds else 0
    return (
        is_control,
        -int(job.train_doc_count),
        family_order.get(str(job.family), 0),
        str(job.config.label),
        min_seed,
    )


def _job_completion_keys(
    job: _JobSpec,
) -> Set[
    Tuple[
        str,
        str,
        int,
        int,
        str,
        str,
        int,
        str,
        str,
        str,
        int,
        float,
        float,
        str,
        str,
        str,
    ]
]:
    scope_ids = tuple(str(cell) for cell in job.grid_cell_ids) or (str(job.benchmark),)
    leaf_token_key = (
        0
        if job.config.fixed_leaf_tokens is None
        else int(job.config.fixed_leaf_tokens)
    )
    return {
        (
            str(scope_id),
            str(job.family),
            int(job.train_doc_count),
            int(seed),
            str(job.config.label),
            str(job.tuning_stage),
            int(leaf_token_key),
            str(job.study_name),
            str(job.study_axis),
            str(job.axis_value),
            int(job.budget_total_calls),
            float(job.budget_total_calls_per_doc),
            float(job.full_doc_budget_share),
            str(job.doc_consumption_mode),
            str(job.local_split_mode),
            str(job.local_allocation_policy),
        )
        for scope_id in scope_ids
        for seed in job.seeds
    }


def _load_completed_run_keys(
    output_root: Path,
) -> Set[
    Tuple[
        str,
        str,
        int,
        int,
        str,
        str,
        int,
        str,
        str,
        str,
        int,
        float,
        float,
        str,
        str,
        str,
    ]
]:
    completed: Set[
        Tuple[
            str,
            str,
            int,
            int,
            str,
            str,
            int,
            str,
            str,
            str,
            int,
            float,
            float,
            str,
            str,
            str,
        ]
    ] = set()
    for path in sorted(Path(output_root).glob("**/runs/*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scope_id = str(run.get("cell_id") or run.get("benchmark") or "").strip()
        family = str(run.get("baseline_family") or "").strip()
        if not scope_id or not family:
            continue
        try:
            train_doc_count = int(run.get("train_doc_count"))
            seed = int(run.get("seed"))
        except (TypeError, ValueError):
            continue
        stored_leaf_tokens = (
            int(run.get("fixed_leaf_tokens"))
            if run.get("fixed_leaf_tokens") not in {"", None}
            else 0
        )
        leaf_token_keys = {int(stored_leaf_tokens)}
        if int(stored_leaf_tokens) > 0:
            leaf_token_keys.add(0)
        for leaf_token_key in leaf_token_keys:
            raw_budget_total_calls = run.get("budget_total_calls", 0)
            raw_budget_calls_per_doc = run.get("budget_total_calls_per_doc", 0.0)
            raw_full_doc_budget_share = run.get("full_doc_budget_share", 1.0)
            completed.add(
                (
                    scope_id,
                    family,
                    train_doc_count,
                    seed,
                    str(run.get("config_label", "")),
                    str(run.get("tuning_stage", "")),
                    int(leaf_token_key),
                    str(run.get("study_name", "")),
                    str(run.get("study_axis", "")),
                    str(run.get("axis_value", "")),
                    (
                        0
                        if raw_budget_total_calls in {"", None}
                        else int(raw_budget_total_calls)
                    ),
                    (
                        0.0
                        if raw_budget_calls_per_doc in {"", None}
                        else float(raw_budget_calls_per_doc)
                    ),
                    (
                        1.0
                        if raw_full_doc_budget_share in {"", None}
                        else float(raw_full_doc_budget_share)
                    ),
                    str(run.get("doc_consumption_mode", "")),
                    str(run.get("local_split_mode", "")),
                    str(run.get("local_allocation_policy", "")),
                )
            )
    return completed


def _build_jobs_for_configs(
    *,
    families: Sequence[str],
    train_doc_counts: Sequence[int],
    benchmark: str,
    hardness_grid: str,
    grid_cell_ids: Sequence[str],
    seeds: Sequence[int],
    job_granularity: str,
    repeat_closed_form_controls: bool,
    configs: Sequence[_RunConfigSpec],
    tuning_stage: str = "",
    test_metrics_hidden_during_selection: bool = False,
    study_name: str = "",
    study_axis: str = "",
    axis_value: str = "",
    locked_tree_neural_config_label: str = "",
    selection_metric: str = "",
    budget_total_calls: int = 0,
    budget_total_calls_per_doc: float = 0.0,
    mass_target_per_doc: float = float("nan"),
    full_doc_budget_share: float = 1.0,
    doc_consumption_mode: str = "",
    local_split_mode: str = "",
    local_allocation_policy: str = "",
    package_semantics: str = "",
    depth_discount_gamma: float = 1.0,
) -> List[_JobSpec]:
    jobs: List[_JobSpec] = []
    family_list = [str(family) for family in families]
    family_order = {
        str(family): idx for idx, family in enumerate(family_list)
    }
    seed_values = [int(seed) for seed in seeds]
    for config in configs:
        effective_config = _with_run_intent_overrides(
            config,
            budget_total_calls=int(budget_total_calls),
            budget_total_calls_per_doc=float(budget_total_calls_per_doc),
            mass_target_per_doc=float(mass_target_per_doc),
            full_doc_budget_share=float(full_doc_budget_share),
            doc_consumption_mode=str(doc_consumption_mode),
            local_split_mode=str(local_split_mode),
            local_allocation_policy=str(local_allocation_policy),
            package_semantics=str(package_semantics),
            depth_discount_gamma=float(depth_discount_gamma),
        )
        for train_doc_count in [int(value) for value in train_doc_counts]:
            for family in family_list:
                if bool(repeat_closed_form_controls) or str(family) not in CLOSED_FORM_CONTROL_FAMILIES:
                    job_seeds = list(seed_values)
                else:
                    job_seeds = [int(seed_values[0])]
                if str(job_granularity) == "family_train_seed":
                    for seed in job_seeds:
                        jobs.append(
                            _JobSpec(
                                family=str(family),
                                train_doc_count=int(train_doc_count),
                                benchmark=str(benchmark),
                                hardness_grid=str(hardness_grid),
                                grid_cell_ids=tuple(str(cell) for cell in grid_cell_ids),
                                seeds=(int(seed),),
                                config=effective_config,
                                tuning_stage=str(tuning_stage),
                                test_metrics_hidden_during_selection=bool(
                                    test_metrics_hidden_during_selection
                                ),
                                study_name=str(study_name),
                                study_axis=str(study_axis),
                                axis_value=str(axis_value),
                                locked_tree_neural_config_label=str(
                                    locked_tree_neural_config_label
                                ),
                                selection_metric=str(selection_metric),
                            )
                        )
                    continue
                jobs.append(
                    _JobSpec(
                        family=str(family),
                        train_doc_count=int(train_doc_count),
                        benchmark=str(benchmark),
                        hardness_grid=str(hardness_grid),
                        grid_cell_ids=tuple(str(cell) for cell in grid_cell_ids),
                        seeds=tuple(int(seed) for seed in job_seeds),
                        config=effective_config,
                        tuning_stage=str(tuning_stage),
                        test_metrics_hidden_during_selection=bool(
                            test_metrics_hidden_during_selection
                        ),
                        study_name=str(study_name),
                        study_axis=str(study_axis),
                        axis_value=str(axis_value),
                        locked_tree_neural_config_label=str(
                            locked_tree_neural_config_label
                        ),
                        selection_metric=str(selection_metric),
                    )
                )
    return sorted(jobs, key=lambda job: _job_priority(job, family_order=family_order))


def _build_jobs(args: argparse.Namespace) -> List[_JobSpec]:
    return _build_jobs_for_configs(
        families=[str(family) for family in args.families],
        train_doc_counts=[int(value) for value in args.train_doc_counts],
        benchmark=str(args.benchmark),
        hardness_grid=str(args.hardness_grid),
        grid_cell_ids=tuple(str(cell) for cell in args.grid_cell_ids),
        seeds=[int(seed) for seed in args.seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=bool(args.repeat_closed_form_controls),
        configs=(_default_run_config(args),),
    )


def build_budget_frontier_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    tree_config, capacity_root_value = _budget_frontier_tree_config(args)
    reference_config = _default_run_config(args, label="budget_reference_default")
    jobs: List[_JobSpec] = []
    train_doc_count = int(args.train_doc_count)
    hardness_grid = str(getattr(args, "hardness_grid", "") or "").strip()
    grid_cell_ids = tuple(
        str(cell).strip()
        for cell in getattr(args, "grid_cell_ids", ())
        if str(cell).strip()
    )
    budget_axis = "budget_total_calls_per_doc__full_doc_budget_share"
    tree_families = [str(family) for family in args.tree_families]
    reference_families = [str(family) for family in args.reference_families]
    local_allocation_policy = str(args.local_allocation_policy)

    for budget_per_doc in [float(value) for value in args.budget_calls_per_doc]:
        budget_total_calls = int(round(float(budget_per_doc) * float(train_doc_count)))
        for full_doc_share in [float(value) for value in args.full_doc_budget_shares]:
            if abs(float(full_doc_share) - 1.0) <= 1e-12:
                axis_value = _budget_frontier_axis_value(
                    budget_total_calls_per_doc=float(budget_per_doc),
                    full_doc_budget_share=float(full_doc_share),
                    doc_consumption_mode="full_doc_only",
                    local_split_mode="inactive_for_family",
                )
                jobs.extend(
                    _build_jobs_for_configs(
                        families=reference_families,
                        train_doc_counts=(train_doc_count,),
                        benchmark=str(args.benchmark),
                        hardness_grid=hardness_grid,
                        grid_cell_ids=grid_cell_ids,
                        seeds=[int(seed) for seed in args.seeds],
                        job_granularity=str(args.job_granularity),
                        repeat_closed_form_controls=True,
                        configs=(reference_config,),
                        study_name=ORACLE_BUDGET_STUDY_NAME,
                        study_axis=budget_axis,
                        axis_value=axis_value,
                        selection_metric="val_root_mae_mean",
                        budget_total_calls=int(budget_total_calls),
                        budget_total_calls_per_doc=float(budget_per_doc),
                        full_doc_budget_share=float(full_doc_share),
                        doc_consumption_mode="full_doc_only",
                        local_split_mode="inactive_for_family",
                        local_allocation_policy=str(local_allocation_policy),
                    )
                )

            doc_modes = [str(mode) for mode in args.doc_consumption_modes]
            if float(full_doc_share) <= 0.0:
                doc_modes = ["root_only"]
            local_split_modes = [str(mode) for mode in args.local_split_modes]
            if abs(float(full_doc_share) - 1.0) <= 1e-12:
                local_split_modes = ["balanced"]

            for doc_mode in doc_modes:
                for local_split_mode in local_split_modes:
                    axis_value = _budget_frontier_axis_value(
                        budget_total_calls_per_doc=float(budget_per_doc),
                        full_doc_budget_share=float(full_doc_share),
                        doc_consumption_mode=str(doc_mode),
                        local_split_mode=str(local_split_mode),
                    )
                    jobs.extend(
                        _build_jobs_for_configs(
                            families=tree_families,
                            train_doc_counts=(train_doc_count,),
                            benchmark=str(args.benchmark),
                            hardness_grid=hardness_grid,
                            grid_cell_ids=grid_cell_ids,
                            seeds=[int(seed) for seed in args.seeds],
                            job_granularity=str(args.job_granularity),
                            repeat_closed_form_controls=True,
                            configs=(tree_config,),
                            study_name=ORACLE_BUDGET_STUDY_NAME,
                            study_axis=budget_axis,
                            axis_value=axis_value,
                            selection_metric="val_root_mae_mean",
                            budget_total_calls=int(budget_total_calls),
                            budget_total_calls_per_doc=float(budget_per_doc),
                            full_doc_budget_share=float(full_doc_share),
                            doc_consumption_mode=str(doc_mode),
                            local_split_mode=str(local_split_mode),
                            local_allocation_policy=str(local_allocation_policy),
                        )
                    )
    return {
        "output_root": output_root,
        "jobs": jobs,
        "tree_config": tree_config,
        "reference_config": reference_config,
        "capacity_root": capacity_root_value,
        "manifest_payload": {
            "mode": "budget_frontier",
            "study_name": ORACLE_BUDGET_STUDY_NAME,
            "benchmark": str(args.benchmark),
            "hardness_grid": hardness_grid,
            "grid_cell_ids": [str(cell) for cell in grid_cell_ids],
            "train_doc_count": int(train_doc_count),
            "budget_calls_per_doc": [float(value) for value in args.budget_calls_per_doc],
            "full_doc_budget_shares": [
                float(value) for value in args.full_doc_budget_shares
            ],
            "doc_consumption_modes": [str(mode) for mode in args.doc_consumption_modes],
            "local_split_modes": [str(mode) for mode in args.local_split_modes],
            "local_allocation_policy": str(local_allocation_policy),
            "tree_families": tree_families,
            "reference_families": reference_families,
            "budget_tree_config_mode": str(args.budget_tree_config_mode),
            "capacity_root": capacity_root_value,
            "tree_config": asdict(tree_config),
            "reference_config": asdict(reference_config),
            "jobs": [asdict(job) for job in jobs],
        },
    }


def finalize_budget_frontier_output(output_root: Path) -> Dict[str, Any]:
    payload = _write_summary_outputs(output_root)
    report_script = REPO_ROOT / "scripts" / "report_tree_oracle_budget_frontier_pdf.py"
    report_pdf = output_root / "tree_oracle_budget_frontier_report.pdf"
    summary_json = Path(str(payload.get("summary_json", output_root / "summary.json")))
    if report_script.exists() and summary_json.exists():
        subprocess.run(
            [
                sys.executable,
                str(report_script),
                "--summary-json",
                str(summary_json),
                "--output-pdf",
                str(report_pdf),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
    return {
        "output_root": str(output_root),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "tree_oracle_budget_frontier_summary_json": str(
            output_root / "tree_oracle_budget_frontier_summary.json"
        ),
        "tree_oracle_budget_frontier_summary_md": str(
            output_root / "tree_oracle_budget_frontier_summary.md"
        ),
        "tree_oracle_budget_frontier_report_pdf": str(report_pdf),
    }


def build_parity_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    parity_tree_families = _parse_name_list(
        getattr(args, "tree_families", None),
        PARITY_TREE_FAMILIES,
    )
    parity_fno_families = _parse_name_list(
        getattr(args, "fno_families", None),
        PARITY_FNO_FAMILIES,
    )
    parity_comparison_families = [
        *parity_fno_families,
        *parity_tree_families,
    ]
    if str(getattr(args, "capacity_root", "")).strip():
        locked_capacity_root = Path(str(args.capacity_root))
        locked_capacity_config, capacity_summary = _locked_tree_neural_config_from_capacity_root(
            locked_capacity_root
        )
        parity_tree_base = _parity_tree_config_from_base(
            locked_capacity_config,
            config_label=FAIR_FNO_PARITY_CONFIG_LABEL,
            doc_sequence_train_fraction=0.0,
        )
        capacity_root_value = str(locked_capacity_root)
    else:
        capacity_summary = {}
        parity_tree_base = _fair_fno_parity_tree_config(args)
        capacity_root_value = ""
    parity_tree_config = _parity_tree_config_from_base(
        parity_tree_base,
        config_label=FAIR_FNO_PARITY_CONFIG_LABEL,
        doc_sequence_train_fraction=0.0,
    )
    reference_fno_config = _default_run_config(args, label="parity_reference")

    gate_root = output_root / "gate"
    gate_jobs = [
        *_build_jobs_for_configs(
            families=parity_fno_families,
            train_doc_counts=(int(args.gate_train_doc_count),),
            benchmark=str(args.benchmark),
            hardness_grid="",
            grid_cell_ids=(),
            seeds=[int(seed) for seed in args.seeds],
            job_granularity=str(args.job_granularity),
            repeat_closed_form_controls=True,
            configs=(reference_fno_config,),
        ),
        *_build_jobs_for_configs(
            families=parity_tree_families,
            train_doc_counts=(int(args.gate_train_doc_count),),
            benchmark=str(args.benchmark),
            hardness_grid="",
            grid_cell_ids=(),
            seeds=[int(seed) for seed in args.seeds],
            job_granularity=str(args.job_granularity),
            repeat_closed_form_controls=True,
            configs=(parity_tree_config,),
        ),
    ]
    upper_bound_configs = [
        _parity_tree_config_from_base(
            parity_tree_base,
            config_label=(
                f"{FAIR_FNO_PARITY_CONFIG_LABEL}_aux{_format_float_label(float(aux_fraction) * 100.0)}"
            ),
            doc_sequence_train_fraction=float(aux_fraction),
        )
        for aux_fraction in [float(value) for value in args.upper_bound_aux_fractions]
    ]
    upper_bound_root = output_root / "upper_bound"
    upper_bound_jobs = _build_jobs_for_configs(
        families=parity_tree_families,
        train_doc_counts=(int(args.gate_train_doc_count),),
        benchmark=str(args.benchmark),
        hardness_grid="",
        grid_cell_ids=(),
        seeds=[int(seed) for seed in args.seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=True,
        configs=tuple(upper_bound_configs),
        tuning_stage="upper_bound",
        test_metrics_hidden_during_selection=False,
    ) if bool(args.run_aux_upper_bound) else []
    backfill_root = output_root / "scale_backfill"
    backfill_jobs = [
        *_build_jobs_for_configs(
            families=parity_fno_families,
            train_doc_counts=[int(value) for value in args.scale_train_doc_counts],
            benchmark=str(args.benchmark),
            hardness_grid="",
            grid_cell_ids=(),
            seeds=[int(seed) for seed in args.seeds],
            job_granularity=str(args.job_granularity),
            repeat_closed_form_controls=True,
            configs=(reference_fno_config,),
        ),
        *_build_jobs_for_configs(
            families=parity_tree_families,
            train_doc_counts=[int(value) for value in args.scale_train_doc_counts],
            benchmark=str(args.benchmark),
            hardness_grid="",
            grid_cell_ids=(),
            seeds=[int(seed) for seed in args.seeds],
            job_granularity=str(args.job_granularity),
            repeat_closed_form_controls=True,
            configs=(parity_tree_config,),
        ),
    ] if bool(args.backfill_on_success) else []
    return {
        "output_root": output_root,
        "gate_root": gate_root,
        "gate_jobs": gate_jobs,
        "upper_bound_root": upper_bound_root,
        "upper_bound_jobs": upper_bound_jobs,
        "backfill_root": backfill_root,
        "backfill_jobs": backfill_jobs,
        "parity_tree_config": parity_tree_config,
        "reference_fno_config": reference_fno_config,
        "parity_tree_families": list(parity_tree_families),
        "parity_fno_families": list(parity_fno_families),
        "parity_comparison_families": list(parity_comparison_families),
        "capacity_root": capacity_root_value,
        "capacity_summary": capacity_summary,
        "scale_curve_backfilled": bool(args.backfill_on_success),
        "upper_bound_configs": upper_bound_configs,
    }


def finalize_parity_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    gate_failed_jobs: int = 0,
    upper_bound_failed_jobs: int = 0,
    backfill_failed_jobs: int = 0,
    parity_tree_config: _RunConfigSpec,
    reference_fno_config: _RunConfigSpec,
    parity_tree_families: Sequence[str],
    parity_fno_families: Sequence[str],
    parity_comparison_families: Sequence[str],
    capacity_root_value: str,
) -> Dict[str, Any]:
    final_payload = _write_summary_outputs(output_root)
    final_parity_summary = dict(final_payload.get("tree_fno_fair_parity_summary") or {})
    summary = {
        "benchmark": str(args.benchmark),
        "gate_train_doc_count": int(args.gate_train_doc_count),
        "scale_train_doc_counts": [int(value) for value in args.scale_train_doc_counts],
        "families": list(parity_comparison_families),
        "parity_tree_families": list(parity_tree_families),
        "parity_fno_families": list(parity_fno_families),
        "parity_config_label": str(parity_tree_config.label),
        "parity_tree_config": asdict(parity_tree_config),
        "capacity_root": capacity_root_value,
        "reference_fno_config": asdict(reference_fno_config),
        "gate_summary_json": str(output_root / "gate" / "summary.json"),
        "final_summary_json": str(output_root / "summary.json"),
        "final_summary_md": str(output_root / "summary.md"),
        "scale_curve_backfilled": bool(args.backfill_on_success),
        "scale_curve_backfill_policy": "always_when_enabled",
        "parity_summary": final_parity_summary,
        "gate_failed_jobs": int(gate_failed_jobs),
        "upper_bound_failed_jobs": int(upper_bound_failed_jobs),
        "backfill_failed_jobs": int(backfill_failed_jobs),
    }
    summary_json = output_root / "fair_parity_run_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md = output_root / "fair_parity_run_summary.md"
    summary_md.write_text(
        _render_parity_summary_markdown(summary),
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "fair_parity_run_summary_json": str(summary_json),
        "fair_parity_run_summary_md": str(summary_md),
        "primary_success_met": bool(
            final_parity_summary.get("primary_success_met", False)
        ),
        "secondary_success_met": bool(
            final_parity_summary.get("secondary_success_met", False)
        ),
        "tree_fno_upper_bound_summary_json": str(
            output_root / "tree_fno_upper_bound_summary.json"
        ),
        "tree_fno_upper_bound_summary_md": str(
            output_root / "tree_fno_upper_bound_summary.md"
        ),
        "scale_curve_backfilled": bool(args.backfill_on_success),
        "failed_jobs": int(
            int(gate_failed_jobs)
            + int(upper_bound_failed_jobs)
            + int(backfill_failed_jobs)
        ),
    }


def build_capacity_screen_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    locked_configs = _capacity_grid(args)
    capacity_axis_metadata = _resolved_capacity_axis_metadata(args)
    screen_configs = [
        _apply_capacity_screen_runtime_overrides(config, args=args)
        for config in locked_configs
    ]
    config_by_label = {str(config.label): config for config in locked_configs}
    effective_policy = _capacity_screen_effective_policy(
        args=args,
        screen_configs=screen_configs,
    )
    screen_worker_slots = list(getattr(args, "mig_uuids_resolved", ()) or ())
    if not screen_worker_slots:
        screen_worker_slots = (
            _parse_mig_uuids(args.mig_uuids)
            if str(getattr(args, "mig_uuids", "")).strip()
            else _discover_mig_uuids()
        )
    if bool(effective_policy.get("effective_gpu_runtime_allow_multi_worker_screen", False)):
        worker_multiplier = max(
            1,
            int(
                effective_policy.get(
                    "effective_gpu_runtime_capacity_workers_per_mig",
                    1,
                )
                or 1
            ),
        )
        if worker_multiplier > 1:
            screen_worker_slots = [
                token
                for token in screen_worker_slots
                for _ in range(int(worker_multiplier))
            ]
    try:
        mig_layout = _discover_mig_layout()
    except Exception:
        mig_layout = []
    layout_by_uuid = _mig_layout_by_uuid(mig_layout)
    ordered_screen_worker_slots = _apply_screen_device_order(
        screen_worker_slots,
        layout_by_uuid=layout_by_uuid,
        order_mode=str(
            effective_policy.get("effective_screen_device_order", "input")
            or "input"
        ),
    )
    constrained_screen_worker_slots = _limit_devices_per_physical_gpu(
        ordered_screen_worker_slots,
        layout_by_uuid=layout_by_uuid,
        max_per_physical_gpu=int(
            effective_policy.get(
                "effective_screen_max_concurrent_per_physical_gpu",
                0,
            )
            or 0
        ),
    )
    screen_root = output_root / "screen"
    screen_jobs = _build_jobs_for_configs(
        families=(str(args.priority_family),),
        train_doc_counts=(int(args.train_doc_count),),
        benchmark=str(args.benchmark),
        hardness_grid="",
        grid_cell_ids=(),
        seeds=[int(seed) for seed in args.screen_seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=True,
        configs=screen_configs,
        tuning_stage="capacity_screen",
        test_metrics_hidden_during_selection=True,
    )
    screen_jobs = _reorder_capacity_screen_jobs(
        screen_jobs,
        strong_guard=bool(effective_policy.get("strong_guard_enabled", False)),
    )
    screen_preflight = _capacity_screen_preflight(
        args=args,
        screen_jobs=screen_jobs,
        raw_screen_worker_slots=screen_worker_slots,
        ordered_screen_worker_slots=ordered_screen_worker_slots,
        active_screen_worker_slots=constrained_screen_worker_slots,
        screen_configs=screen_configs,
        mig_layout=mig_layout,
        effective_policy=effective_policy,
    )
    benchmark_payload = _resolved_benchmark_payload(str(args.benchmark))
    return {
        "output_root": output_root,
        "screen_root": screen_root,
        "screen_jobs": screen_jobs,
        "screen_worker_slots": constrained_screen_worker_slots,
        "screen_allowed_devices": tuple(
            str(token) for token in constrained_screen_worker_slots
        ),
        "config_by_label": config_by_label,
        "effective_policy": dict(effective_policy),
        "screen_preflight": screen_preflight,
        "screen_manifest_payload": {
            "mode": "capacity_screen",
            **benchmark_payload,
            "train_doc_count": int(args.train_doc_count),
            "priority_family": str(args.priority_family),
            "screen_seeds": [int(seed) for seed in args.screen_seeds],
            "selection_metric": "val_root_mae_mean",
            "test_metrics_hidden_during_selection": True,
            **capacity_axis_metadata,
            "screen_device_order": str(
                getattr(args, "screen_device_order", "input")
            ),
            "screen_max_concurrent_per_physical_gpu": int(
                getattr(args, "screen_max_concurrent_per_physical_gpu", 0) or 0
            ),
            "effective_screen_device_order": str(
                effective_policy.get("effective_screen_device_order", "input")
                or "input"
            ),
            "effective_screen_max_concurrent_per_physical_gpu": int(
                effective_policy.get(
                    "effective_screen_max_concurrent_per_physical_gpu",
                    0,
                )
                or 0
            ),
            "auto_safe_applied": bool(
                effective_policy.get("auto_safe_applied", False)
            ),
            "screen_runtime_overrides": _capacity_screen_runtime_override_values(args),
            "screen_job_ordering_strategy": (
                "seed_then_descending_capacity_complexity"
                if bool(effective_policy.get("strong_guard_enabled", False))
                else "default_priority"
            ),
            "gpu_runtime_screen_workers_per_mig": int(
                max(
                    1,
                    int(
                        effective_policy.get(
                            "effective_gpu_runtime_capacity_workers_per_mig",
                            1,
                        )
                        or 1
                    ),
                )
            ),
            "screen_preflight": dict(screen_preflight),
            "jobs": [asdict(job) for job in screen_jobs],
        },
    }


def finalize_capacity_screen_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    screen_root: Path,
    config_by_label: Mapping[str, _RunConfigSpec],
) -> Dict[str, Any]:
    benchmark_payload = _resolved_benchmark_payload(str(args.benchmark))
    screen_payload = _load_or_write_summary_outputs(screen_root)
    screen_rankings = _rank_config_rows(
        screen_payload,
        baseline_family=str(args.priority_family),
        tuning_stage="capacity_screen",
        train_doc_count=int(args.train_doc_count),
        metric_key="val_root_mae_mean",
    )
    if not screen_rankings:
        raise RuntimeError("capacity screen stage produced no ranked configs")
    top_rankings = screen_rankings[: max(int(args.top_k), 1)]
    locked_configs = [
        config_by_label[str(row.get("config_label", ""))]
        for row in top_rankings
        if str(row.get("config_label", "")) in config_by_label
    ]
    if not locked_configs:
        raise RuntimeError("capacity screen stage produced no lockable configs")
    capacity_axis_metadata = _resolved_capacity_axis_metadata(args)
    screen_summary = {
        **benchmark_payload,
        "train_doc_count": int(args.train_doc_count),
        "priority_family": str(args.priority_family),
        "selection_metric": "val_root_mae_mean",
        "test_metrics_hidden_during_selection": True,
        **capacity_axis_metadata,
        "screen_summary_json": str(screen_root / "summary.json"),
        "screen_summary_md": str(screen_root / "summary.md"),
        "screen_rankings": screen_rankings,
        "top_config_specs": {
            str(row.get("config_label", "")): asdict(
                config_by_label[str(row.get("config_label", ""))]
            )
            for row in top_rankings
            if str(row.get("config_label", "")) in config_by_label
        },
    }
    screen_summary_json = output_root / "tree_fno_capacity_screen_summary.json"
    screen_summary_json.write_text(
        json.dumps(screen_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    screen_summary_md = output_root / "tree_fno_capacity_screen_summary.md"
    screen_summary_md.write_text(
        _render_capacity_screen_summary_markdown(screen_summary),
        encoding="utf-8",
    )
    return {
        "screen_rankings": screen_rankings,
        "top_rankings": top_rankings,
        "locked_configs": locked_configs,
        "screen_summary_json": str(screen_summary_json),
        "screen_summary_md": str(screen_summary_md),
    }


def build_capacity_locked_job_bundle(
    args: argparse.Namespace,
    *,
    locked_configs: Sequence[_RunConfigSpec],
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    locked_root = output_root / "locked"
    capacity_axis_metadata = _resolved_capacity_axis_metadata(args)
    locked_jobs = _build_jobs_for_configs(
        families=(str(args.priority_family),),
        train_doc_counts=(int(args.train_doc_count),),
        benchmark=str(args.benchmark),
        hardness_grid="",
        grid_cell_ids=(),
        seeds=[int(seed) for seed in args.locked_seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=True,
        configs=list(locked_configs),
        tuning_stage="capacity_locked",
        test_metrics_hidden_during_selection=True,
    )
    benchmark_payload = _resolved_benchmark_payload(str(args.benchmark))
    return {
        "locked_root": locked_root,
        "locked_jobs": locked_jobs,
        "locked_manifest_payload": {
            "mode": "capacity_locked",
            **benchmark_payload,
            "train_doc_count": int(args.train_doc_count),
            "priority_family": str(args.priority_family),
            "locked_seeds": [int(seed) for seed in args.locked_seeds],
            "selection_metric": "val_root_mae_mean",
            "test_metrics_hidden_during_selection": True,
            **capacity_axis_metadata,
            "jobs": [asdict(job) for job in locked_jobs],
        },
    }


def finalize_capacity_locked_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    screen_root: Path,
    locked_root: Path,
    screen_rankings: Sequence[Mapping[str, Any]],
    config_by_label: Mapping[str, _RunConfigSpec],
) -> Dict[str, Any]:
    benchmark_payload = _resolved_benchmark_payload(str(args.benchmark))
    _load_or_write_summary_outputs(screen_root)
    locked_payload = _load_or_write_summary_outputs(locked_root)
    locked_rankings = _rank_config_rows(
        locked_payload,
        baseline_family=str(args.priority_family),
        tuning_stage="capacity_locked",
        train_doc_count=int(args.train_doc_count),
        metric_key="val_root_mae_mean",
    )
    if not locked_rankings:
        raise RuntimeError("capacity locked stage produced no ranked configs")
    winning = dict(locked_rankings[0])
    winning_label = str(winning.get("config_label", ""))
    capacity_axis_metadata = _resolved_capacity_axis_metadata(args)
    locked_summary = {
        **benchmark_payload,
        "train_doc_count": int(args.train_doc_count),
        "priority_family": str(args.priority_family),
        "selection_metric": "val_root_mae_mean",
        "top_k": int(max(args.top_k, 1)),
        **capacity_axis_metadata,
        "screen_summary_json": str(screen_root / "summary.json"),
        "screen_summary_md": str(screen_root / "summary.md"),
        "locked_summary_json": str(locked_root / "summary.json"),
        "locked_summary_md": str(locked_root / "summary.md"),
        "screen_rankings": list(screen_rankings),
        "locked_rankings": locked_rankings,
        "winning_config": winning,
        "winning_config_label": winning_label,
        "winning_config_spec": (
            asdict(config_by_label[winning_label]) if winning_label in config_by_label else {}
        ),
    }
    locked_summary_json = output_root / "tree_fno_capacity_locked_summary.json"
    locked_summary_json.write_text(
        json.dumps(locked_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    locked_summary_md = output_root / "tree_fno_capacity_locked_summary.md"
    locked_summary_md.write_text(
        _render_capacity_locked_summary_markdown(locked_summary),
        encoding="utf-8",
    )
    _write_summary_outputs(output_root)
    return {
        "output_root": str(output_root),
        "screen_summary_json": str(screen_root / "summary.json"),
        "locked_summary_json": str(locked_root / "summary.json"),
        "tree_fno_capacity_screen_summary_json": str(output_root / "tree_fno_capacity_screen_summary.json"),
        "tree_fno_capacity_locked_summary_json": str(locked_summary_json),
        "tree_fno_capacity_locked_summary_md": str(locked_summary_md),
        "winning_config_label": winning_label,
    }


def build_controller_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    jobs = _build_jobs(args)
    if not jobs:
        raise ValueError("No jobs were generated.")
    return {
        "output_root": output_root,
        "jobs": jobs,
        "manifest_payload": {
            "mode": "controller",
            "jobs": [asdict(job) for job in jobs],
            "seeds": [int(seed) for seed in args.seeds],
            "job_granularity": str(args.job_granularity),
            "train_doc_counts": [int(value) for value in args.train_doc_counts],
            "families": [str(family) for family in args.families],
            "benchmark": str(args.benchmark),
            "hardness_grid": str(args.hardness_grid),
            "grid_cell_ids": [str(cell) for cell in args.grid_cell_ids],
            "config": asdict(_default_run_config(args)),
        },
    }


def finalize_controller_output(output_root: Path) -> Dict[str, Any]:
    payload = _write_summary_outputs(output_root)
    return {
        "output_root": str(output_root),
        "summary_json": str(payload.get("summary_json", output_root / "summary.json")),
        "summary_md": str(payload.get("summary_md", output_root / "summary.md")),
    }


def build_exact_sanity_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    jobs: List[_JobSpec] = []
    configs_by_train_doc_count: Dict[str, Dict[str, Any]] = {}
    for train_doc_count in [int(value) for value in args.train_doc_counts]:
        configs = _exact_sanity_configs_for_train_doc_count(
            args,
            train_doc_count=int(train_doc_count),
        )
        configs_by_train_doc_count[str(int(train_doc_count))] = {
            str(config.label): asdict(config) for config in configs
        }
        for config in configs:
            jobs.extend(
                _build_jobs_for_configs(
                    families=(EXACT_SANITY_FAMILY,),
                    train_doc_counts=(int(train_doc_count),),
                    benchmark=str(args.benchmark),
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=[int(seed) for seed in args.seeds],
                    job_granularity=str(args.job_granularity),
                    repeat_closed_form_controls=True,
                    configs=(config,),
                    study_name=EXACT_SANITY_STUDY_NAME,
                    study_axis="exact_sanity_condition",
                    axis_value=str(config.label),
                    selection_metric="exact_sketch_diagnostic_only",
                )
            )
        if int(train_doc_count) <= 1024:
            jobs.extend(
                _build_jobs_for_configs(
                    families=("official_fno",),
                    train_doc_counts=(int(train_doc_count),),
                    benchmark=str(args.benchmark),
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=[int(seed) for seed in args.seeds],
                    job_granularity=str(args.job_granularity),
                    repeat_closed_form_controls=True,
                    configs=(
                        _default_run_config(
                            args,
                            label="official_fno_root_probe_reference",
                        ),
                    ),
                    study_name=EXACT_SANITY_STUDY_NAME,
                    study_axis="exact_sanity_condition",
                    axis_value="official_fno_root_probe_reference",
                    selection_metric="root_probe_reference_only",
                )
            )
    return {
        "output_root": output_root,
        "jobs": jobs,
        "manifest_payload": {
            "mode": "exact_sanity",
            "study_name": EXACT_SANITY_STUDY_NAME,
            "benchmark": str(args.benchmark),
            "train_doc_counts": [int(value) for value in args.train_doc_counts],
            "seeds": [int(seed) for seed in args.seeds],
            "job_granularity": str(args.job_granularity),
            "configs_by_train_doc_count": configs_by_train_doc_count,
            "jobs": [asdict(job) for job in jobs],
        },
    }


def finalize_exact_sanity_output(output_root: Path) -> Dict[str, Any]:
    payload = _write_summary_outputs(output_root)
    exact_summary = _tree_neural_exact_sanity_summary(dict(payload or {}))
    exact_summary_json = output_root / "tree_neural_exact_sanity_summary.json"
    exact_summary_md = output_root / "tree_neural_exact_sanity_summary.md"
    exact_summary_json.write_text(
        json.dumps(exact_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    exact_summary_md.write_text(
        _render_exact_sanity_summary_markdown(exact_summary)
        if exact_summary
        else "# Tree-Neural Exact-Sketch Sanity Summary\n\nNo exact-sanity runs found.\n",
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "tree_neural_exact_sanity_summary_json": str(exact_summary_json),
        "tree_neural_exact_sanity_summary_md": str(exact_summary_md),
    }


def _representation_metric_stats(
    runs: Sequence[Mapping[str, Any]],
    *,
    field: str | None = None,
    path: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if field is not None:
        return _finite_summary_stats(
            [dict(run).get(str(field), float("nan")) for run in runs]
        )
    if path is not None:
        return _finite_summary_stats(
            [_nested_mapping_value(run, tuple(path)) for run in runs]
        )
    return {"mean": float("nan"), "std": float("nan"), "n": 0}


def _representation_mapping_mean(
    runs: Sequence[Mapping[str, Any]],
    *,
    path: Sequence[str],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for run in runs:
        value = _nested_mapping_value(run, tuple(path))
        if not isinstance(value, Mapping):
            continue
        for key, raw_metric in dict(value).items():
            metric = float(raw_metric)
            if not np.isfinite(metric):
                continue
            key_text = str(key)
            totals[key_text] = float(totals.get(key_text, 0.0) + metric)
            counts[key_text] = int(counts.get(key_text, 0) + 1)
    ordered_keys = sorted(
        totals.keys(),
        key=lambda key: (int(key) if str(key).isdigit() else str(key)),
    )
    return {
        str(key): float(totals[str(key)] / float(max(1, counts[str(key)])))
        for key in ordered_keys
        if int(counts.get(str(key), 0)) > 0
    }


def _finite_or_inf(value: Any) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("inf")


def _finite_or_neg_inf(value: Any) -> float:
    numeric = float(value)
    return -numeric if np.isfinite(numeric) else float("inf")


def _validate_representation_theorem_metrics(
    entry: Mapping[str, Any],
) -> None:
    if str(entry.get("baseline_family", "")).strip() != REPRESENTATION_SUFFICIENCY_FAMILY:
        return
    required_fields = (
        "test_exact_sketch_markov_sufficiency_gap_score_mean",
        "test_exact_projected_root_mae_mean",
        "test_certified_projected_root_mae_mean",
        "test_root_mae_predicted_counts_predicted_endpoints_mean",
        "test_root_mae_oracle_counts_predicted_endpoints_mean",
        "test_root_mae_predicted_counts_oracle_endpoints_mean",
        "test_learned_merger_gap_mean",
        "test_leaf_first_accuracy_mean",
        "test_leaf_last_accuracy_mean",
        "test_merge_first_accuracy_mean",
        "test_merge_last_accuracy_mean",
    )
    for field_name in required_fields:
        value = float(entry.get(field_name, float("nan")))
        if not np.isfinite(value):
            raise RuntimeError(
                "representation sufficiency reducer requires finite "
                f"{field_name} for {str(entry.get('config_label', ''))}"
            )


def _representation_sufficiency_stage_entries(
    payload: Mapping[str, Any],
    *,
    tuning_stage: str,
    train_doc_count: int,
    config_by_label: Mapping[str, _RunConfigSpec],
    promotion_stage: str,
    study_name: str = REPRESENTATION_SUFFICIENCY_STUDY_NAME,
) -> List[Dict[str, Any]]:
    relevant_runs = [
        dict(run)
        for run in list(payload.get("runs") or [])
        if str(run.get("study_name", "")).strip()
        == str(study_name)
        and str(run.get("tuning_stage", "")).strip() == str(tuning_stage)
        and int(run.get("train_doc_count", 0)) == int(train_doc_count)
    ]
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for run in relevant_runs:
        family = str(run.get("baseline_family", "")).strip()
        label = str(run.get("config_label", "") or run.get("axis_value", "")).strip()
        if not family or not label:
            continue
        grouped.setdefault((family, label), []).append(run)

    entries: List[Dict[str, Any]] = []
    for (family, label), runs in sorted(grouped.items()):
        config = config_by_label.get(str(label))
        metadata = (
            _representation_sufficiency_config_metadata(
                config,
                baseline_family=str(family),
                promotion_stage=str(promotion_stage),
            )
            if config is not None
            else {
                "representation_family": "unknown",
                "representation_variant": "unknown",
                "representation_size": "unknown",
                "promotion_eligible": False,
                "control_only": False,
                "reference_only": False,
                "promotion_stage": str(promotion_stage),
                "state_dim": 0,
                "hidden_dim": 0,
                "theorem_feature_dim": 0,
                "theorem_feature_hidden_dim": 0,
                "merge_hidden_dim": 0,
                "carrier_merge_input_dim": 0,
                "slot_count": 0,
                "tree_theorem_surface_mode": "",
                "tree_theorem_count_head_mode": "",
                "tree_c2_mode": "",
                "exact_lane": False,
                "tree_theorem_score_dim": 0,
                "tree_theorem_fiber_dim": 0,
                "tree_theorem_aux_dim": 0,
            }
        )
        metrics = _representation_metrics_for_runs(runs)
        entry = {
            "baseline_family": str(family),
            "config_label": str(label),
            "train_doc_count": int(train_doc_count),
            "tuning_stage": str(tuning_stage),
            "n_runs": int(len(runs)),
            "seed_values": sorted(
                {int(run.get("seed", 0)) for run in runs if "seed" in run}
            ),
            "config_spec": asdict(config) if config is not None else {},
            "metrics": metrics,
            "val_exact_sketch_direct_mean": float(
                dict(metrics["val_exact_sketch_direct"]).get("mean", float("nan"))
            ),
            "test_exact_sketch_markov_sufficiency_gap_score_mean": float(
                dict(
                    metrics["test_exact_sketch_markov_sufficiency_gap_score"]
                ).get("mean", float("nan"))
            ),
            "test_probe_leaf_exact_summary_match_rate_mean": float(
                dict(metrics["test_probe_leaf_exact_summary_match_rate"]).get(
                    "mean", float("nan")
                )
            ),
            "test_probe_merge_exact_summary_match_rate_mean": float(
                dict(metrics["test_probe_merge_exact_summary_match_rate"]).get(
                    "mean", float("nan")
                )
            ),
            "test_probe_merge_count_mae_mean": float(
                dict(metrics["test_probe_merge_count_mae"]).get(
                    "mean", float("nan")
                )
            ),
            "test_probe_merge_first_accuracy_mean": float(
                dict(metrics["test_probe_merge_first_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_probe_merge_last_accuracy_mean": float(
                dict(metrics["test_probe_merge_last_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_merge_join_bit_accuracy_mean": float(
                dict(metrics["test_merge_join_bit_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_merge_decoded_consistency_count_mae_mean": float(
                dict(metrics["test_merge_decoded_consistency_count_mae"]).get(
                    "mean", float("nan")
                )
            ),
            "test_merge_decoded_consistency_first_accuracy_mean": float(
                dict(
                    metrics["test_merge_decoded_consistency_first_accuracy"]
                ).get("mean", float("nan"))
            ),
            "test_merge_decoded_consistency_last_accuracy_mean": float(
                dict(
                    metrics["test_merge_decoded_consistency_last_accuracy"]
                ).get("mean", float("nan"))
            ),
            "test_c2_on_range_exact_match_mean": float(
                dict(metrics["test_c2_on_range_exact_match"]).get(
                    "mean", float("nan")
                )
            ),
            "test_exact_projected_root_mae_mean": float(
                dict(metrics["test_exact_projected_root_mae"]).get(
                    "mean", float("nan")
                )
            ),
            "test_certified_projected_root_mae_mean": float(
                dict(metrics["test_certified_projected_root_mae"]).get(
                    "mean", float("nan")
                )
            ),
            "test_root_mae_predicted_counts_predicted_endpoints_mean": float(
                dict(metrics["test_root_mae_predicted_counts_predicted_endpoints"]).get(
                    "mean", float("nan")
                )
            ),
            "test_root_mae_oracle_counts_predicted_endpoints_mean": float(
                dict(metrics["test_root_mae_oracle_counts_predicted_endpoints"]).get(
                    "mean", float("nan")
                )
            ),
            "test_root_mae_predicted_counts_oracle_endpoints_mean": float(
                dict(metrics["test_root_mae_predicted_counts_oracle_endpoints"]).get(
                    "mean", float("nan")
                )
            ),
            "test_learned_merger_gap_mean": float(
                dict(metrics["test_learned_merger_gap"]).get(
                    "mean", float("nan")
                )
            ),
            "test_leaf_first_accuracy_mean": float(
                dict(metrics["test_leaf_first_accuracy"]).get("mean", float("nan"))
            ),
            "test_leaf_last_accuracy_mean": float(
                dict(metrics["test_leaf_last_accuracy"]).get("mean", float("nan"))
            ),
            "test_merge_first_accuracy_mean": float(
                dict(metrics["test_merge_first_accuracy"]).get("mean", float("nan"))
            ),
            "test_merge_last_accuracy_mean": float(
                dict(metrics["test_merge_last_accuracy"]).get("mean", float("nan"))
            ),
            "test_leaf_count_off_by_k_histogram_mean": dict(
                metrics["test_leaf_count_off_by_k_histogram"].get("mean", {}) or {}
            ),
            "test_merge_exact_summary_match_rate_by_depth_mean": dict(
                metrics["test_merge_exact_summary_match_rate_by_depth"].get(
                    "mean",
                    {},
                )
                or {}
            ),
            "test_root_mae_mean": float(
                dict(metrics["test_root_mae"]).get("mean", float("nan"))
            ),
            "test_mean_leaves_per_doc_mean": float(
                dict(metrics["test_mean_leaves_per_doc"]).get(
                    "mean", float("nan")
                )
            ),
            **metadata,
        }
        _validate_representation_theorem_metrics(entry)
        entries.append(entry)
    return entries


def _representation_run_selection_metric_value(run: Mapping[str, Any]) -> float:
    for curve_key in (
        "training_selection_metric_curve",
        "stage2_selection_metric_curve",
        "stage1_selection_metric_curve",
    ):
        curve = list(run.get(curve_key) or [])
        finite_values = [
            float(value)
            for value in curve
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        if finite_values:
            return float(min(finite_values))
    nested_value = _nested_mapping_value(
        run,
        (
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "val",
            "val_exact_sketch_direct",
        ),
    )
    if math.isfinite(float(nested_value)):
        return float(nested_value)
    return float("nan")


def _representation_metrics_for_runs(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {
        "val_exact_sketch_direct": {
            "mean": float(
                np.mean(
                    np.asarray(
                        [
                            _representation_run_selection_metric_value(run)
                            for run in runs
                            if math.isfinite(
                                _representation_run_selection_metric_value(run)
                            )
                        ],
                        dtype=np.float64,
                    )
                )
            )
            if any(
                math.isfinite(_representation_run_selection_metric_value(run))
                for run in runs
            )
            else float("nan"),
            "std": float(
                np.std(
                    np.asarray(
                        [
                            _representation_run_selection_metric_value(run)
                            for run in runs
                            if math.isfinite(
                                _representation_run_selection_metric_value(run)
                            )
                        ],
                        dtype=np.float64,
                    )
                )
            )
            if any(
                math.isfinite(_representation_run_selection_metric_value(run))
                for run in runs
            )
            else float("nan"),
        },
        "test_exact_sketch_markov_sufficiency_gap_score": _representation_metric_stats(
            runs,
            field="exact_sketch_markov_sufficiency_gap_score",
        ),
        "test_probe_leaf_exact_summary_match_rate": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "leaf",
                "probe",
                "exact_summary_match_rate",
            ),
        ),
        "test_probe_merge_exact_summary_match_rate": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "probe",
                "exact_summary_match_rate",
            ),
        ),
        "test_probe_merge_count_mae": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "probe",
                "count_mae",
            ),
        ),
        "test_probe_merge_first_accuracy": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "probe",
                "first_accuracy",
            ),
        ),
        "test_probe_merge_last_accuracy": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "probe",
                "last_accuracy",
            ),
        ),
        "test_merge_join_bit_accuracy": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "decoded_consistency",
                "merge_join_bit_accuracy",
            ),
        ),
        "test_merge_decoded_consistency_count_mae": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "decoded_consistency",
                "merge_decoded_consistency_count_mae",
            ),
        ),
        "test_merge_decoded_consistency_first_accuracy": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "decoded_consistency",
                "merge_decoded_consistency_first_accuracy",
            ),
        ),
        "test_merge_decoded_consistency_last_accuracy": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "tree_neural",
                "test",
                "merge",
                "decoded_consistency",
                "merge_decoded_consistency_last_accuracy",
            ),
        ),
        "test_c2_on_range_exact_match": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "direct_selection_metrics",
                "test",
                "c2_on_range_exact_match",
            ),
        ),
        "test_exact_projected_root_mae": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "direct_selection_metrics",
                "test",
                "exact_projected_root_mae",
            ),
        ),
        "test_certified_projected_root_mae": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "direct_selection_metrics",
                "test",
                "certified_projected_root_mae",
            ),
        ),
        "test_learned_merger_gap": _representation_metric_stats(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "direct_selection_metrics",
                "test",
                "learned_merger_gap",
            ),
        ),
        "test_root_mae": _representation_metric_stats(
            runs,
            field="test_root_mae",
        ),
        "test_mean_leaves_per_doc": _representation_metric_stats(
            runs,
            field="test_mean_leaves_per_doc",
        ),
    }
    metrics["test_root_mae_predicted_counts_predicted_endpoints"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "root_mae_predicted_counts_predicted_endpoints",
        ),
    )
    metrics["test_root_mae_oracle_counts_predicted_endpoints"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "root_mae_oracle_counts_predicted_endpoints",
        ),
    )
    metrics["test_root_mae_predicted_counts_oracle_endpoints"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "root_mae_predicted_counts_oracle_endpoints",
        ),
    )
    metrics["test_leaf_first_accuracy"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "leaf_first_accuracy",
        ),
    )
    metrics["test_leaf_last_accuracy"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "leaf_last_accuracy",
        ),
    )
    metrics["test_merge_first_accuracy"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "merge_first_accuracy",
        ),
    )
    metrics["test_merge_last_accuracy"] = _representation_metric_stats(
        runs,
        path=(
            "exact_sketch_diagnostics",
            "direct_selection_metrics",
            "test",
            "merge_last_accuracy",
        ),
    )
    metrics["test_leaf_count_off_by_k_histogram"] = {
        "mean": _representation_mapping_mean(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "direct_selection_metrics",
                "test",
                "leaf_count_off_by_k_histogram",
            ),
        )
    }
    metrics["test_merge_exact_summary_match_rate_by_depth"] = {
        "mean": _representation_mapping_mean(
            runs,
            path=(
                "exact_sketch_diagnostics",
                "direct_selection_metrics",
                "test",
                "merge_exact_summary_match_rate_by_depth",
            ),
        )
    }
    return metrics


def _representation_screen_sort_key(
    entry: Mapping[str, Any],
) -> tuple[float, float, float, float, str]:
    return (
        _finite_or_inf(entry.get("val_exact_sketch_direct_mean", float("nan"))),
        _finite_or_inf(
            entry.get("test_exact_sketch_markov_sufficiency_gap_score_mean", float("nan"))
        ),
        _finite_or_neg_inf(
            entry.get("test_probe_leaf_exact_summary_match_rate_mean", float("nan"))
        ),
        _finite_or_inf(entry.get("test_root_mae_mean", float("nan"))),
        str(entry.get("config_label", "")),
    )


def _representation_lock_sort_key(
    entry: Mapping[str, Any],
) -> tuple[float, float, float, float, str]:
    return (
        _finite_or_inf(entry.get("val_exact_sketch_direct_mean", float("nan"))),
        _finite_or_inf(
            entry.get("test_exact_sketch_markov_sufficiency_gap_score_mean", float("nan"))
        ),
        _finite_or_neg_inf(
            entry.get("test_probe_leaf_exact_summary_match_rate_mean", float("nan"))
        ),
        _finite_or_inf(entry.get("test_root_mae_mean", float("nan"))),
        str(entry.get("config_label", "")),
    )


def _write_representation_sufficiency_stage_summary(
    *,
    output_root: Path,
    name: str,
    payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    json_path = output_root / f"{name}.json"
    json_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path = output_root / f"{name}.md"
    lines = [
        f"# {str(payload.get('stage_title', name.replace('_', ' ').title()))}",
        "",
        f"- stage: `{str(payload.get('stage', ''))}`",
        f"- benchmark: `{str(payload.get('benchmark', ''))}`",
        f"- train_doc_count: `{int(payload.get('train_doc_count', 0))}`",
        f"- selection_metric: `{str(payload.get('selection_metric', ''))}`",
    ]
    if str(payload.get("winner_config_label", "")).strip():
        lines.append(
            f"- winner_config_label: `{str(payload.get('winner_config_label', ''))}`"
        )
    if list(payload.get("selected_learned_labels") or []):
        lines.append(
            f"- selected_learned_labels: `{list(payload.get('selected_learned_labels') or [])}`"
        )
    if list(payload.get("selected_control_labels") or []):
        lines.append(
            f"- selected_control_labels: `{list(payload.get('selected_control_labels') or [])}`"
        )
    if str(payload.get("official_fno_label", "")).strip():
        lines.append(
            f"- official_fno_label: `{str(payload.get('official_fno_label', ''))}`"
        )
    entries = list(payload.get("entries") or [])
    if entries:
        lines.extend(
            [
                "",
                "| config_label | family | head | exact_lane | eligible | control | val_exact | suff_gap | exact_root | count_only_root | endpoint_only_root | probe_leaf | probe_merge | c2_exact | root_mae |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for entry in entries:
            lines.append(
                "| "
                f"{str(entry.get('config_label', ''))} | "
                f"{str(entry.get('baseline_family', ''))} | "
                f"{str(entry.get('tree_theorem_count_head_mode', ''))} | "
                f"{bool(entry.get('exact_lane', False))} | "
                f"{bool(entry.get('promotion_eligible', False))} | "
                f"{bool(entry.get('control_only', False))} | "
                f"{float(entry.get('val_exact_sketch_direct_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_exact_sketch_markov_sufficiency_gap_score_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_exact_projected_root_mae_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_root_mae_predicted_counts_oracle_endpoints_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_root_mae_oracle_counts_predicted_endpoints_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_probe_leaf_exact_summary_match_rate_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_probe_merge_exact_summary_match_rate_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_c2_on_range_exact_match_mean', float('nan'))):.6g} | "
                f"{float(entry.get('test_root_mae_mean', float('nan'))):.6g} |"
            )
    md_path.write_text("\n".join([*lines, ""]), encoding="utf-8")
    return json_path, md_path


def _representation_sufficiency_jobs_for_configs(
    *,
    args: argparse.Namespace,
    configs: Sequence[_RunConfigSpec],
    train_doc_count: int,
    seeds: Sequence[int],
    tuning_stage: str,
    official_fno_label: str,
    study_name: str = REPRESENTATION_SUFFICIENCY_STUDY_NAME,
) -> List[_JobSpec]:
    jobs: List[_JobSpec] = []
    tree_configs = [
        config for config in configs if str(config.label) != str(official_fno_label)
    ]
    official_configs = [
        config for config in configs if str(config.label) == str(official_fno_label)
    ]
    if tree_configs:
        jobs.extend(
            _build_jobs_for_configs(
                families=(REPRESENTATION_SUFFICIENCY_FAMILY,),
                train_doc_counts=(int(train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="",
                grid_cell_ids=(),
                seeds=[int(seed) for seed in seeds],
                job_granularity=str(args.job_granularity),
                repeat_closed_form_controls=True,
                configs=tuple(tree_configs),
                tuning_stage=str(tuning_stage),
                study_name=str(study_name),
                study_axis="representation_config",
                selection_metric=REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
            )
        )
    if official_configs:
        jobs.extend(
            _build_jobs_for_configs(
                families=("official_fno",),
                train_doc_counts=(int(train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="",
                grid_cell_ids=(),
                seeds=[int(seed) for seed in seeds],
                job_granularity=str(args.job_granularity),
                repeat_closed_form_controls=True,
                configs=tuple(official_configs),
                tuning_stage=str(tuning_stage),
                study_name=str(study_name),
                study_axis="representation_config",
                selection_metric="official_fno_reference_only",
            )
        )
    return jobs


def build_representation_sufficiency_screen_job_bundle(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    screen_root = output_root / "screen"
    screen_specs = _representation_sufficiency_screen_config_specs(args)
    jobs = _representation_sufficiency_jobs_for_configs(
        args=args,
        configs=list(screen_specs["config_by_label"].values()),
        train_doc_count=int(args.screen_train_doc_count),
        seeds=[int(seed) for seed in args.screen_seeds],
        tuning_stage=REPRESENTATION_SUFFICIENCY_SCREEN_STAGE,
        official_fno_label=str(screen_specs["official_fno_label"]),
    )
    return {
        "output_root": output_root,
        "screen_root": screen_root,
        "screen_jobs": jobs,
        "config_by_label": dict(screen_specs["config_by_label"]),
        "config_metadata_by_label": dict(screen_specs["config_metadata_by_label"]),
        "slotwise_control_labels_by_state_dim": dict(
            screen_specs["slotwise_control_labels_by_state_dim"]
        ),
        "official_fno_label": str(screen_specs["official_fno_label"]),
        "screen_manifest_payload": {
            "mode": "representation_sufficiency_screen",
            **_resolved_benchmark_payload(str(args.benchmark)),
            "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
            "train_doc_count": int(args.screen_train_doc_count),
            "screen_seeds": [int(seed) for seed in args.screen_seeds],
            "top_k": int(args.top_k),
            "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
            "configs_by_label": {
                str(label): asdict(config)
                for label, config in screen_specs["config_by_label"].items()
            },
            "config_metadata_by_label": dict(
                screen_specs["config_metadata_by_label"]
            ),
            "slotwise_control_labels_by_state_dim": dict(
                screen_specs["slotwise_control_labels_by_state_dim"]
            ),
            "official_fno_label": str(screen_specs["official_fno_label"]),
            "jobs": [asdict(job) for job in jobs],
        },
    }


def finalize_representation_sufficiency_screen_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    screen_root: Path,
    config_by_label: Mapping[str, _RunConfigSpec],
    slotwise_control_labels_by_state_dim: Mapping[int, str],
    official_fno_label: str,
) -> Dict[str, Any]:
    payload = _load_or_write_summary_outputs(screen_root)
    entries = _representation_sufficiency_stage_entries(
        payload,
        tuning_stage=REPRESENTATION_SUFFICIENCY_SCREEN_STAGE,
        train_doc_count=int(args.screen_train_doc_count),
        config_by_label=config_by_label,
        promotion_stage="screen",
    )
    if not entries:
        raise RuntimeError("representation sufficiency screen stage produced no entries")
    eligible_entries = [
        dict(entry)
        for entry in entries
        if str(entry.get("baseline_family", "")) == REPRESENTATION_SUFFICIENCY_FAMILY
        and bool(entry.get("promotion_eligible", False))
    ]
    eligible_entries.sort(key=_representation_screen_sort_key)
    selected_learned = eligible_entries[: max(int(args.top_k), 1)]
    if not selected_learned:
        raise RuntimeError(
            "representation sufficiency screen stage produced no promotion-eligible learned configs"
        )
    selected_control_labels = sorted(
        {
            str(
                slotwise_control_labels_by_state_dim.get(
                    int(entry.get("state_dim", 0)),
                    "",
                )
            )
            for entry in selected_learned
            if int(entry.get("state_dim", 0)) in slotwise_control_labels_by_state_dim
        }
    )
    if not selected_control_labels:
        raise RuntimeError(
            "representation sufficiency screen stage could not find matched slotwise controls"
        )
    locked_labels = [
        *[str(entry.get("config_label", "")) for entry in selected_learned],
        *selected_control_labels,
        str(official_fno_label),
    ]
    locked_labels = [
        label
        for idx, label in enumerate(locked_labels)
        if label and label in config_by_label and label not in locked_labels[:idx]
    ]
    locked_configs = [config_by_label[str(label)] for label in locked_labels]
    stage_summary = {
        "stage_title": "Representation Sufficiency Screen Summary",
        "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
        "stage": "screen",
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.screen_train_doc_count),
        "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
        "screen_summary_json": str(screen_root / "summary.json"),
        "screen_summary_md": str(screen_root / "summary.md"),
        "entries": entries,
        "promotion_rankings": eligible_entries,
        "selected_learned_labels": [
            str(entry.get("config_label", "")) for entry in selected_learned
        ],
        "selected_control_labels": list(selected_control_labels),
        "official_fno_label": str(official_fno_label),
    }
    stage_json, stage_md = _write_representation_sufficiency_stage_summary(
        output_root=output_root,
        name="representation_sufficiency_screen_summary",
        payload=stage_summary,
    )
    return {
        "screen_summary": stage_summary,
        "screen_summary_json": str(stage_json),
        "screen_summary_md": str(stage_md),
        "locked_configs": locked_configs,
    }


def build_representation_sufficiency_lock_job_bundle(
    args: argparse.Namespace,
    *,
    locked_configs: Sequence[_RunConfigSpec],
    official_fno_label: str,
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    locked_root = output_root / "lock"
    jobs = _representation_sufficiency_jobs_for_configs(
        args=args,
        configs=list(locked_configs),
        train_doc_count=int(args.lock_train_doc_count),
        seeds=[int(seed) for seed in args.lock_seeds],
        tuning_stage=REPRESENTATION_SUFFICIENCY_LOCK_STAGE,
        official_fno_label=str(official_fno_label),
    )
    return {
        "locked_root": locked_root,
        "locked_jobs": jobs,
        "locked_manifest_payload": {
            "mode": "representation_sufficiency_lock",
            **_resolved_benchmark_payload(str(args.benchmark)),
            "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
            "train_doc_count": int(args.lock_train_doc_count),
            "lock_seeds": [int(seed) for seed in args.lock_seeds],
            "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
            "official_fno_label": str(official_fno_label),
            "jobs": [asdict(job) for job in jobs],
        },
    }


def finalize_representation_sufficiency_lock_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    locked_root: Path,
    config_by_label: Mapping[str, _RunConfigSpec],
    slotwise_control_labels_by_state_dim: Mapping[int, str],
    official_fno_label: str,
) -> Dict[str, Any]:
    payload = _load_or_write_summary_outputs(locked_root)
    entries = _representation_sufficiency_stage_entries(
        payload,
        tuning_stage=REPRESENTATION_SUFFICIENCY_LOCK_STAGE,
        train_doc_count=int(args.lock_train_doc_count),
        config_by_label=config_by_label,
        promotion_stage="lock",
    )
    if not entries:
        raise RuntimeError("representation sufficiency lock stage produced no entries")
    eligible_entries = [
        dict(entry)
        for entry in entries
        if str(entry.get("baseline_family", "")) == REPRESENTATION_SUFFICIENCY_FAMILY
        and bool(entry.get("promotion_eligible", False))
    ]
    eligible_entries.sort(key=_representation_lock_sort_key)
    if not eligible_entries:
        raise RuntimeError(
            "representation sufficiency lock stage produced no promotion-eligible learned configs"
        )
    winner = dict(eligible_entries[0])
    winner_label = str(winner.get("config_label", ""))
    matched_control_label = str(
        slotwise_control_labels_by_state_dim.get(int(winner.get("state_dim", 0)), "")
    )
    if not matched_control_label or matched_control_label not in config_by_label:
        raise RuntimeError(
            f"representation sufficiency lock stage could not find slotwise control for state_dim={int(winner.get('state_dim', 0))}"
        )
    promotion_labels = [
        str(winner_label),
        str(matched_control_label),
        str(official_fno_label),
    ]
    promotion_configs = [
        config_by_label[str(label)]
        for label in promotion_labels
        if str(label) in config_by_label
    ]
    stage_summary = {
        "stage_title": "Representation Sufficiency Lock Summary",
        "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
        "stage": "lock",
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.lock_train_doc_count),
        "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
        "locked_summary_json": str(locked_root / "summary.json"),
        "locked_summary_md": str(locked_root / "summary.md"),
        "entries": entries,
        "promotion_rankings": eligible_entries,
        "winner_config_label": str(winner_label),
        "winner_entry": winner,
        "matched_control_label": str(matched_control_label),
        "official_fno_label": str(official_fno_label),
    }
    stage_json, stage_md = _write_representation_sufficiency_stage_summary(
        output_root=output_root,
        name="representation_sufficiency_lock_summary",
        payload=stage_summary,
    )
    return {
        "lock_summary": stage_summary,
        "lock_summary_json": str(stage_json),
        "lock_summary_md": str(stage_md),
        "promotion_configs": promotion_configs,
        "winner_label": str(winner_label),
        "matched_control_label": str(matched_control_label),
    }


def build_representation_sufficiency_promotion_job_bundle(
    args: argparse.Namespace,
    *,
    promotion_configs: Sequence[_RunConfigSpec],
    official_fno_label: str,
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    promotion_root = output_root / "promotion"
    jobs = _representation_sufficiency_jobs_for_configs(
        args=args,
        configs=list(promotion_configs),
        train_doc_count=int(args.promotion_train_doc_count),
        seeds=[int(seed) for seed in args.promotion_seeds],
        tuning_stage=REPRESENTATION_SUFFICIENCY_PROMOTION_STAGE,
        official_fno_label=str(official_fno_label),
    )
    return {
        "promotion_root": promotion_root,
        "promotion_jobs": jobs,
        "promotion_manifest_payload": {
            "mode": "representation_sufficiency_promotion",
            **_resolved_benchmark_payload(str(args.benchmark)),
            "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
            "train_doc_count": int(args.promotion_train_doc_count),
            "promotion_seeds": [int(seed) for seed in args.promotion_seeds],
            "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
            "official_fno_label": str(official_fno_label),
            "jobs": [asdict(job) for job in jobs],
        },
    }


def _representation_control_health_report(
    control_entry: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(control_entry, Mapping):
        return {
            "healthy": False,
            "all_required_metrics_finite": False,
            "reason": "missing_control_entry",
            "thresholds": {
                "min_exact_match": float(
                    REPRESENTATION_SUFFICIENCY_CONTROL_MIN_EXACT_MATCH
                ),
                "max_sufficiency_gap": float(
                    REPRESENTATION_SUFFICIENCY_CONTROL_MAX_SUFFICIENCY_GAP
                ),
            },
        }
    required_values = {
        "test_exact_sketch_markov_sufficiency_gap_score_mean": float(
            control_entry.get(
                "test_exact_sketch_markov_sufficiency_gap_score_mean",
                float("nan"),
            )
        ),
        "test_probe_leaf_exact_summary_match_rate_mean": float(
            control_entry.get(
                "test_probe_leaf_exact_summary_match_rate_mean",
                float("nan"),
            )
        ),
        "test_probe_merge_exact_summary_match_rate_mean": float(
            control_entry.get(
                "test_probe_merge_exact_summary_match_rate_mean",
                float("nan"),
            )
        ),
        "test_c2_on_range_exact_match_mean": float(
            control_entry.get("test_c2_on_range_exact_match_mean", float("nan"))
        ),
        "test_root_mae_mean": float(
            control_entry.get("test_root_mae_mean", float("nan"))
        ),
        "test_exact_projected_root_mae_mean": float(
            control_entry.get("test_exact_projected_root_mae_mean", float("nan"))
        ),
        "test_certified_projected_root_mae_mean": float(
            control_entry.get("test_certified_projected_root_mae_mean", float("nan"))
        ),
        "test_learned_merger_gap_mean": float(
            control_entry.get("test_learned_merger_gap_mean", float("nan"))
        ),
    }
    all_finite = all(np.isfinite(value) for value in required_values.values())
    meets_leaf = (
        required_values["test_probe_leaf_exact_summary_match_rate_mean"]
        >= float(REPRESENTATION_SUFFICIENCY_CONTROL_MIN_EXACT_MATCH)
    )
    meets_merge = (
        required_values["test_probe_merge_exact_summary_match_rate_mean"]
        >= float(REPRESENTATION_SUFFICIENCY_CONTROL_MIN_EXACT_MATCH)
    )
    meets_c2 = (
        required_values["test_c2_on_range_exact_match_mean"]
        >= float(REPRESENTATION_SUFFICIENCY_CONTROL_MIN_EXACT_MATCH)
    )
    meets_gap = (
        required_values["test_exact_sketch_markov_sufficiency_gap_score_mean"]
        <= float(REPRESENTATION_SUFFICIENCY_CONTROL_MAX_SUFFICIENCY_GAP)
    )
    healthy = bool(all_finite and meets_leaf and meets_merge and meets_c2 and meets_gap)
    return {
        "healthy": bool(healthy),
        "all_required_metrics_finite": bool(all_finite),
        "thresholds": {
            "min_exact_match": float(
                REPRESENTATION_SUFFICIENCY_CONTROL_MIN_EXACT_MATCH
            ),
            "max_sufficiency_gap": float(
                REPRESENTATION_SUFFICIENCY_CONTROL_MAX_SUFFICIENCY_GAP
            ),
        },
        "metrics": required_values,
        "checks": {
            "probe_leaf_exact_summary_match_rate": bool(meets_leaf),
            "probe_merge_exact_summary_match_rate": bool(meets_merge),
            "c2_on_range_exact_match": bool(meets_c2),
            "markov_sufficiency_gap": bool(meets_gap),
        },
    }


def _representation_learned_success_report(
    learned_entry: Mapping[str, Any] | None,
    *,
    control_entry: Mapping[str, Any] | None,
    official_fno_entry: Mapping[str, Any] | None,
    control_health: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(learned_entry, Mapping):
        return {
            "success": False,
            "reason": "missing_learned_entry",
            "criteria": {},
        }
    if not isinstance(control_entry, Mapping):
        return {
            "success": False,
            "reason": "missing_control_entry",
            "criteria": {},
        }
    if not isinstance(official_fno_entry, Mapping):
        return {
            "success": False,
            "reason": "missing_official_fno_entry",
            "criteria": {},
        }
    criteria = {
        "test_exact_sketch_markov_sufficiency_gap_score_mean": {
            "lhs": float(
                learned_entry.get(
                    "test_exact_sketch_markov_sufficiency_gap_score_mean",
                    float("nan"),
                )
            ),
            "rhs": float(
                control_entry.get(
                    "test_exact_sketch_markov_sufficiency_gap_score_mean",
                    float("nan"),
                )
            )
            + float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            "direction": "<=",
        },
        "test_probe_leaf_exact_summary_match_rate_mean": {
            "lhs": float(
                learned_entry.get(
                    "test_probe_leaf_exact_summary_match_rate_mean",
                    float("nan"),
                )
            ),
            "rhs": float(
                control_entry.get(
                    "test_probe_leaf_exact_summary_match_rate_mean",
                    float("nan"),
                )
            )
            - float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            "direction": ">=",
        },
        "test_probe_merge_exact_summary_match_rate_mean": {
            "lhs": float(
                learned_entry.get(
                    "test_probe_merge_exact_summary_match_rate_mean",
                    float("nan"),
                )
            ),
            "rhs": float(
                control_entry.get(
                    "test_probe_merge_exact_summary_match_rate_mean",
                    float("nan"),
                )
            )
            - float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            "direction": ">=",
        },
        "test_c2_on_range_exact_match_mean": {
            "lhs": float(
                learned_entry.get("test_c2_on_range_exact_match_mean", float("nan"))
            ),
            "rhs": float(
                control_entry.get("test_c2_on_range_exact_match_mean", float("nan"))
            )
            - float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            "direction": ">=",
        },
        "test_root_mae_mean": {
            "lhs": float(learned_entry.get("test_root_mae_mean", float("nan"))),
            "rhs": float(official_fno_entry.get("test_root_mae_mean", float("nan")))
            + float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            "direction": "<=",
        },
    }
    checks: Dict[str, Any] = {}
    success = bool(control_health.get("healthy", False))
    for key, criterion in criteria.items():
        lhs = float(criterion["lhs"])
        rhs = float(criterion["rhs"])
        if str(criterion["direction"]) == "<=":
            passed = bool(np.isfinite(lhs) and np.isfinite(rhs) and lhs <= rhs)
        else:
            passed = bool(np.isfinite(lhs) and np.isfinite(rhs) and lhs >= rhs)
        checks[key] = {
            **criterion,
            "passed": bool(passed),
            "delta": float(lhs - rhs)
            if np.isfinite(lhs) and np.isfinite(rhs)
            else float("nan"),
        }
        success = bool(success and passed)
    return {
        "success": bool(success),
        "criteria": checks,
    }


def _render_representation_sufficiency_summary_markdown(
    summary: Mapping[str, Any],
) -> str:
    lines = [
        "# Tree-Neural Representation Sufficiency Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- study_name: `{str(summary.get('study_name', ''))}`",
        f"- selection_metric: `{str(summary.get('selection_metric', ''))}`",
        f"- primary_question: `{str(summary.get('primary_question', ''))}`",
        f"- final_status: `{str(summary.get('final_status', ''))}`",
        f"- winning_config_label: `{str(summary.get('winning_config_label', ''))}`",
        f"- matched_control_label: `{str(summary.get('matched_control_label', ''))}`",
        f"- official_fno_label: `{str(summary.get('official_fno_label', ''))}`",
    ]
    for stage_name in ("screen", "lock", "promotion"):
        stage = dict(summary.get(stage_name) or {})
        entries = list(stage.get("entries") or [])
        lines.extend(
            [
                "",
                f"## {stage_name.title()}",
                "",
                f"- train_doc_count: `{int(stage.get('train_doc_count', 0))}`",
                f"- stage_summary_json: `{str(stage.get('stage_summary_json', ''))}`",
                f"- stage_summary_md: `{str(stage.get('stage_summary_md', ''))}`",
            ]
        )
        if str(stage.get("winner_config_label", "")).strip():
            lines.append(
                f"- winner_config_label: `{str(stage.get('winner_config_label', ''))}`"
            )
        if list(stage.get("selected_learned_labels") or []):
            lines.append(
                f"- selected_learned_labels: `{list(stage.get('selected_learned_labels') or [])}`"
            )
        if list(stage.get("selected_control_labels") or []):
            lines.append(
                f"- selected_control_labels: `{list(stage.get('selected_control_labels') or [])}`"
            )
        if entries:
            lines.extend(
                [
                    "",
                    "| config_label | family | head | exact_lane | eligible | control | val_exact | suff_gap | exact_root | count_only_root | endpoint_only_root | probe_leaf | probe_merge | c2_exact | root_mae |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for entry in entries:
                lines.append(
                    "| "
                    f"{str(entry.get('config_label', ''))} | "
                    f"{str(entry.get('baseline_family', ''))} | "
                    f"{str(entry.get('tree_theorem_count_head_mode', ''))} | "
                    f"{bool(entry.get('exact_lane', False))} | "
                    f"{bool(entry.get('promotion_eligible', False))} | "
                    f"{bool(entry.get('control_only', False))} | "
                    f"{float(entry.get('val_exact_sketch_direct_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_exact_sketch_markov_sufficiency_gap_score_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_exact_projected_root_mae_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_root_mae_predicted_counts_oracle_endpoints_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_root_mae_oracle_counts_predicted_endpoints_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_probe_leaf_exact_summary_match_rate_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_probe_merge_exact_summary_match_rate_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_c2_on_range_exact_match_mean', float('nan'))):.6g} | "
                    f"{float(entry.get('test_root_mae_mean', float('nan'))):.6g} |"
                )
    promotion = dict(summary.get("promotion") or {})
    lines.extend(
        [
            "",
            "## Promotion Gate",
            "",
            f"- slotwise_control_health: `{dict(promotion.get('slotwise_control_health') or {})}`",
            f"- learned_shared_surface_success: `{dict(promotion.get('learned_shared_surface_success') or {})}`",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_representation_sufficiency_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    promotion_root: Path,
    config_by_label: Mapping[str, _RunConfigSpec],
    screen_summary: Mapping[str, Any],
    lock_summary: Mapping[str, Any],
    official_fno_label: str,
) -> Dict[str, Any]:
    combined_payload = _write_summary_outputs(output_root)
    promotion_payload = _load_or_write_summary_outputs(promotion_root)
    promotion_entries = _representation_sufficiency_stage_entries(
        promotion_payload,
        tuning_stage=REPRESENTATION_SUFFICIENCY_PROMOTION_STAGE,
        train_doc_count=int(args.promotion_train_doc_count),
        config_by_label=config_by_label,
        promotion_stage="promotion",
    )
    if not promotion_entries:
        raise RuntimeError(
            "representation sufficiency promotion stage produced no entries"
        )
    winner_label = str(dict(lock_summary).get("winner_config_label", ""))
    matched_control_label = str(dict(lock_summary).get("matched_control_label", ""))
    learned_entry = next(
        (
            dict(entry)
            for entry in promotion_entries
            if str(entry.get("baseline_family", "")) == REPRESENTATION_SUFFICIENCY_FAMILY
            and str(entry.get("config_label", "")) == str(winner_label)
        ),
        None,
    )
    control_entry = next(
        (
            dict(entry)
            for entry in promotion_entries
            if str(entry.get("baseline_family", "")) == REPRESENTATION_SUFFICIENCY_FAMILY
            and str(entry.get("config_label", "")) == str(matched_control_label)
        ),
        None,
    )
    official_fno_entry = next(
        (
            dict(entry)
            for entry in promotion_entries
            if str(entry.get("baseline_family", "")) == "official_fno"
            and str(entry.get("config_label", "")) == str(official_fno_label)
        ),
        None,
    )
    control_health = _representation_control_health_report(control_entry)
    learned_success = _representation_learned_success_report(
        learned_entry,
        control_entry=control_entry,
        official_fno_entry=official_fno_entry,
        control_health=control_health,
    )
    if not bool(control_health.get("healthy", False)):
        final_status = "invalid_slotwise_control"
    elif bool(learned_success.get("success", False)):
        final_status = "representation_flexible_enough"
    else:
        final_status = "shared_surface_bottleneck"
    promotion_summary = {
        "stage_title": "Representation Sufficiency Promotion Summary",
        "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
        "stage": "promotion",
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.promotion_train_doc_count),
        "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
        "entries": promotion_entries,
        "winner_config_label": str(winner_label),
        "matched_control_label": str(matched_control_label),
        "official_fno_label": str(official_fno_label),
        "slotwise_control_health": control_health,
        "learned_shared_surface_success": learned_success,
        "final_status": str(final_status),
    }
    promotion_stage_json, promotion_stage_md = _write_representation_sufficiency_stage_summary(
        output_root=output_root,
        name="representation_sufficiency_promotion_summary",
        payload=promotion_summary,
    )
    final_summary = {
        "study_name": REPRESENTATION_SUFFICIENCY_STUDY_NAME,
        "study_title": "Tree-Neural Representation Sufficiency Summary",
        "benchmark": str(args.benchmark),
        "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
        "primary_question": (
            "Can a learned shared FNO/tree representation recover the Markov "
            "exact sketch well enough that topology reruns are justified?"
        ),
        "screen": {
            **dict(screen_summary),
            "stage_summary_json": str(
                output_root / "representation_sufficiency_screen_summary.json"
            ),
            "stage_summary_md": str(
                output_root / "representation_sufficiency_screen_summary.md"
            ),
        },
        "lock": {
            **dict(lock_summary),
            "stage_summary_json": str(
                output_root / "representation_sufficiency_lock_summary.json"
            ),
            "stage_summary_md": str(
                output_root / "representation_sufficiency_lock_summary.md"
            ),
        },
        "promotion": {
            **promotion_summary,
            "stage_summary_json": str(promotion_stage_json),
            "stage_summary_md": str(promotion_stage_md),
        },
        "winning_config_label": str(winner_label),
        "winning_config_spec": (
            asdict(config_by_label[winner_label])
            if str(winner_label) in config_by_label
            else {}
        ),
        "matched_control_label": str(matched_control_label),
        "official_fno_label": str(official_fno_label),
        "final_status": str(final_status),
        "topology_rerun_recommended": bool(
            final_status == "representation_flexible_enough"
        ),
        "summary_json": str(combined_payload.get("summary_json", output_root / "summary.json")),
        "summary_md": str(combined_payload.get("summary_md", output_root / "summary.md")),
    }
    final_summary_json = (
        output_root / "tree_neural_representation_sufficiency_summary.json"
    )
    final_summary_json.write_text(
        json.dumps(final_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    final_summary_md = (
        output_root / "tree_neural_representation_sufficiency_summary.md"
    )
    final_summary_md.write_text(
        _render_representation_sufficiency_summary_markdown(final_summary),
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "summary_json": str(combined_payload.get("summary_json", output_root / "summary.json")),
        "summary_md": str(combined_payload.get("summary_md", output_root / "summary.md")),
        "representation_sufficiency_screen_summary_json": str(
            output_root / "representation_sufficiency_screen_summary.json"
        ),
        "representation_sufficiency_lock_summary_json": str(
            output_root / "representation_sufficiency_lock_summary.json"
        ),
        "representation_sufficiency_promotion_summary_json": str(
            promotion_stage_json
        ),
        "tree_neural_representation_sufficiency_summary_json": str(
            final_summary_json
        ),
        "tree_neural_representation_sufficiency_summary_md": str(final_summary_md),
        "winning_config_label": str(winner_label),
        "matched_control_label": str(matched_control_label),
        "final_status": str(final_status),
    }


def build_representation_learnability_winner_job_bundle(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    winner_root = output_root / "winner_freeze"
    screen_specs = _representation_sufficiency_screen_config_specs(args)
    jobs = _representation_sufficiency_jobs_for_configs(
        args=args,
        configs=list(screen_specs["config_by_label"].values()),
        train_doc_count=int(args.winner_train_doc_count),
        seeds=[int(seed) for seed in args.winner_seeds],
        tuning_stage=REPRESENTATION_LEARNABILITY_WINNER_STAGE,
        official_fno_label=str(screen_specs["official_fno_label"]),
        study_name=REPRESENTATION_LEARNABILITY_STUDY_NAME,
    )
    return {
        "output_root": output_root,
        "winner_root": winner_root,
        "winner_jobs": jobs,
        "config_by_label": dict(screen_specs["config_by_label"]),
        "config_metadata_by_label": dict(screen_specs["config_metadata_by_label"]),
        "slotwise_control_labels_by_state_dim": dict(
            screen_specs["slotwise_control_labels_by_state_dim"]
        ),
        "official_fno_label": str(screen_specs["official_fno_label"]),
        "winner_manifest_payload": {
            "mode": "representation_learnability_winner",
            **_resolved_benchmark_payload(str(args.benchmark)),
            "study_name": REPRESENTATION_LEARNABILITY_STUDY_NAME,
            "train_doc_count": int(args.winner_train_doc_count),
            "winner_seeds": [int(seed) for seed in args.winner_seeds],
            "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
            "configs_by_label": {
                str(label): asdict(config)
                for label, config in screen_specs["config_by_label"].items()
            },
            "config_metadata_by_label": dict(
                screen_specs["config_metadata_by_label"]
            ),
            "slotwise_control_labels_by_state_dim": dict(
                screen_specs["slotwise_control_labels_by_state_dim"]
            ),
            "official_fno_label": str(screen_specs["official_fno_label"]),
            "jobs": [asdict(job) for job in jobs],
        },
    }


def finalize_representation_learnability_winner_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    winner_root: Path,
    config_by_label: Mapping[str, _RunConfigSpec],
    slotwise_control_labels_by_state_dim: Mapping[int, str],
    official_fno_label: str,
) -> Dict[str, Any]:
    payload = _load_or_write_summary_outputs(winner_root)
    entries = _representation_sufficiency_stage_entries(
        payload,
        tuning_stage=REPRESENTATION_LEARNABILITY_WINNER_STAGE,
        train_doc_count=int(args.winner_train_doc_count),
        config_by_label=config_by_label,
        promotion_stage="winner",
        study_name=REPRESENTATION_LEARNABILITY_STUDY_NAME,
    )
    if not entries:
        raise RuntimeError(
            "representation learnability winner-freeze stage produced no entries"
        )
    eligible_entries = [
        dict(entry)
        for entry in entries
        if str(entry.get("baseline_family", "")) == REPRESENTATION_SUFFICIENCY_FAMILY
        and bool(entry.get("promotion_eligible", False))
    ]
    eligible_entries.sort(key=_representation_screen_sort_key)
    if not eligible_entries:
        raise RuntimeError(
            "representation learnability winner-freeze stage produced no promotion-eligible learned configs"
        )
    winner = dict(eligible_entries[0])
    winner_label = str(winner.get("config_label", ""))
    matched_control_label = str(
        slotwise_control_labels_by_state_dim.get(int(winner.get("state_dim", 0)), "")
    )
    if not matched_control_label or matched_control_label not in config_by_label:
        raise RuntimeError(
            f"representation learnability winner-freeze stage could not find slotwise control for state_dim={int(winner.get('state_dim', 0))}"
        )
    selected_labels = [winner_label, matched_control_label, str(official_fno_label)]
    selected_configs = [
        config_by_label[str(label)]
        for label in selected_labels
        if str(label) in config_by_label
    ]
    stage_summary = {
        "stage_title": "Representation Learnability Winner Freeze Summary",
        "study_name": REPRESENTATION_LEARNABILITY_STUDY_NAME,
        "stage": "winner_freeze",
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.winner_train_doc_count),
        "selection_metric": REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
        "winner_summary_json": str(winner_root / "summary.json"),
        "winner_summary_md": str(winner_root / "summary.md"),
        "entries": entries,
        "winner_rankings": eligible_entries,
        "winner_config_label": str(winner_label),
        "winner_entry": winner,
        "matched_control_label": str(matched_control_label),
        "official_fno_label": str(official_fno_label),
    }
    stage_json, stage_md = _write_representation_sufficiency_stage_summary(
        output_root=output_root,
        name="representation_learnability_winner_summary",
        payload=stage_summary,
    )
    return {
        "winner_summary": stage_summary,
        "winner_summary_json": str(stage_json),
        "winner_summary_md": str(stage_md),
        "winner_label": str(winner_label),
        "matched_control_label": str(matched_control_label),
        "selected_configs": selected_configs,
    }


def _representation_learnability_jobs_for_configs(
    *,
    args: argparse.Namespace,
    configs: Sequence[_RunConfigSpec],
    benchmarks: Sequence[Any],
    train_doc_counts: Sequence[int],
    seeds: Sequence[int],
    tuning_stage: str,
    official_fno_label: str,
) -> List[_JobSpec]:
    jobs: List[_JobSpec] = []
    tree_configs = [
        config for config in configs if str(config.label) != str(official_fno_label)
    ]
    official_configs = [
        config for config in configs if str(config.label) == str(official_fno_label)
    ]
    for benchmark in benchmarks:
        benchmark_name = str(benchmark.name)
        grid_name = str(benchmark.grid_name or "")
        grid_cell_ids = (
            (str(benchmark.cell_id),)
            if grid_name and str(benchmark.cell_id or "").strip()
            else ()
        )
        if tree_configs:
            jobs.extend(
                _build_jobs_for_configs(
                    families=(REPRESENTATION_SUFFICIENCY_FAMILY,),
                    train_doc_counts=tuple(int(value) for value in train_doc_counts),
                    benchmark=benchmark_name,
                    hardness_grid=grid_name,
                    grid_cell_ids=grid_cell_ids,
                    seeds=[int(seed) for seed in seeds],
                    job_granularity=str(args.job_granularity),
                    repeat_closed_form_controls=True,
                    configs=tuple(tree_configs),
                    tuning_stage=str(tuning_stage),
                    study_name=REPRESENTATION_LEARNABILITY_STUDY_NAME,
                    study_axis="representation_config",
                    selection_metric=REPRESENTATION_SUFFICIENCY_SELECTION_METRIC,
                )
            )
        if official_configs:
            jobs.extend(
                _build_jobs_for_configs(
                    families=("official_fno",),
                    train_doc_counts=tuple(int(value) for value in train_doc_counts),
                    benchmark=benchmark_name,
                    hardness_grid=grid_name,
                    grid_cell_ids=grid_cell_ids,
                    seeds=[int(seed) for seed in seeds],
                    job_granularity=str(args.job_granularity),
                    repeat_closed_form_controls=True,
                    configs=tuple(official_configs),
                    tuning_stage=str(tuning_stage),
                    study_name=REPRESENTATION_LEARNABILITY_STUDY_NAME,
                    study_axis="representation_config",
                    selection_metric="official_fno_reference_only",
                )
            )
    return jobs


def _representation_learnability_sweep_entries(
    payload: Mapping[str, Any],
    *,
    tuning_stage: str,
    config_by_label: Mapping[str, _RunConfigSpec],
) -> List[Dict[str, Any]]:
    relevant_runs = [
        dict(run)
        for run in list(payload.get("runs") or [])
        if str(run.get("study_name", "")).strip()
        == REPRESENTATION_LEARNABILITY_STUDY_NAME
        and str(run.get("tuning_stage", "")).strip() == str(tuning_stage)
    ]
    grouped: Dict[tuple[str, int, str, str], List[Dict[str, Any]]] = {}
    for run in relevant_runs:
        benchmark_name = str(run.get("benchmark", "")).strip()
        family = str(run.get("baseline_family", "")).strip()
        label = str(run.get("config_label", "") or run.get("axis_value", "")).strip()
        train_doc_count = int(run.get("train_doc_count", 0))
        if not benchmark_name or not family or not label or train_doc_count <= 0:
            continue
        grouped.setdefault(
            (benchmark_name, int(train_doc_count), family, label),
            [],
        ).append(run)

    entries: List[Dict[str, Any]] = []
    for (benchmark_name, train_doc_count, family, label), runs in sorted(grouped.items()):
        config = config_by_label.get(str(label))
        metadata = (
            _representation_sufficiency_config_metadata(
                config,
                baseline_family=str(family),
                promotion_stage="learnability",
            )
            if config is not None
            else {
                "representation_family": "unknown",
                "representation_variant": "unknown",
                "representation_size": "unknown",
                "promotion_eligible": False,
                "control_only": False,
                "reference_only": False,
                "promotion_stage": "learnability",
                "state_dim": 0,
                "hidden_dim": 0,
                "theorem_feature_dim": 0,
                "theorem_feature_hidden_dim": 0,
                "slot_count": 0,
                "tree_theorem_surface_mode": "",
                "tree_c2_mode": "",
                "tree_theorem_score_dim": 0,
                "tree_theorem_fiber_dim": 0,
                "tree_theorem_aux_dim": 0,
            }
        )
        benchmark_metadata = _representation_learnability_benchmark_metadata(
            benchmark_name
        )
        metrics = _representation_metrics_for_runs(runs)
        entry = {
            "baseline_family": str(family),
            "config_label": str(label),
            "benchmark": str(benchmark_name),
            "train_doc_count": int(train_doc_count),
            "tuning_stage": str(tuning_stage),
            "n_runs": int(len(runs)),
            "seed_values": sorted(
                {int(run.get("seed", 0)) for run in runs if "seed" in run}
            ),
            "config_spec": asdict(config) if config is not None else {},
            "metrics": metrics,
            "val_exact_sketch_direct_mean": float(
                dict(metrics["val_exact_sketch_direct"]).get("mean", float("nan"))
            ),
            "test_exact_sketch_markov_sufficiency_gap_score_mean": float(
                dict(
                    metrics["test_exact_sketch_markov_sufficiency_gap_score"]
                ).get("mean", float("nan"))
            ),
            "test_probe_leaf_exact_summary_match_rate_mean": float(
                dict(metrics["test_probe_leaf_exact_summary_match_rate"]).get(
                    "mean", float("nan")
                )
            ),
            "test_probe_merge_exact_summary_match_rate_mean": float(
                dict(metrics["test_probe_merge_exact_summary_match_rate"]).get(
                    "mean", float("nan")
                )
            ),
            "test_c2_on_range_exact_match_mean": float(
                dict(metrics["test_c2_on_range_exact_match"]).get(
                    "mean", float("nan")
                )
            ),
            "test_exact_projected_root_mae_mean": float(
                dict(metrics["test_exact_projected_root_mae"]).get(
                    "mean", float("nan")
                )
            ),
            "test_certified_projected_root_mae_mean": float(
                dict(metrics["test_certified_projected_root_mae"]).get(
                    "mean", float("nan")
                )
            ),
            "test_root_mae_predicted_counts_predicted_endpoints_mean": float(
                dict(metrics["test_root_mae_predicted_counts_predicted_endpoints"]).get(
                    "mean", float("nan")
                )
            ),
            "test_root_mae_oracle_counts_predicted_endpoints_mean": float(
                dict(metrics["test_root_mae_oracle_counts_predicted_endpoints"]).get(
                    "mean", float("nan")
                )
            ),
            "test_root_mae_predicted_counts_oracle_endpoints_mean": float(
                dict(metrics["test_root_mae_predicted_counts_oracle_endpoints"]).get(
                    "mean", float("nan")
                )
            ),
            "test_learned_merger_gap_mean": float(
                dict(metrics["test_learned_merger_gap"]).get(
                    "mean", float("nan")
                )
            ),
            "test_leaf_first_accuracy_mean": float(
                dict(metrics["test_leaf_first_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_leaf_last_accuracy_mean": float(
                dict(metrics["test_leaf_last_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_merge_first_accuracy_mean": float(
                dict(metrics["test_merge_first_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_merge_last_accuracy_mean": float(
                dict(metrics["test_merge_last_accuracy"]).get(
                    "mean", float("nan")
                )
            ),
            "test_leaf_count_off_by_k_histogram_mean": dict(
                metrics["test_leaf_count_off_by_k_histogram"].get("mean", {}) or {}
            ),
            "test_merge_exact_summary_match_rate_by_depth_mean": dict(
                metrics["test_merge_exact_summary_match_rate_by_depth"].get(
                    "mean", {}
                )
                or {}
            ),
            "test_root_mae_mean": float(
                dict(metrics["test_root_mae"]).get("mean", float("nan"))
            ),
            **metadata,
            **benchmark_metadata,
        }
        _validate_representation_theorem_metrics(entry)
        entries.append(entry)
    return entries


def _representation_threshold_report(
    lhs_stats: Mapping[str, Any],
    rhs_stats: Mapping[str, Any],
    *,
    direction: str,
    tolerance: float,
    conservative: bool,
) -> Dict[str, Any]:
    lhs_mean = float(lhs_stats.get("mean", float("nan")))
    rhs_mean = float(rhs_stats.get("mean", float("nan")))
    lhs_se = _representation_metric_se(lhs_stats)
    rhs_se = _representation_metric_se(rhs_stats)
    if str(direction) == "<=":
        lhs_value = lhs_mean + (1.96 * lhs_se if conservative and np.isfinite(lhs_se) else 0.0)
        rhs_value = rhs_mean + float(tolerance)
        if conservative and np.isfinite(rhs_se):
            rhs_value += 1.96 * rhs_se
        passed = bool(
            np.isfinite(lhs_value)
            and np.isfinite(rhs_value)
            and lhs_value <= rhs_value
        )
    else:
        lhs_value = lhs_mean - (1.96 * lhs_se if conservative and np.isfinite(lhs_se) else 0.0)
        rhs_value = rhs_mean - float(tolerance)
        if conservative and np.isfinite(rhs_se):
            rhs_value -= 1.96 * rhs_se
        passed = bool(
            np.isfinite(lhs_value)
            and np.isfinite(rhs_value)
            and lhs_value >= rhs_value
        )
    return {
        "lhs_mean": float(lhs_mean),
        "rhs_mean": float(rhs_mean),
        "lhs_se": float(lhs_se),
        "rhs_se": float(rhs_se),
        "lhs_value": float(lhs_value),
        "rhs_value": float(rhs_value),
        "direction": str(direction),
        "tolerance": float(tolerance),
        "conservative": bool(conservative),
        "passed": bool(passed),
    }


def _representation_learnability_point_report(
    learned_entry: Mapping[str, Any] | None,
    *,
    control_entry: Mapping[str, Any] | None,
    official_fno_entry: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(learned_entry, Mapping):
        return {
            "valid_control": False,
            "pass_mean": False,
            "pass_conservative": False,
            "status": "missing_learned_entry",
        }
    if not isinstance(control_entry, Mapping):
        return {
            "valid_control": False,
            "pass_mean": False,
            "pass_conservative": False,
            "status": "missing_control_entry",
        }
    if not isinstance(official_fno_entry, Mapping):
        return {
            "valid_control": False,
            "pass_mean": False,
            "pass_conservative": False,
            "status": "missing_official_fno_entry",
        }
    control_health = _representation_control_health_report(control_entry)
    criteria_specs = {
        "test_exact_sketch_markov_sufficiency_gap_score": {
            "direction": "<=",
            "rhs_entry": control_entry,
        },
        "test_probe_leaf_exact_summary_match_rate": {
            "direction": ">=",
            "rhs_entry": control_entry,
        },
        "test_probe_merge_exact_summary_match_rate": {
            "direction": ">=",
            "rhs_entry": control_entry,
        },
        "test_c2_on_range_exact_match": {
            "direction": ">=",
            "rhs_entry": control_entry,
        },
        "test_root_mae": {
            "direction": "<=",
            "rhs_entry": official_fno_entry,
        },
    }
    mean_criteria: Dict[str, Any] = {}
    conservative_criteria: Dict[str, Any] = {}
    pass_mean = bool(control_health.get("healthy", False))
    pass_conservative = bool(control_health.get("healthy", False))
    for metric_key, spec in criteria_specs.items():
        rhs_entry = cast(Mapping[str, Any], spec["rhs_entry"])
        mean_report = _representation_threshold_report(
            _representation_metric_lookup(learned_entry, metric_key),
            _representation_metric_lookup(rhs_entry, metric_key),
            direction=str(spec["direction"]),
            tolerance=float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            conservative=False,
        )
        conservative_report = _representation_threshold_report(
            _representation_metric_lookup(learned_entry, metric_key),
            _representation_metric_lookup(rhs_entry, metric_key),
            direction=str(spec["direction"]),
            tolerance=float(REPRESENTATION_SUFFICIENCY_DELTA_TOLERANCE),
            conservative=True,
        )
        mean_criteria[str(metric_key)] = mean_report
        conservative_criteria[str(metric_key)] = conservative_report
        pass_mean = bool(pass_mean and mean_report["passed"])
        pass_conservative = bool(pass_conservative and conservative_report["passed"])
    if not bool(control_health.get("healthy", False)):
        status = "invalid_control"
    elif pass_conservative:
        status = "pass_conservative"
    elif pass_mean:
        status = "pass_mean_only"
    else:
        status = "failed_with_healthy_control"
    return {
        "valid_control": bool(control_health.get("healthy", False)),
        "control_health": control_health,
        "pass_mean": bool(pass_mean),
        "pass_conservative": bool(pass_conservative),
        "mean_criteria": mean_criteria,
        "conservative_criteria": conservative_criteria,
        "status": str(status),
    }


def _render_representation_learnability_summary_markdown(
    summary: Mapping[str, Any],
) -> str:
    lines = [
        "# Tree-Neural Representation Learnability Summary",
        "",
        f"- study_name: `{str(summary.get('study_name', ''))}`",
        f"- winner_label: `{str(summary.get('winner_label', ''))}`",
        f"- matched_control_label: `{str(summary.get('matched_control_label', ''))}`",
        f"- official_fno_label: `{str(summary.get('official_fno_label', ''))}`",
        f"- final_status: `{str(summary.get('final_status', ''))}`",
        "",
        "| benchmark_cell | regimes | density | recoverable | slotwise_control_healthy | pass_mean | pass_conservative | n_min_mean | n_min_conservative | status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell_summary in list(summary.get("cell_summaries") or []):
        lines.append(
            "| "
            f"{str(cell_summary.get('benchmark_cell', ''))} | "
            f"{int(cell_summary.get('regime_count', 0))} | "
            f"{str(cell_summary.get('segment_density_band', ''))} | "
            f"{bool(cell_summary.get('lean_recoverable_in_principle', False))} | "
            f"{bool(cell_summary.get('slotwise_control_healthy', False))} | "
            f"{bool(cell_summary.get('pass_mean', False))} | "
            f"{bool(cell_summary.get('pass_conservative', False))} | "
            f"{cell_summary.get('n_min_mean', None)} | "
            f"{cell_summary.get('n_min_conservative', None)} | "
            f"{str(cell_summary.get('status', ''))} |"
        )
    return "\n".join([*lines, ""])


def finalize_representation_learnability_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    sweep_root: Path,
    config_by_label: Mapping[str, _RunConfigSpec],
    winner_summary: Mapping[str, Any],
    winner_label: str,
    matched_control_label: str,
    official_fno_label: str,
) -> Dict[str, Any]:
    combined_payload = _write_summary_outputs(output_root)
    sweep_payload = _load_or_write_summary_outputs(sweep_root)
    sweep_entries = _representation_learnability_sweep_entries(
        sweep_payload,
        tuning_stage=REPRESENTATION_LEARNABILITY_SWEEP_STAGE,
        config_by_label=config_by_label,
    )
    if not sweep_entries:
        raise RuntimeError("representation learnability sweep stage produced no entries")
    point_reports: List[Dict[str, Any]] = []
    grouped_by_point: Dict[tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for entry in sweep_entries:
        grouped_by_point.setdefault(
            (str(entry.get("benchmark_cell", "")), int(entry.get("train_doc_count", 0))),
            {},
        )[f"{str(entry.get('baseline_family', ''))}::{str(entry.get('config_label', ''))}"] = dict(entry)

    for (benchmark_cell, train_doc_count), entries_by_key in sorted(grouped_by_point.items()):
        learned_entry = entries_by_key.get(f"{REPRESENTATION_SUFFICIENCY_FAMILY}::{winner_label}")
        control_entry = entries_by_key.get(
            f"{REPRESENTATION_SUFFICIENCY_FAMILY}::{matched_control_label}"
        )
        official_entry = entries_by_key.get(f"official_fno::{official_fno_label}")
        point_report = _representation_learnability_point_report(
            learned_entry,
            control_entry=control_entry,
            official_fno_entry=official_entry,
        )
        metadata_source = (
            learned_entry
            if isinstance(learned_entry, Mapping)
            else control_entry
            if isinstance(control_entry, Mapping)
            else official_entry
            if isinstance(official_entry, Mapping)
            else {}
        )
        point_reports.append(
            {
                "benchmark": str(metadata_source.get("benchmark", "")),
                "benchmark_cell": str(benchmark_cell),
                "regime_count": int(metadata_source.get("regime_count", 0)),
                "segment_density_band": str(
                    metadata_source.get("segment_density_band", "")
                ),
                "train_doc_count": int(train_doc_count),
                "winner_label": str(winner_label),
                "matched_control_label": str(matched_control_label),
                "official_fno_label": str(official_fno_label),
                "lean_recoverable_in_principle": bool(
                    metadata_source.get("lean_recoverable_in_principle", False)
                ),
                "lean_bayes_error_zero": bool(
                    metadata_source.get("lean_bayes_error_zero", False)
                ),
                "learned_entry": dict(learned_entry or {}),
                "control_entry": dict(control_entry or {}),
                "official_fno_entry": dict(official_entry or {}),
                **point_report,
            }
        )

    cell_summaries: List[Dict[str, Any]] = []
    grouped_cell_reports: Dict[str, List[Dict[str, Any]]] = {}
    for report in point_reports:
        grouped_cell_reports.setdefault(str(report["benchmark_cell"]), []).append(report)
    for benchmark_cell, reports in sorted(grouped_cell_reports.items()):
        reports = sorted(reports, key=lambda row: int(row.get("train_doc_count", 0)))
        valid_reports = [row for row in reports if bool(row.get("valid_control", False))]
        n_min_mean = next(
            (
                int(row.get("train_doc_count", 0))
                for row in reports
                if bool(row.get("valid_control", False))
                and bool(row.get("pass_mean", False))
            ),
            None,
        )
        n_min_conservative = next(
            (
                int(row.get("train_doc_count", 0))
                for row in reports
                if bool(row.get("valid_control", False))
                and bool(row.get("pass_conservative", False))
            ),
            None,
        )
        sample = reports[0]
        if not valid_reports:
            status = "invalid_control"
        elif n_min_conservative is not None:
            status = "pass_conservative"
        elif n_min_mean is not None:
            status = "pass_mean_only"
        elif bool(sample.get("lean_recoverable_in_principle", False)):
            status = "recoverable_in_principle_not_yet_learned"
        else:
            status = "failed_without_recoverability_certificate"
        cell_summary = {
            "benchmark": str(sample.get("benchmark", "")),
            "benchmark_cell": str(benchmark_cell),
            "regime_count": int(sample.get("regime_count", 0)),
            "segment_density_band": str(sample.get("segment_density_band", "")),
            "winner_label": str(winner_label),
            "matched_control_label": str(matched_control_label),
            "official_fno_label": str(official_fno_label),
            "lean_recoverable_in_principle": bool(
                sample.get("lean_recoverable_in_principle", False)
            ),
            "lean_bayes_error_zero": bool(sample.get("lean_bayes_error_zero", False)),
            "slotwise_control_healthy": bool(valid_reports),
            "pass_mean": bool(n_min_mean is not None),
            "pass_conservative": bool(n_min_conservative is not None),
            "n_min_mean": n_min_mean,
            "n_min_conservative": n_min_conservative,
            "status": str(status),
            "point_reports": reports,
        }
        cell_summaries.append(cell_summary)
        cell_json = output_root / f"representation_learnability__{benchmark_cell}_summary.json"
        cell_json.write_text(
            json.dumps(cell_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cell_md = output_root / f"representation_learnability__{benchmark_cell}_summary.md"
        cell_md.write_text(
            _render_representation_learnability_summary_markdown(
                {
                    "study_name": REPRESENTATION_LEARNABILITY_STUDY_NAME,
                    "winner_label": str(winner_label),
                    "matched_control_label": str(matched_control_label),
                    "official_fno_label": str(official_fno_label),
                    "final_status": str(status),
                    "cell_summaries": [cell_summary],
                }
            ),
            encoding="utf-8",
        )

    if any(bool(cell.get("pass_conservative", False)) for cell in cell_summaries):
        final_status = "threshold_estimated_conservative"
    elif any(bool(cell.get("pass_mean", False)) for cell in cell_summaries):
        final_status = "threshold_estimated_mean_only"
    elif any(bool(cell.get("slotwise_control_healthy", False)) for cell in cell_summaries):
        final_status = "recoverable_in_principle_not_yet_learned"
    else:
        final_status = "invalid_control"
    final_summary = {
        "study_name": REPRESENTATION_LEARNABILITY_STUDY_NAME,
        "study_title": "Tree-Neural Representation Learnability Summary",
        "primary_question": (
            "When should the current Markov representation test work in principle, "
            "and how many training documents does the learned shared state need "
            "before it clears the sufficiency-first gate?"
        ),
        "winner_freeze": {
            **dict(winner_summary),
            "stage_summary_json": str(
                output_root / "representation_learnability_winner_summary.json"
            ),
            "stage_summary_md": str(
                output_root / "representation_learnability_winner_summary.md"
            ),
        },
        "winner_label": str(winner_label),
        "matched_control_label": str(matched_control_label),
        "official_fno_label": str(official_fno_label),
        "sweep_train_doc_counts": [
            int(value) for value in tuple(args.sweep_train_doc_counts or ())
        ],
        "sweep_seeds": [int(seed) for seed in args.sweep_seeds],
        "cell_summaries": cell_summaries,
        "point_reports": point_reports,
        "final_status": str(final_status),
        "summary_json": str(combined_payload.get("summary_json", output_root / "summary.json")),
        "summary_md": str(combined_payload.get("summary_md", output_root / "summary.md")),
    }
    final_summary_json = (
        output_root / "tree_neural_representation_learnability_summary.json"
    )
    final_summary_json.write_text(
        json.dumps(final_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    final_summary_md = (
        output_root / "tree_neural_representation_learnability_summary.md"
    )
    final_summary_md.write_text(
        _render_representation_learnability_summary_markdown(final_summary),
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "summary_json": str(combined_payload.get("summary_json", output_root / "summary.json")),
        "summary_md": str(combined_payload.get("summary_md", output_root / "summary.md")),
        "representation_learnability_winner_summary_json": str(
            output_root / "representation_learnability_winner_summary.json"
        ),
        "tree_neural_representation_learnability_summary_json": str(
            final_summary_json
        ),
        "tree_neural_representation_learnability_summary_md": str(final_summary_md),
        "winner_label": str(winner_label),
        "matched_control_label": str(matched_control_label),
        "final_status": str(final_status),
    }


def _write_summary_outputs(output_root: Path) -> Dict[str, Any]:
    payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(output_root)
    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md = output_root / "summary.md"
    summary_md.write_text(
        render_full_doc_anchor_diagnostic_markdown(payload),
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_json)
    payload["summary_md"] = str(summary_md)
    return payload


def _load_or_write_summary_outputs(output_root: Path) -> Dict[str, Any]:
    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    if not summary_json.exists():
        return _write_summary_outputs(output_root)
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    if not summary_md.exists():
        summary_md.write_text(
            render_full_doc_anchor_diagnostic_markdown(payload),
            encoding="utf-8",
        )
    payload["summary_json"] = str(summary_json)
    payload["summary_md"] = str(summary_md)
    return payload


def _worker_command_for_job(
    job: _JobSpec,
    *,
    output_dir: Path,
    torch_threads: int,
    use_cuda: bool,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_family = str(getattr(job.config, "baseline_family", "") or "").strip()
    if config_family != str(job.family).strip():
        raise ValueError(
            "job config baseline_family must match job.family before worker launch "
            f"(config={config_family!r}, job={str(job.family).strip()!r})"
        )
    config_spec_path = output_dir / "requested_run_config.json"
    _write_run_config_spec(config_spec_path, job.config)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--job-name",
        str(job.job_name),
        "--output-dir",
        str(output_dir),
        "--memory-probe-jsonl",
        str(output_dir / "memory_probe.jsonl"),
        "--family",
        str(job.family),
        "--train-doc-count",
        str(int(job.train_doc_count)),
        "--benchmark",
        str(job.benchmark),
        "--hardness-grid",
        str(job.hardness_grid),
        "--state-dim",
        str(int(job.config.state_dim)),
        "--hidden-dim",
        str(int(job.config.hidden_dim)),
        "--n-epochs",
        str(int(job.config.n_epochs)),
        "--batch-size",
        str(int(job.config.batch_size)),
        "--lr",
        str(float(job.config.lr)),
        "--weight-decay",
        str(float(job.config.weight_decay)),
        "--torch-threads",
        str(int(torch_threads)),
        "--config-label",
        str(job.config.label),
        "--config-spec-json-path",
        str(config_spec_path),
        "--tuning-stage",
        str(job.tuning_stage),
    ]
    if bool(job.grid_cell_ids):
        cmd.extend(["--grid-cell-ids", *[str(cell) for cell in job.grid_cell_ids]])
    if job.config.tree_local_law_weight is not None:
        cmd.extend(
            ["--tree-local-law-weight", str(float(job.config.tree_local_law_weight))]
        )
    if job.config.fixed_leaf_tokens is not None:
        cmd.extend(
            ["--fixed-leaf-tokens", str(int(job.config.fixed_leaf_tokens))]
        )
    if job.config.tree_task_objective_weight is not None:
        cmd.extend(
            [
                "--tree-task-objective-weight",
                str(float(job.config.tree_task_objective_weight)),
            ]
        )
    if str(job.config.tree_local_weighting_mode).strip():
        cmd.extend(
            [
                "--tree-local-weighting-mode",
                str(job.config.tree_local_weighting_mode),
            ]
        )
    if str(job.config.tree_exact_collapse_mode).strip():
        cmd.extend(
            [
                "--tree-exact-collapse-mode",
                str(job.config.tree_exact_collapse_mode),
            ]
        )
    if bool(job.config.official_fno_preserve_requested_leaf_tokens):
        cmd.append("--official-fno-preserve-requested-leaf-tokens")
    if bool(job.config.preserve_requested_leaf_tokens):
        cmd.append("--preserve-requested-leaf-tokens")
    if str(job.config.comparison_mode).strip():
        cmd.extend(["--comparison-mode", str(job.config.comparison_mode)])
    if float(job.config.tree_c1_relative_weight) != 1.0:
        cmd.extend(
            [
                "--tree-c1-relative-weight",
                str(float(job.config.tree_c1_relative_weight)),
            ]
        )
    if float(job.config.tree_c2_relative_weight) != 1.0:
        cmd.extend(
            [
                "--tree-c2-relative-weight",
                str(float(job.config.tree_c2_relative_weight)),
            ]
        )
    if float(job.config.tree_c3_relative_weight) != 1.0:
        cmd.extend(
            [
                "--tree-c3-relative-weight",
                str(float(job.config.tree_c3_relative_weight)),
            ]
        )
    if job.config.tree_leaf_fno_width is not None:
        cmd.extend(
            ["--tree-leaf-fno-width", str(int(job.config.tree_leaf_fno_width))]
        )
    if job.config.tree_leaf_fno_n_modes is not None:
        cmd.extend(
            [
                "--tree-leaf-fno-n-modes",
                str(int(job.config.tree_leaf_fno_n_modes)),
            ]
        )
    if job.config.tree_leaf_fno_n_layers is not None:
        cmd.extend(
            [
                "--tree-leaf-fno-n-layers",
                str(int(job.config.tree_leaf_fno_n_layers)),
            ]
        )
    if str(job.config.tree_model_version).strip():
        cmd.extend(
            [
                "--tree-model-version",
                str(job.config.tree_model_version),
            ]
        )
    if str(job.config.tree_batch_runtime_mode).strip():
        cmd.extend(
            [
                "--tree-batch-runtime-mode",
                str(job.config.tree_batch_runtime_mode),
            ]
        )
    if str(job.config.tree_root_supervision_kind).strip():
        cmd.extend(
            [
                "--tree-root-supervision-kind",
                str(job.config.tree_root_supervision_kind),
            ]
        )
    if str(job.config.tree_document_loss_normalization_mode).strip():
        cmd.extend(
            [
                "--tree-document-loss-normalization-mode",
                str(job.config.tree_document_loss_normalization_mode),
            ]
        )
    if str(job.config.tree_supervision_source).strip():
        cmd.extend(
            [
                "--tree-supervision-source",
                str(job.config.tree_supervision_source),
            ]
        )
    if str(job.config.tree_checkpoint_metric).strip():
        cmd.extend(
            [
                "--tree-checkpoint-metric",
                str(job.config.tree_checkpoint_metric),
            ]
        )
    if str(job.config.tree_stage1_checkpoint_metric).strip():
        cmd.extend(
            [
                "--tree-stage1-checkpoint-metric",
                str(job.config.tree_stage1_checkpoint_metric),
            ]
        )
    if str(job.config.tree_stage1_eval_mode).strip():
        cmd.extend(
            [
                "--tree-stage1-eval-mode",
                str(job.config.tree_stage1_eval_mode),
            ]
        )
    if int(job.config.tree_stage1_screen_doc_limit) != 0:
        cmd.extend(
            [
                "--tree-stage1-screen-doc-limit",
                str(int(job.config.tree_stage1_screen_doc_limit)),
            ]
        )
    if int(job.config.tree_stage1_final_exact_doc_limit) != 0:
        cmd.extend(
            [
                "--tree-stage1-final-exact-doc-limit",
                str(int(job.config.tree_stage1_final_exact_doc_limit)),
            ]
        )
    if int(job.config.exact_metric_selection_doc_limit) != 0:
        cmd.extend(
            [
                "--exact-metric-selection-doc-limit",
                str(int(job.config.exact_metric_selection_doc_limit)),
            ]
        )
    if int(job.config.exact_metric_selection_interval) != 1:
        cmd.extend(
            [
                "--exact-metric-selection-interval",
                str(int(job.config.exact_metric_selection_interval)),
            ]
        )
    if int(job.config.tree_exact_eval_max_docs) != 0:
        cmd.extend(
            [
                "--tree-exact-eval-max-docs",
                str(int(job.config.tree_exact_eval_max_docs)),
            ]
        )
    if int(job.config.tree_posttrain_train_doc_limit) != 0:
        cmd.extend(
            [
                "--tree-posttrain-train-doc-limit",
                str(int(job.config.tree_posttrain_train_doc_limit)),
            ]
        )
    if str(job.config.tree_batch_pack_mode).strip():
        cmd.extend(
            [
                "--tree-batch-pack-mode",
                str(job.config.tree_batch_pack_mode),
            ]
        )
    if int(job.config.tree_batch_token_budget) != 0:
        cmd.extend(
            [
                "--tree-batch-token-budget",
                str(int(job.config.tree_batch_token_budget)),
            ]
        )
    if int(job.config.tree_batch_node_budget) != 0:
        cmd.extend(
            [
                "--tree-batch-node-budget",
                str(int(job.config.tree_batch_node_budget)),
            ]
        )
    cmd.append(
        "--tree-batch-autotune"
        if bool(job.config.tree_batch_autotune)
        else "--no-tree-batch-autotune"
    )
    if float(job.config.tree_batch_structural_pad_limit) != 0.5:
        cmd.extend(
            [
                "--tree-batch-structural-pad-limit",
                str(float(job.config.tree_batch_structural_pad_limit)),
            ]
        )
    if int(job.config.tree_batch_auto_queue_min_docs) != 8:
        cmd.extend(
            [
                "--tree-batch-auto-queue-min-docs",
                str(int(job.config.tree_batch_auto_queue_min_docs)),
            ]
        )
    if float(job.config.tree_batch_auto_queue_min_fill_ratio) != 0.5:
        cmd.extend(
            [
                "--tree-batch-auto-queue-min-fill-ratio",
                str(float(job.config.tree_batch_auto_queue_min_fill_ratio)),
            ]
        )
    if int(job.config.tree_eval_workers_per_mig) != 0:
        cmd.extend(
            [
                "--tree-eval-workers-per-mig",
                str(int(job.config.tree_eval_workers_per_mig)),
            ]
        )
    cmd.extend(
        [
            "--gpu-runtime-data-mode",
            str(job.config.gpu_runtime_data_mode),
            "--gpu-runtime-bucket-mode",
            str(job.config.gpu_runtime_bucket_mode),
            "--gpu-runtime-preload-splits",
            *[str(value) for value in job.config.gpu_runtime_preload_splits],
            (
                "--gpu-runtime-preload-targets"
                if bool(job.config.gpu_runtime_preload_targets)
                else "--no-gpu-runtime-preload-targets"
            ),
            "--gpu-runtime-workers-per-mig",
            str(int(job.config.gpu_runtime_workers_per_mig)),
            (
                "--gpu-runtime-allow-multi-worker-screen"
                if bool(job.config.gpu_runtime_allow_multi_worker_screen)
                else "--no-gpu-runtime-allow-multi-worker-screen"
            ),
            "--gpu-runtime-capacity-workers-per-mig",
            str(int(job.config.gpu_runtime_capacity_workers_per_mig)),
        ]
    )
    if str(job.config.tree_stage1_artifact_dir).strip():
        cmd.extend(
            [
                "--tree-stage1-artifact-dir",
                str(job.config.tree_stage1_artifact_dir),
            ]
        )
    if str(job.config.prepared_data_root).strip():
        cmd.extend(
            [
                "--prepared-data-root",
                str(job.config.prepared_data_root),
            ]
        )
    cmd.append(
        "--prepared-data-allow-create"
        if bool(job.config.prepared_data_allow_create)
        else "--no-prepared-data-allow-create"
    )
    if str(job.config.base_bundle_path).strip():
        cmd.extend(
            [
                "--base-bundle-path",
                str(job.config.base_bundle_path),
            ]
        )
    if str(job.config.diagnostic_detail_mode).strip():
        cmd.extend(
            [
                "--diagnostic-detail-mode",
                str(job.config.diagnostic_detail_mode),
            ]
        )
    if str(job.config.posttrain_diagnostics_mode).strip():
        cmd.extend(
            [
                "--posttrain-diagnostics-mode",
                str(job.config.posttrain_diagnostics_mode),
            ]
        )
    if str(job.config.raw_diagnostic_artifact_dir).strip():
        cmd.extend(
            [
                "--raw-diagnostic-artifact-dir",
                str(job.config.raw_diagnostic_artifact_dir),
            ]
        )
    if float(job.config.tree_stage1_root_weight) > 0.0:
        cmd.extend(
            [
                "--tree-stage1-root-weight",
                str(float(job.config.tree_stage1_root_weight)),
            ]
        )
    if float(job.config.tree_join_bit_weight) > 0.0:
        cmd.extend(
            [
                "--tree-join-bit-weight",
                str(float(job.config.tree_join_bit_weight)),
            ]
        )
    if str(job.config.tree_training_schedule).strip():
        cmd.extend(
            [
                "--tree-training-schedule",
                str(job.config.tree_training_schedule),
                "--tree-stage1-epochs",
                str(int(job.config.tree_stage1_epochs)),
                "--tree-stage2-epochs",
                str(int(job.config.tree_stage2_epochs)),
                "--tree-task-head-mode",
                str(job.config.tree_task_head_mode),
                "--tree-theorem-surface-mode",
                str(job.config.tree_theorem_surface_mode),
                "--tree-theorem-count-head-mode",
                str(job.config.tree_theorem_count_head_mode),
                "--tree-theorem-count-ordinal-weight",
                str(float(job.config.tree_theorem_count_ordinal_weight)),
                "--tree-theorem-count-scalar-aux-weight",
                str(float(job.config.tree_theorem_count_scalar_aux_weight)),
                "--tree-theorem-feature-dim",
                str(int(job.config.tree_theorem_feature_dim)),
                "--tree-theorem-feature-hidden-dim",
                str(int(job.config.tree_theorem_feature_hidden_dim)),
                "--tree-merge-hidden-dim",
                str(int(job.config.tree_merge_hidden_dim)),
                "--tree-theorem-score-dim",
                str(int(job.config.tree_theorem_score_dim)),
                "--tree-theorem-fiber-dim",
                str(int(job.config.tree_theorem_fiber_dim)),
                "--tree-theorem-aux-dim",
                str(int(job.config.tree_theorem_aux_dim)),
                "--tree-score-merge-mode",
                str(job.config.tree_score_merge_mode),
                "--tree-phi-compose-weight",
                str(float(job.config.tree_phi_compose_weight)),
                "--tree-phi-contrastive-weight",
                str(float(job.config.tree_phi_contrastive_weight)),
                "--tree-phi-alignment-loss",
                str(job.config.tree_phi_alignment_loss),
                "--tree-c2-mode",
                str(job.config.tree_c2_mode),
                "--tree-summary-spec-root-mode",
                str(job.config.tree_summary_spec_root_mode),
                "--leaf-supervision-kind",
                str(job.config.leaf_supervision_kind),
            ]
        )
        if not bool(job.config.tree_theorem_count_threshold_balance):
            cmd.append("--no-tree-theorem-count-threshold-balance")
    if str(job.config.aligned_sketch_surface).strip():
        cmd.extend(
            [
                "--aligned-sketch-surface",
                str(job.config.aligned_sketch_surface),
            ]
        )
    if str(job.config.summary_spec_name).strip():
        cmd.extend(
            [
                "--summary-spec-name",
                str(job.config.summary_spec_name),
                "--slot-count",
                str(int(job.config.slot_count)),
                "--tree-theorem-count-dim",
                str(int(job.config.tree_theorem_count_dim)),
                "--tree-theorem-first-dim",
                str(int(job.config.tree_theorem_first_dim)),
                "--tree-theorem-last-dim",
                str(int(job.config.tree_theorem_last_dim)),
                "--leaf-label-rate",
                str(float(job.config.leaf_label_rate)),
            ]
        )
    cmd.extend(
        [
            "--theorem-feature-adapter",
            str(job.config.theorem_feature_adapter),
        ]
    )
    if str(job.config.oracle_metric_name).strip():
        cmd.extend(
            [
                "--oracle-metric-name",
                str(job.config.oracle_metric_name),
                "--oracle-same-threshold",
                str(float(job.config.oracle_same_threshold)),
                "--oracle-diff-threshold",
                str(float(job.config.oracle_diff_threshold)),
            ]
        )
    if job.config.theorem_pair_same_threshold is not None:
        cmd.extend(
            [
                "--theorem-pair-same-threshold",
                str(float(job.config.theorem_pair_same_threshold)),
            ]
        )
    if job.config.theorem_pair_diff_threshold is not None:
        cmd.extend(
            [
                "--theorem-pair-diff-threshold",
                str(float(job.config.theorem_pair_diff_threshold)),
            ]
        )
    cmd.extend(
        [
            "--internal-supervision-kind",
            str(job.config.internal_supervision_kind),
            "--internal-label-rate",
            str(float(job.config.internal_label_rate)),
            "--max-internal-depth",
            str(int(job.config.max_internal_depth)),
        ]
    )
    if bool(job.config.leaf_exact_supervision):
        cmd.append("--leaf-exact-supervision")
    if float(job.config.root_weight) != 1.0:
        cmd.extend(["--root-weight", str(float(job.config.root_weight))])
    if float(job.config.schedule_consistency_weight) != 0.0:
        cmd.extend(
            [
                "--schedule-consistency-weight",
                str(float(job.config.schedule_consistency_weight)),
            ]
        )
    if float(job.config.endpoint_loss_scale) != 1.0:
        cmd.extend(
            ["--endpoint-loss-scale", str(float(job.config.endpoint_loss_scale))]
        )
    if bool(job.test_metrics_hidden_during_selection):
        cmd.append("--test-metrics-hidden-during-selection")
    if str(job.study_name).strip():
        cmd.extend(["--study-name", str(job.study_name)])
    if str(job.study_axis).strip():
        cmd.extend(["--study-axis", str(job.study_axis)])
    if str(job.axis_value).strip():
        cmd.extend(["--axis-value", str(job.axis_value)])
    if str(job.locked_tree_neural_config_label).strip():
        cmd.extend(
            [
                "--locked-tree-neural-config-label",
                str(job.locked_tree_neural_config_label),
            ]
        )
    if str(job.selection_metric).strip():
        cmd.extend(["--selection-metric", str(job.selection_metric)])
    if int(job.budget_total_calls) > 0:
        cmd.extend(["--budget-total-calls", str(int(job.budget_total_calls))])
    if float(job.budget_total_calls_per_doc) > 0.0:
        cmd.extend(
            [
                "--budget-total-calls-per-doc",
                str(float(job.budget_total_calls_per_doc)),
            ]
        )
    if math.isfinite(float(job.mass_target_per_doc)):
        cmd.extend(["--mass-target-per-doc", str(float(job.mass_target_per_doc))])
    cmd.extend(
        [
            "--full-doc-budget-share",
            str(float(job.full_doc_budget_share)),
            "--doc-consumption-mode",
            str(job.doc_consumption_mode),
            "--local-split-mode",
            str(job.local_split_mode),
            "--local-allocation-policy",
            str(job.local_allocation_policy),
        ]
    )
    if str(job.package_semantics).strip():
        cmd.extend(["--package-semantics", str(job.package_semantics)])
    if not math.isclose(
        float(job.config.depth_discount_gamma),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        cmd.extend(
            ["--depth-discount-gamma", str(float(job.config.depth_discount_gamma))]
        )
    cmd.extend(["--seeds", *[str(seed) for seed in job.seeds]])
    if bool(use_cuda):
        cmd.append("--use-cuda")
    return cmd


def _worker_env_for_token(
    token: str,
    *,
    use_cuda: bool,
) -> dict[str, str]:
    env = dict(os.environ)
    if bool(use_cuda):
        env["CUDA_VISIBLE_DEVICES"] = str(token)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _run_job_batch(
    *,
    output_root: Path,
    jobs: Sequence[_JobSpec],
    mig_uuids: Sequence[str],
    resume_enabled: bool,
    use_cuda: bool,
    torch_threads: int,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    job_root = output_root / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mig_job_manifest.json").write_text(
        json.dumps(dict(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    completed_run_keys = _load_completed_run_keys(output_root) if bool(resume_enabled) else set()
    skipped_jobs: List[Dict[str, Any]] = []
    pending: List[_JobSpec] = []
    for job in jobs:
        required_keys = _job_completion_keys(job)
        if required_keys and required_keys.issubset(completed_run_keys):
            skipped_jobs.append(
                {
                    "job_name": job.job_name,
                    "family": job.family,
                    "train_doc_count": int(job.train_doc_count),
                    "config_label": str(job.config.label),
                    "tuning_stage": str(job.tuning_stage),
                    "seeds": [int(seed) for seed in job.seeds],
                    "reason": "already_completed",
                }
            )
            continue
        pending.append(job)

    active: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    available_tokens = list(mig_uuids)
    stop_requested = False
    force_terminate_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, force_terminate_requested
        if not stop_requested:
            stop_requested = True
            print(
                f"received signal {int(signum)}; pausing launch queue and waiting for active jobs to finish",
                flush=True,
            )
            return
        if force_terminate_requested:
            return
        force_terminate_requested = True
        print(
            f"received signal {int(signum)} again; terminating {len(active)} active workers",
            flush=True,
        )
        for entry in active:
            proc = entry.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    continue

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if skipped_jobs:
        print(
            f"skipping {len(skipped_jobs)} completed jobs already present under {output_root}",
            flush=True,
        )

    while pending or active:
        while pending and available_tokens and not stop_requested:
            token = available_tokens.pop(0)
            job = pending.pop(0)
            job_output_dir = job_root / _job_output_dir_name(job.job_name)
            job_output_dir.mkdir(parents=True, exist_ok=True)
            log_path = job_output_dir / "worker.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            cmd = _worker_command_for_job(
                job,
                output_dir=job_output_dir,
                torch_threads=int(torch_threads),
                use_cuda=bool(use_cuda),
            )
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(token)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=log_fh,
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
            )
            active.append(
                {
                    "job": job,
                    "proc": proc,
                    "log_path": log_path,
                    "log_fh": log_fh,
                    "mig_uuid": token,
                }
            )
            print(
                f"launched {job.job_name} seeds={list(job.seeds)} on {token[:18]} pid={proc.pid}",
                flush=True,
            )

        time.sleep(1.0)
        still_active: List[Dict[str, Any]] = []
        for entry in active:
            proc = entry["proc"]
            if proc.poll() is None:
                still_active.append(entry)
                continue
            stdout_text = proc.stdout.read() if proc.stdout is not None else ""
            entry["log_fh"].close()
            available_tokens.append(str(entry["mig_uuid"]))
            if int(proc.returncode) != 0:
                failed.append(
                    {
                        "job_name": entry["job"].job_name,
                        "family": entry["job"].family,
                        "train_doc_count": int(entry["job"].train_doc_count),
                        "config_label": str(entry["job"].config.label),
                        "tuning_stage": str(entry["job"].tuning_stage),
                        "returncode": int(proc.returncode),
                        "log_path": str(entry["log_path"]),
                        "stdout_tail": stdout_text[-500:],
                    }
                )
                print(
                    f"failed {entry['job'].job_name} rc={proc.returncode} log={entry['log_path']}",
                    flush=True,
                )
                continue
            result = json.loads(stdout_text.strip().splitlines()[-1])
            completed.append(result)
            seed_label = ",".join(str(seed) for seed in list(result.get("job_seeds") or []))
            if bool(result.get("test_metrics_hidden_during_selection", False)):
                print(
                    "completed "
                    f"{result['job_name']} "
                    f"seeds=[{seed_label}] "
                    f"val_root_mae={result['val_root_mae']:.6g} "
                    f"selection={result['selection_metric_name'] or 'val_root_mae'} "
                    f"cfg={result['config_label']} "
                    "(test hidden for selection)",
                    flush=True,
                )
            elif bool(result.get("objective_weights_active", False)):
                print(
                    "completed "
                    f"{result['job_name']} "
                    f"seeds=[{seed_label}] "
                    f"root_mae={result['test_root_mae']:.6g} "
                    f"param={result['parameterization']} "
                    f"weights=({result['local_law_c1_weight']:.4g},"
                    f"{result['local_law_c2_weight']:.4g},"
                    f"{result['local_law_c3_weight']:.4g})",
                    flush=True,
                )
            else:
                print(
                    "completed "
                    f"{result['job_name']} "
                    f"seeds=[{seed_label}] "
                    f"root_mae={result['test_root_mae']:.6g} "
                    "(closed_form_control; local-law weights inactive)",
                    flush=True,
                )
        active = still_active

    controller_summary = {
        "completed_jobs": completed,
        "failed_jobs": failed,
        "skipped_jobs": skipped_jobs,
        "resume_enabled": bool(resume_enabled),
        "stop_requested": bool(stop_requested),
    }
    (output_root / "controller_results.json").write_text(
        json.dumps(controller_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        payload = _write_summary_outputs(output_root)
    except FileNotFoundError:
        payload = {"runs": [], "aggregate_rows": []}
    return {
        "payload": payload,
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "skipped_jobs": skipped_jobs,
        "resume_enabled": bool(resume_enabled),
        "stop_requested": bool(stop_requested),
        "output_root": str(output_root),
    }


def _scheduler_item_for_job(
    *,
    phase: str,
    item_id: str,
    output_root: Path,
    job: _JobSpec,
    torch_threads: int,
    use_cuda: bool,
    gpu_slots: int = 1,
    allowed_devices: Sequence[str] = (),
) -> SchedulerItem:
    def _scheduler_scope() -> str:
        if str(job.hardness_grid).strip():
            return str(job.hardness_grid)
        return str(job.benchmark)

    def _scheduler_package() -> str:
        cfg = job.config
        if float(job.full_doc_budget_share) < 0.999999:
            return ""
        if str(job.doc_consumption_mode or "root_only") not in {"", "root_only"}:
            return ""
        if str(job.local_split_mode or "balanced") not in {"", "balanced"}:
            return ""
        if str(cfg.leaf_supervision_kind or "count_only") != "count_only":
            return ""
        if abs(float(cfg.leaf_label_rate)) > 1e-9:
            return ""
        if str(cfg.internal_supervision_kind or "none") != "none":
            return ""
        if abs(float(cfg.internal_label_rate)) > 1e-9:
            return ""
        return "full100"

    job_output_dir = output_root / "jobs" / _job_output_dir_name(str(job.job_name))
    metadata: Dict[str, Any] = {
        "job_name": str(job.job_name),
        "task_name": str(job.job_name),
        "train_docs": int(job.train_doc_count),
        "model_family": str(job.family),
        "worker_kind": "full_doc_diagnostics",
        "n_epochs": int(job.config.n_epochs),
    }
    scope = _scheduler_scope().strip()
    if scope:
        metadata["scope"] = scope
    package = _scheduler_package().strip()
    if package:
        metadata["package"] = package
    return SchedulerItem(
        item_id=str(item_id),
        phase=str(phase),
        kind="gpu_command",
        expected_outputs=(str(job_output_dir / "summary.json"),),
        command=tuple(
            str(arg)
            for arg in _worker_command_for_job(
                job,
                output_dir=job_output_dir,
                torch_threads=int(torch_threads),
                use_cuda=bool(use_cuda),
            )
        ),
        log_path=str(job_output_dir / "worker.log"),
        metadata=metadata,
        gpu_slots=max(1, int(gpu_slots)),
        allowed_devices=tuple(str(token) for token in allowed_devices if str(token).strip()),
    )


def _scheduler_result_from_summary(
    *,
    output_root: Path,
    scheduler_summary: Mapping[str, Any],
    resume_enabled: bool,
) -> Dict[str, Any]:
    def _iter_item_infos(name: str) -> List[Mapping[str, Any]]:
        raw = dict(scheduler_summary).get(name)
        if isinstance(raw, Mapping):
            return [dict(info) for info in raw.values()]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [dict(info) for info in raw if isinstance(info, Mapping)]
        return []

    completed_jobs: List[Dict[str, Any]] = []
    skipped_jobs: List[Dict[str, Any]] = []
    failed_jobs: List[Dict[str, Any]] = []
    for info in _iter_item_infos("completed_items"):
        if str(info.get("kind", "")) != "gpu_command":
            continue
        payload = {
            "item_id": str(info.get("item_id", "")),
            "phase": str(info.get("phase", "")),
            "job_name": str(dict(info.get("metadata") or {}).get("job_name", "")),
            "log_path": str(info.get("log_path", "")),
            "expected_outputs": [str(path) for path in list(info.get("expected_outputs") or [])],
            "gpu_slots": int(info.get("gpu_slots", 1) or 1),
        }
        if bool(info.get("reused", False)):
            payload["reason"] = "already_completed"
            skipped_jobs.append(payload)
        else:
            completed_jobs.append(payload)
    for info in _iter_item_infos("failed_items"):
        if str(info.get("kind", "")) != "gpu_command":
            continue
        failed_jobs.append(
            {
                "item_id": str(info.get("item_id", "")),
                "phase": str(info.get("phase", "")),
                "job_name": str(dict(info.get("metadata") or {}).get("job_name", "")),
                "returncode": int(info.get("returncode", 1) or 1),
                "log_path": str(info.get("log_path", "")),
                "expected_outputs": [
                    str(path) for path in list(info.get("expected_outputs") or [])
                ],
                "gpu_slots": int(info.get("gpu_slots", 1) or 1),
            }
        )
    return {
        "payload": (
            json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            if (output_root / "summary.json").exists()
            else {}
        ),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "resume_enabled": bool(resume_enabled),
        "stop_requested": False,
        "output_root": str(output_root),
        "scheduler_summary": dict(scheduler_summary),
    }


def _run_scheduler_bundle(
    *,
    output_root: Path,
    items: Sequence[SchedulerItem],
    devices: Sequence[str],
    max_gpu_items_per_mig: int,
    launch_stagger_seconds: float,
    cleanup_stale_children: bool,
    resume_enabled: bool,
    manifest_payload: Mapping[str, Any],
    min_mem_available_kib: int = 128 * 1024 * 1024,
    min_swap_free_kib: int = 2 * 1024 * 1024,
    cancel_on_failure: bool = True,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "mig_job_manifest.json"
    scheduler_status_path = output_root / "scheduler_status.json"
    scheduler_event_log_path = output_root / "scheduler_events.jsonl"
    scheduler_failure_snapshot_path = output_root / "scheduler_failure_snapshot.json"
    manifest_path.write_text(
        json.dumps(dict(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        scheduler_summary = run_scheduler(
            items,
            config=SchedulerConfig(
                devices=tuple(str(device) for device in devices),
                max_gpu_items_per_mig=int(max(1, int(max_gpu_items_per_mig))),
                launch_stagger_seconds=float(max(0.0, float(launch_stagger_seconds))),
                cleanup_stale_children=bool(cleanup_stale_children),
                cancel_on_failure=bool(cancel_on_failure),
                min_mem_available_kib=int(max(0, int(min_mem_available_kib))),
                min_swap_free_kib=int(max(0, int(min_swap_free_kib))),
                root_markers=(str(output_root),),
                status_path=str(scheduler_status_path),
                event_log_path=str(scheduler_event_log_path),
                failure_snapshot_path=str(scheduler_failure_snapshot_path),
            ),
        )
    except SchedulerRunError as exc:
        scheduler_summary = dict(exc.summary)
    result = _scheduler_result_from_summary(
        output_root=output_root,
        scheduler_summary=scheduler_summary,
        resume_enabled=bool(resume_enabled),
    )
    memory_probe_summary = _write_memory_probe_summary(output_root)
    controller_summary = {
        "completed_jobs": list(result["completed_jobs"]),
        "failed_jobs": list(result["failed_jobs"]),
        "skipped_jobs": list(result["skipped_jobs"]),
        "resume_enabled": bool(result["resume_enabled"]),
        "stop_requested": bool(result["stop_requested"]),
        "scheduler": dict(scheduler_summary),
        "scheduler_status_json": str(scheduler_status_path),
        "scheduler_events_jsonl": str(scheduler_event_log_path),
        "scheduler_failure_snapshot_json": str(scheduler_failure_snapshot_path),
        "memory_probe_summary_json": str(memory_probe_summary["summary_json"]),
    }
    (output_root / "controller_results.json").write_text(
        json.dumps(controller_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["memory_probe_summary_json"] = str(memory_probe_summary["summary_json"])
    return result


def _scheduler_cli_payload(
    *,
    items: Sequence[SchedulerItem],
    devices: Sequence[str],
    max_gpu_items_per_mig: int,
    launch_stagger_seconds: float,
    min_mem_available_kib: int,
    min_swap_free_kib: int,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scheduler": {
            **summarize_scheduler_plan(
                items,
                devices=tuple(str(device) for device in devices),
                max_gpu_items_per_mig=int(max(1, int(max_gpu_items_per_mig))),
                launch_stagger_seconds=float(max(0.0, float(launch_stagger_seconds))),
            ),
            "min_mem_available_kib": int(max(0, int(min_mem_available_kib))),
            "min_swap_free_kib": int(max(0, int(min_swap_free_kib))),
        },
        "manifest": dict(manifest_payload),
    }


def _launch_controller(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(result["summary_json"]),
                "summary_md": str(result["summary_md"]),
                "completed_jobs": len(list(result["completed_jobs"])),
                "failed_jobs": len(list(result["failed_jobs"])),
                "skipped_jobs": len(list(result["skipped_jobs"])),
                "resume_enabled": bool(result["resume_enabled"]),
                "stop_requested": bool(result["stop_requested"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _select_top_config_rows(
    payload: Mapping[str, Any],
    *,
    baseline_family: str,
    tuning_stage: str,
    train_doc_count: int,
    metric_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    rows = _rank_config_rows(
        payload,
        baseline_family=baseline_family,
        tuning_stage=tuning_stage,
        train_doc_count=train_doc_count,
        metric_key=metric_key,
    )
    return rows[: int(top_k)]


def _rank_config_rows(
    payload: Mapping[str, Any],
    *,
    baseline_family: str,
    tuning_stage: str,
    train_doc_count: int,
    metric_key: str,
) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(payload.get("aggregate_rows") or [])
        if str(row.get("baseline_family", "")) == str(baseline_family)
        and str(row.get("tuning_stage", "")) == str(tuning_stage)
        and int(row.get("train_doc_count", 0)) == int(train_doc_count)
    ]
    rows.sort(
        key=lambda row: (
            float(row.get(metric_key, float("inf"))),
            str(row.get("config_label", "")),
        )
    )
    return rows


def _run_config_from_mapping(mapping: Mapping[str, Any]) -> _RunConfigSpec:
    shared = _shared_run_config_from_mapping(mapping)
    return _RunConfigSpec(**asdict(shared))


def _load_tuning_summary(tuning_root: Path) -> Dict[str, Any]:
    summary_path = tuning_root / "tuning_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing tuning summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_capacity_locked_summary(capacity_root: Path) -> Dict[str, Any]:
    summary_path = capacity_root / "tree_fno_capacity_locked_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing capacity locked summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _locked_tree_neural_config_from_tuning_root(
    tuning_root: Path,
) -> tuple[_RunConfigSpec, Dict[str, Any]]:
    summary = _load_tuning_summary(tuning_root)
    config_spec = summary.get("winning_config_spec")
    if isinstance(config_spec, Mapping):
        return _run_config_from_mapping(config_spec), summary

    winning = dict(summary.get("winning_config") or {})
    winning_label = str(
        summary.get("winning_config_label", "") or winning.get("config_label", "")
    ).strip()
    if not winning_label:
        raise ValueError(
            f"tuning summary at {tuning_root / 'tuning_summary.json'} is missing a winning config label"
        )

    locked_summary_path = Path(
        str(
            summary.get("locked_summary_json", "")
            or summary.get("final_locked_summary_json", "")
        )
    )
    candidate_paths = [locked_summary_path]
    candidate_paths.extend(
        [
            tuning_root / "locked" / "summary.json",
            tuning_root / "final_locked" / "summary.json",
            tuning_root / "screen" / "summary.json",
        ]
    )
    for path in candidate_paths:
        if not str(path).strip() or not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in list(dict(payload).get("runs") or []):
            if (
                str(run.get("baseline_family", "")) == "tree_neural"
                and str(run.get("config_label", "")).strip() == winning_label
                and isinstance(run.get("config"), Mapping)
            ):
                config_dict = dict(run["config"])
                config_dict["label"] = winning_label
                return _run_config_from_mapping(config_dict), summary
    raise RuntimeError(
        f"unable to reconstruct winning tree_neural config '{winning_label}' from {tuning_root}"
    )


def _locked_tree_neural_config_from_capacity_root(
    capacity_root: Path,
) -> tuple[_RunConfigSpec, Dict[str, Any]]:
    summary = _load_capacity_locked_summary(capacity_root)
    config_spec = summary.get("winning_config_spec")
    if isinstance(config_spec, Mapping):
        return _run_config_from_mapping(config_spec), summary
    winning_label = str(summary.get("winning_config_label", "")).strip()
    if not winning_label:
        raise ValueError(
            f"capacity summary at {capacity_root / 'tree_fno_capacity_locked_summary.json'} "
            "is missing a winning config label"
        )
    locked_summary_json = Path(str(summary.get("locked_summary_json", "")).strip())
    candidate_paths = [
        locked_summary_json,
        capacity_root / "locked" / "summary.json",
        capacity_root / "screen" / "summary.json",
    ]
    for path in candidate_paths:
        if not str(path).strip() or not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in list(dict(payload).get("runs") or []):
            if (
                str(run.get("baseline_family", "")) == CAPACITY_PRIORITY_FAMILY
                and str(run.get("config_label", "")).strip() == winning_label
                and isinstance(run.get("config"), Mapping)
            ):
                config_dict = dict(run["config"])
                config_dict["label"] = winning_label
                return _run_config_from_mapping(config_dict), summary
    raise RuntimeError(
        f"unable to reconstruct winning tree_neural capacity config '{winning_label}' from {capacity_root}"
    )


def _comparison_study_config(args: argparse.Namespace) -> _RunConfigSpec:
    preload_splits = tuple(
        str(item)
        for item in list(getattr(args, "gpu_runtime_preload_splits", ("train", "val", "test")))
        if str(item).strip()
    )
    return _RunConfigSpec(
        label="comparison_default",
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.comparison_n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.comparison_lr),
        weight_decay=float(args.weight_decay),
        fixed_leaf_tokens=None,
        tree_local_law_weight=float(args.comparison_tree_local_law_weight),
        tree_task_objective_weight=(
            None
            if args.tree_task_objective_weight is None
            else float(args.tree_task_objective_weight)
        ),
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
        doc_sequence_train_fraction=float(args.doc_sequence_train_fraction),
    )


def _study_focus_rows(
    payload: Mapping[str, Any],
    *,
    family: str,
    tuning_stage: str = "",
) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(payload.get("aggregate_rows") or [])
        if str(row.get("baseline_family", "")) == str(family)
    ]
    if str(tuning_stage).strip():
        rows = [
            row
            for row in rows
            if str(row.get("tuning_stage", "")) == str(tuning_stage)
        ]
    return rows


def _select_representative_structural_cells(
    payload: Mapping[str, Any],
    *,
    family: str,
    tuning_stage: str,
    train_doc_count: int,
) -> Dict[str, Dict[str, Any]]:
    rows = [
        dict(row)
        for row in _study_focus_rows(payload, family=family, tuning_stage=tuning_stage)
        if int(row.get("train_doc_count", 0)) == int(train_doc_count)
        and str(row.get("cell_id", "")).strip()
    ]
    rows.sort(
        key=lambda row: (
            float(row.get("test_root_mae_mean", float("inf"))),
            str(row.get("cell_id", "")),
        )
    )
    if not rows:
        return {}
    easiest = rows[0]
    hardest = rows[-1]
    median = rows[len(rows) // 2]
    selected = {
        "easiest": easiest,
        "median": median,
        "hardest": hardest,
    }
    return {
        role: {
            "cell_id": str(row.get("cell_id", "")),
            "test_root_mae_mean": float(row.get("test_root_mae_mean", float("nan"))),
            "n_regimes": int(row.get("n_regimes", 0)),
            "segment_density_band": str(row.get("segment_density_band", "")),
        }
        for role, row in selected.items()
    }


def _write_combined_runs_output(
    *,
    output_root: Path,
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(runs):
        family = str(run.get("baseline_family", "run"))
        seed = int(run.get("seed", index))
        config_label = _sanitize_label(str(run.get("config_label", "")) or "default")
        stage_label = _sanitize_label(str(run.get("tuning_stage", "")) or "final")
        cell_id = _sanitize_label(str(run.get("cell_id", "") or run.get("benchmark", "")))
        study_axis = _sanitize_label(str(run.get("study_axis", "")))
        axis_value = _sanitize_label(str(run.get("axis_value", "")))
        leaf_tokens = (
            ""
            if run.get("fixed_leaf_tokens") in {"", None}
            else f"__leaf_{int(run.get('fixed_leaf_tokens', 0))}"
        )
        study_suffix = ""
        if study_axis and axis_value:
            study_suffix = f"__{study_axis}_{axis_value}"
        stem = (
            f"{family}__{cell_id}__cfg_{config_label}__stage_{stage_label}"
            f"{leaf_tokens}{study_suffix}__seed_{seed}"
        )
        (runs_dir / f"{stem}.json").write_text(
            json.dumps(dict(run), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return _write_summary_outputs(output_root)


def _render_tuning_summary_markdown(summary: Mapping[str, Any]) -> str:
    screen_rankings = list(summary.get("screen_rankings") or [])
    locked_rankings = list(summary.get("locked_rankings") or [])
    winning = dict(summary.get("winning_config") or {})
    lines = [
        "# Tree-Neural 10k Full-Law Tuning Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- train_doc_count: `{int(summary.get('train_doc_count', 0))}`",
        f"- priority_family: `{str(summary.get('priority_family', ''))}`",
        f"- dev_selection_metric: `{str(summary.get('dev_selection_metric', ''))}`",
        (
            "- test metrics hidden during config selection: "
            f"`{bool(summary.get('test_metrics_hidden_during_selection', False))}`"
        ),
        "",
        "## Screen Rankings",
        "",
        "| config | val_root_mae_mean | train_root_mae_mean | n_runs |",
        "|---|---:|---:|---:|",
    ]
    for row in screen_rankings:
        lines.append(
            "| "
            f"{str(row.get('config_label', ''))} | "
            f"{float(row.get('val_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('train_root_mae_mean', float('nan'))):.6g} | "
            f"{int(row.get('n_runs', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Locked Candidate Rankings",
            "",
            "| config | val_root_mae_mean | test_root_mae_mean | n_runs |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in locked_rankings:
        lines.append(
            "| "
            f"{str(row.get('config_label', ''))} | "
            f"{float(row.get('val_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
            f"{int(row.get('n_runs', 0))} |"
        )
    if winning:
        lines.extend(
            [
                "",
                "## Winning Config",
                "",
                f"- config_label: `{str(winning.get('config_label', ''))}`",
                f"- val_root_mae_mean: `{float(winning.get('val_root_mae_mean', float('nan'))):.6g}`",
                f"- test_root_mae_mean: `{float(winning.get('test_root_mae_mean', float('nan'))):.6g}`",
                f"- final_locked_summary_json: `{str(summary.get('final_locked_summary_json', ''))}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _render_parity_summary_markdown(summary: Mapping[str, Any]) -> str:
    parity = dict(summary.get("parity_summary") or {})
    comparisons = list(parity.get("comparisons") or [])
    lines = [
        "# Tree/FNO Fair-Parity Runner Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- gate_train_doc_count: `{int(summary.get('gate_train_doc_count', 0))}`",
        f"- parity_config_label: `{str(summary.get('parity_config_label', ''))}`",
        f"- scale_curve_backfilled: `{bool(summary.get('scale_curve_backfilled', False))}`",
        f"- primary_success_met: `{bool(parity.get('primary_success_met', False))}`",
        f"- secondary_success_met: `{bool(parity.get('secondary_success_met', False))}`",
        f"- best_full_doc_fno_family_at_gate: `{str(parity.get('best_full_doc_fno_family_at_gate', ''))}`",
        f"- best_parity_tree_family_at_gate: `{str(parity.get('best_parity_tree_family_at_gate', ''))}`",
        f"- tree_neural_gap_ratio_vs_best_fno_at_gate: `{float(parity.get('tree_neural_gap_ratio_vs_best_fno_at_gate', float('nan'))):.6g}`",
        f"- comparison_interpretation: `{str(parity.get('comparison_interpretation', ''))}`",
        "",
        "## Outputs",
        "",
        f"- gate_summary_json: `{str(summary.get('gate_summary_json', ''))}`",
        f"- final_summary_json: `{str(summary.get('final_summary_json', ''))}`",
        f"- final_summary_md: `{str(summary.get('final_summary_md', ''))}`",
    ]
    if comparisons:
        lines.extend(
            [
                "",
                "## Comparisons",
                "",
                "| train_docs | best_fno | best_fno_mae | tree_neural_mae | best_parity_tree | best_parity_tree_mae | primary<=10% | secondary<=10% |",
                "|---|---|---:|---:|---|---:|---:|---:|",
            ]
        )
        for row in comparisons:
            lines.append(
                "| "
                f"{int(row.get('train_doc_count', 0))} | "
                f"{str(row.get('best_full_doc_fno_family', ''))} | "
                f"{float(row.get('best_full_doc_fno_test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('tree_neural_test_root_mae_mean', float('nan'))):.6g} | "
                f"{str(row.get('best_parity_tree_family', ''))} | "
                f"{float(row.get('best_parity_tree_test_root_mae_mean', float('nan'))):.6g} | "
                f"{bool(row.get('primary_success_within_10pct', False))} | "
                f"{bool(row.get('secondary_success_within_10pct', False))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _format_capacity_axis_values(values: Sequence[Any]) -> str:
    formatted: List[str] = []
    for value in list(values or []):
        if value is None:
            formatted.append("none")
        elif isinstance(value, float):
            formatted.append(f"{float(value):.6g}")
        else:
            formatted.append(str(value))
    return "[" + ", ".join(formatted) + "]"


def _capacity_axis_markdown_lines(summary: Mapping[str, Any]) -> List[str]:
    return [
        f"- capacity_profile: `{str(summary.get('capacity_profile', ROOT_ONLY_CAPACITY_PROFILE_DEFAULT))}`",
        f"- width axis: `{_format_capacity_axis_values(list(summary.get('capacity_widths') or []))}`",
        f"- modes axis: `{_format_capacity_axis_values(list(summary.get('capacity_modes') or []))}`",
        f"- layers axis: `{_format_capacity_axis_values(list(summary.get('capacity_layers') or []))}`",
        f"- state_dim axis: `{_format_capacity_axis_values(list(summary.get('capacity_state_dims') or []))}`",
        f"- hidden_dim axis: `{_format_capacity_axis_values(list(summary.get('capacity_hidden_dims') or []))}`",
        f"- n_epochs axis: `{_format_capacity_axis_values(list(summary.get('capacity_n_epochs') or []))}`",
        f"- tree_training_schedule axis: `{_format_capacity_axis_values(list(summary.get('capacity_tree_training_schedules') or []))}`",
        f"- tree_checkpoint_metric axis: `{_format_capacity_axis_values(list(summary.get('capacity_tree_checkpoint_metrics') or []))}`",
        f"- tree_stage1_checkpoint_metric axis: `{_format_capacity_axis_values(list(summary.get('capacity_tree_stage1_checkpoint_metrics') or []))}`",
        f"- tree_stage1_root_weight axis: `{_format_capacity_axis_values(list(summary.get('capacity_tree_stage1_root_weights') or []))}`",
        f"- slot_count axis: `{_format_capacity_axis_values(list(summary.get('capacity_slot_counts') or []))}`",
        f"- fixed_leaf_tokens axis: `{_format_capacity_axis_values(list(summary.get('capacity_fixed_leaf_tokens') or []))}`",
    ]


def _capacity_recipe_markdown_lines(
    config_spec: Mapping[str, Any],
) -> List[str]:
    if not config_spec:
        return []
    return [
        f"- state_dim: `{int(config_spec.get('state_dim', 0))}`",
        f"- hidden_dim: `{int(config_spec.get('hidden_dim', 0))}`",
        f"- n_epochs: `{int(config_spec.get('n_epochs', 0))}`",
        f"- tree_training_schedule: `{str(config_spec.get('tree_training_schedule', ''))}`",
        f"- tree_checkpoint_metric: `{str(config_spec.get('tree_checkpoint_metric', ''))}`",
        f"- tree_stage1_checkpoint_metric: `{str(config_spec.get('tree_stage1_checkpoint_metric', ''))}`",
        f"- tree_stage1_root_weight: `{float(config_spec.get('tree_stage1_root_weight', 0.0)):.6g}`",
        f"- slot_count: `{int(config_spec.get('slot_count', 0))}`",
        f"- fixed_leaf_tokens: `{'none' if config_spec.get('fixed_leaf_tokens') is None else int(config_spec.get('fixed_leaf_tokens', 0))}`",
    ]


def _render_capacity_screen_summary_markdown(summary: Mapping[str, Any]) -> str:
    rankings = list(summary.get("screen_rankings") or [])
    lines = [
        "# Tree-FNO Capacity Screen Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- train_doc_count: `{int(summary.get('train_doc_count', 0))}`",
        f"- priority_family: `{str(summary.get('priority_family', ''))}`",
        f"- selection_metric: `{str(summary.get('selection_metric', ''))}`",
        f"- test metrics hidden during selection: `{bool(summary.get('test_metrics_hidden_during_selection', False))}`",
        *_capacity_axis_markdown_lines(summary),
        "",
        "| config | width | modes | layers | state | hidden | epochs | schedule | ckpt | s1_ckpt | s1rw | slots | leaf_tokens | val_root_mae_mean | elapsed_s_mean | n_runs |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rankings:
        lines.append(
            "| "
            f"{str(row.get('config_label', ''))} | "
            f"{int(row.get('tree_leaf_fno_width', 0))} | "
            f"{int(row.get('tree_leaf_fno_n_modes', 0))} | "
            f"{int(row.get('tree_leaf_fno_n_layers', 0))} | "
            f"{int(row.get('state_dim', 0))} | "
            f"{int(row.get('hidden_dim', 0))} | "
            f"{int(row.get('n_epochs', 0))} | "
            f"{str(row.get('tree_training_schedule', ''))} | "
            f"{str(row.get('tree_checkpoint_metric', ''))} | "
            f"{str(row.get('tree_stage1_checkpoint_metric', ''))} | "
            f"{float(row.get('tree_stage1_root_weight', 0.0)):.6g} | "
            f"{int(row.get('slot_count', 0))} | "
            f"{'none' if row.get('fixed_leaf_tokens') is None else int(row.get('fixed_leaf_tokens', 0))} | "
            f"{float(row.get('val_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('elapsed_s_mean', float('nan'))):.6g} | "
            f"{int(row.get('n_runs', 0))} |"
        )
    top_specs = dict(summary.get("top_config_specs") or {})
    if top_specs:
        lines.extend(
            [
                "",
                "## Top Config Specs",
                "",
            ]
        )
        for label, config_spec in sorted(top_specs.items()):
            lines.extend(
                [
                    f"### {str(label)}",
                    "",
                    *_capacity_recipe_markdown_lines(dict(config_spec or {})),
                    "",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def _render_capacity_locked_summary_markdown(summary: Mapping[str, Any]) -> str:
    locked_rankings = list(summary.get("locked_rankings") or [])
    winning = dict(summary.get("winning_config") or {})
    lines = [
        "# Tree-FNO Capacity Locked Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- train_doc_count: `{int(summary.get('train_doc_count', 0))}`",
        f"- priority_family: `{str(summary.get('priority_family', ''))}`",
        f"- selection_metric: `{str(summary.get('selection_metric', ''))}`",
        f"- top_k: `{int(summary.get('top_k', 0))}`",
        *_capacity_axis_markdown_lines(summary),
        f"- screen_summary_json: `{str(summary.get('screen_summary_json', ''))}`",
        f"- locked_summary_json: `{str(summary.get('locked_summary_json', ''))}`",
        "",
        "| config | width | modes | layers | state | hidden | epochs | schedule | ckpt | s1_ckpt | s1rw | slots | leaf_tokens | val_root_mae_mean | test_root_mae_mean | elapsed_s_mean | n_runs |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in locked_rankings:
        lines.append(
            "| "
            f"{str(row.get('config_label', ''))} | "
            f"{int(row.get('tree_leaf_fno_width', 0))} | "
            f"{int(row.get('tree_leaf_fno_n_modes', 0))} | "
            f"{int(row.get('tree_leaf_fno_n_layers', 0))} | "
            f"{int(row.get('state_dim', 0))} | "
            f"{int(row.get('hidden_dim', 0))} | "
            f"{int(row.get('n_epochs', 0))} | "
            f"{str(row.get('tree_training_schedule', ''))} | "
            f"{str(row.get('tree_checkpoint_metric', ''))} | "
            f"{str(row.get('tree_stage1_checkpoint_metric', ''))} | "
            f"{float(row.get('tree_stage1_root_weight', 0.0)):.6g} | "
            f"{int(row.get('slot_count', 0))} | "
            f"{'none' if row.get('fixed_leaf_tokens') is None else int(row.get('fixed_leaf_tokens', 0))} | "
            f"{float(row.get('val_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('elapsed_s_mean', float('nan'))):.6g} | "
            f"{int(row.get('n_runs', 0))} |"
        )
    if winning:
        lines.extend(
            [
                "",
                "## Winning Config",
                "",
                f"- config_label: `{str(winning.get('config_label', ''))}`",
                f"- width/modes/layers: `{int(winning.get('tree_leaf_fno_width', 0))}/{int(winning.get('tree_leaf_fno_n_modes', 0))}/{int(winning.get('tree_leaf_fno_n_layers', 0))}`",
                f"- val_root_mae_mean: `{float(winning.get('val_root_mae_mean', float('nan'))):.6g}`",
                f"- test_root_mae_mean: `{float(winning.get('test_root_mae_mean', float('nan'))):.6g}`",
                f"- elapsed_s_mean: `{float(winning.get('elapsed_s_mean', float('nan'))):.6g}`",
            ]
        )
    winning_spec = dict(summary.get("winning_config_spec") or {})
    if winning_spec:
        lines.extend(
            [
                "",
                "## Winning Config Spec",
                "",
                *_capacity_recipe_markdown_lines(winning_spec),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _render_study_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {str(summary.get('study_title', 'Tree Study Summary'))}",
        "",
        f"- study_name: `{str(summary.get('study_name', ''))}`",
        f"- study_axis: `{str(summary.get('study_axis', ''))}`",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- train_doc_count: `{int(summary.get('train_doc_count', 0))}`",
        f"- locked_tree_neural_config_label: `{str(summary.get('locked_tree_neural_config_label', ''))}`",
        f"- selection_metric: `{str(summary.get('selection_metric', ''))}`",
        (
            "- test metrics hidden during full-law tuning: "
            f"`{bool(summary.get('tuning_test_metrics_hidden_during_selection', False))}`"
        ),
    ]
    axis_values = list(summary.get("axis_values") or [])
    if axis_values:
        lines.append(f"- axis_values: `{axis_values}`")
    lines.extend(["", "## Outputs", ""])
    for key in (
        "study_summary_json",
        "summary_json",
        "summary_md",
        "screen_summary_json",
        "representative_summary_json",
    ):
        value = str(summary.get(key, "")).strip()
        if value:
            lines.append(f"- {key}: `{value}`")
    rankings = list(summary.get("rankings") or [])
    if rankings:
        lines.extend(
            [
                "",
                "## Rankings",
                "",
                "| axis_value | family | test_root_mae_mean | val_root_mae_mean | n_runs |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rankings:
            lines.append(
                "| "
                f"{str(row.get('axis_value', ''))} | "
                f"{str(row.get('baseline_family', ''))} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('val_root_mae_mean', float('nan'))):.6g} | "
                f"{int(row.get('n_runs', 0))} |"
            )
    representative_cells = dict(summary.get("representative_cells") or {})
    if representative_cells:
        lines.extend(
            [
                "",
                "## Representative Cells",
                "",
                "| role | cell_id | test_root_mae_mean | regimes | density |",
                "|---|---|---:|---:|---|",
            ]
        )
        for role in ("easiest", "median", "hardest"):
            row = dict(representative_cells.get(role) or {})
            if not row:
                continue
            lines.append(
                "| "
                f"{role} | "
                f"{str(row.get('cell_id', ''))} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{int(row.get('n_regimes', 0))} | "
                f"{str(row.get('segment_density_band', ''))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_study_summary(
    *,
    output_root: Path,
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    summary_json = output_root / "study_summary.json"
    summary_json.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md = output_root / "study_summary.md"
    summary_md.write_text(
        _render_study_summary_markdown(summary),
        encoding="utf-8",
    )
    return summary_json, summary_md


def build_tune_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    screen_configs = _tuning_grid(args)
    config_by_label = {str(config.label): config for config in screen_configs}
    screen_root = output_root / "screen"
    screen_jobs = _build_jobs_for_configs(
        families=(str(args.priority_family),),
        train_doc_counts=(int(args.train_doc_count),),
        benchmark=str(args.benchmark),
        hardness_grid="",
        grid_cell_ids=(),
        seeds=[int(seed) for seed in args.screen_seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=True,
        configs=screen_configs,
        tuning_stage="screen",
        test_metrics_hidden_during_selection=True,
    )
    comparison_root = output_root / "comparison"
    comparison_config = _RunConfigSpec(
        label="comparison_default",
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.comparison_n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.comparison_lr),
        weight_decay=float(args.weight_decay),
        tree_local_law_weight=float(args.comparison_tree_local_law_weight),
        tree_task_objective_weight=(
            None
            if args.tree_task_objective_weight is None
            else float(args.tree_task_objective_weight)
        ),
        doc_sequence_train_fraction=float(args.doc_sequence_train_fraction),
    )
    comparison_jobs = _build_jobs_for_configs(
        families=[str(family) for family in args.comparison_families],
        train_doc_counts=(int(args.train_doc_count),),
        benchmark=str(args.benchmark),
        hardness_grid="",
        grid_cell_ids=(),
        seeds=[int(seed) for seed in args.locked_seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=bool(args.repeat_closed_form_controls),
        configs=(comparison_config,),
        tuning_stage="comparison",
        test_metrics_hidden_during_selection=False,
    )
    return {
        "output_root": output_root,
        "screen_root": screen_root,
        "screen_jobs": screen_jobs,
        "comparison_root": comparison_root,
        "comparison_jobs": comparison_jobs,
        "screen_configs": screen_configs,
        "config_by_label": config_by_label,
        "comparison_config": comparison_config,
    }


def build_tune_locked_job_bundle(
    args: argparse.Namespace,
    *,
    locked_configs: Sequence[_RunConfigSpec],
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    locked_root = output_root / "locked"
    locked_jobs = _build_jobs_for_configs(
        families=(str(args.priority_family),),
        train_doc_counts=(int(args.train_doc_count),),
        benchmark=str(args.benchmark),
        hardness_grid="",
        grid_cell_ids=(),
        seeds=[int(seed) for seed in args.locked_seeds],
        job_granularity=str(args.job_granularity),
        repeat_closed_form_controls=True,
        configs=list(locked_configs),
        tuning_stage="locked",
        test_metrics_hidden_during_selection=True,
    )
    return {
        "locked_root": locked_root,
        "locked_jobs": locked_jobs,
    }


def finalize_tune_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    screen_root: Path,
    comparison_root: Path,
    locked_root: Path,
    screen_rankings: Sequence[Mapping[str, Any]],
    config_by_label: Mapping[str, _RunConfigSpec],
) -> Dict[str, Any]:
    _load_or_write_summary_outputs(screen_root)
    locked_payload = _load_or_write_summary_outputs(locked_root)
    locked_rankings = _select_top_config_rows(
        locked_payload,
        baseline_family=str(args.priority_family),
        tuning_stage="locked",
        train_doc_count=int(args.train_doc_count),
        metric_key="val_root_mae_mean",
        top_k=max(int(args.top_k), 1),
    )
    if not locked_rankings:
        raise RuntimeError("locked stage produced no ranked configs")
    winning = dict(locked_rankings[0])
    winning_label = str(winning.get("config_label", ""))

    comparison_payload = _load_or_write_summary_outputs(comparison_root)
    winning_runs = [
        dict(run)
        for run in list(locked_payload.get("runs") or [])
        if str(run.get("baseline_family", "")) == str(args.priority_family)
        and str(run.get("tuning_stage", "")) == "locked"
        and str(run.get("config_label", "")) == winning_label
    ]
    comparison_runs = [dict(run) for run in list(comparison_payload.get("runs") or [])]
    final_locked_root = output_root / "final_locked"
    final_payload = _write_combined_runs_output(
        output_root=final_locked_root,
        runs=[*comparison_runs, *winning_runs],
    )

    summary = {
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.train_doc_count),
        "priority_family": str(args.priority_family),
        "comparison_families": [str(family) for family in args.comparison_families],
        "dev_selection_metric": "val_root_mae_mean",
        "primary_report_metric": "test_root_mae_mean",
        "test_metrics_hidden_during_selection": True,
        "screen_rankings": list(screen_rankings),
        "locked_rankings": locked_rankings,
        "winning_config": winning,
        "winning_config_label": winning_label,
        "winning_config_spec": (
            asdict(config_by_label[winning_label]) if winning_label in config_by_label else {}
        ),
        "screen_summary_json": str(screen_root / "summary.json"),
        "comparison_summary_json": str(comparison_root / "summary.json"),
        "locked_summary_json": str(locked_root / "summary.json"),
        "final_locked_summary_json": str(final_locked_root / "summary.json"),
        "final_locked_summary_md": str(final_locked_root / "summary.md"),
        "final_locked_payload_benchmark": str(final_payload.get("benchmark", "")),
    }
    summary_json = output_root / "tuning_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md = output_root / "tuning_summary.md"
    summary_md.write_text(_render_tuning_summary_markdown(summary), encoding="utf-8")
    return {
        "output_root": str(output_root),
        "tuning_summary_json": str(summary_json),
        "tuning_summary_md": str(summary_md),
        "screen_summary_json": str(screen_root / "summary.json"),
        "comparison_summary_json": str(comparison_root / "summary.json"),
        "locked_summary_json": str(locked_root / "summary.json"),
        "final_locked_summary_json": str(final_locked_root / "summary.json"),
        "winning_config_label": winning_label,
    }


def build_study_job_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    tuning_root = Path(str(args.tuning_root))
    locked_tree_config, tuning_summary = _locked_tree_neural_config_from_tuning_root(
        tuning_root
    )
    comparison_config = _comparison_study_config(args)
    locked_label = str(locked_tree_config.label)
    selection_metric = str(
        tuning_summary.get("dev_selection_metric", "val_root_mae_mean")
    )
    tuning_hidden = bool(
        tuning_summary.get("test_metrics_hidden_during_selection", False)
    )
    return {
        "output_root": output_root,
        "tuning_root": tuning_root,
        "locked_tree_config": locked_tree_config,
        "comparison_config": comparison_config,
        "locked_label": locked_label,
        "selection_metric": selection_metric,
        "tuning_hidden": tuning_hidden,
    }


def finalize_leaf_geometry_study_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    locked_label: str,
    selection_metric: str,
    tuning_hidden: bool,
    axis_values: Sequence[int],
) -> Dict[str, Any]:
    payload = _write_summary_outputs(output_root)
    rankings = sorted(
        [
            {
                "axis_value": row.get("axis_value", ""),
                "baseline_family": str(row.get("baseline_family", "")),
                "test_root_mae_mean": float(
                    row.get("test_root_mae_mean", float("nan"))
                ),
                "val_root_mae_mean": float(
                    row.get("val_root_mae_mean", float("nan"))
                ),
                "n_runs": int(row.get("n_runs", 0)),
            }
            for row in list(payload.get("aggregate_rows") or [])
        ],
        key=lambda row: (
            str(row.get("axis_value", "")),
            str(row.get("baseline_family", "")),
        ),
    )
    study_summary = {
        "study_title": "Tree Leaf Geometry Study Summary",
        "study_name": "leaf_geometry",
        "study_axis": "fixed_leaf_tokens",
        "axis_values": [int(value) for value in axis_values],
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.train_doc_count),
        "locked_tree_neural_config_label": locked_label,
        "selection_metric": selection_metric,
        "tuning_test_metrics_hidden_during_selection": tuning_hidden,
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "rankings": rankings,
    }
    study_summary_json, study_summary_md = _write_study_summary(
        output_root=output_root,
        summary=study_summary,
    )
    return {
        "output_root": str(output_root),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "study_summary_json": str(study_summary_json),
        "study_summary_md": str(study_summary_md),
    }


def build_structural_study_representative_job_bundle(
    args: argparse.Namespace,
    *,
    locked_tree_config: _RunConfigSpec,
    comparison_config: _RunConfigSpec,
    locked_label: str,
    selection_metric: str,
    representative_cells: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    comparison_families = [
        str(family) for family in args.families if str(family) != "tree_neural"
    ]
    representative_jobs: List[_JobSpec] = []
    for cell_id in [str(details.get("cell_id", "")) for details in representative_cells.values()]:
        representative_jobs.extend(
            _build_jobs_for_configs(
                families=("tree_neural",),
                train_doc_counts=(int(args.train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="structural_core_v1",
                grid_cell_ids=(cell_id,),
                seeds=[int(seed) for seed in args.locked_seeds],
                job_granularity=str(args.job_granularity),
                repeat_closed_form_controls=True,
                configs=(locked_tree_config,),
                tuning_stage="study_representative",
                study_name="structural_complexity",
                study_axis="structural_core_cell",
                axis_value=str(cell_id),
                locked_tree_neural_config_label=locked_label,
                selection_metric=selection_metric,
            )
        )
        if comparison_families:
            representative_jobs.extend(
                _build_jobs_for_configs(
                    families=comparison_families,
                    train_doc_counts=(int(args.train_doc_count),),
                    benchmark=str(args.benchmark),
                    hardness_grid="structural_core_v1",
                    grid_cell_ids=(cell_id,),
                    seeds=[int(seed) for seed in args.locked_seeds],
                    job_granularity=str(args.job_granularity),
                    repeat_closed_form_controls=bool(args.repeat_closed_form_controls),
                    configs=(comparison_config,),
                    tuning_stage="study_representative",
                    study_name="structural_complexity",
                    study_axis="structural_core_cell",
                    axis_value=str(cell_id),
                    locked_tree_neural_config_label=locked_label,
                    selection_metric=selection_metric,
                )
            )
    return {
        "representative_root": output_root / "representative",
        "representative_jobs": representative_jobs,
    }


def finalize_structural_study_output(
    *,
    args: argparse.Namespace,
    output_root: Path,
    locked_label: str,
    selection_metric: str,
    tuning_hidden: bool,
    representative_cells: Mapping[str, Mapping[str, Any]],
    cell_ids: Sequence[str],
) -> Dict[str, Any]:
    _load_or_write_summary_outputs(output_root / "screen")
    _load_or_write_summary_outputs(output_root / "representative")
    study_summary = {
        "study_title": "Tree Structural Complexity Study Summary",
        "study_name": "structural_complexity",
        "study_axis": "structural_core_cell",
        "axis_values": list(cell_ids),
        "benchmark": str(args.benchmark),
        "train_doc_count": int(args.train_doc_count),
        "locked_tree_neural_config_label": locked_label,
        "selection_metric": selection_metric,
        "tuning_test_metrics_hidden_during_selection": tuning_hidden,
        "representative_cells": representative_cells,
        "screen_summary_json": str(output_root / "screen" / "summary.json"),
        "representative_summary_json": str(output_root / "representative" / "summary.json"),
    }
    study_summary_json, study_summary_md = _write_study_summary(
        output_root=output_root,
        summary=study_summary,
    )
    return {
        "output_root": str(output_root),
        "screen_summary_json": str(output_root / "screen" / "summary.json"),
        "representative_summary_json": str(output_root / "representative" / "summary.json"),
        "study_summary_json": str(study_summary_json),
        "study_summary_md": str(study_summary_md),
    }


def _scheduler_max_slots(
    args: argparse.Namespace,
    *,
    screen_workers_per_mig: int = 1,
) -> int:
    return max(
        1,
        int(getattr(args, "max_gpu_items_per_mig", 1) or 1),
        int(max(1, int(screen_workers_per_mig))),
    )


def _gpu_slots_for_workers_per_mig(
    *,
    scheduler_max: int,
    workers_per_mig: int,
) -> int:
    # Each Markov fit is still a single-device worker. Oversubscription belongs in the
    # scheduler's per-device slot pool, not in the per-item visible device list.
    # Reserving multiple MIG tokens for one fit only wastes devices because the worker
    # still trains on cuda:0 inside its local CUDA_VISIBLE_DEVICES namespace.
    _ = int(scheduler_max)
    _ = int(workers_per_mig)
    return 1


def _discover_scheduler_devices(args: argparse.Namespace) -> List[str]:
    mig_uuids = (
        _parse_mig_uuids(args.mig_uuids)
        if str(getattr(args, "mig_uuids", "")).strip()
        else _discover_mig_uuids()
    )
    if not mig_uuids:
        raise RuntimeError(
            "No MIG UUIDs discovered. Pass --mig-uuids explicitly or configure MIGs first."
        )
    return list(mig_uuids)


def _build_scheduler_graph(
    args: argparse.Namespace,
    *,
    output_root: Path,
    mig_uuids: Sequence[str],
) -> Dict[str, Any]:
    mode = str(args.mode or "controller")
    items: List[SchedulerItem] = []
    scheduler_max = _scheduler_max_slots(args)
    manifest_payload: Dict[str, Any] = {
        "mode": mode,
        "mig_uuids": [str(token) for token in mig_uuids],
        "scheduler_mode": str(getattr(args, "scheduler_mode", "global_per_run")),
        "job_granularity": str(getattr(args, "job_granularity", "family_train_seed")),
        "cleanup_stale_children": bool(getattr(args, "cleanup_stale_children", True)),
        "max_gpu_items_per_mig": int(getattr(args, "max_gpu_items_per_mig", 1) or 1),
        "scheduler_launch_stagger_seconds": float(
            max(0.0, float(getattr(args, "scheduler_launch_stagger_seconds", 0.0)))
        ),
    }

    if mode == "controller":
        bundle = build_controller_job_bundle(args)
        manifest_payload.update(dict(bundle["manifest_payload"]))
        gpu_ids: List[str] = []
        for job in list(bundle["jobs"]):
            item = _scheduler_item_for_job(
                phase="controller",
                item_id=f"controller::{job.job_name}",
                output_root=output_root,
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=1,
            )
            gpu_ids.append(str(item.item_id))
            items.append(item)

        items.append(
            SchedulerItem(
                item_id="controller::reduce",
                phase="controller",
                kind="cpu_callback",
                deps=tuple(gpu_ids),
                expected_outputs=(str(output_root / "summary.json"),),
                callback=lambda: {"result": dict(finalize_controller_output(output_root))},
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "exact_sanity":
        bundle = build_exact_sanity_job_bundle(args)
        manifest_payload.update(dict(bundle["manifest_payload"]))
        gpu_ids: List[str] = []
        for job in list(bundle["jobs"]):
            item = _scheduler_item_for_job(
                phase="exact_sanity",
                item_id=f"exact_sanity::{job.job_name}",
                output_root=output_root,
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=1,
            )
            gpu_ids.append(str(item.item_id))
            items.append(item)

        items.append(
            SchedulerItem(
                item_id="exact_sanity::reduce",
                phase="exact_sanity",
                kind="cpu_callback",
                deps=tuple(gpu_ids),
                expected_outputs=(str(output_root / "tree_neural_exact_sanity_summary.json"),),
                callback=lambda: {"result": dict(finalize_exact_sanity_output(output_root))},
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "representation_sufficiency":
        screen_workers_per_mig = (
            int(getattr(args, "gpu_runtime_capacity_workers_per_mig", 2) or 1)
            if bool(getattr(args, "gpu_runtime_allow_multi_worker_screen", True))
            else 1
        )
        scheduler_max = _scheduler_max_slots(
            args,
            screen_workers_per_mig=screen_workers_per_mig,
        )
        screen_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=int(screen_workers_per_mig),
        )
        locked_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=1,
        )
        bundle = build_representation_sufficiency_screen_job_bundle(args)
        manifest_payload.update(
            {
                **dict(bundle["screen_manifest_payload"]),
                "benchmark": str(args.benchmark),
                "screen_train_doc_count": int(args.screen_train_doc_count),
                "lock_train_doc_count": int(args.lock_train_doc_count),
                "promotion_train_doc_count": int(args.promotion_train_doc_count),
                "screen_seeds": [int(seed) for seed in args.screen_seeds],
                "lock_seeds": [int(seed) for seed in args.lock_seeds],
                "promotion_seeds": [int(seed) for seed in args.promotion_seeds],
                "top_k": int(args.top_k),
            }
        )
        screen_item_ids: List[str] = []
        for job in list(bundle["screen_jobs"]):
            item = _scheduler_item_for_job(
                phase="representation_sufficiency",
                item_id=f"representation_sufficiency::screen::{job.job_name}",
                output_root=bundle["screen_root"],
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=int(screen_slots),
            )
            screen_item_ids.append(str(item.item_id))
            items.append(item)

        def _representation_screen_reduce() -> Mapping[str, Any]:
            screen_result = finalize_representation_sufficiency_screen_output(
                args=args,
                output_root=output_root,
                screen_root=bundle["screen_root"],
                config_by_label=bundle["config_by_label"],
                slotwise_control_labels_by_state_dim=bundle[
                    "slotwise_control_labels_by_state_dim"
                ],
                official_fno_label=str(bundle["official_fno_label"]),
            )
            lock_bundle = build_representation_sufficiency_lock_job_bundle(
                args,
                locked_configs=screen_result["locked_configs"],
                official_fno_label=str(bundle["official_fno_label"]),
            )
            lock_item_ids: List[str] = []
            new_items: List[SchedulerItem] = []
            for job in list(lock_bundle["locked_jobs"]):
                item = _scheduler_item_for_job(
                    phase="representation_sufficiency",
                    item_id=f"representation_sufficiency::lock::{job.job_name}",
                    output_root=lock_bundle["locked_root"],
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=int(locked_slots),
                )
                lock_item_ids.append(str(item.item_id))
                new_items.append(item)

            def _representation_lock_reduce() -> Mapping[str, Any]:
                lock_result = finalize_representation_sufficiency_lock_output(
                    args=args,
                    output_root=output_root,
                    locked_root=lock_bundle["locked_root"],
                    config_by_label=bundle["config_by_label"],
                    slotwise_control_labels_by_state_dim=bundle[
                        "slotwise_control_labels_by_state_dim"
                    ],
                    official_fno_label=str(bundle["official_fno_label"]),
                )
                promotion_bundle = (
                    build_representation_sufficiency_promotion_job_bundle(
                        args,
                        promotion_configs=lock_result["promotion_configs"],
                        official_fno_label=str(bundle["official_fno_label"]),
                    )
                )
                promotion_item_ids: List[str] = []
                promotion_items: List[SchedulerItem] = []
                for job in list(promotion_bundle["promotion_jobs"]):
                    item = _scheduler_item_for_job(
                        phase="representation_sufficiency",
                        item_id=f"representation_sufficiency::promotion::{job.job_name}",
                        output_root=promotion_bundle["promotion_root"],
                        job=job,
                        torch_threads=int(args.torch_threads),
                        use_cuda=bool(args.use_cuda),
                        gpu_slots=int(locked_slots),
                    )
                    promotion_item_ids.append(str(item.item_id))
                    promotion_items.append(item)

                def _representation_promotion_reduce() -> Mapping[str, Any]:
                    final = finalize_representation_sufficiency_output(
                        args=args,
                        output_root=output_root,
                        promotion_root=promotion_bundle["promotion_root"],
                        config_by_label=bundle["config_by_label"],
                        screen_summary=screen_result["screen_summary"],
                        lock_summary=lock_result["lock_summary"],
                        official_fno_label=str(bundle["official_fno_label"]),
                    )
                    return {"result": dict(final)}

                promotion_items.append(
                    SchedulerItem(
                        item_id="representation_sufficiency::promotion::reduce",
                        phase="representation_sufficiency",
                        kind="cpu_callback",
                        deps=tuple(promotion_item_ids),
                        expected_outputs=(
                            str(
                                output_root
                                / "tree_neural_representation_sufficiency_summary.json"
                            ),
                        ),
                        callback=_representation_promotion_reduce,
                        reuse_existing=False,
                        run_on_failed_dependencies=True,
                    )
                )
                return {
                    "new_items": promotion_items,
                    "result": {
                        "lock_summary_json": str(lock_result["lock_summary_json"]),
                        "winner_label": str(lock_result["winner_label"]),
                    },
                }

            new_items.append(
                SchedulerItem(
                    item_id="representation_sufficiency::lock::reduce",
                    phase="representation_sufficiency",
                    kind="cpu_callback",
                    deps=tuple(lock_item_ids),
                    expected_outputs=(
                        str(output_root / "representation_sufficiency_lock_summary.json"),
                    ),
                    callback=_representation_lock_reduce,
                    reuse_existing=False,
                    run_on_failed_dependencies=True,
                )
            )
            return {
                "new_items": new_items,
                "result": {
                    "screen_summary_json": str(screen_result["screen_summary_json"]),
                },
            }

        items.append(
            SchedulerItem(
                item_id="representation_sufficiency::screen::reduce",
                phase="representation_sufficiency",
                kind="cpu_callback",
                deps=tuple(screen_item_ids),
                expected_outputs=(
                    str(output_root / "representation_sufficiency_screen_summary.json"),
                ),
                callback=_representation_screen_reduce,
                reuse_existing=False,
                run_on_failed_dependencies=True,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "representation_learnability":
        screen_workers_per_mig = (
            int(getattr(args, "gpu_runtime_capacity_workers_per_mig", 2) or 1)
            if bool(getattr(args, "gpu_runtime_allow_multi_worker_screen", True))
            else 1
        )
        scheduler_max = _scheduler_max_slots(
            args,
            screen_workers_per_mig=screen_workers_per_mig,
        )
        screen_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=int(screen_workers_per_mig),
        )
        locked_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=1,
        )
        benchmark_specs = _representation_learnability_benchmark_specs(args)
        bundle = build_representation_learnability_winner_job_bundle(args)
        manifest_payload.update(
            {
                **dict(bundle["winner_manifest_payload"]),
                "winner_train_doc_count": int(args.winner_train_doc_count),
                "winner_seeds": [int(seed) for seed in args.winner_seeds],
                "sweep_train_doc_counts": [
                    int(value) for value in args.sweep_train_doc_counts
                ],
                "sweep_seeds": [int(seed) for seed in args.sweep_seeds],
                "benchmark_cells": [
                    str(spec.cell_id or spec.name) for spec in benchmark_specs
                ],
                "benchmark_names": [str(spec.name) for spec in benchmark_specs],
            }
        )
        winner_item_ids: List[str] = []
        for job in list(bundle["winner_jobs"]):
            item = _scheduler_item_for_job(
                phase="representation_learnability",
                item_id=f"representation_learnability::winner::{job.job_name}",
                output_root=bundle["winner_root"],
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=int(screen_slots),
            )
            winner_item_ids.append(str(item.item_id))
            items.append(item)

        def _representation_learnability_winner_reduce() -> Mapping[str, Any]:
            winner_result = finalize_representation_learnability_winner_output(
                args=args,
                output_root=output_root,
                winner_root=bundle["winner_root"],
                config_by_label=bundle["config_by_label"],
                slotwise_control_labels_by_state_dim=bundle[
                    "slotwise_control_labels_by_state_dim"
                ],
                official_fno_label=str(bundle["official_fno_label"]),
            )
            sweep_root = output_root / "sweep"
            sweep_jobs = _representation_learnability_jobs_for_configs(
                args=args,
                configs=winner_result["selected_configs"],
                benchmarks=benchmark_specs,
                train_doc_counts=[
                    int(value) for value in args.sweep_train_doc_counts
                ],
                seeds=[int(seed) for seed in args.sweep_seeds],
                tuning_stage=REPRESENTATION_LEARNABILITY_SWEEP_STAGE,
                official_fno_label=str(bundle["official_fno_label"]),
            )
            sweep_item_ids: List[str] = []
            new_items: List[SchedulerItem] = []
            for job in sweep_jobs:
                item = _scheduler_item_for_job(
                    phase="representation_learnability",
                    item_id=f"representation_learnability::sweep::{job.job_name}",
                    output_root=sweep_root,
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=int(locked_slots),
                )
                sweep_item_ids.append(str(item.item_id))
                new_items.append(item)

            def _representation_learnability_sweep_reduce() -> Mapping[str, Any]:
                final = finalize_representation_learnability_output(
                    args=args,
                    output_root=output_root,
                    sweep_root=sweep_root,
                    config_by_label=bundle["config_by_label"],
                    winner_summary=winner_result["winner_summary"],
                    winner_label=str(winner_result["winner_label"]),
                    matched_control_label=str(winner_result["matched_control_label"]),
                    official_fno_label=str(bundle["official_fno_label"]),
                )
                return {"result": dict(final)}

            new_items.append(
                SchedulerItem(
                    item_id="representation_learnability::sweep::reduce",
                    phase="representation_learnability",
                    kind="cpu_callback",
                    deps=tuple(sweep_item_ids),
                    expected_outputs=(
                        str(
                            output_root
                            / "tree_neural_representation_learnability_summary.json"
                        ),
                    ),
                    callback=_representation_learnability_sweep_reduce,
                    reuse_existing=False,
                    run_on_failed_dependencies=True,
                )
            )
            return {
                "new_items": new_items,
                "result": {
                    "winner_summary_json": str(
                        winner_result["winner_summary_json"]
                    ),
                    "winner_label": str(winner_result["winner_label"]),
                },
            }

        items.append(
            SchedulerItem(
                item_id="representation_learnability::winner::reduce",
                phase="representation_learnability",
                kind="cpu_callback",
                deps=tuple(winner_item_ids),
                expected_outputs=(
                    str(output_root / "representation_learnability_winner_summary.json"),
                ),
                callback=_representation_learnability_winner_reduce,
                reuse_existing=False,
                run_on_failed_dependencies=True,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "budget_frontier":
        bundle = build_budget_frontier_job_bundle(args)
        manifest_payload.update(dict(bundle["manifest_payload"]))
        gpu_ids: List[str] = []
        for job in list(bundle["jobs"]):
            item = _scheduler_item_for_job(
                phase="budget_frontier",
                item_id=f"budget_frontier::{job.job_name}",
                output_root=output_root,
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=1,
            )
            gpu_ids.append(str(item.item_id))
            items.append(item)

        items.append(
            SchedulerItem(
                item_id="budget_frontier::reduce",
                phase="budget_frontier",
                kind="cpu_callback",
                deps=tuple(gpu_ids),
                expected_outputs=(str(output_root / "tree_oracle_budget_frontier_summary.json"),),
                callback=lambda: {"result": dict(finalize_budget_frontier_output(output_root))},
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "parity":
        bundle = build_parity_job_bundle(args)
        manifest_payload.update(
            {
                "benchmark": str(args.benchmark),
                "gate_train_doc_count": int(args.gate_train_doc_count),
                "scale_train_doc_counts": [int(value) for value in args.scale_train_doc_counts],
                "parity_tree_config": asdict(bundle["parity_tree_config"]),
                "reference_fno_config": asdict(bundle["reference_fno_config"]),
                "capacity_root": str(bundle["capacity_root"]),
            }
        )
        all_gpu_ids: List[str] = []
        for prefix, root_key, jobs_key in (
            ("gate", "gate_root", "gate_jobs"),
            ("upper", "upper_bound_root", "upper_bound_jobs"),
            ("backfill", "backfill_root", "backfill_jobs"),
        ):
            for job in list(bundle[jobs_key]):
                item = _scheduler_item_for_job(
                    phase="parity",
                    item_id=f"parity::{prefix}::{job.job_name}",
                    output_root=Path(str(bundle[root_key])),
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=1,
                )
                all_gpu_ids.append(str(item.item_id))
                items.append(item)

        def _parity_reduce() -> Mapping[str, Any]:
            result = finalize_parity_output(
                args=args,
                output_root=output_root,
                gate_failed_jobs=0,
                upper_bound_failed_jobs=0,
                backfill_failed_jobs=0,
                parity_tree_config=bundle["parity_tree_config"],
                reference_fno_config=bundle["reference_fno_config"],
                parity_tree_families=bundle["parity_tree_families"],
                parity_fno_families=bundle["parity_fno_families"],
                parity_comparison_families=bundle["parity_comparison_families"],
                capacity_root_value=str(bundle["capacity_root"]),
            )
            return {"result": dict(result)}

        items.append(
            SchedulerItem(
                item_id="parity::reduce",
                phase="parity",
                kind="cpu_callback",
                deps=tuple(all_gpu_ids),
                expected_outputs=(str(output_root / "fair_parity_run_summary.json"),),
                callback=_parity_reduce,
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "capacity":
        screen_workers_per_mig = (
            int(getattr(args, "gpu_runtime_capacity_workers_per_mig", 2) or 1)
            if bool(getattr(args, "gpu_runtime_allow_multi_worker_screen", True))
            else 1
        )
        scheduler_max = _scheduler_max_slots(
            args,
            screen_workers_per_mig=screen_workers_per_mig,
        )
        screen_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=int(screen_workers_per_mig),
        )
        locked_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=1,
        )
        bundle = _cached_capacity_screen_job_bundle(args)
        manifest_payload.update(dict(bundle["screen_manifest_payload"]))
        screen_item_ids: List[str] = []
        for job in list(bundle["screen_jobs"]):
            item = _scheduler_item_for_job(
                phase="capacity",
                item_id=f"capacity::screen::{job.job_name}",
                output_root=bundle["screen_root"],
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=int(screen_slots),
                allowed_devices=tuple(bundle.get("screen_allowed_devices") or ()),
            )
            screen_item_ids.append(str(item.item_id))
            items.append(item)

        def _capacity_screen_reduce() -> Mapping[str, Any]:
            result = finalize_capacity_screen_output(
                args=args,
                output_root=output_root,
                screen_root=bundle["screen_root"],
                config_by_label=bundle["config_by_label"],
            )
            locked_bundle = build_capacity_locked_job_bundle(
                args,
                locked_configs=result["locked_configs"],
            )
            locked_item_ids: List[str] = []
            new_items: List[SchedulerItem] = []
            for job in list(locked_bundle["locked_jobs"]):
                item = _scheduler_item_for_job(
                    phase="capacity",
                    item_id=f"capacity::locked::{job.job_name}",
                    output_root=locked_bundle["locked_root"],
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=int(locked_slots),
                )
                locked_item_ids.append(str(item.item_id))
                new_items.append(item)

            def _capacity_locked_reduce() -> Mapping[str, Any]:
                final = finalize_capacity_locked_output(
                    args=args,
                    output_root=output_root,
                    screen_root=bundle["screen_root"],
                    locked_root=locked_bundle["locked_root"],
                    screen_rankings=result["top_rankings"],
                    config_by_label=bundle["config_by_label"],
                )
                return {"result": dict(final)}

            new_items.append(
                SchedulerItem(
                    item_id="capacity::locked::reduce",
                    phase="capacity",
                    kind="cpu_callback",
                    deps=tuple(locked_item_ids),
                    expected_outputs=(str(output_root / "tree_fno_capacity_locked_summary.json"),),
                    callback=_capacity_locked_reduce,
                    reuse_existing=False,
                )
            )
            return {"new_items": new_items, "result": dict(result)}

        items.append(
            SchedulerItem(
                item_id="capacity::screen::reduce",
                phase="capacity",
                kind="cpu_callback",
                deps=tuple(screen_item_ids),
                expected_outputs=(str(output_root / "tree_fno_capacity_screen_summary.json"),),
                callback=_capacity_screen_reduce,
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "tune":
        screen_workers_per_mig = (
            int(getattr(args, "gpu_runtime_capacity_workers_per_mig", 2) or 1)
            if bool(getattr(args, "gpu_runtime_allow_multi_worker_screen", True))
            else 1
        )
        scheduler_max = _scheduler_max_slots(
            args,
            screen_workers_per_mig=screen_workers_per_mig,
        )
        screen_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=int(screen_workers_per_mig),
        )
        locked_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=1,
        )
        bundle = build_tune_job_bundle(args)
        manifest_payload.update(
            {
                "benchmark": str(args.benchmark),
                "train_doc_count": int(args.train_doc_count),
                "priority_family": str(args.priority_family),
                "comparison_families": [str(family) for family in args.comparison_families],
                "dev_selection_metric": "val_root_mae_mean",
                "test_metrics_hidden_during_selection": True,
            }
        )
        comparison_item_ids: List[str] = []
        screen_item_ids: List[str] = []
        for job in list(bundle["screen_jobs"]):
            item = _scheduler_item_for_job(
                phase="tune",
                item_id=f"tune::screen::{job.job_name}",
                output_root=bundle["screen_root"],
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=int(screen_slots),
            )
            screen_item_ids.append(str(item.item_id))
            items.append(item)
        for job in list(bundle["comparison_jobs"]):
            item = _scheduler_item_for_job(
                phase="tune",
                item_id=f"tune::comparison::{job.job_name}",
                output_root=bundle["comparison_root"],
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=int(locked_slots),
            )
            comparison_item_ids.append(str(item.item_id))
            items.append(item)

        def _tune_screen_reduce() -> Mapping[str, Any]:
            screen_payload = _load_or_write_summary_outputs(bundle["screen_root"])
            screen_rankings = _select_top_config_rows(
                screen_payload,
                baseline_family=str(args.priority_family),
                tuning_stage="screen",
                train_doc_count=int(args.train_doc_count),
                metric_key="val_root_mae_mean",
                top_k=max(int(args.top_k), 1),
            )
            if not screen_rankings:
                raise RuntimeError("screen stage produced no ranked configs")
            locked_configs = [
                bundle["config_by_label"][str(row.get("config_label", ""))]
                for row in screen_rankings
                if str(row.get("config_label", "")) in bundle["config_by_label"]
            ]
            locked_bundle = build_tune_locked_job_bundle(
                args,
                locked_configs=locked_configs,
            )
            new_items: List[SchedulerItem] = []
            locked_item_ids: List[str] = []
            for job in list(locked_bundle["locked_jobs"]):
                item = _scheduler_item_for_job(
                    phase="tune",
                    item_id=f"tune::locked::{job.job_name}",
                    output_root=locked_bundle["locked_root"],
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=int(locked_slots),
                )
                locked_item_ids.append(str(item.item_id))
                new_items.append(item)

            def _tune_reduce() -> Mapping[str, Any]:
                final = finalize_tune_output(
                    args=args,
                    output_root=output_root,
                    screen_root=bundle["screen_root"],
                    comparison_root=bundle["comparison_root"],
                    locked_root=locked_bundle["locked_root"],
                    screen_rankings=screen_rankings,
                    config_by_label=bundle["config_by_label"],
                )
                return {"result": dict(final)}

            new_items.append(
                SchedulerItem(
                    item_id="tune::reduce",
                    phase="tune",
                    kind="cpu_callback",
                    deps=tuple([*comparison_item_ids, *locked_item_ids]),
                    expected_outputs=(str(output_root / "tuning_summary.json"),),
                    callback=_tune_reduce,
                    reuse_existing=False,
                )
            )
            return {"new_items": new_items, "result": {"screen_rankings": list(screen_rankings)}}

        items.append(
            SchedulerItem(
                item_id="tune::screen::reduce",
                phase="tune",
                kind="cpu_callback",
                deps=tuple(screen_item_ids),
                expected_outputs=(),
                callback=_tune_screen_reduce,
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    if mode == "study":
        bundle = build_study_job_bundle(args)
        screen_workers_per_mig = (
            int(getattr(args, "gpu_runtime_capacity_workers_per_mig", 2) or 1)
            if bool(getattr(args, "gpu_runtime_allow_multi_worker_screen", True))
            else 1
        )
        scheduler_max = _scheduler_max_slots(
            args,
            screen_workers_per_mig=screen_workers_per_mig,
        )
        screen_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=int(screen_workers_per_mig),
        )
        locked_slots = _gpu_slots_for_workers_per_mig(
            scheduler_max=int(scheduler_max),
            workers_per_mig=1,
        )
        manifest_payload.update(
            {
                "study_name": str(args.study_name),
                "benchmark": str(args.benchmark),
                "train_doc_count": int(args.train_doc_count),
                "families": [str(family) for family in args.families],
                "tuning_root": str(bundle["tuning_root"]),
                "locked_tree_neural_config_label": str(bundle["locked_label"]),
                "selection_metric": str(bundle["selection_metric"]),
                "tuning_test_metrics_hidden_during_selection": bool(bundle["tuning_hidden"]),
            }
        )

        if str(args.study_name) == "leaf_geometry":
            axis_values = [int(value) for value in args.leaf_tokens]
            jobs: List[_JobSpec] = []
            for leaf_tokens in axis_values:
                tree_config = _RunConfigSpec(
                    **{**asdict(bundle["locked_tree_config"]), "fixed_leaf_tokens": int(leaf_tokens)}
                )
                comparison_leaf_config = _RunConfigSpec(
                    **{**asdict(bundle["comparison_config"]), "fixed_leaf_tokens": int(leaf_tokens)}
                )
                jobs.extend(
                    _build_jobs_for_configs(
                        families=("tree_neural",),
                        train_doc_counts=(int(args.train_doc_count),),
                        benchmark=str(args.benchmark),
                        hardness_grid="",
                        grid_cell_ids=(),
                        seeds=[int(seed) for seed in args.seeds],
                        job_granularity=str(args.job_granularity),
                        repeat_closed_form_controls=True,
                        configs=(tree_config,),
                        tuning_stage="study_locked",
                        study_name="leaf_geometry",
                        study_axis="fixed_leaf_tokens",
                        axis_value=str(int(leaf_tokens)),
                        locked_tree_neural_config_label=str(bundle["locked_label"]),
                        selection_metric=str(bundle["selection_metric"]),
                    )
                )
                comparison_families = [
                    str(family)
                    for family in args.families
                    if str(family) != "tree_neural"
                ]
                if comparison_families:
                    jobs.extend(
                        _build_jobs_for_configs(
                            families=comparison_families,
                            train_doc_counts=(int(args.train_doc_count),),
                            benchmark=str(args.benchmark),
                            hardness_grid="",
                            grid_cell_ids=(),
                            seeds=[int(seed) for seed in args.seeds],
                            job_granularity=str(args.job_granularity),
                            repeat_closed_form_controls=bool(args.repeat_closed_form_controls),
                            configs=(comparison_leaf_config,),
                            tuning_stage="study_comparison",
                            study_name="leaf_geometry",
                            study_axis="fixed_leaf_tokens",
                            axis_value=str(int(leaf_tokens)),
                            locked_tree_neural_config_label=str(bundle["locked_label"]),
                            selection_metric=str(bundle["selection_metric"]),
                        )
                    )
            gpu_ids: List[str] = []
            for job in jobs:
                item = _scheduler_item_for_job(
                    phase="study",
                    item_id=f"study::leaf_geometry::{job.job_name}",
                    output_root=output_root,
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=int(locked_slots),
                )
                gpu_ids.append(str(item.item_id))
                items.append(item)

            def _leaf_reduce() -> Mapping[str, Any]:
                final = finalize_leaf_geometry_study_output(
                    args=args,
                    output_root=output_root,
                    locked_label=str(bundle["locked_label"]),
                    selection_metric=str(bundle["selection_metric"]),
                    tuning_hidden=bool(bundle["tuning_hidden"]),
                    axis_values=axis_values,
                )
                return {"result": dict(final)}

            items.append(
                SchedulerItem(
                    item_id="study::leaf_geometry::reduce",
                    phase="study",
                    kind="cpu_callback",
                    deps=tuple(gpu_ids),
                    expected_outputs=(str(output_root / "study_summary.json"),),
                    callback=_leaf_reduce,
                    reuse_existing=False,
                )
            )
            return {
                "items": items,
                "manifest_payload": manifest_payload,
                "scheduler_max_gpu_items_per_mig": int(scheduler_max),
            }

        cell_ids = [
            str(cell.cell_id)
            for cell in resolve_full_doc_diagnostic_grid("structural_core_v1")
            if str(cell.cell_id).strip()
        ]
        comparison_families = [
            str(family) for family in args.families if str(family) != "tree_neural"
        ]
        screen_root = output_root / "screen"
        screen_jobs: List[_JobSpec] = []
        for cell_id in cell_ids:
            screen_jobs.extend(
                _build_jobs_for_configs(
                    families=("tree_neural",),
                    train_doc_counts=(int(args.train_doc_count),),
                    benchmark=str(args.benchmark),
                    hardness_grid="structural_core_v1",
                    grid_cell_ids=(cell_id,),
                    seeds=[int(seed) for seed in args.screen_seeds],
                    job_granularity=str(args.job_granularity),
                    repeat_closed_form_controls=True,
                    configs=(bundle["locked_tree_config"],),
                    tuning_stage="study_screen",
                    study_name="structural_complexity",
                    study_axis="structural_core_cell",
                    axis_value=str(cell_id),
                    locked_tree_neural_config_label=str(bundle["locked_label"]),
                    selection_metric=str(bundle["selection_metric"]),
                )
            )
            if comparison_families:
                screen_jobs.extend(
                    _build_jobs_for_configs(
                        families=comparison_families,
                        train_doc_counts=(int(args.train_doc_count),),
                        benchmark=str(args.benchmark),
                        hardness_grid="structural_core_v1",
                        grid_cell_ids=(cell_id,),
                        seeds=[int(seed) for seed in args.screen_seeds],
                        job_granularity=str(args.job_granularity),
                        repeat_closed_form_controls=bool(args.repeat_closed_form_controls),
                        configs=(bundle["comparison_config"],),
                        tuning_stage="study_screen",
                        study_name="structural_complexity",
                        study_axis="structural_core_cell",
                        axis_value=str(cell_id),
                        locked_tree_neural_config_label=str(bundle["locked_label"]),
                        selection_metric=str(bundle["selection_metric"]),
                    )
                )
        screen_item_ids: List[str] = []
        for job in screen_jobs:
            item = _scheduler_item_for_job(
                phase="study",
                item_id=f"study::screen::{job.job_name}",
                output_root=screen_root,
                job=job,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                gpu_slots=int(screen_slots),
            )
            screen_item_ids.append(str(item.item_id))
            items.append(item)

        def _study_screen_reduce() -> Mapping[str, Any]:
            screen_payload = _load_or_write_summary_outputs(screen_root)
            representative_cells = _select_representative_structural_cells(
                screen_payload,
                family="tree_neural",
                tuning_stage="study_screen",
                train_doc_count=int(args.train_doc_count),
            )
            if not representative_cells:
                raise RuntimeError("structural screen stage produced no representative cells")
            representative_bundle = build_structural_study_representative_job_bundle(
                args,
                locked_tree_config=bundle["locked_tree_config"],
                comparison_config=bundle["comparison_config"],
                locked_label=str(bundle["locked_label"]),
                selection_metric=str(bundle["selection_metric"]),
                representative_cells=representative_cells,
            )
            representative_item_ids: List[str] = []
            new_items: List[SchedulerItem] = []
            for job in list(representative_bundle["representative_jobs"]):
                item = _scheduler_item_for_job(
                    phase="study",
                    item_id=f"study::representative::{job.job_name}",
                    output_root=representative_bundle["representative_root"],
                    job=job,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                    gpu_slots=int(locked_slots),
                )
                representative_item_ids.append(str(item.item_id))
                new_items.append(item)

            def _study_reduce() -> Mapping[str, Any]:
                final = finalize_structural_study_output(
                    args=args,
                    output_root=output_root,
                    locked_label=str(bundle["locked_label"]),
                    selection_metric=str(bundle["selection_metric"]),
                    tuning_hidden=bool(bundle["tuning_hidden"]),
                    representative_cells=representative_cells,
                    cell_ids=cell_ids,
                )
                return {"result": dict(final)}

            new_items.append(
                SchedulerItem(
                    item_id="study::structural_complexity::reduce",
                    phase="study",
                    kind="cpu_callback",
                    deps=tuple(representative_item_ids),
                    expected_outputs=(str(output_root / "study_summary.json"),),
                    callback=_study_reduce,
                    reuse_existing=False,
                )
            )
            return {"new_items": new_items, "result": {"representative_cells": representative_cells}}

        items.append(
            SchedulerItem(
                item_id="study::screen::reduce",
                phase="study",
                kind="cpu_callback",
                deps=tuple(screen_item_ids),
                expected_outputs=(),
                callback=_study_screen_reduce,
                reuse_existing=False,
            )
        )
        return {
            "items": items,
            "manifest_payload": manifest_payload,
            "scheduler_max_gpu_items_per_mig": int(scheduler_max),
        }

    raise ValueError(f"unsupported mode for scheduler graph: {mode}")


def _run_scheduler_mode(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(str(args.output_root))
    mig_uuids = _discover_scheduler_devices(args)
    setattr(args, "mig_uuids_resolved", list(mig_uuids))
    scheduler_min_mem_available_kib = int(
        max(
            0.0,
            float(getattr(args, "scheduler_min_mem_available_gib", 128.0) or 0.0),
        )
        * 1024.0
        * 1024.0
    )
    scheduler_min_swap_free_kib = int(
        max(
            0.0,
            float(getattr(args, "scheduler_min_swap_free_gib", 2.0) or 0.0),
        )
        * 1024.0
        * 1024.0
    )
    graph = _build_scheduler_graph(
        args,
        output_root=output_root,
        mig_uuids=mig_uuids,
    )
    if bool(getattr(args, "plan_only", False)):
        payload = _scheduler_cli_payload(
            items=graph["items"],
            devices=mig_uuids,
            max_gpu_items_per_mig=int(graph["scheduler_max_gpu_items_per_mig"]),
            launch_stagger_seconds=float(
                max(0.0, float(getattr(args, "scheduler_launch_stagger_seconds", 0.0)))
            ),
            min_mem_available_kib=int(scheduler_min_mem_available_kib),
            min_swap_free_kib=int(scheduler_min_swap_free_kib),
            manifest_payload=graph["manifest_payload"],
        )
        payload["output_root"] = str(output_root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return {"plan_only": True, "failed_jobs": [], "output_root": str(output_root)}
    return _run_scheduler_bundle(
        output_root=output_root,
        items=graph["items"],
        devices=mig_uuids,
        max_gpu_items_per_mig=int(graph["scheduler_max_gpu_items_per_mig"]),
        launch_stagger_seconds=float(
            max(0.0, float(getattr(args, "scheduler_launch_stagger_seconds", 0.0)))
        ),
        cleanup_stale_children=bool(getattr(args, "cleanup_stale_children", True)),
        resume_enabled=bool(getattr(args, "resume", True)),
        manifest_payload=graph["manifest_payload"],
        min_mem_available_kib=int(scheduler_min_mem_available_kib),
        min_swap_free_kib=int(scheduler_min_swap_free_kib),
        cancel_on_failure=bool(
            getattr(args, "mode", "")
            not in {"representation_sufficiency", "representation_learnability"}
        ),
    )


def _nested_mapping_value(
    mapping: Mapping[str, Any],
    path: Sequence[str],
    *,
    default: Any = float("nan"),
) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(str(key))
    return cur if cur is not None else default


def _finite_summary_stats(values: Sequence[Any]) -> Dict[str, Any]:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "n": int(finite.size),
    }


def _exact_sanity_metric_summary(
    runs: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> Dict[str, Any]:
    return _finite_summary_stats(
        [_nested_mapping_value(run, path) for run in runs]
    )


def _exact_sanity_condition_kind(run: Mapping[str, Any]) -> str:
    config_label = str(run.get("config_label", "")).strip()
    task_split_suffix = "_task_split_ablation"
    task_split_ablation = config_label.endswith(task_split_suffix)
    base_config_label = (
        config_label[: -len(task_split_suffix)]
        if task_split_ablation and len(config_label) > len(task_split_suffix)
        else config_label
    )
    exact_label_map = {
        FAIR_FNO_PARITY_CONFIG_LABEL: "legacy_fair_fno_root_only",
        "tree_neural_slot_align_v1_root_only": "slot_root_only",
        "tree_neural_slot_align_v1_leaf_sampled": "slot_leaf_sampled_r0p25",
        "tree_neural_slot_align_v1_leaf_dense": "slot_leaf_dense",
        "tree_neural_slot_align_v1_internal_count_r0p25": "slot_internal_count_only_r0p25",
        "tree_neural_slot_align_v1_internal_full_r0p25": "slot_internal_full_sketch_r0p25",
        "tree_neural_slot_align_v1_internal_count_dense": "slot_internal_count_only_dense",
        "tree_neural_slot_align_v1_internal_full_dense": "slot_internal_full_sketch_dense",
        "tree_neural_slot_align_v1_internal_full_r0p5": "slot_internal_full_sketch_r0p5",
    }
    if base_config_label in exact_label_map:
        base_kind = exact_label_map[base_config_label]
        return (
            f"{base_kind}__task_split_ablation"
            if task_split_ablation
            else base_kind
        )
    summary_spec_name = str(run.get("summary_spec_name", "")).strip()
    leaf_label_rate = float(run.get("leaf_label_rate", 1.0) or 0.0)
    internal_kind = str(run.get("internal_supervision_kind", "none")).strip() or "none"
    internal_rate = float(run.get("internal_label_rate", 0.0) or 0.0)
    if summary_spec_name != "markov_count_sketch":
        return config_label or "legacy_unknown"
    if internal_kind == "none" and leaf_label_rate <= 0.0:
        return "slot_root_only"
    if internal_kind == "none":
        rate_label = _format_float_label(float(leaf_label_rate))
        if rate_label == "1":
            return "slot_leaf_dense"
        return f"slot_leaf_sampled_r{rate_label}"
    rate_label = _format_float_label(float(internal_rate))
    if internal_kind == "count_only":
        if rate_label == "1":
            return "slot_internal_count_only_dense"
        return f"slot_internal_count_only_r{rate_label}"
    if internal_kind == "full_sketch":
        if rate_label == "1":
            base_kind = "slot_internal_full_sketch_dense"
        else:
            base_kind = f"slot_internal_full_sketch_r{rate_label}"
        return (
            f"{base_kind}__task_split_ablation"
            if task_split_ablation
            else base_kind
        )
    return config_label or "aligned_unknown"


def _exact_sanity_condition_id(run: Mapping[str, Any]) -> str:
    config_label = str(run.get("config_label", "")).strip()
    if config_label:
        return config_label
    return _exact_sanity_condition_kind(run)


def _exact_sanity_condition_title(condition_id: str) -> str:
    fixed_titles = {
        FAIR_FNO_PARITY_CONFIG_LABEL: "Legacy Fair-FNO Root-Only",
        "tree_neural_slot_align_v1_root_only": "Slot-Aligned Root-Only",
        "tree_neural_slot_align_v1_leaf_sampled": "Slot-Aligned Leaf Sampled @ 0.25",
        "tree_neural_slot_align_v1_leaf_dense": "Slot-Aligned Leaf Dense",
        "tree_neural_slot_align_v1_internal_count_r0p25": "Slot-Aligned Internal Count-Only @ 0.25",
        "tree_neural_slot_align_v1_internal_full_r0p25": "Slot-Aligned Internal Full-Sketch @ 0.25",
        "tree_neural_slot_align_v1_internal_count_dense": "Slot-Aligned Internal Count-Only Dense",
        "tree_neural_slot_align_v1_internal_full_dense": "Slot-Aligned Internal Full-Sketch Dense",
        "tree_neural_slot_align_v1_internal_full_r0p5": "Slot-Aligned Internal Full-Sketch @ 0.5",
        "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation": (
            "Slot-Aligned Internal Full-Sketch @ 0.25 (Task-Split Ablation)"
        ),
        "tree_neural_slot_align_v1_balanced_full_r0p25": "Slot-Aligned Rebalanced Full-Sketch @ 0.25",
        "tree_neural_slot_align_v1_leaf_ep_count_r0p25": "Slot-Aligned Leaf Full-Sketch + Internal Count @ 0.25",
        "legacy_fair_fno_root_only": "Legacy Fair-FNO Root-Only",
        "slot_root_only": "Slot-Aligned Root-Only",
        "slot_leaf_sampled_r0p25": "Slot-Aligned Leaf Sampled @ 0.25",
        "slot_leaf_dense": "Slot-Aligned Leaf Dense",
        "slot_internal_count_only_r0p25": "Slot-Aligned Internal Count-Only @ 0.25",
        "slot_internal_full_sketch_r0p25": "Slot-Aligned Internal Full-Sketch @ 0.25",
        "slot_internal_full_sketch_r0p25__task_split_ablation": (
            "Slot-Aligned Internal Full-Sketch @ 0.25 (Task-Split Ablation)"
        ),
        "slot_internal_count_only_dense": "Slot-Aligned Internal Count-Only Dense",
        "slot_internal_full_sketch_dense": "Slot-Aligned Internal Full-Sketch Dense",
        "slot_internal_full_sketch_r0p5": "Slot-Aligned Internal Full-Sketch @ 0.5",
    }
    normalized = str(condition_id)
    if normalized in fixed_titles:
        return fixed_titles[normalized]
    if normalized.startswith("slot_leaf_sampled_r"):
        return f"Slot-Aligned Leaf Sampled @ {normalized.split('_r', 1)[1].replace('p', '.')}"
    if normalized.startswith("slot_internal_count_only_r"):
        return f"Slot-Aligned Internal Count-Only @ {normalized.split('_r', 1)[1].replace('p', '.')}"
    if normalized.startswith("slot_internal_full_sketch_r"):
        if normalized.endswith("__task_split_ablation"):
            base_id = normalized[: -len("__task_split_ablation")]
            return (
                f"{_exact_sanity_condition_title(base_id)} "
                "(Task-Split Ablation)"
            )
        return f"Slot-Aligned Internal Full-Sketch @ {normalized.split('_r', 1)[1].replace('p', '.')}"
    return normalized


def _exact_sanity_condition_summary(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not runs:
        return {}
    exemplar = dict(runs[0])
    failure_bucket_counts: Dict[str, int] = {}
    for run in runs:
        bucket = str(run.get("exact_sketch_failure_bucket", "")).strip()
        if bucket:
            failure_bucket_counts[bucket] = failure_bucket_counts.get(bucket, 0) + 1
    tree_neural: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        tree_neural[split] = {}
        for level in EXACT_SANITY_LEVELS:
            tree_neural[split][level] = {
                branch: {
                    metric: _exact_sanity_metric_summary(
                        runs,
                        (
                            "exact_sketch_diagnostics",
                            "tree_neural",
                            split,
                            level,
                            branch,
                            metric,
                        ),
                    )
                    for metric in EXACT_SANITY_COMPONENT_METRICS
                }
                for branch in ("direct", "probe")
            }
            if level == "merge":
                tree_neural[split][level]["decoded_consistency"] = {
                    metric: _exact_sanity_metric_summary(
                        runs,
                        (
                            "exact_sketch_diagnostics",
                            "tree_neural",
                            split,
                            level,
                            "decoded_consistency",
                            metric,
                        ),
                    )
                    for metric in EXACT_SANITY_MERGE_CONSISTENCY_METRICS
                }
    test_tree = dict(tree_neural.get("test") or {})
    test_leaf_probe = dict((test_tree.get("leaf") or {}).get("probe") or {})
    test_merge_probe = dict((test_tree.get("merge") or {}).get("probe") or {})
    test_root_direct = dict((test_tree.get("root") or {}).get("direct") or {})
    test_root_probe = dict((test_tree.get("root") or {}).get("probe") or {})
    test_merge_consistency = dict(
        ((test_tree.get("merge") or {}).get("decoded_consistency") or {})
    )
    condition_id = _exact_sanity_condition_id(exemplar)
    condition_kind = _exact_sanity_condition_kind(exemplar)
    return {
        "condition_id": condition_id,
        "condition_kind": condition_kind,
        "condition_title": _exact_sanity_condition_title(condition_id),
        "config_label": str(exemplar.get("config_label", "")),
        "n_runs": int(len(runs)),
        "seed_values": sorted(
            {int(run.get("seed", 0)) for run in runs if "seed" in run}
        ),
        "aligned_sketch_surface": str(exemplar.get("aligned_sketch_surface", "")),
        "weighting_scheme": str(exemplar.get("weighting_scheme", "")),
        "optimization_root_weight": float(
            exemplar.get("optimization_root_weight", float("nan"))
        ),
        "local_law_c1_weight": float(exemplar.get("local_law_c1_weight", float("nan"))),
        "local_law_c2_weight": float(exemplar.get("local_law_c2_weight", float("nan"))),
        "local_law_c3_weight": float(exemplar.get("local_law_c3_weight", float("nan"))),
        "summary_spec_name": str(exemplar.get("summary_spec_name", "")),
        "slot_count": int(exemplar.get("slot_count", 0)),
        "tree_theorem_count_dim": int(exemplar.get("tree_theorem_count_dim", 0)),
        "tree_theorem_first_dim": int(exemplar.get("tree_theorem_first_dim", 0)),
        "tree_theorem_last_dim": int(exemplar.get("tree_theorem_last_dim", 0)),
        "tree_theorem_count_head_mode": str(
            exemplar.get("tree_theorem_count_head_mode", "")
        ),
        "tree_theorem_count_ordinal_weight": float(
            exemplar.get("tree_theorem_count_ordinal_weight", 1.0)
        ),
        "tree_theorem_count_scalar_aux_weight": float(
            exemplar.get("tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        "tree_theorem_count_threshold_balance": bool(
            exemplar.get("tree_theorem_count_threshold_balance", True)
        ),
        "leaf_supervision_kind": str(exemplar.get("leaf_supervision_kind", "")),
        "internal_supervision_kind": str(
            exemplar.get("internal_supervision_kind", "none")
        ),
        "internal_label_rate": float(exemplar.get("internal_label_rate", 0.0)),
        "leaf_exact_supervision": bool(exemplar.get("leaf_exact_supervision", False)),
        "leaf_label_rate": float(exemplar.get("leaf_label_rate", 1.0)),
        "tree_training_schedule": str(exemplar.get("tree_training_schedule", "")),
        "tree_stage1_epochs": int(exemplar.get("tree_stage1_epochs", 0)),
        "tree_stage2_epochs": int(exemplar.get("tree_stage2_epochs", 0)),
        "tree_root_supervision_kind": str(
            exemplar.get("tree_root_supervision_kind", "")
        ),
        "tree_checkpoint_metric": str(exemplar.get("tree_checkpoint_metric", "")),
        "tree_stage1_checkpoint_metric": str(
            exemplar.get("tree_stage1_checkpoint_metric", "")
        ),
        "tree_stage1_eval_mode": str(exemplar.get("tree_stage1_eval_mode", "")),
        "tree_stage1_screen_doc_limit": int(
            exemplar.get("tree_stage1_screen_doc_limit", 0)
        ),
        "tree_stage1_final_exact_doc_limit": int(
            exemplar.get("tree_stage1_final_exact_doc_limit", 0)
        ),
        "tree_stage1_artifact_dir": str(
            exemplar.get("tree_stage1_artifact_dir", "")
        ),
        "tree_stage1_root_weight": float(
            exemplar.get("tree_stage1_root_weight", 0.0)
        ),
        "tree_summary_spec_root_mode": str(
            exemplar.get("tree_summary_spec_root_mode", "")
        ),
        "failure_bucket_counts": dict(failure_bucket_counts),
        "failure_gap_scores": {
            "leaf_boundary_encoding_gap_score": _finite_summary_stats(
                [run.get("exact_sketch_leaf_gap_score", float("nan")) for run in runs]
            ),
            "count_composition_gap_score": _finite_summary_stats(
                [run.get("exact_sketch_merge_gap_score", float("nan")) for run in runs]
            ),
            "subtree_label_value_gap_score": _finite_summary_stats(
                [
                    (
                        _nested_mapping_value(
                            run,
                            (
                                "exact_sketch_diagnostics",
                                "failure_attribution",
                                "subtree_label_value_gap_score",
                            ),
                        )
                        if np.isfinite(
                            float(
                                _nested_mapping_value(
                                    run,
                                    (
                                        "exact_sketch_diagnostics",
                                        "failure_attribution",
                                        "subtree_label_value_gap_score",
                                    ),
                                )
                            )
                        )
                        else _nested_mapping_value(
                            run,
                            (
                                "exact_sketch_diagnostics",
                                "failure_attribution",
                                "internal_label_value_gap_score",
                            ),
                        )
                    )
                    for run in runs
                ]
            ),
            "legacy_readout_gap_score": _finite_summary_stats(
                [run.get("exact_sketch_readout_gap_score", float("nan")) for run in runs]
            ),
        },
        "tree_neural": tree_neural,
        "acceptance_readout": {
            "test_probe_leaf_exact_summary_match_rate_mean": float(
                (test_leaf_probe.get("exact_summary_match_rate") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_exact_summary_match_rate_mean": float(
                (test_merge_probe.get("exact_summary_match_rate") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_direct_root_count_mae_mean": float(
                (test_root_direct.get("count_mae") or {}).get("mean", float("nan"))
            ),
            "test_task_root_mae_ablation_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "task_root_mae_ablation",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_task_root_mae_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "task_root_mae",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_probe_root_count_mae_mean": float(
                (test_root_probe.get("count_mae") or {}).get("mean", float("nan"))
            ),
            "test_merge_join_bit_accuracy_mean": float(
                (test_merge_consistency.get("merge_join_bit_accuracy") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_c2_on_range_exact_match_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "c2_on_range_exact_match",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_theorem_bootstrap_direct_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "val_theorem_bootstrap_direct",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_probe_merge_count_match_rate_mean": float(
                (test_merge_probe.get("count_match_rate") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_first_accuracy_mean": float(
                (test_merge_probe.get("first_accuracy") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_last_accuracy_mean": float(
                (test_merge_probe.get("last_accuracy") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_count_mae_mean": float(
                (test_merge_probe.get("count_mae") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
        },
    }


def _condition_acceptance_value(
    condition: Mapping[str, Any],
    key: str,
) -> float:
    return float(
        dict(condition.get("acceptance_readout") or {}).get(key, float("nan"))
    )


def _tree_neural_exact_sanity_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    all_runs = [
        dict(run)
        for run in list(payload.get("runs") or [])
        if str(run.get("study_name", "")).strip() == EXACT_SANITY_STUDY_NAME
    ]
    runs = [
        dict(run)
        for run in all_runs
        if str(run.get("baseline_family", "")) == EXACT_SANITY_FAMILY
        and isinstance(run.get("exact_sketch_diagnostics"), Mapping)
    ]
    if not runs:
        return {}

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(int(run.get("train_doc_count", 0)), []).append(run)

    groups: List[Dict[str, Any]] = []
    for train_doc_count in sorted(grouped):
        group_runs = list(grouped[train_doc_count])
        fno_reference_runs = [
            run
            for run in all_runs
            if int(run.get("train_doc_count", 0)) == int(train_doc_count)
            and str(run.get("baseline_family", "")) == "official_fno"
            and isinstance(run.get("root_summary_probe_audit"), Mapping)
        ]
        failure_bucket_counts: Dict[str, int] = {}
        for run in group_runs:
            bucket = str(run.get("exact_sketch_failure_bucket", "")).strip()
            if bucket:
                failure_bucket_counts[bucket] = failure_bucket_counts.get(bucket, 0) + 1
        runs_by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for run in group_runs:
            runs_by_condition.setdefault(_exact_sanity_condition_id(run), []).append(run)
        exact_witness: Dict[str, Any] = {}
        for split in ("train", "val", "test"):
            exact_witness[split] = {
                "law_metrics": {
                    metric: _exact_sanity_metric_summary(
                        group_runs,
                        (
                            "exact_sketch_diagnostics",
                            "exact_witness",
                            split,
                            "law_metrics",
                            metric,
                        ),
                    )
                    for metric in EXACT_SANITY_LAW_METRICS
                }
            }
            for level in EXACT_SANITY_LEVELS:
                exact_witness[split][level] = {
                    "direct": {
                        metric: _exact_sanity_metric_summary(
                            group_runs,
                            (
                                "exact_sketch_diagnostics",
                                "exact_witness",
                                split,
                                level,
                                "direct",
                                metric,
                            ),
                        )
                        for metric in EXACT_SANITY_COMPONENT_METRICS
                    },
                    "probe_control": {
                        metric: _exact_sanity_metric_summary(
                            group_runs,
                            (
                                "exact_sketch_diagnostics",
                                "exact_witness",
                                split,
                                level,
                                "probe_control",
                                metric,
                            ),
                        )
                        for metric in EXACT_SANITY_COMPONENT_METRICS
                    },
                }
        conditions = [
            _exact_sanity_condition_summary(condition_runs)
            for _condition_id, condition_runs in sorted(runs_by_condition.items())
        ]
        condition_by_id = {
            str(condition.get("condition_id", "")): condition for condition in conditions
        }
        condition_by_kind = {
            str(condition.get("condition_kind", "")): condition for condition in conditions
        }
        exact_test_laws = exact_witness["test"]["law_metrics"]
        exact_witness_near_zero = all(
            abs(float(exact_test_laws[metric]["mean"])) <= 1e-9
            for metric in EXACT_SANITY_LAW_METRICS
            if np.isfinite(float(exact_test_laws[metric]["mean"]))
        )
        legacy_condition = condition_by_kind.get("legacy_fair_fno_root_only")
        slot_root_only = condition_by_kind.get("slot_root_only")
        slot_leaf_sampled = condition_by_kind.get("slot_leaf_sampled_r0p25")
        slot_leaf_dense = condition_by_kind.get("slot_leaf_dense")
        legacy_vs_slot_root_only: Dict[str, Any] = {}
        if legacy_condition is not None and slot_root_only is not None:
            legacy_vs_slot_root_only = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        legacy_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "leaf_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        legacy_condition,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                ),
                "direct_root_count_mae_delta": float(
                    _condition_acceptance_value(
                        slot_root_only,
                        "test_direct_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        legacy_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
                "slot_root_only_improves_over_legacy": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            legacy_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            legacy_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        legacy_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        slot_root_only,
                        "test_direct_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        legacy_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
            }
        leaf_sampled_value: Dict[str, Any] = {}
        leaf_value_by_rate: Dict[str, Any] = {}
        for rate_label in ("0p25", "0p5", "0p75", "dense"):
            leaf_condition = condition_by_kind.get(
                "slot_leaf_dense"
                if rate_label == "dense"
                else f"slot_leaf_sampled_r{rate_label}"
            )
            if slot_root_only is None or leaf_condition is None:
                continue
            payload = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "leaf_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                ),
                "root_probe_count_mae_delta": float(
                    _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_root_count_mae_mean",
                    )
                ),
                "leaf_rate_improves_over_root_only": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            leaf_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            leaf_condition,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_root_count_mae_mean",
                    )
                ),
            }
            leaf_value_by_rate[rate_label] = payload
            if rate_label == "0p25":
                leaf_sampled_value = dict(payload)
        dense_leaf_value: Dict[str, Any] = {}
        if slot_leaf_sampled is not None and slot_leaf_dense is not None:
            dense_leaf_value = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "root_probe_count_mae_delta": float(
                    _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_root_count_mae_mean",
                    )
                ),
                "leaf_dense_improves_over_leaf_sampled": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_dense,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_sampled,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_dense,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_sampled,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_root_count_mae_mean",
                    )
                ),
            }
        subtree_label_value_by_rate: Dict[str, Any] = {}
        for rate_label in ("0p25", "0p5", "0p75", "dense"):
            count_condition = condition_by_kind.get(
                "slot_internal_count_only_dense"
                if rate_label == "dense"
                else f"slot_internal_count_only_r{rate_label}"
            )
            full_condition = condition_by_kind.get(
                "slot_internal_full_sketch_dense"
                if rate_label == "dense"
                else f"slot_internal_full_sketch_r{rate_label}"
            )
            if count_condition is None or full_condition is None:
                continue
            subtree_label_value_by_rate[rate_label] = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        full_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        count_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "merge_join_bit_accuracy_delta": float(
                    _condition_acceptance_value(
                        full_condition,
                        "test_merge_join_bit_accuracy_mean",
                    )
                    - _condition_acceptance_value(
                        count_condition,
                        "test_merge_join_bit_accuracy_mean",
                    )
                ),
                "direct_root_count_mae_delta": float(
                    _condition_acceptance_value(
                        full_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        count_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
                "full_sketch_improves_over_count_only": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            full_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            count_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            full_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            count_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        full_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        count_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        full_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        count_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
            }
        root_mode_alignment_by_base_config: Dict[str, Any] = {}
        for condition in conditions:
            condition_id = str(condition.get("condition_id", ""))
            if not condition_id.endswith("_task_split_ablation"):
                continue
            base_condition_id = condition_id[: -len("_task_split_ablation")]
            primary_condition = condition_by_id.get(base_condition_id)
            if primary_condition is None:
                continue
            root_mode_alignment_by_base_config[base_condition_id] = {
                "aligned_primary_condition_id": base_condition_id,
                "theorem_primary_condition_id": base_condition_id,
                "task_split_ablation_condition_id": condition_id,
                "aligned_primary_root_mode": str(
                    primary_condition.get("tree_summary_spec_root_mode", "")
                ),
                "theorem_primary_root_mode": str(
                    primary_condition.get("tree_summary_spec_root_mode", "")
                ),
                "task_split_root_mode": str(
                    condition.get("tree_summary_spec_root_mode", "")
                ),
                "theorem_root_count_mae_delta": float(
                    _condition_acceptance_value(
                        primary_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
                "task_root_mae_ablation_delta": float(
                    _condition_acceptance_value(
                        primary_condition,
                        "test_task_root_mae_ablation_mean",
                    )
                    - _condition_acceptance_value(
                        condition,
                        "test_task_root_mae_ablation_mean",
                    )
                ),
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        primary_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "aligned_primary_improves_or_matches_theorem_root": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            primary_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        primary_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
            }
        groups.append(
            {
                "train_doc_count": int(train_doc_count),
                "n_runs": int(len(group_runs)),
                "seed_values": sorted(
                    {int(run.get("seed", 0)) for run in group_runs if "seed" in run}
                ),
                "config_labels": sorted(
                    {
                        str(run.get("config_label", "")).strip()
                        for run in group_runs
                        if str(run.get("config_label", "")).strip()
                    }
                ),
                "failure_bucket_counts": dict(failure_bucket_counts),
                "failure_gap_scores": {
                    "leaf_boundary_encoding_gap_score": _finite_summary_stats(
                        [
                            run.get("exact_sketch_leaf_gap_score", float("nan"))
                            for run in group_runs
                        ]
                    ),
                    "count_composition_gap_score": _finite_summary_stats(
                        [
                            run.get("exact_sketch_merge_gap_score", float("nan"))
                            for run in group_runs
                        ]
                    ),
                    "subtree_label_value_gap_score": _finite_summary_stats(
                        [
                            (
                                _nested_mapping_value(
                                    run,
                                    (
                                        "exact_sketch_diagnostics",
                                        "failure_attribution",
                                        "subtree_label_value_gap_score",
                                    ),
                                )
                                if np.isfinite(
                                    float(
                                        _nested_mapping_value(
                                            run,
                                            (
                                                "exact_sketch_diagnostics",
                                                "failure_attribution",
                                                "subtree_label_value_gap_score",
                                            ),
                                        )
                                    )
                                )
                                else _nested_mapping_value(
                                    run,
                                    (
                                        "exact_sketch_diagnostics",
                                        "failure_attribution",
                                        "internal_label_value_gap_score",
                                    ),
                                )
                            )
                            for run in group_runs
                        ]
                    ),
                    "legacy_readout_gap_score": _finite_summary_stats(
                        [
                            run.get("exact_sketch_readout_gap_score", float("nan"))
                            for run in group_runs
                        ]
                    ),
                },
                "exact_witness": exact_witness,
                "conditions": conditions,
                "full_doc_fno_reference": {
                    split: {
                        metric: _exact_sanity_metric_summary(
                            fno_reference_runs,
                            ("root_summary_probe_audit", split, metric),
                        )
                        for metric in EXACT_SANITY_COMPONENT_METRICS
                    }
                    for split in ("train", "val", "test")
                }
                if fno_reference_runs
                else {},
                "acceptance_readout": {
                    "exact_witness_test_laws_near_zero": bool(exact_witness_near_zero),
                    "legacy_vs_slot_root_only": legacy_vs_slot_root_only,
                    "leaf_sampled_value": leaf_sampled_value,
                    "leaf_value_by_rate": leaf_value_by_rate,
                    "dense_leaf_value": dense_leaf_value,
                    "subtree_label_value_by_rate": subtree_label_value_by_rate,
                    "root_mode_alignment_by_base_config": root_mode_alignment_by_base_config,
                },
            }
        )
    return {
        "study_name": EXACT_SANITY_STUDY_NAME,
        "benchmark": str(payload.get("benchmark", "")),
        "baseline_family": EXACT_SANITY_FAMILY,
        "primary_question": (
            "Can tree_neural fair root-only recover the Lean-style exact sketch "
            "(count, first, last) well enough that the local-law gap is attributable?"
        ),
        "paper_to_lean_local_law_mapping": {
            "C1": "L1",
            "C2": "L3",
            "C3": "L2",
        },
        "theorem_contract": dict(
            ((runs[0].get("exact_sketch_diagnostics") or {}).get("theorem_contract") or {})
        ),
        "groups": groups,
    }


def _render_exact_sanity_summary_markdown(
    summary: Mapping[str, Any],
) -> str:
    groups = list(summary.get("groups") or [])
    lines = [
        "# Tree-Neural Exact-Sketch Sanity Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- baseline_family: `{str(summary.get('baseline_family', ''))}`",
        f"- study_name: `{str(summary.get('study_name', ''))}`",
        f"- primary_question: `{str(summary.get('primary_question', ''))}`",
        f"- paper_to_lean_local_law_mapping: `{dict(summary.get('paper_to_lean_local_law_mapping') or {})}`",
        f"- theorem_contract: `{dict(summary.get('theorem_contract') or {})}`",
    ]
    for group in groups:
        acceptance = dict(group.get("acceptance_readout") or {})
        exact_witness = dict(group.get("exact_witness") or {})
        fno_reference = dict(group.get("full_doc_fno_reference") or {})
        witness_test_laws = dict((exact_witness.get("test") or {}).get("law_metrics") or {})
        lines.extend(
            [
                "",
                f"## train_doc_count = {int(group.get('train_doc_count', 0))}",
                "",
                f"- n_runs: `{int(group.get('n_runs', 0))}`",
                f"- seeds: `{list(group.get('seed_values') or [])}`",
                f"- config_labels: `{list(group.get('config_labels') or [])}`",
                f"- failure_bucket_counts: `{dict(group.get('failure_bucket_counts') or {})}`",
                (
                    "- exact witness test laws near zero: "
                    f"`{bool(acceptance.get('exact_witness_test_laws_near_zero', False))}`"
                ),
                "",
                "### Exact Witness Test Laws",
                "",
                "| metric | mean | std |",
                "|---|---:|---:|",
            ]
        )
        for metric in EXACT_SANITY_LAW_METRICS:
            stats = dict(witness_test_laws.get(metric) or {})
            lines.append(
                "| "
                f"{metric} | "
                f"{float(stats.get('mean', float('nan'))):.6g} | "
                f"{float(stats.get('std', float('nan'))):.6g} |"
            )
        for condition in list(group.get("conditions") or []):
            condition = dict(condition)
            tree_test = dict((condition.get("tree_neural") or {}).get("test") or {})
            condition_acceptance = dict(condition.get("acceptance_readout") or {})
            lines.extend(
                [
                    "",
                    f"### {str(condition.get('condition_title', 'Condition'))}",
                    "",
                    f"- condition_id: `{str(condition.get('condition_id', ''))}`",
                    f"- condition_kind: `{str(condition.get('condition_kind', ''))}`",
                    f"- config_label: `{str(condition.get('config_label', ''))}`",
                    f"- aligned_sketch_surface: `{str(condition.get('aligned_sketch_surface', ''))}`",
                    f"- weighting_scheme: `{str(condition.get('weighting_scheme', ''))}`",
                    f"- optimization_root_weight: `{float(condition.get('optimization_root_weight', float('nan'))):.6g}`",
                    f"- local_law_c1_weight: `{float(condition.get('local_law_c1_weight', float('nan'))):.6g}`",
                    f"- local_law_c2_weight: `{float(condition.get('local_law_c2_weight', float('nan'))):.6g}`",
                    f"- local_law_c3_weight: `{float(condition.get('local_law_c3_weight', float('nan'))):.6g}`",
                    f"- summary_spec_name: `{str(condition.get('summary_spec_name', ''))}`",
                    f"- slot_count: `{int(condition.get('slot_count', 0))}`",
                    f"- tree_theorem_count_dim: `{int(condition.get('tree_theorem_count_dim', 0))}`",
                    f"- tree_theorem_first_dim: `{int(condition.get('tree_theorem_first_dim', 0))}`",
                    f"- tree_theorem_last_dim: `{int(condition.get('tree_theorem_last_dim', 0))}`",
                    f"- tree_theorem_count_head_mode: `{str(condition.get('tree_theorem_count_head_mode', ''))}`",
                    f"- tree_theorem_count_ordinal_weight: `{float(condition.get('tree_theorem_count_ordinal_weight', 1.0)):.6g}`",
                    f"- tree_theorem_count_scalar_aux_weight: `{float(condition.get('tree_theorem_count_scalar_aux_weight', 0.25)):.6g}`",
                    f"- tree_theorem_count_threshold_balance: `{bool(condition.get('tree_theorem_count_threshold_balance', True))}`",
                    f"- leaf_supervision_kind: `{str(condition.get('leaf_supervision_kind', ''))}`",
                    f"- internal_supervision_kind: `{str(condition.get('internal_supervision_kind', ''))}`",
                    f"- internal_label_rate: `{float(condition.get('internal_label_rate', 0.0)):.6g}`",
                    f"- leaf_exact_supervision: `{bool(condition.get('leaf_exact_supervision', False))}`",
                    f"- leaf_label_rate: `{float(condition.get('leaf_label_rate', 1.0)):.6g}`",
                    f"- tree_training_schedule: `{str(condition.get('tree_training_schedule', ''))}`",
                    f"- tree_stage1_epochs: `{int(condition.get('tree_stage1_epochs', 0))}`",
                    f"- tree_stage2_epochs: `{int(condition.get('tree_stage2_epochs', 0))}`",
                    f"- tree_root_supervision_kind: `{str(condition.get('tree_root_supervision_kind', ''))}`",
                    f"- tree_checkpoint_metric: `{str(condition.get('tree_checkpoint_metric', ''))}`",
                    f"- tree_stage1_checkpoint_metric: `{str(condition.get('tree_stage1_checkpoint_metric', ''))}`",
                    f"- tree_stage1_root_weight: `{float(condition.get('tree_stage1_root_weight', 0.0)):.6g}`",
                    f"- tree_task_head_mode: `{str(condition.get('tree_task_head_mode', ''))}`",
                    f"- tree_theorem_surface_mode: `{str(condition.get('tree_theorem_surface_mode', ''))}`",
                    f"- tree_summary_spec_root_mode: `{str(condition.get('tree_summary_spec_root_mode', ''))}`",
                    f"- failure_bucket_counts: `{dict(condition.get('failure_bucket_counts') or {})}`",
                    "",
                    "| level | branch | count_mae | count_match | first_acc | last_acc | exact_match |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for level in EXACT_SANITY_LEVELS:
                level_payload = dict(tree_test.get(level) or {})
                for branch in ("direct", "probe"):
                    branch_payload = dict(level_payload.get(branch) or {})
                    lines.append(
                        "| "
                        f"{level} | "
                        f"{branch} | "
                        f"{float((branch_payload.get('count_mae') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('count_match_rate') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('first_accuracy') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('last_accuracy') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('exact_summary_match_rate') or {}).get('mean', float('nan'))):.6g} |"
                    )
            merge_consistency = dict((tree_test.get("merge") or {}).get("decoded_consistency") or {})
            if merge_consistency:
                lines.extend(
                    [
                        "",
                        "| merge_consistency_metric | mean | std |",
                        "|---|---:|---:|",
                    ]
                )
                for metric in EXACT_SANITY_MERGE_CONSISTENCY_METRICS:
                    stats = dict(merge_consistency.get(metric) or {})
                    lines.append(
                        "| "
                        f"{metric} | "
                        f"{float(stats.get('mean', float('nan'))):.6g} | "
                        f"{float(stats.get('std', float('nan'))):.6g} |"
                    )
            lines.extend(
                [
                    "",
                    f"- test probe leaf exact summary match rate mean: `{float(condition_acceptance.get('test_probe_leaf_exact_summary_match_rate_mean', float('nan'))):.6g}`",
                    f"- test probe merge exact summary match rate mean: `{float(condition_acceptance.get('test_probe_merge_exact_summary_match_rate_mean', float('nan'))):.6g}`",
                    f"- test theorem root direct count mae mean: `{float(condition_acceptance.get('test_direct_root_count_mae_mean', float('nan'))):.6g}`",
                    f"- test task root mae ablation mean: `{float(condition_acceptance.get('test_task_root_mae_ablation_mean', float('nan'))):.6g}`",
                    f"- test probe root count mae mean: `{float(condition_acceptance.get('test_probe_root_count_mae_mean', float('nan'))):.6g}`",
                    f"- test merge join bit accuracy mean: `{float(condition_acceptance.get('test_merge_join_bit_accuracy_mean', float('nan'))):.6g}`",
                    f"- test C2/L3 on-range exact match mean: `{float(condition_acceptance.get('test_c2_on_range_exact_match_mean', float('nan'))):.6g}`",
                    f"- test theorem bootstrap direct mean: `{float(condition_acceptance.get('test_theorem_bootstrap_direct_mean', float('nan'))):.6g}`",
                ]
            )
        if fno_reference:
            ref_test = dict(fno_reference.get("test") or {})
            lines.extend(
                [
                    "",
                    "### Full-Doc FNO Root Probe Reference",
                    "",
                    "| metric | mean | std |",
                    "|---|---:|---:|",
                ]
            )
            for metric in EXACT_SANITY_COMPONENT_METRICS:
                stats = dict(ref_test.get(metric) or {})
                lines.append(
                    "| "
                    f"{metric} | "
                    f"{float(stats.get('mean', float('nan'))):.6g} | "
                    f"{float(stats.get('std', float('nan'))):.6g} |"
                )
        lines.extend(
            [
                "",
                "### Acceptance Readout",
                "",
                f"- legacy_vs_slot_root_only: `{dict(acceptance.get('legacy_vs_slot_root_only') or {})}`",
                f"- leaf_sampled_value: `{dict(acceptance.get('leaf_sampled_value') or {})}`",
                f"- leaf_value_by_rate: `{dict(acceptance.get('leaf_value_by_rate') or {})}`",
                f"- dense_leaf_value: `{dict(acceptance.get('dense_leaf_value') or {})}`",
                f"- subtree_label_value_by_rate: `{dict(acceptance.get('subtree_label_value_by_rate') or {})}`",
                f"- root_mode_alignment_by_base_config: `{dict(acceptance.get('root_mode_alignment_by_base_config') or {})}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _budget_frontier_axis_value(
    *,
    budget_total_calls_per_doc: float,
    full_doc_budget_share: float,
    doc_consumption_mode: str,
    local_split_mode: str,
) -> str:
    return _sanitize_label(
        "_".join(
            [
                f"b{_format_float_label(float(budget_total_calls_per_doc))}",
                f"a{_format_float_label(float(full_doc_budget_share))}",
                str(doc_consumption_mode or "none"),
                str(local_split_mode or "none"),
            ]
        )
    )


def _launch_exact_sanity(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(result["summary_json"]),
                "summary_md": str(result["summary_md"]),
                "tree_neural_exact_sanity_summary_json": str(
                    output_root / "tree_neural_exact_sanity_summary.json"
                ),
                "tree_neural_exact_sanity_summary_md": str(
                    output_root / "tree_neural_exact_sanity_summary.md"
                ),
                "completed_jobs": len(list(result["completed_jobs"])),
                "failed_jobs": len(list(result["failed_jobs"])),
                "skipped_jobs": len(list(result["skipped_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_representation_sufficiency(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    final_summary_json = (
        output_root / "tree_neural_representation_sufficiency_summary.json"
    )
    if not final_summary_json.exists():
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "summary_json": str(output_root / "summary.json"),
                    "summary_md": str(output_root / "summary.md"),
                    "representation_sufficiency_screen_summary_json": str(
                        output_root / "representation_sufficiency_screen_summary.json"
                    ),
                    "representation_sufficiency_lock_summary_json": str(
                        output_root / "representation_sufficiency_lock_summary.json"
                    ),
                    "tree_neural_representation_sufficiency_summary_json": str(
                        final_summary_json
                    ),
                    "failed_jobs": len(list(result["failed_jobs"])),
                    "status": "missing_representation_sufficiency_summary",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    final_summary = json.loads(final_summary_json.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(output_root / "summary.json"),
                "summary_md": str(output_root / "summary.md"),
                "representation_sufficiency_screen_summary_json": str(
                    output_root / "representation_sufficiency_screen_summary.json"
                ),
                "representation_sufficiency_lock_summary_json": str(
                    output_root / "representation_sufficiency_lock_summary.json"
                ),
                "representation_sufficiency_promotion_summary_json": str(
                    output_root / "representation_sufficiency_promotion_summary.json"
                ),
                "tree_neural_representation_sufficiency_summary_json": str(
                    final_summary_json
                ),
                "tree_neural_representation_sufficiency_summary_md": str(
                    output_root / "tree_neural_representation_sufficiency_summary.md"
                ),
                "winning_config_label": str(
                    final_summary.get("winning_config_label", "")
                ),
                "matched_control_label": str(
                    final_summary.get("matched_control_label", "")
                ),
                "final_status": str(final_summary.get("final_status", "")),
                "topology_rerun_recommended": bool(
                    final_summary.get("topology_rerun_recommended", False)
                ),
                "failed_jobs": len(list(result["failed_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_representation_learnability(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    final_summary_json = (
        output_root / "tree_neural_representation_learnability_summary.json"
    )
    if not final_summary_json.exists():
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "summary_json": str(output_root / "summary.json"),
                    "summary_md": str(output_root / "summary.md"),
                    "representation_learnability_winner_summary_json": str(
                        output_root / "representation_learnability_winner_summary.json"
                    ),
                    "tree_neural_representation_learnability_summary_json": str(
                        final_summary_json
                    ),
                    "failed_jobs": len(list(result["failed_jobs"])),
                    "status": "missing_representation_learnability_summary",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    final_summary = json.loads(final_summary_json.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(output_root / "summary.json"),
                "summary_md": str(output_root / "summary.md"),
                "representation_learnability_winner_summary_json": str(
                    output_root / "representation_learnability_winner_summary.json"
                ),
                "tree_neural_representation_learnability_summary_json": str(
                    final_summary_json
                ),
                "tree_neural_representation_learnability_summary_md": str(
                    output_root / "tree_neural_representation_learnability_summary.md"
                ),
                "winner_label": str(final_summary.get("winner_label", "")),
                "matched_control_label": str(
                    final_summary.get("matched_control_label", "")
                ),
                "final_status": str(final_summary.get("final_status", "")),
                "failed_jobs": len(list(result["failed_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_budget_frontier(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    report_pdf = output_root / "tree_oracle_budget_frontier_report.pdf"
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(result["summary_json"]),
                "summary_md": str(result["summary_md"]),
                "tree_oracle_budget_frontier_summary_json": str(
                    output_root / "tree_oracle_budget_frontier_summary.json"
                ),
                "tree_oracle_budget_frontier_summary_md": str(
                    output_root / "tree_oracle_budget_frontier_summary.md"
                ),
                "tree_oracle_budget_frontier_report_pdf": str(report_pdf),
                "completed_jobs": len(list(result["completed_jobs"])),
                "failed_jobs": len(list(result["failed_jobs"])),
                "skipped_jobs": len(list(result["skipped_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_parity(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(output_root / "summary.json"),
                "summary_md": str(output_root / "summary.md"),
                "fair_parity_run_summary_json": str(output_root / "fair_parity_run_summary.json"),
                "fair_parity_run_summary_md": str(output_root / "fair_parity_run_summary.md"),
                "primary_success_met": bool(
                    dict(result.get("payload") or {}).get("tree_fno_fair_parity_summary", {}).get("primary_success_met", False)
                ),
                "secondary_success_met": bool(
                    dict(result.get("payload") or {}).get("tree_fno_fair_parity_summary", {}).get("secondary_success_met", False)
                ),
                "tree_fno_upper_bound_summary_json": str(
                    output_root / "tree_fno_upper_bound_summary.json"
                ),
                "tree_fno_upper_bound_summary_md": str(
                    output_root / "tree_fno_upper_bound_summary.md"
                ),
                "scale_curve_backfilled": bool(args.backfill_on_success),
                "failed_jobs": len(list(result["failed_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_tune(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    tuning_summary = json.loads((output_root / "tuning_summary.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "tuning_summary_json": str(output_root / "tuning_summary.json"),
                "tuning_summary_md": str(output_root / "tuning_summary.md"),
                "screen_summary_json": str(tuning_summary.get("screen_summary_json", "")),
                "comparison_summary_json": str(tuning_summary.get("comparison_summary_json", "")),
                "locked_summary_json": str(tuning_summary.get("locked_summary_json", "")),
                "final_locked_summary_json": str(tuning_summary.get("final_locked_summary_json", "")),
                "winning_config_label": str(tuning_summary.get("winning_config_label", "")),
                "failed_jobs": len(list(result["failed_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_capacity(args: argparse.Namespace) -> int:
    bundle = _cached_capacity_screen_job_bundle(args)
    screen_preflight = dict(bundle.get("screen_preflight") or {})
    if (
        not bool(getattr(args, "plan_only", False))
        and str(screen_preflight.get("status", "")).strip()
        == "unsafe_capacity_screen_layout"
    ):
        output_root = Path(str(args.output_root))
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "screen_summary_json": str(output_root / "screen" / "summary.json"),
                    "locked_summary_json": str(output_root / "locked" / "summary.json"),
                    "tree_fno_capacity_screen_summary_json": str(
                        output_root / "tree_fno_capacity_screen_summary.json"
                    ),
                    "tree_fno_capacity_locked_summary_json": str(
                        output_root / "tree_fno_capacity_locked_summary.json"
                    ),
                    "tree_fno_capacity_locked_summary_md": str(
                        output_root / "tree_fno_capacity_locked_summary.md"
                    ),
                    "status": "unsafe_capacity_screen_layout",
                    "screen_preflight": dict(screen_preflight),
                    "recommended_safe_rerun_flags": list(
                        screen_preflight.get("recommended_safe_rerun_flags") or []
                    ),
                    "failed_jobs": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    locked_summary_json = output_root / "tree_fno_capacity_locked_summary.json"
    if not locked_summary_json.exists():
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "screen_summary_json": str(output_root / "screen" / "summary.json"),
                    "locked_summary_json": str(output_root / "locked" / "summary.json"),
                    "tree_fno_capacity_screen_summary_json": str(
                        output_root / "tree_fno_capacity_screen_summary.json"
                    ),
                    "tree_fno_capacity_locked_summary_json": str(locked_summary_json),
                    "tree_fno_capacity_locked_summary_md": str(
                        output_root / "tree_fno_capacity_locked_summary.md"
                    ),
                    "failed_jobs": len(list(result["failed_jobs"])),
                    "status": "missing_locked_summary",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    locked_summary = json.loads(locked_summary_json.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "screen_summary_json": str(output_root / "screen" / "summary.json"),
                "locked_summary_json": str(output_root / "locked" / "summary.json"),
                "tree_fno_capacity_screen_summary_json": str(output_root / "tree_fno_capacity_screen_summary.json"),
                "tree_fno_capacity_locked_summary_json": str(locked_summary_json),
                "tree_fno_capacity_locked_summary_md": str(output_root / "tree_fno_capacity_locked_summary.md"),
                "winning_config_label": str(locked_summary.get("winning_config_label", "")),
                "failed_jobs": len(list(result["failed_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_study(args: argparse.Namespace) -> int:
    result = _run_scheduler_mode(args)
    if bool(result.get("plan_only", False)):
        return 0
    output_root = Path(str(args.output_root))
    study_summary = json.loads((output_root / "study_summary.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary_json": str(output_root / "summary.json") if (output_root / "summary.json").exists() else "",
                "summary_md": str(output_root / "summary.md") if (output_root / "summary.md").exists() else "",
                "screen_summary_json": str(study_summary.get("screen_summary_json", "")),
                "representative_summary_json": str(study_summary.get("representative_summary_json", "")),
                "study_summary_json": str(output_root / "study_summary.json"),
                "study_summary_md": str(output_root / "study_summary.md"),
                "failed_jobs": len(list(result["failed_jobs"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not result["failed_jobs"] else 1


def _launch_prepare_data(args: argparse.Namespace) -> int:
    payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name=str(getattr(args, "benchmark", "recoverable_v4")),
        hardness_grid=str(getattr(args, "hardness_grid", "")),
        grid_cell_ids=tuple(
            str(value) for value in list(getattr(args, "grid_cell_ids", ()) or ())
        ),
        seeds=tuple(int(seed) for seed in list(getattr(args, "seeds", ()) or ())),
        train_doc_counts=tuple(
            int(value) for value in list(getattr(args, "train_doc_counts", ()) or ())
        ),
        use_cuda=bool(getattr(args, "use_cuda", False)),
        cuda_device=getattr(args, "cuda_device", None),
        torch_threads=int(getattr(args, "torch_threads", 1)),
        config_overrides={
            "prepared_data_root": str(getattr(args, "prepared_data_root", "")),
            "prepared_data_allow_create": bool(
                getattr(args, "prepared_data_allow_create", True)
            ),
            "tree_exact_eval_max_docs": int(
                getattr(args, "tree_exact_eval_max_docs", 0)
            ),
            "max_internal_depth": int(getattr(args, "max_internal_depth", 0)),
            **(
                {"fixed_leaf_tokens": int(args.fixed_leaf_tokens)}
                if getattr(args, "fixed_leaf_tokens", None) is not None
                else {}
            ),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-doc tree-baseline diagnostics across MIG slices."
    )
    subparsers = parser.add_subparsers(dest="mode", required=False)

    def _add_gpu_runtime_args(
        subparser: argparse.ArgumentParser,
        *,
        allow_multi_worker_screen_default: bool = True,
    ) -> None:
        subparser.add_argument(
            "--gpu-runtime-data-mode",
            choices=("resident", "cpu_debug"),
            default="resident",
        )
        subparser.add_argument(
            "--gpu-runtime-bucket-mode",
            choices=("exact_then_bucketed", "leaf_count_auto_queue"),
            default="exact_then_bucketed",
        )
        subparser.add_argument(
            "--gpu-runtime-preload-splits",
            nargs="*",
            default=("train", "val", "test"),
        )
        subparser.add_argument(
            "--gpu-runtime-preload-targets",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        subparser.add_argument("--gpu-runtime-workers-per-mig", type=int, default=1)
        subparser.add_argument(
            "--gpu-runtime-allow-multi-worker-screen",
            action=argparse.BooleanOptionalAction,
            default=allow_multi_worker_screen_default,
        )
        subparser.add_argument(
            "--gpu-runtime-capacity-workers-per-mig",
            type=int,
            default=2,
        )

    def _add_capacity_screen_gpu_runtime_args(
        subparser: argparse.ArgumentParser,
    ) -> None:
        subparser.add_argument(
            "--screen-gpu-runtime-data-mode",
            choices=("resident", "cpu_debug"),
            default=None,
        )
        subparser.add_argument(
            "--screen-gpu-runtime-bucket-mode",
            choices=("exact_then_bucketed", "leaf_count_auto_queue"),
            default=None,
        )
        subparser.add_argument(
            "--screen-gpu-runtime-preload-splits",
            nargs="*",
            default=None,
        )
        subparser.add_argument(
            "--screen-gpu-runtime-preload-targets",
            action=argparse.BooleanOptionalAction,
            default=None,
        )

    def _add_scheduler_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--scheduler-mode",
            choices=("global_per_run",),
            default="global_per_run",
        )
        subparser.add_argument(
            "--scheduler-launch-stagger-seconds",
            type=float,
            default=0.0,
        )
        subparser.add_argument(
            "--cleanup-stale-children",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        subparser.add_argument(
            "--max-gpu-items-per-mig",
            type=int,
            default=1,
        )
        subparser.add_argument(
            "--scheduler-min-mem-available-gib",
            type=float,
            default=128.0,
        )
        subparser.add_argument(
            "--scheduler-min-swap-free-gib",
            type=float,
            default=2.0,
        )
        subparser.add_argument(
            "--plan-only",
            action=argparse.BooleanOptionalAction,
            default=False,
        )

    def _add_tree_stage1_screen_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--tree-stage1-eval-mode", type=str, default="per_epoch")
        subparser.add_argument("--tree-stage1-screen-doc-limit", type=int, default=0)
        subparser.add_argument("--tree-stage1-final-exact-doc-limit", type=int, default=0)
        subparser.add_argument("--exact-metric-selection-doc-limit", type=int, default=0)
        subparser.add_argument("--exact-metric-selection-interval", type=int, default=1)

    def _add_tree_memory_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--tree-exact-eval-max-docs", type=int, default=0)
        subparser.add_argument("--prepared-data-root", type=str, default="")
        subparser.add_argument(
            "--prepared-data-allow-create",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        subparser.add_argument("--base-bundle-path", type=str, default="")
        subparser.add_argument(
            "--diagnostic-detail-mode",
            type=str,
            default="summary",
            choices=("summary", "debug_raw"),
        )
        subparser.add_argument("--raw-diagnostic-artifact-dir", type=str, default="")

    controller = subparsers.add_parser("controller")
    controller.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_neural_full_doc_mig",
    )
    controller.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    controller.add_argument(
        "--hardness-grid",
        type=str,
        default="",
    )
    controller.add_argument(
        "--grid-cell-ids",
        nargs="*",
        default=(),
    )
    controller.add_argument(
        "--train-doc-counts",
        nargs="*",
        type=int,
        default=(1024, 10240),
    )
    controller.add_argument(
        "--families",
        nargs="*",
        choices=list(VALID_BASELINE_FAMILIES),
        default=(
            "tree_ridge_leaf",
            "tree_doc_ridge",
            "tree_neural_c2",
            "tree_neural_c2c3",
            "tree_neural",
        ),
    )
    controller.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    controller.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
        help=(
            "How to shard work across MIGs. 'family_train_seed' launches one worker per seed, "
            "which better saturates many MIG slices."
        ),
    )
    controller.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip jobs whose per-seed run JSONs already exist under the output root. "
            "This allows stopping and resuming seed-sharded runs without rerunning completed seeds."
        ),
    )
    controller.add_argument("--mig-uuids", type=str, default="")
    controller.add_argument("--state-dim", type=int, default=128)
    controller.add_argument("--hidden-dim", type=int, default=512)
    controller.add_argument("--n-epochs", type=int, default=32)
    controller.add_argument("--batch-size", type=int, default=64)
    controller.add_argument("--lr", type=float, default=5e-4)
    controller.add_argument("--weight-decay", type=float, default=0.0)
    controller.add_argument("--tree-local-law-weight", type=float, default=None)
    controller.add_argument("--tree-task-objective-weight", type=float, default=None)
    controller.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    controller.add_argument("--torch-threads", type=int, default=1)
    controller.add_argument(
        "--repeat-closed-form-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    controller.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(controller)
    _add_tree_memory_args(controller)
    _add_scheduler_args(controller)
    _add_gpu_runtime_args(controller)

    exact_sanity = subparsers.add_parser("exact_sanity")
    exact_sanity.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_neural_exact_sanity",
    )
    exact_sanity.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    exact_sanity.add_argument(
        "--train-doc-counts",
        nargs="*",
        type=int,
        default=(1024,),
    )
    exact_sanity.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(0,),
    )
    exact_sanity.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    exact_sanity.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    exact_sanity.add_argument("--mig-uuids", type=str, default="")
    exact_sanity.add_argument("--state-dim", type=int, default=128)
    exact_sanity.add_argument("--hidden-dim", type=int, default=512)
    exact_sanity.add_argument("--n-epochs", type=int, default=32)
    exact_sanity.add_argument("--batch-size", type=int, default=64)
    exact_sanity.add_argument("--lr", type=float, default=5e-4)
    exact_sanity.add_argument("--weight-decay", type=float, default=0.0)
    exact_sanity.add_argument("--tree-local-law-weight", type=float, default=0.8)
    exact_sanity.add_argument("--tree-task-objective-weight", type=float, default=None)
    exact_sanity.add_argument("--tree-c1-relative-weight", type=float, default=1.0)
    exact_sanity.add_argument("--tree-c2-relative-weight", type=float, default=1.0)
    exact_sanity.add_argument("--tree-c3-relative-weight", type=float, default=1.0)
    exact_sanity.add_argument(
        "--extra-high-rates",
        nargs="*",
        type=float,
        default=(),
    )
    exact_sanity.add_argument(
        "--tree-checkpoint-metric",
        type=str,
        default="val_exact_sketch_direct",
    )
    exact_sanity.add_argument(
        "--tree-stage1-checkpoint-metric",
        type=str,
        default="val_theorem_bootstrap_direct",
    )
    exact_sanity.add_argument("--tree-stage1-artifact-dir", type=str, default="")
    exact_sanity.add_argument("--tree-stage1-root-weight", type=float, default=0.0)
    exact_sanity.add_argument("--tree-join-bit-weight", type=float, default=1.0)
    exact_sanity.add_argument(
        "--tree-training-schedule",
        type=str,
        default="two_stage",
    )
    exact_sanity.add_argument("--tree-stage1-epochs", type=int, default=12)
    exact_sanity.add_argument("--tree-stage2-epochs", type=int, default=20)
    exact_sanity.add_argument(
        "--tree-task-head-mode",
        type=str,
        default="theorem_feature_scalar",
    )
    exact_sanity.add_argument(
        "--tree-theorem-surface-mode",
        type=str,
        default="shared_bottleneck",
    )
    exact_sanity.add_argument(
        "--tree-theorem-count-head-mode",
        type=str,
        default="scalar_mse",
    )
    exact_sanity.add_argument("--tree-theorem-feature-dim", type=int, default=48)
    exact_sanity.add_argument(
        "--tree-theorem-feature-hidden-dim", type=int, default=256
    )
    exact_sanity.add_argument("--tree-merge-hidden-dim", type=int, default=0)
    exact_sanity.add_argument("--tree-phi-compose-weight", type=float, default=1.0)
    exact_sanity.add_argument(
        "--tree-phi-contrastive-weight", type=float, default=0.25
    )
    exact_sanity.add_argument(
        "--tree-phi-alignment-loss", type=str, default="cosine_mse"
    )
    exact_sanity.add_argument("--tree-c2-mode", type=str, default="reconstruction")
    exact_sanity.add_argument(
        "--theorem-feature-adapter", type=str, default="markov_count_sketch"
    )
    exact_sanity.add_argument(
        "--theorem-pair-same-threshold", type=float, default=None
    )
    exact_sanity.add_argument(
        "--theorem-pair-diff-threshold", type=float, default=None
    )
    exact_sanity.add_argument(
        "--tree-theorem-count-ordinal-weight",
        type=float,
        default=1.0,
    )
    exact_sanity.add_argument(
        "--tree-theorem-count-scalar-aux-weight",
        type=float,
        default=0.25,
    )
    exact_sanity.add_argument(
        "--tree-theorem-count-threshold-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    exact_sanity.add_argument(
        "--tree-summary-spec-root-mode",
        type=str,
        default="factored_theorem_readout",
    )
    exact_sanity.add_argument("--tree-theorem-count-dim", type=int, default=8)
    exact_sanity.add_argument("--tree-theorem-first-dim", type=int, default=8)
    exact_sanity.add_argument("--tree-theorem-last-dim", type=int, default=8)
    exact_sanity.add_argument(
        "--leaf-supervision-kind",
        type=str,
        default="full_sketch",
    )
    exact_sanity.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    exact_sanity.add_argument("--torch-threads", type=int, default=1)
    exact_sanity.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_scheduler_args(exact_sanity)
    _add_gpu_runtime_args(exact_sanity, allow_multi_worker_screen_default=False)

    representation_sufficiency = subparsers.add_parser("representation_sufficiency")
    representation_sufficiency.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_neural_representation_sufficiency",
    )
    representation_sufficiency.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    representation_sufficiency.add_argument(
        "--screen-train-doc-count",
        type=int,
        default=REPRESENTATION_SUFFICIENCY_DEFAULT_SCREEN_DOC_COUNT,
    )
    representation_sufficiency.add_argument(
        "--lock-train-doc-count",
        type=int,
        default=REPRESENTATION_SUFFICIENCY_DEFAULT_LOCK_DOC_COUNT,
    )
    representation_sufficiency.add_argument(
        "--promotion-train-doc-count",
        type=int,
        default=REPRESENTATION_SUFFICIENCY_DEFAULT_PROMOTION_DOC_COUNT,
    )
    representation_sufficiency.add_argument(
        "--screen-seeds",
        nargs="*",
        type=int,
        default=(0, 1),
    )
    representation_sufficiency.add_argument(
        "--lock-seeds",
        nargs="*",
        type=int,
        default=(0, 1),
    )
    representation_sufficiency.add_argument(
        "--promotion-seeds",
        nargs="*",
        type=int,
        default=(0, 1),
    )
    representation_sufficiency.add_argument(
        "--representation-state-dims",
        nargs="*",
        type=int,
        default=(128,),
    )
    representation_sufficiency.add_argument(
        "--representation-count-head-modes",
        nargs="*",
        type=str,
        default=REPRESENTATION_SUFFICIENCY_DEFAULT_COUNT_HEAD_MODES,
    )
    representation_sufficiency.add_argument(
        "--top-k",
        type=int,
        default=REPRESENTATION_SUFFICIENCY_DEFAULT_TOP_K,
    )
    representation_sufficiency.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    representation_sufficiency.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    representation_sufficiency.add_argument("--mig-uuids", type=str, default="")
    representation_sufficiency.add_argument("--n-epochs", type=int, default=32)
    representation_sufficiency.add_argument("--batch-size", type=int, default=64)
    representation_sufficiency.add_argument("--lr", type=float, default=5e-4)
    representation_sufficiency.add_argument("--weight-decay", type=float, default=0.0)
    representation_sufficiency.add_argument(
        "--tree-local-law-weight",
        type=float,
        default=0.8,
    )
    representation_sufficiency.add_argument(
        "--tree-task-objective-weight",
        type=float,
        default=None,
    )
    representation_sufficiency.add_argument(
        "--tree-c1-relative-weight",
        type=float,
        default=1.0,
    )
    representation_sufficiency.add_argument(
        "--tree-c2-relative-weight",
        type=float,
        default=1.0,
    )
    representation_sufficiency.add_argument(
        "--tree-c3-relative-weight",
        type=float,
        default=1.0,
    )
    representation_sufficiency.add_argument(
        "--tree-stage1-artifact-dir",
        type=str,
        default="",
    )
    representation_sufficiency.add_argument(
        "--tree-join-bit-weight",
        type=float,
        default=1.0,
    )
    representation_sufficiency.add_argument(
        "--tree-stage1-epochs",
        type=int,
        default=12,
    )
    representation_sufficiency.add_argument(
        "--tree-stage2-epochs",
        type=int,
        default=20,
    )
    representation_sufficiency.add_argument(
        "--tree-theorem-count-head-mode",
        type=str,
        default="scalar_mse",
    )
    representation_sufficiency.add_argument(
        "--tree-theorem-count-ordinal-weight",
        type=float,
        default=1.0,
    )
    representation_sufficiency.add_argument(
        "--tree-theorem-count-scalar-aux-weight",
        type=float,
        default=0.25,
    )
    representation_sufficiency.add_argument(
        "--tree-theorem-count-threshold-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    representation_sufficiency.add_argument(
        "--tree-merge-hidden-dim",
        type=int,
        default=0,
    )
    representation_sufficiency.add_argument(
        "--tree-phi-compose-weight",
        type=float,
        default=1.0,
    )
    representation_sufficiency.add_argument(
        "--tree-phi-contrastive-weight",
        type=float,
        default=0.25,
    )
    representation_sufficiency.add_argument(
        "--tree-phi-alignment-loss",
        type=str,
        default="cosine_mse",
    )
    representation_sufficiency.add_argument("--torch-threads", type=int, default=1)
    representation_sufficiency.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(representation_sufficiency)
    _add_tree_memory_args(representation_sufficiency)
    _add_scheduler_args(representation_sufficiency)
    _add_gpu_runtime_args(
        representation_sufficiency,
        allow_multi_worker_screen_default=False,
    )

    representation_learnability = subparsers.add_parser("representation_learnability")
    representation_learnability.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_neural_representation_learnability",
    )
    representation_learnability.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    representation_learnability.add_argument(
        "--winner-train-doc-count",
        type=int,
        default=REPRESENTATION_LEARNABILITY_DEFAULT_WINNER_DOC_COUNT,
    )
    representation_learnability.add_argument(
        "--sweep-train-doc-counts",
        nargs="*",
        type=int,
        default=REPRESENTATION_LEARNABILITY_DEFAULT_SWEEP_DOC_COUNTS,
    )
    representation_learnability.add_argument(
        "--winner-seeds",
        nargs="*",
        type=int,
        default=(0, 1),
    )
    representation_learnability.add_argument(
        "--sweep-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3),
    )
    representation_learnability.add_argument(
        "--benchmark-cells",
        nargs="*",
        type=str,
        default=REPRESENTATION_LEARNABILITY_DEFAULT_BENCHMARK_CELLS,
    )
    representation_learnability.add_argument(
        "--full-structural-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    representation_learnability.add_argument(
        "--representation-state-dims",
        nargs="*",
        type=int,
        default=(128, 256),
    )
    representation_learnability.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    representation_learnability.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    representation_learnability.add_argument("--mig-uuids", type=str, default="")
    representation_learnability.add_argument("--n-epochs", type=int, default=32)
    representation_learnability.add_argument("--batch-size", type=int, default=64)
    representation_learnability.add_argument("--lr", type=float, default=5e-4)
    representation_learnability.add_argument("--weight-decay", type=float, default=0.0)
    representation_learnability.add_argument(
        "--tree-local-law-weight",
        type=float,
        default=0.8,
    )
    representation_learnability.add_argument(
        "--tree-task-objective-weight",
        type=float,
        default=None,
    )
    representation_learnability.add_argument(
        "--tree-c1-relative-weight",
        type=float,
        default=1.0,
    )
    representation_learnability.add_argument(
        "--tree-c2-relative-weight",
        type=float,
        default=1.0,
    )
    representation_learnability.add_argument(
        "--tree-c3-relative-weight",
        type=float,
        default=1.0,
    )
    representation_learnability.add_argument(
        "--tree-stage1-artifact-dir",
        type=str,
        default="",
    )
    representation_learnability.add_argument(
        "--tree-join-bit-weight",
        type=float,
        default=1.0,
    )
    representation_learnability.add_argument(
        "--tree-stage1-epochs",
        type=int,
        default=12,
    )
    representation_learnability.add_argument(
        "--tree-stage2-epochs",
        type=int,
        default=20,
    )
    representation_learnability.add_argument(
        "--tree-theorem-count-head-mode",
        type=str,
        default="scalar_mse",
    )
    representation_learnability.add_argument(
        "--tree-theorem-count-ordinal-weight",
        type=float,
        default=1.0,
    )
    representation_learnability.add_argument(
        "--tree-theorem-count-scalar-aux-weight",
        type=float,
        default=0.25,
    )
    representation_learnability.add_argument(
        "--tree-theorem-count-threshold-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    representation_learnability.add_argument(
        "--tree-merge-hidden-dim",
        type=int,
        default=0,
    )
    representation_learnability.add_argument(
        "--tree-phi-compose-weight",
        type=float,
        default=1.0,
    )
    representation_learnability.add_argument(
        "--tree-phi-contrastive-weight",
        type=float,
        default=0.25,
    )
    representation_learnability.add_argument(
        "--tree-phi-alignment-loss",
        type=str,
        default="cosine_mse",
    )
    representation_learnability.add_argument("--torch-threads", type=int, default=1)
    representation_learnability.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(representation_learnability)
    _add_tree_memory_args(representation_learnability)
    _add_scheduler_args(representation_learnability)
    _add_gpu_runtime_args(
        representation_learnability,
        allow_multi_worker_screen_default=False,
    )

    budget_frontier = subparsers.add_parser("budget_frontier")
    budget_frontier.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_oracle_budget_frontier",
    )
    budget_frontier.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    budget_frontier.add_argument(
        "--hardness-grid",
        type=str,
        default="",
    )
    budget_frontier.add_argument(
        "--grid-cell-ids",
        nargs="*",
        default=(),
    )
    budget_frontier.add_argument(
        "--train-doc-count",
        type=int,
        default=10240,
    )
    budget_frontier.add_argument(
        "--tree-families",
        nargs="*",
        choices=list(VALID_BASELINE_FAMILIES),
        default=BUDGET_FRONTIER_TREE_FAMILIES,
    )
    budget_frontier.add_argument(
        "--reference-families",
        nargs="*",
        choices=list(VALID_BASELINE_FAMILIES),
        default=BUDGET_FRONTIER_REFERENCE_FAMILIES,
    )
    budget_frontier.add_argument(
        "--budget-calls-per-doc",
        nargs="*",
        type=float,
        default=BUDGET_FRONTIER_BUDGETS_PER_DOC,
    )
    budget_frontier.add_argument(
        "--full-doc-budget-shares",
        nargs="*",
        type=float,
        default=BUDGET_FRONTIER_FULL_DOC_SHARES,
    )
    budget_frontier.add_argument(
        "--doc-consumption-modes",
        nargs="*",
        choices=BUDGET_FRONTIER_DOC_CONSUMPTION_MODES,
        default=BUDGET_FRONTIER_DOC_CONSUMPTION_MODES,
    )
    budget_frontier.add_argument(
        "--local-split-modes",
        nargs="*",
        choices=BUDGET_FRONTIER_LOCAL_SPLIT_MODES,
        default=BUDGET_FRONTIER_LOCAL_SPLIT_MODES,
    )
    budget_frontier.add_argument(
        "--local-allocation-policy",
        choices=(BUDGET_FRONTIER_ALLOCATION_POLICY,),
        default=BUDGET_FRONTIER_ALLOCATION_POLICY,
    )
    budget_frontier.add_argument(
        "--budget-tree-config-mode",
        choices=("parity", "default"),
        default="parity",
        help=(
            "Tree config used for the budget study. 'parity' uses the fair_fno_v1 "
            "tree setup by default; 'default' preserves the legacy mse tree setup."
        ),
    )
    budget_frontier.add_argument("--capacity-root", type=str, default="")
    budget_frontier.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    budget_frontier.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    budget_frontier.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    budget_frontier.add_argument("--mig-uuids", type=str, default="")
    budget_frontier.add_argument("--state-dim", type=int, default=128)
    budget_frontier.add_argument("--hidden-dim", type=int, default=512)
    budget_frontier.add_argument("--n-epochs", type=int, default=32)
    budget_frontier.add_argument("--batch-size", type=int, default=64)
    budget_frontier.add_argument("--lr", type=float, default=5e-4)
    budget_frontier.add_argument("--weight-decay", type=float, default=0.0)
    budget_frontier.add_argument("--tree-local-law-weight", type=float, default=0.3)
    budget_frontier.add_argument("--tree-task-objective-weight", type=float, default=None)
    budget_frontier.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    budget_frontier.add_argument("--torch-threads", type=int, default=1)
    budget_frontier.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(budget_frontier)
    _add_tree_memory_args(budget_frontier)
    _add_scheduler_args(budget_frontier)
    _add_gpu_runtime_args(budget_frontier, allow_multi_worker_screen_default=False)

    parity = subparsers.add_parser("parity")
    parity.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_fno_fair_parity",
    )
    parity.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    parity.add_argument(
        "--gate-train-doc-count",
        type=int,
        default=PARITY_GATE_TRAIN_DOC_COUNT,
    )
    parity.add_argument(
        "--scale-train-doc-counts",
        nargs="*",
        type=int,
        default=PARITY_SCALE_CURVE_TRAIN_DOC_COUNTS,
    )
    parity.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    parity.add_argument(
        "--tree-families",
        nargs="*",
        type=str,
        choices=list(VALID_BASELINE_FAMILIES),
        default=PARITY_TREE_FAMILIES,
    )
    parity.add_argument(
        "--fno-families",
        nargs="*",
        type=str,
        choices=list(VALID_BASELINE_FAMILIES),
        default=PARITY_FNO_FAMILIES,
    )
    parity.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    parity.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parity.add_argument(
        "--backfill-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Legacy flag name. When enabled, always run the parity scale "
            "backfill after the gate so publication runs emit the full "
            "multi-scale comparison curve."
        ),
    )
    parity.add_argument("--capacity-root", type=str, default="")
    parity.add_argument(
        "--run-aux-upper-bound",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parity.add_argument(
        "--upper-bound-aux-fractions",
        nargs="*",
        type=float,
        default=(0.25, 1.0),
    )
    parity.add_argument("--mig-uuids", type=str, default="")
    parity.add_argument("--state-dim", type=int, default=128)
    parity.add_argument("--hidden-dim", type=int, default=512)
    parity.add_argument("--n-epochs", type=int, default=32)
    parity.add_argument("--batch-size", type=int, default=64)
    parity.add_argument("--lr", type=float, default=5e-4)
    parity.add_argument("--weight-decay", type=float, default=0.0)
    parity.add_argument("--tree-local-law-weight", type=float, default=None)
    parity.add_argument("--tree-task-objective-weight", type=float, default=None)
    parity.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    parity.add_argument("--torch-threads", type=int, default=1)
    parity.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(parity)
    _add_tree_memory_args(parity)
    _add_scheduler_args(parity)
    _add_gpu_runtime_args(parity, allow_multi_worker_screen_default=False)

    tune = subparsers.add_parser("tune")
    tune.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_neural_full_doc_tuning",
    )
    tune.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    tune.add_argument(
        "--train-doc-count",
        type=int,
        default=10240,
    )
    tune.add_argument(
        "--priority-family",
        type=str,
        choices=("tree_neural",),
        default="tree_neural",
    )
    tune.add_argument(
        "--comparison-families",
        nargs="*",
        choices=list(VALID_BASELINE_FAMILIES),
        default=(
            "tree_ridge_leaf",
            "tree_doc_ridge",
            "tree_neural_c2",
            "tree_neural_c2c3",
        ),
    )
    tune.add_argument(
        "--screen-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2),
    )
    tune.add_argument(
        "--locked-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    tune.add_argument(
        "--top-k",
        type=int,
        default=2,
    )
    tune.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    tune.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    tune.add_argument("--mig-uuids", type=str, default="")
    tune.add_argument("--state-dim", type=int, default=128)
    tune.add_argument("--hidden-dim", type=int, default=512)
    tune.add_argument("--batch-size", type=int, default=64)
    tune.add_argument("--weight-decay", type=float, default=0.0)
    tune.add_argument(
        "--screen-n-epochs",
        nargs="*",
        type=int,
        default=(32, 64),
    )
    tune.add_argument(
        "--screen-lrs",
        nargs="*",
        type=float,
        default=(5e-4, 2e-4),
    )
    tune.add_argument(
        "--screen-tree-local-law-weights",
        nargs="*",
        type=float,
        default=(0.15, 0.3, 0.45),
    )
    tune.add_argument("--comparison-n-epochs", type=int, default=32)
    tune.add_argument("--comparison-lr", type=float, default=5e-4)
    tune.add_argument("--comparison-tree-local-law-weight", type=float, default=0.3)
    tune.add_argument("--tree-task-objective-weight", type=float, default=None)
    tune.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    tune.add_argument("--torch-threads", type=int, default=1)
    tune.add_argument(
        "--repeat-closed-form-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    tune.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(tune)
    _add_tree_memory_args(tune)
    _add_scheduler_args(tune)
    _add_gpu_runtime_args(tune)

    capacity = subparsers.add_parser("capacity")
    capacity.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_fno_capacity",
    )
    capacity.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    capacity.add_argument(
        "--base-config-preset",
        type=str,
        default="",
    )
    capacity.add_argument(
        "--capacity-profile",
        type=str,
        choices=CAPACITY_PROFILE_CHOICES,
        default=ROOT_ONLY_CAPACITY_PROFILE_DEFAULT,
    )
    capacity.add_argument(
        "--train-doc-count",
        type=int,
        default=10240,
    )
    capacity.add_argument(
        "--priority-family",
        type=str,
        choices=(CAPACITY_PRIORITY_FAMILY,),
        default=CAPACITY_PRIORITY_FAMILY,
    )
    capacity.add_argument(
        "--screen-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2),
    )
    capacity.add_argument(
        "--locked-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    capacity.add_argument("--top-k", type=int, default=3)
    capacity.add_argument(
        "--capacity-widths",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-modes",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-layers",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-state-dims",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-hidden-dims",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-n-epochs",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-tree-training-schedules",
        nargs="*",
        type=str,
        default=None,
    )
    capacity.add_argument(
        "--capacity-tree-checkpoint-metrics",
        nargs="*",
        type=str,
        default=None,
    )
    capacity.add_argument(
        "--capacity-tree-stage1-checkpoint-metrics",
        nargs="*",
        type=str,
        default=None,
    )
    capacity.add_argument(
        "--capacity-tree-stage1-root-weights",
        nargs="*",
        type=float,
        default=None,
    )
    capacity.add_argument(
        "--capacity-slot-counts",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--capacity-fixed-leaf-tokens",
        nargs="*",
        type=int,
        default=None,
    )
    capacity.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    capacity.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    capacity.add_argument("--mig-uuids", type=str, default="")
    capacity.add_argument("--state-dim", type=int, default=None)
    capacity.add_argument("--hidden-dim", type=int, default=None)
    capacity.add_argument("--n-epochs", type=int, default=None)
    capacity.add_argument("--batch-size", type=int, default=None)
    capacity.add_argument("--lr", type=float, default=None)
    capacity.add_argument("--weight-decay", type=float, default=None)
    capacity.add_argument("--tree-local-law-weight", type=float, default=None)
    capacity.add_argument("--tree-task-objective-weight", type=float, default=None)
    capacity.add_argument("--leaf-supervision-kind", type=str, default=None)
    capacity.add_argument("--leaf-label-rate", type=float, default=None)
    capacity.add_argument("--internal-supervision-kind", type=str, default=None)
    capacity.add_argument("--internal-label-rate", type=float, default=None)
    capacity.add_argument("--max-internal-depth", type=int, default=None)
    capacity.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    capacity.add_argument("--torch-threads", type=int, default=1)
    capacity.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(capacity)
    _add_tree_memory_args(capacity)
    _add_scheduler_args(capacity)
    _add_gpu_runtime_args(capacity)
    _add_capacity_screen_gpu_runtime_args(capacity)
    capacity.add_argument(
        "--screen-max-concurrent-per-physical-gpu",
        type=int,
        default=0,
    )
    capacity.add_argument(
        "--screen-device-order",
        choices=("input", "interleave_by_physical_gpu"),
        default="input",
    )
    capacity.set_defaults(
        gpu_runtime_allow_multi_worker_screen=False,
        gpu_runtime_capacity_workers_per_mig=1,
        gpu_runtime_data_mode=None,
        gpu_runtime_bucket_mode=None,
    )

    prepare_data = subparsers.add_parser("prepare_data")
    prepare_data.add_argument("--benchmark", type=str, default="recoverable_v4")
    prepare_data.add_argument("--hardness-grid", type=str, default="")
    prepare_data.add_argument("--grid-cell-ids", nargs="*", default=())
    prepare_data.add_argument("--train-doc-counts", nargs="*", type=int, default=())
    prepare_data.add_argument("--seeds", nargs="*", type=int, default=(0, 1, 2, 3, 4))
    prepare_data.add_argument("--fixed-leaf-tokens", type=int, default=None)
    prepare_data.add_argument("--max-internal-depth", type=int, default=0)
    prepare_data.add_argument("--torch-threads", type=int, default=1)
    prepare_data.add_argument("--cuda-device", type=int, default=None)
    prepare_data.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    _add_tree_memory_args(prepare_data)

    study = subparsers.add_parser("study")
    study.add_argument(
        "--output-root",
        type=str,
        default="outputs/tree_neural_full_doc_study",
    )
    study.add_argument(
        "--tuning-root",
        type=str,
        required=True,
    )
    study.add_argument(
        "--study-name",
        type=str,
        choices=("leaf_geometry", "structural_complexity"),
        required=True,
    )
    study.add_argument(
        "--benchmark",
        type=str,
        default="recoverable_v4",
    )
    study.add_argument(
        "--train-doc-count",
        type=int,
        default=10240,
    )
    study.add_argument(
        "--families",
        nargs="*",
        choices=list(VALID_BASELINE_FAMILIES),
        default=(
            "tree_ridge_leaf",
            "tree_doc_ridge",
            "tree_neural_c2",
            "tree_neural_c2c3",
            "tree_neural",
        ),
    )
    study.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    study.add_argument(
        "--leaf-tokens",
        nargs="*",
        type=int,
        default=(8, 16, 32),
    )
    study.add_argument(
        "--screen-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2),
    )
    study.add_argument(
        "--locked-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2, 3, 4),
    )
    study.add_argument(
        "--job-granularity",
        choices=("family_train_seed", "family_train"),
        default="family_train_seed",
    )
    study.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    study.add_argument("--mig-uuids", type=str, default="")
    study.add_argument("--state-dim", type=int, default=128)
    study.add_argument("--hidden-dim", type=int, default=512)
    study.add_argument("--batch-size", type=int, default=64)
    study.add_argument("--weight-decay", type=float, default=0.0)
    study.add_argument("--comparison-n-epochs", type=int, default=32)
    study.add_argument("--comparison-lr", type=float, default=5e-4)
    study.add_argument("--comparison-tree-local-law-weight", type=float, default=0.3)
    study.add_argument("--tree-task-objective-weight", type=float, default=None)
    study.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    study.add_argument("--torch-threads", type=int, default=1)
    study.add_argument(
        "--repeat-closed-form-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    study.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_tree_stage1_screen_args(study)
    _add_tree_memory_args(study)
    _add_scheduler_args(study)
    _add_gpu_runtime_args(study)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--job-name", type=str, required=True)
    worker.add_argument("--output-dir", type=str, required=True)
    worker.add_argument("--family", type=str, required=True)
    worker.add_argument("--train-doc-count", type=int, required=True)
    worker.add_argument("--benchmark", type=str, default="recoverable_v4")
    worker.add_argument("--hardness-grid", type=str, default="")
    worker.add_argument("--grid-cell-ids", nargs="*", default=())
    worker.add_argument("--seeds", nargs="*", type=int, default=(0, 1, 2, 3, 4))
    worker.add_argument("--state-dim", type=int, required=True)
    worker.add_argument("--hidden-dim", type=int, required=True)
    worker.add_argument("--n-epochs", type=int, required=True)
    worker.add_argument("--batch-size", type=int, required=True)
    worker.add_argument("--lr", type=float, required=True)
    worker.add_argument("--weight-decay", type=float, required=True)
    worker.add_argument("--fixed-leaf-tokens", type=int, default=None)
    worker.add_argument("--config-label", type=str, default="")
    worker.add_argument(
        "--config-spec-json-path",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--tuning-stage", type=str, default="")
    worker.add_argument(
        "--test-metrics-hidden-during-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    worker.add_argument("--study-name", type=str, default="")
    worker.add_argument("--study-axis", type=str, default="")
    worker.add_argument("--axis-value", type=str, default="")
    worker.add_argument("--locked-tree-neural-config-label", type=str, default="")
    worker.add_argument("--selection-metric", type=str, default="")
    worker.add_argument("--tree-local-law-weight", type=float, default=None)
    worker.add_argument("--tree-task-objective-weight", type=float, default=None)
    worker.add_argument(
        "--tree-local-weighting-mode",
        type=str,
        default="fixed_k_hajek",
    )
    worker.add_argument("--tree-exact-collapse-mode", type=str, default="")
    worker.add_argument(
        "--official-fno-preserve-requested-leaf-tokens",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    worker.add_argument(
        "--preserve-requested-leaf-tokens",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--comparison-mode", type=str, default="legacy")
    worker.add_argument("--tree-c1-relative-weight", type=float, default=1.0)
    worker.add_argument("--tree-c2-relative-weight", type=float, default=1.0)
    worker.add_argument("--tree-c3-relative-weight", type=float, default=1.0)
    worker.add_argument("--tree-leaf-fno-width", type=int, default=None)
    worker.add_argument("--tree-leaf-fno-n-modes", type=int, default=None)
    worker.add_argument("--tree-leaf-fno-n-layers", type=int, default=None)
    worker.add_argument("--tree-model-version", type=str, default="")
    worker.add_argument("--tree-batch-runtime-mode", type=str, default="")
    worker.add_argument("--tree-root-supervision-kind", type=str, default="mse")
    worker.add_argument(
        "--tree-document-loss-normalization-mode",
        type=str,
        default="auto",
    )
    worker.add_argument("--tree-supervision-source", type=str, default="rate")
    worker.add_argument("--tree-checkpoint-metric", type=str, default="val_root_mae")
    worker.add_argument(
        "--tree-stage1-checkpoint-metric",
        type=str,
        default="val_root_mae",
    )
    worker.add_argument("--tree-stage1-eval-mode", type=str, default="per_epoch")
    worker.add_argument("--tree-stage1-screen-doc-limit", type=int, default=0)
    worker.add_argument("--tree-stage1-final-exact-doc-limit", type=int, default=0)
    worker.add_argument("--exact-metric-selection-doc-limit", type=int, default=0)
    worker.add_argument("--exact-metric-selection-interval", type=int, default=1)
    worker.add_argument("--tree-exact-eval-max-docs", type=int, default=0)
    worker.add_argument("--tree-posttrain-train-doc-limit", type=int, default=0)
    worker.add_argument("--tree-batch-pack-mode", type=str, default="")
    worker.add_argument("--tree-batch-token-budget", type=int, default=0)
    worker.add_argument("--tree-batch-node-budget", type=int, default=0)
    worker.add_argument(
        "--tree-batch-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    worker.add_argument("--tree-batch-structural-pad-limit", type=float, default=0.5)
    worker.add_argument("--tree-batch-auto-queue-min-docs", type=int, default=8)
    worker.add_argument(
        "--tree-batch-auto-queue-min-fill-ratio",
        type=float,
        default=0.5,
    )
    worker.add_argument("--tree-eval-workers-per-mig", type=int, default=0)
    worker.add_argument("--tree-stage1-artifact-dir", type=str, default="")
    worker.add_argument("--prepared-data-root", type=str, default="")
    worker.add_argument(
        "--prepared-data-allow-create",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    worker.add_argument("--base-bundle-path", type=str, default="")
    worker.add_argument(
        "--diagnostic-detail-mode",
        type=str,
        default="summary",
        choices=("summary", "debug_raw"),
    )
    worker.add_argument(
        "--posttrain-diagnostics-mode",
        type=str,
        default="",
        choices=("", "full", "minimal"),
    )
    worker.add_argument("--raw-diagnostic-artifact-dir", type=str, default="")
    worker.add_argument("--tree-stage1-root-weight", type=float, default=0.0)
    worker.add_argument("--tree-join-bit-weight", type=float, default=0.0)
    worker.add_argument("--tree-training-schedule", type=str, default="two_stage")
    worker.add_argument("--tree-stage1-epochs", type=int, default=0)
    worker.add_argument("--tree-stage2-epochs", type=int, default=0)
    worker.add_argument("--tree-task-head-mode", type=str, default="full_state_scalar")
    worker.add_argument("--tree-theorem-surface-mode", type=str, default="slotwise")
    worker.add_argument(
        "--tree-theorem-count-head-mode",
        type=str,
        default="scalar_mse",
    )
    worker.add_argument("--tree-theorem-feature-dim", type=int, default=48)
    worker.add_argument("--tree-theorem-feature-hidden-dim", type=int, default=256)
    worker.add_argument("--tree-merge-hidden-dim", type=int, default=0)
    worker.add_argument("--tree-theorem-score-dim", type=int, default=0)
    worker.add_argument("--tree-theorem-fiber-dim", type=int, default=0)
    worker.add_argument("--tree-theorem-aux-dim", type=int, default=0)
    worker.add_argument("--tree-score-merge-mode", type=str, default="gated_affine")
    worker.add_argument("--tree-phi-compose-weight", type=float, default=1.0)
    worker.add_argument("--tree-phi-contrastive-weight", type=float, default=0.25)
    worker.add_argument(
        "--tree-phi-alignment-loss", type=str, default="cosine_mse"
    )
    worker.add_argument("--tree-c2-mode", type=str, default="reconstruction")
    worker.add_argument(
        "--theorem-feature-adapter", type=str, default="markov_count_sketch"
    )
    worker.add_argument("--oracle-metric-name", type=str, default="")
    worker.add_argument("--oracle-same-threshold", type=float, default=0.0)
    worker.add_argument("--oracle-diff-threshold", type=float, default=0.0)
    worker.add_argument("--theorem-pair-same-threshold", type=float, default=None)
    worker.add_argument("--theorem-pair-diff-threshold", type=float, default=None)
    worker.add_argument(
        "--tree-theorem-count-ordinal-weight",
        type=float,
        default=1.0,
    )
    worker.add_argument(
        "--tree-theorem-count-scalar-aux-weight",
        type=float,
        default=0.25,
    )
    worker.add_argument(
        "--tree-theorem-count-threshold-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    worker.add_argument(
        "--tree-summary-spec-root-mode",
        type=str,
        default="task_split_ablation",
    )
    worker.add_argument("--aligned-sketch-surface", type=str, default="")
    worker.add_argument("--summary-spec-name", type=str, default="")
    worker.add_argument("--slot-count", type=int, default=0)
    worker.add_argument("--tree-theorem-count-dim", type=int, default=0)
    worker.add_argument("--tree-theorem-first-dim", type=int, default=0)
    worker.add_argument("--tree-theorem-last-dim", type=int, default=0)
    worker.add_argument("--internal-supervision-kind", type=str, default="none")
    worker.add_argument("--internal-label-rate", type=float, default=0.0)
    worker.add_argument("--max-internal-depth", type=int, default=0)
    worker.add_argument("--leaf-supervision-kind", type=str, default="full_sketch")
    worker.add_argument("--leaf-label-rate", type=float, default=1.0)
    worker.add_argument(
        "--leaf-exact-supervision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    worker.add_argument("--root-weight", type=float, default=1.0)
    worker.add_argument("--schedule-consistency-weight", type=float, default=0.0)
    worker.add_argument("--endpoint-loss-scale", type=float, default=1.0)
    worker.add_argument("--doc-sequence-train-fraction", type=float, default=0.0)
    worker.add_argument("--budget-total-calls", type=int, default=0)
    worker.add_argument("--budget-total-calls-per-doc", type=float, default=0.0)
    worker.add_argument("--mass-target-per-doc", type=float, default=float("nan"))
    worker.add_argument("--full-doc-budget-share", type=float, default=1.0)
    worker.add_argument("--doc-consumption-mode", type=str, default="")
    worker.add_argument("--local-split-mode", type=str, default="")
    worker.add_argument("--local-allocation-policy", type=str, default="")
    worker.add_argument("--package-semantics", type=str, default="")
    worker.add_argument("--depth-discount-gamma", type=float, default=1.0)
    worker.add_argument("--debug-snapshot-json", type=str, default="")
    worker.add_argument("--memory-probe-jsonl", type=str, default="")
    worker.add_argument(
        "--debug-stop-after-snapshot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    worker.add_argument("--torch-threads", type=int, default=1)
    worker.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_gpu_runtime_args(worker)

    replay_worker_snapshot = subparsers.add_parser("replay_worker_snapshot")
    replay_worker_snapshot.add_argument("--snapshot-json", type=str, required=True)
    replay_worker_snapshot.add_argument("--output-dir", type=str, default="")
    replay_worker_snapshot.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    replay_worker_snapshot.add_argument("--cuda-device", type=int, default=None)
    replay_worker_snapshot.add_argument("--torch-threads", type=int, default=None)
    replay_worker_snapshot.add_argument("--memory-probe-jsonl", type=str, default="")

    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    mode = str(args.mode or "controller")
    if mode == "worker":
        print(json.dumps(_worker_payload(args), sort_keys=True))
        return 0
    if mode == "replay_worker_snapshot":
        print(json.dumps(_replay_worker_snapshot_payload(args), sort_keys=True))
        return 0
    if mode == "exact_sanity":
        return _launch_exact_sanity(args)
    if mode == "representation_sufficiency":
        return _launch_representation_sufficiency(args)
    if mode == "representation_learnability":
        return _launch_representation_learnability(args)
    if mode == "budget_frontier":
        return _launch_budget_frontier(args)
    if mode == "parity":
        return _launch_parity(args)
    if mode == "capacity":
        return _launch_capacity(args)
    if mode == "prepare_data":
        return _launch_prepare_data(args)
    if mode == "tune":
        return _launch_tune(args)
    if mode == "study":
        return _launch_study(args)
    return _launch_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
