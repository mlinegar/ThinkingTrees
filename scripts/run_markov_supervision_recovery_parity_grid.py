#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_parity_grid_io import (  # noqa: E402
    load_parity_grid_root,
    write_materialized_outputs as _write_materialized_outputs,
)
from src.ctreepo.sim.core.tree_reference_presets import (  # noqa: E402
    ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET,
    ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET,
    ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET,
    STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
    UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
    resolve_tree_reference_preset_config,
)
from src.ctreepo.sim.core.tree_neural_facade import (  # noqa: E402
    JobSpec as _JobSpec,
    RunConfigSpec as _RunConfigSpec,
    discover_scheduler_devices as _discover_scheduler_devices,
    job_output_dir_name as _job_output_dir_name,
    run_config_from_mapping as _run_config_from_mapping,
    with_run_intent_overrides as _with_run_intent_overrides,
)
from src.ctreepo.sim.core.tree_neural_execution import (  # noqa: E402
    run_scheduler_bundle as _run_scheduler_bundle,
    scheduler_cli_payload as _scheduler_cli_payload,
    scheduler_item_for_job as _scheduler_item_for_job,
    write_combined_runs_output as _write_combined_runs_output,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    _ensure_prepared_markov_tree_data,
    _official_fno_locked_config_for_benchmark,
    resolve_full_doc_diagnostic_benchmark,
    resolve_full_doc_diagnostic_grid,
)
from src.ctreepo.sim.core.full_doc_config_codec import (  # noqa: E402
    runtime_config_overrides_from_config_like,
)
from src.ctreepo.sim.core.markov_comparison_surface import (  # noqa: E402
    FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    MarkovOPSDataBundle,
    OPSCountConfig,
)
from src.ctreepo.sim.suite.markov_observed_token_policy import (  # noqa: E402
    resolve_markov_observed_token_policy,
)


STUDY_NAME = "supervision_recovery_parity_grid"


def _doc_tokens_for_benchmark(benchmark_name: str) -> int:
    """Derive the document token count from the benchmark's observed-token policy."""
    spec = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    policy = resolve_markov_observed_token_policy(
        profile_name=str(spec.observed_token_profile),
    )
    if int(policy.min_tokens) == int(policy.max_tokens):
        return int(policy.min_tokens)
    return int(round(0.5 * float(int(policy.min_tokens) + int(policy.max_tokens))))


# Fallback constants used when benchmark is not yet resolved (e.g. manifest metadata).
# These match the ``recoverable`` profile defaults (min_tokens=max_tokens=128).
ASSUMED_DOC_TOKENS = 128
ONE_LEAF_TARGET_FIXED_LEAF_TOKENS = 128
TREE_BASELINE_FAMILY = "tree_neural"
PARITY_MANIFEST_NAME = "parity_grid_manifest.json"
PARITY_STATUS_NAME = "parity_grid_status.json"
PARITY_SUMMARY_NAME = "parity_grid_summary.json"
CANONICAL_TRAIN_LADDER = (1024, 4096, 10240)
CLAIM_LEVEL_EMPIRICAL_GEOMETRY = "empirical_geometry"
CLAIM_LEVEL_EXACT_COLLAPSE = "exact_collapse_candidate"
EXACT_COLLAPSE_RECIPE_ID = "exact_collapse_candidate"
EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID = "exact_collapse_legacy_control"
EXACT_COLLAPSE_RUNTIME_MATCH_RECIPE_ID = "exact_collapse_runtime_match"
EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE = "official_fno_one_tree_identity"
EXACT_COLLAPSE_RUNTIME_IDENTITY_MODE = "official_fno_runtime_identity"
EVIDENCE_STATUS_AUTHORITATIVE = "authoritative"
EVIDENCE_STATUS_EXPLORATORY = "exploratory"
EVIDENCE_STATUS_PARTIAL = "partial"
EVIDENCE_STATUS_STOPPED = "stopped"
FNO_BASELINE_FAMILIES = ("official_fno", "official_fno_sumlen")
FNO_RECIPE_ID = "fno_baseline"
FULL_LOCAL_LAWS_TREE_RECIPE_ID = "full_local_laws_tree"
FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS = "full_local_laws_topology_4096"
UNIFIED_G_TOPOLOGY_RECIPE_ID = "unified_g_full_local_laws_tree"
UNIFIED_G_TOPOLOGY_STUDY_AXIS = "unified_g_topology_4096"
TOPOLOGY_STUDY_AXES = (
    FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS,
    UNIFIED_G_TOPOLOGY_STUDY_AXIS,
)
UNIFIED_G_TOPOLOGY_DEFAULT_SEEDS = (0, 1, 2, 3, 4)
UNIFIED_G_TOPOLOGY_DEFAULT_LEAF_TOKENS = (128, 64, 32, 16)
UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_LEAF_TOKENS = (16,)
UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_SEEDS = (0, 1)
VALID_POSTTRAIN_DIAGNOSTICS_MODES = ("", "full", "minimal")


RECIPE_DISPLAY_NAMES: Dict[str, str] = {
    "historical_replay": "Historical replay",
    "optimization_fairness": "Optimization fairness",
    "capacity_fairness": "Capacity fairness",
    "matched_root": "Matched root",
    "fairfno_matched_root": "Fair-FNO matched root",
    EXACT_COLLAPSE_RECIPE_ID: "Exact-collapse candidate",
    EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID: "Legacy exact-collapse control",
    EXACT_COLLAPSE_RUNTIME_MATCH_RECIPE_ID: "Runtime-matched exact collapse",
    "fno_baseline": "FNO baseline",
    FULL_LOCAL_LAWS_TREE_RECIPE_ID: "Tree full local laws",
    UNIFIED_G_TOPOLOGY_RECIPE_ID: "Tree unified_g topology",
}

OFFICIAL_FNO_REFERENCE_FIELDS: tuple[str, ...] = (
    "state_dim",
    "hidden_dim",
    "n_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "fixed_leaf_tokens",
    "tree_model_version",
    "tree_batch_runtime_mode",
    "tree_root_supervision_kind",
    "tree_checkpoint_metric",
    "tree_stage1_checkpoint_metric",
    "tree_stage1_root_weight",
    "tree_training_schedule",
    "tree_stage1_epochs",
    "tree_stage2_epochs",
    "tree_task_head_mode",
    "tree_theorem_surface_mode",
    "tree_theorem_count_head_mode",
    "tree_theorem_count_ordinal_weight",
    "tree_theorem_count_scalar_aux_weight",
    "tree_theorem_count_threshold_balance",
    "tree_summary_spec_root_mode",
    "tree_theorem_feature_dim",
    "tree_theorem_feature_hidden_dim",
    "tree_theorem_score_dim",
    "tree_theorem_fiber_dim",
    "tree_theorem_aux_dim",
    "tree_theorem_count_dim",
    "tree_theorem_first_dim",
    "tree_theorem_last_dim",
    "tree_leaf_fno_width",
    "tree_leaf_fno_n_modes",
    "tree_leaf_fno_n_layers",
    "tree_batch_pack_mode",
    "leaf_supervision_kind",
    "leaf_label_rate",
    "internal_supervision_kind",
    "internal_label_rate",
    "leaf_exact_supervision",
    "local_law_weight",
    "task_objective_weight",
    "c1_relative_weight",
    "c2_relative_weight",
    "c3_relative_weight",
    "tree_local_weighting_mode",
    "doc_sequence_train_fraction",
)

for _budget_label in ("10", "20"):
    for _rate_label in ("0", "10", "20", "50", "100"):
        RECIPE_DISPLAY_NAMES[f"r{_budget_label}_local_{_rate_label}"] = (
            f"R{_budget_label} local {_rate_label}%"
        )


def _local_supervision_sweep_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for root_budget_per_doc, budget_tag in ((0.1, "10"), (0.2, "20")):
        for local_rate, rate_tag in (
            (0.0, "0"),
            (0.1, "10"),
            (0.2, "20"),
            (0.5, "50"),
            (1.0, "100"),
        ):
            specs.append(
                {
                    "recipe_id": f"r{budget_tag}_local_{rate_tag}",
                    "budget_total_calls_per_doc": float(root_budget_per_doc),
                    "full_doc_budget_share": 1.0,
                    "doc_consumption_mode": "root_only",
                    "local_split_mode": "balanced",
                    "leaf_label_rate": float(local_rate),
                    # For DSL/IPW-aligned local supervision, use count-only
                    # labels at both leaves and internals.
                    "leaf_supervision_kind": "count_only",
                    "leaf_exact_supervision": False,
                    "internal_label_rate": float(local_rate),
                    "internal_supervision_kind": (
                        "count_only" if float(local_rate) > 0.0 else "none"
                    ),
                }
            )
    return specs


def _lean_faithful_local_diagnostic_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for recipe_id, root_budget_per_doc, local_rate in (
        ("r10_local_20", 0.1, 0.2),
        ("r20_local_50", 0.2, 0.5),
    ):
        for target_kind in ("count_only", "bounded_full_sketch"):
            for weighting_mode in ("subset_mean", "fixed_k_hajek"):
                specs.append(
                    {
                        "recipe_id": str(recipe_id),
                        "target_kind": str(target_kind),
                        "weighting_mode": str(weighting_mode),
                        # The Lean-faithful diagnostic matrix compares local
                        # target choice and estimator shape under the fixed-k
                        # deterministic subset design. Do not materialize the
                        # older budget manifest here, since explicit budgeted
                        # audit maps would override the subset sampler we are
                        # trying to study.
                        "budget_total_calls_per_doc": 0.0,
                        "full_doc_budget_share": 1.0,
                        "doc_consumption_mode": "",
                        "local_split_mode": "",
                        "recipe_budget_total_calls_per_doc": float(
                            root_budget_per_doc
                        ),
                        "leaf_label_rate": float(local_rate),
                        "internal_label_rate": float(local_rate),
                        "leaf_supervision_kind": str(target_kind),
                        "internal_supervision_kind": (
                            "none"
                            if float(local_rate) <= 0.0
                            else str(target_kind)
                        ),
                        "leaf_exact_supervision": False,
                        "tree_local_law_weight": 0.8,
                    }
                )
    return specs


def _lean_faithful_weight_balance_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for local_law_weight in (0.10, 0.25, 0.50):
        for c1_relative_weight in (1.0, 2.0):
            specs.append(
                {
                    "recipe_id": "r20_local_50",
                    "target_kind": "bounded_full_sketch",
                    "weighting_mode": "fixed_k_hajek",
                    "budget_total_calls_per_doc": 0.0,
                    "full_doc_budget_share": 1.0,
                    "doc_consumption_mode": "",
                    "local_split_mode": "",
                    "recipe_budget_total_calls_per_doc": 0.2,
                    "leaf_label_rate": 0.5,
                    "internal_label_rate": 0.5,
                    "leaf_supervision_kind": "bounded_full_sketch",
                    "internal_supervision_kind": "bounded_full_sketch",
                    "leaf_exact_supervision": False,
                    "tree_local_law_weight": float(local_law_weight),
                    "tree_c1_relative_weight": float(c1_relative_weight),
                    "tree_c2_relative_weight": 1.0,
                    "tree_c3_relative_weight": 1.0,
                }
            )
    return specs


@dataclass(frozen=True)
class ParityGridEntry:
    recipe_id: str
    benchmark: str
    scope_key: str
    scope_label: str
    claim_level: str
    fixed_leaf_tokens: int
    seed: int
    config: _RunConfigSpec
    job: _JobSpec
    official_fno_reference_surface: Dict[str, Any] = field(default_factory=dict)
    nominal_recipe_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        job_family = str(self.job.family or "").strip()
        config_family = str(getattr(self.config, "baseline_family", "") or "").strip()
        if config_family and config_family != job_family:
            raise ValueError(
                f"ParityGridEntry config/job family mismatch: "
                f"config.baseline_family={config_family!r} job.family={job_family!r}"
            )
        if not config_family:
            object.__setattr__(
                self,
                "config",
                replace(self.config, baseline_family=job_family),
            )

    @property
    def job_output_dir_name(self) -> str:
        return _job_output_dir_name(self.job.job_name)

    def manifest_row(
        self, *, output_root: Path, main_train_doc_count: int, epoch_cap: int,
    ) -> Dict[str, Any]:
        doc_tokens = _doc_tokens_for_benchmark(str(self.benchmark))
        row = {
            "job_name": str(self.job.job_name),
            "recipe_id": str(self.recipe_id),
            "recipe_display_name": str(
                RECIPE_DISPLAY_NAMES.get(self.recipe_id, self.recipe_id)
            ),
            "benchmark": str(self.benchmark),
            "scope_key": str(self.scope_key),
            "scope_label": str(self.scope_label),
            "claim_level": str(self.claim_level),
            "train_doc_count": int(self.job.train_doc_count),
            "is_main_comparison": bool(
                int(self.job.train_doc_count) == int(main_train_doc_count)
            ),
            "baseline_family": str(self.job.family),
            "seed": int(self.seed),
            "epoch_cap": int(epoch_cap),
            "fixed_leaf_tokens": int(self.fixed_leaf_tokens),
            "slot_count": int(self.config.slot_count),
            "tuning_stage": str(self.job.tuning_stage or ""),
            "study_axis": str(self.job.study_axis or ""),
            "axis_value": str(self.job.axis_value or ""),
            "one_leaf_target": bool(
                int(self.fixed_leaf_tokens) >= int(doc_tokens)
            ),
            "assumed_doc_tokens": int(doc_tokens),
            "job_output_dir": str(
                output_root / "jobs" / self.job_output_dir_name
            ),
            "config": asdict(self.config),
            "job": asdict(self.job),
            "official_fno_reference_surface": dict(
                self.official_fno_reference_surface or {}
            ),
            "nominal_recipe_metadata": dict(self.nominal_recipe_metadata or {}),
        }
        row.update(
            {
                str(key): value
                for key, value in dict(self.nominal_recipe_metadata or {}).items()
                if str(key).startswith("nominal_recipe_")
            }
        )
        return row


def _nominal_recipe_metadata(
    *,
    recipe_id: str,
    recipe_budget_total_calls_per_doc: float | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "nominal_recipe_id": str(recipe_id),
    }
    if recipe_budget_total_calls_per_doc is not None:
        metadata["nominal_recipe_budget_total_calls_per_doc"] = float(
            recipe_budget_total_calls_per_doc
        )
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch or plan the overnight supervision-recovery parity grid that "
            "adds geometry/parity evidence to the family-grids report lineage."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs")
        / f"markov_supervision_recovery_parity_grid_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--benchmark", default="recoverable_v4")
    parser.add_argument(
        "--structural-benchmark",
        default="structural_core_v1::r12_seg10to12",
    )
    parser.add_argument(
        "--train-doc-counts",
        type=str,
        default="1024 2048 4096 10240 20480",
        help=(
            "Space-separated list of training doc counts forming the learning "
            "curve. The max is the main comparison point; smaller counts show "
            "sample-efficiency. All share the same test set via prefix nesting."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mig-uuids", default="")
    parser.add_argument(
        "--recipe-ids",
        type=str,
        default="",
        help=(
            "Optional space-separated recipe ids to include. "
            "Applies across tree entries, exact-collapse rows, supervision-sweep "
            "rows, and the shared fno_baseline recipe."
        ),
    )
    parser.add_argument(
        "--fixed-leaf-tokens",
        type=str,
        default="",
        help=(
            "Optional space-separated fixed_leaf_tokens values to include. "
            "Useful for smoke subsets such as '16 128'."
        ),
    )
    parser.add_argument(
        "--include-structural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the structural confirmation scope.",
    )
    parser.add_argument("--prepared-data-root", default="")
    parser.add_argument(
        "--corpus-root",
        type=str,
        default="",
        help=(
            "Path to a prepared corpus from prepare_markov_parity_corpus.py. "
            "When set, all jobs load train/val/test from this corpus with "
            "prefix-nested training sets and a shared test set."
        ),
    )
    parser.add_argument(
        "--prepared-data-allow-create",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tree-exact-eval-max-docs", type=int, default=64)
    parser.add_argument("--gpu-runtime-data-mode", default="resident")
    parser.add_argument("--gpu-runtime-bucket-mode", default="leaf_count_auto_queue")
    parser.add_argument(
        "--gpu-runtime-preload-splits",
        nargs="*",
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--gpu-runtime-preload-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Documents per GPU forward pass. Default 512 (8x historical 64).",
    )
    parser.add_argument(
        "--epoch-cap",
        type=int,
        default=0,
        help=(
            "Optional cap on total training epochs for quick runs. "
            "0 keeps the preset epoch counts."
        ),
    )
    parser.add_argument(
        "--exact-metric-selection-doc-limit",
        type=int,
        default=0,
        help=(
            "Optional doc cap for interim exact-metric checkpoint selection. "
            "0 uses the full validation selection set."
        ),
    )
    parser.add_argument(
        "--exact-metric-selection-interval",
        type=int,
        default=1,
        help=(
            "Run interim exact-metric checkpoint selection every N epochs. "
            "The final epoch still runs exact selection."
        ),
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cleanup-stale-children",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-gpu-items-per-mig", type=int, default=1)
    parser.add_argument("--scheduler-launch-stagger-seconds", type=float, default=0.0)
    parser.add_argument(
        "--scheduler-min-mem-available-gib",
        type=float,
        default=128.0,
        help=(
            "Minimum host MemAvailable in GiB required by the nested GPU scheduler. "
            "Set to 0 to disable the floor for a run."
        ),
    )
    parser.add_argument(
        "--scheduler-min-swap-free-gib",
        type=float,
        default=2.0,
        help=(
            "Minimum host SwapFree in GiB required by the nested GPU scheduler. "
            "Set to 0 to disable the floor for a run."
        ),
    )
    parser.add_argument(
        "--skip-fno-baselines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip FNO baseline runs (tree-side only).",
    )
    parser.add_argument(
        "--include-supervision-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include R10/R20 local-supervision-rate sweep entries. "
            "These add matched_root jobs at leaf=16 with varying "
            "leaf_label_rate and internal_label_rate."
        ),
    )
    parser.add_argument(
        "--lean-faithful-diagnostic-matrix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the 12-job recoverable-only diagnostic matrix that crosses "
            "count_only vs bounded_full_sketch with subset_mean vs fixed_k_hajek."
        ),
    )
    parser.add_argument(
        "--lean-faithful-weight-balance-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Augment the Lean-faithful diagnostic matrix with a 6-job "
            "r20_local_50 bounded_full_sketch fixed_k_hajek sweep over "
            "local-law weight and C1:C3 balance."
        ),
    )
    parser.add_argument(
        "--exact-collapse-repair-diagnostic-matrix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the 4-job recoverable-only one-leaf exact-collapse repair "
            "diagnostic: official_fno, legacy exact control, config-matched "
            "exact collapse, and runtime-matched exact collapse."
        ),
    )
    parser.add_argument(
        "--full-local-laws-topology-diagnostic-4096",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the 8-job recoverable-only 4096-doc topology diagnostic "
            "covering official_fno and the canonical factorized full-local-laws "
            "tree surface at fixed_leaf_tokens={64,128} across seeds {0,1}."
        ),
    )
    parser.add_argument(
        "--unified-g-topology-diagnostic-4096",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the recoverable-only 4096-doc unified_g topology diagnostic "
            "anchored at leaf128 and expanded across the requested leaf-token ladder."
        ),
    )
    parser.add_argument(
        "--topology-seeds",
        nargs="*",
        type=int,
        default=UNIFIED_G_TOPOLOGY_DEFAULT_SEEDS,
        help=(
            "Primary seed ladder for --unified-g-topology-diagnostic-4096. "
            "Applies to non-stress leaf-token values."
        ),
    )
    parser.add_argument(
        "--topology-leaf-tokens",
        nargs="*",
        type=int,
        default=UNIFIED_G_TOPOLOGY_DEFAULT_LEAF_TOKENS,
        help=(
            "Leaf-token ladder for --unified-g-topology-diagnostic-4096. "
            "Default 128 64 32 16."
        ),
    )
    parser.add_argument(
        "--topology-stress-leaf-tokens",
        nargs="*",
        type=int,
        default=UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_LEAF_TOKENS,
        help=(
            "Optional stress-only leaf-token values for the unified_g topology study. "
            "These use --topology-stress-seeds instead of the primary seed ladder."
        ),
    )
    parser.add_argument(
        "--topology-stress-seeds",
        nargs="*",
        type=int,
        default=UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_SEEDS,
        help=(
            "Seed ladder for stress-only unified_g topology leaf-token values. "
            "Default 0 1."
        ),
    )
    parser.add_argument(
        "--topology-posttrain-train-doc-limit",
        type=int,
        default=0,
        help=(
            "Optional train-doc cap for post-train diagnostics in the unified_g "
            "topology study. 0 keeps the full training split."
        ),
    )
    parser.add_argument(
        "--topology-posttrain-diagnostics-mode",
        type=str,
        default="",
        choices=VALID_POSTTRAIN_DIAGNOSTICS_MODES,
        help=(
            "Optional post-train diagnostics mode override for unified_g topology "
            "tree runs. Empty keeps the default study behavior."
        ),
    )
    parser.add_argument(
        "--topology-stress-posttrain-diagnostics-mode",
        type=str,
        default="",
        choices=VALID_POSTTRAIN_DIAGNOSTICS_MODES,
        help=(
            "Optional post-train diagnostics mode override for stress-only unified_g "
            "topology leaf-token values."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _parse_train_doc_counts(raw: str) -> List[int]:
    """Parse a space-separated list of train doc counts, sorted ascending."""
    values = sorted({int(v) for v in str(raw).split() if v.strip()})
    if not values:
        raise ValueError("--train-doc-counts must contain at least one value")
    return values


def _parse_optional_token_filter(raw: str) -> set[str] | None:
    values = {str(v).strip() for v in str(raw or "").split() if str(v).strip()}
    return values or None


def _parse_optional_int_filter(raw: str) -> set[int] | None:
    values = {int(v) for v in str(raw or "").split() if str(v).strip()}
    return values or None


def _ordered_unique_ints(values: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: List[int] = []
    for raw in list(values or ()):
        value = int(raw)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _runtime_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    preload_splits = tuple(
        str(value)
        for value in list(getattr(args, "gpu_runtime_preload_splits", ("train", "val", "test")) or ())
        if str(value).strip()
    )
    return {
        "gpu_runtime_data_mode": str(args.gpu_runtime_data_mode),
        "gpu_runtime_bucket_mode": str(args.gpu_runtime_bucket_mode),
        "gpu_runtime_preload_splits": preload_splits or ("train", "val", "test"),
        "gpu_runtime_preload_targets": bool(
            getattr(args, "gpu_runtime_preload_targets", True)
        ),
        "gpu_runtime_workers_per_mig": 1,
        "gpu_runtime_allow_multi_worker_screen": False,
        "gpu_runtime_capacity_workers_per_mig": 1,
    }


def _topology_diagnostic_config_overrides(
    args: argparse.Namespace,
    *,
    fixed_leaf_tokens: int,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    train_doc_limit = int(getattr(args, "topology_posttrain_train_doc_limit", 0) or 0)
    if train_doc_limit > 0:
        overrides["tree_posttrain_train_doc_limit"] = int(train_doc_limit)
    stress_leaf_tokens = {
        int(value)
        for value in list(getattr(args, "topology_stress_leaf_tokens", ()) or ())
    }
    resolved_mode = str(
        getattr(args, "topology_posttrain_diagnostics_mode", "") or ""
    ).strip().lower()
    if int(fixed_leaf_tokens) in stress_leaf_tokens:
        stress_mode = str(
            getattr(args, "topology_stress_posttrain_diagnostics_mode", "") or ""
        ).strip().lower()
        if stress_mode:
            resolved_mode = str(stress_mode)
    if resolved_mode:
        overrides["posttrain_diagnostics_mode"] = str(resolved_mode)
    return overrides


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _apply_epoch_cap_to_mapping(
    mapping: Mapping[str, Any], *, epoch_cap: int,
) -> Dict[str, Any]:
    capped = dict(mapping)
    cap = int(epoch_cap)
    if cap <= 0:
        return capped
    if capped.get("n_epochs") is not None:
        capped["n_epochs"] = min(int(capped.get("n_epochs", cap)), cap)
    schedule = str(capped.get("tree_training_schedule", "") or "").strip().lower()
    if schedule != "two_stage":
        if "tree_stage1_epochs" in capped:
            capped["tree_stage1_epochs"] = 0
        if "tree_stage2_epochs" in capped:
            capped["tree_stage2_epochs"] = 0
        return capped
    stage1_epochs = max(0, int(capped.get("tree_stage1_epochs", 0) or 0))
    stage2_epochs = max(0, int(capped.get("tree_stage2_epochs", 0) or 0))
    total_stage_epochs = stage1_epochs + stage2_epochs
    if total_stage_epochs <= 0:
        return capped
    capped_total = min(total_stage_epochs, cap)
    if capped_total <= 0:
        capped["tree_stage1_epochs"] = 0
        capped["tree_stage2_epochs"] = 0
        return capped
    if stage1_epochs <= 0:
        new_stage1_epochs = 0
        new_stage2_epochs = capped_total
    elif stage2_epochs <= 0:
        new_stage1_epochs = capped_total
        new_stage2_epochs = 0
    else:
        new_stage1_epochs = int(round(capped_total * stage1_epochs / total_stage_epochs))
        if capped_total > 1:
            new_stage1_epochs = max(1, min(capped_total - 1, new_stage1_epochs))
        else:
            new_stage1_epochs = 1
        new_stage2_epochs = capped_total - new_stage1_epochs
    capped["tree_stage1_epochs"] = int(new_stage1_epochs)
    capped["tree_stage2_epochs"] = int(new_stage2_epochs)
    return capped


def _effective_epoch_cap(args: argparse.Namespace) -> int:
    raw_epoch_cap = int(getattr(args, "epoch_cap", 0) or 0)
    if bool(getattr(args, "full_local_laws_topology_diagnostic_4096", False)):
        return int(raw_epoch_cap) if int(raw_epoch_cap) > 0 else 20
    return int(raw_epoch_cap)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scope_label(benchmark: str) -> str:
    return "structural" if str(benchmark).startswith("structural_core_v1::") else "recoverable"


def _canonical_train_ladder_payload() -> List[int]:
    return [int(value) for value in CANONICAL_TRAIN_LADDER]


def _resolve_benchmark_spec(benchmark: str) -> Any:
    text = str(benchmark or "").strip()
    if text.startswith("structural_core_v1::"):
        _, _, cell_id = text.partition("::")
        for candidate in resolve_full_doc_diagnostic_grid("structural_core_v1"):
            if str(candidate.cell_id or "") == str(cell_id):
                return candidate
        raise ValueError(f"unknown structural benchmark cell: {benchmark!r}")
    return resolve_full_doc_diagnostic_benchmark(text)


def _config_mapping_for_run_config(config: _RunConfigSpec) -> Dict[str, Any]:
    return runtime_config_overrides_from_config_like(config)


def _ops_config_from_mapping(mapping: Mapping[str, Any]) -> OPSCountConfig:
    known_fields = set(OPSCountConfig.__dataclass_fields__.keys())
    return OPSCountConfig(
        **{
            key: value
            for key, value in dict(mapping).items()
            if key in known_fields
        }
    )


def _legacy_exact_collapse_reference_surface(
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
    epoch_cap: int = 0,
) -> Dict[str, Any]:
    preset_name = _recipe_preset("matched_root", benchmark)
    preset = resolve_tree_reference_preset_config(preset_name)
    state_dim = int(preset.get("state_dim", 256))
    return _apply_epoch_cap_to_mapping(
        {
        "state_dim": state_dim,
        "hidden_dim": int(preset.get("hidden_dim", 1024)),
        "n_epochs": int(preset.get("n_epochs", 128)),
        "lr": float(preset.get("lr", 5e-4)),
        "weight_decay": float(preset.get("weight_decay", 0.0)),
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "tree_root_supervision_kind": "count_ce",
        "tree_training_schedule": "single_stage",
        "tree_checkpoint_metric": "val_root_mae",
        "tree_stage1_checkpoint_metric": "val_root_mae",
        "tree_stage1_root_weight": 1.0,
        "tree_stage1_epochs": 0,
        "tree_stage2_epochs": 0,
        "local_law_weight": 0.0,
        "c1_relative_weight": 0.0,
        "c2_relative_weight": 0.0,
        "c3_relative_weight": 0.0,
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "doc_sequence_train_fraction": 0.0,
        **_fairfno_leaf_defaults(
            state_dim=state_dim,
            fixed_leaf_tokens=int(fixed_leaf_tokens),
        ),
        },
        epoch_cap=int(epoch_cap),
    )


def _official_fno_reference_run_config(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
    preserve_requested_leaf_tokens: bool = False,
) -> _RunConfigSpec:
    canonical_fixed_leaf_tokens = int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
    preset_name = ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
    base = resolve_tree_reference_preset_config(preset_name)
    overrides: Dict[str, Any] = {
        "label": (
            f"{FNO_RECIPE_ID}__{_scope_label(benchmark)}"
            f"__leaf{int(canonical_fixed_leaf_tokens)}"
        ),
        "benchmark": str(benchmark),
        "comparison_mode": "comparable",
        "fixed_leaf_tokens": int(canonical_fixed_leaf_tokens),
        "batch_size": int(args.batch_size),
        "slot_count": 4,
        "prepared_data_root": str(getattr(args, "prepared_data_root", "") or ""),
        "prepared_data_allow_create": bool(args.prepared_data_allow_create),
        "tree_exact_eval_max_docs": int(args.tree_exact_eval_max_docs),
        "exact_metric_selection_doc_limit": int(
            getattr(args, "exact_metric_selection_doc_limit", 0) or 0
        ),
        "exact_metric_selection_interval": int(
            getattr(args, "exact_metric_selection_interval", 1) or 1
        ),
        **_runtime_config_overrides(args),
    }
    base_mapping = _apply_epoch_cap_to_mapping(
        {**base, **overrides},
        epoch_cap=_effective_epoch_cap(args),
    )
    benchmark_spec = _resolve_benchmark_spec(str(benchmark))
    locked_ops_config = _official_fno_locked_config_for_benchmark(
        benchmark=benchmark_spec,
        config=_ops_config_from_mapping(base_mapping),
    )
    mapping = {
        **base_mapping,
        **asdict(locked_ops_config),
        "baseline_family": "official_fno",
        "label": str(base_mapping.get("label", "") or ""),
        "fixed_leaf_tokens": int(canonical_fixed_leaf_tokens),
        "preserve_requested_leaf_tokens": bool(preserve_requested_leaf_tokens),
        "official_fno_preserve_requested_leaf_tokens": bool(
            preserve_requested_leaf_tokens
        ),
        "tree_c1_relative_weight": float(locked_ops_config.c1_relative_weight),
        "tree_c2_relative_weight": float(locked_ops_config.c2_relative_weight),
        "tree_c3_relative_weight": float(locked_ops_config.c3_relative_weight),
    }
    if locked_ops_config.local_law_weight is not None:
        mapping["tree_local_law_weight"] = float(
            max(0.0, float(locked_ops_config.local_law_weight))
        )
        mapping.pop("tree_task_objective_weight", None)
    else:
        mapping["tree_task_objective_weight"] = float(
            max(0.0, float(locked_ops_config.task_objective_weight or 1.0))
        )
    return _run_config_from_mapping(mapping)


def _full_local_laws_topology_tree_run_config(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
) -> _RunConfigSpec:
    return _topology_tree_run_config(
        args,
        benchmark=benchmark,
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        preset_name=SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
        recipe_id=FULL_LOCAL_LAWS_TREE_RECIPE_ID,
    )


def _topology_tree_run_config(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
    preset_name: str,
    recipe_id: str,
    extra_overrides: Mapping[str, Any] | None = None,
) -> _RunConfigSpec:
    base = resolve_tree_reference_preset_config(preset_name)
    overrides: Dict[str, Any] = {
        "baseline_family": TREE_BASELINE_FAMILY,
        "label": (
            f"{str(recipe_id)}__{_scope_label(benchmark)}"
            f"__leaf{int(fixed_leaf_tokens)}"
        ),
        "benchmark": str(benchmark),
        "comparison_mode": "comparable",
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "preserve_requested_leaf_tokens": True,
        "prepared_data_root": str(getattr(args, "prepared_data_root", "") or ""),
        "prepared_data_allow_create": bool(args.prepared_data_allow_create),
        "tree_exact_eval_max_docs": int(args.tree_exact_eval_max_docs),
        "exact_metric_selection_doc_limit": int(
            getattr(args, "exact_metric_selection_doc_limit", 0) or 0
        ),
        "exact_metric_selection_interval": int(
            getattr(args, "exact_metric_selection_interval", 1) or 1
        ),
        **_runtime_config_overrides(args),
        "slot_count": int(base.get("slot_count", 4) or 4),
        "tree_c1_relative_weight": float(base.get("c1_relative_weight", 1.0) or 1.0),
        "tree_c2_relative_weight": float(base.get("c2_relative_weight", 1.0) or 1.0),
        "tree_c3_relative_weight": float(base.get("c3_relative_weight", 1.0) or 1.0),
    }
    if base.get("local_law_weight") is not None:
        overrides["tree_local_law_weight"] = float(base.get("local_law_weight"))
        overrides.pop("tree_task_objective_weight", None)
    else:
        overrides["tree_task_objective_weight"] = float(
            base.get("task_objective_weight", 1.0) or 1.0
        )
    mapping = _apply_epoch_cap_to_mapping(
        {**base, **overrides, **dict(extra_overrides or {})},
        epoch_cap=_effective_epoch_cap(args),
    )
    return _run_config_from_mapping(mapping)


def _unified_g_topology_tree_run_config(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
) -> _RunConfigSpec:
    return _topology_tree_run_config(
        args,
        benchmark=benchmark,
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        preset_name=UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
        recipe_id=UNIFIED_G_TOPOLOGY_RECIPE_ID,
        extra_overrides=_topology_diagnostic_config_overrides(
            args,
            fixed_leaf_tokens=int(fixed_leaf_tokens),
        ),
    )


def _official_fno_reference_surface(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
) -> Dict[str, Any]:
    mapping = _config_mapping_for_run_config(
        _official_fno_reference_run_config(
            args,
            benchmark=str(benchmark),
            fixed_leaf_tokens=int(fixed_leaf_tokens),
        )
    )
    return {
        str(field_name): mapping.get(field_name)
        for field_name in OFFICIAL_FNO_REFERENCE_FIELDS
        if field_name in mapping
    }


def _config_diff_vs_official_fno(
    *,
    config_mapping: Mapping[str, Any],
    reference_surface: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    reference = dict(reference_surface or {})
    reference_lambda = reference.get("local_law_weight", float("nan"))
    actual_lambda = dict(config_mapping).get("local_law_weight", float("nan"))
    if (
        float(reference_lambda or 0.0) == 0.0
        or float(actual_lambda or 0.0) == 0.0
    ):
        reference.pop("task_objective_weight", None)
    diff: Dict[str, Any] = {}
    for field_name, expected in reference.items():
        actual = config_mapping.get(field_name)
        if expected in {None, ""}:
            if actual not in {None, ""}:
                diff[field_name] = {"expected": expected, "actual": actual}
            continue
        if isinstance(expected, float):
            if actual in {None, ""}:
                actual_value = float("nan")
            else:
                actual_value = float(actual)
            if not math.isfinite(actual_value) or abs(actual_value - float(expected)) > 1e-12:
                diff[field_name] = {"expected": expected, "actual": actual}
            continue
        if actual != expected:
            diff[field_name] = {"expected": expected, "actual": actual}
    return diff



def _recipe_preset(recipe_id: str, benchmark: str) -> str:
    if str(benchmark).startswith("structural_core_v1::"):
        return STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
    mapping = {
        "historical_replay": ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET,
        "optimization_fairness": ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET,
        "capacity_fairness": ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET,
        "matched_root": ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
        "fairfno_matched_root": ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    }
    preset_name = mapping.get(str(recipe_id))
    if not preset_name:
        raise ValueError(f"unsupported recipe_id: {recipe_id!r}")
    return preset_name


def _fairfno_leaf_defaults(*, state_dim: int, fixed_leaf_tokens: int) -> Dict[str, int]:
    return {
        "tree_leaf_fno_width": max(64, int(state_dim)),
        "tree_leaf_fno_n_modes": min(16, max(1, int(fixed_leaf_tokens) // 2)),
        "tree_leaf_fno_n_layers": 4,
    }


def _exact_collapse_config_for_entry(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
    runtime_identity_mode: str = "",
) -> _RunConfigSpec:
    reference_config = _official_fno_reference_run_config(
        args,
        benchmark=str(benchmark),
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        preserve_requested_leaf_tokens=True,
    )
    base = asdict(reference_config)
    base.update(
        {
            "baseline_family": TREE_BASELINE_FAMILY,
            "label": (
                f"{EXACT_COLLAPSE_RECIPE_ID}__{_scope_label(benchmark)}__leaf"
                f"{int(fixed_leaf_tokens)}"
            ),
            "benchmark": str(benchmark),
            "comparison_mode": "exact_collapse",
            "prepared_data_root": str(getattr(args, "prepared_data_root", "") or ""),
            "prepared_data_allow_create": bool(args.prepared_data_allow_create),
            "preserve_requested_leaf_tokens": True,
            "tree_exact_eval_max_docs": int(args.tree_exact_eval_max_docs),
            **_runtime_config_overrides(args),
            "tree_local_law_weight": 0.0,
            "tree_c1_relative_weight": 0.0,
            "tree_c2_relative_weight": 0.0,
            "tree_c3_relative_weight": 0.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.0,
            "leaf_exact_supervision": False,
            "internal_supervision_kind": "none",
            "internal_label_rate": 0.0,
            "tree_exact_collapse_mode": str(runtime_identity_mode or ""),
        }
    )
    base.pop("task_objective_weight", None)
    base.pop("tree_task_objective_weight", None)
    return _run_config_from_mapping(base)


def _legacy_exact_collapse_control_config_for_entry(
    args: argparse.Namespace,
    *,
    benchmark: str,
    fixed_leaf_tokens: int,
) -> _RunConfigSpec:
    reference_surface = _legacy_exact_collapse_reference_surface(
        benchmark=str(benchmark),
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        epoch_cap=int(getattr(args, "epoch_cap", 0) or 0),
    )
    base = resolve_tree_reference_preset_config(ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET)
    base.update(reference_surface)
    base.pop("task_objective_weight", None)
    base.pop("tree_task_objective_weight", None)
    base.update(
        {
            "baseline_family": TREE_BASELINE_FAMILY,
            "label": (
                f"{EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID}__{_scope_label(benchmark)}"
                f"__leaf{int(fixed_leaf_tokens)}"
            ),
            "benchmark": str(benchmark),
            "prepared_data_root": str(getattr(args, "prepared_data_root", "") or ""),
            "prepared_data_allow_create": bool(args.prepared_data_allow_create),
            "tree_exact_eval_max_docs": int(args.tree_exact_eval_max_docs),
            **_runtime_config_overrides(args),
            "batch_size": int(args.batch_size),
            "exact_metric_selection_doc_limit": int(
                getattr(args, "exact_metric_selection_doc_limit", 0) or 0
            ),
            "exact_metric_selection_interval": int(
                getattr(args, "exact_metric_selection_interval", 1) or 1
            ),
            "slot_count": 4,
            "tree_local_law_weight": float(reference_surface["local_law_weight"]),
            "tree_c1_relative_weight": float(reference_surface["c1_relative_weight"]),
            "tree_c2_relative_weight": float(reference_surface["c2_relative_weight"]),
            "tree_c3_relative_weight": float(reference_surface["c3_relative_weight"]),
        }
    )
    return _run_config_from_mapping(base)


def _config_for_entry(
    args: argparse.Namespace,
    *,
    recipe_id: str,
    benchmark: str,
    fixed_leaf_tokens: int,
) -> _RunConfigSpec:
    if str(recipe_id) == EXACT_COLLAPSE_RECIPE_ID:
        return _exact_collapse_config_for_entry(
            args,
            benchmark=str(benchmark),
            fixed_leaf_tokens=int(fixed_leaf_tokens),
            runtime_identity_mode=EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE,
        )
    if str(recipe_id) == EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID:
        return _legacy_exact_collapse_control_config_for_entry(
            args,
            benchmark=str(benchmark),
            fixed_leaf_tokens=int(fixed_leaf_tokens),
        )
    if str(recipe_id) == EXACT_COLLAPSE_RUNTIME_MATCH_RECIPE_ID:
        return _exact_collapse_config_for_entry(
            args,
            benchmark=str(benchmark),
            fixed_leaf_tokens=int(fixed_leaf_tokens),
            runtime_identity_mode=EXACT_COLLAPSE_RUNTIME_IDENTITY_MODE,
        )
    if str(recipe_id) == FNO_RECIPE_ID:
        return _official_fno_reference_run_config(
            args,
            benchmark=str(benchmark),
            fixed_leaf_tokens=int(fixed_leaf_tokens),
            preserve_requested_leaf_tokens=bool(
                bool(getattr(args, "exact_collapse_repair_diagnostic_matrix", False))
                and int(fixed_leaf_tokens)
                >= int(_doc_tokens_for_benchmark(str(benchmark)))
            ),
        )
    preset_name = _recipe_preset(recipe_id, benchmark)
    base = resolve_tree_reference_preset_config(preset_name)
    overrides: Dict[str, Any] = {
        "baseline_family": TREE_BASELINE_FAMILY,
        "label": f"{recipe_id}__{_scope_label(benchmark)}__leaf{int(fixed_leaf_tokens)}",
        "benchmark": str(benchmark),
        "comparison_mode": "comparable",
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "batch_size": int(args.batch_size),
        "slot_count": 4,
        "prepared_data_root": str(getattr(args, "prepared_data_root", "") or ""),
        "prepared_data_allow_create": bool(args.prepared_data_allow_create),
        "tree_exact_eval_max_docs": int(args.tree_exact_eval_max_docs),
        "exact_metric_selection_doc_limit": int(
            getattr(args, "exact_metric_selection_doc_limit", 0) or 0
        ),
        "exact_metric_selection_interval": int(
            getattr(args, "exact_metric_selection_interval", 1) or 1
        ),
        **_runtime_config_overrides(args),
    }
    if str(recipe_id) == "fairfno_matched_root":
        state_dim = int(base.get("state_dim", 256))
        overrides.update(
            _fairfno_leaf_defaults(
                state_dim=state_dim,
                fixed_leaf_tokens=int(fixed_leaf_tokens),
            )
        )
    mapping = _apply_epoch_cap_to_mapping(
        {**base, **overrides},
        epoch_cap=int(getattr(args, "epoch_cap", 0) or 0),
    )
    return _run_config_from_mapping(mapping)

def _corpus_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    """Load the corpus manifest, returning {} if not available."""
    corpus_root = str(getattr(args, "corpus_root", "") or "").strip()
    if not corpus_root:
        return {}
    manifest_path = Path(corpus_root) / "corpus_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _corpus_payload_for_benchmark(
    manifest: Mapping[str, Any],
    *,
    benchmark: str,
) -> Dict[str, Any]:
    benchmarks = dict(manifest.get("benchmarks") or {})
    if benchmarks:
        payload = dict(benchmarks.get(str(benchmark)) or {})
        if payload:
            return payload
        available = sorted(str(name) for name in benchmarks)
        raise FileNotFoundError(
            f"corpus root is missing prepared benchmark {benchmark!r}; "
            f"available benchmarks: {available}"
        )
    manifest_benchmark = str(manifest.get("benchmark", "") or "").strip()
    if manifest_benchmark and str(benchmark).strip() != manifest_benchmark:
        raise FileNotFoundError(
            f"corpus root only contains benchmark {manifest_benchmark!r}, "
            f"but parity grid also requires {benchmark!r}"
        )
    return dict(manifest)


def _corpus_bundle_paths(manifest: Mapping[str, Any]) -> Dict[int, str]:
    """Map each train_doc_count to its corpus bundle path."""
    raw_paths = dict(manifest.get("bundle_paths", {}))
    return {int(k): str(v) for k, v in raw_paths.items() if str(v).strip()}


def _corpus_prepared_data_root(manifest: Mapping[str, Any]) -> str:
    """Return the prepared_data_root from the corpus manifest."""
    return str(manifest.get("prepared_data_root", "") or "")


def _prepared_data_allow_create(
    args: argparse.Namespace,
    *,
    prepared_data_root: str,
) -> bool:
    explicit_allow = bool(getattr(args, "prepared_data_allow_create", True))
    return bool(explicit_allow or not str(prepared_data_root or "").strip())


def _validate_prepared_tree_data_coverage(
    *,
    entries: Sequence[ParityGridEntry],
    manifest: Mapping[str, Any],
) -> None:
    if not manifest:
        return
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if str(entry.job.family) != str(TREE_BASELINE_FAMILY):
            continue
        payload = grouped.setdefault(
            str(entry.benchmark),
            {
                "leaf_tokens": set(),
                "train_doc_counts": set(),
                "seeds": set(),
                "allow_create": False,
            },
        )
        payload["leaf_tokens"].add(int(entry.fixed_leaf_tokens))
        payload["train_doc_counts"].add(int(entry.job.train_doc_count))
        payload["seeds"].add(int(entry.seed))
        payload["allow_create"] = bool(
            payload.get("allow_create", False)
            or bool(entry.config.prepared_data_allow_create)
        )

    for benchmark_name, payload in grouped.items():
        corpus_payload = _corpus_payload_for_benchmark(manifest, benchmark=benchmark_name)
        prepared_data_root = _corpus_prepared_data_root(corpus_payload)
        if not prepared_data_root:
            continue
        bundle_paths = _corpus_bundle_paths(corpus_payload)
        if not bundle_paths:
            raise FileNotFoundError(
                f"corpus payload for {benchmark_name!r} is missing bundle_paths"
            )
        train_doc_counts = sorted(int(value) for value in payload["train_doc_counts"])
        max_train_doc_count = max(train_doc_counts)
        bundle_path = bundle_paths.get(int(max_train_doc_count))
        if not str(bundle_path or "").strip():
            raise FileNotFoundError(
                f"corpus payload for {benchmark_name!r} is missing "
                f"bundle_train{int(max_train_doc_count)}.pkl"
            )
        base_bundle = MarkovOPSDataBundle.load(Path(str(bundle_path)))
        benchmark_spec = _resolve_benchmark_spec(str(benchmark_name))
        for leaf_tokens in sorted(int(value) for value in payload["leaf_tokens"]):
            _ensure_prepared_markov_tree_data(
                benchmark=benchmark_spec,
                base_bundle=base_bundle,
                required_train_docs=int(max_train_doc_count),
                train_prefix_counts=tuple(train_doc_counts),
                fixed_leaf_tokens=int(leaf_tokens),
                max_internal_depth=0,
                seeds=tuple(sorted(int(value) for value in payload["seeds"])),
                prepared_data_root=str(prepared_data_root),
                allow_create=bool(payload.get("allow_create", False)),
            )


def _unified_g_topology_seed_map(
    args: argparse.Namespace,
    *,
    leaf_tokens: Sequence[int],
) -> Dict[int, tuple[int, ...]]:
    primary_seeds = _ordered_unique_ints(
        list(getattr(args, "topology_seeds", ()) or UNIFIED_G_TOPOLOGY_DEFAULT_SEEDS)
    )
    stress_leaf_tokens = set(
        _ordered_unique_ints(
            list(getattr(args, "topology_stress_leaf_tokens", ()) or ())
        )
    )
    stress_seeds = _ordered_unique_ints(
        list(
            getattr(args, "topology_stress_seeds", ())
            or UNIFIED_G_TOPOLOGY_DEFAULT_STRESS_SEEDS
        )
    )
    if not primary_seeds:
        primary_seeds = tuple(int(seed) for seed in UNIFIED_G_TOPOLOGY_DEFAULT_SEEDS)
    if not stress_seeds:
        stress_seeds = primary_seeds
    seed_map: Dict[int, tuple[int, ...]] = {}
    for fixed_leaf_tokens in _ordered_unique_ints(leaf_tokens):
        if int(fixed_leaf_tokens) in stress_leaf_tokens:
            seed_map[int(fixed_leaf_tokens)] = tuple(int(seed) for seed in stress_seeds)
        else:
            seed_map[int(fixed_leaf_tokens)] = tuple(int(seed) for seed in primary_seeds)
    return seed_map


def build_parity_grid_entries(args: argparse.Namespace) -> List[ParityGridEntry]:
    doc_tokens = _doc_tokens_for_benchmark(str(args.benchmark))
    train_doc_counts = _parse_train_doc_counts(str(args.train_doc_counts))
    main_train_doc_count = max(train_doc_counts)
    entries: List[ParityGridEntry] = []
    allowed_recipe_ids = _parse_optional_token_filter(
        getattr(args, "recipe_ids", "")
    )
    allowed_fixed_leaf_tokens = _parse_optional_int_filter(
        getattr(args, "fixed_leaf_tokens", "")
    )
    include_structural = bool(getattr(args, "include_structural", True))
    manifest = _corpus_manifest(args)
    corpus_payload_by_benchmark: Dict[str, Dict[str, Any]] = {}

    def _corpus_assets_for_benchmark(benchmark_name: str) -> tuple[Dict[int, str], str]:
        if not manifest:
            return {}, ""
        payload = corpus_payload_by_benchmark.setdefault(
            str(benchmark_name),
            _corpus_payload_for_benchmark(manifest, benchmark=str(benchmark_name)),
        )
        return _corpus_bundle_paths(payload), _corpus_prepared_data_root(payload)

    if bool(getattr(args, "full_local_laws_topology_diagnostic_4096", False)):
        benchmark = str(args.benchmark)
        if benchmark != "recoverable_v4":
            raise ValueError(
                "--full-local-laws-topology-diagnostic-4096 currently only supports benchmark='recoverable_v4'"
            )
        include_structural = False
        bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(benchmark)
        scope_label = _scope_label(benchmark)
        diagnostic_train_doc_count = 4096
        diagnostic_fixed_leaf_tokens = (64, 128)
        diagnostic_seeds = (0, 1)

        def _selected_topology(recipe_id: str, fixed_leaf_tokens: int) -> bool:
            if allowed_recipe_ids is not None and str(recipe_id) not in allowed_recipe_ids:
                return False
            if (
                allowed_fixed_leaf_tokens is not None
                and int(fixed_leaf_tokens) not in allowed_fixed_leaf_tokens
            ):
                return False
            return True

        prepared_allow_create = _prepared_data_allow_create(
            args,
            prepared_data_root=str(prepared_data_root),
        )
        bundle_path = str(bundle_paths.get(int(diagnostic_train_doc_count), ""))
        for fixed_leaf_tokens in diagnostic_fixed_leaf_tokens:
            if (
                not bool(getattr(args, "skip_fno_baselines", False))
                and _selected_topology(FNO_RECIPE_ID, int(fixed_leaf_tokens))
            ):
                base_fno_config = _official_fno_reference_run_config(
                    args,
                    benchmark=benchmark,
                    fixed_leaf_tokens=int(fixed_leaf_tokens),
                    preserve_requested_leaf_tokens=True,
                )
                for seed in diagnostic_seeds:
                    config = replace(
                        base_fno_config,
                        base_bundle_path=bundle_path,
                        prepared_data_root=str(prepared_data_root),
                        prepared_data_allow_create=prepared_allow_create,
                    )
                    job = _JobSpec(
                        family="official_fno",
                        train_doc_count=int(diagnostic_train_doc_count),
                        benchmark=benchmark,
                        hardness_grid="",
                        grid_cell_ids=(),
                        seeds=(int(seed),),
                        config=config,
                        tuning_stage=FNO_RECIPE_ID,
                        study_name=STUDY_NAME,
                        study_axis=FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS,
                        axis_value=f"official_fno_leaf{int(fixed_leaf_tokens)}",
                        selection_metric="val_root_mae_mean",
                    )
                    entries.append(
                        ParityGridEntry(
                            recipe_id=FNO_RECIPE_ID,
                            benchmark=benchmark,
                            scope_key=benchmark,
                            scope_label=scope_label,
                            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                            fixed_leaf_tokens=int(fixed_leaf_tokens),
                            seed=int(seed),
                            config=config,
                            job=job,
                        )
                    )

            if not _selected_topology(
                FULL_LOCAL_LAWS_TREE_RECIPE_ID,
                int(fixed_leaf_tokens),
            ):
                continue
            base_tree_config = _full_local_laws_topology_tree_run_config(
                args,
                benchmark=benchmark,
                fixed_leaf_tokens=int(fixed_leaf_tokens),
            )
            for seed in diagnostic_seeds:
                config = replace(
                    base_tree_config,
                    base_bundle_path=bundle_path,
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=prepared_allow_create,
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(diagnostic_train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(seed),),
                    config=config,
                    tuning_stage=FULL_LOCAL_LAWS_TREE_RECIPE_ID,
                    study_name=STUDY_NAME,
                    study_axis=FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS,
                    axis_value=f"tree_neural_leaf{int(fixed_leaf_tokens)}",
                    locked_tree_neural_config_label=SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=FULL_LOCAL_LAWS_TREE_RECIPE_ID,
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                        seed=int(seed),
                        config=config,
                        job=job,
                    )
                )

        return entries

    if bool(getattr(args, "unified_g_topology_diagnostic_4096", False)):
        benchmark = str(args.benchmark)
        if benchmark != "recoverable_v4":
            raise ValueError(
                "--unified-g-topology-diagnostic-4096 currently only supports benchmark='recoverable_v4'"
            )
        include_structural = False
        bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(benchmark)
        scope_label = _scope_label(benchmark)
        diagnostic_train_doc_count = 4096
        diagnostic_fixed_leaf_tokens = _ordered_unique_ints(
            list(
                getattr(args, "topology_leaf_tokens", ())
                or UNIFIED_G_TOPOLOGY_DEFAULT_LEAF_TOKENS
            )
        )
        if not diagnostic_fixed_leaf_tokens:
            raise ValueError(
                "--unified-g-topology-diagnostic-4096 requires at least one topology leaf token"
            )
        diagnostic_seed_map = _unified_g_topology_seed_map(
            args,
            leaf_tokens=diagnostic_fixed_leaf_tokens,
        )

        def _selected_topology(recipe_id: str, fixed_leaf_tokens: int) -> bool:
            if allowed_recipe_ids is not None and str(recipe_id) not in allowed_recipe_ids:
                return False
            if (
                allowed_fixed_leaf_tokens is not None
                and int(fixed_leaf_tokens) not in allowed_fixed_leaf_tokens
            ):
                return False
            return True

        prepared_allow_create = _prepared_data_allow_create(
            args,
            prepared_data_root=str(prepared_data_root),
        )
        bundle_path = str(bundle_paths.get(int(diagnostic_train_doc_count), ""))
        one_leaf_leaf_tokens = int(_doc_tokens_for_benchmark(benchmark))
        if (
            not bool(getattr(args, "skip_fno_baselines", False))
            and int(one_leaf_leaf_tokens) in diagnostic_seed_map
            and _selected_topology(FNO_RECIPE_ID, int(one_leaf_leaf_tokens))
        ):
            base_fno_config = _official_fno_reference_run_config(
                args,
                benchmark=benchmark,
                fixed_leaf_tokens=int(one_leaf_leaf_tokens),
                preserve_requested_leaf_tokens=True,
            )
            for seed in diagnostic_seed_map[int(one_leaf_leaf_tokens)]:
                config = replace(
                    base_fno_config,
                    base_bundle_path=bundle_path,
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=prepared_allow_create,
                )
                job = _JobSpec(
                    family="official_fno",
                    train_doc_count=int(diagnostic_train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(seed),),
                    config=config,
                    tuning_stage=FNO_RECIPE_ID,
                    study_name=STUDY_NAME,
                    study_axis=UNIFIED_G_TOPOLOGY_STUDY_AXIS,
                    axis_value=f"official_fno_leaf{int(one_leaf_leaf_tokens)}",
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=FNO_RECIPE_ID,
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=int(one_leaf_leaf_tokens),
                        seed=int(seed),
                        config=config,
                        job=job,
                    )
                )

        for fixed_leaf_tokens in diagnostic_fixed_leaf_tokens:
            if not _selected_topology(
                UNIFIED_G_TOPOLOGY_RECIPE_ID,
                int(fixed_leaf_tokens),
            ):
                continue
            base_tree_config = _unified_g_topology_tree_run_config(
                args,
                benchmark=benchmark,
                fixed_leaf_tokens=int(fixed_leaf_tokens),
            )
            for seed in diagnostic_seed_map.get(int(fixed_leaf_tokens), ()):
                config = replace(
                    base_tree_config,
                    base_bundle_path=bundle_path,
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=prepared_allow_create,
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(diagnostic_train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(seed),),
                    config=config,
                    tuning_stage=UNIFIED_G_TOPOLOGY_RECIPE_ID,
                    study_name=STUDY_NAME,
                    study_axis=UNIFIED_G_TOPOLOGY_STUDY_AXIS,
                    axis_value=f"tree_neural_leaf{int(fixed_leaf_tokens)}",
                    locked_tree_neural_config_label=UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=UNIFIED_G_TOPOLOGY_RECIPE_ID,
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                        seed=int(seed),
                        config=config,
                        job=job,
                    )
                )

        return entries

    if bool(getattr(args, "exact_collapse_repair_diagnostic_matrix", False)):
        benchmark = str(args.benchmark)
        if benchmark != "recoverable_v4":
            raise ValueError(
                "--exact-collapse-repair-diagnostic-matrix currently only supports benchmark='recoverable_v4'"
            )
        include_structural = False
        diagnostic_train_doc_counts = [
            int(value)
            for value in train_doc_counts
            if int(value) == int(main_train_doc_count)
        ]
        if not diagnostic_train_doc_counts:
            diagnostic_train_doc_counts = [int(main_train_doc_count)]
        bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(benchmark)
        scope_label = _scope_label(benchmark)
        diagnostic_fixed_leaf_tokens = int(_doc_tokens_for_benchmark(benchmark))

        def _selected_diagnostic(recipe_id: str, fixed_leaf_tokens: int) -> bool:
            if allowed_recipe_ids is not None and str(recipe_id) not in allowed_recipe_ids:
                return False
            if (
                allowed_fixed_leaf_tokens is not None
                and int(fixed_leaf_tokens) not in allowed_fixed_leaf_tokens
            ):
                return False
            return True

        if (
            not bool(getattr(args, "skip_fno_baselines", False))
            and _selected_diagnostic(FNO_RECIPE_ID, diagnostic_fixed_leaf_tokens)
        ):
            base_config = _config_for_entry(
                args,
                recipe_id=FNO_RECIPE_ID,
                benchmark=benchmark,
                fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
            )
            for train_doc_count in diagnostic_train_doc_counts:
                config = replace(
                    base_config,
                    base_bundle_path=str(bundle_paths.get(int(train_doc_count), "")),
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=_prepared_data_allow_create(
                        args,
                        prepared_data_root=str(prepared_data_root),
                    ),
                )
                job = _JobSpec(
                    family="official_fno",
                    train_doc_count=int(train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(args.seed),),
                    config=config,
                    tuning_stage=FNO_RECIPE_ID,
                    study_name=STUDY_NAME,
                    study_axis="exact_collapse_repair_arm",
                    axis_value="official_fno",
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=FNO_RECIPE_ID,
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                        seed=int(args.seed),
                        config=config,
                        job=job,
                    )
                )

        for recipe_id, claim_level in (
            (
                EXACT_COLLAPSE_LEGACY_CONTROL_RECIPE_ID,
                CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
            ),
            (EXACT_COLLAPSE_RECIPE_ID, CLAIM_LEVEL_EXACT_COLLAPSE),
            (
                EXACT_COLLAPSE_RUNTIME_MATCH_RECIPE_ID,
                CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
            ),
        ):
            if not _selected_diagnostic(str(recipe_id), diagnostic_fixed_leaf_tokens):
                continue
            base_config = _config_for_entry(
                args,
                recipe_id=str(recipe_id),
                benchmark=benchmark,
                fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
            )
            reference_surface = _official_fno_reference_surface(
                args,
                benchmark=benchmark,
                fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
            )
            for train_doc_count in diagnostic_train_doc_counts:
                config = replace(
                    base_config,
                    base_bundle_path=str(bundle_paths.get(int(train_doc_count), "")),
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=_prepared_data_allow_create(
                        args,
                        prepared_data_root=str(prepared_data_root),
                    ),
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(args.seed),),
                    config=config,
                    tuning_stage=str(recipe_id),
                    study_name=STUDY_NAME,
                    study_axis="exact_collapse_repair_arm",
                    axis_value=str(recipe_id),
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=str(recipe_id),
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=str(claim_level),
                        fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                        seed=int(args.seed),
                        config=config,
                        job=job,
                        official_fno_reference_surface=dict(reference_surface),
                    )
                )

        return entries

    if bool(getattr(args, "lean_faithful_diagnostic_matrix", False)):
        benchmark = str(args.benchmark)
        if benchmark != "recoverable_v4":
            raise ValueError(
                "--lean-faithful-diagnostic-matrix currently only supports benchmark='recoverable_v4'"
            )
        if include_structural:
            include_structural = False
        diagnostic_train_doc_counts = [
            int(value)
            for value in train_doc_counts
            if int(value) == int(main_train_doc_count)
        ]
        if not diagnostic_train_doc_counts:
            diagnostic_train_doc_counts = [int(main_train_doc_count)]
        bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(benchmark)
        scope_label = _scope_label(benchmark)
        diagnostic_fixed_leaf_tokens = 16

        def _selected_diagnostic(recipe_id: str, fixed_leaf_tokens: int) -> bool:
            if allowed_recipe_ids is not None and str(recipe_id) not in allowed_recipe_ids:
                return False
            if (
                allowed_fixed_leaf_tokens is not None
                and int(fixed_leaf_tokens) not in allowed_fixed_leaf_tokens
            ):
                return False
            return True

        for recipe_id in ("matched_root", "fairfno_matched_root"):
            if not _selected_diagnostic(str(recipe_id), diagnostic_fixed_leaf_tokens):
                continue
            base_config = _config_for_entry(
                args,
                recipe_id=str(recipe_id),
                benchmark=benchmark,
                fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
            )
            for train_doc_count in diagnostic_train_doc_counts:
                config = replace(
                    base_config,
                    base_bundle_path=str(bundle_paths.get(int(train_doc_count), "")),
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=_prepared_data_allow_create(
                        args,
                        prepared_data_root=str(prepared_data_root),
                    ),
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(args.seed),),
                    config=config,
                    tuning_stage=str(recipe_id),
                    study_name=STUDY_NAME,
                    study_axis="lean_diagnostic_arm",
                    axis_value=str(recipe_id),
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=str(recipe_id),
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                        seed=int(args.seed),
                        config=config,
                        job=job,
                    )
                )

        if not bool(getattr(args, "skip_fno_baselines", False)):
            for fno_family in FNO_BASELINE_FAMILIES:
                if not _selected_diagnostic(FNO_RECIPE_ID, diagnostic_fixed_leaf_tokens):
                    continue
                base_config = _config_for_entry(
                    args,
                    recipe_id=FNO_RECIPE_ID,
                    benchmark=benchmark,
                    fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                )
                for train_doc_count in diagnostic_train_doc_counts:
                    config = replace(
                        base_config,
                        baseline_family=str(fno_family),
                        base_bundle_path=str(bundle_paths.get(int(train_doc_count), "")),
                        prepared_data_root=str(prepared_data_root),
                        prepared_data_allow_create=_prepared_data_allow_create(
                            args,
                            prepared_data_root=str(prepared_data_root),
                        ),
                    )
                    job = _JobSpec(
                        family=str(fno_family),
                        train_doc_count=int(train_doc_count),
                        benchmark=benchmark,
                        hardness_grid="",
                        grid_cell_ids=(),
                        seeds=(int(args.seed),),
                        config=config,
                        tuning_stage=FNO_RECIPE_ID,
                        study_name=STUDY_NAME,
                        study_axis="lean_diagnostic_arm",
                        axis_value=str(fno_family),
                        selection_metric="val_root_mae_mean",
                    )
                    entries.append(
                        ParityGridEntry(
                            recipe_id=FNO_RECIPE_ID,
                            benchmark=benchmark,
                            scope_key=benchmark,
                            scope_label=scope_label,
                            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                            fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                            seed=int(args.seed),
                            config=config,
                            job=job,
                        )
                    )

        base_config = _config_for_entry(
            args,
            recipe_id="matched_root",
            benchmark=benchmark,
            fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
        )
        for diagnostic_spec in _lean_faithful_local_diagnostic_specs():
            if not _selected_diagnostic(
                str(diagnostic_spec["recipe_id"]),
                diagnostic_fixed_leaf_tokens,
            ):
                continue
            variant_slug = (
                f"{str(diagnostic_spec['recipe_id'])}"
                f"__{str(diagnostic_spec['target_kind'])}"
                f"__{str(diagnostic_spec['weighting_mode'])}"
            )
            config_with_diag = replace(
                base_config,
                label=(
                    f"{variant_slug}__{_scope_label(benchmark)}__leaf"
                    f"{int(diagnostic_fixed_leaf_tokens)}"
                ),
                tree_local_weighting_mode=str(diagnostic_spec["weighting_mode"]),
                leaf_label_rate=float(diagnostic_spec["leaf_label_rate"]),
                internal_label_rate=float(diagnostic_spec["internal_label_rate"]),
                leaf_supervision_kind=str(diagnostic_spec["leaf_supervision_kind"]),
                internal_supervision_kind=str(
                    diagnostic_spec["internal_supervision_kind"]
                ),
                leaf_exact_supervision=bool(diagnostic_spec["leaf_exact_supervision"]),
                tree_local_law_weight=float(
                    diagnostic_spec.get("tree_local_law_weight", 0.8)
                ),
                tree_task_objective_weight=None,
            )
            for train_doc_count in diagnostic_train_doc_counts:
                config = _with_run_intent_overrides(
                    replace(
                        config_with_diag,
                        base_bundle_path=str(bundle_paths.get(int(train_doc_count), "")),
                        prepared_data_root=str(prepared_data_root),
                        prepared_data_allow_create=_prepared_data_allow_create(
                            args,
                            prepared_data_root=str(prepared_data_root),
                        ),
                    ),
                    budget_total_calls_per_doc=float(
                        diagnostic_spec["budget_total_calls_per_doc"]
                    ),
                    full_doc_budget_share=float(
                        diagnostic_spec["full_doc_budget_share"]
                    ),
                    doc_consumption_mode=str(
                        diagnostic_spec["doc_consumption_mode"]
                    ),
                    local_split_mode=str(diagnostic_spec["local_split_mode"]),
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(train_doc_count),
                    benchmark=benchmark,
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(args.seed),),
                    config=config,
                    tuning_stage=str(variant_slug),
                    study_name=STUDY_NAME,
                    study_axis="lean_diagnostic_arm",
                    axis_value=str(variant_slug),
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=str(diagnostic_spec["recipe_id"]),
                        benchmark=benchmark,
                        scope_key=benchmark,
                        scope_label=scope_label,
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                        seed=int(args.seed),
                        config=config,
                        job=job,
                        nominal_recipe_metadata=_nominal_recipe_metadata(
                            recipe_id=str(diagnostic_spec["recipe_id"]),
                            recipe_budget_total_calls_per_doc=float(
                                diagnostic_spec["recipe_budget_total_calls_per_doc"]
                            ),
                        ),
                    )
                )

        if bool(getattr(args, "lean_faithful_weight_balance_sweep", False)):
            for diagnostic_spec in _lean_faithful_weight_balance_specs():
                if not _selected_diagnostic(
                    str(diagnostic_spec["recipe_id"]),
                    diagnostic_fixed_leaf_tokens,
                ):
                    continue
                local_weight_pct = int(
                    round(100.0 * float(diagnostic_spec["tree_local_law_weight"]))
                )
                c1_weight = int(round(float(diagnostic_spec["tree_c1_relative_weight"])))
                c3_weight = int(round(float(diagnostic_spec["tree_c3_relative_weight"])))
                variant_slug = (
                    f"{str(diagnostic_spec['recipe_id'])}"
                    f"__weight_sweep"
                    f"__lw{local_weight_pct:02d}"
                    f"__c1_{c1_weight}"
                    f"__c3_{c3_weight}"
                )
                config_with_diag = replace(
                    base_config,
                    label=(
                        f"{variant_slug}__{_scope_label(benchmark)}__leaf"
                        f"{int(diagnostic_fixed_leaf_tokens)}"
                    ),
                    tree_local_weighting_mode=str(diagnostic_spec["weighting_mode"]),
                    leaf_label_rate=float(diagnostic_spec["leaf_label_rate"]),
                    internal_label_rate=float(diagnostic_spec["internal_label_rate"]),
                    leaf_supervision_kind=str(diagnostic_spec["leaf_supervision_kind"]),
                    internal_supervision_kind=str(
                        diagnostic_spec["internal_supervision_kind"]
                    ),
                    leaf_exact_supervision=bool(diagnostic_spec["leaf_exact_supervision"]),
                    tree_local_law_weight=float(
                        diagnostic_spec["tree_local_law_weight"]
                    ),
                    tree_c1_relative_weight=float(
                        diagnostic_spec["tree_c1_relative_weight"]
                    ),
                    tree_c2_relative_weight=float(
                        diagnostic_spec["tree_c2_relative_weight"]
                    ),
                    tree_c3_relative_weight=float(
                        diagnostic_spec["tree_c3_relative_weight"]
                    ),
                )
                for train_doc_count in diagnostic_train_doc_counts:
                    config = _with_run_intent_overrides(
                        replace(
                            config_with_diag,
                            base_bundle_path=str(bundle_paths.get(int(train_doc_count), "")),
                            prepared_data_root=str(prepared_data_root),
                            prepared_data_allow_create=_prepared_data_allow_create(
                                args,
                                prepared_data_root=str(prepared_data_root),
                            ),
                        ),
                        budget_total_calls_per_doc=float(
                            diagnostic_spec["budget_total_calls_per_doc"]
                        ),
                        full_doc_budget_share=float(
                            diagnostic_spec["full_doc_budget_share"]
                        ),
                        doc_consumption_mode=str(
                            diagnostic_spec["doc_consumption_mode"]
                        ),
                        local_split_mode=str(diagnostic_spec["local_split_mode"]),
                    )
                    job = _JobSpec(
                        family=TREE_BASELINE_FAMILY,
                        train_doc_count=int(train_doc_count),
                        benchmark=benchmark,
                        hardness_grid="",
                        grid_cell_ids=(),
                        seeds=(int(args.seed),),
                        config=config,
                        tuning_stage=str(variant_slug),
                        study_name=STUDY_NAME,
                        study_axis="lean_weight_balance",
                        axis_value=str(variant_slug),
                        selection_metric="val_root_mae_mean",
                    )
                    entries.append(
                        ParityGridEntry(
                            recipe_id=str(diagnostic_spec["recipe_id"]),
                            benchmark=benchmark,
                            scope_key=benchmark,
                            scope_label=scope_label,
                            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                            fixed_leaf_tokens=int(diagnostic_fixed_leaf_tokens),
                            seed=int(args.seed),
                            config=config,
                            job=job,
                            nominal_recipe_metadata=_nominal_recipe_metadata(
                                recipe_id=str(diagnostic_spec["recipe_id"]),
                                recipe_budget_total_calls_per_doc=float(
                                    diagnostic_spec["recipe_budget_total_calls_per_doc"]
                                ),
                            ),
                        )
                    )

        return entries

    geometry_matrix = [
        # All five recipes at leaf=16 (8 leaves on 128-token docs)
        ("historical_replay", str(args.benchmark), 16),
        ("optimization_fairness", str(args.benchmark), 16),
        ("capacity_fairness", str(args.benchmark), 16),
        ("matched_root", str(args.benchmark), 16),
        # All five recipes at leaf=64 (2 leaves on 128-token docs)
        ("historical_replay", str(args.benchmark), 64),
        ("optimization_fairness", str(args.benchmark), 64),
        ("capacity_fairness", str(args.benchmark), 64),
        ("matched_root", str(args.benchmark), 64),
        # Geometry sweep for matched_root: 32, 128
        ("matched_root", str(args.benchmark), 32),
        ("matched_root", str(args.benchmark), 128),
        # All five recipes at leaf=128 (1 leaf = single-leaf regime)
        ("historical_replay", str(args.benchmark), 128),
        ("optimization_fairness", str(args.benchmark), 128),
        ("capacity_fairness", str(args.benchmark), 128),
        ("fairfno_matched_root", str(args.benchmark), 128),
        # Fair-FNO geometry sweep: 16, 32, 64
        ("fairfno_matched_root", str(args.benchmark), 16),
        ("fairfno_matched_root", str(args.benchmark), 32),
        ("fairfno_matched_root", str(args.benchmark), 64),
        # Structural confirmation at single-leaf
        ("matched_root", str(args.structural_benchmark), 128),
    ]
    if not include_structural:
        geometry_matrix = [
            row for row in geometry_matrix if str(row[1]) != str(args.structural_benchmark)
        ]
    exact_collapse_matrix = [
        (EXACT_COLLAPSE_RECIPE_ID, str(args.benchmark), int(doc_tokens)),
    ]

    entries: List[ParityGridEntry] = []

    def _selected(recipe_id: str, benchmark: str, fixed_leaf_tokens: int) -> bool:
        if not include_structural and str(benchmark) == str(args.structural_benchmark):
            return False
        if allowed_recipe_ids is not None and str(recipe_id) not in allowed_recipe_ids:
            return False
        if (
            allowed_fixed_leaf_tokens is not None
            and int(fixed_leaf_tokens) not in allowed_fixed_leaf_tokens
        ):
            return False
        return True

    # --- Tree-side entries (swept over train_doc_counts) ---
    for claim_level, matrix in (
        (CLAIM_LEVEL_EMPIRICAL_GEOMETRY, geometry_matrix),
        (CLAIM_LEVEL_EXACT_COLLAPSE, exact_collapse_matrix),
    ):
        for recipe_id, benchmark, fixed_leaf_tokens in matrix:
            if not _selected(
                str(recipe_id),
                str(benchmark),
                int(fixed_leaf_tokens),
            ):
                continue
            scope_label = _scope_label(benchmark)
            bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(str(benchmark))
            base_config = _config_for_entry(
                args,
                recipe_id=str(recipe_id),
                benchmark=str(benchmark),
                fixed_leaf_tokens=int(fixed_leaf_tokens),
            )
            for train_doc_count in train_doc_counts:
                config = replace(
                    base_config,
                    base_bundle_path=str(
                        bundle_paths.get(int(train_doc_count), "")
                    ),
                    prepared_data_root=str(prepared_data_root),
                    prepared_data_allow_create=_prepared_data_allow_create(
                        args,
                        prepared_data_root=str(prepared_data_root),
                    ),
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(train_doc_count),
                    benchmark=str(benchmark),
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(args.seed),),
                    config=config,
                    tuning_stage=str(recipe_id),
                    study_name=STUDY_NAME,
                    study_axis="fixed_leaf_tokens",
                    axis_value=str(int(fixed_leaf_tokens)),
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=str(recipe_id),
                        benchmark=str(benchmark),
                        scope_key=str(benchmark),
                        scope_label=str(scope_label),
                        claim_level=str(claim_level),
                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                        seed=int(args.seed),
                        config=config,
                        job=job,
                        official_fno_reference_surface=(
                            _official_fno_reference_surface(
                                args,
                                benchmark=str(benchmark),
                                fixed_leaf_tokens=int(fixed_leaf_tokens),
                            )
                            if str(recipe_id) == EXACT_COLLAPSE_RECIPE_ID
                            else {}
                        ),
                    )
                )

    # --- FNO baseline entries (same data, same benchmark, swept over train_doc_counts) ---
    if not bool(getattr(args, "skip_fno_baselines", False)):
        benchmark_loop: Sequence[str] = (
            (str(args.benchmark), str(args.structural_benchmark))
            if include_structural
            else (str(args.benchmark),)
        )
        for fno_family in FNO_BASELINE_FAMILIES:
            for benchmark in benchmark_loop:
                if not _selected(FNO_RECIPE_ID, str(benchmark), int(doc_tokens)):
                    continue
                scope_label = _scope_label(benchmark)
                bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(str(benchmark))
                base_config = _config_for_entry(
                    args,
                    recipe_id=FNO_RECIPE_ID,
                    benchmark=str(benchmark),
                    fixed_leaf_tokens=int(doc_tokens),
                )
                for train_doc_count in train_doc_counts:
                    config = replace(
                        base_config,
                        baseline_family=str(fno_family),
                        base_bundle_path=str(
                            bundle_paths.get(int(train_doc_count), "")
                        ),
                        prepared_data_root=str(prepared_data_root),
                        prepared_data_allow_create=_prepared_data_allow_create(
                            args,
                            prepared_data_root=str(prepared_data_root),
                        ),
                    )
                    job = _JobSpec(
                        family=str(fno_family),
                        train_doc_count=int(train_doc_count),
                        benchmark=str(benchmark),
                        hardness_grid="",
                        grid_cell_ids=(),
                        seeds=(int(args.seed),),
                        config=config,
                        tuning_stage=FNO_RECIPE_ID,
                        study_name=STUDY_NAME,
                        study_axis="fno_baseline",
                        axis_value=str(fno_family),
                        selection_metric="val_root_mae_mean",
                    )
                    entries.append(
                        ParityGridEntry(
                            recipe_id=FNO_RECIPE_ID,
                            benchmark=str(benchmark),
                            scope_key=str(benchmark),
                            scope_label=str(scope_label),
                            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                            fixed_leaf_tokens=int(doc_tokens),
                            seed=int(args.seed),
                            config=config,
                            job=job,
                        )
                    )

    # --- R10/R20 local-supervision-rate sweep ---
    #
    # These use the matched_root preset at leaf=16 (full-tree regime where
    # local supervision matters) with varying leaf/internal label rates.
    # Unlike the legacy count-only sweep, supervised leaves now use exact
    # summary-spec labels (`full_sketch`) so the local target is the true leaf
    # answer, not just the scalar count. The R10/R20 job metadata also carries
    # the corresponding sparse root-budget setting.
    if bool(getattr(args, "include_supervision_sweep", False)):
        base_config = _config_for_entry(
            args,
            recipe_id="matched_root",
            benchmark=str(args.benchmark),
            fixed_leaf_tokens=16,
        )
        bundle_paths, prepared_data_root = _corpus_assets_for_benchmark(str(args.benchmark))
        for sweep_spec in _local_supervision_sweep_specs():
            if not _selected(str(sweep_spec["recipe_id"]), str(args.benchmark), 16):
                continue
            config_with_rates = replace(
                base_config,
                leaf_label_rate=float(sweep_spec["leaf_label_rate"]),
                internal_label_rate=float(sweep_spec["internal_label_rate"]),
                leaf_supervision_kind=str(sweep_spec["leaf_supervision_kind"]),
                internal_supervision_kind=str(
                    sweep_spec["internal_supervision_kind"]
                ),
                leaf_exact_supervision=bool(sweep_spec["leaf_exact_supervision"]),
            )
            for train_doc_count in train_doc_counts:
                config = _with_run_intent_overrides(
                    replace(
                        config_with_rates,
                        base_bundle_path=str(
                            bundle_paths.get(int(train_doc_count), "")
                        ),
                        prepared_data_root=str(prepared_data_root),
                        prepared_data_allow_create=_prepared_data_allow_create(
                            args,
                            prepared_data_root=str(prepared_data_root),
                        ),
                    ),
                    budget_total_calls_per_doc=float(
                        sweep_spec["budget_total_calls_per_doc"]
                    ),
                    full_doc_budget_share=float(sweep_spec["full_doc_budget_share"]),
                    doc_consumption_mode=str(sweep_spec["doc_consumption_mode"]),
                    local_split_mode=str(sweep_spec["local_split_mode"]),
                )
                job = _JobSpec(
                    family=TREE_BASELINE_FAMILY,
                    train_doc_count=int(train_doc_count),
                    benchmark=str(args.benchmark),
                    hardness_grid="",
                    grid_cell_ids=(),
                    seeds=(int(args.seed),),
                    config=config,
                    tuning_stage=str(sweep_spec["recipe_id"]),
                    study_name=STUDY_NAME,
                    study_axis="supervision_rate",
                    axis_value=str(sweep_spec["recipe_id"]),
                    selection_metric="val_root_mae_mean",
                )
                entries.append(
                    ParityGridEntry(
                        recipe_id=str(sweep_spec["recipe_id"]),
                        benchmark=str(args.benchmark),
                        scope_key=str(args.benchmark),
                        scope_label="recoverable",
                        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        fixed_leaf_tokens=16,
                        seed=int(args.seed),
                        config=config,
                        job=job,
                    )
                )

    return entries


def _manifest_payload(
    *,
    args: argparse.Namespace,
    output_root: Path,
    entries: Sequence[ParityGridEntry],
    devices: Sequence[str],
) -> Dict[str, Any]:
    entry_train_doc_counts = sorted(
        {
            int(entry.job.train_doc_count)
            for entry in list(entries)
            if int(entry.job.train_doc_count) > 0
        }
    ) or _parse_train_doc_counts(str(args.train_doc_counts))
    effective_epoch_cap = _effective_epoch_cap(args)
    return {
        "generated_at": _utc_now(),
        "study_name": STUDY_NAME,
        "output_root": str(output_root),
        "benchmark": str(args.benchmark),
        "structural_benchmark": str(args.structural_benchmark),
        "train_doc_counts": list(entry_train_doc_counts),
        "main_train_doc_count": max(entry_train_doc_counts),
        "corpus_root": str(getattr(args, "corpus_root", "") or ""),
        "seed": int(args.seed),
        "lean_faithful_diagnostic_matrix": bool(
            getattr(args, "lean_faithful_diagnostic_matrix", False)
        ),
        "exact_collapse_repair_diagnostic_matrix": bool(
            getattr(args, "exact_collapse_repair_diagnostic_matrix", False)
        ),
        "full_local_laws_topology_diagnostic_4096": bool(
            getattr(args, "full_local_laws_topology_diagnostic_4096", False)
        ),
        "unified_g_topology_diagnostic_4096": bool(
            getattr(args, "unified_g_topology_diagnostic_4096", False)
        ),
        "topology_seeds": [
            int(value) for value in list(getattr(args, "topology_seeds", ()) or [])
        ],
        "topology_leaf_tokens": [
            int(value)
            for value in list(getattr(args, "topology_leaf_tokens", ()) or [])
        ],
        "topology_stress_leaf_tokens": [
            int(value)
            for value in list(getattr(args, "topology_stress_leaf_tokens", ()) or [])
        ],
        "topology_stress_seeds": [
            int(value)
            for value in list(getattr(args, "topology_stress_seeds", ()) or [])
        ],
        "topology_posttrain_train_doc_limit": int(
            getattr(args, "topology_posttrain_train_doc_limit", 0) or 0
        ),
        "topology_posttrain_diagnostics_mode": str(
            getattr(args, "topology_posttrain_diagnostics_mode", "") or ""
        ),
        "topology_stress_posttrain_diagnostics_mode": str(
            getattr(args, "topology_stress_posttrain_diagnostics_mode", "") or ""
        ),
        "epoch_cap": int(effective_epoch_cap),
        "exact_metric_selection_doc_limit": int(
            getattr(args, "exact_metric_selection_doc_limit", 0) or 0
        ),
        "exact_metric_selection_interval": int(
            getattr(args, "exact_metric_selection_interval", 1) or 1
        ),
        "assumed_doc_tokens": int(
            _doc_tokens_for_benchmark(str(args.benchmark))
        ),
        "one_leaf_target_fixed_leaf_tokens": int(
            _doc_tokens_for_benchmark(str(args.benchmark))
        ),
        "canonical_train_ladder": _canonical_train_ladder_payload(),
        "requested_devices": [str(device) for device in devices],
        "jobs": [
            entry.manifest_row(
                output_root=output_root,
                main_train_doc_count=max(entry_train_doc_counts),
                epoch_cap=int(effective_epoch_cap),
            )
            for entry in list(entries)
        ],
    }


# Completed-run IO/materialization is package code so validators and CLI
# wrappers share the same row contract.

def _ensure_corpus(args: argparse.Namespace, output_root: Path) -> Path:
    """Return the corpus root, creating it under *output_root* if needed.

    When auto-creating, delegates to ``prepare_markov_parity_corpus`` so
    that the corpus layout (manifest + per-prefix ``MarkovOPSDataBundle``
    JSONs) is identical whether the corpus was prepared ahead of time or
    generated inline.
    """
    explicit = str(getattr(args, "corpus_root", "") or "").strip()
    if explicit:
        corpus_root = Path(explicit).expanduser().resolve()
        manifest = corpus_root / "corpus_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"--corpus-root {corpus_root} does not contain corpus_manifest.json"
            )
        return corpus_root

    # Auto-create under the experiment output root.
    corpus_root = output_root / "corpus"
    if (corpus_root / "corpus_manifest.json").exists():
        return corpus_root

    import subprocess
    train_doc_counts = _parse_train_doc_counts(str(args.train_doc_counts))
    max_train = max(train_doc_counts)
    total_docs = max_train + 1024 + 2048  # 1k test + 2k val

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_markov_parity_corpus.py"),
        "--output-root", str(corpus_root),
        "--benchmark", str(args.benchmark),
        "--benchmarks",
        " ".join((str(args.benchmark), str(args.structural_benchmark))),
        "--total-docs", str(total_docs),
        "--train-doc-counts", " ".join(str(c) for c in train_doc_counts),
        "--test-docs", "1024",
        "--seed", str(args.seed),
    ]
    print(f"[corpus] auto-creating at {corpus_root} ...")
    result = subprocess.run(cmd, check=True)
    return corpus_root


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_root = _ensure_corpus(args, output_root)
    args.corpus_root = str(corpus_root)
    entries = build_parity_grid_entries(args)
    _validate_prepared_tree_data_coverage(
        entries=entries,
        manifest=_corpus_manifest(args),
    )
    devices = _discover_scheduler_devices(args)
    if not devices:
        raise RuntimeError("no MIG devices found; pass --mig-uuids explicitly")
    items = [
        _scheduler_item_for_job(
            phase=STUDY_NAME,
            item_id=f"{STUDY_NAME}::{index:02d}::{entry.job.job_name}",
            output_root=output_root,
            job=entry.job,
            torch_threads=int(args.torch_threads),
            use_cuda=bool(args.use_cuda),
            gpu_slots=1,
        )
        for index, entry in enumerate(entries)
    ]
    manifest_payload = _manifest_payload(
        args=args,
        output_root=output_root,
        entries=entries,
        devices=devices,
    )
    scheduler_plan = _scheduler_cli_payload(
        items=items,
        devices=devices,
        max_gpu_items_per_mig=int(args.max_gpu_items_per_mig),
        launch_stagger_seconds=float(args.scheduler_launch_stagger_seconds),
        min_mem_available_kib=int(
            max(0.0, float(args.scheduler_min_mem_available_gib)) * 1024.0 * 1024.0
        ),
        min_swap_free_kib=int(
            max(0.0, float(args.scheduler_min_swap_free_gib)) * 1024.0 * 1024.0
        ),
        manifest_payload=manifest_payload,
    )
    return {
        "output_root": output_root,
        "entries": entries,
        "devices": devices,
        "items": items,
        "manifest_payload": {
            **manifest_payload,
            "scheduler_plan": scheduler_plan["scheduler"],
        },
    }


def _write_plan_files(plan: Mapping[str, Any]) -> None:
    output_root = Path(str(plan["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / PARITY_MANIFEST_NAME, dict(plan["manifest_payload"]))
    _write_materialized_outputs(output_root)


def main() -> int:
    args = _parse_args()
    plan = build_plan(args)
    output_root = Path(str(plan["output_root"]))
    _write_plan_files(plan)
    if bool(args.plan_only):
        print(output_root)
        return 0

    result = _run_scheduler_bundle(
        output_root=output_root,
        items=list(plan["items"]),
        devices=list(plan["devices"]),
        max_gpu_items_per_mig=int(args.max_gpu_items_per_mig),
        launch_stagger_seconds=float(args.scheduler_launch_stagger_seconds),
        cleanup_stale_children=bool(args.cleanup_stale_children),
        resume_enabled=bool(args.resume),
        manifest_payload=dict(plan["manifest_payload"]),
        min_mem_available_kib=int(
            max(0.0, float(args.scheduler_min_mem_available_gib)) * 1024.0 * 1024.0
        ),
        min_swap_free_kib=int(
            max(0.0, float(args.scheduler_min_swap_free_gib)) * 1024.0 * 1024.0
        ),
    )
    combined_runs: List[Mapping[str, Any]] = []
    for entry in list(plan["entries"]):
        job_output_dir = output_root / "jobs" / entry.job_output_dir_name
        summary_payload = _load_json(job_output_dir / "summary.json")
        combined_runs.extend(
            dict(run)
            for run in list(summary_payload.get("runs") or [])
            if isinstance(run, Mapping)
        )
    if combined_runs:
        _write_combined_runs_output(output_root=output_root, runs=combined_runs)
    _write_materialized_outputs(output_root)
    print(output_root)
    return 0 if not list(result.get("failed_jobs") or []) else 1


if __name__ == "__main__":
    raise SystemExit(main())
