#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.util import safe_float, safe_int
from src.experiments.script_parse import (
    parse_float_list as _shared_parse_float_list,
    parse_int_list as _shared_parse_int_list,
    parse_str_list as _shared_parse_str_list,
)
from src.ctreepo.contracts import (
    LAW_SET_ALL,
    LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    LAW_SET_ROOT_ONLY,
    LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
    RunAxisSpec,
    assert_public_contract_clean,
    canonical_law_set_id,
    markov_tree_bundle_metadata,
    objective_metadata,
    run_manifest_metadata,
)
from src.experiments import (
    ExperimentSpec,
    ProgressSnapshot,
    ResultRow,
    append_result_rows,
    benchmark_ref_from_parts,
    canonical_artifact_refs_from_paths,
    default_phase_specs,
    merge_artifacts,
    write_experiment_manifest,
    write_experiment_status,
)
from src.experiments.markov_full_doc import method_ref_from_markov_full_doc_run
from src.experiments.scheduler import (
    SchedulerConfig,
    SchedulerItem,
    run_scheduler,
)
from src.ctreepo.sim.core.tree_neural_facade import job_output_dir_name
from src.ctreepo.sim.core.tree_neural_execution import worker_command_for_job
from src.ctreepo.sim.core.markov_comparison_surface import (
    FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS,
    apply_comparable_surface_to_mapping,
    comparison_surface_diff,
    infer_markov_comparison_mode,
    normalize_markov_comparison_mode,
    resolve_markov_comparable_surface,
)
from src.ctreepo.sim.core.full_doc_config_codec import (
    LEGACY_PUBLIC_OBJECTIVE_CONFIG_FIELDS,
    LEGACY_PUBLIC_RUN_AXIS_CONFIG_FIELDS,
    runtime_config_overrides_from_config_like,
    serialize_full_doc_runtime_config,
)
from src.ctreepo.sim.core.device_resolver import (
    build_worker_env as _build_worker_env,
    filter_available_devices as _filter_available_mig_devices,
    set_thread_env_defaults as _set_shared_thread_env_defaults,
)
from src.ctreepo.sim.core.markov_alignment_validation import (
    build_markov_alignment_audit_report,
    write_markov_alignment_audit_report,
)
from src.ctreepo.sim.core.markov_v3_row_contract import (
    annotate_downstream_v3_row,
    filtered_headline_rows,
    filtered_quarantined_rows,
    quarantine_sources_from_rows,
)
from src.ctreepo.sim.core.run_intent import resolve_package_semantics
from src.ctreepo.sim.core.markov_study_names import (
    LAW_PACKAGE_ALIASES as _LAW_PACKAGE_ALIASES,
    SUPERVISION_RECOVERY_PACKAGE_ALIASES as _SUPERVISION_RECOVERY_PACKAGE_ALIASES,
    SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES as _SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES,
    resolve_law_package_names as _resolve_law_package_names,
    resolve_supervision_recovery_package_names as _resolve_supervision_recovery_package_names,
)
from src.ctreepo.sim.core.markov_hazard_panels import (
    panel_to_ops_overrides,
    resolve_markov_hazard_panel,
)
from src.ctreepo.sim.core.tree_reference_presets import (  # noqa: E402
    COMPARISON_GRID_V3_PRESET as _PRESET_COMPARISON_GRID_V3,
    SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET as _PRESET_COMMON,
    UNIFIED_G_FULL_LOCAL_LAWS_PRESET as _PRESET_UNIFIED_G,
    UNIFIED_G_FNO_PARITY_CANARY_PRESET as _PRESET_FNO_CANARY,
    ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET as _PRESET_ROOT_REPLAY,
    ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET as _PRESET_ROOT_OPT_FIX,
    ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET as _PRESET_ROOT_CAP_FIX,
    ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET as _PRESET_ROOT_MATCHED,
    STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET as _PRESET_STRUCTURAL_MATCHED,
    TREE_REFERENCE_OVERRIDE_KEYS as _IMPORTED_OVERRIDE_KEYS,
    TREE_REFERENCE_PRESET_CONFIGS as _IMPORTED_PRESET_CONFIGS,
    resolve_tree_reference_preset as _resolve_tree_reference_preset,
    resolve_tree_reference_preset_recipe as _resolve_tree_reference_preset_recipe,
    tree_reference_label as _imported_tree_reference_label,
)
from src.ctreepo.sim.util import get_str as _gs, norm_str as _ns
from src.experiments.structured_config import load_structured_config, write_structured_config


FULL_DOC_CONFIG_WORKER_KINDS = frozenset({"full_doc_diagnostics", "full_doc_upper_bound"})


def _mass_matched_rate_suffix(rate_percent: float) -> str:
    return f"{float(rate_percent):.1f}".replace(".", "p")


def _mass_matched_package_name(root_share: int, rate_percent: float) -> str:
    return f"r{int(root_share)}_mass_local_eq_{_mass_matched_rate_suffix(rate_percent)}"


def _node_mass_target_package_name(local_mass_percent: float) -> str:
    return f"r100_node_mass_eq_{_mass_matched_rate_suffix(local_mass_percent)}"


def _leaf_mass_preserving_package_name(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return f"r{int(root_share)}_leaf_mass_eq_{_mass_matched_rate_suffix(local_mass_percent)}"


def _depth_equal_mass_preserving_package_name(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return (
        f"r{int(root_share)}_depth_equal_mass_eq_"
        f"{_mass_matched_rate_suffix(local_mass_percent)}"
    )


def _mass_matched_package_order(
    root_share: int,
    local_rate_percents: Sequence[float],
) -> tuple[str, ...]:
    return (
        f"full{int(root_share)}",
        *[
            _mass_matched_package_name(int(root_share), float(rate_percent))
            for rate_percent in local_rate_percents
        ],
    )


def _build_mass_matched_package_specs(
    ladders: Mapping[int, Sequence[float]],
) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for root_share, local_rate_percents in ladders.items():
        target_mass = float(root_share) / 100.0
        for rate_percent in local_rate_percents:
            package_name = _mass_matched_package_name(
                int(root_share),
                float(rate_percent),
            )
            specs[package_name] = {
                "label": f"R{int(root_share)} mass-matched + {float(rate_percent):.1f}% leaf/internal count",
                "description": (
                    f"Tree matches the R{int(root_share)} full-doc-equivalent mass target "
                    f"with equal {float(rate_percent):.1f}% leaf/internal count labels "
                    "and only the residual root supervision budget."
                ),
                "mass_target_per_doc": target_mass,
                "budget_total_calls_per_doc": target_mass,
                "full_doc_budget_share": 1.0,
                "doc_consumption_mode": "root_only",
                "local_split_mode": "balanced",
                "leaf_supervision_kind": "count_only",
                "leaf_label_rate": float(rate_percent) / 100.0,
                "internal_supervision_kind": "count_only",
                "internal_label_rate": float(rate_percent) / 100.0,
                "run_fno": False,
                "fno_reference_package": f"full{int(root_share)}",
                "package_semantics": "mass_matched",
            }
    return specs


def _build_node_mass_target_package_specs(
    local_mass_targets: Sequence[float],
) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for local_mass_percent in local_mass_targets:
        local_mass_target = float(local_mass_percent) / 100.0
        root_mass_target = max(0.0, 1.0 - float(local_mass_target))
        package_name = _node_mass_target_package_name(float(local_mass_percent))
        specs[package_name] = {
            "label": (
                f"{root_mass_target * 100.0:.0f}% root + "
                f"{float(local_mass_percent):.1f}% node mass"
            ),
            "description": (
                "Tree keeps total full-doc-equivalent supervision mass fixed at 100% "
                f"while targeting {float(local_mass_percent):.1f}% of that mass on "
                "covered leaf/internal nodes and the residual on root supervision."
            ),
            "mass_target_per_doc": 1.0,
            "local_mass_target_per_doc": float(local_mass_target),
            "budget_total_calls_per_doc": float(root_mass_target),
            "full_doc_budget_share": 1.0,
            "doc_consumption_mode": "root_only",
            "local_split_mode": "balanced",
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.0,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.0,
            "run_fno": False,
            "fno_reference_package": "full100",
            "package_semantics": "mass_matched",
        }
    return specs


def _build_depth_profile_mass_preserving_package_specs(
    *,
    root_shares: Sequence[int],
    local_split_mode: str,
) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    normalized_mode = str(local_split_mode).strip().lower()
    if normalized_mode not in {"leaf_only", "depth_equal_nonroot"}:
        raise ValueError(
            "local_split_mode for depth-profile mass preserving packages must be "
            f"'leaf_only' or 'depth_equal_nonroot', got {local_split_mode!r}"
        )
    for root_share in root_shares:
        root_mass_target = max(0.0, min(1.0, float(root_share) / 100.0))
        local_mass_target = max(0.0, 1.0 - float(root_mass_target))
        if normalized_mode == "leaf_only":
            package_name = _leaf_mass_preserving_package_name(int(root_share))
            label = f"R{int(root_share)} + leaf-only {local_mass_target * 100.0:.0f}%"
            description = (
                "Tree keeps total full-doc-equivalent supervision mass fixed at 100% "
                f"while retaining {root_mass_target * 100.0:.0f}% root supervision and "
                "placing all remaining local mass on leaves only."
            )
        else:
            package_name = _depth_equal_mass_preserving_package_name(int(root_share))
            label = (
                f"R{int(root_share)} + equal local-depth {local_mass_target * 100.0:.0f}%"
            )
            description = (
                "Tree keeps total full-doc-equivalent supervision mass fixed at 100% "
                f"while retaining {root_mass_target * 100.0:.0f}% root supervision and "
                "distributing the remaining local mass evenly over leaves and the "
                "available non-root merge depths."
            )
        specs[package_name] = {
            "label": label,
            "description": description,
            "mass_target_per_doc": 1.0,
            "local_mass_target_per_doc": float(local_mass_target),
            "budget_total_calls_per_doc": float(root_mass_target),
            "full_doc_budget_share": 1.0,
            "doc_consumption_mode": "root_only",
            "local_split_mode": normalized_mode,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.0,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.0,
            "run_fno": False,
            "fno_reference_package": "full100",
            "package_semantics": "mass_matched",
        }
    return specs


def _nonroot_internal_depth_count(*, n_leaves: int) -> int:
    n = int(max(0, n_leaves))
    if n <= 2:
        return 0
    total_depths = 0
    while n > 1:
        total_depths += 1
        n = int((n + 1) // 2)
    return int(max(0, total_depths - 1))


def _default_supervision_recovery_package_semantics(
    package_name: str,
    package_spec: Mapping[str, Any],
) -> str:
    explicit = str(package_spec.get("package_semantics", "") or "").strip()
    if explicit:
        return explicit
    if (
        "_mass_local_eq_" in str(package_name)
        or "_node_mass_eq_" in str(package_name)
    ):
        return "mass_matched"
    leaf_rate = float(safe_float(package_spec.get("leaf_label_rate"), 0.0))
    internal_rate = float(
        safe_float(package_spec.get("internal_label_rate"), 0.0)
    )
    internal_kind = str(package_spec.get("internal_supervision_kind", "none") or "none")
    local_active = bool(
        leaf_rate > 1e-12
        or (
            internal_kind.strip().lower() != "none"
            and internal_rate > 1e-12
        )
    )
    if local_active and float(safe_float(package_spec.get("budget_total_calls_per_doc"), 0.0)) > 0.0:
        return "superset"
    if local_active:
        return "local_only"
    return "full_doc_only"


PROFILE_SCRIPT = REPO_ROOT / "scripts" / "profile_markov_fixed_fused_autotune.py"
LEARNABILITY_REPORT_SCRIPT = REPO_ROOT / "scripts" / "report_learnability.py"
SUPPORT_SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_markov_local_support_grid.py"
TRADEOFF_REPORT_SCRIPT = REPO_ROOT / "scripts" / "report_markov_optimization_tradeoffs.py"
TREE_FULL_DOC_SCRIPT = REPO_ROOT / "scripts" / "run_tree_neural_full_doc_mig.py"
REPORT_VERSION_MANIFEST_NAME = "report_version_manifest.json"
REPORT_VERSION_SCHEMA_VERSION = 1
CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES = ("official_fno", "official_fno_sumlen")
EFFICIENCY_TREE_METHOD_RUNS = (
    "tree_neural:on_range_idempotence_only",
    "tree_neural:all",
)
EFFICIENCY_TREE_BASELINE_FAMILIES = ("tree_neural_c2", "tree_neural")
EFFICIENCY_DENSE_ANCHOR_FAMILIES = (
    *CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
    *EFFICIENCY_TREE_BASELINE_FAMILIES,
)
DEFAULT_TREE_METHOD_RUNS = (
    "tree_neural:on_range_idempotence_only",
    "tree_neural:all",
)
DEFAULT_REFERENCE_METHOD_RUNS = tuple(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)
EFFICIENCY_STRUCTURAL_BASELINE_FAMILIES = (
    *EFFICIENCY_DENSE_ANCHOR_FAMILIES,
    "palette_block_exact",
)
EFFICIENCY_STRUCTURAL_CORE_CELLS = (
    "r4_p031",
    "r12_p031",
    "r4_p079",
    "r12_p079",
)
SUPERVISION_RECOVERY_TREE_FAMILY = "tree_neural"


def _markov_tradeoff_tree_bundle_contract(
    args: argparse.Namespace,
    *,
    phases: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return markov_tree_bundle_metadata(
        leaf_policy={
            "partition_axis": "synthetic_markov_document",
            "phases": sorted(str(phase) for phase in (phases or ())),
            "preset": str(getattr(args, "preset", "")),
            "fixed_leaf_tokens": int(getattr(args, "fixed_leaf_tokens", 0) or 0),
        },
        state_dim=int(getattr(args, "state_dim", 0) or 0) or None,
        f_init="official_oracle",
        g_init="raw_concat",
        schedule=str(getattr(args, "tree_training_schedule", "balanced") or "balanced"),
        metadata={"runner": "run_markov_optimization_tradeoff_pipeline"},
    )


def _manifest_local_law_weight(
    args: argparse.Namespace,
    *,
    attr: str,
) -> float | None:
    raw = getattr(args, attr, None)
    return float(raw) if raw is not None else None


def _markov_tradeoff_run_manifest(
    *,
    args: argparse.Namespace,
    output_root: Path,
    phases: Sequence[str] | set[str] | None,
    tree_bundle_contract: Mapping[str, Any],
    sources: Mapping[str, Any] | None = None,
    status: str = "completed",
    publication_ready: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    artifacts = [
        {"kind": str(kind), "uri": str(uri)}
        for kind, uri in sorted(dict(sources or {}).items())
        if str(uri or "")
    ]
    artifacts.append({"kind": "pipeline_summary", "uri": str(output_root / "pipeline_summary.json")})
    artifacts.append({"kind": "run_directory", "uri": str(output_root)})
    local_law_weight = _manifest_local_law_weight(
        args,
        attr="local_law_weight",
    )
    root_weight = (
        max(0.0, 1.0 - float(local_law_weight))
        if local_law_weight is not None
        else (
            float(getattr(args, "root_share"))
            if getattr(args, "root_share", None) is not None
            else float(getattr(args, "tree_task_objective_weight"))
            if getattr(args, "tree_task_objective_weight", None) is not None
            else 1.0
        )
    )
    return run_manifest_metadata(
        run_id="markov.tradeoff_pipeline",
        domain="markov",
        role="tradeoff_pipeline",
        backend="fno",
        status=str(status),
        tree_bundle=tree_bundle_contract,
        f_init="official_oracle",
        g_init="raw_concat",
        f_lineage={"init": "official_oracle", "artifact": "synthetic_oracle"},
        g_lineage={"init": "raw_concat", "artifact": "raw_concat"},
        schedule=str(getattr(args, "tree_training_schedule", "balanced") or "balanced"),
        objective=objective_metadata(
            objective_family="markov_tradeoff_pipeline",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
            local_law_weight=local_law_weight,
            root_share=root_weight,
            local_law_component_weights={
                "leaf_preservation": float(local_law_weight or 0.0),
                "merge_preservation": float(local_law_weight or 0.0),
                "on_range_idempotence": float(local_law_weight or 0.0),
            },
            metadata={
                "preset": str(getattr(args, "preset", "")),
                "phases": sorted(str(phase) for phase in (phases or ())),
            },
        ),
        optimizer_config={
            "phases": sorted(str(phase) for phase in (phases or ())),
            "preset": str(getattr(args, "preset", "")),
            "train_docs": int(getattr(args, "train_docs", 0) or 0),
        },
        output_artifacts=artifacts,
        audit_results={"ok": True, "policy": "manifest_contract_required"},
        quarantine={"classification": "valid_treebundle_v1"},
        command=sys.argv,
        allow_legacy=False,
        publication_ready=bool(publication_ready),
        metadata={
            "runner": "scripts/run_markov_optimization_tradeoff_pipeline.py",
            **dict(metadata or {}),
        },
    )


SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK = "recoverable_v5"
SUPERVISION_RECOVERY_STRUCTURAL_GRID = "structural_core_v2"
SUPERVISION_RECOVERY_STRUCTURAL_CELL = "r12_p079"
SUPERVISION_RECOVERY_PACKAGE_ALIASES = _SUPERVISION_RECOVERY_PACKAGE_ALIASES
SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES = _SUPERVISION_RECOVERY_PACKAGE_GROUP_ALIASES
LAW_PACKAGE_ALIASES = _LAW_PACKAGE_ALIASES
# Re-export from tree_reference_presets for shared config/template use.
SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET = _PRESET_COMMON
COMPARISON_GRID_V3_PRESET = _PRESET_COMPARISON_GRID_V3
UNIFIED_G_FULL_LOCAL_LAWS_PRESET = _PRESET_UNIFIED_G
UNIFIED_G_FNO_PARITY_CANARY_PRESET = _PRESET_FNO_CANARY
SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC = "val_root_mae"
SUPERVISION_RECOVERY_CANONICAL_TREE_STAGE1_SELECTION_METRIC = "val_theorem_bootstrap_direct"
SUPERVISION_RECOVERY_CANONICAL_COMPARISON_RULE = (
    "all tree ladder points selected on val_root_mae; local metrics are diagnostics"
)
SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTICS = (
    "leaf_direct_exact_match",
    "merge_direct_exact_match",
    "merge_join_bit_accuracy",
    "leaf_first_accuracy",
    "leaf_last_accuracy",
    "merge_first_accuracy",
    "merge_last_accuracy",
    "c2_on_range_exact_match",
    "phi_merge_alignment",
    "phi_pair_auc",
    "root_direct_count_mae",
    "exact_projected_root_mae",
    "learned_merger_gap",
)
SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTIC_ALIASES = {
    "leaf_direct_exact_match": (
        "leaf_direct_exact_summary_match_rate",
        "test_leaf_direct_exact_summary_match_rate",
    ),
    "merge_direct_exact_match": (
        "merge_direct_exact_summary_match_rate",
        "test_merge_direct_exact_summary_match_rate",
    ),
}
SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM = "primary"
SUPERVISION_RECOVERY_EXACT_COLLAPSE_ONE_TREE_COMPARISON_ARM = (
    "exact_collapse_one_tree_identity"
)
SUPERVISION_RECOVERY_EXACT_COLLAPSE_PACKAGE = "full100"
EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE = "official_fno_one_tree_identity"
SUPERVISION_RECOVERY_V3_PACKAGE_ORDER = (
    "full100",
    "r100_superset_local_eq_10p0",
    "r100_superset_local_eq_15p0",
    "r100_superset_local_eq_20p0",
)
SUPERVISION_RECOVERY_V3_LEAF_TOKEN_LADDER = (32, 16, 8)
SUPERVISION_RECOVERY_V3_DEPTH_DISCOUNT_GAMMAS = (1.0, 0.9)
SUPERVISION_RECOVERY_V3_RECOVERABLE_BENCHMARK = "recoverable_v5_t128"
SUPERVISION_RECOVERY_V3_STRUCTURAL_GRID = "structural_core_v2_t128"
SUPERVISION_RECOVERY_PACKAGE_ORDER = (
    "full100",
    "r100_node_mass_eq_10p0",
    "r100_node_mass_eq_20p0",
    "r100_node_mass_eq_30p0",
    "r100_node_mass_eq_40p0",
    "r100_node_mass_eq_50p0",
    "r100_node_mass_eq_60p0",
    "r100_node_mass_eq_70p0",
    "r100_node_mass_eq_80p0",
    "r100_node_mass_eq_90p0",
    "r100_node_mass_eq_100p0",
    "full50",
    "full30",
    "full20",
    "full10",
    "full10_leaf_count100",
    "full10_leaf_full100",
    "full10_leaf_full100_internal_depth1_count100",
    "full10_leaf_full100_internal_depth2_count100",
    "full10_leaf_full100_internal_count100",
    "full20_leaf_full100_internal_count100",
    "full30_leaf_full100_internal_count100",
    "full40_leaf_full100_internal_count100",
    "full50_leaf_full100_internal_count100",
)
SUPERVISION_RECOVERY_ROOT_SHARE_DECADES = (
    100,
    90,
    80,
    70,
    60,
    50,
    40,
    30,
    20,
    10,
)
SUPERVISION_RECOVERY_DEPTH_PROFILE_ROOT_SHARES = (
    90,
    80,
    70,
    60,
    50,
    40,
    30,
    20,
    10,
    0,
)
SUPERVISION_RECOVERY_ROOT_LADDER_PACKAGE_ORDER = tuple(
    f"full{int(root_share)}" for root_share in SUPERVISION_RECOVERY_ROOT_SHARE_DECADES
)
SUPERVISION_RECOVERY_LEAF_ONLY_MASS_PRESERVING_PACKAGE_ORDER = (
    "full100",
    *(
        _leaf_mass_preserving_package_name(int(root_share))
        for root_share in SUPERVISION_RECOVERY_DEPTH_PROFILE_ROOT_SHARES
    ),
)
SUPERVISION_RECOVERY_DEPTH_EQUAL_MASS_PRESERVING_PACKAGE_ORDER = (
    "full100",
    *(
        _depth_equal_mass_preserving_package_name(int(root_share))
        for root_share in SUPERVISION_RECOVERY_DEPTH_PROFILE_ROOT_SHARES
    ),
)
SUPERVISION_RECOVERY_R10_LOCAL_LAW_RATE_PACKAGE_ORDER = (
    "full10",
    "full10_leaf_count10_internal_count10",
    "full10_leaf_count20_internal_count20",
    "full10_leaf_count50_internal_count50",
    "full10_leaf_count100_internal_count100",
)
SUPERVISION_RECOVERY_R20_LOCAL_LAW_RATE_PACKAGE_ORDER = (
    "full20",
    "full20_leaf_count10_internal_count10",
    "full20_leaf_count20_internal_count20",
    "full20_leaf_count50_internal_count50",
    "full20_leaf_count100_internal_count100",
)
SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS: Dict[int, Sequence[float]] = {
    10: (0.5, 1.0, 1.5, 2.0),
    20: (1.0, 2.0, 3.0, 4.0),
    80: (5.0, 10.0, 15.0, 16.0),
    90: (5.0, 10.0, 15.0, 18.0),
    100: (5.0, 10.0, 15.0, 20.0),
}
SUPERVISION_RECOVERY_REDISTRIBUTION_LOCAL_MASS_TARGETS: tuple[float, ...] = (
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
)
SUPERVISION_RECOVERY_R10_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    10,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[10],
)
SUPERVISION_RECOVERY_R20_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    20,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[20],
)
SUPERVISION_RECOVERY_R80_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    80,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[80],
)
SUPERVISION_RECOVERY_R90_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    90,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[90],
)
SUPERVISION_RECOVERY_R100_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    100,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[100],
)
SUPERVISION_RECOVERY_R100_REDISTRIBUTION_PACKAGE_ORDER = (
    "full100",
    *(
        _node_mass_target_package_name(float(local_mass_percent))
        for local_mass_percent in SUPERVISION_RECOVERY_REDISTRIBUTION_LOCAL_MASS_TARGETS
    ),
)
ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET = _PRESET_ROOT_REPLAY
ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET = _PRESET_ROOT_OPT_FIX
ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET = _PRESET_ROOT_CAP_FIX
ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET = _PRESET_ROOT_MATCHED
STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET = _PRESET_STRUCTURAL_MATCHED
SUPERVISION_RECOVERY_PACKAGE_SPECS: Dict[str, Dict[str, Any]] = {
    "full100": {
        "label": "100% full-doc only",
        "description": "Both FNO and tree train on dense full-doc supervision.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full100",
    },
    "full90": {
        "label": "90% full-doc only",
        "description": "Both FNO and tree train on a 90% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.9,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full90",
    },
    "full80": {
        "label": "80% full-doc only",
        "description": "Both FNO and tree train on a 80% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.8,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full80",
    },
    "full70": {
        "label": "70% full-doc only",
        "description": "Both FNO and tree train on a 70% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.7,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full70",
    },
    "full60": {
        "label": "60% full-doc only",
        "description": "Both FNO and tree train on a 60% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.6,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full60",
    },
    "full50": {
        "label": "50% full-doc only",
        "description": "Both FNO and tree train on a 50% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.5,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full50",
    },
    "full40": {
        "label": "40% full-doc only",
        "description": "Both FNO and tree train on a 40% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.4,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full40",
    },
    "full30": {
        "label": "30% full-doc only",
        "description": "Both FNO and tree train on a 30% reviewed root-supervision subset.",
        "budget_total_calls_per_doc": 0.3,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full30",
    },
    "full20": {
        "label": "20% full-doc only",
        "description": "Both FNO and tree train on a 20% reviewed root-supervision subset.",
        "mass_target_per_doc": 0.2,
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full20",
    },
    "full10": {
        "label": "10% full-doc only",
        "description": "Both FNO and tree train on the same sparse reviewed subset.",
        "mass_target_per_doc": 0.1,
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": True,
        "fno_reference_package": "full10",
    },
    "full0_leaf_full100_internal_count100": {
        "label": "0% full-doc + leaf full + all internal count",
        "description": "Tree gets no root supervision, but still gets full leaf full-sketch labels and full internal count labels.",
        "budget_total_calls_per_doc": 0.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "",
    },
    "full10_leaf_count100": {
        "label": "10% full-doc + leaf count",
        "description": "Tree gets 10% full-doc supervision plus full leaf count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_count10_internal_count10": {
        "label": "10% full-doc + 10% leaf/internal count",
        "description": "Tree gets 10% full-doc supervision plus matched 10% leaf count and 10% internal count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.1,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.1,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_count20_internal_count20": {
        "label": "10% full-doc + 20% leaf/internal count",
        "description": "Tree gets 10% full-doc supervision plus matched 20% leaf count and 20% internal count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.2,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.2,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_count50_internal_count50": {
        "label": "10% full-doc + 50% leaf/internal count",
        "description": "Tree gets 10% full-doc supervision plus matched 50% leaf count and 50% internal count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.5,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.5,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_count100_internal_count100": {
        "label": "10% full-doc + 100% leaf/internal count",
        "description": "Tree gets 10% full-doc supervision plus matched full leaf count and full internal count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full20_leaf_count10_internal_count10": {
        "label": "20% full-doc + 10% leaf/internal count",
        "description": "Tree gets 20% full-doc supervision plus matched 10% leaf count and 10% internal count labels.",
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.1,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.1,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "full20_leaf_count20_internal_count20": {
        "label": "20% full-doc + 20% leaf/internal count",
        "description": "Tree gets 20% full-doc supervision plus matched 20% leaf count and 20% internal count labels.",
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.2,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.2,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "full20_leaf_count50_internal_count50": {
        "label": "20% full-doc + 50% leaf/internal count",
        "description": "Tree gets 20% full-doc supervision plus matched 50% leaf count and 50% internal count labels.",
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.5,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.5,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "full20_leaf_count100_internal_count100": {
        "label": "20% full-doc + 100% leaf/internal count",
        "description": "Tree gets 20% full-doc supervision plus matched full leaf count and full internal count labels.",
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "r10_mass_local_eq_0p5": {
        "label": "R10 mass-matched + 0.5% leaf/internal count",
        "description": "Tree matches the R10 full-doc-equivalent mass target with equal 0.5% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.1,
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.005,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.005,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "r10_mass_local_eq_1p0": {
        "label": "R10 mass-matched + 1.0% leaf/internal count",
        "description": "Tree matches the R10 full-doc-equivalent mass target with equal 1.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.1,
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.01,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.01,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "r10_mass_local_eq_1p5": {
        "label": "R10 mass-matched + 1.5% leaf/internal count",
        "description": "Tree matches the R10 full-doc-equivalent mass target with equal 1.5% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.1,
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.015,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.015,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "r10_mass_local_eq_2p0": {
        "label": "R10 mass-matched + 2.0% leaf/internal count",
        "description": "Tree matches the R10 full-doc-equivalent mass target with equal 2.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.1,
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.02,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.02,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "r20_mass_local_eq_1p0": {
        "label": "R20 mass-matched + 1.0% leaf/internal count",
        "description": "Tree matches the R20 full-doc-equivalent mass target with equal 1.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.2,
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.01,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.01,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "r20_mass_local_eq_2p0": {
        "label": "R20 mass-matched + 2.0% leaf/internal count",
        "description": "Tree matches the R20 full-doc-equivalent mass target with equal 2.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.2,
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.02,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.02,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "r20_mass_local_eq_3p0": {
        "label": "R20 mass-matched + 3.0% leaf/internal count",
        "description": "Tree matches the R20 full-doc-equivalent mass target with equal 3.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.2,
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.03,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.03,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "r20_mass_local_eq_4p0": {
        "label": "R20 mass-matched + 4.0% leaf/internal count",
        "description": "Tree matches the R20 full-doc-equivalent mass target with equal 4.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 0.2,
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.04,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.04,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "r100_mass_local_eq_5p0": {
        "label": "R100 mass-matched + 5.0% leaf/internal count",
        "description": "Tree matches the R100 full-doc-equivalent mass target with equal 5.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 1.0,
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.05,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.05,
        "run_fno": False,
        "fno_reference_package": "full100",
    },
    "r100_mass_local_eq_10p0": {
        "label": "R100 mass-matched + 10.0% leaf/internal count",
        "description": "Tree matches the R100 full-doc-equivalent mass target with equal 10.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 1.0,
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.10,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.10,
        "run_fno": False,
        "fno_reference_package": "full100",
    },
    "r100_mass_local_eq_15p0": {
        "label": "R100 mass-matched + 15.0% leaf/internal count",
        "description": "Tree matches the R100 full-doc-equivalent mass target with equal 15.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 1.0,
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.15,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.15,
        "run_fno": False,
        "fno_reference_package": "full100",
    },
    "r100_mass_local_eq_20p0": {
        "label": "R100 mass-matched + 20.0% leaf/internal count",
        "description": "Tree matches the R100 full-doc-equivalent mass target with equal 20.0% leaf/internal count labels and only the residual root supervision budget.",
        "mass_target_per_doc": 1.0,
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.20,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.20,
        "run_fno": False,
        "fno_reference_package": "full100",
    },
    "r100_superset_local_eq_10p0": {
        "label": "R100 superset + 10.0% leaf/internal count",
        "description": "Tree keeps the full R100 root supervision and adds equal 10.0% leaf/internal count labels on top.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.10,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.10,
        "run_fno": False,
        "fno_reference_package": "full100",
        "package_semantics": "superset",
    },
    "r100_superset_leaf05_internal10p0": {
        "label": "R100 superset + 5.0% leaf / 10.0% internal count",
        "description": "Tree keeps the full R100 root supervision, halves the leaf label rate to 5.0%, and keeps the internal label rate at 10.0%.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.05,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.10,
        "run_fno": False,
        "fno_reference_package": "full100",
        "package_semantics": "superset",
    },
    "r100_superset_local_eq_15p0": {
        "label": "R100 superset + 15.0% leaf/internal count",
        "description": "Tree keeps the full R100 root supervision and adds equal 15.0% leaf/internal count labels on top.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.15,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.15,
        "run_fno": False,
        "fno_reference_package": "full100",
        "package_semantics": "superset",
    },
    "r100_superset_local_eq_20p0": {
        "label": "R100 superset + 20.0% leaf/internal count",
        "description": "Tree keeps the full R100 root supervision and adds equal 20.0% leaf/internal count labels on top.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.20,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.20,
        "run_fno": False,
        "fno_reference_package": "full100",
        "package_semantics": "superset",
    },
    "full10_leaf_full100": {
        "label": "10% full-doc + leaf full",
        "description": "Tree gets 10% full-doc supervision plus full leaf full-sketch labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_full10": {
        "label": "10% full-doc + 10% leaf full-sketch (sparse leaves)",
        "description": "Sparse-everywhere composition test: 10% root + 10% leaf full-sketch labels, no internals.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 0.1,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_full10_internal_count10": {
        "label": "10% full-doc + 10% leaf full + 10% internal count (sparse everywhere)",
        "description": "Sparse-everywhere composition test: 10% root + 10% leaf full-sketch + 10% internal count.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 0.1,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 0.1,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_full10_internal_count100": {
        "label": "10% full-doc + 10% leaf full + 100% internal count",
        "description": "Sparse leaves, full internals: 10% root + 10% leaf full-sketch + 100% internal count.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 0.1,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_full100_internal_depth1_count100": {
        "label": "10% full-doc + leaf full + depth-1 internal count",
        "description": "Tree gets 10% full-doc, full leaf full-sketch, and depth-1 internal count labels only (first pairwise merge level).",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "max_internal_depth": 1,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_full100_internal_depth2_count100": {
        "label": "10% full-doc + leaf full + depth-1+2 internal count",
        "description": "Tree gets 10% full-doc, full leaf full-sketch, and depth-1 and depth-2 internal count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "max_internal_depth": 2,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full10_leaf_full100_internal_count100": {
        "label": "10% full-doc + leaf full + all internal count",
        "description": "Tree gets 10% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.1,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full10",
    },
    "full20_leaf_full100_internal_count100": {
        "label": "20% full-doc + leaf full + all internal count",
        "description": "Tree gets 20% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.2,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full20",
    },
    "full30_leaf_full100_internal_count100": {
        "label": "30% full-doc + leaf full + all internal count",
        "description": "Tree gets 30% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.3,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full30",
    },
    "full40_leaf_full100_internal_count100": {
        "label": "40% full-doc + leaf full + all internal count",
        "description": "Tree gets 40% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.4,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full40",
    },
    "full50_leaf_full100_internal_count100": {
        "label": "50% full-doc + leaf full + all internal count",
        "description": "Tree gets 50% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.5,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full50",
    },
    "full60_leaf_full100_internal_count100": {
        "label": "60% full-doc + leaf full + all internal count",
        "description": "Tree gets 60% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.6,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full60",
    },
    "full70_leaf_full100_internal_count100": {
        "label": "70% full-doc + leaf full + all internal count",
        "description": "Tree gets 70% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.7,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full70",
    },
    "full80_leaf_full100_internal_count100": {
        "label": "80% full-doc + leaf full + all internal count",
        "description": "Tree gets 80% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.8,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full80",
    },
    "full90_leaf_full100_internal_count100": {
        "label": "90% full-doc + leaf full + all internal count",
        "description": "Tree gets 90% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 0.9,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full90",
    },
    "full100_leaf_full100_internal_count100": {
        "label": "100% full-doc + leaf full + all internal count",
        "description": "Tree gets 100% full-doc supervision, full leaf full-sketch labels, and full internal count labels.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full100",
    },
    "full100_leaf_full100_internal_full100": {
        "label": "100% full-doc + leaf full + internal full",
        "description": "Tree gets 100% full-doc supervision, full leaf full-sketch labels, and full internal full-sketch labels.",
        "budget_total_calls_per_doc": 1.0,
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "balanced",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "full_sketch",
        "internal_label_rate": 1.0,
        "run_fno": False,
        "fno_reference_package": "full100",
    },
}
SUPERVISION_RECOVERY_PACKAGE_SPECS.update(
    _build_mass_matched_package_specs(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS)
)
SUPERVISION_RECOVERY_PACKAGE_SPECS.update(
    _build_node_mass_target_package_specs(
        SUPERVISION_RECOVERY_REDISTRIBUTION_LOCAL_MASS_TARGETS
    )
)
SUPERVISION_RECOVERY_PACKAGE_SPECS.update(
    _build_depth_profile_mass_preserving_package_specs(
        root_shares=SUPERVISION_RECOVERY_DEPTH_PROFILE_ROOT_SHARES,
        local_split_mode="leaf_only",
    )
)
SUPERVISION_RECOVERY_PACKAGE_SPECS.update(
    _build_depth_profile_mass_preserving_package_specs(
        root_shares=SUPERVISION_RECOVERY_DEPTH_PROFILE_ROOT_SHARES,
        local_split_mode="depth_equal_nonroot",
    )
)
for _package_name, _package_spec in SUPERVISION_RECOVERY_PACKAGE_SPECS.items():
    _package_spec.setdefault(
        "package_semantics",
        _default_supervision_recovery_package_semantics(
            str(_package_name),
            _package_spec,
        ),
    )


def _resolve_supervision_recovery_package_order(
    package_names: Sequence[str] | None = None,
) -> List[str]:
    if package_names is None:
        return list(SUPERVISION_RECOVERY_PACKAGE_ORDER)
    resolved = _resolve_supervision_recovery_package_names(
        package_names,
        valid_names=tuple(SUPERVISION_RECOVERY_PACKAGE_SPECS.keys()),
    )
    return resolved or list(SUPERVISION_RECOVERY_PACKAGE_ORDER)


def _supervision_recovery_package_order_from_args(
    args: argparse.Namespace,
    default_package_order: Sequence[str] | None = None,
) -> List[str]:
    raw_value = getattr(args, "supervision_recovery_packages", None)
    fallback = default_package_order or SUPERVISION_RECOVERY_PACKAGE_ORDER
    if raw_value is None:
        return _resolve_supervision_recovery_package_order(fallback)
    return _resolve_supervision_recovery_package_order(
        _parse_str_list(raw_value, fallback)
    )


def _resolved_supervision_recovery_leaf_token_batch_sizes(
    args: argparse.Namespace,
) -> Dict[int, int]:
    raw_value = getattr(
        args, "supervision_recovery_leaf_token_batch_sizes", None
    )
    text = str(raw_value or "").strip()
    if not text:
        return {}
    payload: Any = None
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    items: List[Tuple[Any, Any]] = []
    if isinstance(payload, Mapping):
        items = list(payload.items())
    else:
        for chunk in [item.strip() for item in text.split(";") if item.strip()]:
            if "=" not in chunk:
                raise ValueError(
                    "invalid --supervision-recovery-leaf-token-batch-sizes "
                    f"entry {chunk!r}; expected leaf_tokens=batch_size"
                )
            key_text, value_text = chunk.split("=", 1)
            items.append((key_text.strip(), value_text.strip()))
    resolved: Dict[int, int] = {}
    for key, value in items:
        try:
            key_int = int(str(key).strip())
            value_int = int(str(value).strip())
        except Exception as exc:
            raise ValueError(
                "invalid --supervision-recovery-leaf-token-batch-sizes entry "
                f"{key!r}={value!r}; expected integers"
            ) from exc
        if key_int <= 0 or value_int <= 0:
            continue
        resolved[key_int] = value_int
    return resolved


def _resolve_supervision_batch_size_for_leaf_tokens(
    args: argparse.Namespace,
    fixed_leaf_tokens: int,
) -> int:
    overrides = _resolved_supervision_recovery_leaf_token_batch_sizes(args)
    if overrides:
        bs = overrides.get(int(fixed_leaf_tokens))
        if bs is not None and int(bs) > 0:
            return int(bs)
    return int(args.supervision_batch_size)


def _supervision_recovery_leaf_token_ladder_from_args(
    args: argparse.Namespace,
    default_values: Sequence[int] | None = None,
) -> List[int]:
    raw_value = getattr(args, "supervision_recovery_leaf_token_ladder", None)
    parsed_defaults = [int(value) for value in list(default_values or ()) if int(value) > 0]
    values = (
        parsed_defaults
        if raw_value is None
        else [
            int(value)
            for value in _parse_int_list(str(raw_value), parsed_defaults)
            if int(value) > 0
        ]
    )
    seen: set[int] = set()
    resolved: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        resolved.append(int(value))
    return resolved


def _normalize_supervision_recovery_leaf_token_values(
    values: Any,
) -> List[int]:
    if values is None:
        return []
    if isinstance(values, str):
        parsed_values = _parse_int_list(values, [])
    elif isinstance(values, (list, tuple, set)):
        parsed_values = [
            int(value)
            for value in values
            if _safe_int(value, 0) > 0
        ]
    else:
        parsed_values = [int(_safe_int(values, 0))]
    seen: set[int] = set()
    resolved: List[int] = []
    for value in parsed_values:
        if int(value) <= 0 or int(value) in seen:
            continue
        seen.add(int(value))
        resolved.append(int(value))
    return resolved


def _parse_supervision_recovery_package_leaf_token_override_text(
    raw_value: str,
) -> Dict[str, Any]:
    text = str(raw_value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    overrides: Dict[str, Any] = {}
    for chunk in [item.strip() for item in text.split(";") if item.strip()]:
        if "=" not in chunk:
            raise ValueError(
                "invalid supervision_recovery_package_leaf_token_overrides entry "
                f"{chunk!r}; expected package=tokens"
            )
        package_key, token_text = chunk.split("=", 1)
        overrides[str(package_key).strip()] = str(token_text).strip()
    return overrides


def _supervision_recovery_package_leaf_token_overrides_from_args(
    args: argparse.Namespace,
    default_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, List[int]]:
    raw_value = getattr(
        args,
        "supervision_recovery_package_leaf_token_overrides",
        None,
    )
    source: Any = default_overrides if raw_value is None else raw_value
    if source is None:
        return {}
    if isinstance(source, str) and not str(source).strip():
        return {}
    if isinstance(source, Mapping) and not dict(source):
        return {}
    if isinstance(source, Mapping):
        raw_mapping = {str(key): value for key, value in dict(source).items()}
    elif isinstance(source, str):
        raw_mapping = _parse_supervision_recovery_package_leaf_token_override_text(
            source
        )
    else:
        raise ValueError(
            "supervision_recovery_package_leaf_token_overrides must be a mapping "
            "or a semicolon-separated package=tokens string"
        )
    overrides: Dict[str, List[int]] = {}
    valid_names = tuple(SUPERVISION_RECOVERY_PACKAGE_SPECS.keys())
    for raw_key, raw_tokens in raw_mapping.items():
        resolved_packages = _resolve_supervision_recovery_package_names(
            [str(raw_key)],
            valid_names=valid_names,
        )
        token_values = _normalize_supervision_recovery_leaf_token_values(raw_tokens)
        if not token_values:
            continue
        for package_name in resolved_packages:
            overrides[str(package_name)] = list(token_values)
    return overrides


def _resolved_supervision_recovery_package_order(
    args: argparse.Namespace,
) -> List[str]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    return _supervision_recovery_package_order_from_args(
        args,
        preset.get("supervision_recovery_packages"),
    )


def _resolved_supervision_recovery_leaf_token_ladder(
    args: argparse.Namespace,
) -> List[int]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    return _supervision_recovery_leaf_token_ladder_from_args(
        args,
        preset.get("supervision_recovery_leaf_token_ladder"),
    )


def _resolved_supervision_recovery_package_leaf_token_overrides(
    args: argparse.Namespace,
) -> Dict[str, List[int]]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    return _supervision_recovery_package_leaf_token_overrides_from_args(
        args,
        preset.get("supervision_recovery_package_leaf_token_overrides"),
    )


def _supervision_recovery_leafgrid_active(args: argparse.Namespace) -> bool:
    return bool(
        _resolved_supervision_recovery_leaf_token_ladder(args)
        or _resolved_supervision_recovery_package_leaf_token_overrides(args)
    )


def _supervision_recovery_comparison_arm(
    payload: Mapping[str, Any] | None,
) -> str:
    raw_payload = dict(payload or {})
    arm = str(
        raw_payload.get(
            "pipeline_supervision_recovery_comparison_arm",
            raw_payload.get(
                "comparison_arm",
                SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM,
            ),
        )
        or SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM
    ).strip()
    return arm or SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM


def _supervision_recovery_is_one_leaf_geometry(
    payload: Mapping[str, Any] | None,
) -> bool:
    raw_payload = dict(payload or {})
    fixed_leaf_tokens = int(_safe_int(raw_payload.get("fixed_leaf_tokens"), 0))
    assumed_doc_tokens = int(
        _safe_int(raw_payload.get("computed_assumed_doc_tokens"), 0)
    )
    assumed_leaves = int(_safe_int(raw_payload.get("computed_assumed_leaves"), 0))
    if assumed_leaves > 0:
        return assumed_leaves == 1
    return (
        fixed_leaf_tokens > 0
        and assumed_doc_tokens > 0
        and fixed_leaf_tokens >= assumed_doc_tokens
    )


def _supervision_recovery_is_root_only_full_package(
    package_name: str,
    package_spec: Mapping[str, Any] | None,
) -> bool:
    spec = dict(package_spec or {})
    if not _ns(package_name).startswith("full"):
        return False
    if (
        str(spec.get("doc_consumption_mode", "root_only") or "").strip().lower()
        != "root_only"
    ):
        return False
    if abs(_safe_float(spec.get("full_doc_budget_share"), 1.0) - 1.0) > 1e-12:
        return False
    if abs(_safe_float(spec.get("leaf_label_rate"), 0.0)) > 1e-12:
        return False
    if abs(_safe_float(spec.get("internal_label_rate"), 0.0)) > 1e-12:
        return False
    return True


def _one_leaf_package_has_local_supervision(
    package_spec: Mapping[str, Any] | None,
) -> bool:
    spec = dict(package_spec or {})
    if abs(_safe_float(spec.get("leaf_label_rate"), 0.0)) > 1e-12:
        return True
    if abs(_safe_float(spec.get("internal_label_rate"), 0.0)) > 1e-12:
        return True
    if abs(_safe_float(spec.get("local_mass_target_per_doc"), 0.0)) > 1e-12:
        return True
    return False


def _supervision_recovery_requires_exact_full_doc_parity(
    *,
    package_name: str,
    package_spec: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
) -> bool:
    return bool(
        bool(
            payload
            and bool(
                payload.get(
                    "pipeline_supervision_recovery_exact_full_doc_parity_requested",
                    False,
                )
            )
        )
        and _supervision_recovery_is_one_leaf_geometry(payload)
        and _supervision_recovery_is_root_only_full_package(package_name, package_spec)
    )


def _supervision_recovery_requires_matched_root_surface_lock(
    *,
    package_name: str,
    package_spec: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
) -> bool:
    payload_map = dict(payload or {})
    if not _supervision_recovery_is_one_leaf_geometry(payload_map):
        return False
    if not _supervision_recovery_is_root_only_full_package(package_name, package_spec):
        return False
    if str(package_name or "").strip() == "full100":
        return False
    return _gs(payload_map, "pipeline_tree_reference_label") in {
        ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
        STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    }


def _supervision_recovery_prefers_root_checkpoint(
    payload: Mapping[str, Any],
) -> bool:
    doc_mode = _gs(payload, "doc_consumption_mode").lower()
    if doc_mode != "root_only":
        return False
    full_doc_share = _safe_float(payload.get("full_doc_budget_share"), default=0.0)
    coverage = _safe_float(payload.get("budget_total_calls_per_doc"), default=0.0)
    if full_doc_share < 1.0 - 1e-12:
        return False
    if coverage < 1.0 - 1e-12:
        return False
    return True


def _supervision_recovery_tree_checkpoint_metric(
    package_spec: Mapping[str, Any],
    *,
    default_metric: str,
    tree_reference_label: str = "",
) -> str:
    if (
        str(tree_reference_label or "").strip()
        in {
            SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
            UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
            UNIFIED_G_FNO_PARITY_CANARY_PRESET,
        }
    ):
        return SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC
    if _supervision_recovery_prefers_root_checkpoint(package_spec):
        return "val_root_mae"
    return str(default_metric)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _supervision_recovery_geometry_summary(
    *,
    min_tokens: int,
    max_tokens: int,
    fixed_leaf_tokens: int,
    max_internal_depth: int,
) -> Dict[str, Any]:
    assumed_tokens = (
        int(min_tokens)
        if min_tokens > 0 and min_tokens == max_tokens
        else int(round(0.5 * float(min_tokens + max_tokens)))
    )
    internal_mass_full = float("nan")
    leaf_mass_full = float("nan")
    n_leaves = 0
    n_internal_nodes = 0
    if assumed_tokens > 0 and fixed_leaf_tokens > 0:
        from src.ctreepo.sim.core.markov_changepoint_ops_count import (
            _doc_leaf_and_internal_spans,
        )

        leaf_spans, internal_spans = _doc_leaf_and_internal_spans(
            n_tokens=int(assumed_tokens),
            leaf_tokens=int(fixed_leaf_tokens),
            max_internal_depth=int(max_internal_depth),
        )
        n_leaves = int(len(leaf_spans))
        n_internal_nodes = int(len(internal_spans))
        leaf_mass_full = float(
            sum(
                float(max(0, int(end) - int(start))) / float(max(1, assumed_tokens))
                for start, end in leaf_spans
            )
        )
        internal_mass_full = float(
            sum(
                float(max(0, int(end) - int(start))) / float(max(1, assumed_tokens))
                for start, end in internal_spans
            )
        )
    return {
        "assumed_doc_tokens": int(assumed_tokens),
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "max_internal_depth": int(max_internal_depth),
        "assumed_leaves": int(n_leaves),
        "assumed_internal_nodes": int(n_internal_nodes),
        "leaf_mass_full_per_doc": float(leaf_mass_full),
        "internal_mass_full_per_doc": float(internal_mass_full),
    }


def _resolve_supervision_recovery_package_for_scope(
    package_name: str,
    package_spec: Mapping[str, Any],
    *,
    min_tokens: int,
    max_tokens: int,
    fixed_leaf_tokens: int,
    scope_key: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    spec = dict(package_spec or {})
    package_semantics = _default_supervision_recovery_package_semantics(
        str(package_name),
        spec,
    )
    spec["package_semantics"] = str(package_semantics)
    local_split_mode = str(spec.get("local_split_mode", "") or "").strip().lower()
    max_internal_depth = int(_safe_int(spec.get("max_internal_depth"), 0))
    geometry = _supervision_recovery_geometry_summary(
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        max_internal_depth=int(max_internal_depth),
    )
    leaf_mass_full = _safe_float(geometry.get("leaf_mass_full_per_doc"), float("nan"))
    internal_mass_full = _safe_float(
        geometry.get("internal_mass_full_per_doc"),
        float("nan"),
    )
    local_mass_target = _safe_float(
        spec.get("local_mass_target_per_doc"),
        float("nan"),
    )
    leaf_kind = _gs(spec, "leaf_supervision_kind")
    internal_kind = _gs(spec, "internal_supervision_kind")
    leaf_rate = _clamp01(float(spec.get("leaf_label_rate", 0.0)))
    internal_rate = (
        _clamp01(float(spec.get("internal_label_rate", 0.0)))
        if internal_kind != "none"
        else 0.0
    )
    if math.isfinite(local_mass_target):
        if local_split_mode == "leaf_only":
            internal_kind = "none"
            internal_rate = 0.0
            internal_mass_full = 0.0
            max_internal_depth = 0
            spec["internal_supervision_kind"] = "none"
            spec["internal_label_rate"] = 0.0
            spec["max_internal_depth"] = 0
            if leaf_kind == "none" or not math.isfinite(leaf_mass_full) or leaf_mass_full <= 1e-12:
                raise ValueError(
                    f"cannot resolve leaf-only mass-preserving package={package_name!r} "
                    f"scope={scope_key!r}: no finite leaf capacity"
                )
            leaf_rate = float(local_mass_target) / float(leaf_mass_full)
            if leaf_rate < -1e-9 or leaf_rate > 1.0 + 1e-9:
                raise ValueError(
                    f"invalid leaf-only redistribution package={package_name!r} "
                    f"scope={scope_key!r}: local_target={float(local_mass_target):.6g}, "
                    f"leaf_capacity={float(leaf_mass_full):.6g}, required_rate={float(leaf_rate):.6g}"
                )
            leaf_rate = _clamp01(leaf_rate)
            spec["leaf_label_rate"] = float(leaf_rate)
        elif local_split_mode == "depth_equal_nonroot":
            nonroot_depths = _nonroot_internal_depth_count(
                n_leaves=int(geometry.get("assumed_leaves", 0))
            )
            if nonroot_depths > 0:
                max_internal_depth = int(nonroot_depths)
                spec["max_internal_depth"] = int(nonroot_depths)
                geometry = _supervision_recovery_geometry_summary(
                    min_tokens=int(min_tokens),
                    max_tokens=int(max_tokens),
                    fixed_leaf_tokens=int(fixed_leaf_tokens),
                    max_internal_depth=int(max_internal_depth),
                )
                leaf_mass_full = _safe_float(
                    geometry.get("leaf_mass_full_per_doc"),
                    float("nan"),
                )
                internal_mass_full = _safe_float(
                    geometry.get("internal_mass_full_per_doc"),
                    float("nan"),
                )
                if (
                    leaf_kind == "none"
                    or internal_kind == "none"
                    or not math.isfinite(leaf_mass_full)
                    or leaf_mass_full <= 1e-12
                    or not math.isfinite(internal_mass_full)
                    or internal_mass_full <= 1e-12
                ):
                    raise ValueError(
                        f"cannot resolve depth-equal redistribution package={package_name!r} "
                        f"scope={scope_key!r}: insufficient local capacity under "
                        f"fixed_leaf_tokens={int(fixed_leaf_tokens)}"
                    )
                level_count = 1 + int(nonroot_depths)
                per_level_mass_target = float(local_mass_target) / float(level_count)
                target_internal_mass = float(per_level_mass_target) * float(nonroot_depths)
                leaf_rate = float(per_level_mass_target) / float(leaf_mass_full)
                internal_rate = float(target_internal_mass) / float(internal_mass_full)
                if (
                    leaf_rate < -1e-9
                    or leaf_rate > 1.0 + 1e-9
                    or internal_rate < -1e-9
                    or internal_rate > 1.0 + 1e-9
                ):
                    raise ValueError(
                        f"invalid depth-equal redistribution package={package_name!r} "
                        f"scope={scope_key!r}: local_target={float(local_mass_target):.6g}, "
                        f"leaf_capacity={float(leaf_mass_full):.6g}, "
                        f"internal_capacity={float(internal_mass_full):.6g}, "
                        f"leaf_rate={float(leaf_rate):.6g}, internal_rate={float(internal_rate):.6g}"
                    )
                leaf_rate = _clamp01(leaf_rate)
                internal_rate = _clamp01(internal_rate)
                spec["leaf_label_rate"] = float(leaf_rate)
                spec["internal_label_rate"] = float(internal_rate)
            else:
                internal_kind = "none"
                internal_rate = 0.0
                internal_mass_full = 0.0
                max_internal_depth = 0
                spec["internal_supervision_kind"] = "none"
                spec["internal_label_rate"] = 0.0
                spec["max_internal_depth"] = 0
                if leaf_kind == "none" or not math.isfinite(leaf_mass_full) or leaf_mass_full <= 1e-12:
                    raise ValueError(
                        f"cannot resolve depth-equal redistribution package={package_name!r} "
                        f"scope={scope_key!r}: no finite leaf capacity"
                    )
                leaf_rate = float(local_mass_target) / float(leaf_mass_full)
                if leaf_rate < -1e-9 or leaf_rate > 1.0 + 1e-9:
                    raise ValueError(
                        f"invalid depth-equal redistribution package={package_name!r} "
                        f"scope={scope_key!r}: local_target={float(local_mass_target):.6g}, "
                        f"leaf_capacity={float(leaf_mass_full):.6g}, required_rate={float(leaf_rate):.6g}"
                    )
                leaf_rate = _clamp01(leaf_rate)
                spec["leaf_label_rate"] = float(leaf_rate)
        else:
            total_local_mass_full = 0.0
            if leaf_kind != "none" and math.isfinite(leaf_mass_full):
                total_local_mass_full += float(leaf_mass_full)
            if internal_kind != "none" and math.isfinite(internal_mass_full):
                total_local_mass_full += float(internal_mass_full)
            if total_local_mass_full <= 1e-12:
                raise ValueError(
                    f"cannot resolve redistribution supervision_recovery package={package_name!r} "
                    f"scope={scope_key!r}: no finite local mass capacity under "
                    f"min_tokens={int(min_tokens)}, max_tokens={int(max_tokens)}, "
                    f"fixed_leaf_tokens={int(fixed_leaf_tokens)}, max_internal_depth={int(max_internal_depth)}"
                )
            shared_rate = float(local_mass_target) / float(total_local_mass_full)
            if shared_rate < -1e-9 or shared_rate > 1.0 + 1e-9:
                raise ValueError(
                    f"invalid redistribution supervision_recovery package={package_name!r} "
                    f"scope={scope_key!r}: local_target={float(local_mass_target):.6g}, "
                    f"local_capacity={float(total_local_mass_full):.6g}, "
                    f"required_rate={float(shared_rate):.6g}"
                )
            shared_rate = _clamp01(shared_rate)
            if leaf_kind != "none":
                leaf_rate = float(shared_rate)
                spec["leaf_label_rate"] = float(shared_rate)
            if internal_kind != "none":
                internal_rate = float(shared_rate)
                spec["internal_label_rate"] = float(shared_rate)
    leaf_mass = (
        float(leaf_rate) * float(leaf_mass_full)
        if math.isfinite(leaf_mass_full)
        else float("nan")
    )
    internal_mass = (
        float(internal_rate) * float(internal_mass_full)
        if math.isfinite(internal_mass_full)
        else float("nan")
    )
    local_mass = (
        float(leaf_mass + internal_mass)
        if math.isfinite(leaf_mass) and math.isfinite(internal_mass)
        else float("nan")
    )
    mass_target = _safe_float(spec.get("mass_target_per_doc"), float("nan"))
    doc_review_mass = (
        _clamp01(float(spec.get("budget_total_calls_per_doc", 0.0)))
        * _clamp01(float(spec.get("full_doc_budget_share", 1.0)))
    )
    if str(package_semantics) == "mass_matched" and math.isfinite(mass_target):
        if not math.isfinite(local_mass):
            raise ValueError(
                f"cannot resolve mass-matched supervision_recovery package={package_name!r} "
                f"scope={scope_key!r}: missing finite local mass under "
                f"min_tokens={int(min_tokens)}, max_tokens={int(max_tokens)}, "
                f"fixed_leaf_tokens={int(fixed_leaf_tokens)}, max_internal_depth={int(max_internal_depth)}"
            )
        doc_review_mass = float(mass_target - local_mass)
        if doc_review_mass < -1e-9 or doc_review_mass > 1.0 + 1e-9:
            raise ValueError(
                f"invalid mass-matched supervision_recovery package={package_name!r} "
                f"scope={scope_key!r}: target_mass={mass_target:.6g}, "
                f"local_mass={local_mass:.6g}, residual_root_mass={doc_review_mass:.6g}, "
                f"min_tokens={int(min_tokens)}, max_tokens={int(max_tokens)}, "
                f"fixed_leaf_tokens={int(fixed_leaf_tokens)}, max_internal_depth={int(max_internal_depth)}"
            )
        doc_review_mass = _clamp01(doc_review_mass)
        spec["budget_total_calls_per_doc"] = float(doc_review_mass)
        spec["full_doc_budget_share"] = 1.0
        spec["doc_consumption_mode"] = "root_only"
    total_mass = (
        float(doc_review_mass + local_mass)
        if math.isfinite(local_mass)
        else float("nan")
    )
    accounting = {
        **geometry,
        "package_semantics": str(package_semantics),
        "mass_target_per_doc": float(mass_target) if math.isfinite(mass_target) else float("nan"),
        "local_mass_target_per_doc": (
            float(local_mass_target) if math.isfinite(local_mass_target) else float("nan")
        ),
        "computed_doc_review_mass_per_doc": float(doc_review_mass),
        "computed_leaf_mass_per_doc": float(leaf_mass),
        "computed_internal_mass_per_doc": float(internal_mass),
        "computed_local_mass_per_doc": float(local_mass),
        "computed_total_mass_per_doc": float(total_mass),
    }
    return spec, accounting


SUPPORTED_SUPPORT_MODES = ("supported", "unsupported")
SUPERVISION_LEAF_PROFILES: Dict[str, Dict[str, Any]] = {
    "none": {"leaf_supervision_kind": "count_only", "leaf_label_rate": 0.0},
    "count_q25": {"leaf_supervision_kind": "count_only", "leaf_label_rate": 0.25},
    "count_q50": {"leaf_supervision_kind": "count_only", "leaf_label_rate": 0.5},
    "count_q100": {"leaf_supervision_kind": "count_only", "leaf_label_rate": 1.0},
    "full_q50": {"leaf_supervision_kind": "full_sketch", "leaf_label_rate": 0.5},
    "full_q100": {"leaf_supervision_kind": "full_sketch", "leaf_label_rate": 1.0},
}
SUPERVISION_INTERNAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "none": {"internal_supervision_kind": "none", "internal_label_rate": 0.0},
    "count_q25": {"internal_supervision_kind": "count_only", "internal_label_rate": 0.25},
    "count_q50": {"internal_supervision_kind": "count_only", "internal_label_rate": 0.5},
    "count_q100": {"internal_supervision_kind": "count_only", "internal_label_rate": 1.0},
    "full_q50": {"internal_supervision_kind": "full_sketch", "internal_label_rate": 0.5},
    "full_q100": {"internal_supervision_kind": "full_sketch", "internal_label_rate": 1.0},
}
SUPERVISION_LEAF_PROFILE_ORDER = list(SUPERVISION_LEAF_PROFILES.keys())
SUPERVISION_INTERNAL_PROFILE_ORDER = list(SUPERVISION_INTERNAL_PROFILES.keys())
REPORT_SOURCE_SPECS: Dict[str, Dict[str, str]] = {
    "batch_timing_summary": {
        "phase": "batch_timing",
        "alias_relpath": "batch_timing/markov_fixed_fused_leaflaws_batchsize_timing_fullpipeline.json",
    },
    "medium_grid_summary": {
        "phase": "medium_grid",
        "alias_relpath": "medium_grid/aggregate_summary.json",
    },
    "docs_epochs_summary": {
        "phase": "docs_epochs",
        "alias_relpath": "docs_epochs/aggregate_summary.json",
    },
    "learnability_summary": {
        "phase": "learnability",
        "alias_relpath": "learnability_report/learnability_summary.json",
    },
    "weight_ablation_summary": {
        "phase": "weight_ablation",
        "alias_relpath": "weight_ablation_runs/weight_ablation_summary.json",
    },
    "law_comparison_json": {
        "phase": "law_packages",
        "alias_relpath": "law_packages/fno_tree_law_comparison.json",
    },
    "fno_upper_bound_summary": {
        "phase": "full_doc_anchor",
        "alias_relpath": "full_doc_anchor/full_doc_fno_upper_bound_summary.json",
    },
    "oracle_budget_frontier_summary": {
        "phase": "oracle_budget_frontier",
        "alias_relpath": "oracle_budget_frontier/tree_oracle_budget_frontier_summary.json",
    },
    "efficiency_suite_summary": {
        "phase": "efficiency_suite",
        "alias_relpath": "efficiency_suite/summary.json",
    },
    "large_batch_diagnosis_summary": {
        "phase": "large_batch_diagnosis",
        "alias_relpath": "large_batch_diagnosis/aggregate_summary.json",
    },
    "supervision_sweep_summary": {
        "phase": "supervision_sweep",
        "alias_relpath": "supervision_sweep/supervision_sweep_summary.json",
    },
    "support_summary": {
        "phase": "support_grid",
        "alias_relpath": "support_grid/markov_local_support_detailed.summary.json",
    },
    "supervision_recovery_summary": {
        "phase": "supervision_recovery",
        "alias_relpath": "supervision_recovery/summary.json",
    },
}

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

WEIGHT_PROFILE_SPECS: Dict[str, tuple[float, float, float]] = {
    "root_only": (0.0, 0.0, 0.0),
    "pure_c2": (0.0, 1.0, 0.0),
    "c2_trace_c1c3": (0.05, 1.0, 0.05),
    "c2_light_c1c3": (0.1, 1.0, 0.1),
    "c2_mild_c1c3": (0.25, 1.0, 0.25),
    "c2_moderate_c1c3": (0.5, 1.0, 0.5),
    "c2_very_dominant": (1.0, 8.0, 1.0),
    "c2_dominant": (1.0, 4.0, 1.0),
    "c2_heavy": (1.0, 2.0, 1.0),
    "equal": (1.0, 1.0, 1.0),
    "c1c3_heavy": (2.0, 1.0, 2.0),
    "c3_dominant": (1.0, 1.0, 4.0),
    "no_c2": (1.0, 0.0, 4.0),
}
WEIGHT_PROFILE_ORDER = [
    "root_only",
    "pure_c2",
    "c2_trace_c1c3",
    "c2_light_c1c3",
    "c2_mild_c1c3",
    "c2_moderate_c1c3",
    "c2_very_dominant",
    "c2_dominant",
    "c2_heavy",
    "equal",
    "c1c3_heavy",
    "c3_dominant",
    "no_c2",
]
LAW_PACKAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tree_root_only": {
        "law_package": "root_only",
        "local_law_weight": None,
        "c1_relative_weight": 0.0,
        "c2_relative_weight": 0.0,
        "c3_relative_weight": 0.0,
    },
    "tree_c2_only": {
        "law_package": "c2_only",
        "local_law_weight": 0.5,
        "c1_relative_weight": 0.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 0.0,
    },
    "tree_all_laws": {
        "law_package": "all_laws",
        "local_law_weight": 0.5,
        "c1_relative_weight": 1.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 1.0,
    },
}
LAW_SET_CONFIGS: Dict[str, Dict[str, Any]] = {
    LAW_SET_ROOT_ONLY: dict(LAW_PACKAGE_CONFIGS["tree_root_only"]),
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY: dict(LAW_PACKAGE_CONFIGS["tree_c2_only"]),
    LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE: {
        "law_package": "c2c3",
        "local_law_weight": 0.5,
        "c1_relative_weight": 0.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 1.0,
    },
    LAW_SET_ALL: dict(LAW_PACKAGE_CONFIGS["tree_all_laws"]),
}
PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "batch_sizes": [32, 128],
        "medium_batch_sizes": [128, 256],
        "medium_seeds": [0],
        "docs_epochs_train_docs": [256, 1024],
        "docs_epochs_epochs": [1, 3],
        "learnability_train_docs": [1024, 2048],
        "learnability_weights": [0.0, 0.5],
        "learnability_profiles": ["pure_c2", "equal"],
        "weight_ablation_train_docs": [1024],
        "weight_ablation_profiles": ["root_only", "pure_c2", "equal"],
        "full_doc_anchor_train_docs": [256, 1024],
        "full_doc_anchor_seeds": [0],
        "efficiency_anchor_mode": "both",
        "efficiency_train_docs": [1024],
        "efficiency_anchor_train_docs_dense": [256, 512, 1024],
        "efficiency_anchor_seeds": [0],
        "efficiency_hardness_grid": "structural_core_v1",
        "efficiency_structural_cells": ["r4_p031"],
        "oracle_budget_train_docs": 1024,
        "oracle_budget_seeds": [0],
        "oracle_budget_method_runs": ["tree_neural:all"],
        "oracle_budget_reference_method_runs": list(DEFAULT_REFERENCE_METHOD_RUNS),
        "oracle_budget_calls_per_doc": [1.0],
        "oracle_budget_full_doc_shares": [0.5, 1.0],
        "oracle_budget_doc_consumption_modes": ["root_only", "doc_sequence"],
        "oracle_budget_local_split_modes": ["balanced"],
        "large_batch_batch_sizes": [256, 512, 1024],
        "supervision_train_docs": [1024],
        "supervision_leaf_profiles": ["none", "count_q100", "full_q100"],
        "supervision_internal_profiles": ["none", "count_q100"],
        "supervision_seeds": [0],
        "supervision_recovery_train_docs": [1024],
        "supervision_recovery_seeds": [0],
        "supervision_recovery_method_id": SUPERVISION_RECOVERY_TREE_FAMILY,
        "supervision_recovery_recoverable_benchmark": SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
        "supervision_recovery_structural_grid": SUPERVISION_RECOVERY_STRUCTURAL_GRID,
        "supervision_recovery_structural_cell": SUPERVISION_RECOVERY_STRUCTURAL_CELL,
        "supervision_recovery_leaf_token_ladder": [],
        "supervision_recovery_depth_discount_gammas": [1.0],
        "support_leaf_tokens": [8, 16],
        "support_seeds": [0],
    },
    "standard": {
        "batch_sizes": [16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 288, 320, 384, 512, 1024],
        "medium_batch_sizes": [128, 256, 512, 1024],
        "medium_seeds": [0, 1, 2, 3],
        "docs_epochs_train_docs": [1024, 2048, 4096, 10240],
        "docs_epochs_epochs": [1, 2, 5, 10],
        "learnability_train_docs": [2048, 4096, 10240],
        "learnability_weights": [0.0, 0.25, 0.5, 1.0],
        "learnability_profiles": ["pure_c2", "equal"],
        "weight_ablation_train_docs": [2048, 4096, 10240],
        "weight_ablation_profiles": WEIGHT_PROFILE_ORDER,
        "full_doc_anchor_train_docs": [1024, 2048, 4096, 10240],
        "full_doc_anchor_seeds": [0, 1, 2, 3],
        "efficiency_anchor_mode": "both",
        "efficiency_train_docs": [2048, 4096],
        "efficiency_anchor_train_docs_dense": [256, 512, 768, 1024, 1536, 2048, 3072, 4096],
        "efficiency_anchor_seeds": [0, 1, 2, 3, 4],
        "efficiency_hardness_grid": "structural_core_v1",
        "efficiency_structural_cells": list(EFFICIENCY_STRUCTURAL_CORE_CELLS),
        "oracle_budget_train_docs": 10240,
        "oracle_budget_seeds": [0, 1, 2],
        "oracle_budget_method_runs": [
            "tree_neural:on_range_idempotence_only",
            "tree_neural:merge_and_on_range_idempotence",
            "tree_neural:all",
        ],
        "oracle_budget_reference_method_runs": list(DEFAULT_REFERENCE_METHOD_RUNS),
        "oracle_budget_calls_per_doc": [0.5, 1.0, 2.0],
        "oracle_budget_full_doc_shares": [0.0, 0.5, 1.0],
        "oracle_budget_doc_consumption_modes": ["root_only", "doc_sequence"],
        "oracle_budget_local_split_modes": ["balanced"],
        "large_batch_batch_sizes": [256, 512, 1024],
        "supervision_train_docs": [2048, 4096, 10240],
        "supervision_leaf_profiles": list(SUPERVISION_LEAF_PROFILE_ORDER),
        "supervision_internal_profiles": list(SUPERVISION_INTERNAL_PROFILE_ORDER),
        "supervision_seeds": [0, 1],
        "supervision_recovery_train_docs": [1024, 2048, 4096],
        "supervision_recovery_seeds": [0, 1],
        "supervision_recovery_method_id": SUPERVISION_RECOVERY_TREE_FAMILY,
        "supervision_recovery_recoverable_benchmark": SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
        "supervision_recovery_structural_grid": SUPERVISION_RECOVERY_STRUCTURAL_GRID,
        "supervision_recovery_structural_cell": SUPERVISION_RECOVERY_STRUCTURAL_CELL,
        "supervision_recovery_leaf_token_ladder": [],
        "supervision_recovery_depth_discount_gammas": [1.0],
        "support_leaf_tokens": [8, 12, 16, 24, 32],
        "support_seeds": [0, 1],
    },
}

PRESET_DEFAULTS["v3"] = {
    **PRESET_DEFAULTS["standard"],
    "supervision_recovery_train_docs": [1024, 4096, 10240],
    "supervision_recovery_seeds": [0, 1],
    "supervision_recovery_method_id": SUPERVISION_RECOVERY_TREE_FAMILY,
    "supervision_recovery_recoverable_benchmark": SUPERVISION_RECOVERY_V3_RECOVERABLE_BENCHMARK,
    "supervision_recovery_structural_grid": SUPERVISION_RECOVERY_V3_STRUCTURAL_GRID,
    "supervision_recovery_structural_cell": SUPERVISION_RECOVERY_STRUCTURAL_CELL,
    "supervision_recovery_packages": list(SUPERVISION_RECOVERY_V3_PACKAGE_ORDER),
    "supervision_recovery_leaf_token_ladder": list(
        SUPERVISION_RECOVERY_V3_LEAF_TOKEN_LADDER
    ),
    "supervision_recovery_depth_discount_gammas": list(
        SUPERVISION_RECOVERY_V3_DEPTH_DISCOUNT_GAMMAS
    ),
    "support_leaf_tokens": [8, 16, 32],
}

PIPELINE_SELECTION_TEMPLATE: Dict[str, Any] = {
    "tradeoff_pipeline": {
        "preset": "v3",
        "phases": [
            "supervision_recovery",
            "report",
        ],
        "device_mode": "cuda",
        "train_docs": 4096,
        "val_docs": 1024,
        "test_docs": 1024,
        "supervision_recovery_train_docs": [1024, 2048, 4096],
        "supervision_recovery_seeds": [0, 1],
        "supervision_recovery_method_id": SUPERVISION_RECOVERY_TREE_FAMILY,
        "supervision_recovery_recoverable_benchmark": SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
        "supervision_recovery_structural_grid": SUPERVISION_RECOVERY_STRUCTURAL_GRID,
        "supervision_recovery_structural_cell": SUPERVISION_RECOVERY_STRUCTURAL_CELL,
        "supervision_recovery_packages": list(SUPERVISION_RECOVERY_V3_PACKAGE_ORDER),
        "supervision_recovery_leaf_token_ladder": list(
            SUPERVISION_RECOVERY_V3_LEAF_TOKEN_LADDER
        ),
        "supervision_recovery_depth_discount_gammas": list(
            SUPERVISION_RECOVERY_V3_DEPTH_DISCOUNT_GAMMAS
        ),
        "support_leaf_tokens": [8, 16, 32],
        "support_seeds": [0, 1],
        "support_modes": list(SUPPORTED_SUPPORT_MODES),
        "tree_reference": {
            "mode": "preset",
            "capacity_root": "",
            "preset": COMPARISON_GRID_V3_PRESET,
        },
        "structural_tree_reference": {
            "mode": "preset",
            "capacity_root": "",
            "preset": COMPARISON_GRID_V3_PRESET,
        },
        "runtime": {
            "data_mode": "resident",
            "bucket_mode": "leaf_count_auto_queue",
            "tree_batch_structural_pad_limit": 0.5,
            "tree_batch_auto_queue_min_docs": 8,
            "tree_batch_auto_queue_min_fill_ratio": 0.5,
            "preload_splits": ["train", "val", "test"],
            "preload_targets": True,
            "workers_per_mig": 1,
            "allow_multi_worker_screen": True,
            "capacity_workers_per_mig": 2,
        },
        "scheduler": {
            "mode": "global_per_run",
            "default_job_granularity": "family_train_seed",
            "cleanup_stale_children": True,
            "max_gpu_items_per_mig": 1,
        },
    }
}


@dataclass(frozen=True)
class SubprocessTask:
    name: str
    argv: Sequence[str]
    output_path: Path
    log_path: Path
    device_label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress_path: Path | None = None


@dataclass(frozen=True)
class _MigSliceInfo:
    uuid: str
    gpu_index: int
    mig_index: int
    total_mib: int = 0
    used_mib: int = 0
    free_mib: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


_safe_int = safe_int


def _infer_model_family_from_task_name(name: str) -> str:
    parts = [str(part).strip() for part in str(name).split("__") if str(part).strip()]
    if len(parts) >= 2:
        return str(parts[-2])
    return ""


@functools.lru_cache(maxsize=None)
def _resolved_full_doc_benchmark_spec(
    benchmark_name: str,
    hardness_grid: str,
    grid_cell_ids: tuple[str, ...],
):
    from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # type: ignore
        resolve_full_doc_diagnostic_benchmark,
        resolve_full_doc_diagnostic_grid,
    )

    normalized_grid = _ns(hardness_grid)
    if normalized_grid:
        benchmarks = resolve_full_doc_diagnostic_grid(normalized_grid)
        selected_ids = {
            str(value).strip() for value in grid_cell_ids if str(value).strip()
        }
        if selected_ids:
            for benchmark in benchmarks:
                if str(benchmark.cell_id or "").strip() in selected_ids:
                    return benchmark
        return benchmarks[0]
    return resolve_full_doc_diagnostic_benchmark(
        str(benchmark_name or "recoverable_v4") or "recoverable_v4"
    )


@functools.lru_cache(maxsize=None)
def _resolved_full_doc_bundle_token_geometry(
    benchmark_name: str,
    hardness_grid: str,
    grid_cell_ids: tuple[str, ...],
) -> tuple[int, int]:
    benchmark = _resolved_full_doc_benchmark_spec(
        benchmark_name,
        hardness_grid,
        grid_cell_ids,
    )
    candidate_paths = [
        str(getattr(benchmark, "canonical_bundle_path", "") or "").strip(),
        str(getattr(benchmark, "expanded_bundle_path", "") or "").strip(),
    ]
    for raw_path in candidate_paths:
        if not raw_path:
            continue
        bundle_path = Path(raw_path).expanduser()
        if not bundle_path.exists():
            continue
        try:
            from src.ctreepo.sim.core.markov_changepoint_ops_count import (
                MarkovOPSDataBundle,
            )

            bundle = MarkovOPSDataBundle.load(bundle_path)
        except Exception:
            continue
        token_lengths = sorted(
            {
                int(len(doc.tokens))
                for split in (bundle.train_docs, bundle.val_docs, bundle.test_docs)
                for doc in split
            }
        )
        if token_lengths:
            return int(token_lengths[0]), int(token_lengths[-1])
    benchmark_overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    min_tokens = int(_safe_int(benchmark_overrides.get("min_tokens"), 0))
    max_tokens = int(_safe_int(benchmark_overrides.get("max_tokens"), 0))
    if min_tokens > 0 and max_tokens > 0:
        return int(min_tokens), int(max_tokens)
    try:
        from src.ctreepo.sim.suite.markov_observed_token_policy import (
            resolve_markov_observed_token_policy,
        )

        policy = resolve_markov_observed_token_policy(
            profile_name=str(getattr(benchmark, "observed_token_profile", "") or ""),
        )
    except Exception:
        return 0, 0
    return int(getattr(policy, "min_tokens", 0) or 0), int(
        getattr(policy, "max_tokens", 0) or 0
    )


def _supervision_recovery_accounting_tokens(
    *,
    benchmark_name: str,
    hardness_grid: str,
    grid_cell_ids: tuple[str, ...],
    surfaced_min_tokens: int,
    surfaced_max_tokens: int,
) -> tuple[int, int]:
    bundle_min_tokens, bundle_max_tokens = _resolved_full_doc_bundle_token_geometry(
        benchmark_name,
        hardness_grid,
        grid_cell_ids,
    )
    if bundle_min_tokens > 0 and bundle_max_tokens > 0:
        return int(bundle_min_tokens), int(bundle_max_tokens)
    return int(surfaced_min_tokens), int(surfaced_max_tokens)


def _resolve_full_doc_task_benchmark_spec(
    *,
    worker_kind: str,
    task_payload: Mapping[str, Any] | None,
):
    payload = dict(task_payload or {})
    benchmark_name = (
        str(payload.get("template_benchmark", "recoverable_v4") or "recoverable_v4")
        if worker_kind == "full_doc_upper_bound"
        else str(payload.get("benchmark_name", "recoverable_v4") or "recoverable_v4")
    )
    hardness_grid = str(payload.get("hardness_grid", "") or "")
    grid_cell_ids = tuple(
        str(value).strip()
        for value in list(payload.get("grid_cell_ids") or ())
        if str(value).strip()
    )
    return _resolved_full_doc_benchmark_spec(
        benchmark_name,
        hardness_grid,
        grid_cell_ids,
    )


@functools.lru_cache(maxsize=None)
def _benchmark_locked_fno_fields(
    benchmark_name: str,
    hardness_grid: str = "",
    grid_cell_ids: tuple[str, ...] = tuple(),
    *,
    preserve_requested_leaf_tokens: bool = False,
    requested_fixed_leaf_tokens: int = 0,
) -> Dict[str, Any]:
    from src.ctreepo.sim.suite.markov_observed_token_policy import (  # type: ignore
        resolve_markov_observed_token_policy,
    )

    benchmark = _resolved_full_doc_benchmark_spec(
        str(benchmark_name or "recoverable_v4") or "recoverable_v4",
        str(hardness_grid or ""),
        tuple(str(value).strip() for value in grid_cell_ids if str(value).strip()),
    )
    policy = resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )
    observed_token_locked_fields: Dict[str, Any] = {
        "n_regimes": int(policy.n_regimes),
        "vocab_size": int(policy.vocab_size),
        "generator_profile": str(policy.generator_profile),
        "min_tokens": int(policy.min_tokens),
        "max_tokens": int(policy.max_tokens),
        "min_segments": int(policy.min_segments),
        "max_segments": int(policy.max_segments),
        "min_seg_len": int(getattr(policy, "min_seg_len", 0) or 0),
        "max_seg_len": int(getattr(policy, "max_seg_len", 0) or 0),
        "min_distinct_regimes_per_doc": getattr(
            policy, "min_distinct_regimes_per_doc", None
        ),
        "max_distinct_regimes_per_doc": getattr(
            policy, "max_distinct_regimes_per_doc", None
        ),
        "fixed_leaf_tokens": int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS),
    }
    benchmark_overrides = dict(getattr(benchmark, "config_overrides", {}) or {})
    benchmark_overrides.pop("fixed_leaf_tokens", None)
    for key in tuple(observed_token_locked_fields):
        if key in benchmark_overrides and benchmark_overrides[key] is not None:
            observed_token_locked_fields[key] = benchmark_overrides[key]
    return {
        "model_family": "fno",
        "feature_mode": "full",
        "n_regimes": int(observed_token_locked_fields["n_regimes"]),
        "vocab_size": int(observed_token_locked_fields["vocab_size"]),
        "generator_profile": str(observed_token_locked_fields["generator_profile"]),
        "min_tokens": int(observed_token_locked_fields["min_tokens"]),
        "max_tokens": int(observed_token_locked_fields["max_tokens"]),
        "min_segments": int(observed_token_locked_fields["min_segments"]),
        "max_segments": int(observed_token_locked_fields["max_segments"]),
        "min_seg_len": int(observed_token_locked_fields["min_seg_len"]),
        "max_seg_len": int(observed_token_locked_fields["max_seg_len"]),
        "min_distinct_regimes_per_doc": observed_token_locked_fields[
            "min_distinct_regimes_per_doc"
        ],
        "max_distinct_regimes_per_doc": observed_token_locked_fields[
            "max_distinct_regimes_per_doc"
        ],
        "fixed_leaf_tokens": int(
            observed_token_locked_fields["fixed_leaf_tokens"]
        ),
    }


def _full_doc_task_families(task_payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    payload = dict(task_payload or {})
    return tuple(
        str(item).strip()
        for item in list(payload.get("baseline_families") or ())
        if str(item).strip()
    )


def _full_doc_task_benchmark_name(
    *,
    worker_kind: str,
    task_payload: Mapping[str, Any] | None,
) -> str:
    payload = dict(task_payload or {})
    if worker_kind == "full_doc_upper_bound":
        return str(payload.get("template_benchmark", "recoverable_v4") or "recoverable_v4")
    return str(payload.get("benchmark_name", "recoverable_v4") or "recoverable_v4")


def _is_canonical_official_fno_task(
    *,
    worker_kind: str,
    task_payload: Mapping[str, Any] | None,
) -> bool:
    if worker_kind not in {"full_doc_diagnostics", "full_doc_upper_bound"}:
        return False
    families = _full_doc_task_families(task_payload)
    return bool(families) and all(
        str(family) in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES for family in families
    )


def _uses_full_doc_config_codec(worker_kind: str) -> bool:
    return str(worker_kind or "").strip().lower() in FULL_DOC_CONFIG_WORKER_KINDS


def _serialized_worker_config(
    *,
    worker_kind: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if _uses_full_doc_config_codec(worker_kind):
        return serialize_full_doc_runtime_config(
            config,
            metadata=metadata,
            allow_private_tree_aliases=True,
        )
    payload = dict(config)
    if metadata:
        payload.update(dict(metadata))
    return payload


def _resolved_full_doc_task_config(
    *,
    worker_kind: str,
    config: Mapping[str, Any],
    task_payload: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    original_config = (
        runtime_config_overrides_from_config_like(
            config,
            allow_private_tree_aliases=True,
        )
        if _uses_full_doc_config_codec(worker_kind)
        else dict(config)
    )
    resolved = dict(original_config)
    canonical_official_fno_task = _is_canonical_official_fno_task(
        worker_kind=worker_kind,
        task_payload=task_payload,
    )
    families = _full_doc_task_families(task_payload)
    if len(families) == 1:
        requested_family = str(families[0])
        config_family = str(resolved.get("baseline_family", "") or "").strip()
        if config_family and config_family != requested_family:
            raise ValueError(
                "full-doc task config baseline_family drifted from task payload "
                f"(config={config_family!r}, payload={requested_family!r})"
            )
        resolved["baseline_family"] = requested_family
    requested_fixed_leaf_tokens_raw = original_config.get("fixed_leaf_tokens", None)
    requested_fixed_leaf_tokens = int(
        _safe_int(requested_fixed_leaf_tokens_raw, 0)
    )
    requested_leafgrid_tokens = int(
        _safe_int(resolved.get("pipeline_supervision_recovery_leaf_tokens"), 0)
    )
    leafgrid_active = bool(
        resolved.get("pipeline_supervision_recovery_leafgrid_active", False)
    ) or requested_leafgrid_tokens > 0
    leafgrid_preserve_requested_leaf_tokens = bool(
        leafgrid_active and requested_leafgrid_tokens > 0
    )
    requested_comparison_mode = infer_markov_comparison_mode(
        requested_mode=str(resolved.get("comparison_mode", "") or ""),
        baseline_families=families,
        tree_exact_collapse_mode=str(
            resolved.get("tree_exact_collapse_mode", "") or ""
        ),
    )
    benchmark = None
    if canonical_official_fno_task:
        benchmark = _resolve_full_doc_task_benchmark_spec(
            worker_kind=worker_kind,
            task_payload=task_payload,
        )
        resolved.update(
            _benchmark_locked_fno_fields(
                str(getattr(benchmark, "name", "")),
                str(getattr(benchmark, "grid_name", "") or ""),
                tuple(
                    [str(getattr(benchmark, "cell_id", "") or "").strip()]
                    if str(getattr(benchmark, "cell_id", "") or "").strip()
                    else []
                ),
                preserve_requested_leaf_tokens=False,
                requested_fixed_leaf_tokens=0,
            )
        )
        resolved["fixed_leaf_tokens"] = int(
            FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
        )
        resolved["preserve_requested_leaf_tokens"] = bool(
            leafgrid_preserve_requested_leaf_tokens
        )
        resolved["official_fno_preserve_requested_leaf_tokens"] = bool(
            leafgrid_preserve_requested_leaf_tokens
        )
    elif worker_kind in {"full_doc_diagnostics", "full_doc_upper_bound"}:
        if (
            requested_fixed_leaf_tokens_raw not in {"", None}
            and requested_fixed_leaf_tokens > 0
        ):
            resolved["preserve_requested_leaf_tokens"] = True
            resolved["official_fno_preserve_requested_leaf_tokens"] = True
        benchmark = _resolve_full_doc_task_benchmark_spec(
            worker_kind=worker_kind,
            task_payload=task_payload,
        )
    if leafgrid_preserve_requested_leaf_tokens:
        resolved["preserve_requested_leaf_tokens"] = True
        resolved["official_fno_preserve_requested_leaf_tokens"] = True
    if leafgrid_preserve_requested_leaf_tokens:
        if not bool(resolved.get("preserve_requested_leaf_tokens", False)):
            raise ValueError(
                "leaf-grid supervision-recovery tasks must set "
                "preserve_requested_leaf_tokens=True"
            )
        if not bool(
            resolved.get("official_fno_preserve_requested_leaf_tokens", False)
        ):
            raise ValueError(
                "leaf-grid supervision-recovery tasks must set "
                "official_fno_preserve_requested_leaf_tokens=True"
            )
        resolved_fixed_leaf_tokens = int(_safe_int(resolved.get("fixed_leaf_tokens"), 0))
        if (
            resolved_fixed_leaf_tokens > 0
            and resolved_fixed_leaf_tokens != requested_leafgrid_tokens
        ):
            raise ValueError(
                "leaf-grid supervision-recovery fixed_leaf_tokens drifted before task "
                f"serialization: requested={requested_leafgrid_tokens} "
                f"resolved={resolved_fixed_leaf_tokens}"
            )
    if benchmark is not None:
        comparison_mode = requested_comparison_mode
        resolved["comparison_mode"] = str(comparison_mode)
        if comparison_mode in {"comparable", "exact_collapse"}:
            surface = resolve_markov_comparable_surface(
                benchmark=benchmark,
                config=resolved,
                comparison_mode=comparison_mode,
            )
            surfaced = apply_comparable_surface_to_mapping(
                benchmark=benchmark,
                config=resolved,
                surface=surface,
            )
            exact_full_doc_parity = _supervision_recovery_requires_exact_full_doc_parity(
                package_name=str(
                    original_config.get("pipeline_supervision_recovery_package", "") or ""
                ),
                package_spec=original_config,
                payload=original_config,
            )
            matched_root_surface_lock = _supervision_recovery_requires_matched_root_surface_lock(
                package_name=str(
                    original_config.get("pipeline_supervision_recovery_package", "") or ""
                ),
                package_spec=original_config,
                payload=original_config,
            )
            if canonical_official_fno_task or exact_full_doc_parity or matched_root_surface_lock:
                resolved = dict(surfaced)
                if canonical_official_fno_task:
                    resolved["fixed_leaf_tokens"] = int(
                        FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
                    )
                    resolved["preserve_requested_leaf_tokens"] = bool(
                        leafgrid_preserve_requested_leaf_tokens
                    )
                    resolved["official_fno_preserve_requested_leaf_tokens"] = bool(
                        leafgrid_preserve_requested_leaf_tokens
                    )
            else:
                resolved = dict(surfaced)
                resolved.update(original_config)
                resolved["comparison_mode"] = str(comparison_mode)
                resolved["preserve_requested_leaf_tokens"] = bool(
                    surfaced.get("preserve_requested_leaf_tokens", False)
                    or original_config.get("preserve_requested_leaf_tokens", False)
                )
                resolved["official_fno_preserve_requested_leaf_tokens"] = bool(
                    surfaced.get("official_fno_preserve_requested_leaf_tokens", False)
                    or original_config.get(
                        "official_fno_preserve_requested_leaf_tokens",
                        False,
                    )
                )
    # Recompute package_semantics after all mutations to prevent stale values.
    resolved["package_semantics"] = ""
    resolved["package_semantics"] = resolve_package_semantics(resolved)
    return resolved


def _full_doc_runtime_overrides_for_worker(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return full-doc overrides without legacy public run-axis fields."""
    payload = runtime_config_overrides_from_config_like(
        config,
        allow_private_tree_aliases=True,
    )
    for key in (
        set(LEGACY_PUBLIC_RUN_AXIS_CONFIG_FIELDS)
        | set(LEGACY_PUBLIC_OBJECTIVE_CONFIG_FIELDS)
    ):
        payload.pop(str(key), None)
    return payload


def _effective_task_epoch_total(
    *,
    worker_kind: str,
    config: Mapping[str, Any],
    task_payload: Mapping[str, Any] | None,
) -> int:
    resolved = _resolved_full_doc_task_config(
        worker_kind=worker_kind,
        config=config,
        task_payload=task_payload,
    )
    if _is_canonical_official_fno_task(
        worker_kind=worker_kind,
        task_payload=task_payload,
    ):
        return _safe_int(resolved.get("n_epochs"), default=0)
    schedule = _gs(resolved, "tree_training_schedule").lower()
    if schedule == "two_stage":
        total = _safe_int(resolved.get("tree_stage1_epochs"), default=0) + _safe_int(
            resolved.get("tree_stage2_epochs"), default=0
        )
        if total > 0:
            return int(total)
    return _safe_int(resolved.get("n_epochs"), default=0)


def _task_comparison_surface_snapshot(
    *,
    worker_kind: str,
    config: Mapping[str, Any],
    task_payload: Mapping[str, Any] | None,
) -> tuple[str, Dict[str, Any]]:
    if worker_kind not in {"full_doc_diagnostics", "full_doc_upper_bound"}:
        return "legacy", {}
    benchmark = _resolve_full_doc_task_benchmark_spec(
        worker_kind=worker_kind,
        task_payload=task_payload,
    )
    resolved = _resolved_full_doc_task_config(
        worker_kind=worker_kind,
        config=config,
        task_payload=task_payload,
    )
    comparison_mode = normalize_markov_comparison_mode(
        str(resolved.get("comparison_mode", "legacy") or "legacy")
    )
    surface = resolve_markov_comparable_surface(
        benchmark=benchmark,
        config=resolved,
        comparison_mode=comparison_mode,
    )
    return comparison_mode, surface.to_dict()


def _parse_mib_text(value: str | None) -> int:
    token = str(value or "").strip().split(" ", 1)[0]
    return _safe_int(token, default=0)


def _detect_mig_inventory() -> List[_MigSliceInfo]:
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    uuid_by_slot: Dict[tuple[int, int], str] = {}
    current_gpu_index = -1
    for raw_line in listing.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("GPU "):
            prefix = line.split(":", 1)[0]
            current_gpu_index = _safe_int(prefix.split()[1], default=-1)
            continue
        if (
            current_gpu_index >= 0
            and line.startswith("MIG ")
            and "Device" in line
            and "UUID:" in line
        ):
            try:
                device_fragment = line.split("Device", 1)[1]
                mig_index = _safe_int(device_fragment.split(":", 1)[0], default=-1)
                mig_uuid = line.split("UUID:", 1)[1].rstrip(")").strip()
            except Exception:
                continue
            if mig_index >= 0 and mig_uuid:
                uuid_by_slot[(current_gpu_index, mig_index)] = mig_uuid

    inventory: List[_MigSliceInfo] = []
    for gpu_index in sorted({key[0] for key in uuid_by_slot}):
        try:
            xml_result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index), "-q", "-x"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            root = ET.fromstring(xml_result.stdout)
        except Exception:
            root = None

        mig_stats: Dict[int, tuple[int, int, int]] = {}
        if root is not None:
            for gpu_elem in root.findall("gpu"):
                mig_devices = gpu_elem.find("mig_devices")
                if mig_devices is None:
                    continue
                for mig_elem in mig_devices.findall("mig_device"):
                    mig_index = _safe_int(mig_elem.findtext("index"), default=-1)
                    if mig_index < 0:
                        continue
                    fb_usage = mig_elem.find("fb_memory_usage")
                    total_mib = _parse_mib_text(
                        fb_usage.findtext("total") if fb_usage is not None else None
                    )
                    used_mib = _parse_mib_text(
                        fb_usage.findtext("used") if fb_usage is not None else None
                    )
                    free_mib = _parse_mib_text(
                        fb_usage.findtext("free") if fb_usage is not None else None
                    )
                    mig_stats[mig_index] = (total_mib, used_mib, free_mib)

        for (slot_gpu_index, mig_index), mig_uuid in sorted(uuid_by_slot.items()):
            if slot_gpu_index != gpu_index:
                continue
            total_mib, used_mib, free_mib = mig_stats.get(mig_index, (0, 0, 0))
            inventory.append(
                _MigSliceInfo(
                    uuid=str(mig_uuid),
                    gpu_index=int(gpu_index),
                    mig_index=int(mig_index),
                    total_mib=int(total_mib),
                    used_mib=int(used_mib),
                    free_mib=int(free_mib),
                )
            )
    return inventory


def _preparse_worker_task(argv: Sequence[str]) -> Path | None:
    for idx, token in enumerate(argv):
        if token == "--worker-task" and idx + 1 < len(argv):
            return Path(argv[idx + 1]).expanduser()
    return None


def _set_thread_env_defaults() -> None:
    _set_shared_thread_env_defaults()


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return dict(asdict(value))
    return {}


def _run_worker(task_path: Path) -> int:
    _set_thread_env_defaults()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    task = json.loads(task_path.read_text(encoding="utf-8"))
    worker_kind = str(task.get("worker_kind", "ops_count") or "ops_count").strip().lower()
    config = _resolved_full_doc_task_config(
        worker_kind=worker_kind,
        config=dict(task.get("config", {}) or {}),
        task_payload=task,
    )
    task_name = str(task.get("name", task_path.stem) or task_path.stem)
    progress_path_text = _gs(task, "progress_path")
    progress_path = Path(progress_path_text).expanduser() if progress_path_text else None
    expected_epoch_total = _effective_task_epoch_total(
        worker_kind=worker_kind,
        config=config,
        task_payload=task,
    )
    pipeline_metadata: Dict[str, Any] = {}
    for key in (
        "pipeline_law_package_name",
        "pipeline_supervision_leaf_profile",
        "pipeline_supervision_internal_profile",
    ):
        if key in config:
            pipeline_metadata[key] = config.pop(key)
    output_json = Path(task["output_json"]).expanduser()
    artifact_dir = Path(str(config.get("artifact_dir") or output_json.parent / f"{output_json.stem}_artifacts"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config["artifact_dir"] = str(artifact_dir)

    def _write_worker_progress(
        *,
        state: str,
        stage: str,
        **extra: Any,
    ) -> None:
        if progress_path is None:
            return
        current_payload: Dict[str, Any] = {}
        if progress_path.exists():
            try:
                loaded = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current_payload = dict(loaded)
            except Exception:
                current_payload = {}
        current_payload.update(
            {
                "schema_version": 1,
                "updated_at": _utc_now(),
                "state": str(state),
                "stage": str(stage),
                "task_name": str(task_name),
                "worker_kind": str(worker_kind),
                "output_json": str(output_json),
            }
        )
        for key, value in extra.items():
            if value is None:
                continue
            current_payload[str(key)] = value
        _write_json_atomic(progress_path, current_payload)

    def _emit_model_progress(progress: Mapping[str, Any]) -> None:
        payload = dict(progress)
        state = str(payload.pop("state", "running") or "running")
        stage = str(payload.pop("stage", "running") or "running")
        _write_worker_progress(state=state, stage=stage, **payload)

    _write_worker_progress(
        state="running",
        stage="starting",
        epoch_completed=0,
        epochs_total=int(expected_epoch_total),
    )

    try:
        if worker_kind == "ops_count":
            from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # type: ignore
                OPSCountConfig,
                run_markov_changepoint_ops_count_experiment,
            )

            started = time.perf_counter()
            summary = run_markov_changepoint_ops_count_experiment(OPSCountConfig(**config))
            wall_s = time.perf_counter() - started
            payload = json.loads(summary.to_json())
            if pipeline_metadata:
                payload_config = dict(payload.get("config", {}) or {})
                payload_config.update(pipeline_metadata)
                payload["config"] = payload_config
        elif worker_kind == "full_doc_upper_bound":
            from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # type: ignore
                _resolve_device as _resolve_full_doc_device,
                _run_family_with_predictions,
                resolve_full_doc_diagnostic_benchmark,
            )
            from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # type: ignore
                OPSCountConfig,
                build_markov_changepoint_ops_count_data_bundle,
            )

            template_name = str(task.get("template_benchmark", "recoverable_v4") or "recoverable_v4")
            template = resolve_full_doc_diagnostic_benchmark(template_name)
            config.setdefault("state_dim", int(template.official_state_dim))
            config.setdefault("hidden_dim", int(template.official_hidden_dim))
            config.setdefault("n_epochs", int(template.official_epochs))
            config.setdefault("batch_size", int(template.official_batch_size))
            config.setdefault("lr", float(template.official_lr))
            config.setdefault("weight_decay", float(template.official_weight_decay))
            cfg = OPSCountConfig(**config)
            runtime_seeds, device = _resolve_full_doc_device(cfg)
            families = tuple(
                str(item).strip()
                for item in list(task.get("baseline_families") or ())
                if str(item).strip()
            ) or CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
            started = time.perf_counter()
            bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
            rows: List[Dict[str, Any]] = []
            for family in families:
                family_started = time.perf_counter()
                result = _run_family_with_predictions(
                    baseline_family=str(family),
                    config=cfg,
                    benchmark=template,
                    seeds=runtime_seeds,
                    device=device,
                    train_docs=bundle.train_docs,
                    val_docs=bundle.val_docs,
                    test_docs=bundle.test_docs,
                )
                family_wall_s = time.perf_counter() - family_started
                train_metrics = _coerce_mapping(result.get("train_metrics"))
                val_metrics = _coerce_mapping(result.get("val_metrics"))
                test_metrics = _coerce_mapping(result.get("test_metrics"))
                fit_diag = _coerce_mapping(result.get("fit_diag"))
                rows.append(
                    {
                        "baseline_family": str(result.get("baseline_family", family)),
                        "train_doc_count": int(result.get("train_doc_count", config.get("train_docs", 0)) or 0),
                        "seed": int(task.get("seed", config.get("seed", 0)) or 0),
                        "train_root_mae": float(train_metrics.get("root_mae", float("nan"))),
                        "val_root_mae": float(val_metrics.get("root_mae", float("nan"))),
                        "test_root_mae": float(test_metrics.get("root_mae", float("nan"))),
                        "test_exact_match_rate": float(
                            fit_diag.get(
                                "test_exact_match_rate",
                                test_metrics.get("exact_match", float("nan")),
                            )
                        ),
                        "best_epoch": int(fit_diag.get("best_epoch", 0) or 0),
                        "family_wall_clock_s": float(family_wall_s),
                    }
                )
            wall_s = time.perf_counter() - started
            payload = {
                "benchmark": "pipeline_current_markov",
                "template_benchmark": str(template_name),
                "config": _serialized_worker_config(
                    worker_kind=worker_kind,
                    config=config,
                    metadata=pipeline_metadata,
                ),
                "baseline_families": [str(item) for item in families],
                "train_doc_count": int(config.get("train_docs", 0) or 0),
                "seed": int(task.get("seed", config.get("seed", 0)) or 0),
                "rows": rows,
            }
        elif worker_kind == "full_doc_diagnostics":
            from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # type: ignore
                run_markov_full_doc_anchor_diagnostics,
            )

            baseline_families = tuple(
                str(item).strip()
                for item in list(task.get("baseline_families") or ())
                if str(item).strip()
            )
            train_doc_counts = tuple(
                int(value)
                for value in list(task.get("train_doc_counts") or ())
                if int(value) > 0
            )
            seeds = tuple(
                int(value)
                for value in list(task.get("seeds") or ())
                if isinstance(value, (int, float, str))
            )
            started = time.perf_counter()
            payload = run_markov_full_doc_anchor_diagnostics(
                benchmark_name=str(task.get("benchmark_name", "recoverable_v4") or "recoverable_v4"),
                hardness_grid=str(task.get("hardness_grid", "") or ""),
                grid_cell_ids=tuple(
                    str(value).strip()
                    for value in list(task.get("grid_cell_ids") or ())
                    if str(value).strip()
                ),
                seeds=seeds or (int(task.get("seed", config.get("seed", 0)) or 0),),
                train_doc_counts=train_doc_counts,
                baseline_families=baseline_families or None,
                emit_confusion=False,
                output_dir=artifact_dir,
                use_cuda=bool(config.get("use_cuda", False)),
                cuda_device=config.get("cuda_device"),
                torch_threads=int(config.get("torch_threads", 1) or 1),
                config_overrides=_full_doc_runtime_overrides_for_worker(config),
                run_metadata={
                    **pipeline_metadata,
                    "task_name": str(task_name),
                    "hazard_panel_id": str(task.get("hazard_panel_id", "") or ""),
                    "base_bundle_path": str(task.get("base_bundle_path", "") or ""),
                },
                progress_callback=_emit_model_progress,
                base_bundle_path=str(
                    task.get(
                        "base_bundle_path",
                        config.get("pipeline_base_bundle_path", ""),
                    )
                    or ""
                ),
            )
            wall_s = time.perf_counter() - started
            payload = dict(payload)
            payload["config"] = _serialized_worker_config(
                worker_kind=worker_kind,
                config=config,
                metadata=pipeline_metadata,
            )
            payload["wall_clock_s"] = float(wall_s)
        else:
            raise ValueError(f"unsupported worker_kind={worker_kind!r}")

        payload["wall_clock_s"] = float(wall_s)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        existing_epoch_total = int(expected_epoch_total)
        if progress_path is not None and progress_path.exists():
            try:
                existing_progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(existing_progress, dict):
                    existing_epoch_total = max(
                        existing_epoch_total,
                        _safe_int(existing_progress.get("epochs_total"), default=0),
                    )
            except Exception:
                pass
        _write_worker_progress(
            state="completed",
            stage="completed",
            epoch_completed=existing_epoch_total,
            epochs_total=existing_epoch_total,
            wall_clock_s=float(wall_s),
        )
        metadata = {
            "worker_kind": worker_kind,
            "task_name": str(task_name),
            "output_json": str(output_json),
            "wall_clock_s": float(wall_s),
            "config": _serialized_worker_config(
                worker_kind=worker_kind,
                config=config,
                metadata=pipeline_metadata,
            ),
        }
        print(json.dumps(metadata, indent=2))
        return 0
    except Exception as exc:
        _write_worker_progress(
            state="failed",
            stage="failed",
            error_type=str(type(exc).__name__),
            error_message=str(exc),
        )
        raise


def _parse_int_list(text: str | None, default: Sequence[int]) -> List[int]:
    return _shared_parse_int_list(text, default=default, separators=",")


def _parse_float_list(text: str | None, default: Sequence[float]) -> List[float]:
    return _shared_parse_float_list(text, default=default, separators=",")


def _parse_str_list(text: str | None, default: Sequence[str]) -> List[str]:
    return _shared_parse_str_list(text, default=default, separators=",")


def _parse_key_value_text_map(text: str | None) -> Dict[str, str]:
    if text is None:
        return {}
    payload: Dict[str, str] = {}
    normalized = str(text or "").replace(",", ";").replace(" ", ";")
    for raw in normalized.split(";"):
        chunk = raw.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"invalid key=value mapping entry {chunk!r}; expected key=value"
            )
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            payload[key] = value
    return payload


def _run_axis_from_token(
    token: str,
    *,
    problem_id: str = "markov_ops_count",
    role: str = "primary",
) -> Dict[str, Any]:
    text = str(token or "").strip()
    if not text:
        raise ValueError("method run token must be non-empty")
    if ":" in text:
        method_id, law_set_id = text.split(":", 1)
    else:
        method_id, law_set_id = text, LAW_SET_ALL
    return RunAxisSpec(
        problem_id=problem_id,
        method_id=str(method_id).strip(),
        law_set_id=canonical_law_set_id(str(law_set_id).strip() or LAW_SET_ALL),
        role=role,
    ).to_dict()


def _parse_run_axis_list(
    text: Any,
    default: Sequence[Any],
    *,
    role: str,
) -> List[Dict[str, Any]]:
    if text is None:
        raw_items: Sequence[Any] = list(default)
    elif isinstance(text, (list, tuple)):
        raw_items = list(text)
    else:
        raw_items = _parse_str_list(str(text), ())
    if not raw_items:
        raw_items = list(default)
    runs: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            payload = dict(item)
            payload.setdefault("role", role)
            runs.append(RunAxisSpec.from_mapping(payload).to_dict())
        else:
            runs.append(_run_axis_from_token(str(item), role=role))
    return runs


RUN_AXIS_CONFIG_ROLES = {
    "method_runs": "primary",
    "parity_method_runs": "primary",
    "oracle_budget_method_runs": "primary",
    "reference_method_runs": "reference",
    "parity_reference_method_runs": "reference",
    "oracle_budget_reference_method_runs": "reference",
    "full_doc_anchor_reference_method_runs": "reference",
}


def _normalize_run_axis_config_aliases(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def _clean(value: Any, key: str = "") -> Any:
        role = RUN_AXIS_CONFIG_ROLES.get(str(key))
        if role is not None:
            return _parse_run_axis_list(value, (), role=role)
        if isinstance(value, Mapping):
            return {str(child_key): _clean(child_value, str(child_key)) for child_key, child_value in value.items()}
        if isinstance(value, list):
            return [_clean(item) for item in value]
        if isinstance(value, tuple):
            return [_clean(item) for item in value]
        return value

    return dict(_clean(payload))


def _method_ids_from_run_axes(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    return [str(run.get("method_id") or "").strip() for run in runs if str(run.get("method_id") or "").strip()]


def _legacy_family_from_run_axis(run: Mapping[str, Any]) -> str:
    method_id = str(run.get("method_id") or "").strip()
    law_set_id = canonical_law_set_id(str(run.get("law_set_id") or LAW_SET_ALL))
    if method_id == "tree_neural":
        if law_set_id == LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY:
            return "tree_neural_c2"
        if law_set_id == LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE:
            return "tree_neural_c2c3"
    return method_id


def _legacy_families_from_run_axes(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    return [_legacy_family_from_run_axis(run) for run in runs]


def _stringify_cli_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)) and all(isinstance(item, Mapping) for item in value):
        return list(value)
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return value


def _optional_path_text(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if raw in {"", "."}:
        return ""
    return raw


@functools.lru_cache(maxsize=1)
def _ops_count_supported_config_keys() -> set[str]:
    from dataclasses import fields

    from src.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig

    return {str(field.name) for field in fields(OPSCountConfig)}


def _reject_legacy_public_run_axis_config(payload: Mapping[str, Any], *, path: Path) -> None:
    assert_public_contract_clean(payload, surface=str(path))


def _load_selection_config(
    path: Path | None,
    *,
    section_names: Sequence[str],
) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = load_structured_config(path)
    _reject_legacy_public_run_axis_config(
        _normalize_run_axis_config_aliases(payload),
        path=path,
    )
    for section_name in section_names:
        section = payload.get(section_name)
        if isinstance(section, Mapping):
            flat = dict(section)
            runtime_section = section.get("runtime")
            if isinstance(runtime_section, Mapping):
                for key, value in runtime_section.items():
                    flat[f"runtime_{str(key)}"] = value
            scheduler_section = section.get("scheduler")
            if isinstance(scheduler_section, Mapping):
                for key, value in scheduler_section.items():
                    flat[f"scheduler_{str(key)}"] = value
            tree_reference_section = section.get("tree_reference")
            if isinstance(tree_reference_section, Mapping):
                for key, value in tree_reference_section.items():
                    flat[f"tree_reference_{str(key)}"] = value
            structural_tree_reference_section = section.get("structural_tree_reference")
            if isinstance(structural_tree_reference_section, Mapping):
                for key, value in structural_tree_reference_section.items():
                    flat[f"structural_tree_reference_{str(key)}"] = value
            one_leaf_tree_reference_section = section.get("one_leaf_tree_reference")
            if isinstance(one_leaf_tree_reference_section, Mapping):
                for key, value in one_leaf_tree_reference_section.items():
                    flat[f"one_leaf_tree_reference_{str(key)}"] = value
            return flat
    flat = dict(payload)
    runtime_section = payload.get("runtime")
    if isinstance(runtime_section, Mapping):
        for key, value in runtime_section.items():
            flat[f"runtime_{str(key)}"] = value
    scheduler_section = payload.get("scheduler")
    if isinstance(scheduler_section, Mapping):
        for key, value in scheduler_section.items():
            flat[f"scheduler_{str(key)}"] = value
    tree_reference_section = payload.get("tree_reference")
    if isinstance(tree_reference_section, Mapping):
        for key, value in tree_reference_section.items():
            flat[f"tree_reference_{str(key)}"] = value
    structural_tree_reference_section = payload.get("structural_tree_reference")
    if isinstance(structural_tree_reference_section, Mapping):
        for key, value in structural_tree_reference_section.items():
            flat[f"structural_tree_reference_{str(key)}"] = value
    one_leaf_tree_reference_section = payload.get("one_leaf_tree_reference")
    if isinstance(one_leaf_tree_reference_section, Mapping):
        for key, value in one_leaf_tree_reference_section.items():
            flat[f"one_leaf_tree_reference_{str(key)}"] = value
    return flat


def _preparse_config_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection-config", "--config", dest="selection_config", type=Path, default=None)
    parser.add_argument(
        "--write-selection-template",
        "--write-config-template",
        dest="write_selection_template",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-source",
        dest="report_sources",
        action="append",
        default=None,
        help="Stage an external report source into this version root as key=path. Repeatable.",
    )
    parsed, _ = parser.parse_known_args(list(argv))
    return parsed


def _build_parser(*, config_defaults: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the full optimized Markov tradeoff grid and render an updated consolidated report.\n\n"
            "Recommended workflow:\n"
            "  1. Start from config/markov/tradeoff_pipeline.standard.toml or generate\n"
            "     a custom TOML with --write-config-template.\n"
            "  2. Inspect the resolved run with --config ... --plan-only.\n"
            "  3. Launch the real run with --config ... plus any needed output/device overrides.\n\n"
            "Direct grid/model flags remain available as advanced overrides on top of\n"
            "the resolved config."
        ),
        epilog=(
            "Examples:\n"
            "  python3 scripts/run_markov_optimization_tradeoff_pipeline.py \\\n"
            "    --config config/markov/tradeoff_pipeline.standard.toml \\\n"
            "    --plan-only\n\n"
            "  python3 scripts/run_markov_optimization_tradeoff_pipeline.py \\\n"
            "    --config config/markov/tradeoff_pipeline.standard.toml \\\n"
            "    --output-root outputs/markov_tradeoff_$(date +%Y%m%d_%H%M%S)\n\n"
            "  python3 scripts/run_markov_optimization_tradeoff_pipeline.py \\\n"
            "    --write-config-template outputs/markov_tradeoff_pipeline.custom.toml"
        ),
    )
    parser.add_argument(
        "--selection-config",
        "--config",
        dest="selection_config",
        type=Path,
        default=None,
        help="Path to a .toml or .json run config file. Recommended primary interface.",
    )
    parser.add_argument(
        "--write-selection-template",
        "--write-config-template",
        dest="write_selection_template",
        type=Path,
        default=None,
        help="Write a starter .toml or .json config template and exit. Prefer committing important run configs under config/markov/.",
    )
    parser.add_argument("--write-run-plan", type=Path, default=None, help="Write the fully resolved run plan JSON and exit or continue.")
    parser.add_argument("--plan-only", action=argparse.BooleanOptionalAction, default=False, help="Print the fully resolved run plan and exit.")
    parser.add_argument(
        "--refresh-existing-output-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild aggregate summaries and the report for an existing output root without rerunning worker cells.",
    )
    parser.add_argument("--worker-task", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / f"markov_optimization_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--preset", choices=sorted(PRESET_DEFAULTS), default="standard")
    parser.add_argument(
        "--phases",
        type=str,
        default="batch_timing,medium_grid,docs_epochs,learnability,weight_ablation,law_packages,full_doc_anchor,oracle_budget_frontier,efficiency_suite,large_batch_diagnosis,support_grid,report",
        help="Comma-separated phases to run.",
    )
    parser.add_argument(
        "--report-source",
        dest="report_sources",
        action="append",
        default=None,
        help="Stage an external report source into this version root as key=path. Repeatable.",
    )
    parser.add_argument("--device-mode", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-workers", type=int, default=0, help="0 uses all detected MIG slices, or 1 on CPU.")
    parser.add_argument("--migs", type=str, default="", help="Optional explicit comma-separated MIG UUIDs.")

    parser.add_argument("--train-docs", type=int, default=10240, help="Default train-doc count for medium/law/support phases when not overridden by a grid.")
    parser.add_argument("--val-docs", type=int, default=1024)
    parser.add_argument("--test-docs", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--fixed-leaf-tokens", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--fno-width", type=int, default=16)
    parser.add_argument("--fno-n-modes", type=int, default=8)
    parser.add_argument("--fno-n-layers", type=int, default=2)
    parser.add_argument("--theorem-feature-dim", type=int, default=16)
    parser.add_argument("--theorem-feature-hidden-dim", type=int, default=32)
    parser.add_argument("--medium-epochs", type=int, default=5)
    parser.add_argument("--medium-val-docs", type=int, default=1024)
    parser.add_argument("--medium-exact-doc-limit", type=int, default=128)
    parser.add_argument("--docs-epochs-batch-size", type=int, default=256)
    parser.add_argument("--law-batch-size", type=int, default=256)
    parser.add_argument("--law-epochs", type=int, default=10)
    parser.add_argument("--support-batch-size", type=int, default=256)
    parser.add_argument("--support-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seeds", type=str, default="0 1")

    parser.add_argument("--batch-sizes", type=str, default=None)
    parser.add_argument("--medium-batch-sizes", type=str, default=None)
    parser.add_argument("--medium-seeds", type=str, default=None)
    parser.add_argument("--docs-epochs-train-docs", type=str, default=None)
    parser.add_argument("--docs-epochs-epochs", type=str, default=None)
    parser.add_argument("--learnability-train-docs", type=str, default=None)
    parser.add_argument("--learnability-weights", type=str, default=None)
    parser.add_argument("--learnability-profiles", type=str, default=None)
    parser.add_argument("--weight-ablation-train-docs", type=str, default=None)
    parser.add_argument("--weight-ablation-profiles", type=str, default=None)
    parser.add_argument(
        "--law-set-ids",
        type=str,
        default=None,
        help="Optional comma- or space-separated generic law_set_id list.",
    )
    parser.add_argument("--support-leaf-tokens", type=str, default=None)
    parser.add_argument("--support-seeds", type=str, default=None)
    parser.add_argument("--support-modes", type=str, default=None)
    parser.add_argument("--full-doc-anchor-train-docs", type=str, default=None)
    parser.add_argument("--full-doc-anchor-seeds", type=str, default=None)
    parser.add_argument(
        "--full-doc-anchor-reference-method-runs",
        type=str,
        default="official_fno official_fno_sumlen",
    )
    parser.add_argument("--efficiency-anchor-mode", choices=("both", "fno_only", "tree_only"), default="both")
    parser.add_argument("--efficiency-train-docs", type=str, default=None)
    parser.add_argument("--efficiency-anchor-train-docs-dense", type=str, default=None)
    parser.add_argument("--efficiency-anchor-seeds", type=str, default=None)
    parser.add_argument("--efficiency-hardness-grid", type=str, default="structural_core_v1")
    parser.add_argument("--efficiency-structural-cells", type=str, default=None)
    parser.add_argument("--oracle-budget-train-docs", type=int, default=None)
    parser.add_argument("--oracle-budget-seeds", type=str, default=None)
    parser.add_argument("--oracle-budget-method-runs", type=str, default=None)
    parser.add_argument("--oracle-budget-reference-method-runs", type=str, default="official_fno official_fno_sumlen")
    parser.add_argument("--oracle-budget-calls-per-doc", type=str, default=None)
    parser.add_argument("--oracle-budget-full-doc-shares", type=str, default=None)
    parser.add_argument("--oracle-budget-doc-consumption-modes", type=str, default=None)
    parser.add_argument("--oracle-budget-local-split-modes", type=str, default=None)
    parser.add_argument("--oracle-budget-tree-config-mode", choices=("parity", "default"), default="parity")
    parser.add_argument("--oracle-budget-capacity-root", type=Path, default=None)
    parser.add_argument(
        "--tree-reference-mode",
        choices=("default", "capacity_locked", "package_capacity_locked", "preset"),
        default="default",
        help="Shared tree reference source for direct tree-study phases.",
    )
    parser.add_argument(
        "--tree-reference-capacity-root",
        type=Path,
        default=None,
        help="Capacity root whose locked winner should define the shared tree reference when --tree-reference-mode=capacity_locked.",
    )
    parser.add_argument(
        "--tree-reference-preset",
        type=str,
        default="",
        help="Named tree reference preset used when --tree-reference-mode=preset.",
    )
    parser.add_argument(
        "--structural-tree-reference-mode",
        choices=("default", "capacity_locked", "package_capacity_locked", "preset"),
        default="default",
        help="Optional structural-only tree reference source for supervision_recovery; defaults to the shared tree reference.",
    )
    parser.add_argument(
        "--structural-tree-reference-capacity-root",
        type=Path,
        default=None,
        help="Capacity root whose locked winner should define the structural-only tree reference when --structural-tree-reference-mode=capacity_locked.",
    )
    parser.add_argument(
        "--structural-tree-reference-preset",
        type=str,
        default="",
        help="Named structural-only tree reference preset used when --structural-tree-reference-mode=preset.",
    )
    parser.add_argument(
        "--one-leaf-tree-reference-mode",
        choices=("default", "preset"),
        default="default",
        help="Optional tree reference override for 1-leaf geometries (leaf_tokens >= doc_tokens). When set to 'preset', uses the canary preset for the 1-leaf point.",
    )
    parser.add_argument(
        "--one-leaf-tree-reference-preset",
        type=str,
        default="",
        help="Named preset used for 1-leaf geometries when --one-leaf-tree-reference-mode=preset.",
    )
    parser.add_argument("--large-batch-batch-sizes", type=str, default=None)
    parser.add_argument("--large-batch-fixed-epochs", type=int, default=5)
    parser.add_argument("--large-batch-target-steps", type=int, default=200)
    parser.add_argument("--large-batch-lrs", type=str, default="0.001 0.002 0.004")
    parser.add_argument("--supervision-train-docs", type=str, default=None)
    parser.add_argument("--supervision-leaf-profiles", type=str, default=None)
    parser.add_argument("--supervision-internal-profiles", type=str, default=None)
    parser.add_argument("--supervision-seeds", type=str, default=None)
    parser.add_argument("--supervision-batch-size", type=int, default=256)
    parser.add_argument(
        "--supervision-recovery-leaf-token-batch-sizes",
        type=str,
        default=None,
        help=(
            "Optional per-leaf-tokens batch-size override for the supervision "
            "recovery sweep. JSON dict (e.g. '{\"16\":8,\"64\":32}') or "
            "semicolon-delimited list (e.g. '16=8;64=32;128=64'). Keys are "
            "fixed_leaf_tokens, values are batch sizes (in docs). Any leaf "
            "size not in the map falls back to --supervision-batch-size."
        ),
    )
    parser.add_argument("--supervision-epochs", type=int, default=10)
    parser.add_argument("--exact-metric-final-doc-limit", type=int, default=0)
    parser.add_argument("--tree-posttrain-train-doc-limit", type=int, default=0)
    parser.add_argument(
        "--tree-training-schedule",
        choices=("single_stage", "two_stage"),
        default=None,
    )
    parser.add_argument("--tree-stage1-epochs", type=int, default=None)
    parser.add_argument("--tree-stage2-epochs", type=int, default=None)
    parser.add_argument("--tree-stage1-artifact-root", type=Path, default=None)
    parser.add_argument(
        "--tree-stage1-resume-if-available",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tree-exact-eval-max-docs", type=int, default=0)
    parser.add_argument("--prepared-data-root", type=Path, default=None)
    parser.add_argument(
        "--prepared-data-allow-create",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diagnostic-detail-mode",
        choices=("summary", "debug_raw"),
        default="summary",
    )
    parser.add_argument("--raw-diagnostic-artifact-dir", type=Path, default=None)
    parser.add_argument("--supervision-min-tokens", type=int, default=64)
    parser.add_argument("--supervision-max-tokens", type=int, default=64)
    parser.add_argument("--supervision-min-segments", type=int, default=2)
    parser.add_argument("--supervision-max-segments", type=int, default=6)
    parser.add_argument("--supervision-fixed-leaf-tokens", type=int, default=8)
    parser.add_argument("--supervision-recovery-train-docs", type=str, default=None)
    parser.add_argument("--supervision-recovery-seeds", type=str, default=None)
    parser.add_argument(
        "--supervision-recovery-depth-discount-gammas",
        type=str,
        default=None,
        help=(
            "Optional comma- or space-separated gamma values for supervision_recovery "
            "depth-discount sweeps."
        ),
    )
    parser.add_argument(
        "--supervision-recovery-packages",
        type=str,
        default=None,
        help=(
            "Optional ordered supervision-recovery package list. "
            "Accepts comma- or space-separated canonical package ids, "
            "public aliases like root100_extra_local10, and group aliases "
            "like comparison_grid_v3 or mass_r100."
        ),
    )
    parser.add_argument(
        "--supervision-recovery-method-id",
        choices=("tree_neural",),
        default=SUPERVISION_RECOVERY_TREE_FAMILY,
    )
    parser.add_argument(
        "--supervision-recovery-recoverable-benchmark",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--supervision-recovery-structural-grid",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--supervision-recovery-structural-cell",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--supervision-recovery-scope-keys",
        type=str,
        default=None,
        help=(
            "Optional comma- or space-separated supervision_recovery scope keys to run. "
            "When omitted, both the recoverable benchmark and the structural cell are run."
        ),
    )
    parser.add_argument(
        "--supervision-recovery-hazard-panel-ids",
        type=str,
        default=None,
        help=(
            "Optional comma- or space-separated Markov hazard panel ids to expose as "
            "supervision_recovery scopes."
        ),
    )
    parser.add_argument(
        "--supervision-recovery-hazard-panel-bundle-map",
        type=str,
        default=None,
        help=(
            "Optional panel_id=base_bundle.json mapping for hazard panel scopes. "
            "Separate entries with semicolons, commas, or spaces."
        ),
    )
    parser.add_argument(
        "--supervision-recovery-leaf-token-ladder",
        type=str,
        default=None,
        help=(
            "Optional explicit leaf-token ladder for supervision_recovery. "
            "Accepts comma- or space-separated values. When set, the caller's "
            "fixed_leaf_tokens survive tree-reference preset overrides."
        ),
    )
    parser.add_argument(
        "--supervision-recovery-package-leaf-token-overrides",
        type=str,
        default=None,
        help=(
            "Optional package-specific leaf-token ladders. Accepts either a JSON "
            "object like '{\"root_ladder_deciles\": [64,32,16,8]}' or a "
            "semicolon-separated string like "
            "'root_ladder_deciles=64,32,16,8;mass_preserving_leaf_only_deciles=64,32,16,8'."
        ),
    )
    parser.add_argument(
        "--tree-batch-autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--runtime-data-mode", choices=("resident", "cpu_debug"), default="resident")
    parser.add_argument(
        "--runtime-bucket-mode",
        choices=("exact_then_bucketed", "leaf_count_auto_queue"),
        default="exact_then_bucketed",
    )
    parser.add_argument(
        "--runtime-tree-batch-structural-pad-limit",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--runtime-tree-batch-auto-queue-min-docs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--runtime-tree-batch-auto-queue-min-fill-ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument("--runtime-preload-splits", type=str, default="train val test")
    parser.add_argument("--runtime-preload-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-workers-per-mig", type=int, default=1)
    parser.add_argument("--runtime-allow-multi-worker-screen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-capacity-workers-per-mig", type=int, default=2)
    parser.add_argument("--scheduler-mode", choices=("global_per_run",), default="global_per_run")
    parser.add_argument("--default-job-granularity", choices=("family_train_seed", "family_train"), default="family_train_seed")
    parser.add_argument("--cleanup-stale-children", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gpu-items-per-mig", type=int, default=1)

    if config_defaults:
        valid_dests = {action.dest for action in parser._actions}
        normalized = {
            str(key): _stringify_cli_default(value)
            for key, value in dict(config_defaults).items()
            if str(key) in valid_dests
        }
        if normalized:
            parser.set_defaults(**normalized)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    meta_args = _preparse_config_args(raw_argv)
    config_defaults = _load_selection_config(
        meta_args.selection_config,
        section_names=("tradeoff_pipeline", "markov_tradeoff_pipeline"),
    )
    parser = _build_parser(config_defaults=config_defaults)
    return parser.parse_args(raw_argv)


_safe_float = safe_float
_safe_int = safe_int


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_mig_devices() -> List[str]:
    return [str(row.uuid) for row in _detect_mig_inventory() if str(row.uuid).strip()]


def _detect_available_mig_devices() -> List[str]:
    return _filter_available_mig_devices(
        _detect_mig_inventory(),
        min_free_fraction=0.8,
        max_used_mib=1024,
    )


def _resolve_devices(args: argparse.Namespace) -> List[str]:
    mode = str(args.device_mode).strip().lower()
    if mode == "cpu":
        return [""]
    explicit = [item.strip() for item in str(args.migs or "").replace(",", " ").split() if item.strip()]
    if explicit:
        devices = explicit
    else:
        devices = _detect_available_mig_devices()
        if not devices:
            devices = _detect_mig_devices()
    if not devices:
        return [""] if mode == "auto" else []
    max_workers = int(args.max_workers)
    if max_workers > 0:
        return devices[:max_workers]
    return devices


def _phase_set(text: str) -> set[str]:
    return {item.strip() for item in str(text).replace(",", " ").split() if item.strip()}


def _default_output_subdir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _phase_execution_root(root: Path, phase: str) -> Path:
    if root.parent.name == "attempts" and root.parent.parent.name == phase:
        root.mkdir(parents=True, exist_ok=True)
        return root
    return _default_output_subdir(root, phase)


def _version_root_for_phase_root(root: Path, phase: str) -> Path:
    if root.parent.name == "attempts" and root.parent.parent.name == phase:
        return root.parents[2]
    return root


def _report_version_manifest_path(output_root: Path) -> Path:
    return output_root / REPORT_VERSION_MANIFEST_NAME


def _new_attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _phase_attempt_root(output_root: Path, phase: str, attempt_id: str) -> Path:
    path = output_root / phase / "attempts" / str(attempt_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), indent=None, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_config_fingerprint(args: argparse.Namespace, phase: str) -> str:
    resolved = _resolved_tradeoff_selection(args)
    return _stable_fingerprint({"phase": str(phase), "selection": resolved})


def _canonical_alias_path(output_root: Path, source_key: str) -> Path:
    spec = REPORT_SOURCE_SPECS.get(str(source_key))
    if spec is None:
        raise KeyError(f"unknown report source key: {source_key}")
    return output_root / str(spec["alias_relpath"])


def _copy_selected_artifact(src: Path, dst: Path) -> None:
    src = src.expanduser()
    dst = dst.expanduser()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _empty_selected_source_entry(key: str) -> Dict[str, Any]:
    spec = REPORT_SOURCE_SPECS.get(str(key), {})
    return {
        "relpath": "",
        "origin": "",
        "phase": str(spec.get("phase", "")),
        "sha256": "",
        "config_fingerprint": "",
        "status": "missing",
        "reason": "no local artifact selected",
        "selected_attempt_id": "",
    }


def _load_report_version_manifest(output_root: Path) -> Dict[str, Any]:
    path = _report_version_manifest_path(output_root)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": REPORT_VERSION_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version_root": str(output_root),
            "selected_sources": {},
            "phase_attempts": {},
            "report_outputs": {},
        }
    manifest["schema_version"] = int(manifest.get("schema_version", REPORT_VERSION_SCHEMA_VERSION))
    manifest["version_root"] = str(output_root)
    selected_sources = dict(manifest.get("selected_sources") or {})
    for key in REPORT_SOURCE_SPECS:
        selected_sources.setdefault(key, _empty_selected_source_entry(key))
    manifest["selected_sources"] = selected_sources
    manifest["phase_attempts"] = dict(manifest.get("phase_attempts") or {})
    manifest["report_outputs"] = dict(manifest.get("report_outputs") or {})
    return manifest


def _write_report_version_manifest(output_root: Path, manifest: Mapping[str, Any]) -> Path:
    path = _report_version_manifest_path(output_root)
    _write_json(path, dict(manifest))
    return path


def _record_phase_attempt(
    manifest: Dict[str, Any],
    *,
    phase: str,
    attempt_id: str,
    config_fingerprint: str,
    summary_path: Path,
    log_path: Path | None = None,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    phase_attempts = dict(manifest.get("phase_attempts") or {})
    phase_payload = dict(phase_attempts.get(str(phase)) or {})
    attempts = dict(phase_payload.get("attempts") or {})
    payload = {
        "attempt_id": str(attempt_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": str(config_fingerprint),
        "summary_relpath": str(summary_path.relative_to(Path(manifest["version_root"]))),
        "log_relpath": (
            str(log_path.relative_to(Path(manifest["version_root"])))
            if log_path is not None and log_path.exists()
            else ""
        ),
        "status": str(status),
    }
    if extra:
        payload.update(dict(extra))
    attempts[str(attempt_id)] = payload
    phase_payload["attempts"] = attempts
    phase_attempts[str(phase)] = phase_payload
    manifest["phase_attempts"] = phase_attempts


def _select_manifest_source(
    manifest: Dict[str, Any],
    *,
    output_root: Path,
    source_key: str,
    artifact_path: Path,
    origin: str,
    phase: str,
    status: str,
    reason: str,
    selected_attempt_id: str = "",
    config_fingerprint: str = "",
    extra: Mapping[str, Any] | None = None,
) -> None:
    entry = _empty_selected_source_entry(source_key)
    entry.update(
        {
            "relpath": str(artifact_path.relative_to(output_root)),
            "origin": str(origin),
            "phase": str(phase),
            "sha256": _sha256_path(artifact_path) if artifact_path.exists() else "",
            "config_fingerprint": str(config_fingerprint or ""),
            "status": str(status),
            "reason": str(reason or ""),
            "selected_attempt_id": str(selected_attempt_id or ""),
        }
    )
    if extra:
        entry.update(dict(extra))
    selected_sources = dict(manifest.get("selected_sources") or {})
    selected_sources[str(source_key)] = entry
    manifest["selected_sources"] = selected_sources


def _refresh_selected_source_statuses(
    manifest: Dict[str, Any],
    *,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    active_phases = _phase_set(getattr(args, "phases", None))
    refresh_existing = bool(getattr(args, "refresh_existing_output_root", False))
    selected_sources = dict(manifest.get("selected_sources") or {})
    for key, raw_entry in selected_sources.items():
        entry = dict(raw_entry or {})
        relpath = _gs(entry, "relpath")
        artifact_path = output_root / relpath if relpath else None
        previous_status = str(entry.get("status", "") or "")
        if artifact_path is None or not artifact_path.exists():
            entry["status"] = "missing"
            entry["reason"] = "selected artifact is missing from this version root"
            entry["sha256"] = ""
            selected_sources[str(key)] = entry
            continue
        current_sha = _sha256_path(artifact_path)
        if str(entry.get("origin", "")) == "rerun":
            phase = str(entry.get("phase", "") or REPORT_SOURCE_SPECS.get(str(key), {}).get("phase", ""))
            expected_fingerprint = (
                _phase_config_fingerprint(args, phase)
                if phase and str(phase) in active_phases and not refresh_existing
                else ""
            )
            if expected_fingerprint and str(entry.get("config_fingerprint", "")) != expected_fingerprint:
                entry["status"] = "stale"
                entry["reason"] = "stored config fingerprint does not match the current phase selection"
            elif str(entry.get("sha256", "")) and str(entry.get("sha256", "")) != current_sha:
                entry["status"] = "stale"
                entry["reason"] = "selected artifact contents changed after selection"
            elif previous_status not in {"incompatible", "suspicious", "unavailable"}:
                entry["status"] = "ready"
                entry["reason"] = ""
        elif str(entry.get("origin", "")) == "staged_copy":
            if str(entry.get("sha256", "")) and str(entry.get("sha256", "")) != current_sha:
                entry["status"] = "stale"
                entry["reason"] = "staged copy contents changed after staging"
            elif previous_status not in {"incompatible", "suspicious", "unavailable"}:
                entry["status"] = "ready"
                entry["reason"] = ""
        entry["sha256"] = current_sha
        selected_sources[str(key)] = entry
    manifest["selected_sources"] = selected_sources


def _parse_report_source_overrides(values: Sequence[str] | None) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for raw in list(values or []):
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"invalid --report-source {text!r}; expected key=path with key in {', '.join(sorted(REPORT_SOURCE_SPECS))}"
            )
        key, raw_path = text.split("=", 1)
        key = str(key).strip()
        if key not in REPORT_SOURCE_SPECS:
            raise ValueError(
                f"unknown --report-source key {key!r}; valid keys are {', '.join(sorted(REPORT_SOURCE_SPECS))}"
            )
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"--report-source {key} path does not exist: {path}")
        overrides[key] = path
    return overrides


def _stage_report_sources(
    *,
    output_root: Path,
    manifest: Dict[str, Any],
    overrides: Mapping[str, Path],
) -> None:
    if not overrides:
        return
    for key, src_path in sorted(overrides.items()):
        stage_id = _new_attempt_id()
        staged_path = output_root / "report_sources" / str(key) / str(stage_id) / src_path.name
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, staged_path)
        _select_manifest_source(
            manifest,
            output_root=output_root,
            source_key=str(key),
            artifact_path=staged_path,
            origin="staged_copy",
            phase=str(REPORT_SOURCE_SPECS[str(key)]["phase"]),
            status="ready",
            reason="",
            selected_attempt_id="",
            extra={"staged_from": str(src_path)},
        )


def _register_phase_source(
    manifest: Dict[str, Any],
    *,
    output_root: Path,
    phase: str,
    source_key: str,
    attempt_id: str,
    config_fingerprint: str,
    artifact_path: Path,
    alias_path: Path | None = None,
    log_path: Path | None = None,
    extra_attempt: Mapping[str, Any] | None = None,
    extra_source: Mapping[str, Any] | None = None,
) -> None:
    if alias_path is not None and artifact_path.exists():
        _copy_selected_artifact(artifact_path, alias_path)
    _record_phase_attempt(
        manifest,
        phase=phase,
        attempt_id=attempt_id,
        config_fingerprint=config_fingerprint,
        summary_path=artifact_path,
        log_path=log_path,
        status="ready" if artifact_path.exists() else "missing",
        extra=extra_attempt,
    )
    _select_manifest_source(
        manifest,
        output_root=output_root,
        source_key=source_key,
        artifact_path=artifact_path,
        origin="rerun",
        phase=phase,
        status="ready" if artifact_path.exists() else "missing",
        reason="" if artifact_path.exists() else "phase completed without the expected artifact",
        selected_attempt_id=attempt_id,
        config_fingerprint=config_fingerprint,
        extra=extra_source,
    )


def _register_report_outputs(
    manifest: Dict[str, Any],
    *,
    output_root: Path,
    attempt_id: str,
    config_fingerprint: str,
    attempt_root: Path,
) -> None:
    summary_path = attempt_root / "summary.json"
    markdown_path = attempt_root / "report.md"
    pdf_path = attempt_root / "report.pdf"
    for path in (summary_path, markdown_path, pdf_path):
        if path.exists():
            _copy_selected_artifact(path, output_root / "tradeoff_report" / path.name)
    report_outputs = dict(manifest.get("report_outputs") or {})
    attempts = dict(report_outputs.get("attempts") or {})
    attempts[str(attempt_id)] = {
        "attempt_id": str(attempt_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": str(config_fingerprint),
        "summary_relpath": str(summary_path.relative_to(output_root)) if summary_path.exists() else "",
        "markdown_relpath": str(markdown_path.relative_to(output_root)) if markdown_path.exists() else "",
        "pdf_relpath": str(pdf_path.relative_to(output_root)) if pdf_path.exists() else "",
        "status": "ready" if summary_path.exists() and pdf_path.exists() else "missing",
    }
    report_outputs.update(
        {
            "selected_attempt_id": str(attempt_id),
            "summary": str((output_root / "tradeoff_report" / "summary.json")),
            "markdown": str((output_root / "tradeoff_report" / "report.md")),
            "pdf": str((output_root / "tradeoff_report" / "report.pdf")),
            "attempts": attempts,
        }
    )
    manifest["report_outputs"] = report_outputs


def _resolved_tradeoff_selection(args: argparse.Namespace) -> Dict[str, Any]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    phases = sorted(_phase_set(args.phases))
    train_docs, val_docs, test_docs = _law_phase_doc_counts(args)
    tree_reference = _resolve_tree_reference(args)
    structural_tree_reference = _resolve_tree_reference(
        args,
        prefix="structural_tree_reference",
        fallback=tree_reference,
    )
    one_leaf_tree_reference = _resolve_tree_reference(
        args,
        prefix="one_leaf_tree_reference",
        fallback=None,
    )
    recoverable_one_leaf_root_only_reference = _tree_reference_from_preset_name(
        ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
    )
    structural_one_leaf_root_only_reference = _tree_reference_from_preset_name(
        STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
    )
    return {
        "preset": str(args.preset),
        "phases": phases,
        "selection_config": str(args.selection_config) if getattr(args, "selection_config", None) else "",
        "device_mode": str(args.device_mode),
        "max_workers": int(args.max_workers),
        "train_docs": int(args.train_docs),
        "val_docs": int(args.val_docs),
        "test_docs": int(args.test_docs),
        "fixed_leaf_tokens": int(args.fixed_leaf_tokens),
        "seed": int(args.seed),
        "data_seeds": _parse_int_list(args.data_seeds, [0, 1]),
        "batch_sizes": _parse_int_list(args.batch_sizes, preset["batch_sizes"]),
        "medium_batch_sizes": _parse_int_list(args.medium_batch_sizes, preset["medium_batch_sizes"]),
        "medium_seeds": _parse_int_list(args.medium_seeds, preset["medium_seeds"]),
        "docs_epochs_train_docs": _parse_int_list(args.docs_epochs_train_docs, preset["docs_epochs_train_docs"]),
        "docs_epochs_epochs": _parse_int_list(args.docs_epochs_epochs, preset["docs_epochs_epochs"]),
        "learnability_train_docs": _parse_int_list(args.learnability_train_docs, preset["learnability_train_docs"]),
        "learnability_weights": _parse_float_list(args.learnability_weights, preset["learnability_weights"]),
        "learnability_profiles": _parse_str_list(args.learnability_profiles, preset["learnability_profiles"]),
        "weight_ablation_train_docs": _parse_int_list(args.weight_ablation_train_docs, preset["weight_ablation_train_docs"]),
        "weight_ablation_profiles": _parse_str_list(args.weight_ablation_profiles, preset["weight_ablation_profiles"]),
        "law_set_ids": _parse_str_list(
            getattr(args, "law_set_ids", None),
            [LAW_SET_ROOT_ONLY, LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY, LAW_SET_ALL],
        ),
        "full_doc_anchor_train_docs": _parse_int_list(args.full_doc_anchor_train_docs, preset["full_doc_anchor_train_docs"]),
        "full_doc_anchor_seeds": _parse_int_list(args.full_doc_anchor_seeds, preset["full_doc_anchor_seeds"]),
        "full_doc_anchor_reference_method_runs": _parse_run_axis_list(
            args.full_doc_anchor_reference_method_runs,
            DEFAULT_REFERENCE_METHOD_RUNS,
            role="reference",
        ),
        "efficiency_anchor_mode": str(getattr(args, "efficiency_anchor_mode", preset["efficiency_anchor_mode"])),
        "efficiency_train_docs": _parse_int_list(
            getattr(args, "efficiency_train_docs", None),
            preset["efficiency_train_docs"],
        ),
        "efficiency_anchor_train_docs_dense": _parse_int_list(
            getattr(args, "efficiency_anchor_train_docs_dense", None),
            preset["efficiency_anchor_train_docs_dense"],
        ),
        "efficiency_anchor_seeds": _parse_int_list(
            getattr(args, "efficiency_anchor_seeds", None),
            preset["efficiency_anchor_seeds"],
        ),
        "efficiency_hardness_grid": str(
            getattr(args, "efficiency_hardness_grid", preset["efficiency_hardness_grid"])
        ),
        "efficiency_structural_cells": _parse_str_list(
            getattr(args, "efficiency_structural_cells", None),
            preset["efficiency_structural_cells"],
        ),
        "oracle_budget_train_docs": int(
            args.oracle_budget_train_docs
            if args.oracle_budget_train_docs is not None
            else preset["oracle_budget_train_docs"]
        ),
        "oracle_budget_seeds": _parse_int_list(args.oracle_budget_seeds, preset["oracle_budget_seeds"]),
        "oracle_budget_method_runs": _parse_run_axis_list(
            args.oracle_budget_method_runs,
            preset["oracle_budget_method_runs"],
            role="primary",
        ),
        "oracle_budget_reference_method_runs": _parse_run_axis_list(
            args.oracle_budget_reference_method_runs,
            preset["oracle_budget_reference_method_runs"],
            role="reference",
        ),
        "oracle_budget_calls_per_doc": _parse_float_list(
            args.oracle_budget_calls_per_doc,
            preset["oracle_budget_calls_per_doc"],
        ),
        "oracle_budget_full_doc_shares": _parse_float_list(
            args.oracle_budget_full_doc_shares,
            preset["oracle_budget_full_doc_shares"],
        ),
        "oracle_budget_doc_consumption_modes": _parse_str_list(
            args.oracle_budget_doc_consumption_modes,
            preset["oracle_budget_doc_consumption_modes"],
        ),
        "oracle_budget_local_split_modes": _parse_str_list(
            args.oracle_budget_local_split_modes,
            preset["oracle_budget_local_split_modes"],
        ),
        "oracle_budget_tree_config_mode": str(args.oracle_budget_tree_config_mode),
        "oracle_budget_capacity_root": (
            str(args.oracle_budget_capacity_root.expanduser())
            if getattr(args, "oracle_budget_capacity_root", None) is not None
            else ""
        ),
        "tree_reference": tree_reference,
        "structural_tree_reference": structural_tree_reference,
        "one_leaf_tree_reference": one_leaf_tree_reference,
        "large_batch_batch_sizes": _parse_int_list(args.large_batch_batch_sizes, preset["large_batch_batch_sizes"]),
        "large_batch_fixed_epochs": int(args.large_batch_fixed_epochs),
        "large_batch_target_steps": int(args.large_batch_target_steps),
        "large_batch_lrs": _parse_float_list(args.large_batch_lrs, [1e-3, 2e-3, 4e-3]),
        "supervision_train_docs": _parse_int_list(
            getattr(args, "supervision_train_docs", None),
            preset["supervision_train_docs"],
        ),
        "supervision_leaf_profiles": _parse_str_list(
            getattr(args, "supervision_leaf_profiles", None),
            preset["supervision_leaf_profiles"],
        ),
        "supervision_internal_profiles": _parse_str_list(
            getattr(args, "supervision_internal_profiles", None),
            preset["supervision_internal_profiles"],
        ),
        "supervision_seeds": _parse_int_list(
            getattr(args, "supervision_seeds", None),
            preset["supervision_seeds"],
        ),
        "supervision_recovery_train_docs": _parse_int_list(
            getattr(args, "supervision_recovery_train_docs", None),
            preset["supervision_recovery_train_docs"],
        ),
        "supervision_recovery_seeds": _parse_int_list(
            getattr(args, "supervision_recovery_seeds", None),
            preset["supervision_recovery_seeds"],
        ),
        "supervision_recovery_depth_discount_gammas": _parse_float_list(
            getattr(args, "supervision_recovery_depth_discount_gammas", None),
            preset["supervision_recovery_depth_discount_gammas"],
        ),
        "supervision_recovery_leaf_token_ladder": _resolved_supervision_recovery_leaf_token_ladder(args),
        "supervision_recovery_package_leaf_token_overrides": _resolved_supervision_recovery_package_leaf_token_overrides(args),
        "tree_exact_eval_max_docs": int(
            getattr(args, "tree_exact_eval_max_docs", 0)
        ),
        "prepared_data_root": _optional_path_text(
            getattr(args, "prepared_data_root", None)
        ),
        "prepared_data_allow_create": bool(
            getattr(args, "prepared_data_allow_create", True)
        ),
        "diagnostic_detail_mode": str(
            getattr(args, "diagnostic_detail_mode", "summary")
        ),
        "raw_diagnostic_artifact_dir": _optional_path_text(
            getattr(args, "raw_diagnostic_artifact_dir", None)
        ),
        "supervision_recovery_packages": _resolved_supervision_recovery_package_order(args),
        "supervision_recovery_method_id": str(
            getattr(
                args,
                "supervision_recovery_method_id",
                preset["supervision_recovery_method_id"],
            )
        ),
        "supervision_recovery_scope_keys": list(
            _supervision_recovery_scope_keys(args)
        ),
        "supervision_recovery_hazard_panel_ids": _parse_str_list(
            getattr(args, "supervision_recovery_hazard_panel_ids", None),
            (),
        ),
        "supervision_recovery_hazard_panel_bundle_map": _parse_key_value_text_map(
            getattr(args, "supervision_recovery_hazard_panel_bundle_map", None)
        ),
        "supervision_recovery_recoverable_benchmark": _supervision_recovery_recoverable_benchmark_name(args),
        "supervision_recovery_structural_grid": _supervision_recovery_structural_grid_name(args),
        "supervision_recovery_structural_cell": _supervision_recovery_structural_cell_name(args),
        "support_leaf_tokens": _parse_int_list(args.support_leaf_tokens, preset["support_leaf_tokens"]),
        "support_seeds": _parse_int_list(args.support_seeds, preset["support_seeds"]),
        "support_modes": _parse_str_list(getattr(args, "support_modes", None), SUPPORTED_SUPPORT_MODES),
        "runtime": {
            "data_mode": str(getattr(args, "runtime_data_mode", "resident")),
            "bucket_mode": str(
                getattr(args, "runtime_bucket_mode", "exact_then_bucketed")
            ),
            "tree_batch_structural_pad_limit": float(
                getattr(args, "runtime_tree_batch_structural_pad_limit", 0.5)
            ),
            "tree_batch_auto_queue_min_docs": int(
                getattr(args, "runtime_tree_batch_auto_queue_min_docs", 8)
            ),
            "tree_batch_auto_queue_min_fill_ratio": float(
                getattr(args, "runtime_tree_batch_auto_queue_min_fill_ratio", 0.5)
            ),
            "preload_splits": _parse_str_list(
                getattr(args, "runtime_preload_splits", None),
                ("train", "val", "test"),
            ),
            "preload_targets": bool(
                getattr(args, "runtime_preload_targets", True)
            ),
            "workers_per_mig": int(
                getattr(args, "runtime_workers_per_mig", 1)
            ),
            "allow_multi_worker_screen": bool(
                getattr(args, "runtime_allow_multi_worker_screen", True)
            ),
            "capacity_workers_per_mig": int(
                getattr(args, "runtime_capacity_workers_per_mig", 2)
            ),
        },
        "scheduler": {
            "mode": str(getattr(args, "scheduler_mode", "global_per_run")),
            "default_job_granularity": str(
                getattr(args, "default_job_granularity", "family_train_seed")
            ),
            "cleanup_stale_children": bool(
                getattr(args, "cleanup_stale_children", True)
            ),
            "max_gpu_items_per_mig": int(
                getattr(args, "max_gpu_items_per_mig", 1)
            ),
        },
        "law_phase_doc_counts": {
            "train_docs": int(train_docs),
            "val_docs": int(val_docs),
            "test_docs": int(test_docs),
        },
    }


def build_run_plan(
    args: argparse.Namespace,
    *,
    devices: Sequence[str] | None = None,
) -> Dict[str, Any]:
    resolved = _resolved_tradeoff_selection(args)
    phases = set(resolved["phases"])
    if "supervision_recovery" in phases:
        _validate_supervision_recovery_tree_setup(args)
    device_list = list(devices) if devices is not None else _resolve_devices(args)
    phase_counts: Dict[str, Dict[str, Any]] = {}
    total_worker_tasks = 0

    def _record(name: str, task_count: int, details: Mapping[str, Any], output: str) -> None:
        nonlocal total_worker_tasks
        phase_counts[name] = {
            "worker_tasks": int(task_count),
            "details": dict(details),
            "summary_output": output,
        }
        total_worker_tasks += int(task_count)

    output_root = Path(args.output_root).expanduser()
    data_seed_count = len(resolved["data_seeds"])
    if "batch_timing" in phases:
        _record(
            "batch_timing",
            len(resolved["batch_sizes"]),
            {"batch_sizes": resolved["batch_sizes"]},
            str(output_root / "batch_timing" / "markov_fixed_fused_leaflaws_batchsize_timing_fullpipeline.json"),
        )
    if "medium_grid" in phases:
        _record(
            "medium_grid",
            len(resolved["medium_batch_sizes"]) * len(resolved["medium_seeds"]),
            {
                "batch_sizes": resolved["medium_batch_sizes"],
                "seeds": resolved["medium_seeds"],
            },
            str(output_root / "medium_grid" / "aggregate_summary.json"),
        )
    if "docs_epochs" in phases:
        _record(
            "docs_epochs",
            len(resolved["docs_epochs_train_docs"]) * len(resolved["docs_epochs_epochs"]),
            {
                "train_docs": resolved["docs_epochs_train_docs"],
                "epochs": resolved["docs_epochs_epochs"],
            },
            str(output_root / "docs_epochs" / "aggregate_summary.json"),
        )
    if "learnability" in phases:
        _record(
            "learnability",
            len(resolved["learnability_train_docs"])
            * len(resolved["learnability_weights"])
            * len(resolved["learnability_profiles"])
            * data_seed_count,
            {
                "train_docs": resolved["learnability_train_docs"],
                "weights": resolved["learnability_weights"],
                "profiles": resolved["learnability_profiles"],
                "data_seeds": resolved["data_seeds"],
            },
            str(output_root / "learnability_report" / "learnability_summary.json"),
        )
    if "weight_ablation" in phases:
        _record(
            "weight_ablation",
            len(resolved["weight_ablation_train_docs"])
            * len(resolved["weight_ablation_profiles"])
            * data_seed_count,
            {
                "train_docs": resolved["weight_ablation_train_docs"],
                "profiles": resolved["weight_ablation_profiles"],
                "data_seeds": resolved["data_seeds"],
            },
            str(output_root / "weight_ablation_runs" / "weight_ablation_summary.json"),
        )
    if "law_packages" in phases:
        _record(
            "law_packages",
            len(resolved["law_set_ids"]),
            {
                "law_set_ids": resolved["law_set_ids"],
                "doc_counts": resolved["law_phase_doc_counts"],
            },
            str(output_root / "law_packages" / "fno_tree_law_comparison.json"),
        )
    if "full_doc_anchor" in phases:
        _record(
            "full_doc_anchor",
            len(resolved["full_doc_anchor_train_docs"])
            * len(resolved["full_doc_anchor_seeds"])
            * len(resolved["full_doc_anchor_reference_method_runs"]),
            {
                "train_docs": resolved["full_doc_anchor_train_docs"],
                "seeds": resolved["full_doc_anchor_seeds"],
                "reference_method_runs": resolved["full_doc_anchor_reference_method_runs"],
            },
            str(output_root / "full_doc_anchor" / "full_doc_fno_upper_bound_summary.json"),
        )
    if "oracle_budget_frontier" in phases:
        budget_jobs = 0
        for full_doc_share in resolved["oracle_budget_full_doc_shares"]:
            doc_modes = list(resolved["oracle_budget_doc_consumption_modes"])
            if float(full_doc_share) <= 0.0:
                doc_modes = ["root_only"]
            local_split_modes = list(resolved["oracle_budget_local_split_modes"])
            if abs(float(full_doc_share) - 1.0) <= 1e-12:
                local_split_modes = ["balanced"]
                budget_jobs += len(resolved["oracle_budget_reference_method_runs"])
            budget_jobs += (
                len(resolved["oracle_budget_method_runs"])
                * len(doc_modes)
                * len(local_split_modes)
            )
        _record(
            "oracle_budget_frontier",
            budget_jobs * len(resolved["oracle_budget_seeds"]) * len(resolved["oracle_budget_calls_per_doc"]),
            {
                "train_docs": int(resolved["oracle_budget_train_docs"]),
                "seeds": resolved["oracle_budget_seeds"],
                "method_runs": resolved["oracle_budget_method_runs"],
                "reference_method_runs": resolved["oracle_budget_reference_method_runs"],
                "budget_calls_per_doc": resolved["oracle_budget_calls_per_doc"],
                "full_doc_budget_shares": resolved["oracle_budget_full_doc_shares"],
                "doc_consumption_modes": resolved["oracle_budget_doc_consumption_modes"],
                "local_split_modes": resolved["oracle_budget_local_split_modes"],
                "tree_config_mode": resolved["oracle_budget_tree_config_mode"],
            },
            str(output_root / "oracle_budget_frontier" / "tree_oracle_budget_frontier_summary.json"),
        )
    if "efficiency_suite" in phases:
        recoverable_families = _efficiency_anchor_families(
            mode=str(resolved["efficiency_anchor_mode"]),
            structural=False,
        )
        structural_families = _efficiency_anchor_families(
            mode=str(resolved["efficiency_anchor_mode"]),
            structural=True,
        )
        dense_anchor_tasks = (
            len(resolved["efficiency_anchor_train_docs_dense"])
            * len(resolved["efficiency_anchor_seeds"])
            * len(recoverable_families)
        )
        structural_anchor_docs = [
            int(train_docs)
            for train_docs in resolved["efficiency_anchor_train_docs_dense"]
            if int(train_docs) >= 1024
        ] or [1024]
        structural_anchor_tasks = (
            len(structural_anchor_docs)
            * len(resolved["efficiency_structural_cells"])
            * len(resolved["efficiency_anchor_seeds"])
            * len(structural_families)
        )
        budget_jobs = 0
        for full_doc_share in resolved["oracle_budget_full_doc_shares"]:
            doc_modes = list(resolved["oracle_budget_doc_consumption_modes"])
            if float(full_doc_share) <= 0.0:
                doc_modes = ["root_only"]
            local_split_modes = list(resolved["oracle_budget_local_split_modes"])
            if abs(float(full_doc_share) - 1.0) <= 1e-12:
                local_split_modes = ["balanced"]
                budget_jobs += len(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)
            budget_jobs += len(EFFICIENCY_TREE_BASELINE_FAMILIES) * len(doc_modes) * len(local_split_modes)
        recoverable_budget_tasks = budget_jobs * len(resolved["efficiency_train_docs"]) * len(resolved["oracle_budget_seeds"])
        structural_budget_tasks = (
            budget_jobs
            * len(resolved["efficiency_train_docs"])
            * len(resolved["oracle_budget_seeds"])
            * max(1, len(resolved["efficiency_structural_cells"]))
        )
        _record(
            "efficiency_suite",
            dense_anchor_tasks + structural_anchor_tasks + recoverable_budget_tasks + structural_budget_tasks,
            {
                "anchor_mode": resolved["efficiency_anchor_mode"],
                "dense_anchor_train_docs": resolved["efficiency_anchor_train_docs_dense"],
                "budget_train_docs": resolved["efficiency_train_docs"],
                "anchor_seeds": resolved["efficiency_anchor_seeds"],
                "hardness_grid": resolved["efficiency_hardness_grid"],
                "structural_cells": resolved["efficiency_structural_cells"],
                "oracle_budget_calls_per_doc": resolved["oracle_budget_calls_per_doc"],
                "oracle_budget_full_doc_shares": resolved["oracle_budget_full_doc_shares"],
            },
            str(output_root / "efficiency_suite" / "summary.json"),
        )
    if "large_batch_diagnosis" in phases:
        task_count = 2 * len(resolved["large_batch_batch_sizes"]) + len(resolved["large_batch_lrs"])
        _record(
            "large_batch_diagnosis",
            task_count,
            {
                "batch_sizes": resolved["large_batch_batch_sizes"],
                "fixed_epochs": resolved["large_batch_fixed_epochs"],
                "target_steps": resolved["large_batch_target_steps"],
                "lrs": resolved["large_batch_lrs"],
            },
            str(output_root / "large_batch_diagnosis" / "aggregate_summary.json"),
        )
    if "supervision_sweep" in phases:
        task_count = (
            len(resolved["supervision_train_docs"])
            * len(resolved["supervision_leaf_profiles"])
            * len(resolved["supervision_internal_profiles"])
            * len(resolved["supervision_seeds"])
        )
        _record(
            "supervision_sweep",
            task_count,
            {
                "train_docs": resolved["supervision_train_docs"],
                "leaf_profiles": resolved["supervision_leaf_profiles"],
                "internal_profiles": resolved["supervision_internal_profiles"],
                "seeds": resolved["supervision_seeds"],
            },
            str(output_root / "supervision_sweep" / "supervision_sweep_summary.json"),
        )
    if "supervision_recovery" in phases:
        resolved_package_order = list(resolved["supervision_recovery_packages"])
        leaf_token_ladder = list(
            resolved.get("supervision_recovery_leaf_token_ladder") or []
        )
        package_leaf_token_overrides = {
            str(key): [
                int(value)
                for value in list(values or [])
                if int(value) > 0
            ]
            for key, values in dict(
                resolved.get("supervision_recovery_package_leaf_token_overrides")
                or {}
            ).items()
        }
        depth_discount_gammas = [
            float(value)
            for value in list(
                resolved.get("supervision_recovery_depth_discount_gammas") or [1.0]
            )
        ]
        gamma_count = max(1, len(depth_discount_gammas))
        fno_package_count = sum(
            1
            for package_name in resolved_package_order
            if bool(SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name]["run_fno"])
        )
        tree_geometry_count = sum(
            max(
                1,
                len(
                    list(
                        package_leaf_token_overrides.get(
                            str(package_name),
                            leaf_token_ladder,
                        )
                        or []
                    )
                ),
            )
            for package_name in resolved_package_order
        )
        plan_scope_specs = _supervision_recovery_scope_specs(
            recoverable_benchmark=str(
                resolved["supervision_recovery_recoverable_benchmark"]
            ),
            structural_grid=str(resolved["supervision_recovery_structural_grid"]),
            structural_cell=str(resolved["supervision_recovery_structural_cell"]),
            requested_scope_keys=list(
                resolved.get("supervision_recovery_scope_keys") or []
            ),
            hazard_panel_ids=list(
                resolved.get("supervision_recovery_hazard_panel_ids") or []
            ),
            hazard_panel_bundle_map=dict(
                resolved.get("supervision_recovery_hazard_panel_bundle_map") or {}
            ),
        )
        scope_count = len(plan_scope_specs)
        supervision_min_tokens = int(
            resolved.get(
                "supervision_min_tokens",
                getattr(args, "supervision_min_tokens", 0),
            )
            or 0
        )
        supervision_max_tokens = int(
            resolved.get(
                "supervision_max_tokens",
                getattr(args, "supervision_max_tokens", 0),
            )
            or 0
        )
        assumed_doc_tokens = (
            int(supervision_max_tokens)
            if int(supervision_min_tokens) == int(supervision_max_tokens)
            else 0
        )
        task_count = (
            len(resolved["supervision_recovery_train_docs"])
            * len(resolved["supervision_recovery_seeds"])
            * scope_count
            * gamma_count
            * (
                fno_package_count
                + tree_geometry_count
            )
        )
        _plan_tree_reference = _resolve_tree_reference(args)
        _plan_structural_tree_reference = _resolve_tree_reference(
            args,
            prefix="structural_tree_reference",
            fallback=_plan_tree_reference,
        )
        _plan_one_leaf_tree_reference = _resolve_tree_reference(
            args,
            prefix="one_leaf_tree_reference",
            fallback=None,
        )
        _parity_ref_active = bool(
            (
                _plan_one_leaf_tree_reference
                and _plan_one_leaf_tree_reference.get("config")
            )
            or _tree_reference_label(_plan_tree_reference)
            == UNIFIED_G_FNO_PARITY_CANARY_PRESET
            or _tree_reference_label(_plan_structural_tree_reference)
            == UNIFIED_G_FNO_PARITY_CANARY_PRESET
        )
        _record(
            "supervision_recovery",
            task_count,
            {
                "train_docs": resolved["supervision_recovery_train_docs"],
                "seeds": resolved["supervision_recovery_seeds"],
                "method_id": resolved["supervision_recovery_method_id"],
                "recoverable_benchmark": resolved["supervision_recovery_recoverable_benchmark"],
                "structural_grid": resolved["supervision_recovery_structural_grid"],
                "structural_cell": resolved["supervision_recovery_structural_cell"],
                "packages": resolved_package_order,
                "leaf_token_ladder": leaf_token_ladder,
                "package_leaf_token_overrides": package_leaf_token_overrides,
                "depth_discount_gammas": depth_discount_gammas,
                "exact_full_doc_parity_leaf_tokens": [
                    int(leaf_tokens)
                    for leaf_tokens in sorted(
                        {
                            int(leaf_tokens)
                            for package_name in resolved_package_order
                            for leaf_tokens in list(
                                package_leaf_token_overrides.get(
                                    str(package_name),
                                    leaf_token_ladder,
                                )
                                or []
                            )
                        }
                    )
                    if _parity_ref_active
                    and assumed_doc_tokens > 0
                    and int(leaf_tokens) >= int(assumed_doc_tokens)
                ],
                "benchmarks": [
                    str(spec.get("scope_key", "")) for spec in plan_scope_specs
                ],
                "hazard_panel_ids": list(
                    resolved.get("supervision_recovery_hazard_panel_ids") or []
                ),
                "hazard_panel_bundle_map": dict(
                    resolved.get("supervision_recovery_hazard_panel_bundle_map") or {}
                ),
            },
            str(output_root / "supervision_recovery" / "summary.json"),
        )
    if "support_grid" in phases:
        _record(
            "support_grid",
            len(resolved["support_leaf_tokens"]) * len(resolved["support_modes"]) * len(resolved["support_seeds"]),
            {
                "leaf_tokens": resolved["support_leaf_tokens"],
                "modes": resolved["support_modes"],
                "seeds": resolved["support_seeds"],
            },
            str(output_root / "support_grid" / "markov_local_support_detailed.summary.json"),
        )
    if "report" in phases:
        phase_counts["report"] = {
            "worker_tasks": 0,
            "details": {"sources": sorted(phases - {"report"})},
            "summary_output": str(output_root / "tradeoff_report" / "summary.json"),
        }

    scheduler_plan = {
        "scheduler_mode": str(resolved["scheduler"]["mode"]),
        "default_job_granularity": str(resolved["scheduler"]["default_job_granularity"]),
        "cleanup_stale_children": bool(resolved["scheduler"]["cleanup_stale_children"]),
        "max_gpu_items_per_mig": int(resolved["scheduler"]["max_gpu_items_per_mig"]),
        "device_count": len(device_list),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "resolved_selection": resolved,
        "devices": list(device_list),
        "device_count": len(device_list),
        "scheduler": scheduler_plan,
        "phase_task_counts": phase_counts,
        "total_worker_tasks": int(total_worker_tasks),
    }


# Imported from src.ctreepo.sim.core.tree_reference_presets
TREE_REFERENCE_OVERRIDE_KEYS = _IMPORTED_OVERRIDE_KEYS

TREE_REFERENCE_PRESET_CONFIGS = _IMPORTED_PRESET_CONFIGS

# Legacy inline definitions removed. To add or modify presets, edit
# src/ctreepo/sim/core/tree_reference_presets.py instead.


_tree_reference_label = _imported_tree_reference_label


def _tree_reference_recipe(tree_reference: Mapping[str, Any] | None) -> str:
    reference = dict(tree_reference or {})
    for key in ("preset_recipe", "recipe_name"):
        recipe = str(reference.get(key, "") or "").strip()
        if recipe:
            return recipe
    for key in (
        "preset_requested_name",
        "preset_public_name",
        "preset_display_name",
        "preset",
    ):
        preset_name = str(reference.get(key, "") or "").strip()
        if not preset_name:
            continue
        try:
            return str(_resolve_tree_reference_preset_recipe(preset_name))
        except ValueError:
            continue
    return ""


def _supervision_recovery_recoverable_benchmark_name(
    args: argparse.Namespace,
) -> str:
    preset = PRESET_DEFAULTS[str(args.preset)]
    raw = getattr(
        args,
        "supervision_recovery_recoverable_benchmark",
        preset["supervision_recovery_recoverable_benchmark"],
    )
    text = str(raw or "").strip()
    if text.lower() in {"", "none"}:
        return str(preset["supervision_recovery_recoverable_benchmark"]).strip()
    return text


def _supervision_recovery_structural_grid_name(
    args: argparse.Namespace,
) -> str:
    preset = PRESET_DEFAULTS[str(args.preset)]
    raw = getattr(
        args,
        "supervision_recovery_structural_grid",
        preset["supervision_recovery_structural_grid"],
    )
    text = str(raw or "").strip()
    if text.lower() in {"", "none"}:
        return str(preset["supervision_recovery_structural_grid"]).strip()
    return text


def _supervision_recovery_structural_cell_name(
    args: argparse.Namespace,
) -> str:
    preset = PRESET_DEFAULTS[str(args.preset)]
    raw = getattr(
        args,
        "supervision_recovery_structural_cell",
        preset["supervision_recovery_structural_cell"],
    )
    text = str(raw or "").strip()
    if text.lower() in {"", "none"}:
        return str(preset["supervision_recovery_structural_cell"]).strip()
    return text


def _supervision_recovery_scope_keys(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    values = _parse_str_list(
        getattr(args, "supervision_recovery_scope_keys", None),
        (),
    )
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _default_hazard_panel_bundle_path(panel_id: str) -> str:
    return str(
        REPO_ROOT
        / "outputs"
        / "_bundles"
        / "markov_hazard_panels"
        / str(panel_id)
        / "seed_0"
        / "base_bundle.json"
    )


def _hazard_panel_scope_spec(
    panel_id: str,
    *,
    bundle_path: str,
) -> dict[str, Any]:
    panel = resolve_markov_hazard_panel(panel_id)
    max_doc_tokens = int(max(condition.doc_tokens for condition in panel.conditions))
    template_benchmark = (
        "recoverable_v5_t2048"
        if max_doc_tokens >= 2048
        else "recoverable_v5_t128"
    )
    ops_overrides = dict(panel_to_ops_overrides(panel))
    supported_ops_keys = _ops_count_supported_config_keys()
    ops_overrides = {
        str(key): value
        for key, value in ops_overrides.items()
        if str(key) in supported_ops_keys
    }
    return {
        "scope_key": str(panel.panel_id),
        "scope_label": str(panel.display_name),
        "scope_kind": "hazard_panel",
        "benchmark_name": str(template_benchmark),
        "hardness_grid": "",
        "grid_cell_ids": [],
        "base_bundle_path": str(bundle_path or _default_hazard_panel_bundle_path(panel.panel_id)),
        "ops_config_overrides": ops_overrides,
        "hazard_panel_id": str(panel.panel_id),
    }


def _supervision_recovery_scope_specs(
    *,
    recoverable_benchmark: str,
    structural_grid: str,
    structural_cell: str,
    requested_scope_keys: Sequence[str] = (),
    hazard_panel_ids: Sequence[str] = (),
    hazard_panel_bundle_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    scope_specs = [
        {
            "scope_key": str(recoverable_benchmark),
            "scope_label": str(recoverable_benchmark),
            "scope_kind": "recoverable",
            "benchmark_name": str(recoverable_benchmark),
            "hardness_grid": "",
            "grid_cell_ids": [],
            "base_bundle_path": "",
            "ops_config_overrides": {},
            "hazard_panel_id": "",
        },
        {
            "scope_key": str(structural_cell),
            "scope_label": _supervision_recovery_scope_label(
                str(structural_cell),
                recoverable_scope_key=str(recoverable_benchmark),
                structural_grid=str(structural_grid),
            ),
            "scope_kind": "structural",
            "benchmark_name": _structural_supervision_recovery_benchmark_name(
                str(structural_cell),
                structural_grid=str(structural_grid),
            ),
            "hardness_grid": str(structural_grid),
            "grid_cell_ids": [str(structural_cell)],
            "base_bundle_path": "",
            "ops_config_overrides": {},
            "hazard_panel_id": "",
        },
    ]
    bundle_map = {str(key): str(value) for key, value in dict(hazard_panel_bundle_map or {}).items()}
    for panel_id in _parse_str_list(" ".join(str(value) for value in hazard_panel_ids), ()):
        panel = resolve_markov_hazard_panel(panel_id)
        scope_specs.append(
            _hazard_panel_scope_spec(
                str(panel.panel_id),
                bundle_path=bundle_map.get(
                    str(panel.panel_id),
                    bundle_map.get(str(panel_id), _default_hazard_panel_bundle_path(panel.panel_id)),
                ),
            )
        )
    requested = {
        str(value or "").strip()
        for value in requested_scope_keys
        if str(value or "").strip()
    }
    if not requested:
        return scope_specs
    filtered = [
        dict(spec)
        for spec in scope_specs
        if str(spec.get("scope_key", "") or "").strip() in requested
    ]
    if filtered:
        return filtered
    raise ValueError(
        "supervision_recovery_scope_keys did not match any available supervision_recovery scopes: "
        f"requested={sorted(requested)}, available="
        f"{[str(spec['scope_key']) for spec in scope_specs]}"
    )


def _structural_supervision_recovery_benchmark_name(
    structural_cell: str,
    *,
    structural_grid: str = SUPERVISION_RECOVERY_STRUCTURAL_GRID,
) -> str:
    return f"{str(structural_grid).strip()}::{str(structural_cell).strip()}"


def _supervision_recovery_scope_label(
    scope_key: str,
    *,
    recoverable_scope_key: str,
    structural_grid: str,
) -> str:
    normalized_scope_key = str(scope_key).strip()
    if not normalized_scope_key:
        return ""
    if normalized_scope_key == str(recoverable_scope_key).strip():
        return normalized_scope_key
    return f"{str(structural_grid).strip()}::{normalized_scope_key}"

SUPERVISION_TREE_REFERENCE_PRESERVE_KEYS: frozenset[str] = frozenset(
    {
        "leaf_supervision_kind",
        "leaf_label_rate",
        "leaf_exact_supervision",
        "internal_supervision_kind",
        "internal_label_rate",
        "root_weight",
        "depth_discount_gamma",
        "schedule_consistency_weight",
        "tree_batch_autotune",
        "tree_batch_structural_pad_limit",
        "tree_batch_auto_queue_min_docs",
        "tree_batch_auto_queue_min_fill_ratio",
        "gpu_runtime_data_mode",
        "gpu_runtime_bucket_mode",
        "gpu_runtime_preload_splits",
        "gpu_runtime_preload_targets",
        "gpu_runtime_workers_per_mig",
        "gpu_runtime_allow_multi_worker_screen",
        "gpu_runtime_capacity_workers_per_mig",
    }
)


def _supervision_tree_reference_preserve_keys(
    args: argparse.Namespace,
    *,
    preserve_schedule: bool = True,
) -> frozenset[str]:
    keys = set(SUPERVISION_TREE_REFERENCE_PRESERVE_KEYS)
    if preserve_schedule:
        explicit_schedule = str(getattr(args, "tree_training_schedule", "") or "").strip()
        if explicit_schedule:
            keys.add("tree_training_schedule")
        if getattr(args, "tree_stage1_epochs", None) is not None:
            keys.add("tree_stage1_epochs")
        if getattr(args, "tree_stage2_epochs", None) is not None:
            keys.add("tree_stage2_epochs")
    return frozenset(keys)


def _supervision_recovery_tree_learning_errors(
    config: Mapping[str, Any],
    *,
    preset: str,
    scope_kind: str,
    tree_reference: Mapping[str, Any],
    allow_leafgrid_geometry: bool = False,
) -> List[str]:
    # Previously 180+ lines re-asserting preset defaults as runtime checks.
    # The preset system already sets these values correctly; redundant
    # validation only creates fragile coupling. Retained: smoke guard.
    errors: List[str] = []
    if _ns(preset) == "smoke":
        errors.append(
            "preset='smoke' is not allowed for supervision_recovery; "
            "use a non-smoke preset or explicit strong overrides"
        )
    theorem_surface_mode = str(
        config.get("tree_theorem_surface_mode", "") or ""
    ).strip().lower()
    task_head_mode = _gs(config, "tree_task_head_mode").lower()
    summary_root_mode = str(
        config.get("tree_summary_spec_root_mode", "") or ""
    ).strip().lower()
    uses_factorized_theorem_fiber = (
        theorem_surface_mode == "factorized_score_fiber"
        and task_head_mode == "theorem_feature_scalar"
        and summary_root_mode == "factored_theorem_readout"
    )
    if not uses_factorized_theorem_fiber:
        if int(_safe_int(config.get("tree_leaf_fno_width"), 0)) <= 0:
            errors.append("tree_leaf_fno_width must be positive")
        if int(_safe_int(config.get("tree_leaf_fno_n_modes"), 0)) <= 0:
            errors.append("tree_leaf_fno_n_modes must be positive")
        if int(_safe_int(config.get("tree_leaf_fno_n_layers"), 0)) <= 0:
            errors.append("tree_leaf_fno_n_layers must be positive")
        if not _gs(config, "summary_spec_name"):
            errors.append("summary_spec_name must be non-empty")
        if int(_safe_int(config.get("slot_count"), 0)) <= 0:
            errors.append("slot_count must be positive")
    return errors


def _validated_supervision_recovery_tree_config(
    config: Mapping[str, Any],
    *,
    preset: str,
    scope_kind: str,
    context: str,
    tree_reference: Mapping[str, Any],
    allow_leafgrid_geometry: bool = False,
) -> Dict[str, Any]:
    payload = dict(config or {})
    errors = _supervision_recovery_tree_learning_errors(
        payload,
        preset=preset,
        scope_kind=scope_kind,
        tree_reference=tree_reference,
        allow_leafgrid_geometry=allow_leafgrid_geometry,
    )
    if errors:
        reference_mode = str(tree_reference.get("mode", "") or "")
        reference_label = str(
            tree_reference.get("preset")
            or tree_reference.get("winning_config_label")
            or tree_reference.get("capacity_root")
            or ""
        )
        message = (
            f"supervision_recovery tree config is not learning-capable for {context}: "
            + "; ".join(errors)
        )
        if reference_mode:
            message += f" (tree_reference_mode={reference_mode}"
            if reference_label:
                message += f", tree_reference={reference_label}"
            message += ")"
        raise ValueError(message)
    return payload


def _build_supervision_recovery_scope_config(
    args: argparse.Namespace,
    *,
    base_config: Mapping[str, Any],
    package_name: str,
    package_spec: Mapping[str, Any],
    scope_key: str,
    scope_label: str,
    tree_reference: Mapping[str, Any],
    preserve_schedule: bool,
    preserve_fixed_leaf_tokens: bool = False,
    preserve_requested_leaf_tokens: bool = False,
    comparison_surface_source: Mapping[str, Any] | None = None,
    extra_updates: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = dict(base_config)
    config.update({
        "budget_total_calls": 0,
        "budget_total_calls_per_doc": float(package_spec.get("budget_total_calls_per_doc", 0.0)),
        "full_doc_budget_share": float(package_spec.get("full_doc_budget_share", 1.0)),
        "doc_consumption_mode": str(package_spec.get("doc_consumption_mode", "root_only")),
        "local_split_mode": str(package_spec.get("local_split_mode", "balanced")),
        "leaf_supervision_kind": str(package_spec.get("leaf_supervision_kind", "count_only")),
        "leaf_label_rate": float(package_spec.get("leaf_label_rate", 0.0)),
        "internal_supervision_kind": str(package_spec.get("internal_supervision_kind", "none")),
        "internal_label_rate": float(package_spec.get("internal_label_rate", 0.0)),
        "max_internal_depth": int(package_spec.get("max_internal_depth", 0)),
        "pipeline_supervision_recovery_package": str(package_name),
        "pipeline_supervision_recovery_scope": str(scope_key),
        "pipeline_supervision_recovery_scope_label": str(scope_label),
    })
    if extra_updates:
        config.update(dict(extra_updates))
    tree_config = _apply_tree_reference_overrides(
        dict(config),
        tree_reference,
        preserve_fixed_leaf_tokens=preserve_fixed_leaf_tokens,
        preserve_keys=_supervision_tree_reference_preserve_keys(
            args,
            preserve_schedule=preserve_schedule,
        ),
    )
    if preserve_requested_leaf_tokens:
        tree_config["preserve_requested_leaf_tokens"] = True
        tree_config["official_fno_preserve_requested_leaf_tokens"] = True
    comparison_mode = normalize_markov_comparison_mode(
        str(tree_config.get("comparison_mode", "legacy") or "legacy")
    )
    if comparison_mode in {"comparable", "exact_collapse"}:
        pre_surface_tree_config = dict(tree_config)
        surface_source = dict(comparison_surface_source or tree_config)
        benchmark = _resolved_full_doc_benchmark_spec(
            str(
                config.get("pipeline_benchmark_name", scope_key)
                or scope_key
            ),
            str(config.get("pipeline_hardness_grid", "") or ""),
            tuple(
                str(value).strip()
                for value in list(config.get("pipeline_grid_cell_ids") or ())
                if str(value).strip()
            ),
        )
        comparable_surface = resolve_markov_comparable_surface(
            benchmark=benchmark,
            config=surface_source,
            comparison_mode=comparison_mode,
        )
        surfaced_tree_config = apply_comparable_surface_to_mapping(
            benchmark=benchmark,
            config=tree_config,
            surface=comparable_surface,
        )
        exact_full_doc_parity = _supervision_recovery_requires_exact_full_doc_parity(
            package_name=str(package_name),
            package_spec=package_spec,
            payload=pre_surface_tree_config,
        )
        if exact_full_doc_parity:
            tree_config = dict(surfaced_tree_config)
        else:
            # Surface first (aligns shared FNO params), then overlay pre-surface
            # tree-specific values. Preservation flags merge with OR.
            tree_config = dict(surfaced_tree_config)
            tree_config.update(pre_surface_tree_config)
            tree_config["comparison_mode"] = str(comparison_mode)
            for _preserve_key in ("preserve_requested_leaf_tokens", "official_fno_preserve_requested_leaf_tokens"):
                tree_config[_preserve_key] = bool(
                    surfaced_tree_config.get(_preserve_key, False)
                    or pre_surface_tree_config.get(_preserve_key, False)
                )
    package_accounting_source = dict(
        surfaced_tree_config
        if "surfaced_tree_config" in locals()
        else tree_config
    )
    accounting_min_tokens, accounting_max_tokens = _supervision_recovery_accounting_tokens(
        benchmark_name=str(
            config.get("pipeline_benchmark_name", scope_key)
            or scope_key
        ),
        hardness_grid=str(config.get("pipeline_hardness_grid", "") or ""),
        grid_cell_ids=tuple(
            str(value).strip()
            for value in list(config.get("pipeline_grid_cell_ids") or ())
            if str(value).strip()
        ),
        surfaced_min_tokens=int(
            _safe_int(package_accounting_source.get("min_tokens"), 0)
        ),
        surfaced_max_tokens=int(
            _safe_int(package_accounting_source.get("max_tokens"), 0)
        ),
    )
    package_accounting_source["min_tokens"] = int(accounting_min_tokens)
    package_accounting_source["max_tokens"] = int(accounting_max_tokens)
    resolved_package_spec, accounting = _resolve_supervision_recovery_package_for_scope(
        str(package_name),
        package_spec,
        min_tokens=int(accounting_min_tokens),
        max_tokens=int(accounting_max_tokens),
        fixed_leaf_tokens=int(_safe_int(tree_config.get("fixed_leaf_tokens"), 0)),
        scope_key=str(scope_key),
    )
    tree_config.update(
        {
            "budget_total_calls_per_doc": float(
                resolved_package_spec.get("budget_total_calls_per_doc", 0.0)
            ),
            "full_doc_budget_share": float(
                resolved_package_spec.get("full_doc_budget_share", 1.0)
            ),
            "doc_consumption_mode": str(
                resolved_package_spec.get("doc_consumption_mode", "root_only")
            ),
            "package_semantics": str(
                resolved_package_spec.get("package_semantics", "")
            ),
            "local_split_mode": str(
                resolved_package_spec.get("local_split_mode", "balanced")
            ),
            "tree_supervision_source": str(
                resolved_package_spec.get("tree_supervision_source", "manifest")
            ),
            "leaf_supervision_kind": str(
                resolved_package_spec.get("leaf_supervision_kind", "count_only")
            ),
            "leaf_label_rate": float(
                resolved_package_spec.get("leaf_label_rate", 0.0)
            ),
            "internal_supervision_kind": str(
                resolved_package_spec.get("internal_supervision_kind", "none")
            ),
            "internal_label_rate": float(
                resolved_package_spec.get("internal_label_rate", 0.0)
            ),
            "max_internal_depth": int(
                resolved_package_spec.get("max_internal_depth", 0)
            ),
            "tree_local_weighting_mode": str(
                resolved_package_spec.get(
                    "tree_local_weighting_mode",
                    "span_mass_ipw_sum",
                )
            ),
            "depth_discount_gamma": _safe_float(
                resolved_package_spec.get(
                    "depth_discount_gamma",
                    tree_config.get("depth_discount_gamma", 1.0),
                ),
                1.0,
            ),
            "mass_target_per_doc": _safe_float(
                accounting.get("mass_target_per_doc"),
                float("nan"),
            ),
            "computed_doc_review_mass_per_doc": _safe_float(
                accounting.get("computed_doc_review_mass_per_doc"),
                float("nan"),
            ),
            "computed_local_mass_per_doc": _safe_float(
                accounting.get("computed_local_mass_per_doc"),
                float("nan"),
            ),
            "computed_leaf_mass_per_doc": _safe_float(
                accounting.get("computed_leaf_mass_per_doc"),
                float("nan"),
            ),
            "computed_internal_mass_per_doc": _safe_float(
                accounting.get("computed_internal_mass_per_doc"),
                float("nan"),
            ),
            "computed_total_mass_per_doc": _safe_float(
                accounting.get("computed_total_mass_per_doc"),
                float("nan"),
            ),
            "computed_assumed_doc_tokens": int(
                _safe_int(accounting.get("assumed_doc_tokens"), 0)
            ),
            "computed_assumed_leaves": int(
                _safe_int(accounting.get("assumed_leaves"), 0)
            ),
            "computed_assumed_internal_nodes": int(
                _safe_int(accounting.get("assumed_internal_nodes"), 0)
            ),
            "computed_leaf_mass_full_per_doc": _safe_float(
                accounting.get("leaf_mass_full_per_doc"),
                float("nan"),
            ),
            "computed_internal_mass_full_per_doc": _safe_float(
                accounting.get("internal_mass_full_per_doc"),
                float("nan"),
            ),
        }
    )
    return tree_config, resolved_package_spec, accounting


def _validate_supervision_recovery_tree_setup(args: argparse.Namespace) -> None:
    preset = PRESET_DEFAULTS[str(args.preset)]
    recoverable_benchmark = _supervision_recovery_recoverable_benchmark_name(args)
    structural_grid = _supervision_recovery_structural_grid_name(args)
    structural_cell = _supervision_recovery_structural_cell_name(args)
    requested_scope_keys = set(_supervision_recovery_scope_keys(args))
    run_recoverable = (
        not requested_scope_keys
        or str(recoverable_benchmark) in requested_scope_keys
    )
    run_structural = (
        not requested_scope_keys
        or str(structural_cell) in requested_scope_keys
    )
    train_docs_values = _parse_int_list(
        getattr(args, "supervision_recovery_train_docs", None),
        preset["supervision_recovery_train_docs"],
    )
    recovery_seeds = _parse_int_list(
        getattr(args, "supervision_recovery_seeds", None),
        preset["supervision_recovery_seeds"],
    )
    depth_discount_gammas = _parse_float_list(
        getattr(args, "supervision_recovery_depth_discount_gammas", None),
        preset["supervision_recovery_depth_discount_gammas"],
    )
    include_gamma_tag = bool(
        len(depth_discount_gammas) > 1
        or any(
            not math.isclose(_safe_float(gamma, 1.0), 1.0, rel_tol=0.0, abs_tol=1e-9)
            for gamma in depth_discount_gammas
        )
    )
    depth_discount_gammas = _parse_float_list(
        getattr(args, "supervision_recovery_depth_discount_gammas", None),
        preset["supervision_recovery_depth_discount_gammas"],
    )
    if not train_docs_values or not recovery_seeds:
        return
    resolved_package_order = _resolved_supervision_recovery_package_order(args)
    explicit_leaf_tokens = _resolved_supervision_recovery_leaf_token_ladder(args)
    leaf_tokens_values = explicit_leaf_tokens or [int(args.supervision_fixed_leaf_tokens)]
    leafgrid_active = bool(explicit_leaf_tokens)
    seed = int(args.seed) + int(recovery_seeds[0])
    train_docs = int(train_docs_values[0])
    for fixed_leaf_tokens in leaf_tokens_values:
        for package_name in resolved_package_order:
            package_spec = dict(SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name])
            tree_reference = _resolve_tree_reference(
                args,
                package_name=str(package_name),
            )
            structural_tree_reference = _resolve_tree_reference(
                args,
                prefix="structural_tree_reference",
                package_name=str(package_name),
                fallback=tree_reference,
            )
            config = _base_ops_config(
                args,
                seed=seed,
                data_seed=int(recovery_seeds[0]),
                train_docs=train_docs,
                val_docs=min(int(args.val_docs), max(2, int(train_docs) // 8)),
                test_docs=min(int(args.test_docs), max(2, int(train_docs) // 8)),
                batch_size=_resolve_supervision_batch_size_for_leaf_tokens(
                    args, int(fixed_leaf_tokens)
                ),
                n_epochs=int(args.supervision_epochs),
                fixed_leaf_tokens=int(fixed_leaf_tokens),
            )
            config.update(
                {
                    "min_tokens": int(args.supervision_min_tokens),
                    "max_tokens": int(args.supervision_max_tokens),
                    "min_segments": int(args.supervision_min_segments),
                    "max_segments": int(args.supervision_max_segments),
                    "law_package": "all_laws",
                    "local_law_weight": 0.5,
                    "c1_relative_weight": 1.0,
                    "c2_relative_weight": 1.0,
                    "c3_relative_weight": 1.0,
                    "leaf_supervision_kind": str(package_spec["leaf_supervision_kind"]),
                    "leaf_label_rate": float(package_spec["leaf_label_rate"]),
                    "internal_supervision_kind": str(package_spec["internal_supervision_kind"]),
                    "internal_label_rate": float(package_spec["internal_label_rate"]),
                    "max_internal_depth": int(package_spec.get("max_internal_depth", 0)),
                    "budget_total_calls_per_doc": float(
                        package_spec["budget_total_calls_per_doc"]
                    ),
                    "pipeline_supervision_recovery_package": str(package_name),
                    "pipeline_supervision_recovery_recoverable_benchmark": str(
                        recoverable_benchmark
                    ),
                    "pipeline_supervision_recovery_structural_grid": str(
                        structural_grid
                    ),
                    "pipeline_supervision_recovery_scope": str(recoverable_benchmark),
                    "pipeline_supervision_recovery_scope_label": str(recoverable_benchmark),
                }
            )
            if run_recoverable:
                recoverable_tree_config, recoverable_package_spec, _ = _build_supervision_recovery_scope_config(
                    args,
                    base_config=config,
                    package_name=str(package_name),
                    package_spec=package_spec,
                    scope_key=str(recoverable_benchmark),
                    scope_label=str(recoverable_benchmark),
                    tree_reference=tree_reference,
                    preserve_schedule=True,
                    preserve_fixed_leaf_tokens=leafgrid_active,
                )
                recoverable_tree_reference_label = _tree_reference_label(tree_reference)
                if recoverable_tree_reference_label in {
                    SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
                    UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
                    UNIFIED_G_FNO_PARITY_CANARY_PRESET,
                    "structural_factorized_sketch_v3",
                }:
                    recoverable_tree_config["tree_checkpoint_metric"] = _supervision_recovery_tree_checkpoint_metric(
                        recoverable_package_spec,
                        default_metric=str(
                            recoverable_tree_config.get(
                                "tree_checkpoint_metric",
                                "val_exact_sketch_direct",
                            )
                        ),
                        tree_reference_label=recoverable_tree_reference_label,
                    )
                _validated_supervision_recovery_tree_config(
                    recoverable_tree_config,
                    preset=str(args.preset),
                    scope_kind="recoverable",
                    context=(
                        "representative supervision_recovery recoverable tree row "
                        f"(package={package_name}, fixed_leaf_tokens={int(fixed_leaf_tokens)})"
                    ),
                    tree_reference=tree_reference,
                    allow_leafgrid_geometry=leafgrid_active,
                )
            if run_structural:
                structural_config = dict(config)
                structural_config.update(
                    {
                        "pipeline_supervision_recovery_scope": str(structural_cell),
                        "pipeline_supervision_recovery_scope_label": (
                            _supervision_recovery_scope_label(
                                str(structural_cell),
                                recoverable_scope_key=str(recoverable_benchmark),
                                structural_grid=str(structural_grid),
                            )
                        ),
                        "pipeline_supervision_recovery_recoverable_benchmark": str(
                            recoverable_benchmark
                        ),
                        "pipeline_supervision_recovery_structural_grid": str(
                            structural_grid
                        ),
                        "pipeline_benchmark_name": _structural_supervision_recovery_benchmark_name(
                            structural_cell,
                            structural_grid=str(structural_grid),
                        ),
                        "pipeline_hardness_grid": str(structural_grid),
                        "pipeline_grid_cell_ids": [structural_cell],
                    }
                )
                structural_tree_config, structural_package_spec, _ = _build_supervision_recovery_scope_config(
                    args,
                    base_config=structural_config,
                    package_name=str(package_name),
                    package_spec=package_spec,
                    scope_key=structural_cell,
                    scope_label=str(
                        structural_config.get("pipeline_supervision_recovery_scope_label", "")
                        or _supervision_recovery_scope_label(
                            str(structural_cell),
                            recoverable_scope_key=str(recoverable_benchmark),
                            structural_grid=str(structural_grid),
                        )
                    ),
                    tree_reference=structural_tree_reference,
                    preserve_schedule=False,
                    preserve_fixed_leaf_tokens=True,
                )
                structural_tree_reference_label = _tree_reference_label(structural_tree_reference)
                if structural_tree_reference_label in {
                    SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
                    UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
                    UNIFIED_G_FNO_PARITY_CANARY_PRESET,
                    "structural_factorized_sketch_v3",
                }:
                    structural_tree_config["tree_checkpoint_metric"] = _supervision_recovery_tree_checkpoint_metric(
                        structural_package_spec,
                        default_metric=str(
                            structural_tree_config.get(
                                "tree_checkpoint_metric",
                                "val_exact_sketch_direct",
                            )
                        ),
                        tree_reference_label=structural_tree_reference_label,
                    )
                _validated_supervision_recovery_tree_config(
                    structural_tree_config,
                    preset=str(args.preset),
                    scope_kind="structural",
                    context=(
                        "representative supervision_recovery structural tree row "
                        f"(package={package_name}, fixed_leaf_tokens={int(fixed_leaf_tokens)})"
                    ),
                    tree_reference=structural_tree_reference,
                    allow_leafgrid_geometry=leafgrid_active,
                )


@functools.lru_cache(maxsize=8)
def _load_capacity_locked_tree_reference(capacity_root_text: str) -> Dict[str, Any]:
    capacity_root = Path(capacity_root_text).expanduser()
    summary_path = capacity_root / "tree_fno_capacity_locked_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing capacity locked summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winning_label = str(summary.get("winning_config_label", "")).strip()
    if not winning_label:
        raise ValueError(f"capacity summary at {summary_path} is missing winning_config_label")
    candidate_paths = []
    locked_summary_json = str(summary.get("locked_summary_json", "")).strip()
    if locked_summary_json:
        candidate_paths.append(Path(locked_summary_json))
    candidate_paths.extend(
        [
            capacity_root / "locked" / "summary.json",
            capacity_root / "screen" / "summary.json",
        ]
    )
    for path in candidate_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in list(dict(payload).get("runs") or []):
            if (
                str(run.get("baseline_family", "")).strip() == "tree_neural"
                and str(run.get("config_label", "")).strip() == winning_label
                and isinstance(run.get("config"), Mapping)
            ):
                return {
                    "mode": "capacity_locked",
                    "capacity_root": str(capacity_root),
                    "winning_config_label": winning_label,
                    "config": dict(run.get("config") or {}),
                }
    raise RuntimeError(
        f"unable to reconstruct winning tree reference '{winning_label}' from {capacity_root}"
    )


def _resolve_tree_reference(
    args: argparse.Namespace,
    *,
    prefix: str = "tree_reference",
    package_name: str | None = None,
    fallback: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    mode = str(getattr(args, f"{prefix}_mode", "default") or "default").strip().lower()
    if mode == "default":
        if fallback is not None:
            return dict(fallback)
        return {
            "mode": "default",
            "capacity_root": "",
            "preset": "",
            "winning_config_label": "",
            "config": {},
        }
    if mode == "package_capacity_locked":
        capacity_root = getattr(args, f"{prefix}_capacity_root", None)
        capacity_root_text = _optional_path_text(capacity_root)
        if not capacity_root_text:
            raise ValueError(
                f"{prefix}_mode=package_capacity_locked requires --{prefix.replace('_', '-')}-capacity-root"
            )
        package_name_text = str(package_name or "").strip()
        if not package_name_text:
            raise ValueError(
                f"{prefix}_mode=package_capacity_locked requires a supervision_recovery package name"
            )
        package_capacity_root = Path(capacity_root_text).expanduser() / package_name_text
        tree_reference = _load_capacity_locked_tree_reference(str(package_capacity_root))
        tree_reference.update(
            {
                "mode": "package_capacity_locked",
                "package_name": package_name_text,
                "package_capacity_root": str(package_capacity_root),
                "package_capacity_base_root": str(Path(capacity_root_text).expanduser()),
            }
        )
        return tree_reference
    if mode == "preset":
        preset_name = str(getattr(args, f"{prefix}_preset", "") or "").strip()
        if not preset_name:
            raise ValueError(
                f"{prefix}_mode=preset requires --{prefix.replace('_', '-')}-preset"
            )
        preset_record = _resolve_tree_reference_preset(preset_name)
        return {
            "mode": "preset",
            "capacity_root": "",
            "preset": preset_name,
            "preset_requested_name": str(preset_record["requested_name"]),
            "preset_public_name": str(preset_record["public_name"]),
            "preset_display_name": str(preset_record["public_name"]),
            "preset_recipe": str(preset_record["recipe_name"]),
            "winning_config_label": preset_name,
            "config": dict(preset_record["config"]),
        }
    if mode != "capacity_locked":
        raise ValueError(
            f"unsupported {prefix}_mode={mode!r}; expected one of default, capacity_locked, package_capacity_locked, preset"
        )
    capacity_root = getattr(args, f"{prefix}_capacity_root", None)
    capacity_root_text = _optional_path_text(capacity_root)
    if not capacity_root_text:
        raise ValueError(
            f"{prefix}_mode=capacity_locked requires --{prefix.replace('_', '-')}-capacity-root"
        )
    return _load_capacity_locked_tree_reference(str(Path(capacity_root_text).expanduser()))


def _tree_reference_from_preset_name(preset_name: str) -> Dict[str, Any]:
    preset_record = _resolve_tree_reference_preset(str(preset_name))
    return {
        "mode": "preset",
        "capacity_root": "",
        "preset": str(preset_name),
        "preset_requested_name": str(preset_record["requested_name"]),
        "preset_public_name": str(preset_record["public_name"]),
        "preset_display_name": str(preset_record["public_name"]),
        "preset_recipe": str(preset_record["recipe_name"]),
        "winning_config_label": str(preset_name),
        "config": dict(preset_record["config"]),
    }


def _apply_tree_reference_overrides(
    config: Dict[str, Any],
    tree_reference: Mapping[str, Any],
    *,
    preserve_fixed_leaf_tokens: bool = False,
    preserve_keys: Iterable[str] = (),
    frozen_keys: Iterable[str] = (),
) -> Dict[str, Any]:
    """Apply preset overrides from tree_reference onto config.

    Uses a deny-list approach: any key in ref_config that is a valid
    OPSCountConfig field and NOT in TREE_REFERENCE_DENY_KEYS is applied.
    Keys in *frozen_keys* or *preserve_keys* are additionally protected.
    *preserve_fixed_leaf_tokens* is shorthand for freezing "fixed_leaf_tokens".
    """
    from src.ctreepo.sim.core.tree_reference_presets import TREE_REFERENCE_DENY_KEYS
    ref_config = dict(tree_reference.get("config") or {})
    if not ref_config:
        return config
    out = dict(config)
    supported_keys = _ops_count_supported_config_keys()
    frozen = set(frozen_keys) | set(preserve_keys) | TREE_REFERENCE_DENY_KEYS
    if preserve_fixed_leaf_tokens:
        frozen.add("fixed_leaf_tokens")
    for key, value in ref_config.items():
        if key in frozen or key not in supported_keys:
            continue
        if value in ("", None):
            continue
        out[key] = value
    return out


def _common_worker_env(device_label: str) -> Dict[str, str]:
    return _build_worker_env(device_label, use_cuda=True)


def _run_subprocess_tasks(tasks: Sequence[SubprocessTask], devices: Sequence[str]) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    device_pool = list(devices or [""])
    results: List[Dict[str, Any]] = []
    pending_tasks: List[SubprocessTask] = []
    for task in tasks:
        if task.output_path.exists():
            results.append(
                {
                    "name": task.name,
                    "device": "",
                    "returncode": 0,
                    "output_path": str(task.output_path),
                    "log_path": str(task.log_path),
                    "wall_clock_s": 0.0,
                    "reused": True,
                }
            )
            continue
        pending_tasks.append(task)
    if not pending_tasks:
        return results
    pool_size = max(1, len(device_pool))
    for wave_start in range(0, len(pending_tasks), pool_size):
        wave = list(pending_tasks[wave_start : wave_start + pool_size])
        procs = []
        for task, device_label in zip(wave, device_pool):
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(task.log_path, "w", encoding="utf-8")
            env = _common_worker_env(device_label if task.device_label == "" else task.device_label)
            started = time.perf_counter()
            proc = subprocess.Popen(
                list(task.argv),
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            procs.append((task, device_label, handle, proc, started))
        for task, device_label, handle, proc, started in procs:
            rc = proc.wait()
            handle.close()
            finished = time.perf_counter()
            result = {
                "name": task.name,
                "device": device_label,
                "returncode": int(rc),
                "output_path": str(task.output_path),
                "log_path": str(task.log_path),
                "wall_clock_s": float(finished - started),
                "reused": False,
            }
            if rc != 0:
                raise RuntimeError(
                    f"task {task.name} failed with rc={rc}; see {task.log_path}"
                )
            results.append(result)
    return results


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _public_payload_for_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def _clean(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            if path and path[-1] == "config" and any("tree_reference" in part for part in path[:-1]):
                encoded = json.dumps(dict(value), sort_keys=True, default=str).encode("utf-8")
                return {"backend_config_digest": hashlib.sha256(encoded).hexdigest()}
            return {str(key): _clean(item, (*path, str(key))) for key, item in value.items()}
        if isinstance(value, list):
            return [_clean(item, (*path, str(index))) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return [_clean(item, (*path, str(index))) for index, item in enumerate(value)]
        return value

    return dict(_clean(payload, ()))


def _tradeoff_experiment_spec(
    *,
    args: argparse.Namespace,
    output_root: Path,
    run_plan: Mapping[str, Any],
) -> ExperimentSpec:
    resolved = dict(run_plan.get("resolved_selection", {}) or {})
    recoverable_benchmark = str(
        resolved.get(
            "supervision_recovery_recoverable_benchmark",
            SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
        )
        or SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK
    )
    structural_grid = str(
        resolved.get(
            "supervision_recovery_structural_grid",
            SUPERVISION_RECOVERY_STRUCTURAL_GRID,
        )
        or SUPERVISION_RECOVERY_STRUCTURAL_GRID
    )
    structural_cell = str(resolved.get("supervision_recovery_structural_cell", "") or "")
    benchmark_refs = (
        benchmark_ref_from_parts(
            family="markov_full_doc",
            scope=str(recoverable_benchmark),
            name=str(recoverable_benchmark),
        ),
        benchmark_ref_from_parts(
            family="markov_full_doc",
            scope=str(structural_grid),
            cell=structural_cell,
            name=(
                f"{structural_grid}::{structural_cell}"
                if structural_cell
                else str(structural_grid)
            ),
        ),
    )
    method_refs = (
        method_ref_from_markov_full_doc_run(
            family=str(
                resolved.get(
                    "supervision_recovery_method_id",
                    SUPERVISION_RECOVERY_TREE_FAMILY,
                )
                or SUPERVISION_RECOVERY_TREE_FAMILY
            ),
            variant="tree_reference_family",
            adapter="markov_tree",
            metadata={"scope": "tree"},
        ),
        method_ref_from_markov_full_doc_run(
            family="official_fno",
            variant="canonical_full_doc",
            adapter="markov_tree",
            metadata={"scope": "fno"},
        ),
    )
    phase_counts = dict(run_plan.get("phase_task_counts", {}) or {})
    phase_specs = default_phase_specs(phase_counts.keys())
    return ExperimentSpec.create(
        adapter_id="markov_tree",
        output_root=str(output_root),
        title="markov_tradeoff_pipeline",
        benchmark_refs=benchmark_refs,
        method_refs=method_refs,
        phases=phase_specs,
        report_profiles=("tradeoff", "supervision_recovery"),
        launch_command=[sys.executable, "scripts/run_markov_optimization_tradeoff_pipeline.py", *sys.argv[1:]],
        resume_command=[sys.executable, "scripts/run_markov_optimization_tradeoff_pipeline.py", *sys.argv[1:]],
        metadata={
            "legacy_script": "run_markov_optimization_tradeoff_pipeline.py",
            "phases": list(phase_counts.keys()),
            "devices": list(run_plan.get("devices", []) or []),
            "selection_preset": str(getattr(args, "preset", "") or ""),
        },
    )


def _write_tradeoff_experiment_state(
    *,
    output_root: Path,
    spec: ExperimentSpec,
    state: str,
    active_phase: str = "",
    items_total: int = 0,
    completed_items: int = 0,
    failed_items: int = 0,
    active_items: int = 0,
    pending_items: int = 0,
) -> None:
    finished = int(completed_items) + int(failed_items)
    percent_complete = (
        100.0 * float(finished) / float(items_total)
        if int(items_total) > 0
        else 0.0
    )
    write_experiment_status(
        output_root,
        ProgressSnapshot(
            experiment_id=str(spec.experiment_id),
            state=str(state),
            active_phase=str(active_phase),
            items_total=int(items_total),
            completed_items=int(completed_items),
            failed_items=int(failed_items),
            active_items=int(active_items),
            pending_items=int(pending_items),
            percent_complete=percent_complete,
            artifact_targets=(
                "pipeline_summary_json",
                "supervision_recovery_summary_json",
                "tradeoff_report_summary_json",
                "tradeoff_report_pdf",
            ),
            metadata={"adapter": "markov_tree"},
        ),
    )


def _scheduler_item_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return int(len(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        return int(len(value))
    return int(_safe_int(value, 0))


def _tradeoff_artifacts(output_root: Path) -> list[object]:
    return canonical_artifact_refs_from_paths(
        {
            "pipeline_summary_json": str(output_root / "pipeline_summary.json"),
            "report_version_manifest_json": str(output_root / REPORT_VERSION_MANIFEST_NAME),
            "supervision_recovery_summary_json": str(output_root / "supervision_recovery" / "summary.json"),
            "tradeoff_report_summary_json": str(output_root / "tradeoff_report" / "summary.json"),
            "tradeoff_report_pdf": str(output_root / "tradeoff_report" / "report.pdf"),
            "alignment_audit_json": str(output_root / "markov_alignment_audit.json"),
            "alignment_audit_markdown": str(output_root / "markov_alignment_audit.md"),
        },
        phase_id="aggregate",
        required=False,
    )


def _tradeoff_result_rows(
    *,
    spec: ExperimentSpec,
    manifest: Mapping[str, Any],
) -> list[ResultRow]:
    benchmark_ref = benchmark_ref_from_parts(
        family="markov_full_doc",
        scope="pipeline",
        name="tradeoff_pipeline",
    )
    method_ref = method_ref_from_markov_full_doc_run(
        family="tree_neural",
        variant="pipeline_summary",
        adapter="markov_tree",
    )
    rows: list[ResultRow] = []
    phase_counts = dict(manifest.get("phase_task_counts", {}) or {})
    for phase_name, payload in phase_counts.items():
        rows.append(
            ResultRow(
                experiment_id=str(spec.experiment_id),
                phase=str(phase_name),
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                metric_name="worker_tasks",
                metric_value=int(dict(payload).get("worker_tasks", 0) or 0),
                artifact_refs=("pipeline_summary_json",),
                metadata={"source": "phase_task_counts"},
            )
        )
    selected_sources = dict(manifest.get("selected_sources", {}) or {})
    for source_name, payload in selected_sources.items():
        rows.append(
            ResultRow(
                experiment_id=str(spec.experiment_id),
                phase="report",
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                metric_name=f"source_status::{source_name}",
                metric_value=str(dict(payload).get("status", "") or ""),
                artifact_refs=("pipeline_summary_json",),
                metadata={"source": "selected_sources"},
            )
        )
    return rows


def _supervision_recovery_result_rows_from_summary(
    *,
    spec: ExperimentSpec,
    summary: Mapping[str, Any],
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    scopes = dict(summary.get("scopes") or {})
    tree_family = str(summary.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY) or SUPERVISION_RECOVERY_TREE_FAMILY)
    recoverable_scope_key = str(
        summary.get("recoverable_scope_key", SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK)
        or SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK
    )
    structural_hardness_grid = str(
        summary.get("structural_hardness_grid", SUPERVISION_RECOVERY_STRUCTURAL_GRID)
        or SUPERVISION_RECOVERY_STRUCTURAL_GRID
    )
    for scope_key, scope_payload in scopes.items():
        scope_data = dict(scope_payload or {})
        scope_label = str(scope_data.get("scope_label", "") or "")
        benchmark_scope = (
            str(recoverable_scope_key)
            if str(scope_key) == str(recoverable_scope_key)
            else str(structural_hardness_grid)
        )
        benchmark_cell = "" if str(scope_key) == str(recoverable_scope_key) else str(scope_key)
        benchmark_ref = benchmark_ref_from_parts(
            family="markov_full_doc",
            scope=benchmark_scope,
            cell=benchmark_cell,
            name=scope_label or benchmark_scope,
        )
        for item in list(scope_data.get("rows_by_train_docs") or []):
            payload = dict(item or {})
            train_doc_count = int(_safe_int(payload.get("train_doc_count"), 0))
            for row_payload in list(payload.get("rows") or []):
                row = dict(row_payload or {})
                package_name = str(row.get("package_name", "") or "")
                tree_method_ref = method_ref_from_markov_full_doc_run(
                    family=tree_family,
                    variant=package_name or "unknown_package",
                    adapter="markov_tree",
                    config_like=row,
                    package_name=package_name,
                    metadata={
                        "scope_key": str(scope_key),
                        "scope_label": scope_label,
                        "baseline_family": tree_family,
                    },
                    mean_leaves_per_doc=_safe_float(
                        row.get("test_mean_leaves_per_doc"), None
                    ),
                )
                metric_map = {
                    "test_root_mae": _safe_float(row.get("tree_test_root_mae"), float("nan")),
                    "val_root_mae": _safe_float(row.get("tree_val_root_mae"), float("nan")),
                    "test_leaf_mae": _safe_float(row.get("tree_test_leaf_mae"), float("nan")),
                    "test_merge_mae": _safe_float(row.get("tree_test_merge_mae"), float("nan")),
                    "test_full_law_objective": _safe_float(row.get("tree_test_full_law_objective"), float("nan")),
                    "val_full_law_objective": _safe_float(row.get("tree_val_full_law_objective"), float("nan")),
                    "test_active_objective": _safe_float(row.get("tree_test_active_objective"), float("nan")),
                    "val_active_objective": _safe_float(row.get("tree_val_active_objective"), float("nan")),
                    "best_epoch": _safe_float(row.get("tree_best_epoch"), float("nan")),
                }
                for metric_name, metric_value in metric_map.items():
                    if not math.isfinite(float(metric_value)):
                        continue
                    rows.append(
                        ResultRow(
                            experiment_id=str(spec.experiment_id),
                            phase="supervision_recovery",
                            benchmark_ref=benchmark_ref,
                            method_ref=tree_method_ref,
                            split="test" if metric_name.startswith("test_") else ("validation" if metric_name.startswith("val_") else ""),
                            train_docs=train_doc_count,
                            supervision_ref=tree_method_ref.supervision,
                            metric_name=str(metric_name),
                            metric_value=float(metric_value),
                            artifact_refs=("supervision_recovery_summary_json",),
                            metadata={
                                "package_name": package_name,
                                "scope_key": str(scope_key),
                                "scope_label": scope_label,
                            },
                        )
                    )
                fno_rows = dict(row.get("fno_family_rows") or {})
                for family_name, family_payload in fno_rows.items():
                    fno_data = dict(family_payload or {})
                    fno_method_ref = method_ref_from_markov_full_doc_run(
                        family=str(family_name),
                        variant=str(package_name or "matched_reference"),
                        adapter="markov_tree",
                        config_like=row,
                        package_name=package_name,
                        metadata={
                            "scope_key": str(scope_key),
                            "scope_label": scope_label,
                            "reference_package": package_name,
                        },
                        mean_leaves_per_doc=_safe_float(
                            row.get("test_mean_leaves_per_doc"), None
                        ),
                    )
                    for metric_name in ("test_root_mae", "val_root_mae"):
                        metric_value = _safe_float(fno_data.get(metric_name), float("nan"))
                        if not math.isfinite(float(metric_value)):
                            continue
                        rows.append(
                            ResultRow(
                                experiment_id=str(spec.experiment_id),
                                phase="supervision_recovery",
                                benchmark_ref=benchmark_ref,
                                method_ref=fno_method_ref,
                                split="test" if metric_name.startswith("test_") else "validation",
                                train_docs=train_doc_count,
                                supervision_ref=fno_method_ref.supervision,
                                metric_name=str(metric_name),
                                metric_value=float(metric_value),
                                artifact_refs=("supervision_recovery_summary_json",),
                                metadata={
                                    "package_name": package_name,
                                    "scope_key": str(scope_key),
                                    "scope_label": scope_label,
                                    "reference_family": str(family_name),
                                },
                            )
                        )
    return rows


def _profile_task(
    *,
    name: str,
    output_dir: Path,
    train_docs: int,
    val_docs: int,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    exact_doc_limit: int,
    leaf_tokens: int,
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    device_mode: str,
) -> SubprocessTask:
    argv = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--train-docs",
        str(train_docs),
        "--val-docs",
        str(val_docs),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--pack-mode",
        "fixed_fused",
        "--cases",
        "no_autotune",
        "--c1-weight",
        "1.0",
        "--c2-weight",
        "1.0",
        "--c3-weight",
        "0.0",
        "--root-weight",
        "1.0",
        "--leaf-supervision-kind",
        "full_sketch",
        "--internal-supervision-kind",
        "count_only",
        "--leaf-tokens",
        str(leaf_tokens),
        "--min-tokens",
        str(min_tokens),
        "--max-tokens",
        str(max_tokens),
        "--min-segments",
        str(min_segments),
        "--max-segments",
        str(max_segments),
        "--screen-doc-limit",
        str(exact_doc_limit),
        "--exact-doc-limit",
        str(exact_doc_limit),
        "--torch-threads",
        "1",
        "--torch-interop-threads",
        "1",
    ]
    if device_mode in {"cpu", "cuda"}:
        argv.extend(["--device", device_mode])
    return SubprocessTask(
        name=name,
        argv=argv,
        output_path=output_dir / "summary.json",
        log_path=output_dir / "run.log",
    )


def _base_ops_config(args: argparse.Namespace, *, seed: int, data_seed: int, train_docs: int, val_docs: int, test_docs: int, batch_size: int, n_epochs: int, fixed_leaf_tokens: int | None = None) -> Dict[str, Any]:
    feature_dim = int(args.theorem_feature_dim)
    preload_splits = _parse_str_list(
        getattr(args, "runtime_preload_splits", None),
        ("train", "val", "test"),
    )
    config = {
        "problem_id": "markov_ops_count",
        "method_id": "tree_neural",
        "law_set_id": LAW_SET_ALL,
        "model_family": "fno",
        "n_regimes": int(args.n_regimes) if hasattr(args, "n_regimes") else 4,
        "vocab_size": 32,
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "min_segments": int(args.min_segments),
        "max_segments": int(args.max_segments),
        "fixed_leaf_tokens": int(fixed_leaf_tokens or args.fixed_leaf_tokens),
        "train_docs": int(train_docs),
        "val_docs": int(val_docs),
        "test_docs": int(test_docs),
        "feature_mode": "full",
        "state_dim": int(args.state_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_epochs": int(n_epochs),
        "batch_size": int(batch_size),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "fno_width": int(args.fno_width),
        "fno_n_modes": int(args.fno_n_modes),
        "fno_n_layers": int(args.fno_n_layers),
        "tree_model_version": "v2",
        "tree_batch_runtime_mode": "unified_v2",
        "tree_batch_pack_mode": "fixed_fused",
        "tree_batch_autotune": bool(
            getattr(args, "tree_batch_autotune", False)
        ),
        "tree_batch_structural_pad_limit": float(
            getattr(args, "runtime_tree_batch_structural_pad_limit", 0.5)
        ),
        "tree_batch_auto_queue_min_docs": int(
            getattr(args, "runtime_tree_batch_auto_queue_min_docs", 8)
        ),
        "tree_batch_auto_queue_min_fill_ratio": float(
            getattr(args, "runtime_tree_batch_auto_queue_min_fill_ratio", 0.5)
        ),
        "tree_training_schedule": str(
            getattr(args, "tree_training_schedule", "single_stage") or "single_stage"
        ),
        "exact_metric_final_doc_limit": int(
            getattr(args, "exact_metric_final_doc_limit", 0)
        ),
        "tree_posttrain_train_doc_limit": int(
            getattr(args, "tree_posttrain_train_doc_limit", 0)
        ),
        "tree_stage1_artifact_root": _optional_path_text(
            getattr(args, "tree_stage1_artifact_root", None)
        ),
        "tree_stage1_resume_if_available": bool(
            getattr(args, "tree_stage1_resume_if_available", True)
        ),
        "tree_exact_eval_max_docs": int(
            getattr(args, "tree_exact_eval_max_docs", 0)
        ),
        "prepared_data_root": _optional_path_text(
            getattr(args, "prepared_data_root", None)
        ),
        "prepared_data_allow_create": bool(
            getattr(args, "prepared_data_allow_create", True)
        ),
        "diagnostic_detail_mode": str(
            getattr(args, "diagnostic_detail_mode", "summary")
        ),
        "raw_diagnostic_artifact_dir": _optional_path_text(
            getattr(args, "raw_diagnostic_artifact_dir", None)
        ),
        "tree_document_loss_normalization_mode": "auto",
        "tree_task_head_mode": "theorem_feature_scalar",
        "tree_theorem_surface_mode": "factorized_score_fiber",
        "tree_summary_spec_root_mode": "factored_theorem_readout",
        "tree_theorem_feature_dim": int(feature_dim),
        "tree_theorem_feature_hidden_dim": int(args.theorem_feature_hidden_dim),
        "tree_theorem_score_dim": 1,
        "tree_theorem_fiber_dim": max(1, feature_dim - 1),
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
        "root_weight": 1.0,
        "schedule_consistency_weight": 0.0,
        "audit_policy": "fraction",
        "audit_fraction": 1.0,
        "c3_audit_strategy": "uniform",
        "c3_include_root": True,
        "leaf_query_rate": 1.0,
        "include_root_query": True,
        "tree_stage1_screen_doc_limit": 0,
        "tree_stage1_final_exact_doc_limit": 0,
        "gpu_runtime_data_mode": str(getattr(args, "runtime_data_mode", "resident")),
        "gpu_runtime_bucket_mode": str(
            getattr(args, "runtime_bucket_mode", "exact_then_bucketed")
        ),
        "gpu_runtime_preload_splits": tuple(str(item) for item in preload_splits),
        "gpu_runtime_preload_targets": bool(
            getattr(args, "runtime_preload_targets", True)
        ),
        "gpu_runtime_workers_per_mig": int(
            getattr(args, "runtime_workers_per_mig", 1)
        ),
        "gpu_runtime_allow_multi_worker_screen": bool(
            getattr(args, "runtime_allow_multi_worker_screen", True)
        ),
        "gpu_runtime_capacity_workers_per_mig": int(
            getattr(args, "runtime_capacity_workers_per_mig", 2)
        ),
        "use_cuda": bool(args.device_mode != "cpu"),
        "cuda_device": 0 if args.device_mode != "cpu" else None,
        "torch_threads": 1,
        "seed": int(seed),
        "data_seed": int(data_seed),
        "model_seed": int(seed),
    }
    if getattr(args, "tree_stage1_epochs", None) is not None:
        config["tree_stage1_epochs"] = int(args.tree_stage1_epochs)
    if getattr(args, "tree_stage2_epochs", None) is not None:
        config["tree_stage2_epochs"] = int(args.tree_stage2_epochs)
    return config


def _law_phase_doc_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    if str(args.preset).strip().lower() == "standard":
        return (
            int(args.train_docs),
            int(args.val_docs),
            int(args.test_docs),
        )
    return (
        min(int(args.train_docs), 512),
        min(int(args.val_docs), 64),
        min(int(args.test_docs), 64),
    )


def _direct_task(
    *,
    root: Path,
    name: str,
    config: Mapping[str, Any],
    output_filename: str = "summary.json",
    worker_kind: str = "ops_count",
    extra_payload: Mapping[str, Any] | None = None,
) -> SubprocessTask:
    task_dir = root / name
    task_dir.mkdir(parents=True, exist_ok=True)
    output_json = task_dir / str(output_filename)
    task_json = task_dir / "task.request"
    progress_json = task_dir / "progress.json"
    extra_payload_dict = dict(extra_payload or {})
    effective_config = _resolved_full_doc_task_config(
        worker_kind=str(worker_kind),
        config=config,
        task_payload=extra_payload_dict,
    )
    if str(worker_kind) == "full_doc_diagnostics":
        from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # type: ignore
            _effective_config_for_family,
            _effective_train_config_for_full_doc_run,
            resolve_full_doc_diagnostic_benchmark,
            resolve_full_doc_diagnostic_grid,
        )
        from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # type: ignore
            OPSCountConfig,
        )

        families = tuple(
            str(item).strip()
            for item in list(extra_payload_dict.get("baseline_families") or ())
            if str(item).strip()
        )
        if len(families) == 1:
            hardness_grid = _gs(extra_payload_dict, "hardness_grid")
            if hardness_grid:
                selected_ids = tuple(
                    str(value).strip()
                    for value in list(extra_payload_dict.get("grid_cell_ids") or ())
                    if str(value).strip()
                )
                benchmarks = resolve_full_doc_diagnostic_grid(hardness_grid)
                benchmark = next(
                    (
                        candidate
                        for candidate in benchmarks
                        if not selected_ids
                        or str(candidate.cell_id or "").strip() in selected_ids
                    ),
                    benchmarks[0],
                )
            else:
                benchmark = resolve_full_doc_diagnostic_benchmark(
                    str(
                        extra_payload_dict.get("benchmark_name", "recoverable_v4")
                        or "recoverable_v4"
                    )
                )
            supported_keys = _ops_count_supported_config_keys()
            resolved_config_dict = dict(effective_config)
            effective_family_config = _effective_config_for_family(
                benchmark=benchmark,
                baseline_family=str(families[0]),
                config=OPSCountConfig(
                    **{
                        str(key): value
                        for key, value in resolved_config_dict.items()
                        if str(key) in supported_keys
                    }
                ),
            )
            effective_config = asdict(
                _effective_train_config_for_full_doc_run(
                    benchmark=benchmark,
                    baseline_family=str(families[0]),
                    train_doc_count=_safe_int(
                        dict(effective_config).get("train_docs"),
                        default=0,
                    ),
                    config=effective_family_config,
                )
            )
            for key, value in resolved_config_dict.items():
                if str(key) not in supported_keys:
                    effective_config[str(key)] = value
            comparison_mode = normalize_markov_comparison_mode(
                str(effective_config.get("comparison_mode", "legacy") or "legacy")
            )
            comparison_surface = resolve_markov_comparable_surface(
                benchmark=benchmark,
                config=effective_config,
                comparison_mode=comparison_mode,
            )
            payload_surface = comparison_surface.to_dict()
            payload_diff = comparison_surface_diff(
                expected_surface=comparison_surface,
                actual_config=effective_config,
            )
            effective_config["comparison_mode"] = str(comparison_mode)
            effective_config["comparison_surface_snapshot"] = dict(payload_surface)
            effective_config["comparison_surface_diff"] = dict(payload_diff)
    payload = {
        "name": name,
        "config": _serialized_worker_config(
            worker_kind=str(worker_kind),
            config=effective_config,
        ),
        "output_json": str(output_json),
        "worker_kind": str(worker_kind),
        "progress_path": str(progress_json),
    }
    if str(worker_kind) == "full_doc_diagnostics":
        payload["comparison_mode"] = str(
            dict(effective_config).get("comparison_mode", "legacy")
        )
        payload["comparison_surface_snapshot"] = dict(
            dict(effective_config).get("comparison_surface_snapshot") or {}
        )
        payload["comparison_surface_diff"] = dict(
            dict(effective_config).get("comparison_surface_diff") or {}
        )
    if extra_payload_dict:
        payload.update(dict(extra_payload_dict))
    task_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    baseline_families = tuple(
        str(item).strip()
        for item in list(extra_payload_dict.get("baseline_families") or ())
        if str(item).strip()
    )
    metadata: Dict[str, Any] = {
        "task_name": str(name),
        "worker_kind": str(worker_kind),
        "progress_path": str(progress_json),
        "train_docs": _safe_int(dict(effective_config).get("train_docs"), default=0),
        "n_epochs": _effective_task_epoch_total(
            worker_kind=str(worker_kind),
            config=effective_config,
            task_payload=extra_payload_dict,
        ),
    }
    if str(worker_kind) == "full_doc_diagnostics":
        metadata["comparison_mode"] = str(
            dict(effective_config).get("comparison_mode", "legacy")
        )
    scope = str(
        dict(effective_config).get("pipeline_supervision_recovery_scope", "") or ""
    ).strip()
    package = str(
        dict(effective_config).get("pipeline_supervision_recovery_package", "") or ""
    ).strip()
    if scope:
        metadata["scope"] = scope
    if package:
        metadata["package"] = package
    model_family = _infer_model_family_from_task_name(name)
    if not model_family and len(baseline_families) == 1:
        model_family = str(baseline_families[0])
    if model_family:
        metadata["model_family"] = model_family
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task",
        str(task_json),
    ]
    return SubprocessTask(
        name=name,
        argv=argv,
        output_path=output_json,
        log_path=task_dir / "run.log",
        metadata=metadata,
        progress_path=progress_json,
    )


def _load_profile_run(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    return dict((payload.get("runs", {}) or {}).get("no_autotune", {}) or {})


def _aggregate_batch_timing(task_infos: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def _runtime_payload(summary: Mapping[str, Any]) -> Dict[str, Any]:
        runtime = dict(summary.get("runtime_efficiency", {}) or {})
        metrics = dict(summary.get("batching_metrics", {}) or {})

        def _pick(name: str, default: Any) -> Any:
            if name in runtime:
                return runtime.get(name)
            return metrics.get(name, default)

        return {
            "runtime_data_mode": str(_pick("runtime_data_mode", "")),
            "runtime_bucket_mode": str(_pick("runtime_bucket_mode", "")),
            "runtime_workers_per_mig": int(_pick("runtime_workers_per_mig", 0) or 0),
            "resident_store_build_time_s": float(
                _pick("resident_store_build_time_s", 0.0) or 0.0
            ),
            "steady_state_h2d_bytes": int(
                _pick("steady_state_h2d_bytes", 0) or 0
            ),
            "steady_state_h2d_events": int(
                _pick("steady_state_h2d_events", 0) or 0
            ),
            "steady_state_h2d_time_s": float(
                _pick("steady_state_h2d_time_s", 0.0) or 0.0
            ),
            "resident_store_hits": int(_pick("resident_store_hits", 0) or 0),
            "resident_store_misses": int(_pick("resident_store_misses", 0) or 0),
            "auto_queue_family_count": int(
                _pick("auto_queue_family_count", 0) or 0
            ),
            "structural_padding_waste_ratio": float(
                _pick("structural_padding_waste_ratio", 0.0) or 0.0
            ),
            "auto_queue_fused_batches": int(
                _pick("auto_queue_fused_batches", 0) or 0
            ),
            "auto_queue_generic_fallback_batches": int(
                _pick("auto_queue_generic_fallback_batches", 0) or 0
            ),
            "fixed_shape_dense_bucket_store_hits": int(
                _pick("fixed_shape_dense_bucket_store_hits", 0) or 0
            ),
            "cpu_fallback_reason_counts": dict(
                _pick("cpu_fallback_reason_counts", {}) or {}
            ),
        }

    rows = []
    runtime_rows: List[Dict[str, Any]] = []
    for info in task_infos:
        payload = _read_json(Path(str(info["output_path"])))
        summary = dict((payload.get("runs", {}) or {}).get("no_autotune", {}) or {})
        timing = dict(summary.get("timing_breakdown", {}) or {})
        metrics = dict(summary.get("batching_metrics", {}) or {})
        runtime_metrics = _runtime_payload(summary)
        config = dict(payload.get("config", {}) or {})
        output_path = Path(str(info["output_path"]))
        parent_name = output_path.parent.name
        digits = "".join(ch for ch in parent_name if ch.isdigit())
        batch_size = int(digits) if digits else int(summary.get("config_batch_size", 0) or 0)
        train_docs = float(config.get("train_docs", 0.0) or 0.0)
        wall = max(float(summary.get("wall_clock_s", 0.0)), 1e-9)
        train_loop = max(float(timing.get("train_loop_s", 0.0)), 1e-9)
        rows.append(
            {
                "batch_size": batch_size,
                "wall_clock_s": float(summary.get("wall_clock_s", 0.0)),
                "train_loop_s": float(timing.get("train_loop_s", 0.0)),
                "docs_per_s_wall": float(train_docs / wall) if train_docs > 0.0 else 0.0,
                "docs_per_s_train_loop": float(train_docs / train_loop) if train_docs > 0.0 else 0.0,
                "gpu_reserved_mem_peak_gb": float(metrics.get("gpu_reserved_mem_peak_gb", 0.0)),
                "mean_docs_per_batch": float(metrics.get("mean_docs_per_batch", 0.0)),
                "mean_nodes_per_batch": float(metrics.get("mean_nodes_per_batch", 0.0)),
                "train_forward_time_s": float(metrics.get("train_forward_time_s", 0.0)),
                "train_backward_time_s": float(metrics.get("train_backward_time_s", 0.0)),
                "screen_eval_s": float(timing.get("screen_eval_s", 0.0)),
                "exact_metric_eval_s": float(timing.get("exact_metric_eval_s", 0.0)),
                "eval_total_s": float(timing.get("eval_total_s", 0.0)),
                **runtime_metrics,
            }
        )
        runtime_rows.append(runtime_metrics)
    rows.sort(key=lambda row: int(row["batch_size"]))
    runtime_counter = Counter(
        row["runtime_data_mode"]
        for row in runtime_rows
        if str(row.get("runtime_data_mode", "")).strip()
    )
    bucket_counter = Counter(
        row["runtime_bucket_mode"]
        for row in runtime_rows
        if str(row.get("runtime_bucket_mode", "")).strip()
    )
    cpu_fallback_reason_counts: Dict[str, int] = {}
    for row in runtime_rows:
        for key, value in dict(row.get("cpu_fallback_reason_counts", {}) or {}).items():
            cpu_fallback_reason_counts[str(key)] = (
                int(cpu_fallback_reason_counts.get(str(key), 0)) + int(value)
            )
    runtime_efficiency = {
        "runtime_data_mode": runtime_counter.most_common(1)[0][0] if runtime_counter else "",
        "runtime_bucket_mode": bucket_counter.most_common(1)[0][0] if bucket_counter else "",
        "runtime_workers_per_mig_mean": (
            sum(int(row.get("runtime_workers_per_mig", 0) or 0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "resident_store_build_time_s_mean": (
            sum(float(row.get("resident_store_build_time_s", 0.0) or 0.0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "steady_state_h2d_bytes_mean": (
            sum(int(row.get("steady_state_h2d_bytes", 0) or 0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "steady_state_h2d_events_mean": (
            sum(int(row.get("steady_state_h2d_events", 0) or 0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "steady_state_h2d_time_s_mean": (
            sum(float(row.get("steady_state_h2d_time_s", 0.0) or 0.0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "resident_store_hits_total": sum(
            int(row.get("resident_store_hits", 0) or 0) for row in runtime_rows
        ),
        "resident_store_misses_total": sum(
            int(row.get("resident_store_misses", 0) or 0) for row in runtime_rows
        ),
        "auto_queue_family_count_mean": (
            sum(int(row.get("auto_queue_family_count", 0) or 0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "structural_padding_waste_ratio_mean": (
            sum(float(row.get("structural_padding_waste_ratio", 0.0) or 0.0) for row in runtime_rows)
            / len(runtime_rows)
            if runtime_rows
            else 0.0
        ),
        "auto_queue_fused_batches_total": sum(
            int(row.get("auto_queue_fused_batches", 0) or 0) for row in runtime_rows
        ),
        "auto_queue_generic_fallback_batches_total": sum(
            int(row.get("auto_queue_generic_fallback_batches", 0) or 0)
            for row in runtime_rows
        ),
        "fixed_shape_dense_bucket_store_hits_total": sum(
            int(row.get("fixed_shape_dense_bucket_store_hits", 0) or 0)
            for row in runtime_rows
        ),
        "cpu_fallback_reason_counts": cpu_fallback_reason_counts,
    }
    return {"summary": rows, "runtime_efficiency": runtime_efficiency}


def _aggregate_medium_grid(task_infos: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    runs = []
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    train_docs_value = 0
    epochs_value = 0
    for info in task_infos:
        payload = _read_json(Path(str(info["output_path"])))
        summary = dict((payload.get("runs", {}) or {}).get("no_autotune", {}) or {})
        timing = dict(summary.get("timing_breakdown", {}) or {})
        runtime = dict(summary.get("runtime_efficiency", {}) or {})
        output_path = Path(str(info["output_path"]))
        name = output_path.parent.name
        parts = name.split("_")
        batch_size = int(parts[0].replace("bs", ""))
        seed = int(parts[1].replace("seed", ""))
        train_docs = float((payload.get("config", {}) or {}).get("train_docs", 0.0) or 0.0)
        if not train_docs_value:
            train_docs_value = int(train_docs)
        epochs_completed = max(int(summary.get("epochs_completed", 0) or 0), 1)
        if not epochs_value:
            epochs_value = int(epochs_completed)
        wall = max(float(summary.get("wall_clock_s", 0.0)), 1e-9)
        row = {
            "run": name,
            "batch_size": batch_size,
            "seed": seed,
            "best_val_mae": float(summary.get("best_val_mae", float("nan"))),
            "val_root_mae": float((summary.get("val", {}) or {}).get("root_mae", float("nan"))),
            "wall_clock_s": float(summary.get("wall_clock_s", 0.0)),
            "docs_per_s_wall_effective": float(train_docs * epochs_completed / wall),
            "exact_metric_eval_s": float(timing.get("exact_metric_eval_s", 0.0)),
            "eval_total_s": float(timing.get("eval_total_s", 0.0)),
            "gpu_reserved_mem_peak_gb": float((summary.get("batching_metrics", {}) or {}).get("gpu_reserved_mem_peak_gb", 0.0)),
            "runtime_data_mode": str(runtime.get("runtime_data_mode", "")),
            "runtime_bucket_mode": str(runtime.get("runtime_bucket_mode", "")),
            "resident_store_build_time_s": float(runtime.get("resident_store_build_time_s", 0.0) or 0.0),
            "steady_state_h2d_bytes": int(runtime.get("steady_state_h2d_bytes", 0) or 0),
            "steady_state_h2d_time_s": float(runtime.get("steady_state_h2d_time_s", 0.0) or 0.0),
        }
        runs.append(row)
        grouped.setdefault(batch_size, []).append(row)
    by_batch_size: Dict[str, Dict[str, Any]] = {}
    for batch_size, rows in sorted(grouped.items()):
        best_run = min(rows, key=lambda row: _safe_float(row.get("best_val_mae"), float("inf")))
        by_batch_size[str(batch_size)] = {
            "n_runs": len(rows),
            "mean_best_val_mae": sum(_safe_float(r.get("best_val_mae"), 0.0) for r in rows) / len(rows),
            "mean_val_root_mae": sum(_safe_float(r.get("val_root_mae"), 0.0) for r in rows) / len(rows),
            "mean_wall_clock_s": sum(_safe_float(r.get("wall_clock_s"), 0.0) for r in rows) / len(rows),
            "mean_docs_per_s_wall_effective": sum(_safe_float(r.get("docs_per_s_wall_effective"), 0.0) for r in rows) / len(rows),
            "mean_exact_metric_eval_s": sum(_safe_float(r.get("exact_metric_eval_s"), 0.0) for r in rows) / len(rows),
            "mean_eval_total_s": sum(_safe_float(r.get("eval_total_s"), 0.0) for r in rows) / len(rows),
            "mean_gpu_reserved_mem_peak_gb": sum(_safe_float(r.get("gpu_reserved_mem_peak_gb"), 0.0) for r in rows) / len(rows),
            "mean_resident_store_build_time_s": sum(_safe_float(r.get("resident_store_build_time_s"), 0.0) for r in rows) / len(rows),
            "mean_steady_state_h2d_bytes": sum(_safe_float(r.get("steady_state_h2d_bytes"), 0.0) for r in rows) / len(rows),
            "mean_steady_state_h2d_time_s": sum(_safe_float(r.get("steady_state_h2d_time_s"), 0.0) for r in rows) / len(rows),
            "best_run": str(best_run.get("run")),
            "best_run_val_mae": float(best_run.get("best_val_mae", float("nan"))),
        }
    return {
        "runs": runs,
        "by_batch_size": by_batch_size,
        "train_docs": int(train_docs_value),
        "epochs": int(epochs_value),
    }


def _aggregate_docs_epochs(task_infos: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for info in task_infos:
        payload = _read_json(Path(str(info["output_path"])))
        summary = dict((payload.get("runs", {}) or {}).get("no_autotune", {}) or {})
        timing = dict(summary.get("timing_breakdown", {}) or {})
        runtime = dict(summary.get("runtime_efficiency", {}) or {})
        name = Path(str(info["output_path"])).parent.name
        parts = name.split("_")
        train_docs = int(parts[0].replace("train", ""))
        epochs = int(parts[1].replace("ep", ""))
        epochs_completed = max(int(summary.get("epochs_completed", 0) or 0), 1)
        wall = max(float(summary.get("wall_clock_s", 0.0)), 1e-9)
        row = {
            "run": name,
            "train_docs": train_docs,
            "epochs": epochs,
            "wall_clock_s": float(summary.get("wall_clock_s", 0.0)),
            "epochs_completed": epochs_completed,
            "best_val_mae": float(summary.get("best_val_mae", float("nan"))),
            "val_root_mae": float((summary.get("val", {}) or {}).get("root_mae", float("nan"))),
            "val_exact_match": float((summary.get("val", {}) or {}).get("exact_match", float("nan"))),
            "train_root_mae": float((summary.get("train", {}) or {}).get("root_mae", float("nan"))),
            "train_loop_s": float(timing.get("train_loop_s", 0.0)),
            "exact_metric_eval_s": float(timing.get("exact_metric_eval_s", 0.0)),
            "eval_total_s": float(timing.get("eval_total_s", 0.0)),
            "docs_per_s_wall_effective": float(train_docs * epochs_completed / wall),
            "gpu_reserved_mem_peak_gb": float((summary.get("batching_metrics", {}) or {}).get("gpu_reserved_mem_peak_gb", 0.0)),
            "runtime_data_mode": str(runtime.get("runtime_data_mode", "")),
            "runtime_bucket_mode": str(runtime.get("runtime_bucket_mode", "")),
            "resident_store_build_time_s": float(runtime.get("resident_store_build_time_s", 0.0) or 0.0),
            "steady_state_h2d_bytes": int(runtime.get("steady_state_h2d_bytes", 0) or 0),
            "steady_state_h2d_time_s": float(runtime.get("steady_state_h2d_time_s", 0.0) or 0.0),
        }
        rows.append(row)
        grouped.setdefault(train_docs, []).append(row)
    rows.sort(key=lambda row: (int(row["train_docs"]), int(row["epochs"])))
    by_train_docs = {}
    for train_docs, subrows in sorted(grouped.items()):
        best = min(subrows, key=lambda row: _safe_float(row.get("best_val_mae"), float("inf")))
        fastest = max(subrows, key=lambda row: _safe_float(row.get("docs_per_s_wall_effective"), float("-inf")))
        by_train_docs[str(train_docs)] = {
            "rows": sorted(subrows, key=lambda row: int(row["epochs"])),
            "best_val_run": str(best["run"]),
            "best_val_mae": float(best["best_val_mae"]),
            "best_val_epochs": int(best["epochs"]),
            "fastest_run": str(fastest["run"]),
            "fastest_docs_per_s": float(fastest["docs_per_s_wall_effective"]),
            "fastest_epochs": int(fastest["epochs"]),
        }
    return {"rows": rows, "by_train_docs": by_train_docs}


def _load_ops_payloads(root: Path) -> List[Mapping[str, Any]]:
    payloads = []
    for path in sorted(root.rglob("summary.json")):
        try:
            payload = dict(_read_json(path))
            payload["source_summary_json"] = str(path)
            payloads.append(payload)
        except Exception:
            continue
    return payloads


def _load_supervision_recovery_refresh_payloads(root: Path) -> List[Mapping[str, Any]]:
    run_payloads: List[Mapping[str, Any]] = []
    for path in sorted(root.rglob("summary_artifacts/runs/*.json")):
        try:
            run_payload = dict(_read_json(path))
        except Exception:
            continue
        if not run_payload:
            continue
        run_payloads.append(
            {
                "config": dict(run_payload.get("config") or {}),
                "aggregate_rows": [dict(run_payload)],
                "runs": [dict(run_payload)],
                "benchmark": str(run_payload.get("benchmark", "") or ""),
                "hardness_grid": str(run_payload.get("hardness_grid", "") or ""),
                "source_summary_json": str(path),
            }
        )
    if run_payloads:
        return run_payloads
    return _load_ops_payloads(root)


def _refresh_phase_config_fingerprint(
    *,
    manifest: Mapping[str, Any],
    phase: str,
) -> str:
    selected_sources = dict(manifest.get("selected_sources") or {})
    for source_key, spec in REPORT_SOURCE_SPECS.items():
        if str(spec.get("phase", "")) != str(phase):
            continue
        entry = dict(selected_sources.get(str(source_key)) or {})
        fingerprint = str(entry.get("config_fingerprint", "") or "").strip()
        if fingerprint:
            return fingerprint
    phase_attempts = dict(
        dict(manifest.get("phase_attempts") or {}).get(str(phase)) or {}
    )
    attempts = dict(phase_attempts.get("attempts") or {})
    for attempt_id in sorted(attempts.keys(), reverse=True):
        fingerprint = str(
            dict(attempts.get(attempt_id) or {}).get("config_fingerprint", "") or ""
        ).strip()
        if fingerprint:
            return fingerprint
    return _stable_fingerprint({"phase": str(phase), "mode": "refresh_existing"})


def _latest_phase_attempt_dirs(output_root: Path, phase: str) -> List[Path]:
    attempts_root = output_root / str(phase) / "attempts"
    if not attempts_root.exists():
        return []
    return sorted(
        [path for path in attempts_root.iterdir() if path.is_dir()],
        key=lambda path: str(path.name),
        reverse=True,
    )


def _best_existing_raw_root(output_root: Path, phase: str) -> Path | None:
    candidates: List[tuple[int, str, Path]] = []
    for attempt_dir in _latest_phase_attempt_dirs(output_root, phase):
        raw_root = attempt_dir / "raw"
        if not raw_root.exists():
            continue
        payload_count = len(list(raw_root.rglob("summary.json")))
        if payload_count <= 0:
            continue
        candidates.append((int(payload_count), str(attempt_dir.name), raw_root))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (int(item[0]), str(item[1])), reverse=True)
    return candidates[0][2]


def _is_terminal_progress_payload(payload: Mapping[str, Any] | None) -> bool:
    payload_map = dict(payload or {})
    state = str(payload_map.get("state", "") or "").strip().lower()
    active_items = int(_safe_int(payload_map.get("active_items", 0), 0))
    pending_items = int(_safe_int(payload_map.get("pending_items", 0), 0))
    return state in {"completed", "failed"} and active_items <= 0 and pending_items <= 0


def _terminal_status_items_total(
    *payloads: Mapping[str, Any] | None,
) -> int:
    for payload in payloads:
        payload_map = dict(payload or {})
        items_total = int(
            _safe_int(
                payload_map.get(
                    "items_total",
                    payload_map.get("total_items", 0),
                ),
                0,
            )
        )
        if items_total > 0:
            return int(items_total)
    return 0


def _reset_terminal_bucket_payload(
    buckets: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    refreshed: Dict[str, Dict[str, Any]] = {}
    for bucket_name, raw_bucket in dict(buckets or {}).items():
        if not isinstance(raw_bucket, Mapping):
            continue
        bucket = dict(raw_bucket)
        total = int(_safe_int(bucket.get("total", 0), 0))
        if total <= 0:
            total = (
                int(_safe_int(bucket.get("completed", 0), 0))
                + int(_safe_int(bucket.get("failed", 0), 0))
                + int(_safe_int(bucket.get("active", 0), 0))
                + int(_safe_int(bucket.get("pending", 0), 0))
            )
        bucket["total"] = int(total)
        bucket["completed"] = int(total)
        bucket["failed"] = 0
        bucket["active"] = 0
        bucket["pending"] = 0
        bucket["percent_complete"] = 100.0
        epochs_total = int(_safe_int(bucket.get("epochs_total", 0), 0))
        if epochs_total > 0:
            bucket["epochs_total"] = int(epochs_total)
            bucket["epochs_completed"] = int(epochs_total)
            bucket["epoch_percent"] = 100.0
        refreshed[str(bucket_name)] = bucket
    return refreshed


def _refresh_terminal_tradeoff_status_files(
    *,
    args: argparse.Namespace,
    output_root: Path,
    run_plan: Mapping[str, Any],
) -> bool:
    experiment_status_path = output_root / "experiment_status.json"
    scheduler_status_path = output_root / "scheduler_status.json"
    experiment_payload = (
        _read_json(experiment_status_path) if experiment_status_path.exists() else {}
    )
    scheduler_payload = (
        _read_json(scheduler_status_path) if scheduler_status_path.exists() else {}
    )
    if not (
        _is_terminal_progress_payload(experiment_payload)
        or _is_terminal_progress_payload(scheduler_payload)
    ):
        return False

    items_total = _terminal_status_items_total(
        scheduler_payload,
        experiment_payload,
    )
    if items_total <= 0:
        return False

    spec = _tradeoff_experiment_spec(
        args=args,
        output_root=output_root,
        run_plan=run_plan,
    )
    _write_tradeoff_experiment_state(
        output_root=output_root,
        spec=spec,
        state="completed",
        active_phase="",
        items_total=items_total,
        completed_items=items_total,
        failed_items=0,
        active_items=0,
        pending_items=0,
    )

    refreshed_scheduler_payload = dict(scheduler_payload or {})
    refreshed_scheduler_payload["generated_at"] = _utc_now()
    refreshed_scheduler_payload["state"] = "completed"
    refreshed_scheduler_payload["active_phase"] = ""
    refreshed_scheduler_payload["items_total"] = int(items_total)
    refreshed_scheduler_payload["initial_items_total"] = int(
        max(
            int(_safe_int(refreshed_scheduler_payload.get("initial_items_total", 0), 0)),
            int(items_total),
        )
    )
    refreshed_scheduler_payload["dynamic_items_added"] = int(
        _safe_int(refreshed_scheduler_payload.get("dynamic_items_added", 0), 0)
    )
    refreshed_scheduler_payload["completed_items"] = int(items_total)
    refreshed_scheduler_payload["failed_items"] = 0
    refreshed_scheduler_payload["active_items"] = 0
    refreshed_scheduler_payload["pending_items"] = 0
    refreshed_scheduler_payload["percent_complete"] = 100.0
    refreshed_scheduler_payload["progress_bar"] = "#" * 20
    refreshed_scheduler_payload["status_kind"] = str(
        refreshed_scheduler_payload.get("status_kind", "experiment_progress")
        or "experiment_progress"
    )
    refreshed_scheduler_payload.pop("first_failed_item", None)
    refreshed_scheduler_payload["active_item_details"] = []
    for bucket_key in (
        "phase_progress",
        "by_scope",
        "by_train_docs",
        "by_model_family",
        "by_package",
        "by_worker_kind",
    ):
        refreshed_scheduler_payload[bucket_key] = _reset_terminal_bucket_payload(
            refreshed_scheduler_payload.get(bucket_key)
        )
    _write_json(scheduler_status_path, refreshed_scheduler_payload)
    return True


def _refresh_existing_tradeoff_outputs(
    args: argparse.Namespace,
    *,
    output_root: Path,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    phases = _phase_set(args.phases)
    version_manifest = _load_report_version_manifest(output_root)
    existing_pipeline_summary_path = output_root / "pipeline_summary.json"
    if existing_pipeline_summary_path.exists():
        pipeline_manifest = dict(_read_json(existing_pipeline_summary_path))
    else:
        pipeline_manifest = {}
    run_plan = build_run_plan(args, devices=[])
    pipeline_manifest.setdefault(
        "phase_task_counts",
        dict(run_plan.get("phase_task_counts") or {}),
    )
    pipeline_manifest["refresh_mode"] = "aggregate_only"
    pipeline_manifest["output_root"] = str(output_root)
    refreshed: Dict[str, str] = {}

    if "supervision_recovery" in phases:
        raw_root = _best_existing_raw_root(output_root, "supervision_recovery")
        if raw_root is None:
            raise SystemExit(
                f"could not find existing supervision_recovery raw outputs under {output_root}"
            )
        payloads = _load_supervision_recovery_refresh_payloads(raw_root)
        if not payloads:
            raise SystemExit(
                f"no supervision_recovery payload summaries found under {raw_root}"
            )
        summary = _aggregate_supervision_recovery_from_payloads(payloads)
        attempt_id = _new_attempt_id()
        config_fingerprint = _refresh_phase_config_fingerprint(
            manifest=version_manifest,
            phase="supervision_recovery",
        )
        attempt_root = _phase_attempt_root(output_root, "supervision_recovery", attempt_id)
        summary_path = attempt_root / "summary.json"
        _write_json(summary_path, summary)
        _register_phase_source(
            version_manifest,
            output_root=output_root,
            phase="supervision_recovery",
            source_key="supervision_recovery_summary",
            attempt_id=attempt_id,
            config_fingerprint=config_fingerprint,
            artifact_path=summary_path,
            alias_path=_canonical_alias_path(output_root, "supervision_recovery_summary"),
            extra_source={
                "expected_train_doc_counts": list(summary.get("train_doc_counts") or []),
                "expected_package_order": list(summary.get("package_order") or []),
                "expected_tree_family": str(
                    summary.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)
                    or SUPERVISION_RECOVERY_TREE_FAMILY
                ),
                "expected_recoverable_benchmark": str(
                    summary.get(
                        "recoverable_scope_key",
                        SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
                    )
                    or SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK
                ),
                "expected_structural_grid": str(
                    summary.get(
                        "structural_hardness_grid",
                        SUPERVISION_RECOVERY_STRUCTURAL_GRID,
                    )
                    or SUPERVISION_RECOVERY_STRUCTURAL_GRID
                ),
                "expected_structural_cell": str(
                    summary.get(
                        "structural_scope_key",
                        SUPERVISION_RECOVERY_STRUCTURAL_CELL,
                    )
                    or SUPERVISION_RECOVERY_STRUCTURAL_CELL
                ),
            },
        )
        refreshed["supervision_recovery_summary"] = str(
            output_root / "supervision_recovery" / "summary.json"
        )

    if "report" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _refresh_phase_config_fingerprint(
            manifest=version_manifest,
            phase="report",
        )
        report_root = _phase_attempt_root(output_root, "tradeoff_report", attempt_id)
        _refresh_selected_source_statuses(
            version_manifest,
            output_root=output_root,
            args=args,
        )
        _write_report_version_manifest(output_root, version_manifest)
        _invoke_report(
            TRADEOFF_REPORT_SCRIPT,
            [
                "--manifest",
                str(_report_version_manifest_path(output_root)),
                "--version-root",
                str(output_root),
                "--output-dir",
                str(report_root),
            ],
        )
        _register_report_outputs(
            version_manifest,
            output_root=output_root,
            attempt_id=attempt_id,
            config_fingerprint=config_fingerprint,
            attempt_root=report_root,
        )
        refreshed["tradeoff_report_summary"] = str(output_root / "tradeoff_report" / "summary.json")

    _refresh_selected_source_statuses(version_manifest, output_root=output_root, args=args)
    manifest_path = _write_report_version_manifest(output_root, version_manifest)
    pipeline_manifest["report_version_manifest"] = str(manifest_path)
    pipeline_manifest["selected_sources"] = dict(version_manifest.get("selected_sources") or {})
    report_summary_path = output_root / "tradeoff_report" / "summary.json"
    if report_summary_path.exists():
        report = build_markov_alignment_audit_report(
            family_grids_summary_json=report_summary_path,
            run_lean_build=False,
        )
        audit_outputs = write_markov_alignment_audit_report(
            report,
            output_json=output_root / "markov_alignment_audit.json",
            output_markdown=output_root / "markov_alignment_audit.md",
        )
        pipeline_manifest["alignment_audit_json"] = str(audit_outputs["output_json"])
        pipeline_manifest["alignment_audit_markdown"] = str(audit_outputs["output_markdown"])
        pipeline_manifest["contract_gate_status"] = (
            "fail" if int(report.summary.get("n_fail", 0)) > 0 else "pass"
        )
    supervision_summary_path = output_root / "supervision_recovery" / "summary.json"
    if supervision_summary_path.exists():
        supervision_summary = json.loads(
            supervision_summary_path.read_text(encoding="utf-8")
        )
        pipeline_manifest["quarantined_row_count"] = int(
            _safe_int(supervision_summary.get("quarantined_row_count"), 0)
        )
        pipeline_manifest["quarantined_sources"] = list(
            supervision_summary.get("quarantined_sources") or []
        )
    pipeline_summary_path = output_root / "pipeline_summary.json"
    _write_json(pipeline_summary_path, pipeline_manifest)
    merge_artifacts(output_root, _tradeoff_artifacts(output_root))
    _refresh_terminal_tradeoff_status_files(
        args=args,
        output_root=output_root,
        run_plan=run_plan,
    )
    return {
        "pipeline_summary_path": str(pipeline_summary_path),
        "report_version_manifest": str(manifest_path),
        "refreshed": refreshed,
    }


def _weight_profile_name(c1_ratio: float, c2_ratio: float, c3_ratio: float, local_law_weight: float) -> str:
    if abs(local_law_weight) <= 1e-12:
        return "root_only"
    rounded = (round(float(c1_ratio), 2), round(float(c2_ratio), 2), round(float(c3_ratio), 2))
    for name, triple in WEIGHT_PROFILE_SPECS.items():
        if rounded == tuple(round(float(v), 2) for v in triple):
            return name
    return f"c1={c1_ratio}_c2={c2_ratio}_c3={c3_ratio}"


def _normalize_target_leaf_counts(value: Any) -> List[int]:
    if isinstance(value, (list, tuple, set)):
        return sorted(
            {
                int(_safe_int(item))
                for item in value
                if int(_safe_int(item)) > 0
            }
        )
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = None
        if loaded is not None and loaded is not value:
            return _normalize_target_leaf_counts(loaded)
        values = [
            int(_safe_int(part))
            for part in text.replace(",", " ").split()
            if int(_safe_int(part)) > 0
        ]
        return sorted(set(values))
    return []


def _effective_supervision_recovery_run_epochs(
    run: Mapping[str, Any],
    *,
    task_config: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
) -> int:
    progress_payload = dict(progress or {})
    progress_total = int(_safe_int(progress_payload.get("epochs_total"), 0))
    if progress_total > 0:
        return int(progress_total)
    fit = dict(run.get("fit_diagnostics") or {})
    epochs_completed = int(_safe_int(fit.get("epochs_completed"), 0))
    if epochs_completed > 0:
        return int(epochs_completed)
    config = dict(task_config or {})
    schedule = str(
        run.get("tree_training_schedule")
        or run.get("training_schedule")
        or config.get("tree_training_schedule")
        or ""
    ).strip().lower()
    if schedule == "two_stage":
        stage1 = int(
            _safe_int(
                run.get("tree_stage1_epochs", config.get("tree_stage1_epochs")),
                0,
            )
        )
        stage2 = int(
            _safe_int(
                run.get("tree_stage2_epochs", config.get("tree_stage2_epochs")),
                0,
            )
        )
        if stage1 + stage2 > 0:
            return int(stage1 + stage2)
    n_epochs = int(
        _safe_int(
            run.get("n_epochs", config.get("n_epochs")),
            0,
        )
    )
    return int(max(1, n_epochs))


def _classify_supervision_recovery_fast_path(
    *,
    runtime_data_mode: str,
    runtime_bucket_mode: str,
    tree_batch_pack_mode: str,
    steady_state_h2d_bytes: float,
    steady_state_h2d_events: float,
    resident_store_hits: float,
    auto_queue_fused_batches: float,
    fixed_shape_dense_bucket_store_hits: float,
) -> str:
    normalized_data_mode = _ns(runtime_data_mode)
    normalized_bucket_mode = _ns(runtime_bucket_mode)
    normalized_pack_mode = _ns(tree_batch_pack_mode)
    correct_modes = (
        normalized_data_mode == "resident"
        and normalized_bucket_mode == "leaf_count_auto_queue"
        and normalized_pack_mode == "fixed_fused"
    )
    zero_h2d = (
        _safe_float(steady_state_h2d_bytes, 0.0) <= 0.0
        and _safe_float(steady_state_h2d_events, 0.0) <= 0.0
    )
    resident_hits_ok = _safe_float(resident_store_hits, 0.0) > 0.0
    fused_hits_ok = (
        _safe_float(auto_queue_fused_batches, 0.0) > 0.0
        or _safe_float(fixed_shape_dense_bucket_store_hits, 0.0) > 0.0
    )
    if correct_modes and zero_h2d and resident_hits_ok and fused_hits_ok:
        return "fast_path_confirmed"
    if correct_modes and (zero_h2d or resident_hits_ok or fused_hits_ok):
        return "fast_path_partial"
    return "fallback_or_unconfirmed"


def _supervision_recovery_runtime_row_from_payload(
    task_summary: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    progress: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    task_config = dict(task_summary.get("config") or {})
    run_config = dict(run.get("config") or {})
    runtime = dict(run.get("runtime_efficiency") or {})
    timing = dict(run.get("timing_breakdown") or {})
    autotuned = dict(run.get("autotuned_batch_budgets") or {})
    scope_key = str(
        run.get("cell_id")
        or task_config.get("pipeline_supervision_recovery_scope")
        or ""
    ).strip()
    recoverable_scope_key = str(
        task_config.get(
            "pipeline_supervision_recovery_recoverable_benchmark",
            task_config.get(
                "pipeline_supervision_recovery_scope",
                SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
            ),
        )
        or SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK
    ).strip()
    structural_grid = str(
        task_config.get(
            "pipeline_supervision_recovery_structural_grid",
            SUPERVISION_RECOVERY_STRUCTURAL_GRID,
        )
        or SUPERVISION_RECOVERY_STRUCTURAL_GRID
    ).strip()
    scope_label = str(
        task_config.get("pipeline_supervision_recovery_scope_label")
        or _supervision_recovery_scope_label(
            scope_key,
            recoverable_scope_key=recoverable_scope_key,
            structural_grid=structural_grid,
        )
    ).strip()
    package_name = str(
        task_config.get("pipeline_supervision_recovery_package", "") or ""
    ).strip()
    comparison_arm = _supervision_recovery_comparison_arm(task_config)
    baseline_family = _gs(run, "baseline_family")
    train_doc_count = int(
        _safe_int(run.get("train_doc_count", task_config.get("train_docs")), 0)
    )
    effective_train_epochs = _effective_supervision_recovery_run_epochs(
        run,
        task_config=task_config,
        progress=progress,
    )
    wall_clock_s = _safe_float(
        dict(progress or {}).get("wall_clock_s", task_summary.get("wall_clock_s")),
        float("nan"),
    )
    runtime_data_mode = str(
        runtime.get(
            "runtime_data_mode",
            run_config.get(
                "gpu_runtime_data_mode",
                task_config.get("gpu_runtime_data_mode", ""),
            ),
        )
        or ""
    )
    runtime_bucket_mode = str(
        runtime.get(
            "runtime_bucket_mode",
            run_config.get(
                "gpu_runtime_bucket_mode",
                task_config.get("gpu_runtime_bucket_mode", ""),
            ),
        )
        or ""
    )
    tree_batch_pack_mode = str(
        run.get(
            "tree_batch_pack_mode",
            run_config.get(
                "tree_batch_pack_mode",
                task_config.get("tree_batch_pack_mode", ""),
            ),
        )
        or ""
    )
    tree_document_loss_normalization_mode = str(
        run.get(
            "tree_document_loss_normalization_mode",
            runtime.get(
                "tree_document_loss_normalization_mode",
                run_config.get(
                    "tree_document_loss_normalization_mode",
                    task_config.get("tree_document_loss_normalization_mode", ""),
                ),
            ),
        )
        or ""
    )
    effective_tree_document_loss_normalization_mode = str(
        run.get(
            "effective_tree_document_loss_normalization_mode",
            runtime.get(
                "effective_tree_document_loss_normalization_mode",
                tree_document_loss_normalization_mode,
            ),
        )
        or tree_document_loss_normalization_mode
        or ""
    )
    geometry_info = _supervision_recovery_geometry_info(run, config=run_config or task_config)
    row: Dict[str, Any] = {
        **geometry_info,
        "scope_key": str(scope_key),
        "scope_label": str(scope_label),
        "package_name": str(package_name),
        "package_semantics": str(
            run.get(
                "package_semantics",
                run_config.get(
                    "package_semantics",
                    task_config.get(
                        "package_semantics",
                        _default_supervision_recovery_package_semantics(
                            str(package_name),
                            dict(
                                SUPERVISION_RECOVERY_PACKAGE_SPECS.get(
                                    str(package_name),
                                    {},
                                )
                            ),
                        ),
                    ),
                ),
            )
            or ""
        ),
        "baseline_family": str(baseline_family),
        "comparison_mode": str(
            run.get(
                "comparison_mode",
                run_config.get(
                    "comparison_mode",
                    task_config.get("comparison_mode", ""),
                ),
            )
            or ""
        ),
        "comparison_semantics": str(run.get("comparison_semantics", "") or ""),
        "comparison_semantics_label": str(
            run.get("comparison_semantics_label", "") or ""
        ),
        "run_intent_hash": str(run.get("run_intent_hash", "") or ""),
        "run_intent_validation_status": str(
            run.get("run_intent_validation_status", "") or ""
        ),
        "comparison_arm": str(comparison_arm),
        "seed": int(_safe_int(run.get("seed", task_config.get("seed")), 0)),
        "train_doc_count": int(train_doc_count),
        "depth_discount_gamma": _safe_float(
            run.get(
                "depth_discount_gamma",
                run_config.get(
                    "depth_discount_gamma",
                    task_config.get("depth_discount_gamma", 1.0),
                ),
            ),
            1.0,
        ),
        "effective_train_epochs": int(max(1, effective_train_epochs)),
        "wall_clock_s": float(wall_clock_s),
        "runtime_data_mode": str(runtime_data_mode),
        "runtime_bucket_mode": str(runtime_bucket_mode),
        "tree_batch_pack_mode": str(tree_batch_pack_mode),
        "tree_document_loss_normalization_mode": str(
            tree_document_loss_normalization_mode
        ),
        "effective_tree_document_loss_normalization_mode": str(
            effective_tree_document_loss_normalization_mode
        ),
        "tree_reference_mode": str(
            run.get(
                "tree_reference_mode",
                run_config.get(
                    "pipeline_tree_reference_mode",
                    task_config.get("pipeline_tree_reference_mode", ""),
                ),
            )
            or ""
        ),
        "tree_reference_label": str(
            run.get(
                "tree_reference_label",
                run_config.get(
                    "pipeline_tree_reference_label",
                    task_config.get("pipeline_tree_reference_label", ""),
                ),
            )
            or ""
        ),
        "tree_training_schedule": str(
            run.get(
                "tree_training_schedule",
                run_config.get(
                    "tree_training_schedule",
                    task_config.get("tree_training_schedule", ""),
                ),
            )
            or ""
        ),
        "tree_checkpoint_metric": str(
            run.get(
                "tree_checkpoint_metric",
                run_config.get(
                    "tree_checkpoint_metric",
                    task_config.get("tree_checkpoint_metric", ""),
                ),
            )
            or ""
        ),
        "tree_stage1_checkpoint_metric": str(
            run.get(
                "tree_stage1_checkpoint_metric",
                run_config.get(
                    "tree_stage1_checkpoint_metric",
                    task_config.get("tree_stage1_checkpoint_metric", ""),
                ),
            )
            or ""
        ),
        "tree_exact_collapse_mode": str(
            run.get(
                "tree_exact_collapse_mode",
                run_config.get(
                    "tree_exact_collapse_mode",
                    task_config.get("tree_exact_collapse_mode", ""),
                ),
            )
            or ""
        ),
        "tree_model_version": str(
            run.get(
                "tree_model_version",
                run_config.get(
                    "tree_model_version",
                    task_config.get("tree_model_version", ""),
                ),
            )
            or ""
        ),
        "tree_runtime_merge_kind": str(
            run.get(
                "tree_runtime_merge_kind",
                run_config.get(
                    "tree_runtime_merge_kind",
                    task_config.get("tree_runtime_merge_kind", ""),
                ),
            )
            or ""
        ),
        "tree_exact_projected_merge_is_runtime_merge": bool(
            run.get(
                "tree_exact_projected_merge_is_runtime_merge",
                run_config.get(
                    "tree_exact_projected_merge_is_runtime_merge",
                    task_config.get(
                        "tree_exact_projected_merge_is_runtime_merge",
                        False,
                    ),
                ),
            )
        ),
        "summary_spec_name": str(
            run.get(
                "summary_spec_name",
                run_config.get(
                    "summary_spec_name",
                    task_config.get("summary_spec_name", ""),
                ),
            )
            or ""
        ),
        "slot_count": int(
            _safe_int(
                run.get(
                    "slot_count",
                    run_config.get(
                        "slot_count",
                        task_config.get("slot_count", 0),
                    ),
                ),
                0,
            )
        ),
        "state_dim": int(
            _safe_int(
                run.get(
                    "state_dim",
                    run_config.get(
                        "state_dim",
                        task_config.get("state_dim", 0),
                    ),
                ),
                0,
            )
        ),
        "hidden_dim": int(
            _safe_int(
                run.get(
                    "hidden_dim",
                    run_config.get(
                        "hidden_dim",
                        task_config.get("hidden_dim", 0),
                    ),
                ),
                0,
            )
        ),
        "fixed_leaf_tokens": int(
            _safe_int(
                run.get(
                    "fixed_leaf_tokens",
                    run_config.get(
                        "fixed_leaf_tokens",
                        task_config.get("fixed_leaf_tokens", 0),
                    ),
                ),
                0,
            )
        ),
        "requested_fixed_leaf_tokens": int(
            _safe_int(
                run.get(
                    "requested_fixed_leaf_tokens",
                    run_config.get(
                        "requested_fixed_leaf_tokens",
                        run_config.get(
                            "fixed_leaf_tokens",
                            task_config.get(
                                "requested_fixed_leaf_tokens",
                                task_config.get("fixed_leaf_tokens", 0),
                            ),
                        ),
                    ),
                ),
                0,
            )
        ),
        "executed_fixed_leaf_tokens": int(
            _safe_int(
                run.get(
                    "executed_fixed_leaf_tokens",
                    run.get(
                        "fixed_leaf_tokens",
                        run_config.get(
                            "executed_fixed_leaf_tokens",
                            task_config.get("fixed_leaf_tokens", 0),
                        ),
                    ),
                ),
                0,
            )
        ),
        "executed_leaves_per_doc": int(
            _safe_int(
                run.get(
                    "executed_leaves_per_doc",
                    run.get("test_mean_leaves_per_doc", 0),
                ),
                0,
            )
        ),
        "executed_internal_nodes_per_doc": int(
            _safe_int(
                run.get("executed_internal_nodes_per_doc"),
                0,
            )
        ),
        "parity_mode": str(run.get("parity_mode", "") or ""),
        "is_exact_full_doc_parity_row": bool(
            run.get("is_exact_full_doc_parity_row", False)
        ),
        "tree_supervision_source": str(
            run.get(
                "tree_supervision_source",
                task_config.get("tree_supervision_source", ""),
            )
            or ""
        ),
        "local_estimand_mode": str(run.get("local_estimand_mode", "") or ""),
        "c2_pair_weighting_mode": str(
            run.get("c2_pair_weighting_mode", "") or ""
        ),
        "c2_same_pair_count": _safe_float(
            run.get("c2_same_pair_count"),
            float("nan"),
        ),
        "c2_different_pair_count": _safe_float(
            run.get("c2_different_pair_count"),
            float("nan"),
        ),
        "c2_pair_weight_ess": _safe_float(
            run.get("c2_pair_weight_ess"),
            float("nan"),
        ),
        "c2_pair_weight_max": _safe_float(
            run.get("c2_pair_weight_max"),
            float("nan"),
        ),
        **{
            metric_name: _safe_float(run.get(metric_name), float("nan"))
            for metric_name in SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTICS
        },
        **{
            f"{metric_name}_mean": _safe_float(
                run.get(f"{metric_name}_mean", run.get(metric_name)),
                float("nan"),
            )
            for metric_name in SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTICS
        },
        "steady_state_h2d_bytes": _safe_float(
            runtime.get("steady_state_h2d_bytes"),
            0.0,
        ),
        "steady_state_h2d_events": _safe_float(
            runtime.get("steady_state_h2d_events"),
            0.0,
        ),
        "resident_store_hits": _safe_float(
            runtime.get("resident_store_hits"),
            0.0,
        ),
        "resident_store_misses": _safe_float(
            runtime.get("resident_store_misses"),
            0.0,
        ),
        "auto_queue_family_count": _safe_float(
            runtime.get("auto_queue_family_count"),
            0.0,
        ),
        "auto_queue_target_leaf_counts": _normalize_target_leaf_counts(
            autotuned.get(
                "auto_queue_target_leaf_counts",
                runtime.get("auto_queue_target_leaf_counts", ()),
            )
        ),
        "structural_padding_waste_ratio": _safe_float(
            runtime.get("structural_padding_waste_ratio"),
            float("nan"),
        ),
        "auto_queue_fused_batches": _safe_float(
            runtime.get("auto_queue_fused_batches"),
            0.0,
        ),
        "auto_queue_generic_fallback_batches": _safe_float(
            runtime.get("auto_queue_generic_fallback_batches"),
            0.0,
        ),
        "fixed_shape_dense_bucket_store_hits": _safe_float(
            runtime.get("fixed_shape_dense_bucket_store_hits"),
            0.0,
        ),
        "document_supervision_docs_total": int(
            _safe_int(
                run.get(
                    "document_supervision_docs_total",
                    runtime.get("document_supervision_docs_total", 0),
                ),
                0,
            )
        ),
        "root_supervision_docs_total": int(
            _safe_int(
                run.get(
                    "root_supervision_docs_total",
                    runtime.get("root_supervision_docs_total", 0),
                ),
                0,
            )
        ),
        "doc_sequence_supervision_docs_total": int(
            _safe_int(
                run.get(
                    "doc_sequence_supervision_docs_total",
                    runtime.get("doc_sequence_supervision_docs_total", 0),
                ),
                0,
            )
        ),
        "document_supervision_coverage_rate": _safe_float(
            run.get(
                "document_supervision_coverage_rate",
                runtime.get("document_supervision_coverage_rate", float("nan")),
            ),
            float("nan"),
        ),
        "effective_full_doc_mass_per_doc": _safe_float(
            run.get("effective_full_doc_mass_per_doc"),
            float("nan"),
        ),
        "requested_root_mass_per_doc": _safe_float(
            run.get("requested_root_mass_per_doc"),
            float("nan"),
        ),
        "mass_target_per_doc": _safe_float(
            run.get(
                "mass_target_per_doc",
                run_config.get(
                    "mass_target_per_doc",
                    task_config.get("mass_target_per_doc"),
                ),
            ),
            float("nan"),
        ),
        "document_loss_mean_batch_scale": _safe_float(
            run.get(
                "document_loss_mean_batch_scale",
                runtime.get("document_loss_mean_batch_scale", float("nan")),
            ),
            float("nan"),
        ),
        "normalized_root_contribution_final": _safe_float(
            run.get(
                "normalized_root_contribution_final",
                runtime.get("normalized_root_contribution_final", float("nan")),
            ),
            float("nan"),
        ),
        "train_loop_s": _safe_float(
            timing.get("train_loop_s", run.get("elapsed_s_train_loop")),
            float("nan"),
        ),
        "stage1_train_loop_s": _safe_float(
            timing.get("stage1_train_loop_s"),
            float("nan"),
        ),
        "stage2_train_loop_s": _safe_float(
            timing.get("stage2_train_loop_s"),
            float("nan"),
        ),
        "exact_metric_eval_s": _safe_float(
            timing.get("exact_metric_eval_s", run.get("elapsed_s_exact_metric_eval")),
            float("nan"),
        ),
        "source_summary_json": str(
            task_summary.get("source_summary_json", "") or ""
        ),
    }
    row["fast_path_classification"] = _classify_supervision_recovery_fast_path(
        runtime_data_mode=str(row["runtime_data_mode"]),
        runtime_bucket_mode=str(row["runtime_bucket_mode"]),
        tree_batch_pack_mode=str(row["tree_batch_pack_mode"]),
        steady_state_h2d_bytes=_safe_float(row.get("steady_state_h2d_bytes"), 0.0),
        steady_state_h2d_events=_safe_float(row.get("steady_state_h2d_events"), 0.0),
        resident_store_hits=_safe_float(row.get("resident_store_hits"), 0.0),
        auto_queue_fused_batches=_safe_float(row.get("auto_queue_fused_batches"), 0.0),
        fixed_shape_dense_bucket_store_hits=_safe_float(
            row.get("fixed_shape_dense_bucket_store_hits"),
            0.0,
        ),
    )
    epochs = float(max(1, int(row["effective_train_epochs"])))
    train_docs_k = max(float(row["train_doc_count"]) / 1000.0, 1e-9)
    train_loop_s = _safe_float(row.get("train_loop_s"), float("nan"))
    wall_s = _safe_float(row.get("wall_clock_s"), float("nan"))
    row["train_loop_s_per_epoch"] = (
        float(train_loop_s / epochs) if math.isfinite(train_loop_s) else float("nan")
    )
    row["train_loop_s_per_epoch_per_1k_docs"] = (
        float(train_loop_s / epochs / train_docs_k)
        if math.isfinite(train_loop_s)
        else float("nan")
    )
    row["wall_clock_s_per_epoch"] = (
        float(wall_s / epochs) if math.isfinite(wall_s) else float("nan")
    )
    return row


def _summarize_supervision_recovery_runtime_diagnosis(
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    tree_family: str = SUPERVISION_RECOVERY_TREE_FAMILY,
) -> Dict[str, Any]:
    tree_rows = [
        dict(row)
        for row in runtime_rows
        if _gs(row, "baseline_family") == str(tree_family)
        and _supervision_recovery_comparison_arm(row)
        == SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM
    ]
    fno_rows = [
        dict(row)
        for row in runtime_rows
        if _gs(row, "baseline_family")
        in set(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)
    ]
    if not tree_rows:
        return {
            "status": "missing",
            "reason": "no completed tree-neural runtime rows were available",
            "tree_family": str(tree_family),
            "tree_rows": [],
            "grouped_rows": [],
            "fno_context_rows": fno_rows,
        }

    def _median_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        values = [
            _safe_float(row.get(field), float("nan"))
            for row in rows
            if math.isfinite(_safe_float(row.get(field), float("nan")))
        ]
        return float(median(values)) if values else float("nan")

    confirmed_rows = [
        row for row in tree_rows if str(row.get("fast_path_classification")) == "fast_path_confirmed"
    ]
    partial_or_fallback_rows = [
        row for row in tree_rows if str(row.get("fast_path_classification")) != "fast_path_confirmed"
    ]
    zero_h2d_rows = [
        row
        for row in tree_rows
        if _safe_float(row.get("steady_state_h2d_bytes"), 0.0) <= 0.0
        and _safe_float(row.get("steady_state_h2d_events"), 0.0) <= 0.0
    ]
    grouped: Dict[tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in tree_rows:
        grouped[
            (
                str(row.get("scope_key", "")),
                int(_safe_int(row.get("train_doc_count"), 0)),
                str(row.get("package_name", "")),
            )
        ].append(dict(row))

    grouped_rows: List[Dict[str, Any]] = []
    for (scope_key, train_doc_count, package_name), items in sorted(
        grouped.items(),
        key=lambda item: (
            str(item[0][0]),
            int(item[0][1]),
            SUPERVISION_RECOVERY_PACKAGE_ORDER.index(str(item[0][2]))
            if str(item[0][2]) in SUPERVISION_RECOVERY_PACKAGE_ORDER
            else len(SUPERVISION_RECOVERY_PACKAGE_ORDER),
            str(item[0][2]),
        ),
    ):
        class_counter = Counter(str(item.get("fast_path_classification", "")) for item in items)
        grouped_rows.append(
            {
                "scope_key": str(scope_key),
                "scope_label": str(items[0].get("scope_label", scope_key)),
                "train_doc_count": int(train_doc_count),
                "package_name": str(package_name),
                "n_seeds_completed": int(len(items)),
                "seed_values": sorted(
                    {
                        int(_safe_int(item.get("seed"), 0))
                        for item in items
                        if int(_safe_int(item.get("seed"), 0)) >= 0
                    }
                ),
                "fast_path_classification": str(
                    class_counter.most_common(1)[0][0] if class_counter else ""
                ),
                "fast_path_confirmed_rate": float(
                    sum(
                        1
                        for item in items
                        if str(item.get("fast_path_classification")) == "fast_path_confirmed"
                    )
                    / len(items)
                ),
                "zero_h2d_rate": float(
                    sum(
                        1
                        for item in items
                        if _safe_float(item.get("steady_state_h2d_bytes"), 0.0) <= 0.0
                        and _safe_float(item.get("steady_state_h2d_events"), 0.0) <= 0.0
                    )
                    / len(items)
                ),
                "runtime_data_mode": str(items[0].get("runtime_data_mode", "")),
                "runtime_bucket_mode": str(items[0].get("runtime_bucket_mode", "")),
                "tree_batch_pack_mode": str(items[0].get("tree_batch_pack_mode", "")),
                "tree_document_loss_normalization_mode": str(
                    items[0].get("tree_document_loss_normalization_mode", "")
                ),
                "effective_tree_document_loss_normalization_mode": str(
                    items[0].get("effective_tree_document_loss_normalization_mode", "")
                ),
                "tree_reference_mode": str(items[0].get("tree_reference_mode", "")),
                "tree_reference_label": str(items[0].get("tree_reference_label", "")),
                "tree_training_schedule": str(items[0].get("tree_training_schedule", "")),
                "tree_checkpoint_metric": str(items[0].get("tree_checkpoint_metric", "")),
                "tree_stage1_checkpoint_metric": str(
                    items[0].get("tree_stage1_checkpoint_metric", "")
                ),
                "summary_spec_name": str(items[0].get("summary_spec_name", "")),
                "slot_count": int(_safe_int(items[0].get("slot_count"), 0)),
                "state_dim": int(_safe_int(items[0].get("state_dim"), 0)),
                "hidden_dim": int(_safe_int(items[0].get("hidden_dim"), 0)),
                "fixed_leaf_tokens": int(_safe_int(items[0].get("fixed_leaf_tokens"), 0)),
                "wall_clock_s_per_epoch_median": _median_field(items, "wall_clock_s_per_epoch"),
                "train_loop_s_per_epoch_median": _median_field(items, "train_loop_s_per_epoch"),
                "train_loop_s_per_epoch_per_1k_docs_median": _median_field(
                    items,
                    "train_loop_s_per_epoch_per_1k_docs",
                ),
                "resident_store_hits_median": _median_field(items, "resident_store_hits"),
                "fixed_shape_dense_bucket_store_hits_median": _median_field(
                    items,
                    "fixed_shape_dense_bucket_store_hits",
                ),
                "auto_queue_fused_batches_median": _median_field(
                    items,
                    "auto_queue_fused_batches",
                ),
                "document_supervision_coverage_rate_median": _median_field(
                    items,
                    "document_supervision_coverage_rate",
                ),
                "document_loss_mean_batch_scale_median": _median_field(
                    items,
                    "document_loss_mean_batch_scale",
                ),
                "normalized_root_contribution_final_median": _median_field(
                    items,
                    "normalized_root_contribution_final",
                ),
            }
        )

    completion_rate = float(len(confirmed_rows) / len(tree_rows)) if tree_rows else 0.0
    zero_h2d_rate = float(len(zero_h2d_rows) / len(tree_rows)) if tree_rows else 0.0
    likely_fast = bool(
        completion_rate >= 0.9
        and zero_h2d_rate >= 0.9
        and len(confirmed_rows) > 0
    )
    return {
        "status": "ready",
        "reason": "",
        "tree_family": str(tree_family),
        "tree_rows": tree_rows,
        "grouped_rows": grouped_rows,
        "fno_context_rows": fno_rows,
        "tree_fast_path_completion_rate": float(completion_rate),
        "tree_zero_h2d_rate": float(zero_h2d_rate),
        "tree_median_train_loop_s_per_epoch": _median_field(
            tree_rows,
            "train_loop_s_per_epoch",
        ),
        "tree_median_train_loop_s_per_epoch_per_1k_docs": _median_field(
            tree_rows,
            "train_loop_s_per_epoch_per_1k_docs",
        ),
        "tree_median_wall_clock_s_per_epoch": _median_field(
            tree_rows,
            "wall_clock_s_per_epoch",
        ),
        "tree_median_resident_store_hits": _median_field(
            tree_rows,
            "resident_store_hits",
        ),
        "tree_median_dense_bucket_hits": _median_field(
            tree_rows,
            "fixed_shape_dense_bucket_store_hits",
        ),
        "tree_median_auto_queue_fused_batches": _median_field(
            tree_rows,
            "auto_queue_fused_batches",
        ),
        "tree_median_document_loss_batch_scale": _median_field(
            tree_rows,
            "document_loss_mean_batch_scale",
        ),
        "tree_fast_path_confirmed_runs": int(len(confirmed_rows)),
        "tree_partial_or_fallback_runs": int(len(partial_or_fallback_rows)),
        "current_evidence_status": (
            "fast_path_engaged_and_likely_materially_helping"
            if likely_fast
            else "strict_causal_ab_proof_pending"
        ),
    }


def _aggregate_weight_ablation_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    records = []
    for payload in payloads:
        config = dict(payload.get("config", {}) or {})
        objective = dict(payload.get("objective", {}) or {})
        learned = dict((payload.get("metrics", {}) or {}).get("learned", {}) or {})
        root_error = _safe_float(learned.get("root_mae"))
        if not math.isfinite(root_error):
            continue
        profile = _weight_profile_name(
            _safe_float(objective.get("local_law_c1_share", config.get("c1_relative_weight", 0.0)), 0.0),
            _safe_float(objective.get("local_law_c2_share", config.get("c2_relative_weight", 0.0)), 0.0),
            _safe_float(objective.get("local_law_c3_share", config.get("c3_relative_weight", 0.0)), 0.0),
            _safe_float(objective.get("local_law_weight", config.get("local_law_weight", 0.0)), 0.0),
        )
        scenario = (
            f"train_{_safe_int(config.get('train_docs'))}"
            f"_data_{_safe_int(config.get('data_seed'))}"
            f"_state_{_safe_int(config.get('state_dim'))}"
            f"_hidden_{_safe_int(config.get('hidden_dim'))}"
        )
        records.append(
            {
                "profile": profile,
                "scenario": scenario,
                "root_error": root_error,
                "path": str(config.get("artifact_dir", "")),
            }
        )

    by_profile: Dict[str, List[float]] = {}
    by_scenario: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        by_profile.setdefault(str(record["profile"]), []).append(float(record["root_error"]))
        by_scenario.setdefault(str(record["scenario"]), {}).setdefault(str(record["profile"]), []).append(float(record["root_error"]))

    profile_summaries = []
    for profile in WEIGHT_PROFILE_ORDER:
        vals = by_profile.get(profile, [])
        if not vals:
            continue
        ordered = sorted(vals)
        profile_summaries.append(
            {
                "profile": profile,
                "n": len(vals),
                "n_with_root": len(vals),
                "mean_root_error": float(sum(vals) / len(vals)),
                "median_root_error": float(ordered[len(ordered) // 2]),
            }
        )

    matched_pairs = []
    for scenario, profiles in sorted(by_scenario.items()):
        baseline = profiles.get("root_only", [])
        if not baseline:
            continue
        baseline_mae = sum(baseline) / len(baseline)
        if baseline_mae <= 0.0:
            continue
        for profile in WEIGHT_PROFILE_ORDER:
            if profile == "root_only":
                continue
            vals = profiles.get(profile, [])
            if not vals:
                continue
            treatment_mae = sum(vals) / len(vals)
            ratio = treatment_mae / baseline_mae
            matched_pairs.append(
                {
                    "scenario": scenario,
                    "profile": profile,
                    "baseline_mae": baseline_mae,
                    "treatment_mae": treatment_mae,
                    "ratio": ratio,
                    "gain_pct": (1.0 - ratio) * 100.0,
                }
            )

    matched_summaries = []
    for profile in WEIGHT_PROFILE_ORDER:
        if profile == "root_only":
            continue
        rows = [row for row in matched_pairs if row["profile"] == profile]
        if not rows:
            continue
        ratios = [float(row["ratio"]) for row in rows]
        gains = [float(row["gain_pct"]) for row in rows]
        matched_summaries.append(
            {
                "profile": profile,
                "n_matched": len(rows),
                "mean_ratio": float(sum(ratios) / len(ratios)),
                "median_ratio": float(sorted(ratios)[len(ratios) // 2]),
                "min_ratio": float(min(ratios)),
                "max_ratio": float(max(ratios)),
                "mean_gain_pct": float(sum(gains) / len(gains)),
                "primary_pass_rate": float(sum(1 for ratio in ratios if ratio < 1.0) / len(ratios)),
            }
        )

    return {
        "profile_summaries": profile_summaries,
        "matched_summaries": matched_summaries,
        "matched_pairs": matched_pairs,
        "n_total": len(records),
        "n_scenarios": len(by_scenario),
    }


def _aggregate_law_packages_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for payload in payloads:
        config = dict(payload.get("config", {}) or {})
        learned = dict((payload.get("metrics", {}) or {}).get("learned", {}) or {})
        fno_test = dict((payload.get("metrics", {}) or {}).get("fno", {}) or {})
        fno_train = dict((payload.get("metrics", {}) or {}).get("fno_train", {}) or {})
        fno_training = dict((payload.get("metrics", {}) or {}).get("fno_training", {}) or {})
        objective = dict(payload.get("objective", {}) or {})
        key = str(config.get("pipeline_law_package_name", config.get("law_package", "")) or "")
        if not key:
            continue
        tree_root_mae = float(learned.get("root_mae", float("nan")))
        doc_fno_root_mae = float(fno_test.get("root_mae", float("nan")))
        tree_vs_doc_gap = (
            float(tree_root_mae - doc_fno_root_mae)
            if math.isfinite(tree_root_mae) and math.isfinite(doc_fno_root_mae)
            else float("nan")
        )
        out[key] = {
            "test_root_mae": tree_root_mae,
            "test_leaf_mae": float(learned.get("leaf_mae", float("nan"))),
            "test_merge_mae": float(learned.get("merge_mae", float("nan"))),
            "test_c2_mae": float(learned.get("c2_idempotence_mae", float("nan"))),
            "train_root_mae": float(learned.get("train_root_mae", float("nan"))),
            "train_leaf_mae": float(learned.get("train_leaf_mae", float("nan"))),
            "train_merge_mae": float(learned.get("train_merge_mae", float("nan"))),
            "doc_fno_test_root_mae": doc_fno_root_mae,
            "doc_fno_train_root_mae": float(fno_train.get("root_mae", float("nan"))),
            "doc_fno_best_epoch": int(fno_training.get("best_epoch", -1) or -1),
            "tree_vs_doc_fno_root_mae_gap": tree_vs_doc_gap,
            "epochs": int(learned.get("epochs_completed", config.get("n_epochs", 0)) or 0),
            "best_epoch": int(learned.get("training_selection_best_epoch", -1) or -1),
            "c1_weight": float(objective.get("local_law_c1_weight", float("nan"))),
            "c2_weight": float(objective.get("local_law_c2_weight", float("nan"))),
            "c3_weight": float(objective.get("local_law_c3_weight", float("nan"))),
            "wall_seconds": float(payload.get("wall_clock_s", float("nan")) or float("nan")),
        }
    return out


def _aggregate_full_doc_upper_bound_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[int, Dict[str, List[Mapping[str, Any]]]] = {}
    benchmark = ""
    template_benchmark = ""
    for payload in payloads:
        benchmark = benchmark or str(payload.get("benchmark", ""))
        template_benchmark = template_benchmark or str(payload.get("template_benchmark", ""))
        for row in list(payload.get("rows") or []):
            train_docs = int(_safe_int(row.get("train_doc_count")))
            family = str(row.get("baseline_family", "") or "")
            if train_docs <= 0 or not family:
                continue
            grouped.setdefault(train_docs, {}).setdefault(family, []).append(dict(row))
    rows: List[Dict[str, Any]] = []
    for train_docs, family_rows in sorted(grouped.items()):
        family_summaries: List[Dict[str, Any]] = []
        for family, items in sorted(family_rows.items()):
            test_values = [_safe_float(item.get("test_root_mae")) for item in items]
            train_values = [_safe_float(item.get("train_root_mae")) for item in items]
            val_values = [_safe_float(item.get("val_root_mae")) for item in items]
            wall_values = [_safe_float(item.get("family_wall_clock_s")) for item in items]
            best_epoch_values = [_safe_float(item.get("best_epoch")) for item in items]
            family_summaries.append(
                {
                    "baseline_family": family,
                    "n_runs": len(items),
                    "test_root_mae_mean": float(sum(test_values) / len(test_values)),
                    "train_root_mae_mean": float(sum(train_values) / len(train_values)),
                    "val_root_mae_mean": float(sum(val_values) / len(val_values)),
                    "family_wall_clock_s_mean": float(sum(wall_values) / len(wall_values)),
                    "best_epoch_mean": float(sum(best_epoch_values) / len(best_epoch_values)),
                }
            )
        family_summaries.sort(key=lambda item: _safe_float(item.get("test_root_mae_mean"), float("inf")))
        best = dict(family_summaries[0]) if family_summaries else {}
        second = dict(family_summaries[1]) if len(family_summaries) > 1 else {}
        rows.append(
            {
                "train_docs": int(train_docs),
                "family_summaries": family_summaries,
                "best_full_doc_fno_family": str(best.get("baseline_family", "")),
                "best_full_doc_fno_test_root_mae": float(best.get("test_root_mae_mean", float("nan"))),
                "second_best_full_doc_fno_family": str(second.get("baseline_family", "")),
                "second_best_full_doc_fno_test_root_mae": float(second.get("test_root_mae_mean", float("nan"))),
            }
        )
    return {
        "benchmark": benchmark,
        "template_benchmark": template_benchmark,
        "rows": rows,
    }


def _normalize_efficiency_anchor_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    control_exactness: Dict[str, Any] = {}
    benchmark_name = ""
    hardness_grid = ""
    for payload in payloads:
        benchmark_name = benchmark_name or str(payload.get("benchmark", "") or "")
        hardness_grid = hardness_grid or str(payload.get("hardness_grid", "") or "")
        diag_summary = dict(payload.get("grid_diagnostic_summary") or {})
        exactness = dict(diag_summary.get("control_exactness") or {})
        if exactness:
            for key, value in exactness.items():
                control_exactness[str(key)] = value
        for row in list(payload.get("aggregate_rows") or []):
            if not isinstance(row, Mapping):
                continue
            normalized = dict(row)
            normalized.setdefault("cell_id", str(row.get("cell_id") or payload.get("benchmark") or ""))
            normalized.setdefault("benchmark", str(payload.get("benchmark", "") or ""))
            normalized["hardness_grid"] = str(payload.get("hardness_grid", "") or "")
            normalized["train_doc_count"] = int(_safe_int(row.get("train_doc_count")))
            rows.append(normalized)
    rows.sort(
        key=lambda row: (
            str(row.get("cell_id", "")),
            str(row.get("baseline_family", "")),
            int(_safe_int(row.get("train_doc_count"))),
        )
    )
    return {
        "benchmark": benchmark_name,
        "hardness_grid": hardness_grid,
        "rows": rows,
        "baseline_families": sorted(
            {str(row.get("baseline_family", "")) for row in rows if str(row.get("baseline_family", "")).strip()}
        ),
        "train_doc_counts": sorted(
            {int(_safe_int(row.get("train_doc_count"))) for row in rows if int(_safe_int(row.get("train_doc_count"))) > 0}
        ),
        "cell_ids": sorted(
            {str(row.get("cell_id", "")) for row in rows if str(row.get("cell_id", "")).strip()}
        ),
        "control_exactness": control_exactness,
    }


def _best_budget_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        cell_id = str(row.get("cell_id", "") or row.get("benchmark", "") or "")
        key = (
            cell_id,
            int(_safe_int(row.get("train_doc_count"))),
            float(_safe_float(row.get("budget_total_calls_per_doc"))),
            float(_safe_float(row.get("full_doc_budget_share"))),
            str(row.get("doc_consumption_mode", "")),
            str(row.get("local_split_mode", "")),
        )
        incumbent = grouped.get(key)
        value = _safe_float(row.get("test_root_mae_mean"), float("inf"))
        incumbent_value = _safe_float((incumbent or {}).get("test_root_mae_mean"), float("inf"))
        if incumbent is None or value < incumbent_value:
            grouped[key] = dict(row)
    return sorted(
        grouped.values(),
        key=lambda row: (
            str(row.get("cell_id", "")),
            int(_safe_int(row.get("train_doc_count"))),
            float(_safe_float(row.get("budget_total_calls_per_doc"))),
            float(_safe_float(row.get("full_doc_budget_share"))),
            str(row.get("doc_consumption_mode", "")),
            str(row.get("local_split_mode", "")),
        ),
    )


def _normalize_efficiency_budget_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    tree_rows: List[Dict[str, Any]] = []
    reference_rows: List[Dict[str, Any]] = []
    benchmark_name = ""
    hardness_grid = ""
    for payload in payloads:
        benchmark_name = benchmark_name or str(payload.get("benchmark", "") or "")
        hardness_grid = hardness_grid or str(payload.get("hardness_grid", "") or "")
        inferred_train_doc_count = int(_safe_int(payload.get("train_doc_count")))
        for source_key, target in (("tree_rows", tree_rows), ("reference_rows", reference_rows)):
            for row in list(payload.get(source_key) or []):
                if not isinstance(row, Mapping):
                    continue
                normalized = dict(row)
                normalized.setdefault(
                    "cell_id",
                    str(row.get("cell_id") or payload.get("benchmark") or ""),
                )
                normalized.setdefault("benchmark", str(payload.get("benchmark", "") or ""))
                normalized["hardness_grid"] = str(payload.get("hardness_grid", "") or "")
                if int(_safe_int(normalized.get("train_doc_count"))) <= 0 and inferred_train_doc_count > 0:
                    normalized["train_doc_count"] = inferred_train_doc_count
                target.append(normalized)
    tree_rows.sort(
        key=lambda row: (
            str(row.get("cell_id", "")),
            int(_safe_int(row.get("train_doc_count"))),
            float(_safe_float(row.get("budget_total_calls_per_doc"))),
            float(_safe_float(row.get("full_doc_budget_share"))),
            str(row.get("doc_consumption_mode", "")),
            str(row.get("local_split_mode", "")),
            str(row.get("baseline_family", "")),
        )
    )
    reference_rows.sort(
        key=lambda row: (
            str(row.get("cell_id", "")),
            int(_safe_int(row.get("train_doc_count"))),
            float(_safe_float(row.get("budget_total_calls_per_doc"))),
            float(_safe_float(row.get("full_doc_budget_share"))),
            str(row.get("baseline_family", "")),
        )
    )
    return {
        "benchmark": benchmark_name,
        "hardness_grid": hardness_grid,
        "tree_rows": tree_rows,
        "reference_rows": reference_rows,
        "best_tree_by_budget": _best_budget_rows(tree_rows),
        "train_doc_counts": sorted(
            {
                int(_safe_int(row.get("train_doc_count")))
                for row in tree_rows + reference_rows
                if int(_safe_int(row.get("train_doc_count"))) > 0
            }
        ),
        "cell_ids": sorted(
            {
                str(row.get("cell_id", ""))
                for row in tree_rows + reference_rows
                if str(row.get("cell_id", "")).strip()
            }
        ),
    }


def _aggregate_efficiency_suite(
    *,
    recoverable_anchor_payloads: Sequence[Mapping[str, Any]],
    structural_anchor_payloads: Sequence[Mapping[str, Any]],
    recoverable_budget_payloads: Sequence[Mapping[str, Any]],
    structural_budget_payloads: Sequence[Mapping[str, Any]],
    tree_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "tree_reference": dict(tree_reference),
        "recoverable_dense_anchor": _normalize_efficiency_anchor_payloads(recoverable_anchor_payloads),
        "recoverable_budget": _normalize_efficiency_budget_payloads(recoverable_budget_payloads),
        "structural_dense_anchor": _normalize_efficiency_anchor_payloads(structural_anchor_payloads),
        "structural_budget": _normalize_efficiency_budget_payloads(structural_budget_payloads),
    }


def _ceil_div(numer: int, denom: int) -> int:
    numer_i = int(numer)
    denom_i = max(1, int(denom))
    return int((numer_i + denom_i - 1) // denom_i)


def _parse_large_batch_task_name(name: str) -> Dict[str, Any]:
    parts = [item for item in str(name).split("__") if item]
    if not parts:
        return {}
    out: Dict[str, Any] = {"study_block": str(parts[0])}
    for token in parts[1:]:
        if token.startswith("bs"):
            out["batch_size"] = int(token.replace("bs", ""))
        elif token.startswith("ep"):
            out["epochs"] = int(token.replace("ep", ""))
        elif token.startswith("lr"):
            out["lr"] = float(token.replace("lr", ""))
    return out


def _aggregate_large_batch_diagnosis(
    task_infos: Sequence[Mapping[str, Any]],
    *,
    target_total_steps: int = 200,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for info in task_infos:
        meta = _parse_large_batch_task_name(str(info.get("name", "")))
        payload = _read_json(Path(str(info["output_path"])))
        summary = dict((payload.get("runs", {}) or {}).get("no_autotune", {}) or {})
        timing = dict(summary.get("timing_breakdown", {}) or {})
        metrics = dict(summary.get("batching_metrics", {}) or {})
        config = dict(payload.get("config", {}) or {})
        batch_size = int(meta.get("batch_size", config.get("batch_size", 0)) or 0)
        epochs = int(meta.get("epochs", summary.get("epochs_completed", 0)) or 0)
        train_docs = int(config.get("train_docs", 0) or 0)
        steps_per_epoch = _ceil_div(train_docs, batch_size) if batch_size > 0 else 0
        total_steps = int(steps_per_epoch * max(1, epochs))
        wall_s = max(_safe_float(summary.get("wall_clock_s"), 0.0), 1e-9)
        train_loop_s = max(_safe_float(timing.get("train_loop_s"), 0.0), 1e-9)
        rows.append(
            {
                "run": str(info.get("name", "")),
                "study_block": str(meta.get("study_block", "")),
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": float(meta.get("lr", config.get("lr", float("nan"))) or float("nan")),
                "train_docs": train_docs,
                "steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total_steps,
                "best_val_mae": float(summary.get("best_val_mae", float("nan"))),
                "best_epoch": int(summary.get("best_epoch", 0) or 0),
                "docs_per_s_wall": float(train_docs * max(1, epochs) / wall_s) if train_docs > 0 else 0.0,
                "docs_per_s_train_loop": float(train_docs * max(1, epochs) / train_loop_s) if train_docs > 0 else 0.0,
                "gpu_reserved_mem_peak_gb": float(metrics.get("gpu_reserved_mem_peak_gb", float("nan"))),
                "wall_clock_s": float(summary.get("wall_clock_s", 0.0)),
            }
        )
    rows.sort(key=lambda row: (str(row["study_block"]), int(row["batch_size"]), float(row["lr"])))
    by_key = {
        (str(row["study_block"]), int(row["batch_size"]), round(float(row["lr"]), 6)): row
        for row in rows
    }
    baseline = by_key.get(("constant_steps", 512, round(1e-3, 6)))
    fixed_1024 = by_key.get(("fixed_epoch", 1024, round(1e-3, 6)))
    const_1024 = by_key.get(("constant_steps", 1024, round(1e-3, 6)))
    retuned_rows = [row for row in rows if str(row["study_block"]) == "retune_1024"]
    best_retuned_1024 = (
        min(retuned_rows, key=lambda row: _safe_float(row.get("best_val_mae"), float("inf")))
        if retuned_rows
        else None
    )

    classification = "unresolved"
    if baseline and const_1024:
        baseline_mae = _safe_float(baseline.get("best_val_mae"), float("inf"))
        const_mae = _safe_float(const_1024.get("best_val_mae"), float("inf"))
        retuned_mae = _safe_float((best_retuned_1024 or {}).get("best_val_mae"), float("inf"))
        threshold = 1.25 * baseline_mae if math.isfinite(baseline_mae) else float("inf")
        if const_mae <= threshold:
            classification = "update_budget_limited"
        elif retuned_mae <= threshold:
            classification = "optimizer_scale_limited"

    recommendation_batch_size = 512
    recommendation_reason = "retuned_1024 does not clear the quality/speed gate"
    if baseline and best_retuned_1024:
        baseline_mae = _safe_float(baseline.get("best_val_mae"), float("inf"))
        candidate_mae = _safe_float(best_retuned_1024.get("best_val_mae"), float("inf"))
        baseline_speed = _safe_float(baseline.get("docs_per_s_wall"), float("-inf"))
        candidate_speed = _safe_float(best_retuned_1024.get("docs_per_s_wall"), float("-inf"))
        if candidate_mae <= 1.25 * baseline_mae and candidate_speed > baseline_speed:
            recommendation_batch_size = 1024
            recommendation_reason = "retuned_1024 is within 25% of the bs512 constant-step MAE and faster on wall throughput"
    return {
        "rows": rows,
        "train_docs": int(rows[0]["train_docs"]) if rows else 0,
        "target_total_steps": int(target_total_steps),
        "fixed_epoch_reference": fixed_1024,
        "constant_steps_reference": const_1024,
        "best_retuned_1024": best_retuned_1024,
        "classification": classification,
        "recommendation": {
            "recommended_max_batch_size": int(recommendation_batch_size),
            "reason": str(recommendation_reason),
        },
    }


def _aggregate_supervision_sweep_from_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    grouped: Dict[tuple[int, str, str], List[Dict[str, Any]]] = {}
    for payload in payloads:
        config = dict(payload.get("config", {}) or {})
        metrics = dict(payload.get("metrics", {}) or {})
        learned_test = dict(metrics.get("learned_test", metrics.get("learned", {})) or {})
        learned_val = dict(metrics.get("learned_val", {}) or {})
        train_docs = int(_safe_int(config.get("train_docs"), 0))
        leaf_profile = _gs(config, "pipeline_supervision_leaf_profile")
        internal_profile = _gs(config, "pipeline_supervision_internal_profile")
        if train_docs <= 0 or not leaf_profile or not internal_profile:
            continue
        grouped.setdefault((train_docs, leaf_profile, internal_profile), []).append(
            {
                "train_doc_count": train_docs,
                "leaf_profile": leaf_profile,
                "internal_profile": internal_profile,
                "leaf_supervision_kind": str(config.get("leaf_supervision_kind", "")),
                "leaf_label_rate": float(config.get("leaf_label_rate", 0.0) or 0.0),
                "internal_supervision_kind": str(config.get("internal_supervision_kind", "")),
                "internal_label_rate": float(config.get("internal_label_rate", 0.0) or 0.0),
                "data_seed": int(_safe_int(config.get("data_seed"), 0)),
                "test_root_mae": _safe_float(
                    learned_test.get("test_root_mae", learned_test.get("root_mae")),
                    float("nan"),
                ),
                "val_root_mae": _safe_float(
                    learned_test.get("val_root_mae", learned_val.get("root_mae")),
                    float("nan"),
                ),
                "test_leaf_mae": _safe_float(
                    learned_test.get("test_leaf_mae", learned_test.get("leaf_mae")),
                    float("nan"),
                ),
                "test_merge_mae": _safe_float(
                    learned_test.get("test_merge_mae", learned_test.get("merge_mae")),
                    float("nan"),
                ),
                "wall_clock_s": _safe_float(payload.get("wall_clock_s"), float("nan")),
            }
        )

    def _mean(items: Sequence[float]) -> float:
        vals = [float(item) for item in items if math.isfinite(float(item))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _profile_rank(row: Mapping[str, Any]) -> tuple[int, int]:
        return (
            SUPERVISION_LEAF_PROFILE_ORDER.index(str(row.get("leaf_profile")))
            if str(row.get("leaf_profile")) in SUPERVISION_LEAF_PROFILE_ORDER
            else len(SUPERVISION_LEAF_PROFILE_ORDER),
            SUPERVISION_INTERNAL_PROFILE_ORDER.index(str(row.get("internal_profile")))
            if str(row.get("internal_profile")) in SUPERVISION_INTERNAL_PROFILE_ORDER
            else len(SUPERVISION_INTERNAL_PROFILE_ORDER),
        )

    rows: List[Dict[str, Any]] = []
    by_train_docs: Dict[str, Dict[str, Any]] = {}
    for (train_docs, leaf_profile, internal_profile), items in sorted(grouped.items()):
        row = {
            "train_doc_count": int(train_docs),
            "leaf_profile": str(leaf_profile),
            "internal_profile": str(internal_profile),
            "leaf_supervision_kind": str(items[0]["leaf_supervision_kind"]),
            "leaf_label_rate": float(items[0]["leaf_label_rate"]),
            "internal_supervision_kind": str(items[0]["internal_supervision_kind"]),
            "internal_label_rate": float(items[0]["internal_label_rate"]),
            "n_runs": len(items),
            "mean_test_root_mae": _mean([float(item["test_root_mae"]) for item in items]),
            "mean_val_root_mae": _mean([float(item["val_root_mae"]) for item in items]),
            "mean_test_leaf_mae": _mean([float(item["test_leaf_mae"]) for item in items]),
            "mean_test_merge_mae": _mean([float(item["test_merge_mae"]) for item in items]),
            "mean_wall_clock_s": _mean([float(item["wall_clock_s"]) for item in items]),
        }
        rows.append(row)
        by_train_docs.setdefault(str(train_docs), {"rows": []})["rows"].append(row)

    for train_docs, payload in by_train_docs.items():
        subrows = list(payload.get("rows") or [])
        subrows.sort(
            key=lambda row: (
                _safe_float(row.get("mean_test_root_mae"), float("inf")),
                *_profile_rank(row),
            )
        )
        best_root = dict(subrows[0]) if subrows else {}
        payload["rows"] = subrows
        payload["best_root_row"] = best_root
        payload["best_profile"] = (
            f"{best_root.get('leaf_profile')} / {best_root.get('internal_profile')}"
            if best_root
            else ""
        )

    rows.sort(
        key=lambda row: (
            int(row["train_doc_count"]),
            _safe_float(row.get("mean_test_root_mae"), float("inf")),
            *_profile_rank(row),
        )
    )
    best_overall = min(
        rows,
        key=lambda row: (
            _safe_float(row.get("mean_test_root_mae"), float("inf")),
            int(_safe_int(row.get("train_doc_count"), 0)),
            *_profile_rank(row),
        ),
        default={},
    )
    return {
        "rows": rows,
        "by_train_docs": by_train_docs,
        "best_by_train_docs": {
            train_docs: dict(payload.get("best_root_row") or {})
            for train_docs, payload in sorted(
                by_train_docs.items(),
                key=lambda item: int(_safe_int(item[0], 0)),
            )
        },
        "best_overall": dict(best_overall),
        "leaf_profiles": list(SUPERVISION_LEAF_PROFILE_ORDER),
        "internal_profiles": list(SUPERVISION_INTERNAL_PROFILE_ORDER),
    }


def _supervision_recovery_leaf_geometry_tag(fixed_leaf_tokens: int) -> str:
    value = int(max(0, int(fixed_leaf_tokens)))
    if value <= 0:
        return ""
    width = max(3, len(str(value)))
    return f"leaf{value:0{width}d}"


def _supervision_recovery_geometry_info(
    row: Mapping[str, Any] | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    row_map = dict(row or {})
    config_map = dict(config or {})
    baseline_family = str(
        row_map.get("baseline_family", config_map.get("baseline_family", "")) or ""
    ).strip()
    pipeline_leaf_tokens = int(
        _safe_int(
            row_map.get(
                "pipeline_supervision_recovery_leaf_tokens",
                config_map.get("pipeline_supervision_recovery_leaf_tokens"),
            ),
            0,
        )
    )
    requested_fixed_leaf_tokens = int(
        _safe_int(
            row_map.get(
                "requested_fixed_leaf_tokens",
                row_map.get(
                    "fixed_leaf_tokens",
                    config_map.get(
                        "requested_fixed_leaf_tokens",
                        config_map.get("fixed_leaf_tokens"),
                    ),
                ),
            ),
            0,
        )
    )
    executed_fixed_leaf_tokens = int(
        _safe_int(
            row_map.get(
                "executed_fixed_leaf_tokens",
                row_map.get(
                    "fixed_leaf_tokens",
                    config_map.get("executed_fixed_leaf_tokens", config_map.get("fixed_leaf_tokens")),
                ),
            ),
            0,
        )
    )
    computed_assumed_doc_tokens = int(
        _safe_int(
            row_map.get(
                "computed_assumed_doc_tokens",
                config_map.get("computed_assumed_doc_tokens"),
            ),
            0,
        )
    )
    computed_assumed_leaves = int(
        _safe_int(
            row_map.get(
                "computed_assumed_leaves",
                config_map.get("computed_assumed_leaves"),
            ),
            0,
        )
    )
    executed_leaves_per_doc = int(
        _safe_int(
            row_map.get("executed_leaves_per_doc"),
            _safe_int(
                row_map.get("test_mean_leaves_per_doc"),
                config_map.get("executed_leaves_per_doc", computed_assumed_leaves),
            ),
        )
    )
    if executed_leaves_per_doc <= 0 and computed_assumed_doc_tokens > 0 and executed_fixed_leaf_tokens > 0:
        executed_leaves_per_doc = int(
            math.ceil(
                float(computed_assumed_doc_tokens)
                / float(max(1, executed_fixed_leaf_tokens))
            )
        )
    canonical_fno_geometry = bool(
        baseline_family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
        and executed_fixed_leaf_tokens == int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
    )
    label_leaf_tokens = int(
        FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
        if canonical_fno_geometry
        else (
            pipeline_leaf_tokens
            or requested_fixed_leaf_tokens
            or executed_fixed_leaf_tokens
        )
    )
    geometry_label = _supervision_recovery_leaf_geometry_tag(label_leaf_tokens)
    key_bits: List[str] = [str(geometry_label)] if str(geometry_label).strip() else []
    if requested_fixed_leaf_tokens > 0 and not canonical_fno_geometry:
        key_bits.append(f"req{int(requested_fixed_leaf_tokens)}")
    if executed_fixed_leaf_tokens > 0:
        key_bits.append(f"exec{int(executed_fixed_leaf_tokens)}")
    if executed_leaves_per_doc > 0:
        key_bits.append(f"n{int(executed_leaves_per_doc)}")
    geometry_key = "__".join(key_bits)
    return {
        "pipeline_supervision_recovery_leaf_tokens": int(pipeline_leaf_tokens),
        "supervision_recovery_geometry_key": str(geometry_key),
        "supervision_recovery_geometry_label": str(
            geometry_label
            or (
                f"req{int(requested_fixed_leaf_tokens)}"
                if requested_fixed_leaf_tokens > 0
                else (
                    f"exec{int(executed_fixed_leaf_tokens)}"
                    if executed_fixed_leaf_tokens > 0
                    else ""
                )
            )
        ),
        "requested_fixed_leaf_tokens": int(requested_fixed_leaf_tokens),
        "executed_fixed_leaf_tokens": int(executed_fixed_leaf_tokens),
        "executed_leaves_per_doc": int(executed_leaves_per_doc),
        "computed_assumed_doc_tokens": int(computed_assumed_doc_tokens),
    }


def _aggregate_supervision_recovery_from_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    tree_family: str = SUPERVISION_RECOVERY_TREE_FAMILY,
    recoverable_benchmark: str = SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK,
    structural_grid: str = SUPERVISION_RECOVERY_STRUCTURAL_GRID,
    structural_cell: str = SUPERVISION_RECOVERY_STRUCTURAL_CELL,
    package_order: Sequence[str] | None = None,
) -> Dict[str, Any]:
    resolved_package_order = _resolve_supervision_recovery_package_order(package_order)
    grouped: Dict[
        tuple[str, int, str, str, str, float, str, str, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)
    runtime_grouped: Dict[
        tuple[str, int, str, str, str, float, str, str, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)
    completed_runtime_rows: List[Dict[str, Any]] = []
    scope_labels: Dict[str, str] = {}
    scope_hardness: Dict[str, str] = {}
    observed_packages: set[str] = set()
    observed_train_docs: set[int] = set()
    observed_seeds: set[int] = set()
    scope_package_accounting: Dict[str, Dict[tuple[str, str], Dict[str, Any]]] = defaultdict(dict)

    def _diagnostic_metric_value(row: Mapping[str, Any], metric_name: str) -> float:
        keys = [
            f"{metric_name}_mean",
            metric_name,
            f"test_{metric_name}",
            *SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTIC_ALIASES.get(
                str(metric_name),
                (),
            ),
        ]
        for key in keys:
            value = _safe_float(row.get(key), float("nan"))
            if math.isfinite(value):
                return float(value)
        return float("nan")

    for payload in payloads:
        config = dict(payload.get("config") or {})
        package_name = _gs(config, "pipeline_supervision_recovery_package")
        scope_key_config = _gs(config, "pipeline_supervision_recovery_scope")
        scope_label_config = _gs(config, "pipeline_supervision_recovery_scope_label")
        if not package_name:
            continue
        observed_packages.add(package_name)
        observed_seeds.add(int(_safe_int(config.get("data_seed"), 0)))
        saw_tree_row = False
        for row in list(payload.get("aggregate_rows") or []):
            if not isinstance(row, Mapping):
                continue
            train_doc_count = int(
                _safe_int(row.get("train_doc_count", config.get("train_docs")), 0)
            )
            baseline_family = _gs(row, "baseline_family")
            geometry_info = _supervision_recovery_geometry_info(row, config=config)
            geometry_key = str(geometry_info["supervision_recovery_geometry_key"])
            comparison_arm = _supervision_recovery_comparison_arm(
                {**config, **dict(row)}
            )
            comparison_mode = str(
                row.get("comparison_mode", config.get("comparison_mode", "")) or ""
            )
            comparison_semantics = str(row.get("comparison_semantics", "") or "")
            comparison_semantics_label = str(
                row.get("comparison_semantics_label", "") or ""
            )
            run_intent_hash = str(row.get("run_intent_hash", "") or "")
            run_intent_validation_status = str(
                row.get("run_intent_validation_status", "") or ""
            )
            depth_discount_gamma = round(
                _safe_float(
                    row.get(
                        "depth_discount_gamma",
                        config.get("depth_discount_gamma", 1.0),
                    ),
                    1.0,
                ),
                6,
            )
            if (
                baseline_family == str(tree_family)
                and comparison_arm == SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM
            ):
                saw_tree_row = True
            scope_key = str(
                row.get("cell_id", "")
                or scope_key_config
                or payload.get("benchmark", "")
                or ""
            ).strip()
            if train_doc_count <= 0 or not baseline_family or not scope_key:
                continue
            observed_train_docs.add(train_doc_count)
            scope_labels.setdefault(
                scope_key,
                scope_label_config
                or _supervision_recovery_scope_label(
                    scope_key,
                    recoverable_scope_key=str(recoverable_benchmark),
                    structural_grid=str(structural_grid),
                ),
            )
            scope_hardness.setdefault(scope_key, _ns(row.get("hardness_grid", payload.get("hardness_grid", ""))))
            grouped[
                (
                    scope_key,
                    train_doc_count,
                    package_name,
                    baseline_family,
                    comparison_arm,
                    float(depth_discount_gamma),
                    geometry_key,
                    comparison_semantics_label,
                    run_intent_hash,
                )
            ].append(
                {
                    **geometry_info,
                    "source_summary_json": str(
                        payload.get("source_summary_json", "") or ""
                    ),
                    "comparison_mode": str(comparison_mode),
                    "comparison_semantics": str(comparison_semantics),
                    "comparison_semantics_label": str(comparison_semantics_label),
                    "run_intent_hash": str(run_intent_hash),
                    "run_intent_validation_status": str(
                        run_intent_validation_status
                    ),
                    "comparison_arm": str(comparison_arm),
                    "depth_discount_gamma": float(depth_discount_gamma),
                    "test_root_mae_mean": _safe_float(
                        row.get("test_root_mae_mean", row.get("test_root_mae")),
                        float("nan"),
                    ),
                    "test_leaf_mae_mean": _safe_float(
                        row.get("test_leaf_mae_mean", row.get("test_leaf_mae")),
                        float("nan"),
                    ),
                    "test_merge_mae_mean": _safe_float(
                        row.get("test_merge_mae_mean", row.get("test_merge_mae")),
                        float("nan"),
                    ),
                    "train_root_mae_mean": _safe_float(
                        row.get("train_root_mae_mean", row.get("train_root_mae")),
                        float("nan"),
                    ),
                    "val_root_mae_mean": _safe_float(
                        row.get("val_root_mae_mean", row.get("val_root_mae")),
                        float("nan"),
                    ),
                    "best_epoch_mean": _safe_float(
                        row.get("best_epoch_mean", row.get("best_epoch")),
                        float("nan"),
                    ),
                    "selection_metric_name": str(
                        row.get("selection_metric", "")
                        or config.get("tree_checkpoint_metric", "")
                        or ""
                    ),
                    "selection_metric_value_mean": _safe_float(
                        row.get("selection_metric_value_mean", row.get("selection_metric_value")),
                        float("nan"),
                    ),
                    "test_unweighted_full_law_objective_mean": _safe_float(
                        row.get(
                            "test_unweighted_full_law_objective_mean",
                            row.get("test_unweighted_full_law_objective"),
                        ),
                        float("nan"),
                    ),
                    "val_unweighted_full_law_objective_mean": _safe_float(
                        row.get(
                            "val_unweighted_full_law_objective_mean",
                            row.get("val_unweighted_full_law_objective"),
                        ),
                        float("nan"),
                    ),
                    "test_unweighted_active_objective_mean": _safe_float(
                        row.get(
                            "test_unweighted_active_objective_mean",
                            row.get("test_unweighted_active_objective"),
                        ),
                        float("nan"),
                    ),
                    "val_unweighted_active_objective_mean": _safe_float(
                        row.get(
                            "val_unweighted_active_objective_mean",
                            row.get("val_unweighted_active_objective"),
                        ),
                        float("nan"),
                    ),
                    "elapsed_s_mean": _safe_float(
                        row.get("elapsed_s_mean", row.get("family_wall_clock_s_mean")),
                        float("nan"),
                    ),
                    "n_runs": max(1, int(_safe_int(row.get("n_runs"), 1))),
                    "budget_total_calls_per_doc": _safe_float(
                        row.get("budget_total_calls_per_doc", config.get("budget_total_calls_per_doc", 0.0)),
                        0.0,
                    ),
                    "full_doc_budget_share": _safe_float(
                        row.get("full_doc_budget_share", config.get("full_doc_budget_share", 1.0)),
                        1.0,
                    ),
                    "doc_consumption_mode": str(
                        row.get("doc_consumption_mode", config.get("doc_consumption_mode", ""))
                        or ""
                    ),
                    "package_semantics": str(
                        row.get(
                            "package_semantics",
                            config.get(
                                "package_semantics",
                                _default_supervision_recovery_package_semantics(
                                    str(package_name),
                                    config,
                                ),
                            ),
                        )
                        or ""
                    ),
                    "local_split_mode": str(
                        row.get("local_split_mode", config.get("local_split_mode", ""))
                        or ""
                    ),
                    "leaf_supervision_kind": str(
                        row.get("leaf_supervision_kind", config.get("leaf_supervision_kind", ""))
                        or ""
                    ),
                    "leaf_label_rate": _safe_float(
                        row.get("leaf_label_rate", config.get("leaf_label_rate", 0.0)),
                        0.0,
                    ),
                    "internal_supervision_kind": str(
                        row.get("internal_supervision_kind", config.get("internal_supervision_kind", ""))
                        or ""
                    ),
                    "internal_label_rate": _safe_float(
                        row.get("internal_label_rate", config.get("internal_label_rate", 0.0)),
                        0.0,
                    ),
                    "max_internal_depth": int(
                        _safe_int(
                            row.get(
                                "max_internal_depth",
                                config.get("max_internal_depth", 0),
                            ),
                            0,
                        )
                    ),
                    "effective_full_doc_mass_per_doc_mean": _safe_float(
                        row.get(
                            "effective_full_doc_mass_per_doc_mean",
                            row.get("effective_full_doc_mass_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "requested_root_mass_per_doc": _safe_float(
                        row.get("requested_root_mass_per_doc"),
                        float("nan"),
                    ),
                    "root_supervision_docs_total": _safe_float(
                        row.get("root_supervision_docs_total"),
                        float("nan"),
                    ),
                    "mass_target_per_doc": _safe_float(
                        row.get("mass_target_per_doc", config.get("mass_target_per_doc")),
                        float("nan"),
                    ),
                    "computed_doc_review_mass_per_doc": _safe_float(
                        row.get(
                            "computed_doc_review_mass_per_doc",
                            config.get("computed_doc_review_mass_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_local_mass_per_doc": _safe_float(
                        row.get(
                            "computed_local_mass_per_doc",
                            config.get("computed_local_mass_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_leaf_mass_per_doc": _safe_float(
                        row.get(
                            "computed_leaf_mass_per_doc",
                            config.get("computed_leaf_mass_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_internal_mass_per_doc": _safe_float(
                        row.get(
                            "computed_internal_mass_per_doc",
                            config.get("computed_internal_mass_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_total_mass_per_doc": _safe_float(
                        row.get(
                            "computed_total_mass_per_doc",
                            config.get("computed_total_mass_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_leaf_mass_full_per_doc": _safe_float(
                        row.get(
                            "computed_leaf_mass_full_per_doc",
                            config.get("computed_leaf_mass_full_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_internal_mass_full_per_doc": _safe_float(
                        row.get(
                            "computed_internal_mass_full_per_doc",
                            config.get("computed_internal_mass_full_per_doc"),
                        ),
                        float("nan"),
                    ),
                    "computed_assumed_doc_tokens": int(
                        _safe_int(
                            row.get(
                                "computed_assumed_doc_tokens",
                                config.get("computed_assumed_doc_tokens"),
                            ),
                            0,
                        )
                    ),
                    "computed_assumed_leaves": int(
                        _safe_int(
                            row.get(
                                "computed_assumed_leaves",
                                config.get("computed_assumed_leaves"),
                            ),
                            0,
                        )
                    ),
                    "computed_assumed_internal_nodes": int(
                        _safe_int(
                            row.get(
                                "computed_assumed_internal_nodes",
                                config.get("computed_assumed_internal_nodes"),
                            ),
                            0,
                        )
                    ),
                    "train_mean_leaves_per_doc": _safe_float(
                        row.get("train_mean_leaves_per_doc"),
                        float("nan"),
                    ),
                    "val_mean_leaves_per_doc": _safe_float(
                        row.get("val_mean_leaves_per_doc"),
                        float("nan"),
                    ),
                    "test_mean_leaves_per_doc": _safe_float(
                        row.get("test_mean_leaves_per_doc"),
                        float("nan"),
                    ),
                    "fixed_leaf_tokens": int(
                        _safe_int(
                            row.get(
                                "fixed_leaf_tokens",
                                config.get("fixed_leaf_tokens"),
                            ),
                            0,
                        )
                    ),
                    "requested_fixed_leaf_tokens": int(
                        geometry_info["requested_fixed_leaf_tokens"]
                    ),
                    "executed_fixed_leaf_tokens": int(
                        geometry_info["executed_fixed_leaf_tokens"]
                    ),
                    "executed_leaves_per_doc": int(
                        geometry_info["executed_leaves_per_doc"]
                    ),
                    "executed_internal_nodes_per_doc": int(
                        _safe_int(row.get("executed_internal_nodes_per_doc"), 0)
                    ),
                    "tree_exact_collapse_mode": str(
                        row.get(
                            "tree_exact_collapse_mode",
                            config.get("tree_exact_collapse_mode", ""),
                        )
                        or ""
                    ),
                    "parity_mode": str(row.get("parity_mode", "") or ""),
                    "is_exact_full_doc_parity_row": bool(
                        row.get("is_exact_full_doc_parity_row", False)
                    ),
                    "tree_supervision_source": str(
                        row.get(
                            "tree_supervision_source",
                            config.get("tree_supervision_source", ""),
                        )
                        or ""
                    ),
                    "local_estimand_mode": str(
                        row.get(
                            "local_estimand_mode",
                            config.get("tree_local_weighting_mode", ""),
                        )
                        or ""
                    ),
                    "c2_pair_weighting_mode": str(
                        row.get("c2_pair_weighting_mode", "") or ""
                    ),
                    "tree_model_version": str(
                        row.get(
                            "tree_model_version",
                            config.get("tree_model_version", ""),
                        )
                        or ""
                    ),
                    "tree_runtime_merge_kind": str(
                        row.get(
                            "tree_runtime_merge_kind",
                            config.get("tree_runtime_merge_kind", ""),
                        )
                        or ""
                    ),
                    "tree_exact_projected_merge_is_runtime_merge": bool(
                        row.get(
                            "tree_exact_projected_merge_is_runtime_merge",
                            config.get(
                                "tree_exact_projected_merge_is_runtime_merge",
                                False,
                            ),
                        )
                    ),
                    **{
                        f"{metric_name}_mean": _diagnostic_metric_value(
                            row,
                            metric_name,
                        )
                        for metric_name in SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTICS
                    },
                }
            )
        for run in list(payload.get("runs") or []):
            if not isinstance(run, Mapping):
                continue
            train_doc_count = int(
                _safe_int(run.get("train_doc_count", config.get("train_docs")), 0)
            )
            baseline_family = _gs(run, "baseline_family")
            scope_key = str(
                run.get("cell_id", "")
                or scope_key_config
                or payload.get("benchmark", "")
                or ""
            ).strip()
            if train_doc_count <= 0 or not baseline_family or not scope_key:
                continue
            runtime = dict(run.get("runtime_efficiency") or {})
            autotuned = dict(run.get("autotuned_batch_budgets") or {})
            run_config = dict(run.get("config") or {})
            geometry_info = _supervision_recovery_geometry_info(run, config=run_config or config)
            geometry_key = str(geometry_info["supervision_recovery_geometry_key"])
            comparison_arm = _supervision_recovery_comparison_arm(
                run_config or config
            )
            comparison_semantics_label = str(
                run.get("comparison_semantics_label", "") or ""
            )
            run_intent_hash = str(run.get("run_intent_hash", "") or "")
            depth_discount_gamma = round(
                _safe_float(
                    run.get(
                        "depth_discount_gamma",
                        run_config.get(
                            "depth_discount_gamma",
                            config.get("depth_discount_gamma", 1.0),
                        ),
                    ),
                    1.0,
                ),
                6,
            )
            runtime_grouped[
                (
                    scope_key,
                    train_doc_count,
                    package_name,
                    baseline_family,
                    comparison_arm,
                    float(depth_discount_gamma),
                    geometry_key,
                    comparison_semantics_label,
                    run_intent_hash,
                )
            ].append(
                {
                    **geometry_info,
                    "comparison_arm": str(comparison_arm),
                    "depth_discount_gamma": float(depth_discount_gamma),
                    "runtime_data_mode": str(
                        runtime.get(
                            "runtime_data_mode",
                            run_config.get(
                                "gpu_runtime_data_mode",
                                config.get("gpu_runtime_data_mode", ""),
                            ),
                        )
                        or ""
                    ),
                    "runtime_bucket_mode": str(
                        runtime.get(
                            "runtime_bucket_mode",
                            run_config.get(
                                "gpu_runtime_bucket_mode",
                                config.get("gpu_runtime_bucket_mode", ""),
                            ),
                        )
                        or ""
                    ),
                    "steady_state_h2d_bytes": _safe_float(
                        runtime.get("steady_state_h2d_bytes"),
                        0.0,
                    ),
                    "steady_state_h2d_events": _safe_float(
                        runtime.get("steady_state_h2d_events"),
                        0.0,
                    ),
                    "resident_store_hits": _safe_float(
                        runtime.get("resident_store_hits"),
                        0.0,
                    ),
                    "resident_store_misses": _safe_float(
                        runtime.get("resident_store_misses"),
                        0.0,
                    ),
                    "auto_queue_family_count": _safe_float(
                        runtime.get("auto_queue_family_count"),
                        0.0,
                    ),
                    "auto_queue_target_leaf_counts": _normalize_target_leaf_counts(
                        autotuned.get(
                            "auto_queue_target_leaf_counts",
                            runtime.get("auto_queue_target_leaf_counts", ()),
                        )
                    ),
                    "structural_padding_waste_ratio": _safe_float(
                        runtime.get("structural_padding_waste_ratio"),
                        float("nan"),
                    ),
                    "auto_queue_fused_batches": _safe_float(
                        runtime.get("auto_queue_fused_batches"),
                        0.0,
                    ),
                    "auto_queue_generic_fallback_batches": _safe_float(
                        runtime.get("auto_queue_generic_fallback_batches"),
                        0.0,
                    ),
                    "fixed_shape_dense_bucket_store_hits": _safe_float(
                        runtime.get("fixed_shape_dense_bucket_store_hits"),
                        0.0,
                    ),
                }
            )
            completed_runtime_rows.append(
                _supervision_recovery_runtime_row_from_payload(
                    payload,
                    run,
                )
            )
        if scope_key_config:
            config_geometry = _supervision_recovery_geometry_info(config=config)
            accounting_key = (
                str(package_name),
                str(config_geometry["supervision_recovery_geometry_key"]),
            )
            existing_accounting = dict(
                (scope_package_accounting.get(scope_key_config) or {}).get(accounting_key) or {}
            )
            if (not existing_accounting) or (
                saw_tree_row and not bool(existing_accounting.get("from_tree"))
            ):
                scope_package_accounting[str(scope_key_config)][accounting_key] = {
                    "from_tree": bool(saw_tree_row),
                    **config_geometry,
                    "package_semantics": str(
                        config.get(
                            "package_semantics",
                            _default_supervision_recovery_package_semantics(
                                str(package_name),
                                config,
                            ),
                        )
                        or ""
                    ),
                    "mass_target_per_doc": _safe_float(
                        config.get("mass_target_per_doc"),
                        float("nan"),
                    ),
                    "computed_doc_review_mass_per_doc": _safe_float(
                        config.get("computed_doc_review_mass_per_doc"),
                        float("nan"),
                    ),
                    "computed_local_mass_per_doc": _safe_float(
                        config.get("computed_local_mass_per_doc"),
                        float("nan"),
                    ),
                    "computed_leaf_mass_per_doc": _safe_float(
                        config.get("computed_leaf_mass_per_doc"),
                        float("nan"),
                    ),
                    "computed_internal_mass_per_doc": _safe_float(
                        config.get("computed_internal_mass_per_doc"),
                        float("nan"),
                    ),
                    "computed_total_mass_per_doc": _safe_float(
                        config.get("computed_total_mass_per_doc"),
                        float("nan"),
                    ),
                    "computed_leaf_mass_full_per_doc": _safe_float(
                        config.get("computed_leaf_mass_full_per_doc"),
                        float("nan"),
                    ),
                    "computed_internal_mass_full_per_doc": _safe_float(
                        config.get("computed_internal_mass_full_per_doc"),
                        float("nan"),
                    ),
                    "computed_assumed_doc_tokens": int(
                        _safe_int(config.get("computed_assumed_doc_tokens"), 0)
                    ),
                    "computed_assumed_leaves": int(
                        _safe_int(config.get("computed_assumed_leaves"), 0)
                    ),
                    "computed_assumed_internal_nodes": int(
                        _safe_int(config.get("computed_assumed_internal_nodes"), 0)
                    ),
                    "fixed_leaf_tokens": int(_safe_int(config.get("fixed_leaf_tokens"), 0)),
                    "requested_fixed_leaf_tokens": int(
                        _safe_int(config.get("fixed_leaf_tokens"), 0)
                    ),
                    "max_internal_depth": int(_safe_int(config.get("max_internal_depth"), 0)),
                }

    def _weighted_mean(items: Sequence[Mapping[str, Any]], field: str) -> float:
        numer = 0.0
        denom = 0.0
        for item in items:
            value = _safe_float(item.get(field), float("nan"))
            weight = max(1.0, float(_safe_int(item.get("n_runs"), 1)))
            if not math.isfinite(value):
                continue
            numer += weight * value
            denom += weight
        return float(numer / denom) if denom > 0.0 else float("nan")

    def _representative_string(items: Sequence[Mapping[str, Any]], field: str) -> str:
        counter = Counter(
            str(item.get(field, "") or "").strip()
            for item in items
            if str(item.get(field, "") or "").strip()
        )
        return str(counter.most_common(1)[0][0]) if counter else ""

    def _weighted_mean_with_fallback(
        primary_items: Sequence[Mapping[str, Any]],
        fallback_items: Sequence[Mapping[str, Any]],
        field: str,
    ) -> float:
        value = _weighted_mean(primary_items, field)
        if math.isfinite(value):
            return value
        return _weighted_mean(fallback_items, field)

    def _representative_mapping(
        items: Sequence[Mapping[str, Any]],
        field: str,
    ) -> Dict[str, Any]:
        for item in items:
            value = item.get(field)
            if isinstance(value, Mapping) and value:
                return dict(value)
        return {}

    def _monotone_lower_envelope(
        rows: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        envelope: List[Dict[str, Any]] = []
        best_so_far = float("inf")
        for row in sorted(
            (dict(item) for item in rows),
            key=lambda item: int(_safe_int(item.get("train_doc_count"), 0)),
        ):
            train_doc_count = int(_safe_int(row.get("train_doc_count"), 0))
            if train_doc_count <= 0:
                continue
            mae = _safe_float(row.get("test_root_mae_mean"), float("nan"))
            if not math.isfinite(mae):
                continue
            best_so_far = min(best_so_far, float(mae))
            envelope.append(
                {
                    "train_doc_count": int(train_doc_count),
                    "test_root_mae_mean": float(best_so_far),
                    **row,
                }
            )
        return envelope

    def _interpolate_canonical_equivalent_train_docs(
        curve_rows: Sequence[Mapping[str, Any]],
        *,
        target_mae: float,
    ) -> Dict[str, Any]:
        rows = _monotone_lower_envelope(curve_rows)
        if not rows:
            return {
                "equivalent_train_docs": float("nan"),
                "relation": "unavailable",
                "anchor_saturated": False,
            }
        maes = [_safe_float(row.get("test_root_mae_mean"), float("nan")) for row in rows]
        docs = [int(_safe_int(row.get("train_doc_count"), 0)) for row in rows]
        anchor_saturated = all(
            math.isfinite(mae) and abs(mae) <= 1e-12 for mae in maes
        )
        if anchor_saturated:
            return {
                "equivalent_train_docs": float("nan"),
                "relation": "anchor_saturated",
                "anchor_saturated": True,
                "min_train_docs": int(min(docs)),
                "max_train_docs": int(max(docs)),
            }
        min_docs = int(min(docs))
        max_docs = int(max(docs))
        worst_mae = float(maes[0])
        best_mae = float(maes[-1])
        if not math.isfinite(target_mae):
            relation = "unavailable"
            value = float("nan")
        elif target_mae < best_mae - 1e-12:
            relation = "below_floor"
            value = float(max_docs)
        elif target_mae > worst_mae + 1e-12:
            relation = "above_range"
            value = float(min_docs)
        else:
            relation = "interpolated"
            value = float(min_docs)
            for idx in range(len(rows) - 1):
                left_docs = float(int(_safe_int(rows[idx].get("train_doc_count"), 0)))
                right_docs = float(
                    int(_safe_int(rows[idx + 1].get("train_doc_count"), 0))
                )
                left_mae = float(_safe_float(rows[idx].get("test_root_mae_mean")))
                right_mae = float(
                    _safe_float(rows[idx + 1].get("test_root_mae_mean"))
                )
                if target_mae > left_mae + 1e-12 or target_mae < right_mae - 1e-12:
                    continue
                if abs(left_mae - right_mae) <= 1e-12:
                    value = float(left_docs)
                else:
                    left_x = math.log2(max(left_docs, 1.0))
                    right_x = math.log2(max(right_docs, 1.0))
                    ratio = float((target_mae - left_mae) / (right_mae - left_mae))
                    value = float(2.0 ** (left_x + ratio * (right_x - left_x)))
                break
        return {
            "equivalent_train_docs": float(value),
            "relation": relation,
            "anchor_saturated": False,
            "min_train_docs": min_docs,
            "max_train_docs": max_docs,
        }

    completed_runtime_grouped: Dict[
        tuple[str, int, str, str, str, float, str, str, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)
    for row in completed_runtime_rows:
        geometry_key = str(
            _supervision_recovery_geometry_info(row).get(
                "supervision_recovery_geometry_key",
                "",
            )
        )
        completed_runtime_grouped[
            (
                str(row.get("scope_key", "")),
                int(_safe_int(row.get("train_doc_count"), 0)),
                str(row.get("package_name", "")),
                str(row.get("baseline_family", "")),
                _supervision_recovery_comparison_arm(row),
                round(_safe_float(row.get("depth_discount_gamma"), 1.0), 6),
                geometry_key,
                str(row.get("comparison_semantics_label", "") or ""),
                str(row.get("run_intent_hash", "") or ""),
            )
        ].append(dict(row))

    family_rows: List[Dict[str, Any]] = []
    exact_collapse_family_rows: List[Dict[str, Any]] = []
    for (
        scope_key,
        train_doc_count,
        package_name,
        baseline_family,
        comparison_arm,
        depth_discount_gamma,
        geometry_key,
        comparison_semantics_label,
        run_intent_hash,
    ), items in sorted(grouped.items()):
        representative = dict(items[0]) if items else {}
        runtime_items = list(
            runtime_grouped.get(
                (
                    scope_key,
                    train_doc_count,
                    package_name,
                    baseline_family,
                    comparison_arm,
                    depth_discount_gamma,
                    geometry_key,
                    comparison_semantics_label,
                    run_intent_hash,
                ),
                [],
            )
        )
        completed_items = list(
            completed_runtime_grouped.get(
                (
                    scope_key,
                    train_doc_count,
                    package_name,
                    baseline_family,
                    comparison_arm,
                    depth_discount_gamma,
                    geometry_key,
                    comparison_semantics_label,
                    run_intent_hash,
                ),
                [],
            )
        )
        runtime_data_mode = _representative_string(
            runtime_items or completed_items,
            "runtime_data_mode",
        )
        runtime_bucket_mode = _representative_string(
            runtime_items or completed_items,
            "runtime_bucket_mode",
        )
        tree_batch_pack_mode = _representative_string(
            completed_items,
            "tree_batch_pack_mode",
        )
        detail_items = list(completed_items or items)
        computed_assumed_doc_tokens = int(
            _safe_int(_weighted_mean(items, "computed_assumed_doc_tokens"), 0)
        )
        computed_assumed_leaves = int(
            _safe_int(_weighted_mean(items, "computed_assumed_leaves"), 0)
        )
        computed_assumed_internal_nodes = int(
            _safe_int(_weighted_mean(items, "computed_assumed_internal_nodes"), 0)
        )
        executed_fixed_leaf_tokens = int(
            _safe_int(_weighted_mean(completed_items, "fixed_leaf_tokens"), 0)
        )
        if executed_fixed_leaf_tokens <= 0:
            executed_fixed_leaf_tokens = int(
                _safe_int(_weighted_mean(items, "fixed_leaf_tokens"), 0)
            )
        requested_fixed_leaf_tokens = int(
            _safe_int(_weighted_mean(items, "requested_fixed_leaf_tokens"), 0)
        )
        if requested_fixed_leaf_tokens <= 0:
            requested_fixed_leaf_tokens = int(
                _safe_int(_weighted_mean(items, "fixed_leaf_tokens"), 0)
            )
        train_mean_leaves_per_doc = _weighted_mean(items, "train_mean_leaves_per_doc")
        val_mean_leaves_per_doc = _weighted_mean(items, "val_mean_leaves_per_doc")
        test_mean_leaves_per_doc = _weighted_mean(items, "test_mean_leaves_per_doc")
        if math.isfinite(test_mean_leaves_per_doc) and test_mean_leaves_per_doc > 0.0:
            executed_leaves_per_doc = int(round(test_mean_leaves_per_doc))
        elif executed_fixed_leaf_tokens > 0 and computed_assumed_doc_tokens > 0:
            executed_leaves_per_doc = int(
                math.ceil(
                    float(computed_assumed_doc_tokens)
                    / float(max(1, executed_fixed_leaf_tokens))
                )
            )
        else:
            executed_leaves_per_doc = int(computed_assumed_leaves)
        executed_internal_nodes_per_doc = (
            max(0, int(executed_leaves_per_doc) - 1)
            if int(executed_leaves_per_doc) > 0
            else int(computed_assumed_internal_nodes)
        )
        tree_exact_collapse_mode = (
            _representative_string(completed_items, "tree_exact_collapse_mode")
            or _representative_string(items, "tree_exact_collapse_mode")
        )
        parity_mode = _representative_string(items, "parity_mode")
        is_exact_full_doc_parity_row = bool(
            str(parity_mode).strip() == "exact_full_doc"
            or (
                str(tree_exact_collapse_mode).strip()
                == EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
                and int(executed_leaves_per_doc) == 1
            )
        )
        is_fno_equivalent_geometry = bool(
            int(executed_leaves_per_doc) == 1
            and bool(is_exact_full_doc_parity_row)
        )
        is_canonical_full_doc_geometry = bool(
            int(executed_fixed_leaf_tokens)
            == int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
            and (
                int(executed_leaves_per_doc) in {0, 1}
                or int(computed_assumed_doc_tokens) in {0, FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS}
            )
        )
        is_authoritative_parity_row = bool(
            _representative_string(detail_items, "tree_supervision_source") == "manifest"
            and _representative_string(detail_items, "local_estimand_mode")
            == "span_mass_ipw_sum"
            and (
                _representative_string(detail_items, "c2_pair_weighting_mode")
                in {"", "pair_ipw_geomean"}
            )
        )
        report_row_geometry_label = (
            ""
            if is_canonical_full_doc_geometry
            else str(
                representative.get("supervision_recovery_geometry_label", "") or ""
            ).strip()
        )
        row_payload = {
            "source_summary_json": _representative_string(
                items,
                "source_summary_json",
            ),
            "pipeline_supervision_recovery_leaf_tokens": int(
                _safe_int(
                    representative.get("pipeline_supervision_recovery_leaf_tokens"),
                    0,
                )
            ),
            "supervision_recovery_geometry_key": str(
                representative.get("supervision_recovery_geometry_key", geometry_key) or geometry_key
            ),
            "supervision_recovery_geometry_label": str(
                representative.get("supervision_recovery_geometry_label", "") or ""
            ),
            "scope_key": str(scope_key),
            "scope_label": str(
                scope_labels.get(scope_key)
                or _supervision_recovery_scope_label(
                    scope_key,
                    recoverable_scope_key=str(recoverable_benchmark),
                    structural_grid=str(structural_grid),
                )
            ),
            "hardness_grid": str(scope_hardness.get(scope_key, "") or ""),
            "train_doc_count": int(train_doc_count),
            "package_name": str(package_name),
            "package_semantics": str(
                representative.get(
                    "package_semantics",
                    _default_supervision_recovery_package_semantics(
                        str(package_name),
                        dict(
                            SUPERVISION_RECOVERY_PACKAGE_SPECS.get(
                                str(package_name),
                                {},
                            )
                        ),
                    ),
                )
                or ""
            ),
            "report_row_key": (
                (
                    f"{package_name}__"
                    f"{report_row_geometry_label}__"
                    f"g{float(depth_discount_gamma):0.2f}"
                ).replace(".", "p").replace("____", "__").strip("_")
            ),
            "baseline_family": str(baseline_family),
            "comparison_mode": _representative_string(items, "comparison_mode"),
            "comparison_semantics": _representative_string(
                items,
                "comparison_semantics",
            ),
            "comparison_semantics_label": str(comparison_semantics_label),
            "run_intent_hash": str(run_intent_hash),
            "run_intent_validation_status": _representative_string(
                items,
                "run_intent_validation_status",
            ),
            "comparison_arm": str(comparison_arm),
            "depth_discount_gamma": float(depth_discount_gamma),
            "n_runs": int(
                sum(max(1, int(_safe_int(item.get("n_runs"), 1))) for item in items)
            ),
            "test_root_mae_mean": _weighted_mean(items, "test_root_mae_mean"),
            "test_leaf_mae_mean": _weighted_mean(items, "test_leaf_mae_mean"),
            "test_merge_mae_mean": _weighted_mean(items, "test_merge_mae_mean"),
            "train_root_mae_mean": _weighted_mean(items, "train_root_mae_mean"),
            "val_root_mae_mean": _weighted_mean(items, "val_root_mae_mean"),
            "best_epoch_mean": _weighted_mean(items, "best_epoch_mean"),
            "selection_metric_name": (
                _representative_string(items, "selection_metric_name")
                or _representative_string(completed_items, "tree_checkpoint_metric")
            ),
            "selection_metric_value_mean": _weighted_mean(
                items,
                "selection_metric_value_mean",
            ),
            "test_unweighted_full_law_objective_mean": _weighted_mean(
                items,
                "test_unweighted_full_law_objective_mean",
            ),
            "val_unweighted_full_law_objective_mean": _weighted_mean(
                items,
                "val_unweighted_full_law_objective_mean",
            ),
            "test_unweighted_active_objective_mean": _weighted_mean(
                items,
                "test_unweighted_active_objective_mean",
            ),
            "val_unweighted_active_objective_mean": _weighted_mean(
                items,
                "val_unweighted_active_objective_mean",
            ),
            "elapsed_s_mean": _weighted_mean(items, "elapsed_s_mean"),
            "budget_total_calls_per_doc": _safe_float(
                representative.get("budget_total_calls_per_doc"),
                0.0,
            ),
            "full_doc_budget_share": _safe_float(
                representative.get("full_doc_budget_share"),
                1.0,
            ),
            "doc_consumption_mode": str(
                representative.get("doc_consumption_mode", "") or ""
            ),
            "local_split_mode": str(
                representative.get("local_split_mode", "") or ""
            ),
            "leaf_supervision_kind": str(
                representative.get("leaf_supervision_kind", "") or ""
            ),
            "leaf_label_rate": _safe_float(representative.get("leaf_label_rate"), 0.0),
            "internal_supervision_kind": str(
                representative.get("internal_supervision_kind", "") or ""
            ),
            "internal_label_rate": _safe_float(
                representative.get("internal_label_rate"),
                0.0,
            ),
            "max_internal_depth": int(
                _safe_int(representative.get("max_internal_depth"), 0)
            ),
            "effective_full_doc_mass_per_doc_mean": _weighted_mean(
                items,
                "effective_full_doc_mass_per_doc_mean",
            ),
            "requested_root_mass_per_doc_mean": _weighted_mean_with_fallback(
                items,
                completed_items,
                "requested_root_mass_per_doc",
            ),
            "root_supervision_docs_total_mean": _weighted_mean_with_fallback(
                items,
                completed_items,
                "root_supervision_docs_total",
            ),
            "mass_target_per_doc": _weighted_mean(items, "mass_target_per_doc"),
            "computed_doc_review_mass_per_doc": _weighted_mean(
                items,
                "computed_doc_review_mass_per_doc",
            ),
            "computed_local_mass_per_doc": _weighted_mean(
                items,
                "computed_local_mass_per_doc",
            ),
            "computed_leaf_mass_per_doc": _weighted_mean(
                items,
                "computed_leaf_mass_per_doc",
            ),
            "computed_internal_mass_per_doc": _weighted_mean(
                items,
                "computed_internal_mass_per_doc",
            ),
            "computed_total_mass_per_doc": _weighted_mean(
                items,
                "computed_total_mass_per_doc",
            ),
            "computed_leaf_mass_full_per_doc": _weighted_mean(
                items,
                "computed_leaf_mass_full_per_doc",
            ),
            "computed_internal_mass_full_per_doc": _weighted_mean(
                items,
                "computed_internal_mass_full_per_doc",
            ),
            "computed_assumed_doc_tokens": int(computed_assumed_doc_tokens),
            "computed_assumed_leaves": int(computed_assumed_leaves),
            "computed_assumed_internal_nodes": int(computed_assumed_internal_nodes),
            "train_mean_leaves_per_doc": float(train_mean_leaves_per_doc),
            "val_mean_leaves_per_doc": float(val_mean_leaves_per_doc),
            "test_mean_leaves_per_doc": float(test_mean_leaves_per_doc),
            "requested_fixed_leaf_tokens": int(requested_fixed_leaf_tokens),
            "executed_fixed_leaf_tokens": int(executed_fixed_leaf_tokens),
            "executed_leaves_per_doc": int(executed_leaves_per_doc),
            "executed_internal_nodes_per_doc": int(executed_internal_nodes_per_doc),
            "leaves_per_doc": int(executed_leaves_per_doc),
            "internal_nodes_per_doc": int(executed_internal_nodes_per_doc),
            "runtime_data_mode": str(runtime_data_mode),
            "runtime_bucket_mode": str(runtime_bucket_mode),
            "tree_batch_pack_mode": str(tree_batch_pack_mode),
            "tree_reference_mode": _representative_string(
                completed_items,
                "tree_reference_mode",
            ),
            "tree_reference_label": _representative_string(
                completed_items,
                "tree_reference_label",
            ),
            "tree_training_schedule": _representative_string(
                completed_items,
                "tree_training_schedule",
            ),
            "tree_checkpoint_metric": _representative_string(
                completed_items,
                "tree_checkpoint_metric",
            ),
            "tree_stage1_checkpoint_metric": _representative_string(
                completed_items,
                "tree_stage1_checkpoint_metric",
            ),
            "summary_spec_name": _representative_string(
                completed_items,
                "summary_spec_name",
            ),
            "slot_count": int(
                _safe_int(_weighted_mean(completed_items, "slot_count"), 0)
            ),
            "state_dim": int(
                _safe_int(_weighted_mean(completed_items, "state_dim"), 0)
            ),
            "hidden_dim": int(
                _safe_int(_weighted_mean(completed_items, "hidden_dim"), 0)
            ),
            "fixed_leaf_tokens": int(executed_fixed_leaf_tokens),
            "tree_exact_collapse_mode": str(tree_exact_collapse_mode),
            "parity_mode": str("exact_full_doc" if is_exact_full_doc_parity_row else ""),
            "is_exact_full_doc_parity_row": bool(is_exact_full_doc_parity_row),
            "is_fno_equivalent_geometry": bool(is_fno_equivalent_geometry),
            "is_authoritative_parity_row": bool(is_authoritative_parity_row),
            "tree_local_weighting_mode": _representative_string(
                detail_items,
                "tree_local_weighting_mode",
            ),
            "tree_supervision_source": _representative_string(
                detail_items,
                "tree_supervision_source",
            ),
            "local_estimand_mode": _representative_string(
                detail_items,
                "local_estimand_mode",
            ),
            "c2_pair_weighting_mode": _representative_string(
                detail_items,
                "c2_pair_weighting_mode",
            ),
            "tree_model_version": _representative_string(
                detail_items,
                "tree_model_version",
            ),
            "tree_runtime_merge_kind": _representative_string(
                detail_items,
                "tree_runtime_merge_kind",
            ),
            "tree_exact_projected_merge_is_runtime_merge_rate": _weighted_mean(
                detail_items,
                "tree_exact_projected_merge_is_runtime_merge",
            ),
            **{
                f"{metric_name}_mean": _weighted_mean(
                    detail_items,
                    f"{metric_name}_mean",
                )
                for metric_name in SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTICS
            },
            "local_sampling_design_name": _representative_string(
                detail_items,
                "local_sampling_design_name",
            ),
            "leaf_population_size": _weighted_mean(
                detail_items,
                "leaf_population_size",
            ),
            "leaf_sample_size": _weighted_mean(detail_items, "leaf_sample_size"),
            "leaf_effective_propensity": _weighted_mean(
                detail_items,
                "leaf_effective_propensity",
            ),
            "merge_population_size": _weighted_mean(
                detail_items,
                "merge_population_size",
            ),
            "merge_sample_size": _weighted_mean(detail_items, "merge_sample_size"),
            "merge_effective_propensity": _weighted_mean(
                detail_items,
                "merge_effective_propensity",
            ),
            "c2_same_pair_count": _weighted_mean(detail_items, "c2_same_pair_count"),
            "c2_different_pair_count": _weighted_mean(
                detail_items,
                "c2_different_pair_count",
            ),
            "c2_pair_weight_ess": _weighted_mean(
                detail_items,
                "c2_pair_weight_ess",
            ),
            "c2_pair_weight_max": _weighted_mean(
                detail_items,
                "c2_pair_weight_max",
            ),
            "local_objective_audit": _representative_mapping(
                detail_items,
                "local_objective_audit",
            ),
            "steady_state_h2d_bytes": _weighted_mean(
                runtime_items,
                "steady_state_h2d_bytes",
            ),
            "steady_state_h2d_events": _weighted_mean(
                runtime_items,
                "steady_state_h2d_events",
            ),
            "resident_store_hits": _weighted_mean(runtime_items, "resident_store_hits"),
            "resident_store_misses": _weighted_mean(
                runtime_items,
                "resident_store_misses",
            ),
            "auto_queue_family_count": _weighted_mean(
                runtime_items,
                "auto_queue_family_count",
            ),
            "auto_queue_target_leaf_counts": sorted(
                {
                    int(value)
                    for item in runtime_items
                    for value in list(item.get("auto_queue_target_leaf_counts") or [])
                    if int(_safe_int(value)) > 0
                }
            ),
            "structural_padding_waste_ratio": _weighted_mean(
                runtime_items,
                "structural_padding_waste_ratio",
            ),
            "auto_queue_fused_batches": _weighted_mean(
                runtime_items,
                "auto_queue_fused_batches",
            ),
            "auto_queue_generic_fallback_batches": _weighted_mean(
                runtime_items,
                "auto_queue_generic_fallback_batches",
            ),
            "fixed_shape_dense_bucket_store_hits": _weighted_mean(
                runtime_items,
                "fixed_shape_dense_bucket_store_hits",
            ),
            "wall_clock_s_mean": _weighted_mean(completed_items, "wall_clock_s"),
            "effective_train_epochs_mean": _weighted_mean(
                completed_items,
                "effective_train_epochs",
            ),
            "train_loop_s_mean": _weighted_mean(completed_items, "train_loop_s"),
            "stage1_train_loop_s_mean": _weighted_mean(
                completed_items,
                "stage1_train_loop_s",
            ),
            "stage2_train_loop_s_mean": _weighted_mean(
                completed_items,
                "stage2_train_loop_s",
            ),
            "exact_metric_eval_s_mean": _weighted_mean(
                completed_items,
                "exact_metric_eval_s",
            ),
            "train_loop_s_per_epoch_mean": _weighted_mean(
                completed_items,
                "train_loop_s_per_epoch",
            ),
            "train_loop_s_per_epoch_per_1k_docs_mean": _weighted_mean(
                completed_items,
                "train_loop_s_per_epoch_per_1k_docs",
            ),
            "wall_clock_s_per_epoch_mean": _weighted_mean(
                completed_items,
                "wall_clock_s_per_epoch",
            ),
            "fast_path_classification": _classify_supervision_recovery_fast_path(
                runtime_data_mode=str(runtime_data_mode),
                runtime_bucket_mode=str(runtime_bucket_mode),
                tree_batch_pack_mode=str(tree_batch_pack_mode),
                steady_state_h2d_bytes=_weighted_mean(
                    runtime_items,
                    "steady_state_h2d_bytes",
                ),
                steady_state_h2d_events=_weighted_mean(
                    runtime_items,
                    "steady_state_h2d_events",
                ),
                resident_store_hits=_weighted_mean(
                    runtime_items,
                    "resident_store_hits",
                ),
                auto_queue_fused_batches=_weighted_mean(
                    runtime_items,
                    "auto_queue_fused_batches",
                ),
                fixed_shape_dense_bucket_store_hits=_weighted_mean(
                    runtime_items,
                    "fixed_shape_dense_bucket_store_hits",
                ),
            ),
        }
        target_rows = (
            family_rows
            if comparison_arm == SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM
            else exact_collapse_family_rows
        )
        target_rows.append(
            annotate_downstream_v3_row(
                row_payload,
                canonical_fno_families=CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
                canonical_fno_fixed_leaf_tokens=FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS,
            )
        )

    all_family_rows = [dict(row) for row in family_rows]
    all_exact_collapse_family_rows = [dict(row) for row in exact_collapse_family_rows]
    quarantined_family_rows = filtered_quarantined_rows(all_family_rows)
    quarantined_exact_collapse_rows = filtered_quarantined_rows(
        all_exact_collapse_family_rows
    )
    family_rows = filtered_headline_rows(all_family_rows)
    exact_collapse_family_rows = filtered_headline_rows(
        all_exact_collapse_family_rows
    )

    scopes: Dict[str, Dict[str, Any]] = {}
    expected_packages = list(resolved_package_order)
    package_doc_equivalent: Dict[str, Dict[str, Any]] = {}
    scope_package_doc_equivalent: Dict[
        str,
        Dict[str, Dict[str, Dict[str, Any]]],
    ] = {}
    scope_keys_for_doc_equiv = sorted(
        {str(row["scope_key"]) for row in family_rows}
        | {str(recoverable_benchmark), str(structural_cell)}
    )
    for scope_key in scope_keys_for_doc_equiv:
        scope_entries: Dict[str, Dict[str, Dict[str, Any]]] = {}
        scope_accounting = dict(scope_package_accounting.get(str(scope_key)) or {})
        for (package_name, geometry_key), raw_accounting in sorted(
            scope_accounting.items(),
            key=lambda item: (str(item[0][0]), str(item[0][1])),
        ):
            accounting = dict(raw_accounting or {})
            if not accounting or str(package_name) not in expected_packages:
                continue
            entry = {
                "package_semantics": str(
                    accounting.get(
                        "package_semantics",
                        _default_supervision_recovery_package_semantics(
                            str(package_name),
                            dict(
                                SUPERVISION_RECOVERY_PACKAGE_SPECS.get(
                                    str(package_name),
                                    {},
                                )
                            ),
                        ),
                    )
                    or ""
                ),
                "assumed_doc_tokens": int(
                    _safe_int(accounting.get("computed_assumed_doc_tokens"), 0)
                ),
                "fixed_leaf_tokens": int(
                    _safe_int(accounting.get("fixed_leaf_tokens"), 0)
                ),
                "max_internal_depth": int(
                    _safe_int(accounting.get("max_internal_depth"), 0)
                ),
                "assumed_leaves": int(
                    _safe_int(accounting.get("computed_assumed_leaves"), 0)
                ),
                "assumed_internal_nodes": int(
                    _safe_int(accounting.get("computed_assumed_internal_nodes"), 0)
                ),
                "mass_target_per_doc": _safe_float(
                    accounting.get("mass_target_per_doc"),
                    float("nan"),
                ),
                "doc_review_mass_per_doc": _safe_float(
                    accounting.get("computed_doc_review_mass_per_doc"),
                    float("nan"),
                ),
                "leaf_mass_full_per_doc": _safe_float(
                    accounting.get("computed_leaf_mass_full_per_doc"),
                    float("nan"),
                ),
                "internal_mass_full_per_doc": _safe_float(
                    accounting.get("computed_internal_mass_full_per_doc"),
                    float("nan"),
                ),
                "leaf_mass_per_doc": _safe_float(
                    accounting.get("computed_leaf_mass_per_doc"),
                    float("nan"),
                ),
                "internal_mass_per_doc": _safe_float(
                    accounting.get("computed_internal_mass_per_doc"),
                    float("nan"),
                ),
                "local_mass_per_doc": _safe_float(
                    accounting.get("computed_local_mass_per_doc"),
                    float("nan"),
                ),
                "total_mass_per_doc": _safe_float(
                    accounting.get("computed_total_mass_per_doc"),
                    float("nan"),
                ),
                "pipeline_supervision_recovery_leaf_tokens": int(
                    _safe_int(
                        accounting.get("pipeline_supervision_recovery_leaf_tokens"),
                        0,
                    )
                ),
                "supervision_recovery_geometry_key": str(
                    accounting.get("supervision_recovery_geometry_key", geometry_key)
                    or geometry_key
                ),
                "supervision_recovery_geometry_label": str(
                    accounting.get("supervision_recovery_geometry_label", "") or ""
                ),
            }
            scope_entries.setdefault(str(package_name), {})[str(geometry_key)] = dict(entry)
            package_doc_equivalent.setdefault(
                str(package_name),
                dict(entry),
            )
        if scope_entries:
            scope_package_doc_equivalent[str(scope_key)] = scope_entries
    for scope_key in sorted(
        {str(row["scope_key"]) for row in family_rows}
        | {str(recoverable_benchmark), str(structural_cell)}
    ):
        scope_label = str(
            scope_labels.get(scope_key)
            or _supervision_recovery_scope_label(
                scope_key,
                recoverable_scope_key=str(recoverable_benchmark),
                structural_grid=str(structural_grid),
            )
        )
        scope_rows = [
            dict(row)
            for row in family_rows
            if str(row.get("scope_key", "")) == scope_key
        ]
        rows_by_train_docs: List[Dict[str, Any]] = []
        best_tree_by_train_docs: Dict[str, Dict[str, Any]] = {}
        available_train_docs = sorted(
            {
                int(_safe_int(row.get("train_doc_count")))
                for row in scope_rows
                if int(_safe_int(row.get("train_doc_count"))) > 0
            }
        )
        by_train_docs: Dict[
            int,
            Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for row in scope_rows:
            train_doc_count = int(_safe_int(row.get("train_doc_count")))
            package_name = str(row.get("package_name", "") or "")
            geometry_key = str(row.get("supervision_recovery_geometry_key", "") or "")
            baseline_family = str(row.get("baseline_family", "") or "")
            by_train_docs[train_doc_count][package_name][geometry_key][baseline_family] = dict(row)

        def _family_row_at(
            candidate_train_doc_count: int,
            package_name: str,
            family: str,
            geometry_key: str,
        ) -> Dict[str, Any]:
            candidate_package_map = dict(
                (by_train_docs.get(int(candidate_train_doc_count), {}) or {}).get(
                    str(package_name),
                    {},
                )
                or {}
            )
            exact_row = dict(
                dict(candidate_package_map.get(str(geometry_key), {}) or {}).get(
                    str(family),
                    {},
                )
                or {}
            )
            if exact_row:
                return exact_row
            fallback_rows = [
                dict(dict(family_map or {}).get(str(family), {}) or {})
                for family_map in candidate_package_map.values()
                if dict(dict(family_map or {}).get(str(family), {}) or {})
            ]
            if not fallback_rows:
                return {}
            return min(
                fallback_rows,
                key=lambda row: _safe_float(
                    row.get("test_root_mae_mean"),
                    float("inf"),
                ),
            )

        for train_doc_count in available_train_docs:
            package_map = by_train_docs.get(train_doc_count, {})
            comparisons: List[Dict[str, Any]] = []

            def _family_row_for(
                package_name: str,
                family: str,
                geometry_key: str,
            ) -> Dict[str, Any]:
                package_geometry_map = dict(package_map.get(str(package_name), {}) or {})
                exact_row = dict(
                    dict(package_geometry_map.get(str(geometry_key), {}) or {}).get(
                        str(family),
                        {},
                    )
                    or {}
                )
                if exact_row:
                    return exact_row
                fallback_rows = [
                    dict(dict(family_map or {}).get(str(family), {}) or {})
                    for family_map in package_geometry_map.values()
                    if dict(dict(family_map or {}).get(str(family), {}) or {})
                ]
                if not fallback_rows:
                    return {}
                return min(
                    fallback_rows,
                    key=lambda row: _safe_float(
                        row.get("test_root_mae_mean"),
                        float("inf"),
                    ),
                )

            def _best_fno_row_for(package_name: str, geometry_key: str) -> Dict[str, Any]:
                candidates = [
                    _family_row_for(str(package_name), family, str(geometry_key))
                    for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
                ]
                finite_candidates = [
                    dict(candidate)
                    for candidate in candidates
                    if dict(candidate)
                ]
                if not finite_candidates:
                    return {}
                return min(
                    finite_candidates,
                    key=lambda row: _safe_float(
                        row.get("test_root_mae_mean"),
                        float("inf"),
                    ),
                )

            for package_name in expected_packages:
                package_spec = dict(SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name])
                package_geometry_map = dict(package_map.get(package_name, {}) or {})
                tree_rows_for_package = [
                    dict(dict(family_map or {}).get(tree_family, {}) or {})
                    for family_map in package_geometry_map.values()
                    if dict(dict(family_map or {}).get(tree_family, {}) or {})
                ]
                tree_rows_for_package.sort(
                    key=lambda row: (
                        str(row.get("supervision_recovery_geometry_label", "") or ""),
                        int(_safe_int(row.get("executed_leaves_per_doc"), 0)),
                        int(_safe_int(row.get("executed_fixed_leaf_tokens"), 0)),
                    )
                )
                for tree_row in tree_rows_for_package:
                    geometry_key = str(
                        tree_row.get("supervision_recovery_geometry_key", "") or ""
                    )
                    package_doc_equiv = dict(
                        (
                            (
                                scope_package_doc_equivalent.get(str(scope_key))
                                or {}
                            ).get(str(package_name))
                            or {}
                        ).get(str(geometry_key))
                        or {}
                    )
                    fno_reference_package = str(
                        package_spec.get("fno_reference_package", "full10") or "full10"
                    )
                    fno_reference = dict(
                        _best_fno_row_for(str(fno_reference_package), str(geometry_key))
                    )
                    fno_ceiling = dict(_best_fno_row_for("full100", str(geometry_key)))
                    canonical_official_fno_full100 = dict(
                        _family_row_for("full100", "official_fno", str(geometry_key))
                    )
                    canonical_official_fno_full100_mae = _safe_float(
                        canonical_official_fno_full100.get("test_root_mae_mean"),
                        float("nan"),
                    )
                    tree_mae = _safe_float(
                        tree_row.get("test_root_mae_mean"),
                        float("nan"),
                    )
                    full10_fno_mae = _safe_float(
                        fno_reference.get("test_root_mae_mean"),
                        float("nan"),
                    )
                    full100_fno_mae = _safe_float(
                        fno_ceiling.get("test_root_mae_mean"),
                        float("nan"),
                    )
                    def _fno_breakdown_entry(
                        package_name: str,
                        family: str,
                    ) -> Dict[str, Any]:
                        family_row = dict(
                            _family_row_for(
                                str(package_name),
                                str(family),
                                str(geometry_key),
                            )
                        )
                        return {
                            "baseline_family": str(family),
                            "package_name": str(package_name),
                            "n_runs": int(_safe_int(family_row.get("n_runs"), 0)),
                            "test_root_mae": _safe_float(
                                family_row.get("test_root_mae_mean"),
                                float("nan"),
                            ),
                            "comparison_mode": str(
                                family_row.get("comparison_mode", "") or ""
                            ),
                            "comparison_semantics": str(
                                family_row.get("comparison_semantics", "") or ""
                            ),
                            "comparison_semantics_label": str(
                                family_row.get(
                                    "comparison_semantics_label",
                                    "",
                                )
                                or ""
                            ),
                            "run_intent_hash": str(
                                family_row.get("run_intent_hash", "") or ""
                            ),
                            "run_intent_validation_status": str(
                                family_row.get(
                                    "run_intent_validation_status",
                                    "",
                                )
                                or ""
                            ),
                            "requested_fixed_leaf_tokens": int(
                                _safe_int(
                                    family_row.get("requested_fixed_leaf_tokens"),
                                    0,
                                )
                            ),
                            "executed_fixed_leaf_tokens": int(
                                _safe_int(
                                    family_row.get("executed_fixed_leaf_tokens"),
                                    0,
                                )
                            ),
                            "contract_status": str(
                                family_row.get("contract_status", "") or ""
                            ),
                        }
                    matched_fno_family_rows = {
                        family: _fno_breakdown_entry(
                            str(fno_reference_package),
                            str(family),
                        )
                        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
                    }
                    for family_row in matched_fno_family_rows.values():
                        value = _safe_float(family_row.get("test_root_mae"), float("nan"))
                        family_row["delta_vs_tree"] = (
                            float(tree_mae - value)
                            if math.isfinite(tree_mae) and math.isfinite(value)
                            else float("nan")
                        )
                    full100_fno_family_rows = {
                        family: _fno_breakdown_entry("full100", str(family))
                        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
                    }
                    for family_row in full100_fno_family_rows.values():
                        value = _safe_float(family_row.get("test_root_mae"), float("nan"))
                        family_row["delta_vs_tree"] = (
                            float(tree_mae - value)
                            if math.isfinite(tree_mae) and math.isfinite(value)
                            else float("nan")
                        )
                    target_budget_equivalent_docs = (
                        float(train_doc_count)
                        * _safe_float(
                            package_doc_equiv.get("total_mass_per_doc"),
                            float("nan"),
                        )
                    )
                    realized_budget_equivalent_docs = (
                        float(train_doc_count)
                        * _safe_float(
                            tree_row.get("effective_full_doc_mass_per_doc_mean"),
                            float("nan"),
                        )
                    )
                    comparisons.append(
                        {
                            "scope_key": str(scope_key),
                            "scope_label": str(scope_label),
                            "train_doc_count": int(train_doc_count),
                            "package_name": str(package_name),
                            "report_row_key": str(
                                tree_row.get("report_row_key", "") or ""
                            ),
                            "source_summary_json": str(
                                tree_row.get("source_summary_json", "") or ""
                            ),
                            "pipeline_supervision_recovery_leaf_tokens": int(
                                _safe_int(
                                    tree_row.get(
                                        "pipeline_supervision_recovery_leaf_tokens"
                                    ),
                                    0,
                                )
                            ),
                            "supervision_recovery_geometry_key": str(geometry_key),
                            "supervision_recovery_geometry_label": str(
                                tree_row.get(
                                    "supervision_recovery_geometry_label",
                                    "",
                                )
                                or ""
                            ),
                            "package_label": str(package_spec.get("label", package_name)),
                            "package_semantics": str(
                                package_spec.get(
                                    "package_semantics",
                                    _default_supervision_recovery_package_semantics(
                                        str(package_name),
                                        package_spec,
                                    ),
                                )
                                or ""
                            ),
                            "doc_equiv_total_mass_per_doc": _safe_float(
                                package_doc_equiv.get("total_mass_per_doc"),
                                float("nan"),
                            ),
                            "doc_equiv_train_docs": float(train_doc_count)
                            * _safe_float(
                                package_doc_equiv.get("total_mass_per_doc"),
                                float("nan"),
                            ),
                            "target_budget_equivalent_docs": target_budget_equivalent_docs,
                            "doc_equiv_doc_review_mass_per_doc": _safe_float(
                                package_doc_equiv.get("doc_review_mass_per_doc"),
                                float("nan"),
                            ),
                            "doc_equiv_leaf_mass_per_doc": _safe_float(
                                package_doc_equiv.get("leaf_mass_per_doc"),
                                float("nan"),
                            ),
                            "doc_equiv_internal_mass_per_doc": _safe_float(
                                package_doc_equiv.get("internal_mass_per_doc"),
                                float("nan"),
                            ),
                            "doc_equiv_local_mass_per_doc": _safe_float(
                                package_doc_equiv.get("local_mass_per_doc"),
                                float("nan"),
                            ),
                            "doc_equiv_mass_target_per_doc": _safe_float(
                                package_doc_equiv.get("mass_target_per_doc"),
                                float("nan"),
                            ),
                            "tree_family": str(tree_family),
                            "comparison_mode": str(
                                tree_row.get("comparison_mode", "") or ""
                            ),
                            "comparison_semantics": str(
                                tree_row.get("comparison_semantics", "") or ""
                            ),
                            "comparison_semantics_label": str(
                                tree_row.get("comparison_semantics_label", "") or ""
                            ),
                            "run_intent_hash": str(
                                tree_row.get("run_intent_hash", "") or ""
                            ),
                            "run_intent_validation_status": str(
                                tree_row.get(
                                    "run_intent_validation_status",
                                    "",
                                )
                                or ""
                            ),
                            "contract_status": str(
                                tree_row.get("contract_status", "") or ""
                            ),
                            "tree_n_runs": int(_safe_int(tree_row.get("n_runs"), 0)),
                            "tree_test_root_mae": tree_mae,
                            "tree_effective_full_doc_mass_per_doc": _safe_float(
                                tree_row.get("effective_full_doc_mass_per_doc_mean"),
                                float("nan"),
                            ),
                            "realized_budget_equivalent_docs": realized_budget_equivalent_docs,
                            "tree_mass_target_per_doc": _safe_float(
                                tree_row.get("mass_target_per_doc"),
                                _safe_float(package_doc_equiv.get("mass_target_per_doc"), float("nan")),
                            ),
                            "tree_computed_doc_review_mass_per_doc": _safe_float(
                                tree_row.get("computed_doc_review_mass_per_doc"),
                                _safe_float(package_doc_equiv.get("doc_review_mass_per_doc"), float("nan")),
                            ),
                            "tree_computed_local_mass_per_doc": _safe_float(
                                tree_row.get("computed_local_mass_per_doc"),
                                _safe_float(package_doc_equiv.get("local_mass_per_doc"), float("nan")),
                            ),
                            "tree_computed_leaf_mass_per_doc": _safe_float(
                                tree_row.get("computed_leaf_mass_per_doc"),
                                _safe_float(package_doc_equiv.get("leaf_mass_per_doc"), float("nan")),
                            ),
                            "tree_computed_internal_mass_per_doc": _safe_float(
                                tree_row.get("computed_internal_mass_per_doc"),
                                _safe_float(package_doc_equiv.get("internal_mass_per_doc"), float("nan")),
                            ),
                            "tree_train_root_mae": _safe_float(
                                tree_row.get("train_root_mae_mean"),
                                float("nan"),
                            ),
                            "tree_val_root_mae": _safe_float(
                                tree_row.get("val_root_mae_mean"),
                                float("nan"),
                            ),
                            "tree_test_leaf_mae": _safe_float(tree_row.get("test_leaf_mae_mean"), float("nan")),
                            "tree_test_merge_mae": _safe_float(tree_row.get("test_merge_mae_mean"), float("nan")),
                            "tree_test_full_law_objective": _safe_float(
                                tree_row.get("test_unweighted_full_law_objective_mean"),
                                float("nan"),
                            ),
                            "tree_val_full_law_objective": _safe_float(
                                tree_row.get("val_unweighted_full_law_objective_mean"),
                                float("nan"),
                            ),
                            "tree_test_active_objective": _safe_float(
                                tree_row.get("test_unweighted_active_objective_mean"),
                                float("nan"),
                            ),
                            "tree_val_active_objective": _safe_float(
                                tree_row.get("val_unweighted_active_objective_mean"),
                                float("nan"),
                            ),
                            "tree_best_epoch": _safe_float(
                                tree_row.get("best_epoch_mean"),
                                float("nan"),
                            ),
                            "tree_selection_metric_name": str(
                                tree_row.get("selection_metric_name", "") or ""
                            ),
                            "tree_selection_metric_value": _safe_float(
                                tree_row.get("selection_metric_value_mean"),
                                float("nan"),
                            ),
                            "tree_checkpoint_metric": str(
                                tree_row.get("tree_checkpoint_metric", "") or ""
                            ),
                            "tree_stage1_checkpoint_metric": str(
                                tree_row.get("tree_stage1_checkpoint_metric", "") or ""
                            ),
                            "tree_reference_label": str(
                                tree_row.get("tree_reference_label", "") or ""
                            ),
                            "requested_fixed_leaf_tokens": int(
                                _safe_int(tree_row.get("requested_fixed_leaf_tokens"), 0)
                            ),
                            "executed_fixed_leaf_tokens": int(
                                _safe_int(tree_row.get("executed_fixed_leaf_tokens"), 0)
                            ),
                            "fixed_leaf_tokens": int(
                                _safe_int(tree_row.get("fixed_leaf_tokens"), 0)
                            ),
                            "computed_assumed_doc_tokens": int(
                                _safe_int(
                                    tree_row.get("computed_assumed_doc_tokens"),
                                    0,
                                )
                            ),
                            "leaves_per_doc": int(
                                _safe_int(tree_row.get("leaves_per_doc"), 0)
                            ),
                            "executed_leaves_per_doc": int(
                                _safe_int(tree_row.get("executed_leaves_per_doc"), 0)
                            ),
                            "internal_nodes_per_doc": int(
                                _safe_int(tree_row.get("internal_nodes_per_doc"), 0)
                            ),
                            "executed_internal_nodes_per_doc": int(
                                _safe_int(
                                    tree_row.get("executed_internal_nodes_per_doc"),
                                    0,
                                )
                            ),
                            "parity_mode": str(tree_row.get("parity_mode", "") or ""),
                            "is_exact_full_doc_parity_row": bool(
                                tree_row.get("is_exact_full_doc_parity_row")
                            ),
                            "is_authoritative_parity_row": bool(
                                tree_row.get("is_authoritative_parity_row")
                            ),
                            "is_fno_equivalent_geometry": bool(
                                tree_row.get("is_fno_equivalent_geometry")
                            ),
                            "tree_local_weighting_mode": str(
                                tree_row.get("tree_local_weighting_mode", "") or ""
                            ),
                            "tree_supervision_source": str(
                                tree_row.get("tree_supervision_source", "") or ""
                            ),
                            "local_estimand_mode": str(
                                tree_row.get("local_estimand_mode", "") or ""
                            ),
                            "depth_discount_gamma": _safe_float(
                                tree_row.get("depth_discount_gamma"),
                                1.0,
                            ),
                            "c2_pair_weighting_mode": str(
                                tree_row.get("c2_pair_weighting_mode", "") or ""
                            ),
                            "local_sampling_design_name": str(
                                tree_row.get("local_sampling_design_name", "") or ""
                            ),
                            "leaf_population_size": _safe_float(
                                tree_row.get("leaf_population_size"),
                                float("nan"),
                            ),
                            "leaf_sample_size": _safe_float(
                                tree_row.get("leaf_sample_size"),
                                float("nan"),
                            ),
                            "leaf_effective_propensity": _safe_float(
                                tree_row.get("leaf_effective_propensity"),
                                float("nan"),
                            ),
                            "merge_population_size": _safe_float(
                                tree_row.get("merge_population_size"),
                                float("nan"),
                            ),
                            "merge_sample_size": _safe_float(
                                tree_row.get("merge_sample_size"),
                                float("nan"),
                            ),
                            "merge_effective_propensity": _safe_float(
                                tree_row.get("merge_effective_propensity"),
                                float("nan"),
                            ),
                            "c2_same_pair_count": _safe_float(
                                tree_row.get("c2_same_pair_count"),
                                float("nan"),
                            ),
                            "c2_different_pair_count": _safe_float(
                                tree_row.get("c2_different_pair_count"),
                                float("nan"),
                            ),
                            "c2_pair_weight_ess": _safe_float(
                                tree_row.get("c2_pair_weight_ess"),
                                float("nan"),
                            ),
                            "c2_pair_weight_max": _safe_float(
                                tree_row.get("c2_pair_weight_max"),
                                float("nan"),
                            ),
                            "local_objective_audit": dict(
                                tree_row.get("local_objective_audit") or {}
                            ),
                            "fno_reference_package": str(fno_reference_package),
                            "fno_reference_family": str(fno_reference.get("baseline_family", "") or ""),
                            "fno_reference_n_runs": int(_safe_int(fno_reference.get("n_runs"), 0)),
                            "fno_reference_test_root_mae": full10_fno_mae,
                            "matched_fno_family_rows": matched_fno_family_rows,
                            "full100_fno_family": str(fno_ceiling.get("baseline_family", "") or ""),
                            "full100_fno_n_runs": int(_safe_int(fno_ceiling.get("n_runs"), 0)),
                            "full100_fno_test_root_mae": full100_fno_mae,
                            "full100_fno_family_rows": full100_fno_family_rows,
                            "best_full100_fno_family": str(
                                fno_ceiling.get("baseline_family", "") or ""
                            ),
                            "best_full100_fno_test_root_mae": full100_fno_mae,
                            "delta_vs_best_full100_fno": (
                                float(tree_mae - full100_fno_mae)
                                if math.isfinite(tree_mae) and math.isfinite(full100_fno_mae)
                                else float("nan")
                            ),
                            "canonical_official_fno_full100_family": "official_fno",
                            "canonical_official_fno_full100_n_runs": int(
                                _safe_int(canonical_official_fno_full100.get("n_runs"), 0)
                            ),
                            "canonical_official_fno_full100_test_root_mae": (
                                canonical_official_fno_full100_mae
                            ),
                            "delta_vs_full10_fno": (
                                float(tree_mae - full10_fno_mae)
                                if math.isfinite(tree_mae) and math.isfinite(full10_fno_mae)
                                else float("nan")
                            ),
                            "delta_vs_full100_fno_ceiling": (
                                float(tree_mae - full100_fno_mae)
                                if math.isfinite(tree_mae) and math.isfinite(full100_fno_mae)
                                else float("nan")
                            ),
                            "delta_vs_canonical_official_fno_full100": (
                                float(tree_mae - canonical_official_fno_full100_mae)
                                if math.isfinite(tree_mae)
                                and math.isfinite(canonical_official_fno_full100_mae)
                                else float("nan")
                            ),
                        }
                    )
            comparisons.sort(
                key=lambda row: (
                    expected_packages.index(str(row.get("package_name", ""))),
                    str(row.get("supervision_recovery_geometry_label", "") or ""),
                    int(_safe_int(row.get("executed_leaves_per_doc"), 0)),
                    int(_safe_int(row.get("executed_fixed_leaf_tokens"), 0)),
                )
            )
            canonical_curve_rows = [
                {
                    "train_doc_count": int(candidate_docs),
                    "test_root_mae_mean": _safe_float(
                        _family_row_at(
                            int(candidate_docs),
                            "full100",
                            "official_fno",
                            "",
                        ).get(
                            "test_root_mae_mean"
                        ),
                        float("nan"),
                    ),
                }
                for candidate_docs in available_train_docs
                if math.isfinite(
                    _safe_float(
                        _family_row_at(
                            int(candidate_docs),
                            "full100",
                            "official_fno",
                            "",
                        ).get("test_root_mae_mean"),
                        float("nan"),
                    )
                )
            ]
            canonical_curve_rows = [
                row
                for row in canonical_curve_rows
                if math.isfinite(_safe_float(row.get("test_root_mae_mean"), float("nan")))
            ]
            for row in comparisons:
                geometry_key = str(
                    row.get("supervision_recovery_geometry_key", "") or ""
                )
                if geometry_key:
                    geometry_curve_rows = [
                        {
                            "train_doc_count": int(candidate_docs),
                            "test_root_mae_mean": _safe_float(
                                _family_row_at(
                                    int(candidate_docs),
                                    "full100",
                                    "official_fno",
                                    str(geometry_key),
                                ).get("test_root_mae_mean"),
                                float("nan"),
                            ),
                        }
                        for candidate_docs in available_train_docs
                    ]
                    geometry_curve_rows = [
                        item
                        for item in geometry_curve_rows
                        if math.isfinite(
                            _safe_float(item.get("test_root_mae_mean"), float("nan"))
                        )
                    ]
                else:
                    geometry_curve_rows = list(canonical_curve_rows)
                interp = _interpolate_canonical_equivalent_train_docs(
                    geometry_curve_rows or canonical_curve_rows,
                    target_mae=_safe_float(row.get("tree_test_root_mae"), float("nan")),
                )
                row["canonical_official_fno_equivalent_train_docs"] = _safe_float(
                    interp.get("equivalent_train_docs"),
                    float("nan"),
                )
                row["canonical_official_fno_equivalent_train_docs_relation"] = str(
                    interp.get("relation", "") or ""
                )
                row["canonical_official_fno_equivalent_train_docs_min_train_docs"] = int(
                    _safe_int(interp.get("min_train_docs"), 0)
                )
                row["canonical_official_fno_equivalent_train_docs_max_train_docs"] = int(
                    _safe_int(interp.get("max_train_docs"), 0)
                )
            rows_by_train_docs.append(
                {
                    "train_doc_count": int(train_doc_count),
                    "rows": comparisons,
                }
            )
            finite_tree_rows = [
                dict(row)
                for row in comparisons
                if math.isfinite(_safe_float(row.get("tree_test_root_mae"), float("nan")))
            ]
            best_tree = (
                min(
                    finite_tree_rows,
                    key=lambda row: (
                        _safe_float(row.get("tree_test_root_mae"), float("inf")),
                        expected_packages.index(str(row.get("package_name", ""))),
                    ),
                )
                if finite_tree_rows
                else {}
            )
            best_tree_by_train_docs[str(train_doc_count)] = dict(best_tree)

        scopes[scope_key] = {
            "scope_key": str(scope_key),
            "scope_label": str(scope_label),
            "hardness_grid": str(scope_hardness.get(scope_key, "") or ""),
            "available_train_docs": available_train_docs,
            "rows_by_train_docs": rows_by_train_docs,
            "dense_anchor_rows": [
                dict(row)
                for payload in rows_by_train_docs
                for row in list(payload.get("rows") or [])
                if str(row.get("package_name", "")) == "full100"
            ],
            "best_tree_by_train_docs": best_tree_by_train_docs,
        }

    best_tree_summary = []
    for scope_key, scope_payload in sorted(scopes.items()):
        for train_doc_count, row in sorted(
            (scope_payload.get("best_tree_by_train_docs") or {}).items(),
            key=lambda item: int(_safe_int(item[0], 0)),
        ):
            if row:
                best_tree_summary.append(
                    {
                        "scope_key": str(scope_key),
                        "scope_label": str(scope_payload.get("scope_label", scope_key)),
                        "train_doc_count": int(_safe_int(train_doc_count)),
                        **dict(row),
                    }
                )

    primary_family_lookup = {
        (
            str(row.get("scope_key", "")),
            int(_safe_int(row.get("train_doc_count"), 0)),
            str(row.get("package_name", "")),
            str(row.get("baseline_family", "")),
            str(row.get("supervision_recovery_geometry_key", "") or ""),
        ): dict(row)
        for row in family_rows
    }
    exact_collapse_rows: List[Dict[str, Any]] = []
    for row in sorted(
        exact_collapse_family_rows,
        key=lambda item: (
            str(item.get("scope_key", "")),
            int(_safe_int(item.get("train_doc_count"), 0)),
            str(item.get("package_name", "")),
            str(item.get("comparison_arm", "")),
        ),
    ):
        scope_key = str(row.get("scope_key", ""))
        train_doc_count = int(_safe_int(row.get("train_doc_count"), 0))
        package_name = str(row.get("package_name", ""))
        geometry_key = str(
            row.get("supervision_recovery_geometry_key", "") or ""
        )
        official_fno_row = dict(
            primary_family_lookup.get(
                (
                    scope_key,
                    train_doc_count,
                    package_name,
                    "official_fno",
                    geometry_key,
                )
            )
            or {}
        )
        if not official_fno_row:
            exact_row_is_canonical_full_doc_geometry = bool(
                int(_safe_int(row.get("executed_leaves_per_doc"), 0)) == 1
                and int(_safe_int(row.get("executed_fixed_leaf_tokens"), 0))
                == int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
            )
            fno_candidates = [
                dict(candidate)
                for candidate in family_rows
                if str(candidate.get("scope_key", "")) == scope_key
                and int(_safe_int(candidate.get("train_doc_count"), 0))
                == train_doc_count
                and str(candidate.get("package_name", "")) == package_name
                and (
                    str(candidate.get("supervision_recovery_geometry_key", "") or "")
                    == geometry_key
                    or not geometry_key
                    or (
                        exact_row_is_canonical_full_doc_geometry
                        and int(
                            _safe_int(candidate.get("executed_leaves_per_doc"), 0)
                        )
                        == 1
                        and int(
                            _safe_int(
                                candidate.get("executed_fixed_leaf_tokens"),
                                0,
                            )
                        )
                        == int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
                    )
                )
                and str(candidate.get("baseline_family", ""))
                in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
            ]
            if fno_candidates:
                official_fno_row = min(
                    fno_candidates,
                    key=lambda candidate: _safe_float(
                        candidate.get("test_root_mae_mean"),
                        float("inf"),
                    ),
                )
        ordinary_tree_row = dict(
            primary_family_lookup.get(
                (
                    scope_key,
                    train_doc_count,
                    package_name,
                    str(tree_family),
                    geometry_key,
                )
            )
            or {}
        )
        if not ordinary_tree_row:
            ordinary_tree_candidates = [
                dict(candidate)
                for candidate in family_rows
                if str(candidate.get("scope_key", "")) == scope_key
                and int(_safe_int(candidate.get("train_doc_count"), 0))
                == train_doc_count
                and str(candidate.get("package_name", "")) == package_name
                and str(candidate.get("baseline_family", "")) == str(tree_family)
            ]
            if ordinary_tree_candidates:
                ordinary_tree_row = min(
                    ordinary_tree_candidates,
                    key=lambda candidate: (
                        str(candidate.get("comparison_arm", "") or "")
                        != SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM,
                        _safe_float(
                            candidate.get("test_root_mae_mean"),
                            float("inf"),
                        ),
                    ),
                )
        exact_test_root_mae = _safe_float(row.get("test_root_mae_mean"), float("nan"))
        official_fno_test_root_mae = _safe_float(
            official_fno_row.get("test_root_mae_mean"),
            float("nan"),
        )
        ordinary_tree_test_root_mae = _safe_float(
            ordinary_tree_row.get("test_root_mae_mean"),
            float("nan"),
        )
        exact_collapse_rows.append(
            {
                **dict(row),
                "claim_level": "exact_collapse_candidate",
                "official_fno_family": str(
                    official_fno_row.get("baseline_family", "") or ""
                ),
                "official_fno_test_root_mae": official_fno_test_root_mae,
                "delta_vs_official_fno": (
                    float(exact_test_root_mae - official_fno_test_root_mae)
                    if math.isfinite(exact_test_root_mae)
                    and math.isfinite(official_fno_test_root_mae)
                    else float("nan")
                ),
                "ordinary_tree_family": str(
                    ordinary_tree_row.get("baseline_family", "") or ""
                ),
                "ordinary_tree_test_root_mae": ordinary_tree_test_root_mae,
                "delta_vs_ordinary_tree": (
                    float(exact_test_root_mae - ordinary_tree_test_root_mae)
                    if math.isfinite(exact_test_root_mae)
                    and math.isfinite(ordinary_tree_test_root_mae)
                    else float("nan")
                ),
            }
        )

    scope_tree_references: Dict[str, Dict[str, Any]] = {}
    for row in family_rows:
        if _gs(row, "baseline_family") != str(tree_family):
            continue
        scope_key = _gs(row, "scope_key")
        if not scope_key or scope_key in scope_tree_references:
            continue
        scope_tree_references[scope_key] = {
            "scope_key": scope_key,
            "scope_label": str(
                row.get("scope_label")
                or scope_labels.get(scope_key)
                or _supervision_recovery_scope_label(
                    scope_key,
                    recoverable_scope_key=str(recoverable_benchmark),
                    structural_grid=str(structural_grid),
                )
            ),
            "tree_reference_mode": str(row.get("tree_reference_mode", "") or ""),
            "tree_reference_label": str(row.get("tree_reference_label", "") or ""),
            "tree_training_schedule": str(row.get("tree_training_schedule", "") or ""),
            "tree_checkpoint_metric": str(row.get("tree_checkpoint_metric", "") or ""),
            "tree_stage1_checkpoint_metric": str(
                row.get("tree_stage1_checkpoint_metric", "") or ""
            ),
            "summary_spec_name": str(row.get("summary_spec_name", "") or ""),
            "slot_count": int(_safe_int(row.get("slot_count"), 0)),
            "state_dim": int(_safe_int(row.get("state_dim"), 0)),
            "hidden_dim": int(_safe_int(row.get("hidden_dim"), 0)),
            "requested_fixed_leaf_tokens": int(
                _safe_int(row.get("requested_fixed_leaf_tokens"), 0)
            ),
            "executed_fixed_leaf_tokens": int(
                _safe_int(row.get("executed_fixed_leaf_tokens"), 0)
            ),
            "fixed_leaf_tokens": int(_safe_int(row.get("fixed_leaf_tokens"), 0)),
            "computed_assumed_doc_tokens": int(
                _safe_int(row.get("computed_assumed_doc_tokens"), 0)
            ),
            "executed_leaves_per_doc": int(
                _safe_int(row.get("executed_leaves_per_doc"), 0)
            ),
            "leaves_per_doc": int(_safe_int(row.get("leaves_per_doc"), 0)),
            "executed_internal_nodes_per_doc": int(
                _safe_int(row.get("executed_internal_nodes_per_doc"), 0)
            ),
            "internal_nodes_per_doc": int(
                _safe_int(row.get("internal_nodes_per_doc"), 0)
            ),
            "parity_mode": str(row.get("parity_mode", "") or ""),
            "is_exact_full_doc_parity_row": bool(
                row.get("is_exact_full_doc_parity_row")
            ),
            "is_fno_equivalent_geometry": bool(
                row.get("is_fno_equivalent_geometry")
            ),
        }
    scope_tree_reference_labels = sorted(
        {
            _gs(reference, "tree_reference_label")
            for reference in scope_tree_references.values()
            if _gs(reference, "tree_reference_label")
        }
    )
    tree_reference_labels = sorted(
        {
            _gs(row, "tree_reference_label")
            for row in family_rows
            if _gs(row, "baseline_family") == str(tree_family)
            and _gs(row, "tree_reference_label")
        }
    )
    tree_checkpoint_metrics = sorted(
        {
            _gs(row, "tree_checkpoint_metric")
            for row in family_rows
            if _gs(row, "baseline_family") == str(tree_family)
            and _gs(row, "tree_checkpoint_metric")
        }
    )
    common_tree_reference_label = (
        str(tree_reference_labels[0]) if len(tree_reference_labels) == 1 else ""
    )
    canonical_tree_reference_labels = {
        SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
        UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
        UNIFIED_G_FNO_PARITY_CANARY_PRESET,
    }
    comparator_alignment_status = "aligned" if len(tree_reference_labels) <= 1 else "mixed"
    comparator_alignment_warning = (
        ""
        if comparator_alignment_status == "aligned"
        else "tree references differ across supervision_recovery rows: "
        + ", ".join(tree_reference_labels)
    )
    comparator_selection_status = (
        "root_comparable"
        if tree_checkpoint_metrics == [SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC]
        else "mixed_selection"
    )
    comparator_selection_warning = (
        ""
        if comparator_selection_status == "root_comparable"
        else "tree ladder checkpoint metrics differ: " + ", ".join(tree_checkpoint_metrics)
    )
    canonical_tree_selection_metric = (
        SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC
        if common_tree_reference_label in canonical_tree_reference_labels
        else ""
    )
    canonical_tree_stage1_checkpoint_metric = (
        SUPERVISION_RECOVERY_CANONICAL_TREE_STAGE1_SELECTION_METRIC
        if common_tree_reference_label in canonical_tree_reference_labels
        else ""
    )
    canonical_comparison_rule = (
        SUPERVISION_RECOVERY_CANONICAL_COMPARISON_RULE
        if common_tree_reference_label in canonical_tree_reference_labels
        else ""
    )

    return {
        "status": "ready" if family_rows else "missing",
        "reason": "" if family_rows else "no supervision-recovery payload rows were aggregated",
        "contract_gate_status": (
            "fail"
            if (quarantined_family_rows or quarantined_exact_collapse_rows)
            else "pass"
        ),
        "quarantined_row_count": int(
            len(quarantined_family_rows) + len(quarantined_exact_collapse_rows)
        ),
        "quarantined_sources": quarantine_sources_from_rows(
            [*quarantined_family_rows, *quarantined_exact_collapse_rows]
        ),
        "tree_family": str(tree_family),
        "canonical_fno_families": list(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES),
        "train_doc_counts": sorted(observed_train_docs),
        "seeds": sorted(observed_seeds),
        "seed_count": len(observed_seeds),
        "package_order": list(expected_packages),
        "package_definitions": {
            key: dict(SUPERVISION_RECOVERY_PACKAGE_SPECS[key]) for key in expected_packages
        },
        "package_doc_equivalent": package_doc_equivalent,
        "scope_package_doc_equivalent": scope_package_doc_equivalent,
        "recoverable_scope_key": str(recoverable_benchmark),
        "recoverable_scope_label": str(recoverable_benchmark),
        "structural_hardness_grid": str(structural_grid),
        "structural_scope_key": str(structural_cell),
        "structural_scope_label": _supervision_recovery_scope_label(
            str(structural_cell),
            recoverable_scope_key=str(recoverable_benchmark),
            structural_grid=str(structural_grid),
        ),
        "scope_tree_references": scope_tree_references,
        "scope_tree_reference_labels": scope_tree_reference_labels,
        "tree_reference_labels": tree_reference_labels,
        "common_tree_reference_label": common_tree_reference_label,
        "comparator_alignment_status": comparator_alignment_status,
        "comparator_alignment_warning": comparator_alignment_warning,
        "tree_checkpoint_metrics": tree_checkpoint_metrics,
        "comparator_selection_status": comparator_selection_status,
        "comparator_selection_warning": comparator_selection_warning,
        "canonical_tree_selection_metric": canonical_tree_selection_metric,
        "canonical_tree_stage1_checkpoint_metric": canonical_tree_stage1_checkpoint_metric,
        "canonical_comparison_rule": canonical_comparison_rule,
        "theorem_state_diagnostic_metric_names": list(
            SUPERVISION_RECOVERY_THEOREM_STATE_DIAGNOSTICS
        ),
        "lean_alignment_contract": (
            "root_mae is scalar task performance; Markov Lean alignment also "
            "requires movement in learned state/merge diagnostics."
        ),
        "scopes": scopes,
        "best_tree_summary": best_tree_summary,
        "family_rows": family_rows,
        "quarantined_family_rows": quarantined_family_rows,
        "all_family_rows": all_family_rows,
        "exact_collapse_rows": exact_collapse_rows,
        "quarantined_exact_collapse_rows": quarantined_exact_collapse_rows,
        "runtime_diagnosis": _summarize_supervision_recovery_runtime_diagnosis(
            completed_runtime_rows,
            tree_family=str(tree_family),
        ),
    }


def _invoke_report(script: Path, argv: Sequence[str]) -> None:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, str(script), *list(argv)], cwd=REPO_ROOT, check=True, env=env)


def _run_logged_command(argv: Sequence[str], *, log_path: Path) -> None:
    env = _common_worker_env("")
    env["MPLBACKEND"] = "Agg"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(part) for part in argv) + "\n\n")
        handle.flush()
        subprocess.run(
            list(argv),
            cwd=REPO_ROOT,
            check=True,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _hydrate_existing_report_sources(output_root: Path, sources: Dict[str, str]) -> Dict[str, str]:
    candidates = {
        "batch_timing_summary": output_root / "batch_timing" / "markov_fixed_fused_leaflaws_batchsize_timing_fullpipeline.json",
        "medium_grid_summary": output_root / "medium_grid" / "aggregate_summary.json",
        "docs_epochs_summary": output_root / "docs_epochs" / "aggregate_summary.json",
        "learnability_summary": output_root / "learnability_report" / "learnability_summary.json",
        "weight_ablation_summary": output_root / "weight_ablation_runs" / "weight_ablation_summary.json",
        "law_comparison_json": output_root / "law_packages" / "fno_tree_law_comparison.json",
        "fno_upper_bound_summary": output_root / "full_doc_anchor" / "full_doc_fno_upper_bound_summary.json",
        "oracle_budget_frontier_summary": output_root / "oracle_budget_frontier" / "tree_oracle_budget_frontier_summary.json",
        "efficiency_suite_summary": output_root / "efficiency_suite" / "summary.json",
        "large_batch_diagnosis_summary": output_root / "large_batch_diagnosis" / "aggregate_summary.json",
        "supervision_sweep_summary": output_root / "supervision_sweep" / "supervision_sweep_summary.json",
        "support_summary": output_root / "support_grid" / "markov_local_support_detailed.summary.json",
        "supervision_recovery_summary": output_root / "supervision_recovery" / "summary.json",
    }
    hydrated = dict(sources)
    for key, path in candidates.items():
        if key not in hydrated and path.exists():
            hydrated[key] = str(path)
    return hydrated


def _build_batch_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    batch_sizes = _parse_int_list(args.batch_sizes, preset["batch_sizes"])
    phase_root = _phase_execution_root(root, "batch_timing")
    tasks = []
    for batch_size in batch_sizes:
        name = f"bs{batch_size:04d}"
        tasks.append(
            _profile_task(
                name=name,
                output_dir=phase_root / name,
                train_docs=1000,
                val_docs=128,
                epochs=1,
                batch_size=batch_size,
                seed=int(args.seed),
                lr=1e-3,
                exact_doc_limit=128,
                leaf_tokens=int(args.fixed_leaf_tokens),
                min_tokens=int(args.min_tokens),
                max_tokens=int(args.max_tokens),
                min_segments=int(args.min_segments),
                max_segments=int(args.max_segments),
                device_mode=str(args.device_mode),
            )
        )
    return tasks, phase_root


def _build_medium_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    batch_sizes = _parse_int_list(args.medium_batch_sizes, preset["medium_batch_sizes"])
    seeds = _parse_int_list(args.medium_seeds, preset["medium_seeds"])
    phase_root = _phase_execution_root(root, "medium_grid")
    tasks = []
    for batch_size in batch_sizes:
        for seed in seeds:
            name = f"bs{batch_size:04d}_seed{seed}"
            tasks.append(
                _profile_task(
                    name=name,
                    output_dir=phase_root / name,
                    train_docs=int(args.train_docs),
                    val_docs=int(args.medium_val_docs),
                    epochs=int(args.medium_epochs),
                    batch_size=batch_size,
                    seed=seed,
                    lr=1e-3,
                    exact_doc_limit=int(args.medium_exact_doc_limit),
                    leaf_tokens=int(args.fixed_leaf_tokens),
                    min_tokens=int(args.min_tokens),
                    max_tokens=int(args.max_tokens),
                    min_segments=int(args.min_segments),
                    max_segments=int(args.max_segments),
                    device_mode=str(args.device_mode),
                )
            )
    return tasks, phase_root


def _build_docs_epochs_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs_values = _parse_int_list(args.docs_epochs_train_docs, preset["docs_epochs_train_docs"])
    epoch_values = _parse_int_list(args.docs_epochs_epochs, preset["docs_epochs_epochs"])
    phase_root = _phase_execution_root(root, "docs_epochs")
    tasks = []
    for train_docs in train_docs_values:
        for epochs in epoch_values:
            name = f"train{train_docs:05d}_ep{epochs:02d}"
            tasks.append(
                _profile_task(
                    name=name,
                    output_dir=phase_root / name,
                    train_docs=train_docs,
                    val_docs=int(args.val_docs),
                    epochs=epochs,
                    batch_size=int(args.docs_epochs_batch_size),
                    seed=int(args.seed),
                    lr=1e-3,
                    exact_doc_limit=128,
                    leaf_tokens=int(args.fixed_leaf_tokens),
                    min_tokens=int(args.min_tokens),
                    max_tokens=int(args.max_tokens),
                    min_segments=int(args.min_segments),
                    max_segments=int(args.max_segments),
                    device_mode=str(args.device_mode),
                )
            )
    return tasks, phase_root


def _build_learnability_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs_values = _parse_int_list(args.learnability_train_docs, preset["learnability_train_docs"])
    llw_values = _parse_float_list(args.learnability_weights, preset["learnability_weights"])
    profiles = _parse_str_list(args.learnability_profiles, preset["learnability_profiles"])
    data_seeds = _parse_int_list(args.data_seeds, _parse_int_list(None, [0, 1]))
    tree_reference = _resolve_tree_reference(args)
    phase_root = _phase_execution_root(root, "learnability")
    tasks = []
    for train_docs in train_docs_values:
        for llw in llw_values:
            for profile in profiles:
                ratios = WEIGHT_PROFILE_SPECS[profile]
                for data_seed in data_seeds:
                    seed = int(args.seed) + int(data_seed)
                    name = f"train{train_docs:05d}_llw{llw:g}_{profile}_d{data_seed}"
                    config = _base_ops_config(
                        args,
                        seed=seed,
                        data_seed=data_seed,
                        train_docs=train_docs,
                        val_docs=min(int(args.val_docs), max(2, train_docs // 8)),
                        test_docs=min(int(args.test_docs), max(2, train_docs // 8)),
                        batch_size=int(args.law_batch_size),
                        n_epochs=int(args.law_epochs),
                    )
                    config.update(
                        {
                            "local_law_weight": float(llw),
                            "c1_relative_weight": float(ratios[0]),
                            "c2_relative_weight": float(ratios[1]),
                            "c3_relative_weight": float(ratios[2]),
                            "law_package": "" if profile not in {"tree_root_only"} else "root_only",
                        }
                    )
                    config = _apply_tree_reference_overrides(config, tree_reference)
                    tasks.append(
                        _direct_task(
                            root=phase_root,
                            name=name,
                            config=config,
                            output_filename=f"seed_{data_seed}.json",
                        )
                    )
    return tasks, phase_root


def _build_weight_ablation_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs_values = _parse_int_list(args.weight_ablation_train_docs, preset["weight_ablation_train_docs"])
    profiles = _parse_str_list(args.weight_ablation_profiles, preset["weight_ablation_profiles"])
    data_seeds = _parse_int_list(args.data_seeds, _parse_int_list(None, [0, 1]))
    tree_reference = _resolve_tree_reference(args)
    phase_root = _phase_execution_root(root, "weight_ablation")
    tasks = []
    for train_docs in train_docs_values:
        for profile in profiles:
            ratios = WEIGHT_PROFILE_SPECS[profile]
            llw = 0.0 if profile == "root_only" else 1.0
            for data_seed in data_seeds:
                seed = int(args.seed) + int(data_seed)
                name = f"train{train_docs:05d}_{profile}_d{data_seed}"
                config = _base_ops_config(
                    args,
                    seed=seed,
                    data_seed=data_seed,
                    train_docs=train_docs,
                    val_docs=min(int(args.val_docs), max(2, train_docs // 8)),
                    test_docs=min(int(args.test_docs), max(2, train_docs // 8)),
                    batch_size=int(args.law_batch_size),
                    n_epochs=int(args.law_epochs),
                )
                config.update(
                    {
                        "local_law_weight": float(llw),
                        "c1_relative_weight": float(ratios[0]),
                        "c2_relative_weight": float(ratios[1]),
                        "c3_relative_weight": float(ratios[2]),
                    }
                )
                config = _apply_tree_reference_overrides(config, tree_reference)
                tasks.append(_direct_task(root=phase_root, name=name, config=config))
    return tasks, phase_root


def _build_law_package_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    phase_root = _phase_execution_root(root, "law_packages")
    tasks = []
    train_docs, val_docs, test_docs = _law_phase_doc_counts(args)
    tree_reference = _resolve_tree_reference(args)
    law_set_ids = [
        canonical_law_set_id(value)
        for value in _parse_str_list(
            getattr(args, "law_set_ids", None),
            list(LAW_SET_CONFIGS.keys()),
        )
    ]
    for index, law_set_id in enumerate(law_set_ids):
        if law_set_id not in LAW_SET_CONFIGS:
            raise ValueError(
                f"unsupported law_set_id={law_set_id!r}; expected one of {sorted(LAW_SET_CONFIGS)}"
            )
        spec = dict(LAW_SET_CONFIGS[law_set_id])
        config = _base_ops_config(
            args,
            seed=int(args.seed) + index,
            data_seed=int(args.seed),
            train_docs=int(train_docs),
            val_docs=int(val_docs),
            test_docs=int(test_docs),
            batch_size=int(args.law_batch_size),
            n_epochs=int(args.law_epochs),
        )
        config.update(spec)
        config["law_set_id"] = law_set_id
        config["include_fno_baseline"] = True
        config["pipeline_law_set_id"] = law_set_id
        config = _apply_tree_reference_overrides(config, tree_reference)
        tasks.append(_direct_task(root=phase_root / "raw", name=law_set_id, config=config))
    return tasks, phase_root


def _build_full_doc_anchor_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs_values = _parse_int_list(
        args.full_doc_anchor_train_docs, preset["full_doc_anchor_train_docs"]
    )
    seeds = _parse_int_list(args.full_doc_anchor_seeds, preset["full_doc_anchor_seeds"])
    family_runs = _parse_run_axis_list(
        args.full_doc_anchor_reference_method_runs,
        DEFAULT_REFERENCE_METHOD_RUNS,
        role="reference",
    )
    families = _method_ids_from_run_axes(family_runs)
    template_benchmark = "recoverable_v4" if str(args.preset) == "standard" else "smoke"
    phase_root = _phase_execution_root(root, "full_doc_anchor")
    tasks: List[SubprocessTask] = []
    for train_docs in train_docs_values:
        for seed in seeds:
            for family in families:
                name = f"train{train_docs:05d}_{family}_seed{seed}"
                config = {
                    "n_regimes": 4,
                    "vocab_size": 32,
                    "generator_profile": "piecewise_markov",
                    "min_tokens": int(args.min_tokens),
                    "max_tokens": int(args.max_tokens),
                    "min_segments": int(args.min_segments),
                    "max_segments": int(args.max_segments),
                    "fixed_leaf_tokens": int(args.fixed_leaf_tokens),
                    "train_docs": int(train_docs),
                    "val_docs": int(args.val_docs),
                    "test_docs": int(args.test_docs),
                    "feature_mode": "token_full",
                    "doc_sequence_objective": "count_ce_only",
                    "use_cuda": bool(args.device_mode != "cpu"),
                    "cuda_device": 0 if args.device_mode != "cpu" else None,
                    "torch_threads": 1,
                    "seed": int(seed),
                    "data_seed": int(seed),
                    "model_seed": int(seed),
                }
                tasks.append(
                    _direct_task(
                        root=phase_root / "raw",
                        name=name,
                        config=config,
                        worker_kind="full_doc_upper_bound",
                        extra_payload={
                            "template_benchmark": template_benchmark,
                            "baseline_families": [str(family)],
                            "seed": int(seed),
                        },
                    )
                )
    return tasks, phase_root


def _efficiency_anchor_families(*, mode: str, structural: bool = False) -> List[str]:
    normalized_mode = str(mode or "both").strip().lower()
    families: List[str]
    if normalized_mode == "fno_only":
        families = list(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)
    elif normalized_mode == "tree_only":
        families = list(EFFICIENCY_TREE_BASELINE_FAMILIES)
    else:
        families = list(EFFICIENCY_DENSE_ANCHOR_FAMILIES)
    if structural and "palette_block_exact" not in families:
        families.append("palette_block_exact")
    return families


def _efficiency_diagnostic_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = {
        "n_regimes": 4,
        "vocab_size": 32,
        "generator_profile": "piecewise_markov",
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "min_segments": int(args.min_segments),
        "max_segments": int(args.max_segments),
        "fixed_leaf_tokens": int(args.fixed_leaf_tokens),
        "val_docs": int(args.val_docs),
        "test_docs": int(args.test_docs),
        "feature_mode": "token_full",
        "doc_sequence_objective": "count_ce_only",
        "use_cuda": bool(args.device_mode != "cpu"),
        "cuda_device": 0 if args.device_mode != "cpu" else None,
        "torch_threads": 1,
        "state_dim": int(args.state_dim),
        "hidden_dim": int(args.hidden_dim),
    }
    tree_reference = _resolve_tree_reference(args)
    return _apply_tree_reference_overrides(config, tree_reference)


def _existing_recoverable_full_doc_anchor_train_docs(
    output_root: Path,
    *,
    train_doc_counts: Sequence[int],
) -> set[int]:
    summary_path = output_root / "full_doc_anchor" / "full_doc_fno_upper_bound_summary.json"
    if not summary_path.exists():
        return set()
    try:
        payload = _read_json(summary_path)
    except Exception:
        return set()
    requested = {int(train_docs) for train_docs in train_doc_counts if int(train_docs) > 0}
    return {
        int(_safe_int((row or {}).get("train_docs")))
        for row in list(payload.get("rows") or [])
        if int(_safe_int((row or {}).get("train_docs"))) in requested
    }


def _existing_recoverable_anchor_payloads(
    output_root: Path,
    *,
    train_doc_counts: Sequence[int],
) -> List[Mapping[str, Any]]:
    summary_path = output_root / "full_doc_anchor" / "full_doc_fno_upper_bound_summary.json"
    if not summary_path.exists():
        return []
    try:
        payload = _read_json(summary_path)
    except Exception:
        return []
    requested = {int(train_docs) for train_docs in train_doc_counts if int(train_docs) > 0}
    aggregate_rows: List[Dict[str, Any]] = []
    for row in list(payload.get("rows") or []):
        train_docs = int(_safe_int((row or {}).get("train_docs")))
        if train_docs <= 0 or train_docs not in requested:
            continue
        aggregate_rows.append(
            {
                "benchmark": str(payload.get("benchmark", "recoverable_v4") or "recoverable_v4"),
                "cell_id": "recoverable_v4",
                "baseline_family": str(row.get("best_full_doc_fno_family", "")),
                "train_doc_count": train_docs,
                "test_root_mae_mean": _safe_float(row.get("best_full_doc_fno_test_root_mae"), float("nan")),
            }
        )
    if not aggregate_rows:
        return []
    return [
        {
            "benchmark": str(payload.get("benchmark", "recoverable_v4") or "recoverable_v4"),
            "aggregate_rows": aggregate_rows,
            "reuse_source": str(summary_path),
        }
    ]


def _build_efficiency_anchor_tasks(
    args: argparse.Namespace,
    root: Path,
) -> tuple[Dict[str, List[SubprocessTask]], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    phase_root = _phase_execution_root(root, "efficiency_suite")
    version_root = _version_root_for_phase_root(root, "efficiency_suite")
    dense_train_docs = _parse_int_list(
        getattr(args, "efficiency_anchor_train_docs_dense", None),
        preset["efficiency_anchor_train_docs_dense"],
    )
    seeds = _parse_int_list(
        getattr(args, "efficiency_anchor_seeds", None),
        preset["efficiency_anchor_seeds"],
    )
    anchor_mode = str(getattr(args, "efficiency_anchor_mode", preset["efficiency_anchor_mode"]))
    structural_cells = _parse_str_list(
        getattr(args, "efficiency_structural_cells", None),
        preset["efficiency_structural_cells"],
    )
    structural_train_docs = sorted(
        {
            int(train_docs)
            for train_docs in dense_train_docs
            if int(train_docs) >= 1024
        }
    ) or [1024]
    config = _efficiency_diagnostic_config(args)
    tasks: Dict[str, List[SubprocessTask]] = {
        "recoverable_dense_anchor": [],
        "structural_dense_anchor": [],
    }
    recoverable_families = _efficiency_anchor_families(mode=anchor_mode, structural=False)
    structural_families = _efficiency_anchor_families(mode=anchor_mode, structural=True)
    reusable_recoverable_fno_train_docs = _existing_recoverable_full_doc_anchor_train_docs(
        version_root,
        train_doc_counts=dense_train_docs,
    )
    if "full_doc_anchor" in _phase_set(getattr(args, "phases", "")):
        reusable_recoverable_fno_train_docs = reusable_recoverable_fno_train_docs.union(
            {
                int(train_docs)
                for train_docs in _parse_int_list(
                    getattr(args, "full_doc_anchor_train_docs", None),
                    preset["full_doc_anchor_train_docs"],
                )
            }
        )

    for train_docs in dense_train_docs:
        for family in recoverable_families:
            if (
                int(train_docs) in reusable_recoverable_fno_train_docs
                and str(family) in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
            ):
                continue
            for seed in seeds:
                name = f"recoverable__train{int(train_docs):05d}__{family}__seed{int(seed)}"
                tasks["recoverable_dense_anchor"].append(
                    _direct_task(
                        root=phase_root / "recoverable_dense_anchor" / "raw",
                        name=name,
                        config=config,
                        worker_kind="full_doc_diagnostics",
                        extra_payload={
                            "benchmark_name": "recoverable_v4",
                            "train_doc_counts": [int(train_docs)],
                            "baseline_families": [str(family)],
                            "seeds": [int(seed)],
                        },
                    )
                )

    for cell_id in structural_cells:
        for train_docs in structural_train_docs:
            for family in structural_families:
                for seed in seeds:
                    name = (
                        f"structural__{cell_id}__train{int(train_docs):05d}"
                        f"__{family}__seed{int(seed)}"
                    )
                    tasks["structural_dense_anchor"].append(
                        _direct_task(
                            root=phase_root / "structural_dense_anchor" / "raw",
                            name=name,
                            config=config,
                            worker_kind="full_doc_diagnostics",
                            extra_payload={
                                "benchmark_name": "recoverable_v4",
                                "hardness_grid": str(
                                    getattr(args, "efficiency_hardness_grid", preset["efficiency_hardness_grid"])
                                ),
                                "grid_cell_ids": [str(cell_id)],
                                "train_doc_counts": [int(train_docs)],
                                "baseline_families": [str(family)],
                                "seeds": [int(seed)],
                            },
                        )
                    )
    return tasks, phase_root


def _oracle_budget_frontier_command(
    args: argparse.Namespace,
    *,
    phase_root: Path,
    devices: Sequence[str],
    benchmark_name: str = "recoverable_v4",
    train_docs_override: int | None = None,
    tree_families_override: Sequence[str] | None = None,
    reference_families_override: Sequence[str] | None = None,
    hardness_grid: str = "",
    grid_cell_ids: Sequence[str] = (),
) -> List[str]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs = int(
        train_docs_override
        if train_docs_override is not None
        else (
            args.oracle_budget_train_docs
            if args.oracle_budget_train_docs is not None
            else preset["oracle_budget_train_docs"]
        )
    )
    seeds = _parse_int_list(args.oracle_budget_seeds, preset["oracle_budget_seeds"])
    tree_runs = (
        [_run_axis_from_token(value, role="primary") for value in tree_families_override]
        if tree_families_override is not None
        else _parse_run_axis_list(
            args.oracle_budget_method_runs,
            preset["oracle_budget_method_runs"],
            role="primary",
        )
    )
    reference_runs = (
        [_run_axis_from_token(value, role="reference") for value in reference_families_override]
        if reference_families_override is not None
        else _parse_run_axis_list(
            args.oracle_budget_reference_method_runs,
            preset["oracle_budget_reference_method_runs"],
            role="reference",
        )
    )
    tree_families = _legacy_families_from_run_axes(tree_runs)
    reference_families = _method_ids_from_run_axes(reference_runs)
    budget_calls = _parse_float_list(
        args.oracle_budget_calls_per_doc,
        preset["oracle_budget_calls_per_doc"],
    )
    full_doc_shares = _parse_float_list(
        args.oracle_budget_full_doc_shares,
        preset["oracle_budget_full_doc_shares"],
    )
    doc_modes = _parse_str_list(
        args.oracle_budget_doc_consumption_modes,
        preset["oracle_budget_doc_consumption_modes"],
    )
    local_split_modes = _parse_str_list(
        args.oracle_budget_local_split_modes,
        preset["oracle_budget_local_split_modes"],
    )
    cmd = [
        sys.executable,
        str(TREE_FULL_DOC_SCRIPT),
        "budget_frontier",
        "--output-root",
        str(phase_root),
        "--benchmark",
        str(benchmark_name),
        "--train-doc-count",
        str(train_docs),
        "--tree-families",
        *[str(value) for value in tree_families],
        "--reference-families",
        *[str(value) for value in reference_families],
        "--budget-calls-per-doc",
        *[str(value) for value in budget_calls],
        "--full-doc-budget-shares",
        *[str(value) for value in full_doc_shares],
        "--doc-consumption-modes",
        *[str(value) for value in doc_modes],
        "--local-split-modes",
        *[str(value) for value in local_split_modes],
        "--budget-tree-config-mode",
        str(args.oracle_budget_tree_config_mode),
        "--seeds",
        *[str(value) for value in seeds],
        "--use-cuda" if str(args.device_mode) != "cpu" else "--no-use-cuda",
        "--resume",
    ]
    if str(hardness_grid).strip():
        cmd.extend(["--hardness-grid", str(hardness_grid)])
    if grid_cell_ids:
        cmd.extend(["--grid-cell-ids", *[str(value) for value in grid_cell_ids]])
    oracle_capacity_root = (
        Path(args.oracle_budget_capacity_root).expanduser()
        if getattr(args, "oracle_budget_capacity_root", None) is not None
        else None
    )
    if oracle_capacity_root is None:
        tree_reference = _resolve_tree_reference(args)
        capacity_root_text = str(tree_reference.get("capacity_root", "")).strip()
        if str(tree_reference.get("mode", "")).strip().lower() == "capacity_locked" and capacity_root_text:
            oracle_capacity_root = Path(capacity_root_text)
    if oracle_capacity_root is not None:
        cmd.extend(["--capacity-root", str(oracle_capacity_root)])
    if devices:
        cmd.extend(["--mig-uuids", ",".join(str(device) for device in devices)])
    return cmd


def _efficiency_budget_specs(
    args: argparse.Namespace,
    *,
    phase_root: Path,
    devices: Sequence[str],
) -> List[Dict[str, Any]]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs_values = _parse_int_list(
        getattr(args, "efficiency_train_docs", None),
        preset["efficiency_train_docs"],
    )
    structural_cells = _parse_str_list(
        getattr(args, "efficiency_structural_cells", None),
        preset["efficiency_structural_cells"],
    )
    specs: List[Dict[str, Any]] = []
    for train_docs in train_docs_values:
        recoverable_root = phase_root / "recoverable_budget" / f"train{int(train_docs):05d}"
        specs.append(
            {
                "name": f"recoverable_budget__train{int(train_docs):05d}",
                "summary_path": recoverable_root / "tree_oracle_budget_frontier_summary.json",
                "command": _oracle_budget_frontier_command(
                    args,
                    phase_root=recoverable_root,
                    devices=devices,
                    benchmark_name="recoverable_v4",
                    train_docs_override=int(train_docs),
                    tree_families_override=EFFICIENCY_TREE_METHOD_RUNS,
                    reference_families_override=CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
                ),
                "log_path": recoverable_root / "oracle_budget_frontier.log",
            }
        )
        structural_root = phase_root / "structural_budget" / f"train{int(train_docs):05d}"
        specs.append(
            {
                "name": f"structural_budget__train{int(train_docs):05d}",
                "summary_path": structural_root / "tree_oracle_budget_frontier_summary.json",
                "command": _oracle_budget_frontier_command(
                    args,
                    phase_root=structural_root,
                    devices=devices,
                    benchmark_name="recoverable_v4",
                    train_docs_override=int(train_docs),
                    tree_families_override=EFFICIENCY_TREE_METHOD_RUNS,
                    reference_families_override=CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
                    hardness_grid=str(
                        getattr(args, "efficiency_hardness_grid", preset["efficiency_hardness_grid"])
                    ),
                    grid_cell_ids=tuple(structural_cells),
                ),
                "log_path": structural_root / "oracle_budget_frontier.log",
            }
        )
    return specs


def _existing_recoverable_budget_payloads(
    output_root: Path,
    *,
    train_doc_counts: Sequence[int],
) -> tuple[List[Mapping[str, Any]], set[int]]:
    summary_path = output_root / "oracle_budget_frontier" / "tree_oracle_budget_frontier_summary.json"
    if not summary_path.exists():
        return [], set()
    try:
        payload = _read_json(summary_path)
    except Exception:
        return [], set()
    requested = {int(train_docs) for train_docs in train_doc_counts if int(train_docs) > 0}
    tree_rows = [
        dict(row)
        for row in list(payload.get("tree_rows") or [])
        if int(_safe_int((row or {}).get("train_doc_count"))) in requested
    ]
    reference_rows = [
        dict(row)
        for row in list(payload.get("reference_rows") or [])
        if int(_safe_int((row or {}).get("train_doc_count"))) in requested
    ]
    reusable_train_docs = {
        int(_safe_int(row.get("train_doc_count")))
        for row in tree_rows + reference_rows
        if int(_safe_int(row.get("train_doc_count"))) > 0
    }
    if not reusable_train_docs:
        return [], set()
    best_rows = [
        dict(row)
        for row in list(payload.get("best_tree_by_budget") or [])
        if any(
            int(_safe_int(candidate.get("train_doc_count"))) in reusable_train_docs
            and str(candidate.get("baseline_family", "")) == str(row.get("baseline_family", ""))
            and float(_safe_float(candidate.get("budget_total_calls_per_doc"))) == float(_safe_float(row.get("budget_total_calls_per_doc")))
            and float(_safe_float(candidate.get("full_doc_budget_share"))) == float(_safe_float(row.get("full_doc_budget_share")))
            and str(candidate.get("doc_consumption_mode", "")) == str(row.get("doc_consumption_mode", ""))
            and str(candidate.get("local_split_mode", "")) == str(row.get("local_split_mode", ""))
            for candidate in tree_rows
        )
    ]
    return (
        [
            {
                "benchmark": str(payload.get("benchmark", "recoverable_v4") or "recoverable_v4"),
                "tree_rows": tree_rows,
                "reference_rows": reference_rows,
                "best_tree_by_budget": best_rows,
                "reuse_source": str(summary_path),
            }
        ],
        reusable_train_docs,
    )


def _build_large_batch_diagnosis_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    batch_sizes = _parse_int_list(
        args.large_batch_batch_sizes, preset["large_batch_batch_sizes"]
    )
    fixed_epochs = int(args.large_batch_fixed_epochs)
    target_steps = int(args.large_batch_target_steps)
    lrs = _parse_float_list(args.large_batch_lrs, [1e-3, 2e-3, 4e-3])
    phase_root = _phase_execution_root(root, "large_batch_diagnosis")
    tasks: List[SubprocessTask] = []
    train_docs = int(args.train_docs)
    for batch_size in batch_sizes:
        tasks.append(
            _profile_task(
                name=f"fixed_epoch__bs{batch_size:04d}__ep{fixed_epochs:02d}__lr{1e-3:.4f}",
                output_dir=phase_root / f"fixed_epoch_bs{batch_size:04d}",
                train_docs=train_docs,
                val_docs=int(args.medium_val_docs),
                epochs=fixed_epochs,
                batch_size=batch_size,
                seed=int(args.seed),
                lr=1e-3,
                exact_doc_limit=int(args.medium_exact_doc_limit),
                leaf_tokens=int(args.fixed_leaf_tokens),
                min_tokens=int(args.min_tokens),
                max_tokens=int(args.max_tokens),
                min_segments=int(args.min_segments),
                max_segments=int(args.max_segments),
                device_mode=str(args.device_mode),
            )
        )
        steps_per_epoch = _ceil_div(train_docs, batch_size)
        constant_epochs = max(1, _ceil_div(target_steps, max(1, steps_per_epoch)))
        tasks.append(
            _profile_task(
                name=f"constant_steps__bs{batch_size:04d}__ep{constant_epochs:02d}__lr{1e-3:.4f}",
                output_dir=phase_root / f"constant_steps_bs{batch_size:04d}",
                train_docs=train_docs,
                val_docs=int(args.medium_val_docs),
                epochs=constant_epochs,
                batch_size=batch_size,
                seed=int(args.seed),
                lr=1e-3,
                exact_doc_limit=int(args.medium_exact_doc_limit),
                leaf_tokens=int(args.fixed_leaf_tokens),
                min_tokens=int(args.min_tokens),
                max_tokens=int(args.max_tokens),
                min_segments=int(args.min_segments),
                max_segments=int(args.max_segments),
                device_mode=str(args.device_mode),
            )
        )
    for lr in lrs:
        tasks.append(
            _profile_task(
                name=f"retune_1024__bs1024__ep20__lr{lr:.4f}",
                output_dir=phase_root / f"retune_1024_lr{lr:.4f}",
                train_docs=train_docs,
                val_docs=int(args.medium_val_docs),
                epochs=20,
                batch_size=1024,
                seed=int(args.seed),
                lr=lr,
                exact_doc_limit=int(args.medium_exact_doc_limit),
                leaf_tokens=int(args.fixed_leaf_tokens),
                min_tokens=int(args.min_tokens),
                max_tokens=int(args.max_tokens),
                min_segments=int(args.min_segments),
                max_segments=int(args.max_segments),
                device_mode=str(args.device_mode),
            )
        )
    return tasks, phase_root


def _build_supervision_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    train_docs_values = _parse_int_list(args.supervision_train_docs, preset["supervision_train_docs"])
    leaf_profiles = _parse_str_list(args.supervision_leaf_profiles, preset["supervision_leaf_profiles"])
    internal_profiles = _parse_str_list(
        args.supervision_internal_profiles,
        preset["supervision_internal_profiles"],
    )
    supervision_seeds = _parse_int_list(args.supervision_seeds, preset["supervision_seeds"])
    unknown_leaf = [name for name in leaf_profiles if name not in SUPERVISION_LEAF_PROFILES]
    if unknown_leaf:
        raise ValueError(
            f"unknown supervision leaf profiles: {', '.join(sorted(unknown_leaf))}; "
            f"valid options are {', '.join(SUPERVISION_LEAF_PROFILE_ORDER)}"
        )
    unknown_internal = [name for name in internal_profiles if name not in SUPERVISION_INTERNAL_PROFILES]
    if unknown_internal:
        raise ValueError(
            f"unknown supervision internal profiles: {', '.join(sorted(unknown_internal))}; "
            f"valid options are {', '.join(SUPERVISION_INTERNAL_PROFILE_ORDER)}"
        )
    tree_reference = _resolve_tree_reference(args)
    phase_root = _phase_execution_root(root, "supervision_sweep")
    tasks: List[SubprocessTask] = []
    for train_docs in train_docs_values:
        for leaf_profile in leaf_profiles:
            leaf_spec = dict(SUPERVISION_LEAF_PROFILES[leaf_profile])
            for internal_profile in internal_profiles:
                internal_spec = dict(SUPERVISION_INTERNAL_PROFILES[internal_profile])
                for data_seed in supervision_seeds:
                    seed = int(args.seed) + int(data_seed)
                    name = (
                        f"train{int(train_docs):05d}_leaf{leaf_profile}"
                        f"_internal{internal_profile}_d{int(data_seed)}"
                    )
                    config = _base_ops_config(
                        args,
                        seed=seed,
                        data_seed=data_seed,
                        train_docs=int(train_docs),
                        val_docs=min(int(args.val_docs), max(2, int(train_docs) // 8)),
                        test_docs=min(int(args.test_docs), max(2, int(train_docs) // 8)),
                        batch_size=int(args.supervision_batch_size),
                        n_epochs=int(args.supervision_epochs),
                        fixed_leaf_tokens=int(args.supervision_fixed_leaf_tokens),
                    )
                    config.update(
                        {
                            "min_tokens": int(args.supervision_min_tokens),
                            "max_tokens": int(args.supervision_max_tokens),
                            "min_segments": int(args.supervision_min_segments),
                            "max_segments": int(args.supervision_max_segments),
                            "law_package": "all_laws",
                            "local_law_weight": 0.5,
                            "c1_relative_weight": 1.0,
                            "c2_relative_weight": 1.0,
                            "c3_relative_weight": 1.0,
                            **leaf_spec,
                            **internal_spec,
                            "pipeline_supervision_leaf_profile": str(leaf_profile),
                            "pipeline_supervision_internal_profile": str(internal_profile),
                        }
                    )
                    config = _apply_tree_reference_overrides(
                        config,
                        tree_reference,
                        preserve_keys=_supervision_tree_reference_preserve_keys(args),
                    )
                    tasks.append(_direct_task(root=phase_root / "raw", name=name, config=config))
    return tasks, phase_root


def _build_supervision_recovery_phase(
    args: argparse.Namespace,
    root: Path,
    *,
    package_order: Sequence[str] | None = None,
) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    recoverable_benchmark = _supervision_recovery_recoverable_benchmark_name(args)
    structural_grid = _supervision_recovery_structural_grid_name(args)
    train_docs_values = _parse_int_list(
        getattr(args, "supervision_recovery_train_docs", None),
        preset["supervision_recovery_train_docs"],
    )
    recovery_seeds = _parse_int_list(
        getattr(args, "supervision_recovery_seeds", None),
        preset["supervision_recovery_seeds"],
    )
    depth_discount_gammas = _parse_float_list(
        getattr(args, "supervision_recovery_depth_discount_gammas", None),
        preset["supervision_recovery_depth_discount_gammas"],
    )
    include_gamma_tag = bool(
        len(depth_discount_gammas) > 1
        or any(
            not math.isclose(
                _safe_float(gamma, 1.0),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for gamma in depth_discount_gammas
        )
    )
    tree_family = str(
        getattr(
            args,
            "supervision_recovery_method_id",
            preset["supervision_recovery_method_id"],
        )
        or preset["supervision_recovery_method_id"]
    ).strip()
    if tree_family not in {"tree_neural"}:
        raise ValueError(
            "supervision_recovery_method_id must be 'tree_neural'"
        )
    if not recoverable_benchmark:
        raise ValueError("supervision_recovery_recoverable_benchmark must be non-empty")
    if not structural_grid:
        raise ValueError("supervision_recovery_structural_grid must be non-empty")
    try:
        from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
            resolve_full_doc_diagnostic_grid,
        )

        valid_structural_cells = {
            str(benchmark.cell_id or "").strip()
            for benchmark in resolve_full_doc_diagnostic_grid(structural_grid)
            if str(benchmark.cell_id or "").strip()
        }
    except Exception as exc:
        raise ValueError(
            f"unknown supervision recovery structural grid: {structural_grid!r}"
        ) from exc
    from src.ctreepo.sim.core.markov_hazard_panels import (
        canonicalize_structural_v2_cell_id,
    )

    structural_cell = (
        canonicalize_structural_v2_cell_id(
            _supervision_recovery_structural_cell_name(args)
        )
        or _supervision_recovery_structural_cell_name(args)
    )
    if structural_cell not in valid_structural_cells:
        raise ValueError(
            f"unknown supervision recovery structural cell: {structural_cell!r}; "
            f"valid options are {', '.join(sorted(valid_structural_cells))}"
        )
    _validate_supervision_recovery_tree_setup(args)
    resolved_package_order = (
        _resolve_supervision_recovery_package_order(package_order)
        if package_order is not None
        else _resolved_supervision_recovery_package_order(args)
    )
    explicit_leaf_tokens = _resolved_supervision_recovery_leaf_token_ladder(args)
    package_leaf_token_overrides = _resolved_supervision_recovery_package_leaf_token_overrides(
        args
    )
    default_leaf_tokens_values = explicit_leaf_tokens or [
        int(args.supervision_fixed_leaf_tokens)
    ]
    leafgrid_active = bool(explicit_leaf_tokens or package_leaf_token_overrides)
    # Optional per-leaf-size override: use a different tree reference for the
    # 1-leaf geometry (e.g. canary CE preset) when configured via
    # [tradeoff_pipeline.one_leaf_tree_reference] in the TOML.
    one_leaf_tree_reference = _resolve_tree_reference(
        args, prefix="one_leaf_tree_reference", fallback=None,
    )
    recoverable_one_leaf_root_only_reference = _tree_reference_from_preset_name(
        ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
    )
    structural_one_leaf_root_only_reference = _tree_reference_from_preset_name(
        STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET
    )
    _one_leaf_ref_active = bool(
        one_leaf_tree_reference
        and one_leaf_tree_reference.get("config")
    )
    _assumed_doc_tokens = (
        int(args.supervision_max_tokens)
        if int(args.supervision_min_tokens) == int(args.supervision_max_tokens)
        else 0
    )
    phase_root = _phase_execution_root(root, "supervision_recovery")
    scope_specs = _supervision_recovery_scope_specs(
        recoverable_benchmark=str(recoverable_benchmark),
        structural_grid=str(structural_grid),
        structural_cell=str(structural_cell),
        requested_scope_keys=_supervision_recovery_scope_keys(args),
        hazard_panel_ids=_parse_str_list(
            getattr(args, "supervision_recovery_hazard_panel_ids", None),
            (),
        ),
        hazard_panel_bundle_map=_parse_key_value_text_map(
            getattr(args, "supervision_recovery_hazard_panel_bundle_map", None)
        ),
    )
    tasks: List[SubprocessTask] = []
    emitted_fno_tasks: Dict[
        tuple[str, int, int, float, str],
        tuple[Dict[str, Any], Dict[str, Any]],
    ] = {}
    for scope in scope_specs:
        scope_kind = str(scope.get("scope_kind", "") or "").strip() or (
            "recoverable"
            if not str(scope["hardness_grid"]).strip()
            else "structural"
        )
        for train_docs in train_docs_values:
            for data_seed in recovery_seeds:
                seed = int(args.seed) + int(data_seed)
                for depth_discount_gamma in depth_discount_gammas:
                    gamma_tag = (
                        f"__g{float(depth_discount_gamma):0.2f}".replace(".", "p")
                        if include_gamma_tag
                        else ""
                    )
                    for package_name in resolved_package_order:
                        package_leaf_tokens_values = list(
                            package_leaf_token_overrides.get(
                                str(package_name),
                                default_leaf_tokens_values,
                            )
                            or default_leaf_tokens_values
                        )
                        for fixed_leaf_tokens in package_leaf_tokens_values:
                            base_config = _base_ops_config(
                                args,
                                seed=seed,
                                data_seed=data_seed,
                                train_docs=int(train_docs),
                                val_docs=min(int(args.val_docs), max(2, int(train_docs) // 8)),
                                test_docs=min(int(args.test_docs), max(2, int(train_docs) // 8)),
                                batch_size=_resolve_supervision_batch_size_for_leaf_tokens(
                                    args, int(fixed_leaf_tokens)
                                ),
                                n_epochs=int(args.supervision_epochs),
                                fixed_leaf_tokens=int(fixed_leaf_tokens),
                            )
                            base_config.update(
                                {
                                    "min_tokens": int(args.supervision_min_tokens),
                                    "max_tokens": int(args.supervision_max_tokens),
                                    "min_segments": int(args.supervision_min_segments),
                                    "max_segments": int(args.supervision_max_segments),
                                    "law_package": "all_laws",
                                    "local_law_weight": 0.5,
                                    "c1_relative_weight": 1.0,
                                    "c2_relative_weight": 1.0,
                                    "c3_relative_weight": 1.0,
                                    "depth_discount_gamma": float(depth_discount_gamma),
                                    "pipeline_supervision_recovery_depth_discount_gamma": float(
                                        depth_discount_gamma
                                    ),
                                }
                            )
                            scope_ops_overrides = {
                                str(key): value
                                for key, value in dict(
                                    scope.get("ops_config_overrides") or {}
                                ).items()
                                if str(key) in _ops_count_supported_config_keys()
                            }
                            if scope_ops_overrides:
                                base_config.update(scope_ops_overrides)
                            if str(scope.get("base_bundle_path", "") or "").strip():
                                base_config["pipeline_base_bundle_path"] = str(
                                    scope.get("base_bundle_path", "")
                                )
                            if str(scope.get("hazard_panel_id", "") or "").strip():
                                base_config["pipeline_hazard_panel_id"] = str(
                                    scope.get("hazard_panel_id", "")
                                )
                            if leafgrid_active:
                                base_config["pipeline_supervision_recovery_leaf_tokens"] = int(
                                    fixed_leaf_tokens
                                )
                            leaf_tag = (
                                f"__leaf{int(fixed_leaf_tokens):03d}"
                                if leafgrid_active
                                else ""
                            )
                            package_spec = dict(
                                SUPERVISION_RECOVERY_PACKAGE_SPECS[package_name]
                            )
                            tree_reference = _resolve_tree_reference(
                                args,
                                package_name=str(package_name),
                            )
                            structural_tree_reference = _resolve_tree_reference(
                                args,
                                prefix="structural_tree_reference",
                                package_name=str(package_name),
                                fallback=tree_reference,
                            )
                            _is_one_leaf = (
                                _one_leaf_ref_active
                                and _assumed_doc_tokens > 0
                                and int(fixed_leaf_tokens) >= _assumed_doc_tokens
                            )
                            _is_root_only_one_leaf = (
                                _is_one_leaf
                                and str(package_spec.get("doc_consumption_mode", "") or "")
                                == "root_only"
                                and not _one_leaf_package_has_local_supervision(
                                    package_spec
                                )
                            )
                            _is_exact_collapse_one_leaf = bool(
                                _is_root_only_one_leaf
                            )
                            scope_tree_reference = (
                                one_leaf_tree_reference
                                if _is_exact_collapse_one_leaf
                                else (
                                    recoverable_one_leaf_root_only_reference
                                    if _is_root_only_one_leaf
                                    and scope_kind == "recoverable"
                                    else (
                                        structural_one_leaf_root_only_reference
                                        if _is_root_only_one_leaf
                                        else (
                                            tree_reference
                                            if scope_kind == "recoverable"
                                            else structural_tree_reference
                                        )
                                    )
                                )
                            )
                            scope_tree_reference_label = _tree_reference_label(
                                scope_tree_reference
                            )
                            scope_tree_reference_recipe = _tree_reference_recipe(
                                scope_tree_reference
                            )
                            exact_parity_requested = bool(
                                _is_exact_collapse_one_leaf
                                or scope_tree_reference_label
                                == UNIFIED_G_FNO_PARITY_CANARY_PRESET
                                or scope_tree_reference_recipe
                                == UNIFIED_G_FNO_PARITY_CANARY_PRESET
                            )
                            config = dict(base_config)
                            config.update(
                                {
                                    "comparison_mode": "comparable",
                                    "budget_total_calls": 0,
                                    "budget_total_calls_per_doc": float(
                                        package_spec["budget_total_calls_per_doc"]
                                    ),
                                    "full_doc_budget_share": float(
                                        package_spec["full_doc_budget_share"]
                                    ),
                                    "doc_consumption_mode": str(
                                        package_spec["doc_consumption_mode"]
                                    ),
                                    "local_split_mode": str(
                                        package_spec["local_split_mode"]
                                    ),
                                    "leaf_supervision_kind": str(
                                        package_spec["leaf_supervision_kind"]
                                    ),
                                    "leaf_label_rate": float(
                                        package_spec["leaf_label_rate"]
                                    ),
                                    "internal_supervision_kind": str(
                                        package_spec["internal_supervision_kind"]
                                    ),
                                    "internal_label_rate": float(
                                        package_spec["internal_label_rate"]
                                    ),
                                    "max_internal_depth": int(
                                        package_spec.get("max_internal_depth", 0)
                                    ),
                                    "pipeline_supervision_recovery_package": str(
                                        package_name
                                    ),
                                    "pipeline_supervision_recovery_recoverable_benchmark": str(
                                        recoverable_benchmark
                                    ),
                                    "pipeline_supervision_recovery_structural_grid": str(
                                        structural_grid
                                    ),
                                    "pipeline_supervision_recovery_scope": str(
                                        scope["scope_key"]
                                    ),
                                    "pipeline_supervision_recovery_scope_label": str(
                                        scope["scope_label"]
                                    ),
                                    "pipeline_benchmark_name": str(scope["benchmark_name"]),
                                    "pipeline_hardness_grid": str(
                                        scope["hardness_grid"]
                                    ),
                                    "pipeline_grid_cell_ids": list(scope["grid_cell_ids"]),
                                    "pipeline_supervision_recovery_exact_full_doc_parity_requested": bool(
                                        exact_parity_requested
                                    ),
                                }
                            )
                            fno_compare_config: Dict[str, Any] | None = None
                            fno_compare_payload: Dict[str, Any] | None = None
                            if bool(package_spec.get("run_fno")):
                                fno_task_key = (
                                    str(scope["scope_key"]),
                                    int(train_docs),
                                    int(data_seed),
                                    float(depth_discount_gamma),
                                    str(package_name),
                                )
                                if fno_task_key not in emitted_fno_tasks:
                                    fno_is_one_leaf = bool(
                                        _one_leaf_ref_active
                                        and _assumed_doc_tokens > 0
                                        and int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
                                        >= _assumed_doc_tokens
                                    )
                                    fno_is_root_only_one_leaf = bool(
                                        fno_is_one_leaf
                                        and str(
                                            package_spec.get("doc_consumption_mode", "") or ""
                                        )
                                        == "root_only"
                                        and not _one_leaf_package_has_local_supervision(
                                            package_spec
                                        )
                                    )
                                    fno_is_exact_collapse_one_leaf = bool(
                                        fno_is_root_only_one_leaf
                                    )
                                    fno_scope_tree_reference = (
                                        one_leaf_tree_reference
                                        if fno_is_exact_collapse_one_leaf
                                        else (
                                            recoverable_one_leaf_root_only_reference
                                            if fno_is_root_only_one_leaf
                                            and scope_kind == "recoverable"
                                            else (
                                                structural_one_leaf_root_only_reference
                                                if fno_is_root_only_one_leaf
                                                else (
                                                    tree_reference
                                                    if scope_kind == "recoverable"
                                                    else structural_tree_reference
                                                )
                                            )
                                        )
                                    )
                                    fno_config = dict(config)
                                    fno_config["fixed_leaf_tokens"] = int(
                                        FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS
                                    )
                                    fno_config[
                                        "pipeline_supervision_recovery_leaf_tokens"
                                    ] = int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
                                    fno_config["preserve_requested_leaf_tokens"] = bool(
                                        leafgrid_active
                                    )
                                    fno_config[
                                        "official_fno_preserve_requested_leaf_tokens"
                                    ] = bool(leafgrid_active)
                                    _tree_ref_cfg = (
                                        fno_scope_tree_reference.get("config") or {}
                                    )
                                    from src.ctreepo.sim.core.fno_arch_config import (
                                        resolve_fno_arch_from_mapping,
                                    )

                                    _ref_fno = resolve_fno_arch_from_mapping(_tree_ref_cfg)
                                    fno_config["tree_leaf_fno_width"] = _ref_fno.width
                                    fno_config["tree_leaf_fno_n_modes"] = _ref_fno.n_modes
                                    fno_config["tree_leaf_fno_n_layers"] = _ref_fno.n_layers
                                    _tv_rsk = _tree_ref_cfg.get("tree_root_supervision_kind")
                                    if _tv_rsk:
                                        fno_config["tree_root_supervision_kind"] = str(_tv_rsk)
                                    for _hp_key in (
                                        "state_dim",
                                        "hidden_dim",
                                        "batch_size",
                                        "lr",
                                        "weight_decay",
                                    ):
                                        _hp_val = _tree_ref_cfg.get(_hp_key)
                                        if _hp_val is not None:
                                            fno_config[_hp_key] = _hp_val
                                    fno_leaf_tag = (
                                        f"__leaf{int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS):03d}"
                                        if leafgrid_active
                                        else ""
                                    )
                                    fno_name = (
                                        f"{scope['scope_key']}__train{int(train_docs):05d}"
                                        f"__{package_name}{fno_leaf_tag}{gamma_tag}__fno__d{int(data_seed)}"
                                    )
                                    fno_compare_config = dict(fno_config)
                                    fno_compare_payload = {
                                        "benchmark_name": str(scope["benchmark_name"]),
                                        "hardness_grid": str(scope["hardness_grid"]),
                                        "grid_cell_ids": list(scope["grid_cell_ids"]),
                                        "base_bundle_path": str(
                                            scope.get("base_bundle_path", "") or ""
                                        ),
                                        "hazard_panel_id": str(
                                            scope.get("hazard_panel_id", "") or ""
                                        ),
                                        "train_doc_counts": [int(train_docs)],
                                        "baseline_families": list(
                                            CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
                                        ),
                                        "seeds": [int(seed)],
                                    }
                                    emitted_fno_tasks[fno_task_key] = (
                                        fno_compare_config,
                                        fno_compare_payload,
                                    )
                                    tasks.append(
                                        _direct_task(
                                            root=phase_root / "raw",
                                            name=fno_name,
                                            config=fno_config,
                                            worker_kind="full_doc_diagnostics",
                                            extra_payload=fno_compare_payload,
                                        )
                                    )
                                else:
                                    (
                                        fno_compare_config,
                                        fno_compare_payload,
                                    ) = emitted_fno_tasks[fno_task_key]
                            tree_config, resolved_package_spec, _ = (
                                _build_supervision_recovery_scope_config(
                                    args,
                                    base_config=config,
                                    package_name=str(package_name),
                                    package_spec=package_spec,
                                    scope_key=str(scope["scope_key"]),
                                    scope_label=str(scope["scope_label"]),
                                    tree_reference=scope_tree_reference,
                                    preserve_schedule=(scope_kind == "recoverable"),
                                    preserve_fixed_leaf_tokens=(
                                        leafgrid_active or scope_kind == "structural"
                                    ),
                                    preserve_requested_leaf_tokens=leafgrid_active,
                                    comparison_surface_source=(
                                        fno_compare_config
                                        if fno_compare_config is not None
                                        else config
                                    ),
                                    extra_updates={
                                        "pipeline_benchmark_name": str(
                                            scope["benchmark_name"]
                                        ),
                                        "pipeline_hardness_grid": str(
                                            scope["hardness_grid"]
                                        ),
                                        "pipeline_grid_cell_ids": list(
                                            scope["grid_cell_ids"]
                                        ),
                                    },
                                )
                            )
                            if scope_tree_reference_label in {
                                SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
                                UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
                                UNIFIED_G_FNO_PARITY_CANARY_PRESET,
                                "structural_factorized_sketch_v3",
                            }:
                                tree_config["tree_checkpoint_metric"] = (
                                    _supervision_recovery_tree_checkpoint_metric(
                                        resolved_package_spec,
                                        default_metric=str(
                                            tree_config.get(
                                                "tree_checkpoint_metric",
                                                "val_exact_sketch_direct",
                                            )
                                        ),
                                        tree_reference_label=scope_tree_reference_label,
                                    )
                                )
                            exact_full_doc_parity = bool(
                                bool(
                                    tree_config.get(
                                        "pipeline_supervision_recovery_exact_full_doc_parity_requested",
                                        False,
                                    )
                                )
                                and _supervision_recovery_requires_exact_full_doc_parity(
                                    package_name=str(package_name),
                                    package_spec=resolved_package_spec,
                                    payload=tree_config,
                                )
                            )
                            if exact_full_doc_parity:
                                tree_config.update(
                                    {
                                        "comparison_mode": "exact_collapse",
                                        "official_fno_preserve_requested_leaf_tokens": True,
                                        "preserve_requested_leaf_tokens": True,
                                        "tree_exact_collapse_mode": (
                                            EXACT_COLLAPSE_ONE_TREE_IDENTITY_MODE
                                        ),
                                        "pipeline_supervision_recovery_parity_mode": "exact_full_doc",
                                        "pipeline_supervision_recovery_is_exact_full_doc_parity_row": True,
                                    }
                                )
                            tree_config = _validated_supervision_recovery_tree_config(
                                tree_config,
                                preset=str(args.preset),
                                scope_kind=scope_kind,
                                context=(
                                    f"supervision_recovery scope={scope['scope_key']} "
                                    f"train_docs={int(train_docs)} package={package_name} "
                                    f"fixed_leaf_tokens={int(fixed_leaf_tokens)}"
                                ),
                                tree_reference=scope_tree_reference,
                                allow_leafgrid_geometry=leafgrid_active,
                            )
                            tree_config.update(
                                {
                                    "pipeline_tree_reference_mode": str(
                                        scope_tree_reference.get("mode", "") or ""
                                    ),
                                    "pipeline_tree_reference_label": scope_tree_reference_label,
                                    "pipeline_tree_scope_kind": str(scope_kind),
                                    "pipeline_supervision_recovery_leafgrid_active": bool(
                                        leafgrid_active
                                    ),
                                }
                            )
                            if fno_compare_config is not None and fno_compare_payload is not None:
                                tree_compare_payload = {
                                    "benchmark_name": str(scope["benchmark_name"]),
                                    "hardness_grid": str(scope["hardness_grid"]),
                                    "grid_cell_ids": list(scope["grid_cell_ids"]),
                                    "base_bundle_path": str(
                                        scope.get("base_bundle_path", "") or ""
                                    ),
                                    "hazard_panel_id": str(
                                        scope.get("hazard_panel_id", "") or ""
                                    ),
                                    "train_doc_counts": [int(train_docs)],
                                    "baseline_families": [str(tree_family)],
                                    "seeds": [int(seed)],
                                }
                                fno_mode, fno_surface = _task_comparison_surface_snapshot(
                                    worker_kind="full_doc_diagnostics",
                                    config=fno_compare_config,
                                    task_payload=fno_compare_payload,
                                )
                                tree_mode, tree_surface = _task_comparison_surface_snapshot(
                                    worker_kind="full_doc_diagnostics",
                                    config=tree_config,
                                    task_payload=tree_compare_payload,
                                )
                                _ignored_surface_keys = {
                                    "comparison_mode",
                                    "fixed_leaf_tokens",
                                    "state_dim",
                                    "hidden_dim",
                                    "n_epochs",
                                    "batch_size",
                                    "lr",
                                    "weight_decay",
                                    "tree_leaf_fno_width",
                                    "tree_leaf_fno_n_modes",
                                    "tree_leaf_fno_n_layers",
                                    "tree_root_supervision_kind",
                                }
                                _fno_surf_cmp = {
                                    k: v
                                    for k, v in (fno_surface or {}).items()
                                    if k not in _ignored_surface_keys
                                }
                                _tree_surf_cmp = {
                                    k: v
                                    for k, v in (tree_surface or {}).items()
                                    if k not in _ignored_surface_keys
                                }
                                canonical_leafgrid_fno_reference = bool(
                                    leafgrid_active
                                    and str(package_name)
                                    == SUPERVISION_RECOVERY_EXACT_COLLAPSE_PACKAGE
                                    and int(fixed_leaf_tokens)
                                    != int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS)
                                )
                                if (
                                    fno_mode == "comparable"
                                    or tree_mode == "comparable"
                                ) and _fno_surf_cmp != _tree_surf_cmp:
                                    tree_reference_mode = str(
                                        scope_tree_reference.get("mode", "") or ""
                                    ).strip().lower()
                                    if (
                                        not canonical_leafgrid_fno_reference
                                        and tree_reference_mode not in {
                                        "capacity_locked",
                                        "package_capacity_locked",
                                        }
                                    ):
                                        raise ValueError(
                                            "comparable full-doc surface drift detected for "
                                            f"scope={scope['scope_key']} train_docs={int(train_docs)} "
                                            f"package={package_name}: fno_surface={fno_surface} "
                                            f"tree_surface={tree_surface}"
                                        )
                            tree_name = (
                                f"{scope['scope_key']}__train{int(train_docs):05d}"
                                f"__{package_name}{leaf_tag}{gamma_tag}__{tree_family}__d{int(data_seed)}"
                            )
                            tasks.append(
                                _direct_task(
                                    root=phase_root / "raw",
                                    name=tree_name,
                                    config=tree_config,
                                    worker_kind="full_doc_diagnostics",
                                    extra_payload={
                                        "benchmark_name": str(scope["benchmark_name"]),
                                        "hardness_grid": str(scope["hardness_grid"]),
                                        "grid_cell_ids": list(scope["grid_cell_ids"]),
                                        "base_bundle_path": str(
                                            scope.get("base_bundle_path", "") or ""
                                        ),
                                        "hazard_panel_id": str(
                                            scope.get("hazard_panel_id", "") or ""
                                        ),
                                        "train_doc_counts": [int(train_docs)],
                                        "baseline_families": [str(tree_family)],
                                        "seeds": [int(seed)],
                                    },
                                )
                            )
    return tasks, phase_root


def _build_support_phase(args: argparse.Namespace, root: Path) -> tuple[List[SubprocessTask], Path]:
    preset = PRESET_DEFAULTS[str(args.preset)]
    leaf_tokens_values = _parse_int_list(args.support_leaf_tokens, preset["support_leaf_tokens"])
    support_seeds = _parse_int_list(args.support_seeds, preset["support_seeds"])
    tree_reference = _resolve_tree_reference(args)
    phase_root = _phase_execution_root(root, "support_grid")
    tasks = []
    support_mode_specs = {
        "supported": {"leaf_query_rate": 1.0, "audit_fraction": 1.0, "include_root_query": True},
        "unsupported": {"leaf_query_rate": 0.0, "audit_fraction": 0.0, "include_root_query": True},
    }
    selected_modes = _parse_str_list(
        getattr(args, "support_modes", None),
        SUPPORTED_SUPPORT_MODES,
    )
    unknown_modes = [name for name in selected_modes if name not in support_mode_specs]
    if unknown_modes:
        raise ValueError(
            f"unknown support modes: {', '.join(sorted(unknown_modes))}; "
            f"valid options are {', '.join(sorted(support_mode_specs))}"
        )
    for fixed_leaf_tokens in leaf_tokens_values:
        for mode_name in selected_modes:
            mode_spec = dict(support_mode_specs[mode_name])
            for data_seed in support_seeds:
                seed = int(args.seed) + int(data_seed)
                name = f"leaf{fixed_leaf_tokens:03d}_{mode_name}_d{data_seed}"
                config = _base_ops_config(
                    args,
                    seed=seed,
                    data_seed=data_seed,
                    train_docs=4096 if str(args.preset) == "standard" else 1024,
                    val_docs=256 if str(args.preset) == "standard" else 64,
                    test_docs=256 if str(args.preset) == "standard" else 64,
                    batch_size=int(args.support_batch_size),
                    n_epochs=int(args.support_epochs),
                    fixed_leaf_tokens=fixed_leaf_tokens,
                )
                config.update(
                    {
                        "local_law_weight": 1.0,
                        "c1_relative_weight": 1.0,
                        "c2_relative_weight": 1.0,
                        "c3_relative_weight": 1.0,
                        **mode_spec,
                    }
                )
                config = _apply_tree_reference_overrides(
                    config,
                    tree_reference,
                    preserve_fixed_leaf_tokens=True,
                )
                tasks.append(_direct_task(root=phase_root / "raw", name=name, config=config))
    return tasks, phase_root


def _report_argv_from_sources(output_root: Path, sources: Mapping[str, str]) -> List[str]:
    report_root = _default_output_subdir(output_root, "tradeoff_report")
    argv = [
        "--output-dir",
        str(report_root),
    ]
    for key, path in sources.items():
        if key == "law_comparison_json":
            argv.extend(["--law-comparison-json", path])
        elif key == "batch_timing_summary":
            argv.extend(["--batch-timing-summary", path])
        elif key == "medium_grid_summary":
            argv.extend(["--medium-grid-summary", path])
        elif key == "docs_epochs_summary":
            argv.extend(["--docs-epochs-summary", path])
        elif key == "learnability_summary":
            argv.extend(["--learnability-summary", path])
        elif key == "weight_ablation_summary":
            argv.extend(["--weight-ablation-summary", path])
        elif key == "fno_upper_bound_summary":
            argv.extend(["--fno-upper-bound-summary", path])
        elif key == "oracle_budget_frontier_summary":
            argv.extend(["--oracle-budget-frontier-summary", path])
        elif key == "efficiency_suite_summary":
            argv.extend(["--efficiency-suite-summary", path])
        elif key == "large_batch_diagnosis_summary":
            argv.extend(["--large-batch-diagnosis-summary", path])
        elif key == "supervision_sweep_summary":
            argv.extend(["--supervision-sweep-summary", path])
        elif key == "supervision_recovery_summary":
            argv.extend(["--supervision-recovery-summary", path])
        elif key == "support_summary":
            argv.extend(["--support-summary", path])
    return argv


def _scheduler_item_from_subprocess_task(phase: str, task: SubprocessTask) -> SchedulerItem:
    env: Dict[str, str] = {}
    if str(task.device_label).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(task.device_label)
    metadata = {"task_name": str(task.name), **dict(task.metadata)}
    if task.progress_path is not None:
        metadata["progress_path"] = str(task.progress_path)
    return SchedulerItem(
        item_id=f"{phase}::{task.name}",
        phase=str(phase),
        kind="gpu_command",
        expected_outputs=(str(task.output_path),),
        command=tuple(str(arg) for arg in task.argv),
        log_path=str(task.log_path),
        env=env,
        metadata=metadata,
    )


def _full_doc_job_scheduler_item(
    *,
    phase: str,
    item_id: str,
    output_root: Path,
    job: Any,
    torch_threads: int,
    use_cuda: bool,
) -> SchedulerItem:
    job_output_dir = output_root / "jobs" / job_output_dir_name(str(job.job_name))
    return SchedulerItem(
        item_id=item_id,
        phase=str(phase),
        kind="gpu_command",
        expected_outputs=(str(job_output_dir / "summary.json"),),
        command=tuple(
            str(arg)
            for arg in worker_command_for_job(
                job,
                output_dir=job_output_dir,
                torch_threads=int(torch_threads),
                use_cuda=bool(use_cuda),
            )
        ),
        log_path=str(job_output_dir / "worker.log"),
        metadata={"job_name": str(job.job_name)},
    )


def _budget_frontier_namespace(
    args: argparse.Namespace,
    *,
    phase_root: Path,
    benchmark_name: str = "recoverable_v4",
    train_docs_override: int | None = None,
    tree_families_override: Sequence[str] | None = None,
    reference_families_override: Sequence[str] | None = None,
    hardness_grid: str = "",
    grid_cell_ids: Sequence[str] = (),
) -> argparse.Namespace:
    preset = PRESET_DEFAULTS[str(args.preset)]
    namespace = argparse.Namespace(**vars(args))
    namespace.output_root = str(phase_root)
    namespace.benchmark = str(benchmark_name)
    namespace.hardness_grid = str(hardness_grid or "")
    namespace.grid_cell_ids = tuple(str(cell) for cell in grid_cell_ids)
    namespace.train_doc_count = int(
        train_docs_override
        if train_docs_override is not None
        else (
            args.oracle_budget_train_docs
            if args.oracle_budget_train_docs is not None
            else preset["oracle_budget_train_docs"]
        )
    )
    tree_runs = (
        [_run_axis_from_token(value, role="primary") for value in tree_families_override]
        if tree_families_override is not None
        else _parse_run_axis_list(
            args.oracle_budget_method_runs,
            preset["oracle_budget_method_runs"],
            role="primary",
        )
    )
    reference_runs = (
        [_run_axis_from_token(value, role="reference") for value in reference_families_override]
        if reference_families_override is not None
        else _parse_run_axis_list(
            args.oracle_budget_reference_method_runs,
            preset["oracle_budget_reference_method_runs"],
            role="reference",
        )
    )
    namespace.tree_families = _legacy_families_from_run_axes(tree_runs)
    namespace.reference_families = _method_ids_from_run_axes(reference_runs)
    namespace.budget_calls_per_doc = _parse_float_list(
        args.oracle_budget_calls_per_doc,
        preset["oracle_budget_calls_per_doc"],
    )
    namespace.full_doc_budget_shares = _parse_float_list(
        args.oracle_budget_full_doc_shares,
        preset["oracle_budget_full_doc_shares"],
    )
    namespace.doc_consumption_modes = _parse_str_list(
        args.oracle_budget_doc_consumption_modes,
        preset["oracle_budget_doc_consumption_modes"],
    )
    namespace.local_split_modes = _parse_str_list(
        args.oracle_budget_local_split_modes,
        preset["oracle_budget_local_split_modes"],
    )
    namespace.local_allocation_policy = "breadth_first"
    namespace.budget_tree_config_mode = str(args.oracle_budget_tree_config_mode)
    namespace.seeds = _parse_int_list(args.oracle_budget_seeds, preset["oracle_budget_seeds"])
    namespace.job_granularity = str(getattr(args, "default_job_granularity", "family_train_seed"))
    namespace.resume = True
    namespace.use_cuda = bool(args.device_mode != "cpu")
    namespace.torch_threads = 1
    namespace.mig_uuids = ""
    namespace.state_dim = int(getattr(args, "state_dim", 128))
    namespace.hidden_dim = int(getattr(args, "hidden_dim", 512))
    namespace.n_epochs = int(getattr(args, "n_epochs", 32))
    namespace.batch_size = int(getattr(args, "batch_size", 64))
    namespace.lr = float(getattr(args, "lr", 5e-4))
    namespace.weight_decay = float(getattr(args, "weight_decay", 0.0))
    namespace.local_law_weight = getattr(args, "local_law_weight", getattr(args, "tree_local_law_weight", 0.3))
    namespace.root_share = getattr(args, "root_share", getattr(args, "tree_task_objective_weight", None))
    namespace.tree_local_law_weight = namespace.local_law_weight
    namespace.tree_task_objective_weight = namespace.root_share
    namespace.doc_sequence_train_fraction = float(
        getattr(args, "doc_sequence_train_fraction", 0.0)
    )
    namespace.gpu_runtime_data_mode = str(
        getattr(args, "gpu_runtime_data_mode", getattr(args, "runtime_data_mode", "resident"))
    )
    namespace.gpu_runtime_bucket_mode = str(
        getattr(
            args,
            "gpu_runtime_bucket_mode",
            getattr(args, "runtime_bucket_mode", "exact_then_bucketed"),
        )
    )
    namespace.gpu_runtime_preload_splits = tuple(
        _parse_str_list(
            getattr(args, "gpu_runtime_preload_splits", getattr(args, "runtime_preload_splits", None)),
            ("train", "val", "test"),
        )
    )
    namespace.gpu_runtime_preload_targets = bool(
        getattr(args, "gpu_runtime_preload_targets", getattr(args, "runtime_preload_targets", True))
    )
    namespace.gpu_runtime_workers_per_mig = int(
        getattr(args, "gpu_runtime_workers_per_mig", getattr(args, "runtime_workers_per_mig", 1))
    )
    namespace.gpu_runtime_allow_multi_worker_screen = bool(
        getattr(
            args,
            "gpu_runtime_allow_multi_worker_screen",
            getattr(args, "runtime_allow_multi_worker_screen", True),
        )
    )
    namespace.gpu_runtime_capacity_workers_per_mig = int(
        getattr(
            args,
            "gpu_runtime_capacity_workers_per_mig",
            getattr(args, "runtime_capacity_workers_per_mig", 2),
        )
    )
    return namespace


def _build_tradeoff_scheduler_graph(
    args: argparse.Namespace,
    *,
    output_root: Path,
    devices: Sequence[str],
) -> Dict[str, Any]:
    phases = _phase_set(args.phases)
    tree_bundle_contract = _markov_tradeoff_tree_bundle_contract(
        args,
        phases=sorted(phases),
    )
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "preset": str(args.preset),
        "phases": sorted(phases),
        "devices": list(devices),
        "tree_bundle_contract": tree_bundle_contract,
    }
    manifest["run_manifest"] = _markov_tradeoff_run_manifest(
        args=args,
        output_root=output_root,
        phases=sorted(phases),
        tree_bundle_contract=tree_bundle_contract,
        status="running",
        publication_ready=False,
        metadata={"scheduler_mode": "global"},
    )
    version_manifest = _load_report_version_manifest(output_root)
    _stage_report_sources(
        output_root=output_root,
        manifest=version_manifest,
        overrides=_parse_report_source_overrides(getattr(args, "report_sources", None)),
    )
    _refresh_selected_source_statuses(version_manifest, output_root=output_root, args=args)
    reducer_ids: List[str] = []
    items: List[SchedulerItem] = []

    def _register_direct_phase(
        *,
        phase: str,
        tasks: Sequence[SubprocessTask],
        expected_summary: Path,
        callback: Any,
    ) -> None:
        gpu_ids: List[str] = []
        for task in tasks:
            item = _scheduler_item_from_subprocess_task(phase, task)
            gpu_ids.append(str(item.item_id))
            items.append(item)
        items.append(
            SchedulerItem(
                item_id=f"{phase}::reduce",
                phase=str(phase),
                kind="cpu_callback",
                deps=tuple(gpu_ids),
                expected_outputs=(str(expected_summary),),
                callback=callback,
                reuse_existing=False,
            )
        )
        reducer_ids.append(f"{phase}::reduce")

    if "batch_timing" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "batch_timing")
        attempt_root = _phase_attempt_root(output_root, "batch_timing", attempt_id)
        tasks, phase_root = _build_batch_phase(args, attempt_root)
        summary_path = phase_root / "markov_fixed_fused_leaflaws_batchsize_timing_fullpipeline.json"
        combined_path = phase_root / "markov_fixed_fused_leaflaws_batchsize_combined_fullpipeline.json"
        alias_summary_path = _canonical_alias_path(output_root, "batch_timing_summary")

        def _batch_reduce(
            *,
            tasks: Sequence[SubprocessTask] = tasks,
            summary_path: Path = summary_path,
            combined_path: Path = combined_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            task_infos = [
                {
                    "name": task.name,
                    "output_path": str(task.output_path),
                    "log_path": str(task.log_path),
                }
                for task in tasks
            ]
            summary = _aggregate_batch_timing(task_infos)
            _write_json(summary_path, summary)
            simple_rows = []
            for row in list(summary.get("summary", []) or []):
                simple = dict(row)
                simple.pop("screen_eval_s", None)
                simple.pop("exact_metric_eval_s", None)
                simple.pop("eval_total_s", None)
                simple_rows.append(simple)
            _write_json(combined_path, {"summary": simple_rows})
            manifest["batch_timing"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="batch_timing",
                source_key="batch_timing_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
                extra_attempt={"combined_summary_relpath": str(combined_path.relative_to(output_root))},
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="batch_timing",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_batch_reduce,
        )

    if "medium_grid" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "medium_grid")
        attempt_root = _phase_attempt_root(output_root, "medium_grid", attempt_id)
        tasks, phase_root = _build_medium_phase(args, attempt_root)
        summary_path = phase_root / "aggregate_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "medium_grid_summary")

        def _medium_reduce(
            *,
            tasks: Sequence[SubprocessTask] = tasks,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            task_infos = [{"output_path": str(task.output_path)} for task in tasks]
            aggregate = _aggregate_medium_grid(task_infos)
            _write_json(summary_path, aggregate)
            manifest["medium_grid"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="medium_grid",
                source_key="medium_grid_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="medium_grid",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_medium_reduce,
        )

    if "docs_epochs" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "docs_epochs")
        attempt_root = _phase_attempt_root(output_root, "docs_epochs", attempt_id)
        tasks, phase_root = _build_docs_epochs_phase(args, attempt_root)
        summary_path = phase_root / "aggregate_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "docs_epochs_summary")

        def _docs_reduce(
            *,
            tasks: Sequence[SubprocessTask] = tasks,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            task_infos = [{"output_path": str(task.output_path)} for task in tasks]
            aggregate = _aggregate_docs_epochs(task_infos)
            _write_json(summary_path, aggregate)
            manifest["docs_epochs"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="docs_epochs",
                source_key="docs_epochs_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="docs_epochs",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_docs_reduce,
        )

    if "learnability" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "learnability")
        attempt_root = _phase_attempt_root(output_root, "learnability", attempt_id)
        tasks, phase_root = _build_learnability_phase(args, attempt_root)
        report_root = _default_output_subdir(attempt_root, "learnability_report")
        summary_path = report_root / "learnability_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "learnability_summary")

        def _learnability_reduce(
            *,
            phase_root: Path = phase_root,
            report_root: Path = report_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            _invoke_report(
                LEARNABILITY_REPORT_SCRIPT,
                [
                    "--family",
                    "markov",
                    "--input-root",
                    str(phase_root),
                    "--output-dir",
                    str(report_root),
                ],
            )
            manifest["learnability"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="learnability",
                source_key="learnability_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="learnability",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_learnability_reduce,
        )

    if "weight_ablation" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "weight_ablation")
        attempt_root = _phase_attempt_root(output_root, "weight_ablation", attempt_id)
        tasks, phase_root = _build_weight_ablation_phase(args, attempt_root)
        summary_path = phase_root / "weight_ablation_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "weight_ablation_summary")

        def _weight_reduce(
            *,
            phase_root: Path = phase_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            payloads = _load_ops_payloads(phase_root)
            summary = _aggregate_weight_ablation_from_payloads(payloads)
            _write_json(summary_path, summary)
            manifest["weight_ablation"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="weight_ablation",
                source_key="weight_ablation_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="weight_ablation",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_weight_reduce,
        )

    if "law_packages" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "law_packages")
        attempt_root = _phase_attempt_root(output_root, "law_packages", attempt_id)
        tasks, phase_root = _build_law_package_phase(args, attempt_root)
        summary_path = phase_root / "fno_tree_law_comparison.json"
        alias_summary_path = _canonical_alias_path(output_root, "law_comparison_json")

        def _law_reduce(
            *,
            phase_root: Path = phase_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            payloads = _load_ops_payloads(phase_root / "raw")
            summary = _aggregate_law_packages_from_payloads(payloads)
            _write_json(summary_path, summary)
            manifest["law_packages"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="law_packages",
                source_key="law_comparison_json",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="law_packages",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_law_reduce,
        )

    if "full_doc_anchor" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "full_doc_anchor")
        attempt_root = _phase_attempt_root(output_root, "full_doc_anchor", attempt_id)
        tasks, phase_root = _build_full_doc_anchor_phase(args, attempt_root)
        summary_path = phase_root / "full_doc_fno_upper_bound_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "fno_upper_bound_summary")

        def _full_doc_anchor_reduce(
            *,
            phase_root: Path = phase_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            payloads = _load_ops_payloads(phase_root / "raw")
            summary = _aggregate_full_doc_upper_bound_from_payloads(payloads)
            _write_json(summary_path, summary)
            manifest["full_doc_anchor"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="full_doc_anchor",
                source_key="fno_upper_bound_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="full_doc_anchor",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_full_doc_anchor_reduce,
        )

    if "oracle_budget_frontier" in phases:
        from scripts.run_tree_neural_full_doc_mig import (  # type: ignore
            build_budget_frontier_job_bundle,
            finalize_budget_frontier_output,
        )

        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "oracle_budget_frontier")
        phase_root = _phase_attempt_root(output_root, "oracle_budget_frontier", attempt_id)
        budget_args = _budget_frontier_namespace(args, phase_root=phase_root)
        budget_bundle = build_budget_frontier_job_bundle(budget_args)
        gpu_ids: List[str] = []
        for job in list(budget_bundle.get("jobs") or []):
            item = _full_doc_job_scheduler_item(
                phase="oracle_budget_frontier",
                item_id=f"oracle_budget_frontier::{job.job_name}",
                output_root=phase_root,
                job=job,
                torch_threads=1,
                use_cuda=bool(args.device_mode != "cpu"),
            )
            gpu_ids.append(str(item.item_id))
            items.append(item)
        summary_path = phase_root / "tree_oracle_budget_frontier_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "oracle_budget_frontier_summary")

        def _oracle_reduce(
            *,
            phase_root: Path = phase_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            result = finalize_budget_frontier_output(phase_root)
            manifest["oracle_budget_frontier"] = result
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="oracle_budget_frontier",
                source_key="oracle_budget_frontier_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
                extra_attempt={"pdf_relpath": str((phase_root / "tree_oracle_budget_frontier_report.pdf").relative_to(output_root))},
            )
            return {"result": dict(result)}

        items.append(
            SchedulerItem(
                item_id="oracle_budget_frontier::reduce",
                phase="oracle_budget_frontier",
                kind="cpu_callback",
                deps=tuple(gpu_ids),
                expected_outputs=(str(summary_path),),
                callback=_oracle_reduce,
                reuse_existing=False,
            )
        )
        reducer_ids.append("oracle_budget_frontier::reduce")

    if "efficiency_suite" in phases:
        from scripts.run_tree_neural_full_doc_mig import (  # type: ignore
            build_budget_frontier_job_bundle,
            finalize_budget_frontier_output,
        )

        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "efficiency_suite")
        phase_root = _phase_attempt_root(output_root, "efficiency_suite", attempt_id)
        dense_train_docs = _parse_int_list(
            getattr(args, "efficiency_anchor_train_docs_dense", None),
            PRESET_DEFAULTS[str(args.preset)]["efficiency_anchor_train_docs_dense"],
        )
        efficiency_train_docs = _parse_int_list(
            getattr(args, "efficiency_train_docs", None),
            PRESET_DEFAULTS[str(args.preset)]["efficiency_train_docs"],
        )
        anchor_tasks, _ = _build_efficiency_anchor_tasks(args, phase_root)
        recoverable_anchor_items = [
            _scheduler_item_from_subprocess_task("efficiency_suite", task)
            for task in anchor_tasks.get("recoverable_dense_anchor", [])
        ]
        structural_anchor_items = [
            _scheduler_item_from_subprocess_task("efficiency_suite", task)
            for task in anchor_tasks.get("structural_dense_anchor", [])
        ]
        for item in recoverable_anchor_items + structural_anchor_items:
            items.append(item)

        budget_reducer_ids: List[str] = []
        recoverable_budget_train_docs_reused: set[int]
        _, recoverable_budget_train_docs_reused = _existing_recoverable_budget_payloads(
            output_root,
            train_doc_counts=efficiency_train_docs,
        )
        structural_cells = _parse_str_list(
            getattr(args, "efficiency_structural_cells", None),
            PRESET_DEFAULTS[str(args.preset)]["efficiency_structural_cells"],
        )
        for train_docs in efficiency_train_docs:
            recoverable_root = phase_root / "recoverable_budget" / f"train{int(train_docs):05d}"
            if int(train_docs) not in recoverable_budget_train_docs_reused:
                recoverable_args = _budget_frontier_namespace(
                    args,
                    phase_root=recoverable_root,
                    train_docs_override=int(train_docs),
                    tree_families_override=EFFICIENCY_TREE_METHOD_RUNS,
                    reference_families_override=CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
                )
                recoverable_bundle = build_budget_frontier_job_bundle(recoverable_args)
                recoverable_gpu_ids: List[str] = []
                for job in list(recoverable_bundle.get("jobs") or []):
                    item = _full_doc_job_scheduler_item(
                        phase="efficiency_suite",
                        item_id=f"efficiency_suite::recoverable_budget::{job.job_name}",
                        output_root=recoverable_root,
                        job=job,
                        torch_threads=1,
                        use_cuda=bool(args.device_mode != "cpu"),
                    )
                    recoverable_gpu_ids.append(str(item.item_id))
                    items.append(item)
                reducer_id = f"efficiency_suite::recoverable_budget::train{int(train_docs):05d}::reduce"

                def _make_recoverable_budget_reduce(root: Path) -> Any:
                    def _callback() -> Mapping[str, Any]:
                        result = finalize_budget_frontier_output(root)
                        return {"result": dict(result)}

                    return _callback

                items.append(
                    SchedulerItem(
                        item_id=reducer_id,
                        phase="efficiency_suite",
                        kind="cpu_callback",
                        deps=tuple(recoverable_gpu_ids),
                        expected_outputs=(str(recoverable_root / "tree_oracle_budget_frontier_summary.json"),),
                        callback=_make_recoverable_budget_reduce(recoverable_root),
                        reuse_existing=False,
                    )
                )
                budget_reducer_ids.append(reducer_id)
            structural_root = phase_root / "structural_budget" / f"train{int(train_docs):05d}"
            structural_args = _budget_frontier_namespace(
                args,
                phase_root=structural_root,
                train_docs_override=int(train_docs),
                tree_families_override=EFFICIENCY_TREE_METHOD_RUNS,
                reference_families_override=CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
                hardness_grid=str(
                    getattr(
                        args,
                        "efficiency_hardness_grid",
                        PRESET_DEFAULTS[str(args.preset)]["efficiency_hardness_grid"],
                    )
                ),
                grid_cell_ids=tuple(structural_cells),
            )
            structural_bundle = build_budget_frontier_job_bundle(structural_args)
            structural_gpu_ids: List[str] = []
            for job in list(structural_bundle.get("jobs") or []):
                item = _full_doc_job_scheduler_item(
                    phase="efficiency_suite",
                    item_id=f"efficiency_suite::structural_budget::{job.job_name}",
                    output_root=structural_root,
                    job=job,
                    torch_threads=1,
                    use_cuda=bool(args.device_mode != "cpu"),
                )
                structural_gpu_ids.append(str(item.item_id))
                items.append(item)
            reducer_id = f"efficiency_suite::structural_budget::train{int(train_docs):05d}::reduce"

            def _make_structural_budget_reduce(root: Path) -> Any:
                def _callback() -> Mapping[str, Any]:
                    result = finalize_budget_frontier_output(root)
                    return {"result": dict(result)}

                return _callback

            items.append(
                SchedulerItem(
                    item_id=reducer_id,
                    phase="efficiency_suite",
                    kind="cpu_callback",
                    deps=tuple(structural_gpu_ids),
                    expected_outputs=(str(structural_root / "tree_oracle_budget_frontier_summary.json"),),
                    callback=_make_structural_budget_reduce(structural_root),
                    reuse_existing=False,
                )
            )
            budget_reducer_ids.append(reducer_id)

        efficiency_summary_path = phase_root / "summary.json"
        efficiency_deps = [
            *[str(item.item_id) for item in recoverable_anchor_items],
            *[str(item.item_id) for item in structural_anchor_items],
            *budget_reducer_ids,
        ]
        if "full_doc_anchor" in phases:
            efficiency_deps.append("full_doc_anchor::reduce")

        def _efficiency_reduce(
            *,
            phase_root: Path = phase_root,
            dense_train_docs: Sequence[int] = tuple(dense_train_docs),
            efficiency_train_docs: Sequence[int] = tuple(efficiency_train_docs),
            efficiency_summary_path: Path = efficiency_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            recoverable_anchor_payloads = (
                _load_ops_payloads(phase_root / "recoverable_dense_anchor" / "raw")
                + list(
                    _existing_recoverable_anchor_payloads(
                        output_root,
                        train_doc_counts=dense_train_docs,
                    )
                )
            )
            structural_anchor_payloads = _load_ops_payloads(phase_root / "structural_dense_anchor" / "raw")
            recoverable_budget_payloads, _ = _existing_recoverable_budget_payloads(
                output_root,
                train_doc_counts=efficiency_train_docs,
            )
            for train_docs in efficiency_train_docs:
                recoverable_summary = (
                    phase_root
                    / "recoverable_budget"
                    / f"train{int(train_docs):05d}"
                    / "tree_oracle_budget_frontier_summary.json"
                )
                if recoverable_summary.exists():
                    recoverable_budget_payloads.append(_read_json(recoverable_summary))
            structural_budget_payloads: List[Mapping[str, Any]] = []
            for train_docs in efficiency_train_docs:
                structural_summary = (
                    phase_root
                    / "structural_budget"
                    / f"train{int(train_docs):05d}"
                    / "tree_oracle_budget_frontier_summary.json"
                )
                if structural_summary.exists():
                    structural_budget_payloads.append(_read_json(structural_summary))
            efficiency_summary = _aggregate_efficiency_suite(
                recoverable_anchor_payloads=recoverable_anchor_payloads,
                structural_anchor_payloads=structural_anchor_payloads,
                recoverable_budget_payloads=recoverable_budget_payloads,
                structural_budget_payloads=structural_budget_payloads,
                tree_reference=_resolve_tree_reference(args),
            )
            _write_json(efficiency_summary_path, efficiency_summary)
            manifest["efficiency_suite"] = {"summary": str(efficiency_summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="efficiency_suite",
                source_key="efficiency_suite_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=efficiency_summary_path,
                alias_path=_canonical_alias_path(output_root, "efficiency_suite_summary"),
            )
            return {"result": {"summary": str(efficiency_summary_path)}}

        items.append(
            SchedulerItem(
                item_id="efficiency_suite::reduce",
                phase="efficiency_suite",
                kind="cpu_callback",
                deps=tuple(efficiency_deps),
                expected_outputs=(str(efficiency_summary_path),),
                callback=_efficiency_reduce,
                reuse_existing=False,
            )
        )
        reducer_ids.append("efficiency_suite::reduce")

    if "large_batch_diagnosis" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "large_batch_diagnosis")
        attempt_root = _phase_attempt_root(output_root, "large_batch_diagnosis", attempt_id)
        tasks, phase_root = _build_large_batch_diagnosis_phase(args, attempt_root)
        summary_path = phase_root / "aggregate_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "large_batch_diagnosis_summary")

        def _large_batch_reduce(
            *,
            tasks: Sequence[SubprocessTask] = tasks,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            task_infos = [{"name": task.name, "output_path": str(task.output_path)} for task in tasks]
            summary = _aggregate_large_batch_diagnosis(
                task_infos,
                target_total_steps=int(args.large_batch_target_steps),
            )
            _write_json(summary_path, summary)
            manifest["large_batch_diagnosis"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="large_batch_diagnosis",
                source_key="large_batch_diagnosis_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="large_batch_diagnosis",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_large_batch_reduce,
        )

    if "supervision_sweep" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "supervision_sweep")
        attempt_root = _phase_attempt_root(output_root, "supervision_sweep", attempt_id)
        tasks, phase_root = _build_supervision_phase(args, attempt_root)
        summary_path = phase_root / "supervision_sweep_summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "supervision_sweep_summary")

        def _supervision_reduce(
            *,
            phase_root: Path = phase_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            payloads = _load_ops_payloads(phase_root / "raw")
            summary = _aggregate_supervision_sweep_from_payloads(payloads)
            _write_json(summary_path, summary)
            manifest["supervision_sweep"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="supervision_sweep",
                source_key="supervision_sweep_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
                extra_source={
                    "expected_train_doc_counts": _parse_int_list(
                        args.supervision_train_docs,
                        PRESET_DEFAULTS[str(args.preset)]["supervision_train_docs"],
                    )
                },
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="supervision_sweep",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_supervision_reduce,
        )

    if "supervision_recovery" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "supervision_recovery")
        attempt_root = _phase_attempt_root(output_root, "supervision_recovery", attempt_id)
        tasks, phase_root = _build_supervision_recovery_phase(args, attempt_root)
        summary_path = phase_root / "summary.json"
        alias_summary_path = _canonical_alias_path(output_root, "supervision_recovery_summary")

        def _supervision_recovery_reduce(
            *,
            phase_root: Path = phase_root,
            summary_path: Path = summary_path,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            payloads = _load_ops_payloads(phase_root / "raw")
            summary = _aggregate_supervision_recovery_from_payloads(
                payloads,
                tree_family=str(
                    getattr(
                        args,
                        "supervision_recovery_method_id",
                        PRESET_DEFAULTS[str(args.preset)]["supervision_recovery_method_id"],
                    )
                ),
                recoverable_benchmark=_supervision_recovery_recoverable_benchmark_name(
                    args
                ),
                structural_grid=_supervision_recovery_structural_grid_name(args),
                structural_cell=_supervision_recovery_structural_cell_name(args),
                package_order=_resolved_supervision_recovery_package_order(args),
            )
            _write_json(summary_path, summary)
            manifest["supervision_recovery"] = {"summary": str(summary_path)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="supervision_recovery",
                source_key="supervision_recovery_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_path,
                alias_path=alias_summary_path,
                extra_source={
                    "expected_train_doc_counts": _parse_int_list(
                        getattr(args, "supervision_recovery_train_docs", None),
                        PRESET_DEFAULTS[str(args.preset)]["supervision_recovery_train_docs"],
                    ),
                    "expected_package_order": _resolved_supervision_recovery_package_order(args),
                    "expected_method_id": str(
                        getattr(
                            args,
                            "supervision_recovery_method_id",
                            PRESET_DEFAULTS[str(args.preset)]["supervision_recovery_method_id"],
                        )
                    ),
                    "expected_recoverable_benchmark": _supervision_recovery_recoverable_benchmark_name(
                        args
                    ),
                    "expected_structural_grid": _supervision_recovery_structural_grid_name(
                        args
                    ),
                    "expected_structural_cell": _supervision_recovery_structural_cell_name(args),
                },
            )
            return {"result": {"summary": str(summary_path)}}

        _register_direct_phase(
            phase="supervision_recovery",
            tasks=tasks,
            expected_summary=summary_path,
            callback=_supervision_recovery_reduce,
        )

    if "support_grid" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "support_grid")
        attempt_root = _phase_attempt_root(output_root, "support_grid", attempt_id)
        tasks, phase_root = _build_support_phase(args, attempt_root)
        summary_json = phase_root / "markov_local_support_detailed.summary.json"
        summary_csv = phase_root / "markov_local_support_detailed.summary.csv"
        alias_summary_path = _canonical_alias_path(output_root, "support_summary")

        def _support_reduce(
            *,
            phase_root: Path = phase_root,
            summary_json: Path = summary_json,
            summary_csv: Path = summary_csv,
            alias_summary_path: Path = alias_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            _invoke_report(
                SUPPORT_SUMMARY_SCRIPT,
                [
                    "--input-root",
                    str(phase_root / "raw"),
                    "--output-json",
                    str(summary_json),
                    "--output-csv",
                    str(summary_csv),
                ],
            )
            manifest["support_grid"] = {"summary": str(summary_json)}
            _register_phase_source(
                version_manifest,
                output_root=output_root,
                phase="support_grid",
                source_key="support_summary",
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                artifact_path=summary_json,
                alias_path=alias_summary_path,
                extra_attempt={"csv_relpath": str(summary_csv.relative_to(output_root))},
            )
            return {"result": {"summary": str(summary_json)}}

        _register_direct_phase(
            phase="support_grid",
            tasks=tasks,
            expected_summary=summary_json,
            callback=_support_reduce,
        )

    if "report" in phases:
        attempt_id = _new_attempt_id()
        config_fingerprint = _phase_config_fingerprint(args, "report")
        report_root = _phase_attempt_root(output_root, "tradeoff_report", attempt_id)
        report_summary_path = report_root / "summary.json"

        def _report_reduce(
            *,
            report_root: Path = report_root,
            report_summary_path: Path = report_summary_path,
            attempt_id: str = attempt_id,
            config_fingerprint: str = config_fingerprint,
        ) -> Mapping[str, Any]:
            _refresh_selected_source_statuses(version_manifest, output_root=output_root, args=args)
            _write_report_version_manifest(output_root, version_manifest)
            _invoke_report(
                TRADEOFF_REPORT_SCRIPT,
                [
                    "--manifest",
                    str(_report_version_manifest_path(output_root)),
                    "--version-root",
                    str(output_root),
                    "--output-dir",
                    str(report_root),
                ],
            )
            _register_report_outputs(
                version_manifest,
                output_root=output_root,
                attempt_id=attempt_id,
                config_fingerprint=config_fingerprint,
                attempt_root=report_root,
            )
            manifest["tradeoff_report"] = dict(version_manifest.get("report_outputs") or {})
            return {"result": {"summary": str(report_summary_path)}}

        items.append(
            SchedulerItem(
                item_id="report::reduce",
                phase="report",
                kind="cpu_callback",
                deps=tuple(reducer_ids),
                expected_outputs=(str(report_summary_path),),
                callback=_report_reduce,
                reuse_existing=False,
            )
        )

    return {
        "manifest": manifest,
        "report_version_manifest": version_manifest,
        "items": items,
    }


def _run_tradeoff_scheduler(
    args: argparse.Namespace,
    *,
    output_root: Path,
    devices: Sequence[str],
    run_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    graph = _build_tradeoff_scheduler_graph(
        args,
        output_root=output_root,
        devices=devices,
    )
    manifest = graph["manifest"]
    tree_bundle_contract = dict(manifest.get("tree_bundle_contract") or {})
    version_manifest = graph["report_version_manifest"]
    items = list(graph["items"])
    phases = _phase_set(args.phases)
    experiment_spec = _tradeoff_experiment_spec(
        args=args,
        output_root=output_root,
        run_plan=run_plan,
    )
    write_experiment_manifest(output_root, experiment_spec)
    _write_tradeoff_experiment_state(
        output_root=output_root,
        spec=experiment_spec,
        state="running",
        active_phase=(str(items[0].phase) if items else ""),
        items_total=len(items),
        pending_items=len(items),
    )
    scheduler_summary = run_scheduler(
        items,
        config=SchedulerConfig(
            devices=tuple(devices),
            max_gpu_items_per_mig=int(getattr(args, "max_gpu_items_per_mig", 1) or 1),
            cleanup_stale_children=bool(getattr(args, "cleanup_stale_children", True)),
            cancel_on_failure=False,
            raise_on_failure=False,
            root_markers=(str(output_root),),
            status_path=str(output_root / "experiment_status.json"),
            status_alias_paths=(str(output_root / "scheduler_status.json"),),
            status_metadata={
                "experiment_id": str(experiment_spec.experiment_id),
                "experiment_adapter": str(experiment_spec.adapter_id),
                "experiment_title": str(experiment_spec.title),
                "artifact_targets": [
                    "pipeline_summary_json",
                    "supervision_recovery_summary_json",
                    "tradeoff_report_summary_json",
                    "tradeoff_report_pdf",
                    "alignment_audit_json",
                    "alignment_audit_markdown",
                ],
            },
            event_log_path=str(output_root / "event_log.jsonl"),
        ),
    )
    manifest["scheduler"] = scheduler_summary
    _refresh_selected_source_statuses(version_manifest, output_root=output_root, args=args)
    manifest["report_version_manifest"] = str(
        _write_report_version_manifest(output_root, version_manifest)
    )
    manifest["selected_sources"] = dict(version_manifest.get("selected_sources") or {})
    supervision_summary_path = output_root / "supervision_recovery" / "summary.json"
    report_summary_path = output_root / "tradeoff_report" / "summary.json"
    if supervision_summary_path.exists():
        try:
            supervision_summary = json.loads(
                supervision_summary_path.read_text(encoding="utf-8")
            )
            manifest["contract_gate_status"] = str(
                supervision_summary.get("contract_gate_status", "") or ""
            )
            manifest["quarantined_row_count"] = int(
                _safe_int(supervision_summary.get("quarantined_row_count"), 0)
            )
            manifest["quarantined_sources"] = list(
                supervision_summary.get("quarantined_sources") or []
            )
        except Exception:
            pass
    if report_summary_path.exists():
        try:
            report = build_markov_alignment_audit_report(
                family_grids_summary_json=report_summary_path,
                run_lean_build=False,
            )
            audit_outputs = write_markov_alignment_audit_report(
                report,
                output_json=output_root / "markov_alignment_audit.json",
                output_markdown=output_root / "markov_alignment_audit.md",
            )
            manifest["alignment_audit_json"] = str(audit_outputs["output_json"])
            manifest["alignment_audit_markdown"] = str(
                audit_outputs["output_markdown"]
            )
            manifest["contract_gate_status"] = (
                "fail" if int(report.summary.get("n_fail", 0)) > 0 else "pass"
            )
        except Exception:
            pass
    pipeline_summary_path = output_root / "pipeline_summary.json"
    scheduler_state = str(scheduler_summary.get("state", "completed") or "completed")
    manifest["run_manifest"] = _markov_tradeoff_run_manifest(
        args=args,
        output_root=output_root,
        phases=sorted(phases),
        tree_bundle_contract=tree_bundle_contract,
        sources={
            **dict(version_manifest.get("selected_sources") or {}),
            "alignment_audit_json": manifest.get("alignment_audit_json", ""),
            "alignment_audit_markdown": manifest.get("alignment_audit_markdown", ""),
        },
        status="completed" if scheduler_state == "completed" else "partial",
        publication_ready=scheduler_state == "completed",
        metadata={"scheduler_mode": "global", "scheduler_state": scheduler_state},
    )
    _write_json(pipeline_summary_path, manifest)
    merge_artifacts(output_root, _tradeoff_artifacts(output_root))
    canonical_rows = list(
        _tradeoff_result_rows(
            spec=experiment_spec,
            manifest=manifest,
        )
    )
    if supervision_summary_path.exists():
        try:
            canonical_rows.extend(
                _supervision_recovery_result_rows_from_summary(
                    spec=experiment_spec,
                    summary=json.loads(supervision_summary_path.read_text(encoding="utf-8")),
                )
            )
        except Exception:
            pass
    append_result_rows(
        output_root,
        canonical_rows,
    )
    _write_tradeoff_experiment_state(
        output_root=output_root,
        spec=experiment_spec,
        state=str(scheduler_summary.get("state", "completed") or "completed"),
        active_phase=str(scheduler_summary.get("active_phase", "") or ""),
        items_total=_scheduler_item_count(
            scheduler_summary.get("items_total", len(items))
        )
        or len(items),
        completed_items=_scheduler_item_count(
            scheduler_summary.get("completed_items", 0)
        ),
        failed_items=_scheduler_item_count(
            scheduler_summary.get("failed_items", 0)
        ),
        active_items=_scheduler_item_count(
            scheduler_summary.get("active_items", 0)
        ),
        pending_items=_scheduler_item_count(
            scheduler_summary.get("pending_items", 0)
        ),
    )
    return {
        "manifest": manifest,
        "pipeline_summary_path": pipeline_summary_path,
    }


def _write_selection_template(path: Path) -> None:
    assert_public_contract_clean(
        PIPELINE_SELECTION_TEMPLATE,
        surface="markov tradeoff config template",
    )
    write_structured_config(path, PIPELINE_SELECTION_TEMPLATE)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    worker_task = _preparse_worker_task(raw_argv)
    if worker_task is not None:
        return _run_worker(worker_task)

    args = _parse_args(raw_argv)
    if args.write_selection_template is not None:
        _write_selection_template(Path(args.write_selection_template))
        print(json.dumps({"selection_template": str(Path(args.write_selection_template).expanduser())}, indent=2))
        return 0
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "refresh_existing_output_root", False)):
        result = _refresh_existing_tradeoff_outputs(
            args,
            output_root=output_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    phases = _phase_set(args.phases)
    devices = _resolve_devices(args)
    run_plan = build_run_plan(args, devices=devices)
    public_run_plan = _public_payload_for_contract(run_plan)
    assert_public_contract_clean(public_run_plan, surface="markov tradeoff run plan")
    if args.write_run_plan is not None:
        _write_json(Path(args.write_run_plan).expanduser(), public_run_plan)
    if bool(args.plan_only):
        print(json.dumps(public_run_plan, indent=2, sort_keys=True))
        return 0
    if not devices:
        raise SystemExit("No devices resolved. Use --device-mode cpu or provide MIG UUIDs with --migs.")
    if str(getattr(args, "scheduler_mode", "global_per_run")) == "global_per_run":
        result = _run_tradeoff_scheduler(
            args,
            output_root=output_root,
            devices=devices,
            run_plan=run_plan,
        )
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "pipeline_summary": str(result["pipeline_summary_path"]),
                },
                indent=2,
            )
        )
        return 0

    tree_bundle_contract = _markov_tradeoff_tree_bundle_contract(
        args,
        phases=sorted(phases),
    )
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "preset": str(args.preset),
        "phases": sorted(phases),
        "devices": list(devices),
        "tree_bundle_contract": tree_bundle_contract,
    }
    sources: Dict[str, str] = {}

    if "batch_timing" in phases:
        tasks, phase_root = _build_batch_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        summary = _aggregate_batch_timing(runs)
        timing_path = phase_root / "markov_fixed_fused_leaflaws_batchsize_timing_fullpipeline.json"
        combined_path = phase_root / "markov_fixed_fused_leaflaws_batchsize_combined_fullpipeline.json"
        _write_json(timing_path, summary)
        simple_rows = []
        for row in list(summary.get("summary", []) or []):
            simple = dict(row)
            simple.pop("screen_eval_s", None)
            simple.pop("exact_metric_eval_s", None)
            simple.pop("eval_total_s", None)
            simple_rows.append(simple)
        _write_json(combined_path, {"summary": simple_rows})
        manifest["batch_timing"] = {"runs": runs, "summary": str(timing_path)}
        sources["batch_timing_summary"] = str(timing_path)

    if "medium_grid" in phases:
        tasks, phase_root = _build_medium_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        aggregate = _aggregate_medium_grid(runs)
        aggregate_path = phase_root / "aggregate_summary.json"
        _write_json(aggregate_path, aggregate)
        manifest["medium_grid"] = {"runs": runs, "summary": str(aggregate_path)}
        sources["medium_grid_summary"] = str(aggregate_path)

    if "docs_epochs" in phases:
        tasks, phase_root = _build_docs_epochs_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        aggregate = _aggregate_docs_epochs(runs)
        aggregate_path = phase_root / "aggregate_summary.json"
        _write_json(aggregate_path, aggregate)
        manifest["docs_epochs"] = {"runs": runs, "summary": str(aggregate_path)}
        sources["docs_epochs_summary"] = str(aggregate_path)

    if "learnability" in phases:
        tasks, phase_root = _build_learnability_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        report_root = _default_output_subdir(output_root, "learnability_report")
        _invoke_report(
            LEARNABILITY_REPORT_SCRIPT,
            [
                "--family",
                "markov",
                "--input-root",
                str(phase_root),
                "--output-dir",
                str(report_root),
            ],
        )
        summary_path = report_root / "learnability_summary.json"
        manifest["learnability"] = {"runs": runs, "summary": str(summary_path)}
        sources["learnability_summary"] = str(summary_path)

    if "weight_ablation" in phases:
        tasks, phase_root = _build_weight_ablation_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        payloads = _load_ops_payloads(phase_root)
        summary = _aggregate_weight_ablation_from_payloads(payloads)
        summary_path = phase_root / "weight_ablation_summary.json"
        _write_json(summary_path, summary)
        manifest["weight_ablation"] = {"runs": runs, "summary": str(summary_path)}
        sources["weight_ablation_summary"] = str(summary_path)

    if "law_packages" in phases:
        tasks, phase_root = _build_law_package_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        payloads = _load_ops_payloads(phase_root / "raw")
        summary = _aggregate_law_packages_from_payloads(payloads)
        summary_path = phase_root / "fno_tree_law_comparison.json"
        _write_json(summary_path, summary)
        manifest["law_packages"] = {"runs": runs, "summary": str(summary_path)}
        sources["law_comparison_json"] = str(summary_path)

    if "full_doc_anchor" in phases:
        tasks, phase_root = _build_full_doc_anchor_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        payloads = _load_ops_payloads(phase_root / "raw")
        summary = _aggregate_full_doc_upper_bound_from_payloads(payloads)
        summary_path = phase_root / "full_doc_fno_upper_bound_summary.json"
        _write_json(summary_path, summary)
        manifest["full_doc_anchor"] = {"runs": runs, "summary": str(summary_path)}
        sources["fno_upper_bound_summary"] = str(summary_path)

    if "oracle_budget_frontier" in phases:
        phase_root = _default_output_subdir(output_root, "oracle_budget_frontier")
        phase_root.mkdir(parents=True, exist_ok=True)
        log_path = phase_root / "oracle_budget_frontier.log"
        command = _oracle_budget_frontier_command(args, phase_root=phase_root, devices=devices)
        _run_logged_command(command, log_path=log_path)
        summary_path = phase_root / "tree_oracle_budget_frontier_summary.json"
        manifest["oracle_budget_frontier"] = {
            "command": [str(item) for item in command],
            "log": str(log_path),
            "summary": str(summary_path),
            "pdf": str(phase_root / "tree_oracle_budget_frontier_report.pdf"),
        }
        sources["oracle_budget_frontier_summary"] = str(summary_path)

    if "efficiency_suite" in phases:
        phase_root = _default_output_subdir(output_root, "efficiency_suite")
        preset = PRESET_DEFAULTS[str(args.preset)]
        dense_train_docs = _parse_int_list(
            getattr(args, "efficiency_anchor_train_docs_dense", None),
            preset["efficiency_anchor_train_docs_dense"],
        )
        efficiency_train_docs = _parse_int_list(
            getattr(args, "efficiency_train_docs", None),
            preset["efficiency_train_docs"],
        )
        anchor_tasks, _ = _build_efficiency_anchor_tasks(args, output_root)
        recoverable_anchor_runs = _run_subprocess_tasks(
            anchor_tasks.get("recoverable_dense_anchor", []),
            devices,
        )
        structural_anchor_runs = _run_subprocess_tasks(
            anchor_tasks.get("structural_dense_anchor", []),
            devices,
        )
        recoverable_anchor_payloads = (
            _load_ops_payloads(phase_root / "recoverable_dense_anchor" / "raw")
            + list(
                _existing_recoverable_anchor_payloads(
                    output_root,
                    train_doc_counts=dense_train_docs,
                )
            )
        )
        structural_anchor_payloads = _load_ops_payloads(phase_root / "structural_dense_anchor" / "raw")

        recoverable_budget_payloads, reusable_recoverable_budget_train_docs = _existing_recoverable_budget_payloads(
            output_root,
            train_doc_counts=efficiency_train_docs,
        )
        structural_budget_payloads: List[Mapping[str, Any]] = []
        budget_runs: List[Dict[str, Any]] = []
        for spec in _efficiency_budget_specs(args, phase_root=phase_root, devices=devices):
            spec_name = str(spec.get("name", ""))
            if spec_name.startswith("recoverable_budget__train"):
                train_docs_token = spec_name.rsplit("train", 1)[-1]
                train_docs_value = int(_safe_int(train_docs_token))
                if train_docs_value in reusable_recoverable_budget_train_docs:
                    budget_runs.append(
                        {
                            "name": spec_name,
                            "command": [],
                            "log": "",
                            "summary": str(spec.get("summary_path", "")),
                            "reused": True,
                        }
                    )
                    continue
            command = [str(item) for item in list(spec.get("command") or [])]
            log_path = Path(str(spec.get("log_path")))
            summary_path = Path(str(spec.get("summary_path")))
            if not summary_path.exists():
                _run_logged_command(command, log_path=log_path)
            if summary_path.exists():
                payload = _read_json(summary_path)
                if spec_name.startswith("recoverable_budget"):
                    recoverable_budget_payloads.append(payload)
                else:
                    structural_budget_payloads.append(payload)
            budget_runs.append(
                {
                    "name": spec_name,
                    "command": command,
                    "log": str(log_path),
                    "summary": str(summary_path),
                    "reused": bool(summary_path.exists() and not command),
                }
            )

        efficiency_summary = _aggregate_efficiency_suite(
            recoverable_anchor_payloads=recoverable_anchor_payloads,
            structural_anchor_payloads=structural_anchor_payloads,
            recoverable_budget_payloads=recoverable_budget_payloads,
            structural_budget_payloads=structural_budget_payloads,
            tree_reference=_resolve_tree_reference(args),
        )
        summary_path = phase_root / "summary.json"
        _write_json(summary_path, efficiency_summary)
        manifest["efficiency_suite"] = {
            "recoverable_dense_anchor_runs": recoverable_anchor_runs,
            "structural_dense_anchor_runs": structural_anchor_runs,
            "budget_runs": budget_runs,
            "summary": str(summary_path),
        }
        sources["efficiency_suite_summary"] = str(summary_path)

    if "large_batch_diagnosis" in phases:
        tasks, phase_root = _build_large_batch_diagnosis_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        summary = _aggregate_large_batch_diagnosis(
            runs,
            target_total_steps=int(args.large_batch_target_steps),
        )
        summary_path = phase_root / "aggregate_summary.json"
        _write_json(summary_path, summary)
        manifest["large_batch_diagnosis"] = {"runs": runs, "summary": str(summary_path)}
        sources["large_batch_diagnosis_summary"] = str(summary_path)

    if "supervision_sweep" in phases:
        tasks, phase_root = _build_supervision_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        payloads = _load_ops_payloads(phase_root / "raw")
        summary = _aggregate_supervision_sweep_from_payloads(payloads)
        summary_path = phase_root / "supervision_sweep_summary.json"
        _write_json(summary_path, summary)
        manifest["supervision_sweep"] = {"runs": runs, "summary": str(summary_path)}
        sources["supervision_sweep_summary"] = str(summary_path)

    if "support_grid" in phases:
        tasks, phase_root = _build_support_phase(args, output_root)
        runs = _run_subprocess_tasks(tasks, devices)
        summary_json = phase_root / "markov_local_support_detailed.summary.json"
        summary_csv = phase_root / "markov_local_support_detailed.summary.csv"
        _invoke_report(
            SUPPORT_SUMMARY_SCRIPT,
            [
                "--input-root",
                str(phase_root / "raw"),
                "--output-json",
                str(summary_json),
                "--output-csv",
                str(summary_csv),
            ],
        )
        manifest["support_grid"] = {"runs": runs, "summary": str(summary_json)}
        sources["support_summary"] = str(summary_json)

    if "report" in phases:
        report_root = _default_output_subdir(output_root, "tradeoff_report")
        sources = _hydrate_existing_report_sources(output_root, sources)
        argv = [
            "--output-dir",
            str(report_root),
        ]
        for key, path in sources.items():
            if key == "law_comparison_json":
                argv.extend(["--law-comparison-json", path])
            elif key == "batch_timing_summary":
                argv.extend(["--batch-timing-summary", path])
            elif key == "medium_grid_summary":
                argv.extend(["--medium-grid-summary", path])
            elif key == "docs_epochs_summary":
                argv.extend(["--docs-epochs-summary", path])
            elif key == "learnability_summary":
                argv.extend(["--learnability-summary", path])
            elif key == "weight_ablation_summary":
                argv.extend(["--weight-ablation-summary", path])
            elif key == "fno_upper_bound_summary":
                argv.extend(["--fno-upper-bound-summary", path])
            elif key == "oracle_budget_frontier_summary":
                argv.extend(["--oracle-budget-frontier-summary", path])
            elif key == "efficiency_suite_summary":
                argv.extend(["--efficiency-suite-summary", path])
            elif key == "large_batch_diagnosis_summary":
                argv.extend(["--large-batch-diagnosis-summary", path])
            elif key == "supervision_sweep_summary":
                argv.extend(["--supervision-sweep-summary", path])
            elif key == "support_summary":
                argv.extend(["--support-summary", path])
        _invoke_report(TRADEOFF_REPORT_SCRIPT, argv)
        manifest["tradeoff_report"] = {
            "summary": str(report_root / "summary.json"),
            "markdown": str(report_root / "report.md"),
            "pdf": str(report_root / "report.pdf"),
        }

    pipeline_summary_path = output_root / "pipeline_summary.json"
    manifest["sources"] = sources
    manifest["run_manifest"] = _markov_tradeoff_run_manifest(
        args=args,
        output_root=output_root,
        phases=sorted(phases),
        tree_bundle_contract=tree_bundle_contract,
        sources=sources,
        status="completed",
        publication_ready=True,
        metadata={"scheduler_mode": "direct"},
    )
    _write_json(pipeline_summary_path, manifest)
    print(json.dumps({"output_root": str(output_root), "pipeline_summary": str(pipeline_summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
