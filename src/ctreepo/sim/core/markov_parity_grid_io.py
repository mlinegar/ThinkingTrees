from __future__ import annotations

"""Public IO helpers for Markov supervision-recovery parity-grid outputs."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from src.experiments import (
    append_result_rows,
    benchmark_ref_from_parts,
    canonical_artifact_refs_from_paths,
    merge_artifacts,
)
from src.experiments.contracts import ResultRow
from src.experiments.markov_full_doc import method_ref_from_markov_full_doc_run
from src.ctreepo.sim.core.full_doc_config_codec import (
    runtime_config_overrides_from_config_like,
)
from src.ctreepo.sim.core.markov_comparison_surface import (
    FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS,
)
from src.ctreepo.sim.core.markov_v3_row_contract import annotate_downstream_v3_row
from src.ctreepo.sim.core.tree_reference_presets import (
    ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET,
    ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET,
    ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET,
    STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    resolve_tree_reference_preset_config,
)


STUDY_NAME = "supervision_recovery_parity_grid"



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


def _canonical_train_ladder_payload() -> List[int]:
    return [int(value) for value in CANONICAL_TRAIN_LADDER]


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


def _load_prepared_metadata(config_mapping: Mapping[str, Any]) -> Dict[str, Any]:
    prepared_root = str(config_mapping.get("prepared_data_root", "") or "").strip()
    prepared_signature = str(config_mapping.get("prepared_data_signature", "") or "").strip()
    if not prepared_root or not prepared_signature:
        return {}
    metadata_path = (
        Path(prepared_root).expanduser()
        / f"prepared_{prepared_signature}"
        / "metadata.json"
    )
    payload = _load_json(metadata_path)
    if not payload:
        return {}
    return {
        "metadata_json": str(metadata_path),
        "train_prefix_counts": [
            int(value) for value in list(payload.get("train_prefix_counts") or [])
        ],
        "train_prefix_signatures": {
            str(key): str(value)
            for key, value in dict(payload.get("train_prefix_signatures") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        "train_corpus_signature": str(payload.get("train_corpus_signature", "") or ""),
        "val_corpus_signature": str(payload.get("val_corpus_signature", "") or ""),
        "test_corpus_signature": str(payload.get("test_corpus_signature", "") or ""),
    }


def _infer_root_evidence_status(
    *,
    output_root: Path,
    state: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    root_text = str(output_root).lower()
    state_text = str(state or "").strip().lower()
    if state_text in {"stopped", "stop_requested", "cancelled", "canceled", "aborted"}:
        return EVIDENCE_STATUS_STOPPED
    if any(token in root_text for token in ("exploratory", "diagnostic", "smoke", "check")):
        return EVIDENCE_STATUS_EXPLORATORY
    if state_text == "completed" and rows and all(
        str((row or {}).get("state", "")).strip().lower() == "completed"
        for row in rows
    ):
        return EVIDENCE_STATUS_AUTHORITATIVE
    return EVIDENCE_STATUS_PARTIAL


def _controller_failed_job_names(output_root: Path) -> set[str]:
    payload = _load_json(output_root / "controller_results.json")
    failed = set()
    for row in list(payload.get("failed_jobs") or []):
        job_name = str((row or {}).get("job_name", "")).strip()
        if job_name:
            failed.add(job_name)
    return failed


def summary_metrics_for_job(job_output_dir: Path) -> Dict[str, Any]:
    payload = _load_json(job_output_dir / "summary.json")
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    runs = list(payload.get("runs") or [])
    run = dict(runs[0] if runs else {})
    if aggregate_rows:
        aggregate = dict(aggregate_rows[0])
        aggregate.update(
            {
                "bundle_source": run.get("bundle_source"),
                "train_corpus_signature": run.get("train_corpus_signature"),
                "val_corpus_signature": run.get("val_corpus_signature"),
                "test_corpus_signature": run.get("test_corpus_signature"),
                "collapse_runtime_delegate_family": run.get(
                    "collapse_runtime_delegate_family"
                ),
                "collapse_runtime_mode": run.get("collapse_runtime_mode"),
                "tree_local_weighting_mode": run.get("tree_local_weighting_mode"),
                "local_loss_kind": run.get("local_loss_kind"),
                "local_sampling_design_name": run.get(
                    "local_sampling_design_name"
                ),
                "leaf_population_size": run.get("leaf_population_size"),
                "leaf_sample_size": run.get("leaf_sample_size"),
                "leaf_effective_propensity": run.get("leaf_effective_propensity"),
                "merge_population_size": run.get("merge_population_size"),
                "merge_sample_size": run.get("merge_sample_size"),
                "merge_effective_propensity": run.get(
                    "merge_effective_propensity"
                ),
                "local_objective_audit": dict(
                    run.get("local_objective_audit", {}) or {}
                ),
                "optimization_root_weight": run.get("optimization_root_weight"),
                "local_law_c1_weight": run.get("local_law_c1_weight"),
                "local_law_c2_weight": run.get("local_law_c2_weight"),
                "local_law_c3_weight": run.get("local_law_c3_weight"),
                "leaf_supervision_kind": run.get("leaf_supervision_kind"),
                "internal_supervision_kind": run.get("internal_supervision_kind"),
                "leaf_label_rate": run.get("leaf_label_rate"),
                "internal_label_rate": run.get("internal_label_rate"),
                "comparison_mode": aggregate.get("comparison_mode", run.get("comparison_mode")),
                "comparison_semantics": aggregate.get(
                    "comparison_semantics",
                    run.get("comparison_semantics"),
                ),
                "comparison_semantics_label": aggregate.get(
                    "comparison_semantics_label",
                    run.get("comparison_semantics_label"),
                ),
                "run_intent_hash": aggregate.get(
                    "run_intent_hash",
                    run.get("run_intent_hash"),
                ),
                "run_intent_validation_status": aggregate.get(
                    "run_intent_validation_status",
                    run.get("run_intent_validation_status"),
                ),
                "requested_fixed_leaf_tokens": aggregate.get(
                    "requested_fixed_leaf_tokens",
                    run.get("requested_fixed_leaf_tokens"),
                ),
                "executed_fixed_leaf_tokens": aggregate.get(
                    "executed_fixed_leaf_tokens",
                    run.get("executed_fixed_leaf_tokens"),
                ),
                "config": dict(run.get("config") or {}),
            }
        )
        return aggregate
    if runs:
        return {
            "test_root_mae_mean": run.get("test_root_mae"),
            "test_leaf_mae_mean": run.get("test_leaf_mae"),
            "test_merge_mae_mean": run.get("test_merge_mae"),
            "val_root_mae_mean": run.get("val_root_mae"),
            "best_epoch_mean": run.get("best_epoch"),
            "elapsed_s_mean": run.get("elapsed_s"),
            "selection_metric_name": run.get("selection_metric_name"),
            "bundle_source": run.get("bundle_source"),
            "train_corpus_signature": run.get("train_corpus_signature"),
            "val_corpus_signature": run.get("val_corpus_signature"),
            "test_corpus_signature": run.get("test_corpus_signature"),
            "collapse_runtime_delegate_family": run.get(
                "collapse_runtime_delegate_family"
            ),
            "collapse_runtime_mode": run.get("collapse_runtime_mode"),
            "tree_local_weighting_mode": run.get("tree_local_weighting_mode"),
            "local_loss_kind": run.get("local_loss_kind"),
            "local_sampling_design_name": run.get("local_sampling_design_name"),
            "leaf_population_size": run.get("leaf_population_size"),
            "leaf_sample_size": run.get("leaf_sample_size"),
            "leaf_effective_propensity": run.get("leaf_effective_propensity"),
            "merge_population_size": run.get("merge_population_size"),
            "merge_sample_size": run.get("merge_sample_size"),
            "merge_effective_propensity": run.get("merge_effective_propensity"),
            "local_objective_audit": dict(
                run.get("local_objective_audit", {}) or {}
            ),
            "optimization_root_weight": run.get("optimization_root_weight"),
            "local_law_c1_weight": run.get("local_law_c1_weight"),
            "local_law_c2_weight": run.get("local_law_c2_weight"),
            "local_law_c3_weight": run.get("local_law_c3_weight"),
            "leaf_supervision_kind": run.get("leaf_supervision_kind"),
            "internal_supervision_kind": run.get("internal_supervision_kind"),
            "leaf_label_rate": run.get("leaf_label_rate"),
            "internal_label_rate": run.get("internal_label_rate"),
            "comparison_mode": run.get("comparison_mode"),
            "comparison_semantics": run.get("comparison_semantics"),
            "comparison_semantics_label": run.get("comparison_semantics_label"),
            "run_intent_hash": run.get("run_intent_hash"),
            "run_intent_validation_status": run.get(
                "run_intent_validation_status"
            ),
            "requested_fixed_leaf_tokens": run.get(
                "requested_fixed_leaf_tokens"
            ),
            "executed_fixed_leaf_tokens": run.get(
                "executed_fixed_leaf_tokens"
            ),
            "config": dict(run.get("config") or {}),
        }
    return {}


def _config_mapping_from_manifest_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_config_overrides_from_config_like(
        config,
        allow_private_tree_aliases=True,
    )


def row_from_manifest_job(
    job_payload: Mapping[str, Any],
    *,
    prior_row: Mapping[str, Any] | None = None,
    failed_job_names: set[str],
) -> Dict[str, Any]:
    config = dict(job_payload.get("config") or {})
    job_name = str(job_payload.get("job_name", "") or "")
    job_output_dir = Path(str(job_payload.get("job_output_dir", "")))
    metrics = summary_metrics_for_job(job_output_dir)
    completed = bool(metrics)
    prior = dict(prior_row or {})
    state = "planned"
    if completed:
        state = "completed"
    elif job_name in failed_job_names:
        state = "failed"
    elif str(prior.get("state", "")).strip():
        state = str(prior.get("state"))

    def _pick_metric(name: str) -> Any:
        if name in metrics:
            return metrics.get(name)
        return prior.get(name)

    claim_level = str(
        job_payload.get("claim_level", prior.get("claim_level", CLAIM_LEVEL_EMPIRICAL_GEOMETRY))
        or CLAIM_LEVEL_EMPIRICAL_GEOMETRY
    )
    runtime_config = dict(metrics.get("config") or {})
    config_mapping = (
        runtime_config
        if runtime_config
        else _config_mapping_from_manifest_config(config)
    )
    prepared_metadata = _load_prepared_metadata(config_mapping)
    reference_surface = dict(job_payload.get("official_fno_reference_surface") or {})
    config_diff = (
        _config_diff_vs_official_fno(
            config_mapping=config_mapping,
            reference_surface=(
                reference_surface
                or _legacy_exact_collapse_reference_surface(
                    benchmark=str(job_payload.get("benchmark", "") or ""),
                    fixed_leaf_tokens=int(job_payload.get("fixed_leaf_tokens", 0) or 0),
                    epoch_cap=int(job_payload.get("epoch_cap", 0) or 0),
                )
            ),
        )
        if claim_level == CLAIM_LEVEL_EXACT_COLLAPSE
        else {}
    )
    train_doc_count = int(job_payload.get("train_doc_count", 0) or 0)
    reference_bundle_source = str(
        _pick_metric("bundle_source")
        or config_mapping.get("base_bundle_path", "")
        or ""
    ).strip()
    train_prefix_signature = str(
        (prepared_metadata.get("train_prefix_signatures") or {}).get(
            str(train_doc_count),
            "",
        )
        or ""
    ).strip()
    runtime_train_signature = str(
        _pick_metric("train_corpus_signature")
        or prepared_metadata.get("train_corpus_signature", "")
        or ""
    ).strip()
    runtime_val_signature = str(
        _pick_metric("val_corpus_signature")
        or prepared_metadata.get("val_corpus_signature", "")
        or ""
    ).strip()
    runtime_test_signature = str(
        _pick_metric("test_corpus_signature")
        or prepared_metadata.get("test_corpus_signature", "")
        or ""
    ).strip()
    optimization_root_weight = _pick_metric("optimization_root_weight")
    local_law_c1_weight = _pick_metric("local_law_c1_weight")
    local_law_c2_weight = _pick_metric("local_law_c2_weight")
    local_law_c3_weight = _pick_metric("local_law_c3_weight")
    try:
        local_objective_inactive = (
            math.isfinite(float(optimization_root_weight))
            and abs(float(optimization_root_weight) - 1.0) <= 1e-12
            and abs(float(local_law_c1_weight or 0.0)) <= 1e-12
            and abs(float(local_law_c2_weight or 0.0)) <= 1e-12
            and abs(float(local_law_c3_weight or 0.0)) <= 1e-12
        )
    except (TypeError, ValueError):
        local_objective_inactive = False
    root_metric = _pick_metric("test_root_mae_mean")
    root_metric_present = False
    try:
        root_metric_present = math.isfinite(float(root_metric))
    except (TypeError, ValueError):
        root_metric_present = False
    strict_collapse_pass = bool(
        claim_level == CLAIM_LEVEL_EXACT_COLLAPSE
        and bool(completed)
        and not config_diff
        and bool(reference_bundle_source)
        and bool(train_prefix_signature)
        and runtime_train_signature == train_prefix_signature
        and runtime_val_signature
        == str(prepared_metadata.get("val_corpus_signature", "") or "").strip()
        and runtime_test_signature
        == str(prepared_metadata.get("test_corpus_signature", "") or "").strip()
        and bool(local_objective_inactive)
        and bool(root_metric_present)
    )
    nominal_recipe_metadata = dict(job_payload.get("nominal_recipe_metadata") or {})

    row = {
        "job_name": job_name,
        "recipe_id": str(job_payload.get("recipe_id", "") or ""),
        "recipe_display_name": str(
            job_payload.get("recipe_display_name", "")
            or RECIPE_DISPLAY_NAMES.get(str(job_payload.get("recipe_id", "")), "")
        ),
        "tuning_stage": str(job_payload.get("tuning_stage", "") or ""),
        "study_axis": str(job_payload.get("study_axis", "") or ""),
        "axis_value": str(job_payload.get("axis_value", "") or ""),
        "scope_key": str(job_payload.get("scope_key", "") or ""),
        "scope_label": str(job_payload.get("scope_label", "") or ""),
        "nominal_recipe_metadata": dict(nominal_recipe_metadata),
        "nominal_recipe_id": str(
            nominal_recipe_metadata.get("nominal_recipe_id", "")
            or job_payload.get("nominal_recipe_id", "")
            or ""
        ),
        "nominal_recipe_budget_total_calls_per_doc": (
            float(
                nominal_recipe_metadata.get(
                    "nominal_recipe_budget_total_calls_per_doc",
                    job_payload.get(
                        "nominal_recipe_budget_total_calls_per_doc",
                        float("nan"),
                    ),
                )
            )
            if (
                nominal_recipe_metadata.get("nominal_recipe_budget_total_calls_per_doc")
                is not None
                or job_payload.get("nominal_recipe_budget_total_calls_per_doc")
                is not None
            )
            else float("nan")
        ),
        "claim_level": claim_level,
        "benchmark": str(job_payload.get("benchmark", "") or ""),
        "train_doc_count": int(train_doc_count),
        "baseline_family": str(
            job_payload.get("baseline_family", TREE_BASELINE_FAMILY) or TREE_BASELINE_FAMILY
        ),
        "comparison_mode": str(
            metrics.get("comparison_mode", config.get("comparison_mode", "")) or ""
        ),
        "comparison_semantics": str(
            metrics.get("comparison_semantics", "") or ""
        ),
        "comparison_semantics_label": str(
            metrics.get("comparison_semantics_label", "") or ""
        ),
        "run_intent_hash": str(metrics.get("run_intent_hash", "") or ""),
        "run_intent_validation_status": str(
            metrics.get("run_intent_validation_status", "") or ""
        ),
        "seed": int(job_payload.get("seed", 0) or 0),
        "config_label": str(config.get("label", "") or ""),
        "state_dim": int(config.get("state_dim", 0) or 0),
        "hidden_dim": int(config.get("hidden_dim", 0) or 0),
        "tree_training_schedule": str(config.get("tree_training_schedule", "") or ""),
        "tree_checkpoint_metric": str(config.get("tree_checkpoint_metric", "") or ""),
        "tree_stage1_checkpoint_metric": str(
            config.get("tree_stage1_checkpoint_metric", "") or ""
        ),
        "tree_stage1_root_weight": float(config.get("tree_stage1_root_weight", 0.0) or 0.0),
        "tree_local_weighting_mode": str(
            _pick_metric("tree_local_weighting_mode")
            or config.get("tree_local_weighting_mode", "fixed_k_hajek")
            or "fixed_k_hajek"
        ),
        "local_loss_kind": str(_pick_metric("local_loss_kind") or ""),
        "local_sampling_design_name": str(
            _pick_metric("local_sampling_design_name") or ""
        ),
        "leaf_population_size": _pick_metric("leaf_population_size"),
        "leaf_sample_size": _pick_metric("leaf_sample_size"),
        "leaf_effective_propensity": _pick_metric("leaf_effective_propensity"),
        "merge_population_size": _pick_metric("merge_population_size"),
        "merge_sample_size": _pick_metric("merge_sample_size"),
        "merge_effective_propensity": _pick_metric("merge_effective_propensity"),
        "local_objective_audit": dict(
            _pick_metric("local_objective_audit") or {}
        ),
        "optimization_root_weight": optimization_root_weight,
        "local_law_c1_weight": local_law_c1_weight,
        "local_law_c2_weight": local_law_c2_weight,
        "local_law_c3_weight": local_law_c3_weight,
        "leaf_supervision_kind": str(
            _pick_metric("leaf_supervision_kind")
            or config.get("leaf_supervision_kind", "")
            or ""
        ),
        "internal_supervision_kind": str(
            _pick_metric("internal_supervision_kind")
            or config.get("internal_supervision_kind", "none")
            or "none"
        ),
        "leaf_label_rate": _pick_metric("leaf_label_rate"),
        "internal_label_rate": _pick_metric("internal_label_rate"),
        "tree_leaf_fno_width": int(config.get("tree_leaf_fno_width", 0) or 0),
        "tree_leaf_fno_n_modes": int(config.get("tree_leaf_fno_n_modes", 0) or 0),
        "tree_leaf_fno_n_layers": int(config.get("tree_leaf_fno_n_layers", 0) or 0),
        "depth_discount_gamma": float(config.get("depth_discount_gamma", 1.0) or 1.0),
        "fixed_leaf_tokens": int(job_payload.get("fixed_leaf_tokens", 0) or 0),
        "requested_fixed_leaf_tokens": int(
            metrics.get(
                "requested_fixed_leaf_tokens",
                config.get("fixed_leaf_tokens", 0),
            )
            or 0
        ),
        "executed_fixed_leaf_tokens": int(
            metrics.get(
                "executed_fixed_leaf_tokens",
                job_payload.get("fixed_leaf_tokens", 0),
            )
            or 0
        ),
        "slot_count": int(job_payload.get("slot_count", 0) or 0),
        "assumed_doc_tokens": int(job_payload.get("assumed_doc_tokens", ASSUMED_DOC_TOKENS) or ASSUMED_DOC_TOKENS),
        "one_leaf_target": bool(job_payload.get("one_leaf_target", False)),
        "state": str(state),
        "test_root_mae_mean": _pick_metric("test_root_mae_mean"),
        "test_leaf_mae_mean": _pick_metric("test_leaf_mae_mean"),
        "test_merge_mae_mean": _pick_metric("test_merge_mae_mean"),
        "val_root_mae_mean": _pick_metric("val_root_mae_mean"),
        "reference_bundle_source": str(reference_bundle_source),
        "train_prefix_counts": list(prepared_metadata.get("train_prefix_counts") or []),
        "train_prefix_signatures": dict(prepared_metadata.get("train_prefix_signatures") or {}),
        "train_corpus_signature": str(runtime_train_signature),
        "val_corpus_signature": str(runtime_val_signature),
        "test_corpus_signature": str(runtime_test_signature),
        "collapse_runtime_delegate_family": str(
            _pick_metric("collapse_runtime_delegate_family") or ""
        ),
        "collapse_runtime_mode": str(_pick_metric("collapse_runtime_mode") or ""),
        "local_objective_inactive": bool(local_objective_inactive),
        "config_diff_vs_official_fno": dict(config_diff),
        "strict_collapse_pass": bool(strict_collapse_pass),
        "selection_metric_name": str(
            _pick_metric("selection_metric_name") or "val_root_mae_mean"
        ),
        "best_epoch_mean": _pick_metric("best_epoch_mean"),
        "wall_clock_s_mean": _pick_metric("elapsed_s_mean"),
        "prepared_data_metadata_json": str(prepared_metadata.get("metadata_json", "") or ""),
        "source_summary_json": str(job_output_dir / "summary.json"),
        "job_output_dir": str(job_output_dir),
    }
    return annotate_downstream_v3_row(
        row,
        canonical_fno_families=FNO_BASELINE_FAMILIES,
        canonical_fno_fixed_leaf_tokens=FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS,
    )


def _overall_state(
    rows: Sequence[Mapping[str, Any]],
    scheduler_payload: Mapping[str, Any],
) -> str:
    scheduler_state = str(scheduler_payload.get("state", "") or "").strip()
    if scheduler_state:
        return scheduler_state
    if any(str((row or {}).get("state", "")) == "failed" for row in rows):
        return "failed"
    if rows and all(str((row or {}).get("state", "")) == "completed" for row in rows):
        return "completed"
    if any(str((row or {}).get("state", "")) == "completed" for row in rows):
        return "running"
    return "planned"


def load_parity_grid_root(output_root: Path) -> Dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    manifest_path = root / PARITY_MANIFEST_NAME
    summary_path = root / PARITY_SUMMARY_NAME
    manifest = _load_json(manifest_path)
    existing_summary = _load_json(summary_path)
    prior_rows = {
        str((row or {}).get("job_name", "")): dict(row)
        for row in list(existing_summary.get("rows") or [])
        if str((row or {}).get("job_name", "")).strip()
    }
    failed_job_names = _controller_failed_job_names(root)
    rows: List[Dict[str, Any]] = []
    for raw_job in list(manifest.get("jobs") or []):
        if not isinstance(raw_job, Mapping):
            continue
        job_name = str(raw_job.get("job_name", "") or "")
        rows.append(
            row_from_manifest_job(
                raw_job,
                prior_row=prior_rows.get(job_name),
                failed_job_names=failed_job_names,
            )
        )
    scheduler_payload = _load_json(root / "scheduler_status.json")
    completed_items = sum(1 for row in rows if str(row.get("state", "")) == "completed")
    failed_items = sum(1 for row in rows if str(row.get("state", "")) == "failed")
    pending_items = max(0, len(rows) - completed_items - failed_items)
    state = _overall_state(rows, scheduler_payload)
    evidence_status = _infer_root_evidence_status(
        output_root=root,
        state=state,
        rows=rows,
    )
    rows = [{**dict(row), "evidence_status": evidence_status} for row in rows]
    quarantined_rows = [
        dict(row)
        for row in rows
        if str(row.get("contract_status", "") or "") == "legacy_quarantined"
    ]
    percent_complete = (
        float(completed_items) / float(len(rows)) * 100.0
        if rows
        else 0.0
    )
    return {
        "generated_at": _utc_now(),
        "study_name": STUDY_NAME,
        "output_root": str(root),
        "assumed_doc_tokens": int(
            manifest.get("assumed_doc_tokens", ASSUMED_DOC_TOKENS) or ASSUMED_DOC_TOKENS
        ),
        "canonical_train_ladder": [
            int(value)
            for value in list(
                manifest.get("canonical_train_ladder") or _canonical_train_ladder_payload()
            )
        ],
        "one_leaf_target_fixed_leaf_tokens": int(
            manifest.get(
                "one_leaf_target_fixed_leaf_tokens",
                ONE_LEAF_TARGET_FIXED_LEAF_TOKENS,
            )
            or ONE_LEAF_TARGET_FIXED_LEAF_TOKENS
        ),
        "evidence_status": str(evidence_status),
        "contract_gate_status": "fail" if quarantined_rows else "pass",
        "quarantined_row_count": int(len(quarantined_rows)),
        "quarantined_sources": [
            str(row.get("source_summary_json", "") or "")
            for row in quarantined_rows
            if str(row.get("source_summary_json", "") or "").strip()
        ],
        "rows": rows,
        "items_total": int(len(rows)),
        "completed_items": int(
            scheduler_payload.get("completed_items", completed_items) or completed_items
        ),
        "failed_items": int(
            scheduler_payload.get("failed_items", failed_items) or failed_items
        ),
        "active_items": int(scheduler_payload.get("active_items", 0) or 0),
        "pending_items": int(
            scheduler_payload.get("pending_items", pending_items) or pending_items
        ),
        "percent_complete": float(
            scheduler_payload.get("percent_complete", percent_complete) or percent_complete
        ),
        "state": state,
        "scheduler_status_json": str(root / "scheduler_status.json"),
        "parity_grid_manifest_json": str(manifest_path),
        "parity_grid_summary_json": str(summary_path),
        "results_json": str(root / "results.jsonl"),
    }


def parity_results_rows(payload: Mapping[str, Any]) -> List[ResultRow]:
    experiment_id = f"{STUDY_NAME}:{Path(str(payload.get('output_root', ''))).name}"
    result_rows: List[ResultRow] = []
    for row in list(payload.get("rows") or []):
        entry = dict(row or {})
        if str(entry.get("state", "")) != "completed":
            continue
        benchmark_name = str(entry.get("benchmark", "") or "")
        scope_label = str(entry.get("scope_label", "") or "")
        cell_id = ""
        if "::" in benchmark_name:
            _, _, cell_id = benchmark_name.partition("::")
        benchmark_ref = benchmark_ref_from_parts(
            family="markov_supervision_recovery",
            scope=scope_label,
            cell=cell_id,
            dataset_id="markov_full_doc",
            name=benchmark_name,
            metadata={"study_name": STUDY_NAME},
        )
        method_ref = method_ref_from_markov_full_doc_run(
            family=str(entry.get("baseline_family", TREE_BASELINE_FAMILY)),
            variant=(
                f"{str(entry.get('recipe_id', ''))}_leaf"
                f"{int(entry.get('fixed_leaf_tokens', 0) or 0)}"
            ),
            adapter="markov_tree",
            config_like={
                "root_label_rate": 1.0,
                "leaf_supervision_kind": "none",
                "leaf_label_rate": 0.0,
                "internal_supervision_kind": "none",
                "internal_label_rate": 0.0,
                "fixed_leaf_tokens": int(entry.get("fixed_leaf_tokens", 0) or 0),
            },
            package_name="full100",
            metadata={
                "recipe_id": str(entry.get("recipe_id", "") or ""),
                "fixed_leaf_tokens": int(entry.get("fixed_leaf_tokens", 0) or 0),
                "slot_count": int(entry.get("slot_count", 0) or 0),
                "one_leaf_target": bool(entry.get("one_leaf_target", False)),
                "study_name": STUDY_NAME,
            },
        )
        supervision_ref = method_ref.supervision
        artifact_refs = tuple(
            item
            for item in (
                str(entry.get("source_summary_json", "") or "").strip(),
                str(entry.get("job_output_dir", "") or "").strip(),
            )
            if item
        )
        metric_specs = (
            ("test", "root_mae", entry.get("test_root_mae_mean")),
            ("test", "leaf_mae", entry.get("test_leaf_mae_mean")),
            ("test", "merge_mae", entry.get("test_merge_mae_mean")),
            ("val", "root_mae", entry.get("val_root_mae_mean")),
        )
        for split, metric_name, metric_value in metric_specs:
            if metric_value in {None, ""}:
                continue
            result_rows.append(
                ResultRow(
                    experiment_id=experiment_id,
                    phase=STUDY_NAME,
                    benchmark_ref=benchmark_ref,
                    method_ref=method_ref,
                    split=str(split),
                    seed=int(entry.get("seed", 0) or 0),
                    train_docs=int(entry.get("train_doc_count", 0) or 0),
                    supervision_ref=supervision_ref,
                    metric_name=str(metric_name),
                    metric_value=float(metric_value),
                    artifact_refs=artifact_refs,
                    metadata={
                        "recipe_id": str(entry.get("recipe_id", "") or ""),
                        "claim_level": str(entry.get("claim_level", "") or ""),
                        "fixed_leaf_tokens": int(entry.get("fixed_leaf_tokens", 0) or 0),
                        "selection_metric_name": str(
                            entry.get("selection_metric_name", "") or ""
                        ),
                        "best_epoch_mean": entry.get("best_epoch_mean"),
                        "wall_clock_s_mean": entry.get("wall_clock_s_mean"),
                        "scope_key": str(entry.get("scope_key", "") or ""),
                        "scope_label": str(entry.get("scope_label", "") or ""),
                        "reference_bundle_source": str(
                            entry.get("reference_bundle_source", "") or ""
                        ),
                        "strict_collapse_pass": bool(
                            entry.get("strict_collapse_pass", False)
                        ),
                        "evidence_status": str(entry.get("evidence_status", "") or ""),
                    },
                )
            )
    return result_rows


def write_materialized_outputs(output_root: Path) -> Dict[str, Any]:
    payload = load_parity_grid_root(output_root)
    _write_json(output_root / PARITY_SUMMARY_NAME, payload)
    status_payload = {
        "generated_at": payload["generated_at"],
        "study_name": payload["study_name"],
        "output_root": payload["output_root"],
        "state": payload["state"],
        "evidence_status": payload["evidence_status"],
        "items_total": payload["items_total"],
        "completed_items": payload["completed_items"],
        "failed_items": payload["failed_items"],
        "active_items": payload["active_items"],
        "pending_items": payload["pending_items"],
        "percent_complete": payload["percent_complete"],
        "canonical_train_ladder": list(payload.get("canonical_train_ladder") or []),
        "scheduler_status_json": payload["scheduler_status_json"],
        "parity_grid_manifest_json": payload["parity_grid_manifest_json"],
        "parity_grid_summary_json": payload["parity_grid_summary_json"],
        "results_json": payload["results_json"],
        "rows_by_scope": {
            scope: sum(
                1
                for row in list(payload.get("rows") or [])
                if str((row or {}).get("scope_label", "")) == scope
            )
            for scope in ("recoverable", "structural")
        },
        "rows_by_recipe": {
            recipe: sum(
                1
                for row in list(payload.get("rows") or [])
                if str((row or {}).get("recipe_id", "")) == recipe
            )
            for recipe in sorted(
                {
                    str((row or {}).get("recipe_id", ""))
                    for row in list(payload.get("rows") or [])
                    if str((row or {}).get("recipe_id", "")).strip()
                }
            )
        },
        "rows_by_claim_level": {
            claim_level: sum(
                1
                for row in list(payload.get("rows") or [])
                if str((row or {}).get("claim_level", "")) == claim_level
            )
            for claim_level in sorted(
                {
                    str((row or {}).get("claim_level", ""))
                    for row in list(payload.get("rows") or [])
                    if str((row or {}).get("claim_level", "")).strip()
                }
            )
        },
    }
    _write_json(output_root / PARITY_STATUS_NAME, status_payload)
    result_rows = parity_results_rows(payload)
    results_path = output_root / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    append_result_rows(output_root, result_rows)
    merge_artifacts(
        output_root,
        canonical_artifact_refs_from_paths(
            {
                "parity_grid_manifest_json": str(output_root / PARITY_MANIFEST_NAME),
                "parity_grid_status_json": str(output_root / PARITY_STATUS_NAME),
                "parity_grid_summary_json": str(output_root / PARITY_SUMMARY_NAME),
                "parity_grid_results_jsonl": str(output_root / "results.jsonl"),
                "scheduler_status_json": str(output_root / "scheduler_status.json"),
            },
            phase_id=STUDY_NAME,
            required=False,
        ),
    )
    return payload
