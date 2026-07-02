#!/usr/bin/env python3
"""
Train neural operator families with one command.

This orchestrator runs:
1) CTreePO operator training (`scripts/train_ctreepo.py`)
2) Mergeable embedding sketch training (`scripts/train_rile_embedding_sketch.py`)

Use `--ctreepo-args` and `--mergeable-args` for full passthrough flexibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.experiments import (
    ARTIFACT_BEST_CHECKPOINT_PATH,
    ARTIFACT_FINAL_CHECKPOINT_PATH,
    ARTIFACT_METRICS_JSON,
    ARTIFACT_PREDICTIONS_CSV,
    ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON,
    ARTIFACT_SUMMARY_JSON,
    ARTIFACT_TRAINING_RESULT_JSON,
    ExperimentSpec,
    ProgressSnapshot,
    ResultRow,
    append_result_rows,
    benchmark_ref_from_parts,
    canonical_artifact_refs_from_paths,
    default_phase_specs,
    merge_artifacts,
    metadata_with_roles,
    method_ref_from_parts,
    oracle_ref,
    prefixed_artifact_key,
    state_model_role_ref,
    write_experiment_manifest,
    write_experiment_status,
)
from src.experiments.normalization import (
    control_ref_from_ctreepo_local_law_config,
    result_rows_from_scalar_metrics,
    supervision_ref_from_treepo_supervision_spec,
)
from src.training.reproducibility import (
    configure_reproducibility,
    write_reproducibility_manifest,
)
from src.training.search_trace import (
    SearchSpec,
    expand_search_trials,
    fixed_search_spec,
    load_search_spec,
    select_best_trial,
    write_json as write_search_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _operator_method_metadata(family: str, *, task: str) -> Dict[str, Any]:
    return metadata_with_roles(
        {"task": str(task), "method_family": str(family)},
        roles={
            "scorer": {
                "role": "scorer",
                "surface": "native",
                "engine": "pytorch",
                "model": str(family),
            },
            "state_model": state_model_role_ref(
                engine="pytorch",
                model=str(family),
                execution_mode="training",
            ),
        },
        oracle=oracle_ref(kind="training_labels", source=str(task)),
    )


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_output_dir(raw: str | None) -> Path:
    if raw:
        out = Path(raw).expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        return out
    run_id = datetime.now().strftime("neural_operators_%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / run_id).resolve()


def _run_command(label: str, cmd: List[str], log_path: Path) -> Dict[str, Any]:
    logger.info("[%s] running: %s", label, " ".join(shlex.quote(x) for x in cmd))
    started = datetime.now().isoformat()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    ended = datetime.now().isoformat()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"label: {label}",
                f"started_at: {started}",
                f"ended_at: {ended}",
                f"returncode: {proc.returncode}",
                "",
                "=== STDOUT ===",
                proc.stdout or "",
                "",
                "=== STDERR ===",
                proc.stderr or "",
            ]
        ),
        encoding="utf-8",
    )

    if proc.returncode != 0:
        logger.error("[%s] failed (code=%d), see %s", label, proc.returncode, log_path)
    else:
        logger.info("[%s] completed successfully", label)
    return {
        "label": label,
        "returncode": int(proc.returncode),
        "log": str(log_path),
        "started_at": started,
        "ended_at": ended,
    }


def _detect_artifacts(label: str, run_dir: Path) -> Dict[str, Any]:
    """
    Detect primary model artifact paths for each operator family.
    """
    if label == "ctreepo":
        best_path = run_dir / "best.pt"
        final_path = run_dir / "final.pt"
        training_result = run_dir / "training_result.json"
        repro_manifest = run_dir / "reproducibility_manifest.json"
        training_payload = _read_json_if_exists(training_result) or {}
        local_law_summary = (
            training_payload.get("local_law_summary")
            if isinstance(training_payload.get("local_law_summary"), dict)
            else None
        )
        compositional_learning_problem = (
            training_payload.get("compositional_learning_problem")
            if isinstance(training_payload.get("compositional_learning_problem"), dict)
            else (
                local_law_summary.get("compositional_learning_problem")
                if isinstance(local_law_summary, dict)
                and isinstance(local_law_summary.get("compositional_learning_problem"), dict)
                else None
            )
        )
        return {
            "primary_model_path": str(best_path) if best_path.exists() else (str(final_path) if final_path.exists() else None),
            "best_model_path": str(best_path) if best_path.exists() else None,
            ARTIFACT_BEST_CHECKPOINT_PATH: str(best_path) if best_path.exists() else None,
            "final_model_path": str(final_path) if final_path.exists() else None,
            ARTIFACT_FINAL_CHECKPOINT_PATH: str(final_path) if final_path.exists() else None,
            "training_result_path": str(training_result) if training_result.exists() else None,
            ARTIFACT_TRAINING_RESULT_JSON: str(training_result) if training_result.exists() else None,
            "reproducibility_manifest_path": str(repro_manifest) if repro_manifest.exists() else None,
            ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON: str(repro_manifest) if repro_manifest.exists() else None,
            "local_law_summary": local_law_summary,
            "compositional_learning_problem": compositional_learning_problem,
        }
    if label == "mergeable_sketch":
        best_path = run_dir / "checkpoint_best.pt"
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"
        repro_manifest = run_dir / "reproducibility_manifest.json"
        return {
            "primary_model_path": str(best_path) if best_path.exists() else None,
            "best_model_path": str(best_path) if best_path.exists() else None,
            ARTIFACT_BEST_CHECKPOINT_PATH: str(best_path) if best_path.exists() else None,
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            ARTIFACT_METRICS_JSON: str(metrics_path) if metrics_path.exists() else None,
            "predictions_path": str(predictions_path) if predictions_path.exists() else None,
            ARTIFACT_PREDICTIONS_CSV: str(predictions_path) if predictions_path.exists() else None,
            "reproducibility_manifest_path": str(repro_manifest) if repro_manifest.exists() else None,
            ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON: str(repro_manifest) if repro_manifest.exists() else None,
        }
    return {
        "primary_model_path": None,
    }


def _build_ctreepo_local_law_config(args: argparse.Namespace) -> Dict[str, Any]:
    from src.training.local_law_oracles import normalize_local_law_oracle_spec

    oracle_spec = normalize_local_law_oracle_spec(args.ctreepo_local_law_oracle_spec)
    return {
        "root_weight": float(args.ctreepo_root_weight) if args.ctreepo_root_weight is not None else None,
        "leaf_audit_weight": (
            float(args.ctreepo_leaf_audit_weight) if args.ctreepo_leaf_audit_weight is not None else None
        ),
        "merge_audit_weight": (
            float(args.ctreepo_merge_audit_weight) if args.ctreepo_merge_audit_weight is not None else None
        ),
        "violation_threshold": (
            float(args.ctreepo_local_law_violation_threshold)
            if args.ctreepo_local_law_violation_threshold is not None
            else None
        ),
        "require_supervision": bool(args.ctreepo_require_local_law_supervision),
        "oracle_module": oracle_spec,
        "label_source_kind": (
            "task_oracle"
            if oracle_spec == "task"
            else "oracle_callback"
            if oracle_spec
            else "model_backed_teacher"
            if args.ctreepo_local_law_score_port is not None
            else "none"
        ),
        "teacher_port": int(args.ctreepo_local_law_score_port) if args.ctreepo_local_law_score_port is not None else None,
        "teacher_model": str(args.ctreepo_local_law_score_model).strip() if args.ctreepo_local_law_score_model else None,
        "score_port": int(args.ctreepo_local_law_score_port) if args.ctreepo_local_law_score_port is not None else None,
        "score_model": str(args.ctreepo_local_law_score_model).strip() if args.ctreepo_local_law_score_model else None,
        "teacher_max_tokens": (
            int(args.ctreepo_local_law_score_max_tokens)
            if args.ctreepo_local_law_score_max_tokens is not None
            else None
        ),
        "score_max_tokens": (
            int(args.ctreepo_local_law_score_max_tokens)
            if args.ctreepo_local_law_score_max_tokens is not None
            else None
        ),
        "teacher_temperature": (
            float(args.ctreepo_local_law_score_temperature)
            if args.ctreepo_local_law_score_temperature is not None
            else None
        ),
        "score_temperature": (
            float(args.ctreepo_local_law_score_temperature)
            if args.ctreepo_local_law_score_temperature is not None
            else None
        ),
        "allow_model_based_labeling": bool(args.ctreepo_allow_model_based_local_law_scoring),
        "allow_model_based_scoring": bool(args.ctreepo_allow_model_based_local_law_scoring),
    }


def _apply_ctreepo_local_law_args(cmd: List[str], config: Dict[str, Any]) -> None:
    if config.get("root_weight") is not None:
        cmd.extend(["--root-weight", str(float(config["root_weight"]))])
    if config.get("leaf_audit_weight") is not None:
        cmd.extend(["--leaf-audit-weight", str(float(config["leaf_audit_weight"]))])
    if config.get("merge_audit_weight") is not None:
        cmd.extend(["--merge-audit-weight", str(float(config["merge_audit_weight"]))])
    if config.get("violation_threshold") is not None:
        cmd.extend(["--local-law-violation-threshold", str(float(config["violation_threshold"]))])
    if config.get("oracle_module"):
        cmd.extend(["--local-law-oracle", str(config["oracle_module"])])
    if config.get("score_port") is not None:
        cmd.extend(["--local-law-teacher-port", str(int(config["score_port"]))])
    if config.get("score_model"):
        cmd.extend(["--local-law-teacher-model", str(config["score_model"])])
    if config.get("score_max_tokens") is not None:
        cmd.extend(["--local-law-teacher-max-tokens", str(int(config["score_max_tokens"]))])
    if config.get("score_temperature") is not None:
        cmd.extend(["--local-law-teacher-temperature", str(float(config["score_temperature"]))])
    if bool(config.get("require_supervision")):
        cmd.append("--require-local-law-supervision")
    if bool(config.get("allow_model_based_scoring")):
        cmd.append("--allow-model-based-local-law-labeling")


def _treepo_experiment_spec(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    ctreepo_local_law: Mapping[str, Any],
) -> ExperimentSpec:
    benchmark_ref = benchmark_ref_from_parts(
        family="treepo_task",
        scope=str(args.task),
        name=str(args.task),
    )
    control_ref = control_ref_from_ctreepo_local_law_config(
        ctreepo_local_law,
        metadata={"task": str(args.task)},
    )
    supervision_ref = supervision_ref_from_treepo_supervision_spec(
        {
            "unit_selector": "leaves+internal",
            "supervision_kind": "scalar",
            "mode": "label_now" if bool(ctreepo_local_law.get("require_supervision")) else "off",
            "labeler_kind": str(ctreepo_local_law.get("label_source_kind", "") or ""),
            "coverage_label": "tree_local_law_labels" if bool(control_ref and control_ref.enabled) else "",
        },
        metadata={"task": str(args.task)},
    )
    method_refs = []
    if args.which in {"both", "ctreepo"}:
        method_refs.append(
            method_ref_from_parts(
                family="ctreepo",
                variant="local_law_training",
                adapter="treepo_training",
                supervision=supervision_ref,
                control_ref=control_ref,
                metadata=_operator_method_metadata("ctreepo", task=str(args.task)),
            )
        )
    if args.which in {"both", "mergeable_sketch"}:
        method_refs.append(
            method_ref_from_parts(
                family="mergeable_sketch",
                variant="embedding_sketch_training",
                adapter="treepo_training",
                metadata=_operator_method_metadata("mergeable_sketch", task=str(args.task)),
            )
        )
    return ExperimentSpec.create(
        adapter_id="treepo_training",
        output_root=str(output_dir),
        title="train_neural_operators",
        benchmark_refs=(benchmark_ref,),
        method_refs=tuple(method_refs),
        phases=default_phase_specs(("train", "aggregate")),
        report_profiles=("runtime_eval_summary",),
        launch_command=tuple(sys.argv),
        resume_command=tuple(sys.argv),
        metadata={"task": str(args.task), "which": str(args.which)},
    )


def _write_treepo_status(
    output_dir: Path,
    spec: ExperimentSpec,
    *,
    state: str,
    completed_items: int,
    active_items: int,
    pending_items: int,
    failed_items: int,
) -> None:
    items_total = len(spec.method_refs)
    finished = int(completed_items) + int(failed_items)
    percent_complete = 100.0 * float(finished) / float(items_total) if items_total > 0 else 0.0
    write_experiment_status(
        output_dir,
        ProgressSnapshot(
            experiment_id=str(spec.experiment_id),
            state=str(state),
            active_phase="train" if state not in {"completed", "failed"} else "aggregate",
            items_total=int(items_total),
            completed_items=int(completed_items),
            failed_items=int(failed_items),
            active_items=int(active_items),
            pending_items=int(pending_items),
            percent_complete=percent_complete,
            artifact_targets=(
                ARTIFACT_SUMMARY_JSON,
                prefixed_artifact_key("ctreepo", ARTIFACT_TRAINING_RESULT_JSON),
                prefixed_artifact_key("mergeable", ARTIFACT_METRICS_JSON),
                "search_spec_json",
                "search_results_json",
            ),
            metadata={"adapter": "treepo_training"},
        ),
    )


def _treepo_artifacts(output_dir: Path, summary: Mapping[str, Any]) -> list[object]:
    path_map: Dict[str, str] = {
        ARTIFACT_SUMMARY_JSON: str(output_dir / "summary.json"),
    }
    top_level_repro = output_dir / "reproducibility_manifest.json"
    if top_level_repro.exists():
        path_map[ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON] = str(top_level_repro)
    for search_name in ("search_spec.json", "search_results.json"):
        candidate = output_dir / search_name
        if candidate.exists():
            path_map[search_name.replace(".", "_")] = str(candidate)
    for run in list(summary.get("runs") or []):
        if not isinstance(run, dict):
            continue
        label = str(run.get("label", "") or "").strip().lower()
        artifacts = dict(run.get("artifacts") or {})
        if label == "ctreepo":
            for key in (
                "best_model_path",
                "final_model_path",
                "training_result_path",
                ARTIFACT_TRAINING_RESULT_JSON,
                ARTIFACT_BEST_CHECKPOINT_PATH,
                ARTIFACT_FINAL_CHECKPOINT_PATH,
                "reproducibility_manifest_path",
                ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON,
            ):
                value = str(artifacts.get(key, "") or "").strip()
                if value:
                    path_map[f"ctreepo_{key}"] = value
                    if key == ARTIFACT_TRAINING_RESULT_JSON:
                        path_map[prefixed_artifact_key("ctreepo", ARTIFACT_TRAINING_RESULT_JSON)] = value
                    if key == ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON:
                        path_map[prefixed_artifact_key("ctreepo", ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON)] = value
        elif label == "mergeable_sketch":
            for key in (
                "metrics_path",
                ARTIFACT_METRICS_JSON,
                "predictions_path",
                ARTIFACT_PREDICTIONS_CSV,
                "best_model_path",
                ARTIFACT_BEST_CHECKPOINT_PATH,
                "reproducibility_manifest_path",
                ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON,
            ):
                value = str(artifacts.get(key, "") or "").strip()
                if value:
                    path_map[f"mergeable_{key}"] = value
                    if key == ARTIFACT_METRICS_JSON:
                        path_map[prefixed_artifact_key("mergeable", ARTIFACT_METRICS_JSON)] = value
                    if key == ARTIFACT_PREDICTIONS_CSV:
                        path_map[prefixed_artifact_key("mergeable", ARTIFACT_PREDICTIONS_CSV)] = value
                    if key == ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON:
                        path_map[prefixed_artifact_key("mergeable", ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON)] = value
    return canonical_artifact_refs_from_paths(path_map, phase_id="aggregate", required=False)


def _load_operator_search_spec(
    raw_path: Optional[str],
    *,
    label: str,
) -> SearchSpec:
    if raw_path:
        spec = load_search_spec(str(raw_path))
        return SearchSpec(
            mode=spec.mode,
            max_trials=spec.max_trials,
            selection_metric=spec.selection_metric,
            selection_metric_mode=spec.selection_metric_mode,
            tie_breaker_metric=spec.tie_breaker_metric,
            tie_breaker_mode=spec.tie_breaker_mode,
            final_tie_breaker=spec.final_tie_breaker,
            seed_policy=spec.seed_policy,
            dimensions=spec.dimensions,
            metadata={
                **dict(spec.metadata or {}),
                "label": str(label),
            },
        )
    return fixed_search_spec(metadata={"label": str(label)})


def _extract_trial_selection_metrics(
    label: str,
    artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    label = str(label).strip().lower()
    if label == "ctreepo":
        training_result_path = str(artifacts.get("training_result_path", "") or "").strip()
        payload = _read_json_if_exists(Path(training_result_path)) if training_result_path else None
        payload = payload or {}
        eval_metrics = list(payload.get("eval_metrics") or [])
        final_eval = dict(eval_metrics[-1] or {}) if eval_metrics else {}
        validation_mae = payload.get("best_root_mae")
        if validation_mae is None:
            validation_mae = final_eval.get("root_mae")
        return {
            "validation_mae": validation_mae,
            "training_time_seconds": payload.get("training_time_seconds"),
        }
    if label == "mergeable_sketch":
        metrics_path = str(artifacts.get("metrics_path", "") or "").strip()
        payload = _read_json_if_exists(Path(metrics_path)) if metrics_path else None
        payload = payload or {}
        final_payload = dict(payload.get("final", {}) or {})
        final_val = dict(final_payload.get("val", {}) or {})
        validation_mae = payload.get("best_val_mae")
        if validation_mae is None:
            validation_mae = final_val.get("mae")
        return {
            "validation_mae": validation_mae,
            "training_time_seconds": payload.get("training_time_seconds"),
        }
    return {
        "validation_mae": None,
        "training_time_seconds": None,
    }


def _base_method_cmd(
    *,
    py: str,
    script_path: str,
    common_args: Sequence[str],
    extra_args: Optional[str] = None,
) -> List[str]:
    cmd = [str(py), str(script_path), *[str(item) for item in common_args]]
    if extra_args:
        cmd.extend(shlex.split(str(extra_args)))
    return cmd


def _run_method_trials(
    *,
    label: str,
    base_cmd: Sequence[str],
    method_output_dir: Path,
    logs_dir: Path,
    base_seed: int,
    search_spec: SearchSpec,
    fail_fast: bool,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any], bool]:
    search_enabled = bool(search_spec.mode != "fixed" and search_spec.dimensions)
    trials = expand_search_trials(search_spec, base_seed=int(base_seed))
    trial_records: List[Dict[str, Any]] = []
    aborted = False
    for trial in trials:
        trial_id = str(trial.get("trial_id", "trial_000"))
        run_dir = method_output_dir / "trials" / trial_id if search_enabled else method_output_dir
        log_name = f"{label}.{trial_id}.log" if search_enabled else f"{label}.log"
        prefix = [str(item) for item in list(base_cmd[:2])]
        suffix = [str(item) for item in list(base_cmd[2:])]
        cmd = [*prefix, "--output-dir", str(run_dir), *suffix]
        cmd.extend([str(item) for item in list(trial.get("arg_tokens") or ())])
        seed_tokens = [value for value in cmd if str(value) == "--seed"]
        trial_seed = int(trial.get("seed", 42) or 42)
        if search_enabled or not seed_tokens:
            cmd.extend(["--seed", str(trial_seed)])
        result = _run_command(label, cmd, logs_dir / log_name)
        artifacts = _detect_artifacts(label, run_dir)
        selection_metrics = _extract_trial_selection_metrics(label, artifacts)
        trial_record = {
            "trial_id": trial_id,
            "trial_index": int(trial.get("trial_index", 0) or 0),
            "seed": trial_seed,
            "overrides": list(trial.get("overrides") or ()),
            "arg_tokens": list(trial.get("arg_tokens") or ()),
            "command": list(cmd),
            "run_dir": str(run_dir),
            "log": str(result.get("log", "")),
            "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"),
            "returncode": int(result.get("returncode", 1)),
            "success": int(result.get("returncode", 1)) == 0,
            "artifacts": artifacts,
            "selection_metrics": selection_metrics,
            "search_enabled": bool(search_enabled),
        }
        trial_records.append(trial_record)
        if fail_fast and not trial_record["success"]:
            aborted = True
            break

    selected = select_best_trial(
        trial_records,
        selection_metric=str(search_spec.selection_metric),
        selection_metric_mode=str(search_spec.selection_metric_mode),
        tie_breaker_metric=str(search_spec.tie_breaker_metric),
        tie_breaker_mode=str(search_spec.tie_breaker_mode),
    )
    selected_run: Optional[Dict[str, Any]] = None
    if selected is not None:
        selected_run = {
            "label": str(label),
            "returncode": int(selected.get("returncode", 1)),
            "log": str(selected.get("log", "")),
            "started_at": selected.get("started_at"),
            "ended_at": selected.get("ended_at"),
            "run_dir": str(selected.get("run_dir", "")),
            "artifacts": dict(selected.get("artifacts") or {}),
            "selection_metrics": dict(selected.get("selection_metrics") or {}),
            "trial_id": str(selected.get("trial_id", "")),
            "trial_index": int(selected.get("trial_index", 0) or 0),
            "search_enabled": bool(search_enabled),
        }
    payload = {
        "label": str(label),
        "search_enabled": bool(search_enabled),
        "spec": search_spec.to_dict(),
        "selection_rule": {
            "selection_metric": str(search_spec.selection_metric),
            "selection_metric_mode": str(search_spec.selection_metric_mode),
            "tie_breaker_metric": str(search_spec.tie_breaker_metric),
            "tie_breaker_mode": str(search_spec.tie_breaker_mode),
            "final_tie_breaker": str(search_spec.final_tie_breaker),
            "seed_policy": str(search_spec.seed_policy),
        },
        "trials": trial_records,
        "selected_trial_id": None if selected is None else str(selected.get("trial_id", "")),
        "selected_run_dir": None if selected_run is None else str(selected_run.get("run_dir", "")),
        "successful_trials": int(sum(1 for trial in trial_records if bool(trial.get("success", False)))),
        "failed_trials": int(sum(1 for trial in trial_records if not bool(trial.get("success", False)))),
    }
    return selected_run, payload, aborted


def _treepo_result_rows(
    *,
    spec: ExperimentSpec,
    summary: Mapping[str, Any],
) -> list[object]:
    benchmark_ref = benchmark_ref_from_parts(
        family="treepo_task",
        scope=str(summary.get("task", "manifesto_rile") or "manifesto_rile"),
        name=str(summary.get("task", "manifesto_rile") or "manifesto_rile"),
    )
    control_ref = control_ref_from_ctreepo_local_law_config(
        dict(summary.get("ctreepo_local_law") or {}),
        metadata={"task": str(summary.get("task", "") or "")},
    )
    supervision_ref = supervision_ref_from_treepo_supervision_spec(
        {
            "unit_selector": "leaves+internal",
            "supervision_kind": "scalar",
            "mode": "label_now" if bool(dict(summary.get("ctreepo_local_law") or {}).get("require_supervision")) else "off",
            "labeler_kind": str(dict(summary.get("ctreepo_local_law") or {}).get("label_source_kind", "") or ""),
            "coverage_label": "tree_local_law_labels" if bool(control_ref and control_ref.enabled) else "",
        }
    )
    rows: list[object] = []
    for run in list(summary.get("runs") or []):
        if not isinstance(run, dict):
            continue
        label = str(run.get("label", "") or "").strip().lower()
        family = "ctreepo" if label == "ctreepo" else "mergeable_sketch"
        method_ref = method_ref_from_parts(
            family=family,
            variant="local_law_training" if family == "ctreepo" else "embedding_sketch_training",
            adapter="treepo_training",
            supervision=supervision_ref if family == "ctreepo" else None,
            control_ref=control_ref if family == "ctreepo" else None,
            metadata=_operator_method_metadata(
                family,
                task=str(summary.get("task", "manifesto_rile") or "manifesto_rile"),
            ),
        )
        rows.append(
            {
                "experiment_id": str(spec.experiment_id),
                "phase": "train",
                "benchmark_ref": benchmark_ref.to_dict(),
                "method_ref": method_ref.to_dict(),
                "metric_name": "returncode",
                "metric_value": int(run.get("returncode", 1)),
                "artifact_refs": ("summary_json",),
                "metadata": {
                    "label": label,
                    "trial_id": str(run.get("trial_id", "") or ""),
                    "search_enabled": bool(run.get("search_enabled", False)),
                },
            }
        )
        artifacts = dict(run.get("artifacts") or {})
        if family == "ctreepo":
            training_result_path = str(artifacts.get("training_result_path", "") or "").strip()
            if training_result_path and Path(training_result_path).exists():
                training_payload = _read_json_if_exists(Path(training_result_path)) or {}
                base_row = ResultRow(
                    experiment_id=str(spec.experiment_id),
                    phase="train",
                    benchmark_ref=benchmark_ref,
                    method_ref=method_ref,
                    supervision_ref=supervision_ref,
                    control_ref=control_ref,
                    artifact_refs=("ctreepo_training_result_path", "summary_json"),
                )
                rows.extend(
                    result_rows_from_scalar_metrics(
                        base_row=base_row,
                        metrics=training_payload,
                        allowed_keys=("best_epoch", "best_root_mae", "training_time_seconds", "epochs_completed"),
                        metadata={
                            "label": label,
                            "trial_id": str(run.get("trial_id", "") or ""),
                            "search_enabled": bool(run.get("search_enabled", False)),
                        },
                    )
                )
                eval_metrics = list(training_payload.get("eval_metrics") or [])
                if eval_metrics:
                    final_eval = dict(eval_metrics[-1] or {})
                    rows.extend(
                        result_rows_from_scalar_metrics(
                            base_row=ResultRow(
                                experiment_id=str(spec.experiment_id),
                                phase="eval",
                                benchmark_ref=benchmark_ref,
                                method_ref=method_ref,
                                split="validation",
                                supervision_ref=supervision_ref,
                                control_ref=control_ref,
                                artifact_refs=("ctreepo_training_result_path", "summary_json"),
                            ),
                            metrics=final_eval,
                            allowed_keys=(
                                "root_mae",
                                "root_mse",
                                "root_mae_normalized",
                                "node_oracle_label_rate",
                                "node_oracle_mae",
                                "leaf_oracle_mae",
                                "merge_oracle_mae",
                                "leaf_violation_rate",
                                "merge_violation_rate",
                            ),
                            metadata={
                                "label": label,
                                "trial_id": str(run.get("trial_id", "") or ""),
                                "search_enabled": bool(run.get("search_enabled", False)),
                            },
                        )
                    )
        elif family == "mergeable_sketch":
            metrics_path = str(artifacts.get("metrics_path", "") or "").strip()
            if metrics_path and Path(metrics_path).exists():
                metrics_payload = _read_json_if_exists(Path(metrics_path)) or {}
                rows.extend(
                    result_rows_from_scalar_metrics(
                        base_row=ResultRow(
                            experiment_id=str(spec.experiment_id),
                            phase="eval",
                            benchmark_ref=benchmark_ref,
                            method_ref=method_ref,
                            artifact_refs=("mergeable_metrics_path", "summary_json"),
                        ),
                        metrics=metrics_payload,
                        metadata={
                            "label": label,
                            "trial_id": str(run.get("trial_id", "") or ""),
                            "search_enabled": bool(run.get("search_enabled", False)),
                        },
                    )
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CTreePO and mergeable-sketch operators in one run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--task", type=str, default="manifesto_rile")
    parser.add_argument(
        "--which",
        choices=["both", "ctreepo", "mergeable_sketch"],
        default="both",
        help="Which operator family to train.",
    )
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctreepo-root-weight", type=float, default=None)
    parser.add_argument("--ctreepo-leaf-audit-weight", type=float, default=None)
    parser.add_argument("--ctreepo-merge-audit-weight", type=float, default=None)
    parser.add_argument("--ctreepo-local-law-violation-threshold", type=float, default=None)
    parser.add_argument(
        "--ctreepo-local-law-oracle",
        "--ctreepo-local-law-oracle-module",
        dest="ctreepo_local_law_oracle_spec",
        type=str,
        default=None,
        help=(
            "Node-span label source for CTreePO local-law labels. Use 'task' for the task/teacher-provided "
            "oracle, or module.path:function_name for an explicit callback."
        ),
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-port",
        "--ctreepo-local-law-score-port",
        dest="ctreepo_local_law_score_port",
        type=int,
        default=None,
        help="Optional model-backed teacher endpoint for node-span labels. Fallback only.",
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-model",
        "--ctreepo-local-law-score-model",
        dest="ctreepo_local_law_score_model",
        type=str,
        default=None,
        help="Optional model override for the model-backed teacher labeler.",
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-max-tokens",
        "--ctreepo-local-law-score-max-tokens",
        dest="ctreepo_local_law_score_max_tokens",
        type=int,
        default=None,
        help="Max tokens for model-backed teacher labeling.",
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-temperature",
        "--ctreepo-local-law-score-temperature",
        dest="ctreepo_local_law_score_temperature",
        type=float,
        default=None,
        help="Temperature for model-backed teacher labeling.",
    )
    parser.add_argument("--ctreepo-require-local-law-supervision", action="store_true")
    parser.add_argument(
        "--ctreepo-allow-model-based-local-law-labeling",
        "--ctreepo-allow-model-based-local-law-scoring",
        dest="ctreepo_allow_model_based_local_law_scoring",
        action="store_true",
        help="Explicitly allow model-backed teacher labeling for local-law supervision.",
    )
    parser.add_argument(
        "--ctreepo-args",
        type=str,
        default="--pilot",
        help="Extra args forwarded to scripts/train_ctreepo.py",
    )
    parser.add_argument(
        "--mergeable-args",
        type=str,
        default="",
        help="Extra args forwarded to scripts/train_rile_embedding_sketch.py",
    )
    parser.add_argument(
        "--ctreepo-search-spec",
        type=str,
        default=None,
        help=(
            "Optional JSON search spec for CTreePO. When omitted, the run is recorded as a single fixed trial."
        ),
    )
    parser.add_argument(
        "--mergeable-search-spec",
        type=str,
        default=None,
        help=(
            "Optional JSON search spec for mergeable sketch training. When omitted, the run is recorded as a single fixed trial."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failure.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    applied_repro = configure_reproducibility(int(args.seed))
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logger.info("Output dir: %s", output_dir)
    if (
        args.ctreepo_local_law_oracle_spec
        and str(args.ctreepo_local_law_oracle_spec).strip().lower() != "task"
        and args.ctreepo_local_law_score_port is not None
    ):
        parser.error(
            "Choose one CTreePO local-law label source: --ctreepo-local-law-oracle "
            "or --ctreepo-local-law-teacher-port, not both."
        )
    ctreepo_local_law = _build_ctreepo_local_law_config(args)
    experiment_spec = _treepo_experiment_spec(
        output_dir=output_dir,
        args=args,
        ctreepo_local_law=ctreepo_local_law,
    )
    repro_manifest_path = write_reproducibility_manifest(
        output_dir,
        seed=int(args.seed),
        cli_args=vars(args),
        config={"ctreepo_local_law": ctreepo_local_law},
        applied=applied_repro,
        extra={
            "which": str(args.which),
            "ctreepo_args": str(args.ctreepo_args),
            "mergeable_args": str(args.mergeable_args),
            "ctreepo_search_spec": str(args.ctreepo_search_spec or ""),
            "mergeable_search_spec": str(args.mergeable_search_spec or ""),
        },
    )
    logger.info("Reproducibility manifest: %s", repro_manifest_path)
    write_experiment_manifest(output_dir, experiment_spec)
    _write_treepo_status(
        output_dir,
        experiment_spec,
        state="running",
        completed_items=0,
        active_items=1 if args.which else 0,
        pending_items=len(experiment_spec.method_refs),
        failed_items=0,
    )

    py = sys.executable
    common: List[str] = []
    if args.embedding_url:
        common.extend(["--embedding-url", str(args.embedding_url)])
    if args.embedding_model:
        common.extend(["--embedding-model", str(args.embedding_model)])
    if args.task:
        common.extend(["--task", str(args.task)])
    if args.seed is not None:
        common.extend(["--seed", str(int(args.seed))])

    runs: List[Dict[str, Any]] = []
    search_methods: Dict[str, Any] = {}
    ctreepo_search = _load_operator_search_spec(
        args.ctreepo_search_spec,
        label="ctreepo",
    )
    mergeable_search = _load_operator_search_spec(
        args.mergeable_search_spec,
        label="mergeable_sketch",
    )

    if args.which in {"both", "ctreepo"}:
        ctreepo_out = output_dir / "ctreepo"
        ctreepo_base_cmd = _base_method_cmd(
            py=py,
            script_path="scripts/train_ctreepo.py",
            common_args=common,
            extra_args=args.ctreepo_args,
        )
        _apply_ctreepo_local_law_args(ctreepo_base_cmd, ctreepo_local_law)
        selected_run, search_payload, aborted = _run_method_trials(
            label="ctreepo",
            base_cmd=ctreepo_base_cmd,
            method_output_dir=ctreepo_out,
            logs_dir=logs_dir,
            base_seed=int(args.seed),
            search_spec=ctreepo_search,
            fail_fast=bool(args.fail_fast),
        )
        search_methods["ctreepo"] = search_payload
        if selected_run is not None:
            runs.append(selected_run)
        _write_treepo_status(
            output_dir,
            experiment_spec,
            state="running",
            completed_items=sum(1 for row in runs if int(row.get("returncode", 1)) == 0),
            active_items=1 if args.which in {"both", "mergeable_sketch"} else 0,
            pending_items=max(0, len(experiment_spec.method_refs) - len(runs)),
            failed_items=sum(1 for row in runs if int(row.get("returncode", 1)) != 0),
        )
        if aborted:
            failure_summary = {
                "runs": runs,
                "search": {"methods": search_methods},
            }
            write_search_json(output_dir / "search_spec.json", {
                "created_at": datetime.now().isoformat(),
                "methods": {key: value.get("spec", {}) for key, value in search_methods.items()},
            })
            write_search_json(output_dir / "search_results.json", {
                "created_at": datetime.now().isoformat(),
                "methods": search_methods,
            })
            (output_dir / "summary.json").write_text(json.dumps(failure_summary, indent=2), encoding="utf-8")
            merge_artifacts(output_dir, _treepo_artifacts(output_dir, failure_summary))
            _write_treepo_status(
                output_dir,
                experiment_spec,
                state="failed",
                completed_items=sum(1 for row in runs if int(row.get("returncode", 1)) == 0),
                active_items=0,
                pending_items=max(0, len(experiment_spec.method_refs) - len(runs)),
                failed_items=sum(1 for row in runs if int(row.get("returncode", 1)) != 0),
            )
            return 1

    if args.which in {"both", "mergeable_sketch"}:
        merge_out = output_dir / "mergeable_sketch"
        mergeable_base_cmd = _base_method_cmd(
            py=py,
            script_path="scripts/train_rile_embedding_sketch.py",
            common_args=common,
            extra_args=args.mergeable_args,
        )
        selected_run, search_payload, aborted = _run_method_trials(
            label="mergeable_sketch",
            base_cmd=mergeable_base_cmd,
            method_output_dir=merge_out,
            logs_dir=logs_dir,
            base_seed=int(args.seed),
            search_spec=mergeable_search,
            fail_fast=bool(args.fail_fast),
        )
        search_methods["mergeable_sketch"] = search_payload
        if selected_run is not None:
            runs.append(selected_run)
        _write_treepo_status(
            output_dir,
            experiment_spec,
            state="running",
            completed_items=sum(1 for row in runs if int(row.get("returncode", 1)) == 0),
            active_items=0,
            pending_items=max(0, len(experiment_spec.method_refs) - len(runs)),
            failed_items=sum(1 for row in runs if int(row.get("returncode", 1)) != 0),
        )
        if aborted:
            failure_summary = {
                "runs": runs,
                "search": {"methods": search_methods},
            }
            write_search_json(output_dir / "search_spec.json", {
                "created_at": datetime.now().isoformat(),
                "methods": {key: value.get("spec", {}) for key, value in search_methods.items()},
            })
            write_search_json(output_dir / "search_results.json", {
                "created_at": datetime.now().isoformat(),
                "methods": search_methods,
            })
            (output_dir / "summary.json").write_text(json.dumps(failure_summary, indent=2), encoding="utf-8")
            merge_artifacts(output_dir, _treepo_artifacts(output_dir, failure_summary))
            _write_treepo_status(
                output_dir,
                experiment_spec,
                state="failed",
                completed_items=sum(1 for row in runs if int(row.get("returncode", 1)) == 0),
                active_items=0,
                pending_items=max(0, len(experiment_spec.method_refs) - len(runs)),
                failed_items=sum(1 for row in runs if int(row.get("returncode", 1)) != 0),
            )
            return 1

    write_search_json(
        output_dir / "search_spec.json",
        {
            "created_at": datetime.now().isoformat(),
            "methods": {key: value.get("spec", {}) for key, value in search_methods.items()},
        },
    )
    write_search_json(
        output_dir / "search_results.json",
        {
            "created_at": datetime.now().isoformat(),
            "methods": search_methods,
        },
    )
    summary = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "task": str(args.task),
        "which": args.which,
        "common_args": common,
        "ctreepo_local_law": ctreepo_local_law,
        "search": {
            "methods": search_methods,
        },
        "runs": runs,
        "all_success": bool(
            all(
                str(method_ref.family) in search_methods
                and str(search_methods[str(method_ref.family)].get("selected_trial_id", "") or "").strip()
                for method_ref in experiment_spec.method_refs
            )
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    merge_artifacts(output_dir, _treepo_artifacts(output_dir, summary))
    append_result_rows(
        output_dir,
        _treepo_result_rows(
            spec=experiment_spec,
            summary=summary,
        ),
    )
    _write_treepo_status(
        output_dir,
        experiment_spec,
        state="completed" if bool(summary["all_success"]) else "failed",
        completed_items=sum(1 for row in runs if int(row.get("returncode", 1)) == 0),
        active_items=0,
        pending_items=0,
        failed_items=sum(1 for row in runs if int(row.get("returncode", 1)) != 0),
    )

    if summary["all_success"]:
        logger.info("All requested operator trainings completed successfully.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
