#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.structured_config import load_structured_config, write_structured_config
from src.experiments.script_parse import (
    parse_float_list as _shared_parse_float_list,
    parse_int_list as _shared_parse_int_list,
    parse_str_list as _shared_parse_str_list,
)
from src.ctreepo.contracts import (
    LAW_SET_ALL,
    LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
    RunAxisSpec,
    assert_public_contract_clean,
    canonical_law_set_id,
    markov_tree_bundle_metadata,
    objective_metadata,
    run_manifest_metadata,
)
from src.ctreepo.sim.core.tree_reference_presets import (
    COMPARISON_GRID_V3_PRESET,
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
from src.experiments.scheduler import SchedulerConfig, SchedulerItem, run_scheduler
from src.ctreepo.sim.core.tree_neural_facade import job_output_dir_name
from src.ctreepo.sim.core.tree_neural_execution import worker_command_for_job

TRADEOFF_PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"
TREE_FULL_DOC_SCRIPT = REPO_ROOT / "scripts" / "run_tree_neural_full_doc_mig.py"
TREE_FNO_TUNING_PDF_SCRIPT = REPO_ROOT / "scripts" / "report_tree_fno_tuning_pdf.py"
FULL_DOC_DIAGNOSTIC_PDF_SCRIPT = REPO_ROOT / "scripts" / "report_full_doc_anchor_diagnostics_pdf.py"
LONG_JOB_SCRIPT = REPO_ROOT / "scripts" / "long_job.py"
CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES = ("official_fno", "official_fno_sumlen")
DEFAULT_PARITY_METHOD_RUNS = (
    "tree_neural:on_range_idempotence_only",
    "tree_neural:merge_and_on_range_idempotence",
    "tree_neural:all",
)
DEFAULT_REFERENCE_METHOD_RUNS = tuple(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)

DEFAULT_TRADEOFF_PHASES = "supervision_recovery,report"
DEFAULT_CAPACITY_WIDTHS = (64, 128, 256)
DEFAULT_CAPACITY_MODES = (2, 4, 8)
DEFAULT_CAPACITY_LAYERS = (2, 4, 6)
DEFAULT_CAPACITY_SCREEN_SEEDS = (0, 1, 2)
DEFAULT_CAPACITY_LOCKED_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_PARITY_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_PARITY_SCALE_TRAIN_DOCS = (1024, 2048, 3072, 4096, 5120, 8192, 10240)
DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS = (0.25, 1.0)


def _markov_publication_tree_bundle_contract(
    *,
    args: argparse.Namespace,
    phases: Sequence[str] | set[str] | None = None,
    runner: str = "run_markov_publication_bundle",
) -> Dict[str, Any]:
    return markov_tree_bundle_metadata(
        leaf_policy={
            "partition_axis": "synthetic_markov_document",
            "phases": sorted(str(phase) for phase in (phases or ())),
            "preset": str(getattr(args, "preset", "")),
        },
        state_dim=(
            int(getattr(args, "state_dim"))
            if getattr(args, "state_dim", None) is not None
            else None
        ),
        f_init="official_oracle",
        g_init="raw_concat",
        schedule="balanced",
        metadata={"runner": runner},
    )


def _manifest_local_law_weight(
    args: argparse.Namespace,
    *,
    attr: str,
) -> float | None:
    raw = getattr(args, attr, None)
    return float(raw) if raw is not None else None


def _markov_publication_run_manifest(
    *,
    args: argparse.Namespace,
    output_root: Path,
    phases: Sequence[str] | set[str] | None,
    tree_bundle_contract: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    status: str = "completed",
    publication_ready: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    artifact_rows = [
        {"kind": str(kind), "uri": str(uri)}
        for kind, uri in sorted(dict(artifacts or {}).items())
        if str(uri or "")
    ]
    artifact_rows.append({"kind": "run_directory", "uri": str(output_root)})
    local_law_weight = _manifest_local_law_weight(
        args,
        attr="local_law_weight",
    )
    if local_law_weight is None and getattr(args, "tree_local_law_weight", None) is not None:
        local_law_weight = _manifest_local_law_weight(args, attr="tree_local_law_weight")
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
        run_id="markov.publication_bundle",
        domain="markov",
        role="publication_bundle",
        backend="fno",
        status=str(status),
        tree_bundle=tree_bundle_contract,
        f_init="official_oracle",
        g_init="raw_concat",
        f_lineage={"init": "official_oracle", "artifact": "synthetic_oracle"},
        g_lineage={"init": "raw_concat", "artifact": "raw_concat"},
        schedule=str(getattr(args, "tree_training_schedule", "balanced") or "balanced"),
        objective=objective_metadata(
            objective_family="markov_publication_bundle",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
            local_law_weight=local_law_weight,
            root_share=root_weight,
            local_law_component_weights={
                "markov_local_law": float(local_law_weight or 0.0)
            },
            metadata={
                "preset": str(getattr(args, "preset", "")),
                "phases": sorted(str(phase) for phase in (phases or ())),
            },
        ),
        optimizer_config={
            "phases": sorted(str(phase) for phase in (phases or ())),
            "preset": str(getattr(args, "preset", "")),
            "with_preflight": bool(getattr(args, "with_preflight", False)),
            "preflight_only": bool(getattr(args, "preflight_only", False)),
        },
        output_artifacts=artifact_rows,
        audit_results={"ok": True, "policy": "manifest_contract_required"},
        quarantine={"classification": "valid_treebundle_v1"},
        command=sys.argv,
        allow_legacy=False,
        publication_ready=bool(publication_ready),
        metadata={
            "runner": "scripts/run_markov_publication_bundle.py",
            **dict(metadata or {}),
        },
    )


PREFLIGHT_CAPACITY_TRAIN_DOC_COUNT = 1024
PREFLIGHT_CAPACITY_SCREEN_SEEDS = (0,)
PREFLIGHT_CAPACITY_LOCKED_SEEDS = (0,)
PREFLIGHT_CAPACITY_TOP_K = 1
PREFLIGHT_CAPACITY_WIDTHS = (64,)
PREFLIGHT_CAPACITY_MODES = (2,)
PREFLIGHT_CAPACITY_LAYERS = (2,)
PREFLIGHT_TRADEOFF_TRAIN_DOCS = (1024,)
PREFLIGHT_TRADEOFF_SEEDS = (0,)
PREFLIGHT_TRADEOFF_PACKAGES = ("full100",)
PREFLIGHT_PARITY_GATE_TRAIN_DOC_COUNT = 1024
PREFLIGHT_PARITY_SCALE_TRAIN_DOCS = (1024,)
PREFLIGHT_PARITY_SEEDS = (0,)
PREFLIGHT_PARITY_UPPER_BOUND_AUX_FRACTIONS = (1.0,)
DEFAULT_PHASES = (
    "tradeoff",
    "capacity",
    "parity",
    "bundle",
)
ARCHIVED_REPORT_PHASES = frozenset({"tree_fno_pdf", "full_doc_parity_pdf"})


def _template_run_axis(token: str, *, role: str) -> Dict[str, Any]:
    text = str(token or "").strip()
    if ":" in text:
        method_id, law_set_id = text.split(":", 1)
    else:
        method_id, law_set_id = text, LAW_SET_ALL
    return {
        "problem_id": "markov_ops_count",
        "method_id": str(method_id).strip(),
        "law_set_id": canonical_law_set_id(str(law_set_id).strip() or LAW_SET_ALL),
        "role": str(role),
    }


def _template_run_axes(tokens: Sequence[str], *, role: str) -> List[Dict[str, Any]]:
    return [_template_run_axis(token, role=role) for token in tokens]


PUBLICATION_SELECTION_TEMPLATE: Dict[str, Any] = {
    "publication_bundle": {
        "phases": list(DEFAULT_PHASES),
        "with_preflight": True,
        "reuse_existing": True,
        "tradeoff": {
            "preset": "v3",
            "device_mode": "cuda",
            "phases": DEFAULT_TRADEOFF_PHASES.split(","),
            "train_docs": 10240,
            "supervision_recovery_train_docs": [1024, 4096, 10240],
            "supervision_recovery_seeds": [0, 1],
            "supervision_recovery_method_id": "tree_neural",
            "supervision_recovery_structural_cell": "r12_p079",
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
        },
        "capacity": {
            "benchmark": "recoverable_v5",
            "train_doc_count": 10240,
            "screen_seeds": list(DEFAULT_CAPACITY_SCREEN_SEEDS),
            "locked_seeds": list(DEFAULT_CAPACITY_LOCKED_SEEDS),
            "top_k": 3,
            "widths": list(DEFAULT_CAPACITY_WIDTHS),
            "modes": list(DEFAULT_CAPACITY_MODES),
            "layers": list(DEFAULT_CAPACITY_LAYERS),
            "runtime": {
                "data_mode": "resident",
                "bucket_mode": "exact_then_bucketed",
                "preload_splits": ["train", "val", "test"],
                "preload_targets": True,
                "workers_per_mig": 1,
                "allow_multi_worker_screen": True,
                "capacity_workers_per_mig": 2,
            },
        },
        "parity": {
            "benchmark": "recoverable_v5_t128",
            "gate_train_doc_count": 10240,
            "scale_train_doc_counts": list(DEFAULT_PARITY_SCALE_TRAIN_DOCS),
            "seeds": list(DEFAULT_PARITY_SEEDS),
            "method_runs": _template_run_axes(DEFAULT_PARITY_METHOD_RUNS, role="primary"),
            "reference_method_runs": _template_run_axes(DEFAULT_REFERENCE_METHOD_RUNS, role="reference"),
            "run_aux_upper_bound": True,
            "upper_bound_aux_fractions": list(DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS),
            "backfill_on_success": True,
            "runtime": {
                "data_mode": "resident",
                "bucket_mode": "exact_then_bucketed",
                "preload_splits": ["train", "val", "test"],
                "preload_targets": True,
                "workers_per_mig": 1,
                "allow_multi_worker_screen": False,
                "capacity_workers_per_mig": 2,
            },
        },
        "render": {
            "render_full_doc_parity_pdf": True,
        },
        "scheduler": {
            "mode": "global_per_run",
            "default_job_granularity": "family_train_seed",
            "cleanup_stale_children": True,
            "max_gpu_items_per_mig": 1,
        },
    },
    "tradeoff_pipeline": {
        "preset": "v3",
        "phases": [
            "supervision_recovery",
            "report",
        ],
        "train_docs": 10240,
        "val_docs": 1024,
        "test_docs": 1024,
        "supervision_recovery_train_docs": [1024, 4096, 10240],
        "supervision_recovery_seeds": [0, 1],
        "supervision_recovery_method_id": "tree_neural",
            "supervision_recovery_structural_cell": "r12_p079",
        "medium_batch_sizes": [128, 256, 512],
        "medium_seeds": [0, 1, 2, 3],
        "law_set_ids": ["root_only", "on_range_idempotence_only", "all"],
        "support_modes": ["supported", "unsupported"],
        "full_doc_anchor_reference_method_runs": _template_run_axes(DEFAULT_REFERENCE_METHOD_RUNS, role="reference"),
        "efficiency_anchor_mode": "both",
        "efficiency_train_docs": [2048, 4096],
        "efficiency_anchor_train_docs_dense": [256, 512, 768, 1024, 1536, 2048, 3072, 4096],
        "efficiency_anchor_seeds": [0, 1, 2, 3, 4],
        "efficiency_hardness_grid": "structural_core_v2",
        "efficiency_structural_cells": ["r4_p031", "r12_p031", "r4_p079", "r12_p079"],
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
    },
}


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    wall_clock_s: float
    command: List[str]
    log_path: str
    output_root: str
    expected_outputs: List[str]


def _parse_int_list(text: str | None, default: Sequence[int]) -> List[int]:
    return _shared_parse_int_list(text, default=default, separators=",")


def _parse_float_list(text: str | None, default: Sequence[float]) -> List[float]:
    return _shared_parse_float_list(text, default=default, separators=",")


def _parse_phase_set(text: str | None) -> List[str]:
    if text is None:
        return list(DEFAULT_PHASES)
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out or list(DEFAULT_PHASES)


def _parse_str_list(text: str | None, default: Sequence[str]) -> List[str]:
    return _shared_parse_str_list(text, default=default, separators=",")


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


def _parity_backend_tree_families(args: argparse.Namespace) -> List[str]:
    method_runs = _parse_run_axis_list(
        getattr(args, "parity_method_runs", None),
        DEFAULT_PARITY_METHOD_RUNS,
        role="primary",
    )
    return _parse_str_list(
        getattr(args, "parity_tree_families", None),
        _legacy_families_from_run_axes(method_runs),
    )


def _parity_backend_reference_families(args: argparse.Namespace) -> List[str]:
    reference_runs = _parse_run_axis_list(
        getattr(args, "parity_reference_method_runs", None),
        DEFAULT_REFERENCE_METHOD_RUNS,
        role="reference",
    )
    return _parse_str_list(
        getattr(args, "parity_fno_families", None),
        _method_ids_from_run_axes(reference_runs),
    )


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


def _flatten_publication_bundle_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    prefix_map = {
        "tradeoff": "tradeoff_",
        "capacity": "capacity_",
        "parity": "parity_",
        "render": "",
        "launcher": "detach_",
    }
    for key, value in payload.items():
        section_prefix = prefix_map.get(str(key))
        if section_prefix is not None and isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if str(sub_key) in {"runtime", "tree_reference"} and isinstance(sub_value, Mapping):
                    nested_prefix = f"{section_prefix}{str(sub_key)}_"
                    for runtime_key, runtime_value in sub_value.items():
                        flat[f"{nested_prefix}{str(runtime_key)}"] = runtime_value
                    continue
                if str(sub_key) == "scheduler" and isinstance(sub_value, Mapping):
                    for scheduler_key, scheduler_value in sub_value.items():
                        flat[f"scheduler_{str(scheduler_key)}"] = scheduler_value
                    continue
                flat[f"{section_prefix}{sub_key}"] = sub_value
            continue
        flat[str(key)] = value
    return flat


def _load_selection_config(
    path: Path | None,
    *,
    section_names: Sequence[str],
    flatten_sections: bool = False,
) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = load_structured_config(path)
    assert_public_contract_clean(
        _normalize_run_axis_config_aliases(payload),
        surface=str(path),
    )
    selected: Mapping[str, Any] = payload
    for section_name in section_names:
        section = payload.get(section_name)
        if isinstance(section, Mapping):
            selected = section
            break
    if flatten_sections:
        return _flatten_publication_bundle_config(selected)
    return dict(selected)


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
    parsed, _ = parser.parse_known_args(list(argv))
    return parsed


def _strip_detach_args(argv: Sequence[str]) -> List[str]:
    out: List[str] = []
    skip_next = False
    value_flags = {"--detach-name", "--detach-job-root", "--detach-description"}
    for raw in argv:
        if skip_next:
            skip_next = False
            continue
        item = str(raw)
        if item in {"--detach", "--no-detach"}:
            continue
        if any(item.startswith(f"{flag}=") for flag in value_flags):
            continue
        if item in value_flags:
            skip_next = True
            continue
        out.append(item)
    return out


def _launch_detached_bundle(
    *,
    raw_argv: Sequence[str],
    args: argparse.Namespace,
    output_root: Path,
) -> int:
    if bool(args.estimate_only):
        raise ValueError("--detach cannot be combined with --estimate-only")
    python_bin = str(args.python_bin)
    job_root = (
        args.detach_job_root.resolve()
        if args.detach_job_root is not None
        else (output_root / "launcher")
    )
    forwarded = _strip_detach_args(raw_argv)
    cmd = [
        python_bin,
        str(Path(__file__).resolve()),
        *forwarded,
    ]
    launch_cmd = [
        python_bin,
        str(LONG_JOB_SCRIPT),
        "launch",
        "--name",
        str(args.detach_name),
        "--description",
        str(args.detach_description),
        "--cwd",
        str(REPO_ROOT),
        "--job-root",
        str(job_root),
        "--",
        *cmd,
    ]
    result = subprocess.run(
        launch_cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)
    return 0


def _discover_mig_uuids() -> List[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    migs: List[str] = []
    for line in result.stdout.splitlines():
        marker = "MIG-"
        if marker not in line:
            continue
        start = line.find(marker)
        if start < 0:
            continue
        token = line[start:].split(")", 1)[0].strip()
        if token:
            migs.append(token)
    return migs


def _resolved_mig_uuids(args: argparse.Namespace) -> List[str]:
    raw = str(args.migs or "").strip()
    if raw:
        return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    return _discover_mig_uuids()


def _minutes_range(low: float, high: float) -> Dict[str, float]:
    return {
        "eta_low_min": float(max(0.0, low)),
        "eta_high_min": float(max(float(low), high)),
    }


def _estimate_tradeoff_minutes(preset: str) -> Dict[str, float]:
    if str(preset).strip().lower() == "smoke":
        return _minutes_range(4.0, 10.0)
    return _minutes_range(25.0, 40.0)


def _wave_eta_range(
    *,
    total_jobs: int,
    device_count: int,
    low_per_wave_min: float,
    high_per_wave_min: float,
    overhead_min: float,
) -> Dict[str, float]:
    devices = max(1, int(device_count))
    jobs = max(0, int(total_jobs))
    waves = int(math.ceil(float(jobs) / float(devices))) if jobs > 0 else 0
    low = overhead_min + waves * low_per_wave_min
    high = overhead_min + waves * high_per_wave_min
    out = _minutes_range(low, high)
    out["waves"] = float(waves)
    out["jobs"] = float(jobs)
    out["devices"] = float(devices)
    return out


def estimate_publication_runtime(args: argparse.Namespace, *, mig_count: int) -> Dict[str, Any]:
    phases = set(_parse_phase_set(args.phases))
    breakdown: Dict[str, Dict[str, float]] = {}
    with_preflight = bool(getattr(args, "with_preflight", True))
    preflight_only = bool(getattr(args, "preflight_only", False))

    if with_preflight:
        preflight_low = 0.0
        preflight_high = 0.0
        if "tradeoff" in phases:
            preflight_low += 4.0
            preflight_high += 10.0
        if {"capacity", "tree_fno_pdf"} & phases:
            preflight_low += 3.0
            preflight_high += 7.0
        if {"parity", "tree_fno_pdf", "full_doc_parity_pdf"} & phases:
            preflight_low += 3.0
            preflight_high += 7.0
        if {"tree_fno_pdf", "full_doc_parity_pdf", "bundle"} & phases:
            preflight_low += 1.0
            preflight_high += 3.0
        if preflight_high > 0.0:
            breakdown["preflight"] = _minutes_range(preflight_low, preflight_high)

    if not preflight_only and "tradeoff" in phases:
        breakdown["tradeoff"] = _estimate_tradeoff_minutes(str(args.tradeoff_preset))

    if not preflight_only and "capacity" in phases:
        screen_configs = (
            len(_parse_int_list(args.capacity_widths, DEFAULT_CAPACITY_WIDTHS))
            * len(_parse_int_list(args.capacity_modes, DEFAULT_CAPACITY_MODES))
            * len(_parse_int_list(args.capacity_layers, DEFAULT_CAPACITY_LAYERS))
        )
        screen_jobs = screen_configs * len(
            _parse_int_list(args.capacity_screen_seeds, DEFAULT_CAPACITY_SCREEN_SEEDS)
        )
        locked_jobs = max(1, int(args.capacity_top_k)) * len(
            _parse_int_list(args.capacity_locked_seeds, DEFAULT_CAPACITY_LOCKED_SEEDS)
        )
        breakdown["capacity"] = _wave_eta_range(
            total_jobs=screen_jobs + locked_jobs,
            device_count=mig_count,
            low_per_wave_min=2.0,
            high_per_wave_min=3.5,
            overhead_min=2.0,
        )

    if not preflight_only and "parity" in phases:
        seeds = _parse_int_list(args.parity_seeds, DEFAULT_PARITY_SEEDS)
        scale_docs = _parse_int_list(
            args.parity_scale_train_doc_counts, DEFAULT_PARITY_SCALE_TRAIN_DOCS
        )
        method_runs = _parse_run_axis_list(
            getattr(args, "parity_method_runs", None),
            DEFAULT_PARITY_METHOD_RUNS,
            role="primary",
        )
        reference_method_runs = _parse_run_axis_list(
            getattr(args, "parity_reference_method_runs", None),
            DEFAULT_REFERENCE_METHOD_RUNS,
            role="reference",
        )
        aux_fracs = _parse_float_list(
            args.parity_upper_bound_aux_fractions,
            DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS,
        )
        gate_jobs = len(seeds) * (len(method_runs) + len(reference_method_runs))
        upper_jobs = len(seeds) * len(method_runs) * len(aux_fracs) if bool(args.parity_run_aux_upper_bound) else 0
        backfill_jobs = len(seeds) * (len(method_runs) + len(reference_method_runs)) * len(scale_docs) if bool(args.parity_backfill_on_success) else 0
        breakdown["parity"] = _wave_eta_range(
            total_jobs=gate_jobs + upper_jobs + backfill_jobs,
            device_count=mig_count,
            low_per_wave_min=2.0,
            high_per_wave_min=3.5,
            overhead_min=3.0,
        )

    if not preflight_only:
        render_steps = 0
        if "bundle" in phases:
            render_steps += 1
        if render_steps > 0:
            breakdown["render_bundle"] = _minutes_range(1.0, 5.0 + render_steps)

    low_total = sum(float(item.get("eta_low_min", 0.0)) for item in breakdown.values())
    high_total = sum(float(item.get("eta_high_min", 0.0)) for item in breakdown.values())
    return {
        "phases": sorted(phases),
        "mig_count": int(max(1, mig_count)),
        "with_preflight": with_preflight,
        "preflight_only": preflight_only,
        "breakdown": breakdown,
        "total": _minutes_range(low_total, high_total),
        "assumptions": [
            "Preflight uses real smoke/minimal runs to fail fast before the full publication jobs.",
            "Tradeoff pipeline estimate is calibrated from recent full standard runs on this machine.",
            "Capacity/parity estimates assume one job per MIG wave with moderate variance across tree/FNO families.",
            "Parity estimate assumes the full scale backfill runs whenever parity backfill is enabled.",
            "Render/bundle time is small compared with the training phases.",
        ],
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _publication_experiment_spec(
    *,
    args: argparse.Namespace,
    output_root: Path,
    manifest: Mapping[str, Any],
) -> ExperimentSpec:
    tradeoff = dict(manifest.get("tradeoff") or {})
    structural_cell = str(tradeoff.get("supervision_recovery_structural_cell", "") or "")
    benchmark_refs = (
        benchmark_ref_from_parts(
            family="markov_full_doc",
            scope="recoverable_v5",
            name="recoverable_v5",
        ),
        benchmark_ref_from_parts(
            family="markov_full_doc",
            scope="structural_core_v2",
            cell=structural_cell,
            name=(
                f"structural_core_v2::{structural_cell}"
                if structural_cell
                else "structural_core_v2"
            ),
        ),
    )
    method_refs = (
        method_ref_from_markov_full_doc_run(
            family="tree_neural",
            variant="publication_bundle",
            adapter="markov_tree",
        ),
        method_ref_from_markov_full_doc_run(
            family="official_fno",
            variant="publication_bundle",
            adapter="markov_tree",
        ),
    )
    phases = default_phase_specs(sorted(_parse_phase_set(getattr(args, "phases", None))))
    return ExperimentSpec.create(
        adapter_id="markov_tree",
        output_root=str(output_root),
        title="markov_publication_bundle",
        benchmark_refs=benchmark_refs,
        method_refs=method_refs,
        phases=phases,
        report_profiles=("publication_bundle", "tradeoff", "supervision_recovery"),
        launch_command=[sys.executable, "scripts/run_markov_publication_bundle.py", *sys.argv[1:]],
        resume_command=[sys.executable, "scripts/run_markov_publication_bundle.py", *sys.argv[1:]],
        metadata={
            "legacy_script": "run_markov_publication_bundle.py",
            "reuse_existing": bool(getattr(args, "reuse_existing", True)),
        },
    )


def _write_publication_experiment_state(
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
                "publication_manifest_json",
                "tradeoff_report_pdf",
                "capacity_locked_summary_json",
                "supervision_recovery_summary_json",
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


def _publication_artifacts(output_root: Path) -> list[object]:
    return canonical_artifact_refs_from_paths(
        {
            "publication_manifest_json": str(output_root / "publication_bundle" / "publication_manifest.json"),
            "publication_index_md": str(output_root / "publication_bundle" / "INDEX.md"),
            "tradeoff_report_pdf": str(output_root / "tradeoff_pipeline" / "tradeoff_report" / "report.pdf"),
            "supervision_recovery_summary_json": str(output_root / "tradeoff_pipeline" / "supervision_recovery" / "summary.json"),
            "capacity_locked_summary_json": str(output_root / "tree_fno_capacity" / "tree_fno_capacity_locked_summary.json"),
        },
        phase_id="bundle",
        required=False,
    )


def _publication_result_rows(
    *,
    spec: ExperimentSpec,
    manifest: Mapping[str, Any],
) -> list[ResultRow]:
    benchmark_ref = benchmark_ref_from_parts(
        family="markov_full_doc",
        scope="publication_bundle",
        name="publication_bundle",
    )
    method_ref = method_ref_from_markov_full_doc_run(
        family="tree_neural",
        variant="publication_bundle_summary",
        adapter="markov_tree",
    )
    rows: list[ResultRow] = []
    scheduler = dict(manifest.get("scheduler", {}) or {})
    rows.append(
        ResultRow(
            experiment_id=str(spec.experiment_id),
            phase="bundle",
            benchmark_ref=benchmark_ref,
            method_ref=method_ref,
            metric_name="items_total",
            metric_value=int(scheduler.get("items_total", 0) or 0),
            artifact_refs=("publication_manifest_json",),
            metadata={"source": "scheduler"},
        )
    )
    rows.append(
        ResultRow(
            experiment_id=str(spec.experiment_id),
            phase="bundle",
            benchmark_ref=benchmark_ref,
            method_ref=method_ref,
            metric_name="scheduler_state",
            metric_value=str(scheduler.get("state", "") or ""),
            artifact_refs=("publication_manifest_json",),
            metadata={"source": "scheduler"},
        )
    )
    return rows


def _expected_tradeoff_outputs(root: Path) -> List[Path]:
    return [
        root / "pipeline_summary.json",
        root / "tradeoff_report" / "report.pdf",
        root / "learnability_report" / "learnability_report.pdf",
    ]


def _expected_capacity_outputs(root: Path) -> List[Path]:
    return [
        root / "tree_fno_capacity_locked_summary.json",
        root / "locked" / "summary.json",
    ]


def _expected_parity_outputs(root: Path) -> List[Path]:
    return [
        root / "fair_parity_run_summary.json",
        root / "summary.json",
        root / "tree_fno_upper_bound_summary.json",
    ]


def _resolve_tradeoff_tree_reference(
    args: argparse.Namespace,
    *,
    capacity_root: Path | None,
) -> Dict[str, Any]:
    mode = str(getattr(args, "tradeoff_tree_reference_mode", "default") or "default").strip().lower()
    explicit_capacity_root = getattr(args, "tradeoff_tree_reference_capacity_root", None)
    preset = str(getattr(args, "tradeoff_tree_reference_preset", "") or "").strip()
    resolved_capacity_root: Path | None = None
    explicit_capacity_root_text = _optional_path_text(explicit_capacity_root)
    if explicit_capacity_root_text:
        resolved_capacity_root = Path(explicit_capacity_root_text).expanduser().resolve()
    elif mode == "capacity_locked" and capacity_root is not None:
        resolved_capacity_root = Path(capacity_root).expanduser().resolve()
    return {
        "mode": mode,
        "capacity_root": resolved_capacity_root,
        "preset": preset,
    }


def _tradeoff_command(
    *,
    python_bin: str,
    preset: str,
    device_mode: str,
    train_docs: int,
    phases: str,
    output_root: Path,
    mig_uuids: Sequence[str],
    selection_config: Path | None,
    tree_reference_mode: str,
    tree_reference_capacity_root: Path | None,
    tree_reference_preset: str,
    runtime_data_mode: str,
    runtime_bucket_mode: str,
    supervision_recovery_tree_family: str,
    supervision_recovery_structural_cell: str,
    supervision_recovery_train_docs: Sequence[int] | None,
    supervision_recovery_seeds: Sequence[int] | None,
    supervision_recovery_packages: Sequence[str] | None,
    tree_exact_eval_max_docs: int,
    prepared_data_root: Path | None,
    prepared_data_allow_create: bool,
    diagnostic_detail_mode: str,
    raw_diagnostic_artifact_dir: Path | None,
) -> List[str]:
    cmd = [
        python_bin,
        str(TRADEOFF_PIPELINE_SCRIPT),
        "--preset",
        str(preset),
        "--device-mode",
        str(device_mode),
        "--train-docs",
        str(int(train_docs)),
        "--phases",
        str(phases),
        "--output-root",
        str(output_root),
    ]
    if selection_config is not None:
        cmd.extend(["--selection-config", str(selection_config)])
    if str(tree_reference_mode).strip():
        cmd.extend(["--tree-reference-mode", str(tree_reference_mode)])
    if tree_reference_capacity_root is not None:
        cmd.extend(["--tree-reference-capacity-root", str(tree_reference_capacity_root)])
    if str(tree_reference_preset).strip():
        cmd.extend(["--tree-reference-preset", str(tree_reference_preset)])
    if str(runtime_data_mode).strip():
        cmd.extend(["--runtime-data-mode", str(runtime_data_mode)])
    if str(runtime_bucket_mode).strip():
        cmd.extend(["--runtime-bucket-mode", str(runtime_bucket_mode)])
    if str(supervision_recovery_tree_family).strip():
        cmd.extend(
            [
                "--supervision-recovery-method-id",
                str(supervision_recovery_tree_family),
            ]
        )
    if str(supervision_recovery_structural_cell).strip():
        cmd.extend(
            [
                "--supervision-recovery-structural-cell",
                str(supervision_recovery_structural_cell),
            ]
        )
    if supervision_recovery_train_docs:
        cmd.extend(
            [
                "--supervision-recovery-train-docs",
                " ".join(str(int(value)) for value in supervision_recovery_train_docs),
            ]
        )
    if supervision_recovery_seeds:
        cmd.extend(
            [
                "--supervision-recovery-seeds",
                " ".join(str(int(value)) for value in supervision_recovery_seeds),
            ]
        )
    if supervision_recovery_packages:
        cmd.extend(
            [
                "--supervision-recovery-packages",
                " ".join(str(value) for value in supervision_recovery_packages),
            ]
        )
    cmd.extend(["--tree-exact-eval-max-docs", str(int(tree_exact_eval_max_docs))])
    if prepared_data_root is not None:
        cmd.extend(["--prepared-data-root", str(prepared_data_root)])
    cmd.append(
        "--prepared-data-allow-create"
        if bool(prepared_data_allow_create)
        else "--no-prepared-data-allow-create"
    )
    if str(diagnostic_detail_mode).strip():
        cmd.extend(["--diagnostic-detail-mode", str(diagnostic_detail_mode)])
    if raw_diagnostic_artifact_dir is not None:
        cmd.extend(["--raw-diagnostic-artifact-dir", str(raw_diagnostic_artifact_dir)])
    if mig_uuids:
        cmd.extend(["--migs", ",".join(mig_uuids)])
    return cmd


def _capacity_command(
    *,
    python_bin: str,
    output_root: Path,
    benchmark: str,
    train_doc_count: int,
    screen_seeds: Sequence[int],
    locked_seeds: Sequence[int],
    top_k: int,
    widths: Sequence[int],
    modes: Sequence[int],
    layers: Sequence[int],
    runtime_data_mode: str,
    runtime_bucket_mode: str,
    runtime_preload_splits: Sequence[str],
    runtime_preload_targets: bool,
    runtime_workers_per_mig: int,
    runtime_allow_multi_worker_screen: bool,
    runtime_capacity_workers_per_mig: int,
    mig_uuids: Sequence[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(TREE_FULL_DOC_SCRIPT),
        "capacity",
        "--output-root",
        str(output_root),
        "--benchmark",
        str(benchmark),
        "--train-doc-count",
        str(int(train_doc_count)),
        "--screen-seeds",
        *[str(v) for v in screen_seeds],
        "--locked-seeds",
        *[str(v) for v in locked_seeds],
        "--top-k",
        str(int(top_k)),
        "--capacity-widths",
        *[str(v) for v in widths],
        "--capacity-modes",
        *[str(v) for v in modes],
        "--capacity-layers",
        *[str(v) for v in layers],
        "--gpu-runtime-data-mode",
        str(runtime_data_mode),
        "--gpu-runtime-bucket-mode",
        str(runtime_bucket_mode),
        "--gpu-runtime-preload-splits",
        *[str(v) for v in runtime_preload_splits],
        (
            "--gpu-runtime-preload-targets"
            if bool(runtime_preload_targets)
            else "--no-gpu-runtime-preload-targets"
        ),
        "--gpu-runtime-workers-per-mig",
        str(int(runtime_workers_per_mig)),
        (
            "--gpu-runtime-allow-multi-worker-screen"
            if bool(runtime_allow_multi_worker_screen)
            else "--no-gpu-runtime-allow-multi-worker-screen"
        ),
        "--gpu-runtime-capacity-workers-per-mig",
        str(int(runtime_capacity_workers_per_mig)),
        "--use-cuda",
        "--resume",
    ]
    if mig_uuids:
        cmd.extend(["--mig-uuids", ",".join(mig_uuids)])
    return cmd


def _parity_command(
    *,
    python_bin: str,
    output_root: Path,
    benchmark: str,
    gate_train_doc_count: int,
    scale_train_doc_counts: Sequence[int],
    seeds: Sequence[int],
    method_runs: Sequence[Mapping[str, Any]],
    reference_method_runs: Sequence[Mapping[str, Any]],
    capacity_root: Path | None,
    run_aux_upper_bound: bool,
    upper_bound_aux_fractions: Sequence[float],
    backfill_on_success: bool,
    runtime_data_mode: str,
    runtime_bucket_mode: str,
    runtime_preload_splits: Sequence[str],
    runtime_preload_targets: bool,
    runtime_workers_per_mig: int,
    runtime_allow_multi_worker_screen: bool,
    runtime_capacity_workers_per_mig: int,
    mig_uuids: Sequence[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(TREE_FULL_DOC_SCRIPT),
        "parity",
        "--output-root",
        str(output_root),
        "--benchmark",
        str(benchmark),
        "--gate-train-doc-count",
        str(int(gate_train_doc_count)),
        "--scale-train-doc-counts",
        *[str(v) for v in scale_train_doc_counts],
        "--seeds",
        *[str(v) for v in seeds],
        "--method-runs",
        *[
            (
                f"{str(run.get('method_id') or '').strip()}:"
                f"{str(run.get('law_set_id') or LAW_SET_ALL).strip()}"
            )
            for run in method_runs
        ],
        "--reference-method-runs",
        *[
            (
                f"{str(run.get('method_id') or '').strip()}:"
                f"{str(run.get('law_set_id') or LAW_SET_ALL).strip()}"
            )
            for run in reference_method_runs
        ],
        ("--run-aux-upper-bound" if bool(run_aux_upper_bound) else "--no-run-aux-upper-bound"),
        "--upper-bound-aux-fractions",
        *[str(v) for v in upper_bound_aux_fractions],
        ("--backfill-on-success" if bool(backfill_on_success) else "--no-backfill-on-success"),
        "--gpu-runtime-data-mode",
        str(runtime_data_mode),
        "--gpu-runtime-bucket-mode",
        str(runtime_bucket_mode),
        "--gpu-runtime-preload-splits",
        *[str(v) for v in runtime_preload_splits],
        (
            "--gpu-runtime-preload-targets"
            if bool(runtime_preload_targets)
            else "--no-gpu-runtime-preload-targets"
        ),
        "--gpu-runtime-workers-per-mig",
        str(int(runtime_workers_per_mig)),
        (
            "--gpu-runtime-allow-multi-worker-screen"
            if bool(runtime_allow_multi_worker_screen)
            else "--no-gpu-runtime-allow-multi-worker-screen"
        ),
        "--gpu-runtime-capacity-workers-per-mig",
        str(int(runtime_capacity_workers_per_mig)),
        "--use-cuda",
        "--resume",
    ]
    if capacity_root is not None:
        cmd.extend(["--capacity-root", str(capacity_root)])
    if mig_uuids:
        cmd.extend(["--mig-uuids", ",".join(mig_uuids)])
    return cmd


def _tree_fno_pdf_command(
    *,
    python_bin: str,
    capacity_root: Path,
    parity_root: Path,
    output_pdf: Path,
) -> List[str]:
    return [
        python_bin,
        str(TREE_FNO_TUNING_PDF_SCRIPT),
        "--capacity-root",
        str(capacity_root),
        "--parity-root",
        str(parity_root),
        "--output-pdf",
        str(output_pdf),
    ]


def _full_doc_parity_pdf_command(
    *,
    python_bin: str,
    parity_root: Path,
    output_pdf: Path,
) -> List[str]:
    return [
        python_bin,
        str(FULL_DOC_DIAGNOSTIC_PDF_SCRIPT),
        "--summary-json",
        str(parity_root / "summary.json"),
        "--output-pdf",
        str(output_pdf),
    ]


def _write_bundle_outputs(
    *,
    manifest: Mapping[str, Any],
    bundle_root: Path,
) -> tuple[Path, Path]:
    index_path = bundle_root / "publication_index.md"
    manifest_path = bundle_root / "publication_manifest.json"
    _write_json(manifest_path, manifest)
    index_path.write_text(_bundle_markdown(manifest), encoding="utf-8")
    return manifest_path, index_path


def _validate_phase_dependencies(args: argparse.Namespace, phases: set[str]) -> None:
    archived = sorted(str(phase) for phase in phases if phase in ARCHIVED_REPORT_PHASES)
    if archived:
        raise ValueError(
            "archived publication phases requested: "
            f"{archived}. Use the canonical v3 tradeoff/publication report instead."
        )
    if bool(args.preflight_only) and not bool(args.with_preflight):
        raise ValueError("--preflight-only requires --with-preflight")
    if "tree_fno_pdf" in phases:
        if "capacity" not in phases and args.capacity_root is None:
            raise ValueError(
                "tree_fno_pdf requires either the capacity phase or --capacity-root"
            )
        if "parity" not in phases and args.parity_root is None:
            raise ValueError(
                "tree_fno_pdf requires either the parity phase or --parity-root"
            )
    if "full_doc_parity_pdf" in phases:
        if "parity" not in phases and args.parity_root is None:
            raise ValueError(
                "full_doc_parity_pdf requires either the parity phase or --parity-root"
            )
    tradeoff_tree_reference_mode = str(
        getattr(args, "tradeoff_tree_reference_mode", "default") or "default"
    ).strip().lower()
    explicit_tradeoff_tree_reference_root = getattr(
        args, "tradeoff_tree_reference_capacity_root", None
    )
    if (
        "tradeoff" in phases
        and tradeoff_tree_reference_mode == "capacity_locked"
        and explicit_tradeoff_tree_reference_root is None
        and "capacity" not in phases
        and args.capacity_root is None
    ):
        raise ValueError(
            "tradeoff tree_reference mode 'capacity_locked' requires either the capacity phase, "
            "--capacity-root, or --tradeoff-tree-reference-capacity-root"
        )


def _run_command(
    *,
    name: str,
    command: Sequence[str],
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(part) for part in command) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with rc={proc.returncode}; see {log_path}")
    return float(wall_s)


def _maybe_run_step(
    *,
    name: str,
    command: Sequence[str],
    log_path: Path,
    expected_outputs: Sequence[Path],
    output_root: Path,
    cwd: Path,
    reuse_existing: bool,
    env: Mapping[str, str] | None = None,
) -> StepResult:
    if reuse_existing and expected_outputs and all(path.exists() for path in expected_outputs):
        return StepResult(
            name=name,
            status="reused",
            wall_clock_s=0.0,
            command=[str(item) for item in command],
            log_path=str(log_path),
            output_root=str(output_root),
            expected_outputs=[str(path) for path in expected_outputs],
        )
    wall_s = _run_command(
        name=name,
        command=command,
        log_path=log_path,
        cwd=cwd,
        env=env,
    )
    return StepResult(
        name=name,
        status="completed",
        wall_clock_s=float(wall_s),
        command=[str(item) for item in command],
        log_path=str(log_path),
        output_root=str(output_root),
        expected_outputs=[str(path) for path in expected_outputs],
    )


def _bundle_markdown(manifest: Mapping[str, Any]) -> str:
    eta = dict(manifest.get("eta_estimate") or {})
    total = dict(eta.get("total") or {})
    artifacts = dict(manifest.get("artifacts") or {})
    steps = list(manifest.get("steps") or [])
    reference_contract = dict(manifest.get("reference_contract") or {})
    lines = [
        "# Markov Publication Bundle",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Output root: `{manifest.get('output_root', '')}`",
        f"- ETA estimate: `{float(total.get('eta_low_min', 0.0)):.1f}-{float(total.get('eta_high_min', 0.0)):.1f} min`",
        f"- Preflight enabled: `{bool(manifest.get('with_preflight', False))}`",
        f"- Preflight only: `{bool(manifest.get('preflight_only', False))}`",
        "",
        "## Reference Contract",
        "",
        (
            f"- Canonical identifiable-zero reference: "
            f"`{reference_contract.get('identifiable_zero_reference_kind', '')}`"
        ),
        (
            f"- Full-doc FNO families: "
            f"`{', '.join(str(item) for item in reference_contract.get('full_doc_fno_families', []))}`"
        ),
        (
            f"- Full-doc FNO training backend: "
            f"`{reference_contract.get('full_doc_fno_training_backend', '')}`"
        ),
        (
            f"- Note: `{reference_contract.get('note', '')}`"
        ),
        "",
        "## Key Artifacts",
        "",
    ]
    for key in (
        "tradeoff_report_pdf",
        "learnability_report_pdf",
        "full_doc_fno_upper_bound_summary_json",
        "oracle_budget_frontier_summary_json",
        "oracle_budget_frontier_report_pdf",
        "efficiency_suite_summary_json",
        "supervision_recovery_summary_json",
        "tradeoff_report_summary_json",
        "capacity_locked_summary_json",
        "fair_parity_run_summary_json",
        "large_batch_diagnosis_summary_json",
    ):
        value = str(artifacts.get(key, "")).strip()
        if value:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Steps", ""])
    for step in steps:
        lines.append(
            f"- `{step.get('name')}`: `{step.get('status')}` "
            f"(`{float(step.get('wall_clock_s', 0.0)):.1f}s`) "
            f"log=`{step.get('log_path', '')}`"
        )
    return "\n".join(lines) + "\n"


def _build_artifact_map(
    *,
    tradeoff_root: Path,
    capacity_root: Path,
    parity_root: Path,
    bundle_root: Path,
    include_full_doc_parity_pdf: bool,
) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    tradeoff_report_pdf = tradeoff_root / "tradeoff_report" / "report.pdf"
    if tradeoff_report_pdf.exists():
        artifacts["tradeoff_report_pdf"] = str(tradeoff_report_pdf)
    tradeoff_report_summary = tradeoff_root / "tradeoff_report" / "summary.json"
    if tradeoff_report_summary.exists():
        artifacts["tradeoff_report_summary_json"] = str(tradeoff_report_summary)
    learnability_pdf = tradeoff_root / "learnability_report" / "learnability_report.pdf"
    if learnability_pdf.exists():
        artifacts["learnability_report_pdf"] = str(learnability_pdf)
    large_batch_summary = tradeoff_root / "large_batch_diagnosis" / "aggregate_summary.json"
    if large_batch_summary.exists():
        artifacts["large_batch_diagnosis_summary_json"] = str(large_batch_summary)
    full_doc_upper_bound_summary = (
        tradeoff_root / "full_doc_anchor" / "full_doc_fno_upper_bound_summary.json"
    )
    if full_doc_upper_bound_summary.exists():
        artifacts["full_doc_fno_upper_bound_summary_json"] = str(full_doc_upper_bound_summary)
    oracle_budget_frontier_summary = (
        tradeoff_root / "oracle_budget_frontier" / "tree_oracle_budget_frontier_summary.json"
    )
    if oracle_budget_frontier_summary.exists():
        artifacts["oracle_budget_frontier_summary_json"] = str(oracle_budget_frontier_summary)
    oracle_budget_frontier_report = (
        tradeoff_root / "oracle_budget_frontier" / "tree_oracle_budget_frontier_report.pdf"
    )
    if oracle_budget_frontier_report.exists():
        artifacts["oracle_budget_frontier_report_pdf"] = str(oracle_budget_frontier_report)
    efficiency_suite_summary = tradeoff_root / "efficiency_suite" / "summary.json"
    if efficiency_suite_summary.exists():
        artifacts["efficiency_suite_summary_json"] = str(efficiency_suite_summary)
    supervision_recovery_summary = tradeoff_root / "supervision_recovery" / "summary.json"
    if supervision_recovery_summary.exists():
        artifacts["supervision_recovery_summary_json"] = str(supervision_recovery_summary)
    capacity_summary = capacity_root / "tree_fno_capacity_locked_summary.json"
    if capacity_summary.exists():
        artifacts["capacity_locked_summary_json"] = str(capacity_summary)
    parity_summary = parity_root / "fair_parity_run_summary.json"
    if parity_summary.exists():
        artifacts["fair_parity_run_summary_json"] = str(parity_summary)
    return artifacts


def _write_selection_template(path: Path) -> None:
    assert_public_contract_clean(
        PUBLICATION_SELECTION_TEMPLATE,
        surface="markov publication config template",
    )
    write_structured_config(path, PUBLICATION_SELECTION_TEMPLATE)


def _build_tradeoff_plan_for_bundle(
    *,
    args: argparse.Namespace,
    output_root: Path,
    mig_uuids: Sequence[str],
    tree_reference_mode: str,
    tree_reference_capacity_root: Path | None,
    tree_reference_preset: str,
    runtime_data_mode: str,
    runtime_bucket_mode: str,
    supervision_recovery_tree_family: str,
    supervision_recovery_structural_cell: str,
    supervision_recovery_train_docs: Sequence[int] | None,
    supervision_recovery_seeds: Sequence[int] | None,
    supervision_recovery_packages: Sequence[str] | None,
) -> Dict[str, Any]:
    from scripts.run_markov_optimization_tradeoff_pipeline import (  # type: ignore
        _parse_args as _parse_tradeoff_args,
        build_run_plan as _build_tradeoff_run_plan,
    )

    argv: List[str] = [
        "--output-root",
        str(output_root),
        "--preset",
        str(args.tradeoff_preset),
        "--device-mode",
        str(args.tradeoff_device_mode),
        "--train-docs",
        str(int(args.tradeoff_train_docs)),
        "--phases",
        str(args.tradeoff_phases),
        "--tree-exact-eval-max-docs",
        str(int(getattr(args, "tradeoff_tree_exact_eval_max_docs", 0) or 0)),
    ]
    if getattr(args, "selection_config", None) is not None:
        argv.extend(["--selection-config", str(args.selection_config)])
    if str(tree_reference_mode).strip():
        argv.extend(["--tree-reference-mode", str(tree_reference_mode)])
    if tree_reference_capacity_root is not None:
        argv.extend(["--tree-reference-capacity-root", str(tree_reference_capacity_root)])
    if str(tree_reference_preset).strip():
        argv.extend(["--tree-reference-preset", str(tree_reference_preset)])
    if str(runtime_data_mode).strip():
        argv.extend(["--runtime-data-mode", str(runtime_data_mode)])
    if str(runtime_bucket_mode).strip():
        argv.extend(["--runtime-bucket-mode", str(runtime_bucket_mode)])
    if str(supervision_recovery_tree_family).strip():
        argv.extend(
            [
                "--supervision-recovery-method-id",
                str(supervision_recovery_tree_family),
            ]
        )
    if str(supervision_recovery_structural_cell).strip():
        argv.extend(
            [
                "--supervision-recovery-structural-cell",
                str(supervision_recovery_structural_cell),
            ]
        )
    if supervision_recovery_train_docs:
        argv.extend(
            [
                "--supervision-recovery-train-docs",
                " ".join(str(int(value)) for value in supervision_recovery_train_docs),
            ]
        )
    if supervision_recovery_seeds:
        argv.extend(
            [
                "--supervision-recovery-seeds",
                " ".join(str(int(value)) for value in supervision_recovery_seeds),
            ]
        )
    if supervision_recovery_packages:
        argv.extend(
            [
                "--supervision-recovery-packages",
                " ".join(str(value) for value in supervision_recovery_packages),
            ]
        )
    tradeoff_prepared_data_root = getattr(args, "tradeoff_prepared_data_root", None)
    if tradeoff_prepared_data_root is not None and str(tradeoff_prepared_data_root).strip():
        argv.extend(["--prepared-data-root", str(tradeoff_prepared_data_root)])
    argv.append(
        "--prepared-data-allow-create"
        if bool(getattr(args, "tradeoff_prepared_data_allow_create", True))
        else "--no-prepared-data-allow-create"
    )
    argv.extend(
        [
            "--diagnostic-detail-mode",
            str(getattr(args, "tradeoff_diagnostic_detail_mode", "summary")),
        ]
    )
    tradeoff_raw_artifact_dir = getattr(args, "tradeoff_raw_diagnostic_artifact_dir", None)
    if tradeoff_raw_artifact_dir is not None and str(tradeoff_raw_artifact_dir).strip():
        argv.extend(["--raw-diagnostic-artifact-dir", str(tradeoff_raw_artifact_dir)])
    if mig_uuids:
        argv.extend(["--migs", ",".join(mig_uuids)])
    tradeoff_args = _parse_tradeoff_args(argv)
    return _build_tradeoff_run_plan(tradeoff_args, devices=mig_uuids)


def _tradeoff_args_for_bundle(
    *,
    args: argparse.Namespace,
    output_root: Path,
    mig_uuids: Sequence[str],
    tree_reference_mode: str,
    tree_reference_capacity_root: Path | None,
    tree_reference_preset: str,
    runtime_data_mode: str,
    runtime_bucket_mode: str,
    supervision_recovery_tree_family: str,
    supervision_recovery_structural_cell: str,
    supervision_recovery_train_docs: Sequence[int] | None,
    supervision_recovery_seeds: Sequence[int] | None,
    supervision_recovery_packages: Sequence[str] | None,
) -> argparse.Namespace:
    from scripts.run_markov_optimization_tradeoff_pipeline import (  # type: ignore
        _parse_args as _parse_tradeoff_args,
    )

    argv: List[str] = [
        "--output-root",
        str(output_root),
        "--preset",
        str(args.tradeoff_preset),
        "--device-mode",
        str(args.tradeoff_device_mode),
        "--train-docs",
        str(int(args.tradeoff_train_docs)),
        "--phases",
        str(args.tradeoff_phases),
        "--scheduler-mode",
        str(getattr(args, "scheduler_mode", "global_per_run")),
        "--default-job-granularity",
        str(getattr(args, "default_job_granularity", "family_train_seed")),
        (
            "--cleanup-stale-children"
            if bool(getattr(args, "cleanup_stale_children", True))
            else "--no-cleanup-stale-children"
        ),
        "--max-gpu-items-per-mig",
        str(int(getattr(args, "max_gpu_items_per_mig", 1) or 1)),
        "--tree-exact-eval-max-docs",
        str(int(getattr(args, "tradeoff_tree_exact_eval_max_docs", 0) or 0)),
    ]
    if getattr(args, "selection_config", None) is not None:
        argv.extend(["--selection-config", str(args.selection_config)])
    if str(tree_reference_mode).strip():
        argv.extend(["--tree-reference-mode", str(tree_reference_mode)])
    if tree_reference_capacity_root is not None:
        argv.extend(["--tree-reference-capacity-root", str(tree_reference_capacity_root)])
    if str(tree_reference_preset).strip():
        argv.extend(["--tree-reference-preset", str(tree_reference_preset)])
    if str(runtime_data_mode).strip():
        argv.extend(["--runtime-data-mode", str(runtime_data_mode)])
    if str(runtime_bucket_mode).strip():
        argv.extend(["--runtime-bucket-mode", str(runtime_bucket_mode)])
    if str(supervision_recovery_tree_family).strip():
        argv.extend(
            [
                "--supervision-recovery-method-id",
                str(supervision_recovery_tree_family),
            ]
        )
    if str(supervision_recovery_structural_cell).strip():
        argv.extend(
            [
                "--supervision-recovery-structural-cell",
                str(supervision_recovery_structural_cell),
            ]
        )
    if supervision_recovery_train_docs:
        argv.extend(
            [
                "--supervision-recovery-train-docs",
                " ".join(str(int(value)) for value in supervision_recovery_train_docs),
            ]
        )
    if supervision_recovery_seeds:
        argv.extend(
            [
                "--supervision-recovery-seeds",
                " ".join(str(int(value)) for value in supervision_recovery_seeds),
            ]
        )
    if supervision_recovery_packages:
        argv.extend(
            [
                "--supervision-recovery-packages",
                " ".join(str(value) for value in supervision_recovery_packages),
            ]
        )
    tradeoff_prepared_data_root = getattr(args, "tradeoff_prepared_data_root", None)
    if tradeoff_prepared_data_root is not None and str(tradeoff_prepared_data_root).strip():
        argv.extend(["--prepared-data-root", str(tradeoff_prepared_data_root)])
    argv.append(
        "--prepared-data-allow-create"
        if bool(getattr(args, "tradeoff_prepared_data_allow_create", True))
        else "--no-prepared-data-allow-create"
    )
    argv.extend(
        [
            "--diagnostic-detail-mode",
            str(getattr(args, "tradeoff_diagnostic_detail_mode", "summary")),
        ]
    )
    tradeoff_raw_artifact_dir = getattr(args, "tradeoff_raw_diagnostic_artifact_dir", None)
    if tradeoff_raw_artifact_dir is not None and str(tradeoff_raw_artifact_dir).strip():
        argv.extend(["--raw-diagnostic-artifact-dir", str(tradeoff_raw_artifact_dir)])
    if mig_uuids:
        argv.extend(["--migs", ",".join(mig_uuids)])
    return _parse_tradeoff_args(argv)


def _preflight_tradeoff_kwargs(args: argparse.Namespace, *, phases: str) -> Dict[str, Any]:
    phase_set = set(_parse_phase_set(phases))
    if "supervision_recovery" in phase_set:
        return {
            "preset": str(args.tradeoff_preset),
            "selection_config": None,
            "supervision_recovery_tree_family": str(
                getattr(args, "tradeoff_supervision_recovery_tree_family", "tree_neural")
            ),
            "supervision_recovery_structural_cell": str(
                getattr(args, "tradeoff_supervision_recovery_structural_cell", "r12_p079")
            ),
            "supervision_recovery_train_docs": list(PREFLIGHT_TRADEOFF_TRAIN_DOCS),
            "supervision_recovery_seeds": list(PREFLIGHT_TRADEOFF_SEEDS),
            "supervision_recovery_packages": list(PREFLIGHT_TRADEOFF_PACKAGES),
        }
    return {
        "preset": "smoke",
        "selection_config": None,
        "supervision_recovery_tree_family": "",
        "supervision_recovery_structural_cell": "",
        "supervision_recovery_train_docs": None,
        "supervision_recovery_seeds": None,
        "supervision_recovery_packages": None,
    }


def _full_doc_job_item(
    *,
    phase: str,
    item_id: str,
    output_root: Path,
    job: Any,
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
                torch_threads=1,
                use_cuda=bool(use_cuda),
            )
        ),
        log_path=str(job_output_dir / "worker.log"),
        metadata={"job_name": str(job.job_name)},
    )


def _publication_capacity_namespace(
    args: argparse.Namespace,
    *,
    output_root: Path,
    mig_uuids: Sequence[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=str(output_root),
        benchmark=str(args.capacity_benchmark),
        train_doc_count=int(args.capacity_train_doc_count),
        priority_family="tree_neural",
        screen_seeds=_parse_int_list(args.capacity_screen_seeds, DEFAULT_CAPACITY_SCREEN_SEEDS),
        locked_seeds=_parse_int_list(args.capacity_locked_seeds, DEFAULT_CAPACITY_LOCKED_SEEDS),
        top_k=int(args.capacity_top_k),
        capacity_widths=_parse_int_list(args.capacity_widths, DEFAULT_CAPACITY_WIDTHS),
        capacity_modes=_parse_int_list(args.capacity_modes, DEFAULT_CAPACITY_MODES),
        capacity_layers=_parse_int_list(args.capacity_layers, DEFAULT_CAPACITY_LAYERS),
        job_granularity=str(getattr(args, "default_job_granularity", "family_train_seed")),
        resume=True,
        use_cuda=True,
        torch_threads=1,
        mig_uuids=",".join(str(item) for item in mig_uuids),
        mig_uuids_resolved=list(mig_uuids),
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        local_law_weight=0.3,
        root_share=None,
        tree_local_law_weight=0.3,
        tree_task_objective_weight=None,
        doc_sequence_train_fraction=0.0,
        gpu_runtime_data_mode=str(args.capacity_runtime_data_mode),
        gpu_runtime_bucket_mode=str(args.capacity_runtime_bucket_mode),
        gpu_runtime_preload_splits=_parse_str_list(args.capacity_runtime_preload_splits, ("train", "val", "test")),
        gpu_runtime_preload_targets=bool(args.capacity_runtime_preload_targets),
        gpu_runtime_workers_per_mig=int(args.capacity_runtime_workers_per_mig),
        gpu_runtime_allow_multi_worker_screen=bool(args.capacity_runtime_allow_multi_worker_screen),
        gpu_runtime_capacity_workers_per_mig=int(args.capacity_runtime_capacity_workers_per_mig),
    )


def _publication_parity_namespace(
    args: argparse.Namespace,
    *,
    output_root: Path,
    capacity_root: Path | None,
    mig_uuids: Sequence[str],
) -> argparse.Namespace:
    method_runs = _parse_run_axis_list(
        args.parity_method_runs,
        DEFAULT_PARITY_METHOD_RUNS,
        role="primary",
    )
    reference_method_runs = _parse_run_axis_list(
        args.parity_reference_method_runs,
        DEFAULT_REFERENCE_METHOD_RUNS,
        role="reference",
    )
    return argparse.Namespace(
        output_root=str(output_root),
        benchmark=str(args.parity_benchmark),
        gate_train_doc_count=int(args.parity_gate_train_doc_count),
        scale_train_doc_counts=_parse_int_list(args.parity_scale_train_doc_counts, DEFAULT_PARITY_SCALE_TRAIN_DOCS),
        seeds=_parse_int_list(args.parity_seeds, DEFAULT_PARITY_SEEDS),
        tree_families=_legacy_families_from_run_axes(method_runs),
        fno_families=_method_ids_from_run_axes(reference_method_runs),
        job_granularity=str(getattr(args, "default_job_granularity", "family_train_seed")),
        resume=True,
        backfill_on_success=bool(args.parity_backfill_on_success),
        run_aux_upper_bound=bool(args.parity_run_aux_upper_bound),
        upper_bound_aux_fractions=_parse_float_list(
            args.parity_upper_bound_aux_fractions,
            DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS,
        ),
        capacity_root=str(capacity_root) if capacity_root is not None else "",
        use_cuda=True,
        torch_threads=1,
        mig_uuids=",".join(str(item) for item in mig_uuids),
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        local_law_weight=None,
        root_share=None,
        tree_local_law_weight=None,
        tree_task_objective_weight=None,
        doc_sequence_train_fraction=0.0,
        gpu_runtime_data_mode=str(args.parity_runtime_data_mode),
        gpu_runtime_bucket_mode=str(args.parity_runtime_bucket_mode),
        gpu_runtime_preload_splits=_parse_str_list(args.parity_runtime_preload_splits, ("train", "val", "test")),
        gpu_runtime_preload_targets=bool(args.parity_runtime_preload_targets),
        gpu_runtime_workers_per_mig=int(args.parity_runtime_workers_per_mig),
        gpu_runtime_allow_multi_worker_screen=bool(args.parity_runtime_allow_multi_worker_screen),
        gpu_runtime_capacity_workers_per_mig=int(args.parity_runtime_capacity_workers_per_mig),
    )


def _run_publication_global_scheduler(
    *,
    args: argparse.Namespace,
    output_root: Path,
    mig_uuids: Sequence[str],
    tradeoff_root: Path,
    capacity_root: Path,
    parity_root: Path,
    bundle_root: Path,
    phases: Sequence[str],
    eta_estimate: Mapping[str, Any],
) -> Dict[str, Any]:
    from scripts.run_markov_optimization_tradeoff_pipeline import (  # type: ignore
        _build_tradeoff_scheduler_graph,
    )
    from scripts.run_tree_neural_full_doc_mig import (  # type: ignore
        build_capacity_locked_job_bundle,
        build_capacity_screen_job_bundle,
        build_parity_job_bundle,
        finalize_capacity_locked_output,
        finalize_capacity_screen_output,
        finalize_parity_output,
    )

    phase_set = set(phases)
    tradeoff_tree_reference = _resolve_tradeoff_tree_reference(
        args,
        capacity_root=(capacity_root if ("capacity" in phase_set or args.capacity_root is not None) else None),
    )
    tree_bundle_contract = _markov_publication_tree_bundle_contract(
        args=args,
        phases=phase_set,
    )
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "tradeoff_root": str(tradeoff_root),
        "capacity_root": str(capacity_root),
        "parity_root": str(parity_root),
        "bundle_root": str(bundle_root),
        "with_preflight": bool(args.with_preflight),
        "preflight_only": bool(args.preflight_only),
        "eta_estimate": dict(eta_estimate),
        "tree_bundle_contract": tree_bundle_contract,
    }
    manifest["run_manifest"] = _markov_publication_run_manifest(
        args=args,
        output_root=output_root,
        phases=phase_set,
        tree_bundle_contract=tree_bundle_contract,
        status="running",
        publication_ready=False,
        metadata={"scheduler_mode": "global_per_run"},
    )
    items: List[SchedulerItem] = []

    if "tradeoff" in phase_set:
        tradeoff_args = _tradeoff_args_for_bundle(
            args=args,
            output_root=tradeoff_root,
            mig_uuids=mig_uuids,
            tree_reference_mode=str(tradeoff_tree_reference["mode"]),
            tree_reference_capacity_root=tradeoff_tree_reference["capacity_root"],
            tree_reference_preset=str(tradeoff_tree_reference.get("preset", "") or ""),
            runtime_data_mode=str(getattr(args, "tradeoff_runtime_data_mode", "resident")),
            runtime_bucket_mode=str(getattr(args, "tradeoff_runtime_bucket_mode", "leaf_count_auto_queue")),
            supervision_recovery_tree_family=str(
                getattr(args, "tradeoff_supervision_recovery_tree_family", "tree_neural")
            ),
            supervision_recovery_structural_cell=str(
                getattr(args, "tradeoff_supervision_recovery_structural_cell", "r12_p079")
            ),
            supervision_recovery_train_docs=None,
            supervision_recovery_seeds=None,
            supervision_recovery_packages=None,
        )
        tradeoff_graph = _build_tradeoff_scheduler_graph(
            tradeoff_args,
            output_root=tradeoff_root,
            devices=mig_uuids,
        )
        items.extend(list(tradeoff_graph["items"]))

    capacity_final_dep = ""
    if "capacity" in phase_set:
        capacity_args = _publication_capacity_namespace(
            args,
            output_root=capacity_root,
            mig_uuids=mig_uuids,
        )
        capacity_bundle = build_capacity_screen_job_bundle(capacity_args)
        screen_item_ids: List[str] = []
        for job in list(capacity_bundle["screen_jobs"]):
            item = _full_doc_job_item(
                phase="capacity",
                item_id=f"capacity::screen::{job.job_name}",
                output_root=capacity_bundle["screen_root"],
                job=job,
                use_cuda=True,
            )
            screen_item_ids.append(str(item.item_id))
            items.append(item)

        def _capacity_screen_reduce() -> Mapping[str, Any]:
            result = finalize_capacity_screen_output(
                args=capacity_args,
                output_root=capacity_root,
                screen_root=capacity_bundle["screen_root"],
                config_by_label=capacity_bundle["config_by_label"],
            )
            locked_bundle = build_capacity_locked_job_bundle(
                capacity_args,
                locked_configs=result["locked_configs"],
            )
            locked_item_ids: List[str] = []
            new_items: List[SchedulerItem] = []
            for job in list(locked_bundle["locked_jobs"]):
                item = _full_doc_job_item(
                    phase="capacity",
                    item_id=f"capacity::locked::{job.job_name}",
                    output_root=locked_bundle["locked_root"],
                    job=job,
                    use_cuda=True,
                )
                locked_item_ids.append(str(item.item_id))
                new_items.append(item)

            def _capacity_locked_reduce() -> Mapping[str, Any]:
                final = finalize_capacity_locked_output(
                    args=capacity_args,
                    output_root=capacity_root,
                    screen_root=capacity_bundle["screen_root"],
                    locked_root=locked_bundle["locked_root"],
                    screen_rankings=result["top_rankings"],
                    config_by_label=capacity_bundle["config_by_label"],
                )
                manifest["capacity"] = dict(final)
                return {"result": dict(final)}

            new_items.append(
                SchedulerItem(
                    item_id="capacity::locked::reduce",
                    phase="capacity",
                    kind="cpu_callback",
                    deps=tuple(locked_item_ids),
                    expected_outputs=(str(capacity_root / "tree_fno_capacity_locked_summary.json"),),
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
                expected_outputs=(str(capacity_root / "tree_fno_capacity_screen_summary.json"),),
                callback=_capacity_screen_reduce,
                reuse_existing=False,
            )
        )
        capacity_final_dep = "capacity::locked::reduce"

    if "parity" in phase_set:
        parity_expand_deps: List[str] = []
        resolved_capacity_root = capacity_root if ("capacity" in phase_set) else args.capacity_root
        if capacity_final_dep:
            parity_expand_deps.append(capacity_final_dep)

        def _parity_expand() -> Mapping[str, Any]:
            parity_args = _publication_parity_namespace(
                args,
                output_root=parity_root,
                capacity_root=(
                    resolved_capacity_root.resolve()
                    if isinstance(resolved_capacity_root, Path)
                    else None
                ),
                mig_uuids=mig_uuids,
            )
            parity_bundle = build_parity_job_bundle(parity_args)
            new_items: List[SchedulerItem] = []
            all_gpu_ids: List[str] = []
            for prefix, root_key, jobs_key in (
                ("gate", "gate_root", "gate_jobs"),
                ("upper", "upper_bound_root", "upper_bound_jobs"),
                ("backfill", "backfill_root", "backfill_jobs"),
            ):
                phase_root = Path(str(parity_bundle[root_key]))
                for job in list(parity_bundle[jobs_key]):
                    item = _full_doc_job_item(
                        phase="parity",
                        item_id=f"parity::{prefix}::{job.job_name}",
                        output_root=phase_root,
                        job=job,
                        use_cuda=True,
                    )
                    all_gpu_ids.append(str(item.item_id))
                    new_items.append(item)

            def _parity_reduce() -> Mapping[str, Any]:
                final = finalize_parity_output(
                    args=parity_args,
                    output_root=parity_root,
                    gate_failed_jobs=0,
                    upper_bound_failed_jobs=0,
                    backfill_failed_jobs=0,
                    parity_tree_config=parity_bundle["parity_tree_config"],
                    reference_fno_config=parity_bundle["reference_fno_config"],
                    parity_tree_families=parity_bundle["parity_tree_families"],
                    parity_fno_families=parity_bundle["parity_fno_families"],
                    parity_comparison_families=parity_bundle["parity_comparison_families"],
                    capacity_root_value=str(parity_bundle["capacity_root"]),
                )
                manifest["parity"] = dict(final)
                return {"result": dict(final)}

            new_items.append(
                SchedulerItem(
                    item_id="parity::reduce",
                    phase="parity",
                    kind="cpu_callback",
                    deps=tuple(all_gpu_ids),
                    expected_outputs=(str(parity_root / "fair_parity_run_summary.json"),),
                    callback=_parity_reduce,
                    reuse_existing=False,
                )
            )
            return {"new_items": new_items}

        items.append(
            SchedulerItem(
                item_id="parity::expand",
                phase="parity",
                kind="cpu_callback",
                deps=tuple(parity_expand_deps),
                callback=_parity_expand,
                reuse_existing=False,
            )
        )

    if "tree_fno_pdf" in phase_set:
        tree_fno_pdf = bundle_root / "tree_fno_tuning_report.pdf"
        items.append(
            SchedulerItem(
                item_id="tree_fno_pdf::render",
                phase="tree_fno_pdf",
                kind="cpu_command",
                deps=tuple(dep for dep in [capacity_final_dep, "parity::reduce"] if dep),
                expected_outputs=(str(tree_fno_pdf),),
                command=tuple(
                    _tree_fno_pdf_command(
                        python_bin=str(args.python_bin),
                        capacity_root=capacity_root,
                        parity_root=parity_root,
                        output_pdf=tree_fno_pdf,
                    )
                ),
                log_path=str(output_root / "logs" / "tree_fno_pdf.log"),
            )
        )

    if "full_doc_parity_pdf" in phase_set and bool(args.render_full_doc_parity_pdf):
        full_doc_parity_pdf = bundle_root / "full_doc_parity_report.pdf"
        items.append(
            SchedulerItem(
                item_id="full_doc_parity_pdf::render",
                phase="full_doc_parity_pdf",
                kind="cpu_command",
                deps=("parity::reduce",),
                expected_outputs=(str(full_doc_parity_pdf),),
                command=tuple(
                    _full_doc_parity_pdf_command(
                        python_bin=str(args.python_bin),
                        parity_root=parity_root,
                        output_pdf=full_doc_parity_pdf,
                    )
                ),
                log_path=str(output_root / "logs" / "full_doc_parity_pdf.log"),
            )
        )

    if "bundle" in phase_set:
        bundle_manifest_path = bundle_root / "publication_manifest.json"
        bundle_deps = []
        if "tradeoff" in phase_set:
            bundle_deps.append("report::reduce")
        if "tree_fno_pdf" in phase_set:
            bundle_deps.append("tree_fno_pdf::render")
        if "full_doc_parity_pdf" in phase_set and bool(args.render_full_doc_parity_pdf):
            bundle_deps.append("full_doc_parity_pdf::render")

        def _bundle_reduce() -> Mapping[str, Any]:
            artifacts = _build_artifact_map(
                tradeoff_root=tradeoff_root,
                capacity_root=capacity_root,
                parity_root=parity_root,
                bundle_root=bundle_root,
                include_full_doc_parity_pdf=bool(args.render_full_doc_parity_pdf),
            )
            bundle_tree_bundle_contract = _markov_publication_tree_bundle_contract(
                args=args,
                phases=phase_set,
            )
            bundle_manifest = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "output_root": str(output_root),
                "tradeoff_root": str(tradeoff_root),
                "capacity_root": str(capacity_root),
                "parity_root": str(parity_root),
                "bundle_root": str(bundle_root),
                "eta_estimate": dict(eta_estimate),
                "artifacts": artifacts,
                "with_preflight": bool(args.with_preflight),
                "preflight_only": bool(args.preflight_only),
                "steps": [],
                "tree_bundle_contract": bundle_tree_bundle_contract,
                "reference_contract": {
                    "identifiable_zero_reference_kind": "full_doc_fno_upper_bound",
                    "full_doc_fno_families": list(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES),
                    "full_doc_fno_training_backend": "markov_full_doc_runtime_v3",
                    "note": "The publication bundle treats the full-doc FNO upper bound as the canonical identifiable-zero reference.",
                },
            }
            bundle_manifest["run_manifest"] = _markov_publication_run_manifest(
                args=args,
                output_root=output_root,
                phases=phase_set,
                tree_bundle_contract=bundle_tree_bundle_contract,
                artifacts=artifacts,
                publication_ready=not bool(args.preflight_only),
                metadata={"bundle_root": str(bundle_root), "scheduler_mode": "global_per_run"},
            )
            manifest_path, index_path = _write_bundle_outputs(
                manifest=bundle_manifest,
                bundle_root=bundle_root,
            )
            manifest["bundle"] = {
                "bundle_root": str(bundle_root),
                "publication_manifest_json": str(manifest_path),
                "publication_index_md": str(index_path),
            }
            return {"result": {"publication_manifest_json": str(manifest_path)}}

        items.append(
            SchedulerItem(
                item_id="bundle::reduce",
                phase="bundle",
                kind="cpu_callback",
                deps=tuple(bundle_deps),
                expected_outputs=(str(bundle_manifest_path),),
                callback=_bundle_reduce,
                reuse_existing=False,
            )
        )

    experiment_spec = _publication_experiment_spec(
        args=args,
        output_root=output_root,
        manifest=manifest,
    )
    write_experiment_manifest(output_root, experiment_spec)
    _write_publication_experiment_state(
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
            devices=tuple(mig_uuids),
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
                    "publication_manifest_json",
                    "tradeoff_report_pdf",
                    "capacity_locked_summary_json",
                    "supervision_recovery_summary_json",
                ],
            },
            event_log_path=str(output_root / "event_log.jsonl"),
        ),
    )
    manifest["scheduler"] = scheduler_summary
    scheduler_state = str(scheduler_summary.get("state", "completed") or "completed")
    manifest["run_manifest"] = _markov_publication_run_manifest(
        args=args,
        output_root=output_root,
        phases=phase_set,
        tree_bundle_contract=tree_bundle_contract,
        status="completed" if scheduler_state == "completed" else "partial",
        publication_ready=scheduler_state == "completed",
        metadata={"scheduler_mode": "global_per_run", "scheduler_state": scheduler_state},
    )
    merge_artifacts(output_root, _publication_artifacts(output_root))
    append_result_rows(
        output_root,
        _publication_result_rows(
            spec=experiment_spec,
            manifest=manifest,
        ),
    )
    _write_publication_experiment_state(
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
    return manifest


def build_publication_run_plan(
    args: argparse.Namespace,
    *,
    mig_uuids: Sequence[str],
    output_root: Path,
) -> Dict[str, Any]:
    phases = sorted(_parse_phase_set(args.phases))
    preflight_root = output_root / "preflight"
    tradeoff_root = args.tradeoff_root.resolve() if args.tradeoff_root is not None else (output_root / "tradeoff_pipeline")
    capacity_root = args.capacity_root.resolve() if args.capacity_root is not None else (output_root / "tree_fno_capacity")
    parity_root = args.parity_root.resolve() if args.parity_root is not None else (output_root / "tree_fno_parity")
    bundle_root = args.bundle_root.resolve() if args.bundle_root is not None else (output_root / "publication_bundle")
    eta_estimate = estimate_publication_runtime(args, mig_count=len(mig_uuids) if mig_uuids else 1)
    tradeoff_tree_reference = _resolve_tradeoff_tree_reference(
        args,
        capacity_root=(
            capacity_root
            if ("capacity" in phases or args.capacity_root is not None)
            else None
        ),
    )
    tradeoff_plan = _build_tradeoff_plan_for_bundle(
        args=args,
        output_root=tradeoff_root,
        mig_uuids=mig_uuids,
        tree_reference_mode=str(tradeoff_tree_reference["mode"]),
        tree_reference_capacity_root=tradeoff_tree_reference["capacity_root"],
        tree_reference_preset=str(tradeoff_tree_reference.get("preset", "") or ""),
        runtime_data_mode=str(getattr(args, "tradeoff_runtime_data_mode", "resident")),
        runtime_bucket_mode=str(getattr(args, "tradeoff_runtime_bucket_mode", "leaf_count_auto_queue")),
        supervision_recovery_tree_family=str(
            getattr(args, "tradeoff_supervision_recovery_tree_family", "tree_neural")
        ),
        supervision_recovery_structural_cell=str(
            getattr(args, "tradeoff_supervision_recovery_structural_cell", "r12_p079")
        ),
        supervision_recovery_train_docs=None,
        supervision_recovery_seeds=None,
        supervision_recovery_packages=None,
    )
    capacity_runtime = {
        "data_mode": str(getattr(args, "capacity_runtime_data_mode", "resident")),
        "bucket_mode": str(
            getattr(args, "capacity_runtime_bucket_mode", "exact_then_bucketed")
        ),
        "preload_splits": _parse_str_list(
            getattr(args, "capacity_runtime_preload_splits", None),
            ("train", "val", "test"),
        ),
        "preload_targets": bool(
            getattr(args, "capacity_runtime_preload_targets", True)
        ),
        "workers_per_mig": int(
            getattr(args, "capacity_runtime_workers_per_mig", 1)
        ),
        "allow_multi_worker_screen": bool(
            getattr(args, "capacity_runtime_allow_multi_worker_screen", True)
        ),
        "capacity_workers_per_mig": int(
            getattr(args, "capacity_runtime_capacity_workers_per_mig", 2)
        ),
    }
    parity_runtime = {
        "data_mode": str(getattr(args, "parity_runtime_data_mode", "resident")),
        "bucket_mode": str(
            getattr(args, "parity_runtime_bucket_mode", "exact_then_bucketed")
        ),
        "preload_splits": _parse_str_list(
            getattr(args, "parity_runtime_preload_splits", None),
            ("train", "val", "test"),
        ),
        "preload_targets": bool(
            getattr(args, "parity_runtime_preload_targets", True)
        ),
        "workers_per_mig": int(
            getattr(args, "parity_runtime_workers_per_mig", 1)
        ),
        "allow_multi_worker_screen": bool(
            getattr(args, "parity_runtime_allow_multi_worker_screen", False)
        ),
        "capacity_workers_per_mig": int(
            getattr(args, "parity_runtime_capacity_workers_per_mig", 2)
        ),
    }
    resolved_selection = {
        "selection_config": str(args.selection_config) if getattr(args, "selection_config", None) else "",
        "phases": phases,
        "with_preflight": bool(args.with_preflight),
        "preflight_only": bool(args.preflight_only),
        "reuse_existing": bool(args.reuse_existing),
        "tradeoff": {
            "preset": str(args.tradeoff_preset),
            "device_mode": str(args.tradeoff_device_mode),
            "phases": sorted(_parse_phase_set(args.tradeoff_phases)),
            "train_docs": int(args.tradeoff_train_docs),
            "tree_exact_eval_max_docs": int(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("tree_exact_eval_max_docs", 0)
                or 0
            ),
            "prepared_data_root": str(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("prepared_data_root", "")
                or ""
            ),
            "prepared_data_allow_create": bool(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("prepared_data_allow_create", True)
            ),
            "supervision_recovery_train_docs": list(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("supervision_recovery_train_docs", [])
                or []
            ),
            "supervision_recovery_seeds": list(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("supervision_recovery_seeds", [])
                or []
            ),
            "supervision_recovery_method_id": str(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("supervision_recovery_method_id", "")
                or ""
            ),
            "supervision_recovery_structural_cell": str(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("supervision_recovery_structural_cell", "")
                or ""
            ),
            "tree_reference": dict(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("tree_reference", {})
                or {}
            ),
            "runtime": dict(
                (
                    tradeoff_plan.get("resolved_selection", {}) or {}
                ).get("runtime", {})
                or {}
            ),
        },
        "capacity": {
            "benchmark": str(args.capacity_benchmark),
            "train_doc_count": int(args.capacity_train_doc_count),
            "screen_seeds": _parse_int_list(args.capacity_screen_seeds, DEFAULT_CAPACITY_SCREEN_SEEDS),
            "locked_seeds": _parse_int_list(args.capacity_locked_seeds, DEFAULT_CAPACITY_LOCKED_SEEDS),
            "top_k": int(args.capacity_top_k),
            "widths": _parse_int_list(args.capacity_widths, DEFAULT_CAPACITY_WIDTHS),
            "modes": _parse_int_list(args.capacity_modes, DEFAULT_CAPACITY_MODES),
            "layers": _parse_int_list(args.capacity_layers, DEFAULT_CAPACITY_LAYERS),
            "runtime": dict(capacity_runtime),
        },
        "parity": {
            "benchmark": str(args.parity_benchmark),
            "gate_train_doc_count": int(args.parity_gate_train_doc_count),
            "scale_train_doc_counts": _parse_int_list(args.parity_scale_train_doc_counts, DEFAULT_PARITY_SCALE_TRAIN_DOCS),
            "seeds": _parse_int_list(args.parity_seeds, DEFAULT_PARITY_SEEDS),
            "method_runs": _parse_run_axis_list(
                args.parity_method_runs,
                DEFAULT_PARITY_METHOD_RUNS,
                role="primary",
            ),
            "reference_method_runs": _parse_run_axis_list(
                args.parity_reference_method_runs,
                DEFAULT_REFERENCE_METHOD_RUNS,
                role="reference",
            ),
            "run_aux_upper_bound": bool(args.parity_run_aux_upper_bound),
            "upper_bound_aux_fractions": _parse_float_list(args.parity_upper_bound_aux_fractions, DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS),
            "backfill_enabled": bool(args.parity_backfill_on_success),
            "runtime": dict(parity_runtime),
        },
        "render": {
            "render_full_doc_parity_pdf": bool(args.render_full_doc_parity_pdf),
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
    }

    step_commands: List[Dict[str, Any]] = []
    if "tradeoff" in phases:
        step_commands.append(
            {
                "name": "tradeoff",
                "command": _tradeoff_command(
                    python_bin=str(args.python_bin),
                    preset=str(args.tradeoff_preset),
                    device_mode=str(args.tradeoff_device_mode),
                    train_docs=int(args.tradeoff_train_docs),
                    phases=str(args.tradeoff_phases),
                    output_root=tradeoff_root,
                    mig_uuids=mig_uuids,
                    selection_config=args.selection_config,
                    tree_reference_mode=str(tradeoff_tree_reference["mode"]),
                    tree_reference_capacity_root=tradeoff_tree_reference["capacity_root"],
                    tree_reference_preset=str(tradeoff_tree_reference.get("preset", "") or ""),
                    runtime_data_mode=str(getattr(args, "tradeoff_runtime_data_mode", "resident")),
                    runtime_bucket_mode=str(getattr(args, "tradeoff_runtime_bucket_mode", "leaf_count_auto_queue")),
                    supervision_recovery_tree_family=str(
                        getattr(args, "tradeoff_supervision_recovery_tree_family", "tree_neural")
                    ),
                    supervision_recovery_structural_cell=str(
                        getattr(args, "tradeoff_supervision_recovery_structural_cell", "r12_p079")
                    ),
                    supervision_recovery_train_docs=None,
                    supervision_recovery_seeds=None,
                    supervision_recovery_packages=None,
                    tree_exact_eval_max_docs=int(
                        getattr(args, "tradeoff_tree_exact_eval_max_docs", 0) or 0
                    ),
                    prepared_data_root=getattr(args, "tradeoff_prepared_data_root", None),
                    prepared_data_allow_create=bool(
                        getattr(args, "tradeoff_prepared_data_allow_create", True)
                    ),
                    diagnostic_detail_mode=str(
                        getattr(args, "tradeoff_diagnostic_detail_mode", "summary")
                    ),
                    raw_diagnostic_artifact_dir=getattr(
                        args, "tradeoff_raw_diagnostic_artifact_dir", None
                    ),
                ),
                "output_root": str(tradeoff_root),
            }
        )
    if "capacity" in phases:
        step_commands.append(
            {
                "name": "capacity",
                "command": _capacity_command(
                    python_bin=str(args.python_bin),
                    output_root=capacity_root,
                    benchmark=str(args.capacity_benchmark),
                    train_doc_count=int(args.capacity_train_doc_count),
                    screen_seeds=_parse_int_list(args.capacity_screen_seeds, DEFAULT_CAPACITY_SCREEN_SEEDS),
                    locked_seeds=_parse_int_list(args.capacity_locked_seeds, DEFAULT_CAPACITY_LOCKED_SEEDS),
                    top_k=int(args.capacity_top_k),
                    widths=_parse_int_list(args.capacity_widths, DEFAULT_CAPACITY_WIDTHS),
                    modes=_parse_int_list(args.capacity_modes, DEFAULT_CAPACITY_MODES),
                    layers=_parse_int_list(args.capacity_layers, DEFAULT_CAPACITY_LAYERS),
                    runtime_data_mode=str(capacity_runtime["data_mode"]),
                    runtime_bucket_mode=str(capacity_runtime["bucket_mode"]),
                    runtime_preload_splits=capacity_runtime["preload_splits"],
                    runtime_preload_targets=bool(capacity_runtime["preload_targets"]),
                    runtime_workers_per_mig=int(capacity_runtime["workers_per_mig"]),
                    runtime_allow_multi_worker_screen=bool(
                        capacity_runtime["allow_multi_worker_screen"]
                    ),
                    runtime_capacity_workers_per_mig=int(
                        capacity_runtime["capacity_workers_per_mig"]
                    ),
                    mig_uuids=mig_uuids,
                ),
                "output_root": str(capacity_root),
            }
        )
    if "parity" in phases:
        step_commands.append(
            {
                "name": "parity",
                "command": _parity_command(
                    python_bin=str(args.python_bin),
                    output_root=parity_root,
                    benchmark=str(args.parity_benchmark),
                    gate_train_doc_count=int(args.parity_gate_train_doc_count),
                    scale_train_doc_counts=_parse_int_list(args.parity_scale_train_doc_counts, DEFAULT_PARITY_SCALE_TRAIN_DOCS),
                    seeds=_parse_int_list(args.parity_seeds, DEFAULT_PARITY_SEEDS),
                    method_runs=_parse_run_axis_list(
                        getattr(args, "parity_method_runs", None),
                        DEFAULT_PARITY_METHOD_RUNS,
                        role="primary",
                    ),
                    reference_method_runs=_parse_run_axis_list(
                        getattr(args, "parity_reference_method_runs", None),
                        DEFAULT_REFERENCE_METHOD_RUNS,
                        role="reference",
                    ),
                    capacity_root=(capacity_root if ("capacity" in phases or args.capacity_root is not None) else None),
                    run_aux_upper_bound=bool(args.parity_run_aux_upper_bound),
                    upper_bound_aux_fractions=_parse_float_list(args.parity_upper_bound_aux_fractions, DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS),
                    backfill_on_success=bool(args.parity_backfill_on_success),
                    runtime_data_mode=str(parity_runtime["data_mode"]),
                    runtime_bucket_mode=str(parity_runtime["bucket_mode"]),
                    runtime_preload_splits=parity_runtime["preload_splits"],
                    runtime_preload_targets=bool(parity_runtime["preload_targets"]),
                    runtime_workers_per_mig=int(parity_runtime["workers_per_mig"]),
                    runtime_allow_multi_worker_screen=bool(
                        parity_runtime["allow_multi_worker_screen"]
                    ),
                    runtime_capacity_workers_per_mig=int(
                        parity_runtime["capacity_workers_per_mig"]
                    ),
                    mig_uuids=mig_uuids,
                ),
                "output_root": str(parity_root),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "preflight_root": str(preflight_root),
        "tradeoff_root": str(tradeoff_root),
        "capacity_root": str(capacity_root),
        "parity_root": str(parity_root),
        "bundle_root": str(bundle_root),
        "migs": list(mig_uuids),
        "scheduler": dict(resolved_selection["scheduler"]),
        "eta_estimate": eta_estimate,
        "resolved_selection": resolved_selection,
        "tradeoff_run_plan": tradeoff_plan,
        "step_commands": step_commands,
    }


def _build_parser(*, config_defaults: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the full publication-ready Markov report bundle: tradeoff pipeline, "
            "tree-vs-FNO capacity/parity, and final artifact manifest.\n\n"
            "Recommended workflow:\n"
            "  1. Start from config/markov/publication_bundle.standard.toml or generate\n"
            "     a custom TOML with --write-config-template.\n"
            "  2. Inspect the resolved run with --config ... --plan-only.\n"
            "  3. Launch the real job with --config ... --detach.\n\n"
            "Direct --tradeoff-*, --capacity-*, and --parity-* flags remain available\n"
            "as advanced overrides on top of the resolved config."
        ),
        epilog=(
            "Examples:\n"
            "  python3 scripts/run_markov_publication_bundle.py \\\n"
            "    --config config/markov/publication_bundle.standard.toml \\\n"
            "    --plan-only\n\n"
            "  python3 scripts/run_markov_publication_bundle.py \\\n"
            "    --config config/markov/publication_bundle.standard.toml \\\n"
            "    --detach \\\n"
            "    --output-root outputs/markov_publication_bundle_$(date +%Y%m%d_%H%M%S)\n\n"
            "  python3 scripts/run_markov_publication_bundle.py \\\n"
            "    --write-config-template outputs/markov_publication_bundle.custom.toml"
        ),
    )
    parser.add_argument(
        "--selection-config",
        "--config",
        dest="selection_config",
        type=Path,
        default=None,
        help="Path to a .toml or .json publication-run config file. Recommended primary interface.",
    )
    parser.add_argument(
        "--write-selection-template",
        "--write-config-template",
        dest="write_selection_template",
        type=Path,
        default=None,
        help="Write a starter .toml or .json publication config template and exit. Prefer committing important run configs under config/markov/.",
    )
    parser.add_argument("--write-run-plan", type=Path, default=None, help="Write the fully resolved run plan JSON and exit or continue.")
    parser.add_argument("--plan-only", action=argparse.BooleanOptionalAction, default=False, help="Print the fully resolved run plan and exit.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / f"markov_publication_bundle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--phases", type=str, default=",".join(DEFAULT_PHASES))
    parser.add_argument("--estimate-only", action=argparse.BooleanOptionalAction, default=False, help="Print the runtime estimate only.")
    parser.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch via scripts/long_job.py instead of running in the current shell. Recommended for long-running jobs.",
    )
    parser.add_argument("--detach-name", type=str, default="markov_publication_bundle")
    parser.add_argument("--detach-job-root", type=Path, default=None)
    parser.add_argument("--detach-description", type=str, default="Detached Markov publication bundle run.")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--migs", type=str, default="")
    parser.add_argument("--scheduler-mode", choices=("global_per_run",), default="global_per_run")
    parser.add_argument("--default-job-granularity", choices=("family_train_seed", "family_train"), default="family_train_seed")
    parser.add_argument("--cleanup-stale-children", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gpu-items-per-mig", type=int, default=1)
    parser.add_argument("--tradeoff-root", type=Path, default=None)
    parser.add_argument(
        "--tradeoff-preset",
        choices=("smoke", "standard", "v3"),
        default="v3",
    )
    parser.add_argument("--tradeoff-device-mode", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--tradeoff-phases", type=str, default=DEFAULT_TRADEOFF_PHASES)
    parser.add_argument("--tradeoff-train-docs", type=int, default=10240)
    parser.add_argument("--tradeoff-supervision-recovery-method-id", choices=("tree_neural",), default="tree_neural")
    parser.add_argument("--tradeoff-tree-exact-eval-max-docs", type=int, default=0)
    parser.add_argument("--tradeoff-prepared-data-root", type=Path, default=None)
    parser.add_argument(
        "--tradeoff-runtime-data-mode",
        choices=("resident", "cpu_debug"),
        default="resident",
    )
    parser.add_argument(
        "--tradeoff-runtime-bucket-mode",
        choices=("exact_then_bucketed", "leaf_count_auto_queue"),
        default="leaf_count_auto_queue",
    )
    parser.add_argument(
        "--tradeoff-diagnostic-detail-mode",
        choices=("summary", "debug_raw"),
        default="summary",
    )
    parser.add_argument("--tradeoff-raw-diagnostic-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--tradeoff-prepared-data-allow-create",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tradeoff-tree-reference-mode",
        choices=("default", "capacity_locked", "preset"),
        default="default",
    )
    parser.add_argument("--tradeoff-tree-reference-capacity-root", type=Path, default=None)
    parser.add_argument("--tradeoff-tree-reference-preset", type=str, default="")

    parser.add_argument("--capacity-root", type=Path, default=None)
    parser.add_argument("--capacity-benchmark", type=str, default="recoverable_v5")
    parser.add_argument("--capacity-train-doc-count", type=int, default=10240)
    parser.add_argument("--capacity-screen-seeds", type=str, default="0,1,2")
    parser.add_argument("--capacity-locked-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--capacity-top-k", type=int, default=3)
    parser.add_argument("--capacity-widths", type=str, default="64,128,256")
    parser.add_argument("--capacity-modes", type=str, default="2,4,8")
    parser.add_argument("--capacity-layers", type=str, default="2,4,6")
    parser.add_argument("--capacity-runtime-data-mode", choices=("resident", "cpu_debug"), default="resident")
    parser.add_argument("--capacity-runtime-bucket-mode", choices=("exact_then_bucketed",), default="exact_then_bucketed")
    parser.add_argument("--capacity-runtime-preload-splits", type=str, default="train val test")
    parser.add_argument("--capacity-runtime-preload-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capacity-runtime-workers-per-mig", type=int, default=1)
    parser.add_argument("--capacity-runtime-allow-multi-worker-screen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capacity-runtime-capacity-workers-per-mig", type=int, default=2)

    parser.add_argument("--parity-root", type=Path, default=None)
    parser.add_argument("--parity-benchmark", type=str, default="recoverable_v5_t128")
    parser.add_argument("--parity-gate-train-doc-count", type=int, default=10240)
    parser.add_argument("--parity-scale-train-doc-counts", type=str, default="1024,2048,3072,4096,5120,8192,10240")
    parser.add_argument("--parity-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--parity-method-runs", type=str, default=" ".join(DEFAULT_PARITY_METHOD_RUNS))
    parser.add_argument("--parity-reference-method-runs", type=str, default="official_fno official_fno_sumlen")
    parser.add_argument(
        "--parity-backfill-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Legacy flag name. When enabled, always run the parity scale "
            "backfill after the gate so the publication bundle includes "
            "the full multi-scale curve."
        ),
    )
    parser.add_argument("--parity-run-aux-upper-bound", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity-upper-bound-aux-fractions", type=str, default="0.25,1.0")
    parser.add_argument("--parity-runtime-data-mode", choices=("resident", "cpu_debug"), default="resident")
    parser.add_argument("--parity-runtime-bucket-mode", choices=("exact_then_bucketed",), default="exact_then_bucketed")
    parser.add_argument("--parity-runtime-preload-splits", type=str, default="train val test")
    parser.add_argument("--parity-runtime-preload-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity-runtime-workers-per-mig", type=int, default=1)
    parser.add_argument("--parity-runtime-allow-multi-worker-screen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--parity-runtime-capacity-workers-per-mig", type=int, default=2)

    parser.add_argument("--bundle-root", type=Path, default=None)
    parser.add_argument("--render-full-doc-parity-pdf", action=argparse.BooleanOptionalAction, default=True)
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
        section_names=("publication_bundle", "markov_publication_bundle"),
        flatten_sections=True,
    )
    parser = _build_parser(config_defaults=config_defaults)
    args = parser.parse_args(raw_argv)
    method_runs = _parse_run_axis_list(
        getattr(args, "parity_method_runs", None),
        DEFAULT_PARITY_METHOD_RUNS,
        role="primary",
    )
    reference_method_runs = _parse_run_axis_list(
        getattr(args, "parity_reference_method_runs", None),
        DEFAULT_REFERENCE_METHOD_RUNS,
        role="reference",
    )
    # Private adapter aliases for legacy child commands.
    args.parity_tree_families = " ".join(_legacy_families_from_run_axes(method_runs))
    args.parity_fno_families = " ".join(_method_ids_from_run_axes(reference_method_runs))
    args.tradeoff_supervision_recovery_tree_family = str(
        getattr(args, "tradeoff_supervision_recovery_method_id", "tree_neural")
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _parse_args(raw_argv)
    if args.write_selection_template is not None:
        _write_selection_template(Path(args.write_selection_template))
        print(json.dumps({"selection_template": str(Path(args.write_selection_template).expanduser())}, indent=2))
        return 0
    output_root = Path(args.output_root).resolve()
    preflight_root = output_root / "preflight"
    tradeoff_root = (args.tradeoff_root.resolve() if args.tradeoff_root is not None else (output_root / "tradeoff_pipeline"))
    capacity_root = (args.capacity_root.resolve() if args.capacity_root is not None else (output_root / "tree_fno_capacity"))
    parity_root = (args.parity_root.resolve() if args.parity_root is not None else (output_root / "tree_fno_parity"))
    bundle_root = (args.bundle_root.resolve() if args.bundle_root is not None else (output_root / "publication_bundle"))
    preflight_tradeoff_root = preflight_root / "tradeoff_pipeline"
    preflight_capacity_root = preflight_root / "tree_fno_capacity"
    preflight_parity_root = preflight_root / "tree_fno_parity"
    preflight_bundle_root = preflight_root / "publication_bundle"
    logs_root = output_root / "logs"

    mig_uuids = _resolved_mig_uuids(args)
    mig_count = len(mig_uuids) if mig_uuids else 1
    run_plan = build_publication_run_plan(
        args,
        mig_uuids=mig_uuids,
        output_root=output_root,
    )
    public_run_plan = _public_payload_for_contract(run_plan)
    assert_public_contract_clean(public_run_plan, surface="markov publication run plan")
    if args.write_run_plan is not None:
        _write_json(Path(args.write_run_plan).expanduser(), public_run_plan)
    if bool(args.plan_only):
        print(json.dumps(public_run_plan, indent=2, sort_keys=True))
        return 0
    eta_estimate = estimate_publication_runtime(args, mig_count=mig_count)
    if bool(args.estimate_only):
        print(json.dumps(eta_estimate, indent=2, sort_keys=True))
        return 0
    if bool(args.detach):
        return _launch_detached_bundle(
            raw_argv=raw_argv,
            args=args,
            output_root=output_root,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    preflight_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    phases = set(_parse_phase_set(args.phases))
    _validate_phase_dependencies(args, phases)
    steps: List[StepResult] = []
    preflight_steps: List[StepResult] = []

    python_bin = str(args.python_bin)
    mpl_env = {"MPLBACKEND": "Agg"}
    with_preflight = bool(args.with_preflight)
    preflight_tradeoff_phases = str(args.tradeoff_phases)
    preflight_tradeoff_kwargs = _preflight_tradeoff_kwargs(
        args,
        phases=preflight_tradeoff_phases,
    )
    preflight_capacity_needed = bool({"capacity", "tree_fno_pdf"} & phases)
    preflight_parity_needed = bool({"parity", "tree_fno_pdf", "full_doc_parity_pdf"} & phases)

    if with_preflight:
        if preflight_capacity_needed:
            preflight_steps.append(
                _maybe_run_step(
                    name="preflight_capacity",
                    command=_capacity_command(
                        python_bin=python_bin,
                        output_root=preflight_capacity_root,
                        benchmark=str(args.capacity_benchmark),
                        train_doc_count=PREFLIGHT_CAPACITY_TRAIN_DOC_COUNT,
                        screen_seeds=PREFLIGHT_CAPACITY_SCREEN_SEEDS,
                        locked_seeds=PREFLIGHT_CAPACITY_LOCKED_SEEDS,
                        top_k=PREFLIGHT_CAPACITY_TOP_K,
                        widths=PREFLIGHT_CAPACITY_WIDTHS,
                        modes=PREFLIGHT_CAPACITY_MODES,
                        layers=PREFLIGHT_CAPACITY_LAYERS,
                        runtime_data_mode=str(args.capacity_runtime_data_mode),
                        runtime_bucket_mode=str(args.capacity_runtime_bucket_mode),
                        runtime_preload_splits=_parse_str_list(
                            args.capacity_runtime_preload_splits,
                            ("train", "val", "test"),
                        ),
                        runtime_preload_targets=bool(args.capacity_runtime_preload_targets),
                        runtime_workers_per_mig=int(args.capacity_runtime_workers_per_mig),
                        runtime_allow_multi_worker_screen=bool(
                            args.capacity_runtime_allow_multi_worker_screen
                        ),
                        runtime_capacity_workers_per_mig=int(
                            args.capacity_runtime_capacity_workers_per_mig
                        ),
                        mig_uuids=mig_uuids,
                    ),
                    log_path=logs_root / "preflight_tree_fno_capacity.log",
                    expected_outputs=_expected_capacity_outputs(preflight_capacity_root),
                    output_root=preflight_capacity_root,
                    cwd=REPO_ROOT,
                    reuse_existing=bool(args.reuse_existing),
                    env=mpl_env,
                )
            )

        preflight_tradeoff_tree_reference = _resolve_tradeoff_tree_reference(
            args,
            capacity_root=(
                preflight_capacity_root
                if preflight_capacity_needed
                else (
                    capacity_root
                    if args.capacity_root is not None
                    else None
                )
            ),
        )
        if "tradeoff" in phases:
            preflight_steps.append(
                _maybe_run_step(
                    name="preflight_tradeoff",
                    command=_tradeoff_command(
                        python_bin=python_bin,
                        preset=str(preflight_tradeoff_kwargs["preset"]),
                        device_mode=str(args.tradeoff_device_mode),
                        train_docs=int(args.tradeoff_train_docs),
                        phases=preflight_tradeoff_phases,
                        output_root=preflight_tradeoff_root,
                        mig_uuids=mig_uuids,
                        selection_config=preflight_tradeoff_kwargs["selection_config"],
                        tree_reference_mode=str(preflight_tradeoff_tree_reference["mode"]),
                        tree_reference_capacity_root=preflight_tradeoff_tree_reference["capacity_root"],
                        tree_reference_preset=str(preflight_tradeoff_tree_reference.get("preset", "") or ""),
                        runtime_data_mode=str(getattr(args, "tradeoff_runtime_data_mode", "resident")),
                        runtime_bucket_mode=str(getattr(args, "tradeoff_runtime_bucket_mode", "leaf_count_auto_queue")),
                        supervision_recovery_tree_family=str(
                            preflight_tradeoff_kwargs["supervision_recovery_tree_family"]
                        ),
                        supervision_recovery_structural_cell=str(
                            preflight_tradeoff_kwargs["supervision_recovery_structural_cell"]
                        ),
                        supervision_recovery_train_docs=preflight_tradeoff_kwargs[
                            "supervision_recovery_train_docs"
                        ],
                        supervision_recovery_seeds=preflight_tradeoff_kwargs[
                            "supervision_recovery_seeds"
                        ],
                        supervision_recovery_packages=preflight_tradeoff_kwargs[
                            "supervision_recovery_packages"
                        ],
                        tree_exact_eval_max_docs=int(
                            getattr(args, "tradeoff_tree_exact_eval_max_docs", 0) or 0
                        ),
                        prepared_data_root=getattr(args, "tradeoff_prepared_data_root", None),
                        prepared_data_allow_create=bool(
                            getattr(args, "tradeoff_prepared_data_allow_create", True)
                        ),
                        diagnostic_detail_mode=str(
                            getattr(args, "tradeoff_diagnostic_detail_mode", "summary")
                        ),
                        raw_diagnostic_artifact_dir=getattr(
                            args, "tradeoff_raw_diagnostic_artifact_dir", None
                        ),
                    ),
                    log_path=logs_root / "preflight_tradeoff_pipeline.log",
                    expected_outputs=_expected_tradeoff_outputs(preflight_tradeoff_root),
                    output_root=preflight_tradeoff_root,
                    cwd=REPO_ROOT,
                    reuse_existing=bool(args.reuse_existing),
                    env=mpl_env,
                )
            )

        if preflight_parity_needed:
            preflight_capacity_root_arg = (
                preflight_capacity_root if preflight_capacity_needed else None
            )
            preflight_steps.append(
                _maybe_run_step(
                    name="preflight_parity",
                    command=_parity_command(
                        python_bin=python_bin,
                        output_root=preflight_parity_root,
                        benchmark=str(args.parity_benchmark),
                        gate_train_doc_count=PREFLIGHT_PARITY_GATE_TRAIN_DOC_COUNT,
                        scale_train_doc_counts=PREFLIGHT_PARITY_SCALE_TRAIN_DOCS,
                        seeds=PREFLIGHT_PARITY_SEEDS,
                        method_runs=_parse_run_axis_list(
                            getattr(args, "parity_method_runs", None),
                            DEFAULT_PARITY_METHOD_RUNS,
                            role="primary",
                        ),
                        reference_method_runs=_parse_run_axis_list(
                            getattr(args, "parity_reference_method_runs", None),
                            DEFAULT_REFERENCE_METHOD_RUNS,
                            role="reference",
                        ),
                        capacity_root=preflight_capacity_root_arg,
                        run_aux_upper_bound=False,
                        upper_bound_aux_fractions=PREFLIGHT_PARITY_UPPER_BOUND_AUX_FRACTIONS,
                        backfill_on_success=False,
                        runtime_data_mode=str(args.parity_runtime_data_mode),
                        runtime_bucket_mode=str(args.parity_runtime_bucket_mode),
                        runtime_preload_splits=_parse_str_list(
                            args.parity_runtime_preload_splits,
                            ("train", "val", "test"),
                        ),
                        runtime_preload_targets=bool(args.parity_runtime_preload_targets),
                        runtime_workers_per_mig=int(args.parity_runtime_workers_per_mig),
                        runtime_allow_multi_worker_screen=bool(
                            args.parity_runtime_allow_multi_worker_screen
                        ),
                        runtime_capacity_workers_per_mig=int(
                            args.parity_runtime_capacity_workers_per_mig
                        ),
                        mig_uuids=mig_uuids,
                    ),
                    log_path=logs_root / "preflight_tree_fno_parity.log",
                    expected_outputs=_expected_parity_outputs(preflight_parity_root),
                    output_root=preflight_parity_root,
                    cwd=REPO_ROOT,
                    reuse_existing=bool(args.reuse_existing),
                    env=mpl_env,
                )
            )

        preflight_tree_fno_pdf = preflight_bundle_root / "tree_fno_tuning_report.pdf"
        if "tree_fno_pdf" in phases:
            preflight_steps.append(
                _maybe_run_step(
                    name="preflight_tree_fno_pdf",
                    command=_tree_fno_pdf_command(
                        python_bin=python_bin,
                        capacity_root=preflight_capacity_root,
                        parity_root=preflight_parity_root,
                        output_pdf=preflight_tree_fno_pdf,
                    ),
                    log_path=logs_root / "preflight_tree_fno_tuning_pdf.log",
                    expected_outputs=[preflight_tree_fno_pdf],
                    output_root=preflight_bundle_root,
                    cwd=REPO_ROOT,
                    reuse_existing=bool(args.reuse_existing),
                    env=mpl_env,
                )
            )

        preflight_full_doc_pdf = preflight_bundle_root / "full_doc_parity_report.pdf"
        if "full_doc_parity_pdf" in phases and bool(args.render_full_doc_parity_pdf):
            preflight_steps.append(
                _maybe_run_step(
                    name="preflight_full_doc_parity_pdf",
                    command=_full_doc_parity_pdf_command(
                        python_bin=python_bin,
                        parity_root=preflight_parity_root,
                        output_pdf=preflight_full_doc_pdf,
                    ),
                    log_path=logs_root / "preflight_full_doc_parity_pdf.log",
                    expected_outputs=[preflight_full_doc_pdf],
                    output_root=preflight_bundle_root,
                    cwd=REPO_ROOT,
                    reuse_existing=bool(args.reuse_existing),
                    env=mpl_env,
                )
            )

    reference_contract = {
        "identifiable_zero_reference_kind": "full_doc_fno_upper_bound",
        "full_doc_fno_families": list(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES),
        "full_doc_fno_training_backend": (
            "src.ctreepo.sim.core.markov_neural_operator_baselines._train_loop_with_predictions"
        ),
        "note": (
            "The publication bundle treats the full-doc FNO upper bound as the "
            "canonical identifiable-zero reference and intentionally excludes the "
            "older mixed tree/FNO appendix path."
        ),
    }

    if bool(args.preflight_only):
        preflight_artifacts = _build_artifact_map(
            tradeoff_root=preflight_tradeoff_root,
            capacity_root=preflight_capacity_root,
            parity_root=preflight_parity_root,
            bundle_root=preflight_bundle_root,
            include_full_doc_parity_pdf=bool(args.render_full_doc_parity_pdf),
        )
        preflight_tree_bundle_contract = _markov_publication_tree_bundle_contract(
            args=args,
            phases=phases,
        )
        preflight_bundle_manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(preflight_root),
            "bundle_root": str(preflight_bundle_root),
            "migs": mig_uuids,
            "with_preflight": with_preflight,
            "preflight_only": True,
            "eta_estimate": eta_estimate,
            "steps": [asdict(step) for step in preflight_steps],
            "artifacts": preflight_artifacts,
            "tree_bundle_contract": preflight_tree_bundle_contract,
            "reference_contract": dict(reference_contract),
        }
        preflight_bundle_manifest["run_manifest"] = _markov_publication_run_manifest(
            args=args,
            output_root=preflight_root,
            phases=phases,
            tree_bundle_contract=preflight_tree_bundle_contract,
            artifacts=preflight_artifacts,
            status="completed",
            publication_ready=False,
            metadata={"preflight_only": True},
        )
        if "bundle" in phases:
            manifest_path, index_path = _write_bundle_outputs(
                manifest=preflight_bundle_manifest,
                bundle_root=preflight_bundle_root,
            )
            print(
                json.dumps(
                    {
                        "output_root": str(preflight_root),
                        "bundle_root": str(preflight_bundle_root),
                        "publication_manifest_json": str(manifest_path),
                        "publication_index_md": str(index_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(preflight_bundle_manifest, indent=2, sort_keys=True))
        return 0

    if str(getattr(args, "scheduler_mode", "global_per_run")) == "global_per_run":
        manifest = _run_publication_global_scheduler(
            args=args,
            output_root=output_root,
            mig_uuids=mig_uuids,
            tradeoff_root=tradeoff_root,
            capacity_root=capacity_root,
            parity_root=parity_root,
            bundle_root=bundle_root,
            phases=sorted(phases),
            eta_estimate=eta_estimate,
        )
        artifacts = _build_artifact_map(
            tradeoff_root=tradeoff_root,
            capacity_root=capacity_root,
            parity_root=parity_root,
            bundle_root=bundle_root,
            include_full_doc_parity_pdf=bool(args.render_full_doc_parity_pdf),
        )
        manifest["artifacts"] = artifacts
        manifest["reference_contract"] = dict(reference_contract)
        if with_preflight:
            preflight_artifacts = _build_artifact_map(
                tradeoff_root=preflight_tradeoff_root,
                capacity_root=preflight_capacity_root,
                parity_root=preflight_parity_root,
                bundle_root=preflight_bundle_root,
                include_full_doc_parity_pdf=bool(args.render_full_doc_parity_pdf),
            )
            manifest["preflight"] = {
                "output_root": str(preflight_root),
                "bundle_root": str(preflight_bundle_root),
                "steps": [asdict(step) for step in preflight_steps],
                "artifacts": preflight_artifacts,
            }
        if "bundle" in phases:
            print(
                json.dumps(
                    {
                        "output_root": str(output_root),
                        "bundle_root": str(bundle_root),
                        "publication_manifest_json": str(bundle_root / "publication_manifest.json"),
                        "publication_index_md": str(bundle_root / "publication_index.md"),
                        "tradeoff_report_pdf": artifacts.get("tradeoff_report_pdf", ""),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if "capacity" in phases:
        steps.append(
            _maybe_run_step(
                name="capacity",
                command=_capacity_command(
                    python_bin=python_bin,
                    output_root=capacity_root,
                    benchmark=str(args.capacity_benchmark),
                    train_doc_count=int(args.capacity_train_doc_count),
                    screen_seeds=_parse_int_list(args.capacity_screen_seeds, DEFAULT_CAPACITY_SCREEN_SEEDS),
                    locked_seeds=_parse_int_list(args.capacity_locked_seeds, DEFAULT_CAPACITY_LOCKED_SEEDS),
                    top_k=int(args.capacity_top_k),
                    widths=_parse_int_list(args.capacity_widths, DEFAULT_CAPACITY_WIDTHS),
                    modes=_parse_int_list(args.capacity_modes, DEFAULT_CAPACITY_MODES),
                    layers=_parse_int_list(args.capacity_layers, DEFAULT_CAPACITY_LAYERS),
                    runtime_data_mode=str(args.capacity_runtime_data_mode),
                    runtime_bucket_mode=str(args.capacity_runtime_bucket_mode),
                    runtime_preload_splits=_parse_str_list(
                        args.capacity_runtime_preload_splits,
                        ("train", "val", "test"),
                    ),
                    runtime_preload_targets=bool(args.capacity_runtime_preload_targets),
                    runtime_workers_per_mig=int(args.capacity_runtime_workers_per_mig),
                    runtime_allow_multi_worker_screen=bool(
                        args.capacity_runtime_allow_multi_worker_screen
                    ),
                    runtime_capacity_workers_per_mig=int(
                        args.capacity_runtime_capacity_workers_per_mig
                    ),
                    mig_uuids=mig_uuids,
                ),
                log_path=logs_root / "tree_fno_capacity.log",
                expected_outputs=_expected_capacity_outputs(capacity_root),
                output_root=capacity_root,
                cwd=REPO_ROOT,
                reuse_existing=bool(args.reuse_existing),
                env=mpl_env,
            )
        )

    if "tradeoff" in phases:
        main_tradeoff_tree_reference = _resolve_tradeoff_tree_reference(
            args,
            capacity_root=(
                capacity_root
                if ("capacity" in phases or args.capacity_root is not None)
                else None
            ),
        )
        steps.append(
            _maybe_run_step(
                name="tradeoff",
                command=_tradeoff_command(
                    python_bin=python_bin,
                    preset=str(args.tradeoff_preset),
                    device_mode=str(args.tradeoff_device_mode),
                    train_docs=int(args.tradeoff_train_docs),
                    phases=str(args.tradeoff_phases),
                    output_root=tradeoff_root,
                    mig_uuids=mig_uuids,
                    selection_config=args.selection_config,
                    tree_reference_mode=str(main_tradeoff_tree_reference["mode"]),
                    tree_reference_capacity_root=main_tradeoff_tree_reference["capacity_root"],
                    tree_reference_preset=str(main_tradeoff_tree_reference.get("preset", "") or ""),
                    runtime_data_mode=str(getattr(args, "tradeoff_runtime_data_mode", "resident")),
                    runtime_bucket_mode=str(getattr(args, "tradeoff_runtime_bucket_mode", "leaf_count_auto_queue")),
                    supervision_recovery_tree_family=str(
                        getattr(args, "tradeoff_supervision_recovery_tree_family", "tree_neural")
                    ),
                    supervision_recovery_structural_cell=str(
                        getattr(args, "tradeoff_supervision_recovery_structural_cell", "r12_p079")
                    ),
                    supervision_recovery_train_docs=None,
                    supervision_recovery_seeds=None,
                    supervision_recovery_packages=None,
                    tree_exact_eval_max_docs=int(
                        getattr(args, "tradeoff_tree_exact_eval_max_docs", 0) or 0
                    ),
                    prepared_data_root=getattr(args, "tradeoff_prepared_data_root", None),
                    prepared_data_allow_create=bool(
                        getattr(args, "tradeoff_prepared_data_allow_create", True)
                    ),
                    diagnostic_detail_mode=str(
                        getattr(args, "tradeoff_diagnostic_detail_mode", "summary")
                    ),
                    raw_diagnostic_artifact_dir=getattr(
                        args, "tradeoff_raw_diagnostic_artifact_dir", None
                    ),
                ),
                log_path=logs_root / "tradeoff_pipeline.log",
                expected_outputs=_expected_tradeoff_outputs(tradeoff_root),
                output_root=tradeoff_root,
                cwd=REPO_ROOT,
                reuse_existing=bool(args.reuse_existing),
                env=mpl_env,
            )
        )

    if "parity" in phases:
        parity_capacity_root: Path | None = None
        if "capacity" in phases or args.capacity_root is not None:
            parity_capacity_root = capacity_root
        steps.append(
            _maybe_run_step(
                name="parity",
                command=_parity_command(
                    python_bin=python_bin,
                    output_root=parity_root,
                    benchmark=str(args.parity_benchmark),
                    gate_train_doc_count=int(args.parity_gate_train_doc_count),
                    scale_train_doc_counts=_parse_int_list(
                        args.parity_scale_train_doc_counts,
                        DEFAULT_PARITY_SCALE_TRAIN_DOCS,
                    ),
                    seeds=_parse_int_list(args.parity_seeds, DEFAULT_PARITY_SEEDS),
                    method_runs=_parse_run_axis_list(
                        getattr(args, "parity_method_runs", None),
                        DEFAULT_PARITY_METHOD_RUNS,
                        role="primary",
                    ),
                    reference_method_runs=_parse_run_axis_list(
                        getattr(args, "parity_reference_method_runs", None),
                        DEFAULT_REFERENCE_METHOD_RUNS,
                        role="reference",
                    ),
                    capacity_root=parity_capacity_root,
                    run_aux_upper_bound=bool(args.parity_run_aux_upper_bound),
                    upper_bound_aux_fractions=_parse_float_list(
                        args.parity_upper_bound_aux_fractions,
                        DEFAULT_PARITY_UPPER_BOUND_AUX_FRACTIONS,
                    ),
                    backfill_on_success=bool(args.parity_backfill_on_success),
                    runtime_data_mode=str(args.parity_runtime_data_mode),
                    runtime_bucket_mode=str(args.parity_runtime_bucket_mode),
                    runtime_preload_splits=_parse_str_list(
                        args.parity_runtime_preload_splits,
                        ("train", "val", "test"),
                    ),
                    runtime_preload_targets=bool(args.parity_runtime_preload_targets),
                    runtime_workers_per_mig=int(args.parity_runtime_workers_per_mig),
                    runtime_allow_multi_worker_screen=bool(
                        args.parity_runtime_allow_multi_worker_screen
                    ),
                    runtime_capacity_workers_per_mig=int(
                        args.parity_runtime_capacity_workers_per_mig
                    ),
                    mig_uuids=mig_uuids,
                ),
                log_path=logs_root / "tree_fno_parity.log",
                expected_outputs=_expected_parity_outputs(parity_root),
                output_root=parity_root,
                cwd=REPO_ROOT,
                reuse_existing=bool(args.reuse_existing),
                env=mpl_env,
            )
        )

    tree_fno_pdf = bundle_root / "tree_fno_tuning_report.pdf"
    if "tree_fno_pdf" in phases:
        steps.append(
            _maybe_run_step(
                name="tree_fno_pdf",
                command=_tree_fno_pdf_command(
                    python_bin=python_bin,
                    capacity_root=capacity_root,
                    parity_root=parity_root,
                    output_pdf=tree_fno_pdf,
                ),
                log_path=logs_root / "tree_fno_tuning_pdf.log",
                expected_outputs=[tree_fno_pdf],
                output_root=bundle_root,
                cwd=REPO_ROOT,
                reuse_existing=bool(args.reuse_existing),
                env=mpl_env,
            )
        )

    full_doc_parity_pdf = bundle_root / "full_doc_parity_report.pdf"
    if "full_doc_parity_pdf" in phases and bool(args.render_full_doc_parity_pdf):
        steps.append(
            _maybe_run_step(
                name="full_doc_parity_pdf",
                command=_full_doc_parity_pdf_command(
                    python_bin=python_bin,
                    parity_root=parity_root,
                    output_pdf=full_doc_parity_pdf,
                ),
                log_path=logs_root / "full_doc_parity_pdf.log",
                expected_outputs=[full_doc_parity_pdf],
                output_root=bundle_root,
                cwd=REPO_ROOT,
                reuse_existing=bool(args.reuse_existing),
                env=mpl_env,
            )
        )

    artifacts = _build_artifact_map(
        tradeoff_root=tradeoff_root,
        capacity_root=capacity_root,
        parity_root=parity_root,
        bundle_root=bundle_root,
        include_full_doc_parity_pdf=bool(args.render_full_doc_parity_pdf),
    )
    tree_bundle_contract = _markov_publication_tree_bundle_contract(
        args=args,
        phases=phases,
    )
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "preflight_root": str(preflight_root),
        "tradeoff_root": str(tradeoff_root),
        "capacity_root": str(capacity_root),
        "parity_root": str(parity_root),
        "bundle_root": str(bundle_root),
        "migs": mig_uuids,
        "with_preflight": with_preflight,
        "preflight_only": bool(args.preflight_only),
        "eta_estimate": eta_estimate,
        "steps": [asdict(step) for step in steps],
        "artifacts": artifacts,
        "tree_bundle_contract": tree_bundle_contract,
        "reference_contract": dict(reference_contract),
    }
    manifest["run_manifest"] = _markov_publication_run_manifest(
        args=args,
        output_root=output_root,
        phases=phases,
        tree_bundle_contract=tree_bundle_contract,
        artifacts=artifacts,
        status="completed",
        publication_ready=not bool(args.preflight_only),
        metadata={"scheduler_mode": str(getattr(args, "scheduler_mode", ""))},
    )

    if with_preflight:
        preflight_artifacts = _build_artifact_map(
            tradeoff_root=preflight_tradeoff_root,
            capacity_root=preflight_capacity_root,
            parity_root=preflight_parity_root,
            bundle_root=preflight_bundle_root,
            include_full_doc_parity_pdf=bool(args.render_full_doc_parity_pdf),
        )
        preflight_tree_bundle_contract = _markov_publication_tree_bundle_contract(
            args=args,
            phases=phases,
        )
        preflight_manifest: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(preflight_root),
            "tradeoff_root": str(preflight_tradeoff_root),
            "capacity_root": str(preflight_capacity_root),
            "parity_root": str(preflight_parity_root),
            "bundle_root": str(preflight_bundle_root),
            "migs": mig_uuids,
            "with_preflight": True,
            "preflight_only": False,
            "eta_estimate": eta_estimate,
            "steps": [asdict(step) for step in preflight_steps],
            "artifacts": preflight_artifacts,
            "tree_bundle_contract": preflight_tree_bundle_contract,
            "reference_contract": dict(manifest["reference_contract"]),
        }
        preflight_manifest["run_manifest"] = _markov_publication_run_manifest(
            args=args,
            output_root=preflight_root,
            phases=phases,
            tree_bundle_contract=preflight_tree_bundle_contract,
            artifacts=preflight_artifacts,
            status="completed",
            publication_ready=False,
            metadata={"preflight": True},
        )
        if "bundle" in phases:
            preflight_manifest_path, preflight_index_path = _write_bundle_outputs(
                manifest=preflight_manifest,
                bundle_root=preflight_bundle_root,
            )
            preflight_steps.append(
                StepResult(
                    name="preflight_bundle",
                    status="completed",
                    wall_clock_s=0.0,
                    command=[],
                    log_path="",
                    output_root=str(preflight_bundle_root),
                    expected_outputs=[
                        str(preflight_manifest_path),
                        str(preflight_index_path),
                    ],
                )
            )
            preflight_manifest["steps"] = [asdict(step) for step in preflight_steps]
        manifest["preflight"] = {
            "output_root": str(preflight_root),
            "bundle_root": str(preflight_bundle_root),
            "steps": [asdict(step) for step in preflight_steps],
            "artifacts": preflight_artifacts,
        }

    if "bundle" in phases:
        manifest_path, index_path = _write_bundle_outputs(
            manifest=manifest,
            bundle_root=bundle_root,
        )
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "bundle_root": str(bundle_root),
                    "publication_manifest_json": str(manifest_path),
                    "publication_index_md": str(index_path),
                    "tradeoff_report_pdf": artifacts.get("tradeoff_report_pdf", ""),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
