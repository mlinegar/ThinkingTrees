#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PYTHON_BIN = REPO_ROOT / "venv" / "bin" / "python"
LONG_JOB_SCRIPT = REPO_ROOT / "scripts" / "long_job.py"
TRADEOFF_PIPELINE_SCRIPT = (
    REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"
)

from scripts.run_markov_optimization_tradeoff_pipeline import (  # noqa: E402
    _detect_available_mig_devices,
    _detect_mig_devices,
)

TRAIN_DOC_COUNT = 10240
DEFAULT_SEED = 0
TREE_FAMILY = "tree_neural"
FNO_TASK_FAMILY = "fno"
RECOVERABLE_SCOPE_KEY = "recoverable_v5_t128"
STRUCTURAL_SCOPE_KEY = "r12_p079"
STRUCTURAL_GRID_KEY = "structural_core_v2_t128"

SURFACE_FULL_GRID = "full_grid"
SURFACE_PUBLICATION_COMPACT = "publication_compact"
SURFACE_REPAIR_LEAF128_COUNTONLY = "repair_leaf128_countonly"
SURFACE_ALLOCATION_POLICY_GRID = "allocation_policy_grid"
RUN_STYLE_SUPPLEMENTAL = "supplemental_fill"
RUN_STYLE_CLEAN = "clean_rerun"

FULL_GRID_ROOT_SHARES = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)
PUBLICATION_COMPACT_ROOT_SHARES = (100, 50, 20, 10)
REPAIR_LEAF128_COUNTONLY_ROOT_SHARES = (90, 80, 70, 60, 50, 40, 30, 20, 10)
ALLOCATION_REPLACEMENT_ROOT_SHARES = FULL_GRID_ROOT_SHARES
ALLOCATION_MASS_PRESERVING_ROOT_SHARES = (90, 80, 70, 60, 50, 40, 30, 20, 10, 0)
LEAF_TOKEN_LADDER = (64, 32, 16, 8)
ALLOCATION_ROOT_ONLY_LEAF_TOKENS = (128, 64, 32, 16, 8)
ALLOCATION_LOCAL_ONLY_LEAF_TOKENS = (128, 64, 32, 16, 8)
ALLOCATION_INTERNAL_POLICY_LEAF_TOKENS = (32, 16, 8)

COMBINED_RUN_KEY = "combined_scheduler_run"
SUPPLEMENTAL_FULL_GRID_RUN_KEY = "combined_scheduler_full_grid_fill"
SUPPLEMENTAL_COMPACT_RUN_KEY = "combined_scheduler_publication_compact_fill"
REPAIR_RUN_KEY = "combined_scheduler_repair_leaf128_countonly"
ALLOCATION_POLICY_RUN_KEY = "combined_scheduler_allocation_policy_grid"
PRESERVED_COMPLETED_JOB_KEYS = (
    "oneleaf_root_budget_fixed10240_simple",
    "oneleaf_local_law_fixed10240_simple",
)
LEGACY_SUPERSEDED_JOB_KEYS = (
    "root_budget_ladder_large_train",
    "mass_preserving_leaf_only_large_train",
    "oneleaf_leaf_mass_root_sweep_fullval",
)
KNOWN_VISIBLE_JOB_KEYS = (
    COMBINED_RUN_KEY,
    SUPPLEMENTAL_FULL_GRID_RUN_KEY,
    SUPPLEMENTAL_COMPACT_RUN_KEY,
    REPAIR_RUN_KEY,
    ALLOCATION_POLICY_RUN_KEY,
    *PRESERVED_COMPLETED_JOB_KEYS,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_run(cmd: list[str], *, cwd: Path) -> dict:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    idx = 0
    last_obj = None
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        last_obj = obj
        idx = next_idx
    return dict(last_obj) if isinstance(last_obj, dict) else {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _maybe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _parse_explicit_migs(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").replace(",", " ").split() if item.strip()]


def _detect_launch_migs(
    explicit_migs: list[str] | None = None,
    *,
    available_only: bool = False,
) -> list[str]:
    if explicit_migs:
        return list(explicit_migs)
    if available_only:
        devices = list(_detect_available_mig_devices())
        if devices:
            return devices
    devices = list(_detect_mig_devices())
    if devices:
        return devices
    return list(_detect_available_mig_devices())


def _artifact_snapshot(output_root: Path) -> dict:
    artifacts_payload = _maybe_load_json(output_root / "artifacts.json")
    artifacts = dict(artifacts_payload.get("artifacts") or {})
    out = {}
    for artifact_id, payload in artifacts.items():
        path = str(dict(payload).get("path", "") or "").strip()
        if path and Path(path).exists():
            out[str(artifact_id)] = path
    for rel_name in (
        "pipeline_summary.json",
        "supervision_recovery/summary.json",
        "tradeoff_report/summary.json",
        "tradeoff_report/report.pdf",
        "markov_alignment_audit.json",
        "markov_alignment_audit.md",
    ):
        candidate = output_root / rel_name
        if candidate.exists():
            out.setdefault(candidate.name.replace(".", "_"), str(candidate))
    return out


def _custom_job_existing_state(job_root: Path, output_root: Path) -> dict:
    manifest = _maybe_load_json(job_root / "manifest.json")
    status = _maybe_load_json(output_root / "experiment_status.json")
    running = False
    if manifest:
        try:
            status_payload = _json_run(
                [
                    str(DEFAULT_PYTHON_BIN),
                    str(LONG_JOB_SCRIPT),
                    "status",
                    "--job-root",
                    str(job_root),
                ],
                cwd=REPO_ROOT,
            )
            running = bool(status_payload.get("running", False))
        except Exception:
            running = False
    state = str(status.get("state", "") or "")
    return {
        "manifest_path": str(job_root / "manifest.json")
        if (job_root / "manifest.json").exists()
        else "",
        "status_path": str(output_root / "experiment_status.json")
        if (output_root / "experiment_status.json").exists()
        else "",
        "running": running,
        "state": state,
    }


def _stop_long_job_if_running(*, python_bin: Path, job_root: Path) -> dict:
    manifest_path = job_root / "manifest.json"
    if not manifest_path.exists():
        return {
            "job_root": str(job_root),
            "stopped": False,
            "reason": "missing_manifest",
        }
    try:
        status_payload = _json_run(
            [
                str(python_bin),
                str(LONG_JOB_SCRIPT),
                "status",
                "--job-root",
                str(job_root),
            ],
            cwd=REPO_ROOT,
        )
    except Exception as exc:
        return {
            "job_root": str(job_root),
            "stopped": False,
            "reason": f"status_error:{type(exc).__name__}",
        }
    if not bool(status_payload.get("running", False)):
        return {
            "job_root": str(job_root),
            "stopped": False,
            "reason": "not_running",
            "status": status_payload,
        }
    stop_payload = _json_run(
        [
            str(python_bin),
            str(LONG_JOB_SCRIPT),
            "stop",
            "--job-root",
            str(job_root),
        ],
        cwd=REPO_ROOT,
    )
    return {
        "job_root": str(job_root),
        "stopped": True,
        "status": status_payload,
        "stop": stop_payload,
    }


def _surface_root_shares(surface: str) -> tuple[int, ...]:
    normalized = str(surface).strip().lower()
    if normalized == SURFACE_FULL_GRID:
        return tuple(int(v) for v in FULL_GRID_ROOT_SHARES)
    if normalized == SURFACE_PUBLICATION_COMPACT:
        return tuple(int(v) for v in PUBLICATION_COMPACT_ROOT_SHARES)
    if normalized == SURFACE_REPAIR_LEAF128_COUNTONLY:
        return tuple(int(v) for v in REPAIR_LEAF128_COUNTONLY_ROOT_SHARES)
    if normalized == SURFACE_ALLOCATION_POLICY_GRID:
        return tuple(int(v) for v in ALLOCATION_REPLACEMENT_ROOT_SHARES)
    raise ValueError(f"Unsupported surface: {surface!r}")


def _root_only_package(root_share: int) -> str:
    return f"full{int(root_share)}"


def _local_law_package(root_share: int) -> str:
    return f"full{int(root_share)}_leaf_full100_internal_count100"


def _mass_matched_rate_suffix(rate_percent: float) -> str:
    return f"{float(rate_percent):.1f}".replace(".", "p")


def _leaf_mass_package(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return f"r{int(root_share)}_leaf_mass_eq_{_mass_matched_rate_suffix(local_mass_percent)}"


def _depth_equal_mass_package(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return (
        f"r{int(root_share)}_depth_equal_mass_eq_"
        f"{_mass_matched_rate_suffix(local_mass_percent)}"
    )


def _balanced_node_mass_package(root_share: int) -> str:
    local_mass_percent = max(0.0, 100.0 - float(root_share))
    return f"r100_node_mass_eq_{_mass_matched_rate_suffix(local_mass_percent)}"


def _task_name(
    *,
    scope_key: str,
    package_name: str,
    leaf_tokens: int,
    family: str,
    train_docs: int = TRAIN_DOC_COUNT,
    data_seed: int = DEFAULT_SEED,
) -> str:
    return (
        f"{str(scope_key)}__train{int(train_docs):05d}__{str(package_name)}__"
        f"leaf{int(leaf_tokens):03d}__{str(family)}__d{int(data_seed)}"
    )


def _parse_task_name(task_name: str) -> dict[str, Any]:
    parts = str(task_name).split("__")
    if len(parts) < 6:
        return {}
    scope_key, train_part, package_name, leaf_part, family, seed_part = parts[:6]
    train_docs = 0
    leaf_tokens = 0
    data_seed = 0
    if train_part.startswith("train"):
        try:
            train_docs = int(train_part.replace("train", "", 1))
        except ValueError:
            train_docs = 0
    if leaf_part.startswith("leaf"):
        try:
            leaf_tokens = int(leaf_part.replace("leaf", "", 1))
        except ValueError:
            leaf_tokens = 0
    if seed_part.startswith("d"):
        try:
            data_seed = int(seed_part.replace("d", "", 1))
        except ValueError:
            data_seed = 0
    return {
        "scope_key": str(scope_key),
        "train_docs": int(train_docs),
        "package_name": str(package_name),
        "leaf_tokens": int(leaf_tokens),
        "family": str(family),
        "data_seed": int(data_seed),
    }


def _required_task_names_for_surface(surface: str) -> set[str]:
    shares = _surface_root_shares(surface)
    required: set[str] = set()
    if str(surface).strip().lower() == SURFACE_REPAIR_LEAF128_COUNTONLY:
        for root_share in shares:
            required.add(
                _task_name(
                    scope_key=RECOVERABLE_SCOPE_KEY,
                    package_name=_leaf_mass_package(int(root_share)),
                    leaf_tokens=128,
                    family=TREE_FAMILY,
                )
            )
        return required
    if str(surface).strip().lower() == SURFACE_ALLOCATION_POLICY_GRID:
        for scope_key in (RECOVERABLE_SCOPE_KEY, STRUCTURAL_SCOPE_KEY):
            for root_share in ALLOCATION_REPLACEMENT_ROOT_SHARES:
                root_package = _root_only_package(int(root_share))
                required.add(
                    _task_name(
                        scope_key=scope_key,
                        package_name=root_package,
                        leaf_tokens=128,
                        family=TREE_FAMILY,
                    )
                )
                required.add(
                    _task_name(
                        scope_key=scope_key,
                        package_name=root_package,
                        leaf_tokens=128,
                        family=FNO_TASK_FAMILY,
                    )
                )
                for leaf_tokens in tuple(
                    int(value) for value in ALLOCATION_ROOT_ONLY_LEAF_TOKENS if int(value) != 128
                ):
                    required.add(
                        _task_name(
                            scope_key=scope_key,
                            package_name=root_package,
                            leaf_tokens=int(leaf_tokens),
                            family=TREE_FAMILY,
                        )
                    )
            for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES:
                leaf_mass_package = _leaf_mass_package(int(root_share))
                for leaf_tokens in ALLOCATION_LOCAL_ONLY_LEAF_TOKENS:
                    required.add(
                        _task_name(
                            scope_key=scope_key,
                            package_name=leaf_mass_package,
                            leaf_tokens=int(leaf_tokens),
                            family=TREE_FAMILY,
                        )
                    )
                depth_equal_package = _depth_equal_mass_package(int(root_share))
                node_mass_package = _balanced_node_mass_package(int(root_share))
                for leaf_tokens in ALLOCATION_INTERNAL_POLICY_LEAF_TOKENS:
                    required.add(
                        _task_name(
                            scope_key=scope_key,
                            package_name=depth_equal_package,
                            leaf_tokens=int(leaf_tokens),
                            family=TREE_FAMILY,
                        )
                    )
                    required.add(
                        _task_name(
                            scope_key=scope_key,
                            package_name=node_mass_package,
                            leaf_tokens=int(leaf_tokens),
                            family=TREE_FAMILY,
                        )
                    )
        return required
    for scope_key in (RECOVERABLE_SCOPE_KEY, STRUCTURAL_SCOPE_KEY):
        for root_share in shares:
            root_package = _root_only_package(int(root_share))
            for leaf_tokens in LEAF_TOKEN_LADDER:
                required.add(
                    _task_name(
                        scope_key=scope_key,
                        package_name=root_package,
                        leaf_tokens=int(leaf_tokens),
                        family=TREE_FAMILY,
                    )
                )
            required.add(
                _task_name(
                    scope_key=scope_key,
                    package_name=root_package,
                    leaf_tokens=128,
                    family=TREE_FAMILY,
                )
            )
            required.add(
                _task_name(
                    scope_key=scope_key,
                    package_name=root_package,
                    leaf_tokens=128,
                    family=FNO_TASK_FAMILY,
                )
            )
            required.add(
                _task_name(
                    scope_key=scope_key,
                    package_name=_local_law_package(int(root_share)),
                    leaf_tokens=128,
                    family=TREE_FAMILY,
                )
            )
            if int(root_share) == 100:
                continue
            mass_package = _leaf_mass_package(int(root_share))
            for leaf_tokens in LEAF_TOKEN_LADDER:
                required.add(
                    _task_name(
                        scope_key=scope_key,
                        package_name=mass_package,
                        leaf_tokens=int(leaf_tokens),
                        family=TREE_FAMILY,
                    )
                )
            required.add(
                _task_name(
                    scope_key=scope_key,
                    package_name=mass_package,
                    leaf_tokens=128,
                    family=TREE_FAMILY,
                )
            )
    return required


def _known_visible_job_keys(output_root: Path) -> list[str]:
    visible: list[str] = []
    for key in KNOWN_VISIBLE_JOB_KEYS:
        status_path = output_root / key / "experiment_status.json"
        manifest_path = output_root / "_launchers" / key / "manifest.json"
        if key in PRESERVED_COMPLETED_JOB_KEYS or status_path.exists() or manifest_path.exists():
            visible.append(str(key))
    return visible


def _completed_task_names(output_root: Path) -> set[str]:
    names: set[str] = set()
    for key in _known_visible_job_keys(output_root):
        job_output_root = output_root / key
        if not job_output_root.exists():
            continue
        for path in job_output_root.rglob("raw/*/summary.json"):
            if path.parent.name == "summary_artifacts":
                continue
            names.add(str(path.parent.name))
    return names


def _active_task_names(output_root: Path) -> set[str]:
    names: set[str] = set()
    for key in _known_visible_job_keys(output_root):
        status_path = output_root / key / "experiment_status.json"
        payload = _maybe_load_json(status_path)
        for item in list(payload.get("active_item_details") or []):
            task_name = str(dict(item or {}).get("task_name", "") or "").strip()
            if task_name:
                names.add(task_name)
    return names


def _missing_task_names_for_surface(output_root: Path, surface: str) -> set[str]:
    required = _required_task_names_for_surface(surface)
    present = _completed_task_names(output_root) | _active_task_names(output_root)
    return {name for name in required if name not in present}


def _package_order_for_surface(surface: str) -> list[str]:
    shares = _surface_root_shares(surface)
    order: list[str] = []
    if str(surface).strip().lower() == SURFACE_REPAIR_LEAF128_COUNTONLY:
        return [_leaf_mass_package(int(root_share)) for root_share in shares]
    if str(surface).strip().lower() == SURFACE_ALLOCATION_POLICY_GRID:
        for root_share in ALLOCATION_REPLACEMENT_ROOT_SHARES:
            order.append(_root_only_package(int(root_share)))
        for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES:
            order.append(_leaf_mass_package(int(root_share)))
        for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES:
            order.append(_depth_equal_mass_package(int(root_share)))
        for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES:
            order.append(_balanced_node_mass_package(int(root_share)))
        return order
    for root_share in shares:
        order.append(_root_only_package(int(root_share)))
    for root_share in shares:
        order.append(_local_law_package(int(root_share)))
    for root_share in shares:
        if int(root_share) == 100:
            continue
        order.append(_leaf_mass_package(int(root_share)))
    return order


def _package_leaf_overrides_for_missing_tasks(
    missing_task_names: Iterable[str],
    *,
    surface: str,
) -> tuple[list[str], dict[str, list[int]]]:
    by_package: dict[str, set[int]] = {}
    for task_name in missing_task_names:
        parsed = _parse_task_name(str(task_name))
        package_name = str(parsed.get("package_name", "") or "")
        leaf_tokens = int(parsed.get("leaf_tokens", 0) or 0)
        if not package_name or leaf_tokens <= 0:
            continue
        by_package.setdefault(package_name, set()).add(int(leaf_tokens))
    explicit_order = _package_order_for_surface(surface)
    package_names = [name for name in explicit_order if name in by_package]
    overrides = {
        str(package_name): sorted(
            {int(value) for value in by_package.get(package_name, set())},
            reverse=True,
        )
        for package_name in package_names
    }
    return package_names, overrides


def _toml_list(values: Sequence[Any]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _toml_inline_leaf_overrides(mapping: Mapping[str, Sequence[int]]) -> str:
    items = []
    for key, values in mapping.items():
        items.append(f"{str(key)} = {_toml_list([int(v) for v in values])}")
    return "{ " + ", ".join(items) + " }"


def _build_pipeline_config(
    *,
    package_names: Sequence[str],
    package_leaf_token_overrides: Mapping[str, Sequence[int]],
    surface: str,
    output_root: Path,
) -> str:
    normalized_surface = str(surface).strip().lower()
    scope_keys: list[str] = []
    leaf_token_ladder: list[int] = list(int(v) for v in LEAF_TOKEN_LADDER)
    stage1_artifact_root = (
        "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_sticky_simple_fixed10240_full_grid"
    )
    if normalized_surface == SURFACE_REPAIR_LEAF128_COUNTONLY:
        scope_keys = [str(RECOVERABLE_SCOPE_KEY)]
        leaf_token_ladder = [128]
        stage1_artifact_root = str(
            output_root / "_stage1_artifacts" / REPAIR_RUN_KEY
        )
    elif normalized_surface == SURFACE_ALLOCATION_POLICY_GRID:
        scope_keys = [str(RECOVERABLE_SCOPE_KEY), str(STRUCTURAL_SCOPE_KEY)]
        leaf_token_ladder = [int(v) for v in ALLOCATION_ROOT_ONLY_LEAF_TOKENS]
        stage1_artifact_root = str(
            output_root / "_stage1_artifacts" / ALLOCATION_POLICY_RUN_KEY
        )
    return f"""[tradeoff_pipeline]
preset = "standard"
phases = ["supervision_recovery"]
device_mode = "cuda"
train_docs = {TRAIN_DOC_COUNT}
val_docs = 1024
test_docs = 1024
supervision_recovery_train_docs = [{TRAIN_DOC_COUNT}]
supervision_recovery_seeds = [{DEFAULT_SEED}]
supervision_recovery_packages = {_toml_list([str(name) for name in package_names])}
supervision_recovery_tree_family = "{TREE_FAMILY}"
supervision_recovery_structural_cell = "{STRUCTURAL_SCOPE_KEY}"
supervision_recovery_scope_keys = {json.dumps(" ".join(str(value) for value in scope_keys))}
supervision_fixed_leaf_tokens = 8
supervision_recovery_leaf_token_ladder = {_toml_list([int(v) for v in leaf_token_ladder])}
supervision_recovery_package_leaf_token_overrides = {_toml_inline_leaf_overrides(package_leaf_token_overrides)}
supervision_min_tokens = 128
supervision_max_tokens = 128
supervision_epochs = 40
exact_metric_final_doc_limit = 128
tree_posttrain_train_doc_limit = 128
tree_stage1_artifact_root = {json.dumps(str(stage1_artifact_root))}
supervision_recovery_recoverable_benchmark = "{RECOVERABLE_SCOPE_KEY}"
supervision_recovery_structural_grid = "{STRUCTURAL_GRID_KEY}"
tree_stage1_screen_doc_limit = 0
tree_stage1_final_exact_doc_limit = 0
tree_stage1_resume_if_available = false
supervision_recovery_depth_discount_gammas = [1.0]

[tradeoff_pipeline.tree_reference]
mode = "preset"
preset = "unified_g_full_local_laws_v1"

[tradeoff_pipeline.structural_tree_reference]
mode = "preset"
preset = "unified_g_full_local_laws_v1"

[tradeoff_pipeline.one_leaf_tree_reference]
mode = "preset"
preset = "unified_g_fno_parity_canary_v1"

[tradeoff_pipeline.runtime]
data_mode = "resident"
bucket_mode = "leaf_count_auto_queue"
tree_batch_structural_pad_limit = 0.5
tree_batch_auto_queue_min_docs = 8
tree_batch_auto_queue_min_fill_ratio = 0.5
preload_splits = ["train", "val", "test"]
preload_targets = true
workers_per_mig = 1
allow_multi_worker_screen = true
capacity_workers_per_mig = 2

[tradeoff_pipeline.scheduler]
mode = "global_per_run"
default_job_granularity = "family_train_seed"
cleanup_stale_children = true
max_gpu_items_per_mig = 1
"""


def _launch_pipeline_job(
    *,
    python_bin: Path,
    job_root: Path,
    name: str,
    description: str,
    config_path: Path,
    output_root: Path,
    migs: list[str] | None = None,
) -> dict:
    existing = _custom_job_existing_state(job_root=job_root, output_root=output_root)
    if existing["running"]:
        return {
            "key": name,
            "skipped": True,
            "reason": "running",
            "existing_state": existing,
            "job_root": str(job_root),
            "output_root": str(output_root),
        }
    if existing["state"] == "completed":
        return {
            "key": name,
            "skipped": True,
            "reason": "completed",
            "existing_state": existing,
            "job_root": str(job_root),
            "output_root": str(output_root),
        }
    cmd = [
        str(python_bin),
        str(LONG_JOB_SCRIPT),
        "launch",
        "--name",
        name,
        "--description",
        description,
        "--job-root",
        str(job_root),
        "--cwd",
        str(REPO_ROOT),
        "--python-bin",
        str(python_bin),
        "--launch-backend",
        "auto",
        "--no-replace-existing",
        "--",
        str(python_bin),
        str(TRADEOFF_PIPELINE_SCRIPT),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ]
    resolved_migs = [str(item) for item in (migs or []) if str(item).strip()]
    if resolved_migs:
        cmd.extend(["--migs", ",".join(resolved_migs)])
    return _json_run(cmd, cwd=REPO_ROOT)


def _write_combined_progress(*, output_root: Path) -> Path:
    jobs = []
    jobs_completed = 0
    jobs_running = 0
    jobs_failed = 0
    completed_items = 0
    active_items = 0
    pending_items = 0
    failed_items = 0
    items_total = 0
    landed_results = []
    flattened_active_item_details = []

    visible_job_keys = _known_visible_job_keys(output_root)
    for key in visible_job_keys:
        status_path = output_root / key / "experiment_status.json"
        manifest_path = output_root / "_launchers" / key / "manifest.json"
        payload = _maybe_load_json(status_path)
        state = str(payload.get("state", "not_materialized") or "not_materialized")
        artifact_paths = _artifact_snapshot(output_root / key)
        record = {
            "key": str(key),
            "status_path": str(status_path) if status_path.exists() else "",
            "manifest_path": str(manifest_path) if manifest_path.exists() else "",
            "state": state,
            "completed_items": int(payload.get("completed_items", 0) or 0),
            "active_items": int(payload.get("active_items", 0) or 0),
            "pending_items": int(payload.get("pending_items", 0) or 0),
            "failed_items": int(payload.get("failed_items", 0) or 0),
            "items_total": int(payload.get("items_total", 0) or 0),
            "percent_complete": float(payload.get("percent_complete", 0.0) or 0.0),
            "generated_at": str(payload.get("generated_at", "") or ""),
            "by_scope": dict(payload.get("by_scope") or {}),
            "active_item_details": list(payload.get("active_item_details") or []),
            "artifacts": artifact_paths,
            "launcher_owned": key.startswith("combined_scheduler"),
            "preserved_completed_job": key in PRESERVED_COMPLETED_JOB_KEYS,
        }
        jobs.append(record)
        if artifact_paths:
            landed_results.append(
                {
                    "key": str(key),
                    "state": state,
                    "artifacts": artifact_paths,
                }
            )
        items_total += record["items_total"]
        completed_items += record["completed_items"]
        active_items += record["active_items"]
        pending_items += record["pending_items"]
        failed_items += record["failed_items"]
        for item in record["active_item_details"]:
            item_payload = dict(item or {})
            item_payload["job_key"] = str(key)
            flattened_active_item_details.append(item_payload)
        if state == "completed":
            jobs_completed += 1
        elif state == "running":
            jobs_running += 1
        elif state == "failed":
            jobs_failed += 1

    percent_complete = (
        (100.0 * float(completed_items) / float(items_total))
        if items_total > 0
        else 0.0
    )
    filled = int(round((percent_complete / 100.0) * 20.0))
    filled = max(0, min(20, filled))
    progress_bar = ("#" * filled) + ("-" * (20 - filled))

    legacy_superseded_jobs = []
    for key in LEGACY_SUPERSEDED_JOB_KEYS:
        legacy_status_path = output_root / key / "experiment_status.json"
        legacy_manifest_path = output_root / "_launchers" / key / "manifest.json"
        legacy_payload = _maybe_load_json(legacy_status_path)
        legacy_superseded_jobs.append(
            {
                "key": str(key),
                "status_path": str(legacy_status_path)
                if legacy_status_path.exists()
                else "",
                "manifest_path": str(legacy_manifest_path)
                if legacy_manifest_path.exists()
                else "",
                "state": str(
                    legacy_payload.get("state", "not_materialized") or "not_materialized"
                ),
                "completed_items": int(legacy_payload.get("completed_items", 0) or 0),
                "active_items": int(legacy_payload.get("active_items", 0) or 0),
                "pending_items": int(legacy_payload.get("pending_items", 0) or 0),
                "failed_items": int(legacy_payload.get("failed_items", 0) or 0),
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(output_root),
        "active_batch_keys": list(visible_job_keys),
        "active_batch_counts": {
            "jobs_total": len(visible_job_keys),
            "jobs_completed": jobs_completed,
            "jobs_running": jobs_running,
            "jobs_failed": jobs_failed,
            "items_total": items_total,
            "completed_items": completed_items,
            "active_items": active_items,
            "active_items_semantics": (
                "Sum of scheduler-active items across the visible sticky-simple launch "
                "jobs under this output root."
            ),
            "active_scheduler_items": active_items,
            "running_jobs_with_active_items": sum(
                1 for job in jobs if int(job.get("active_items", 0) or 0) > 0
            ),
            "pending_items": pending_items,
            "failed_items": failed_items,
            "percent_complete": percent_complete,
            "progress_bar": progress_bar,
        },
        "active_item_details_count": len(flattened_active_item_details),
        "active_item_details": flattened_active_item_details,
        "landed_results_count": len(landed_results),
        "landed_results": landed_results,
        "legacy_superseded_jobs": legacy_superseded_jobs,
        "jobs": jobs,
    }

    launch_summary = _maybe_load_json(output_root / "sticky_simple_quick_launch_summary.json")
    target_surface = str(launch_summary.get("target_surface", "") or "").strip()
    target_root_shares = list(launch_summary.get("target_root_shares") or [])
    if target_surface:
        required = _required_task_names_for_surface(target_surface)
        completed_task_names = _completed_task_names(output_root)
        active_task_names = _active_task_names(output_root)
        completed_required = required & completed_task_names
        active_required = (required - completed_required) & active_task_names
        remaining_required = required - completed_required - active_required
        summary.update(
            {
                "target_surface": str(target_surface),
                "target_root_shares": [int(v) for v in target_root_shares],
                "required_rows_total": len(required),
                "completed_required_rows": len(completed_required),
                "active_required_rows": len(active_required),
                "remaining_required_rows": len(remaining_required),
                "missing_rows_at_launch": list(launch_summary.get("missing_rows_at_launch") or []),
            }
        )

    path = output_root / "combined_progress.json"
    _write_text(path, json.dumps(_jsonable(summary), indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch or refresh the sticky simple fixed-10240 surface, with "
            "full-grid supplemental-fill support on an existing root."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / f"markov_v5_simple_fixed10240_quick_{_utc_stamp()}",
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument(
        "--surface",
        choices=(
            SURFACE_FULL_GRID,
            SURFACE_PUBLICATION_COMPACT,
            SURFACE_REPAIR_LEAF128_COUNTONLY,
            SURFACE_ALLOCATION_POLICY_GRID,
        ),
        default=SURFACE_FULL_GRID,
        help="Target diagnostic surface for required-row accounting and supplemental fill.",
    )
    parser.add_argument(
        "--run-style",
        choices=(RUN_STYLE_SUPPLEMENTAL, RUN_STYLE_CLEAN),
        default=RUN_STYLE_SUPPLEMENTAL,
        help="Supplement an existing root in place, or launch a clean rerun surface.",
    )
    parser.add_argument(
        "--refresh-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only refresh combined_progress.json for an existing root; do not launch jobs.",
    )
    parser.add_argument(
        "--watch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In refresh-only mode, keep refreshing on an interval until the batch finishes.",
    )
    parser.add_argument(
        "--refresh-interval-seconds",
        type=float,
        default=30.0,
        help="Refresh cadence for --watch.",
    )
    parser.add_argument(
        "--launch-refresher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When launching the batch, also launch a detached combined-progress refresher.",
    )
    parser.add_argument(
        "--launch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the computed supplemental or clean-run scheduler job.",
    )
    parser.add_argument(
        "--migs",
        type=str,
        default="",
        help="Optional explicit comma-separated MIG UUIDs to hand to the launched scheduler job.",
    )
    parser.add_argument(
        "--available-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only MIGs that pass the conservative availability filter.",
    )
    args = parser.parse_args()

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (REPO_ROOT / args.output_root)
    )
    python_bin = (
        args.python_bin
        if args.python_bin.is_absolute()
        else (REPO_ROOT / args.python_bin)
    )

    if args.refresh_only:
        while True:
            combined_progress_path = _write_combined_progress(output_root=output_root)
            if not args.watch:
                print(
                    json.dumps(
                        {"combined_progress_path": str(combined_progress_path)},
                        indent=2,
                    )
                )
                return
            payload = _maybe_load_json(combined_progress_path)
            counts = dict(payload.get("active_batch_counts") or {})
            if int(counts.get("jobs_running", 0) or 0) <= 0:
                print(
                    json.dumps(
                        {
                            "combined_progress_path": str(combined_progress_path),
                            "state": "finished",
                        },
                        indent=2,
                    )
                )
                return
            time.sleep(max(1.0, float(args.refresh_interval_seconds)))

    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    target_root_shares = list(_surface_root_shares(str(args.surface)))
    required_task_names = _required_task_names_for_surface(str(args.surface))
    completed_task_names = _completed_task_names(output_root)
    active_task_names = _active_task_names(output_root)
    present_task_names = completed_task_names | active_task_names

    if str(args.run_style) == RUN_STYLE_SUPPLEMENTAL:
        missing_task_names = sorted(required_task_names - present_task_names)
        if str(args.surface) == SURFACE_FULL_GRID:
            combined_job_key = SUPPLEMENTAL_FULL_GRID_RUN_KEY
        elif str(args.surface) == SURFACE_PUBLICATION_COMPACT:
            combined_job_key = SUPPLEMENTAL_COMPACT_RUN_KEY
        elif str(args.surface) == SURFACE_ALLOCATION_POLICY_GRID:
            combined_job_key = ALLOCATION_POLICY_RUN_KEY
        else:
            combined_job_key = REPAIR_RUN_KEY
    else:
        missing_task_names = sorted(required_task_names)
        combined_job_key = (
            REPAIR_RUN_KEY
            if str(args.surface) == SURFACE_REPAIR_LEAF128_COUNTONLY
            else COMBINED_RUN_KEY
        )

    package_names, package_leaf_token_overrides = _package_leaf_overrides_for_missing_tasks(
        missing_task_names,
        surface=str(args.surface),
    )

    config_path = generated_dir / f"{combined_job_key}.toml"
    launch_result: dict[str, Any] | list[dict[str, Any]] = []
    mig_selection_mode = "explicit" if str(args.migs or "").strip() else "detected"
    mig_pool: list[str] = []
    stopped_legacy_jobs: list[dict[str, Any]] = []

    if package_names:
        _write_text(
            config_path,
            _build_pipeline_config(
                package_names=package_names,
                package_leaf_token_overrides=package_leaf_token_overrides,
                surface=str(args.surface),
                output_root=output_root,
            ),
        )

    existing_visible_jobs = _known_visible_job_keys(output_root)
    any_existing_running = any(
        _custom_job_existing_state(
            job_root=output_root / "_launchers" / key,
            output_root=output_root / key,
        ).get("running", False)
        for key in existing_visible_jobs
    )
    auto_available_only = (
        str(args.run_style) == RUN_STYLE_SUPPLEMENTAL
        and any_existing_running
        and not str(args.migs or "").strip()
    )
    effective_available_only = bool(args.available_only or auto_available_only)
    if auto_available_only and not args.available_only:
        mig_selection_mode = "available_only_auto"
    elif args.available_only:
        mig_selection_mode = "available_only_requested"

    explicit_migs = _parse_explicit_migs(args.migs)
    mig_pool = _detect_launch_migs(
        explicit_migs if explicit_migs else None,
        available_only=effective_available_only,
    )

    if args.launch and str(args.run_style) == RUN_STYLE_CLEAN:
        for key in LEGACY_SUPERSEDED_JOB_KEYS:
            stopped_legacy_jobs.append(
                {
                    "key": str(key),
                    **_stop_long_job_if_running(
                        python_bin=python_bin,
                        job_root=output_root / "_launchers" / key,
                    ),
                }
            )

    combined_job = {
        "key": combined_job_key,
        "description": (
            f"Sticky simple fixed-10240 {str(args.surface)} {str(args.run_style)} "
            "scheduler run"
        ),
        "config_path": config_path,
        "job_root": output_root / "_launchers" / combined_job_key,
        "output_root": output_root / combined_job_key,
        "name": f"markov_v5_simple_10240_{combined_job_key}",
    }

    if args.launch and package_names:
        launch_result = _launch_pipeline_job(
            python_bin=python_bin,
            job_root=combined_job["job_root"],
            name=str(combined_job["name"]),
            description=str(combined_job["description"]),
            config_path=Path(combined_job["config_path"]),
            output_root=Path(combined_job["output_root"]),
            migs=list(mig_pool),
        )
    elif not package_names:
        launch_result = {
            "key": combined_job_key,
            "skipped": True,
            "reason": "no_missing_rows",
            "job_root": str(combined_job["job_root"]),
            "output_root": str(combined_job["output_root"]),
        }
    else:
        launch_result = [dict(combined_job)]

    summary = {
        "output_root": str(output_root),
        "python_bin": str(python_bin),
        "launch": bool(args.launch),
        "target_surface": str(args.surface),
        "target_root_shares": [int(v) for v in target_root_shares],
        "run_style": str(args.run_style),
        "mig_selection_mode": str(mig_selection_mode),
        "available_only_effective": bool(effective_available_only),
        "visible_job_keys": list(_known_visible_job_keys(output_root))
        + ([combined_job_key] if combined_job_key not in _known_visible_job_keys(output_root) else []),
        "preserved_completed_job_keys": list(PRESERVED_COMPLETED_JOB_KEYS),
        "legacy_superseded_job_keys": list(LEGACY_SUPERSEDED_JOB_KEYS),
        "mig_pool": list(mig_pool),
        "generated_configs": [str(config_path)] if package_names else [],
        "required_rows_total": len(required_task_names),
        "missing_rows_at_launch": list(missing_task_names),
        "scheduled_missing_row_count": len(missing_task_names),
        "completed_required_rows_at_launch": len(required_task_names & completed_task_names),
        "active_required_rows_at_launch": len(
            (required_task_names - completed_task_names) & active_task_names
        ),
        "remaining_required_rows_at_launch": len(
            required_task_names - completed_task_names - active_task_names
        ),
        "preserved_completed_jobs": [
            {
                "key": str(key),
                "job_root": str(output_root / "_launchers" / key),
                "output_root": str(output_root / key),
                "existing_state": _custom_job_existing_state(
                    job_root=output_root / "_launchers" / key,
                    output_root=output_root / key,
                ),
            }
            for key in PRESERVED_COMPLETED_JOB_KEYS
        ],
        "stopped_legacy_jobs": stopped_legacy_jobs,
        "combined_job": launch_result,
    }
    summary_path = output_root / "sticky_simple_quick_launch_summary.json"
    _write_text(summary_path, json.dumps(_jsonable(summary), indent=2))
    combined_progress_path = _write_combined_progress(output_root=output_root)

    refresher_result = {}
    if args.launch and args.launch_refresher:
        refresher_job_root = output_root / "_launchers" / "combined_progress_refresher"
        refresher_output_root = output_root / "_status_refresh"
        existing = _custom_job_existing_state(
            job_root=refresher_job_root,
            output_root=refresher_output_root,
        )
        if existing["running"]:
            refresher_result = {
                "skipped": True,
                "reason": "running",
                "job_root": str(refresher_job_root),
            }
        else:
            refresher_cmd = [
                str(python_bin),
                str(LONG_JOB_SCRIPT),
                "launch",
                "--name",
                "markov_v5_simple_fixed10240_combined_progress_refresher",
                "--description",
                "Refresh combined_progress.json for sticky simple fixed-10240 batch",
                "--job-root",
                str(refresher_job_root),
                "--cwd",
                str(REPO_ROOT),
                "--python-bin",
                str(python_bin),
                "--launch-backend",
                "auto",
                "--no-replace-existing",
                "--",
                str(python_bin),
                str(Path(__file__).resolve()),
                "--output-root",
                str(output_root),
                "--refresh-only",
                "--watch",
                "--refresh-interval-seconds",
                str(float(args.refresh_interval_seconds)),
            ]
            refresher_result = _json_run(refresher_cmd, cwd=REPO_ROOT)

    summary["combined_progress_path"] = str(combined_progress_path)
    summary["refresher"] = refresher_result
    _write_text(summary_path, json.dumps(_jsonable(summary), indent=2))
    combined_progress_path = _write_combined_progress(output_root=output_root)

    report_lines = [
        "# Sticky Simple Quick Launch",
        "",
        f"- Output root: `{output_root}`",
        f"- Python: `{python_bin}`",
        f"- Surface: `{args.surface}`",
        f"- Run style: `{args.run_style}`",
        f"- Launch mode: `{'launch' if args.launch else 'plan_only'}`",
        f"- MIG pool size: `{len(mig_pool)}`",
        f"- MIG selection: `{mig_selection_mode}`",
        f"- Required rows: `{len(required_task_names)}`",
        f"- Missing rows at launch: `{len(missing_task_names)}`",
        "",
        "## Preserved Completed Jobs",
    ]
    for key in PRESERVED_COMPLETED_JOB_KEYS:
        report_lines.append(f"- `{key}`: preserved when present")
    report_lines.extend(
        [
            "",
            "## Supplemental / Combined Run",
            f"- Key: `{combined_job_key}`",
            f"- Config: `{config_path}`" if package_names else "- Config: not needed (`no_missing_rows`)",
            f"- Output root: `{combined_job['output_root']}`",
            f"- Launcher root: `{combined_job['job_root']}`",
            "",
            f"- Summary: `{summary_path}`",
            f"- Combined progress: `{combined_progress_path}`",
        ]
    )
    _write_text(
        output_root / "sticky_simple_quick_launch_report.md",
        "\n".join(report_lines) + "\n",
    )

    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
