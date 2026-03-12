#!/usr/bin/env python3
"""Run NVFP4-only manual hybrid bootstrap pipeline (teacher-first, single host)."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.gpu_orchestrator import GPUOrchestrator, OrchestratorConfig, OrchestratorMode
    GPU_ORCHESTRATOR_AVAILABLE = True
except Exception:  # pragma: no cover
    GPUOrchestrator = None  # type: ignore[assignment]
    OrchestratorConfig = None  # type: ignore[assignment]
    OrchestratorMode = None  # type: ignore[assignment]
    GPU_ORCHESTRATOR_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
DEFAULT_STUDENT_MODEL = "/mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


class BootstrapFailure(RuntimeError):
    """Raised when bootstrap orchestration cannot continue."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _run_command(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    dry_run: bool,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(str(part) for part in cmd)

    if dry_run:
        log_path.write_text(f"# dry-run\n{rendered}\n", encoding="utf-8")
        now = _now_iso()
        return {
            "command": [str(part) for part in cmd],
            "log_path": str(log_path),
            "returncode": 0,
            "started_at": now,
            "finished_at": now,
            "dry_run": True,
        }

    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    started = _now_iso()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# cmd={rendered}\n\n")
        handle.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    finished = _now_iso()
    return {
        "command": [str(part) for part in cmd],
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "started_at": started,
        "finished_at": finished,
        "dry_run": False,
    }


def _run_shell(
    command: str,
    *,
    log_path: Path,
    cwd: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    return _run_command(
        ["bash", "-lc", str(command)],
        log_path=log_path,
        cwd=cwd,
        dry_run=dry_run,
    )


def _port_from_base_url(base_url: str, default: int) -> int:
    parsed = urlparse(str(base_url))
    return int(parsed.port or default)


def _is_server_alive(base_url: str, timeout_seconds: float = 3.0) -> bool:
    try:
        response = requests.get(f"{str(base_url).rstrip('/')}/models", timeout=float(timeout_seconds))
        return bool(response.status_code == 200)
    except Exception:
        return False


def _wait_for_server(base_url: str, timeout_seconds: float) -> bool:
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if _is_server_alive(base_url):
            return True
        time.sleep(3.0)
    return False


def _get_model_ids(base_url: str, timeout_seconds: float = 10.0) -> List[str]:
    response = requests.get(f"{str(base_url).rstrip('/')}/models", timeout=float(timeout_seconds))
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for row in data:
        if isinstance(row, dict):
            model_id = row.get("id")
            if model_id is not None:
                ids.append(str(model_id))
    return ids


def _start_vllm_background(
    *,
    profile: str,
    port: int,
    cuda_devices: str,
    log_path: Path,
    cwd: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    args = [f"./scripts/start_vllm.sh {shlex.quote(str(profile))}", f"--port {int(port)}"]
    if str(cuda_devices).strip():
        args.append(f"--cuda-devices {shlex.quote(str(cuda_devices))}")
    cmd = " ".join(args) + f" > {shlex.quote(str(log_path))} 2>&1 &"
    return _run_shell(
        cmd,
        log_path=log_path.with_suffix(log_path.suffix + ".launcher"),
        cwd=cwd,
        dry_run=dry_run,
    )


def _start_embedding_server(
    *,
    port: int,
    cuda_devices: str,
    log_path: Path,
    cwd: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    args = ["./scripts/start_embedding_server.sh", f"--port {int(port)}"]
    if str(cuda_devices).strip():
        args.extend(["--cuda-devices", str(cuda_devices)])
    cmd = [str(part) for part in args]
    return _run_command(
        cmd,
        log_path=log_path,
        cwd=cwd,
        dry_run=dry_run,
    )


def _stop_all_servers(*, cwd: Path, dry_run: bool, log_path: Path) -> Dict[str, Any]:
    return _run_command(
        ["./scripts/stop_small_servers.sh", "--all"],
        log_path=log_path,
        cwd=cwd,
        dry_run=dry_run,
    )


def _safe_rate(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_real_anchor_gate(
    *,
    baseline_metrics: Dict[str, Any],
    post_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = dict(baseline_metrics.get("overall") or {})
    post = dict(post_metrics.get("overall") or {})
    keys = ["c1_pass_rate", "c2_pass_rate", "c3_pass_rate"]

    deltas: Dict[str, Optional[float]] = {}
    for key in keys:
        before = _safe_rate(baseline, key)
        after = _safe_rate(post, key)
        deltas[key] = None if before is None or after is None else float(after - before)

    baseline_values = [_safe_rate(baseline, key) for key in keys]
    post_values = [_safe_rate(post, key) for key in keys]
    if any(value is None for value in baseline_values) or any(value is None for value in post_values):
        avg_delta = None
    else:
        avg_delta = float(sum(post_values) / 3.0 - sum(baseline_values) / 3.0)  # type: ignore[arg-type]

    avg_gain = bool(avg_delta is not None and avg_delta >= 5.0)
    no_regression = bool(all(delta is not None and delta >= -3.0 for delta in deltas.values()))
    return {
        "baseline": baseline,
        "post": post,
        "deltas": deltas,
        "average_delta": avg_delta,
        "average_gain_at_least_5": avg_gain,
        "no_single_law_regresses_by_more_than_3": no_regression,
        "pass": bool(avg_gain and no_regression),
    }


def _extract_best_tune_config(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    best = payload.get("best")
    if not isinstance(best, dict):
        return {}
    config = best.get("config")
    if not isinstance(config, dict):
        return {}
    return dict(config)


def _extract_lawstress_metrics(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if isinstance(payload.get("metrics"), dict):
        return dict(payload.get("metrics") or {})
    return payload


def _build_student_orchestrator(args: argparse.Namespace, repo_root: Path) -> "GPUOrchestrator":
    if not GPU_ORCHESTRATOR_AVAILABLE:
        raise BootstrapFailure("Dynamic mode requested but GPUOrchestrator is unavailable")
    assert OrchestratorConfig is not None
    config_path = repo_root / "config" / "settings.yaml"
    config = OrchestratorConfig.from_yaml(
        config_path,
        task_model_profile_override=str(args.student_profile),
    )
    student_port = _port_from_base_url(args.student_base_url, default=8000)
    embedding_port = _port_from_base_url(args.embedding_base_url, default=8003)

    config.task_primary.port = int(student_port)
    replica_port = int(config.task_replica.port or (student_port + 2))
    blocked_ports = {int(student_port), int(embedding_port)}
    while replica_port in blocked_ports:
        replica_port += 1
    config.task_replica.port = int(replica_port)

    config.task_primary.cuda_devices = str(args.student_cuda_devices)
    config.task_primary.backend = "vllm"
    config.task_replica.backend = "vllm"
    config.genrm.backend = "vllm"
    config.enable_genrm = False
    config.manage_embedding = True
    if config.embedding is None:
        raise BootstrapFailure(
            "Dynamic mode requires embedding management, but embedding config is missing in settings.yaml"
        )
    config.embedding.port = int(embedding_port)
    config.embedding.cuda_devices = str(args.embedding_cuda_devices)
    return GPUOrchestrator(config=config)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual NVFP4 hybrid bootstrap with teacher/student server topology switching."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manage-servers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-wait-seconds", type=float, default=420.0)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-profile", type=str, default="qwen3.5-397b-a17b-nvfp4")
    parser.add_argument("--teacher-cuda-devices", type=str, default="0,1,2,3")

    parser.add_argument("--student-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--student-profile", type=str, default="qwen3.5-35b-a3b")
    parser.add_argument("--student-cuda-devices", type=str, default="0,1")
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument(
        "--student-max-tokens",
        type=int,
        default=0,
        help="Max tokens for student DSPy generations (<=0 uses model/context-window default).",
    )

    parser.add_argument("--embedding-base-url", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cuda-devices", type=str, default="2")
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)

    parser.add_argument("--anchor-id", type=str, default="51320_198306")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-source-chars", type=int, default=1200)
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=0,
        help="Max chars for teacher-trace source text prompts (<=0 disables clipping).",
    )

    parser.add_argument("--lawstress-train-size", type=int, default=48)
    parser.add_argument("--lawstress-val-size", type=int, default=12)
    parser.add_argument("--lawstress-test-size", type=int, default=12)
    parser.add_argument("--lawstress-num-workers", type=int, default=16)
    parser.add_argument("--lawstress-max-attempts", type=int, default=4)
    parser.add_argument("--lawstress-hard-ratio", type=float, default=0.8)
    parser.add_argument("--lawstress-real-anchor-ratio", type=float, default=0.3)
    parser.add_argument("--lawstress-doc-score-tolerance-raw", type=float, default=10.0)
    parser.add_argument("--lawstress-segment-score-tolerance-raw", type=float, default=12.0)
    parser.add_argument(
        "--lawstress-disable-teacher-gates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable teacher gating during LawStress generation (keeps teacher rewrite/reference summary enabled). "
            "Useful when teacher-gate acceptance is too low."
        ),
    )

    parser.add_argument("--teacher-trace-train-size", type=int, default=12)
    parser.add_argument("--teacher-trace-val-size", type=int, default=4)
    parser.add_argument("--teacher-trace-test-size", type=int, default=4)
    parser.add_argument("--teacher-trace-min-accepted", type=int, default=12)
    parser.add_argument("--teacher-trace-num-workers", type=int, default=16)

    parser.add_argument("--gepa-budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--gepa-num-threads", type=int, default=8)
    parser.add_argument(
        "--gepa-objective-aggregate",
        choices=["weighted_mean", "min", "bottleneck_min", "softmin", "floor_then_weighted"],
        default="min",
    )
    parser.add_argument("--gepa-objective-softmin-temperature", type=float, default=0.08)
    parser.add_argument("--gepa-objective-component-floor", type=float, default=0.55)
    parser.add_argument("--enable-prompt-batch-tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-batch-train-pairs", type=Path, default=None)
    parser.add_argument("--prompt-batch-eval-pairs", type=Path, default=None)
    parser.add_argument("--prompt-batch-budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--prompt-batch-num-threads", type=int, default=8)
    parser.add_argument(
        "--prompt-batch-include-score-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--disable-genrm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lawstress-eval-splits", nargs="*", default=["test"])
    parser.add_argument("--real-eval-splits", nargs="*", default=["test"])
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=2,
        help=(
            "Worker count for evaluation/summarize CLIs (LawStress + real-anchor). "
            "Lower values reduce peak memory on the 397B teacher."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not bool(args.disable_genrm):
        LOGGER.error(
            "GenRM is not supported in manual local-law bootstrap. "
            "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM."
        )
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (Path("outputs") / f"bootstrap_poc_manual_{_timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "stage_a", "stage_b", "stage_c", "stage_e", "stage_f"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "bootstrap_manifest.json"
    requested_lawstress = int(args.lawstress_train_size) + int(args.lawstress_val_size) + int(args.lawstress_test_size)
    requested_teacher_traces = (
        int(args.teacher_trace_train_size)
        + int(args.teacher_trace_val_size)
        + int(args.teacher_trace_test_size)
    )
    if int(args.lawstress_num_workers) < 1:
        raise ValueError(f"--lawstress-num-workers must be >= 1 (got {int(args.lawstress_num_workers)})")
    if int(args.teacher_trace_num_workers) < 1:
        raise ValueError(f"--teacher-trace-num-workers must be >= 1 (got {int(args.teacher_trace_num_workers)})")
    if int(args.eval_num_workers) < 1:
        raise ValueError(f"--eval-num-workers must be >= 1 (got {int(args.eval_num_workers)})")
    if requested_lawstress > 1 and int(args.lawstress_num_workers) < 2:
        raise ValueError(
            "Single-worker LawStress generation is disabled for multi-record runs. "
            f"Set --lawstress-num-workers >= 2 (got {int(args.lawstress_num_workers)}, records={requested_lawstress})."
        )
    if requested_teacher_traces > 1 and int(args.teacher_trace_num_workers) < 2:
        raise ValueError(
            "Single-worker teacher-trace generation is disabled for multi-doc runs. "
            f"Set --teacher-trace-num-workers >= 2 (got {int(args.teacher_trace_num_workers)}, docs={requested_teacher_traces})."
        )
    teacher_port = _port_from_base_url(args.teacher_base_url, default=8000)
    student_port = _port_from_base_url(args.student_base_url, default=8000)
    embedding_port = _port_from_base_url(args.embedding_base_url, default=8003)

    manifest: Dict[str, Any] = {
        "created_at": _now_iso(),
        "status": "running",
        "config": {
            "teacher_base_url": str(args.teacher_base_url),
            "teacher_model": str(args.teacher_model),
            "teacher_profile": str(args.teacher_profile),
            "teacher_cuda_devices": str(args.teacher_cuda_devices),
            "student_base_url": str(args.student_base_url),
            "student_model": str(args.student_model),
            "student_profile": str(args.student_profile),
            "student_cuda_devices": str(args.student_cuda_devices),
            "embedding_base_url": str(args.embedding_base_url),
            "embedding_model": str(args.embedding_model),
            "embedding_cuda_devices": str(args.embedding_cuda_devices),
            "anchor_id": str(args.anchor_id),
            "seed": int(args.seed),
            "min_source_chars": int(args.min_source_chars),
            "lawstress_sizes": {
                "train": int(args.lawstress_train_size),
                "val": int(args.lawstress_val_size),
                "test": int(args.lawstress_test_size),
            },
            "lawstress_generation": {
                "num_workers": int(args.lawstress_num_workers),
                "max_attempts": int(args.lawstress_max_attempts),
                "hard_ratio": float(args.lawstress_hard_ratio),
                "real_anchor_ratio": float(args.lawstress_real_anchor_ratio),
                "doc_score_tolerance_raw": float(args.lawstress_doc_score_tolerance_raw),
                "segment_score_tolerance_raw": float(args.lawstress_segment_score_tolerance_raw),
                "disable_teacher_gates": bool(args.lawstress_disable_teacher_gates),
            },
            "teacher_trace_sizes": {
                "train": int(args.teacher_trace_train_size),
                "val": int(args.teacher_trace_val_size),
                "test": int(args.teacher_trace_test_size),
            },
            "teacher_trace_min_accepted": int(args.teacher_trace_min_accepted),
            "teacher_trace_num_workers": int(args.teacher_trace_num_workers),
            "gepa_budget": str(args.gepa_budget),
            "gepa_num_threads": int(args.gepa_num_threads),
            "gepa_objective_aggregate": str(args.gepa_objective_aggregate),
            "gepa_objective_softmin_temperature": float(args.gepa_objective_softmin_temperature),
            "gepa_objective_component_floor": float(args.gepa_objective_component_floor),
            "enable_prompt_batch_tuning": bool(args.enable_prompt_batch_tuning),
            "prompt_batch_budget": str(args.prompt_batch_budget),
            "prompt_batch_num_threads": int(args.prompt_batch_num_threads),
            "prompt_batch_include_score_conditioning": bool(args.prompt_batch_include_score_conditioning),
            "prompt_batch_train_pairs": str(args.prompt_batch_train_pairs) if args.prompt_batch_train_pairs else None,
            "prompt_batch_eval_pairs": str(args.prompt_batch_eval_pairs) if args.prompt_batch_eval_pairs else None,
            "manage_servers": bool(args.manage_servers),
            "dynamic_mode": bool(args.dynamic_mode),
            "student_training_strategy": "proxy_gepa",
            "judge_backend": "large_qwen",
            "tournament_backend": "disabled",
            "disable_genrm": bool(args.disable_genrm),
            "lawstress_eval_splits": [str(value) for value in args.lawstress_eval_splits],
            "real_eval_splits": [str(value) for value in args.real_eval_splits],
            "eval_num_workers": int(args.eval_num_workers),
            "dry_run": bool(args.dry_run),
        },
        "phases": {},
        "gates": {},
        "artifacts": {},
    }
    _write_json(manifest_path, manifest)

    student_orchestrator = None
    try:
        # Stage 0: preflight teacher server.
        stage0: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_0_preflight"] = stage0
        _write_json(manifest_path, manifest)

        stage0["teacher_alive_before"] = _is_server_alive(args.teacher_base_url)
        teacher_model_ids_before: List[str] = []
        if not args.dry_run and bool(stage0["teacher_alive_before"]):
            try:
                teacher_model_ids_before = _get_model_ids(args.teacher_base_url)
            except Exception as exc:  # pragma: no cover - defensive preflight fallback
                stage0["teacher_model_ids_before_error"] = str(exc)
                teacher_model_ids_before = []
        stage0["teacher_model_ids_before"] = teacher_model_ids_before
        teacher_ok = bool(args.dry_run) or (
            bool(stage0["teacher_alive_before"]) and str(args.teacher_model) in set(stage0["teacher_model_ids_before"])
        )

        if not teacher_ok and bool(args.manage_servers):
            stage0["stop_all"] = _stop_all_servers(
                cwd=repo_root,
                dry_run=bool(args.dry_run),
                log_path=output_dir / "logs" / "stage_0_stop_all.log",
            )
            stage0["start_teacher"] = _start_vllm_background(
                profile=str(args.teacher_profile),
                port=int(teacher_port),
                cuda_devices=str(args.teacher_cuda_devices),
                log_path=output_dir / "logs" / "stage_0_teacher_server.log",
                cwd=repo_root,
                dry_run=bool(args.dry_run),
            )
            if not args.dry_run:
                if not _wait_for_server(args.teacher_base_url, timeout_seconds=float(args.server_wait_seconds)):
                    raise BootstrapFailure(f"Teacher server did not become ready at {args.teacher_base_url}")
                ids = _get_model_ids(args.teacher_base_url)
                stage0["teacher_model_ids_after_start"] = ids
                if str(args.teacher_model) not in set(ids):
                    raise BootstrapFailure(
                        "Teacher server is up but does not serve expected model: "
                        f"{args.teacher_model}. Served IDs: {ids}"
                    )
        elif not teacher_ok:
            raise BootstrapFailure(
                f"Teacher server {args.teacher_base_url} does not serve model {args.teacher_model}. "
                "Use --manage-servers or start the correct server manually."
            )

        stage0["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage A: LawStress generation + baseline eval (teacher summarizer/scorer).
        stage_a: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_a_lawstress"] = stage_a
        _write_json(manifest_path, manifest)

        lawstress_data_dir = output_dir / "stage_a" / "lawstress_data"
        baseline_eval_dir = output_dir / "stage_a" / "baseline_eval"
        records_path = lawstress_data_dir / "lawstress_records.jsonl"

        stage_a_generate_cmd = [
            sys.executable,
            "scripts/generate_manifesto_lawstress.py",
            "--output-dir",
            str(lawstress_data_dir),
            "--train-size",
            str(args.lawstress_train_size),
            "--val-size",
            str(args.lawstress_val_size),
            "--test-size",
            str(args.lawstress_test_size),
            "--teacher-base-url",
            str(args.teacher_base_url),
            "--teacher-model",
            str(args.teacher_model),
            "--teacher-score-temperature",
            "0.0",
            "--num-workers",
            str(args.lawstress_num_workers),
            "--max-attempts",
            str(args.lawstress_max_attempts),
            "--hard-ratio",
            str(args.lawstress_hard_ratio),
            "--real-anchor-ratio",
            str(args.lawstress_real_anchor_ratio),
            "--doc-score-tolerance-raw",
            str(args.lawstress_doc_score_tolerance_raw),
            "--segment-score-tolerance-raw",
            str(args.lawstress_segment_score_tolerance_raw),
        ]
        if bool(args.lawstress_disable_teacher_gates):
            stage_a_generate_cmd.append("--disable-teacher-gates")
        stage_a["generate"] = _run_command(
            stage_a_generate_cmd,
            log_path=output_dir / "logs" / "stage_a_generate_lawstress.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_a["generate"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage A generation failed")

        baseline_predictions = baseline_eval_dir / "predictions.jsonl"
        stage_a_summarize_cmd = [
            sys.executable,
            "scripts/eval_manifesto_lawstress.py",
            "--records",
            str(records_path),
            "--output-dir",
            str(baseline_eval_dir),
            "--mode",
            "summarize_only",
            "--predictions-path",
            str(baseline_predictions),
            "--summarizer-base-url",
            str(args.teacher_base_url),
            "--summarizer-model",
            str(args.teacher_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.lawstress_eval_splits:
            stage_a_summarize_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])
        stage_a["baseline_summarize"] = _run_command(
            stage_a_summarize_cmd,
            log_path=output_dir / "logs" / "stage_a_baseline_summarize.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_a["baseline_summarize"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage A baseline summarize failed")

        stage_a_score_cmd = [
            sys.executable,
            "scripts/eval_manifesto_lawstress.py",
            "--records",
            str(records_path),
            "--output-dir",
            str(baseline_eval_dir),
            "--mode",
            "score_and_judge_only",
            "--predictions-path",
            str(baseline_predictions),
            "--scorer-base-url",
            str(args.teacher_base_url),
            "--scorer-model",
            str(args.teacher_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        stage_a_score_cmd.append("--disable-genrm")
        if args.lawstress_eval_splits:
            stage_a_score_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])
        stage_a["baseline_score"] = _run_command(
            stage_a_score_cmd,
            log_path=output_dir / "logs" / "stage_a_baseline_score.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_a["baseline_score"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage A baseline score failed")

        baseline_lawstress_metrics_path = baseline_eval_dir / "eval_metrics.json"
        baseline_lawstress_metrics = {
            "overall": {
                "c1_pass_rate": 50.0,
                "c2_pass_rate": 50.0,
                "c3_pass_rate": 50.0,
                "same_side_of_neutral_pct": 60.0,
                "mae": 0.10,
            },
            "success": {"overall_pass": False},
        }
        if not args.dry_run:
            baseline_lawstress_metrics = _load_json(baseline_lawstress_metrics_path)

        stage_a["generated_records"] = requested_lawstress if args.dry_run else _count_jsonl_rows(records_path)
        stage_a["baseline_eval_metrics_path"] = str(baseline_lawstress_metrics_path)
        stage_a["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage B: tune + teacher traces with retry.
        stage_b: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_b_teacher_traces"] = stage_b
        _write_json(manifest_path, manifest)

        local_tune_dir = output_dir / "stage_b" / "local_tune"
        tune_cmd = [
            sys.executable,
            "scripts/run_single_manifesto_local_law_tune.py",
            "--manifesto-id",
            str(args.anchor_id),
            "--output-root",
            str(local_tune_dir),
            "--teacher-base-url",
            str(args.teacher_base_url),
            "--scorer-base-url",
            str(args.teacher_base_url),
            "--teacher-model",
            str(args.teacher_model),
            "--scorer-model",
            str(args.teacher_model),
            "--seed",
            str(args.seed),
            "--min-source-chars",
            str(args.min_source_chars),
            "--max-source-chars",
            str(args.max_source_chars),
        ]
        stage_b["local_tune"] = _run_command(
            tune_cmd,
            log_path=output_dir / "logs" / "stage_b_local_tune.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_b["local_tune"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage B local tune failed")

        tuned_cfg = {}
        if not args.dry_run:
            tuned_cfg = _extract_best_tune_config(local_tune_dir / "candidate_results.json")
        score_tol = float(tuned_cfg.get("score_tolerance_raw", 20.0) or 20.0)
        max_attempts = int(tuned_cfg.get("max_attempts", 3) or 3)
        use_dspy = bool(tuned_cfg.get("use_dspy_guidance", False))
        dspy_temp = float(tuned_cfg.get("dspy_guidance_temperature", 0.1) or 0.1)
        dspy_max = int(tuned_cfg.get("dspy_guidance_max_tokens", 1600) or 1600)
        summary_temp = float(tuned_cfg.get("summary_temperature", 0.1) or 0.1)

        pass_specs = [
            {"pass_index": 1, "seed": int(args.seed), "score_tol": score_tol, "max_attempts": max_attempts},
            {
                "pass_index": 2,
                "seed": int(args.seed) + 1,
                "score_tol": score_tol + 5.0,
                "max_attempts": max_attempts + 1,
            },
        ]

        selected_manifest_path: Optional[Path] = None
        selected_records_path: Optional[Path] = None
        selected_accepted = 0
        last_manifest_path: Optional[Path] = None

        for spec in pass_specs:
            pass_idx = int(spec["pass_index"])
            pass_dir = output_dir / "stage_b" / f"teacher_traces_pass{pass_idx}"
            pass_cmd = [
                sys.executable,
                "scripts/generate_manifesto_teacher_traces.py",
                "--output-dir",
                str(pass_dir),
                "--train-size",
                str(args.teacher_trace_train_size),
                "--val-size",
                str(args.teacher_trace_val_size),
                "--test-size",
                str(args.teacher_trace_test_size),
                "--num-workers",
                str(args.teacher_trace_num_workers),
                "--seed",
                str(spec["seed"]),
                "--teacher-base-url",
                str(args.teacher_base_url),
                "--teacher-model",
                str(args.teacher_model),
                "--scorer-base-url",
                str(args.teacher_base_url),
                "--scorer-model",
                str(args.teacher_model),
                "--score-tolerance-raw",
                str(spec["score_tol"]),
                "--max-attempts",
                str(spec["max_attempts"]),
                "--summary-temperature",
                str(summary_temp),
                "--dspy-guidance-temperature",
                str(dspy_temp),
                "--dspy-guidance-max-tokens",
                str(dspy_max),
                "--min-source-chars",
                str(args.min_source_chars),
                "--max-source-chars",
                str(args.max_source_chars),
            ]
            pass_cmd.append("--use-dspy-guidance" if use_dspy else "--no-use-dspy-guidance")
            pass_result = _run_command(
                pass_cmd,
                log_path=output_dir / "logs" / f"stage_b_teacher_traces_pass{pass_idx}.log",
                cwd=repo_root,
                dry_run=bool(args.dry_run),
            )
            if int(pass_result.get("returncode", 1)) != 0:
                raise BootstrapFailure(f"Stage B teacher traces pass{pass_idx} failed")

            pass_manifest_path = pass_dir / "manifest.json"
            last_manifest_path = pass_manifest_path
            pass_manifest = {
                "accepted_docs": int(args.teacher_trace_min_accepted) if pass_idx == 1 else 0,
                "paths": {"records": str(pass_dir / "teacher_trace_records.jsonl")},
            }
            if not args.dry_run:
                pass_manifest = _load_json(pass_manifest_path)
            accepted_docs = int(pass_manifest.get("accepted_docs", 0) or 0)
            pass_records = Path(str((pass_manifest.get("paths") or {}).get("records", pass_dir / "teacher_trace_records.jsonl")))

            stage_b[f"pass_{pass_idx}"] = {
                "run": pass_result,
                "manifest_path": str(pass_manifest_path),
                "accepted_docs": accepted_docs,
                "records_path": str(pass_records),
                "score_tolerance_raw": float(spec["score_tol"]),
                "max_attempts": int(spec["max_attempts"]),
            }
            _write_json(manifest_path, manifest)

            if accepted_docs >= int(args.teacher_trace_min_accepted):
                selected_manifest_path = pass_manifest_path
                selected_records_path = pass_records
                selected_accepted = accepted_docs
                break

            if pass_idx == 2:
                selected_manifest_path = pass_manifest_path
                selected_records_path = pass_records
                selected_accepted = accepted_docs

        if selected_records_path is None or selected_manifest_path is None:
            raise BootstrapFailure("Stage B could not resolve selected teacher traces manifest/records")
        if selected_accepted < int(args.teacher_trace_min_accepted):
            stage_b["status"] = "failed"
            stage_b["failure_reason"] = (
                "Teacher-trace accepted-doc floor not met after pass 2 "
                f"(required={int(args.teacher_trace_min_accepted)}, got={selected_accepted})"
            )
            _write_json(manifest_path, manifest)
            raise BootstrapFailure(stage_b["failure_reason"])

        stage_b["selected_manifest_path"] = str(selected_manifest_path)
        stage_b["selected_records_path"] = str(selected_records_path)
        stage_b["selected_accepted_docs"] = int(selected_accepted)
        stage_b["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage C: deterministic split views.
        stage_c: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_c_split_views"] = stage_c
        _write_json(manifest_path, manifest)

        split_views_dir = output_dir / "stage_c" / "split_views"
        stage_c_cmd = [
            sys.executable,
            "scripts/build_teacher_trace_split_views.py",
            "--records",
            str(selected_records_path),
            "--output-dir",
            str(split_views_dir),
        ]
        stage_c["run"] = _run_command(
            stage_c_cmd,
            log_path=output_dir / "logs" / "stage_c_split_views.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_c["run"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage C split view generation failed")
        stage_c["split_ids_path"] = str(split_views_dir / "split_ids.json")
        stage_c["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage E setup: switch to student + embedding topology.
        stage_e_setup: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_e_server_switch_student_embedding"] = stage_e_setup
        _write_json(manifest_path, manifest)

        if bool(args.manage_servers):
            stage_e_setup["stop_all"] = _stop_all_servers(
                cwd=repo_root,
                dry_run=bool(args.dry_run),
                log_path=output_dir / "logs" / "stage_e_stop_all.log",
            )
            if bool(args.dynamic_mode):
                if not GPU_ORCHESTRATOR_AVAILABLE:
                    raise BootstrapFailure(
                        "Dynamic mode requested but GPUOrchestrator is unavailable in this environment"
                    )
                stage_e_setup["dynamic_mode"] = True
                if not args.dry_run:
                    student_orchestrator = _build_student_orchestrator(args, repo_root)
                    assert OrchestratorMode is not None
                    asyncio.run(student_orchestrator.initialize(initial_mode=OrchestratorMode.TASK_DP2))
                    stage_e_setup["dynamic_orchestrator_status_after_init"] = student_orchestrator.get_status()
            else:
                stage_e_setup["dynamic_mode"] = False
                stage_e_setup["start_student"] = _start_vllm_background(
                    profile=str(args.student_profile),
                    port=int(student_port),
                    cuda_devices=str(args.student_cuda_devices),
                    log_path=output_dir / "logs" / "stage_e_student_server.log",
                    cwd=repo_root,
                    dry_run=bool(args.dry_run),
                )
                stage_e_setup["start_embedding"] = _start_embedding_server(
                    port=int(embedding_port),
                    cuda_devices=str(args.embedding_cuda_devices),
                    log_path=output_dir / "logs" / "stage_e_embedding_server.log",
                    cwd=repo_root,
                    dry_run=bool(args.dry_run),
                )

        if not args.dry_run:
            if not _wait_for_server(args.student_base_url, timeout_seconds=float(args.server_wait_seconds)):
                raise BootstrapFailure(f"Student server did not become ready at {args.student_base_url}")
            student_ids = _get_model_ids(args.student_base_url)
            stage_e_setup["student_model_ids"] = student_ids
            if str(args.student_model) not in set(student_ids):
                raise BootstrapFailure(
                    f"Student server at {args.student_base_url} does not serve {args.student_model}. "
                    f"Served IDs: {student_ids}"
                )
            if student_orchestrator is not None:
                ready = asyncio.run(student_orchestrator.ensure_embedding_ready(reason="stage_e_proxy_gepa"))
                if not ready:
                    raise BootstrapFailure("Dynamic orchestrator could not make embedding endpoint ready")
                stage_e_setup["dynamic_orchestrator_status_after_embedding_ready"] = student_orchestrator.get_status()
            else:
                if not _wait_for_server(args.embedding_base_url, timeout_seconds=float(args.server_wait_seconds)):
                    raise BootstrapFailure(f"Embedding server did not become ready at {args.embedding_base_url}")
            embedding_ids = _get_model_ids(args.embedding_base_url)
            stage_e_setup["embedding_model_ids"] = embedding_ids
            if str(args.embedding_model) not in set(embedding_ids):
                raise BootstrapFailure(
                    f"Embedding server at {args.embedding_base_url} does not serve {args.embedding_model}. "
                    f"Served IDs: {embedding_ids}"
                )

        stage_e_setup["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage E training + summarize-only passes.
        stage_e: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_e_training_and_summarize"] = stage_e
        _write_json(manifest_path, manifest)

        baseline_real_eval_dir = output_dir / "stage_f" / "real_anchor_baseline_eval"
        baseline_real_predictions = baseline_real_eval_dir / "predictions.jsonl"
        baseline_real_summarize_cmd = [
            sys.executable,
            "scripts/eval_manifesto_teacher_trace_local_laws.py",
            "--records",
            str(selected_records_path),
            "--output-dir",
            str(baseline_real_eval_dir),
            "--mode",
            "summarize_only",
            "--predictions-path",
            str(baseline_real_predictions),
            "--summarizer-base-url",
            str(args.student_base_url),
            "--summarizer-model",
            str(args.student_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.real_eval_splits:
            baseline_real_summarize_cmd.extend(["--splits", *[str(value) for value in args.real_eval_splits]])
        stage_e["baseline_real_summarize"] = _run_command(
            baseline_real_summarize_cmd,
            log_path=output_dir / "logs" / "stage_e_real_baseline_summarize.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_e["baseline_real_summarize"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage E baseline real-anchor summarize_only failed")

        proxy_gepa_dir = output_dir / "stage_e" / "proxy_gepa"
        proxy_cmd = [
            sys.executable,
            "scripts/bootstrap_lawstress_summarizer.py",
            "--records",
            str(records_path),
            "--output-dir",
            str(proxy_gepa_dir),
            "--student-port",
            str(student_port),
            "--student-model",
            str(args.student_model),
            "--student-temperature",
            str(args.student_temperature),
            "--student-max-tokens",
            str(args.student_max_tokens),
            "--embedding-url",
            str(args.embedding_base_url),
            "--embedding-model",
            str(args.embedding_model),
            "--embedding-api-key",
            str(args.embedding_api_key),
            "--embedding-timeout-seconds",
            str(args.embedding_timeout_seconds),
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--ridge-lambda",
            str(args.ridge_lambda),
            "--gepa-budget",
            str(args.gepa_budget),
            "--num-threads",
            str(args.gepa_num_threads),
            "--objective-aggregate",
            str(args.gepa_objective_aggregate),
            "--objective-softmin-temperature",
            str(args.gepa_objective_softmin_temperature),
            "--objective-component-floor",
            str(args.gepa_objective_component_floor),
            "--seed",
            str(args.seed),
        ]
        stage_e["proxy_gepa"] = _run_command(
            proxy_cmd,
            log_path=output_dir / "logs" / "stage_e_proxy_gepa.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_e["proxy_gepa"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage E proxy+GEPA failed")

        module_path = proxy_gepa_dir / "trained_modules" / "unified_g_final.json"
        proxy_gepa_module_path: Optional[Path] = None
        if not args.dry_run:
            bootstrap_stats = _load_json(proxy_gepa_dir / "bootstrap_stats.json")
            path_value = str(((bootstrap_stats.get("paths") or {}).get("unified_g")) or module_path)
            module_path = Path(path_value)
        proxy_gepa_module_path = module_path
        stage_e["proxy_gepa_module_path"] = str(module_path)

        prompt_batch_module_path: Optional[Path] = None
        if bool(args.enable_prompt_batch_tuning):
            prompt_batch_dir = output_dir / "stage_e" / "prompt_batch"
            prompt_batch_train_pairs = (
                Path(args.prompt_batch_train_pairs)
                if args.prompt_batch_train_pairs is not None
                else split_views_dir / "summary_pairs_train.jsonl"
            )
            prompt_batch_eval_pairs = (
                Path(args.prompt_batch_eval_pairs)
                if args.prompt_batch_eval_pairs is not None
                else split_views_dir / "summary_pairs_val.jsonl"
            )
            prompt_batch_cmd = [
                sys.executable,
                "scripts/tune_manifesto_summary_prompts_batch.py",
                "--train-pairs",
                str(prompt_batch_train_pairs),
                "--eval-pairs",
                str(prompt_batch_eval_pairs),
                "--output-dir",
                str(prompt_batch_dir),
                "--student-port",
                str(student_port),
                "--student-model",
                str(args.student_model),
                "--student-temperature",
                str(args.student_temperature),
                "--student-max-tokens",
                str(args.student_max_tokens),
                "--gepa-budget",
                str(args.prompt_batch_budget),
                "--num-threads",
                str(args.prompt_batch_num_threads),
                "--seed",
                str(args.seed),
            ]
            if bool(args.prompt_batch_include_score_conditioning):
                prompt_batch_cmd.append("--include-score-conditioning")
            stage_e["prompt_batch_tuning"] = _run_command(
                prompt_batch_cmd,
                log_path=output_dir / "logs" / "stage_e_prompt_batch_tuning.log",
                cwd=repo_root,
                dry_run=bool(args.dry_run),
            )
            if int(stage_e["prompt_batch_tuning"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Stage E prompt-batch tuning failed")

            prompt_batch_manifest_path = prompt_batch_dir / "prompt_batch_manifest.json"
            stage_e["prompt_batch_manifest_path"] = str(prompt_batch_manifest_path)
            stage_e["prompt_batch_train_pairs"] = str(prompt_batch_train_pairs)
            stage_e["prompt_batch_eval_pairs"] = str(prompt_batch_eval_pairs)
            if not args.dry_run:
                prompt_batch_manifest = _load_json(prompt_batch_manifest_path)
                artifact_path = str((prompt_batch_manifest.get("artifacts") or {}).get("unified_g") or "").strip()
                if artifact_path:
                    prompt_batch_module_path = Path(artifact_path)
            else:
                prompt_batch_module_path = prompt_batch_dir / "trained_modules" / "unified_g_final.json"
            if prompt_batch_module_path is not None:
                module_path = prompt_batch_module_path
                stage_e["prompt_batch_module_path"] = str(prompt_batch_module_path)
        stage_e["evaluation_module_path"] = str(module_path)

        lawstress_post_eval_dir = output_dir / "stage_f" / "lawstress_post_eval"
        post_lawstress_summarize_cmd = [
            sys.executable,
            "scripts/eval_lawstress_dspy_module.py",
            "--records",
            str(records_path),
            "--module",
            str(module_path),
            "--output-dir",
            str(lawstress_post_eval_dir),
            "--mode",
            "summarize_only",
            "--student-port",
            str(student_port),
            "--student-model",
            str(args.student_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.lawstress_eval_splits:
            post_lawstress_summarize_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])
        stage_e["post_lawstress_summarize"] = _run_command(
            post_lawstress_summarize_cmd,
            log_path=output_dir / "logs" / "stage_e_post_lawstress_summarize.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_e["post_lawstress_summarize"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage E post-lawstress summarize_only failed")

        post_real_eval_dir = output_dir / "stage_f" / "real_anchor_post_eval"
        post_real_predictions = post_real_eval_dir / "predictions.jsonl"
        post_real_summarize_cmd = [
            sys.executable,
            "scripts/eval_manifesto_teacher_trace_local_laws.py",
            "--records",
            str(selected_records_path),
            "--output-dir",
            str(post_real_eval_dir),
            "--mode",
            "summarize_only",
            "--predictions-path",
            str(post_real_predictions),
            "--dspy-module",
            str(module_path),
            "--student-port",
            str(student_port),
            "--student-model",
            str(args.student_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.real_eval_splits:
            post_real_summarize_cmd.extend(["--splits", *[str(value) for value in args.real_eval_splits]])
        stage_e["post_real_summarize"] = _run_command(
            post_real_summarize_cmd,
            log_path=output_dir / "logs" / "stage_e_post_real_summarize.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_e["post_real_summarize"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage E post real-anchor summarize_only failed")

        stage_e["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage F setup: switch back to teacher topology.
        stage_f_setup: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_f_server_switch_teacher"] = stage_f_setup
        _write_json(manifest_path, manifest)

        if bool(args.manage_servers):
            if student_orchestrator is not None:
                stage_f_setup["dynamic_student_shutdown"] = {"requested": True}
                if not args.dry_run:
                    asyncio.run(student_orchestrator.shutdown())
                    student_orchestrator = None
            else:
                stage_f_setup["stop_all"] = _stop_all_servers(
                    cwd=repo_root,
                    dry_run=bool(args.dry_run),
                    log_path=output_dir / "logs" / "stage_f_stop_all.log",
                )
            stage_f_setup["start_teacher"] = _start_vllm_background(
                profile=str(args.teacher_profile),
                port=int(teacher_port),
                cuda_devices=str(args.teacher_cuda_devices),
                log_path=output_dir / "logs" / "stage_f_teacher_server.log",
                cwd=repo_root,
                dry_run=bool(args.dry_run),
            )

        if not args.dry_run:
            if not _wait_for_server(args.teacher_base_url, timeout_seconds=float(args.server_wait_seconds)):
                raise BootstrapFailure(f"Teacher server did not become ready at {args.teacher_base_url}")
            teacher_ids = _get_model_ids(args.teacher_base_url)
            stage_f_setup["teacher_model_ids"] = teacher_ids
            if str(args.teacher_model) not in set(teacher_ids):
                raise BootstrapFailure(
                    f"Teacher server at {args.teacher_base_url} does not serve {args.teacher_model}. "
                    f"Served IDs: {teacher_ids}"
                )

        stage_f_setup["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage F scoring-only passes.
        stage_f: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_f_score_only"] = stage_f
        _write_json(manifest_path, manifest)

        lawstress_score_cmd = [
            sys.executable,
            "scripts/eval_lawstress_dspy_module.py",
            "--records",
            str(records_path),
            "--module",
            str(module_path),
            "--output-dir",
            str(lawstress_post_eval_dir),
            "--mode",
            "score_only",
            "--teacher-base-url",
            str(args.teacher_base_url),
            "--teacher-model",
            str(args.teacher_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.lawstress_eval_splits:
            lawstress_score_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])
        stage_f["lawstress_score_only"] = _run_command(
            lawstress_score_cmd,
            log_path=output_dir / "logs" / "stage_f_lawstress_score_only.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_f["lawstress_score_only"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage F lawstress score_only failed")

        baseline_real_score_cmd = [
            sys.executable,
            "scripts/eval_manifesto_teacher_trace_local_laws.py",
            "--records",
            str(selected_records_path),
            "--output-dir",
            str(baseline_real_eval_dir),
            "--mode",
            "score_only",
            "--predictions-path",
            str(baseline_real_predictions),
            "--scorer-base-url",
            str(args.teacher_base_url),
            "--scorer-model",
            str(args.teacher_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.real_eval_splits:
            baseline_real_score_cmd.extend(["--splits", *[str(value) for value in args.real_eval_splits]])
        stage_f["baseline_real_score_only"] = _run_command(
            baseline_real_score_cmd,
            log_path=output_dir / "logs" / "stage_f_baseline_real_score_only.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_f["baseline_real_score_only"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage F baseline real score_only failed")

        post_real_score_cmd = [
            sys.executable,
            "scripts/eval_manifesto_teacher_trace_local_laws.py",
            "--records",
            str(selected_records_path),
            "--output-dir",
            str(post_real_eval_dir),
            "--mode",
            "score_only",
            "--predictions-path",
            str(post_real_predictions),
            "--scorer-base-url",
            str(args.teacher_base_url),
            "--scorer-model",
            str(args.teacher_model),
            "--num-workers",
            str(args.eval_num_workers),
        ]
        if args.real_eval_splits:
            post_real_score_cmd.extend(["--splits", *[str(value) for value in args.real_eval_splits]])
        stage_f["post_real_score_only"] = _run_command(
            post_real_score_cmd,
            log_path=output_dir / "logs" / "stage_f_post_real_score_only.log",
            cwd=repo_root,
            dry_run=bool(args.dry_run),
        )
        if int(stage_f["post_real_score_only"].get("returncode", 1)) != 0:
            raise BootstrapFailure("Stage F post real score_only failed")

        stage_f["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage 9: final manifest + gates.
        stage_9: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_9_finalize_manifest"] = stage_9
        _write_json(manifest_path, manifest)

        baseline_real_metrics_path = baseline_real_eval_dir / "eval_metrics.json"
        post_real_metrics_path = post_real_eval_dir / "eval_metrics.json"
        post_lawstress_metrics_path = lawstress_post_eval_dir / "optimized_eval_metrics.json"

        lawstress_generated = requested_lawstress if args.dry_run else _count_jsonl_rows(records_path)
        accepted_docs = int(args.teacher_trace_min_accepted) if args.dry_run else int(selected_accepted)

        post_lawstress_metrics = {
            "overall": {"c1_pass_rate": 56.0, "c2_pass_rate": 56.0, "c3_pass_rate": 56.0},
            "success": {"overall_pass": True},
        }
        baseline_real_metrics = {
            "overall": {"c1_pass_rate": 50.0, "c2_pass_rate": 50.0, "c3_pass_rate": 50.0, "avg_law_pass_rate": 50.0}
        }
        post_real_metrics = {
            "overall": {"c1_pass_rate": 56.0, "c2_pass_rate": 56.0, "c3_pass_rate": 56.0, "avg_law_pass_rate": 56.0}
        }
        if not args.dry_run:
            post_lawstress_metrics = _extract_lawstress_metrics(post_lawstress_metrics_path)
            baseline_real_metrics = _load_json(baseline_real_metrics_path)
            post_real_metrics = _load_json(post_real_metrics_path)

        lawstress_success = bool(((post_lawstress_metrics.get("success") or {}).get("overall_pass", False)))
        real_anchor_gate = _compute_real_anchor_gate(
            baseline_metrics=baseline_real_metrics,
            post_metrics=post_real_metrics,
        )

        gates = {
            "data_sufficiency": {
                "lawstress_generated_at_least_requested": bool(lawstress_generated >= requested_lawstress),
                "teacher_trace_accepted_at_least_floor": bool(accepted_docs >= int(args.teacher_trace_min_accepted)),
            },
            "post_training": {
                "lawstress_success_overall_pass": lawstress_success,
                "real_anchor_local_law_improvement": real_anchor_gate,
            },
        }
        gates["overall_pass"] = bool(
            gates["data_sufficiency"]["lawstress_generated_at_least_requested"]
            and gates["data_sufficiency"]["teacher_trace_accepted_at_least_floor"]
            and gates["post_training"]["lawstress_success_overall_pass"]
            and bool(real_anchor_gate.get("pass", False))
        )

        manifest["gates"] = gates
        manifest["artifacts"].update(
            {
                "lawstress_records": str(records_path),
                "teacher_trace_manifest": str(selected_manifest_path),
                "teacher_trace_records": str(selected_records_path),
                "split_ids": str(split_views_dir / "split_ids.json"),
                "baseline_lawstress_metrics": str(baseline_lawstress_metrics_path),
                "baseline_real_anchor_metrics": str(baseline_real_metrics_path),
                "proxy_gepa_module_path": str(proxy_gepa_module_path) if proxy_gepa_module_path is not None else None,
                "evaluation_module_path": str(module_path),
                "prompt_batch_module_path": str(prompt_batch_module_path) if prompt_batch_module_path is not None else None,
                "post_lawstress_metrics": str(post_lawstress_metrics_path),
                "post_real_anchor_metrics": str(post_real_metrics_path),
                "backend_provenance": {
                    "judge_backend": "large_qwen",
                    "tournament_backend": "disabled",
                },
            }
        )

        stage_9["status"] = "completed"
        manifest["status"] = "completed" if gates["overall_pass"] else "failed_gates"
        manifest["finished_at"] = _now_iso()
        _write_json(manifest_path, manifest)
        return 0 if gates["overall_pass"] else 1

    except Exception as exc:  # pylint: disable=broad-except
        if student_orchestrator is not None and not args.dry_run:
            try:
                asyncio.run(student_orchestrator.shutdown())
            except Exception:  # pragma: no cover
                pass
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["finished_at"] = _now_iso()
        _write_json(manifest_path, manifest)
        LOGGER.exception("Manual bootstrap run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
