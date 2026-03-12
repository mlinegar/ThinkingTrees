#!/usr/bin/env python3
"""Run the C-TreePO local-law bootstrap POC (hybrid + SFT wiring)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.teacher_trace_generator import select_seed_manifestos


LOGGER = logging.getLogger(__name__)
DEFAULT_MAIN_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
DEFAULT_STUDENT_MODEL = "/mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4"


class BootstrapFailure(RuntimeError):
    """Raised when bootstrap orchestration cannot continue."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _count_jsonl_rows(path: Path) -> int:
    if not Path(path).exists():
        return 0
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected JSON object at {path}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_command(
    cmd: Sequence[str],
    *,
    log_path: Path,
    dry_run: bool,
    cwd: Path,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_cmd = " ".join(str(part) for part in cmd)

    if dry_run:
        log_path.write_text(f"# dry-run\n{rendered_cmd}\n", encoding="utf-8")
        return {
            "command": [str(part) for part in cmd],
            "log_path": str(log_path),
            "returncode": 0,
            "dry_run": True,
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
        }

    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    started_at = _now_iso()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# cmd={rendered_cmd}\n\n")
        handle.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    finished_at = _now_iso()

    return {
        "command": [str(part) for part in cmd],
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "dry_run": False,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _extract_port_from_base_url(base_url: str, default_port: int) -> int:
    parsed = urlparse(str(base_url))
    return int(parsed.port or default_port)


def _is_server_alive(base_url: str, timeout_seconds: float = 3.0) -> bool:
    try:
        response = requests.get(f"{str(base_url).rstrip('/')}/models", timeout=float(timeout_seconds))
    except Exception:
        return False
    return bool(response.status_code == 200)


def _wait_for_server(base_url: str, timeout_seconds: float = 420.0) -> bool:
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if _is_server_alive(base_url):
            return True
        time.sleep(3.0)
    return False


def _start_vllm_profile_background(
    *,
    profile: str,
    base_url: str,
    log_path: Path,
    cwd: Path,
) -> None:
    port = _extract_port_from_base_url(base_url, default_port=8000)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = (
        f"./scripts/start_vllm.sh {profile} --port {port} "
        f"> {str(log_path)} 2>&1 &"
    )
    subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=str(cwd),
        check=True,
    )


def _start_embedding_server_background(
    *,
    embedding_base_url: str,
    log_path: Path,
    cwd: Path,
) -> None:
    port = _extract_port_from_base_url(embedding_base_url, default_port=8003)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = (
        f"./scripts/start_embedding_server.sh --port {port} "
        f"> {str(log_path)} 2>&1 &"
    )
    subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=str(cwd),
        check=True,
    )


def _ensure_servers_if_requested(args: argparse.Namespace, output_dir: Path, dry_run: bool) -> Dict[str, Any]:
    use_proxy = str(getattr(args, "student_training_strategy", "")).strip().lower() in {"proxy_gepa", "hybrid"}
    info: Dict[str, Any] = {
        "manage_servers": bool(args.manage_servers),
        "teacher_server_started": False,
        "genrm_server_started": False,
        "embedding_server_started": False,
        "teacher_alive_before": _is_server_alive(args.teacher_base_url),
        "genrm_alive_before": _is_server_alive(args.genrm_base_url),
        "embedding_alive_before": _is_server_alive(args.embedding_url) if use_proxy else None,
    }
    if not bool(args.manage_servers):
        return info

    if dry_run:
        info["dry_run"] = True
        return info

    if not _is_server_alive(args.teacher_base_url):
        teacher_log = output_dir / "logs" / "managed_teacher_server.log"
        _start_vllm_profile_background(
            profile=str(args.teacher_profile),
            base_url=str(args.teacher_base_url),
            log_path=teacher_log,
            cwd=Path(__file__).resolve().parents[1],
        )
        info["teacher_server_started"] = True

    if not bool(args.disable_genrm) and not _is_server_alive(args.genrm_base_url):
        genrm_log = output_dir / "logs" / "managed_genrm_server.log"
        _start_vllm_profile_background(
            profile=str(args.genrm_profile),
            base_url=str(args.genrm_base_url),
            log_path=genrm_log,
            cwd=Path(__file__).resolve().parents[1],
        )
        info["genrm_server_started"] = True

    if use_proxy and not _is_server_alive(args.embedding_url):
        embedding_log = output_dir / "logs" / "managed_embedding_server.log"
        _start_embedding_server_background(
            embedding_base_url=str(args.embedding_url),
            log_path=embedding_log,
            cwd=Path(__file__).resolve().parents[1],
        )
        info["embedding_server_started"] = True

    teacher_ready = _wait_for_server(args.teacher_base_url, timeout_seconds=float(args.server_wait_seconds))
    if not teacher_ready:
        raise BootstrapFailure(f"Teacher server did not become ready at {args.teacher_base_url}")

    if not bool(args.disable_genrm):
        genrm_ready = _wait_for_server(args.genrm_base_url, timeout_seconds=float(args.server_wait_seconds))
        if not genrm_ready:
            raise BootstrapFailure(f"GenRM server did not become ready at {args.genrm_base_url}")

    if use_proxy:
        embedding_ready = _wait_for_server(args.embedding_url, timeout_seconds=float(args.server_wait_seconds))
        if not embedding_ready:
            raise BootstrapFailure(f"Embedding server did not become ready at {args.embedding_url}")

    info["teacher_alive_after"] = _is_server_alive(args.teacher_base_url)
    info["genrm_alive_after"] = _is_server_alive(args.genrm_base_url)
    info["embedding_alive_after"] = _is_server_alive(args.embedding_url) if use_proxy else None
    return info


def _pick_anchor_manifesto_id(*, seed: int, min_source_chars: int) -> str:
    dataset = ManifestoDataset(require_text=True)
    selected = select_seed_manifestos(
        dataset,
        n_docs=1,
        seed=int(seed),
        min_source_chars=int(min_source_chars),
        balanced_bins=False,
    )
    if not selected:
        raise BootstrapFailure("Could not pick anchor manifesto ID from dataset")
    return str(selected[0].manifesto_id)


def _extract_best_tune_config(candidate_results_path: Path) -> Dict[str, Any]:
    payload = _load_json(candidate_results_path)
    best = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(best, dict):
        return {}
    cfg = best.get("config")
    return dict(cfg) if isinstance(cfg, dict) else {}


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
        b = _safe_rate(baseline, key)
        p = _safe_rate(post, key)
        deltas[key] = None if b is None or p is None else float(p - b)

    baseline_avg = [value for value in (_safe_rate(baseline, key) for key in keys) if value is not None]
    post_avg = [value for value in (_safe_rate(post, key) for key in keys) if value is not None]

    avg_delta: Optional[float]
    if len(baseline_avg) == 3 and len(post_avg) == 3:
        avg_delta = float(sum(post_avg) / 3.0 - sum(baseline_avg) / 3.0)
    else:
        avg_delta = None

    avg_improve_ok = bool(avg_delta is not None and avg_delta >= 5.0)
    no_large_regression_ok = bool(
        all(delta is not None and delta >= -3.0 for delta in deltas.values())
    )

    return {
        "baseline": baseline,
        "post": post,
        "deltas": deltas,
        "average_delta": avg_delta,
        "average_gain_at_least_5": avg_improve_ok,
        "no_single_law_regresses_by_more_than_3": no_large_regression_ok,
        "pass": bool(avg_improve_ok and no_large_regression_ok),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-law bootstrap POC orchestration")

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manage-servers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--server-wait-seconds", type=float, default=420.0)

    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--teacher-profile", type=str, default="qwen3.5-397b-a17b-nvfp4")

    parser.add_argument("--scorer-base-url", type=str, default=None)
    parser.add_argument("--scorer-model", type=str, default=None)

    parser.add_argument("--genrm-base-url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--genrm-profile", type=str, default="genrm-nvfp4")
    parser.add_argument(
        "--disable-genrm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "GenRM is deprecated for local-law bootstrap and remains disabled by default."
        ),
    )

    parser.add_argument("--baseline-summarizer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--baseline-summarizer-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--post-sft-summarizer-base-url", type=str, default=None)
    parser.add_argument("--post-sft-summarizer-model", type=str, default=None)

    parser.add_argument("--student-base-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument(
        "--student-training-strategy",
        type=str,
        default="proxy_gepa",
        choices=["proxy_gepa", "sft", "hybrid"],
        help="proxy_gepa uses embedding-proxy + GEPA unified-g; sft uses LoRA SFT; hybrid runs both.",
    )

    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--student-model", type=str, default=None)
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument(
        "--student-max-tokens",
        type=int,
        default=0,
        help="Max tokens for student DSPy generations (<=0 uses model/context-window default).",
    )

    parser.add_argument("--embedding-url", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--gepa-budget", type=str, default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--gepa-num-threads", type=int, default=8)
    parser.add_argument(
        "--gepa-objective-aggregate",
        type=str,
        default="min",
        choices=["weighted_mean", "min", "bottleneck_min", "softmin", "floor_then_weighted"],
    )
    parser.add_argument("--gepa-objective-softmin-temperature", type=float, default=0.08)
    parser.add_argument("--gepa-objective-component-floor", type=float, default=0.55)
    parser.add_argument("--enable-prompt-batch-tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-batch-train-pairs", type=Path, default=None)
    parser.add_argument("--prompt-batch-eval-pairs", type=Path, default=None)
    parser.add_argument("--prompt-batch-budget", type=str, default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--prompt-batch-num-threads", type=int, default=8)
    parser.add_argument(
        "--prompt-batch-include-score-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--lawstress-train-size", type=int, default=48)
    parser.add_argument("--lawstress-val-size", type=int, default=12)
    parser.add_argument("--lawstress-test-size", type=int, default=12)
    parser.add_argument("--lawstress-num-workers", type=int, default=16)

    parser.add_argument("--teacher-trace-train-size", type=int, default=12)
    parser.add_argument("--teacher-trace-val-size", type=int, default=4)
    parser.add_argument("--teacher-trace-test-size", type=int, default=4)
    parser.add_argument("--teacher-trace-min-accepted", type=int, default=12)
    parser.add_argument("--teacher-trace-num-workers", type=int, default=16)

    parser.add_argument("--manifesto-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-source-chars", type=int, default=1200)
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=0,
        help="Max chars for teacher-trace source text prompts (<=0 disables clipping).",
    )

    parser.add_argument("--sft-epochs", type=float, default=2.0)
    parser.add_argument("--sft-learning-rate", type=float, default=1e-5)
    parser.add_argument("--sft-per-device-batch-size", type=int, default=1)
    parser.add_argument("--sft-gradient-accumulation-steps", type=int, default=16)

    parser.add_argument(
        "--lawstress-eval-splits",
        nargs="*",
        default=["test"],
        help="Splits to include for LawStress baseline/post evaluation (default: test).",
    )
    parser.add_argument("--real-anchor-eval-splits", nargs="*", default=["test"])
    parser.add_argument(
        "--skip-real-anchor",
        action="store_true",
        help=(
            "Skip teacher-trace generation + real-anchor evaluation stages. "
            "When combined with --student-training-strategy sft|hybrid, the orchestrator will "
            "build SFT pairs from LawStress reference summaries instead."
        ),
    )
    parser.add_argument("--run-single-doc-audit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--audit-manifesto-id", type=str, default=None)

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not bool(args.disable_genrm):
        LOGGER.error(
            "GenRM is not supported in local-law bootstrap POC. "
            "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM."
        )
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (Path("outputs") / f"bootstrap_poc_{_timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "bootstrap_manifest.json"

    scorer_base_url = args.scorer_base_url or args.teacher_base_url
    scorer_model = args.scorer_model or args.teacher_model
    post_sft_base_url = args.post_sft_summarizer_base_url or args.baseline_summarizer_base_url

    requested_lawstress = (
        int(args.lawstress_train_size) + int(args.lawstress_val_size) + int(args.lawstress_test_size)
    )
    requested_teacher_traces = (
        int(args.teacher_trace_train_size)
        + int(args.teacher_trace_val_size)
        + int(args.teacher_trace_test_size)
    )
    if int(args.lawstress_num_workers) < 1:
        raise ValueError(f"--lawstress-num-workers must be >= 1 (got {int(args.lawstress_num_workers)})")
    if int(args.teacher_trace_num_workers) < 1:
        raise ValueError(f"--teacher-trace-num-workers must be >= 1 (got {int(args.teacher_trace_num_workers)})")
    if requested_lawstress > 1 and int(args.lawstress_num_workers) < 2:
        raise ValueError(
            "Single-worker LawStress generation is disabled for multi-record runs. "
            f"Set --lawstress-num-workers >= 2 (got {int(args.lawstress_num_workers)}, records={requested_lawstress})."
        )
    if (
        not bool(args.skip_real_anchor)
        and requested_teacher_traces > 1
        and int(args.teacher_trace_num_workers) < 2
    ):
        raise ValueError(
            "Single-worker teacher-trace generation is disabled for multi-doc runs. "
            f"Set --teacher-trace-num-workers >= 2 (got {int(args.teacher_trace_num_workers)}, docs={requested_teacher_traces})."
        )

    manifest: Dict[str, Any] = {
        "created_at": _now_iso(),
        "status": "running",
        "config": {
            "manage_servers": bool(args.manage_servers),
            "teacher_base_url": str(args.teacher_base_url),
            "teacher_model": str(args.teacher_model),
            "scorer_base_url": str(scorer_base_url),
            "scorer_model": str(scorer_model),
            "judge_backend": "large_qwen",
            "tournament_backend": "disabled",
            "genrm_base_url": str(args.genrm_base_url),
            "disable_genrm": bool(args.disable_genrm),
            "student_base_model": str(args.student_base_model),
            "student_training_strategy": str(args.student_training_strategy),
            "student_port": int(args.student_port),
            "student_model": str(args.student_model or ""),
            "embedding_url": str(args.embedding_url),
            "embedding_model": str(args.embedding_model),
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
            "baseline_summarizer_model": str(args.baseline_summarizer_model),
            "post_sft_summarizer_model": str(args.post_sft_summarizer_model or ""),
            "lawstress_sizes": {
                "train": int(args.lawstress_train_size),
                "val": int(args.lawstress_val_size),
                "test": int(args.lawstress_test_size),
            },
            "lawstress_num_workers": int(args.lawstress_num_workers),
            "lawstress_eval_splits": [str(value) for value in (args.lawstress_eval_splits or [])],
            "teacher_trace_sizes": {
                "train": int(args.teacher_trace_train_size),
                "val": int(args.teacher_trace_val_size),
                "test": int(args.teacher_trace_test_size),
            },
            "teacher_trace_min_accepted": int(args.teacher_trace_min_accepted),
            "teacher_trace_num_workers": int(args.teacher_trace_num_workers),
            "skip_real_anchor": bool(args.skip_real_anchor),
            "dry_run": bool(args.dry_run),
        },
        "phases": {},
        "gates": {},
        "artifacts": {},
    }

    _write_json(manifest_path, manifest)

    try:
        server_info = _ensure_servers_if_requested(args, output_dir, dry_run=bool(args.dry_run))
        manifest["artifacts"]["server_management"] = server_info
        _write_json(manifest_path, manifest)

        # Stage A: LawStress baseline
        stage_a: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_a_lawstress"] = stage_a
        _write_json(manifest_path, manifest)

        stage_a_root = output_dir / "stage_a"
        lawstress_data_dir = stage_a_root / "lawstress_data"
        baseline_eval_dir = stage_a_root / "baseline_eval"
        lawstress_post_eval_dir = output_dir / "stage_f" / "lawstress_post_eval"

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
            "--num-workers",
            str(args.lawstress_num_workers),
            "--seed",
            str(args.seed),
            "--teacher-base-url",
            str(args.teacher_base_url),
            "--teacher-model",
            str(args.teacher_model),
        ]
        stage_a["generate"] = _run_command(
            stage_a_generate_cmd,
            log_path=output_dir / "logs" / "stage_a_generate_lawstress.log",
            dry_run=bool(args.dry_run),
            cwd=repo_root,
        )
        if int(stage_a["generate"].get("returncode", 1)) != 0:
            raise BootstrapFailure("LawStress generation failed")

        records_path = lawstress_data_dir / "lawstress_records.jsonl"
        lawstress_generated = (
            requested_lawstress
            if args.dry_run
            else _count_jsonl_rows(records_path)
        )

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
            str(args.baseline_summarizer_base_url),
            "--summarizer-model",
            str(args.baseline_summarizer_model),
        ]
        if args.lawstress_eval_splits:
            stage_a_summarize_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])
        stage_a["baseline_summarize"] = _run_command(
            stage_a_summarize_cmd,
            log_path=output_dir / "logs" / "stage_a_baseline_summarize.log",
            dry_run=bool(args.dry_run),
            cwd=repo_root,
        )
        if int(stage_a["baseline_summarize"].get("returncode", 1)) != 0:
            raise BootstrapFailure("LawStress baseline summarize_only failed")

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
            str(scorer_base_url),
            "--scorer-model",
            str(scorer_model),
            "--disable-genrm",
        ]
        if args.lawstress_eval_splits:
            stage_a_score_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])

        stage_a["baseline_score_and_judge"] = _run_command(
            stage_a_score_cmd,
            log_path=output_dir / "logs" / "stage_a_baseline_score.log",
            dry_run=bool(args.dry_run),
            cwd=repo_root,
        )
        if int(stage_a["baseline_score_and_judge"].get("returncode", 1)) != 0:
            raise BootstrapFailure("LawStress baseline score_and_judge_only failed")

        baseline_lawstress_metrics_path = baseline_eval_dir / "eval_metrics.json"
        baseline_lawstress_metrics = (
            {
                "overall": {
                    "c1_pass_rate": 50.0,
                    "c2_pass_rate": 50.0,
                    "c3_pass_rate": 50.0,
                    "same_side_of_neutral_pct": 60.0,
                    "mae": 0.10,
                },
                "success": {"overall_pass": False},
            }
            if args.dry_run
            else _load_json(baseline_lawstress_metrics_path)
        )

        stage_a["generated_records"] = lawstress_generated
        stage_a["requested_records"] = requested_lawstress
        stage_a["baseline_eval_metrics"] = str(baseline_lawstress_metrics_path)
        stage_a["status"] = "completed"
        _write_json(manifest_path, manifest)

        anchor_manifesto_id: Optional[str] = None
        final_records_path: Optional[Path] = None
        final_trace_manifest: Dict[str, Any] = {}
        split_views_dir = output_dir / "stage_c" / "split_views"

        baseline_real_metrics_path: Optional[Path] = None
        baseline_real_metrics: Dict[str, Any] = {}

        if bool(args.skip_real_anchor):
            # Stage B: skipped in synthetic-only mode.
            stage_b: Dict[str, Any] = {"status": "skipped", "reason": "skip_real_anchor=true"}
            manifest["phases"]["stage_b_teacher_traces"] = stage_b
            _write_json(manifest_path, manifest)

            # Stage C: build LawStress split views so SFT/hybrid can run fully synthetic.
            stage_c: Dict[str, Any] = {"status": "running", "source": "lawstress"}
            manifest["phases"]["stage_c_split_views"] = stage_c
            _write_json(manifest_path, manifest)

            stage_c_cmd = [
                sys.executable,
                "scripts/build_lawstress_split_views.py",
                "--records",
                str(records_path),
                "--output-dir",
                str(split_views_dir),
            ]
            stage_c["run"] = _run_command(
                stage_c_cmd,
                log_path=output_dir / "logs" / "stage_c_split_views.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_c["run"].get("returncode", 1)) != 0:
                raise BootstrapFailure("LawStress split-view build failed")
            stage_c["split_ids_path"] = str(split_views_dir / "split_ids.json")
            stage_c["status"] = "completed"
            _write_json(manifest_path, manifest)

            # Stage D: skipped in synthetic-only mode.
            stage_d: Dict[str, Any] = {"status": "skipped", "reason": "skip_real_anchor=true"}
            manifest["phases"]["stage_d_real_anchor_baseline_eval"] = stage_d
            _write_json(manifest_path, manifest)
        else:
            # Stage B: Tune + teacher traces with retry
            stage_b = {"status": "running"}
            manifest["phases"]["stage_b_teacher_traces"] = stage_b
            _write_json(manifest_path, manifest)

            anchor_manifesto_id = str(
                args.manifesto_id
                or _pick_anchor_manifesto_id(
                    seed=int(args.seed),
                    min_source_chars=int(args.min_source_chars),
                )
            )
            stage_b["anchor_manifesto_id"] = anchor_manifesto_id

            local_tune_dir = output_dir / "stage_b" / "local_tune"
            local_tune_cmd = [
                sys.executable,
                "scripts/run_single_manifesto_local_law_tune.py",
                "--manifesto-id",
                anchor_manifesto_id,
                "--output-root",
                str(local_tune_dir),
                "--teacher-base-url",
                str(args.teacher_base_url),
                "--scorer-base-url",
                str(scorer_base_url),
                "--teacher-model",
                str(args.teacher_model),
                "--scorer-model",
                str(scorer_model),
                "--seed",
                str(args.seed),
                "--min-source-chars",
                str(args.min_source_chars),
                "--max-source-chars",
                str(args.max_source_chars),
            ]
            stage_b["local_tune"] = _run_command(
                local_tune_cmd,
                log_path=output_dir / "logs" / "stage_b_local_tune.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_b["local_tune"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Single-doc local-law tune failed")

            best_tune_cfg = {} if args.dry_run else _extract_best_tune_config(local_tune_dir / "candidate_results.json")

            tuned_score_tolerance = float(best_tune_cfg.get("score_tolerance_raw", 20.0) or 20.0)
            tuned_max_attempts = int(best_tune_cfg.get("max_attempts", 3) or 3)
            tuned_use_dspy = bool(best_tune_cfg.get("use_dspy_guidance", False))
            tuned_dspy_temp = float(best_tune_cfg.get("dspy_guidance_temperature", 0.1) or 0.1)
            tuned_dspy_max_tokens = int(best_tune_cfg.get("dspy_guidance_max_tokens", 1600) or 1600)
            tuned_summary_temp = float(best_tune_cfg.get("summary_temperature", 0.1) or 0.1)

            pass_records: List[Tuple[int, Path, Dict[str, Any], float, int]] = []
            final_trace_dir: Optional[Path] = None

            pass_configs = [
                {
                    "score_tolerance_raw": tuned_score_tolerance,
                    "max_attempts": tuned_max_attempts,
                },
                {
                    "score_tolerance_raw": tuned_score_tolerance + 5.0,
                    "max_attempts": tuned_max_attempts + 1,
                },
            ]

            for pass_index, pass_cfg in enumerate(pass_configs, start=1):
                pass_dir = output_dir / "stage_b" / f"teacher_traces_pass{pass_index}"
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
                    str(args.seed + pass_index - 1),
                    "--teacher-base-url",
                    str(args.teacher_base_url),
                    "--teacher-model",
                    str(args.teacher_model),
                    "--scorer-base-url",
                    str(scorer_base_url),
                    "--scorer-model",
                    str(scorer_model),
                    "--score-tolerance-raw",
                    str(pass_cfg["score_tolerance_raw"]),
                    "--max-attempts",
                    str(pass_cfg["max_attempts"]),
                    "--summary-temperature",
                    str(tuned_summary_temp),
                    "--dspy-guidance-temperature",
                    str(tuned_dspy_temp),
                    "--dspy-guidance-max-tokens",
                    str(tuned_dspy_max_tokens),
                    "--min-source-chars",
                    str(args.min_source_chars),
                    "--max-source-chars",
                    str(args.max_source_chars),
                ]
                if tuned_use_dspy:
                    pass_cmd.append("--use-dspy-guidance")
                else:
                    pass_cmd.append("--no-use-dspy-guidance")

                pass_result = _run_command(
                    pass_cmd,
                    log_path=output_dir / "logs" / f"stage_b_teacher_traces_pass{pass_index}.log",
                    dry_run=bool(args.dry_run),
                    cwd=repo_root,
                )
                if int(pass_result.get("returncode", 1)) != 0:
                    raise BootstrapFailure(f"Teacher-trace generation failed on pass {pass_index}")

                pass_manifest_path = pass_dir / "manifest.json"
                pass_manifest = {
                    "accepted_docs": int(args.teacher_trace_min_accepted) if args.dry_run else 0,
                    "paths": {"records": str(pass_dir / "teacher_trace_records.jsonl")},
                }
                if not args.dry_run:
                    pass_manifest = _load_json(pass_manifest_path)

                accepted_docs = int(pass_manifest.get("accepted_docs", 0) or 0)
                pass_records.append(
                    (
                        pass_index,
                        pass_dir,
                        pass_manifest,
                        float(pass_cfg["score_tolerance_raw"]),
                        int(pass_cfg["max_attempts"]),
                    )
                )

                stage_b[f"pass_{pass_index}"] = {
                    "run": pass_result,
                    "manifest_path": str(pass_manifest_path),
                    "accepted_docs": accepted_docs,
                    "score_tolerance_raw": float(pass_cfg["score_tolerance_raw"]),
                    "max_attempts": int(pass_cfg["max_attempts"]),
                }

                if accepted_docs >= int(args.teacher_trace_min_accepted):
                    final_trace_dir = pass_dir
                    final_trace_manifest = pass_manifest
                    break

            if final_trace_dir is None:
                # If none passed the floor, use the second pass for artifacts and fail fast.
                _, final_trace_dir, final_trace_manifest, _, _ = pass_records[-1]
                stage_b["status"] = "failed"
                stage_b["failure_reason"] = (
                    "Teacher trace accepted-doc floor not met after pass 2 "
                    f"(required={int(args.teacher_trace_min_accepted)}, "
                    f"got={int(final_trace_manifest.get('accepted_docs', 0) or 0)})"
                )
                _write_json(manifest_path, manifest)
                raise BootstrapFailure(stage_b["failure_reason"])

            stage_b["selected_pass_dir"] = str(final_trace_dir)
            stage_b["selected_records_path"] = str(final_trace_manifest.get("paths", {}).get("records", ""))
            stage_b["selected_accepted_docs"] = int(final_trace_manifest.get("accepted_docs", 0) or 0)
            stage_b["status"] = "completed"
            _write_json(manifest_path, manifest)

            # Stage C: teacher-trace split views.
            stage_c = {"status": "running", "source": "teacher_trace"}
            manifest["phases"]["stage_c_split_views"] = stage_c
            _write_json(manifest_path, manifest)

            final_records_path = Path(
                str(
                    final_trace_manifest.get("paths", {}).get(
                        "records", final_trace_dir / "teacher_trace_records.jsonl"
                    )
                )
            )
            stage_c_cmd = [
                sys.executable,
                "scripts/build_teacher_trace_split_views.py",
                "--records",
                str(final_records_path),
                "--output-dir",
                str(split_views_dir),
            ]
            stage_c["run"] = _run_command(
                stage_c_cmd,
                log_path=output_dir / "logs" / "stage_c_split_views.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_c["run"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Split-view build failed")
            stage_c["split_ids_path"] = str(split_views_dir / "split_ids.json")
            stage_c["status"] = "completed"
            _write_json(manifest_path, manifest)

            # Stage D: baseline real-anchor evaluation
            stage_d = {"status": "running"}
            manifest["phases"]["stage_d_real_anchor_baseline_eval"] = stage_d
            _write_json(manifest_path, manifest)

            baseline_real_eval_dir = output_dir / "stage_d" / "baseline_real_anchor_eval"
            stage_d_cmd = [
                sys.executable,
                "scripts/eval_manifesto_teacher_trace_local_laws.py",
                "--records",
                str(final_records_path),
                "--output-dir",
                str(baseline_real_eval_dir),
                "--mode",
                "full",
                "--summarizer-base-url",
                str(args.baseline_summarizer_base_url),
                "--summarizer-model",
                str(args.baseline_summarizer_model),
                "--scorer-base-url",
                str(scorer_base_url),
                "--scorer-model",
                str(scorer_model),
            ]
            if args.real_anchor_eval_splits:
                stage_d_cmd.extend(["--splits", *[str(value) for value in args.real_anchor_eval_splits]])

            stage_d["run"] = _run_command(
                stage_d_cmd,
                log_path=output_dir / "logs" / "stage_d_real_anchor_baseline_eval.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_d["run"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Baseline real-anchor eval failed")

            baseline_real_metrics_path = baseline_real_eval_dir / "eval_metrics.json"
            baseline_real_metrics = {
                "overall": {
                    "c1_pass_rate": 50.0,
                    "c2_pass_rate": 50.0,
                    "c3_pass_rate": 50.0,
                    "avg_law_pass_rate": 50.0,
                }
            }
            if not args.dry_run:
                baseline_real_metrics = _load_json(baseline_real_metrics_path)

            stage_d["eval_metrics_path"] = str(baseline_real_metrics_path)
            stage_d["status"] = "completed"
            _write_json(manifest_path, manifest)

        # Stage E: student training (proxy GEPA and/or SFT)
        stage_e: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_e_sft"] = stage_e
        _write_json(manifest_path, manifest)

        strategy = str(args.student_training_strategy).strip().lower()
        stage_e["strategy"] = strategy

        sft_dir = output_dir / "stage_e" / "sft"
        proxy_gepa_dir = output_dir / "stage_e" / "proxy_gepa"
        train_pairs = split_views_dir / "summary_pairs_train.jsonl"
        eval_pairs = split_views_dir / "summary_pairs_val.jsonl"

        trained_module_path: Optional[Path] = None
        proxy_gepa_module_path: Optional[Path] = None
        prompt_batch_module_path: Optional[Path] = None
        sft_manifest_path: Optional[Path] = None
        sft_manifest: Dict[str, Any] = {}

        if strategy in {"proxy_gepa", "hybrid"}:
            stage_e_proxy_cmd = [
                sys.executable,
                "scripts/bootstrap_lawstress_summarizer.py",
                "--records",
                str(records_path),
                "--output-dir",
                str(proxy_gepa_dir),
                "--student-port",
                str(args.student_port),
                "--student-temperature",
                str(args.student_temperature),
                "--student-max-tokens",
                str(args.student_max_tokens),
                "--embedding-url",
                str(args.embedding_url),
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
            if args.student_model:
                stage_e_proxy_cmd.extend(["--student-model", str(args.student_model)])

            stage_e["proxy_gepa_run"] = _run_command(
                stage_e_proxy_cmd,
                log_path=output_dir / "logs" / "stage_e_proxy_gepa.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_e["proxy_gepa_run"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Proxy+GEPA training stage failed")

            bootstrap_stats_path = proxy_gepa_dir / "bootstrap_stats.json"
            if args.dry_run:
                trained_module_path = proxy_gepa_dir / "trained_modules" / "unified_g_final.json"
            else:
                bootstrap_stats = _load_json(bootstrap_stats_path)
                path_value = str(
                    ((bootstrap_stats.get("paths") or {}).get("unified_g"))
                    or (proxy_gepa_dir / "trained_modules" / "unified_g_final.json")
                )
                trained_module_path = Path(path_value)
            proxy_gepa_module_path = trained_module_path
            stage_e["proxy_gepa_stats_path"] = str(bootstrap_stats_path)
            stage_e["proxy_gepa_module_path"] = str(trained_module_path)

        if strategy in {"sft", "hybrid"}:
            stage_e_sft_cmd = [
                sys.executable,
                "scripts/train_manifesto_summary_sft.py",
                "--train-pairs",
                str(train_pairs),
                "--eval-pairs",
                str(eval_pairs),
                "--output-dir",
                str(sft_dir),
                "--base-model",
                str(args.student_base_model),
                "--epochs",
                str(args.sft_epochs),
                "--learning-rate",
                str(args.sft_learning_rate),
                "--per-device-batch-size",
                str(args.sft_per_device_batch_size),
                "--gradient-accumulation-steps",
                str(args.sft_gradient_accumulation_steps),
                "--seed",
                str(args.seed),
            ]
            if args.dry_run:
                stage_e_sft_cmd.append("--dry-run")

            stage_e["sft_run"] = _run_command(
                stage_e_sft_cmd,
                log_path=output_dir / "logs" / "stage_e_sft.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_e["sft_run"].get("returncode", 1)) != 0:
                raise BootstrapFailure("SFT training stage failed")

            sft_manifest_path = sft_dir / "sft_manifest.json"
            sft_manifest = {"artifacts": {"adapter_or_model_path": None}, "status": "dry_run"}
            if not args.dry_run:
                sft_manifest = _load_json(sft_manifest_path)
            stage_e["sft_manifest_path"] = str(sft_manifest_path)

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
            stage_e_prompt_batch_cmd = [
                sys.executable,
                "scripts/tune_manifesto_summary_prompts_batch.py",
                "--train-pairs",
                str(prompt_batch_train_pairs),
                "--eval-pairs",
                str(prompt_batch_eval_pairs),
                "--output-dir",
                str(prompt_batch_dir),
                "--student-port",
                str(args.student_port),
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
            if args.student_model:
                stage_e_prompt_batch_cmd.extend(["--student-model", str(args.student_model)])
            if bool(args.prompt_batch_include_score_conditioning):
                stage_e_prompt_batch_cmd.append("--include-score-conditioning")
            if args.dry_run:
                stage_e_prompt_batch_cmd.append("--dry-run")

            stage_e["prompt_batch_run"] = _run_command(
                stage_e_prompt_batch_cmd,
                log_path=output_dir / "logs" / "stage_e_prompt_batch.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_e["prompt_batch_run"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Prompt-batch tuning stage failed")

            prompt_batch_manifest_path = prompt_batch_dir / "prompt_batch_manifest.json"
            stage_e["prompt_batch_manifest_path"] = str(prompt_batch_manifest_path)
            stage_e["prompt_batch_train_pairs"] = str(prompt_batch_train_pairs)
            stage_e["prompt_batch_eval_pairs"] = str(prompt_batch_eval_pairs)
            if args.dry_run:
                prompt_batch_module_path = prompt_batch_dir / "trained_modules" / "unified_g_final.json"
            else:
                prompt_batch_manifest = _load_json(prompt_batch_manifest_path)
                artifact_path = str((prompt_batch_manifest.get("artifacts") or {}).get("unified_g") or "").strip()
                if artifact_path:
                    prompt_batch_module_path = Path(artifact_path)
            if prompt_batch_module_path is not None:
                trained_module_path = prompt_batch_module_path
                stage_e["prompt_batch_module_path"] = str(prompt_batch_module_path)

        stage_e["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Stage F: post-training eval + gates
        stage_f: Dict[str, Any] = {"status": "running"}
        manifest["phases"]["stage_f_post_sft_eval"] = stage_f
        _write_json(manifest_path, manifest)

        inferred_post_model = ""
        if strategy in {"sft", "hybrid"}:
            inferred_post_model = str(
                (sft_manifest.get("artifacts") or {}).get("adapter_or_model_path") or ""
            ).strip()
        post_sft_model = str(args.post_sft_summarizer_model or inferred_post_model or args.baseline_summarizer_model)
        stage_f["post_sft_model_used"] = post_sft_model
        stage_f["post_sft_base_url"] = str(post_sft_base_url)
        stage_f["proxy_gepa_module_path"] = (
            str(proxy_gepa_module_path) if proxy_gepa_module_path is not None else None
        )
        stage_f["evaluation_module_path"] = str(trained_module_path) if trained_module_path is not None else None
        stage_f["prompt_batch_module_path"] = (
            str(prompt_batch_module_path) if prompt_batch_module_path is not None else None
        )

        post_lawstress_metrics_path = lawstress_post_eval_dir / "eval_metrics.json"
        post_lawstress_metrics: Dict[str, Any] = {
            "overall": {
                "c1_pass_rate": 56.0,
                "c2_pass_rate": 56.0,
                "c3_pass_rate": 56.0,
                "same_side_of_neutral_pct": 70.0,
                "mae": 0.08,
            },
            "success": {"overall_pass": True if args.dry_run else False},
        }

        if trained_module_path is not None:
            stage_f_lawstress_eval_cmd = [
                sys.executable,
                "scripts/eval_lawstress_dspy_module.py",
                "--records",
                str(records_path),
                "--module",
                str(trained_module_path),
                "--output-dir",
                str(lawstress_post_eval_dir),
                "--mode",
                "full",
                "--student-port",
                str(args.student_port),
                "--teacher-base-url",
                str(args.teacher_base_url),
                "--teacher-model",
                str(args.teacher_model),
            ]
            if args.student_model:
                stage_f_lawstress_eval_cmd.extend(["--student-model", str(args.student_model)])
            if args.lawstress_eval_splits:
                stage_f_lawstress_eval_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])

            stage_f["lawstress_post_eval"] = _run_command(
                stage_f_lawstress_eval_cmd,
                log_path=output_dir / "logs" / "stage_f_lawstress_post_eval.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_f["lawstress_post_eval"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Post-training LawStress DSPy module eval failed")

            post_lawstress_metrics_path = lawstress_post_eval_dir / "optimized_eval_metrics.json"
            if not args.dry_run:
                optimized_payload = _load_json(post_lawstress_metrics_path)
                if isinstance(optimized_payload.get("metrics"), dict):
                    post_lawstress_metrics = dict(optimized_payload.get("metrics") or {})
                else:
                    post_lawstress_metrics = optimized_payload
        else:
            post_predictions = lawstress_post_eval_dir / "predictions.jsonl"
            stage_f_lawstress_summarize_cmd = [
                sys.executable,
                "scripts/eval_manifesto_lawstress.py",
                "--records",
                str(records_path),
                "--output-dir",
                str(lawstress_post_eval_dir),
                "--mode",
                "summarize_only",
                "--predictions-path",
                str(post_predictions),
                "--summarizer-base-url",
                str(post_sft_base_url),
                "--summarizer-model",
                str(post_sft_model),
            ]
            if args.lawstress_eval_splits:
                stage_f_lawstress_summarize_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])
            stage_f["lawstress_post_summarize"] = _run_command(
                stage_f_lawstress_summarize_cmd,
                log_path=output_dir / "logs" / "stage_f_lawstress_post_summarize.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_f["lawstress_post_summarize"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Post-SFT LawStress summarize_only failed")

            stage_f_lawstress_score_cmd = [
                sys.executable,
                "scripts/eval_manifesto_lawstress.py",
                "--records",
                str(records_path),
                "--output-dir",
                str(lawstress_post_eval_dir),
                "--mode",
                "score_and_judge_only",
                "--predictions-path",
                str(post_predictions),
                "--scorer-base-url",
                str(scorer_base_url),
                "--scorer-model",
                str(scorer_model),
                "--disable-genrm",
                "--baseline-metrics",
                str(baseline_lawstress_metrics_path),
            ]
            if args.lawstress_eval_splits:
                stage_f_lawstress_score_cmd.extend(["--splits", *[str(value) for value in args.lawstress_eval_splits]])

            stage_f["lawstress_post_score_and_judge"] = _run_command(
                stage_f_lawstress_score_cmd,
                log_path=output_dir / "logs" / "stage_f_lawstress_post_score.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_f["lawstress_post_score_and_judge"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Post-SFT LawStress score_and_judge_only failed")

            if not args.dry_run:
                post_lawstress_metrics = _load_json(post_lawstress_metrics_path)

        post_real_metrics_path: Optional[Path] = None
        post_real_metrics: Dict[str, Any] = {}

        if bool(args.skip_real_anchor):
            stage_f["real_anchor_post_eval"] = {"status": "skipped", "reason": "skip_real_anchor=true"}
        else:
            if final_records_path is None:
                raise BootstrapFailure("Final teacher-trace records path missing; cannot run real-anchor eval")

            post_real_eval_dir = output_dir / "stage_f" / "post_real_anchor_eval"
            stage_f_real_cmd = [
                sys.executable,
                "scripts/eval_manifesto_teacher_trace_local_laws.py",
                "--records",
                str(final_records_path),
                "--output-dir",
                str(post_real_eval_dir),
                "--mode",
                "full",
                "--scorer-base-url",
                str(scorer_base_url),
                "--scorer-model",
                str(scorer_model),
            ]
            if trained_module_path is not None:
                stage_f_real_cmd.extend(
                    [
                        "--dspy-module",
                        str(trained_module_path),
                        "--student-port",
                        str(args.student_port),
                    ]
                )
                if args.student_model:
                    stage_f_real_cmd.extend(["--student-model", str(args.student_model)])
            else:
                stage_f_real_cmd.extend(
                    [
                        "--summarizer-base-url",
                        str(post_sft_base_url),
                        "--summarizer-model",
                        str(post_sft_model),
                    ]
                )
            if args.real_anchor_eval_splits:
                stage_f_real_cmd.extend(["--splits", *[str(value) for value in args.real_anchor_eval_splits]])

            stage_f["real_anchor_post_eval"] = _run_command(
                stage_f_real_cmd,
                log_path=output_dir / "logs" / "stage_f_real_anchor_post_eval.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_f["real_anchor_post_eval"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Post-training real-anchor eval failed")

            post_real_metrics_path = post_real_eval_dir / "eval_metrics.json"
            post_real_metrics = {
                "overall": {
                    "c1_pass_rate": 56.0,
                    "c2_pass_rate": 56.0,
                    "c3_pass_rate": 56.0,
                    "avg_law_pass_rate": 56.0,
                }
            }
            if not args.dry_run:
                post_real_metrics = _load_json(post_real_metrics_path)

        stage_f["lawstress_post_eval_metrics_path"] = str(post_lawstress_metrics_path)
        stage_f["real_anchor_post_eval_metrics_path"] = (
            str(post_real_metrics_path) if post_real_metrics_path is not None else None
        )
        stage_f["status"] = "completed"
        _write_json(manifest_path, manifest)

        # Optional Stage G: single-doc TreeBuilder+Auditor sanity on real manifesto.
        stage_g: Optional[Dict[str, Any]] = None
        if bool(args.run_single_doc_audit) and trained_module_path is not None:
            stage_g = {"status": "running"}
            manifest["phases"]["stage_g_single_doc_audit"] = stage_g
            _write_json(manifest_path, manifest)

            audit_manifesto_id = str(args.audit_manifesto_id or anchor_manifesto_id or args.manifesto_id or "").strip()
            if not audit_manifesto_id:
                audit_manifesto_id = _pick_anchor_manifesto_id(
                    seed=int(args.seed),
                    min_source_chars=int(args.min_source_chars),
                )
            audit_dir = output_dir / "stage_g" / "single_doc_audit"

            stage_g_build_cmd = [
                sys.executable,
                "scripts/audit_manifesto_single_doc.py",
                "--id",
                str(audit_manifesto_id),
                "--optimized-module",
                str(trained_module_path),
                "--output-dir",
                str(audit_dir),
                "--mode",
                "build_tree_only",
                "--student-port",
                str(args.student_port),
            ]
            if args.student_model:
                stage_g_build_cmd.extend(["--student-model", str(args.student_model)])

            stage_g["build_tree_only"] = _run_command(
                stage_g_build_cmd,
                log_path=output_dir / "logs" / "stage_g_single_doc_build.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_g["build_tree_only"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Single-doc audit build_tree_only failed")

            stage_g_audit_cmd = [
                sys.executable,
                "scripts/audit_manifesto_single_doc.py",
                "--id",
                str(audit_manifesto_id),
                "--optimized-module",
                str(trained_module_path),
                "--output-dir",
                str(audit_dir),
                "--mode",
                "audit_only",
                "--teacher-base-url",
                str(args.teacher_base_url),
                "--teacher-model",
                str(args.teacher_model),
            ]
            stage_g["audit_only"] = _run_command(
                stage_g_audit_cmd,
                log_path=output_dir / "logs" / "stage_g_single_doc_audit.log",
                dry_run=bool(args.dry_run),
                cwd=repo_root,
            )
            if int(stage_g["audit_only"].get("returncode", 1)) != 0:
                raise BootstrapFailure("Single-doc audit audit_only failed")

            stage_g["output_dir"] = str(audit_dir)
            stage_g["status"] = "completed"
            _write_json(manifest_path, manifest)

        # Gates
        if bool(args.skip_real_anchor):
            teacher_trace_floor_ok = True
            real_anchor_gate: Dict[str, Any] = {
                "skipped": True,
                "reason": "skip_real_anchor=true",
                "pass": True,
            }
        else:
            teacher_trace_floor_ok = bool(
                int(final_trace_manifest.get("accepted_docs", 0) or 0) >= int(args.teacher_trace_min_accepted)
            )
            real_anchor_gate = _compute_real_anchor_gate(
                baseline_metrics=baseline_real_metrics,
                post_metrics=post_real_metrics,
            )

        lawstress_success = bool(
            ((post_lawstress_metrics.get("success") or {}).get("overall_pass", False))
        )

        gates = {
            "data_sufficiency": {
                "lawstress_generated_at_least_requested": bool(lawstress_generated >= requested_lawstress),
                "teacher_trace_accepted_at_least_floor": bool(teacher_trace_floor_ok),
            },
            "post_sft": {
                "lawstress_success_overall_pass": lawstress_success,
                "real_anchor_local_law_improvement": real_anchor_gate,
            },
        }
        gates["overall_pass"] = bool(
            gates["data_sufficiency"]["lawstress_generated_at_least_requested"]
            and gates["data_sufficiency"]["teacher_trace_accepted_at_least_floor"]
            and gates["post_sft"]["lawstress_success_overall_pass"]
            and bool(real_anchor_gate.get("pass", False))
        )

        manifest["gates"] = gates
        manifest["artifacts"].update(
            {
                "lawstress_records": str(records_path),
                "teacher_trace_records": str(final_records_path) if final_records_path is not None else None,
                "split_ids": str(split_views_dir / "split_ids.json"),
                "baseline_lawstress_metrics": str(baseline_lawstress_metrics_path),
                "baseline_real_anchor_metrics": (
                    str(baseline_real_metrics_path) if baseline_real_metrics_path is not None else None
                ),
                "student_training_strategy": str(strategy),
                "sft_manifest": str(sft_manifest_path) if sft_manifest_path is not None else None,
                "proxy_gepa_module_path": (
                    str(proxy_gepa_module_path) if proxy_gepa_module_path is not None else None
                ),
                "evaluation_module_path": str(trained_module_path) if trained_module_path is not None else None,
                "prompt_batch_module_path": (
                    str(prompt_batch_module_path) if prompt_batch_module_path is not None else None
                ),
                "post_lawstress_metrics": str(post_lawstress_metrics_path),
                "post_real_anchor_metrics": str(post_real_metrics_path) if post_real_metrics_path is not None else None,
                "backend_provenance": {
                    "judge_backend": "large_qwen",
                    "tournament_backend": "disabled",
                },
            }
        )

        manifest["status"] = "completed" if gates["overall_pass"] else "failed_gates"
        manifest["finished_at"] = _now_iso()
        _write_json(manifest_path, manifest)

        return 0 if gates["overall_pass"] else 1

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["finished_at"] = _now_iso()
        _write_json(manifest_path, manifest)
        LOGGER.exception("Bootstrap POC failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
