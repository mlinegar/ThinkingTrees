#!/usr/bin/env python3
"""
Run reusable throughput-limit sweeps for key training-pipeline stages.

Example:
  ./venv/bin/python scripts/run_pipeline_throughput_limits.py \
    --steps task_single,task_merge,task_score,task_dp2,genrm_batch,genrm_raw \
    --task-url http://localhost:8000/v1 \
    --task-replica-url http://localhost:8002/v1 \
    --genrm-url http://localhost:8001/v1 \
    --concurrency-grid 1,2,4,8,12,16 \
    --min-requests-per-point 32 \
    --requests-per-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

# Allow running from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.pipeline_limits import (  # noqa: E402
    StepSweepResult,
    default_output_path,
    expand_genrm_steps,
    format_human_summary,
    parse_concurrency_grid,
    parse_genrm_modes,
    run_pipeline_throughput_suite,
    write_suite_csv,
    write_suite_json,
)
from src.benchmark.throughput import VLLMServerManager  # noqa: E402


def _parse_steps(step_csv: str) -> List[str]:
    steps = [part.strip() for part in str(step_csv).split(",") if part.strip()]
    if not steps:
        raise ValueError("At least one step must be provided")
    valid = {
        "task_single",
        "task_merge",
        "task_score",
        "task_dp2",
        "genrm_raw",
        "genrm_batch",
        "genrm_raw_fast",
        "genrm_raw_think",
        "genrm_batch_fast",
        "genrm_batch_think",
    }
    unknown = [s for s in steps if s not in valid]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}. Valid: {sorted(valid)}")
    return steps


def _dedupe_keep_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _extract_local_port(url: str) -> int:
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"URL must start with http:// or https://, got: {url}")
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        raise ValueError(
            f"Auto-start requires local URLs. Got host '{host}' in {url}. "
            "Use --no-auto-start-servers for remote endpoints."
        )
    if parsed.port is not None:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _clear_port_listeners(port: int) -> None:
    logger = logging.getLogger(__name__)
    if shutil.which("fuser"):
        subprocess.run(
            ["fuser", "-k", f"{int(port)}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.5)
        logger.info("Cleared existing listeners on port %d via fuser", int(port))
        return

    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{int(port)}"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(0.5)
        for pid in pids:
            if _pid_exists(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        logger.info("Cleared existing listeners on port %d via lsof", int(port))
        return

    logger.warning("Could not clear port %d: neither fuser nor lsof is available", int(port))


def _kill_stale_vllm_processes() -> None:
    """
    Kill stale/orphaned vLLM worker processes that can hold GPU memory
    even when no API server is listening on a port.
    """
    logger = logging.getLogger(__name__)
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Unable to inspect running processes for stale vLLM state")
        return

    markers = (
        "VLLM::EngineCore",
        "VLLM::Worker",
        # Ensure we also clear any surviving API server parents that can
        # respawn engine cores (e.g., embedding servers on other ports).
        "vllm.entrypoints.openai.api_server",
    )
    candidate_pids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        if any(marker in cmd for marker in markers):
            candidate_pids.append(pid)

    if not candidate_pids:
        return

    pids = sorted(set(candidate_pids))
    logger.warning("Killing stale vLLM GPU processes: %s", ",".join(str(pid) for pid in pids))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(1.0)
    for pid in pids:
        if _pid_exists(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    time.sleep(0.5)


def _clear_vllm_shared_memory() -> None:
    """Remove leaked /dev/shm vLLM shared memory files from prior crashes."""
    logger = logging.getLogger(__name__)
    paths = glob.glob("/dev/shm/vllm*")
    if not paths:
        return

    removed = 0
    for path_str in paths:
        path = Path(path_str)
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
        except Exception:
            # Best effort cleanup; continue.
            continue

    if removed:
        logger.info("Cleared %d stale /dev/shm/vllm* entries", removed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep concurrency limits across task/GenRM pipeline stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        default="task_single,task_merge,task_score,genrm_batch,genrm_raw",
        help=(
            "Comma-separated steps: task_single, task_merge, task_score, task_dp2, "
            "genrm_batch, genrm_raw "
            "(or explicit variants: genrm_batch_fast, genrm_batch_think, "
            "genrm_raw_fast, genrm_raw_think)"
        ),
    )
    parser.add_argument(
        "--genrm-modes",
        default="fast,think",
        help=(
            "Modes used to expand generic GenRM steps (genrm_raw/genrm_batch). "
            "Valid: fast, think"
        ),
    )
    parser.add_argument(
        "--concurrency-grid",
        default="1,2,4,8,12,16",
        help="Comma-separated concurrency sweep values",
    )
    parser.add_argument(
        "--min-requests-per-point",
        type=int,
        default=32,
        help="Minimum number of requests to run for each concurrency point",
    )
    parser.add_argument(
        "--requests-per-concurrency",
        type=int,
        default=4,
        help="Requests multiplier per concurrency (total=max(min, c*multiplier))",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=4,
        help="Warmup requests before each step sweep",
    )

    parser.add_argument("--task-url", default="http://localhost:8000/v1")
    parser.add_argument("--task-replica-url", default=None)
    parser.add_argument("--genrm-url", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--auto-start-servers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start fresh local vLLM servers for required steps (default: enabled).",
    )
    parser.add_argument(
        "--clear-ports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear existing listeners on required ports before auto-starting.",
    )
    parser.add_argument(
        "--clear-vllm-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When auto-starting, kill stale vLLM worker processes and clear "
            "/dev/shm/vllm* before launching servers."
        ),
    )
    parser.add_argument("--task-profile", default="nemotron-30b-nvfp4")
    parser.add_argument(
        "--task-replica-profile",
        default=None,
        help="Profile for task replica server (default: same as --task-profile).",
    )
    parser.add_argument("--genrm-profile", default="genrm-nvfp4")
    parser.add_argument("--task-cuda-devices", default="0,1")
    parser.add_argument("--task-replica-cuda-devices", default="2,3")
    parser.add_argument("--genrm-cuda-devices", default="2,3")
    parser.add_argument("--task-tensor-parallel", type=int, default=2)
    parser.add_argument("--task-replica-tensor-parallel", type=int, default=2)
    parser.add_argument("--genrm-tensor-parallel", type=int, default=2)
    parser.add_argument("--startup-timeout-seconds", type=float, default=480.0)

    parser.add_argument("--task-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--genrm-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--task-max-tokens", type=int, default=512)
    parser.add_argument("--genrm-max-tokens", type=int, default=256)
    parser.add_argument("--task-batch-timeout", type=float, default=0.05)
    parser.add_argument("--task-chars", type=int, default=1200)

    parser.add_argument(
        "--genrm-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request GenRM non-thinking mode when supported",
    )
    parser.add_argument(
        "--genrm-force-json-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request JSON-object response format when supported",
    )
    parser.add_argument("--genrm-temperature", type=float, default=0.6)
    parser.add_argument("--genrm-top-p", type=float, default=0.95)

    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.98,
        help="Stability threshold for recommended settings",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=0.0,
        help="Optional p95 latency threshold; <=0 disables latency filtering",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/throughput_limits_<utc>.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-point metrics",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    requested_steps = _parse_steps(args.steps)
    genrm_modes = parse_genrm_modes(args.genrm_modes)
    steps = _dedupe_keep_order(expand_genrm_steps(requested_steps, genrm_modes))
    concurrency_grid = parse_concurrency_grid(args.concurrency_grid)

    if "task_dp2" in steps and not args.task_replica_url:
        raise ValueError("--task-replica-url is required when step 'task_dp2' is enabled")

    async def _run_suite(selected_steps: List[str]) -> Dict[str, StepSweepResult]:
        if not selected_steps:
            return {}
        return await run_pipeline_throughput_suite(
            steps=selected_steps,
            concurrency_grid=concurrency_grid,
            min_requests_per_point=int(args.min_requests_per_point),
            requests_per_concurrency=int(args.requests_per_concurrency),
            warmup_requests=int(args.warmup_requests),
            task_url=str(args.task_url),
            task_replica_url=(None if args.task_replica_url is None else str(args.task_replica_url)),
            genrm_url=str(args.genrm_url),
            task_timeout_seconds=float(args.task_timeout_seconds),
            genrm_timeout_seconds=float(args.genrm_timeout_seconds),
            task_max_tokens=int(args.task_max_tokens),
            genrm_max_tokens=int(args.genrm_max_tokens),
            task_batch_timeout=float(args.task_batch_timeout),
            task_chars=int(args.task_chars),
            api_key=str(args.api_key),
            genrm_disable_thinking=bool(args.genrm_disable_thinking),
            genrm_force_json_response=bool(args.genrm_force_json_response),
            genrm_temperature=float(args.genrm_temperature),
            genrm_top_p=float(args.genrm_top_p),
            min_success_rate=float(args.min_success_rate),
            max_p95_latency_ms=float(args.max_p95_latency_ms),
        )

    task_steps = [s for s in steps if s.startswith("task_")]
    genrm_steps = [s for s in steps if s.startswith("genrm_")]

    if not bool(args.auto_start_servers):
        results = await _run_suite(steps)
    else:
        task_manager = None
        task_replica_manager = None
        genrm_manager = None
        task_results: Dict[str, StepSweepResult] = {}
        genrm_results: Dict[str, StepSweepResult] = {}
        try:
            task_port = _extract_local_port(str(args.task_url)) if task_steps else None
            genrm_port = _extract_local_port(str(args.genrm_url)) if genrm_steps else None
            task_replica_port = (
                None
                if (args.task_replica_url is None or "task_dp2" not in task_steps)
                else _extract_local_port(str(args.task_replica_url))
            )

            if bool(args.clear_vllm_state):
                _kill_stale_vllm_processes()
                _clear_vllm_shared_memory()

            if bool(args.clear_ports):
                ports_to_clear = set()
                if task_port is not None:
                    ports_to_clear.add(task_port)
                if task_replica_port is not None:
                    ports_to_clear.add(task_replica_port)
                if genrm_port is not None:
                    ports_to_clear.add(genrm_port)
                for port in sorted(ports_to_clear):
                    _clear_port_listeners(int(port))

            if task_steps:
                if task_port is None:
                    raise ValueError("task_port must be set when task steps are selected")
                task_manager = VLLMServerManager(
                    profile=str(args.task_profile),
                    port=task_port,
                    cuda_devices=str(args.task_cuda_devices),
                    tensor_parallel=int(args.task_tensor_parallel),
                    startup_timeout=float(args.startup_timeout_seconds),
                )
                await task_manager.start()

                if "task_dp2" in task_steps:
                    if task_replica_port is None:
                        raise ValueError("--task-replica-url is required when step 'task_dp2' is enabled")
                    replica_profile = str(args.task_replica_profile or args.task_profile)
                    task_replica_manager = VLLMServerManager(
                        profile=replica_profile,
                        port=int(task_replica_port),
                        cuda_devices=str(args.task_replica_cuda_devices),
                        tensor_parallel=int(args.task_replica_tensor_parallel),
                        startup_timeout=float(args.startup_timeout_seconds),
                    )
                    await task_replica_manager.start()

                task_results = await _run_suite(task_steps)

            # task_replica and GenRM commonly share GPUs; stop replica before GenRM phase.
            if task_replica_manager is not None and genrm_steps:
                task_replica_manager.stop()
                task_replica_manager = None

            if genrm_steps:
                if genrm_port is None:
                    raise ValueError("genrm_port must be set when GenRM steps are selected")
                genrm_manager = VLLMServerManager(
                    profile=str(args.genrm_profile),
                    port=genrm_port,
                    cuda_devices=str(args.genrm_cuda_devices),
                    tensor_parallel=int(args.genrm_tensor_parallel),
                    startup_timeout=float(args.startup_timeout_seconds),
                )
                await genrm_manager.start()
                genrm_results = await _run_suite(genrm_steps)

            merged: Dict[str, StepSweepResult] = {}
            for step in steps:
                if step in task_results:
                    merged[step] = task_results[step]
                elif step in genrm_results:
                    merged[step] = genrm_results[step]
            results = merged
        finally:
            if genrm_manager is not None:
                genrm_manager.stop()
            if task_replica_manager is not None:
                task_replica_manager.stop()
            if task_manager is not None:
                task_manager.stop()

    print(format_human_summary(results))

    output_json = args.output_json or default_output_path()
    config_dict = {
        "requested_steps": requested_steps,
        "expanded_steps": steps,
        "genrm_modes": genrm_modes,
        "concurrency_grid": concurrency_grid,
        "min_requests_per_point": int(args.min_requests_per_point),
        "requests_per_concurrency": int(args.requests_per_concurrency),
        "warmup_requests": int(args.warmup_requests),
        "task_url": str(args.task_url),
        "task_replica_url": args.task_replica_url,
        "genrm_url": str(args.genrm_url),
        "auto_start_servers": bool(args.auto_start_servers),
        "clear_ports": bool(args.clear_ports),
        "clear_vllm_state": bool(args.clear_vllm_state),
        "task_profile": str(args.task_profile),
        "task_replica_profile": str(args.task_replica_profile or args.task_profile),
        "genrm_profile": str(args.genrm_profile),
        "task_cuda_devices": str(args.task_cuda_devices),
        "task_replica_cuda_devices": str(args.task_replica_cuda_devices),
        "genrm_cuda_devices": str(args.genrm_cuda_devices),
        "task_tensor_parallel": int(args.task_tensor_parallel),
        "task_replica_tensor_parallel": int(args.task_replica_tensor_parallel),
        "genrm_tensor_parallel": int(args.genrm_tensor_parallel),
        "startup_timeout_seconds": float(args.startup_timeout_seconds),
        "task_timeout_seconds": float(args.task_timeout_seconds),
        "genrm_timeout_seconds": float(args.genrm_timeout_seconds),
        "task_max_tokens": int(args.task_max_tokens),
        "genrm_max_tokens": int(args.genrm_max_tokens),
        "task_batch_timeout": float(args.task_batch_timeout),
        "task_chars": int(args.task_chars),
        "genrm_disable_thinking": bool(args.genrm_disable_thinking),
        "genrm_force_json_response": bool(args.genrm_force_json_response),
        "genrm_temperature": float(args.genrm_temperature),
        "genrm_top_p": float(args.genrm_top_p),
        "min_success_rate": float(args.min_success_rate),
        "max_p95_latency_ms": float(args.max_p95_latency_ms),
    }
    write_suite_json(output_json=output_json, config=config_dict, result=results)
    print(f"Saved JSON results: {output_json}")

    if args.output_csv is not None:
        write_suite_csv(output_csv=args.output_csv, result=results)
        print(f"Saved CSV results: {args.output_csv}")

    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
