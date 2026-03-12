#!/usr/bin/env python3
"""
Run two training-pipeline passes against a shared ConditionalMemory directory.

This script is intended for performance/integration probes where pass 2 should
show improved cache reuse (for example, non-zero L2 hits) versus pass 1.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _extract_conditional_memory(stats: Dict[str, Any]) -> Dict[str, Any]:
    cm = stats.get("conditional_memory", {})
    if not isinstance(cm, dict):
        return {}
    return {
        "mode": cm.get("mode"),
        "hit_rate": _to_float(cm.get("hit_rate")),
        "l1_hits": _to_int(cm.get("l1_hits")),
        "l2_hits": _to_int(cm.get("l2_hits")),
        "misses": _to_int(cm.get("misses")),
        "writes": _to_int(cm.get("writes")),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b - a)


def _run_command(cmd: List[str], log_path: Path, timeout_seconds: Optional[float]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            text=True,
        )
    return int(proc.returncode)


def _build_pass_args(
    *,
    pass_output_dir: Path,
    shared_memory_dir: Path,
    task: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    n_iterations: int,
    optimizer_budget: str,
    routing_policy: str,
    use_engram_memory: bool,
    genrm_init_samples: int,
    start_server: bool,
    start_genrm: bool,
    keep_servers_running: bool,
    extra_args: List[str],
) -> List[str]:
    args: List[str] = [
        "--task",
        str(task),
        "--train-samples",
        str(int(train_samples)),
        "--val-samples",
        str(int(val_samples)),
        "--test-samples",
        str(int(test_samples)),
        "--n-iterations",
        str(int(n_iterations)),
        "--optimizer-budget",
        str(optimizer_budget),
        "--conditional-memory-mode",
        "readwrite",
        "--conditional-memory-dir",
        str(shared_memory_dir),
        "--routing-policy",
        str(routing_policy),
        "--genrm-init-samples",
        str(int(genrm_init_samples)),
        "--output-dir",
        str(pass_output_dir),
    ]
    if use_engram_memory:
        args.append("--engram-memory")
    if start_server:
        args.append("--start-server")
    if start_genrm:
        args.append("--start-genrm")
    if keep_servers_running:
        args.append("--keep-servers-running")
    args.extend(extra_args)
    return args


def _run_pass(
    *,
    run_script: Path,
    pass_name: str,
    pass_output_dir: Path,
    shared_memory_dir: Path,
    task: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    n_iterations: int,
    optimizer_budget: str,
    routing_policy: str,
    use_engram_memory: bool,
    genrm_init_samples: int,
    start_server: bool,
    start_genrm: bool,
    keep_servers_running: bool,
    extra_args: List[str],
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    started_at = _utc_now_iso()
    t0 = time.perf_counter()
    pass_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(run_script)] + _build_pass_args(
        pass_output_dir=pass_output_dir,
        shared_memory_dir=shared_memory_dir,
        task=task,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        n_iterations=n_iterations,
        optimizer_budget=optimizer_budget,
        routing_policy=routing_policy,
        use_engram_memory=use_engram_memory,
        genrm_init_samples=genrm_init_samples,
        start_server=start_server,
        start_genrm=start_genrm,
        keep_servers_running=keep_servers_running,
        extra_args=extra_args,
    )
    log_path = pass_output_dir / "probe_command.log"

    returncode: int
    error: Optional[str] = None
    timed_out = False
    try:
        returncode = _run_command(cmd, log_path, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = 124
        error = "timed out"
    except Exception as exc:
        returncode = 2
        error = f"command execution error: {exc}"

    duration_seconds = float(time.perf_counter() - t0)
    completed_at = _utc_now_iso()

    final_stats_path = pass_output_dir / "final_stats.json"
    stats: Optional[Dict[str, Any]] = None
    cm: Dict[str, Any] = {}
    if final_stats_path.exists():
        try:
            stats = _load_json(final_stats_path)
            cm = _extract_conditional_memory(stats)
        except Exception as exc:
            if error is None:
                error = f"failed to parse final_stats.json: {exc}"
    elif error is None:
        error = f"missing final_stats.json at {final_stats_path}"

    success_flag = bool(stats.get("success")) if isinstance(stats, dict) else False
    status = "ok"
    if timed_out:
        status = "timeout"
    elif returncode != 0:
        status = "failed"
    elif not success_flag:
        status = "failed"

    return {
        "name": pass_name,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "status": status,
        "returncode": returncode,
        "error": error,
        "command": cmd,
        "log_path": str(log_path),
        "run_dir": str(pass_output_dir),
        "final_stats_path": str(final_stats_path),
        "pipeline_success": success_flag,
        "conditional_memory": cm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pass1/pass2 training pipeline probe with shared ConditionalMemory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-script",
        type=Path,
        default=Path("scripts/run_training_pipeline.sh"),
        help="Training pipeline wrapper script to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Probe output directory (will contain pass1/pass2 run dirs).",
    )
    parser.add_argument(
        "--shared-memory-dir",
        type=Path,
        default=None,
        help="Shared ConditionalMemory directory for both passes.",
    )
    parser.add_argument("--pass1-dir", type=Path, default=None, help="Optional explicit pass1 output dir.")
    parser.add_argument("--pass2-dir", type=Path, default=None, help="Optional explicit pass2 output dir.")
    parser.add_argument("--json-out", type=Path, default=None, help="Output JSON path.")

    parser.add_argument("--task", type=str, default="manifesto_rile")
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument("--optimizer-budget", type=str, default="light")
    parser.add_argument("--routing-policy", type=str, default="affinity_load_aware")
    parser.add_argument("--genrm-init-samples", type=int, default=4)

    parser.add_argument("--start-server", action="store_true", help="Use --start-server on pass 1.")
    parser.add_argument("--start-genrm", action="store_true", help="Use --start-genrm on pass 1.")
    parser.add_argument("--engram-memory", action="store_true", help="Enable --engram-memory on both passes.")
    parser.add_argument(
        "--keep-servers-between-passes",
        action="store_true",
        help="Pass --keep-servers-running on pass 1 so pass 2 can reuse live servers.",
    )
    parser.add_argument(
        "--timeout-seconds-per-pass",
        type=float,
        default=21600.0,
        help="Timeout per pass.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg passed through to each run_training_pipeline.sh invocation.",
    )
    parser.add_argument(
        "--require-pass2-hits",
        action="store_true",
        help="Fail if pass2 total hits (l1+l2) are not positive.",
    )
    parser.add_argument(
        "--require-hit-rate-nondecreasing",
        action="store_true",
        help="Fail if pass2 hit_rate is lower than pass1 hit_rate.",
    )
    args = parser.parse_args()

    run_script = args.run_script
    if not run_script.is_absolute():
        run_script = (PROJECT_ROOT / run_script).resolve()
    if not run_script.exists():
        raise FileNotFoundError(f"Run script not found: {run_script}")

    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_memory_dir = args.shared_memory_dir
    if shared_memory_dir is None:
        shared_memory_dir = output_dir / "shared_conditional_memory"
    elif not shared_memory_dir.is_absolute():
        shared_memory_dir = (PROJECT_ROOT / shared_memory_dir).resolve()
    shared_memory_dir.mkdir(parents=True, exist_ok=True)

    pass1_dir = args.pass1_dir
    if pass1_dir is None:
        pass1_dir = output_dir / "pass1"
    elif not pass1_dir.is_absolute():
        pass1_dir = (PROJECT_ROOT / pass1_dir).resolve()

    pass2_dir = args.pass2_dir
    if pass2_dir is None:
        pass2_dir = output_dir / "pass2"
    elif not pass2_dir.is_absolute():
        pass2_dir = (PROJECT_ROOT / pass2_dir).resolve()

    json_out = args.json_out
    if json_out is None:
        json_out = output_dir / "memory_reuse_probe.json"
    elif not json_out.is_absolute():
        json_out = (PROJECT_ROOT / json_out).resolve()

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    pass1 = _run_pass(
        run_script=run_script,
        pass_name="pass1",
        pass_output_dir=pass1_dir,
        shared_memory_dir=shared_memory_dir,
        task=args.task,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        n_iterations=args.n_iterations,
        optimizer_budget=args.optimizer_budget,
        routing_policy=args.routing_policy,
        use_engram_memory=bool(args.engram_memory),
        genrm_init_samples=args.genrm_init_samples,
        start_server=bool(args.start_server),
        start_genrm=bool(args.start_genrm),
        keep_servers_running=bool(args.keep_servers_between_passes),
        extra_args=list(args.extra_arg or []),
        timeout_seconds=float(args.timeout_seconds_per_pass)
        if args.timeout_seconds_per_pass and args.timeout_seconds_per_pass > 0
        else None,
    )

    pass2: Optional[Dict[str, Any]] = None
    if pass1.get("status") == "ok":
        pass2 = _run_pass(
            run_script=run_script,
            pass_name="pass2",
            pass_output_dir=pass2_dir,
            shared_memory_dir=shared_memory_dir,
            task=args.task,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            n_iterations=args.n_iterations,
            optimizer_budget=args.optimizer_budget,
            routing_policy=args.routing_policy,
            use_engram_memory=bool(args.engram_memory),
            genrm_init_samples=args.genrm_init_samples,
            start_server=False,
            start_genrm=False,
            keep_servers_running=False,
            extra_args=list(args.extra_arg or []),
            timeout_seconds=float(args.timeout_seconds_per_pass)
            if args.timeout_seconds_per_pass and args.timeout_seconds_per_pass > 0
            else None,
        )

    pass1_cm = pass1.get("conditional_memory", {}) if isinstance(pass1, dict) else {}
    pass2_cm = (
        pass2.get("conditional_memory", {})
        if isinstance(pass2, dict)
        else {}
    )
    pass1_l1 = _to_int(pass1_cm.get("l1_hits"))
    pass1_l2 = _to_int(pass1_cm.get("l2_hits"))
    pass2_l1 = _to_int(pass2_cm.get("l1_hits"))
    pass2_l2 = _to_int(pass2_cm.get("l2_hits"))
    pass1_total_hits = (
        (pass1_l1 or 0) + (pass1_l2 or 0)
        if pass1_l1 is not None and pass1_l2 is not None
        else None
    )
    pass2_total_hits = (
        (pass2_l1 or 0) + (pass2_l2 or 0)
        if pass2_l1 is not None and pass2_l2 is not None
        else None
    )

    deltas = {
        "hit_rate_delta": _delta(_to_float(pass1_cm.get("hit_rate")), _to_float(pass2_cm.get("hit_rate"))),
        "l1_hits_delta": _delta(_to_float(pass1_l1), _to_float(pass2_l1)),
        "l2_hits_delta": _delta(_to_float(pass1_l2), _to_float(pass2_l2)),
        "total_hits_delta": _delta(_to_float(pass1_total_hits), _to_float(pass2_total_hits)),
        "writes_delta": _delta(_to_float(pass1_cm.get("writes")), _to_float(pass2_cm.get("writes"))),
        "misses_delta": _delta(_to_float(pass1_cm.get("misses")), _to_float(pass2_cm.get("misses"))),
    }
    checks = {
        "pass2_has_hits": bool(pass2_total_hits is not None and pass2_total_hits > 0),
        "pass2_hit_rate_nondecreasing": bool(
            _to_float(pass1_cm.get("hit_rate")) is not None
            and _to_float(pass2_cm.get("hit_rate")) is not None
            and float(_to_float(pass2_cm.get("hit_rate")) or 0.0)
            >= float(_to_float(pass1_cm.get("hit_rate")) or 0.0)
        ),
    }

    completed_at = _utc_now_iso()
    payload: Dict[str, Any] = {
        "created_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": float(time.perf_counter() - t0),
        "run_script": str(run_script),
        "summary": {
            "pass1": pass1,
            "pass2": pass2,
            "deltas": deltas,
            "checks": checks,
            "shared_memory_dir": str(shared_memory_dir),
        },
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved probe JSON: {json_out}")
    if pass2 is None:
        print("Probe status: failed (pass1 did not succeed, pass2 not executed)")
        return 2
    if pass1.get("status") != "ok" or pass2.get("status") != "ok":
        print(f"Probe status: failed (pass1={pass1.get('status')} pass2={pass2.get('status')})")
        return 2
    if args.require_pass2_hits and not checks["pass2_has_hits"]:
        print("Probe status: failed (pass2 has no cache hits)")
        return 3
    if args.require_hit_rate_nondecreasing and not checks["pass2_hit_rate_nondecreasing"]:
        print("Probe status: failed (pass2 hit rate decreased)")
        return 4
    print("Probe status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
