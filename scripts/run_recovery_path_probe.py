#!/usr/bin/env python3
"""Run a pipeline with deliberate server failure injection to measure recovery overhead."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_BATCH_RECOVER_ATTEMPT_RE = re.compile(r"Attempting batch-client server recovery")
_BATCH_RECOVER_SUCCESS_RE = re.compile(r"Batch-client server recovery succeeded")
_BATCH_RECOVER_FAILURE_RE = re.compile(r"Batch-client server recovery (?:reported failure|callback failed)")
_ORCH_RECOVER_RE = re.compile(r"Recovering\s+(\S+)\s+server on port\s+(\d+)")
_TRANSITION_RE = re.compile(r"Transitioned to (\S+) mode in ([0-9.]+)s")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_pids_listening_on_port(port: int) -> List[int]:
    # Prefer lsof when available.
    try:
        proc = subprocess.run(
            ["lsof", "-t", "-i", f"TCP:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            out = []
            for part in proc.stdout.splitlines():
                p = part.strip()
                if p.isdigit():
                    out.append(int(p))
            if out:
                return sorted(set(out))
    except Exception:
        pass

    # Fallback: parse ss output.
    try:
        proc = subprocess.run(
            ["ss", "-ltnp"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        needle = f":{int(port)}"
        out: List[int] = []
        for line in proc.stdout.splitlines():
            if needle not in line:
                continue
            match = re.search(r"pid=(\d+)", line)
            if match:
                out.append(int(match.group(1)))
        return sorted(set(out))
    except Exception:
        return []


def _build_command(
    *,
    run_script: Path,
    run_dir: Path,
    task: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    n_iterations: int,
    optimizer_budget: str,
    routing_policy: str,
    genrm_init_samples: int,
    max_metric_calls: Optional[int],
    start_server: bool,
    start_genrm: bool,
    engram_memory: bool,
    extra_args: List[str],
) -> List[str]:
    cmd: List[str] = [
        str(run_script),
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
        "--routing-policy",
        str(routing_policy),
        "--genrm-init-samples",
        str(int(genrm_init_samples)),
        "--output-dir",
        str(run_dir),
    ]
    if max_metric_calls is not None:
        cmd.extend(["--max-metric-calls", str(int(max_metric_calls))])
    if start_server:
        cmd.append("--start-server")
    if start_genrm:
        cmd.append("--start-genrm")
    if engram_memory:
        cmd.append("--engram-memory")
    cmd.extend(extra_args)
    return cmd


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_recovery_lines(lines: List[str], marker: str) -> Dict[str, Any]:
    after_marker = False
    batch_attempts_after = 0
    batch_success_after = 0
    batch_failure_after = 0
    orch_calls_after = 0
    orch_ports_after: List[int] = []
    transitions_after: List[float] = []

    batch_attempts_total = 0
    batch_success_total = 0
    batch_failure_total = 0

    for line in lines:
        if marker in line:
            after_marker = True
            continue

        if _BATCH_RECOVER_ATTEMPT_RE.search(line):
            batch_attempts_total += 1
            if after_marker:
                batch_attempts_after += 1
        if _BATCH_RECOVER_SUCCESS_RE.search(line):
            batch_success_total += 1
            if after_marker:
                batch_success_after += 1
        if _BATCH_RECOVER_FAILURE_RE.search(line):
            batch_failure_total += 1
            if after_marker:
                batch_failure_after += 1

        m = _ORCH_RECOVER_RE.search(line)
        if m and after_marker:
            orch_calls_after += 1
            try:
                orch_ports_after.append(int(m.group(2)))
            except (TypeError, ValueError):
                pass

        m = _TRANSITION_RE.search(line)
        if m and after_marker:
            sec = _to_float(m.group(2))
            if sec is not None:
                transitions_after.append(sec)

    return {
        "batch_attempts_total": int(batch_attempts_total),
        "batch_successes_total": int(batch_success_total),
        "batch_failures_total": int(batch_failure_total),
        "batch_attempts_after_injection": int(batch_attempts_after),
        "batch_successes_after_injection": int(batch_success_after),
        "batch_failures_after_injection": int(batch_failure_after),
        "orchestrator_recover_calls_after_injection": int(orch_calls_after),
        "orchestrator_recovered_ports_after_injection": sorted(set(orch_ports_after)),
        "transition_count_after_injection": len(transitions_after),
        "transition_mean_seconds_after_injection": (
            float(sum(transitions_after) / len(transitions_after)) if transitions_after else None
        ),
        "transition_max_seconds_after_injection": max(transitions_after) if transitions_after else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject a task-port failure and measure recovery behavior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-script", type=Path, default=Path("scripts/run_training_pipeline.sh"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)

    parser.add_argument("--task", type=str, default="manifesto_rile")
    parser.add_argument("--train-samples", type=int, default=6)
    parser.add_argument("--val-samples", type=int, default=2)
    parser.add_argument("--test-samples", type=int, default=2)
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument("--optimizer-budget", type=str, default="light")
    parser.add_argument("--routing-policy", type=str, default="affinity_load_aware")
    parser.add_argument("--genrm-init-samples", type=int, default=2)
    parser.add_argument("--max-metric-calls", type=int, default=10)

    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--start-genrm", action="store_true")
    parser.add_argument("--engram-memory", action="store_true")

    parser.add_argument("--kill-port", type=int, default=8002)
    parser.add_argument(
        "--inject-after-pattern",
        type=str,
        default="PHASE 1: Processing Documents",
        help="Inject failure once this pattern appears in probe log.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg passed through to run_training_pipeline.sh.",
    )
    parser.add_argument("--require-injection-success", action="store_true")
    parser.add_argument("--require-recovery-success", action="store_true")
    args = parser.parse_args()

    run_script = args.run_script
    if not run_script.is_absolute():
        run_script = (PROJECT_ROOT / run_script).resolve()
    if not run_script.exists():
        raise FileNotFoundError(f"Run script not found: {run_script}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / "pipeline_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    json_out = args.json_out
    if json_out is None:
        json_out = output_dir / "recovery_probe.json"
    elif not json_out.is_absolute():
        json_out = (PROJECT_ROOT / json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(
        run_script=run_script,
        run_dir=run_dir,
        task=args.task,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        n_iterations=args.n_iterations,
        optimizer_budget=args.optimizer_budget,
        routing_policy=args.routing_policy,
        genrm_init_samples=args.genrm_init_samples,
        max_metric_calls=args.max_metric_calls,
        start_server=bool(args.start_server),
        start_genrm=bool(args.start_genrm),
        engram_memory=bool(args.engram_memory),
        extra_args=list(args.extra_arg or []),
    )

    log_path = output_dir / "probe_command.log"
    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    marker = "[PROBE] FAILURE_INJECTED"
    pattern_seen = False
    injected = False
    injected_at: Optional[str] = None
    killed_pids: List[int] = []
    injection_errors: List[str] = []

    returncode = 2
    timeout_hit = False
    pipeline_success = False

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.monotonic() + max(0.0, float(args.timeout_seconds))
        while proc.poll() is None:
            if time.monotonic() > deadline:
                timeout_hit = True
                try:
                    proc.terminate()
                except Exception:
                    pass
                time.sleep(2.0)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break

            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""

            if (not pattern_seen) and args.inject_after_pattern in text:
                pattern_seen = True

            if pattern_seen and (not injected):
                candidate_pids = _find_pids_listening_on_port(int(args.kill_port))
                candidate_pids = [
                    int(pid)
                    for pid in candidate_pids
                    if int(pid) not in {int(proc.pid), int(os.getpid())}
                ]
                if candidate_pids:
                    target_pid = int(candidate_pids[0])
                    try:
                        os.kill(target_pid, signal.SIGKILL)
                        injected = True
                        injected_at = _utc_now_iso()
                        killed_pids.append(target_pid)
                        handle.write(
                            f"\n{marker} port={int(args.kill_port)} pid={target_pid} at={injected_at}\n"
                        )
                        handle.flush()
                    except Exception as exc:
                        injection_errors.append(f"kill failed for pid {target_pid}: {exc}")

            time.sleep(max(0.2, float(args.poll_interval_seconds)))

        if proc.poll() is None:
            returncode = 124
        else:
            returncode = int(proc.returncode)

    duration_seconds = float(time.perf_counter() - t0)

    final_stats_path = run_dir / "final_stats.json"
    final_stats = _load_json(final_stats_path)
    if isinstance(final_stats, dict):
        pipeline_success = bool(final_stats.get("success"))

    lines: List[str] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        lines = []
    recovery = _parse_recovery_lines(lines, marker=marker)

    test_block = final_stats.get("test", {}) if isinstance(final_stats, dict) else {}
    pred_dist = test_block.get("prediction_distribution", {}) if isinstance(test_block, dict) else {}

    recovery_observed = bool(
        int(recovery.get("batch_attempts_after_injection", 0) or 0) > 0
        or int(recovery.get("orchestrator_recover_calls_after_injection", 0) or 0) > 0
    )
    recovery_succeeded = bool(
        int(recovery.get("batch_successes_after_injection", 0) or 0) > 0
        or (
            int(recovery.get("orchestrator_recover_calls_after_injection", 0) or 0) > 0
            and bool(pipeline_success)
        )
    )

    payload: Dict[str, Any] = {
        "created_at": started_at,
        "completed_at": _utc_now_iso(),
        "duration_seconds": duration_seconds,
        "summary": {
            "status": "ok" if (returncode == 0 and pipeline_success) else "failed",
            "command": cmd,
            "returncode": int(returncode),
            "timeout": bool(timeout_hit),
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "final_stats_path": str(final_stats_path),
            "injection": {
                "requested": True,
                "pattern": str(args.inject_after_pattern),
                "pattern_seen": bool(pattern_seen),
                "kill_port": int(args.kill_port),
                "succeeded": bool(injected),
                "succeeded_numeric": 1 if injected else 0,
                "injected_at": injected_at,
                "killed_pids": killed_pids,
                "errors": injection_errors,
            },
            "recovery": recovery,
            "checks": {
                "injection_success": bool(injected),
                "recovery_observed": bool(recovery_observed),
                "recovery_succeeded": bool(recovery_succeeded),
                "pipeline_success": bool(pipeline_success),
                "injection_success_numeric": 1 if injected else 0,
                "recovery_observed_numeric": 1 if recovery_observed else 0,
                "recovery_succeeded_numeric": 1 if recovery_succeeded else 0,
                "pipeline_success_numeric": 1 if pipeline_success else 0,
            },
            "pipeline_metrics": {
                "test_mae": _to_float(test_block.get("mae")) if isinstance(test_block, dict) else None,
                "test_pearson_r": _to_float(test_block.get("pearson_r")) if isinstance(test_block, dict) else None,
                "test_frac_neutral": _to_float(pred_dist.get("frac_neutral"))
                if isinstance(pred_dist, dict)
                else None,
            },
        },
    }

    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved recovery probe JSON: {json_out}")

    if returncode != 0:
        print(f"Probe status: failed (command returncode={returncode})")
        return 2
    if args.require_injection_success and not injected:
        print("Probe status: failed (failure injection did not succeed)")
        return 3
    if args.require_recovery_success and not recovery_succeeded:
        print("Probe status: failed (recovery not observed as successful)")
        return 4
    if not pipeline_success:
        print("Probe status: failed (pipeline reported success=false)")
        return 5

    print("Probe status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
