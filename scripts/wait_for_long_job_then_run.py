#!/usr/bin/env python3
"""Wait for one long_job launcher to finish, then run a follow-on command."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_JOB_SCRIPT = REPO_ROOT / "scripts" / "long_job.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strip_separator(command: Sequence[str]) -> list[str]:
    items = list(command)
    if items and items[0] == "--":
        return items[1:]
    return items


def _status(job_root: str, python_bin: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            python_bin,
            str(LONG_JOB_SCRIPT),
            "status",
            "--job-root",
            job_root,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"long_job status failed with code {result.returncode}: {message}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("long_job status returned non-object JSON")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a scripts/long_job.py job to finish, then run a command."
    )
    parser.add_argument("--job-root", required=True, help="Job root to wait on.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--status-python",
        default=sys.executable,
        help="Python executable used to call scripts/long_job.py status.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Optional wait timeout. Zero means no timeout.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.command = _strip_separator(args.command)
    if not args.command:
        parser.error("provide a command after --")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    print(
        f"{_utc_now()} waiting for {args.job_root}; "
        f"then running: {shlex.join(args.command)}",
        flush=True,
    )

    while True:
        status = _status(args.job_root, args.status_python)
        systemd_state = status.get("systemd_state") or {}
        state = systemd_state.get("active_state") or "unknown"
        sub_state = systemd_state.get("sub_state") or "unknown"
        print(
            f"{_utc_now()} upstream running={bool(status.get('running'))} "
            f"state={state}/{sub_state} pid={status.get('pid')}",
            flush=True,
        )
        if not status.get("running"):
            break
        if args.timeout_seconds and time.monotonic() - started > args.timeout_seconds:
            print(f"{_utc_now()} timed out waiting for {args.job_root}", file=sys.stderr)
            return 124
        time.sleep(args.poll_seconds)

    print(f"{_utc_now()} upstream finished; launching follow-on command", flush=True)
    return subprocess.call(args.command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
