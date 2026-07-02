#!/usr/bin/env python3
"""Wait for detached long_job launchers, then run a command.

This is intentionally small glue for queued experiment follow-ups. It avoids
starting a GPU-heavy job on a device that is already occupied by an earlier
``scripts/long_job.py launch`` process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def _job_is_running(job_root: Path) -> bool:
    manifest_path = job_root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    pid_file = Path(str(manifest.get("pid_file") or job_root / "job.pid"))
    if pid_file.exists():
        try:
            return _pid_is_alive(int(pid_file.read_text().strip()))
        except Exception:
            return False
    try:
        return _pid_is_alive(int(manifest.get("pid") or 0))
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait-job-root",
        action="append",
        default=[],
        help="A long_job.py job root to wait for. May be passed multiple times.",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    wait_roots = [Path(raw) for raw in args.wait_job_root]
    while True:
        running = [root for root in wait_roots if _job_is_running(root)]
        if not running:
            break
        joined = ", ".join(str(root) for root in running)
        _log(f"waiting for {len(running)} job(s): {joined}")
        time.sleep(max(1.0, float(args.poll_seconds)))

    _log("starting queued command: " + " ".join(command))
    return int(subprocess.run(command).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
