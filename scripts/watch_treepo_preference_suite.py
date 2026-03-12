#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh the exact utility transport report while queue jobs are running.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-markdown", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--gpu-pid", type=int, default=None)
    p.add_argument("--cpu-pid", type=int, default=None)
    p.add_argument("--interval-seconds", type=int, default=900)
    p.add_argument("--max-iterations", type=int, default=0, help="0 means run until the watched PIDs exit.")
    return p.parse_args()


def _run_report(output_root: Path, output_markdown: Path, output_pdf: Path | None) -> None:
    cmd = [
        sys.executable,
        "scripts/report_treepo_preference_suite.py",
        "--output-root",
        str(output_root),
        "--output-markdown",
        str(output_markdown),
    ]
    if output_pdf is not None:
        cmd.extend(["--output-pdf", str(output_pdf)])
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    args = parse_args()
    output_markdown = args.output_markdown or (args.output_root / "utility_transport_report.md")
    watched_pids = [pid for pid in (args.gpu_pid, args.cpu_pid) if pid is not None]
    iteration = 0
    while True:
        iteration += 1
        try:
            _run_report(args.output_root, output_markdown, args.output_pdf)
            print(f"refresh_ok | iteration={iteration} | output_root={args.output_root}", flush=True)
        except subprocess.CalledProcessError as exc:
            print(f"refresh_fail | iteration={iteration} | returncode={exc.returncode}", flush=True)
        if int(args.max_iterations) > 0 and iteration >= int(args.max_iterations):
            return 0
        if watched_pids and not any(_pid_alive(pid) for pid in watched_pids):
            try:
                _run_report(args.output_root, output_markdown, args.output_pdf)
                print(f"final_refresh_ok | output_root={args.output_root}", flush=True)
            except subprocess.CalledProcessError as exc:
                print(f"final_refresh_fail | returncode={exc.returncode}", flush=True)
            return 0
        time.sleep(max(30, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
