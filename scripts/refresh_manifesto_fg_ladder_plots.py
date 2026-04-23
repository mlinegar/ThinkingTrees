#!/usr/bin/env python3
"""Refresh manifesto f/g ladder plots until a watched long job completes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_JOB = REPO_ROOT / "scripts" / "long_job.py"
PLOTTER = REPO_ROOT / "scripts" / "plot_manifesto_fg_ladder_grid.py"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh manifesto ladder plots while another long job is running."
    )
    parser.add_argument("--watch-job-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--family", default="dspy")
    parser.add_argument("--figure-title", default="Manifesto economic f/g ladder")
    parser.add_argument("--figure-subtitle", default="")
    parser.add_argument("--external-pearson-min", type=float, default=0.75)
    parser.add_argument("--external-pearson-max", type=float, default=None)
    parser.add_argument("--no-partial", action="store_true")
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument("--stage-label", action="append", default=[])
    return parser.parse_args(argv)


def _run_plot(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PLOTTER),
        "--output-dir",
        str(args.output_dir),
        "--eval-split",
        str(args.eval_split),
        "--family",
        str(args.family),
        "--figure-title",
        str(args.figure_title),
        "--figure-subtitle",
        str(args.figure_subtitle),
    ]
    if args.external_pearson_min is not None:
        cmd.extend(["--external-pearson-min", str(args.external_pearson_min)])
    if args.external_pearson_max is not None:
        cmd.extend(["--external-pearson-max", str(args.external_pearson_max)])
    if args.no_partial:
        cmd.append("--no-partial")
    for root in args.input_root:
        cmd.extend(["--input-root", str(root)])
    for label in args.stage_label:
        cmd.extend(["--stage-label", str(label)])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _job_running(job_root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(LONG_JOB), "status", "--job-root", str(job_root)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return bool(payload.get("running"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    while True:
        _run_plot(args)
        if not _job_running(args.watch_job_root):
            break
        time.sleep(max(1, int(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
