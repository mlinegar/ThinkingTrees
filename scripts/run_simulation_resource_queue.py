#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.resource_queue import detect_gpu_tokens, load_jobs, run_resource_queue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulation command/manifests with device-aware scheduling.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest JSONL file. Repeat as needed.")
    parser.add_argument("--cmd-file", action="append", default=[], help="Command file. Repeat as needed.")
    parser.add_argument("--cpu-workers", type=int, default=0, help="CPU worker count. 0 defaults to nproc().")
    parser.add_argument(
        "--gpu-tokens",
        type=str,
        default="auto",
        help="GPU token spec: auto, none, MIG UUIDs, GPU UUIDs, or GPU indices.",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_paths = [Path(x) for x in (args.manifest or []) if str(x).strip()]
    cmd_files = [Path(x) for x in (args.cmd_file or []) if str(x).strip()]
    if not manifest_paths and not cmd_files:
        raise SystemExit("supply at least one --manifest or --cmd-file")

    cpu_workers = int(args.cpu_workers) if int(args.cpu_workers) > 0 else int(os.cpu_count() or 1)
    gpu_tokens = detect_gpu_tokens(str(args.gpu_tokens))
    jobs = load_jobs(manifest_paths=manifest_paths, cmd_files=cmd_files)
    summary = run_resource_queue(
        jobs,
        cpu_workers=cpu_workers,
        gpu_tokens=gpu_tokens,
        log_dir=Path(args.log_dir),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if int(summary.get("n_fail", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
