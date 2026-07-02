#!/usr/bin/env python3
"""Repair failed cells from the Round 4 HLL JAX overnight grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LONG_JOB = REPO / "scripts" / "long_job.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _exit_code(cell_dir: Path) -> str | None:
    path = cell_dir / "exit_code.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _wait_for_job_root(job_root: Path, *, poll_seconds: float) -> None:
    while True:
        result = subprocess.run(
            [
                str(REPO / "venv" / "bin" / "python"),
                str(LONG_JOB),
                "status",
                "--job-root",
                str(job_root),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to inspect wait job {job_root}: {result.stderr or result.stdout}"
            )
        payload = json.loads(result.stdout)
        if not bool(payload.get("running", False)):
            return
        print(f"[wait] {job_root} still running", flush=True)
        time.sleep(max(1.0, float(poll_seconds)))


def _manifest_cells(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "cell_manifest.json"
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected list manifest at {path}")
    return [row for row in payload if isinstance(row, dict)]


def _selected_failed_cells(
    output_root: Path,
    *,
    shard_index: int,
    num_shards: int,
    names: set[str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, cell in enumerate(_manifest_cells(output_root)):
        name = str(cell.get("name", ""))
        if names is not None and name not in names:
            continue
        if names is None and idx % int(num_shards) != int(shard_index):
            continue
        code = _exit_code(output_root / name)
        if code is not None and code != "0":
            out.append(cell)
    return out


def _run_cell(output_root: Path, cell: dict[str, Any]) -> int:
    name = str(cell["name"])
    cell_dir = output_root / name
    command_path = cell_dir / "command.json"
    if not command_path.exists():
        raise FileNotFoundError(f"missing command.json for failed cell {name}: {command_path}")
    cmd = _load_json(command_path)
    if not isinstance(cmd, list) or not all(isinstance(item, str) for item in cmd):
        raise ValueError(f"invalid command.json for {name}: {command_path}")
    stamp = _utc_stamp()
    repair_command_path = cell_dir / f"repair_command_{stamp}.json"
    repair_log_path = cell_dir / f"repair_run_{stamp}.log"
    repair_command_path.write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    print(f"[repair-run] {name}", flush=True)
    start = time.time()
    with repair_log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start
    (cell_dir / "exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    (cell_dir / "elapsed_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    (cell_dir / "last_repair_log.txt").write_text(str(repair_log_path) + "\n", encoding="utf-8")
    print(f"[repair-done] {name} exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cell-name", action="append", default=[])
    parser.add_argument("--wait-job-root", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(str(args.output_root)).resolve()
    if args.wait_job_root is not None:
        _wait_for_job_root(
            Path(args.wait_job_root).resolve(),
            poll_seconds=float(args.poll_seconds),
        )
    names = {str(name) for name in args.cell_name} if args.cell_name else None
    cells = _selected_failed_cells(
        output_root,
        shard_index=int(args.shard_index),
        num_shards=max(1, int(args.num_shards)),
        names=names,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "selected_failed_cells": [str(cell["name"]) for cell in cells],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    failures = [_run_cell(output_root, cell) for cell in cells]
    return 1 if any(code != 0 for code in failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
