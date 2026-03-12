#!/usr/bin/env python3
"""Run an xargs-style command file with a fixed worker pool."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, wait
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess
import time
from typing import Dict, List


def _load_cmds(path: Path) -> List[str]:
    cmds: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item:
            cmds.append(item)
    return cmds


def _existing_outputs(cmd: str) -> tuple[Path | None, Path | None]:
    tokens = shlex.split(cmd)
    json_path: Path | None = None
    csv_path: Path | None = None
    for idx, token in enumerate(tokens[:-1]):
        if token == "--json-summary":
            json_path = Path(tokens[idx + 1])
        elif token == "--csv-summary":
            csv_path = Path(tokens[idx + 1])
    return json_path, csv_path


def _run_one(idx: int, cmd: str, log_dir: Path) -> Dict[str, object]:
    log_path = log_dir / f"run_{idx:04d}.log"
    json_path, csv_path = _existing_outputs(cmd)
    declared_outputs = [path for path in (json_path, csv_path) if path is not None]
    if declared_outputs and all(path.exists() for path in declared_outputs):
        return {
            "idx": idx,
            "ok": True,
            "returncode": 0,
            "seconds": 0.0,
            "log": str(log_path),
            "skipped_existing": True,
        }
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"cmd={cmd}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
        )
    return {
        "idx": idx,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
        "log": str(log_path),
        "skipped_existing": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command queue with a fixed worker pool.")
    parser.add_argument("--cmd-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cmds = _load_cmds(args.cmd_file)
    if not cmds:
        print("no commands to run")
        return 0
    args.log_dir.mkdir(parents=True, exist_ok=True)

    total = len(cmds)
    next_idx = 0
    inflight: Dict[Future, int] = {}
    with ThreadPoolExecutor(max_workers=int(max(1, args.workers))) as pool:
        while next_idx < total or inflight:
            while next_idx < total and len(inflight) < int(max(1, args.workers)):
                future = pool.submit(_run_one, next_idx, cmds[next_idx], args.log_dir)
                inflight[future] = next_idx
                next_idx += 1

            if not inflight:
                break

            done, _ = wait(inflight.keys(), timeout=float(args.poll_seconds), return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                inflight.pop(future, None)
                result = future.result()
                if result.get("skipped_existing"):
                    prefix = "skip"
                else:
                    prefix = "ok" if result["ok"] else "fail"
                print(
                    f"[{int(result['idx']) + 1}/{total}] idx={result['idx']} {prefix} "
                    f"{result['seconds']}s log={result['log']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
