#!/usr/bin/env python3
"""Run a command file across MIG slices with one worker per slice."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, wait
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Dict, List


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


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


def _run_one(idx: int, cmd: str, mig_uuid: str, log_dir: Path, append_cuda_device_zero: bool) -> Dict[str, object]:
    log_path = log_dir / f"run_{idx:04d}.log"
    json_path, csv_path = _existing_outputs(cmd)
    declared_outputs = [path for path in (json_path, csv_path) if path is not None]
    if declared_outputs and all(path.exists() for path in declared_outputs):
        return {
            "idx": idx,
            "ok": True,
            "returncode": 0,
            "seconds": 0.0,
            "mig": mig_uuid,
            "log": str(log_path),
            "skipped_existing": True,
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = mig_uuid
    final_cmd = cmd
    if append_cuda_device_zero and "--cuda-device" not in cmd:
        final_cmd = f"{cmd} --cuda-device 0"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mig={mig_uuid}\n")
        handle.write(f"cmd={final_cmd}\n")
        handle.flush()
        proc = subprocess.run(
            final_cmd,
            shell=True,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
        )
    return {
        "idx": idx,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
        "mig": mig_uuid,
        "log": str(log_path),
        "skipped_existing": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an xargs-style command file across MIG slices.")
    parser.add_argument("--cmd-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--mig-uuids", type=str, required=True)
    parser.add_argument("--append-cuda-device-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cmds = _load_cmds(args.cmd_file)
    migs = _parse_items(args.mig_uuids)
    if not cmds:
        print("no commands to run", file=sys.stderr)
        return 0
    if not migs:
        raise SystemExit("no MIG UUIDs supplied")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    total = len(cmds)
    next_idx = 0
    inflight: Dict[Future, str] = {}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(migs)) as pool:
        while next_idx < total or inflight:
            while next_idx < total and len(inflight) < len(migs):
                mig_uuid = migs[len(inflight)]
                future = pool.submit(
                    _run_one,
                    next_idx,
                    cmds[next_idx],
                    mig_uuid,
                    args.log_dir,
                    bool(args.append_cuda_device_zero),
                )
                inflight[future] = mig_uuid
                next_idx += 1

            if not inflight:
                break

            done, _ = wait(inflight.keys(), timeout=float(args.poll_seconds), return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                mig_uuid = inflight.pop(future)
                result = future.result()
                if result.get("skipped_existing"):
                    prefix = "skip"
                else:
                    prefix = "ok" if result["ok"] else "fail"
                print(
                    f"[{int(result['idx']) + 1}/{total}] idx={result['idx']} {prefix} "
                    f"{result['seconds']}s mig={mig_uuid} log={result['log']}",
                    flush=True,
                )
                if next_idx < total:
                    new_future = pool.submit(
                        _run_one,
                        next_idx,
                        cmds[next_idx],
                        mig_uuid,
                        args.log_dir,
                        bool(args.append_cuda_device_zero),
                    )
                    inflight[new_future] = mig_uuid
                    next_idx += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
