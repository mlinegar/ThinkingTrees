#!/usr/bin/env python3
"""Spawn a truly detached process (double-fork) and return immediately.

Why this exists:
- In some execution harnesses (including Codex CLI tool calls), background
  processes started via `&`/`nohup` are terminated when the tool call ends.
- A classic double-fork + `setsid()` detaches the grandchild so it is reparented
  to PID 1 and survives after the caller exits.

Typical usage:
  venv/bin/python scripts/spawn_detached_cmd.py \
    --pid-file logs/my_job.pid \
    --cwd /path/to/repo \
    -- bash scripts/run_cpu_megasweep.sh --cmds ... --jobs 24 --log ...
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spawn a detached background process (double-fork).")
    p.add_argument("--pid-file", type=str, required=True, help="Where to write the detached PID.")
    p.add_argument("--cwd", type=str, default=".", help="Working directory for the detached process.")
    p.add_argument("--stdin", type=str, default=os.devnull)
    p.add_argument("--stdout", type=str, default=os.devnull)
    p.add_argument("--stderr", type=str, default=os.devnull)
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to exec. Prefix with `--` to separate from this script's flags.",
    )
    return p.parse_args()


def _strip_remainder(cmd: List[str]) -> List[str]:
    if not cmd:
        return []
    if cmd[0] == "--":
        return cmd[1:]
    return cmd


def main() -> int:
    args = _parse_args()
    cmd = _strip_remainder(list(args.cmd))
    if not cmd:
        raise SystemExit("Missing command. Example: ... spawn_detached_cmd.py --pid-file x.pid -- <cmd> ...")

    pid_path = Path(args.pid_file)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass

    # First fork: parent returns quickly.
    pid = os.fork()
    if pid > 0:
        # Wait briefly for the grandchild to publish its PID.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                text = pid_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                text = ""
            if text:
                print(text, flush=True)
                return 0
            time.sleep(0.05)
        print(str(pid), flush=True)
        return 0

    # Child: detach from controlling terminal and create a new session.
    os.setsid()

    # Second fork: ensure the detached process cannot reacquire a controlling terminal.
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Grandchild: run the requested command.
    os.chdir(str(args.cwd))

    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    # Redirect stdio.
    with open(args.stdin, "rb", buffering=0) as f_in:
        os.dup2(f_in.fileno(), 0)
    with open(args.stdout, "ab", buffering=0) as f_out:
        os.dup2(f_out.fileno(), 1)
    with open(args.stderr, "ab", buffering=0) as f_err:
        os.dup2(f_err.fileno(), 2)

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())

