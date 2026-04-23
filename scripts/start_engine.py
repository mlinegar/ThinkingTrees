#!/usr/bin/env python3
"""Generic engine launcher that delegates to the engine-specific wrapper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engines import EngineRegistry, EngineType


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a configured engine wrapper.")
    parser.add_argument("--engine", required=True, help="Engine name (for example: vllm, sglang).")
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="Print the resolved engine spec and exit without launching.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the engine wrapper.")
    parsed = parser.parse_args()

    spec = EngineRegistry.resolve(EngineType.normalize(parsed.engine))
    if parsed.print_spec:
        print(spec.to_dict())
        return 0
    if not spec.launchable or not spec.launch_script:
        raise SystemExit(f"Engine '{spec.engine.value}' does not provide a launchable local wrapper.")

    passthrough = list(parsed.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    env = os.environ.copy()
    env["TT_START_ENGINE_DIRECT"] = "1"
    cmd = ["/bin/bash", spec.launch_script, *passthrough]
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    raise SystemExit(main())
