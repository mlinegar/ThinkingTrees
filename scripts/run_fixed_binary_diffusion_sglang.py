#!/usr/bin/env python3
"""Backward-compatible SGLang wrapper for the generic diffusion runner."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fixed_binary_diffusion import main as _generic_main


def main() -> int:
    sys.argv = [sys.argv[0], "--backend", "sglang", *sys.argv[1:]]
    return _generic_main()


if __name__ == "__main__":
    raise SystemExit(main())
