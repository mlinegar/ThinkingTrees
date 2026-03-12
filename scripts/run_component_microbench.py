#!/usr/bin/env python3
"""
Run pure-Python microbenchmarks for ThinkingTrees components.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.component_microbench import (  # noqa: E402
    available_benchmarks,
    run_selected_benchmarks,
)


def _parse_bench_arg(value: str) -> list[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return ["all"]
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ThinkingTrees component microbenchmarks.")
    parser.add_argument(
        "--bench",
        default="all",
        help=f"Comma-separated benchmark names. Available: {', '.join(available_benchmarks())}",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Iteration count for iterative microbenchmarks (prompting/memory).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for JSON results.",
    )
    args = parser.parse_args()

    bench_names = _parse_bench_arg(args.bench)
    result = run_selected_benchmarks(bench_names, iterations=max(1, int(args.iterations)))

    payload = json.dumps(result, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
        print(str(args.json_out))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
