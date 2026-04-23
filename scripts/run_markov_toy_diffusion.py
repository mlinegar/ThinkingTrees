#!/usr/bin/env python3
"""Run the theorem-valid fixed-binary Markov toy diffusion track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.markov_toy import run_markov_toy_experiment


def _parse_states(text: str) -> List[str]:
    items = [piece.strip() for piece in text.split(",")]
    return [item for item in items if item]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-binary Markov toy diffusion track.")
    parser.add_argument(
        "--path",
        type=str,
        default="A,B",
        help="Comma-separated Markov states. Default reproduces the count-only counterexample.",
    )
    parser.add_argument("--chunk-size", type=int, default=1, help="Fixed chunk size in states.")
    parser.add_argument("--rounds", type=int, default=1, help="Round index used for the text-budget formula.")
    parser.add_argument("--eps-leaf", type=float, default=0.0, help="Leaf budget for the approximate text formula.")
    parser.add_argument("--eps-merge", type=float, default=0.0, help="Merge budget for the approximate text formula.")
    parser.add_argument("--eps-idemp", type=float, default=0.0, help="Idempotence budget for the approximate text formula.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    states = _parse_states(args.path)
    if not states:
        raise SystemExit("Provide at least one state in --path.")

    payload = run_markov_toy_experiment(
        states,
        chunk_size=args.chunk_size,
        rounds=args.rounds,
        eps_leaf=args.eps_leaf,
        eps_merge=args.eps_merge,
        eps_idemp=args.eps_idemp,
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
