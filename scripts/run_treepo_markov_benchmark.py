#!/usr/bin/env python3
"""Run ThinkingTrees Markov data through the standalone treepo benchmark API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.treepo_bridge.markov import run_markov_benchmark  # noqa: E402
from treepo.bench.io import dump_json, load_yaml_or_json  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ThinkingTrees Markov-paper benchmark through treepo."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--print-json", action="store_true", default=False)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_yaml_or_json(Path(args.config))
    if not isinstance(payload, dict):
        raise SystemExit("--config must contain a JSON/YAML mapping")
    result = run_markov_benchmark(
        config=payload,
        json_out=Path(args.json_out),
        csv_out=Path(args.csv_out),
        print_json=bool(args.print_json),
    )
    if not bool(args.print_json):
        print(dump_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
