#!/usr/bin/env python3
"""Merge family-specific RunSpec manifests into a unified local-law meta manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl, write_manifest_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge RunSpec manifests for the unified local-law protocol.")
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Input RunSpec JSONL manifest.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Merged RunSpec JSONL output path.")
    parser.add_argument("--cmd-file", type=Path, default=None, help="Optional merged command file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged: Dict[str, RunSpec] = {}
    for manifest_path in args.manifest:
        for run in read_manifest_jsonl(manifest_path):
            merged[str(run.id)] = run

    runs: List[RunSpec] = sorted(merged.values(), key=lambda run: (str(run.family), str(run.id)))
    write_manifest_jsonl(args.output_manifest, runs)

    if args.cmd_file is not None:
        args.cmd_file.parent.mkdir(parents=True, exist_ok=True)
        args.cmd_file.write_text(
            "\n".join(str(run.command) for run in runs) + ("\n" if runs else ""),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest),
                "cmd_file": str(args.cmd_file) if args.cmd_file is not None else None,
                "n_runs": len(runs),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
