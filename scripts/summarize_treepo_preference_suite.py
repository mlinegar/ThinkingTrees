#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.utility_transport_expectations import build_utility_transport_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize exact utility transport suite.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = build_utility_transport_report(args.output_root)
    payload = report.to_dict()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("rows", [])
    if rows:
        fieldnames = list(rows[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote_json | {args.output_json}")
    print(f"wrote_csv | {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
