#!/usr/bin/env python3
"""Audit the general C-TreePO run target registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.run_registry import audit_target_records, iter_run_targets  # noqa: E402


def _markdown(records: list[object]) -> str:
    lines = [
        "# C-TreePO Run Target Registry",
        "",
        "| Target | Domain | Role | Backend | Status | Input contract | Audit policy | Publication-ready |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| {target} | {domain} | {role} | {backend} | {status} | {contract} | {policy} | {pub} |".format(
                target=record.target,
                domain=record.domain,
                role=record.role,
                backend=record.backend,
                status=record.status,
                contract=record.expected_input_contract,
                policy=record.audit_policy,
                pub=str(bool(record.publication_ready)).lower(),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", default=[], help="Filter by suite tag.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Exit nonzero on audit errors.")
    args = parser.parse_args(argv)

    records = iter_run_targets(suites=args.suite)
    errors = audit_target_records(records)
    payload = {
        "schema_version": 1,
        "records": [record.to_dict() for record in records],
        "errors": errors,
        "ok": not errors,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(records), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2 if args.check else 0
    print(f"Run target registry audit passed: checked={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
