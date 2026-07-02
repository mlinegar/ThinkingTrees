#!/usr/bin/env python3
"""Compatibility audit for publication-facing targets in the general registry.

The source of truth is now ``src.ctreepo.run_registry``.  This script remains
for older automation that still calls the previous publication-entrypoint audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.run_registry import RunTargetRecord, audit_target_records, iter_run_targets  # noqa: E402


def _publication_records() -> list[RunTargetRecord]:
    return [record for record in iter_run_targets() if record.publication_ready]


def audit_records(records: list[RunTargetRecord] | None = None) -> list[str]:
    return audit_target_records(records or _publication_records())


def _markdown(records: list[RunTargetRecord]) -> str:
    lines = [
        "# Publication-Facing C-TreePO Targets",
        "",
        "| Target | Path | Domain | Role | Backend | Status | Input contract | Audit policy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| {target} | {path} | {domain} | {role} | {backend} | {status} | {contract} | {policy} |".format(
                target=record.target,
                path=record.path,
                domain=record.domain,
                role=record.role,
                backend=record.backend,
                status=record.status,
                contract=record.expected_input_contract,
                policy=record.audit_policy,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Exit nonzero on audit errors.")
    args = parser.parse_args(argv)

    records = _publication_records()
    errors = audit_records(records)
    payload = {
        "schema_version": 1,
        "source_registry": "src.ctreepo.run_registry",
        "records": [record.to_dict() for record in records],
        "errors": errors,
        "ok": not errors,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(records), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2 if args.check else 0
    print(f"Publication target audit passed: checked={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
