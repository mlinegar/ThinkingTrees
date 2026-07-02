#!/usr/bin/env python3
"""Audit C-TreePO RunManifest v1 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.contracts import (  # noqa: E402
    RUN_MANIFEST_SCHEMA_VERSION,
    normalize_run_manifest,
    validate_run_manifest,
)


RUN_MANIFEST_NAMES = {
    "grid_summary.json",
    "manifest.json",
    "paper_bundle_manifest.json",
    "pipeline_summary.json",
    "publication_manifest.json",
    "run_manifest.json",
    "ctreepo_run_manifest.json",
}


def _iter_candidate_files(paths: Iterable[Path]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
            continue
        if not path.exists():
            yield path
            continue
        for candidate in path.rglob("*.json"):
            if candidate.name in RUN_MANIFEST_NAMES:
                yield candidate


def _payload_for_file(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("RunManifest payload must be a JSON object")
    if isinstance(payload.get("run_manifest"), Mapping):
        return payload["run_manifest"]
    return payload


def _has_run_manifest_signal(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("schema_version") or "") == RUN_MANIFEST_SCHEMA_VERSION
        or isinstance(payload.get("run_manifest"), Mapping)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-domain", default=None)
    parser.add_argument("--expected-role", default=None)
    parser.add_argument("--expected-backend", default=None)
    parser.add_argument("--require-tree-bundle", action="store_true")
    parser.add_argument("--require-lineage", action="store_true")
    parser.add_argument("--require-objective", action="store_true")
    parser.add_argument("--require-publication-ready", action="store_true")
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--require-run-manifest", action="store_true")
    args = parser.parse_args(argv)

    errors: list[str] = []
    checked = 0
    for path in _iter_candidate_files(args.paths):
        if not path.exists():
            errors.append(f"{path}: missing path")
            continue
        try:
            payload = _payload_for_file(path)
        except Exception as exc:
            errors.append(f"{path}: failed to parse: {exc}")
            continue
        if not _has_run_manifest_signal(payload) and not bool(args.allow_legacy):
            errors.append(f"{path}: missing RunManifest v1 schema_version")
            continue
        checked += 1
        try:
            validate_run_manifest(
                normalize_run_manifest(payload),
                expected_domain=args.expected_domain,
                expected_role=args.expected_role,
                expected_backend=args.expected_backend,
                require_tree_bundle=bool(args.require_tree_bundle),
                require_lineage=bool(args.require_lineage),
                require_objective=bool(args.require_objective),
                require_publication_ready=bool(args.require_publication_ready),
                allow_legacy=bool(args.allow_legacy),
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if checked == 0 and bool(args.require_run_manifest):
        errors.append("no RunManifest metadata found")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"RunManifest audit passed: checked={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
