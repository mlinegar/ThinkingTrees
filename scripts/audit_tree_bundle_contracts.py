#!/usr/bin/env python3
"""Audit TreeBundle v1 metadata in completed C-TreePO artifacts."""

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
    REDUCER_CONTRACT_BOTTOM_UP,
    STATE_CONTRACT_EXTERNAL_PASSTHROUGH,
    TREE_BUNDLE_SCHEMA_VERSION,
    normalize_tree_bundle_manifest,
    validate_tree_bundle_manifest,
)


def _iter_candidate_files(paths: Iterable[Path]) -> Iterable[Path]:
    names = {
        "fit_result.json",
        "grid_summary.json",
        "manifest.json",
        "paper_bundle_manifest.json",
        "pipeline_summary.json",
        "publication_manifest.json",
        "report_version_manifest.json",
        "run_manifest.json",
        "summary.json",
    }
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
            continue
        if not path.exists():
            yield path
            continue
        for candidate in path.rglob("*.json"):
            if candidate.name in names:
                yield candidate
        for candidate in path.rglob("labeled_trees.jsonl"):
            yield candidate


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_jsonl(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                parsed = json.loads(line)
                if isinstance(parsed, Mapping):
                    metadata = parsed.get("metadata")
                    if isinstance(metadata, Mapping):
                        return metadata
                    return parsed
    return {}


def _payload_for_file(path: Path) -> Mapping[str, Any]:
    if path.name.endswith(".jsonl"):
        return _read_first_jsonl(path)
    payload = _read_json(path)
    if path.name == "manifest.json" and isinstance(payload.get("config"), Mapping):
        return payload["config"]
    return payload


def _has_bundle_signal(payload: Mapping[str, Any]) -> bool:
    schema = str(payload.get("schema_version") or "")
    if isinstance(payload.get("tree_bundle"), Mapping):
        return True
    return any(
        key in payload
        for key in (
            "tree_bundle_manifest",
            "source_kind",
            "tree_bundle_kind",
            "tree_text_source",
            "tree_bundle_contract",
        )
    ) or schema == TREE_BUNDLE_SCHEMA_VERSION


def _iter_bundle_payloads(payload: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if _has_bundle_signal(payload):
            yield payload
        config = payload.get("config")
        if isinstance(config, Mapping) and _has_bundle_signal(config):
            yield config
        for key in (
            "tree_bundle_contract",
            "tree_bundle_manifest",
            "tree_bundle",
            "bundle_contract",
        ):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and nested is not payload:
                if _has_bundle_signal(nested):
                    yield nested
        for value in payload.values():
            if isinstance(value, (Mapping, list, tuple)):
                yield from _iter_bundle_payloads(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            if isinstance(item, (Mapping, list, tuple)):
                yield from _iter_bundle_payloads(item)


def _dedupe_payloads(payloads: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    out: list[Mapping[str, Any]] = []
    for payload in payloads:
        try:
            key = json.dumps(payload, sort_keys=True, default=str)
        except TypeError:
            key = str(id(payload))
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def _lineage_errors(path: Path, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    normalized = normalize_tree_bundle_manifest(payload)
    if normalized.get("source_kind") == "external_state":
        producer = str(normalized.get("external_state_producer") or "").strip()
        if not producer:
            errors.append(f"{path}: external_state source has no external_state_producer")
        if normalized.get("state_contract") != STATE_CONTRACT_EXTERNAL_PASSTHROUGH:
            errors.append(
                f"{path}: external_state source must use state_contract='external_passthrough'"
            )
    for key in ("g_init", "dspy_g_init_mode"):
        value = str(payload.get(key) or "").strip().lower()
        if value == "teacher_passthrough" and normalized.get("source_kind") != "external_state":
            errors.append(
                f"{path}: {key}=teacher_passthrough requires source_kind=external_state"
            )
    if normalized.get("reducer_contract") != REDUCER_CONTRACT_BOTTOM_UP:
        errors.append(
            f"{path}: reducer_contract={normalized.get('reducer_contract')!r}, expected bottom_up"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-domain", default=None)
    parser.add_argument("--expected-source-kind", default=None)
    parser.add_argument("--expected-leaf-unit", default=None)
    parser.add_argument("--expected-dimension", default=None)
    parser.add_argument("--expected-target-scale", default=None)
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Normalize deprecated aliases instead of failing on missing schema_version.",
    )
    parser.add_argument(
        "--require-tree-bundle",
        action="store_true",
        help="Fail if no TreeBundle metadata is discovered in the supplied paths.",
    )
    parser.add_argument(
        "--allow-external-state",
        action="store_true",
        help="Permit source_kind=external_state when it is explicitly labeled.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    checked = 0
    for path in _iter_candidate_files(args.paths):
        if not path.exists():
            errors.append(f"{path}: missing path")
            continue
        try:
            payload = _payload_for_file(path)
        except Exception as exc:  # pragma: no cover - defensive CLI surface
            errors.append(f"{path}: failed to parse: {exc}")
            continue
        if not isinstance(payload, Mapping):
            continue
        bundle_payloads = _dedupe_payloads(_iter_bundle_payloads(payload))
        for bundle_payload in bundle_payloads:
            checked += 1
            normalized = normalize_tree_bundle_manifest(bundle_payload)
            has_v1 = (
                str(bundle_payload.get("schema_version") or "") == TREE_BUNDLE_SCHEMA_VERSION
                or isinstance(bundle_payload.get("tree_bundle_manifest"), Mapping)
                or isinstance(bundle_payload.get("tree_bundle_contract"), Mapping)
            )
            if not args.allow_legacy and not has_v1:
                errors.append(f"{path}: missing TreeBundle v1 schema_version")
            if (
                normalized.get("source_kind") == "external_state"
                and not bool(args.allow_external_state)
            ):
                errors.append(
                    f"{path}: external_state TreeBundle requires --allow-external-state"
                )
            errors.extend(_lineage_errors(path, bundle_payload))
            try:
                validate_tree_bundle_manifest(
                    normalized,
                    expected_domain=args.expected_domain,
                    expected_leaf_unit=args.expected_leaf_unit,
                    expected_source_kind=args.expected_source_kind,
                    expected_dimension=args.expected_dimension,
                    expected_target_scale=args.expected_target_scale,
                )
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    if checked == 0 and bool(args.require_tree_bundle):
        errors.append("no TreeBundle metadata found")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"TreeBundle audit passed: checked={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
