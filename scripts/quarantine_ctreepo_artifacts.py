#!/usr/bin/env python3
"""Classify existing C-TreePO artifacts against the TreeBundle v1 contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.contracts import (  # noqa: E402
    REDUCER_CONTRACT_BOTTOM_UP,
    RUN_MANIFEST_SCHEMA_VERSION,
    SOURCE_KIND_EXTERNAL_STATE,
    TREE_BUNDLE_SCHEMA_VERSION,
    normalize_tree_bundle_manifest,
    normalize_run_manifest,
    tree_bundle_manifest_digest,
    validate_run_manifest,
    validate_tree_bundle_manifest,
)

CLASS_VALID_RUN = "valid_run_manifest_v1"
CLASS_VALID = "valid_treebundle_v1"
CLASS_LEGACY = "legacy_migratable"
CLASS_EXTERNAL = "external_state_compat"
CLASS_MISSING = "missing_contract"
CLASS_INVALID_DIM = "invalid_state_dim"
CLASS_ROOT_SHORTCUT = "root_summary_shortcut_risk"
CLASS_UNKNOWN = "unknown"

ARTIFACT_NAMES = {
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = json.loads(line)
            if isinstance(parsed, Mapping) and isinstance(parsed.get("metadata"), Mapping):
                return parsed["metadata"]
            return parsed
    return {}


def _iter_candidate_files(roots: Iterable[Path]) -> Iterator[Path]:
    for raw in roots:
        root = Path(raw)
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name in ARTIFACT_NAMES:
                yield path
        for path in root.rglob("labeled_trees.jsonl"):
            yield path


def _has_bundle_signal(payload: Mapping[str, Any]) -> bool:
    schema = str(payload.get("schema_version") or "")
    if isinstance(payload.get("tree_bundle"), Mapping):
        return True
    return schema == TREE_BUNDLE_SCHEMA_VERSION or any(
        key in payload
        for key in (
            "tree_bundle_manifest",
            "tree_bundle_contract",
            "source_kind",
            "tree_bundle_kind",
            "tree_text_source",
        )
    )


def _has_run_signal(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("schema_version") or "") == RUN_MANIFEST_SCHEMA_VERSION
        or isinstance(payload.get("run_manifest"), Mapping)
    )


def _iter_payloads(payload: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if _has_bundle_signal(payload):
            yield payload
        config = payload.get("config")
        if isinstance(config, Mapping) and _has_bundle_signal(config):
            yield config
        for value in payload.values():
            if isinstance(value, (Mapping, list, tuple)):
                yield from _iter_payloads(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            if isinstance(value, (Mapping, list, tuple)):
                yield from _iter_payloads(value)


def _dedupe(payloads: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    out: list[Mapping[str, Any]] = []
    for payload in payloads:
        key = json.dumps(payload, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def _read_payload(path: Path) -> Any:
    if path.name.endswith(".jsonl"):
        return _read_first_jsonl(path)
    payload = _read_json(path)
    if path.name == "manifest.json" and isinstance(payload, Mapping) and isinstance(payload.get("config"), Mapping):
        return payload["config"]
    return payload


def _root_shortcut_risk(payload: Mapping[str, Any], normalized: Mapping[str, Any]) -> bool:
    if str(normalized.get("reducer_contract") or "") != REDUCER_CONTRACT_BOTTOM_UP:
        return True
    if str(payload.get("tree_text_source") or "").strip().lower() == "existing_summary":
        return True
    for key in ("g_init", "dspy_g_init_mode", "state_contract"):
        if str(payload.get(key) or "").strip().lower() == "teacher_passthrough":
            return True
    return False


def classify_file(path: Path) -> dict[str, Any]:
    base = {
        "path": str(path),
        "classification": CLASS_UNKNOWN,
        "errors": [],
        "tree_bundle_manifest_digest": "",
        "source_kind": "",
        "domain": "",
        "recommendation": "",
    }
    if not path.exists():
        return {**base, "classification": CLASS_UNKNOWN, "errors": ["missing path"]}
    try:
        payload = _read_payload(path)
    except Exception as exc:
        return {**base, "classification": CLASS_UNKNOWN, "errors": [f"parse error: {exc}"]}
    if not isinstance(payload, Mapping):
        return {**base, "classification": CLASS_UNKNOWN, "errors": ["artifact payload is not a JSON object"]}

    if _has_run_signal(payload):
        try:
            normalized_run = normalize_run_manifest(payload)
            validate_run_manifest(normalized_run)
        except Exception as exc:
            return {
                **base,
                "classification": CLASS_UNKNOWN,
                "errors": [str(exc)],
                "recommendation": "rerun through the general C-TreePO runner or repair RunManifest metadata",
            }
        run_contracts = normalized_run.get("input_contracts") or []
        has_tree_input = any(
            isinstance(contract, Mapping)
            and str(contract.get("kind") or "") == "tree_bundle"
            for contract in run_contracts
        )
        if not has_tree_input:
            return {
                **base,
                "classification": CLASS_VALID_RUN,
                "domain": str(normalized_run.get("domain") or ""),
                "recommendation": "",
            }

    bundle_payloads = _dedupe(_iter_payloads(payload))
    if not bundle_payloads:
        return {
            **base,
            "classification": CLASS_MISSING,
            "recommendation": "rerun through a publication entrypoint that emits TreeBundle v1 metadata",
        }

    classifications: list[str] = []
    errors: list[str] = []
    digests: list[str] = []
    source_kinds: list[str] = []
    domains: list[str] = []
    for bundle_payload in bundle_payloads:
        try:
            normalized = normalize_tree_bundle_manifest(bundle_payload)
            validate_tree_bundle_manifest(normalized)
            digest = tree_bundle_manifest_digest(normalized)
            digests.append(digest)
            source_kinds.append(str(normalized.get("source_kind") or ""))
            domains.append(str(normalized.get("domain") or ""))
            has_v1 = (
                str(bundle_payload.get("schema_version") or "") == TREE_BUNDLE_SCHEMA_VERSION
                or isinstance(bundle_payload.get("tree_bundle_manifest"), Mapping)
                or isinstance(bundle_payload.get("tree_bundle_contract"), Mapping)
            )
            if _root_shortcut_risk(bundle_payload, normalized):
                classifications.append(CLASS_ROOT_SHORTCUT)
            elif normalized.get("source_kind") == SOURCE_KIND_EXTERNAL_STATE:
                classifications.append(CLASS_EXTERNAL)
            elif not has_v1:
                classifications.append(CLASS_LEGACY)
            else:
                classifications.append(CLASS_VALID)
        except Exception as exc:
            message = str(exc)
            errors.append(message)
            if "state_dim must be at least 2 * summary_dim" in message:
                classifications.append(CLASS_INVALID_DIM)
            else:
                classifications.append(CLASS_UNKNOWN)
    priority = [
        CLASS_INVALID_DIM,
        CLASS_ROOT_SHORTCUT,
        CLASS_EXTERNAL,
        CLASS_LEGACY,
        CLASS_MISSING,
        CLASS_UNKNOWN,
        CLASS_VALID,
    ]
    classification = next((c for c in priority if c in classifications), CLASS_UNKNOWN)
    recommendation = ""
    if classification in {CLASS_MISSING, CLASS_INVALID_DIM, CLASS_ROOT_SHORTCUT, CLASS_UNKNOWN}:
        recommendation = "rerun with a canonical TreeBundle v1 publication entrypoint"
    elif classification == CLASS_EXTERNAL:
        recommendation = "treat as explicit external-state compatibility artifact"
    elif classification == CLASS_LEGACY:
        recommendation = "migrate metadata to TreeBundle v1 before publication use"
    return {
        **base,
        "classification": classification,
        "errors": errors,
        "tree_bundle_manifest_digest": ",".join(sorted(set(digests))),
        "source_kind": ",".join(sorted(set(source_kinds))),
        "domain": ",".join(sorted(set(domains))),
        "recommendation": recommendation,
    }


def _write_markdown(path: Path, rows: list[Mapping[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("classification") or CLASS_UNKNOWN)
        counts[key] = counts.get(key, 0) + 1
    lines = ["# C-TreePO Artifact Quarantine Report", ""]
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]}")
    lines.extend(["", "| Classification | Path | Source | Domain | Recommendation |", "|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            "| {cls} | `{path}` | {source} | {domain} | {rec} |".format(
                cls=row.get("classification", ""),
                path=row.get("path", ""),
                source=row.get("source_kind", ""),
                domain=row.get("domain", ""),
                rec=row.get("recommendation", ""),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv", action="store_true", help="Also write a CSV report.")
    parser.add_argument("--write-sidecar", action="store_true", help="Write quarantine_status.json next to each scanned artifact.")
    parser.add_argument("--fail-on", default="", help="Comma-separated classifications that should produce exit code 2.")
    args = parser.parse_args(argv)

    rows = [classify_file(path) for path in sorted(set(_iter_candidate_files(args.roots)))]
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.roots[0])
    if output_dir.is_file():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "artifact_quarantine_report.json"
    md_path = output_dir / "artifact_quarantine_report.md"
    payload = {
        "schema_version": 1,
        "roots": [str(Path(root)) for root in args.roots],
        "counts": {
            key: sum(1 for row in rows if row.get("classification") == key)
            for key in sorted({str(row.get("classification")) for row in rows})
        },
        "artifacts": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, rows)
    if args.csv:
        _write_csv(output_dir / "artifact_quarantine_report.csv", rows)
    if args.write_sidecar:
        for row in rows:
            artifact_path = Path(str(row["path"]))
            sidecar = artifact_path.parent / "quarantine_status.json"
            sidecar.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    fail_on = {part.strip() for part in str(args.fail_on or "").split(",") if part.strip()}
    if fail_on and any(str(row.get("classification")) in fail_on for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
