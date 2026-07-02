#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "config" / "runtime_umbrella_entrypoints.yaml"
CANONICAL_MARKERS = (
    "write_canonical_sidecars",
    "write_experiment_manifest",
    "ExperimentContext",
    "ExperimentSpec.create",
    "merge_artifacts",
    "append_result_rows",
    "run_manifest_metadata",
)


def _load_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
        return dict(payload or {}) if isinstance(payload, Mapping) else {}
    return json.loads(text)


def _script_paths(root: Path) -> List[str]:
    return sorted(
        str(path.relative_to(root))
        for path in (root / "scripts").glob("*.py")
        if path.name != "__init__.py"
    )


def _glob_matches(path: str, entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any] | None:
    for entry in entries:
        pattern = str(entry.get("glob") or "")
        if pattern and fnmatch.fnmatch(path, pattern):
            return dict(entry)
    return None


def build_report(registry_path: Path, *, root: Path) -> Dict[str, Any]:
    registry = _load_json(registry_path)
    supported_entries = [dict(item) for item in list(registry.get("supported") or [])]
    adapter_entries = [dict(item) for item in list(registry.get("adapter_covered") or [])]
    legacy_globs = [
        {"glob": str(item), "status": "legacy"}
        for item in list(registry.get("legacy_globs") or [])
    ]
    supported_by_path = {
        str(entry.get("path") or ""): entry
        for entry in supported_entries
        if str(entry.get("path") or "")
    }
    adapter_by_path = {
        str(entry.get("path") or ""): entry
        for entry in adapter_entries
        if str(entry.get("path") or "")
    }

    scripts = _script_paths(root)
    classified = []
    unclassified = []
    for path in scripts:
        if path in supported_by_path:
            entry = supported_by_path[path]
            classified.append({"path": path, "class": "supported", **entry})
            continue
        if path in adapter_by_path:
            entry = adapter_by_path[path]
            classified.append({"path": path, "class": "adapter_covered", **entry})
            continue
        adapter_match = _glob_matches(path, adapter_entries)
        if adapter_match is not None:
            classified.append({"path": path, "class": "adapter_covered", **adapter_match})
            continue
        legacy_match = _glob_matches(path, legacy_globs)
        if legacy_match is not None:
            classified.append({"path": path, "class": "legacy", **legacy_match})
            continue
        unclassified.append(path)

    missing_supported = [
        path for path in supported_by_path.keys() if not (root / path).exists()
    ]
    supported_policy_violations = []
    for entry in supported_entries:
        path = str(entry.get("path") or "")
        status = str(entry.get("status") or "")
        if not path:
            supported_policy_violations.append(
                {"path": path, "reason": "missing_supported_path"}
            )
            continue
        if "canonical" not in status:
            supported_policy_violations.append(
                {"path": path, "reason": "supported_status_must_be_canonical", "status": status}
            )
            continue
        candidate = root / path
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        if status != "canonical_tool" and not any(marker in text for marker in CANONICAL_MARKERS):
            supported_policy_violations.append(
                {
                    "path": path,
                    "reason": "missing_canonical_sidecar_marker",
                    "markers": list(CANONICAL_MARKERS),
                }
            )
    return {
        "registry_path": str(registry_path),
        "script_count": len(scripts),
        "classified_count": len(classified),
        "unclassified_count": len(unclassified),
        "missing_supported_count": len(missing_supported),
        "supported_policy_violation_count": len(supported_policy_violations),
        "supported": supported_entries,
        "classified": classified,
        "unclassified": unclassified,
        "missing_supported": missing_supported,
        "supported_policy_violations": supported_policy_violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit runtime umbrella entrypoint coverage.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--fail-on-unclassified", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(Path(args.registry).expanduser().resolve(), root=REPO_ROOT)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Runtime umbrella coverage: "
            f"{report['classified_count']}/{report['script_count']} scripts classified; "
            f"{report['unclassified_count']} unclassified; "
            f"{report['missing_supported_count']} missing supported; "
            f"{report['supported_policy_violation_count']} supported policy violations."
        )
        if report["unclassified"]:
            print("Unclassified scripts:")
            for path in report["unclassified"][:50]:
                print(f"  - {path}")
            if len(report["unclassified"]) > 50:
                print(f"  ... {len(report['unclassified']) - 50} more")
    if report["missing_supported"]:
        return 2
    if report["supported_policy_violations"]:
        return 2
    if args.fail_on_unclassified and report["unclassified"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
