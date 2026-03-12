#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.local_law_backfill import load_or_backfill_local_law_payload
from src.ctreepo.sim.objective_backfill import safe_objective_backfill


def _iter_json_files(root: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and "backfill_logs" not in str(path)
    )


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_run_payload(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "family",
            "config",
            "methods",
            "metrics",
            "local_law",
            "stage3",
            "objective",
        )
    )


def _root_blank_lda_package_fallback(root: Path) -> str:
    explicit_packages = set()
    saw_legacy_local_law = False
    for path in _iter_json_files(root):
        payload = _load_json(path)
        if payload is None or not _is_run_payload(payload):
            continue
        local_law = dict(payload.get("local_law", {}) or {})
        if not local_law:
            continue
        saw_legacy_local_law = True
        cfg = dict(payload.get("config", {}) or {})
        local_cfg = dict(local_law.get("config", {}) or {})
        package = str(local_cfg.get("law_package", cfg.get("law_package", "")) or "").strip()
        if package:
            explicit_packages.add(package)
    if saw_legacy_local_law and explicit_packages.issubset({"all_laws"}):
        return "all_laws"
    return ""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill deterministic objective/local-law metadata where safe.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest: Dict[str, Any] = {
        "roots": [str(Path(root)) for root in args.roots],
        "dry_run": bool(args.dry_run),
        "results": [],
    }
    for root in args.roots:
        root = Path(root)
        if not root.exists():
            manifest["results"].append({"root": str(root), "exists": False})
            continue
        blank_fallback = _root_blank_lda_package_fallback(root)
        counts = {
            "root": str(root),
            "exists": True,
            "blank_lda_package_fallback": str(blank_fallback),
            "files_scanned": 0,
            "objective_backfilled": 0,
            "local_law_backfilled": 0,
            "written_files": 0,
        }
        for path in _iter_json_files(root):
            payload = _load_json(path)
            if payload is None or not _is_run_payload(payload):
                continue
            counts["files_scanned"] += 1
            updated = dict(payload)
            changed = False

            if not isinstance(updated.get("objective"), Mapping) or not dict(updated.get("objective", {}) or {}):
                objective = safe_objective_backfill(updated)
                if objective is not None:
                    updated["objective"] = objective
                    counts["objective_backfilled"] += 1
                    changed = True

            if not isinstance(updated.get("local_law_learnability"), Mapping) or not dict(
                updated.get("local_law_learnability", {}) or {}
            ):
                loaded = load_or_backfill_local_law_payload(
                    updated,
                    source_path=str(path),
                    blank_lda_law_package_fallback=str(blank_fallback),
                )
                if loaded is not None:
                    _summary, augmented = loaded
                    if dict(augmented.get("_local_law_backfill", {}) or {}):
                        updated = dict(augmented)
                        counts["local_law_backfilled"] += 1
                        changed = True

            if changed:
                counts["written_files"] += 1
                if not args.dry_run:
                    _write_json(path, updated)

        manifest["results"].append(counts)

    if args.manifest is not None:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
