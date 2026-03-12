#!/usr/bin/env python3
"""Inventory existing local-law outputs and register them into a unified manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.local_law_backfill import load_or_backfill_local_law_payload
from src.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl


PRIMARY_PREFIXES = (
    "markov_law_stress",
    "tree_relevant_lda_local_law",
)
EXPLORATORY_PREFIXES = (
    "markov_local_law_learnability",
    "markov_local_law_journal_suite",
    "tree_relevant_lda_stage3",
    "tree_relevant_lda_best_of",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize existing local-law outputs into a unified inventory.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-root", type=Path, action="append", default=None)
    return parser.parse_args()


def _root_kind(path: Path) -> str:
    name = path.name
    for prefix in PRIMARY_PREFIXES:
        if name.startswith(prefix):
            return "primary"
    for prefix in EXPLORATORY_PREFIXES:
        if name.startswith(prefix):
            return "exploratory"
    return "other"


def _discover_roots(outputs_root: Path, explicit: Sequence[Path] | None) -> List[Path]:
    if explicit:
        return sorted({Path(path).resolve() for path in explicit if Path(path).exists()})
    roots = []
    if outputs_root.exists():
        for path in sorted(outputs_root.iterdir()):
            if path.is_dir() and _root_kind(path) in {"primary", "exploratory"}:
                roots.append(path.resolve())
    return roots


def _load_json(path: Path) -> Dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_non_local_law_stage3_payload(payload: Dict[str, object]) -> bool:
    if str(payload.get("family", "") or "").strip() != "leaf_local_mixture_utility":
        return False
    if "stage3" not in payload:
        return False
    if "local_law" in payload or "local_law_learnability" in payload or "g_artifacts" in payload:
        return False
    return isinstance(payload.get("methods"), dict)


def _inventory_root(root: Path, *, include_in_manifest: bool) -> tuple[Dict[str, object], List[RunSpec]]:
    counts = {
        "json_files": 0,
        "config_json_files": 0,
        "direct_unified": 0,
        "backfilled_legacy": 0,
        "skipped_non_local_law": 0,
        "unsupported": 0,
    }
    manifests: List[RunSpec] = []
    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        counts["json_files"] += 1
        if "config" not in payload:
            continue
        counts["config_json_files"] += 1
        if _is_non_local_law_stage3_payload(payload):
            counts["skipped_non_local_law"] += 1
            continue
        loaded = load_or_backfill_local_law_payload(payload, source_path=str(path))
        if loaded is None:
            counts["unsupported"] += 1
            continue
        summary, augmented = loaded
        backfill = dict(augmented.get("_local_law_backfill", {}) or {})
        if backfill:
            counts["backfilled_legacy"] += 1
            backfill_mode = str(backfill.get("mode", "legacy"))
        else:
            counts["direct_unified"] += 1
            backfill_mode = "direct"
        if include_in_manifest:
            manifests.append(
                RunSpec.create(
                    family=str(summary.family),
                    config={
                        "existing_output": True,
                        "source_root": str(root),
                        "source_path": str(path),
                        "suite_role": str(summary.suite_role),
                        "study_role": str(summary.study_role),
                        "backfill_mode": backfill_mode,
                    },
                    outputs={"json_summary": str(path)},
                    command="true",
                )
            )
    return (
        {
            "root": str(root),
            "root_kind": _root_kind(root),
            "include_in_manifest": bool(include_in_manifest),
            **counts,
        },
        manifests,
    )


def _markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# Existing Local-Law Inventory",
        "",
        "This inventory registers existing outputs into the unified local-law manifest without rerunning simulations.",
        "",
        "## Included Roots",
        "",
    ]
    for row in summary.get("included_roots", []):
        lines.append(
            "- "
            f"`{row['root']}`: "
            f"config-json `{row['config_json_files']}`, "
            f"direct `{row['direct_unified']}`, "
            f"backfilled `{row['backfilled_legacy']}`, "
            f"skipped-non-local-law `{row['skipped_non_local_law']}`, "
            f"unsupported `{row['unsupported']}`."
        )
    lines.extend(["", "## Excluded Roots", ""])
    for row in summary.get("excluded_roots", []):
        lines.append(
            "- "
            f"`{row['root']}` (`{row['root_kind']}`): "
            f"config-json `{row['config_json_files']}`, "
            f"skipped-non-local-law `{row['skipped_non_local_law']}`, "
            f"reason `exploratory or out of primary unified scope`."
        )
    lines.extend(["", "## Totals", ""])
    totals = dict(summary.get("totals", {}) or {})
    for key in (
        "included_config_json_files",
        "included_direct_unified",
        "included_backfilled_legacy",
        "skipped_non_local_law",
        "manifest_runs",
    ):
        lines.append(f"- `{key}`: `{totals.get(key, 0)}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = _discover_roots(args.outputs_root, args.include_root)
    explicit_roots = {Path(path).resolve() for path in (args.include_root or []) if Path(path).exists()}
    included_rows: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []
    manifest_runs: List[RunSpec] = []

    for root in roots:
        kind = _root_kind(root)
        include = kind == "primary" or root in explicit_roots
        row, runs = _inventory_root(root, include_in_manifest=include)
        if include:
            included_rows.append(row)
            manifest_runs.extend(runs)
        else:
            excluded_rows.append(row)

    manifest_path = args.output_dir / "existing_local_law_manifest.jsonl"
    write_manifest_jsonl(manifest_path, manifest_runs)

    summary = {
        "outputs_root": str(args.outputs_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "included_roots": included_rows,
        "excluded_roots": excluded_rows,
        "totals": {
            "included_roots": int(len(included_rows)),
            "excluded_roots": int(len(excluded_rows)),
            "included_config_json_files": int(sum(int(row["config_json_files"]) for row in included_rows)),
            "included_direct_unified": int(sum(int(row["direct_unified"]) for row in included_rows)),
            "included_backfilled_legacy": int(sum(int(row["backfilled_legacy"]) for row in included_rows)),
            "skipped_non_local_law": int(
                sum(int(row["skipped_non_local_law"]) for row in [*included_rows, *excluded_rows])
            ),
            "manifest_runs": int(len(manifest_runs)),
        },
    }
    summary_path = args.output_dir / "existing_local_law_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = args.output_dir / "existing_local_law_inventory.md"
    md_path.write_text(_markdown(summary), encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
