#!/usr/bin/env python3
"""Build and audit the v7 paper asset evidence manifest.

This script is intentionally paper-targeted but contract-generic: it records
which TeX assets are used, where they resolve, whether they live under the
paper asset tree, and what RunManifest/legacy provenance is currently attached.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = PROJECT_ROOT / "paper" / "ctreepo"
DEFAULT_TEX = PAPER_ROOT / "main_v7_cld.tex"
DEFAULT_OUTPUT = PAPER_ROOT / "assets" / "v7_paper_asset_manifest.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.contracts import (  # noqa: E402
    RUN_MANIFEST_SCHEMA_VERSION,
    normalize_run_manifest,
    run_manifest_digest,
)

INCLUDE_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
FIGURE_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")
RUN_MANIFEST_NAMES = ("run_manifest.json", "ctreepo_run_manifest.json", "paper_bundle_manifest.json")

EVIDENCE_ROLES = {
    "01_base_plain.pdf": "mergeable_sketch_framework_diagram",
    "07_local_laws_plain.pdf": "local_law_diagram",
    "classical_sketches_hll_leaf_size.pdf": "hll_exact_state_parity",
    "classical_sketches_summary.pdf": "broad_sketch_suite",
    "learned_sketch_leaf_size_diagnostic.pdf": "learned_state_diagnostic",
    "manifesto_singledim_per_dim_live.pdf": "manifesto_single_dimension",
    "manifesto_fg_combined_ladder_f1g0_f1g1.pdf": "manifesto_universal_summary_ladder",
    "manifesto_fg_combined_audit_gap.pdf": "manifesto_combined_audit_gap",
    "manifesto_singledim_per_dim_live_audit_gap.pdf": "manifesto_single_dimension_audit_gap",
    "benoit_comparison_pearson.tex": "benoit_pearson_table",
    "classical_sketches_compact.tex": "compact_sketch_table",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _paper_rel(path: Path, *, paper_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(paper_root.resolve()))
    except ValueError:
        return str(path)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_tex_reference(ref: str, *, paper_root: Path) -> Path:
    raw = str(ref).strip()
    path = Path(raw)
    if not path.suffix:
        path = path.with_suffix(".tex")
    if path.is_absolute():
        return path
    return paper_root / path


def _collect_tex_sources(tex_paths: Sequence[Path], *, paper_root: Path) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            return
        seen.add(resolved)
        ordered.append(path)
        for ref in INCLUDE_RE.findall(_read_text(path)):
            visit(_resolve_tex_reference(ref, paper_root=paper_root))

    for tex_path in tex_paths:
        visit(Path(tex_path))
    return ordered


def _graphic_paths(*, paper_root: Path) -> list[Path]:
    paths = [paper_root]
    preamble = paper_root / "preamble.tex"
    if not preamble.exists():
        return paths
    in_block = False
    for line in _read_text(preamble).splitlines():
        if "\\graphicspath" in line:
            in_block = True
        if in_block:
            for raw in re.findall(r"\{([^{}]+)\}", line):
                candidate = Path(raw)
                paths.append(candidate if candidate.is_absolute() else paper_root / candidate)
            if line.strip() == "}":
                in_block = False
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _candidate_asset_paths(ref: str, *, paper_root: Path, graphic_paths: Sequence[Path]) -> list[Path]:
    raw = str(ref).strip()
    path = Path(raw)
    suffixes = (path.suffix,) if path.suffix else FIGURE_SUFFIXES
    bases: list[Path] = []
    if path.is_absolute():
        bases.append(path.with_suffix("") if path.suffix else path)
    elif "/" in raw:
        bases.append((paper_root / path).with_suffix("") if path.suffix else paper_root / path)
    else:
        for root in graphic_paths:
            bases.append((root / path).with_suffix("") if path.suffix else root / path)
    candidates: list[Path] = []
    for base in bases:
        for suffix in suffixes:
            candidates.append(base.with_suffix(suffix))
    return candidates


def _resolve_asset(ref: str, *, paper_root: Path, graphic_paths: Sequence[Path]) -> Path | None:
    for candidate in _candidate_asset_paths(ref, paper_root=paper_root, graphic_paths=graphic_paths):
        if candidate.exists():
            return candidate
    candidates = _candidate_asset_paths(ref, paper_root=paper_root, graphic_paths=graphic_paths)
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _run_manifest_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if str(payload.get("schema_version") or "") == RUN_MANIFEST_SCHEMA_VERSION:
        return payload
    nested = payload.get("run_manifest")
    return nested if isinstance(nested, Mapping) else None


def _find_provenance(path: Path, *, paper_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"kind": "missing", "path": "", "run_manifest_digest": ""}
    search_dirs = [path.parent, *path.parents]
    assets_root = paper_root / "assets"
    for directory in search_dirs:
        if not _under(directory, paper_root):
            break
        for name in RUN_MANIFEST_NAMES:
            candidate = directory / name
            payload = _load_json(candidate) if candidate.exists() else None
            if not payload:
                continue
            run_manifest = _run_manifest_from_payload(payload)
            if run_manifest is None:
                continue
            try:
                normalized = normalize_run_manifest(run_manifest)
                digest = run_manifest_digest(normalized)
            except Exception:
                digest = ""
            return {
                "kind": "run_manifest",
                "path": _paper_rel(candidate, paper_root=paper_root),
                "run_manifest_digest": digest,
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            }
        for candidate in sorted(directory.glob("*manifest*.json")):
            if candidate.name == DEFAULT_OUTPUT.name:
                continue
            payload = _load_json(candidate)
            if not payload:
                continue
            run_manifest = _run_manifest_from_payload(payload)
            if run_manifest is not None:
                try:
                    normalized = normalize_run_manifest(run_manifest)
                    digest = run_manifest_digest(normalized)
                except Exception:
                    digest = ""
                return {
                    "kind": "run_manifest",
                    "path": _paper_rel(candidate, paper_root=paper_root),
                    "run_manifest_digest": digest,
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                }
            return {
                "kind": "legacy_manifest",
                "path": _paper_rel(candidate, paper_root=paper_root),
                "run_manifest_digest": "",
                "schema_version": str(payload.get("schema_version") or ""),
            }
        if directory.resolve() == assets_root.resolve():
            break
    return {"kind": "none", "path": "", "run_manifest_digest": ""}


def _regeneration_command(path: Path) -> str:
    rendered = str(path)
    name = path.name
    if "/assets/sketches/" in rendered:
        return "python scripts/run_classical_sketches_paper_bundle.py --out-root outputs/classical_sketches_paper_bundle_$(date +%Y%m%d_%H%M%S)"
    if "/assets/benoit/" in rendered:
        return "bash scripts/run_benoit_combined_dspy_ladder.sh"
    if name in {"01_base_plain.pdf", "07_local_laws_plain.pdf"}:
        return "cp doc/old/figures/cld/{01_base_plain.pdf,07_local_laws_plain.pdf} paper/ctreepo/assets/diagrams/figures/"
    if "/assets/markov/" in rendered:
        return "python scripts/run_ctreepo.py --target markov.publication_bundle --plan-only"
    return ""


def _classification(
    *,
    path: Path | None,
    exists: bool,
    under_assets: bool,
    provenance: Mapping[str, Any],
) -> str:
    if not exists:
        return "missing_contract"
    if provenance.get("kind") == "run_manifest":
        return "valid_treebundle_v1"
    if provenance.get("kind") == "legacy_manifest":
        return "legacy_migratable"
    if path is not None and path.name in {"01_base_plain.pdf", "07_local_laws_plain.pdf"} and under_assets:
        return "legacy_migratable"
    if under_assets:
        return "missing_contract"
    return "unknown"


def _asset_entry(
    *,
    kind: str,
    ref: str,
    tex_source: Path,
    paper_root: Path,
    graphic_paths: Sequence[Path],
) -> dict[str, Any]:
    if kind == "table":
        resolved = _resolve_tex_reference(ref, paper_root=paper_root)
    else:
        resolved = _resolve_asset(ref, paper_root=paper_root, graphic_paths=graphic_paths)
    exists = bool(resolved and resolved.exists())
    under_assets = bool(resolved and _under(resolved, paper_root / "assets"))
    provenance = _find_provenance(resolved, paper_root=paper_root) if resolved else {"kind": "missing"}
    classification = _classification(
        path=resolved,
        exists=exists,
        under_assets=under_assets,
        provenance=provenance,
    )
    return {
        "kind": kind,
        "asset_reference": str(ref),
        "tex_source": _paper_rel(tex_source, paper_root=paper_root),
        "resolved_path": _paper_rel(resolved, paper_root=paper_root) if resolved else "",
        "exists": exists,
        "under_paper_assets": under_assets,
        "evidence_role": EVIDENCE_ROLES.get(Path(str(ref)).name, ""),
        "provenance": provenance,
        "quarantine_status": {"classification": classification},
        "regeneration_command": _regeneration_command(resolved) if resolved else "",
    }


def build_asset_manifest(
    *,
    paper_root: Path = PAPER_ROOT,
    tex_paths: Sequence[Path] = (DEFAULT_TEX,),
) -> dict[str, Any]:
    paper_root = Path(paper_root)
    tex_paths = [Path(path) for path in tex_paths]
    sources = _collect_tex_sources(tex_paths, paper_root=paper_root)
    graphic_paths = _graphic_paths(paper_root=paper_root)
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        if not source.exists():
            continue
        text = _read_text(source)
        for ref in GRAPHICS_RE.findall(text):
            key = ("figure", ref)
            if key not in seen:
                entries.append(
                    _asset_entry(
                        kind="figure",
                        ref=ref,
                        tex_source=source,
                        paper_root=paper_root,
                        graphic_paths=graphic_paths,
                    )
                )
                seen.add(key)
        for ref in INCLUDE_RE.findall(text):
            if not (ref.startswith("assets/") or "/tables/" in ref):
                continue
            key = ("table", ref)
            if key not in seen:
                entries.append(
                    _asset_entry(
                        kind="table",
                        ref=ref,
                        tex_source=source,
                        paper_root=paper_root,
                        graphic_paths=graphic_paths,
                    )
                )
                seen.add(key)
    counts: dict[str, int] = {}
    for entry in entries:
        classification = str((entry.get("quarantine_status") or {}).get("classification") or "")
        counts[classification] = counts.get(classification, 0) + 1
    missing = [entry for entry in entries if not entry.get("exists")]
    outside_assets = [
        entry
        for entry in entries
        if entry.get("exists") and not bool(entry.get("under_paper_assets"))
    ]
    missing_contract = [
        entry
        for entry in entries
        if str((entry.get("quarantine_status") or {}).get("classification") or "")
        in {"missing_contract", "unknown"}
    ]
    return {
        "schema_version": "ctreepo.paper_asset_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_root": str(paper_root),
        "tex_targets": [_paper_rel(path, paper_root=paper_root) for path in tex_paths],
        "repo_vocabulary": {
            "tree_inputs": "TreeBundleManifest",
            "executions": "RunManifest",
            "artifact_status": "ArtifactReport/quarantine",
        },
        "summary": {
            "asset_count": len(entries),
            "classification_counts": counts,
            "missing_count": len(missing),
            "outside_paper_assets_count": len(outside_assets),
            "missing_or_unknown_contract_count": len(missing_contract),
            "publication_ready": not missing and not outside_assets and not missing_contract,
        },
        "assets": entries,
    }


def _write_markdown(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# V7 Paper Asset Manifest",
        "",
        f"- Assets: {manifest['summary']['asset_count']}",
        f"- Missing: {manifest['summary']['missing_count']}",
        f"- Outside paper assets: {manifest['summary']['outside_paper_assets_count']}",
        f"- Missing/unknown contract: {manifest['summary']['missing_or_unknown_contract_count']}",
        "",
        "| Kind | Asset | Status | Provenance |",
        "|---|---|---|---|",
    ]
    for entry in manifest.get("assets", []):
        status = (entry.get("quarantine_status") or {}).get("classification", "")
        provenance = (entry.get("provenance") or {}).get("path", "")
        lines.append(
            f"| {entry.get('kind', '')} | `{entry.get('resolved_path', entry.get('asset_reference', ''))}` | {status} | `{provenance}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-markdown", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-contract", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tex_paths = args.tex or [DEFAULT_TEX]
    manifest = build_asset_manifest(paper_root=PAPER_ROOT, tex_paths=tex_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_markdown is not None:
        args.write_markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.write_markdown, manifest)
    if args.check:
        errors: list[str] = []
        if manifest["summary"]["missing_count"]:
            errors.append(f"missing assets: {manifest['summary']['missing_count']}")
        if manifest["summary"]["outside_paper_assets_count"]:
            errors.append(
                "assets outside paper/ctreepo/assets: "
                f"{manifest['summary']['outside_paper_assets_count']}"
            )
        if args.require_contract and manifest["summary"]["missing_or_unknown_contract_count"]:
            errors.append(
                "assets missing RunManifest/contract provenance: "
                f"{manifest['summary']['missing_or_unknown_contract_count']}"
            )
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
