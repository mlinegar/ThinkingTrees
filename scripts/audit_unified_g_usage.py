#!/usr/bin/env python3
"""Audit manifesto/Benoit code paths for split vs unified g usage.

The audit is intentionally conservative: it records all detected split leaf /
merge summarizer references, then buckets them by whether the file is an active
manifesto/Benoit runtime path, a legacy compatibility definition, an active
unified path, or an expected false positive.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


DEFAULT_DATE = "2026-04-21"

KNOWN_ACTIVE_FILES = {
    "scripts/phase0_economic_pilot.py",
    "scripts/phase2_combined_pipeline.py",
    "scripts/phase3_full_pipeline_optimize.py",
    "scripts/phase3_combined_optimize.py",
    "scripts/run_manifesto_batched_example.py",
    "src/tasks/manifesto/pipeline.py",
    "src/tasks/manifesto_task.py",
}

LEGACY_COMPATIBLE_FILES = {
    "src/tasks/manifesto/pipeline.py",
    "src/tasks/manifesto/summarizer.py",
    "src/tasks/manifesto_task.py",
}

SPLIT_PATTERNS = [
    ("ManifestoSummarizer", re.compile(r"\bManifestoSummarizer\b")),
    ("ManifestoMerger", re.compile(r"\bManifestoMerger\b")),
    ("RILEMerge", re.compile(r"\bRILEMerge\b")),
    ("MergeSummarizer", re.compile(r"\bMergeSummarizer\b")),
    ("GenericMerger", re.compile(r"\bGenericMerger\b")),
    ("create_merge_summarizer", re.compile(r"\bcreate_merge_summarizer\b")),
    ("merge_module", re.compile(r"\bmerge_module\s*=")),
    ("merger attribute", re.compile(r"\bself\.merger\b|\bmerger\s*=")),
    ("summary1/summary2 merge signature", re.compile(r"\bsummary1\b.*\bsummary2\b|\bsummary2\b.*\bsummary1\b")),
]

UNIFIED_PATTERNS = [
    ("UnifiedManifestoG", re.compile(r"\bUnifiedManifestoG\b")),
    ("UnifiedG", re.compile(r"\bUnifiedG\b")),
    ("format_merge_input", re.compile(r"\bformat_merge_input\b")),
    ("unified_mode", re.compile(r"\bunified_mode\s*=\s*True\b")),
    ("default_unified_prompt", re.compile(r"\bdefault_unified_prompt\b")),
]

LEGACY_CLASS_PATTERNS = [
    ("RILEMerge", re.compile(r"^\s*class\s+RILEMerge\b")),
    ("ManifestoMerger", re.compile(r"^\s*class\s+ManifestoMerger\b")),
    ("StrategyCompatibleMerger", re.compile(r"^\s*class\s+StrategyCompatibleMerger\b")),
    ("MergeSummarizer", re.compile(r"^\s*class\s+MergeSummarizer\b")),
    ("create_merge_summarizer", re.compile(r"^\s*def\s+create_merge_summarizer\b|^\s*def\s+create_summarizers\b")),
]

TARGET_SYMBOL_PATTERNS = [
    ("ManifestoPipeline", re.compile(r"^\s*class\s+ManifestoPipeline\b|\bManifestoPipeline\b")),
    ("ManifestoPipelineWithStrategy", re.compile(r"^\s*class\s+ManifestoPipelineWithStrategy\b|\bManifestoPipelineWithStrategy\b")),
    ("manifesto_task.create_merge_summarizer", re.compile(r"^\s*def\s+create_merge_summarizer\b")),
]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    symbol: str
    bucket: str
    reason: str
    snippet: str


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def iter_audit_files(root: Path) -> Iterable[Path]:
    candidates: set[Path] = set()
    for rel in KNOWN_ACTIVE_FILES | LEGACY_COMPATIBLE_FILES:
        path = root / rel
        if path.exists():
            candidates.add(path)
    for glob in (
        "src/tasks/manifesto/**/*.py",
        "src/tasks/manifesto_task.py",
        "scripts/phase*_*.py",
        "scripts/run_manifesto*.py",
        "scripts/*manifesto*.py",
        "scripts/*benoit*.py",
        "src/pipelines/**/*.py",
    ):
        candidates.update(p for p in root.glob(glob) if p.is_file())
    return sorted(candidates)


def classify_text(path: str, text: str) -> list[Finding]:
    """Classify split/unified references in one file.

    This function is kept pure so unit tests can exercise the bucketing logic
    with fixture strings.
    """
    findings: list[Finding] = []
    active = path in KNOWN_ACTIVE_FILES
    legacy_file = path in LEGACY_COMPATIBLE_FILES
    in_tests_or_docs = path.startswith("tests/") or path.startswith("docs/")

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        for symbol, pattern in UNIFIED_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        path=path,
                        line=lineno,
                        symbol=symbol,
                        bucket="active_unified_paths" if active else "false_positives",
                        reason="unified g marker",
                        snippet=stripped[:220],
                    )
                )

        for symbol, pattern in LEGACY_CLASS_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        path=path,
                        line=lineno,
                        symbol=symbol,
                        bucket="legacy_compatible_split_classes",
                        reason="split compatibility definition",
                        snippet=stripped[:220],
                    )
                )

        for symbol, pattern in SPLIT_PATTERNS:
            if not pattern.search(line):
                continue
            if in_tests_or_docs:
                bucket = "false_positives"
                reason = "documentation or test reference"
            elif legacy_file and any(p.search(line) for _, p in LEGACY_CLASS_PATTERNS):
                bucket = "legacy_compatible_split_classes"
                reason = "split compatibility definition"
            elif active:
                bucket = "active_split_paths"
                reason = "active runtime path references split leaf/merge g"
            elif legacy_file:
                bucket = "legacy_compatible_split_classes"
                reason = "legacy split compatibility surface"
            else:
                bucket = "false_positives"
                reason = "non-manifesto or non-runtime split reference"
            findings.append(
                Finding(
                    path=path,
                    line=lineno,
                    symbol=symbol,
                    bucket=bucket,
                    reason=reason,
                    snippet=stripped[:220],
                )
            )

        for symbol, pattern in TARGET_SYMBOL_PATTERNS:
            if pattern.search(line):
                bucket = "active_split_paths" if active else "false_positives"
                if symbol == "manifesto_task.create_merge_summarizer" and path != "src/tasks/manifesto_task.py":
                    continue
                findings.append(
                    Finding(
                        path=path,
                        line=lineno,
                        symbol=symbol,
                        bucket=bucket,
                        reason="required target symbol audit marker",
                        snippet=stripped[:220],
                    )
                )

    return findings


def run_audit(root: Path) -> dict[str, list[dict[str, object]]]:
    buckets = {
        "active_split_paths": [],
        "active_unified_paths": [],
        "legacy_compatible_split_classes": [],
        "false_positives": [],
    }
    for path in iter_audit_files(root):
        rel = _rel(path, root)
        findings = classify_text(rel, path.read_text(encoding="utf-8", errors="replace"))
        for finding in findings:
            buckets[finding.bucket].append(asdict(finding))
    for rows in buckets.values():
        rows.sort(key=lambda r: (str(r["path"]), int(r["line"]), str(r["symbol"])))
    return buckets


def _render_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_None detected._\n"
    lines = ["| Path | Line | Symbol | Reason | Snippet |", "|---|---:|---|---|---|"]
    for row in rows:
        snippet = str(row["snippet"]).replace("|", "\\|")
        lines.append(
            f"| `{row['path']}` | {row['line']} | `{row['symbol']}` | "
            f"{row['reason']} | `{snippet}` |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(buckets: dict[str, list[dict[str, object]]], *, audit_date: str) -> str:
    counts = {name: len(rows) for name, rows in buckets.items()}
    return "\n".join(
        [
            f"# Unified g Audit ({audit_date})",
            "",
            "This artifact records manifesto/Benoit code paths that still mention split leaf and merge summarizers before the unified-g migration.",
            "",
            "## Bucket Counts",
            "",
            "| Bucket | Count |",
            "|---|---:|",
            *[f"| `{name}` | {count} |" for name, count in counts.items()],
            "",
            "## Active Split Paths",
            "",
            _render_table(buckets["active_split_paths"]),
            "## Active Unified Paths",
            "",
            _render_table(buckets["active_unified_paths"]),
            "## Legacy-Compatible Split Classes",
            "",
            _render_table(buckets["legacy_compatible_split_classes"]),
            "## False Positives / Non-Runtime Mentions",
            "",
            _render_table(buckets["false_positives"]),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--date", default=DEFAULT_DATE or date.today().isoformat())
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit_date = args.date
    json_out = args.json_out or root / "docs" / f"unified_g_audit_{audit_date}.json"
    md_out = args.md_out or root / "docs" / f"unified_g_audit_{audit_date}.md"
    buckets = run_audit(root)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(buckets, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(buckets, audit_date=audit_date), encoding="utf-8")
    print(f"Wrote {md_out}")
    print(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
