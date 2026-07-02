#!/usr/bin/env python3
"""Audit paper-facing public contract vocabulary.

Public objective artifacts must use root_share, local_law_weight, and
local_law_component_weights. Older hybrid weight names are allowed only in
explicit legacy/provenance contexts. Public oracle-observation artifacts must
use oracle_observation_design instead of mode/rate fields at the top level.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _term(*parts: str) -> str:
    return "".join(parts)


DEFAULT_PATHS = (
    PROJECT_ROOT / "paper" / "ctreepo" / "appendix" / "v7_cdx",
    PROJECT_ROOT / "lean3" / "FormalProofs" / "OPT" / "MainTheorems.lean",
    PROJECT_ROOT / "lean3" / "FormalProofs" / "OPT" / "UnifiedLocalLawAdjustment.lean",
    PROJECT_ROOT / "config" / "markov",
    PROJECT_ROOT / "scripts" / "report_learnability.py",
)

LEGACY_PATTERNS = (
    re.compile(r"\boracle[- ]recovery\b", re.IGNORECASE),
    re.compile(r"\bscore calibration\b", re.IGNORECASE),
    re.compile(r"\bcalibration channel\b", re.IGNORECASE),
    re.compile(r"L_\{\\mathrm\{cal\}\}"),
    re.compile(r"\blambda_local_law\b"),
    re.compile(rf"\b{re.escape(_term('lambda_', 'eff'))}\b"),
    re.compile(rf"\b{re.escape(_term('lambda_', 'effective'))}\b"),
    re.compile(r"\blambda_local\b"),
    re.compile(r"\bselected_lambda_local\b"),
    re.compile(r"\blambda_nominal\b"),
    re.compile(r"\btask_objective_weight\b"),
    re.compile(r"\btree_local_law_weight\b"),
    re.compile(r"\btree_task_objective_weight\b"),
    re.compile(r"\blaw_task_objective_weight\b"),
    re.compile(r"\blaw_c1_weight\b"),
    re.compile(r"\blaw_c2_proxy_weight\b"),
    re.compile(r"\blaw_c2_weight\b"),
    re.compile(r"\blaw_c3_weight\b"),
    re.compile(r"\broot_weight\b"),
    re.compile(r"\bleaf_weight\b"),
    re.compile(r"\bc1_weight\b"),
    re.compile(r"\bc2_weight\b"),
    re.compile(r"\bc3_weight\b"),
    re.compile(r"\blaw_package(?:_names?)?\b"),
    re.compile(r"\bbaseline_family\b"),
    re.compile(r"\btree_families\b"),
    re.compile(r"\bfno_families\b"),
    re.compile(r"\bfull_doc_anchor_families\b"),
    re.compile(r"\bfull_doc_anchor_mode\b"),
    re.compile(r"\bfull_doc_anchor_target\b"),
    re.compile(r"\boracle_observation_mode\b"),
    re.compile(r"\boracle_budget_tree_families\b"),
    re.compile(r"\boracle_budget_reference_families\b"),
    re.compile(r"\bsupervision_recovery_tree_family\b"),
    re.compile(r"\blocal_law_weights\b"),
    re.compile(r"\bgap_weight\b"),
    re.compile(r"\boracle_gap_weight\b"),
    re.compile(r"\bproxy_weights\b"),
    re.compile(rf"\b{re.escape(_term('relia', 'bility'))}\b"),
    re.compile(r"\bbias_calibration\b"),
    re.compile(r"\bbias_gap\b"),
    re.compile(r"\bbias_excess\b"),
    re.compile(r"\bsignal_scale\b"),
)

ALLOW_PATTERNS = (
    re.compile(r"Backward-compatible", re.IGNORECASE),
    re.compile(r"\balias\b", re.IGNORECASE),
    re.compile(r"compatibility", re.IGNORECASE),
    re.compile(r"legacy", re.IGNORECASE),
    re.compile(rf"\bno (?:additional )?{re.escape(_term('relia', 'bility'))}\b", re.IGNORECASE),
    re.compile(rf"\bwithout {re.escape(_term('relia', 'bility'))}\b", re.IGNORECASE),
    re.compile(r"oracleRecovery"),
    re.compile(r"dr_oracle_recovery"),
    re.compile(r"OracleRecoveredWithin"),
)


def _iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
        elif path.is_dir():
            for suffix in ("*.tex", "*.lean", "*.toml", "*.json", "*.md"):
                yield from sorted(path.rglob(suffix))
        else:
            yield path


def _allowed(line: str) -> bool:
    return any(pattern.search(line) for pattern in ALLOW_PATTERNS)


def audit_paths(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in _iter_files(paths):
        if not path.exists():
            errors.append(f"{path}: missing path")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        allow_comment_block = False
        for lineno, line in enumerate(lines, start=1):
            if "Backward-compatible alias" in line:
                allow_comment_block = True
            if allow_comment_block:
                if "-/" in line:
                    allow_comment_block = False
                continue
            if _allowed(line):
                continue
            for pattern in LEGACY_PATTERNS:
                if pattern.search(line):
                    errors.append(f"{path}:{lineno}: legacy objective vocabulary: {line.strip()}")
                    break
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)

    errors = audit_paths(args.paths or DEFAULT_PATHS)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("Objective vocabulary audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
