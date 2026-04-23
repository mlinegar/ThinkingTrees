#!/usr/bin/env python3
"""Validate Markov full-doc, budget-share, and full-tree IPW alignment semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_alignment_validation import (  # noqa: E402
    build_markov_alignment_audit_report,
    write_markov_alignment_audit_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Markov alignment semantics across full-doc, budget-share, and full-tree IPW surfaces."
    )
    parser.add_argument("--diagnostics-root", type=Path, default=None)
    parser.add_argument("--full-tree-ipw-root", type=Path, default=None)
    parser.add_argument("--ladder-json", type=Path, default=None)
    parser.add_argument("--bundle-manifest", type=Path, default=None)
    parser.add_argument("--family-grids-summary-json", type=Path, default=None)
    parser.add_argument(
        "--parity-grid-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--run-lean-build",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser


def _default_outputs(
    *,
    diagnostics_root: Path | None,
    full_tree_ipw_root: Path | None,
    family_grids_summary_json: Path | None,
    output_json: Path | None,
    output_markdown: Path | None,
) -> tuple[Path, Path]:
    base = (
        diagnostics_root.resolve()
        if diagnostics_root is not None
        else (
            full_tree_ipw_root.resolve()
            if full_tree_ipw_root is not None
            else (
                family_grids_summary_json.expanduser().resolve().parent
                if family_grids_summary_json is not None
                else REPO_ROOT
            )
        )
    )
    json_path = (
        output_json.resolve()
        if output_json is not None
        else (base / "markov_alignment_audit.json")
    )
    markdown_path = (
        output_markdown.resolve()
        if output_markdown is not None
        else (base / "markov_alignment_audit.md")
    )
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.diagnostics_root is None
        and args.full_tree_ipw_root is None
        and args.family_grids_summary_json is None
        and not list(args.parity_grid_root or [])
    ):
        raise SystemExit(
            "at least one of --diagnostics-root, --full-tree-ipw-root, "
            "--family-grids-summary-json, or --parity-grid-root is required"
        )
    report = build_markov_alignment_audit_report(
        diagnostics_root=(
            args.diagnostics_root.resolve() if args.diagnostics_root is not None else None
        ),
        full_tree_ipw_root=(
            args.full_tree_ipw_root.resolve() if args.full_tree_ipw_root is not None else None
        ),
        ladder_json=args.ladder_json.resolve() if args.ladder_json is not None else None,
        bundle_manifest_path=(
            args.bundle_manifest.resolve()
            if args.bundle_manifest is not None
            else None
        ),
        family_grids_summary_json=(
            args.family_grids_summary_json.resolve()
            if args.family_grids_summary_json is not None
            else None
        ),
        parity_grid_roots=tuple(
            path.resolve() for path in list(args.parity_grid_root or [])
        ),
        run_lean_build=bool(args.run_lean_build),
    )
    out_json, out_markdown = _default_outputs(
        diagnostics_root=args.diagnostics_root,
        full_tree_ipw_root=args.full_tree_ipw_root,
        family_grids_summary_json=args.family_grids_summary_json,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    outputs = write_markov_alignment_audit_report(
        report,
        output_json=out_json,
        output_markdown=out_markdown,
    )
    print(
        json.dumps(
            {
                "output_json": outputs["output_json"],
                "output_markdown": outputs["output_markdown"],
                "summary": dict(report.summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if int(report.summary.get("n_fail", 0)) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
