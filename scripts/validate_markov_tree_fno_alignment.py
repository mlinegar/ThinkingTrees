#!/usr/bin/env python3
"""Validate the Markov full-doc tree/FNO lane for provenance and semantic alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_tree_fno_validation import (  # noqa: E402
    build_markov_tree_fno_validation_report,
    write_markov_tree_fno_validation_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate full-doc Markov tree/FNO outputs for provenance and semantics."
    )
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--ladder-json", type=Path, default=None)
    parser.add_argument("--bundle-manifest", type=Path, default=None)
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
    diagnostics_root: Path,
    output_json: Path | None,
    output_markdown: Path | None,
) -> tuple[Path, Path]:
    base = diagnostics_root.resolve()
    json_path = (
        output_json.resolve()
        if output_json is not None
        else (base / "markov_tree_fno_validation.json")
    )
    markdown_path = (
        output_markdown.resolve()
        if output_markdown is not None
        else (base / "markov_tree_fno_validation.md")
    )
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_markov_tree_fno_validation_report(
        diagnostics_root=args.diagnostics_root.resolve(),
        ladder_json=args.ladder_json.resolve() if args.ladder_json is not None else None,
        bundle_manifest_path=(
            args.bundle_manifest.resolve()
            if args.bundle_manifest is not None
            else None
        ),
        run_lean_build=bool(args.run_lean_build),
    )
    out_json, out_markdown = _default_outputs(
        diagnostics_root=args.diagnostics_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    outputs = write_markov_tree_fno_validation_report(
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
