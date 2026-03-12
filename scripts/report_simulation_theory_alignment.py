#!/usr/bin/env python3
"""Build a Lean-aligned simulation-theory report for a formal rerun root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.expectations import build_expectation_report, write_expectation_report  # noqa: E402
from src.ctreepo.sim.theory_alignment import (  # noqa: E402
    build_simulation_theory_alignment_report,
    write_simulation_theory_alignment_report,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a Lean-aligned simulation theory report.")
    p.add_argument(
        "--formal-root",
        type=Path,
        default=None,
        help="Formal rerun root to scan recursively for simulation outputs.",
    )
    p.add_argument(
        "--expectation-json",
        type=Path,
        default=None,
        help="Existing simulation_expectations.json to reuse instead of rescanning.",
    )
    p.add_argument(
        "--bundle-manifest",
        type=Path,
        default=None,
        help="Optional paper_report_bundle_manifest.json for suite status alignment.",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-markdown", type=Path, default=None)
    p.add_argument(
        "--write-expectations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When scanning a formal root, also write simulation_expectations.{json,md} next to the report.",
    )
    return p


def _default_output_paths(
    *,
    formal_root: Path | None,
    output_json: Path | None,
    output_markdown: Path | None,
) -> tuple[Path, Path]:
    if output_json is not None and output_markdown is not None:
        return output_json.resolve(), output_markdown.resolve()
    if formal_root is not None:
        base = formal_root.resolve() / "paper_reports"
    else:
        base = REPO_ROOT
    json_path = output_json.resolve() if output_json is not None else (base / "simulation_theory_alignment.json")
    md_path = output_markdown.resolve() if output_markdown is not None else (base / "simulation_theory_alignment.md")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    if ns.formal_root is None and ns.expectation_json is None:
        raise SystemExit("expected one of --formal-root or --expectation-json")

    formal_root = ns.formal_root.resolve() if ns.formal_root is not None else None
    default_expectation_json = (
        formal_root / "paper_reports" / "simulation_expectations.json" if formal_root is not None else None
    )
    expectation_json_path: Path | None = ns.expectation_json.resolve() if ns.expectation_json is not None else None
    if expectation_json_path is None and default_expectation_json is not None and default_expectation_json.exists():
        expectation_json_path = default_expectation_json

    if expectation_json_path is None:
        report = build_expectation_report(output_root=formal_root)
        if bool(ns.write_expectations):
            exp_json = formal_root / "paper_reports" / "simulation_expectations.json"
            exp_md = formal_root / "paper_reports" / "simulation_expectations.md"
            write_expectation_report(report, output_json=exp_json, output_markdown=exp_md)
            expectation_json_path = exp_json
    else:
        report = None

    out_json, out_md = _default_output_paths(
        formal_root=ns.formal_root,
        output_json=ns.output_json,
        output_markdown=ns.output_markdown,
    )
    theory_report = build_simulation_theory_alignment_report(
        formal_root=formal_root,
        expectation_report=report,
        expectation_json_path=expectation_json_path,
        bundle_manifest_path=ns.bundle_manifest.resolve() if ns.bundle_manifest is not None else None,
    )
    outputs = write_simulation_theory_alignment_report(
        theory_report,
        output_json=out_json,
        output_markdown=out_md,
    )
    print(
        json.dumps(
            {
                "output_json": outputs["output_json"],
                "output_markdown": outputs["output_markdown"],
                "families": [x.family for x in theory_report.family_statuses],
                "summary": theory_report.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
