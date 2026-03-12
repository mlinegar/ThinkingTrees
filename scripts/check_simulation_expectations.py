#!/usr/bin/env python3
"""Scan simulation outputs and emit expectation-check JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.expectations import (  # noqa: E402
    ExpectationConfig,
    VALID_FAMILIES,
    build_expectation_report,
    write_expectation_report,
)


def _parse_family_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for item in str(raw).replace(",", " ").split():
            name = item.strip()
            if not name:
                continue
            if name not in VALID_FAMILIES:
                raise ValueError(
                    f"unsupported family: {name!r}; expected one of {', '.join(VALID_FAMILIES)}"
                )
            out.append(name)
    return sorted(set(out))


def _default_output_paths(
    *,
    output_root: Path | None,
    manifest: Path | None,
    output_json: Path | None,
    output_markdown: Path | None,
) -> tuple[Path | None, Path | None]:
    if output_json is not None and output_markdown is not None:
        return output_json, output_markdown
    if output_root is not None:
        base_dir = output_root.resolve()
    elif manifest is not None:
        base_dir = manifest.resolve().parent
    else:
        base_dir = REPO_ROOT
    json_path = output_json if output_json is not None else (base_dir / "simulation_expectations.json")
    md_path = output_markdown if output_markdown is not None else (base_dir / "simulation_expectations.md")
    return json_path, md_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check whether simulation outputs match the intended qualitative regimes.")
    p.add_argument("--output-root", type=Path, default=None, help="Suite or output root to scan recursively for JSON outputs.")
    p.add_argument("--manifest", type=Path, default=None, help="Optional simulation manifest JSONL.")
    p.add_argument(
        "--family",
        action="append",
        default=[],
        help="Optional family filter. Repeat or pass comma-separated values.",
    )
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-markdown", type=Path, default=None)
    p.add_argument("--min-effect", type=float, default=0.10, help="Relative minimum effect threshold for trend checks.")
    p.add_argument(
        "--adjacent-tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance used when scoring adjacent-step monotonicity.",
    )
    p.add_argument("--seed-aggregate", choices=["median", "mean"], default="median")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    if ns.output_root is None and ns.manifest is None:
        raise SystemExit("expected one of --output-root or --manifest")

    families = _parse_family_args(list(ns.family)) if ns.family else list(VALID_FAMILIES)
    config = ExpectationConfig(
        seed_aggregate=str(ns.seed_aggregate),
        min_effect_rel=float(ns.min_effect),
        adjacent_tolerance=float(ns.adjacent_tolerance),
    )
    report = build_expectation_report(
        output_root=ns.output_root.resolve() if ns.output_root is not None else None,
        manifest_path=ns.manifest.resolve() if ns.manifest is not None else None,
        families=families,
        config=config,
    )
    out_json, out_markdown = _default_output_paths(
        output_root=ns.output_root,
        manifest=ns.manifest,
        output_json=ns.output_json,
        output_markdown=ns.output_markdown,
    )
    outputs = write_expectation_report(report, output_json=out_json, output_markdown=out_markdown)
    payload = {
        "output_json": outputs["output_json"],
        "output_markdown": outputs["output_markdown"],
        "rows_scanned": int(report.rows_scanned),
        "families_scanned": list(report.families_scanned),
        "summary": dict(report.summary),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if bool(ns.strict) and int(report.summary.get("n_fail", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
