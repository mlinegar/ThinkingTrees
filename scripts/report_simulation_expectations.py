#!/usr/bin/env python3
"""Render a Markdown simulation-expectation report from JSON or raw outputs."""

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
    ExpectationReport,
    VALID_FAMILIES,
    build_expectation_report,
    render_expectation_markdown,
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a Markdown report for simulation expectation checks.")
    p.add_argument("--input-json", type=Path, default=None, help="Existing simulation_expectations.json file.")
    p.add_argument("--output-root", type=Path, default=None, help="Optional output root to scan instead of --input-json.")
    p.add_argument("--manifest", type=Path, default=None, help="Optional simulation manifest JSONL.")
    p.add_argument(
        "--family",
        action="append",
        default=[],
        help="Optional family filter when recomputing from raw outputs.",
    )
    p.add_argument("--output-markdown", type=Path, default=None)
    p.add_argument("--min-effect", type=float, default=0.10)
    p.add_argument("--adjacent-tolerance", type=float, default=0.01)
    p.add_argument("--seed-aggregate", choices=["median", "mean"], default="median")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    if ns.input_json is None and ns.output_root is None and ns.manifest is None:
        raise SystemExit("expected --input-json or one of --output-root/--manifest")

    if ns.input_json is not None:
        report = ExpectationReport.from_dict(json.loads(ns.input_json.resolve().read_text(encoding="utf-8")))
    else:
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

    if ns.output_markdown is not None:
        out_path = ns.output_markdown.resolve()
    elif ns.input_json is not None:
        out_path = ns.input_json.resolve().with_suffix(".md")
    elif ns.output_root is not None:
        out_path = ns.output_root.resolve() / "simulation_expectations.md"
    elif ns.manifest is not None:
        out_path = ns.manifest.resolve().parent / "simulation_expectations.md"
    else:
        out_path = REPO_ROOT / "simulation_expectations.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_expectation_markdown(report), encoding="utf-8")
    print(json.dumps({"output_markdown": str(out_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
