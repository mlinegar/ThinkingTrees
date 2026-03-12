#!/usr/bin/env python3
"""
Compare two performance suite run artifacts and detect regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.perf_suite import (  # noqa: E402
    compare_suite_results,
    load_suite_config,
    render_comparison_markdown,
)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate suite result JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("benchmarks/scenarios/performance_suite_full.yaml"),
        help="Suite config path containing metric_rules.",
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline suite_results.json")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate suite_results.json")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional markdown output path")
    parser.add_argument(
        "--allow-regressions",
        action="store_true",
        help="Do not fail exit code when regressions are found.",
    )
    args = parser.parse_args()

    cfg = load_suite_config(args.scenario)
    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)

    comparison = compare_suite_results(cfg, baseline, candidate)
    comparison["baseline_path"] = str(args.baseline.resolve())
    comparison["candidate_path"] = str(args.candidate.resolve())
    comparison["scenario_path"] = str(args.scenario.resolve())

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"Saved comparison JSON: {args.json_out}")

    markdown = render_comparison_markdown(comparison)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
        print(f"Saved comparison markdown: {args.markdown_out}")
    else:
        print(markdown)

    summary = comparison.get("summary", {}) if isinstance(comparison, dict) else {}
    regressions = int(summary.get("checks_regression", 0) or 0)
    print(
        "Checks: total={total} pass={ok} regression={reg} missing={missing}".format(
            total=summary.get("checks_total", 0),
            ok=summary.get("checks_pass", 0),
            reg=regressions,
            missing=summary.get("checks_missing", 0),
        )
    )
    if regressions > 0 and not args.allow_regressions:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
