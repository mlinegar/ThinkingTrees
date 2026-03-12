#!/usr/bin/env python3
"""
Run scenario-driven performance suite benchmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.perf_suite import (  # noqa: E402
    load_suite_config,
    run_performance_suite,
    save_suite_results,
)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run performance suite from YAML/JSON scenario definition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("benchmarks/scenarios/performance_suite_full.yaml"),
        help="Suite config path (YAML/JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for this run.",
    )
    parser.add_argument(
        "--include-layers",
        type=str,
        default="",
        help="Optional comma-separated layer filter (e.g. micro,component,e2e).",
    )
    parser.add_argument(
        "--include-cases",
        type=str,
        default="",
        help="Optional comma-separated case ID whitelist.",
    )
    parser.add_argument(
        "--exclude-cases",
        type=str,
        default="",
        help="Optional comma-separated case ID blacklist.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Run cases marked enabled=false in scenario file.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap number of selected cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render commands without executing them.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop suite execution after first failed case repeat.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional explicit path for output JSON report.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional explicit path for output markdown report.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print final JSON payload to stdout.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_suite_config(Path(args.scenario))
    include_layers = _split_csv(args.include_layers)
    include_cases = _split_csv(args.include_cases)
    exclude_cases = _split_csv(args.exclude_cases)

    payload = run_performance_suite(
        cfg,
        suite_config_path=Path(args.scenario).resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        include_layers=include_layers or None,
        include_case_ids=include_cases or None,
        exclude_case_ids=exclude_cases or None,
        include_disabled=bool(args.include_disabled),
        dry_run=bool(args.dry_run),
        stop_on_failure=bool(args.stop_on_failure),
        max_cases=args.max_cases,
    )

    saved = save_suite_results(
        payload,
        json_path=args.json_out.resolve() if args.json_out is not None else None,
        markdown_path=args.markdown_out.resolve() if args.markdown_out is not None else None,
    )

    print(f"Saved suite JSON: {saved['json']}")
    print(f"Saved suite markdown: {saved['markdown']}")
    summary = payload.get("summary", {})
    print(
        "Summary: total={total} ok={ok} partial={partial} failed={failed} dry_run={dry} skipped={skipped}".format(
            total=summary.get("cases_total", 0),
            ok=summary.get("cases_ok", 0),
            partial=summary.get("cases_partial", 0),
            failed=summary.get("cases_failed", 0),
            dry=summary.get("cases_dry_run", 0),
            skipped=summary.get("cases_skipped", 0),
        )
    )

    if args.print_json:
        print(json.dumps(payload, indent=2))

    if int(summary.get("cases_failed", 0) or 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
