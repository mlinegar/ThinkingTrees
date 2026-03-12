#!/usr/bin/env python3
"""Run manifest-driven performance scenarios across micro/meso/macro layers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.benchmark.perf_harness import (
    build_artifact,
    evaluate_expectation,
    evaluate_regressions,
    extract_metrics,
    has_regression_error,
    load_manifest,
    run_scenario,
    select_scenarios,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ThinkingTrees performance harness.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/perf/perf_matrix.yaml"),
        help="Path to YAML performance manifest.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="ci",
        help="Profile name from manifest (use empty string for all scenarios).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select scenarios and write artifact without executing commands.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print selected scenario IDs and exit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after scenario command failures (default: true).",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any error-severity regression rule fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()

    manifest = load_manifest(manifest_path)
    profile = (args.profile or "").strip() or None
    selected = select_scenarios(manifest, profile)
    if args.list_scenarios:
        for scenario in selected:
            print(f"{scenario.scenario_id}\t{scenario.layer}")
        return 0

    out_path = args.output
    if out_path is None:
        out_path = repo_root / "outputs" / "perf_harness" / f"run_{_utc_stamp()}.json"
    elif not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logs_dir = out_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for scenario in selected:
        result: Dict[str, Any] = {
            "id": scenario.scenario_id,
            "layer": scenario.layer,
            "description": scenario.description,
            "command": scenario.command,
            "metrics_file": scenario.metrics_file,
            "expected_outcome": scenario.expected_outcome,
            "expected_failure_modes": list(scenario.expected_failure_modes),
            "metrics": {},
            "regressions": [],
        }

        if args.dry_run:
            result["status"] = "skipped"
            result["reason"] = "dry_run"
            results.append(result)
            continue

        log_path = logs_dir / f"{scenario.scenario_id}.log"
        try:
            run_info = run_scenario(
                scenario=scenario,
                repo_root=repo_root,
                log_path=log_path,
            )
        except Exception as exc:
            run_info = {
                "started_utc": None,
                "finished_utc": None,
                "wall_seconds": None,
                "exit_code": -1,
                "log_path": str(log_path),
                "error": str(exc),
            }

        result.update(run_info)

        metrics: Dict[str, Any] = {}
        if scenario.metrics_file and scenario.metrics:
            metrics_path = Path(scenario.metrics_file)
            if not metrics_path.is_absolute():
                metrics_path = (repo_root / metrics_path).resolve()
            if metrics_path.exists():
                metrics = extract_metrics(metrics_path, scenario.metrics)
            else:
                metrics = {k: None for k in scenario.metrics}
        result["metrics"] = metrics

        regressions = evaluate_regressions(
            metrics=metrics,
            rules=scenario.regression_rules,
        )
        result["regressions"] = regressions

        command_ok = int(result.get("exit_code", 1)) == 0
        regression_ok = not has_regression_error(regressions)
        expectation = evaluate_expectation(
            expected_outcome=scenario.expected_outcome,
            expected_failure_modes=scenario.expected_failure_modes,
            command_ok=command_ok,
            regression_ok=regression_ok,
        )
        result.update(expectation)
        result["status"] = "passed" if bool(expectation.get("expectation_met")) else "failed"

        results.append(result)
        if (not bool(expectation.get("expectation_met"))) and (not bool(args.continue_on_error)):
            break

    artifact = build_artifact(
        manifest_path=manifest_path,
        profile=profile,
        results=results,
    )
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(str(out_path))

    has_failed = any(row.get("status") == "failed" for row in results)
    has_regression_fail = any(
        has_regression_error(row.get("regressions", []))
        and not (
            str(row.get("expected_outcome", "pass")).lower() == "fail"
            and bool(row.get("expectation_met"))
        )
        for row in results
    )

    if has_failed:
        if args.fail_on_regression:
            return 1
        if any(int(row.get("exit_code", 0)) != 0 for row in results):
            return 1
    if args.fail_on_regression and has_regression_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
