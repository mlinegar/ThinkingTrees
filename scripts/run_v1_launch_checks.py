#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments import (  # noqa: E402
    ResultRow,
    benchmark_ref_from_parts,
    experiment_method_ref,
    oracle_ref,
    role_ref,
    write_canonical_sidecars,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "runtime_eval" / "longbench_v2_smoke.yaml"
DEFAULT_TEST_TARGETS = (
    "tests/experiments",
    "tests/runtime/test_runtime_eval_longbench_cli.py",
    "tests/runtime/test_runtime_call_scheduler.py",
    "tests/runtime/test_runtime_methods.py",
    "tests/runtime/test_inference_context.py",
    "tests/training/test_train_ctreepo_cli.py",
    "tests/training/test_train_neural_operators_cli.py",
    "tests/training/test_run_pipeline_canonical_outputs.py",
)
REQUIRED_RUNTIME_ARTIFACTS = (
    "metrics.json",
    "predictions.jsonl",
    "steps.jsonl",
    "calls.jsonl",
    "experiment_manifest.json",
    "experiment_status.json",
    "artifacts.json",
    "results.jsonl",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _run_command(
    name: str,
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        [str(item) for item in cmd],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    log_path = log_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(str(item) for item in cmd),
                f"returncode={proc.returncode}",
                f"elapsed_seconds={elapsed:.3f}",
                "",
                "=== STDOUT ===",
                proc.stdout or "",
                "",
                "=== STDERR ===",
                proc.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "name": str(name),
        "command": [str(item) for item in cmd],
        "returncode": int(proc.returncode),
        "ok": int(proc.returncode) == 0,
        "elapsed_seconds": float(elapsed),
        "log_path": str(log_path),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _prepare_runtime_config(
    config_path: Path,
    output_dir: Path,
    *,
    scorer_endpoint: str | None = None,
    scorer_model: str | None = None,
) -> Path:
    scorer_endpoint = (scorer_endpoint or "").strip()
    scorer_model = (scorer_model or "").strip()
    if not scorer_endpoint and not scorer_model:
        return config_path

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a mapping: {config_path}")

    scorer_cfg = dict(payload.get("scorer") or {})
    old_endpoint = scorer_cfg.get("endpoint") or scorer_cfg.get("base_url")
    old_model = scorer_cfg.get("model")
    if scorer_endpoint:
        scorer_cfg["endpoint"] = scorer_endpoint
        scorer_cfg.pop("base_url", None)
    if scorer_model:
        scorer_cfg["model"] = scorer_model
    payload["scorer"] = scorer_cfg

    summarizer_cfg = dict(payload.get("summarizer") or {})
    if summarizer_cfg:
        summarizer_endpoint = summarizer_cfg.get("endpoint") or summarizer_cfg.get("base_url")
        summarizer_model = summarizer_cfg.get("model")
        if scorer_endpoint and (not summarizer_endpoint or summarizer_endpoint == old_endpoint):
            summarizer_cfg["endpoint"] = scorer_endpoint
            summarizer_cfg.pop("base_url", None)
        if scorer_model and (not summarizer_model or summarizer_model == old_model):
            summarizer_cfg["model"] = scorer_model
        payload["summarizer"] = summarizer_cfg

    resolved_path = output_dir / "resolved_runtime_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_path


def _artifact_check(run_dir: Path) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_RUNTIME_ARTIFACTS if not (run_dir / name).exists()]
    metrics = _load_json(run_dir / "metrics.json")
    n_predictions = int(metrics.get("n_predictions", 0) or 0)
    n_surface_calls = int(metrics.get("n_surface_calls", 0) or 0)
    ok = not missing and n_predictions > 0 and n_surface_calls > 0
    return {
        "name": "runtime_artifacts",
        "ok": bool(ok),
        "missing": missing,
        "experiment_dir": str(run_dir),
        "n_predictions": n_predictions,
        "n_surface_calls": n_surface_calls,
    }


def _write_sidecars(output_dir: Path, report: Mapping[str, Any]) -> None:
    benchmark_ref = benchmark_ref_from_parts(
        family="v1_launch_readiness",
        scope="runtime_longbench_fixture",
        name="v1 launch checks",
        metadata={
            "runtime_experiment_dir": str(report.get("runtime_experiment_dir", "") or ""),
            "config": str(report.get("config", "") or ""),
        },
    )
    method_ref = experiment_method_ref(
        family="v1_launch_checks",
        variant="preflight",
        adapter="v1_launch_checks",
        roles={
            "scorer": role_ref(
                role="scorer",
                surface="process",
                engine="subprocess",
                model="runtime_eval_fixture",
            )
        },
        oracle=oracle_ref(kind="launch_gate", source="local_checks"),
    )
    rows: List[ResultRow] = []
    for check in list(report.get("checks") or []):
        if not isinstance(check, Mapping):
            continue
        rows.append(
            ResultRow(
                experiment_id="",
                phase="preflight",
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                metric_name=f"{check.get('name', 'check')}_passed",
                metric_value=bool(check.get("ok", False)),
                artifact_refs=("v1_launch_report_json",),
                metadata={key: value for key, value in dict(check).items() if key not in {"ok"}},
            )
        )
    write_canonical_sidecars(
        output_dir,
        title="v1_launch_checks",
        adapter_id="v1_launch_checks",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=("preflight",),
        artifacts={"v1_launch_report_json": str(output_dir / "v1_launch_report.json")},
        result_rows=tuple(rows),
        state="completed" if bool(report.get("ok", False)) else "failed",
        metadata={
            "config": str(report.get("config", "") or ""),
            "runtime_experiment_dir": str(report.get("runtime_experiment_dir", "") or ""),
        },
        launch_command=tuple(sys.argv),
        report_profiles=("runtime_eval_summary",),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v1 launch readiness checks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Runtime eval smoke config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for launch-check logs and sidecars.",
    )
    parser.add_argument(
        "--experiment-id",
        default="longbench_v2_smoke",
        help="Runtime smoke experiment id.",
    )
    parser.add_argument("--max-problems", type=int, default=2)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use configured endpoints instead of --mock-llm for runtime eval.",
    )
    parser.add_argument(
        "--check-endpoints",
        action="store_true",
        help="Probe configured /models endpoints during plan.",
    )
    parser.add_argument(
        "--scorer-endpoint",
        default=None,
        help="Override scorer and default summarizer endpoint in the smoke config.",
    )
    parser.add_argument(
        "--scorer-model",
        default=None,
        help="Override scorer and default summarizer model in the smoke config.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip focused pytest gate.")
    parser.add_argument(
        "--test-target",
        action="append",
        default=None,
        help="Override or append pytest target. Repeatable; defaults to the v1 focused gate.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (REPO_ROOT / "outputs" / f"v1_launch_checks_{_utc_stamp()}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    runtime_output_root = output_dir / "runtime"
    runtime_experiment_dir = runtime_output_root / str(args.experiment_id)
    source_config_path = Path(args.config).expanduser().resolve()
    config_path = _prepare_runtime_config(
        source_config_path,
        output_dir,
        scorer_endpoint=args.scorer_endpoint,
        scorer_model=args.scorer_model,
    ).resolve()

    checks: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []

    audit_cmd = [sys.executable, "scripts/audit_runtime_umbrella_coverage.py", "--json"]
    audit = _run_command("audit_runtime_umbrella", audit_cmd, cwd=REPO_ROOT, log_dir=log_dir)
    commands.append(audit)
    checks.append(
        {
            "name": "umbrella_audit",
            "ok": bool(audit["ok"]),
            "returncode": audit["returncode"],
            "log_path": audit["log_path"],
        }
    )

    plan_cmd = [
        sys.executable,
        "scripts/run_runtime_eval.py",
        "plan",
        "--config",
        str(config_path),
        "--output-dir",
        str(runtime_output_root),
        "--experiment-id",
        str(args.experiment_id),
        "--json",
    ]
    if bool(args.check_endpoints):
        plan_cmd.append("--check-endpoints")
    plan = _run_command("runtime_plan", plan_cmd, cwd=REPO_ROOT, log_dir=log_dir)
    commands.append(plan)
    checks.append(
        {
            "name": "runtime_plan",
            "ok": bool(plan["ok"]),
            "returncode": plan["returncode"],
            "log_path": plan["log_path"],
        }
    )

    init_cmd = [
        sys.executable,
        "scripts/run_runtime_eval.py",
        "init",
        "--config",
        str(config_path),
        "--output-dir",
        str(runtime_output_root),
        "--experiment-id",
        str(args.experiment_id),
    ]
    init = _run_command("runtime_init", init_cmd, cwd=REPO_ROOT, log_dir=log_dir)
    commands.append(init)
    checks.append(
        {
            "name": "runtime_init",
            "ok": bool(init["ok"]),
            "returncode": init["returncode"],
            "log_path": init["log_path"],
        }
    )

    run_cmd = [
        sys.executable,
        "scripts/run_runtime_eval.py",
        "run",
        "--experiment-dir",
        str(runtime_experiment_dir),
        "--max-problems",
        str(int(args.max_problems)),
    ]
    if args.scorer_endpoint:
        run_cmd.extend(["--scorer-endpoint", str(args.scorer_endpoint)])
    if not bool(args.live):
        run_cmd.append("--mock-llm")
    run = _run_command("runtime_run", run_cmd, cwd=REPO_ROOT, log_dir=log_dir)
    commands.append(run)
    checks.append(
        {
            "name": "runtime_run",
            "ok": bool(run["ok"]),
            "returncode": run["returncode"],
            "log_path": run["log_path"],
            "mock_llm": not bool(args.live),
        }
    )

    aggregate_cmd = [
        sys.executable,
        "scripts/run_runtime_eval.py",
        "aggregate",
        "--experiment-dir",
        str(runtime_experiment_dir),
    ]
    aggregate = _run_command("runtime_aggregate", aggregate_cmd, cwd=REPO_ROOT, log_dir=log_dir)
    commands.append(aggregate)
    checks.append(
        {
            "name": "runtime_aggregate",
            "ok": bool(aggregate["ok"]),
            "returncode": aggregate["returncode"],
            "log_path": aggregate["log_path"],
        }
    )
    checks.append(_artifact_check(runtime_experiment_dir))

    if not bool(args.skip_tests):
        targets = tuple(args.test_target or DEFAULT_TEST_TARGETS)
        pytest_cmd = [sys.executable, "-m", "pytest", *targets, "-q"]
        pytest_result = _run_command("focused_pytest", pytest_cmd, cwd=REPO_ROOT, log_dir=log_dir)
        commands.append(pytest_result)
        checks.append(
            {
                "name": "focused_pytest",
                "ok": bool(pytest_result["ok"]),
                "returncode": pytest_result["returncode"],
                "log_path": pytest_result["log_path"],
                "targets": list(targets),
            }
        )

    ok = all(bool(check.get("ok", False)) for check in checks)
    report = {
        "ok": bool(ok),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": str(config_path),
        "source_config": str(source_config_path),
        "output_dir": str(output_dir),
        "runtime_experiment_dir": str(runtime_experiment_dir),
        "checks": checks,
        "commands": commands,
    }
    _write_json(output_dir / "v1_launch_report.json", report)
    _write_sidecars(output_dir, report)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"v1 launch checks: {'PASS' if ok else 'FAIL'}")
        print(f"- Output: {output_dir}")
        print(f"- Runtime experiment: {runtime_experiment_dir}")
        for check in checks:
            print(f"- {check['name']}: {'ok' if check.get('ok') else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
