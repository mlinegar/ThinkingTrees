#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.contracts import ExperimentSpec, ProgressSnapshot  # noqa: E402
from src.experiments.control_plane import (  # noqa: E402
    canonical_artifact_refs_from_paths,
    experiment_paths,
    load_json,
    merge_artifacts,
    write_experiment_manifest,
    write_experiment_status,
)
from src.experiments.registry import (  # noqa: E402
    METHOD_ADAPTERS,
    REPORT_PROFILES,
    ensure_default_method_adapters,
)


def _strip_remainder(items: Sequence[str]) -> list[str]:
    values = list(items)
    if values and values[0] == "--":
        return values[1:]
    return values


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_manifest(path: Path) -> ExperimentSpec:
    payload = load_json(path)
    if not payload:
        raise FileNotFoundError(f"experiment manifest not found or invalid: {path}")
    return ExperimentSpec.from_dict(payload)


def _infer_output_root_from_manifest(path: Path) -> Path:
    spec = _load_manifest(path)
    return Path(spec.output_root).expanduser().resolve()


def _command_script(command: Sequence[str]) -> str:
    parts = [str(item) for item in list(command)]
    if not parts:
        return ""
    first = Path(parts[0]).name
    if "python" in first and len(parts) >= 2:
        if parts[1] == "-m" and len(parts) >= 3:
            return str(parts[2])
        return Path(parts[1]).name
    return Path(parts[0]).name


def _infer_adapter(command: Sequence[str]) -> str:
    script = _command_script(command)
    if not script:
        raise SystemExit("missing legacy command after `--`")
    if script in {"run_runtime_eval.py"}:
        return "runtime_eval"
    if script in {
        "run_markov_publication_bundle.py",
        "run_markov_optimization_tradeoff_pipeline.py",
        "run_markov_supervision_recovery_parity_grid.py",
    }:
        return "markov_tree"
    if script in {"train_neural_operators.py", "train_ctreepo.py", "src.training.run_pipeline"}:
        return "treepo_training"
    if script == "run_experiment.py":
        return "report_only"
    if script.startswith("report_"):
        return "report_only"
    if any(str(item) in {"--manifest", "--runspec-manifest"} for item in command):
        return "ctreepo_sim"
    return "runtime_umbrella_script"


def _resolve_adapter(adapter_key: str | None, command: Sequence[str]) -> Any:
    ensure_default_method_adapters()
    return METHOD_ADAPTERS.resolve(str(adapter_key or _infer_adapter(command)))


def _spec_summary(spec: ExperimentSpec, *, manifest_path: Path | None = None) -> Mapping[str, Any]:
    role_names = sorted(
        {
            str(role)
            for method_ref in spec.method_refs
            for role in dict(method_ref.metadata.get("roles", {}) or {}).keys()
        }
    )
    return {
        "experiment_id": spec.experiment_id,
        "adapter": spec.adapter_id,
        "title": spec.title,
        "output_root": spec.output_root,
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "benchmarks": [item.family for item in spec.benchmark_refs],
        "methods": [item.family for item in spec.method_refs],
        "roles": role_names,
        "phases": [item.phase_id for item in spec.phases],
        "report_profiles": list(spec.report_profiles),
    }


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(f"Experiment: {summary.get('title') or summary.get('experiment_id')}")
    print(f"Adapter:    {summary.get('adapter')}")
    print(f"Output:     {summary.get('output_root')}")
    if summary.get("manifest_path"):
        print(f"Manifest:   {summary.get('manifest_path')}")
    print(f"Benchmarks: {', '.join(summary.get('benchmarks') or []) or 'n/a'}")
    print(f"Methods:    {', '.join(summary.get('methods') or []) or 'n/a'}")
    print(f"Roles:      {', '.join(summary.get('roles') or []) or 'n/a'}")
    print(f"Phases:     {', '.join(summary.get('phases') or []) or 'n/a'}")


def _count_result_rows(output_root: str | Path) -> int:
    results_path = experiment_paths(output_root)["results"]
    if not results_path.exists():
        return 0
    return sum(1 for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _collect_experiment_outputs(
    *,
    adapter: Any,
    spec: ExperimentSpec,
    returncode: int | None = None,
) -> Mapping[str, Any]:
    output_root = Path(spec.output_root).expanduser().resolve()
    artifacts = dict(adapter.collect_artifacts(output_root) or {})
    if artifacts:
        merge_artifacts(
            output_root,
            canonical_artifact_refs_from_paths(artifacts, phase_id="collect", required=False),
        )
    row_count = _count_result_rows(output_root)
    state = (
        "failed"
        if returncode is not None and int(returncode) != 0
        else "completed"
        if returncode is not None
        else "collected"
    )
    status_path = write_experiment_status(
        output_root,
        ProgressSnapshot(
            experiment_id=str(spec.experiment_id),
            state=state,
            active_phase="collect",
            items_total=max(row_count, len(spec.method_refs), 1),
            completed_items=row_count if state != "failed" else 0,
            failed_items=1 if state == "failed" else 0,
            percent_complete=100.0,
            artifact_targets=tuple(str(key) for key in artifacts.keys()),
            metadata={
                "returncode": returncode,
                "collected_artifact_count": len(artifacts),
                "canonical_result_rows": row_count,
            },
        ),
    )
    return {
        "output_root": str(output_root),
        "status_path": str(status_path),
        "state": state,
        "returncode": returncode,
        "artifacts": artifacts,
        "result_rows": row_count,
    }


def cmd_plan(args: argparse.Namespace) -> int:
    command = _strip_remainder(args.command or [])
    if not command:
        raise SystemExit("missing legacy command after `--`")
    adapter = _resolve_adapter(args.adapter, command)
    spec = adapter.build_experiment_spec(command, cwd=Path(args.cwd).resolve())
    manifest_path = write_experiment_manifest(spec.output_root, spec)
    if bool(args.summary):
        _print_summary(_spec_summary(spec, manifest_path=manifest_path))
    else:
        payload = spec.to_dict()
        payload["manifest_path"] = str(manifest_path)
        print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _run_command(command: Sequence[str], *, cwd: Path) -> int:
    result = subprocess.run(list(command), cwd=cwd, check=False)
    return int(result.returncode)


def _launch_via_long_job(
    *,
    name: str,
    cwd: Path,
    description: str,
    job_root: str | None,
    command: Sequence[str],
) -> int:
    long_job = REPO_ROOT / "scripts" / "long_job.py"
    launch_cmd = [
        sys.executable,
        str(long_job),
        "launch",
        "--name",
        str(name),
        "--cwd",
        str(cwd),
    ]
    if description:
        launch_cmd.extend(["--description", str(description)])
    if job_root:
        launch_cmd.extend(["--job-root", str(job_root)])
    launch_cmd.append("--")
    launch_cmd.extend(str(item) for item in command)
    result = subprocess.run(launch_cmd, cwd=REPO_ROOT, check=False)
    return int(result.returncode)


def cmd_launch(args: argparse.Namespace) -> int:
    command = _strip_remainder(args.command or [])
    if not command:
        raise SystemExit("missing legacy command after `--`")
    adapter = _resolve_adapter(args.adapter, command)
    spec = adapter.build_experiment_spec(command, cwd=Path(args.cwd).resolve())
    write_experiment_manifest(spec.output_root, spec)
    if bool(args.detach):
        return _launch_via_long_job(
            name=str(args.name or spec.title or spec.experiment_id),
            cwd=Path(args.cwd).resolve(),
            description=str(args.description or spec.title),
            job_root=args.job_root,
            command=command,
        )
    returncode = _run_command(command, cwd=Path(args.cwd).resolve())
    if bool(args.collect):
        payload = _collect_experiment_outputs(
            adapter=adapter,
            spec=spec,
            returncode=returncode,
        )
        if bool(args.json):
            print(json.dumps(payload, indent=2, sort_keys=False))
    return returncode


def cmd_resume(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    spec = _load_manifest(manifest_path)
    command = list(spec.resume_command or spec.launch_command)
    if not command:
        raise SystemExit(f"manifest {manifest_path} does not declare a launch command")
    if bool(args.detach):
        return _launch_via_long_job(
            name=str(args.name or spec.title or spec.experiment_id),
            cwd=Path(args.cwd).resolve(),
            description=str(args.description or spec.title),
            job_root=args.job_root,
            command=command,
        )
    returncode = _run_command(command, cwd=Path(args.cwd).resolve())
    if bool(args.collect):
        adapter = METHOD_ADAPTERS.resolve(str(spec.adapter_id))
        payload = _collect_experiment_outputs(
            adapter=adapter,
            spec=spec,
            returncode=returncode,
        )
        if bool(args.json):
            print(json.dumps(payload, indent=2, sort_keys=False))
    return returncode


def cmd_collect(args: argparse.Namespace) -> int:
    spec = _load_manifest(Path(args.manifest).expanduser().resolve())
    adapter = METHOD_ADAPTERS.resolve(str(args.adapter or spec.adapter_id))
    payload = _collect_experiment_outputs(adapter=adapter, spec=spec)
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if args.manifest is not None:
        output_root = _infer_output_root_from_manifest(Path(args.manifest).expanduser().resolve())
    else:
        output_root = Path(args.output_root).expanduser().resolve()
    paths = experiment_paths(output_root)
    payload = load_json(paths["status"])
    if not payload:
        raise SystemExit(f"experiment status not found at {paths['status']}")
    payload["status_path"] = str(paths["status"])
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    spec = _load_manifest(manifest_path)
    summary = _spec_summary(spec, manifest_path=manifest_path)
    if bool(args.json):
        print(json.dumps(summary, indent=2, sort_keys=False))
    else:
        _print_summary(summary)
    return 0


def _load_umbrella_registry() -> Mapping[str, Any]:
    path = REPO_ROOT / "config" / "runtime_umbrella_entrypoints.yaml"
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload or {}) if isinstance(payload, Mapping) else {}


def cmd_list(args: argparse.Namespace) -> int:
    registry = _load_umbrella_registry()
    payload = {
        "adapters": list(METHOD_ADAPTERS.available()),
        "report_profiles": list(REPORT_PROFILES.available()),
        "supported": list(registry.get("supported") or []),
        "adapter_covered": list(registry.get("adapter_covered") or []),
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    print("Adapters:")
    for item in payload["adapters"]:
        print(f"  - {item}")
    print("\nSupported entrypoints:")
    for item in payload["supported"]:
        entry = dict(item or {})
        print(f"  - {entry.get('path')} [{entry.get('status', '')}]")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    profile = REPORT_PROFILES.resolve(str(args.profile))
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = Path(args.report_output_dir).expanduser().resolve() if args.report_output_dir else None
    payload = dict(
        profile.render(
            output_root=output_root,
            output_dir=output_dir,
            options={"extra_args": list(args.extra_args or [])},
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _default_v1_check_output_dir() -> Path:
    return (REPO_ROOT / "outputs" / f"v1_launch_checks_{_utc_stamp()}").resolve()


def _v1_check_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    json_output: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_v1_launch_checks.py"),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--output-dir",
        str(output_dir),
        "--experiment-id",
        str(args.experiment_id),
        "--max-problems",
        str(int(args.max_problems)),
    ]
    if bool(json_output):
        cmd.append("--json")
    if bool(args.live):
        cmd.append("--live")
    if bool(args.check_endpoints):
        cmd.append("--check-endpoints")
    if bool(args.skip_tests):
        cmd.append("--skip-tests")
    if args.scorer_endpoint:
        cmd.extend(["--scorer-endpoint", str(args.scorer_endpoint)])
    if args.scorer_model:
        cmd.extend(["--scorer-model", str(args.scorer_model)])
    for target in list(args.test_target or []):
        cmd.extend(["--test-target", str(target)])
    return cmd


def cmd_check(args: argparse.Namespace) -> int:
    if str(args.suite) != "v1":
        raise SystemExit(f"unsupported check suite: {args.suite}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_v1_check_output_dir()
    )
    capture_check_output = bool(args.json or args.report)
    cmd = _v1_check_command(args, output_dir, json_output=capture_check_output)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=capture_check_output,
        text=True,
        check=False,
    )
    if not capture_check_output:
        return int(result.returncode)

    check_payload: Mapping[str, Any]
    try:
        check_payload = json.loads(result.stdout or "{}")
    except Exception:
        check_payload = {}

    report_payload: Mapping[str, Any] = {}
    if bool(args.report) and int(result.returncode) == 0:
        runtime_experiment_dir = Path(
            str(check_payload.get("runtime_experiment_dir") or check_payload.get("runtime_run_dir") or "")
        )
        if runtime_experiment_dir.exists():
            profile = REPORT_PROFILES.resolve("runtime_v1")
            report_output_dir = (
                Path(args.report_output_dir).expanduser().resolve()
                if args.report_output_dir
                else output_dir / "paper_report"
            )
            report_payload = dict(
                profile.render(
                    output_root=runtime_experiment_dir,
                    output_dir=report_output_dir,
                    options={},
                )
            )

    payload = {
        "suite": str(args.suite),
        "returncode": int(result.returncode),
        "ok": int(result.returncode) == 0 and bool(check_payload.get("ok", False)),
        "output_dir": str(output_dir),
        "check": dict(check_payload),
        "report": dict(report_payload),
    }
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=False, default=str))
    else:
        if check_payload:
            print(f"v1 check: {'PASS' if payload['ok'] else 'FAIL'}")
            print(f"- Output: {payload['output_dir']}")
            if check_payload.get("runtime_experiment_dir") or check_payload.get("runtime_run_dir"):
                print(
                    "- Runtime experiment: "
                    f"{check_payload.get('runtime_experiment_dir') or check_payload.get('runtime_run_dir')}"
                )
            for check in list(check_payload.get("checks") or []):
                if isinstance(check, Mapping):
                    print(
                        f"- {check.get('name', 'check')}: "
                        f"{'ok' if check.get('ok') else 'FAILED'}"
                    )
        elif result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if report_payload:
            print(
                "Runtime v1 report: "
                f"{dict(report_payload).get('json_path') or dict(report_payload).get('experiment_dir')}"
            )
    return int(result.returncode)


class _LegacyReportProfile:
    def __init__(self, profile_id: str, script_name: str) -> None:
        self.profile_id = str(profile_id)
        self.aliases: tuple[str, ...] = ()
        self.script_name = str(script_name)

    def render(
        self,
        *,
        output_root: Path,
        output_dir: Path | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        script_path = REPO_ROOT / "scripts" / self.script_name
        extra_args = list((options or {}).get("extra_args") or [])
        cmd = [sys.executable, str(script_path)]
        if output_dir is not None:
            cmd.extend(["--output-dir", str(output_dir)])
        if self.profile_id in {"tradeoff", "supervision_recovery"}:
            source = output_root / "supervision_recovery" / "summary.json"
            if not source.exists():
                source = output_root / "tradeoff_report" / "summary.json"
            if source.exists():
                cmd.extend(["--supervision-recovery-summary", str(source)])
            cmd.extend(["--output-root", str(output_root)])
        elif self.profile_id == "publication_bundle":
            cmd.extend(["--output-root", str(output_root)])
        elif self.profile_id == "runtime_eval_summary":
            metrics_path = output_root / "metrics.json"
            payload = load_json(metrics_path)
            return {
                "profile": self.profile_id,
                "metrics_json": str(metrics_path),
                "metrics": payload,
            }
        cmd.extend(str(item) for item in extra_args)
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return {
            "profile": self.profile_id,
            "returncode": int(result.returncode),
            "stdout": str(result.stdout),
            "stderr": str(result.stderr),
            "command": cmd,
        }


class _RuntimeV1ReportProfile:
    profile_id = "runtime_v1"
    aliases = ("runtime_v1_results", "longbench_v2_runtime")

    def render(
        self,
        *,
        output_root: Path,
        output_dir: Path | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        script_path = REPO_ROOT / "scripts" / "report_runtime_v1_results.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--experiment-dir",
            str(Path(output_root).expanduser().resolve()),
            "--print-json",
        ]
        if output_dir is not None:
            cmd.extend(["--output-dir", str(Path(output_dir).expanduser().resolve())])
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        summary_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else Path(output_root).expanduser().resolve() / "paper_summary"
        )
        payload: dict[str, Any] = {}
        if int(result.returncode) == 0:
            try:
                payload = dict(json.loads(result.stdout or "{}"))
            except Exception:
                payload = {"stdout": result.stdout}
        return {
            **payload,
            "profile": self.profile_id,
            "returncode": int(result.returncode),
            "command": cmd,
            "stderr": str(result.stderr or ""),
            "json_path": str(summary_dir / "runtime_v1_summary.json"),
            "markdown_path": str(summary_dir / "runtime_v1_summary.md"),
        }


def _register_builtin_profiles() -> None:
    if "tradeoff" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("tradeoff", "report_markov_optimization_tradeoffs.py"))
    if "publication_bundle" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("publication_bundle", "report_markov_optimization_tradeoffs.py"))
    if "supervision_recovery" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("supervision_recovery", "report_markov_optimization_tradeoffs.py"))
    if "runtime_eval_summary" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("runtime_eval_summary", "run_runtime_eval.py"))
    if "runtime_v1" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_RuntimeV1ReportProfile())


def _build_parser() -> argparse.ArgumentParser:
    ensure_default_method_adapters()
    _register_builtin_profiles()
    parser = argparse.ArgumentParser(description="Canonical experiment control-plane entrypoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List adapters, report profiles, and supported entrypoints")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(fn=cmd_list)

    p_check = sub.add_parser("check", help="Run centralized readiness checks")
    p_check.add_argument("--suite", default="v1", choices=("v1",))
    p_check.add_argument("--config", default=str(REPO_ROOT / "config" / "runtime_eval" / "longbench_v2_smoke.yaml"))
    p_check.add_argument("--output-dir", default=None)
    p_check.add_argument("--experiment-id", default="longbench_v2_smoke")
    p_check.add_argument("--max-problems", type=int, default=2)
    p_check.add_argument("--live", action="store_true")
    p_check.add_argument("--check-endpoints", action="store_true")
    p_check.add_argument("--scorer-endpoint", default=None)
    p_check.add_argument("--scorer-model", default=None)
    p_check.add_argument("--skip-tests", action="store_true")
    p_check.add_argument("--test-target", action="append", default=None)
    p_check.add_argument("--report", action="store_true", help="Render the runtime_v1 paper summary after a passing check.")
    p_check.add_argument("--report-output-dir", default=None)
    p_check.add_argument("--json", action="store_true")
    p_check.set_defaults(fn=cmd_check)

    p_plan = sub.add_parser("plan", help="Resolve an invocation into experiment_manifest.json")
    p_plan.add_argument("--adapter", default=None, choices=METHOD_ADAPTERS.available())
    p_plan.add_argument("--cwd", default=str(REPO_ROOT))
    p_plan.add_argument("--summary", action="store_true", help="Print a concise human-readable plan.")
    p_plan.add_argument("command", nargs=argparse.REMAINDER)
    p_plan.set_defaults(fn=cmd_plan)

    p_launch = sub.add_parser("launch", help="Launch an invocation through the canonical control plane")
    p_launch.add_argument("--adapter", default=None, choices=METHOD_ADAPTERS.available())
    p_launch.add_argument("--cwd", default=str(REPO_ROOT))
    p_launch.add_argument("--detach", action="store_true")
    p_launch.add_argument("--name", default="")
    p_launch.add_argument("--description", default="")
    p_launch.add_argument("--job-root", default=None)
    p_launch.add_argument("--no-collect", dest="collect", action="store_false")
    p_launch.add_argument("--json", action="store_true", help="Print collection payload after foreground run.")
    p_launch.set_defaults(collect=True)
    p_launch.add_argument("command", nargs=argparse.REMAINDER)
    p_launch.set_defaults(fn=cmd_launch)

    p_resume = sub.add_parser("resume", help="Resume from an existing experiment_manifest.json")
    p_resume.add_argument("--manifest", required=True)
    p_resume.add_argument("--cwd", default=str(REPO_ROOT))
    p_resume.add_argument("--detach", action="store_true")
    p_resume.add_argument("--name", default="")
    p_resume.add_argument("--description", default="")
    p_resume.add_argument("--job-root", default=None)
    p_resume.add_argument("--no-collect", dest="collect", action="store_false")
    p_resume.add_argument("--json", action="store_true", help="Print collection payload after foreground run.")
    p_resume.set_defaults(collect=True)
    p_resume.set_defaults(fn=cmd_resume)

    p_collect = sub.add_parser("collect", help="Collect artifacts and refresh status for a finished run")
    p_collect.add_argument("--manifest", required=True)
    p_collect.add_argument("--adapter", default=None, choices=METHOD_ADAPTERS.available())
    p_collect.set_defaults(fn=cmd_collect)

    p_status = sub.add_parser("status", help="Read canonical experiment_status.json")
    p_status.add_argument("--manifest", default=None)
    p_status.add_argument("--output-root", default=None)
    p_status.set_defaults(fn=cmd_status)

    p_show = sub.add_parser("show", help="Show a concise experiment_manifest.json summary")
    p_show.add_argument("--manifest", required=True)
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(fn=cmd_show)

    p_report = sub.add_parser("report", help="Render a report profile against canonical artifacts")
    p_report.add_argument("--profile", required=True, choices=REPORT_PROFILES.available())
    p_report.add_argument("--output-root", required=True)
    p_report.add_argument("--report-output-dir", default=None)
    p_report.add_argument("extra_args", nargs=argparse.REMAINDER)
    p_report.set_defaults(fn=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "status" and args.manifest is None and args.output_root is None:
        raise SystemExit("status requires either --manifest or --output-root")
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
