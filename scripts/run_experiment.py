#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments import (  # noqa: E402
    METHOD_ADAPTERS,
    REPORT_PROFILES,
    ExperimentSpec,
    experiment_paths,
    write_experiment_manifest,
)
from src.experiments.control_plane import load_json  # noqa: E402


def _strip_remainder(items: Sequence[str]) -> list[str]:
    values = list(items)
    if values and values[0] == "--":
        return values[1:]
    return values


def _load_manifest(path: Path) -> ExperimentSpec:
    payload = load_json(path)
    if not payload:
        raise FileNotFoundError(f"experiment manifest not found or invalid: {path}")
    return ExperimentSpec.from_dict(payload)


def _infer_output_root_from_manifest(path: Path) -> Path:
    spec = _load_manifest(path)
    return Path(spec.output_root).expanduser().resolve()


def cmd_plan(args: argparse.Namespace) -> int:
    adapter = METHOD_ADAPTERS.resolve(str(args.adapter))
    command = _strip_remainder(args.command or [])
    if not command:
        raise SystemExit("missing legacy command after `--`")
    spec = adapter.build_experiment_spec(command, cwd=Path(args.cwd).resolve())
    manifest_path = write_experiment_manifest(spec.output_root, spec)
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
    adapter = METHOD_ADAPTERS.resolve(str(args.adapter))
    command = _strip_remainder(args.command or [])
    if not command:
        raise SystemExit("missing legacy command after `--`")
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
    return _run_command(command, cwd=Path(args.cwd).resolve())


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
    return _run_command(command, cwd=Path(args.cwd).resolve())


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


def _register_builtin_profiles() -> None:
    if "tradeoff" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("tradeoff", "report_markov_optimization_tradeoffs.py"))
    if "publication_bundle" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("publication_bundle", "report_markov_optimization_tradeoffs.py"))
    if "supervision_recovery" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("supervision_recovery", "report_markov_optimization_tradeoffs.py"))
    if "runtime_eval_summary" not in REPORT_PROFILES.available():
        REPORT_PROFILES.register(_LegacyReportProfile("runtime_eval_summary", "run_runtime_eval.py"))


def _build_parser() -> argparse.ArgumentParser:
    _register_builtin_profiles()
    parser = argparse.ArgumentParser(description="Canonical experiment control-plane entrypoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Resolve a legacy invocation into experiment_manifest.json")
    p_plan.add_argument("--adapter", required=True, choices=METHOD_ADAPTERS.available())
    p_plan.add_argument("--cwd", default=str(REPO_ROOT))
    p_plan.add_argument("command", nargs=argparse.REMAINDER)
    p_plan.set_defaults(fn=cmd_plan)

    p_launch = sub.add_parser("launch", help="Launch a legacy invocation through the canonical control plane")
    p_launch.add_argument("--adapter", required=True, choices=METHOD_ADAPTERS.available())
    p_launch.add_argument("--cwd", default=str(REPO_ROOT))
    p_launch.add_argument("--detach", action="store_true")
    p_launch.add_argument("--name", default="")
    p_launch.add_argument("--description", default="")
    p_launch.add_argument("--job-root", default=None)
    p_launch.add_argument("command", nargs=argparse.REMAINDER)
    p_launch.set_defaults(fn=cmd_launch)

    p_resume = sub.add_parser("resume", help="Resume from an existing experiment_manifest.json")
    p_resume.add_argument("--manifest", required=True)
    p_resume.add_argument("--cwd", default=str(REPO_ROOT))
    p_resume.add_argument("--detach", action="store_true")
    p_resume.add_argument("--name", default="")
    p_resume.add_argument("--description", default="")
    p_resume.add_argument("--job-root", default=None)
    p_resume.set_defaults(fn=cmd_resume)

    p_status = sub.add_parser("status", help="Read canonical experiment_status.json")
    p_status.add_argument("--manifest", default=None)
    p_status.add_argument("--output-root", default=None)
    p_status.set_defaults(fn=cmd_status)

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
