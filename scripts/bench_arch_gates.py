#!/usr/bin/env python3
"""
Run architecture quality gates (Manifesto + RULER) and persist a baseline artifact.

Phase 0 utility:
- records command recipes
- executes both gates (unless skipped)
- writes `gate_metrics.json` under outputs/arch_baseline_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

import yaml


DEFAULT_MANIFESTO_IDS = ["51320_198306", "51620_198306", "51320_199705"]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class CommandResult:
    name: str
    command: List[str]
    returncode: int
    log_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "returncode": int(self.returncode),
            "log_path": self.log_path,
        }


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": bool(self.ok), "detail": self.detail}


def _run_command(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(str(part)) for part in command)

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Command: {rendered}\n\n")
        if dry_run:
            handle.write("Dry-run mode: command not executed.\n")
            return CommandResult(name=name, command=[str(p) for p in command], returncode=0, log_path=str(log_path))

        proc = subprocess.run(
            [str(p) for p in command],
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return CommandResult(
        name=name,
        command=[str(p) for p in command],
        returncode=int(proc.returncode),
        log_path=str(log_path),
    )


def _manifesto_mae(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    gaps: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("absolute_gap_rile")
        try:
            if raw is not None:
                gaps.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def _ruler_primary_mean(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(payload.get("primary_mean"))
    except Exception:
        return None


def _endpoint_ready(base_url: str, timeout_seconds: float = 2.0) -> bool:
    target = str(base_url).rstrip("/") + "/models"
    try:
        with urlopen(target, timeout=max(0.25, float(timeout_seconds))) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except (URLError, OSError, ValueError):
        return False


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_ruler_tasks(ruler_cfg: Dict[str, Any], phase_id: str) -> List[str]:
    for phase in ruler_cfg.get("phases", []):
        if str(phase.get("phase_id", "")) == str(phase_id):
            return [str(task) for task in phase.get("tasks", [])]
    return []


def _resolve_task_required_files(*, ruler_dir: Path, task_name: str, synthetic_cfg: Dict[str, Any]) -> List[Path]:
    task_spec = synthetic_cfg.get(task_name, {}) if isinstance(synthetic_cfg, dict) else {}
    task_type = str(task_spec.get("task", "")).strip().lower()
    task_args = task_spec.get("args", {}) if isinstance(task_spec.get("args", {}), dict) else {}
    json_dir = ruler_dir / "scripts" / "data" / "synthetic" / "json"

    required: List[Path] = []
    if task_type in {"niah", "variable_tracking"}:
        haystack = str(task_args.get("type_haystack", "")).strip().lower()
        if haystack == "essay":
            required.append(json_dir / "PaulGrahamEssays.json")
    elif task_type == "qa":
        dataset = str(task_args.get("dataset", "")).strip().lower()
        if dataset == "squad":
            required.append(json_dir / "squad.json")
        elif dataset == "hotpotqa":
            required.append(json_dir / "hotpotqa.json")
    elif task_type == "common_words_extraction":
        required.append(json_dir / "english_words.json")

    return required


def _manifesto_preflight(*, repo_root: Path, port: int, dry_run: bool) -> Dict[str, Any]:
    checks: List[PreflightCheck] = []

    script_path = repo_root / "scripts" / "run_manifesto_batched_example.py"
    checks.append(
        PreflightCheck(
            name="manifesto_script_exists",
            ok=script_path.exists(),
            detail=str(script_path),
        )
    )

    endpoint = f"http://localhost:{int(port)}/v1"
    if dry_run:
        checks.append(
            PreflightCheck(
                name="manifesto_endpoint_ready",
                ok=True,
                detail=f"dry-run: skipped check ({endpoint})",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="manifesto_endpoint_ready",
                ok=_endpoint_ready(endpoint),
                detail=endpoint,
            )
        )

    ok = all(check.ok for check in checks)
    return {"ok": ok, "checks": [check.to_dict() for check in checks]}


def _ruler_preflight(
    *,
    repo_root: Path,
    ruler_config: Path,
    phase_id: str,
    dry_run: bool,
    mock_llm: bool,
    model_base_url_override: Optional[str] = None,
    start_server_profile: Optional[str] = None,
) -> Dict[str, Any]:
    checks: List[PreflightCheck] = []
    config_path = ruler_config if ruler_config.is_absolute() else (repo_root / ruler_config)
    checks.append(
        PreflightCheck(
            name="ruler_config_exists",
            ok=config_path.exists(),
            detail=str(config_path),
        )
    )
    if not config_path.exists():
        return {"ok": False, "checks": [check.to_dict() for check in checks]}

    cfg = _load_yaml(config_path)
    benchmark_cfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    ruler_dir = Path(benchmark_cfg.get("ruler_dir", "outside_data/RULER"))
    if not ruler_dir.is_absolute():
        ruler_dir = (repo_root / ruler_dir).resolve()

    checks.append(
        PreflightCheck(
            name="ruler_dir_exists",
            ok=ruler_dir.exists(),
            detail=str(ruler_dir),
        )
    )
    synth_yaml = ruler_dir / "scripts" / "synthetic.yaml"
    checks.append(
        PreflightCheck(
            name="ruler_synthetic_yaml_exists",
            ok=synth_yaml.exists(),
            detail=str(synth_yaml),
        )
    )
    if not synth_yaml.exists():
        return {"ok": False, "checks": [check.to_dict() for check in checks]}

    synthetic_cfg = _load_yaml(synth_yaml)
    phase_tasks = _resolve_ruler_tasks(cfg, phase_id)
    checks.append(
        PreflightCheck(
            name="ruler_phase_found",
            ok=bool(phase_tasks),
            detail=f"phase={phase_id} tasks={phase_tasks}",
        )
    )

    required_files: List[Path] = []
    for task_name in phase_tasks:
        required_files.extend(
            _resolve_task_required_files(
                ruler_dir=ruler_dir,
                task_name=task_name,
                synthetic_cfg=synthetic_cfg,
            )
        )

    dedup_required = sorted({path.resolve() for path in required_files})
    for required in dedup_required:
        checks.append(
            PreflightCheck(
                name=f"ruler_required_file:{required.name}",
                ok=required.exists(),
                detail=str(required),
            )
        )

    if mock_llm or dry_run or bool(start_server_profile):
        checks.append(
            PreflightCheck(
                name="ruler_endpoint_ready",
                ok=True,
                detail="mock-llm/dry-run/start-server: skipped endpoint check",
            )
        )
    else:
        model_base = str(model_base_url_override or model_cfg.get("base_url", "http://localhost:8000/v1"))
        checks.append(
            PreflightCheck(
                name="ruler_endpoint_ready",
                ok=_endpoint_ready(model_base),
                detail=model_base,
            )
        )

    ok = all(check.ok for check in checks)
    return {"ok": ok, "checks": [check.to_dict() for check in checks]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run architecture gate benchmarks and capture baseline metrics.")
    parser.add_argument("--python-bin", type=Path, default=None, help="Python executable for child benchmark commands.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root output directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory override.")
    parser.add_argument("--manifesto-port", type=int, default=8000, help="Manifesto task model port.")
    parser.add_argument("--manifesto-chunk-size", type=int, default=8000, help="Chunk size for manifesto baseline.")
    parser.add_argument("--manifesto-ids", nargs="+", default=DEFAULT_MANIFESTO_IDS, help="Manifesto IDs.")
    parser.add_argument("--ruler-config", type=Path, default=Path("config/runtime_eval/ruler_8k_freeform.yaml"))
    parser.add_argument("--ruler-phase", type=str, default="S0_smoke")
    parser.add_argument("--ruler-mode", type=str, default="runtime_full")
    parser.add_argument("--ruler-max-units", type=int, default=1)
    parser.add_argument("--ruler-max-problems", type=int, default=25)
    parser.add_argument("--ruler-run-id", type=str, default="baseline_ruler")
    parser.add_argument("--ruler-backend", type=str, choices=["vllm", "sglang"], default=None)
    parser.add_argument("--ruler-backend-fallback", type=str, choices=["none", "vllm", "sglang"], default="vllm")
    parser.add_argument("--ruler-model-base-url", type=str, default=None)
    parser.add_argument("--ruler-start-server", type=str, default=None)
    parser.add_argument("--ruler-server-port", type=int, default=None)
    parser.add_argument("--ruler-cuda-devices", type=str, default=None)
    parser.add_argument("--ruler-vllm-venv-path", type=str, default=None)
    parser.add_argument("--ruler-sglang-venv-path", type=str, default=None)
    parser.add_argument("--auto-prepare-ruler-data", action="store_true")
    parser.add_argument("--skip-manifesto", action="store_true")
    parser.add_argument("--skip-ruler", action="store_true")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM for RULER run.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    default_venv_python = repo_root / "venv" / "bin" / "python"
    if args.python_bin is not None:
        python_bin = str(Path(args.python_bin).resolve())
    elif default_venv_python.exists():
        python_bin = str(default_venv_python)
    else:
        python_bin = str(sys.executable)
    run_dir = args.run_dir or (args.output_root / f"arch_baseline_{_utc_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)

    commands: List[CommandResult] = []
    status: Dict[str, Any] = {"manifesto": "skipped", "ruler": "skipped"}
    preflight: Dict[str, Any] = {"manifesto": {"ok": True, "checks": []}, "ruler": {"ok": True, "checks": []}}

    if args.auto_prepare_ruler_data and not args.skip_ruler:
        setup_cmd: List[str] = [
            python_bin,
            "scripts/setup_ruler_data.py",
            "--ruler-dir",
            str((repo_root / "outside_data" / "RULER").resolve()),
        ]
        setup_result = _run_command(
            name="ruler_setup_data",
            command=setup_cmd,
            cwd=repo_root,
            log_path=run_dir / "ruler_setup_data.log",
            dry_run=bool(args.dry_run),
        )
        commands.append(setup_result)

    if not args.skip_manifesto:
        preflight["manifesto"] = _manifesto_preflight(
            repo_root=repo_root,
            port=int(args.manifesto_port),
            dry_run=bool(args.dry_run),
        )
    if not args.skip_ruler:
        preflight["ruler"] = _ruler_preflight(
            repo_root=repo_root,
            ruler_config=Path(args.ruler_config),
            phase_id=str(args.ruler_phase),
            dry_run=bool(args.dry_run),
            mock_llm=bool(args.mock_llm),
            model_base_url_override=(str(args.ruler_model_base_url) if args.ruler_model_base_url else None),
            start_server_profile=(str(args.ruler_start_server) if args.ruler_start_server else None),
        )

    manifesto_output = run_dir / "manifesto_baseline.json"
    if not args.skip_manifesto:
        if not preflight["manifesto"]["ok"]:
            status["manifesto"] = "failed_preflight"
        else:
            manifesto_cmd: List[str] = [
                python_bin,
                "scripts/run_manifesto_batched_example.py",
                "--ids",
                *[str(doc_id) for doc_id in args.manifesto_ids],
                "--chunk-size",
                str(int(args.manifesto_chunk_size)),
                "--port",
                str(int(args.manifesto_port)),
                "--output",
                str(manifesto_output),
            ]
            result = _run_command(
                name="manifesto_gate",
                command=manifesto_cmd,
                cwd=repo_root,
                log_path=run_dir / "manifesto_gate.log",
                dry_run=bool(args.dry_run),
            )
            commands.append(result)
            status["manifesto"] = "ok" if result.returncode == 0 else "failed"

    ruler_metrics_path = run_dir / "ruler" / args.ruler_run_id / "metrics.json"
    if not args.skip_ruler:
        if not preflight["ruler"]["ok"]:
            status["ruler"] = "failed_preflight"
        else:
            ruler_init_cmd: List[str] = [
                python_bin,
                "scripts/run_runtime_eval.py",
                "init",
                "--config",
                str(args.ruler_config),
                "--output-dir",
                str(run_dir / "ruler"),
                "--run-id",
                str(args.ruler_run_id),
            ]
            commands.append(
                _run_command(
                    name="ruler_init",
                    command=ruler_init_cmd,
                    cwd=repo_root,
                    log_path=run_dir / "ruler_init.log",
                    dry_run=bool(args.dry_run),
                )
            )

            ruler_run_cmd: List[str] = [
                python_bin,
                "scripts/run_runtime_eval.py",
                "run",
                "--run-dir",
                str(run_dir / "ruler" / args.ruler_run_id),
                "--phase-id",
                str(args.ruler_phase),
                "--mode",
                str(args.ruler_mode),
                "--max-units",
                str(int(args.ruler_max_units)),
                "--max-problems",
                str(int(args.ruler_max_problems)),
            ]
            if args.ruler_backend:
                ruler_run_cmd.extend(["--backend", str(args.ruler_backend)])
            if args.ruler_backend_fallback:
                ruler_run_cmd.extend(["--backend-fallback", str(args.ruler_backend_fallback)])
            if args.ruler_model_base_url:
                ruler_run_cmd.extend(["--model-base-url", str(args.ruler_model_base_url)])
            if args.ruler_start_server:
                ruler_run_cmd.extend(["--start-server", str(args.ruler_start_server)])
            if args.ruler_server_port is not None:
                ruler_run_cmd.extend(["--server-port", str(int(args.ruler_server_port))])
            if args.ruler_cuda_devices:
                ruler_run_cmd.extend(["--cuda-devices", str(args.ruler_cuda_devices)])
            if args.ruler_vllm_venv_path:
                ruler_run_cmd.extend(["--vllm-venv-path", str(args.ruler_vllm_venv_path)])
            if args.ruler_sglang_venv_path:
                ruler_run_cmd.extend(["--sglang-venv-path", str(args.ruler_sglang_venv_path)])
            if args.mock_llm:
                ruler_run_cmd.append("--mock-llm")
            commands.append(
                _run_command(
                    name="ruler_run",
                    command=ruler_run_cmd,
                    cwd=repo_root,
                    log_path=run_dir / "ruler_run.log",
                    dry_run=bool(args.dry_run),
                )
            )

            ruler_agg_cmd: List[str] = [
                python_bin,
                "scripts/run_runtime_eval.py",
                "aggregate",
                "--run-dir",
                str(run_dir / "ruler" / args.ruler_run_id),
            ]
            commands.append(
                _run_command(
                    name="ruler_aggregate",
                    command=ruler_agg_cmd,
                    cwd=repo_root,
                    log_path=run_dir / "ruler_aggregate.log",
                    dry_run=bool(args.dry_run),
                )
            )

            ruler_codes = [cmd.returncode for cmd in commands if cmd.name.startswith("ruler_")]
            status["ruler"] = "ok" if all(code == 0 for code in ruler_codes) else "failed"

    manifesto_mae = _manifesto_mae(manifesto_output)
    ruler_primary_mean = _ruler_primary_mean(ruler_metrics_path)
    payload = {
        "run_dir": str(run_dir),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "preflight": preflight,
        "metrics": {
            "manifesto_mae": manifesto_mae,
            "ruler_primary_mean": ruler_primary_mean,
        },
        "artifacts": {
            "manifesto_output": str(manifesto_output),
            "ruler_metrics": str(ruler_metrics_path),
        },
        "commands": [cmd.to_dict() for cmd in commands],
    }

    metrics_path = run_dir / "gate_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(
        "Manifesto MAE:", "n/a" if manifesto_mae is None else f"{manifesto_mae:.4f}",
        "| RULER primary_mean:", "n/a" if ruler_primary_mean is None else f"{ruler_primary_mean:.4f}",
    )

    failed = [cmd for cmd in commands if cmd.returncode != 0]
    if not args.skip_manifesto and status.get("manifesto") == "failed_preflight":
        failed.append(CommandResult(name="manifesto_preflight", command=[], returncode=1, log_path=""))
    if not args.skip_ruler and status.get("ruler") == "failed_preflight":
        failed.append(CommandResult(name="ruler_preflight", command=[], returncode=1, log_path=""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
