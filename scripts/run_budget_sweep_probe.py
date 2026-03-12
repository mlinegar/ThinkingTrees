#!/usr/bin/env python3
"""Run training pipeline across optimizer budgets and summarize quality/cost scaling."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_budgets(raw: str) -> List[str]:
    out: List[str] = []
    for part in str(raw).split(","):
        b = part.strip()
        if b:
            out.append(b)
    return out


def _parse_max_calls_map(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        key = k.strip()
        val = v.strip()
        if not key:
            continue
        try:
            out[key] = int(val)
        except ValueError:
            continue
    return out


def _run_budget(
    *,
    run_script: Path,
    run_dir: Path,
    budget: str,
    task: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    n_iterations: int,
    routing_policy: str,
    genrm_init_samples: int,
    max_metric_calls: Optional[int],
    start_server: bool,
    start_genrm: bool,
    engram_memory: bool,
    keep_servers_running: bool,
    timeout_seconds: Optional[float],
    extra_args: List[str],
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "probe_command.log"

    cmd: List[str] = [
        str(run_script),
        "--task",
        str(task),
        "--train-samples",
        str(int(train_samples)),
        "--val-samples",
        str(int(val_samples)),
        "--test-samples",
        str(int(test_samples)),
        "--n-iterations",
        str(int(n_iterations)),
        "--optimizer-budget",
        str(budget),
        "--routing-policy",
        str(routing_policy),
        "--genrm-init-samples",
        str(int(genrm_init_samples)),
        "--output-dir",
        str(run_dir),
    ]
    if max_metric_calls is not None:
        cmd.extend(["--max-metric-calls", str(int(max_metric_calls))])
    if start_server:
        cmd.append("--start-server")
    if start_genrm:
        cmd.append("--start-genrm")
    if engram_memory:
        cmd.append("--engram-memory")
    if keep_servers_running:
        cmd.append("--keep-servers-running")
    cmd.extend(extra_args)

    started_at = _utc_now_iso()
    t0 = time.perf_counter()
    returncode = 2
    timed_out = False
    error: Optional[str] = None
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                text=True,
            )
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            error = "timed out"
        except Exception as exc:
            returncode = 2
            error = f"command execution error: {exc}"

    duration_seconds = float(time.perf_counter() - t0)
    final_stats_path = run_dir / "final_stats.json"
    stats = _load_json(final_stats_path)
    success = bool(stats.get("success")) if isinstance(stats, dict) else False

    test_block = stats.get("test", {}) if isinstance(stats, dict) else {}
    pred_dist = test_block.get("prediction_distribution", {}) if isinstance(test_block, dict) else {}

    return {
        "budget": str(budget),
        "started_at": started_at,
        "completed_at": _utc_now_iso(),
        "duration_seconds": duration_seconds,
        "status": "ok" if returncode == 0 and success else "failed",
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "error": error,
        "command": cmd,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "final_stats_path": str(final_stats_path),
        "pipeline_success": bool(success),
        "metrics": {
            "test_mae": _to_float(test_block.get("mae")) if isinstance(test_block, dict) else None,
            "test_pearson_r": _to_float(test_block.get("pearson_r")) if isinstance(test_block, dict) else None,
            "test_frac_neutral": _to_float(pred_dist.get("frac_neutral"))
            if isinstance(pred_dist, dict)
            else None,
            "test_n_evaluated": test_block.get("n_evaluated") if isinstance(test_block, dict) else None,
        },
    }


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return float(num / den)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run optimizer budget sweep probe and summarize scaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-script", type=Path, default=Path("scripts/run_training_pipeline.sh"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)

    parser.add_argument("--task", type=str, default="manifesto_rile")
    parser.add_argument("--budgets", type=str, default="light,medium,heavy")
    parser.add_argument("--max-metric-calls-map", type=str, default="light=10,medium=20,heavy=30")
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=3)
    parser.add_argument("--test-samples", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument("--routing-policy", type=str, default="affinity_load_aware")
    parser.add_argument("--genrm-init-samples", type=int, default=3)

    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--start-genrm", action="store_true")
    parser.add_argument("--engram-memory", action="store_true")
    parser.add_argument(
        "--keep-servers-between-runs",
        action="store_true",
        help="Pass --keep-servers-running on all but final run for faster sweeps.",
    )
    parser.add_argument("--timeout-seconds-per-run", type=float, default=21600.0)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg passed through to each run_training_pipeline.sh invocation.",
    )
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--require-heavy-quality-gain", action="store_true")
    args = parser.parse_args()

    run_script = args.run_script
    if not run_script.is_absolute():
        run_script = (PROJECT_ROOT / run_script).resolve()
    if not run_script.exists():
        raise FileNotFoundError(f"Run script not found: {run_script}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = args.json_out
    if json_out is None:
        json_out = output_dir / "budget_sweep_probe.json"
    elif not json_out.is_absolute():
        json_out = (PROJECT_ROOT / json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)

    budgets = _parse_budgets(args.budgets)
    max_calls_map = _parse_max_calls_map(args.max_metric_calls_map)

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    runs: List[Dict[str, Any]] = []
    for idx, budget in enumerate(budgets):
        is_first = idx == 0
        is_last = idx == (len(budgets) - 1)
        keep_for_this_run = bool(args.keep_servers_between_runs) and (not is_last)

        run = _run_budget(
            run_script=run_script,
            run_dir=output_dir / f"budget_{budget}",
            budget=budget,
            task=args.task,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            n_iterations=args.n_iterations,
            routing_policy=args.routing_policy,
            genrm_init_samples=args.genrm_init_samples,
            max_metric_calls=max_calls_map.get(budget),
            start_server=bool(args.start_server) and is_first,
            start_genrm=bool(args.start_genrm) and is_first,
            engram_memory=bool(args.engram_memory),
            keep_servers_running=keep_for_this_run,
            timeout_seconds=float(args.timeout_seconds_per_run)
            if args.timeout_seconds_per_run and args.timeout_seconds_per_run > 0
            else None,
            extra_args=list(args.extra_arg or []),
        )
        runs.append(run)

    by_budget = {str(run.get("budget")): run for run in runs}
    light = by_budget.get("light", {})
    heavy = by_budget.get("heavy", {})

    light_mae = _to_float(((light.get("metrics") or {}).get("test_mae") if isinstance(light, dict) else None))
    heavy_mae = _to_float(((heavy.get("metrics") or {}).get("test_mae") if isinstance(heavy, dict) else None))
    light_pearson = _to_float(
        ((light.get("metrics") or {}).get("test_pearson_r") if isinstance(light, dict) else None)
    )
    heavy_pearson = _to_float(
        ((heavy.get("metrics") or {}).get("test_pearson_r") if isinstance(heavy, dict) else None)
    )
    light_neutral = _to_float(
        ((light.get("metrics") or {}).get("test_frac_neutral") if isinstance(light, dict) else None)
    )
    heavy_neutral = _to_float(
        ((heavy.get("metrics") or {}).get("test_frac_neutral") if isinstance(heavy, dict) else None)
    )
    light_dur = _to_float(light.get("duration_seconds") if isinstance(light, dict) else None)
    heavy_dur = _to_float(heavy.get("duration_seconds") if isinstance(heavy, dict) else None)

    quality_gain_heavy_vs_light = None
    if light_mae is not None and heavy_mae is not None:
        quality_gain_heavy_vs_light = float(light_mae - heavy_mae)

    pearson_gain_heavy_vs_light = None
    if light_pearson is not None and heavy_pearson is not None:
        pearson_gain_heavy_vs_light = float(heavy_pearson - light_pearson)

    neutral_reduction_heavy_vs_light = None
    if light_neutral is not None and heavy_neutral is not None:
        neutral_reduction_heavy_vs_light = float(light_neutral - heavy_neutral)

    duration_ratio_heavy_vs_light = _safe_div(heavy_dur, light_dur)

    success_count = sum(1 for run in runs if str(run.get("status")) == "ok")
    all_ok = success_count == len(runs)
    heavy_quality_not_worse = bool(quality_gain_heavy_vs_light is not None and quality_gain_heavy_vs_light >= 0)

    payload: Dict[str, Any] = {
        "created_at": started_at,
        "completed_at": _utc_now_iso(),
        "duration_seconds": float(time.perf_counter() - t0),
        "summary": {
            "runs": runs,
            "budgets": budgets,
            "success_count": int(success_count),
            "run_count": len(runs),
            "all_success_numeric": 1 if all_ok else 0,
            "quality_gain_heavy_vs_light": quality_gain_heavy_vs_light,
            "pearson_gain_heavy_vs_light": pearson_gain_heavy_vs_light,
            "neutral_reduction_heavy_vs_light": neutral_reduction_heavy_vs_light,
            "duration_ratio_heavy_vs_light": duration_ratio_heavy_vs_light,
            "checks": {
                "all_success": bool(all_ok),
                "heavy_quality_not_worse": bool(heavy_quality_not_worse),
                "all_success_numeric": 1 if all_ok else 0,
                "heavy_quality_not_worse_numeric": 1 if heavy_quality_not_worse else 0,
            },
        },
    }

    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved budget sweep probe JSON: {json_out}")

    if args.require_all_success and not all_ok:
        print("Probe status: failed (one or more budget runs failed)")
        return 2
    if args.require_heavy_quality_gain and not heavy_quality_not_worse:
        print("Probe status: failed (heavy quality did not improve over light)")
        return 3

    print("Probe status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
