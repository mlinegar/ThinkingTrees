#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


START_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+start:\s+(?P<step>.+)$")
DONE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+done:\s+(?P<step>.+)$")
FAIL_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+FAIL\((?P<code>\d+)\):\s+(?P<step>.+)$")
CMD_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+cmd:\s+(?P<cmd>.+)$")
GEPA_PROGRESS_RE = re.compile(
    r"GEPA Optimization:\s+(?P<pct>\d+)%.*?\|\s*(?P<current>\d+)/(?P<total>\d+)\s*\[.*?,\s*(?:(?P<sec>[0-9.]+)s/rollouts|(?P<rps>[0-9.]+)rollouts/s)\]"
)
ITER_RE = re.compile(r"Iteration\s+(?P<iteration>\d+):")
BEST_RE = re.compile(r"Best valset aggregate score so far:\s+(?P<score>[0-9.]+)")
FULL_VAL_RE = re.compile(r"Full valset score for new program:\s+(?P<score>[0-9.]+)")


@dataclass
class ActiveStep:
    name: str
    started_at: Optional[str] = None
    cmd: Optional[str] = None
    output_dir: Optional[str] = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_datetime(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except Exception:
        return None


def _elapsed_seconds(started_at: Optional[str]) -> Optional[float]:
    start_dt = _safe_datetime(started_at)
    if start_dt is None:
        return None
    now = datetime.now(timezone.utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - start_dt.astimezone(timezone.utc)).total_seconds())


def _extract_output_dir_from_cmd(cmd: str) -> Optional[str]:
    try:
        tokens = shlex.split(cmd)
    except Exception:
        return None
    for idx, token in enumerate(tokens):
        if token == "--output-dir" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def _parse_active_step(log_path: Path) -> tuple[list[dict[str, Any]], Optional[ActiveStep]]:
    if not log_path.exists():
        return [], None
    completed: list[dict[str, Any]] = []
    active: Optional[ActiveStep] = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        m = START_RE.match(line)
        if m:
            active = ActiveStep(name=m.group("step").strip(), started_at=m.group("ts").strip())
            continue
        m = CMD_RE.match(line)
        if m and active is not None and active.cmd is None:
            active.cmd = m.group("cmd").strip()
            active.output_dir = _extract_output_dir_from_cmd(active.cmd or "")
            continue
        m = DONE_RE.match(line)
        if m:
            completed.append(
                {
                    "step": m.group("step").strip(),
                    "status": "completed",
                    "finished_at": m.group("ts").strip(),
                }
            )
            if active is not None and active.name == m.group("step").strip():
                active = None
            continue
        m = FAIL_RE.match(line)
        if m:
            completed.append(
                {
                    "step": m.group("step").strip(),
                    "status": f"failed:{m.group('code')}",
                    "finished_at": m.group("ts").strip(),
                }
            )
            if active is not None and active.name == m.group("step").strip():
                active = None
    return completed, active


def _load_phase2_state(step_output_dir: Path) -> Optional[dict[str, Any]]:
    phase2_dir = step_output_dir / "checkpoints" / "phase2_runtime"
    if not phase2_dir.exists():
        return None
    candidates = sorted(phase2_dir.glob("*/state.json"))
    if not candidates:
        return None
    try:
        return _read_json(candidates[-1])
    except Exception:
        return None


def _load_pipeline_runtime_state(step_output_dir: Path) -> Optional[dict[str, Any]]:
    path = step_output_dir / "checkpoints" / "pipeline_runtime_state.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _load_progress_state(step_output_dir: Path) -> Optional[dict[str, Any]]:
    path = step_output_dir / "checkpoints" / "progress.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _load_gepa_snapshot(step_output_dir: Path, component: str) -> Optional[dict[str, Any]]:
    path = step_output_dir / "checkpoints" / "gepa" / str(component) / "gepa_trajectory_snapshot.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _parse_latest_gepa_from_log(log_path: Path) -> dict[str, Any]:
    latest_progress: Optional[dict[str, Any]] = None
    latest_iteration: Optional[int] = None
    latest_best: Optional[float] = None
    latest_full_val: Optional[float] = None
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    for line in text.splitlines():
        m = ITER_RE.search(line)
        if m:
            latest_iteration = int(m.group("iteration"))
        m = BEST_RE.search(line)
        if m:
            latest_best = float(m.group("score"))
        m = FULL_VAL_RE.search(line)
        if m:
            latest_full_val = float(m.group("score"))
        m = GEPA_PROGRESS_RE.search(line)
        if m:
            sec_per_rollout: Optional[float] = None
            if m.group("sec") is not None:
                sec_per_rollout = float(m.group("sec"))
            elif m.group("rps") is not None and float(m.group("rps")) > 0:
                sec_per_rollout = 1.0 / float(m.group("rps"))
            current = int(m.group("current"))
            total = int(m.group("total"))
            eta_seconds = None
            if sec_per_rollout is not None:
                eta_seconds = max(0.0, float(total - current) * sec_per_rollout)
            latest_progress = {
                "percent": int(m.group("pct")),
                "current_rollouts": current,
                "total_rollouts": total,
                "sec_per_rollout": sec_per_rollout,
                "eta_seconds": eta_seconds,
            }
    payload: dict[str, Any] = {}
    if latest_progress is not None:
        payload["gepa_progress"] = latest_progress
    if latest_iteration is not None:
        payload["latest_gepa_iteration"] = latest_iteration
    if latest_best is not None:
        payload["best_valset_aggregate_score"] = latest_best
    if latest_full_val is not None:
        payload["latest_full_valset_score"] = latest_full_val
    return payload


def _build_convergence_from_snapshot(snapshot: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    scores = list(snapshot.get("candidate_scores") or [])
    discovery = list(snapshot.get("discovery_metric_calls") or [])
    points: list[dict[str, Any]] = []
    best_so_far: Optional[float] = None
    for idx, score in enumerate(scores):
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        metric_calls = None
        if idx < len(discovery):
            try:
                metric_calls = int(discovery[idx])
            except (TypeError, ValueError):
                metric_calls = None
        best_so_far = score_f if best_so_far is None else max(best_so_far, score_f)
        points.append(
            {
                "candidate_index": idx,
                "metric_calls": metric_calls,
                "score": score_f,
                "best_score_so_far": best_so_far,
            }
        )
    return points


def summarize_output_root(output_root: Path) -> dict[str, Any]:
    log_path = output_root / "overnight.log"
    completed_steps, active_step = _parse_active_step(log_path)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "completed_steps": completed_steps,
    }
    if active_step is not None:
        payload["active_step"] = {
            "name": active_step.name,
            "started_at": active_step.started_at,
            "elapsed_seconds": _elapsed_seconds(active_step.started_at),
            "output_dir": active_step.output_dir,
        }
    else:
        payload["active_step"] = None

    if active_step is None or not active_step.output_dir:
        payload["live"] = {}
        return payload

    step_output_dir = Path(active_step.output_dir)
    pipeline_state = _load_pipeline_runtime_state(step_output_dir)
    phase2_state = _load_phase2_state(step_output_dir)
    progress_state = _load_progress_state(step_output_dir)
    log_state = _parse_latest_gepa_from_log(log_path)

    live: dict[str, Any] = {
        "step_output_dir": str(step_output_dir),
        "pipeline_runtime_state": pipeline_state or {},
        "progress_state": progress_state or {},
    }

    if isinstance(phase2_state, dict):
        live["phase2_runtime_state"] = {
            "status": phase2_state.get("status"),
            "started_at": phase2_state.get("started_at"),
            "updated_at": phase2_state.get("updated_at"),
            "latest_completed_iteration": phase2_state.get("latest_completed_iteration"),
        }
        iterations = phase2_state.get("iterations") or {}
        current_iter_key = sorted(iterations.keys(), key=lambda x: int(x))[-1] if iterations else None
        if current_iter_key is not None:
            current_iter = iterations.get(current_iter_key) or {}
            components = current_iter.get("components") or {}
            component_summary: dict[str, Any] = {}
            active_component_name: Optional[str] = None
            for component_name, component_state in components.items():
                if not isinstance(component_state, dict):
                    continue
                component_summary[component_name] = {
                    "status": component_state.get("status"),
                    "started_at": component_state.get("started_at"),
                    "updated_at": component_state.get("updated_at"),
                    "metric_before": component_state.get("metric_before"),
                    "metric_after": component_state.get("metric_after"),
                    "optimizer_used": component_state.get("optimizer_used"),
                }
                if component_state.get("status") == "running":
                    active_component_name = str(component_name)
            live["phase2_current_iteration"] = {
                "round": current_iter.get("round"),
                "status": current_iter.get("status"),
                "started_at": current_iter.get("started_at"),
                "components": component_summary,
                "active_component": active_component_name,
            }
            if active_component_name:
                snapshot = _load_gepa_snapshot(step_output_dir, active_component_name)
                if snapshot is not None:
                    live["active_component_gepa_snapshot"] = {
                        "component": snapshot.get("component"),
                        "iteration_index": snapshot.get("iteration_index"),
                        "num_candidates": snapshot.get("num_candidates"),
                        "best_candidate_idx": snapshot.get("best_candidate_idx"),
                        "best_candidate_score": snapshot.get("best_candidate_score"),
                        "total_metric_calls": snapshot.get("total_metric_calls"),
                        "num_full_val_evals": snapshot.get("num_full_val_evals"),
                    }
                    live["active_component_convergence"] = _build_convergence_from_snapshot(snapshot)

    live.update(log_state)
    payload["live"] = live
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize live progress for unified-G suite runs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout.")
    args = parser.parse_args()

    payload = summarize_output_root(args.output_root)
    json_text = json.dumps(payload, indent=2)

    json_out = args.json_out or (args.output_root / "live_progress.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json_text + "\n", encoding="utf-8")

    if args.stdout:
        print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
