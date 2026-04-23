#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.scheduler import cleanup_orphan_processes, matching_processes
from src.ctreepo.sim.util import safe_int as _safe_int


SPAWN_DETACHED_SCRIPT = REPO_ROOT / "scripts" / "spawn_detached_cmd.py"
DEFAULT_ROOT_DIR = REPO_ROOT / "logs" / "long_jobs"
COMBINED_STATUS_NAME = "combined_scheduler_status.json"
DEFAULT_PACKAGE_CAPACITY_ITEMS_ESTIMATE = 98
LAUNCH_BACKEND_AUTO = "auto"
LAUNCH_BACKEND_DOUBLE_FORK = "double_fork"
LAUNCH_BACKEND_SYSTEMD = "systemd"
LAUNCH_BACKEND_CHOICES = (
    LAUNCH_BACKEND_AUTO,
    LAUNCH_BACKEND_DOUBLE_FORK,
    LAUNCH_BACKEND_SYSTEMD,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(text: str) -> str:
    chars = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "job"


def _strip_remainder(cmd: Sequence[str]) -> List[str]:
    items = list(cmd)
    if items and items[0] == "--":
        return items[1:]
    return items


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _pid_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lookup_pgid(pid: int) -> int:
    if pid <= 0:
        return 0
    try:
        return int(os.getpgid(pid))
    except OSError:
        return 0


def _parse_env_assignments(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"invalid env assignment: {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid env assignment: {raw!r}")
        out[key] = value
    return out


def _render_runner_script(
    *,
    path: Path,
    cwd: Path,
    command: Sequence[str],
    env: Mapping[str, str],
    manifest_path: Path | None = None,
    refresh_python_bin: str | None = None,
    refresh_interval_seconds: float = 0.0,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(cwd))}",
    ]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append("cmd=(")
    for item in command:
        lines.append(f"  {shlex.quote(str(item))}")
    lines.append(")")
    refresh_enabled = (
        manifest_path is not None
        and bool(str(refresh_python_bin or "").strip())
        and float(refresh_interval_seconds) > 0.0
    )
    if refresh_enabled:
        refresh_script = REPO_ROOT / "scripts" / "long_job.py"
        lines.extend(
            [
                "refresh_cmd=(",
                f"  {shlex.quote(str(refresh_python_bin))}",
                f"  {shlex.quote(str(refresh_script))}",
                "  refresh",
                "  --manifest",
                f"  {shlex.quote(str(manifest_path))}",
                ")",
                f"refresh_interval={max(1.0, float(refresh_interval_seconds)):.6f}",
                "refresh_once() {",
                f"  if [ -f {shlex.quote(str(manifest_path))} ]; then",
                '    "${refresh_cmd[@]}" >/dev/null 2>&1 || true',
                "  fi",
                "}",
                "refresh_loop() {",
                "  while true; do",
                "    refresh_once",
                '    sleep "$refresh_interval" || break',
                "  done",
                "}",
                "refresh_loop &",
                "refresh_pid=$!",
                "cleanup() {",
                '  if [ "${refresh_pid:-0}" -gt 0 ] 2>/dev/null; then',
                '    kill "${refresh_pid}" 2>/dev/null || true',
                '    wait "${refresh_pid}" 2>/dev/null || true',
                "  fi",
                "  refresh_once",
                "}",
                "trap cleanup EXIT",
            ]
        )
    lines.extend(
        [
            "set +e",
            '"${cmd[@]}"',
            "exit_code=$?",
            "set -e",
            'exit "$exit_code"',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _latest_pointer(root_dir: Path, slug: str) -> Path:
    return root_dir / "by_name" / f"{slug}.latest"


def _resolve_manifest_path(
    *,
    manifest: Path | None,
    job_root: Path | None,
    name: str | None,
    root_dir: Path,
) -> Path:
    if manifest is not None:
        return manifest.resolve()
    if job_root is not None:
        return (job_root.resolve() / "manifest.json")
    if name:
        pointer = _latest_pointer(root_dir.resolve(), _slugify(name))
        if not pointer.exists():
            raise FileNotFoundError(f"no latest manifest pointer for name={name!r} under {root_dir}")
        target = pointer.read_text(encoding="utf-8").strip()
        if not target:
            raise FileNotFoundError(f"latest manifest pointer is empty: {pointer}")
        return Path(target).resolve()
    raise ValueError("provide one of --manifest, --job-root, or --name")


def _ps_snapshot(pid: int) -> Dict[str, str]:
    if pid <= 0:
        return {}
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=", "-o", "etimes=", "-o", "cmd="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}
    line = result.stdout.strip()
    if not line:
        return {}
    parts = line.split(None, 2)
    if len(parts) < 3:
        return {}
    return {
        "pid": parts[0],
        "elapsed_seconds": parts[1],
        "cmd": parts[2],
    }


def _tail_lines(path: Path, count: int) -> List[str]:
    if count <= 0 or not path.exists():
        return []
    try:
        result = subprocess.run(
            ["tail", "-n", str(int(count)), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    return [line.rstrip("\n") for line in result.stdout.splitlines()]


def _systemd_user_available() -> bool:
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except Exception:
        return False
    return int(result.returncode) == 0


def _resolve_launch_backend(requested: str) -> str:
    text = str(requested or "").strip().lower() or LAUNCH_BACKEND_AUTO
    if text == LAUNCH_BACKEND_AUTO:
        return (
            LAUNCH_BACKEND_SYSTEMD
            if _systemd_user_available()
            else LAUNCH_BACKEND_DOUBLE_FORK
        )
    if text == LAUNCH_BACKEND_SYSTEMD and not _systemd_user_available():
        raise SystemExit(
            "systemd --user manager unavailable; use --launch-backend double_fork"
        )
    if text not in LAUNCH_BACKEND_CHOICES:
        raise SystemExit(f"unknown launch backend: {requested!r}")
    return text


def _systemd_unit_base_name(*, slug: str, stamp: str) -> str:
    base = f"codex-long-job-{stamp}-{slug}"
    if len(base) > 200:
        base = base[:200].rstrip("-_")
    return base or "codex-long-job"


def _systemd_unit_name(*, slug: str, stamp: str) -> str:
    return f"{_systemd_unit_base_name(slug=slug, stamp=stamp)}.service"


def _systemd_unit_status(unit_name: str) -> Dict[str, Any]:
    text = str(unit_name or "").strip()
    if not text:
        return {"known": False, "running": False, "main_pid": 0}
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                text,
                "--property=Id",
                "--property=MainPID",
                "--property=ActiveState",
                "--property=SubState",
                "--property=Result",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except Exception:
        return {
            "known": False,
            "running": False,
            "main_pid": 0,
            "unit_name": text,
        }
    payload: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[str(key).strip()] = str(value).strip()
    main_pid = _safe_int(payload.get("MainPID"), default=0)
    active_state = str(payload.get("ActiveState", "") or "").strip()
    sub_state = str(payload.get("SubState", "") or "").strip()
    known = bool(payload.get("Id")) or bool(active_state) or int(result.returncode) == 0
    running = active_state in {"active", "activating", "reloading"}
    return {
        "known": known,
        "running": running,
        "main_pid": main_pid,
        "active_state": active_state,
        "sub_state": sub_state,
        "result": str(payload.get("Result", "") or "").strip(),
        "unit_name": text,
    }


def _resolved_manifest_pid(manifest: Mapping[str, Any]) -> int:
    unit_name = str(manifest.get("systemd_unit", "") or "").strip()
    if unit_name:
        unit_status = _systemd_unit_status(unit_name)
        if int(unit_status.get("main_pid", 0) or 0) > 0:
            return int(unit_status["main_pid"])
    return int(manifest.get("pid", 0) or 0)


def _manifest_is_live(manifest: Mapping[str, Any]) -> bool:
    unit_name = str(manifest.get("systemd_unit", "") or "").strip()
    if unit_name:
        unit_status = _systemd_unit_status(unit_name)
        if bool(unit_status.get("running", False)):
            return True
        main_pid = int(unit_status.get("main_pid", 0) or 0)
        if main_pid > 0 and _pid_is_live(main_pid):
            return True
    pid = int(manifest.get("pid", 0) or 0)
    return _pid_is_live(pid)


def _flag_value(command: Sequence[str], flag: str) -> str:
    items = [str(item) for item in list(command)]
    for idx, token in enumerate(items):
        if token == str(flag) and idx + 1 < len(items):
            return str(items[idx + 1]).strip()
        prefix = f"{flag}="
        if token.startswith(prefix):
            return str(token[len(prefix) :]).strip()
    return ""


def _parse_shell_array_values(command: Sequence[str], name: str) -> List[str]:
    joined = "\n".join(str(item) for item in command)
    match = re.search(
        rf'(^|\n)\s*{re.escape(str(name))}\s*=\s*\((.*?)\)',
        joined,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return []
    body = str(match.group(2) or "").strip()
    if not body:
        return []
    try:
        return [str(item).strip() for item in shlex.split(body) if str(item).strip()]
    except Exception:
        return []


def _infer_output_root(command: Sequence[str]) -> str:
    flagged = _flag_value(command, "--output-root")
    if flagged:
        return flagged
    joined = "\n".join(str(item) for item in command)
    match = re.search(
        r'(^|\n)\s*OUTPUT_ROOT=(["\']?)([^"\n\']+)\2',
        joined,
        flags=re.MULTILINE,
    )
    if match is None:
        return ""
    return str(match.group(3)).strip()


def _render_progress_bar(percent_complete: float, *, width: int = 20) -> str:
    pct = max(0.0, min(float(percent_complete), 100.0))
    filled = int(round((pct / 100.0) * float(width)))
    filled = max(0, min(filled, int(width)))
    return "#" * filled + "-" * (int(width) - filled)


def _candidate_sort_key(path: Path) -> tuple[int, float]:
    payload = _load_json(path)
    state = str(payload.get("state", "") or "").strip().lower()
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return (1 if state == "running" else 0, mtime)


def _planned_package_capacity_status_paths(
    *,
    command: Sequence[str],
    output_root: Path,
) -> List[Path]:
    packages = _parse_shell_array_values(command, "packages")
    if not packages:
        return []
    phase_root = output_root / "package_capacity"
    out: List[Path] = []
    for scope in ("recoverable", "structural"):
        for package_name in packages:
            out.append(phase_root / scope / str(package_name) / "scheduler_status.json")
    return out


def _normalize_progress_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {
        str(key): value for key, value in dict(payload).items()
    }
    items_total = _safe_int(normalized.get("items_total"), default=0)
    initial_items_total = _safe_int(normalized.get("initial_items_total"), default=0)
    dynamic_items_added = _safe_int(normalized.get("dynamic_items_added"), default=0)
    completed_items = _safe_int(normalized.get("completed_items"), default=0)
    failed_items = _safe_int(normalized.get("failed_items"), default=0)
    active_items = _safe_int(normalized.get("active_items"), default=0)
    pending_items = _safe_int(normalized.get("pending_items"), default=0)
    derived_total = max(
        items_total,
        initial_items_total + dynamic_items_added,
        completed_items + failed_items + active_items + pending_items,
    )
    if derived_total > 0:
        normalized["items_total"] = derived_total
        normalized["percent_complete"] = 100.0 * float(completed_items) / float(derived_total)
        normalized["progress_bar"] = _render_progress_bar(float(normalized["percent_complete"]))
    return normalized


def _aggregate_nested_progress(
    *,
    output_root: Path,
    candidates: Sequence[Path],
    planned_candidates: Sequence[Path] = (),
) -> Dict[str, Any]:
    existing_set = {path.resolve() for path in candidates}
    missing_planned = [
        path.resolve()
        for path in planned_candidates
        if path.resolve() not in existing_set
    ]
    relative_parents = [
        path.relative_to(output_root).parent.parts
        for path in sorted(candidates)
    ] + [
        path.relative_to(output_root).parent.parts
        for path in sorted(missing_planned)
        if path.is_absolute() or not str(path).startswith("..")
    ]
    shared_prefix: List[str] = []
    if relative_parents:
        shortest = min(len(parts) for parts in relative_parents)
        for idx in range(shortest):
            column = {parts[idx] for parts in relative_parents}
            if len(column) != 1:
                break
            shared_prefix.append(relative_parents[0][idx])
    child_progress: Dict[str, Any] = {}
    items_total = 0
    completed_items = 0
    failed_items = 0
    active_items = 0
    pending_items = 0
    source_status_paths: List[str] = []
    planned_status_paths: List[str] = []
    child_states: List[str] = []
    observed_child_totals: List[int] = []
    observed_scope_totals: Dict[str, int] = {}
    observed_global_total = 0
    normalized_payloads: Dict[Path, Dict[str, Any]] = {}
    resolved_rel_parts: Dict[Path, Sequence[str]] = {}
    for path in sorted(candidates):
        payload = _load_json(path)
        if not payload:
            continue
        payload = _normalize_progress_payload(_refresh_live_active_progress(payload))
        normalized_payloads[path] = payload
        rel_parts = path.relative_to(output_root).parent.parts
        resolved_rel_parts[path] = rel_parts
        child_total = _safe_int(payload.get("items_total"), default=0)
        observed_child_totals.append(child_total)
        observed_global_total = max(observed_global_total, child_total)
        scope_key = (
            str(rel_parts[len(shared_prefix)])
            if len(rel_parts) > len(shared_prefix)
            else ""
        )
        if scope_key:
            observed_scope_totals[scope_key] = max(
                observed_scope_totals.get(scope_key, 0),
                child_total,
            )

    def _planned_total_for_path(path: Path) -> tuple[int, str]:
        rel_parts = resolved_rel_parts.get(path)
        if rel_parts is None:
            try:
                rel_parts = path.relative_to(output_root).parent.parts
            except Exception:
                rel_parts = path.parts
        scope_key = (
            str(rel_parts[len(shared_prefix)])
            if len(rel_parts) > len(shared_prefix)
            else ""
        )
        scope_total = (
            _safe_int(observed_scope_totals.get(scope_key), default=0)
            if scope_key
            else 0
        )
        if max(scope_total, observed_global_total) > 0:
            if scope_total >= observed_global_total and scope_total > 0:
                return (int(scope_total), "observed_scope_total")
            if observed_global_total > 0:
                return (int(observed_global_total), "observed_global_total")
        if observed_global_total > 0:
            return (int(observed_global_total), "observed_global_total")
        return (int(DEFAULT_PACKAGE_CAPACITY_ITEMS_ESTIMATE), "default_fallback")

    for path in sorted(candidates):
        payload = normalized_payloads.get(path)
        if not payload:
            continue
        rel_parts = resolved_rel_parts.get(path) or path.relative_to(output_root).parent.parts
        rel_key = "/".join(rel_parts[len(shared_prefix) :]) or str(path.relative_to(output_root).parent)
        source_status_paths.append(str(path))
        child_states.append(str(payload.get("state", "") or ""))
        observed_total = _safe_int(payload.get("items_total"), default=0)
        planned_total, total_source = _planned_total_for_path(path)
        child_total = max(observed_total, planned_total)
        child_completed = _safe_int(payload.get("completed_items"), default=0)
        child_failed = _safe_int(payload.get("failed_items"), default=0)
        child_active = _safe_int(payload.get("active_items"), default=0)
        child_pending = max(
            _safe_int(payload.get("pending_items"), default=0),
            max(0, child_total - child_completed - child_failed - child_active),
        )
        items_total += child_total
        completed_items += child_completed
        failed_items += child_failed
        active_items += child_active
        pending_items += child_pending
        child_progress[rel_key] = {
            "state": str(payload.get("state", "") or ""),
            "items_total": child_total,
            "observed_items_total": observed_total,
            "completed_items": child_completed,
            "failed_items": child_failed,
            "active_items": child_active,
            "pending_items": child_pending,
            "percent_complete": (
                100.0 * float(child_completed) / float(child_total)
                if child_total > 0
                else 0.0
            ),
            "items_total_source": total_source,
            "status_path": str(path),
            "generated_at": str(payload.get("generated_at", "") or ""),
        }
    for path in sorted(missing_planned):
        try:
            rel_parts = path.relative_to(output_root).parent.parts
        except Exception:
            rel_parts = path.parts
        rel_key = "/".join(rel_parts[len(shared_prefix) :]) or str(path)
        planned_status_paths.append(str(path))
        child_states.append("planned")
        planned_total, total_source = _planned_total_for_path(path)
        items_total += planned_total
        pending_items += planned_total
        child_progress[rel_key] = {
            "state": "planned",
            "items_total": planned_total,
            "completed_items": 0,
            "failed_items": 0,
            "active_items": 0,
            "pending_items": planned_total,
            "percent_complete": 0.0,
            "items_total_source": total_source,
            "status_path": "",
            "expected_status_path": str(path),
            "generated_at": "",
        }
    if any(state == "running" for state in child_states):
        state = "running"
    elif any(state == "planned" for state in child_states):
        state = "running"
    elif any(state == "failed" for state in child_states):
        state = "failed"
    elif child_states and all(state == "completed" for state in child_states):
        state = "completed"
    else:
        state = "unknown"
    percent_complete = (
        100.0 * float(completed_items) / float(items_total)
        if items_total > 0
        else 0.0
    )
    combined = {
        "generated_at": _utc_now(),
        "state": state,
        "status_kind": "combined_scheduler_progress",
        "active_phase": "package_capacity",
        "items_total": items_total,
        "completed_items": completed_items,
        "failed_items": failed_items,
        "active_items": active_items,
        "pending_items": pending_items,
        "percent_complete": percent_complete,
        "progress_bar": _render_progress_bar(percent_complete),
        "phase_progress": child_progress,
        "source_status_paths": source_status_paths,
        "planned_status_paths": planned_status_paths,
    }
    combined_path = output_root / COMBINED_STATUS_NAME
    _write_json(combined_path, combined)
    combined["status_path"] = str(combined_path)
    return combined


def _sibling_progress_path(log_path: str) -> Path | None:
    text = str(log_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.name != "run.log":
        return None
    return path.with_name("progress.json")


def _refresh_bucket_epoch_percent(bucket: Mapping[str, Any]) -> Dict[str, Any]:
    refreshed = dict(bucket)
    epochs_total = _safe_int(refreshed.get("epochs_total"), default=0)
    epochs_completed = _safe_int(refreshed.get("epochs_completed"), default=0)
    if epochs_total > 0:
        refreshed["epochs_completed"] = max(0, min(epochs_completed, epochs_total))
        refreshed["epoch_percent"] = (
            100.0 * float(refreshed["epochs_completed"]) / float(epochs_total)
        )
    return refreshed


def _apply_epoch_delta(
    buckets: Mapping[str, Any],
    bucket_name: str,
    delta: int,
) -> Dict[str, Any]:
    updated = {
        str(key): dict(value) if isinstance(value, Mapping) else value
        for key, value in buckets.items()
    }
    if bucket_name not in updated:
        return updated
    current = updated[str(bucket_name)]
    if not isinstance(current, Mapping):
        return updated
    current_payload = dict(current)
    current_payload["epochs_completed"] = _safe_int(
        current_payload.get("epochs_completed"),
        default=0,
    ) + int(delta)
    updated[str(bucket_name)] = _refresh_bucket_epoch_percent(current_payload)
    return updated


def _refresh_live_active_progress(payload: Mapping[str, Any]) -> Dict[str, Any]:
    progress = {
        str(key): value for key, value in dict(payload).items()
    }
    active_items = list(progress.get("active_item_details") or [])
    if not active_items:
        return progress
    phase_progress = dict(progress.get("phase_progress") or {})
    by_scope = dict(progress.get("by_scope") or {})
    by_train_docs = dict(progress.get("by_train_docs") or {})
    by_model_family = dict(progress.get("by_model_family") or {})
    by_package = dict(progress.get("by_package") or {})
    by_worker_kind = dict(progress.get("by_worker_kind") or {})
    refreshed_items: List[Dict[str, Any]] = []
    changed = False
    for item in active_items:
        if not isinstance(item, Mapping):
            refreshed_items.append(dict(item))
            continue
        item_payload = dict(item)
        live_progress_path = _sibling_progress_path(str(item_payload.get("log_path", "")))
        if live_progress_path is None or not live_progress_path.exists():
            refreshed_items.append(item_payload)
            continue
        live_progress = _load_json(live_progress_path)
        if not live_progress:
            refreshed_items.append(item_payload)
            continue
        recorded_progress = dict(item_payload.get("progress") or {})
        recorded_epoch_completed = _safe_int(
            recorded_progress.get("epoch_completed"),
            default=0,
        )
        live_epoch_completed = _safe_int(
            live_progress.get("epoch_completed"),
            default=recorded_epoch_completed,
        )
        delta = int(live_epoch_completed - recorded_epoch_completed)
        if delta != 0:
            phase_name = str(item_payload.get("phase", "") or "").strip()
            if phase_name:
                phase_progress = _apply_epoch_delta(phase_progress, phase_name, delta)
            scope = str(item_payload.get("scope", "") or "").strip()
            if scope:
                by_scope = _apply_epoch_delta(by_scope, scope, delta)
            train_docs = str(item_payload.get("train_docs", "") or "").strip()
            if train_docs:
                by_train_docs = _apply_epoch_delta(by_train_docs, train_docs, delta)
            model_family = str(item_payload.get("model_family", "") or "").strip()
            if model_family:
                by_model_family = _apply_epoch_delta(by_model_family, model_family, delta)
            package = str(item_payload.get("package", "") or "").strip()
            if package:
                by_package = _apply_epoch_delta(by_package, package, delta)
            worker_kind = str(item_payload.get("worker_kind", "") or "").strip()
            if worker_kind:
                by_worker_kind = _apply_epoch_delta(by_worker_kind, worker_kind, delta)
            changed = True
        item_payload["progress"] = live_progress
        refreshed_items.append(item_payload)
    if not changed:
        progress["active_item_details"] = refreshed_items
        return progress
    progress["active_item_details"] = refreshed_items
    progress["phase_progress"] = phase_progress
    progress["by_scope"] = by_scope
    progress["by_train_docs"] = by_train_docs
    progress["by_model_family"] = by_model_family
    progress["by_package"] = by_package
    progress["by_worker_kind"] = by_worker_kind
    progress["live_progress_refreshed_at"] = _utc_now()
    return progress


def _load_job_progress(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    command = [str(item) for item in list(manifest.get("command") or ())]
    output_root_text = _infer_output_root(command)
    if not output_root_text:
        return {}
    cwd = Path(str(manifest.get("cwd", REPO_ROOT) or REPO_ROOT)).resolve()
    output_root = Path(output_root_text).expanduser()
    if not output_root.is_absolute():
        output_root = (cwd / output_root).resolve()
    status_path = output_root / "experiment_status.json"
    if not status_path.exists():
        status_path = output_root / "scheduler_status.json"
    if not status_path.exists():
        candidates = sorted(output_root.rglob("experiment_status.json"))
        if not candidates:
            candidates = sorted(output_root.rglob("scheduler_status.json"))
        planned_candidates = _planned_package_capacity_status_paths(
            command=command,
            output_root=output_root,
        )
        if not candidates and not planned_candidates:
            return {}
        if len(candidates) > 1 or planned_candidates:
            return _aggregate_nested_progress(
                output_root=output_root,
                candidates=candidates,
                planned_candidates=planned_candidates,
            )
        status_path = max(candidates, key=_candidate_sort_key)
    payload = _load_json(status_path)
    if payload:
        payload = _normalize_progress_payload(_refresh_live_active_progress(payload))
        payload["status_path"] = str(status_path)
    return payload


def _numeric_bucket_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except Exception:
        return (1, text)


def _display_friendly_status(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, Mapping):
        items = list(value.items())
        if str(parent_key) == "by_train_docs":
            items = sorted(items, key=lambda item: _numeric_bucket_sort_key(str(item[0])))
        formatted: Dict[str, Any] = {}
        for key, subvalue in items:
            key_text = str(key)
            if key_text in {"percent_complete", "epoch_percent"} and isinstance(
                subvalue, (int, float)
            ):
                formatted[key_text] = f"{float(subvalue):.1f}%"
            else:
                formatted[key_text] = _display_friendly_status(
                    subvalue,
                    parent_key=key_text,
                )
        return formatted
    if isinstance(value, list):
        return [
            _display_friendly_status(item, parent_key=parent_key) for item in value
        ]
    return value


def _launch(args: argparse.Namespace) -> int:
    cmd = _strip_remainder(args.cmd)
    if not cmd:
        raise SystemExit("missing command after `--`")
    root_dir = Path(args.root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(args.name)
    stamp = _utc_stamp()
    job_root = Path(args.job_root).resolve() if args.job_root else (root_dir / f"{stamp}_{slug}")
    manifest_path = job_root / "manifest.json"
    pid_path = job_root / "job.pid"
    log_path = job_root / "job.log"
    runner_path = job_root / "runner.sh"
    env = _parse_env_assignments(args.env or [])
    launch_backend = _resolve_launch_backend(str(getattr(args, "launch_backend", LAUNCH_BACKEND_AUTO)))
    systemd_unit = ""

    latest_path = _latest_pointer(root_dir, slug)
    if latest_path.exists() and not bool(args.replace_existing):
        try:
            latest_manifest = Path(latest_path.read_text(encoding="utf-8").strip()).resolve()
            payload = _load_json(latest_manifest)
            old_pid = _resolved_manifest_pid(payload)
            old_live = _manifest_is_live(payload)
        except Exception:
            latest_manifest = None
            old_pid = 0
            old_live = False
        if old_live:
            raise SystemExit(
                f"name={args.name!r} already has a live job pid={old_pid}; use --replace-existing to override"
            )

    job_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    _render_runner_script(
        path=runner_path,
        cwd=Path(args.cwd).resolve(),
        command=cmd,
        env=env,
        manifest_path=manifest_path,
        refresh_python_bin=(
            str(Path(args.python_bin).resolve())
            if os.path.sep in str(args.python_bin)
            else str(args.python_bin)
        ),
        refresh_interval_seconds=float(
            max(0.0, float(getattr(args, "progress_refresh_interval_seconds", 30.0) or 0.0))
        ),
    )
    python_bin = str(
        Path(args.python_bin).resolve()
        if os.path.sep in str(args.python_bin)
        else Path(args.python_bin)
    )
    if launch_backend == LAUNCH_BACKEND_SYSTEMD:
        systemd_unit = _systemd_unit_name(slug=slug, stamp=stamp)
        launcher_cmd = [
            "systemd-run",
            "--user",
            "--collect",
            "--no-block",
            "--quiet",
            f"--unit={systemd_unit.removesuffix('.service')}",
            f"--description={str(args.description or args.name)}",
            "--service-type=simple",
            f"--working-directory={str(Path(args.cwd).resolve())}",
            "bash",
            "-lc",
            (
                f"exec bash {shlex.quote(str(runner_path))}"
                f" >> {shlex.quote(str(log_path))} 2>&1"
            ),
        ]
        subprocess.run(
            launcher_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        pid = 0
        deadline = time.time() + 5.0
        while time.time() < deadline:
            unit_status = _systemd_unit_status(systemd_unit)
            pid = int(unit_status.get("main_pid", 0) or 0)
            if bool(unit_status.get("running", False)) or pid > 0:
                break
            time.sleep(0.1)
    else:
        spawn_cmd = [
            python_bin,
            str(SPAWN_DETACHED_SCRIPT),
            "--pid-file",
            str(pid_path),
            "--cwd",
            str(Path(args.cwd).resolve()),
            "--stdin",
            os.devnull,
            "--stdout",
            str(log_path),
            "--stderr",
            str(log_path),
            "--",
            "bash",
            str(runner_path),
        ]
        result = subprocess.run(
            spawn_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        pid = int((result.stdout or "0").strip() or "0")
    deadline = time.time() + 5.0
    pgid = 0
    while time.time() < deadline:
        pgid = _lookup_pgid(pid)
        if pgid > 0:
            break
        time.sleep(0.05)
    if pgid <= 0:
        pgid = pid
    manifest = {
        "schema_version": 1,
        "name": str(args.name),
        "slug": slug,
        "description": str(args.description or ""),
        "launched_at": _utc_now(),
        "cwd": str(Path(args.cwd).resolve()),
        "command": [str(item) for item in cmd],
        "python_bin": str(args.python_bin),
        "launch_backend": str(launch_backend),
        "systemd_unit": str(systemd_unit),
        "pid": int(pid),
        "pgid": int(pgid),
        "job_root": str(job_root),
        "manifest_path": str(manifest_path),
        "pid_file": str(pid_path),
        "log_path": str(log_path),
        "runner_script": str(runner_path),
        "env": dict(env),
        "launcher": "scripts/long_job.py",
        "progress_refresh_interval_seconds": float(
            max(0.0, float(getattr(args, "progress_refresh_interval_seconds", 30.0) or 0.0))
        ),
        "tail_command": f"tail -f {log_path}",
        "status_command": f"python3 scripts/long_job.py status --manifest {manifest_path}",
        "stop_command": f"python3 scripts/long_job.py stop --manifest {manifest_path}",
    }
    _write_json(manifest_path, manifest)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(manifest_path) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _status(args: argparse.Namespace) -> int:
    manifest_path = _resolve_manifest_path(
        manifest=args.manifest,
        job_root=args.job_root,
        name=args.name,
        root_dir=Path(args.root_dir),
    )
    manifest = _load_json(manifest_path)
    systemd_unit = str(manifest.get("systemd_unit", "") or "").strip()
    unit_status = _systemd_unit_status(systemd_unit) if systemd_unit else {}
    pid = int(unit_status.get("main_pid", 0) or 0) or int(manifest.get("pid", 0) or 0)
    live = bool(unit_status.get("running", False)) or _pid_is_live(pid)
    snapshot = _ps_snapshot(pid) if live else {}
    status = {
        "manifest_path": str(manifest_path),
        "name": str(manifest.get("name", "")),
        "description": str(manifest.get("description", "")),
        "job_root": str(manifest.get("job_root", "")),
        "launch_backend": str(manifest.get("launch_backend", LAUNCH_BACKEND_DOUBLE_FORK)),
        "systemd_unit": systemd_unit,
        "systemd_state": {
            "active_state": str(unit_status.get("active_state", "") or ""),
            "sub_state": str(unit_status.get("sub_state", "") or ""),
            "result": str(unit_status.get("result", "") or ""),
        }
        if systemd_unit
        else {},
        "pid": pid,
        "pgid": int(manifest.get("pgid", 0) or 0),
        "running": bool(live),
        "launched_at": str(manifest.get("launched_at", "")),
        "log_path": str(manifest.get("log_path", "")),
        "command": list(manifest.get("command") or []),
        "process": snapshot,
        "tail": _tail_lines(Path(str(manifest.get("log_path", ""))), int(args.tail_lines)),
    }
    progress = _load_job_progress(manifest)
    if progress:
        status["progress"] = progress
    print(
        json.dumps(
            _display_friendly_status(status),
            indent=2,
            sort_keys=False,
        )
    )
    return 0


def _refresh(args: argparse.Namespace) -> int:
    manifest_path = _resolve_manifest_path(
        manifest=args.manifest,
        job_root=args.job_root,
        name=args.name,
        root_dir=Path(args.root_dir),
    )
    manifest = _load_json(manifest_path)
    progress = _load_job_progress(manifest)
    if bool(getattr(args, "emit_json", False)):
        payload = {
            "manifest_path": str(manifest_path),
            "refreshed_at": _utc_now(),
            "has_progress": bool(progress),
            "status_path": str(progress.get("status_path", "")) if progress else "",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _signal_from_name(name: str) -> signal.Signals:
    key = str(name).strip().upper()
    if not key:
        raise ValueError("signal name cannot be empty")
    if not key.startswith("SIG"):
        key = f"SIG{key}"
    return signal.Signals[key]


def _stop(args: argparse.Namespace) -> int:
    manifest_path = _resolve_manifest_path(
        manifest=args.manifest,
        job_root=args.job_root,
        name=args.name,
        root_dir=Path(args.root_dir),
    )
    manifest = _load_json(manifest_path)
    systemd_unit = str(manifest.get("systemd_unit", "") or "").strip()
    unit_status_before = _systemd_unit_status(systemd_unit) if systemd_unit else {}
    pid = int(unit_status_before.get("main_pid", 0) or 0) or int(manifest.get("pid", 0) or 0)
    pgid = _lookup_pgid(pid) or int(manifest.get("pgid", 0) or 0)
    sig = _signal_from_name(str(args.signal))
    target_kind = "pid"
    target_value = pid
    if pgid > 0:
        target_kind = "pgid"
        target_value = pgid
    if systemd_unit:
        target_kind = "systemd_unit"
        target_value = systemd_unit
    already_dead = not (
        bool(unit_status_before.get("running", False)) or (pid > 0 and _pid_is_live(pid))
    )
    if not already_dead:
        if systemd_unit:
            if sig == signal.SIGTERM:
                subprocess.run(
                    ["systemctl", "--user", "stop", systemd_unit],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                subprocess.run(
                    ["systemctl", "--user", "kill", f"--signal={sig.name}", systemd_unit],
                    capture_output=True,
                    text=True,
                    check=False,
                )
        elif target_kind == "pgid":
            os.killpg(target_value, sig)
        else:
            os.kill(target_value, sig)
        deadline = time.time() + float(args.wait_seconds)
        while time.time() < deadline:
            unit_status_now = _systemd_unit_status(systemd_unit) if systemd_unit else {}
            live_now = bool(unit_status_now.get("running", False)) or (
                pid > 0 and _pid_is_live(pid)
            )
            if not live_now:
                break
            time.sleep(0.1)
        if bool(args.force_kill) and (
            bool((_systemd_unit_status(systemd_unit) if systemd_unit else {}).get("running", False))
            or (pid > 0 and _pid_is_live(pid))
        ):
            if systemd_unit:
                subprocess.run(
                    ["systemctl", "--user", "kill", "--signal=KILL", systemd_unit],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            elif target_kind == "pgid":
                os.killpg(target_value, signal.SIGKILL)
            else:
                os.kill(target_value, signal.SIGKILL)
    root_markers = [str(manifest.get("job_root", ""))]
    orphan_cleanup_events = cleanup_orphan_processes(root_markers)
    matching_after_stop = matching_processes(root_markers)
    status = {
        "manifest_path": str(manifest_path),
        "name": str(manifest.get("name", "")),
        "launch_backend": str(manifest.get("launch_backend", LAUNCH_BACKEND_DOUBLE_FORK)),
        "systemd_unit": systemd_unit,
        "pid": pid,
        "pgid": pgid,
        "signal": sig.name,
        "target_kind": target_kind,
        "target_value": target_value,
        "already_dead": already_dead,
        "running_after_stop": bool(
            (_systemd_unit_status(systemd_unit) if systemd_unit else {}).get("running", False)
        )
        or _pid_is_live(pid),
        "orphan_cleanup_events": orphan_cleanup_events,
        "matching_processes_after_stop": matching_after_stop,
        "verified_children_gone": not bool(matching_after_stop),
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official detached launcher for long-running jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Launch a detached job and write a manifest.")
    launch.add_argument("--name", type=str, required=True)
    launch.add_argument("--description", type=str, default="")
    launch.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    launch.add_argument("--job-root", type=Path, default=None)
    launch.add_argument("--cwd", type=Path, default=REPO_ROOT)
    launch.add_argument("--python-bin", type=str, default=sys.executable)
    launch.add_argument(
        "--launch-backend",
        type=str,
        choices=list(LAUNCH_BACKEND_CHOICES),
        default=LAUNCH_BACKEND_AUTO,
        help="Detach backend. 'auto' prefers systemd --user when available, else falls back to the legacy double-fork launcher.",
    )
    launch.add_argument("--progress-refresh-interval-seconds", type=float, default=30.0)
    launch.add_argument("--replace-existing", action=argparse.BooleanOptionalAction, default=False)
    launch.add_argument("--env", action="append", default=[])
    launch.add_argument("cmd", nargs=argparse.REMAINDER)

    status = subparsers.add_parser("status", help="Report status for a launched job.")
    status.add_argument("--manifest", type=Path, default=None)
    status.add_argument("--job-root", type=Path, default=None)
    status.add_argument("--name", type=str, default=None)
    status.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    status.add_argument("--tail-lines", type=int, default=20)

    refresh = subparsers.add_parser("refresh", help="Refresh derived progress files for a launched job.")
    refresh.add_argument("--manifest", type=Path, default=None)
    refresh.add_argument("--job-root", type=Path, default=None)
    refresh.add_argument("--name", type=str, default=None)
    refresh.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    refresh.add_argument("--emit-json", action=argparse.BooleanOptionalAction, default=False)

    stop = subparsers.add_parser("stop", help="Stop a launched job by manifest, job root, or latest name.")
    stop.add_argument("--manifest", type=Path, default=None)
    stop.add_argument("--job-root", type=Path, default=None)
    stop.add_argument("--name", type=str, default=None)
    stop.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    stop.add_argument("--signal", type=str, default="TERM")
    stop.add_argument("--wait-seconds", type=float, default=5.0)
    stop.add_argument("--force-kill", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "launch":
        return _launch(args)
    if args.command == "status":
        return _status(args)
    if args.command == "refresh":
        return _refresh(args)
    if args.command == "stop":
        return _stop(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
