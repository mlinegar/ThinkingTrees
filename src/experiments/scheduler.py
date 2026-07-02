#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import time
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_int as _safe_int


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _thread_env_defaults() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key in THREAD_ENV_KEYS:
        env[key] = os.environ.get(key, "1")
    env["PYTHONUNBUFFERED"] = os.environ.get("PYTHONUNBUFFERED", "1")
    return env


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(dict(payload)), sort_keys=False) + "\n")
        handle.flush()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_proc_status(pid: int) -> Dict[str, Any]:
    path = Path("/proc") / str(int(pid)) / "status"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    rows: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        rows[str(key).strip()] = str(value).strip()
    payload: Dict[str, Any] = {}
    for field in ("Name", "State", "VmRSS", "VmHWM", "VmSize", "VmPeak", "Threads"):
        if field in rows:
            payload[str(field)] = rows[str(field)]
    return payload


def _read_proc_text_file(pid: int, name: str) -> str:
    path = Path("/proc") / str(int(pid)) / str(name)
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_proc_smaps_rollup(pid: int) -> Dict[str, Any]:
    text = _read_proc_text_file(pid, "smaps_rollup")
    if not text:
        return {}
    rows: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        rows[str(key).strip()] = str(value).strip()
    payload: Dict[str, Any] = {}
    for field in (
        "Rss",
        "Pss",
        "Shared_Clean",
        "Shared_Dirty",
        "Private_Clean",
        "Private_Dirty",
        "Referenced",
        "Anonymous",
        "Swap",
    ):
        if field in rows:
            payload[str(field)] = rows[str(field)]
    return payload


def _system_memory_snapshot() -> Dict[str, Any]:
    path = Path("/proc/meminfo")
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    rows: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        rows[str(key).strip()] = str(value).strip()
    payload: Dict[str, Any] = {}
    for field in (
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SwapTotal",
        "SwapFree",
        "Committed_AS",
        "CommitLimit",
        "AnonPages",
        "Shmem",
    ):
        if field in rows:
            payload[str(field)] = rows[str(field)]
    return payload


def _nvidia_compute_snapshot() -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if int(result.returncode) != 0:
        return []
    rows: List[Dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = str(raw_line).strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        pid_text, gpu_uuid, used_memory_text, process_name = parts
        rows.append(
            {
                "pid": _safe_int(pid_text, default=0),
                "gpu_uuid": str(gpu_uuid),
                "used_gpu_memory_mib": _safe_int(used_memory_text, default=0),
                "process_name": str(process_name),
            }
        )
    return rows


def _progress_bar(percent: float, *, width: int = 20) -> str:
    bounded = max(0.0, min(100.0, float(percent)))
    filled = int(round((bounded / 100.0) * float(width)))
    filled = max(0, min(int(width), int(filled)))
    return "#" * filled + "-" * (int(width) - filled)


def _bucket_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except Exception:
        return (1, text)


@dataclass
class SchedulerItem:
    item_id: str
    phase: str
    kind: str
    deps: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    log_path: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Callable[[], Mapping[str, Any] | None] | None = None
    gpu_slots: int = 1
    allowed_devices: tuple[str, ...] = ()
    reuse_existing: bool = True
    run_on_failed_dependencies: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    devices: tuple[str, ...]
    max_gpu_items_per_mig: int = 1
    cleanup_stale_children: bool = True
    cancel_on_failure: bool = True
    raise_on_failure: bool = True
    root_markers: tuple[str, ...] = ()
    status_path: str = ""
    status_alias_paths: tuple[str, ...] = ()
    status_metadata: Dict[str, Any] = field(default_factory=dict)
    event_log_path: str = ""
    failure_snapshot_path: str = ""
    launch_stagger_seconds: float = 0.0
    min_mem_available_kib: int = 0
    min_swap_free_kib: int = 0


class SchedulerRunError(RuntimeError):
    def __init__(self, message: str, *, summary: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


def _expected_outputs_exist(item: SchedulerItem) -> bool:
    if not bool(item.reuse_existing):
        return False
    outputs = [str(path).strip() for path in item.expected_outputs if str(path).strip()]
    if not outputs:
        return False
    return all(Path(path).exists() for path in outputs)


def _compatible_available_tokens(
    available_tokens: Sequence[str],
    item: SchedulerItem,
) -> List[str]:
    allowed = tuple(str(token) for token in tuple(item.allowed_devices or ()) if str(token).strip())
    if not allowed:
        return [str(token) for token in available_tokens]
    compatible: List[str] = []
    available = [str(token) for token in available_tokens]
    for allowed_token in allowed:
        for token in available:
            if str(token) == str(allowed_token):
                compatible.append(str(token))
    return compatible


def _pop_compatible_tokens(
    available_tokens: List[str],
    item: SchedulerItem,
    *,
    slot_count: int,
) -> List[str]:
    compatible = _compatible_available_tokens(available_tokens, item)
    if int(slot_count) > len(compatible):
        raise RuntimeError(
            f"item {item.item_id} requested {int(slot_count)} slots but only "
            f"{len(compatible)} compatible devices are available"
        )
    selected = compatible[: int(slot_count)]
    remaining: List[str] = []
    selected_counts: Dict[str, int] = {}
    for token in selected:
        selected_counts[str(token)] = int(selected_counts.get(str(token), 0)) + 1
    for token in available_tokens:
        key = str(token)
        if int(selected_counts.get(key, 0)) > 0:
            selected_counts[key] = int(selected_counts[key]) - 1
            continue
        remaining.append(key)
    available_tokens[:] = remaining
    return [str(token) for token in selected]


def _ps_rows() -> List[tuple[int, int, str]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=", "-o", "ppid=", "-o", "args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    rows: List[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except Exception:
            continue
        rows.append((pid, ppid, str(parts[2])))
    return rows


def matching_processes(root_markers: Sequence[str]) -> List[Dict[str, Any]]:
    markers = [str(marker).strip() for marker in root_markers if str(marker).strip()]
    if not markers:
        return []
    matches: List[Dict[str, Any]] = []
    for pid, ppid, cmd in _ps_rows():
        if not any(marker in cmd for marker in markers):
            continue
        matches.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "pgid": int(_lookup_pgid(pid)),
                "command": str(cmd),
            }
        )
    return matches


def cleanup_orphan_processes(root_markers: Sequence[str]) -> List[Dict[str, Any]]:
    markers = [str(marker).strip() for marker in root_markers if str(marker).strip()]
    if not markers:
        return []
    events: List[Dict[str, Any]] = []
    current_pid = os.getpid()
    orphan_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    orphan_pids: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in matching_processes(markers):
        pid = int(row["pid"])
        ppid = int(row["ppid"])
        pgid = int(row.get("pgid", 0) or 0)
        if pid == current_pid:
            continue
        if _pid_is_live(ppid):
            continue
        if pgid > 0:
            orphan_groups[int(pgid)].append(dict(row))
        else:
            orphan_pids[int(pid)].append(dict(row))

    def _kill_target(
        *,
        target_kind: str,
        target_value: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        first_row = dict(rows[0]) if rows else {}
        member_pids = sorted({int(row.get("pid", 0) or 0) for row in rows if int(row.get("pid", 0) or 0) > 0})
        event: Dict[str, Any] = {
            "pid": int(first_row.get("pid", 0) or 0),
            "ppid": int(first_row.get("ppid", 0) or 0),
            "target_kind": str(target_kind),
            "target_value": int(target_value),
            "pgid": int(target_value) if str(target_kind) == "pgid" else int(rows[0].get("pgid", 0) or 0),
            "pids": member_pids,
            "member_count": len(member_pids),
            "commands": [str(row.get("command", "")) for row in rows],
            "signal": "SIGTERM",
            "killed": False,
            "escalated_to_sigkill": False,
        }
        try:
            if str(target_kind) == "pgid":
                os.killpg(int(target_value), signal.SIGTERM)
            else:
                os.kill(int(target_value), signal.SIGTERM)
            event["killed"] = True
            deadline = time.time() + 2.0
            while time.time() < deadline and any(_pid_is_live(pid) for pid in member_pids):
                time.sleep(0.05)
            if any(_pid_is_live(pid) for pid in member_pids):
                if str(target_kind) == "pgid":
                    os.killpg(int(target_value), signal.SIGKILL)
                else:
                    os.kill(int(target_value), signal.SIGKILL)
                event["signal"] = "SIGKILL"
                event["escalated_to_sigkill"] = True
        except OSError as exc:
            event["error"] = str(exc)
        event["live_pids_after_cleanup"] = [
            int(pid) for pid in member_pids if _pid_is_live(pid)
        ]
        event["verified_children_gone"] = not bool(event["live_pids_after_cleanup"])
        events.append(event)

    for pgid, rows in sorted(orphan_groups.items()):
        _kill_target(target_kind="pgid", target_value=int(pgid), rows=rows)
    for pid, rows in sorted(orphan_pids.items()):
        _kill_target(target_kind="pid", target_value=int(pid), rows=rows)
    return events


def _open_log(path: Path, command: Sequence[str]) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    if command:
        handle.write("$ " + " ".join(str(part) for part in command) + "\n\n")
        handle.flush()
    return handle


def _launch_command_item(
    item: SchedulerItem,
    *,
    tokens: Sequence[str],
) -> Dict[str, Any]:
    env = dict(os.environ)
    env.update(_thread_env_defaults())
    env.update({str(key): str(value) for key, value in dict(item.env).items()})
    visible_tokens = [str(token).strip() for token in tokens if str(token).strip()]
    if visible_tokens:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_tokens)
    log_path = Path(str(item.log_path))
    handle = _open_log(log_path, list(item.command))
    proc = subprocess.Popen(
        list(item.command),
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return {
        "item": item,
        "proc": proc,
        "log_fh": handle,
        "log_path": str(log_path),
        "started_at": time.perf_counter(),
        "tokens": list(visible_tokens),
    }


def _run_callback_item(item: SchedulerItem) -> Dict[str, Any]:
    started = time.perf_counter()
    if _expected_outputs_exist(item):
        return {
            "status": "reused",
            "wall_clock_s": 0.0,
            "new_items": [],
            "result": {},
        }
    callback = item.callback
    if callback is None:
        raise ValueError(f"callback item {item.item_id} missing callback")
    payload = dict(callback() or {})
    return {
        "status": str(payload.get("status", "completed")),
        "wall_clock_s": float(time.perf_counter() - started),
        "new_items": list(payload.get("new_items") or []),
        "result": dict(payload.get("result") or {}),
    }


def summarize_scheduler_plan(
    items: Sequence[SchedulerItem],
    *,
    devices: Sequence[str],
    max_gpu_items_per_mig: int,
    launch_stagger_seconds: float = 0.0,
) -> Dict[str, Any]:
    phase_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        phase_counts[str(item.phase)][str(item.kind)] += 1
    phase_payload = {
        phase: dict(sorted(counts.items()))
        for phase, counts in sorted(phase_counts.items())
    }
    return {
        "scheduler_mode": "global_per_run",
        "default_job_granularity": "family_train_seed",
        "device_count": len(list(devices)),
        "max_gpu_items_per_mig": int(max_gpu_items_per_mig),
        "launch_stagger_seconds": float(max(0.0, float(launch_stagger_seconds))),
        "items_total": len(list(items)),
        "phase_item_counts": phase_payload,
    }


def run_scheduler(
    items: Sequence[SchedulerItem],
    *,
    config: SchedulerConfig,
) -> Dict[str, Any]:
    device_slots: List[str] = []
    resolved_devices = [str(token) for token in list(config.devices)]
    max_slots_per_device = max(1, int(config.max_gpu_items_per_mig))
    launch_stagger_seconds = max(0.0, float(config.launch_stagger_seconds))
    min_mem_available_kib = max(0, int(getattr(config, "min_mem_available_kib", 0)))
    min_swap_free_kib = max(0, int(getattr(config, "min_swap_free_kib", 0)))
    cancel_on_failure = bool(getattr(config, "cancel_on_failure", True))
    raise_on_failure = bool(getattr(config, "raise_on_failure", True))
    for _slot_idx in range(max_slots_per_device):
        device_slots.extend(resolved_devices)
    all_items: MutableMapping[str, SchedulerItem] = {
        str(item.item_id): item for item in items
    }
    initial_items_total = len(all_items)
    pending: MutableMapping[str, SchedulerItem] = dict(all_items)
    completed: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, Dict[str, Any]] = {}
    reverse_deps: Dict[str, set[str]] = defaultdict(set)
    active: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    queue_wait_s: Dict[str, float] = {}
    created_at: Dict[str, float] = {str(item.item_id): time.perf_counter() for item in items}
    available_tokens = list(device_slots)
    stale_cleanup_events = (
        cleanup_orphan_processes(config.root_markers)
        if bool(config.cleanup_stale_children)
        else []
    )
    reused_count = 0
    executed_count = 0
    idle_slot_seconds = 0.0
    last_tick = time.perf_counter()
    next_gpu_launch_at = last_tick
    failure_cleanup_events: List[Dict[str, Any]] = []
    status_path = Path(str(config.status_path)).expanduser() if str(config.status_path).strip() else None
    status_alias_paths = tuple(
        Path(str(path)).expanduser()
        for path in tuple(getattr(config, "status_alias_paths", ()) or ())
        if str(path).strip()
    )
    status_metadata = dict(getattr(config, "status_metadata", {}) or {})
    event_log_path = (
        Path(str(config.event_log_path)).expanduser()
        if str(config.event_log_path).strip()
        else None
    )
    failure_snapshot_path = (
        Path(str(config.failure_snapshot_path)).expanduser()
        if str(config.failure_snapshot_path).strip()
        else None
    )
    abort_reason: Dict[str, Any] | None = None
    last_status_write = 0.0
    last_status_payload: Dict[str, Any] = {}
    first_failure_recorded = False

    for item in items:
        item_id = str(item.item_id)
        for dep in item.deps:
            reverse_deps[str(dep)].add(item_id)

    def _meminfo_kib(snapshot: Mapping[str, Any], field: str) -> int:
        raw = str(snapshot.get(field, "") or "").strip()
        if not raw:
            return 0
        return _safe_int(raw.split()[0], default=0)

    def _write_event(event_type: str, payload: Mapping[str, Any]) -> None:
        if event_log_path is None:
            return
        _append_jsonl(
            event_log_path,
            {
                "generated_at": _utc_now(),
                "event": str(event_type),
                **dict(payload),
            },
        )

    def _active_process_payload(now: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for entry in active:
            item = entry["item"]
            proc = entry["proc"]
            pid = int(getattr(proc, "pid", 0) or 0)
            rows.append(
                {
                    "item_id": str(item.item_id),
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "pid": int(pid),
                    "pgid": int(_lookup_pgid(int(pid))),
                    "tokens": [str(token) for token in list(entry.get("tokens") or ())],
                    "log_path": str(entry["log_path"]),
                    "elapsed_seconds": float(now - float(entry["started_at"])),
                    "metadata": _json_safe(dict(item.metadata)),
                    "proc_status": _read_proc_status(int(pid)),
                    "proc_smaps_rollup": _read_proc_smaps_rollup(int(pid)),
                    "oom_score": _read_proc_text_file(int(pid), "oom_score"),
                    "oom_score_adj": _read_proc_text_file(int(pid), "oom_score_adj"),
                }
            )
        return sorted(rows, key=lambda row: str(row["item_id"]))

    def _write_failure_snapshot(*, reason: str) -> None:
        if failure_snapshot_path is None:
            return
        now = time.perf_counter()
        _write_json_atomic(
            failure_snapshot_path,
            {
                "generated_at": _utc_now(),
                "reason": str(reason),
                "pending_items": sorted(str(item_id) for item_id in pending.keys()),
                "available_tokens": [str(token) for token in available_tokens],
                "active_processes": _active_process_payload(now),
                "completed_item_ids": sorted(str(item_id) for item_id in completed.keys()),
                "failed_items": _json_safe(dict(failed)),
                "failure_cleanup_events": _json_safe(list(failure_cleanup_events)),
                "timeline_tail": _json_safe(list(timeline[-200:])),
                "last_status": _json_safe(dict(last_status_payload)),
                "system_memory_snapshot": _json_safe(_system_memory_snapshot()),
                "nvidia_compute_snapshot": _json_safe(_nvidia_compute_snapshot()),
            },
        )

    def _item_progress(metadata: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        progress_path = str(metadata.get("progress_path", "") or "").strip()
        if not progress_path:
            return {}
        return _load_json(Path(progress_path))

    def _update_bucket(
        buckets: MutableMapping[str, Dict[str, Any]],
        *,
        bucket_name: str,
        state: str,
        epoch_completed: int,
        epoch_total: int,
    ) -> None:
        bucket = buckets.setdefault(
            str(bucket_name),
            {
                "total": 0,
                "completed": 0,
                "active": 0,
                "pending": 0,
                "failed": 0,
                "epochs_completed": 0,
                "epochs_total": 0,
            },
        )
        bucket["total"] += 1
        state_key = str(state).strip().lower()
        if state_key == "completed":
            bucket["completed"] += 1
        elif state_key == "active":
            bucket["active"] += 1
        elif state_key == "failed":
            bucket["failed"] += 1
        else:
            bucket["pending"] += 1
        if int(epoch_total) > 0:
            bucket["epochs_total"] += int(epoch_total)
            bucket["epochs_completed"] += max(0, min(int(epoch_completed), int(epoch_total)))

    def _bucket_payload(
        buckets: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        for name, bucket in sorted(buckets.items(), key=lambda item: _bucket_sort_key(item[0])):
            total = int(bucket.get("total", 0) or 0)
            completed = int(bucket.get("completed", 0) or 0)
            active_count = int(bucket.get("active", 0) or 0)
            failed_count = int(bucket.get("failed", 0) or 0)
            pending_count = int(bucket.get("pending", 0) or 0)
            epochs_total = int(bucket.get("epochs_total", 0) or 0)
            epochs_completed = int(bucket.get("epochs_completed", 0) or 0)
            payload[str(name)] = {
                "total": total,
                "completed": completed,
                "active": active_count,
                "pending": pending_count,
                "failed": failed_count,
                "percent_complete": (
                    (100.0 * float(completed + failed_count) / float(total))
                    if total > 0
                    else 0.0
                ),
            }
            if epochs_total > 0:
                payload[str(name)]["epochs_completed"] = epochs_completed
                payload[str(name)]["epochs_total"] = epochs_total
                payload[str(name)]["epoch_percent"] = (
                    100.0 * float(epochs_completed) / float(epochs_total)
                )
        return payload

    def _write_live_status(*, force: bool = False) -> None:
        nonlocal last_status_write, last_status_payload
        now = time.perf_counter()
        if not force and (now - last_status_write) < 1.0:
            return
        active_by_id = {
            str(entry["item"].item_id): entry
            for entry in active
        }
        phase_buckets: Dict[str, Dict[str, Any]] = {}
        scope_buckets: Dict[str, Dict[str, Any]] = {}
        train_doc_buckets: Dict[str, Dict[str, Any]] = {}
        model_buckets: Dict[str, Dict[str, Any]] = {}
        package_buckets: Dict[str, Dict[str, Any]] = {}
        worker_kind_buckets: Dict[str, Dict[str, Any]] = {}
        active_payload: List[Dict[str, Any]] = []
        items_total = len(all_items)
        dynamic_items_added = max(0, int(items_total) - int(initial_items_total))
        finished_count = len(completed) + len(failed)

        for item in all_items.values():
            item_id = str(item.item_id)
            metadata = _json_safe(dict(item.metadata))
            state = "pending"
            epoch_progress = {}
            if item_id in failed:
                state = "failed"
            elif item_id in completed:
                state = "completed"
            elif item_id in active_by_id:
                state = "active"
                epoch_progress = _item_progress(metadata)

            configured_epochs = _safe_int(metadata.get("n_epochs"), default=0)
            progress_epoch_total = _safe_int(epoch_progress.get("epochs_total"), default=0)
            epoch_total = progress_epoch_total if progress_epoch_total > 0 else configured_epochs
            epoch_completed = 0
            if state == "completed" and epoch_total > 0:
                epoch_completed = epoch_total
            elif state in {"active", "failed"}:
                epoch_completed = _safe_int(epoch_progress.get("epoch_completed"), default=0)

            _update_bucket(
                phase_buckets,
                bucket_name=str(item.phase),
                state=state,
                epoch_completed=epoch_completed,
                epoch_total=epoch_total,
            )

            axis_values = {
                "scope": metadata.get("scope"),
                "train_docs": metadata.get("train_docs"),
                "model_family": metadata.get("model_family"),
                "package": metadata.get("package"),
                "worker_kind": metadata.get("worker_kind"),
            }
            for axis_name, axis_value in axis_values.items():
                if axis_value in {None, ""}:
                    continue
                target = {
                    "scope": scope_buckets,
                    "train_docs": train_doc_buckets,
                    "model_family": model_buckets,
                    "package": package_buckets,
                    "worker_kind": worker_kind_buckets,
                }[axis_name]
                _update_bucket(
                    target,
                    bucket_name=str(axis_value),
                    state=state,
                    epoch_completed=epoch_completed,
                    epoch_total=epoch_total,
                )

            if state == "active":
                entry = active_by_id[item_id]
                elapsed_s = float(now - float(entry["started_at"]))
                item_payload: Dict[str, Any] = {
                    "item_id": item_id,
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "elapsed_seconds": elapsed_s,
                    "log_path": str(entry["log_path"]),
                    "task_name": str(metadata.get("task_name", item_id)),
                }
                for key in ("scope", "train_docs", "model_family", "package", "worker_kind"):
                    value = metadata.get(key)
                    if value not in {None, ""}:
                        item_payload[str(key)] = value
                if epoch_progress:
                    item_payload["progress"] = epoch_progress
                elif epoch_total > 0:
                    item_payload["progress"] = {
                        "state": "running",
                        "epoch_completed": 0,
                        "epochs_total": int(epoch_total),
                    }
                active_payload.append(item_payload)

        state = "completed"
        if pending or active:
            state = "running"
        elif failed:
            state = "failed"

        percent_complete = (
            100.0 * float(finished_count) / float(items_total)
            if items_total > 0
            else 100.0
        )
        active_phase = ""
        if active_payload:
            active_phase = str(active_payload[0].get("phase", "") or "")
        elif pending:
            next_item = next(iter(pending.values()))
            active_phase = str(next_item.phase)
        elif completed:
            last_item = next(reversed(completed.values()))
            active_phase = str(last_item.get("phase", "") or "")
        artifact_targets = [
            str(item)
            for item in list(status_metadata.get("artifact_targets") or ())
            if str(item).strip()
        ]
        payload = {
            "generated_at": _utc_now(),
            "state": state,
            "status_kind": "experiment_progress",
            "experiment_id": str(status_metadata.get("experiment_id", "") or ""),
            "experiment_adapter": str(status_metadata.get("experiment_adapter", "") or ""),
            "experiment_title": str(status_metadata.get("experiment_title", "") or ""),
            "active_phase": active_phase,
            "items_total": int(items_total),
            "initial_items_total": int(initial_items_total),
            "dynamic_items_added": int(dynamic_items_added),
            "completed_items": len(completed),
            "failed_items": len(failed),
            "active_items": len(active),
            "pending_items": len(pending),
            "percent_complete": percent_complete,
            "progress_bar": _progress_bar(percent_complete),
            "artifact_targets": artifact_targets,
            "phase_progress": _bucket_payload(phase_buckets),
            "by_scope": _bucket_payload(scope_buckets),
            "by_train_docs": _bucket_payload(train_doc_buckets),
            "by_model_family": _bucket_payload(model_buckets),
            "by_package": _bucket_payload(package_buckets),
            "by_worker_kind": _bucket_payload(worker_kind_buckets),
            "active_item_details": sorted(active_payload, key=lambda row: str(row["item_id"])),
        }
        if status_metadata:
            payload["status_metadata"] = _json_safe(dict(status_metadata))
        if failed:
            payload["first_failed_item"] = _json_safe(next(iter(failed.values())))
        last_status_payload = dict(payload)
        if status_path is not None:
            _write_json_atomic(status_path, payload)
        for alias_path in status_alias_paths:
            _write_json_atomic(alias_path, payload)
        last_status_write = now

    def _record_timeline() -> None:
        timeline.append(
            {
                "timestamp": _utc_now(),
                "pending_items": len(pending),
                "active_gpu_items": sum(
                    1 for entry in active if str(entry["item"].kind) == "gpu_command"
                ),
                "active_cpu_items": sum(
                    1 for entry in active if str(entry["item"].kind) == "cpu_command"
                ),
                "active_gpu_slots": sum(
                    len(list(entry.get("tokens") or []))
                    for entry in active
                    if str(entry["item"].kind) == "gpu_command"
                ),
                "available_gpu_slots": len(available_tokens),
            }
        )

    _write_live_status(force=True)

    def _ready_items(kind: str) -> List[SchedulerItem]:
        ready: List[SchedulerItem] = []
        for item_id, item in list(pending.items()):
            if str(item.kind) != str(kind):
                continue
            if bool(item.run_on_failed_dependencies):
                if any(dep not in completed and dep not in failed for dep in item.deps):
                    continue
            elif any(dep not in completed for dep in item.deps):
                continue
            ready.append(item)
        return ready

    def _mark_failed_dependency_tree(*, failed_item_id: str, reason: str) -> None:
        queue: List[str] = [str(failed_item_id)]
        seen: set[str] = set()
        while queue:
            upstream_id = str(queue.pop(0))
            if upstream_id in seen:
                continue
            seen.add(upstream_id)
            for child_id in sorted(reverse_deps.get(upstream_id, ())):
                child_item = pending.pop(child_id, None)
                if child_item is not None:
                    if bool(child_item.run_on_failed_dependencies):
                        pending[str(child_item.item_id)] = child_item
                        continue
                    existing = failed.get(child_id)
                    upstream_failed_items: List[str] = []
                    if existing is not None:
                        upstream_failed_items.extend(
                            str(item)
                            for item in list(existing.get("upstream_failed_items") or [])
                            if str(item).strip()
                        )
                    if upstream_id not in upstream_failed_items:
                        upstream_failed_items.append(upstream_id)
                    failed[child_id] = {
                        "item_id": str(child_item.item_id),
                        "phase": str(child_item.phase),
                        "kind": str(child_item.kind),
                        "returncode": -1,
                        "failure_reason": str(reason),
                        "log_path": str(child_item.log_path),
                        "expected_outputs": [str(path) for path in child_item.expected_outputs],
                        "metadata": _json_safe(dict(child_item.metadata)),
                        "gpu_slots": int(max(1, int(child_item.gpu_slots))),
                        "upstream_failed_items": upstream_failed_items,
                    }
                    _write_event(
                        "dependency_failed",
                        {
                            "item_id": str(child_item.item_id),
                            "phase": str(child_item.phase),
                            "kind": str(child_item.kind),
                            "failure_reason": str(reason),
                            "upstream_failed_items": list(upstream_failed_items),
                            "log_path": str(child_item.log_path),
                        },
                    )
                if child_id in failed:
                    queue.append(child_id)

    def _finish_command(entry: Mapping[str, Any]) -> None:
        nonlocal reused_count, executed_count
        proc = entry["proc"]
        item = entry["item"]
        handle = entry["log_fh"]
        handle.close()
        finished = time.perf_counter()
        rc = int(proc.returncode)
        wall_s = float(finished - float(entry["started_at"]))
        queue_wait_s[str(item.item_id)] = float(entry["started_at"] - created_at[str(item.item_id)])
        if str(item.kind) == "gpu_command":
            available_tokens.extend([str(token) for token in list(entry.get("tokens") or ())])
        if rc != 0:
            failed[str(item.item_id)] = {
                "item_id": str(item.item_id),
                "phase": str(item.phase),
                "kind": str(item.kind),
                "returncode": rc,
                "log_path": str(entry["log_path"]),
                "expected_outputs": [str(path) for path in item.expected_outputs],
                "metadata": _json_safe(dict(item.metadata)),
                "gpu_slots": int(max(1, int(item.gpu_slots))),
            }
            _write_event(
                "command_failed",
                {
                    "item_id": str(item.item_id),
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "returncode": rc,
                    "tokens": [str(token) for token in list(entry.get("tokens") or ())],
                    "pid": int(getattr(proc, "pid", 0) or 0),
                    "pgid": int(_lookup_pgid(int(getattr(proc, "pid", 0) or 0))),
                    "log_path": str(entry["log_path"]),
                    "pending_items_after_failure": len(pending),
                    "available_gpu_slots_after_failure": len(available_tokens),
                },
            )
            _mark_failed_dependency_tree(
                failed_item_id=str(item.item_id),
                reason="failed_dependency",
            )
            return
        executed_count += 1
        completed[str(item.item_id)] = {
            "item_id": str(item.item_id),
            "status": "completed",
            "phase": str(item.phase),
            "kind": str(item.kind),
            "wall_clock_s": wall_s,
            "log_path": str(entry["log_path"]),
            "expected_outputs": [str(path) for path in item.expected_outputs],
            "reused": False,
            "metadata": _json_safe(dict(item.metadata)),
            "gpu_slots": int(max(1, int(item.gpu_slots))),
        }
        _write_event(
            "command_completed",
            {
                "item_id": str(item.item_id),
                "phase": str(item.phase),
                "kind": str(item.kind),
                "tokens": [str(token) for token in list(entry.get("tokens") or ())],
                "pid": int(getattr(proc, "pid", 0) or 0),
                "pgid": int(_lookup_pgid(int(getattr(proc, "pid", 0) or 0))),
                "log_path": str(entry["log_path"]),
                "wall_clock_s": wall_s,
            },
        )

    def _cancel_active_items(*, reason: str) -> None:
        nonlocal active
        if not active:
            return
        survivors: List[Dict[str, Any]] = []
        for entry in active:
            proc = entry["proc"]
            if proc.poll() is not None:
                _finish_command(entry)
                continue
            survivors.append(entry)
        active = survivors
        if not active:
            return

        for entry in active:
            proc = entry["proc"]
            try:
                proc.terminate()
            except OSError as exc:
                entry["terminate_error"] = str(exc)

        deadline = time.time() + 5.0
        for entry in active:
            proc = entry["proc"]
            remaining = max(0.0, deadline - time.time())
            if proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:
                entry["wait_error"] = str(exc)

        for entry in active:
            proc = entry["proc"]
            if proc.poll() is not None:
                continue
            try:
                proc.kill()
            except OSError as exc:
                entry["kill_error"] = str(exc)

        for entry in active:
            proc = entry["proc"]
            item = entry["item"]
            handle = entry["log_fh"]
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass
            if str(item.kind) == "gpu_command":
                available_tokens.extend(
                    [str(token) for token in list(entry.get("tokens") or ())]
                )
            wall_s = float(time.perf_counter() - float(entry["started_at"]))
            queue_wait_s[str(item.item_id)] = float(
                entry["started_at"] - created_at[str(item.item_id)]
            )
            cancel_event = {
                "item_id": str(item.item_id),
                "phase": str(item.phase),
                "kind": str(item.kind),
                "reason": str(reason),
                "returncode": (
                    int(proc.returncode)
                    if proc.returncode is not None
                    else -int(signal.SIGKILL)
                ),
                "wall_clock_s": wall_s,
                "log_path": str(entry["log_path"]),
                "expected_outputs": [str(path) for path in item.expected_outputs],
                "metadata": _json_safe(dict(item.metadata)),
                "gpu_slots": int(max(1, int(item.gpu_slots))),
            }
            if entry.get("terminate_error"):
                cancel_event["terminate_error"] = str(entry["terminate_error"])
            if entry.get("wait_error"):
                cancel_event["wait_error"] = str(entry["wait_error"])
            if entry.get("kill_error"):
                cancel_event["kill_error"] = str(entry["kill_error"])
            failed[str(item.item_id)] = cancel_event
            failure_cleanup_events.append(cancel_event)
            _write_event("command_cancelled", cancel_event)
            _mark_failed_dependency_tree(
                failed_item_id=str(item.item_id),
                reason="cancelled_dependency",
            )
        active = []

    while pending or active:
        if min_mem_available_kib > 0 or min_swap_free_kib > 0:
            system_snapshot = _system_memory_snapshot()
            mem_available_kib = _meminfo_kib(system_snapshot, "MemAvailable")
            swap_free_kib = _meminfo_kib(system_snapshot, "SwapFree")
            if (
                (min_mem_available_kib > 0 and mem_available_kib < min_mem_available_kib)
                or (min_swap_free_kib > 0 and swap_free_kib < min_swap_free_kib)
            ):
                abort_reason = {
                    "status": "host_memory_floor_breached",
                    "mem_available_kib": int(mem_available_kib),
                    "swap_free_kib": int(swap_free_kib),
                    "min_mem_available_kib": int(min_mem_available_kib),
                    "min_swap_free_kib": int(min_swap_free_kib),
                }
                _write_failure_snapshot(reason="host_memory_floor_breached")
                _cancel_active_items(reason="host_memory_floor_breached")
                _write_live_status(force=True)
                break
        now = time.perf_counter()
        ready_gpu = _ready_items("gpu_command")
        has_fittable_gpu_item = any(
            int(max(1, int(item.gpu_slots))) <= len(available_tokens)
            for item in ready_gpu
        )
        if has_fittable_gpu_item and len(available_tokens) > 0:
            idle_slot_seconds += float(now - last_tick) * float(len(available_tokens))
        last_tick = now

        progressed = False

        while available_tokens:
            launch_now = time.perf_counter()
            if launch_now < next_gpu_launch_at:
                break
            ready_gpu = [
                item
                for item in _ready_items("gpu_command")
                if int(max(1, int(item.gpu_slots)))
                <= len(_compatible_available_tokens(available_tokens, item))
            ]
            if not ready_gpu:
                break
            item = ready_gpu[0]
            pending.pop(str(item.item_id), None)
            if _expected_outputs_exist(item):
                reused_count += 1
                completed[str(item.item_id)] = {
                    "item_id": str(item.item_id),
                    "status": "reused",
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "wall_clock_s": 0.0,
                    "log_path": str(item.log_path),
                    "expected_outputs": [str(path) for path in item.expected_outputs],
                    "reused": True,
                    "metadata": dict(item.metadata),
                    "gpu_slots": int(max(1, int(item.gpu_slots))),
                }
                _write_event(
                    "command_reused",
                    {
                        "item_id": str(item.item_id),
                        "phase": str(item.phase),
                        "kind": str(item.kind),
                        "log_path": str(item.log_path),
                    },
                )
                progressed = True
                continue
            slot_count = int(max(1, int(item.gpu_slots)))
            tokens = _pop_compatible_tokens(
                available_tokens,
                item,
                slot_count=int(slot_count),
            )
            active.append(_launch_command_item(item, tokens=tokens))
            launched = active[-1]
            proc = launched["proc"]
            _write_event(
                "launch",
                {
                    "item_id": str(item.item_id),
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "pid": int(getattr(proc, "pid", 0) or 0),
                    "pgid": int(_lookup_pgid(int(getattr(proc, "pid", 0) or 0))),
                    "tokens": [str(token) for token in list(tokens)],
                    "log_path": str(launched["log_path"]),
                    "pending_items_after_launch": len(pending),
                    "available_gpu_slots_after_launch": len(available_tokens),
                },
            )
            if launch_stagger_seconds > 0.0:
                next_gpu_launch_at = time.perf_counter() + float(launch_stagger_seconds)
            progressed = True

        cpu_callbacks = _ready_items("cpu_callback")
        for item in cpu_callbacks:
            pending.pop(str(item.item_id), None)
            callback_result = _run_callback_item(item)
            status = str(callback_result.get("status", "completed"))
            if status == "reused":
                reused_count += 1
            else:
                executed_count += 1
            completed[str(item.item_id)] = {
                "item_id": str(item.item_id),
                "status": status,
                "phase": str(item.phase),
                "kind": str(item.kind),
                "wall_clock_s": float(callback_result.get("wall_clock_s", 0.0)),
                "log_path": str(item.log_path),
                "expected_outputs": [str(path) for path in item.expected_outputs],
                "reused": status == "reused",
                "result": _json_safe(dict(callback_result.get("result") or {})),
                "metadata": _json_safe(dict(item.metadata)),
                "gpu_slots": int(max(1, int(item.gpu_slots))),
            }
            for new_item in list(callback_result.get("new_items") or []):
                all_items[str(new_item.item_id)] = new_item
                pending[str(new_item.item_id)] = new_item
                created_at[str(new_item.item_id)] = time.perf_counter()
                for dep in new_item.deps:
                    reverse_deps[str(dep)].add(str(new_item.item_id))
                failed_deps = [
                    str(dep)
                    for dep in new_item.deps
                    if str(dep) in failed
                ]
                if failed_deps and not bool(new_item.run_on_failed_dependencies):
                    pending.pop(str(new_item.item_id), None)
                    failed[str(new_item.item_id)] = {
                        "item_id": str(new_item.item_id),
                        "phase": str(new_item.phase),
                        "kind": str(new_item.kind),
                        "returncode": -1,
                        "failure_reason": "failed_dependency",
                        "log_path": str(new_item.log_path),
                        "expected_outputs": [str(path) for path in new_item.expected_outputs],
                        "metadata": _json_safe(dict(new_item.metadata)),
                        "gpu_slots": int(max(1, int(new_item.gpu_slots))),
                        "upstream_failed_items": list(failed_deps),
                    }
                    _write_event(
                        "dependency_failed",
                        {
                            "item_id": str(new_item.item_id),
                            "phase": str(new_item.phase),
                            "kind": str(new_item.kind),
                            "failure_reason": "failed_dependency",
                            "upstream_failed_items": list(failed_deps),
                            "log_path": str(new_item.log_path),
                        },
                    )
                    _mark_failed_dependency_tree(
                        failed_item_id=str(new_item.item_id),
                        reason="failed_dependency",
                    )
            _write_event(
                "callback_complete",
                {
                    "item_id": str(item.item_id),
                    "phase": str(item.phase),
                    "status": status,
                    "new_items": [
                        str(new_item.item_id)
                        for new_item in list(callback_result.get("new_items") or [])
                    ],
                },
            )
            progressed = True

        cpu_commands = _ready_items("cpu_command")
        for item in cpu_commands:
            pending.pop(str(item.item_id), None)
            if _expected_outputs_exist(item):
                reused_count += 1
                completed[str(item.item_id)] = {
                    "item_id": str(item.item_id),
                    "status": "reused",
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "wall_clock_s": 0.0,
                    "log_path": str(item.log_path),
                    "expected_outputs": [str(path) for path in item.expected_outputs],
                    "reused": True,
                    "metadata": _json_safe(dict(item.metadata)),
                    "gpu_slots": int(max(1, int(item.gpu_slots))),
                }
                _write_event(
                    "command_reused",
                    {
                        "item_id": str(item.item_id),
                        "phase": str(item.phase),
                        "kind": str(item.kind),
                        "log_path": str(item.log_path),
                    },
                )
                progressed = True
                continue
            active.append(_launch_command_item(item, tokens=()))
            launched = active[-1]
            proc = launched["proc"]
            _write_event(
                "launch",
                {
                    "item_id": str(item.item_id),
                    "phase": str(item.phase),
                    "kind": str(item.kind),
                    "pid": int(getattr(proc, "pid", 0) or 0),
                    "pgid": int(_lookup_pgid(int(getattr(proc, "pid", 0) or 0))),
                    "tokens": [],
                    "log_path": str(launched["log_path"]),
                    "pending_items_after_launch": len(pending),
                    "available_gpu_slots_after_launch": len(available_tokens),
                },
            )
            progressed = True

        _record_timeline()
        _write_live_status()
        time.sleep(0.1 if active else 0.02)
        still_active: List[Dict[str, Any]] = []
        for entry in active:
            proc = entry["proc"]
            if proc.poll() is None:
                still_active.append(entry)
                continue
            _finish_command(entry)
            progressed = True
        active = still_active

        _record_timeline()
        _write_live_status()
        if failed and not first_failure_recorded:
            _write_failure_snapshot(reason="first_failure_detected")
            first_failure_recorded = True
            if cancel_on_failure:
                _cancel_active_items(reason="peer_failure")
                _write_live_status(force=True)
                break
        ready_gpu_launch_pending = bool(
            available_tokens
            and any(
                int(max(1, int(item.gpu_slots))) <= len(available_tokens)
                for item in _ready_items("gpu_command")
            )
        )
        if not progressed and not active and pending and not ready_gpu_launch_pending:
            blocked_items = sorted(pending)
            raise RuntimeError(
                "scheduler deadlock: pending items have unsatisfied dependencies: "
                + ", ".join(blocked_items[:10])
            )

    phase_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase_reused: Dict[str, int] = defaultdict(int)
    for info in completed.values():
        phase = str(info.get("phase", ""))
        kind = str(info.get("kind", ""))
        phase_counts[phase][kind] += 1
        if bool(info.get("reused", False)):
            phase_reused[phase] += 1

    summary = {
        "generated_at": _utc_now(),
        "device_count": len(tuple(config.devices)),
        "max_gpu_items_per_mig": int(config.max_gpu_items_per_mig),
        "cancel_on_failure": bool(cancel_on_failure),
        "raise_on_failure": bool(raise_on_failure),
        "launch_stagger_seconds": float(launch_stagger_seconds),
        "min_mem_available_kib": int(min_mem_available_kib),
        "min_swap_free_kib": int(min_swap_free_kib),
        "items_total": len(all_items),
        "initial_items_total": int(initial_items_total),
        "dynamic_items_added": max(0, int(len(all_items)) - int(initial_items_total)),
        "executed_items": int(executed_count),
        "reused_items": int(reused_count),
        "idle_mig_slots_due_to_scheduler_seconds": float(idle_slot_seconds),
        "completed_items": completed,
        "failed_items": failed,
        "phase_counts": {
            phase: {
                "items": int(sum(counts.values())),
                "by_kind": dict(sorted(counts.items())),
                "reused_items": int(phase_reused.get(phase, 0)),
            }
            for phase, counts in sorted(phase_counts.items())
        },
        "queue_wait_seconds": {key: float(value) for key, value in sorted(queue_wait_s.items())},
        "timeline": timeline,
        "stale_cleanup_events": stale_cleanup_events,
        "failure_cleanup_events": failure_cleanup_events,
    }
    if status_path is not None:
        summary["live_status_path"] = str(status_path)
    if event_log_path is not None:
        summary["event_log_path"] = str(event_log_path)
    if failure_snapshot_path is not None:
        summary["failure_snapshot_path"] = str(failure_snapshot_path)
    if abort_reason is not None:
        summary["abort_reason"] = _json_safe(dict(abort_reason))
    _write_live_status(force=True)
    if abort_reason is not None:
        raise SchedulerRunError(
            "scheduler aborted before host memory collapsed",
            summary=summary,
        )
    if failed and raise_on_failure:
        first_failure = next(iter(failed.values()))
        raise SchedulerRunError(
            f"scheduler item {first_failure['item_id']} failed; see {first_failure['log_path']}",
            summary=summary,
        )
    return summary
