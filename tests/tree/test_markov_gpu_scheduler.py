from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import pytest

import scripts.markov_gpu_scheduler as scheduler
from scripts.markov_gpu_scheduler import (
    SchedulerConfig,
    SchedulerItem,
    cleanup_orphan_processes,
    run_scheduler,
)


def _touch_command(path: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            "time.sleep(0.05); "
            "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
            "Path(sys.argv[1]).write_text('ok', encoding='utf-8')"
        ),
        str(path),
    )


def _delayed_touch_command(path: Path, *, sleep_s: float) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            f"time.sleep({float(sleep_s)}); "
            "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
            "Path(sys.argv[1]).write_text('ok', encoding='utf-8')"
        ),
        str(path),
    )


def _record_visible_device_command(path: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import os, sys; "
            "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
            "Path(sys.argv[1]).write_text(os.environ.get('CUDA_VISIBLE_DEVICES', ''), encoding='utf-8')"
        ),
        str(path),
    )


def _record_start_time_command(path: Path, *, sleep_s: float = 0.0) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
            "Path(sys.argv[1]).write_text(str(time.time()), encoding='utf-8'); "
            f"time.sleep({float(sleep_s)})"
        ),
        str(path),
    )


def _exit_command(*, exit_code: int, sleep_s: float = 0.0) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        f"import sys, time; time.sleep({float(sleep_s)}); sys.exit({int(exit_code)})",
    )


def test_scheduler_fills_all_gpu_slots(tmp_path: Path) -> None:
    items = []
    for idx in range(8):
        output_path = tmp_path / f"gpu_{idx}" / "summary.json"
        items.append(
            SchedulerItem(
                item_id=f"gpu::{idx}",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(output_path),),
                command=_touch_command(output_path),
                log_path=str(tmp_path / f"gpu_{idx}" / "run.log"),
            )
        )
    summary = run_scheduler(
        items,
        config=SchedulerConfig(
            devices=("MIG-0", "MIG-1", "MIG-2", "MIG-3"),
            max_gpu_items_per_mig=1,
            cleanup_stale_children=False,
        ),
    )
    assert summary["device_count"] == 4
    assert summary["executed_items"] == 8
    assert max(int(row["active_gpu_items"]) for row in summary["timeline"]) >= 4


def test_scheduler_distributes_initial_gpu_assignments_round_robin(tmp_path: Path) -> None:
    items = []
    output_paths = []
    for idx in range(4):
        output_path = tmp_path / f"gpu_{idx}" / "summary.json"
        output_paths.append(output_path)
        items.append(
            SchedulerItem(
                item_id=f"gpu::{idx}",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(output_path),),
                command=_record_visible_device_command(output_path),
                log_path=str(tmp_path / f"gpu_{idx}" / "run.log"),
            )
        )
    run_scheduler(
        items,
        config=SchedulerConfig(
            devices=("MIG-0", "MIG-1", "MIG-2", "MIG-3"),
            max_gpu_items_per_mig=4,
            cleanup_stale_children=False,
        ),
    )
    assigned = [path.read_text(encoding="utf-8") for path in output_paths]
    assert assigned == ["MIG-0", "MIG-1", "MIG-2", "MIG-3"]


def test_scheduler_respects_allowed_device_order(tmp_path: Path) -> None:
    output_path = tmp_path / "gpu" / "assigned.txt"
    run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::allowed",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(output_path),),
                command=_record_visible_device_command(output_path),
                log_path=str(tmp_path / "gpu" / "run.log"),
                allowed_devices=("MIG-2", "MIG-0"),
            )
        ],
        config=SchedulerConfig(
            devices=("MIG-0", "MIG-1", "MIG-2"),
            max_gpu_items_per_mig=1,
            cleanup_stale_children=False,
        ),
    )
    assert output_path.read_text(encoding="utf-8") == "MIG-2"


def test_scheduler_can_stagger_gpu_launches(tmp_path: Path) -> None:
    first = tmp_path / "gpu_0" / "started.txt"
    second = tmp_path / "gpu_1" / "started.txt"
    run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::0",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(first),),
                command=_record_start_time_command(first, sleep_s=0.05),
                log_path=str(tmp_path / "gpu_0" / "run.log"),
            ),
            SchedulerItem(
                item_id="gpu::1",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(second),),
                command=_record_start_time_command(second, sleep_s=0.05),
                log_path=str(tmp_path / "gpu_1" / "run.log"),
            ),
        ],
        config=SchedulerConfig(
            devices=("MIG-0", "MIG-1"),
            max_gpu_items_per_mig=1,
            cleanup_stale_children=False,
            launch_stagger_seconds=0.2,
        ),
    )
    first_started = float(first.read_text(encoding="utf-8").strip())
    second_started = float(second.read_text(encoding="utf-8").strip())
    assert second_started - first_started >= 0.15


def test_scheduler_respects_dependencies_and_reuse(tmp_path: Path) -> None:
    gpu_output = tmp_path / "gpu" / "summary.json"
    gpu_output.parent.mkdir(parents=True, exist_ok=True)
    gpu_output.write_text("done", encoding="utf-8")
    callback_output = tmp_path / "callback" / "summary.json"

    def _callback() -> dict[str, object]:
        callback_output.parent.mkdir(parents=True, exist_ok=True)
        callback_output.write_text("reduced", encoding="utf-8")
        return {"result": {"summary": str(callback_output)}}

    summary = run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::existing",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(gpu_output),),
                command=_touch_command(gpu_output),
                log_path=str(tmp_path / "gpu" / "run.log"),
            ),
            SchedulerItem(
                item_id="reduce::callback",
                phase="reduce_phase",
                kind="cpu_callback",
                deps=("gpu::existing",),
                expected_outputs=(str(callback_output),),
                callback=_callback,
            ),
        ],
        config=SchedulerConfig(
            devices=("MIG-0",),
            cleanup_stale_children=False,
        ),
    )
    assert summary["reused_items"] == 1
    assert callback_output.exists()
    assert "reduce::callback" in summary["completed_items"]


def test_scheduler_can_force_callback_refresh_with_existing_summary(tmp_path: Path) -> None:
    gpu_output = tmp_path / "gpu" / "summary.json"
    gpu_output.parent.mkdir(parents=True, exist_ok=True)
    gpu_output.write_text("done", encoding="utf-8")
    callback_output = tmp_path / "callback" / "summary.json"
    callback_output.parent.mkdir(parents=True, exist_ok=True)
    callback_output.write_text("stale", encoding="utf-8")

    def _callback() -> dict[str, object]:
        callback_output.write_text("fresh", encoding="utf-8")
        return {"result": {"summary": str(callback_output)}}

    summary = run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::existing",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(gpu_output),),
                command=_touch_command(gpu_output),
                log_path=str(tmp_path / "gpu" / "run.log"),
            ),
            SchedulerItem(
                item_id="reduce::callback",
                phase="reduce_phase",
                kind="cpu_callback",
                deps=("gpu::existing",),
                expected_outputs=(str(callback_output),),
                callback=_callback,
                reuse_existing=False,
            ),
        ],
        config=SchedulerConfig(
            devices=("MIG-0",),
            cleanup_stale_children=False,
        ),
    )
    assert callback_output.read_text(encoding="utf-8") == "fresh"
    assert summary["completed_items"]["gpu::existing"]["reused"] is True
    assert summary["completed_items"]["reduce::callback"]["reused"] is False


def test_cleanup_orphan_processes_kills_stale_children(monkeypatch) -> None:
    marker = "/tmp/markov_orphan_marker"
    live = {1111: True, 9999: False}
    kill_events: list[tuple[int, int]] = []

    monkeypatch.setattr(
        scheduler,
        "_ps_rows",
        lambda: [(1111, 9999, f"python worker {marker}")],
    )
    monkeypatch.setattr(scheduler, "_pid_is_live", lambda pid: bool(live.get(pid, False)))

    def _fake_kill(pid: int, sig: int) -> None:
        kill_events.append((int(pid), int(sig)))
        if int(sig) != 0:
            live[int(pid)] = False

    monkeypatch.setattr(scheduler.os, "kill", _fake_kill)
    events = cleanup_orphan_processes([marker])
    assert events
    assert int(events[0]["pid"]) == 1111
    assert kill_events
    assert kill_events[0][0] == 1111


@dataclass
class _FakeSpec:
    label: str
    width: int


def test_scheduler_summary_metadata_is_json_safe(tmp_path: Path) -> None:
    output_path = tmp_path / "gpu" / "summary.json"
    summary = run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::jsonsafe",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(output_path),),
                command=_touch_command(output_path),
                log_path=str(tmp_path / "gpu" / "run.log"),
                metadata={"config": _FakeSpec(label="cfg", width=128)},
            )
        ],
        config=SchedulerConfig(
            devices=("MIG-0",),
            cleanup_stale_children=False,
        ),
    )
    encoded = json.dumps(summary, sort_keys=True)
    assert '"label": "cfg"' in encoded


def test_scheduler_status_uses_live_progress_path_for_active_items(tmp_path: Path) -> None:
    output_path = tmp_path / "gpu" / "summary.json"
    progress_path = tmp_path / "gpu" / "progress.json"
    status_path = tmp_path / "scheduler_status.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "state": "running",
                "epoch_completed": 4,
                "epochs_total": 9,
            }
        ),
        encoding="utf-8",
    )
    item = SchedulerItem(
        item_id="gpu::progress",
        phase="gpu_phase",
        kind="gpu_command",
        expected_outputs=(str(output_path),),
        command=(
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys, time; "
                "time.sleep(1.4); "
                "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
                "Path(sys.argv[1]).write_text('ok', encoding='utf-8')"
            ),
            str(output_path),
        ),
        log_path=str(tmp_path / "gpu" / "run.log"),
        metadata={"progress_path": str(progress_path)},
    )
    result: dict[str, object] = {}

    def _runner() -> None:
        result["summary"] = run_scheduler(
            [item],
            config=SchedulerConfig(
                devices=("MIG-0",),
                cleanup_stale_children=False,
                status_path=str(status_path),
            ),
        )

    thread = threading.Thread(target=_runner)
    thread.start()
    deadline = time.time() + 5.0
    observed = None
    while time.time() < deadline:
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            active = list(payload.get("active_item_details") or [])
            if active:
                observed = payload
                progress = dict(active[0].get("progress") or {})
                if int(progress.get("epoch_completed", 0) or 0) == 4:
                    break
        time.sleep(0.1)
    thread.join(timeout=10.0)
    assert thread.is_alive() is False
    assert observed is not None
    active = list(observed.get("active_item_details") or [])
    assert active
    assert int(active[0]["progress"]["epoch_completed"]) == 4
    assert int(observed["phase_progress"]["gpu_phase"]["epochs_completed"]) == 4


def test_scheduler_status_expands_totals_for_callback_added_items(tmp_path: Path) -> None:
    status_path = tmp_path / "scheduler_status.json"
    output_path = tmp_path / "gpu_dynamic" / "summary.json"
    observed = None
    result: dict[str, object] = {}

    def _callback() -> dict[str, object]:
        return {
            "new_items": [
                SchedulerItem(
                    item_id="gpu::dynamic",
                    phase="gpu_phase",
                    kind="gpu_command",
                    expected_outputs=(str(output_path),),
                    command=_delayed_touch_command(output_path, sleep_s=1.5),
                    log_path=str(tmp_path / "gpu_dynamic" / "run.log"),
                )
            ],
            "result": {"added": 1},
        }

    def _runner() -> None:
        result["summary"] = run_scheduler(
            [
                SchedulerItem(
                    item_id="reduce::seed_graph",
                    phase="reduce_phase",
                    kind="cpu_callback",
                    callback=_callback,
                )
            ],
            config=SchedulerConfig(
                devices=("MIG-0",),
                cleanup_stale_children=False,
                status_path=str(status_path),
            ),
        )

    thread = threading.Thread(target=_runner)
    thread.start()
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if int(payload.get("items_total", 0) or 0) == 2:
                if int(payload.get("pending_items", 0) or 0) + int(payload.get("active_items", 0) or 0) >= 1:
                    observed = payload
                    break
        time.sleep(0.1)
    thread.join(timeout=10.0)
    assert thread.is_alive() is False
    assert observed is not None
    assert int(observed["items_total"]) == 2
    assert int(observed["initial_items_total"]) == 1
    assert int(observed["dynamic_items_added"]) == 1
    assert observed["percent_complete"] == pytest.approx(50.0)
    summary = result["summary"]
    assert int(summary["items_total"]) == 2
    assert int(summary["initial_items_total"]) == 1
    assert int(summary["dynamic_items_added"]) == 1


def test_scheduler_writes_event_log_and_failure_snapshot_on_failure(tmp_path: Path) -> None:
    ok_output = tmp_path / "gpu_ok" / "summary.json"
    status_path = tmp_path / "scheduler_status.json"
    event_log_path = tmp_path / "scheduler_events.jsonl"
    failure_snapshot_path = tmp_path / "scheduler_failure_snapshot.json"

    items = [
        SchedulerItem(
            item_id="gpu::fail",
            phase="gpu_phase",
            kind="gpu_command",
            expected_outputs=(str(tmp_path / "gpu_fail" / "summary.json"),),
            command=_exit_command(exit_code=3, sleep_s=0.1),
            log_path=str(tmp_path / "gpu_fail" / "run.log"),
        ),
        SchedulerItem(
            item_id="gpu::peer",
            phase="gpu_phase",
            kind="gpu_command",
            expected_outputs=(str(ok_output),),
            command=_delayed_touch_command(ok_output, sleep_s=5.0),
            log_path=str(tmp_path / "gpu_ok" / "run.log"),
        ),
    ]

    with pytest.raises(scheduler.SchedulerRunError):
        run_scheduler(
            items,
            config=SchedulerConfig(
                devices=("MIG-0", "MIG-1"),
                cleanup_stale_children=False,
                status_path=str(status_path),
                event_log_path=str(event_log_path),
                failure_snapshot_path=str(failure_snapshot_path),
            ),
        )

    assert status_path.exists()
    assert event_log_path.exists()
    assert failure_snapshot_path.exists()

    failure_snapshot = json.loads(failure_snapshot_path.read_text(encoding="utf-8"))
    assert failure_snapshot["reason"] == "first_failure_detected"
    assert failure_snapshot["failed_items"]["gpu::fail"]["returncode"] == 3
    assert failure_snapshot["active_processes"]
    assert "system_memory_snapshot" in failure_snapshot
    assert any(
        str(row["item_id"]) == "gpu::peer"
        for row in list(failure_snapshot["active_processes"])
    )
    peer_rows = [
        dict(row)
        for row in list(failure_snapshot["active_processes"])
        if str(row.get("item_id")) == "gpu::peer"
    ]
    assert peer_rows


def test_scheduler_can_continue_after_failure_and_skip_dependency_subtree(tmp_path: Path) -> None:
    peer_output = tmp_path / "gpu_peer" / "summary.json"
    skipped_output = tmp_path / "reduce_after_fail" / "summary.json"
    event_log_path = tmp_path / "scheduler_events.jsonl"

    def _callback() -> dict[str, object]:
        skipped_output.parent.mkdir(parents=True, exist_ok=True)
        skipped_output.write_text("should_not_exist", encoding="utf-8")
        return {"result": {"summary": str(skipped_output)}}

    summary = run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::fail",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(tmp_path / "gpu_fail" / "summary.json"),),
                command=_exit_command(exit_code=3, sleep_s=0.1),
                log_path=str(tmp_path / "gpu_fail" / "run.log"),
            ),
            SchedulerItem(
                item_id="gpu::peer",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(peer_output),),
                command=_delayed_touch_command(peer_output, sleep_s=0.2),
                log_path=str(tmp_path / "gpu_peer" / "run.log"),
            ),
            SchedulerItem(
                item_id="reduce::after_fail",
                phase="reduce_phase",
                kind="cpu_callback",
                deps=("gpu::fail",),
                expected_outputs=(str(skipped_output),),
                callback=_callback,
            ),
        ],
        config=SchedulerConfig(
            devices=("MIG-0", "MIG-1"),
            cleanup_stale_children=False,
            cancel_on_failure=False,
            raise_on_failure=False,
            event_log_path=str(event_log_path),
        ),
    )

    assert peer_output.exists()
    assert not skipped_output.exists()
    assert "gpu::peer" in summary["completed_items"]
    assert summary["failed_items"]["gpu::fail"]["returncode"] == 3
    skipped = summary["failed_items"]["reduce::after_fail"]
    assert skipped["failure_reason"] == "failed_dependency"
    assert skipped["upstream_failed_items"] == ["gpu::fail"]
    assert summary["cancel_on_failure"] is False
    assert summary["raise_on_failure"] is False

    events = [
        json.loads(line)
        for line in event_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "dependency_failed"
        and event.get("item_id") == "reduce::after_fail"
        for event in events
    )
    assert not any(
        event.get("event") == "command_cancelled"
        and event.get("item_id") == "gpu::peer"
        for event in events
    )


def test_scheduler_can_run_callback_after_failed_dependencies_when_enabled(
    tmp_path: Path,
) -> None:
    peer_output = tmp_path / "gpu_peer" / "summary.json"
    callback_output = tmp_path / "reduce_after_fail" / "summary.json"

    def _callback() -> dict[str, object]:
        callback_output.parent.mkdir(parents=True, exist_ok=True)
        callback_output.write_text("reduced", encoding="utf-8")
        return {"result": {"summary": str(callback_output)}}

    summary = run_scheduler(
        [
            SchedulerItem(
                item_id="gpu::fail",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(tmp_path / "gpu_fail" / "summary.json"),),
                command=_exit_command(exit_code=3, sleep_s=0.05),
                log_path=str(tmp_path / "gpu_fail" / "run.log"),
            ),
            SchedulerItem(
                item_id="gpu::peer",
                phase="gpu_phase",
                kind="gpu_command",
                expected_outputs=(str(peer_output),),
                command=_delayed_touch_command(peer_output, sleep_s=0.1),
                log_path=str(tmp_path / "gpu_peer" / "run.log"),
            ),
            SchedulerItem(
                item_id="reduce::after_fail",
                phase="reduce_phase",
                kind="cpu_callback",
                deps=("gpu::fail", "gpu::peer"),
                expected_outputs=(str(callback_output),),
                callback=_callback,
                run_on_failed_dependencies=True,
            ),
        ],
        config=SchedulerConfig(
            devices=("MIG-0", "MIG-1"),
            cleanup_stale_children=False,
            cancel_on_failure=False,
            raise_on_failure=False,
        ),
    )

    assert peer_output.exists()
    assert callback_output.exists()
    assert "gpu::fail" in summary["failed_items"]
    assert "gpu::peer" in summary["completed_items"]
    assert "reduce::after_fail" in summary["completed_items"]


def test_scheduler_status_stays_running_while_continuing_after_failure(tmp_path: Path) -> None:
    status_path = tmp_path / "scheduler_status.json"
    peer_output = tmp_path / "gpu_peer" / "summary.json"
    observed = None
    result: dict[str, object] = {}

    def _runner() -> None:
        result["summary"] = run_scheduler(
            [
                SchedulerItem(
                    item_id="gpu::fail",
                    phase="gpu_phase",
                    kind="gpu_command",
                    expected_outputs=(str(tmp_path / "gpu_fail" / "summary.json"),),
                    command=_exit_command(exit_code=3, sleep_s=0.1),
                    log_path=str(tmp_path / "gpu_fail" / "run.log"),
                ),
                SchedulerItem(
                    item_id="gpu::peer",
                    phase="gpu_phase",
                    kind="gpu_command",
                    expected_outputs=(str(peer_output),),
                    command=_delayed_touch_command(peer_output, sleep_s=1.2),
                    log_path=str(tmp_path / "gpu_peer" / "run.log"),
                ),
            ],
            config=SchedulerConfig(
                devices=("MIG-0", "MIG-1"),
                cleanup_stale_children=False,
                cancel_on_failure=False,
                raise_on_failure=False,
                status_path=str(status_path),
            ),
        )

    thread = threading.Thread(target=_runner)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                int(payload.get("failed_items", 0) or 0) >= 1
                and int(payload.get("active_items", 0) or 0) >= 1
            ):
                observed = payload
                break
        time.sleep(0.1)
    thread.join(timeout=10.0)
    assert thread.is_alive() is False
    assert observed is not None
    assert observed["state"] == "running"
    assert int(observed["failed_items"]) >= 1
    assert int(observed["active_items"]) >= 1


def test_scheduler_aborts_when_host_memory_floor_is_breached(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failure_snapshot_path = tmp_path / "scheduler_failure_snapshot.json"
    ok_output = tmp_path / "gpu_ok" / "summary.json"
    monkeypatch.setattr(
        scheduler,
        "_system_memory_snapshot",
        lambda: {
            "MemAvailable": "1024 kB",
            "SwapFree": "1024 kB",
        },
    )

    with pytest.raises(scheduler.SchedulerRunError):
        run_scheduler(
            [
                SchedulerItem(
                    item_id="gpu::0",
                    phase="gpu_phase",
                    kind="gpu_command",
                    expected_outputs=(str(ok_output),),
                    command=_delayed_touch_command(ok_output, sleep_s=5.0),
                    log_path=str(tmp_path / "gpu_ok" / "run.log"),
                )
            ],
            config=SchedulerConfig(
                devices=("MIG-0",),
                cleanup_stale_children=False,
                failure_snapshot_path=str(failure_snapshot_path),
                min_mem_available_kib=2048,
                min_swap_free_kib=2048,
            ),
        )

    failure_snapshot = json.loads(failure_snapshot_path.read_text(encoding="utf-8"))
    assert failure_snapshot["reason"] == "host_memory_floor_breached"
