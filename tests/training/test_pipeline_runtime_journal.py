import argparse
from pathlib import Path

from src.training.run_pipeline import (
    _initialize_pipeline_runtime_state,
    _record_pipeline_runtime_phase,
    _write_json_atomic,
)


def _args(*, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(resume=resume)


def test_initialize_pipeline_runtime_state_fresh(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline_runtime_state.json"
    state, resumed = _initialize_pipeline_runtime_state(
        state_path=state_path,
        output_dir=tmp_path,
        args=_args(resume=False),
    )

    assert resumed is False
    assert state["status"] == "running"
    assert state["current_phase"] == "setup"
    assert state["resume_count"] == 0
    assert isinstance(state.get("events"), list)
    assert state["events"][-1]["message"] == "start"


def test_initialize_pipeline_runtime_state_resume_marks_running_as_interrupted(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline_runtime_state.json"
    existing = {
        "version": 1,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:10:00",
        "resume_count": 2,
        "status": "running",
        "current_phase": "phase2",
        "phases": {
            "phase1": {"status": "completed", "updated_at": "2026-01-01T00:05:00"},
            "phase2": {"status": "running", "updated_at": "2026-01-01T00:10:00"},
        },
        "events": [],
    }
    _write_json_atomic(state_path, existing)

    state, resumed = _initialize_pipeline_runtime_state(
        state_path=state_path,
        output_dir=tmp_path,
        args=_args(resume=True),
    )

    assert resumed is True
    assert state["resume_count"] == 3
    assert state["status"] == "running"
    assert state["current_phase"] == "setup"
    assert state["phases"]["phase2"]["status"] == "interrupted"
    assert state["events"][-1]["message"] == "resume"


def test_record_pipeline_runtime_phase_tracks_attempts_details_and_errors(tmp_path: Path) -> None:
    state, _ = _initialize_pipeline_runtime_state(
        state_path=tmp_path / "pipeline_runtime_state.json",
        output_dir=tmp_path,
        args=_args(resume=False),
    )

    state = _record_pipeline_runtime_phase(
        state,
        phase="phase2",
        phase_status="running",
        message="phase2_start",
    )
    state = _record_pipeline_runtime_phase(
        state,
        phase="phase2",
        phase_status="completed",
        details={"artifact_count": 3},
    )
    state = _record_pipeline_runtime_phase(
        state,
        phase="phase2",
        phase_status="running",
        message="phase2_retry",
    )
    state = _record_pipeline_runtime_phase(
        state,
        phase="phase2",
        phase_status="failed",
        pipeline_status="failed",
        error="boom",
    )

    phase_state = state["phases"]["phase2"]
    assert phase_state["attempts"] == 2
    assert phase_state["status"] == "failed"
    assert phase_state["details"]["artifact_count"] == 3
    assert phase_state["error"] == "boom"
    assert state["status"] == "failed"
    assert state["last_error"] == "boom"
    assert state["events"][-1]["phase_status"] == "failed"

