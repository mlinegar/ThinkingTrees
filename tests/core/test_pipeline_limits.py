"""Unit tests for throughput-limit sweep utilities."""

import pytest

from src.benchmark.pipeline_limits import (
    SweepPoint,
    _is_valid_rile_score_response,
    expand_genrm_steps,
    parse_concurrency_grid,
    parse_genrm_modes,
    summarize_step,
)


def _point(step: str, concurrency: int, success: int, total: int, req_per_s: float, p95_ms: float) -> SweepPoint:
    failed = total - success
    return SweepPoint(
        step=step,
        concurrency=concurrency,
        total_requests=total,
        successful_requests=success,
        failed_requests=failed,
        timeout_errors=0,
        network_errors=0,
        server_errors=failed,
        wall_seconds=10.0,
        requests_per_second=req_per_s,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        tokens_per_second=0.0,
        latency_avg_ms=p95_ms,
        latency_p50_ms=p95_ms,
        latency_p95_ms=p95_ms,
    )


def test_parse_concurrency_grid_dedupes_and_sorts():
    assert parse_concurrency_grid("8,1,4,4,2") == [1, 2, 4, 8]


def test_parse_concurrency_grid_rejects_non_positive():
    with pytest.raises(ValueError):
        parse_concurrency_grid("0,1,2")


def test_parse_genrm_modes_and_expand_steps():
    modes = parse_genrm_modes("fast,think")
    expanded = expand_genrm_steps(["task_single", "genrm_batch", "genrm_raw"], modes)
    assert expanded == [
        "task_single",
        "genrm_batch_fast",
        "genrm_batch_think",
        "genrm_raw_fast",
        "genrm_raw_think",
    ]


def test_summarize_step_prefers_best_stable_throughput():
    points = [
        _point("task_single", 1, 10, 10, 1.0, 100.0),
        _point("task_single", 2, 10, 10, 1.9, 120.0),
        _point("task_single", 4, 9, 10, 2.3, 150.0),  # unstable at 90%
    ]
    summary = summarize_step(
        step="task_single",
        points=points,
        min_success_rate=0.95,
        max_p95_latency_ms=0.0,
    )
    assert summary.recommended_concurrency == 2
    assert summary.max_stable_concurrency == 2
    assert summary.peak_req_per_s_concurrency == 4


def test_summarize_step_obeys_latency_threshold():
    points = [
        _point("genrm_batch", 1, 10, 10, 0.5, 500.0),
        _point("genrm_batch", 2, 10, 10, 0.7, 1500.0),
    ]
    summary = summarize_step(
        step="genrm_batch",
        points=points,
        min_success_rate=0.99,
        max_p95_latency_ms=1000.0,
    )
    assert summary.recommended_concurrency == 1
    assert summary.max_stable_concurrency == 1


def test_score_response_validator_accepts_clean_numeric_only():
    assert _is_valid_rile_score_response("0")
    assert _is_valid_rile_score_response("-72.5")
    assert _is_valid_rile_score_response("  43  ")
    assert _is_valid_rile_score_response("score: 42")
    assert _is_valid_rile_score_response("42\nextra")


def test_score_response_validator_rejects_non_numeric_or_out_of_range():
    assert not _is_valid_rile_score_response("no score present")
    assert not _is_valid_rile_score_response("101")
