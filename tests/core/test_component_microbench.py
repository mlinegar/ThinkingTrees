"""Tests for component-level microbenchmarks."""

import pytest

from src.benchmark.component_microbench import (
    available_benchmarks,
    run_selected_benchmarks,
)


def test_available_benchmarks_expected():
    names = set(available_benchmarks())
    assert names == {"chunker", "conditional_memory", "prompting", "tree_builder"}


def test_run_prompting_microbench_shape():
    payload = run_selected_benchmarks(["prompting"], iterations=30)
    assert "created_at" in payload
    benchmarks = payload["benchmarks"]
    assert set(benchmarks.keys()) == {"prompting"}

    prompting = benchmarks["prompting"]
    assert prompting["name"] == "prompting"
    metrics = prompting["metrics"]
    assert metrics["iterations_per_second"] > 0
    assert 0.0 <= metrics["parse_success_rate"] <= 1.0


def test_run_conditional_memory_microbench_shape():
    payload = run_selected_benchmarks(["conditional_memory"], iterations=40)
    memory = payload["benchmarks"]["conditional_memory"]

    assert memory["name"] == "conditional_memory"
    metrics = memory["metrics"]
    assert metrics["ops_per_second"] > 0
    assert metrics["lookup_latency_ms_p95"] >= 0
    assert 0.0 <= metrics["lookup_hit_rate_runtime"] <= 1.0
    assert isinstance(metrics["report"], dict)


def test_run_all_benchmarks():
    payload = run_selected_benchmarks(["all"], iterations=8)
    names = set(payload["benchmarks"].keys())
    assert names == {"chunker", "conditional_memory", "prompting", "tree_builder"}


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError):
        run_selected_benchmarks(["does_not_exist"], iterations=10)
