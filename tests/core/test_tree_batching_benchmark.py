"""Tests for tree-summary batching benchmark utilities."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.benchmark.tree_batching import (
    TreeBatchPointConfig,
    compute_tree_batch_budget,
    expand_tree_batch_grid,
    parse_positive_float_grid,
    parse_positive_int_grid,
    render_tree_batch_markdown,
    run_tree_batching_suite,
    summarize_metrics_snapshots,
    summarize_tree_batch_results,
    write_tree_batch_jsonl,
    write_tree_batch_markdown,
)
from src.core import async_utils


def test_parse_positive_grids_dedupe_and_sort() -> None:
    assert parse_positive_int_grid("8,1,4,4,2", name="x") == [1, 2, 4, 8]
    assert parse_positive_int_grid("0,4", name="workers", allow_zero=True) == [0, 4]
    assert parse_positive_float_grid("0.05,0.01,0.01", name="timeout") == [0.01, 0.05]


def test_parse_positive_grids_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        parse_positive_int_grid("0,1", name="x")
    with pytest.raises(ValueError):
        parse_positive_float_grid("0,0.1", name="x")


def test_expand_tree_batch_grid() -> None:
    points = expand_tree_batch_grid(
        leaf_tokens=[1000, 2000],
        summary_max_tokens=[256],
        max_concurrent_requests=[16],
        batch_sizes=[8, 16],
        batch_timeouts=[0.01],
        dspy_workers=[None, 4],
    )
    assert len(points) == 8
    assert points[0] == TreeBatchPointConfig(
        leaf_tokens=1000,
        summary_max_tokens=256,
        max_concurrent_requests=16,
        batch_size=8,
        batch_timeout=0.01,
        dspy_workers=None,
    )


def test_compute_tree_batch_budget_recommends_safe_request_density() -> None:
    report = compute_tree_batch_budget(
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        prompt_overhead_tokens=128,
        leaf_tokens=512,
        summary_max_tokens=256,
        safety_fraction=0.90,
    )
    assert report.leaf_request_tokens == 896
    assert report.merge_request_tokens == 896
    assert report.leaf_context_fits
    assert report.merge_context_fits
    assert report.recommended_leaf_concurrency == 4
    assert report.recommended_merge_concurrency == 4
    assert report.recommended_max_concurrent_requests == 4


def test_compute_tree_batch_budget_flags_context_overflow() -> None:
    report = compute_tree_batch_budget(
        max_model_len=1024,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        prompt_overhead_tokens=128,
        leaf_tokens=1200,
        summary_max_tokens=256,
    )
    assert not report.leaf_context_fits
    assert report.recommended_max_concurrent_requests == 0
    assert "exceeds max_model_len" in report.notes


def test_missing_metrics_summary_is_empty_but_valid() -> None:
    summary = summarize_metrics_snapshots([])
    assert summary.samples == 0
    assert summary.reachable_samples == 0
    assert summary.max_kv_cache_usage_pct is None
    assert summary.max_requests_waiting is None
    assert summary.avg_prefix_cache_hit_rate is None


def test_to_thread_worker_env_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    async_utils.configure_to_thread_max_workers(None)
    monkeypatch.setenv("TT_TO_THREAD_MAX_WORKERS", "7")
    assert async_utils.resolve_to_thread_max_workers() == 7

    async_utils.configure_to_thread_max_workers(3)
    assert async_utils.resolve_to_thread_max_workers() == 3

    async_utils.configure_to_thread_max_workers(None)
    monkeypatch.delenv("TT_TO_THREAD_MAX_WORKERS", raising=False)


def test_fake_tree_batching_suite_writes_artifacts(tmp_path: Path) -> None:
    points = expand_tree_batch_grid(
        leaf_tokens=[500, 1000],
        summary_max_tokens=[128, 256],
        max_concurrent_requests=[8],
        batch_sizes=[8],
        batch_timeouts=[0.02],
        dspy_workers=[None],
    )
    suite = asyncio.run(
        run_tree_batching_suite(
            points=points,
            base_url="http://localhost:8000/v1",
            total_input_tokens=4000,
            document_count=2,
            request_timeout_seconds=30.0,
            await_response_timeout_seconds=None,
            metrics_poll_seconds=0.0,
            api_key="EMPTY",
            temperature=0.2,
            rubric="Preserve policy facts.",
            fake=True,
            max_model_len=32768,
            max_num_seqs=128,
            max_num_batched_tokens=16384,
            prompt_overhead_tokens=256,
            budget_safety_fraction=0.9,
        )
    )

    assert len(suite.points) == 4
    assert suite.summary.best_tokens_point is not None
    assert suite.summary.leaf_stability

    output_jsonl = tmp_path / "tree_batch.jsonl"
    output_md = tmp_path / "tree_batch.md"
    write_tree_batch_jsonl(output_jsonl=output_jsonl, suite=suite)
    write_tree_batch_markdown(output_markdown=output_md, suite=suite)

    jsonl_lines = output_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 4
    markdown = output_md.read_text(encoding="utf-8")
    assert "Tree Batching Throughput Sweep" in markdown
    assert "Leaf-Size Stability" in markdown


def test_summarize_tree_batch_results_ranks_best_point() -> None:
    points = expand_tree_batch_grid(
        leaf_tokens=[500, 1000],
        summary_max_tokens=[128],
        max_concurrent_requests=[8],
        batch_sizes=[8],
        batch_timeouts=[0.02],
        dspy_workers=[None],
    )
    suite = asyncio.run(
        run_tree_batching_suite(
            points=points,
            base_url="http://localhost:8000/v1",
            total_input_tokens=4000,
            document_count=2,
            request_timeout_seconds=30.0,
            await_response_timeout_seconds=None,
            metrics_poll_seconds=0.0,
            api_key="EMPTY",
            temperature=0.2,
            rubric="Preserve policy facts.",
            fake=True,
            max_model_len=32768,
            max_num_seqs=128,
            max_num_batched_tokens=16384,
            prompt_overhead_tokens=256,
            budget_safety_fraction=0.9,
        )
    )
    summary = summarize_tree_batch_results(suite.points)
    assert summary.best_tokens_point is not None
    assert summary.best_docs_point is not None
    assert summary.best_tokens_point.tokens_per_second > 0.0
    assert "Top" in render_tree_batch_markdown(suite)

