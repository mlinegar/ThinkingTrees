import logging
import urllib.request

from src.training.run_pipeline import (
    _aggregate_batch_run_telemetry,
    _log_server_metrics_sync,
    _parse_prom_value_any,
)


def test_parse_prom_value_any_supports_sglang_names():
    text = "\n".join(
        [
            "sglang:num_requests_waiting 7",
            "sglang:num_requests_running 3",
        ]
    )
    waiting, waiting_name = _parse_prom_value_any(
        text,
        ["vllm:num_requests_waiting", "sglang:num_requests_waiting"],
    )
    running, running_name = _parse_prom_value_any(
        text,
        ["vllm:num_requests_running", "sglang:num_requests_running"],
    )
    assert waiting == 7.0
    assert waiting_name == "sglang:num_requests_waiting"
    assert running == 3.0
    assert running_name == "sglang:num_requests_running"


def test_log_server_metrics_sync_parses_sglang_aliases(monkeypatch):
    metrics_text = "\n".join(
        [
            "sglang:kv_cache_usage 0.33",
            "sglang:cache_hit_rate 0.42",
            "sglang:num_requests_waiting 12",
            "sglang:num_requests_running 2",
        ]
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return metrics_text.encode("utf-8")

    def _fake_urlopen(*args, **kwargs):
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    rows = _log_server_metrics_sync([30000], logging.getLogger("test.telemetry"), label="unit")
    assert len(rows) == 1
    row = rows[0]
    assert row["reachable"] is True
    assert row["kv_cache_usage_pct"] == 0.33
    assert row["prefix_cache_hit_rate"] == 0.42
    assert row["queue_waiting"] == 12
    assert row["queue_running"] == 2
    assert row["metric_names"]["prefix_cache_hit_rate"] == "sglang:cache_hit_rate"


def test_aggregate_batch_run_telemetry_summarizes_cache_routing_and_recovery():
    runs = [
        {
            "phase": "train",
            "docs_total": 10,
            "docs_successful": 9,
            "elapsed_seconds": 5.0,
            "llm_stats": {
                "total_tokens": 1000,
                "cache_hits": 20,
                "cache_misses": 30,
                "cache_writes": 15,
            },
            "diagnostics": {
                "routing": {"by_server": {"http://localhost:8000/v1": 12}},
                "servers": [
                    {
                        "recovery": {
                            "attempts": 2,
                            "successes": 1,
                            "failures": 1,
                            "skipped_cooldown": 0,
                            "retry_attempts": 3,
                            "retry_after_recovery": 1,
                        },
                        "errors": {
                            "status_counts": {"500": 1},
                            "type_counts": {"TimeoutError": 2},
                        },
                    }
                ],
            },
        },
        {
            "phase": "val",
            "docs_total": 4,
            "docs_successful": 4,
            "elapsed_seconds": 2.0,
            "llm_stats": {
                "total_tokens": 600,
                "cache_hits": 10,
                "cache_misses": 10,
                "cache_writes": 8,
            },
            "diagnostics": {
                "routing": {"by_server": {"http://localhost:8002/v1": 7}},
                "servers": [
                    {
                        "recovery": {
                            "attempts": 1,
                            "successes": 1,
                            "failures": 0,
                            "skipped_cooldown": 1,
                            "retry_attempts": 1,
                            "retry_after_recovery": 1,
                        },
                        "errors": {
                            "status_counts": {"503": 2},
                            "type_counts": {"ClientConnectorError": 1},
                        },
                    }
                ],
            },
        },
    ]

    agg = _aggregate_batch_run_telemetry(runs)
    assert agg["run_count"] == 2
    assert agg["docs_total"] == 14
    assert agg["docs_successful"] == 13
    assert agg["llm"]["total_tokens"] == 1600
    assert agg["llm"]["cache_hits"] == 30
    assert agg["llm"]["cache_misses"] == 40
    assert agg["llm"]["cache_writes"] == 23
    assert agg["recovery"]["attempts"] == 3
    assert agg["recovery"]["successes"] == 2
    assert agg["recovery"]["failures"] == 1
    assert agg["recovery"]["retry_attempts"] == 4
    assert agg["routing"]["by_server"]["http://localhost:8000/v1"] == 12
    assert agg["routing"]["by_server"]["http://localhost:8002/v1"] == 7
    assert agg["errors"]["status_counts"]["500"] == 1
    assert agg["errors"]["status_counts"]["503"] == 2
