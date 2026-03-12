from src.core.vllm_metrics import ServerMetrics, VLLMMetricsCollector


def test_parse_prometheus_accepts_vllm_names():
    text = "\n".join(
        [
            "vllm:gpu_cache_usage_perc 0.75",
            "vllm:num_requests_waiting 8",
            "vllm:num_requests_running 3",
            "vllm:prefix_cache_hit_rate 0.40",
        ]
    )
    m = ServerMetrics(port=8000)
    VLLMMetricsCollector._parse_prometheus(text, m)

    assert m.kv_cache_usage_pct == 0.75
    assert m.num_requests_waiting == 8
    assert m.num_requests_running == 3
    assert m.prefix_cache_hit_rate == 0.40


def test_parse_prometheus_accepts_sglang_names():
    text = "\n".join(
        [
            "sglang:kv_cache_usage 0.62",
            "sglang:num_requests_waiting 11",
            "sglang:num_requests_running 4",
            "sglang:cache_hit_rate 0.55",
        ]
    )
    m = ServerMetrics(port=30000)
    VLLMMetricsCollector._parse_prometheus(text, m)

    assert m.kv_cache_usage_pct == 0.62
    assert m.num_requests_waiting == 11
    assert m.num_requests_running == 4
    assert m.prefix_cache_hit_rate == 0.55
