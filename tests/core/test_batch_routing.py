from src.core.batch_processor import (
    BatchRequest,
    MultiServerBatchClient,
    RoutingPolicy,
)


def _set_pending(client, n: int) -> None:
    client._pending_futures = {f"r{i}": object() for i in range(int(n))}


def test_round_robin_routing_cycles_servers():
    router = MultiServerBatchClient(
        servers=["http://localhost:8000/v1", "http://localhost:8002/v1"],
        routing_policy=RoutingPolicy.ROUND_ROBIN,
    )
    req_a = BatchRequest(request_id="a", messages=[])
    req_b = BatchRequest(request_id="b", messages=[])
    req_c = BatchRequest(request_id="c", messages=[])

    c1 = router._get_client_for_request(req_a)
    c2 = router._get_client_for_request(req_b)
    c3 = router._get_client_for_request(req_c)

    assert c1.base_url == "http://localhost:8000/v1"
    assert c2.base_url == "http://localhost:8002/v1"
    assert c3.base_url == "http://localhost:8000/v1"


def test_document_affinity_is_stable_for_same_key():
    router = MultiServerBatchClient(
        servers=["http://localhost:8000/v1", "http://localhost:8002/v1", "http://localhost:8004/v1"],
        routing_policy=RoutingPolicy.DOCUMENT_AFFINITY,
    )
    req_1 = BatchRequest(request_id="r1", messages=[], document_id="doc_123")
    req_2 = BatchRequest(request_id="r2", messages=[], document_id="doc_123")

    c1 = router._get_client_for_request(req_1)
    c2 = router._get_client_for_request(req_2)
    assert c1.base_url == c2.base_url


def test_routing_key_takes_precedence_over_document_id():
    router = MultiServerBatchClient(
        servers=["http://localhost:8000/v1", "http://localhost:8002/v1", "http://localhost:8004/v1"],
        routing_policy=RoutingPolicy.DOCUMENT_AFFINITY,
    )
    req_a = BatchRequest(request_id="r1", messages=[], document_id="doc_a", routing_key="group_1")
    req_b = BatchRequest(request_id="r2", messages=[], document_id="doc_b", routing_key="group_1")

    c1 = router._get_client_for_request(req_a)
    c2 = router._get_client_for_request(req_b)
    assert c1.base_url == c2.base_url


def test_affinity_load_aware_spills_to_least_loaded_server():
    router = MultiServerBatchClient(
        servers=["http://localhost:8000/v1", "http://localhost:8002/v1", "http://localhost:8004/v1"],
        routing_policy=RoutingPolicy.AFFINITY_LOAD_AWARE,
    )
    req = BatchRequest(request_id="r1", messages=[], document_id="doc_hot")

    preferred = router._affinity_client(req, load_aware=False)
    for client in router.clients:
        _set_pending(client, 50)
    _set_pending(preferred, 200)

    routed = router._get_client_for_request(req)
    assert routed.base_url != preferred.base_url
    assert routed.pending_count == 50
    assert router.routing_stats["affinity_spillovers"] >= 1


def test_multi_server_stats_aggregate_cache_counters():
    router = MultiServerBatchClient(
        servers=["http://localhost:8000/v1", "http://localhost:8002/v1"],
        routing_policy=RoutingPolicy.ROUND_ROBIN,
    )
    router.clients[0].stats.cache_hits = 3
    router.clients[0].stats.cache_misses = 7
    router.clients[0].stats.cache_writes = 2
    router.clients[1].stats.cache_hits = 5
    router.clients[1].stats.cache_misses = 11
    router.clients[1].stats.cache_writes = 4

    aggregate = router.stats
    assert aggregate.cache_hits == 8
    assert aggregate.cache_misses == 18
    assert aggregate.cache_writes == 6


def test_routing_stats_track_policy_and_server_distribution():
    router = MultiServerBatchClient(
        servers=["http://localhost:8000/v1", "http://localhost:8002/v1"],
        routing_policy=RoutingPolicy.ROUND_ROBIN,
    )
    router._get_client_for_request(BatchRequest(request_id="r1", messages=[]))
    router._get_client_for_request(BatchRequest(request_id="r2", messages=[]))
    router._get_client_for_request(BatchRequest(request_id="r3", messages=[]))

    stats = router.routing_stats
    assert stats["policy"] == "round_robin"
    assert stats["policy_counts"]["round_robin"] == 3
    assert sum(stats["by_server"].values()) == 3

    diagnostics = router.diagnostics
    assert diagnostics["routing"]["policy"] == "round_robin"
    assert len(diagnostics["servers"]) == 2
