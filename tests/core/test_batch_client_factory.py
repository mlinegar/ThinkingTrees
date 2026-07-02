from __future__ import annotations

import pytest

from src.core.batch_client_factory import build_batch_client
from src.core.batch_processor import AsyncBatchLLMClient, MultiServerBatchClient


def test_factory_builds_single_server_client_with_transport_settings() -> None:
    client = build_batch_client(
        base_url="http://localhost:8000/v1/",
        model="demo-model",
        api_key="secret",
        max_concurrent=0,
        batch_size=0,
        batch_timeout=-1.0,
        request_timeout=0.5,
    )

    assert isinstance(client, AsyncBatchLLMClient)
    assert client.base_url == "http://localhost:8000/v1"
    assert client.model == "demo-model"
    assert client.api_key == "secret"
    assert client.max_concurrent == 1
    assert client.batch_size == 1
    assert client.batch_timeout == 0.0
    assert client.request_timeout == 1.0


def test_factory_builds_multi_server_client_with_routing_and_model_propagation() -> None:
    client = build_batch_client(
        server_urls=["http://localhost:8000/v1/", "http://localhost:8001/v1"],
        model="fleet-model",
        api_key="fleet-key",
        routing_policy="round_robin",
        max_concurrent=2,
        batch_size=3,
        request_timeout=4.0,
    )

    assert isinstance(client, MultiServerBatchClient)
    assert client.servers == ["http://localhost:8000/v1", "http://localhost:8001/v1"]
    assert client.routing_policy.value == "round_robin"
    assert [inner.model for inner in client.clients] == ["fleet-model", "fleet-model"]
    assert [inner.api_key for inner in client.clients] == ["fleet-key", "fleet-key"]
    assert [inner.max_concurrent for inner in client.clients] == [2, 2]
    assert [inner.batch_size for inner in client.clients] == [3, 3]
    assert [inner.request_timeout for inner in client.clients] == [4.0, 4.0]


def test_factory_normalizes_comma_separated_base_urls() -> None:
    client = build_batch_client(
        api_base="http://localhost:8000/v1, http://localhost:8001/v1",
        routing_policy="document_affinity",
    )

    assert isinstance(client, MultiServerBatchClient)
    assert client.servers == ["http://localhost:8000/v1", "http://localhost:8001/v1"]
    assert client.routing_policy.value == "document_affinity"


def test_factory_requires_endpoint() -> None:
    with pytest.raises(ValueError, match="at least one endpoint"):
        build_batch_client()
