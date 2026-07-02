from __future__ import annotations

from src.core.engines import EngineSurface
from src.core.inference_engine import build_inference_engine
from src.runtime.contracts import EmbeddingInput, EmbeddingOutput, InferenceRequest
from treepo.llm import OpenAICompatibleEmbeddingClient, build_embedding_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeEmbeddingSession:
    def __init__(self) -> None:
        self.posts = []

    def get(self, url, *, headers=None, timeout=None):
        return _FakeResponse({"data": [{"id": "fake-embedding-model"}]})

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        inputs = list((json or {}).get("input", []))
        # Return out of order to prove the client restores OpenAI's index order.
        data = [
            {"index": idx, "embedding": [float(len(text)), float(idx)]}
            for idx, text in reversed(list(enumerate(inputs)))
        ]
        return _FakeResponse({"data": data})


def test_vllm_embedding_surface_mock_executes_openai_compatible_request() -> None:
    engine = build_inference_engine(
        "vllm",
        surface=EngineSurface.EMBEDDING,
        base_url="http://localhost:8003/v1",
        model="mock-embedding",
        mock=True,
    )

    response = engine.execute(
        InferenceRequest(
            surface=EngineSurface.EMBEDDING,
            input=EmbeddingInput(texts=["alpha beta", "beta gamma"]),
        )
    )

    assert response.surface is EngineSurface.EMBEDDING
    assert response.model_id == "mock-embedding"
    assert isinstance(response.output, EmbeddingOutput)
    assert len(response.output.vectors) == 2
    assert len(response.output.vectors[0]) == response.telemetry["embedding_dim"]
    assert response.usage["input_count"] == 2


def test_sglang_embedding_surface_can_use_same_simple_protocol() -> None:
    engine = build_inference_engine(
        "sglang",
        surface=EngineSurface.EMBEDDING,
        model="mock-embedding",
        mock=True,
    )

    response = engine.execute(
        InferenceRequest(
            surface=EngineSurface.EMBEDDING,
            input=EmbeddingInput(texts=["alpha", "beta"]),
        )
    )

    assert response.surface is EngineSurface.EMBEDDING
    assert isinstance(response.output, EmbeddingOutput)
    assert len(response.output.vectors) == 2


def test_canonical_hash_embedding_client_is_deterministic() -> None:
    client = build_embedding_client("hashing", embedding_dim=8)

    first = client.embed_texts(["alpha beta", "alpha beta"])
    second = client.embed_texts(["alpha beta"])

    assert first[0] == first[1] == second[0]
    assert len(first[0]) == 8


def test_openai_compatible_embedding_client_restores_response_order_and_caches() -> None:
    session = _FakeEmbeddingSession()
    client = OpenAICompatibleEmbeddingClient(
        api_base="http://localhost:8003/v1",
        model="fake-embedding-model",
        session=session,
        cache_enabled=True,
    )

    first = client.embed_texts(["a", "abcd"])
    second = client.embed_texts(["a", "abcd"])

    assert first == [[1.0, 0.0], [4.0, 1.0]]
    assert second == first
    assert len(session.posts) == 1
    assert session.posts[0]["json"] == {
        "model": "fake-embedding-model",
        "input": ["a", "abcd"],
    }
