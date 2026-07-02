from __future__ import annotations

import pytest

from src.core.engines import EngineSurface, EngineType
from src.runtime.contracts import ChatInput, EmbeddingInput, EmbeddingOutput
from src.runtime.inference_context import (
    RuntimeInferenceContext,
    normalize_role_config,
    normalize_surface_config,
)


@pytest.mark.parametrize(
    "key",
    ["answerer", "state_operator", "model", "resources", "surfaces", "run", "run_id", "typo"],
)
def test_normalize_surface_config_rejects_unknown_public_keys(key: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported runtime-eval config key: {key}"):
        normalize_surface_config({key: {}})


def test_normalize_surface_config_accepts_paper_facing_runtime_names() -> None:
    spec = {
        "scorer": {
            "engine": "vllm",
            "endpoint": "http://localhost:8000/v1",
            "model": "scorer-model",
            "batch_size": 4,
        },
        "embedder": {
            "engine": "vllm",
            "endpoint": "http://localhost:8003/v1",
            "model": "embed-model",
        },
        "state_model": {
            "kind": "neural_operator",
            "model": "state-model",
            "checkpoint": "/tmp/state.pt",
            "device": "cuda",
        },
    }

    surfaces = normalize_surface_config(spec)
    roles = normalize_role_config(spec)

    assert surfaces["chat_openai"]["base_url"] == "http://localhost:8000/v1"
    assert surfaces["chat_openai"]["model"] == "scorer-model"
    assert surfaces["chat_openai"]["batch_size"] == 4
    assert roles["summarizer"]["defaulted_from"] == "scorer"
    assert roles["summarizer"]["model"] == "scorer-model"
    assert surfaces["embedding"]["base_url"] == "http://localhost:8003/v1"
    assert surfaces["embedding"]["model"] == "embed-model"
    assert surfaces["operator"]["engine"] == EngineType.NATIVE_OPERATOR.value
    assert surfaces["operator"]["model"] == "state-model"
    assert surfaces["operator"]["checkpoint_path"] == "/tmp/state.pt"
    assert surfaces["operator"]["kind"] == "neural_operator"
    assert surfaces["operator"]["device"] == "cuda"


def test_separate_summarizer_endpoint_is_preserved() -> None:
    roles = normalize_role_config(
        {
            "scorer": {"endpoint": "http://localhost:8000/v1", "model": "scorer"},
            "summarizer": {"endpoint": "http://localhost:8001/v1", "model": "summarizer"},
        }
    )

    assert roles["scorer"]["base_url"] == "http://localhost:8000/v1"
    assert roles["summarizer"]["base_url"] == "http://localhost:8001/v1"
    assert "defaulted_from" not in roles["summarizer"]


def test_runtime_inference_context_lazy_surfaces_and_capabilities() -> None:
    ctx = RuntimeInferenceContext(
        {
            "surfaces": {
                "chat_openai": {
                    "engine": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "model": "mock-chat",
                },
                "embedding": {
                    "engine": "vllm",
                    "base_url": "http://localhost:8003/v1",
                    "model": "mock-embed",
                    "batch_size": 8,
                },
            }
        },
        mock=True,
    )

    assert ctx.has_surface(EngineSurface.CHAT_OPENAI) is True
    assert ctx.has_surface(EngineSurface.EMBEDDING) is True
    assert ctx.has_surface(EngineSurface.OPERATOR) is False
    assert ctx.capabilities(EngineSurface.CHAT_OPENAI).supports_logprobs is False
    assert ctx.capabilities(EngineSurface.EMBEDDING).supports_batching is True

    chat_response = ctx.execute(
        EngineSurface.CHAT_OPENAI,
        ChatInput(messages=[{"role": "user", "content": "hello"}], max_tokens=4),
    )
    assert chat_response.model_id == "mock"
    assert "hello" in chat_response.to_model_response().text

    embed_response = ctx.execute(
        EngineSurface.EMBEDDING,
        EmbeddingInput(texts=["alpha beta", "beta gamma"]),
    )
    assert isinstance(embed_response.output, EmbeddingOutput)
    assert len(embed_response.output.vectors) == 2


def test_runtime_inference_context_reports_missing_surface() -> None:
    ctx = RuntimeInferenceContext({}, mock=True)

    with pytest.raises(RuntimeError, match="chat_openai"):
        ctx.surface_config(EngineSurface.CHAT_OPENAI)
