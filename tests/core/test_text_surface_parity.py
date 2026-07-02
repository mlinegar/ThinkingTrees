"""Parity tests for the unified text-generation surface.

After the LM-interface unification, `/v1/chat/completions` and `/generate` are two
*transports* under one canonical `CHAT_OPENAI` surface. These tests pin the invariant
the unification promises: the same `ChatInput` produces the same `InferenceResponse`
*shape* regardless of which transport serves it, and transport selection routes as
documented (omni → generate; default chat engines → openai-chat).

See docs/lm_interface_unification_plan.md (verification items 1 and 4).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.core.engines import EngineSurface
from src.core.inference_engine import build_inference_engine
from src.runtime.contracts import ChatInput, InferenceRequest, TextOutput


class _FakeChatClient:
    """Minimal `ChatCompatibleClient` returning a controlled chat completion.

    `supports_batch_client = False` forces the engine's synchronous `.chat()`
    path (same mechanism `GenerateChatClient` uses), so the test never touches a
    real batch endpoint.
    """

    supports_batch_client = False

    def __init__(self, config: Any) -> None:
        self.config = config

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            content="openai-chat text",
            model="chat-model",
            prompt_tokens=3,
            completion_tokens=5,
            raw_response={"messages": list(messages), "kwargs": dict(kwargs)},
        )


class _FakeGenerateResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeGenerateSession:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, json: Dict[str, Any], timeout: float) -> _FakeGenerateResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeGenerateResponse(self.payload)


def _request() -> InferenceRequest:
    return InferenceRequest(
        surface=EngineSurface.CHAT_OPENAI,
        input=ChatInput(
            messages=[{"role": "user", "content": "Summarize this."}],
            max_tokens=32,
            temperature=0.0,
        ),
    )


def test_chat_and_generate_transports_share_response_shape() -> None:
    """Same ChatInput → same InferenceResponse shape across both transports."""

    chat_engine = build_inference_engine(
        "vllm",
        surface=EngineSurface.CHAT_OPENAI,
        base_url="http://localhost:8000/v1",
        model="chat-model",
        llm_client=_FakeChatClient(SimpleNamespace(model="chat-model")),
    )
    generate_engine = build_inference_engine(
        "vllm_omni",
        surface=EngineSurface.CHAT_OPENAI,
        base_url="http://localhost:8004",
        model="served-model",
        session=_FakeGenerateSession({"choices": [{"text": "generate text"}], "model": "served-model"}),
        transport="generate",
    )

    chat_response = chat_engine.execute(_request())
    generate_response = generate_engine.execute(_request())

    # Same surface, same output type, both convertible to the ModelResponse contract.
    assert chat_response.surface is EngineSurface.CHAT_OPENAI
    assert generate_response.surface is EngineSurface.CHAT_OPENAI
    assert isinstance(chat_response.output, TextOutput)
    assert isinstance(generate_response.output, TextOutput)
    assert chat_response.output.text == "openai-chat text"
    assert generate_response.output.text == "generate text"
    # The downstream ModelResponse projection works identically for both.
    assert chat_response.to_model_response().text == "openai-chat text"
    assert generate_response.to_model_response().text == "generate text"


def test_omni_engine_autoselects_generate_transport_without_explicit_flag() -> None:
    """vLLM-Omni has no chat surface, so it must route to /generate by default."""

    session = _FakeGenerateSession({"choices": [{"text": "auto"}], "model": "m"})
    engine = build_inference_engine(
        "vllm_omni",
        surface=EngineSurface.CHAT_OPENAI,
        base_url="http://localhost:8004",
        session=session,
    )
    response = engine.execute(_request())

    assert isinstance(response.output, TextOutput)
    assert session.calls and session.calls[0]["url"].endswith("/generate")


def test_default_chat_engine_does_not_use_generate_transport() -> None:
    """A plain vLLM chat engine must use the openai-chat client, never /generate."""

    session = _FakeGenerateSession({"choices": [{"text": "should-not-be-used"}]})
    engine = build_inference_engine(
        "vllm",
        surface=EngineSurface.CHAT_OPENAI,
        base_url="http://localhost:8000/v1",
        model="chat-model",
        llm_client=_FakeChatClient(SimpleNamespace(model="chat-model")),
        session=session,
    )
    response = engine.execute(_request())

    assert response.output.text == "openai-chat text"
    assert session.calls == []  # generate transport untouched


def test_diffusion_generate_surface_is_retired_for_engine_construction() -> None:
    with pytest.raises(ValueError, match="DIFFUSION_GENERATE has been retired"):
        build_inference_engine(
            "sglang",
            surface=EngineSurface.DIFFUSION_GENERATE,
            base_url="http://localhost:30000",
        )
