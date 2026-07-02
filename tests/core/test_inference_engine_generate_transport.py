from __future__ import annotations

from typing import Any, Dict, List

from src.core.engines import EngineSurface
from src.core.inference_engine import build_inference_engine
from src.runtime.contracts import ChatInput, InferenceRequest, TextOutput


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self.payload)


def test_generate_transport_returns_chat_text_output_shape() -> None:
    session = _FakeSession({"choices": [{"text": "generated text"}], "model": "served-model"})
    engine = build_inference_engine(
        "vllm_omni",
        surface=EngineSurface.CHAT_OPENAI,
        base_url="http://localhost:8004",
        model="requested-model",
        session=session,
        transport="generate",
    )

    response = engine.execute(
        InferenceRequest(
            surface=EngineSurface.CHAT_OPENAI,
            input=ChatInput(
                messages=[{"role": "user", "content": "Summarize this."}],
                max_tokens=32,
                temperature=0.0,
            ),
        )
    )

    assert isinstance(response.output, TextOutput)
    assert response.output.text == "generated text"
    assert response.surface is EngineSurface.CHAT_OPENAI
    assert session.calls[0]["url"] == "http://localhost:8004/generate"
    assert session.calls[0]["json"]["text"] == "Summarize this."
    assert session.calls[0]["json"]["max_tokens"] == 32
    assert session.calls[0]["json"]["temperature"] == 0.0
