"""Tests for DSPy LM transport backed by AsyncBatchLLMClient."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

from src.config.dspy_config import (
    create_local_engine_lm,
    create_local_engine_lm_with_manager,
    create_vllm_lm,
    create_vllm_lm_multi,
)
from src.core.batch_processor import BatchResponse
from src.core.dspy_batch_client import (
    BatchedDSPyLM,
    _infer_single_dspy_json_output_field,
    _infer_single_dspy_output_field,
    _maybe_wrap_bare_dspy_field_response,
)
from src.core.engines import EngineType, LocalChatEndpoints


class FakeBridge:
    def __init__(self, *, content: str = "batched response") -> None:
        self.content = content
        self.calls: List[Dict[str, Any]] = []
        self.closed = False

    def request(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: Optional[Dict[str, Any]],
        extra_request_params: Optional[Dict[str, Any]],
        timeout: Optional[float],
    ) -> BatchResponse:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": chat_template_kwargs,
                "extra_request_params": extra_request_params,
                "timeout": timeout,
            }
        )
        return BatchResponse(
            request_id="fake",
            content=self.content,
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            latency_ms=12.0,
        )

    def submit(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: Optional[Dict[str, Any]],
        extra_request_params: Optional[Dict[str, Any]],
    ) -> Future[BatchResponse]:
        future: Future[BatchResponse] = Future()
        future.set_result(
            self.request(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                chat_template_kwargs=chat_template_kwargs,
                extra_request_params=extra_request_params,
                timeout=None,
            )
        )
        return future

    def close(self) -> None:
        self.closed = True


def test_bare_dspy_completion_payload_is_wrapped_for_single_output_field() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "[[ ## prompt ## ]]\ninput\n\n"
                "[[ ## completion ## ]]\noutput\n\n"
                "[[ ## completed ## ]]"
            ),
        }
    ]
    raw = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'

    assert _infer_single_dspy_output_field(messages) == "completion"
    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert wrapped.startswith("[[ ## completion ## ]]\n")
    assert raw in wrapped
    assert wrapped.rstrip().endswith("[[ ## completed ## ]]")


def test_bare_dspy_payload_wrapper_does_not_guess_multi_output_signature() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "[[ ## summary ## ]]\ninput\n\n"
                "[[ ## reasoning ## ]]\nreason\n\n"
                "[[ ## score ## ]]\nscore\n\n"
                "[[ ## completed ## ]]"
            ),
        }
    ]
    raw = "0.7"

    assert _infer_single_dspy_output_field(messages) is None
    assert _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages) == raw


def test_bare_dspy_payload_is_wrapped_for_json_adapter_single_output_field() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "Outputs will be a JSON object with the following fields.\n"
                '{\n  "completion": "str"\n}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`completion`."
            ),
        },
    ]
    raw = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'

    assert _infer_single_dspy_json_output_field(messages) == "completion"
    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    payload = json.loads(wrapped)
    assert payload == {"completion": raw}


def test_bare_dspy_payload_is_wrapped_for_json_adapter_scores_field() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`scores_json`."
            ),
        }
    ]
    raw = '{"rile": 0.5, "domain_1": 0.25}'

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert json.loads(wrapped) == {"scores_json": raw}


def test_json_adapter_payload_wrapper_does_not_double_wrap() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`completion`."
            ),
        }
    ]
    raw = '{"completion": "{\\"cmp_state\\": {}}"}'

    assert _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages) == raw


def test_json_adapter_payload_wrapper_cleans_single_field_envelope() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`completion`."
            ),
        }
    ]
    inner = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'
    raw = json.dumps({"completion": inner}) + "\n```"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert json.loads(wrapped) == {"completion": inner}


def test_json_adapter_payload_wrapper_repairs_unescaped_single_field_json_string() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`completion`."
            ),
        }
    ]
    raw = """{
  \"completion\": "{
    \"cmp_state\": {
      \"compact_targets\": {
        \"rile\": 0.5
      }
    }
  }"
}"""

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    payload = json.loads(wrapped)
    assert set(payload) == {"completion"}
    assert json.loads(payload["completion"]) == {
        "cmp_state": {"compact_targets": {"rile": 0.5}}
    }


def test_bare_dspy_payload_wrapper_normalizes_extra_bracket_chat_header() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "[[ ## prompt ## ]]\ninput\n\n"
                "[[ ## completion ## ]]\noutput\n\n"
                "[[ ## completed ## ]]"
            ),
        }
    ]
    inner = "{\"cmp_state\": {\"compact_targets\": {\"rile\": 0.5}}}"
    raw = f"[[ [[ ## completion ## ]]\n{inner} [[ ## completed ## ]]"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert wrapped.startswith("[[ ## completion ## ]]\n")
    assert wrapped.count("[[ ## completion ## ]]") == 1
    assert "[[ [[ ## completion ## ]]" not in wrapped
    assert inner in wrapped
    assert wrapped.rstrip().endswith("[[ ## completed ## ]]")


def test_bare_dspy_payload_wrapper_normalizes_double_hash_chat_header() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "[[ ## prompt ## ]]\ninput\n\n"
                "[[ ## completion ## ]]\noutput\n\n"
                "[[ ## completed ## ]]"
            ),
        }
    ]
    inner = "{\"cmp_state\": {\"compact_targets\": {\"rile\": 0.5}}}"
    raw = f"[[ ## ## completion ## ]]\n{inner}\n\n[[ ## completed ## ]]"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert wrapped.startswith("[[ ## completion ## ]]\n")
    assert wrapped.count("[[ ## completion ## ]]") == 1
    assert "[[ ## ## completion ## ]]" not in wrapped
    assert inner in wrapped
    assert wrapped.rstrip().endswith("[[ ## completed ## ]]")



def test_bare_dspy_payload_wrapper_normalizes_missing_prefix_hash_chat_header() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "[[ ## prompt ## ]]\ninput\n\n"
                "[[ ## completion ## ]]\noutput\n\n"
                "[[ ## completed ## ]]"
            ),
        }
    ]
    inner = "{\"cmp_state\": {\"compact_targets\": {\"rile\": 0.5}}}"
    raw = f"[[ completion ## ]]\n{inner}\n\n[[ ## completed ## ]]"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert wrapped.startswith("[[ ## completion ## ]]\n")
    assert wrapped.count("[[ ## completion ## ]]") == 1
    assert "[[ completion ## ]]" not in wrapped
    assert inner in wrapped
    assert wrapped.rstrip().endswith("[[ ## completed ## ]]")


def test_json_adapter_payload_wrapper_normalizes_broken_chat_header() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`completion`."
            ),
        }
    ]
    inner = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'
    raw = f"## completion ## ]]\n{inner}"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert json.loads(wrapped) == {"completion": inner}



def test_json_adapter_payload_wrapper_normalizes_missing_prefix_hash_chat_header() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Respond with a JSON object in the following order of fields: "
                "`completion`."
            ),
        }
    ]
    inner = "{\"cmp_state\": {\"compact_targets\": {\"rile\": 0.5}}}"
    raw = f"[[ completion ## ]]\n{inner}"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert json.loads(wrapped) == {"completion": inner}


def test_bare_dspy_payload_wrapper_normalizes_broken_chat_header() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "[[ ## prompt ## ]]\ninput\n\n"
                "[[ ## completion ## ]]\noutput\n\n"
                "[[ ## completed ## ]]"
            ),
        }
    ]
    inner = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'
    raw = f"## completion ## ]]\n{inner}"

    wrapped = _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages)

    assert wrapped.startswith("[[ ## completion ## ]]\n")
    assert wrapped.count("[[ ## completion ## ]]") == 1
    assert "\n## completion ## ]]" not in wrapped
    assert inner in wrapped
    assert wrapped.rstrip().endswith("[[ ## completed ## ]]")


def test_dspy_predict_accepts_repaired_bare_payload_with_chat_adapter() -> None:
    import dspy

    raw = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'
    bridge = FakeBridge(content=raw)
    lm = BatchedDSPyLM(
        model="openai/test-model",
        api_base="http://localhost:8000/v1",
        bridge=bridge,
    )

    class SingleOutputSignature(dspy.Signature):
        prompt: str = dspy.InputField()
        completion: str = dspy.OutputField()

    with dspy.context(lm=lm):
        result = dspy.Predict(SingleOutputSignature)(prompt="emit compact state")

    assert result.completion == raw


def test_dspy_predict_accepts_repaired_bare_payload_with_json_adapter() -> None:
    import dspy
    from dspy.adapters.json_adapter import JSONAdapter

    raw = '{"cmp_state": {"compact_targets": {"rile": 0.5}}}'
    bridge = FakeBridge(content=raw)
    lm = BatchedDSPyLM(
        model="openai/test-model",
        api_base="http://localhost:8000/v1",
        bridge=bridge,
    )

    class SingleOutputSignature(dspy.Signature):
        prompt: str = dspy.InputField()
        completion: str = dspy.OutputField()

    with dspy.context(lm=lm, adapter=JSONAdapter()):
        result = dspy.Predict(SingleOutputSignature)(prompt="emit compact state")

    assert result.completion == raw


def test_bare_dspy_payload_wrapper_has_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("TT_DSPY_WRAP_BARE_FIELD_OUTPUT", "0")
    messages = [
        {
            "role": "system",
            "content": "[[ ## prompt ## ]]\ninput\n[[ ## completion ## ]]\noutput",
        }
    ]
    raw = '{"cmp_state": {}}'

    assert _maybe_wrap_bare_dspy_field_response(content=raw, messages=messages) == raw


def test_batched_dspy_lm_routes_sync_calls_through_bridge() -> None:
    bridge = FakeBridge()
    lm = BatchedDSPyLM(
        model="openai/test-model",
        api_base="http://localhost:8000/v1",
        temperature=0.2,
        max_tokens=11,
        bridge=bridge,
    )

    output = lm(prompt="hello")

    assert output == ["batched response"]
    assert bridge.calls[-1]["messages"] == [{"role": "user", "content": "hello"}]
    assert bridge.calls[-1]["max_tokens"] == 11
    assert bridge.calls[-1]["temperature"] == 0.2


def test_batched_dspy_lm_copy_shares_bridge_and_overrides_generation_kwargs() -> None:
    bridge = FakeBridge()
    lm = BatchedDSPyLM(
        model="openai/test-model",
        api_base="http://localhost:8000/v1",
        temperature=0.2,
        max_tokens=11,
        bridge=bridge,
    )
    copied = lm.copy(temperature=0.9, max_tokens=7)

    output = copied(messages=[{"role": "user", "content": "copy call"}])

    assert output == ["batched response"]
    assert copied._batch_bridge is bridge
    assert bridge.calls[-1]["messages"] == [{"role": "user", "content": "copy call"}]
    assert bridge.calls[-1]["max_tokens"] == 7
    assert bridge.calls[-1]["temperature"] == 0.9


def test_batched_dspy_lm_async_forward_uses_bridge_submit() -> None:
    bridge = FakeBridge()
    lm = BatchedDSPyLM(
        model="openai/test-model",
        api_base="http://localhost:8000/v1",
        temperature=0.3,
        max_tokens=13,
        bridge=bridge,
    )

    response = asyncio.run(lm.aforward(prompt="async hello"))

    assert response.choices[0].message.content == "batched response"
    assert bridge.calls[-1]["messages"] == [{"role": "user", "content": "async hello"}]
    assert bridge.calls[-1]["max_tokens"] == 13
    assert bridge.calls[-1]["temperature"] == 0.3


def test_batched_dspy_lm_forwards_openai_request_options() -> None:
    bridge = FakeBridge()
    lm = BatchedDSPyLM(
        model="openai/test-model",
        api_base="http://localhost:8000/v1",
        temperature=0.3,
        max_tokens=13,
        bridge=bridge,
        extra_body={"seed": 7, "top_p": 0.9},
    )

    output = lm(prompt="options", stop=["</done>"])

    assert output == ["batched response"]
    assert bridge.calls[-1]["extra_request_params"] == {
        "seed": 7,
        "top_p": 0.9,
        "stop": ["</done>"],
    }


def test_vllm_factories_can_create_batched_dspy_lm_without_starting_server() -> None:
    lm = create_vllm_lm(
        port=8123,
        model="fake-model",
        batch_max_concurrent=17,
        batch_size=5,
        batch_timeout=0.03,
    )
    assert isinstance(lm, BatchedDSPyLM)
    assert lm._batch_max_concurrent == 17
    assert lm._batch_size == 5
    assert lm._batch_timeout == 0.03

    multi_lm = create_vllm_lm_multi(
        ports=[8123, 8124],
        model="fake-model",
        batch_max_concurrent=19,
        batch_size=7,
    )
    assert isinstance(multi_lm, BatchedDSPyLM)
    assert multi_lm._batch_api_bases == [
        "http://localhost:8123/v1",
        "http://localhost:8124/v1",
    ]
    assert multi_lm._batch_max_concurrent == 19
    assert multi_lm._batch_size == 7


def test_local_engine_factory_supports_sglang_batch_transport_without_starting_server() -> None:
    lm = create_local_engine_lm(
        engine="sglang",
        port=30000,
        model="fake-model",
        max_tokens=23,
        batch_max_concurrent=29,
        batch_size=11,
        batch_timeout=0.05,
    )

    assert isinstance(lm, BatchedDSPyLM)
    assert lm._batch_api_bases == ["http://localhost:30000/v1"]
    assert lm._batch_max_concurrent == 29
    assert lm._batch_size == 11
    assert lm._batch_timeout == 0.05


def test_local_engine_factory_accepts_resolved_endpoint_contract() -> None:
    endpoints = LocalChatEndpoints(
        engine=EngineType.VLLM,
        ports=(8123, 8124),
        base_urls=("http://localhost:8123/v1", "http://localhost:8124/v1"),
    )

    lm = create_local_engine_lm(
        endpoints=endpoints,
        model="fake-model",
        batch_size=9,
    )

    assert isinstance(lm, BatchedDSPyLM)
    assert lm._batch_api_bases == [
        "http://localhost:8123/v1",
        "http://localhost:8124/v1",
    ]
    assert lm._batch_size == 9


def test_local_engine_manager_factory_uses_batch_transport_without_starting_server() -> None:
    lm, manager = create_local_engine_lm_with_manager(
        engine="sglang",
        port=30000,
        model="fake-model",
        task="summarizer",
        batch_size=13,
    )

    assert isinstance(lm, BatchedDSPyLM)
    assert lm._batch_api_bases == ["http://localhost:30000/v1"]
    assert lm._batch_size == 13
    assert lm.kwargs["max_tokens"] == manager.max_output_tokens
