from __future__ import annotations

from src.core.engines import EngineSurface
from src.core.inference_engine import NativeOperatorRegistry
from src.runtime.calls import RuntimeCallScheduler
from src.runtime.contracts import (
    ChatInput,
    OperatorInput,
    OperatorOutput,
    RuntimeSurfaceCall,
    STATE_OPERATOR_MERGE_STATE,
)
from src.runtime.inference_context import RuntimeInferenceContext


def test_runtime_call_scheduler_executes_chat_and_emits_compact_record() -> None:
    records: list[dict] = []
    ctx = RuntimeInferenceContext(
        {
            "surfaces": {
                "chat_openai": {
                    "engine": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "model": "mock-chat",
                }
            }
        },
        mock=True,
        call_sink=records.append,
    )
    ctx.set_call_scope(
        run_id="r",
        unit_id="u000001",
        method_id="full_context",
        runner_id="llm_direct_official",
        problem_id="p",
    )

    result = ctx.scheduler.schedule(
        RuntimeSurfaceCall(
            surface=EngineSurface.CHAT_OPENAI,
            input=ChatInput(messages=[{"role": "user", "content": "hello"}], max_tokens=4),
            request_kind="answer",
        )
    )

    assert result.response.model_id == "mock"
    assert len(records) == 1
    assert records[0]["experiment_id"] == "r"
    assert "run_id" not in records[0]
    assert records[0]["method_id"] == "full_context"
    assert records[0]["runner_id"] == "llm_direct_official"
    assert records[0]["surface"] == "chat_openai"
    assert records[0]["role"] == ""
    assert records[0]["input_summary"]["kind"] == "chat"
    assert "hello" not in str(records[0])


def test_runtime_call_scheduler_runs_many_chat_calls_in_order() -> None:
    ctx = RuntimeInferenceContext(
        {
            "surfaces": {
                "chat_openai": {
                    "engine": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "model": "mock-chat",
                }
            }
        },
        mock=True,
    )
    calls = [
        RuntimeSurfaceCall(
            surface=EngineSurface.CHAT_OPENAI,
            input=ChatInput(messages=[{"role": "user", "content": f"hello {idx}"}]),
        )
        for idx in range(3)
    ]

    results = ctx.scheduler.schedule_many(calls)

    assert [result.call.input.messages[0]["content"] for result in results] == [
        "hello 0",
        "hello 1",
        "hello 2",
    ]
    assert all(result.response.model_id == "mock" for result in results)


def test_runtime_call_scheduler_coalesces_embedding_texts_and_preserves_order() -> None:
    records: list[dict] = []
    ctx = RuntimeInferenceContext(
        {
            "surfaces": {
                "embedding": {
                    "engine": "vllm",
                    "base_url": "http://localhost:8003/v1",
                    "model": "mock-embed",
                    "mock": True,
                }
            }
        },
        mock=True,
        call_sink=records.append,
    )

    vectors = ctx.scheduler.embed_texts(["alpha", "beta", "alpha"])

    assert len(vectors) == 3
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]
    assert len(records) == 1
    assert records[0]["surface"] == "embedding"
    assert records[0]["role"] == "embedder"
    assert records[0]["input_summary"]["text_count"] == 3


def test_runtime_call_scheduler_dispatches_batched_native_operator() -> None:
    NativeOperatorRegistry.clear()
    records: list[dict] = []

    def _merge(payload: OperatorInput) -> OperatorOutput:
        return OperatorOutput(
            data={"batch_size": len(payload.batch), "operation": payload.operation},
            artifacts={"handler": "fixture"},
        )

    NativeOperatorRegistry.register(STATE_OPERATOR_MERGE_STATE, _merge)
    try:
        ctx = RuntimeInferenceContext(
            {
                "surfaces": {
                    "operator": {
                        "engine": "native_operator",
                        "model": "state-fixture",
                    }
                }
            },
            call_sink=records.append,
        )
        scheduler = RuntimeCallScheduler(ctx, sink=records.append)

        result = scheduler.batch_operator(
            STATE_OPERATOR_MERGE_STATE,
            [{"left": "a", "right": "b"}, {"left": "c", "right": "d"}],
        )
    finally:
        NativeOperatorRegistry.clear()

    assert result.response.model_id == "state-fixture"
    assert isinstance(result.response.output, OperatorOutput)
    assert result.response.output.data["batch_size"] == 2
    assert records[0]["surface"] == "operator"
    assert records[0]["role"] == "state_model"
    assert records[0]["request_kind"] == f"state_model:{STATE_OPERATOR_MERGE_STATE}"
    assert records[0]["input_summary"]["operation"] == STATE_OPERATOR_MERGE_STATE
    assert records[0]["input_summary"]["batch_count"] == 2
