from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.core.documents import DocumentSample
from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig
from src.tasks.prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt


class _DummyStrategy:
    async def summarize(self, content: str, rubric: str, temperature: float = 0.7) -> str:
        return "summary"

    async def merge(self, left: str, right: str, rubric: str, temperature: float = 0.7) -> str:
        return "merged"


class _FakeOrchestrator:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.config = config

    async def process_documents(
        self,
        documents,
        rubric,
        get_text_fn,
        get_id_fn,
        progress_callback=None,
    ):
        results = []
        for sample in documents:
            doc_id = get_id_fn(sample)
            leaf = SimpleNamespace(
                summary="leaf summary",
                raw_text_span=get_text_fn(sample),
            )
            tree = SimpleNamespace(
                metadata={"doc_id": doc_id},
                final_summary="final summary",
                leaves=[leaf],
                root=SimpleNamespace(content="final summary", metadata={"doc_id": doc_id}),
                height=1,
                leaf_count=1,
            )
            results.append(
                SimpleNamespace(
                    tree=tree,
                    errors=[],
                    content_weights=None,
                )
            )
        return results


class _FakeAsyncBatchLLMClient:
    def __init__(
        self,
        base_url,
        max_concurrent,
        batch_size,
        batch_timeout,
        request_timeout=None,
        recover_base_url_callback=None,
        recovery_cooldown_seconds=0.0,
        **_kwargs,
    ):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.request_timeout = request_timeout
        self._responses = {}
        self.submitted = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def submit(self, request):
        self.submitted.append(request)
        content = "12.5" if getattr(request, "request_type", "") == "score" else "0.0"
        self._responses[getattr(request, "request_id")] = SimpleNamespace(
            content=content,
            error=None,
        )

    async def await_response(self, request_id, timeout=None, **_kwargs):
        del timeout
        return self._responses[request_id]


def _score_prompt(summary: str, task_context: str):
    return [
        {"role": "system", "content": "Return one numeric score."},
        {"role": "user", "content": summary},
    ]


def test_process_batch_with_external_strategy_still_computes_llm_backend_score(monkeypatch) -> None:
    created_clients = []

    class _RecordingFakeClient(_FakeAsyncBatchLLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_clients.append(self)

    monkeypatch.setattr("src.pipelines.batched.BatchTreeOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr("src.pipelines.batched.AsyncBatchLLMClient", _RecordingFakeClient)

    config = BatchedPipelineConfig(
        task_model_url="http://localhost:8000/v1",
        representation_backends=["llm"],
        primary_representation_backend="llm",
        fallback_to_available_backend=True,
        run_baseline=False,
        show_progress=False,
        prompt_builders=PromptBuilders(
            summarize=default_summarize_prompt,
            merge=default_merge_prompt,
            score=_score_prompt,
            audit=None,
        ),
        score_parser=lambda raw: float(str(raw).strip()),
    )
    pipeline = BatchedDocPipeline(config=config)

    sample = DocumentSample(doc_id="doc_1", text="This is test content.")
    results = asyncio.run(
        pipeline.process_batch_with_strategy([sample], strategy=_DummyStrategy())
    )

    assert len(created_clients) == 1
    assert len(created_clients[0].submitted) >= 1

    assert len(results) == 1
    result = results[0]
    assert result.estimated_score == 12.5
    assert result.metadata["representation_selected_backend"] == "llm"
    assert result.metadata["representation_backend_scores"]["llm"] == 12.5
