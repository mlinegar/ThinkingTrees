from __future__ import annotations

from src.tasks.manifesto.pipeline import ManifestoPipeline


class _RecordingG:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, content: str, rubric: str) -> str:
        self.calls.append((content, rubric))
        return f"summary-{len(self.calls)}"


class _Scorer:
    def __call__(self, *, summary: str, task_context: str) -> dict:
        return {"score": 0.5, "reasoning": f"scored {len(summary)} chars"}


def test_manifesto_pipeline_uses_same_unified_g_for_leaves_and_merges():
    pipeline = ManifestoPipeline(chunk_size=120)
    recorder = _RecordingG()
    pipeline.g = recorder
    pipeline.scorer = _Scorer()

    text = (
        "The party supports public investment, regional policy, climate action, "
        "and social services while discussing taxation and migration. "
    ) * 10

    result = pipeline(text=text, rubric="preserve policy positions")

    assert result.score == 0.5
    raw_calls = [content for content, _ in recorder.calls if not content.startswith("PART 1:")]
    merge_calls = [content for content, _ in recorder.calls if content.startswith("PART 1:")]
    assert raw_calls, "expected unified g to be called on raw chunks"
    assert merge_calls, "expected unified g to be called on formatted merge inputs"
    assert all("PART 2:" in content for content in merge_calls)
