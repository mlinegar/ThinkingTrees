from __future__ import annotations

from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig
from src.tasks.prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt


def _make_pipeline(**kwargs) -> BatchedDocPipeline:
    config = BatchedPipelineConfig(
        prompt_builders=PromptBuilders(
            summarize=default_summarize_prompt,
            merge=default_merge_prompt,
            score=None,
            audit=None,
        ),
        score_parser=lambda _: 0.0,
        **kwargs,
    )
    return BatchedDocPipeline(config=config)


def test_embedding_score_falls_back_to_proxy_metadata() -> None:
    pipeline = _make_pipeline()
    score, support, source = pipeline._score_from_embedding_sources(
        payload=None,
        metadata={"embedding_proxy_score": 12.5},
    )
    assert support == 0
    assert source == "metadata:embedding_proxy_score"
    assert score == 12.5


def test_hybrid_oracle_seeded_ensemble_clamps_llm_and_boosts_operators() -> None:
    pipeline = _make_pipeline(
        hybrid_oracle_seeded_ensemble=True,
        hybrid_seed_llm_min_weight=0.2,
        hybrid_seed_llm_max_weight=0.5,
        hybrid_operator_boost=1.5,
        representation_weights={"llm": 1.0, "ctreepo": 1.0, "mergeable_sketch": 1.0},
    )
    weights, diagnostics = pipeline._resolve_ensemble_weights(
        {"llm": 20.0, "ctreepo": 25.0, "mergeable_sketch": 24.0},
        metadata={"ctreepo_confidence": 0.9, "mergeable_sketch_window_count": 8},
    )
    assert diagnostics["mode"] == "hybrid_oracle_seeded"
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert 0.2 <= float(weights["llm"]) <= 0.5
    assert float(weights["ctreepo"]) > float(weights["llm"])


def test_static_ensemble_weights_when_hybrid_disabled() -> None:
    pipeline = _make_pipeline(
        hybrid_oracle_seeded_ensemble=False,
        representation_weights={"llm": 1.0, "ctreepo": 1.0},
    )
    weights, diagnostics = pipeline._resolve_ensemble_weights(
        {"llm": 15.0, "ctreepo": 18.0},
        metadata={},
    )
    assert diagnostics["mode"] == "static"
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert abs(float(weights["llm"]) - 0.5) < 1e-6
    assert abs(float(weights["ctreepo"]) - 0.5) < 1e-6
