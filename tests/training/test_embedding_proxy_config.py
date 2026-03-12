"""Tests for adapter-aware embedding proxy configuration resolution."""

from argparse import Namespace

from src.preprocessing.chunker import AdaptiveChunkingConfig
from src.training.run_pipeline import (
    EmbeddingProxyConfig,
    resolve_embedding_model_for_adapter,
    resolve_embedding_proxy_config,
)


def _embedding_args(**overrides):
    """Build a Namespace with embedding-related CLI args."""
    base = {
        "adaptive_embedding_proxy": None,
        "adaptive_embedding_api_base": None,
        "adaptive_embedding_model": None,
        "adaptive_embedding_models_by_adapter": None,
        "adaptive_embedding_batch_size": None,
        "adaptive_embedding_timeout_sec": None,
        "adaptive_embedding_min_samples": None,
        "adaptive_embedding_ridge_lambda": None,
        "adaptive_embedding_head_method": None,
        "adaptive_embedding_head_epochs": None,
        "adaptive_embedding_head_lr": None,
        "adaptive_embedding_head_weight_decay": None,
        "adaptive_embedding_full_finetune": None,
        "adaptive_embedding_finetune_command": None,
        "adaptive_embedding_max_text_chars": None,
        "adaptive_embedding_retrain_rounds": None,
        "adaptive_embedding_include_val": None,
        "adaptive_embedding_truth_sources": None,
        "adaptive_embedding_score_key": None,
        "port": 8000,
    }
    base.update(overrides)
    return Namespace(**base)


def test_resolve_embedding_model_for_adapter_prefers_adapter_specific_model():
    cfg = EmbeddingProxyConfig(
        model="global-model",
        model_by_adapter={
            "text_char": "text-model",
            "time_segment": "video-model",
        },
    )
    assert (
        resolve_embedding_model_for_adapter(cfg, adapter_name="time_segment", fallback="fallback-model")
        == "video-model"
    )
    assert (
        resolve_embedding_model_for_adapter(cfg, adapter_name="video_time", fallback="fallback-model")
        == "video-model"
    )


def test_resolve_embedding_model_for_adapter_falls_back_global_then_fallback():
    cfg = EmbeddingProxyConfig(model="global-model", model_by_adapter={})
    assert (
        resolve_embedding_model_for_adapter(cfg, adapter_name="text_page", fallback="fallback-model")
        == "global-model"
    )

    cfg_no_global = EmbeddingProxyConfig(model=None, model_by_adapter={})
    assert (
        resolve_embedding_model_for_adapter(cfg_no_global, adapter_name="text_page", fallback="fallback-model")
        == "fallback-model"
    )


def test_resolve_embedding_proxy_config_parses_models_by_adapter_json():
    args = _embedding_args(
        adaptive_embedding_models_by_adapter=(
            '{"text_char":"text-model","time_segment":"video-model"}'
        )
    )
    settings = {
        "chunking": {"adaptive": {"embedding_proxy": {}}},
        "servers": {"embedding_url": "http://localhost:8003/v1"},
    }
    adaptive_cfg = AdaptiveChunkingConfig(
        enabled=True,
        window_adapter="time_segment",
        proxy_model="fallback-model",
    )

    cfg = resolve_embedding_proxy_config(args, settings=settings, adaptive_cfg=adaptive_cfg)
    assert cfg.model_by_adapter == {
        "text_char": "text-model",
        "time_segment": "video-model",
    }
    assert (
        resolve_embedding_model_for_adapter(
            cfg,
            adapter_name=adaptive_cfg.window_adapter,
            fallback=adaptive_cfg.proxy_model,
        )
        == "video-model"
    )
