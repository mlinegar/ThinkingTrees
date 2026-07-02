"""Standalone diffusion research prototype surfaces."""

from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings

from src.diffusion.markov_toy import (
    MarkovToySketch,
    changepoint_count,
    chunk_states,
    count_only_feature,
    encode_markov_path,
    merge_markov_sketch,
    run_markov_toy_experiment,
)


__all__ = [
    "MarkovToySketch",
    "changepoint_count",
    "chunk_states",
    "count_only_feature",
    "encode_markov_path",
    "format_diffusion_chat_prompt",
    "merge_markov_sketch",
    "run_markov_toy_experiment",
]


_ARCHIVED_BACKEND_EXPORTS = {
    "DiffusionBackend",
    "DiffusionBatchResponse",
    "DiffusionGeneration",
    "HTTPGenerateDiffusionBackend",
    "SGLangDiffusionBackend",
    "SGLangDiffusionClient",
    "VLLMOmniDiffusionBackend",
    "build_diffusion_backend",
}

_ARCHIVED_TREE_EXPORTS = {
    "DiffusionOperationTrace",
    "DiffusionPromptTemplates",
    "DiffusionRunResult",
    "DiffusionTreeEngine",
    "FixedBinaryDiffusionTreeEngine",
    "format_diffusion_chat_prompt",
}


def __getattr__(name: str) -> Any:
    if name in _ARCHIVED_BACKEND_EXPORTS:
        warnings.warn(
            f"src.diffusion.{name} is archived compatibility API; use "
            "src.core.inference_engine.build_inference_engine(..., "
            "surface=EngineSurface.CHAT_OPENAI, transport='generate') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module("src.diffusion.backends")
        if name == "SGLangDiffusionClient":
            shim = import_module("src.diffusion.sglang_client")
            return getattr(shim, name)
        return getattr(module, name)
    if name in _ARCHIVED_TREE_EXPORTS:
        warnings.warn(
            f"src.diffusion.{name} is archived compatibility API; use "
            "src.tree.async_operator.AsyncFromInferenceEngine with the canonical "
            "text surface instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module("src.diffusion.tree_engine")
        return getattr(module, name)
    raise AttributeError(name)
