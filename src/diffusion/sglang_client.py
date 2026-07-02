"""Compatibility alias for the SGLang `/generate` diffusion backend."""

from __future__ import annotations

from src.diffusion.backends import (
    DiffusionBatchResponse,
    DiffusionGeneration,
    SGLangDiffusionBackend,
)


SGLangDiffusionClient = SGLangDiffusionBackend


__all__ = [
    "DiffusionBatchResponse",
    "DiffusionGeneration",
    "SGLangDiffusionBackend",
    "SGLangDiffusionClient",
]
