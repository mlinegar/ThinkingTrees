"""Compatibility exports for the vLLM-Omni diffusion backend."""

from src.diffusion.backends import (
    DiffusionBatchResponse,
    DiffusionGeneration,
    VLLMOmniDiffusionBackend,
)


__all__ = [
    "DiffusionBatchResponse",
    "DiffusionGeneration",
    "VLLMOmniDiffusionBackend",
]
