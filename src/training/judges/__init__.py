"""
Judge implementations for pairwise comparison.

Provides different judge implementations for comparing summaries:
- DSPyJudge: LLM-based judge using DSPy
- GenRMJudge: NVIDIA GenRM model wrapper
- OracleJudge: Oracle scoring function wrapper
"""

from src.training.judges.base import (
    JudgeResult,
    JudgeConfig,
    JudgeError,
    BaseJudge,
    AsyncJudge,
    CompilableJudge,
)

from src.training.judges.dspy import DSPyJudge
from src.training.judges.genrm import GenRMJudgeWrapper
from src.training.judges.oracle import OracleJudge

__all__ = [
    # Base types
    "JudgeResult",
    "JudgeConfig",
    "JudgeError",
    "BaseJudge",
    "AsyncJudge",
    "CompilableJudge",
    # Implementations
    "DSPyJudge",
    "GenRMJudgeWrapper",
    "OracleJudge",
]
