"""Public judge backends and capability helpers for supervision collection."""

from src.training.judges.base import (
    JudgeResult,
    JudgeConfig,
    JudgeError,
    BaseJudge,
    AsyncJudge,
    CompilableJudge,
)

from src.training.judges.dspy import DSPyJudge
from src.training.judges.genrm import GenRMJudge, GenRMJudgeWrapper
from src.training.judges.oracle import OracleJudge
from src.training.judges.large_dspy import (
    LargeJudgeComparisonModule,
    LargeJudgeListwiseModule,
)
from src.training.judges.oracle_pairwise import OracleJudgeResult, OraclePairwiseJudge
from src.training.supervision.judge_capabilities import (
    ComparativeJudgeResult,
    PairwiseJudgeResult,
    invoke_comparative_judgment_async,
    invoke_comparative_judgment_sync,
    invoke_pairwise_judgment_async,
    invoke_pairwise_judgment_sync,
    judge_backend_name,
    supports_direct_comparative_judging,
    supports_pairwise_judging,
)

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
    "GenRMJudge",
    "GenRMJudgeWrapper",
    "LargeJudgeComparisonModule",
    "LargeJudgeListwiseModule",
    "OracleJudge",
    "OracleJudgeResult",
    "OraclePairwiseJudge",
    # Capability helpers
    "ComparativeJudgeResult",
    "PairwiseJudgeResult",
    "invoke_comparative_judgment_async",
    "invoke_comparative_judgment_sync",
    "invoke_pairwise_judgment_async",
    "invoke_pairwise_judgment_sync",
    "judge_backend_name",
    "supports_direct_comparative_judging",
    "supports_pairwise_judging",
]
