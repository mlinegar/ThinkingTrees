"""Judge-facing wrapper over the internal oracle pairwise judge."""

from src.training.preference.oracle_judge import (
    OracleJudgeResult,
    OraclePairwiseJudge,
)

__all__ = [
    "OracleJudgeResult",
    "OraclePairwiseJudge",
]
