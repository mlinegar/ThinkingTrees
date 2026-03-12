"""
Preference learning system for OPS training.

This module provides infrastructure for collecting and using pairwise preferences
to train summarization models and judges.
"""

# Types and protocols
from src.training.preference.types import (
    PreferenceDerivationResult,
    PreferenceDeriver,
    PreferencePair,
    GenerationConfig,
    PreferenceDataset,
    compute_propensity_diagnostics,
    get_deriver,
    list_derivers,
    register_deriver,
    JudgeDeriver,
    GenRMDeriver,
    OracleDeriver,
)

# Base collector
from src.training.preference.base import (
    BasePreferenceCollector,
    CandidateInfo,
    PreferenceResult,
    CollectionStatistics,
)

# Collectors
from src.training.preference.collector import (
    PreferenceCollector,
    PairwiseJudge,
)

# Engine
from src.training.preference.engine import (
    PreferenceEngine,
    PreferenceEngineConfig,
    PreferenceDerivationStrategy,
)

# GenRM
from src.training.preference.genrm import (
    GenRMJudge,
    GenRMResult,
    is_genrm_error,
)

# GenRM Batching
from src.training.preference.genrm_batch import (
    AsyncBatchGenRMClient,
    GenRMComparisonRequest,
    GenRMBatchStats,
    create_genrm_batch_client,
)

# Large-model DSPy judge (GenRM-free tournament path)
from src.training.preference.large_judge_dspy import (
    LargeJudgeComparisonModule,
)
from src.training.preference.oracle_judge import (
    OracleJudgeResult,
    OraclePairwiseJudge,
)

__all__ = [
    # Types
    "PreferenceDerivationResult",
    "PreferenceDeriver",
    "PreferencePair",
    "GenerationConfig",
    "PreferenceDataset",
    "compute_propensity_diagnostics",
    "get_deriver",
    "list_derivers",
    "register_deriver",
    "JudgeDeriver",
    "GenRMDeriver",
    "OracleDeriver",
    # Base
    "BasePreferenceCollector",
    "CandidateInfo",
    "PreferenceResult",
    "CollectionStatistics",
    # Collector
    "PreferenceCollector",
    "PairwiseJudge",
    # Engine
    "PreferenceEngine",
    "PreferenceEngineConfig",
    "PreferenceDerivationStrategy",
    # GenRM
    "GenRMJudge",
    "GenRMResult",
    "is_genrm_error",
    # Human (interactive) judge
    "HumanGenRMJudge",
    # GenRM Batching
    "AsyncBatchGenRMClient",
    "GenRMComparisonRequest",
    "GenRMBatchStats",
    "create_genrm_batch_client",
    # Large-model DSPy judge
    "LargeJudgeComparisonModule",
    # Oracle scorer judge
    "OracleJudgeResult",
    "OraclePairwiseJudge",
]


def __getattr__(name: str):
    # Lazy import to avoid importing the submodule during package import.
    # This prevents runpy warnings when running `python -m src.training.preference.human_judge`.
    if name == "HumanGenRMJudge":
        from src.training.preference.human_judge import HumanGenRMJudge

        return HumanGenRMJudge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
