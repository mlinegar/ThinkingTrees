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

__all__ = [
    # Types
    "PreferenceDerivationResult",
    "PreferenceDeriver",
    "PreferencePair",
    "GenerationConfig",
    "PreferenceDataset",
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
]
