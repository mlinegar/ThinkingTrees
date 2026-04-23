"""Internal compatibility wrapper over ``src.training.supervision.base``."""

from src.training.supervision.base import (
    BasePreferenceCollector,
    CandidateInfo,
    CollectionStatistics,
    PreferenceResult,
)

__all__ = [
    "BasePreferenceCollector",
    "CandidateInfo",
    "CollectionStatistics",
    "PreferenceResult",
]
