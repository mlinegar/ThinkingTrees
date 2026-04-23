"""Internal compatibility wrapper over ``src.training.supervision.collector``."""

from src.training.supervision.collector import (
    PairwiseJudge,
    PreferenceCollector,
)
from src.training.supervision.comparative_types import (
    GenerationConfig,
    PreferenceDataset,
)

__all__ = [
    "GenerationConfig",
    "PairwiseJudge",
    "PreferenceDataset",
    "PreferenceCollector",
]
