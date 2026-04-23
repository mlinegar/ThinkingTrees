"""Supervision-facing wrappers for optimizer reward backends."""

from src.training.preference.genrm_reward import (
    create_genrm_dspy_metric,
    create_genrm_reward_func,
)
from src.training.preference.oracle_reward import (
    create_oracle_alignment_reward_func,
)

__all__ = [
    "create_genrm_dspy_metric",
    "create_genrm_reward_func",
    "create_oracle_alignment_reward_func",
]
