"""Compatibility shim for canonical local-law training arithmetic.

The scalar training-row implementation lives in ``treepo.training.local_law``.
ThinkingTrees keeps these public names for older callers and diagnostics, but
new objective arithmetic should be added upstream.
"""

from __future__ import annotations

from treepo.training.local_law import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    MIN_PROPENSITY,
    LocalLawTrainingAggregate,
    LocalLawTrainingRow,
    aggregate_local_law_training_rows,
    corrected_local_law_loss,
    depth_discount,
    local_law_training_objective_mean,
    normalize_local_law_objective_mode,
)

VALID_LOCAL_LAW_OBJECTIVE_MODES: tuple[str, ...] = (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
)

LocalLawObservation = LocalLawTrainingRow
LocalLawAggregate = LocalLawTrainingAggregate
local_law_objective_mean = local_law_training_objective_mean
aggregate_local_law_observations = aggregate_local_law_training_rows

__all__ = [
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "MIN_PROPENSITY",
    "LocalLawAggregate",
    "LocalLawObservation",
    "VALID_LOCAL_LAW_OBJECTIVE_MODES",
    "aggregate_local_law_observations",
    "corrected_local_law_loss",
    "depth_discount",
    "local_law_objective_mean",
    "normalize_local_law_objective_mode",
]
