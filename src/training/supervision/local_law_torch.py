"""Compatibility shim for canonical torch local-law objectives."""

from __future__ import annotations

from treepo.training.local_law import (
    corrected_local_law_loss_tensor,
    corrected_local_law_target_mse,
    local_law_objective_from_losses,
    local_law_objective_target_mse,
)

__all__ = [
    "corrected_local_law_loss_tensor",
    "corrected_local_law_target_mse",
    "local_law_objective_from_losses",
    "local_law_objective_target_mse",
]
