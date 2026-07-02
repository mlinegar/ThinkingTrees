from __future__ import annotations

import pytest
import torch

from src.training.supervision.local_law_torch import (
    corrected_local_law_loss_tensor,
    local_law_objective_from_losses,
)


def test_corrected_local_law_loss_tensor_rejects_zero_observed_propensity() -> None:
    with pytest.raises(ValueError, match="finite propensity in \\(0, 1\\]"):
        corrected_local_law_loss_tensor(
            proxy_loss=torch.tensor([0.4]),
            oracle_loss=torch.tensor([0.1]),
            observed=torch.tensor([True]),
            propensity=torch.tensor([0.0]),
        )


def test_corrected_local_law_loss_tensor_allows_unobserved_zero_propensity() -> None:
    corrected = corrected_local_law_loss_tensor(
        proxy_loss=torch.tensor([0.4]),
        oracle_loss=torch.tensor([0.1]),
        observed=torch.tensor([False]),
        propensity=torch.tensor([0.0]),
    )

    assert torch.allclose(corrected, torch.tensor([0.4]))


def test_local_law_objective_from_losses_rejects_out_of_range_observed_propensity() -> None:
    with pytest.raises(ValueError, match="finite propensity in \\(0, 1\\]"):
        local_law_objective_from_losses(
            proxy_loss=torch.tensor([0.4]),
            oracle_loss=torch.tensor([0.1]),
            observed=torch.tensor([True]),
            propensity=torch.tensor([1.5]),
            depths=torch.tensor([0]),
        )
