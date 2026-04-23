"""Unit tests for Markov OPS-count inference-time guidance override modes."""

import torch

from src.tree.markov_changepoint_ops_count_simulation import (
    AdditiveCountSketch,
    _override_state_with_oracle_count,
)


def test_additive_guidance_override_modes_are_equivalent():
    model = AdditiveCountSketch(
        feature_dim=2 * 2 + 3,
        hidden_dim=8,
        target_scale=10.0,
        n_regimes=2,
        use_endpoints=True,
    )
    count0 = torch.tensor(0.12, dtype=torch.float32)
    first = torch.tensor([1.0, 0.0], dtype=torch.float32)
    last = torch.tensor([0.0, 1.0], dtype=torch.float32)
    state = torch.cat([count0.unsqueeze(0), first, last], dim=0)

    out_reset = _override_state_with_oracle_count(model, state, target_count=4.0, override_mode="reset")
    out_adjust = _override_state_with_oracle_count(model, state, target_count=4.0, override_mode="adjust")

    assert torch.allclose(out_reset, out_adjust)
    count1, first1, last1 = model._split_state(out_reset)
    assert abs(float(count1.detach().cpu()) - 0.4) <= 1e-6
    assert torch.allclose(first1, first)
    assert torch.allclose(last1, last)
