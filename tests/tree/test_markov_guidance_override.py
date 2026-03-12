"""Unit tests for Markov OPS-count inference-time guidance override modes."""

import torch

from src.tree.markov_changepoint_ops_count_simulation import (
    AdditiveCountSketch,
    LearnedCountSketch,
    _override_state_with_oracle_count,
)


def _orth_component(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(w * w)
    if float(denom.detach().cpu()) <= 1e-20:
        return x
    proj = (torch.sum(x * w) / denom) * w
    return x - proj


def test_guidance_override_reset_sets_h_parallel_to_readout():
    model = LearnedCountSketch(
        feature_dim=2 * 2 + 3,  # endpoints + dummy core
        state_dim=4,
        hidden_dim=8,
        target_scale=10.0,
        n_regimes=2,
        use_endpoints=True,
    )
    with torch.no_grad():
        model.readout.weight.copy_(torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=model.readout.weight.dtype))
        model.readout.bias.copy_(torch.tensor([0.1], dtype=model.readout.bias.dtype))

    h0 = torch.tensor([0.3, -0.4, 0.2, 0.1], dtype=model.readout.weight.dtype)
    first = torch.tensor([1.0, 0.0], dtype=model.readout.weight.dtype)
    last = torch.tensor([0.0, 1.0], dtype=model.readout.weight.dtype)
    state = torch.cat([h0, first, last], dim=0)

    out = _override_state_with_oracle_count(model, state, target_count=3.0, override_mode="reset")
    h1, first1, last1 = model._split_state(out)

    w = model.readout.weight.squeeze(0)
    orth = _orth_component(h1, w)
    assert float(torch.linalg.norm(orth).detach().cpu()) <= 1e-6

    pred = float(model.predict_norm_from_state(out).detach().cpu())
    assert abs(pred - 0.3) <= 1e-6
    assert torch.allclose(first1, first)
    assert torch.allclose(last1, last)


def test_guidance_override_adjust_preserves_orthogonal_component():
    model = LearnedCountSketch(
        feature_dim=2 * 2 + 3,
        state_dim=4,
        hidden_dim=8,
        target_scale=10.0,
        n_regimes=2,
        use_endpoints=True,
    )
    with torch.no_grad():
        model.readout.weight.copy_(torch.tensor([[0.2, 1.5, -0.7, 0.9]], dtype=model.readout.weight.dtype))
        model.readout.bias.copy_(torch.tensor([-0.3], dtype=model.readout.bias.dtype))

    h0 = torch.tensor([0.8, -0.1, 0.4, 0.2], dtype=model.readout.weight.dtype)
    first = torch.tensor([0.0, 1.0], dtype=model.readout.weight.dtype)
    last = torch.tensor([1.0, 0.0], dtype=model.readout.weight.dtype)
    state = torch.cat([h0, first, last], dim=0)

    w = model.readout.weight.squeeze(0)
    orth0 = _orth_component(h0, w)

    out = _override_state_with_oracle_count(model, state, target_count=7.0, override_mode="adjust")
    h1, _first1, _last1 = model._split_state(out)
    orth1 = _orth_component(h1, w)

    assert torch.allclose(orth1, orth0, atol=1e-6, rtol=0.0)
    pred = float(model.predict_norm_from_state(out).detach().cpu())
    assert abs(pred - 0.7) <= 1e-6


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

