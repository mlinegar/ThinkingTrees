"""The FNO neural-operator merge CAN learn simple binary ops.

Guards the conclusion that the manifesto g failure is an INPUT BOTTLENECK (mass
not fed to the merge), not a capacity gap: the operator learns average/max
trivially, learns mass-weighted average WHEN mass is in the state, and provably
cannot when mass is hidden from it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralop")

from src.ctreepo.embedding_fno import EmbeddingCoordinateFNOTreeRegressor


def _model(*, dim: int, mode: str):
    return EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=dim,
        hidden_channels=16,
        n_modes=min(dim, 8),
        n_layers=2,
        head_hidden_dim=16,
        target_min=0.0,
        target_max=1.0,
        merge_mode=mode,
        merge_gate_hidden_dim=16,
    )


def _fit_merge(model, make_batch, *, steps=600, lr=3e-3):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        l, r, tgt = make_batch(256)
        loss = torch.nn.functional.l1_loss(model.merge(l, r), tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        l, r, tgt = make_batch(2048)
        return torch.nn.functional.l1_loss(model.merge(l, r), tgt).item()


def test_fno_merge_learns_average():
    dim = 8

    def batch(B):
        l, r = torch.rand(B, 1, dim), torch.rand(B, 1, dim)
        return l, r, 0.5 * (l + r)

    mae = _fit_merge(_model(dim=dim, mode="gated"), batch)
    assert mae < 0.01, f"gated FNO merge should learn average; MAE={mae}"


def test_fno_merge_learns_max():
    dim = 8

    def batch(B):
        l, r = torch.rand(B, 1, dim), torch.rand(B, 1, dim)
        return l, r, torch.maximum(l, r)

    mae = _fit_merge(_model(dim=dim, mode="maxpool"), batch)
    assert mae < 0.02, f"maxpool FNO merge should learn max; MAE={mae}"


def test_fno_merge_mass_weighted_needs_mass_in_state():
    """The crux: mass-weighted avg is learnable WHEN mass is a state dim, and
    NOT when mass is hidden from the merge (the manifesto situation)."""
    dim = 8

    def batch_mass_in(B):
        # mass = state dim 0; weighted avg over all dims by those masses
        l, r = torch.rand(B, 1, dim), torch.rand(B, 1, dim)
        ml = l[:, :, 0:1].clamp_min(1e-3)
        mr = r[:, :, 0:1].clamp_min(1e-3)
        w = ml / (ml + mr)
        return l, r, w * l + (1.0 - w) * r

    def batch_mass_out(B):
        # mass is a SEPARATE scalar the merge never sees
        l, r = torch.rand(B, 1, dim), torch.rand(B, 1, dim)
        ml = torch.rand(B, 1, 1) + 0.1
        mr = torch.rand(B, 1, 1) + 0.1
        w = ml / (ml + mr)
        return l, r, w * l + (1.0 - w) * r

    mae_in = _fit_merge(_model(dim=dim, mode="gated"), batch_mass_in)
    mae_out = _fit_merge(_model(dim=dim, mode="gated"), batch_mass_out)
    # mass visible -> learnable; mass hidden -> stuck, materially worse
    assert mae_in < 0.02, f"mass-weighted avg should be learnable w/ mass in state; MAE={mae_in}"
    assert mae_out > 2 * mae_in, (
        f"mass-weighted avg must FAIL when mass is hidden (input bottleneck): "
        f"mae_in={mae_in} mae_out={mae_out}"
    )
