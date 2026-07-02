"""The learned 'extent' latent: mass-aware general g for the FNO merge.

Guards the design that every node state carries an extra scalar 'extent'
coordinate (a free latent, laws-only) which the merge gate reads so g can weight
children by information density — without breaking the channel invariant or
old checkpoints. The extent is excised before every FNO/score op and re-injected
only as a gate feature + a propagated scalar.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralop")

from src.ctreepo.embedding_fno import EmbeddingCoordinateFNOTreeRegressor


def _model(*, extent=False, init="neutral", mode="gated"):
    return EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=16,
        hidden_channels=8,
        n_modes=8,
        n_layers=2,
        head_hidden_dim=16,
        target_min=0.0,
        target_max=1.0,
        merge_mode=mode,
        merge_gate_hidden_dim=16,
        extent_enabled=extent,
        extent_merge_init=init,
    )


def test_extent_off_is_width_d_and_unchanged():
    """extent_enabled=False: state stays width D, no new params, behavior intact."""
    m = _model(extent=False)
    assert m.leaf_extent_head is None and m.extent_merge_head is None
    x = torch.randn(4, 16)
    s = m.encode_leaves(x)
    assert s.shape == (4, 1, 16)
    mg = m.merge(s[:2], s[2:])
    assert mg.shape == (2, 1, 16)
    assert m.predict_normalized(s).shape == (4,)


def test_extent_on_widens_state_to_d_plus_1():
    m = _model(extent=True)
    x = torch.randn(4, 16)
    s = m.encode_leaves(x)
    assert s.shape == (4, 1, 17)
    mg = m.merge(s[:2], s[2:])
    assert mg.shape == (2, 1, 17)


def test_extent_requires_gated_merge():
    with pytest.raises(ValueError, match="merge_mode='gated'"):
        _model(extent=True, mode="mean")


def test_predict_ignores_extent_coordinate():
    """The score reads only the D state coords; the extent is a routing latent."""
    m = _model(extent=True)
    x = torch.randn(5, 16)
    s = m.encode_leaves(x)
    base = m.predict_normalized(s)
    scrambled = s.clone()
    scrambled[..., 16] += 7.5  # only the extent coord
    assert torch.allclose(base, m.predict_normalized(scrambled), atol=1e-6)


def test_additive_init_parent_extent_is_sum():
    """additive init: untrained parent extent == m_left + m_right (mass prior)."""
    m = _model(extent=True, init="additive")
    m.initialize_as_identity()
    sL, sR = torch.randn(3, 1, 17), torch.randn(3, 1, 17)
    sL[..., 16] = torch.tensor([0.2, 0.5, 1.0]).view(3, 1)
    sR[..., 16] = torch.tensor([0.3, 0.1, 2.0]).view(3, 1)
    out = m.merge(sL, sR)
    assert torch.allclose(out[..., 16], (sL[..., 16] + sR[..., 16]), atol=1e-5)


def test_neutral_init_parent_extent_is_zero():
    """neutral init: untrained parent extent == 0 (the collapse basin / arm A)."""
    m = _model(extent=True, init="neutral")
    m.initialize_as_identity()
    sL, sR = torch.randn(3, 1, 17), torch.randn(3, 1, 17)
    sL[..., 16] = torch.tensor([0.2, 0.5, 1.0]).view(3, 1)
    sR[..., 16] = torch.tensor([0.3, 0.1, 2.0]).view(3, 1)
    out = m.merge(sL, sR)
    assert torch.allclose(out[..., 16], torch.zeros(3, 1), atol=1e-6)


def test_gate_alpha_responds_to_child_extent():
    """Mass visibility: changing only a child's extent shifts the blended state.

    This is the whole point — the gate can weight children by extent (density),
    which the two child STATE vectors alone cannot encode.
    """
    m = _model(extent=True, init="additive")
    # Perturb the gate's final layer so alpha != 0.5 and depends on its inputs.
    for p in m.merge_gate[-1].parameters():
        torch.nn.init.normal_(p, std=0.5)
    sL, sR = torch.randn(4, 1, 17), torch.randn(4, 1, 17)
    out1 = m.merge(sL, sR)
    sL2 = sL.clone()
    sL2[..., 16] += 3.0  # only the left extent changes
    out2 = m.merge(sL2, sR)
    state_delta = (out1[..., :16] - out2[..., :16]).abs().max().item()
    assert state_delta > 1e-4, "gate is extent-blind"


def test_merge_self_is_identity_on_state_at_init():
    """merge(a,a) == a on the STATE coords at identity init, for any gate alpha.

    The invariant holds at init (FNO residual zeroed); the extent coordinate does
    not disturb it because it's excised before the state blend / FNO.
    """
    m = _model(extent=True, init="additive")
    m.initialize_as_identity()
    # Even with a perturbed gate (alpha != 0.5), merge(a,a) keeps the state since
    # alpha*a + (1-alpha)*a = a and the FNO residual is zeroed at init.
    for p in m.merge_gate[-1].parameters():
        torch.nn.init.normal_(p, std=0.3)
    a = torch.randn(3, 1, 17)
    out_aa = m.merge(a, a)
    assert torch.allclose(out_aa[..., :16], a[..., :16], atol=1e-4)


def test_merge_learns_mass_weighting_via_extent_coordinate():
    """The mechanism: with mass carried in the EXTENT coordinate, the gated merge
    learns the mass-weighted average of the D state coords.

    Mirrors test_fno_merge_mass_weighted_needs_mass_in_state in the probe, but the
    mass lives in the dedicated extent slot (index D) the gate reads as a feature —
    the design's claim that the extent makes mass-weighting learnable WITHOUT
    polluting a state dimension. The target is on the D state coords only.
    """
    torch.manual_seed(0)
    m = _model(extent=True, init="additive")
    m.initialize_as_identity()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def batch(B):
        # state coords ~ U[0,1]; extent (index 16) = the per-child mass.
        l = torch.rand(B, 1, 17)
        r = torch.rand(B, 1, 17)
        ml = l[:, :, 16:17].clamp_min(1e-3)
        mr = r[:, :, 16:17].clamp_min(1e-3)
        w = ml / (ml + mr)
        tgt_state = w * l[:, :, :16] + (1.0 - w) * r[:, :, :16]
        return l, r, tgt_state

    for _ in range(600):
        opt.zero_grad()
        l, r, tgt = batch(256)
        out = m.merge(l, r)
        loss = torch.nn.functional.l1_loss(out[:, :, :16], tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        l, r, tgt = batch(2048)
        mae = torch.nn.functional.l1_loss(m.merge(l, r)[:, :, :16], tgt).item()
    assert mae < 0.03, f"extent-fed mass-weighted merge should be learnable; MAE={mae}"


def test_extent_round_trips_through_forward_tree_states():
    """A small tree forwarded level-synchronously keeps width D+1 end to end."""
    from src.ctreepo.embedding_fno import _PreparedTree
    from src.tree.labeled import LabeledNode, LabeledTree

    # 2 leaves -> 1 root, balanced.
    root = "r0"
    tree = LabeledTree(
        doc_id="d0", document_text="", document_score=0.5, metadata={}
    )
    _n = lambda **kw: LabeledNode(doc_id="d0", text="", **kw)
    tree.add_node(_n(node_id="l0", level=0, score=0.1))
    tree.add_node(_n(node_id="l1", level=0, score=0.9))
    tree.add_node(
        _n(node_id="r0", level=1, score=0.5, left_child_id="l0", right_child_id="l1")
    )
    item = _PreparedTree(
        tree=tree,
        split="test",
        leaf_embeddings=torch.randn(2, 16),
        node_order=["l0", "l1", "r0"],
        leaf_ranges={},
        root_node_id=root,
    )
    from src.ctreepo.embedding_fno import _forward_tree_states

    m = _model(extent=True)
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    for nid in ["l0", "l1", "r0"]:
        assert states[nid].shape == (1, 1, 17), (nid, states[nid].shape)
    # score is finite and reads only the D coords
    score = m.predict_normalized(states["r0"])
    assert torch.isfinite(score).all()
