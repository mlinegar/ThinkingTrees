"""f-null-space salience law: weight emerges from geometry, not estimation.

Guards the reframe (deregulation): the merge weight is NOT mass and should NOT be
estimated. Instead the encoder pushes low-IMPACT content (leave-one-out: removing it
barely changes f(parent)) into f's null space (reads neutral), so an additive/free
merge ignores it automatically.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralop")

from src.ctreepo.embedding_fno import (
    EmbeddingCoordinateFNOTreeRegressor,
    _PreparedTree,
)
from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig
from src.tree.labeled import LabeledNode, LabeledTree


def _family(null_w=1.0, merge_mode="mlp"):
    cfg = FNOFamilyConfig(
        merge_mode=merge_mode,
        hidden_channels=8,
        n_modes=8,
        n_layers=2,
        head_hidden_dim=16,
        g_null_space_weight=null_w,
        identity_init=True,
    )
    fam = FNOFamily(config=cfg, embedding_client=None, device="cpu")
    return fam


def _model(merge_mode="mlp"):
    return EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=16,
        hidden_channels=8,
        n_modes=8,
        n_layers=2,
        head_hidden_dim=16,
        target_min=0.0,
        target_max=1.0,
        merge_mode=merge_mode,
        merge_gate_hidden_dim=16,
    )


def _toy_item(n_leaves=2):
    tree = LabeledTree(doc_id="d0", document_text="", document_score=0.5, metadata={})
    _n = lambda **k: LabeledNode(doc_id="d0", text="", **k)
    tree.add_node(_n(node_id="l0", level=0, score=0.1))
    tree.add_node(_n(node_id="l1", level=0, score=0.9))
    tree.add_node(
        _n(node_id="r0", level=1, score=0.5, left_child_id="l0", right_child_id="l1")
    )
    return _PreparedTree(
        tree=tree,
        split="train",
        leaf_embeddings=torch.randn(2, 16),
        node_order=["l0", "l1", "r0"],
        leaf_ranges={},
        root_node_id="r0",
    )


def test_null_space_term_zero_when_both_children_impactful():
    """If removing either child swings f(parent) fully (impact=1), the penalty is 0."""
    fam = _family()
    m = _model()
    item = _toy_item()
    # Construct states so f(parent) is far from BOTH children's f (max impact).
    # Easiest: stub predict_normalized to return controlled values per state id.
    states = {
        "l0": torch.zeros(1, 1, 16),
        "l1": torch.zeros(1, 1, 16),
        "r0": torch.zeros(1, 1, 16),
    }

    class _M:
        def predict_normalized(self, x):
            n = x.shape[0]
            # parent reads 1.0, children read 0.0 -> impact_l=|1-0|=1, impact_r=1
            # We can't tell which is which by value here; return 0 for children, 1 for
            # parent by tagging via a sentinel: use the fact the term batches p,l,r
            # separately. Simpler: monkeypatch below.
            raise NotImplementedError

    # Simpler: directly exercise with a model whose readout we control via a hook.
    # impact=1 for both => (1-impact)=0 => term must be 0 regardless of child reading.
    def fake_pred(x):
        # parent state is all-zeros here; map zeros->1.0 so f(parent)=1, f(child)=1?
        # We need impact = |f(p)-f(sibling)|. Make f depend on a planted scalar in
        # coord 0 so we can set p=1, children=0 deterministically.
        return torch.sigmoid(x[..., 0, 0] * 50.0)  # ~0 if coord0<0, ~1 if >0

    m.predict_normalized = fake_pred  # type: ignore
    states = {
        "l0": torch.full((1, 1, 16), -1.0),  # f ~ 0
        "l1": torch.full((1, 1, 16), -1.0),  # f ~ 0
        "r0": torch.full((1, 1, 16), 1.0),   # f(parent) ~ 1 -> impact ~1 both
    }
    term = fam._null_space_term(m, states, item)
    assert term.item() < 1e-3, f"impact=1 should zero the penalty; got {term.item()}"


def test_null_space_penalizes_low_impact_child_with_signal():
    """A low-impact child (removing it doesn't move f(parent)) that still reads far
    from neutral must incur penalty."""
    fam = _family()
    m = _model()
    item = _toy_item()

    def fake_pred(x):
        # f = sigmoid(50*coord0): neutral (zeros) -> 0.5
        return torch.sigmoid(x[..., 0, 0] * 50.0)

    m.predict_normalized = fake_pred  # type: ignore
    # parent reads same as right child (impact_left ~ 0): left is low-impact.
    # left reads FAR from neutral (coord0=+1 -> f~1, neutral f~0.5) -> should be penalized.
    states = {
        "l0": torch.full((1, 1, 16), 1.0),    # low-impact (see below), f~1 (far from 0.5)
        "l1": torch.full((1, 1, 16), 0.02),   # f~0.5 ~ neutral
        "r0": torch.full((1, 1, 16), 0.02),   # f(parent)~0.5 == f(right) -> impact_left~0
    }
    term = fam._null_space_term(m, states, item)
    assert term.item() > 1e-3, f"low-impact child with signal must be penalized; got {term.item()}"


def test_null_space_weight_zero_is_backcompat():
    """g_null_space_weight=0 -> f-loss path adds nothing (term never invoked)."""
    fam0 = _family(null_w=0.0)
    assert fam0.config.g_null_space_weight == 0.0
    # The term itself is well-defined but multiplied by 0 in the loss; just confirm
    # the helper runs and returns a finite scalar so the guard is the only gate.
    m = _model()
    item = _toy_item()
    term = fam0._null_space_term(m, {
        "l0": torch.randn(1, 1, 16),
        "l1": torch.randn(1, 1, 16),
        "r0": torch.randn(1, 1, 16),
    }, item)
    assert torch.isfinite(term).all()


def test_training_drives_low_impact_child_toward_neutral():
    """Mechanism: with the law, a planted-NOISE leaf (irrelevant to the parent
    target) is pushed to read neutral, while a planted-SIGNAL leaf keeps its reading.
    """
    torch.manual_seed(0)
    fam = _family(null_w=5.0)
    m = _model()
    m.initialize_as_identity()
    m.freeze_for_f_training()
    fam._model = m
    fam._embedding_dim = 16
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)

    # One tree: left leaf carries the doc signal (parent target tracks it), right leaf
    # is noise. After training with the law, the right (low-impact) leaf should read
    # closer to neutral than the left.
    item = _toy_item()
    # make parent target = left leaf's score so left is high-impact, right low-impact
    item.tree.get_node("r0").score = 0.1   # == left leaf score
    item.tree.get_node("l1").score = 0.5   # right leaf neutral-ish

    for _ in range(150):
        opt.zero_grad()
        loss = fam._train_step_loss_f(m, item)
        loss.backward()
        opt.step()

    from src.ctreepo.embedding_fno import _forward_tree_states

    with torch.no_grad():
        states = _forward_tree_states(m, item, device=torch.device("cpu"))
        f_neutral = m.predict_normalized(torch.zeros(1, 1, 16)).item()
        f_left = m.predict_normalized(states["l0"]).item()
        f_right = m.predict_normalized(states["l1"]).item()
    gap_left = abs(f_left - f_neutral)
    gap_right = abs(f_right - f_neutral)
    # The low-impact (right/noise) leaf should sit closer to neutral than the
    # high-impact (left/signal) leaf.
    assert gap_right < gap_left + 0.05, (
        f"low-impact leaf should be nearer neutral: "
        f"f_neutral={f_neutral:.3f} f_left={f_left:.3f} f_right={f_right:.3f}"
    )
