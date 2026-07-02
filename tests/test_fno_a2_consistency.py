"""Lean A2 merge-consistency law + A3 readout-merge on the FNO.

A2: f(parent) == merge of child readings, through f (D f*(u.v, g(g u . g v))=0).
A3: the merge factors through the readout via the Aczel phi-form, associative +
commutative BY CONSTRUCTION (the proven Lean merge_assoc / merge_comm).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralop")

from src.ctreepo.embedding_fno import (
    EmbeddingCoordinateFNOTreeRegressor,
    _PreparedTree,
    _forward_tree_states,
)
from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig
from src.tree.labeled import LabeledNode, LabeledTree


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


def _family(**kw):
    cfg = FNOFamilyConfig(
        merge_mode="mlp",
        hidden_channels=8,
        n_modes=8,
        n_layers=2,
        head_hidden_dim=16,
        identity_init=True,
        target_min=0.0,  # match _model() + the toy's [0,1] scores
        target_max=1.0,
        **kw,
    )
    return FNOFamily(config=cfg, embedding_client=None, device="cpu")


def _toy(i=0):
    t = LabeledTree(doc_id=f"d{i}", document_text="", document_score=0.5, metadata={})
    _n = lambda **k: LabeledNode(doc_id=f"d{i}", text="", **k)
    t.add_node(_n(node_id="l0", level=0, score=0.1))
    t.add_node(_n(node_id="l1", level=0, score=0.9))
    t.add_node(_n(node_id="r0", level=1, score=0.5, left_child_id="l0", right_child_id="l1"))
    return _PreparedTree(
        tree=t, split="train", leaf_embeddings=torch.randn(2, 16),
        node_order=["l0", "l1", "r0"], leaf_ranges={}, root_node_id="r0",
    )


# --- A3 readout-merge: assoc + comm by construction (the proven Lean property) ---

def test_readout_merge_commutative_and_associative_by_construction():
    m = _model()
    with torch.no_grad():
        m.readout_merge_offset.fill_(0.6)  # arbitrary nonzero
    a, b, c = torch.rand(7), torch.rand(7), torch.rand(7)
    assert torch.allclose(m.readout_merge(a, b), m.readout_merge(b, a), atol=1e-6)
    left = m.readout_merge(m.readout_merge(a, b), c)
    right = m.readout_merge(a, m.readout_merge(b, c))
    assert torch.allclose(left, right, atol=1e-5)


def test_readout_merge_stays_in_unit_interval():
    m = _model()
    with torch.no_grad():
        m.readout_merge_offset.fill_(-2.0)
    out = m.readout_merge(torch.rand(50), torch.rand(50))
    assert (out > 0).all() and (out < 1).all()


# --- merge-consistency law f*(A.B) = f*(g(A).g(B)) ---
# LEFT side = INDEPENDENT reading of the parent text (NOT the merge route).


def test_a2_state_zero_at_identity_init():
    """At identity init f(.)=0.5 everywhere, so both sides read 0.5 -> term 0."""
    fam = _family(g_a2_weight=1.0, a2_mode="state")
    m = _model()
    m.initialize_as_identity()
    item = _toy()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    assert fam._a2_term(m, states, item).item() < 1e-6


def test_a2_readout_zero_at_identity_init():
    fam = _family(g_a2_weight=1.0, a2_mode="readout")
    m = _model()
    m.initialize_as_identity()
    item = _toy()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    # offset 0 -> M(0.5,0.5)=sigmoid(0)=0.5 == f*(A.B)=0.5 -> term 0
    assert fam._a2_term(m, states, item).item() < 1e-6


def test_a2_state_is_nonvacuous_unlike_old_self_comparison():
    """THE FIX: the old term compared f(parent_state) to f(merge(l,r)); since the
    forward pass DEFINES parent_state = merge(l,r), that residual was identically
    zero -- a no-op regardless of f or g. The corrected term compares the merge
    route to the INDEPENDENT parent-text reading f*(A.B), which is generally
    different, so the term is positive for a non-trivial (non-identity) model."""
    torch.manual_seed(0)
    fam = _family(g_a2_weight=1.0, a2_mode="state")
    m = _model()  # NO identity init -> random score head, non-constant f
    item = _toy()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    # Structural fact: parent_state IS the merge of children, so the OLD residual
    # f(parent_state) - f(merge(l,r)) is ~0 even for this non-trivial model.
    f_parent_state = m.predict_normalized(states["r0"])
    f_merge = m.predict_normalized(m.merge(states["l0"], states["l1"]))
    assert torch.allclose(f_parent_state, f_merge, atol=1e-6)
    # The CORRECTED term is non-zero: the merge route disagrees with the text read.
    assert fam._a2_term(m, states, item).item() > 1e-5


def test_a2_zero_when_both_sides_read_constant():
    """If f reads a constant, the merge route and the text reading agree -> term 0,
    in both modes (offset 0 -> M(c,c)=c for c=0.5)."""
    fam_s = _family(g_a2_weight=1.0, a2_mode="state")
    fam_r = _family(g_a2_weight=1.0, a2_mode="readout")
    m = _model()
    m.predict_normalized = lambda x: torch.full((x.reshape(-1, x.shape[-1]).shape[0],), 0.5)  # type: ignore
    item = _toy()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    assert fam_s._a2_term(m, states, item).item() < 1e-6
    assert fam_r._a2_term(m, states, item).item() < 1e-6


def _toy3(i=0):
    """3 leaves -> one UNOBSERVED interior (score=None) -> observed root."""
    t = LabeledTree(doc_id=f"d{i}", document_text="", document_score=0.5, metadata={})
    _n = lambda **k: LabeledNode(doc_id=f"d{i}", text="", **k)
    t.add_node(_n(node_id="l0", level=0, score=0.1))
    t.add_node(_n(node_id="l1", level=0, score=0.9))
    t.add_node(_n(node_id="l2", level=0, score=0.5))
    # interior merge of l0,l1 is UNSUPERVISED (score=None) -> only the text-read
    # proxy can constrain it.
    t.add_node(_n(node_id="m0", level=1, score=None, left_child_id="l0", right_child_id="l1"))
    t.add_node(_n(node_id="r0", level=2, score=0.6, left_child_id="m0", right_child_id="l2"))
    return _PreparedTree(
        tree=t, split="train", leaf_embeddings=torch.randn(3, 16),
        node_order=["l0", "l1", "l2", "m0", "r0"], leaf_ranges={}, root_node_id="r0",
    )


def test_a2_constrains_unsupervised_interior_via_text_read():
    """The novel content of the corrected law: an UNOBSERVED interior merge (score
    None) still contributes, through the AIPW proxy = independent parent-text read.
    With a non-trivial f the term is positive even though the interior has no gold."""
    torch.manual_seed(0)
    fam = _family(g_a2_weight=1.0)
    m = _model()  # non-identity -> f non-constant
    item = _toy3()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    assert fam._a2_term(m, states, item).item() > 1e-5


# --- Lambda = root/law convex split (the canonical ObjectiveSpec knob) ---

def test_lambda_zero_is_root_only():
    """Lambda=0 -> pure rootLoss: only the observed root merge is supervised; the
    unsupervised interior contributes nothing (its text-read law has weight 0)."""
    torch.manual_seed(0)
    fam_root = _family(local_law_weight=0.0)
    fam_law = _family(local_law_weight=1.0)
    m = _model()
    item = _toy3()  # has an UNOBSERVED interior m0
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    root_only = fam_root._a2_term(m, states, item).item()
    pure_law = fam_law._a2_term(m, states, item).item()
    # Pure law ignores the root; pure root ignores the interior -> they differ.
    assert root_only > 1e-6 and pure_law > 1e-6
    assert abs(root_only - pure_law) > 1e-6


def test_lambda_default_trains_merge_law():
    """Default Lambda -> the canonical root/law-split merge objective is positive
    when the merge route disagrees with gold/text."""
    torch.manual_seed(0)
    fam = _family()  # local_law_weight defaults to 0.5
    m = _model()
    fam._model = m
    fam._embedding_dim = 16
    item = _toy3()
    loss = fam._train_step_loss_g(m, item)
    assert torch.isfinite(loss).all() and loss.item() > 0


def test_gamma_depth_zero_collapses_law_to_shallowest():
    """Lean convention: depth = max_level - level so the ROOT is depth 0. With
    Lambda=1 (pure law, root excluded) and gamma=0, every non-root law node sits at
    depth>=1 and is zeroed -> the law objective collapses to ~0."""
    torch.manual_seed(0)
    fam = _family(local_law_weight=1.0, gamma_depth=0.0)
    m = _model()
    item = _toy3()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    assert fam._a2_term(m, states, item).item() < 1e-6


# --- A3 readout factorization: SEPARATE law from A2 ---

def test_a3_factorization_zero_at_identity_init():
    fam = _family(a3_factorization_weight=1.0)
    m = _model()
    m.initialize_as_identity()
    item = _toy()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    # f(merge)=0.5 and M(0.5,0.5)=0.5 -> factorization residual 0
    assert fam._a3_factorization_term(m, states, item).item() < 1e-6


def test_a3_factorization_positive_when_merge_does_not_factor():
    torch.manual_seed(0)
    fam = _family(a3_factorization_weight=1.0)
    m = _model()  # non-identity: state merge generally != phi-merge of readouts
    item = _toy()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    assert fam._a3_factorization_term(m, states, item).item() > 1e-6


def test_assoc_diagnostic_runs_and_is_finite():
    fam = _family(g_assoc_weight=0.5)
    m = _model()
    m.initialize_as_identity()
    item = _toy3()
    states = _forward_tree_states(m, item, device=torch.device("cpu"))
    assert torch.isfinite(fam._assoc_term(m, states, item)).all()
