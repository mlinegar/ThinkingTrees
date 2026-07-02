"""Tests for per-level lopsidedness-weighted merge evaluation.

Establishes the non-additive merge structure the learned g must capture: the
dim/RILE targets are mass-weighted (ratio) merges, so equal-averaging is wrong
exactly when sibling masses are lopsided, and the error grows with depth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.eval_qsentence_merge_by_level as E
from src.ctreepo.manifesto_qsentence_dspy_family import _lopsidedness_weight


def _node(node_id, level, scores, mass, left=None, right=None):
    return {
        "node_id": node_id,
        "level": level,
        "dimension_scores": scores,
        "left_child_id": left,
        "right_child_id": right,
        "metadata": {"total_non_header": mass},
    }


def _write_tree(tmp_path: Path, nodes: dict) -> str:
    rec = {"doc_id": "d0", "nodes": nodes}
    p = tmp_path / "trees.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    return str(p)


def test_lopsidedness_weight_monotonic_and_clamped():
    assert _lopsidedness_weight(0.0, strength=4.0) == 1.0
    assert _lopsidedness_weight(1.0, strength=4.0) == 5.0
    assert _lopsidedness_weight(0.5, strength=4.0) == 3.0
    # clamp lopsidedness into [0,1]
    assert _lopsidedness_weight(9.0, strength=4.0) == 5.0
    assert _lopsidedness_weight(-1.0, strength=4.0) == 1.0
    # strength 0 disables weighting
    assert _lopsidedness_weight(1.0, strength=0.0) == 1.0


def test_node_mass_from_metadata_and_teacher_summary():
    # evaluator's dict-based mass reader (used on labeled-tree JSON nodes)
    assert E._node_mass({"metadata": {"total_non_header": 7}}) == 7.0
    ts = json.dumps({"cmp_state": {"total_non_header": 12}})
    assert E._node_mass({"metadata": {"teacher_summary": ts}}) == 12.0
    assert E._node_mass({"metadata": {}}) is None


def test_mass_weighted_is_exact_equal_average_is_wrong_when_lopsided(tmp_path):
    # Parent ratio for a single dim with lopsided children.
    # left: 9 qsentences all coded to the dim -> ratio 1.0
    # right: 1 qsentence coded 0 -> ratio 0.0
    # gold parent ratio = 9/10 = 0.9 (mass-weighted), NOT 0.5 (equal-average).
    dim = {"rile": 0.9, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    left = {"rile": 1.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    right = {"rile": 0.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    nodes = {
        "L": _node("L", 0, left, 9),
        "R": _node("R", 0, right, 1),
        "P": _node("P", 1, dim, 10, left="L", right="R"),
    }
    path = _write_tree(tmp_path, nodes)
    res = E.evaluate(path, strength=4.0)
    lvl1 = [r for r in res["by_level"] if r["level"] == 1][0]
    # mass-weighted reproduces the gold ratio exactly
    assert lvl1["mass_wtd_wmae"] == pytest.approx(0.0, abs=1e-9)
    # equal-average is off by 0.4 on the rile dim (|0.5 - 0.9|), averaged over 8
    # dims (7 of which are 0 with no error) -> 0.4/8 = 0.05 unweighted; the
    # lopsidedness weight (lop=0.8 -> w=4.2) cancels in a single-node wmae.
    assert lvl1["equal_avg_wmae"] > 0.04
    # lopsidedness recorded
    assert lvl1["lop_p90"] == pytest.approx(0.8, abs=1e-6)


def test_node_lopsidedness_balanced_is_zero(tmp_path):
    # balanced masses -> lopsidedness 0 -> equal-average is correct
    dim = {"rile": 0.5, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    left = {"rile": 1.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    right = {"rile": 0.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    nodes = {
        "L": _node("L", 0, left, 5),
        "R": _node("R", 0, right, 5),
        "P": _node("P", 1, dim, 10, left="L", right="R"),
    }
    path = _write_tree(tmp_path, nodes)
    res = E.evaluate(path, strength=4.0)
    lvl1 = [r for r in res["by_level"] if r["level"] == 1][0]
    # balanced: both methods are exact, lopsidedness 0
    assert lvl1["equal_avg_wmae"] == pytest.approx(0.0, abs=1e-9)
    assert lvl1["mass_wtd_wmae"] == pytest.approx(0.0, abs=1e-9)
    assert lvl1["lop_p90"] == pytest.approx(0.0, abs=1e-6)


def test_learned_g_scored_against_yardstick(tmp_path):
    dim = {"rile": 0.9, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    left = {"rile": 1.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    right = {"rile": 0.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    nodes = {
        "L": _node("L", 0, left, 9),
        "R": _node("R", 0, right, 1),
        "P": _node("P", 1, dim, 10, left="L", right="R"),
    }
    path = _write_tree(tmp_path, nodes)
    # a perfect learned g emits the mass-weighted ratio at the parent
    g_states = tmp_path / "g.jsonl"
    g_states.write_text(
        json.dumps(
            {
                "doc_id": "d0",
                "node_id": "P",
                "compact_targets": {"rile": 0.9, **{f"domain_{i}": 0.0 for i in range(1, 8)}},
            }
        )
        + "\n"
    )
    res = E.evaluate(path, strength=4.0, g_states_path=str(g_states))
    p = res["pooled_weighted"]
    assert res["g_states"]["nodes_found"] == 1
    # perfect g matches the ceiling (mass-weighted) and beats equal-average
    assert p["learned_g_wmae"] == pytest.approx(0.0, abs=1e-9)
    assert p["learned_g_wmae"] < p["equal_avg_wmae"]


def test_learned_g_via_f_scored_and_rescues_offschema(tmp_path):
    """f_readout scores g THROUGH f, and rescues a node whose compact_targets is
    null (off-schema) — the whole point of scoring the way g is actually used."""
    dim = {"rile": 0.9, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    left = {"rile": 1.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    right = {"rile": 0.0, **{f"domain_{i}": 0.0 for i in range(1, 8)}}
    nodes = {
        "L": _node("L", 0, left, 9),
        "R": _node("R", 0, right, 1),
        "P": _node("P", 1, dim, 10, left="L", right="R"),
    }
    path = _write_tree(tmp_path, nodes)
    # g's raw state did NOT direct-parse (compact_targets=null), but f read it as
    # the mass-weighted ratio. The via-f path must score it (and rescue it).
    g_states = tmp_path / "g.jsonl"
    g_states.write_text(
        json.dumps({
            "doc_id": "d0",
            "node_id": "P",
            "compact_targets": None,
            "f_readout": {"rile": 0.9, **{f"domain_{i}": 0.0 for i in range(1, 8)}},
        }) + "\n"
    )
    res = E.evaluate(path, strength=4.0, g_states_path=str(g_states))
    p = res["pooled_weighted"]
    assert res["g_states"]["has_via_f"] is True
    assert res["g_states"]["via_f_found"] == 1
    # direct-parse found nothing (off-schema), via-f rescued it perfectly
    assert res["g_states"]["nodes_found"] == 0
    assert p.get("learned_g_wmae") is None
    assert p["learned_g_via_f_wmae"] == pytest.approx(0.0, abs=1e-9)
    assert p["learned_g_via_f_wmae"] < p["equal_avg_wmae"]
