"""Acceptance matrix for the current ``treepo`` methods surface.

Built-in families run end-to-end through ``treepo.methods.run`` across a range
of leaf sizes. Application families stay outside the standalone package and
must fail through the same dispatch surface with a clear extension error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest

import treepo
from treepo.methods.families import list_families, resolve_family
from treepo.methods.fixtures import make_hll_item_trees


LEAF_SIZES = (2, 4, 8, 16)
BUILTIN_FAMILIES = (
    "oracle",
    "fno",
    "neural_operator",
    "learnable_constant",
    "classical_sketch",
    "llm",
    "dspy",
)
EXTENSION_FAMILIES = ("diffusion", "dgemma", "diffusiongemma", "trl")
NEURAL_OPERATOR_CASES = (
    ("fno", "fno"),
    ("neural_operator", "fno"),
    ("neural_operator", "conv1d"),
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _hll_trees(leaves: int):
    return list(
        make_hll_item_trees(
            n_trees=6, leaves_per_tree=leaves, leaf_unit_count=24, seed=0
        )
    )


def _make_text_trees(n_leaves: int, n_trees: int = 6) -> List:
    """Perfect binary text trees with exactly ``n_leaves`` leaves."""
    from src.tree.labeled import LabeledNode, LabeledTree

    assert n_leaves >= 2 and (n_leaves & (n_leaves - 1)) == 0, "n_leaves must be a power of two >= 2"
    trees: List = []
    for i in range(n_trees):
        doc_id = f"doc_{n_leaves}_{i}"
        base = 3.0 + 0.4 * i
        leaf_rows = []
        for j in range(n_leaves):
            text = f"{doc_id} leaf{j} policy evidence about investment jobs taxation welfare {j}"
            score = base + 0.05 * (j - n_leaves / 2.0)
            leaf_rows.append((f"l0_{j}", text, score))

        root_text = " ".join(t for _, t, _ in leaf_rows)
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=root_text,
            document_score=base,
            metadata={
                "split": "train" if i % 2 == 0 else "test",
                "teacher_score_1_7": base,
                "expert_score_1_7": base + 0.1,
                "observed": True,
                "propensity": 1.0,
            },
            label_source="test",
        )
        for nid, text, score in leaf_rows:
            tree.add_node(LabeledNode(node_id=nid, doc_id=doc_id, level=0, text=text, score=score))

        current = list(leaf_rows)
        level = 1
        while len(current) > 1:
            nxt = []
            for k in range(0, len(current), 2):
                lid, lt, ls = current[k]
                rid, rt, rs = current[k + 1]
                text = f"{lt} {rt}"
                score = (ls + rs) / 2.0
                nid = "root" if len(current) == 2 else f"l{level}_{k // 2}"
                tree.add_node(
                    LabeledNode(
                        node_id=nid,
                        doc_id=doc_id,
                        level=level,
                        text=text,
                        score=score,
                        left_child_id=lid,
                        right_child_id=rid,
                    )
                )
                nxt.append((nid, text, score))
            current = nxt
            level += 1
        trees.append(tree)
    return trees


# --------------------------------------------------------------------------- #
# Registry / wiring
# --------------------------------------------------------------------------- #


def test_builtin_families_registered() -> None:
    names = set(list_families())
    for fam in BUILTIN_FAMILIES:
        assert fam in names, f"family {fam!r} missing from treepo.methods registry"


@pytest.mark.parametrize("family", EXTENSION_FAMILIES)
def test_extension_families_fail_clearly_without_registration(family: str) -> None:
    with pytest.raises(KeyError, match="not registered"):
        resolve_family(family, {})


# --------------------------------------------------------------------------- #
# Offline end-to-end fits across leaf sizes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("leaves", LEAF_SIZES)
def test_oracle_across_leaf_sizes(tmp_path: Path, leaves: int) -> None:
    trees = _hll_trees(leaves)
    result = treepo.fit(
        {
            "family": "oracle",
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "oracle_name": "hll_exact",
                "output_dir": str(tmp_path / f"oracle_{leaves}"),
            },
        }
    )
    assert result.status == "success"
    metrics = dict(result.metrics or {})
    assert _finite(metrics.get("internal_f_mae")), metrics
    assert float(metrics["internal_f_mae"]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("leaves", LEAF_SIZES)
def test_learnable_constant_across_leaf_sizes(tmp_path: Path, leaves: int) -> None:
    trees = _hll_trees(leaves)
    result = treepo.fit(
        {
            "family": "learnable_constant",
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {"output_dir": str(tmp_path / f"lc_{leaves}")},
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )
    assert result.status == "success"
    metrics = dict(result.metrics or {})
    assert _finite(metrics.get("internal_f_mae")), metrics


@pytest.mark.parametrize(("family", "operator_kind"), NEURAL_OPERATOR_CASES)
@pytest.mark.parametrize("leaves", LEAF_SIZES)
def test_neural_operator_families_across_leaf_sizes(
    tmp_path: Path,
    leaves: int,
    family: str,
    operator_kind: str,
) -> None:
    trees = _make_text_trees(leaves)
    result = treepo.fit(
        {
            "family": family,
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "operator_kind": operator_kind,
                "embedding_dim": 16,
                "hidden_channels": 4,
                "n_modes": 2,
                "n_layers": 1,
                "head_hidden_dim": 8,
                "epochs_per_iteration": 1,
                "batch_size": 2,
                "device": "cpu",
                "output_dir": str(tmp_path / f"{family}_{operator_kind}_{leaves}"),
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
        },
    )
    assert result.status == "success"
    metrics = dict(result.metrics or {})
    finite_metrics = [k for k, v in metrics.items() if _finite(v)]
    assert finite_metrics, (
        f"no finite metrics for {family}/{operator_kind} @ leaves={leaves}: {metrics}"
    )
