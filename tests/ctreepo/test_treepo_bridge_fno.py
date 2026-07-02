from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.ctreepo.treepo_bridge.fno import (
    THINKINGTREES_FNO_FAMILY,
    HashingEmbeddingClient,
    register_fno_family,
)
from src.tree.labeled import LabeledNode, LabeledTree


def _tiny_tree(doc_id: str, *, score: float, split: str = "test") -> LabeledTree:
    left_text = f"{doc_id} left policy evidence jobs investment"
    right_text = f"{doc_id} right policy evidence taxation welfare"
    text = f"{left_text} {right_text}"
    tree = LabeledTree(
        doc_id=doc_id,
        document_text=text,
        document_score=float(score),
        metadata={
            "split": split,
            "teacher_score_1_7": float(score),
            "expert_score_1_7": float(score) + 0.05,
        },
        label_source="test",
    )
    tree.add_node(
        LabeledNode(
            node_id="leaf_0",
            doc_id=doc_id,
            level=0,
            text=left_text,
            score=float(score) - 0.1,
        )
    )
    tree.add_node(
        LabeledNode(
            node_id="leaf_1",
            doc_id=doc_id,
            level=0,
            text=right_text,
            score=float(score) + 0.1,
        )
    )
    tree.add_node(
        LabeledNode(
            node_id="root",
            doc_id=doc_id,
            level=1,
            text=text,
            score=float(score),
            left_child_id="leaf_0",
            right_child_id="leaf_1",
        )
    )
    return tree


def _tiny_trees() -> list[LabeledTree]:
    return [
        _tiny_tree("doc0", score=3.8),
        _tiny_tree("doc1", score=4.1),
        _tiny_tree("doc2", score=4.4),
    ]


def test_latest_treepo_builtin_fno_is_available() -> None:
    import treepo
    from treepo.methods.families import list_families, resolve_family

    assert str(treepo.__file__).endswith("/treepo/src/treepo/__init__.py")
    assert "fno" in set(list_families())
    runtime = resolve_family(
        "fno",
        {
            "embedding_dim": 16,
            "hidden_channels": 4,
            "n_modes": 2,
            "n_layers": 1,
            "device": "cpu",
        },
    )
    assert runtime.name == "fno"


def test_register_thinkingtrees_fno_uses_separate_family_name() -> None:
    from treepo.methods.families import list_families, resolve_family
    from src.ctreepo.fno_family import FNOFamilyConfig

    family_name = register_fno_family()
    assert family_name == THINKINGTREES_FNO_FAMILY
    assert THINKINGTREES_FNO_FAMILY in set(list_families())
    runtime = resolve_family(
        THINKINGTREES_FNO_FAMILY,
        {
            "fno_config": FNOFamilyConfig(
                hidden_channels=4,
                n_modes=4,
                n_layers=1,
                head_hidden_dim=8,
                epochs_per_iteration=1,
                batch_size=2,
                leaf_size_tokens=64,
                effective_embedding_dim=16,
                embedding_max_length_tokens=None,
                identity_init=True,
                seed=5,
            ),
            "embedding_client": HashingEmbeddingClient(dim=16),
            "device": "cpu",
        },
    )
    assert runtime.name == "fno"


def test_treepo_methods_fit_runs_builtin_fno(tmp_path: Path) -> None:
    import treepo

    trees = _tiny_trees()
    result = treepo.fit(
        {
            "family": "fno",
            "train_data": trees,
            "eval_data": trees,
            "axis": {"max_iterations": 1, "axis_value": 0},
            "backend_config": {
                "embedding_dim": 16,
                "hidden_channels": 4,
                "n_modes": 2,
                "n_layers": 1,
                "head_hidden_dim": 8,
                "epochs_per_iteration": 1,
                "batch_size": 2,
                "device": "cpu",
                "output_dir": str(tmp_path / "builtin_fno_fit"),
            },
        },
    )
    assert result.status == "success"
    metrics = dict(result.metrics or {})
    assert metrics["n"] == 3.0
    assert math.isfinite(float(metrics["internal_f_mae"]))
    assert result.artifacts["f"]["kind"] == "treepo_fno"
    assert result.manifest_path is not None
    assert Path(result.manifest_path).exists()


def test_treepo_methods_fit_runs_thinkingtrees_fno(tmp_path: Path) -> None:
    import treepo
    from src.ctreepo.fno_family import FNOFamilyConfig

    family_name = register_fno_family()
    trees = _tiny_trees()
    result = treepo.fit(
        {
            "family": family_name,
            "train_data": trees,
            "eval_data": trees,
            "initial_artifacts": {"f": "identity", "g": "identity"},
            "axis": {"max_iterations": 0, "axis_value": 0},
            "backend_config": {
                "fno_config": FNOFamilyConfig(
                    hidden_channels=4,
                    n_modes=4,
                    n_layers=1,
                    head_hidden_dim=8,
                    epochs_per_iteration=1,
                    batch_size=2,
                    leaf_size_tokens=64,
                    effective_embedding_dim=16,
                    embedding_max_length_tokens=None,
                    identity_init=True,
                    seed=5,
                ),
                "embedding_client": HashingEmbeddingClient(dim=16),
                "device": "cpu",
                "output_dir": str(tmp_path / "thinkingtrees_fno_fit"),
            },
        },
    )
    assert result.status == "success"
    metrics = dict(result.metrics or {})
    assert metrics["n"] == 3.0
    assert math.isfinite(float(metrics["internal_f_mae"]))
    assert result.manifest_path is not None
    assert Path(result.manifest_path).exists()
