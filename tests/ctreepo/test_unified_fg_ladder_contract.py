from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.ctreepo.distillation import write_labeled_trees_jsonl
from src.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig
from src.ctreepo.alternating import run_alternating_family
from src.ctreepo.embedding_fno import (
    EmbeddingCoordinateFNOTreeRegressor,
    _prepare_trees,
)
from src.ctreepo.fno_family import FNOFamilyConfig
from src.ctreepo.trl_family import TRLFamily, TRLFamilyConfig
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.dimensions import PolicyDimension, get_dimension
from src.tree.labeled import LabeledNode, LabeledTree


class _FakeEmbeddingClient:
    def __init__(self, dim: int = 5) -> None:
        self.dim = int(dim)

    def embed_texts(self, texts):
        return [
            [float((len(str(text)) + idx) % 7) for idx in range(self.dim)]
            for text in texts
        ]


def _tiny_tree(doc_id: str, *, split: str = "test", score: float = 4.0) -> LabeledTree:
    text = (
        f"{doc_id} left policy evidence about investment and jobs. "
        f"{doc_id} right policy evidence about taxation and welfare."
    )
    tree = LabeledTree(
        doc_id=doc_id,
        document_text=text,
        document_score=float(score),
        metadata={
            "split": split,
            "expert_score_1_7": float(score) + 0.1,
            "teacher_score_1_7": float(score),
        },
        label_source="test",
    )
    left = LabeledNode(
        node_id="leaf_0",
        doc_id=doc_id,
        level=0,
        text=f"{doc_id} left policy evidence about investment and jobs.",
        score=float(score) - 0.2,
        metadata={"teacher_summary": "left summary", "target_summary": "left summary"},
    )
    right = LabeledNode(
        node_id="leaf_1",
        doc_id=doc_id,
        level=0,
        text=f"{doc_id} right policy evidence about taxation and welfare.",
        score=float(score) + 0.2,
        metadata={"teacher_summary": "right summary", "target_summary": "right summary"},
    )
    root = LabeledNode(
        node_id="root",
        doc_id=doc_id,
        level=1,
        text=text,
        score=float(score),
        left_child_id="leaf_0",
        right_child_id="leaf_1",
        metadata={"teacher_summary": "root summary", "target_summary": "root summary"},
    )
    tree.add_node(left)
    tree.add_node(right)
    tree.add_node(root)
    return tree


class _ToyAlternatingFamily:
    name = "toy"

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"f_iter_{iteration:02d}.json"
        artifact.write_text('{"kind": "f"}\n', encoding="utf-8")
        return str(artifact)

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"g_iter_{iteration:02d}.json"
        artifact.write_text('{"kind": "g"}\n', encoding="utf-8")
        return str(artifact)

    def score_roots_with_f(self, *, f, g, trees):
        return [float(tree.document_score) for tree in trees]


def test_fno_default_shape_contract_is_embedding_axis_with_fixed_channels() -> None:
    pytest.importorskip("neuralop")

    cfg = FNOFamilyConfig(
        leaf_size_tokens=512,
        embedding_max_length_tokens=2048,
        effective_embedding_dim=768,
    )
    assert cfg.chunks_per_leaf == 1
    assert cfg.effective_embedding_dim == 768

    model = EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=768,
        hidden_channels=4,
        n_modes=8,
        n_layers=1,
        head_hidden_dim=8,
        target_min=1.0,
        target_max=7.0,
    )
    leaf_inputs = torch.randn(3, 768)
    leaf_states = model.encode_leaves(leaf_inputs)
    assert tuple(leaf_states.shape) == (3, 1, 768)
    merged = model.merge(leaf_states[:2], leaf_states[1:])
    assert tuple(merged.shape) == (2, 1, 768)

    future = FNOFamilyConfig(
        leaf_size_tokens=4096,
        embedding_max_length_tokens=2048,
        effective_embedding_dim=1536,
    )
    assert future.chunks_per_leaf == 2


def test_fno_prepare_trees_raises_instead_of_truncating_oversized_leaf() -> None:
    text = " ".join(f"oversized_token_{idx}" for idx in range(60))
    tree = LabeledTree(
        doc_id="oversized",
        document_text=text,
        document_score=4.0,
        metadata={"split": "test", "expert_score_1_7": 4.0},
    )
    tree.add_node(
        LabeledNode(
            node_id="leaf_0",
            doc_id="oversized",
            level=0,
            text=text,
            score=4.0,
        )
    )

    with pytest.raises(RuntimeError, match="needs .* embedding chunks"):
        _prepare_trees(
            [tree],
            embedding_client=_FakeEmbeddingClient(),
            embedding_max_tokens=4,
            chunks_per_leaf=1,
            enforce_no_truncation=True,
        )


def test_dspy_actual_record_budget_guard_hard_errors_before_optimizer() -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            leaf_size_tokens=1,
            lm_context_window_tokens=16,
            max_completion_tokens=4,
            prompt_template_overhead_tokens=1,
            lm_config={"model": "openai/test", "api_base": "http://localhost:9/v1"},
        )
    )

    with pytest.raises(RuntimeError, match="DSPy no-truncation guard failed"):
        family._check_training_record_budgets(
            [
                {
                    "prompt": " ".join(f"budget_token_{idx}" for idx in range(50)),
                    "response": "short",
                }
            ],
            role="f",
        )


def test_dimension_scorer_loads_legacy_score_key_as_predictor() -> None:
    scorer = DimensionScorer(get_dimension(PolicyDimension.ECONOMIC))
    legacy_state = {"score": scorer.dump_state()["predictor"]}

    restored = DimensionScorer(get_dimension(PolicyDimension.ECONOMIC))
    restored.load_state(legacy_state)

    assert callable(restored.predictor)
    assert not callable(getattr(restored, "score", None))


def test_trl_validate_artifact_requires_hf_load_markers(tmp_path: Path) -> None:
    family = TRLFamily(
        config=TRLFamilyConfig(
            leaf_size_tokens=8,
            max_completion_tokens=32,
            lm_context_window_tokens=128,
            prompt_template_overhead_tokens=16,
        )
    )
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

    family.validate_artifact(kind="f", artifact=str(model_dir))

    bad_dir = tmp_path / "bad_model"
    bad_dir.mkdir()
    with pytest.raises(RuntimeError, match="no HuggingFace load markers"):
        family.validate_artifact(kind="g", artifact=str(bad_dir))


def test_alternating_ladder_writes_step_checkpoints_with_current_artifacts(tmp_path: Path) -> None:
    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.1) for idx in range(4)]

    records = run_alternating_family(
        family=_ToyAlternatingFamily(),
        f_init="f0",
        g_init="g0",
        traces=trees,
        eval_trees=trees,
        max_iterations=2,
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path,
    )

    assert [record.stage_name for record in records] == ["fg", "fgf", "fgfg"]
    post_train = json.loads(
        (tmp_path / "step_checkpoints" / "iter_01_post_train.json").read_text(
            encoding="utf-8"
        )
    )
    assert post_train["phase"] == "post_train"
    assert post_train["trained"] == "f"
    assert post_train["f_artifact"].endswith("iter_01_train_f/f_iter_01.json")
    assert post_train["g_artifact"] == "g0"

    latest = json.loads(
        (tmp_path / "step_checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["phase"] == "post_eval"
    assert latest["iteration"] == 2
    assert latest["g_artifact"].endswith("iter_02_train_g/g_iter_02.json")
    assert records[1].f_artifact == post_train["f_artifact"]


def test_alternating_ladder_size_axis_smoke_with_fake_trees(tmp_path: Path) -> None:
    pytest.importorskip("neuralop")

    import scripts.run_alternating_ladder as cli

    fg_dir = tmp_path / "fg_grid"
    write_labeled_trees_jsonl(
        fg_dir / "leaf0008tok" / "labeled_trees.jsonl",
        [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.2) for idx in range(4)],
    )
    output_dir = tmp_path / "alternating"

    rc = cli.main(
        [
            "--families",
            "fno",
            "--teacher-dir",
            str(fg_dir),
            "--output-dir",
            str(output_dir),
            "--leaf-size-tokens",
            "8",
            "--max-iterations",
            "2",
            "--embedding-backend",
            "hashing",
            "--hashing-embedding-dim",
            "8",
            "--fno-hidden-channels",
            "4",
            "--fno-n-modes",
            "4",
            "--fno-n-layers",
            "1",
            "--fno-head-hidden-dim",
            "8",
            "--fno-epochs",
            "1",
            "--fno-batch-size",
            "2",
            "--embedding-dim",
            "8",
        ]
    )

    assert rc == 0
    summary = json.loads((output_dir / "grid_summary.json").read_text(encoding="utf-8"))
    assert summary["topology_axis"] == "leaf_size_tokens"
    assert summary["leaf_grid"] is None
    assert summary["leaf_size_tokens"] == [8]
    assert summary["per_row_paths"] == ["fno/leaf0008tok/iteration_history.json"]
    assert {row["axis_kind"] for row in summary["rows"]} == {"leaf_size_tokens"}
    assert {row["leaf_size_tokens"] for row in summary["rows"]} == {8}

    history = json.loads(
        (output_dir / "fno" / "leaf0008tok" / "iteration_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history["row_label"] == "leaf0008tok"
    assert history["leaf_count"] is None
    assert history["leaf_size_tokens"] == 8
    assert [row["stage_name"] for row in history["iterations"]] == ["fg", "fgf", "fgfg"]

    latest = json.loads(
        (
            output_dir
            / "fno"
            / "leaf0008tok"
            / "step_checkpoints"
            / "latest.json"
        ).read_text(encoding="utf-8")
    )
    assert latest["phase"] == "post_eval"
    assert latest["artifact_validation"]["g"] == "passed"
    assert latest["g_artifact"].endswith(".pt")
