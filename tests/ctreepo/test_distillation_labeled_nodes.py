from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

from src.ctreepo.distillation import (
    DistillationContractConfig,
    DistillationTrainConfig,
    FEmbeddingConfig,
    FLMConfig,
    GLMConfig,
    ScoreTargetConfig,
    SummaryTargetConfig,
    attach_labeled_tree_scores,
    build_embedding_tree_from_labeled_tree,
    build_f_embedding_examples,
    build_f_lm_regression_records,
    build_g_sft_records,
    build_labeled_tree_from_text,
    evaluate_labeled_tree_local_laws,
    fit_f_embedding_proxy,
    fit,
    load_labeled_trees,
    repair_labeled_tree_missing_summaries,
    write_labeled_trees_jsonl,
)
from src.training.config_sections import RunConfig, RuntimeConfig, TrainConfig, ValidationConfig
from src.training.ctreepo_trainer import (
    CTreePOTrainer,
    CTreePOTrainingConfig,
    TreeOperatorDataConfig,
    TreeOperatorObjectiveConfig,
)
from src.tree.ctreepo_model import CTreePOConfig
from src.tree.embedding_tree import build_embedding_tree


class FakeEmbeddingClient:
    def resolve_model(self):
        return "fake-embedding-model"

    def embed_texts(self, texts):
        return [
            [
                float(len(text)),
                float(sum(ord(ch) for ch in text) % 17),
                float(idx),
                1.0,
            ]
            for idx, text in enumerate(texts)
        ]


def _score_span(text: str) -> float:
    return float(len(text))


def _distillation_config(
    *,
    train_targets=("tree_operator",),
    student_model_class="ctreepo_embedding_tree",
    supervision_source="labeled_tree_artifact",
    teacher_model_spec=None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    summary_targets: SummaryTargetConfig | None = None,
    score_targets: ScoreTargetConfig | None = None,
    g_lm: GLMConfig | None = None,
    f_embedding: FEmbeddingConfig | None = None,
    f_lm: FLMConfig | None = None,
) -> DistillationTrainConfig:
    return DistillationTrainConfig(
        contract=DistillationContractConfig(
            train_targets=tuple(train_targets),
            student_model_class=student_model_class,
            supervision_source=supervision_source,
            teacher_model_spec=teacher_model_spec,
        ),
        run=RunConfig(output_dir=output_dir, dry_run=dry_run),
        summary_targets=summary_targets or SummaryTargetConfig(),
        score_targets=score_targets or ScoreTargetConfig(),
        g_lm=g_lm or GLMConfig(),
        f_embedding=f_embedding or FEmbeddingConfig(),
        f_lm=f_lm or FLMConfig(),
    )


def _ctreepo_config() -> CTreePOTrainingConfig:
    return CTreePOTrainingConfig(
        model=CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
        data=TreeOperatorDataConfig(window_size=7, window_overlap=0),
        train=TrainConfig(epochs=1, batch_size=1),
        validation=ValidationConfig(eval_every=1),
        runtime=RuntimeConfig(device="cpu"),
        objective=TreeOperatorObjectiveConfig(
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
            idempotence_weight=0.3,
        ),
    )


def test_teacher_trace_labeled_tree_round_trip(tmp_path):
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghijklmnopqrstuvwxyz",
        document_score=12.0,
        split="train",
        score_fn=_score_span,
        window_size=10,
        window_overlap=0,
        label_source="teacher",
        root_summary="summary",
        resummary_target="summary again",
        fill_missing_summaries_from_span=True,
    )

    assert len(tree.get_leaves()) == 3
    assert len(tree.get_merge_nodes()) == 3
    assert tree.metadata["sibling_triples"]
    assert tree.metadata["idempotence_pairs"]
    assert tree.metadata["distillation_state_contract"]["f_input_kind"] == "summary_embedding"
    assert all(node.metadata.get("teacher_summary") for node in tree.nodes.values())
    assert all(node.metadata.get("f_input_kind") == "summary_embedding" for node in tree.nodes.values())
    root = tree.get_node(tree.levels[-1][0])
    assert root is not None
    assert root.metadata["teacher_summary"] == "summary"
    assert root.metadata["teacher_resummary"] == "summary again"

    path = write_labeled_trees_jsonl(tmp_path / "labeled_trees.jsonl", [tree])
    loaded = load_labeled_trees(path)

    assert len(loaded) == 1
    assert loaded[0].doc_id == "doc1"
    assert loaded[0].metadata["artifact_version"] == "ctreepo_labeled_node_distillation_v1"
    assert len(loaded[0].nodes) == len(tree.nodes)


def test_labeled_tree_scores_attach_to_runtime_nodes():
    text = "abcdefghijklmnopqrstuvwxyz"
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text=text,
        document_score=12.0,
        split="train",
        score_fn=_score_span,
        window_size=10,
        window_overlap=0,
        label_source="teacher",
    )
    windows = [(0, 10), (10, 20), (20, 26)]
    embeddings = [[1.0, 0.0, 0.0, 0.0] for _ in windows]
    nodes = build_embedding_tree(text, embeddings, windows)

    stats = attach_labeled_tree_scores(nodes, tree)

    assert stats["attached"] == len(nodes)
    assert stats["missing"] == 0
    assert stats["leaf_attached"] == 3
    assert stats["internal_attached"] == 3
    assert all("rile" in node.oracle_scores for node in nodes)


def test_target_leaves_policy_writes_exact_topology_metadata():
    text = "".join(chr(65 + (idx % 26)) for idx in range(160))
    tree = build_labeled_tree_from_text(
        doc_id="doc16",
        text=text,
        document_score=12.0,
        split="train",
        score_fn=_score_span,
        window_size=999,
        target_leaves_per_doc=16,
        fill_missing_summaries_from_span=True,
    )

    leaves = tree.get_leaves()

    assert len(leaves) == 16
    assert tree.metadata["topology_policy"]["kind"] == "target_leaves_per_doc"
    assert tree.metadata["topology_replay"] == "exact_artifact_spans"
    assert leaves[0].metadata["char_start"] == 0
    assert leaves[-1].metadata["char_end"] == len(text)


def test_generate_teacher_trace_labeled_tree_flags_parse():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / "generate_manifesto_teacher_traces.py"
    spec = importlib.util.spec_from_file_location("generate_manifesto_teacher_traces", str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--output-dir",
            "outputs/test_teacher_trace_flags",
            "--train-size",
            "1",
            "--val-size",
            "0",
            "--test-size",
            "0",
            "--emit-labeled-trees",
            "--target-leaves-per-doc",
            "16",
            "--labeled-tree-node-summary-mode",
            "partial",
        ]
    )

    assert args.emit_labeled_trees is True
    assert args.labeled_tree_target_leaves_per_doc == 16
    assert args.labeled_tree_node_summary_mode == "partial"


def test_exact_labeled_tree_replay_ignores_current_chunk_flags():
    text = "abcdefghijklmnopqrstuvwxyz"
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text=text,
        document_score=12.0,
        split="train",
        score_fn=_score_span,
        window_size=10,
        window_overlap=0,
        fill_missing_summaries_from_span=True,
    )

    nodes, stats = build_embedding_tree_from_labeled_tree(
        tree,
        embedding_client=FakeEmbeddingClient(),
    )

    assert stats["topology_replay"] == "exact_artifact_spans"
    assert [(node.char_start, node.char_end) for node in nodes if node.is_leaf] == [
        (0, 10),
        (10, 20),
        (20, 26),
    ]
    assert [node.children for node in nodes if not node.is_leaf][-1] == (3, 4)


def test_fit_dry_run_does_not_require_teacher_or_embedding_calls():
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
    )

    result = fit([tree], _distillation_config(dry_run=True))

    assert result.train_count == 1
    assert result.val_count == 0
    assert result.metadata["dry_run"] is True
    assert result.train_targets == ("tree_operator",)
    assert result.student_model_class == "ctreepo_embedding_tree"
    assert result.supervision_source == "labeled_tree_artifact"
    assert result.metadata["distillation_contract"]["train_targets"] == ["tree_operator"]


def test_distillation_explicit_contract_metadata():
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
    )

    result = fit(
        [tree],
        _distillation_config(
            train_targets=("f",),
            student_model_class="lm_scalar_regression",
            supervision_source="labeled_tree_artifact",
            teacher_model_spec={"model": "teacher-llm"},
            dry_run=True,
        ),
    )

    assert result.train_targets == ("f",)
    assert result.student_model_class == "lm_scalar_regression"
    assert result.supervision_source == "labeled_tree_artifact"
    assert result.teacher_model_spec == {"model": "teacher-llm"}
    assert result.metadata["distillation_contract"] == {
        "train_targets": ["f"],
        "student_model_class": "lm_scalar_regression",
        "supervision_source": "labeled_tree_artifact",
        "teacher_model_spec": {"model": "teacher-llm"},
    }


def test_g_sft_records_export_from_cached_labeled_trees(tmp_path):
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
        root_summary="teacher root summary",
        resummary_target="teacher root resummary",
        fill_missing_summaries_from_span=True,
    )

    records = build_g_sft_records([tree])
    result = fit(
        [tree],
        _distillation_config(
            train_targets=("g",),
            student_model_class="lm_sft",
            supervision_source="labeled_tree_artifact",
            output_dir=tmp_path,
            summary_targets=SummaryTargetConfig(include_identity_targets=False),
        ),
    )

    assert len(records) == 4
    assert {row["metadata"]["law_role"] for row in records} == {
        "leaf_g",
        "merge_g",
        "idempotence_proxy",
    }
    assert result.student_model_class == "lm_sft"
    assert result.train_targets == ("g",)
    assert result.metadata["distillation_contract"]["train_targets"] == ["g"]
    assert result.metadata["sft_train_records"] == 4
    assert (tmp_path / "g_sft_train.jsonl").exists()
    assert (tmp_path / "g_sft_val.jsonl").exists()


def test_f_embedding_examples_and_proxy_fit(tmp_path):
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
        root_summary="teacher root summary",
        resummary_target="teacher root resummary",
        fill_missing_summaries_from_span=True,
    )

    examples = build_f_embedding_examples([tree])
    result = fit_f_embedding_proxy(
        [tree],
        embedding_client=FakeEmbeddingClient(),
        output_path=tmp_path / "f_embedding_proxy.json",
        ridge_lambda=1e-8,
    )
    unified_result = fit(
        [tree],
        _distillation_config(
            train_targets=("f",),
            student_model_class="embedding_ridge_proxy",
            supervision_source="labeled_tree_artifact",
            output_dir=tmp_path / "unified_f",
            f_embedding=FEmbeddingConfig(ridge_lambda=1e-8),
        ),
        embedding_client=FakeEmbeddingClient(),
    )

    assert len(examples) == len(tree.nodes)
    assert all(0.0 <= ex.target_score <= 1.0 for ex in examples)
    assert result.train_targets == ("f",)
    assert result.metadata["train_examples"] == len(tree.nodes)
    assert result.metadata["train_metrics"]["count"] == len(tree.nodes)
    assert (tmp_path / "f_embedding_proxy.json").exists()
    assert unified_result.train_targets == ("f",)
    assert unified_result.student_model_class == "embedding_ridge_proxy"
    assert unified_result.metadata["train_examples"] == len(tree.nodes)
    assert unified_result.metadata["distillation_contract"]["student_model_class"] == "embedding_ridge_proxy"
    assert "local_law_eval" in unified_result.metadata
    assert unified_result.metadata["local_law_eval"]["C1"]["scored_count"] == 2
    assert (tmp_path / "unified_f" / "f_embedding_proxy.json").exists()


def test_partial_labeled_tree_can_be_repaired_for_g_targets():
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
        root_summary=None,
        fill_missing_summaries_from_span=False,
    )

    assert tree.metadata["summary_coverage"]["partial_artifact"] is True
    assert build_g_sft_records([tree]) == []

    repaired = repair_labeled_tree_missing_summaries(
        [tree],
        lambda text, context: f"{context['node_id']}:{text[:3]}",
    )[0]

    assert repaired.metadata["summary_coverage"]["partial_artifact"] is False
    assert len(build_g_sft_records([repaired])) == len(repaired.nodes)


def test_f_lm_regression_records_and_local_law_eval(tmp_path):
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
        root_summary="teacher root summary",
        resummary_target="teacher root resummary",
        fill_missing_summaries_from_span=True,
    )

    records = build_f_lm_regression_records([tree])
    result = fit(
        [tree],
        _distillation_config(
            train_targets=("f",),
            student_model_class="lm_scalar_regression",
            supervision_source="labeled_tree_artifact",
            output_dir=tmp_path,
        ),
    )
    metrics = evaluate_labeled_tree_local_laws(
        [tree],
        score_fn=lambda text: float(len(text)) / 20.0,
    )

    assert len(records) == len(tree.nodes)
    assert all("score" in row and "response" in row for row in records)
    assert result.train_targets == ("f",)
    assert result.student_model_class == "lm_scalar_regression"
    assert result.metadata["lm_regression_train_records"] == len(tree.nodes)
    assert (tmp_path / "f_lm_regression_train.jsonl").exists()
    assert metrics["C1"]["count"] == 2
    assert metrics["C2"]["idempotence_pairs"] == 1
    assert metrics["C3"]["count"] == 1


def test_distill_ctreepo_students_export_only_cli(tmp_path):
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
        root_summary="teacher root summary",
        resummary_target="teacher root resummary",
        fill_missing_summaries_from_span=True,
    )
    artifact = write_labeled_trees_jsonl(tmp_path / "labeled_trees.jsonl", [tree])
    output_dir = tmp_path / "distill"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/distill_ctreepo_students.py",
            "--labeled-tree-artifacts",
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--export-only",
        ],
        cwd="/home/mlinegar/ThinkingTrees",
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output_dir / "distillation_manifest.json").read_text())
    assert manifest["counts"]["g_train_records"] == 4
    assert manifest["counts"]["f_train_examples"] == len(tree.nodes)
    assert manifest["counts"]["f_lm_train_records"] == len(tree.nodes)
    assert manifest["distillation_contracts"]["g"]["train_targets"] == ["g"]
    assert manifest["distillation_contracts"]["g"]["student_model_class"] == "lm_sft"
    assert manifest["distillation_contracts"]["f_lm_regression"]["train_targets"] == ["f"]
    assert (
        manifest["distillation_contracts"]["f_lm_regression"]["student_model_class"]
        == "lm_scalar_regression"
    )
    assert (output_dir / "g_sft_train.jsonl").exists()
    assert (output_dir / "f_embedding_train_examples.jsonl").exists()
    assert (output_dir / "f_lm_regression_train.jsonl").exists()


def test_ctreepo_trainer_consumes_labeled_trees_and_reports_proxy_idempotence():
    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghijklmnopqrstuvwxyz",
        document_score=12.0,
        split="train",
        score_fn=_score_span,
        window_size=10,
        window_overlap=0,
        label_source="teacher",
    )
    trainer = CTreePOTrainer(
        _ctreepo_config(),
        embedding_client=FakeEmbeddingClient(),
    )

    built = trainer.prepare_trees_from_labeled_trees([tree], split="train")
    nodes, _rile, _doc_id = trainer.train_trees[0]
    summary = trainer._tree_local_law_summary(trainer.train_trees)

    assert built == 1
    assert all("rile" in node.oracle_scores for node in nodes)
    assert [(node.char_start, node.char_end) for node in nodes if node.is_leaf] == [
        (0, 10),
        (10, 20),
        (20, 26),
    ]
    assert summary["labeled_leaves"] == 3
    assert summary["labeled_internal"] == 3
    assert summary["objective"]["idempotence_supervision"] is False
    assert summary["objective"]["proxy_idempotence_penalty"] is True

    optimizer = trainer._make_optimizer()
    loss = trainer.train_step(trainer.train_trees, optimizer)
    wrapper = trainer._last_wrapper_regularization_stats

    assert math.isfinite(loss)
    assert wrapper["idempotence_regularization_active"] is True
    assert wrapper["idempotence_proxy_only"] is True
    assert wrapper["idempotence_term_count"] == len(nodes)
