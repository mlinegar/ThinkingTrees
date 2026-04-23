from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.ctreepo.distillation import build_labeled_tree_from_text, write_labeled_trees_jsonl


class _FakeEmbeddingClient:
    def __init__(self, *args, **kwargs):
        pass

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


class _FakeSample:
    manifesto_id = "doc1"
    text = "abcdefghij"
    rile = 1.0
    party_abbrev = "FAKE"
    country_name = "Nowhere"


class _FakeManifestoDataset:
    def __init__(self, *args, **kwargs):
        pass

    def get_sample(self, manifesto_id):
        sample = _FakeSample()
        sample.manifesto_id = str(manifesto_id)
        return sample


def _load_module():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / "train_ctreepo.py"
    spec = importlib.util.spec_from_file_location("train_ctreepo", str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_ctreepo_blocks_model_based_local_law_scoring_by_default(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--pilot",
            "--local-law-teacher-port",
            "8001",
        ],
    )

    rc = int(mod.main())

    assert rc == 2


def test_train_ctreepo_blocks_conflicting_local_law_sources(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--pilot",
            "--local-law-oracle-module",
            "tests.training.fake_oracle:score_span",
            "--local-law-teacher-port",
            "8001",
        ],
    )

    rc = int(mod.main())

    assert rc == 2


def test_train_ctreepo_blocks_task_model_based_local_law_oracle_by_default(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--pilot",
            "--local-law-oracle",
            "task",
            "--local-law-teacher-port",
            "8001",
        ],
    )

    rc = int(mod.main())

    assert rc == 2


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--teacher-rile-cache", "cache.jsonl"],
        ["--teacher-rile-cache-nonstrict"],
        ["--dump-teacher-rile-cache", "cache.jsonl"],
    ],
)
def test_train_ctreepo_rejects_removed_teacher_rile_cache_flags(
    monkeypatch,
    removed_args,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["train_ctreepo.py", "--pilot", *removed_args],
    )

    with pytest.raises(SystemExit) as exc_info:
        mod.main()

    assert exc_info.value.code == 2


def test_train_ctreepo_labeled_tree_artifacts_route_through_distillation_fit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import src.training.embedding_proxy as embedding_proxy

    tree = build_labeled_tree_from_text(
        doc_id="doc1",
        text="abcdefghij",
        document_score=1.0,
        split="train",
        score_fn=_score_span,
        window_size=5,
        window_overlap=0,
        fill_missing_summaries_from_span=True,
    )
    artifact = write_labeled_trees_jsonl(tmp_path / "labeled_trees.jsonl", [tree])
    output_dir = tmp_path / "ctreepo_fit"

    monkeypatch.setattr(embedding_proxy, "VLLMEmbeddingClient", _FakeEmbeddingClient)
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--labeled-tree-artifacts",
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--embedding-dim",
            "4",
            "--tree-model-version",
            "legacy",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--window-size",
            "5",
            "--window-overlap",
            "0",
            "--eval-every",
            "1",
            "--device",
            "cpu",
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "reproducibility_manifest.json").exists()
    repro = json.loads((output_dir / "reproducibility_manifest.json").read_text())
    assert repro["extra"]["distillation_contract"] == {
        "train_targets": ["tree_operator"],
        "student_model_class": "ctreepo_embedding_tree",
        "supervision_source": "labeled_tree_artifact",
        "teacher_model_spec": None,
    }
    result = json.loads((output_dir / "training_result.json").read_text())
    assert result["local_law_summary"]["root_supervision_source"] == "labeled_tree_artifact"
    assert result["local_law_summary"]["node_supervision_source"] == "labeled_tree_artifact"


def test_train_ctreepo_root_only_empirical_training_reports_empirical_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import src.tasks.manifesto.data_loader as data_loader
    import src.training.embedding_proxy as embedding_proxy

    output_dir = tmp_path / "ctreepo_root_only"
    monkeypatch.setattr(embedding_proxy, "VLLMEmbeddingClient", _FakeEmbeddingClient)
    monkeypatch.setattr(data_loader, "ManifestoDataset", _FakeManifestoDataset)
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--train-ids",
            "doc1",
            "--output-dir",
            str(output_dir),
            "--embedding-dim",
            "4",
            "--tree-model-version",
            "legacy",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--window-size",
            "5",
            "--window-overlap",
            "0",
            "--eval-every",
            "1",
            "--device",
            "cpu",
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    payload = json.loads((output_dir / "training_result.json").read_text())
    summary = payload["local_law_summary"]
    assert summary["supervision_source"] == "empirical_root_labels"
    assert summary["root_supervision_source"] == "empirical_root_labels"
    assert summary["node_supervision_source"] == "none"
    assert summary["objective"]["root_supervision"] is True
    assert summary["labeled_leaves"] == 0
    assert summary["labeled_internal"] == 0


def test_train_ctreepo_online_human_only_creates_feedback_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import src.tasks.manifesto.data_loader as data_loader
    import src.training.embedding_proxy as embedding_proxy

    output_dir = tmp_path / "ctreepo_online_human"
    monkeypatch.setattr(embedding_proxy, "VLLMEmbeddingClient", _FakeEmbeddingClient)
    monkeypatch.setattr(data_loader, "ManifestoDataset", _FakeManifestoDataset)
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--train-ids",
            "doc1",
            "--output-dir",
            str(output_dir),
            "--embedding-dim",
            "4",
            "--tree-model-version",
            "legacy",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--window-size",
            "5",
            "--window-overlap",
            "0",
            "--eval-every",
            "1",
            "--device",
            "cpu",
            "--online-human-only",
            "--online-leaf-query-budget-per-epoch",
            "1",
            "--online-merge-query-budget-per-epoch",
            "1",
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    assert (output_dir / "online_feedback_store.json").exists()
    assert (output_dir / "training_result.json").exists()


def test_train_ctreepo_online_teacher_worker_smoke(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import src.tasks.manifesto.data_loader as data_loader
    import src.training.embedding_proxy as embedding_proxy

    output_dir = tmp_path / "ctreepo_online_teacher"
    monkeypatch.setattr(embedding_proxy, "VLLMEmbeddingClient", _FakeEmbeddingClient)
    monkeypatch.setattr(data_loader, "ManifestoDataset", _FakeManifestoDataset)
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--train-ids",
            "doc1",
            "--output-dir",
            str(output_dir),
            "--embedding-dim",
            "4",
            "--tree-model-version",
            "legacy",
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--window-size",
            "5",
            "--window-overlap",
            "0",
            "--eval-every",
            "1",
            "--device",
            "cpu",
            "--local-law-oracle",
            "tests.training.fake_oracle:score_span",
            "--online-local-law-supervision",
            "--online-teacher-worker",
            "on",
            "--online-leaf-query-budget-per-epoch",
            "2",
            "--online-merge-query-budget-per-epoch",
            "1",
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    store_payload = (output_dir / "online_feedback_store.json").read_text()
    assert "completed" in store_payload
