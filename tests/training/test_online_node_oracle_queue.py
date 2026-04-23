from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.feedback.store import FeedbackStore
from src.feedback.types import FeedbackDimension, FeedbackRequest, FeedbackResponse
from src.training.config_sections import RuntimeConfig, TrainConfig, ValidationConfig
from src.training.ctreepo_trainer import (
    CTreePOTrainer,
    CTreePOTrainingConfig,
    OnlineLocalLawSupervisionConfig,
    LocalLawSupervisionConfig,
    TreeOperatorDataConfig,
    TreeOperatorObjectiveConfig,
)
from src.training.online_node_oracle import (
    OnlineNodeOracleQueue,
    OnlineNodeOracleQueueConfig,
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


@dataclass
class FakeSample:
    manifesto_id: str
    text: str
    rile: float


def _tree_item():
    text = "abcdefghij"
    windows = [(0, 5), (5, 10)]
    nodes = build_embedding_tree(
        text,
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        windows,
    )
    return nodes, 1.0, "doc1"


def _training_config(*, epochs: int, batch_size: int = 1) -> CTreePOTrainingConfig:
    return CTreePOTrainingConfig(
        model=CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
        data=TreeOperatorDataConfig(window_size=5, window_overlap=0),
        train=TrainConfig(epochs=epochs, batch_size=batch_size),
        validation=ValidationConfig(eval_every=1),
        runtime=RuntimeConfig(device="cpu"),
        objective=TreeOperatorObjectiveConfig(
            leaf_audit_weight=0.1,
            merge_audit_weight=0.1,
        ),
        supervision=LocalLawSupervisionConfig(
            online=OnlineLocalLawSupervisionConfig(enabled=True)
        ),
    )


def test_feedback_store_file_backed_duplicate_and_reload(tmp_path: Path):
    path = tmp_path / "feedback_store.json"
    store = FeedbackStore(storage_path=path, autosave=True)
    request = FeedbackRequest(
        request_id="req1",
        text_a="abc",
        dimensions=[FeedbackDimension(kind="scalar", name="score")],
    )

    assert store.enqueue(request) == "req1"
    assert store.enqueue(request) == "req1"
    assert len(store.get_pending(limit=10)) == 1

    reloaded = FeedbackStore(storage_path=path, autosave=True)
    assert len(reloaded.get_pending(limit=10)) == 1
    assert reloaded.submit(
        "req1",
        FeedbackResponse(request_id="req1", scores={"score": 3.0}, source="oracle"),
    )
    assert reloaded.submit(
        "req1",
        FeedbackResponse(request_id="req1", scores={"score": 3.0}, source="oracle"),
    )

    final = FeedbackStore(storage_path=path, autosave=True)
    assert final.get_statistics()["pending"] == 0
    assert final.get_statistics()["completed"] == 1


def test_online_node_oracle_queue_samples_requests_and_attaches_completed(tmp_path: Path):
    store = FeedbackStore(storage_path=tmp_path / "feedback_store.json", autosave=True)
    queue = OnlineNodeOracleQueue(
        store=store,
        config=OnlineNodeOracleQueueConfig(
            leaf_budget_per_epoch=1,
            merge_budget_per_epoch=1,
            source_kind="oracle_callback",
            source_spec="tests.training.fake_oracle:score_span",
        ),
    )
    nodes, target, doc_id = _tree_item()

    stats = queue.enqueue_epoch_requests([(nodes, target, doc_id)], split="train", epoch=0)
    pending = store.get_pending(limit=10)

    assert stats["enqueued"] == 2
    assert len(pending) == 2
    assert {req.context["node_kind"] for req in pending} == {"leaf", "internal"}
    assert all(req.context["ctreepo_online_node_oracle"] for req in pending)
    assert all(
        req.context["supervision_timing"]["activation_barrier"] == "epoch_boundary"
        for req in pending
    )
    assert all(req.sampling.policy_name == "budgeted_random_node_feedback" for req in pending)

    for request in pending:
        store.submit(
            request.request_id,
            FeedbackResponse(
                request_id=request.request_id,
                scores={"score": float(len(request.text_a))},
                source="oracle",
            ),
        )

    attach = queue.attach_completed([(nodes, target, doc_id)], split="train")

    assert attach.attached == 2
    assert attach.leaf_attached == 1
    assert attach.merge_attached == 1
    assert len(attach.observations) == 2
    assert sum(1 for node in nodes if "rile" in node.oracle_scores) == 2


def test_ctreepo_trainer_online_mode_without_worker_does_not_block(tmp_path: Path):
    store = FeedbackStore(storage_path=tmp_path / "feedback_store.json", autosave=True)
    queue = OnlineNodeOracleQueue(
        store=store,
        config=OnlineNodeOracleQueueConfig(leaf_budget_per_epoch=1, merge_budget_per_epoch=1),
    )
    trainer = CTreePOTrainer(
        _training_config(epochs=1, batch_size=1),
        embedding_client=FakeEmbeddingClient(),
        online_node_oracle_queue=queue,
    )

    trainer.prepare_trees_from_samples(
        [FakeSample("doc1", "abcdefghij", 1.0)],
        split="train",
    )
    result = trainer.train(output_dir=tmp_path / "train")

    assert result.epochs_completed == 1
    assert store.get_statistics()["pending"] > 0
    assert all("rile" not in node.oracle_scores for node in trainer.train_trees[0][0])
    payload = json.loads((tmp_path / "train" / "training_result.json").read_text())
    assert payload["local_law_summary"]["online_node_oracle_queue"]["enabled"] is True
    timing = payload["local_law_summary"]["supervision_timing"]
    assert timing["acquisition_policy"] == "async_feedback_queue"
    assert timing["activation_barrier"] == "epoch_boundary"
    assert timing["blocking"] is False


def test_ctreepo_trainer_online_teacher_worker_attaches_at_epoch_boundary(tmp_path: Path):
    store = FeedbackStore(storage_path=tmp_path / "feedback_store.json", autosave=True)
    queue = OnlineNodeOracleQueue(
        store=store,
        config=OnlineNodeOracleQueueConfig(leaf_budget_per_epoch=2, merge_budget_per_epoch=1),
    )
    trainer = CTreePOTrainer(
        _training_config(epochs=2, batch_size=1),
        embedding_client=FakeEmbeddingClient(),
        node_oracle_predictor=lambda text: float(len(text)),
        node_oracle_source_kind="oracle_callback",
        online_node_oracle_queue=queue,
        online_teacher_worker=True,
        online_worker_concurrency=2,
    )

    trainer.prepare_trees_from_samples(
        [FakeSample("doc1", "abcdefghij", 1.0)],
        split="train",
    )
    trainer.train(output_dir=tmp_path / "train")

    labeled = sum(1 for node in trainer.train_trees[0][0] if "rile" in node.oracle_scores)
    assert labeled > 0
    assert store.get_statistics()["completed"] > 0
