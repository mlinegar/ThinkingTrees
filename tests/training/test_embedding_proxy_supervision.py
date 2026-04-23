import pytest

from src.training.embedding_proxy import LabeledEmbeddingExample, fit_embedding_ridge_proxy
from src.training.supervision import (
    OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
    REPRESENTATION_EMBEDDING_VECTOR,
    TARGET_SCALAR,
)


class _FakeEmbeddingClient:
    def resolve_model(self) -> str:
        return "fake-embedding-model"

    def embed_texts(self, texts):
        mapping = {
            "a": [0.0, 0.0],
            "b": [1.0, 0.0],
            "c": [0.0, 1.0],
            "d": [1.0, 1.0],
        }
        return [mapping[str(text)] for text in texts]


def test_fit_embedding_ridge_proxy_smoke() -> None:
    model = fit_embedding_ridge_proxy(
        [
            LabeledEmbeddingExample(doc_id="a", text="a", target_score=0.0),
            LabeledEmbeddingExample(doc_id="b", text="b", target_score=0.25),
            LabeledEmbeddingExample(doc_id="c", text="c", target_score=0.75),
            LabeledEmbeddingExample(doc_id="d", text="d", target_score=1.0),
        ],
        embedding_client=_FakeEmbeddingClient(),
        ridge_lambda=1e-8,
        model_id="unit_test_embedding_proxy",
    )
    assert model.embedding_model == "fake-embedding-model"
    assert model.embedding_dim == 2
    assert model.train_size == 4
    assert len(model.weights) == 2
    assert model.training_contract is not None
    assert model.training_contract["representation_kind"] == REPRESENTATION_EMBEDDING_VECTOR
    assert model.training_contract["target_kind"] == TARGET_SCALAR
    assert model.training_contract["optimizer_family"] == OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR
    assert model.predict_from_embedding([0.5, 0.5]) == pytest.approx(0.5, abs=1e-4)
