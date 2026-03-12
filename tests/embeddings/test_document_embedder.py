from __future__ import annotations

import numpy as np

from src.core.doc_metadata import DocMetadata, format_doc_meta_embedding_text
from src.embeddings.document_embedder import DocumentEmbedder, DocumentEmbeddingConfig


class FakeEmbeddingClient:
    def __init__(self, dim: int = 2):
        self.dim = int(dim)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        batch = [str(t) for t in texts]
        self.calls.append(batch)
        out = []
        for text in batch:
            n = float(len(text))
            s = float(sum(ord(ch) for ch in text) % 100)
            vec = [n, s]
            if self.dim > 2:
                vec = vec + [0.0] * (self.dim - 2)
            out.append([float(x) for x in vec[: self.dim]])
        return out


def _fake_vec(text: str, dim: int = 2) -> np.ndarray:
    n = float(len(text))
    s = float(sum(ord(ch) for ch in text) % 100)
    vec = np.array([n, s], dtype=np.float32)
    if dim > 2:
        vec = np.concatenate([vec, np.zeros((dim - 2,), dtype=np.float32)], axis=0)
    return vec[:dim]


def test_build_windows_respects_max_windows_and_tail_coverage():
    client = FakeEmbeddingClient()
    cfg = DocumentEmbeddingConfig(window_chars=10, overlap_chars=0, max_windows=2)
    embedder = DocumentEmbedder(client, config=cfg)

    text = "x" * 101
    windows = embedder.build_windows(text)
    assert len(windows) <= 2
    assert int(windows[-1].end) == len(text)
    assert int(windows[0].start) == 0


def test_build_windows_unlimited_keeps_all_windows():
    client = FakeEmbeddingClient()
    cfg = DocumentEmbeddingConfig(window_chars=10, overlap_chars=0, max_windows=0)
    embedder = DocumentEmbedder(client, config=cfg)

    text = "x" * 25
    windows = embedder.build_windows(text)
    assert [(int(w.start), int(w.end)) for w in windows] == [(0, 10), (10, 20), (20, 25)]


def test_embed_document_pools_and_combines_without_normalization():
    client = FakeEmbeddingClient(dim=2)
    cfg = DocumentEmbeddingConfig(
        window_chars=4,
        overlap_chars=0,
        max_windows=0,
        pooling="mean",
        l2_normalize=False,
        embed_metadata=True,
        text_weight=1.0,
        meta_weight=0.25,
    )
    embedder = DocumentEmbedder(client, config=cfg)

    text = "abcdefghij"  # windows: abcd, efgh, ij
    meta = DocMetadata(doc_id="doc1", source="manifesto", country="Sweden", party="SAP", year=1998)
    result = embedder.embed_document(text, meta=meta)

    expected_text_vec = (_fake_vec("abcd") + _fake_vec("efgh") + _fake_vec("ij")) / 3.0
    assert result.text_vector is not None
    assert np.allclose(result.text_vector, expected_text_vec, atol=1e-6)

    meta_text = format_doc_meta_embedding_text(meta)
    expected_meta_vec = _fake_vec(meta_text)
    assert result.meta_vector is not None
    assert np.allclose(result.meta_vector, expected_meta_vec, atol=1e-6)

    expected_combined = expected_text_vec + 0.25 * expected_meta_vec
    assert result.combined_vector is not None
    assert np.allclose(result.combined_vector, expected_combined, atol=1e-6)
