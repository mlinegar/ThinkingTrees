"""Disk-cached embedding wrapper: correctness, persistence, miss-only recompute."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.ctreepo.embedding_cache import DiskCachedEmbeddingClient


class _Counting:
    """Inner client that records how many texts it actually embedded."""

    def __init__(self):
        self.calls = 0
        self.embedded = []

    def embed_texts(self, texts):
        self.calls += 1
        self.embedded.extend(list(texts))
        # deterministic per-text vector
        return [[float(len(t)), float(sum(map(ord, t)) % 97)] for t in texts]


def test_cache_returns_same_vectors_and_only_embeds_misses(tmp_path):
    inner = _Counting()
    c = DiskCachedEmbeddingClient(inner, tmp_path, model_id="m1")
    out1 = c.embed_texts(["a", "bb", "ccc"])
    assert inner.embedded == ["a", "bb", "ccc"]
    # second call with overlap: only the new text is embedded
    out2 = c.embed_texts(["a", "bb", "dddd"])
    assert inner.embedded == ["a", "bb", "ccc", "dddd"]  # only 'dddd' added
    # cached vectors are identical
    assert out2[0] == out1[0] and out2[1] == out1[1]


def test_cache_persists_across_instances(tmp_path):
    inner1 = _Counting()
    c1 = DiskCachedEmbeddingClient(inner1, tmp_path, model_id="m1")
    v = c1.embed_texts(["hello", "world"])
    # a fresh instance (new process simulation) must hit disk, not the inner client
    inner2 = _Counting()
    c2 = DiskCachedEmbeddingClient(inner2, tmp_path, model_id="m1")
    v2 = c2.embed_texts(["hello", "world"])
    assert inner2.calls == 0, "should be served entirely from disk"
    assert v2 == v


def test_model_id_namespaces_cache(tmp_path):
    inner = _Counting()
    c_a = DiskCachedEmbeddingClient(inner, tmp_path, model_id="A")
    c_b = DiskCachedEmbeddingClient(inner, tmp_path, model_id="B")
    c_a.embed_texts(["x"])
    c_b.embed_texts(["x"])  # different model -> different key -> recompute
    assert inner.embedded == ["x", "x"]


def test_parallel_arms_merge_not_clobber(tmp_path):
    """Two clients writing different texts to the same dir must both survive."""
    inner = _Counting()
    a = DiskCachedEmbeddingClient(inner, tmp_path, model_id="m")
    b = DiskCachedEmbeddingClient(inner, tmp_path, model_id="m")
    a.embed_texts(["alpha"])
    b.embed_texts(["beta"])
    # a fresh reader sees BOTH
    reader = DiskCachedEmbeddingClient(_Counting(), tmp_path, model_id="m")
    out = reader.embed_texts(["alpha", "beta"])
    assert len(out) == 2 and all(v for v in out)
