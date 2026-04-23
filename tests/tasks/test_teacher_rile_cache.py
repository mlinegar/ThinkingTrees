from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tasks.manifesto.teacher_rile_cache import (
    TeacherRILECache,
    _hash_text,
    create_cached_rile_oracle,
    dump_labeled_trees_to_cache,
    dump_nodes_to_cache,
)
from src.tree.labeled import LabeledNode, LabeledTree


class _FakeNode:
    def __init__(self, text_span: str, rile: float, is_leaf: bool):
        self.text_span = text_span
        self.oracle_scores = {"rile": rile}
        self.is_leaf = is_leaf


def _legacy_cache(path: Path) -> TeacherRILECache:
    with pytest.warns(DeprecationWarning, match="TeacherRILECache is deprecated"):
        return TeacherRILECache(path)


def _legacy_oracle(path: Path, **kwargs):
    with pytest.warns(DeprecationWarning, match="TeacherRILECache is deprecated"):
        return create_cached_rile_oracle(path, **kwargs)


def test_teacher_rile_cache_is_deprecated_compatibility_path(tmp_path: Path) -> None:
    cache = _legacy_cache(tmp_path / "rile_cache.jsonl")
    assert cache.deprecated is True


def test_cache_put_and_get_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "rile_cache.jsonl"
    cache = _legacy_cache(cache_path).load()
    cache.put("hello world", 1.25)
    cache.put("another span", -3.5)

    reloaded = _legacy_cache(cache_path).load()
    assert reloaded.get("hello world") == pytest.approx(1.25)
    assert reloaded.get("another span") == pytest.approx(-3.5)
    assert reloaded.get("missing") is None
    assert "hello world" in reloaded
    assert len(reloaded) == 2


def test_cache_is_append_only_and_deduped(tmp_path: Path) -> None:
    cache_path = tmp_path / "rile_cache.jsonl"
    cache = _legacy_cache(cache_path).load()
    cache.put("same text", 2.0)
    cache.put("same text", 2.0)  # exact duplicate — should not re-write
    cache.put("same text", 2.5)  # different score — updates in-memory and appends

    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "Exact-duplicate put should not append; changed score should."
    first = json.loads(lines[0])
    assert first["key"] == _hash_text("same text")
    assert cache.get("same text") == pytest.approx(2.5)


def test_create_cached_oracle_strict_raises_on_miss(tmp_path: Path) -> None:
    cache_path = tmp_path / "rile_cache.jsonl"
    cache = _legacy_cache(cache_path).load()
    cache.put("known span", 7.0)

    oracle = _legacy_oracle(cache_path, strict=True)
    assert oracle("known span") == pytest.approx(7.0)
    with pytest.raises(KeyError):
        oracle("unknown span")


def test_create_cached_oracle_fallback_is_called_and_optionally_writes(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "rile_cache.jsonl"
    _legacy_cache(cache_path).load().put("known", 1.0)

    calls: list[str] = []

    def fallback(text: str) -> float:
        calls.append(text)
        return 42.0

    oracle = _legacy_oracle(
        cache_path, fallback=fallback, write_back=True
    )
    assert oracle("known") == pytest.approx(1.0)
    assert oracle("novel") == pytest.approx(42.0)
    assert calls == ["novel"]

    # write_back=True should persist the fallback result
    reloaded = _legacy_cache(cache_path).load()
    assert reloaded.get("novel") == pytest.approx(42.0)


def test_create_cached_oracle_nonstrict_returns_zero_on_miss(tmp_path: Path) -> None:
    cache_path = tmp_path / "rile_cache.jsonl"
    _legacy_cache(cache_path).load().put("known", 1.0)
    oracle = _legacy_oracle(cache_path, strict=False)
    assert oracle("missing") == pytest.approx(0.0)


def test_dump_nodes_to_cache_walks_trees(tmp_path: Path) -> None:
    cache_path = tmp_path / "rile_cache.jsonl"
    cache = _legacy_cache(cache_path).load()

    leaf_a = _FakeNode("leaf A text", 1.0, is_leaf=True)
    leaf_b = _FakeNode("leaf B text", -1.0, is_leaf=True)
    merge_ab = _FakeNode("merged AB text", 0.25, is_leaf=False)
    # Node with no oracle label should be skipped
    unlabeled = _FakeNode("unlabeled text", None, is_leaf=True)
    unlabeled.oracle_scores = {}
    # Node with empty text should be skipped
    empty = _FakeNode("", 5.0, is_leaf=True)

    trees = [
        ([leaf_a, leaf_b, merge_ab, unlabeled, empty], 0.0, "doc1"),
    ]
    summary = dump_nodes_to_cache(trees, cache)

    assert summary == {"leaf": 2, "merge": 1, "skipped": 2, "total": 3}

    reloaded = _legacy_cache(cache_path).load()
    assert reloaded.get("leaf A text") == pytest.approx(1.0)
    assert reloaded.get("leaf B text") == pytest.approx(-1.0)
    assert reloaded.get("merged AB text") == pytest.approx(0.25)
    assert reloaded.get("unlabeled text") is None
    assert reloaded.get("") is None


def test_dump_labeled_trees_to_cache_projects_legacy_score_examples(
    tmp_path: Path,
) -> None:
    tree = LabeledTree(
        doc_id="doc1",
        document_text="leaf A text\nleaf B text",
        document_score=0.25,
    )
    tree.add_node(
        LabeledNode(
            node_id="leaf_a",
            doc_id="doc1",
            level=0,
            text="leaf A text",
            score=1.0,
            metadata={"teacher_summary": "teacher summary A"},
        )
    )
    tree.add_node(
        LabeledNode(
            node_id="merge_ab",
            doc_id="doc1",
            level=1,
            text="merged AB text",
            score=0.25,
            metadata={"teacher_summary": "teacher merge summary"},
        )
    )

    cache_path = tmp_path / "rile_cache.jsonl"
    cache = _legacy_cache(cache_path).load()
    summary = dump_labeled_trees_to_cache([tree], cache)

    assert summary == {"leaf": 1, "merge": 1, "skipped": 0, "total": 2}
    reloaded = _legacy_cache(cache_path).load()
    assert reloaded.get("leaf A text") == pytest.approx(1.0)
    assert reloaded.get("merged AB text") == pytest.approx(0.25)


def test_oracle_signature_matches_trainer_expectation(tmp_path: Path) -> None:
    """CTreePOTrainer expects Callable[[str], float]. Regression guard."""
    cache_path = tmp_path / "rile_cache.jsonl"
    _legacy_cache(cache_path).load().put("span", 0.5)
    oracle = _legacy_oracle(cache_path)
    result = oracle("span")
    assert isinstance(result, float)
    # Check the provenance attrs survive so trainer logs / repro can reflect them.
    assert getattr(oracle, "__ctreepo_oracle_kind__", None) == "cached_rile"
    assert getattr(oracle, "__ctreepo_cache_path__", None) == str(cache_path)
    assert getattr(oracle, "__ctreepo_oracle_deprecated__", None) is True
    assert getattr(oracle, "__ctreepo_oracle_replacement__", None) == "labeled_tree_artifacts"
