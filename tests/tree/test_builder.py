"""
Tests for OPS tree builder.

These tests focus on the sync convenience functions (build, build_test_tree).
The TreeBuilder class is async-first and tested via integration tests.
"""

import asyncio
import pytest
from pathlib import Path
from src.tree.builder import (
    BuildConfig,
    TreeBuilder,
    IdentitySummarizer, TruncatingSummarizer, ConcatenatingSummarizer,
    build, build_test_tree
)
from src.preprocessing.chunker import TextChunk
from src.core.data_models import Tree


class TestSummarizers:
    """Tests for built-in summarizers."""

    def test_identity_summarizer(self):
        """Identity returns content unchanged."""
        summarizer = IdentitySummarizer()
        content = "Original content here."
        result = summarizer(content, "any rubric")
        assert result == content

    def test_truncating_summarizer(self):
        """Truncating shortens long content."""
        summarizer = TruncatingSummarizer(max_length=20)
        short = "Short."
        long = "This is a very long piece of content that exceeds the limit."

        assert summarizer(short, "") == short
        assert len(summarizer(long, "")) == 20
        assert summarizer(long, "").endswith("...")

    def test_concatenating_summarizer(self):
        """Concatenating adds prefix."""
        summarizer = ConcatenatingSummarizer(prefix="[SUM] ")
        result = summarizer("Some content", "")
        assert result.startswith("[SUM] ")


class TestBuildConfig:
    """Tests for BuildConfig."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = BuildConfig()
        assert config.max_chunk_chars > 0
        assert config.max_chunk_tokens is None
        assert config.min_chunk_chars > 0
        assert config.chunk_strategy in ("axis", "sentence", "paragraph")

    def test_custom_config(self):
        """Custom config values work."""
        config = BuildConfig(
            max_chunk_chars=500,
            min_chunk_chars=50,
            chunk_strategy="paragraph",
            verbose=True
        )
        assert config.max_chunk_chars == 500
        assert config.verbose is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_build_simple(self, medium_text):
        """build creates valid tree."""
        tree = build(medium_text, rubric="Test rubric")

        assert isinstance(tree, Tree)
        assert tree.root is not None
        assert tree.validate() == []
        assert "chunk_boundaries" in tree.metadata
        assert len(tree.metadata["chunk_boundaries"]) >= 1
        leaves = tree.leaves
        assert leaves
        for leaf_node, boundary in zip(leaves, tree.metadata["chunk_boundaries"]):
            assert isinstance(leaf_node.metadata.get("char_start"), int)
            assert isinstance(leaf_node.metadata.get("char_end"), int)
            assert leaf_node.metadata.get("char_start") == boundary.get("start_char")
            assert leaf_node.metadata.get("char_end") == boundary.get("end_char")

    def test_build_with_custom_summarizer(self, long_text):
        """Custom summarizer used."""
        call_count = 0

        def counting(content, rubric):
            nonlocal call_count
            call_count += 1
            return content[:50]

        tree = build(long_text, summarizer=counting, max_chars=200)

        assert tree is not None
        if tree.node_count > 1:
            assert call_count > 0

    def test_build_test_tree(self):
        """build_test_tree creates predictable tree."""
        tree = build_test_tree(num_leaves=4)

        assert tree.leaf_count == 4
        assert tree.validate() == []

    def test_build_test_tree_various_sizes(self):
        """Test tree with various leaf counts."""
        for n in [1, 2, 3, 4, 5, 8, 15]:
            tree = build_test_tree(num_leaves=n)
            assert tree.leaf_count == n
            assert tree.validate() == []


class TestTreeProperties:
    """Property-based tests for tree invariants using build()."""

    def test_all_nodes_reachable_from_root(self, long_text):
        """All nodes reachable via traversal from root."""
        tree = build(long_text, max_chars=200)

        traversed = set(node.id for node in tree.traverse_preorder())
        all_nodes_count = tree.node_count

        assert len(traversed) == all_nodes_count

    def test_parent_child_consistency(self, long_text):
        """Parent-child references are consistent."""
        tree = build(long_text, max_chars=200)

        for node in tree.traverse_preorder():
            for child in node.children:
                assert child.parent is node


class _BulkAwareStrategy:
    def __init__(self):
        self.summarize_calls = 0
        self.merge_calls = 0
        self.summarize_many_calls = 0
        self.merge_many_calls = 0

    async def summarize(self, content: str, rubric: str) -> str:
        self.summarize_calls += 1
        return f"S:{content[:8]}"

    async def merge(self, left: str, right: str, rubric: str) -> str:
        self.merge_calls += 1
        return f"M:{left}|{right}"

    async def summarize_many(self, items):
        self.summarize_many_calls += 1
        return [f"S:{str(item['content'])[:8]}" for item in items]

    async def merge_many(self, items):
        self.merge_many_calls += 1
        return [
            f"M:{str(item['left'])}|{str(item['right'])}"
            for item in items
        ]


def test_tree_builder_unified_v2_uses_bulk_leaf_and_merge_methods():
    strategy = _BulkAwareStrategy()
    builder = TreeBuilder(
        strategy=strategy,
        config=BuildConfig(max_chunk_chars=50, runtime_mode="unified_v2"),
    )
    chunks = [
        TextChunk(text=f"chunk-{idx}", start_char=idx * 10, end_char=(idx + 1) * 10, chunk_index=idx)
        for idx in range(4)
    ]

    result = asyncio.run(builder.build_from_chunks(chunks, rubric="preserve"))

    assert result.tree.leaf_count == 4
    assert strategy.summarize_many_calls == 1
    assert strategy.merge_many_calls >= 1
    assert strategy.summarize_calls == 0
    assert strategy.merge_calls == 0
    assert result.tree.final_summary.startswith("M:")


def test_tree_builder_honors_token_chunk_budget():
    strategy = _BulkAwareStrategy()
    builder = TreeBuilder(
        strategy=strategy,
        config=BuildConfig(max_chunk_chars=8000, max_chunk_tokens=32),
    )
    text = ("tree builder token chunk budget " * 300).strip()

    result = builder.build_sync(text, rubric="preserve")

    boundaries = result.tree.metadata.get("chunk_boundaries", [])
    assert boundaries
    assert all(int(boundary.get("token_count", 0) or 0) <= 32 for boundary in boundaries)
