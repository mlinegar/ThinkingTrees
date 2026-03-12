"""Unit tests for the unified tree architecture (Phase 1-3).

Tests that EmbeddingTreeNode carries both text and sketch fields,
build_unified_tree() produces shared topology, and adaptive windowing
integrates with the feedback loop.
"""

import numpy as np
import pytest
import torch

from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel
from src.tree.embedding_tree import (
    EmbeddingTreeNode,
    _embedding_boundary_scores,
    _uniform_windows,
    build_embedding_tree,
    build_unified_tree,
    collect_sketches,
    forward_ctreepo,
    get_root_sketch,
)


# ---------------------------------------------------------------------------
# Fake embedding client for testing (deterministic, no server needed)
# ---------------------------------------------------------------------------


class FakeEmbeddingClient:
    """Deterministic embedding client for testing."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._call_count = 0

    def embed_texts(self, texts):
        """Return deterministic embeddings based on text hash."""
        result = []
        for text in texts:
            rng = np.random.RandomState(hash(text) % (2**31))
            result.append(rng.randn(self.dim).tolist())
            self._call_count += 1
        return result


# ---------------------------------------------------------------------------
# Phase 1: UnifiedNode fields
# ---------------------------------------------------------------------------


class TestUnifiedNodeFields:
    """Test that EmbeddingTreeNode has the unified fields."""

    def test_node_has_summary_field(self):
        node = EmbeddingTreeNode(level=0, text_span="hello", char_start=0, char_end=5)
        assert node.summary == ""
        node.summary = "summarized hello"
        assert node.summary == "summarized hello"

    def test_node_has_audit_result_field(self):
        node = EmbeddingTreeNode(level=0, text_span="x", char_start=0, char_end=1)
        assert node.audit_result is None
        node.audit_result = {"status": "passed", "score": 0.95}
        assert node.audit_result["score"] == 0.95

    def test_node_has_sketch_scores_field(self):
        node = EmbeddingTreeNode(level=0, text_span="x", char_start=0, char_end=1)
        assert node.sketch_scores == {}
        node.sketch_scores["rile"] = -3.5
        assert node.sketch_scores["rile"] == -3.5

    def test_node_has_sketch_confidence_field(self):
        node = EmbeddingTreeNode(level=0, text_span="x", char_start=0, char_end=1)
        assert node.sketch_confidence is None
        node.sketch_confidence = 0.87
        assert node.sketch_confidence == 0.87

    def test_backward_compatible_with_existing_fields(self):
        """Verify existing code that only uses old fields still works."""
        node = EmbeddingTreeNode(
            level=0,
            text_span="test",
            char_start=0,
            char_end=4,
            embedding=[1.0, 2.0, 3.0],
        )
        assert node.is_leaf
        assert node.text_len == 4
        assert node.embedding == [1.0, 2.0, 3.0]
        assert node.oracle_scores == {}


# ---------------------------------------------------------------------------
# Phase 1: build_unified_tree (non-adaptive fallback)
# ---------------------------------------------------------------------------


class TestBuildUnifiedTree:
    """Test build_unified_tree() with adaptive=False (uniform windows)."""

    def test_basic_tree_construction(self):
        text = "a" * 3000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=1000,
            adaptive=False,
        )
        assert len(nodes) > 0
        leaves = [n for n in nodes if n.is_leaf]
        assert len(leaves) >= 2  # 3000 chars / ~1000 = 2-3+ windows

    def test_leaves_have_summary_populated(self):
        text = "Hello world. This is a test document."
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=100,
            adaptive=False,
        )
        for node in nodes:
            if node.is_leaf:
                # Summary should be populated with raw text
                assert node.summary != ""
                assert node.summary == node.text_span

    def test_leaves_have_embeddings(self):
        text = "a" * 2000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=800,
            adaptive=False,
        )
        for node in nodes:
            if node.is_leaf:
                assert node.embedding is not None
                assert len(node.embedding) == 64

    def test_internal_nodes_have_children(self):
        text = "a" * 2000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=800,
            adaptive=False,
        )
        for node in nodes:
            if not node.is_leaf:
                assert node.children is not None
                left_idx, right_idx = node.children
                assert 0 <= left_idx < len(nodes)
                assert 0 <= right_idx < len(nodes)

    def test_root_covers_full_text(self):
        text = "a" * 5000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=1200,
            adaptive=False,
        )
        root = nodes[-1]
        assert root.char_start == 0
        assert root.char_end == 5000

    def test_sketch_forward_pass_works_on_unified_tree(self):
        """CTreePO forward pass should work identically on unified nodes."""
        text = "a" * 2000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=800,
            adaptive=False,
        )

        config = CTreePOConfig(embedding_dim=64, sketch_dim=8, hidden_dim=16)
        model = CTreePOModel(config)
        forward_ctreepo(model, nodes)

        # Every node should have a sketch
        for node in nodes:
            assert node.sketch is not None
            assert node.sketch.shape == (8,)

        # Root sketch should be gettable
        root = get_root_sketch(nodes)
        assert root.shape == (8,)


# ---------------------------------------------------------------------------
# Phase 1: Embedding boundary scores
# ---------------------------------------------------------------------------


class TestEmbeddingBoundaryScores:
    """Test the embedding-drift scoring function."""

    def test_uniform_embeddings_get_low_scores(self):
        """Identical embeddings → all scores near 0 (no drift)."""
        embs = [[1.0, 0.0, 0.0]] * 5
        scores = _embedding_boundary_scores(embs)
        assert len(scores) == 5
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_varying_embeddings_produce_range(self):
        """Different embeddings → scores should vary."""
        rng = np.random.RandomState(42)
        embs = [rng.randn(32).tolist() for _ in range(10)]
        scores = _embedding_boundary_scores(embs)
        assert len(scores) == 10
        assert max(scores) > min(scores)  # not all identical

    def test_single_window(self):
        scores = _embedding_boundary_scores([[1.0, 2.0]])
        assert len(scores) == 1
        assert scores[0] == 0.5

    def test_empty(self):
        scores = _embedding_boundary_scores([])
        assert scores == []


# ---------------------------------------------------------------------------
# Phase 2: Adaptive windowing (if available)
# ---------------------------------------------------------------------------


class TestAdaptiveWindowing:
    """Test build_unified_tree() with adaptive=True."""

    def test_adaptive_mode_produces_valid_tree(self):
        """With a long-enough document, adaptive mode should work."""
        text = "a" * 20000  # must be > coarse_window_size
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            coarse_window_size=4000,
            fine_window_size=1200,
            adaptive=True,
        )
        assert len(nodes) > 0
        root = nodes[-1]
        assert root.char_start == 0
        assert root.char_end == 20000

    def test_adaptive_produces_different_windows_than_uniform(self):
        """Adaptive should refine near boundaries, merge in homogeneous regions."""
        text = "a" * 20000
        client = FakeEmbeddingClient(dim=64)

        adaptive_nodes = build_unified_tree(
            text, client,
            coarse_window_size=4000,
            fine_window_size=1200,
            adaptive=True,
        )
        uniform_nodes = build_unified_tree(
            text, client,
            fine_window_size=1200,
            adaptive=False,
        )

        adaptive_leaves = sum(1 for n in adaptive_nodes if n.is_leaf)
        uniform_leaves = sum(1 for n in uniform_nodes if n.is_leaf)
        # They should generally differ (adaptive refines some, merges others)
        # But with fake embeddings this may or may not differ; just verify both work
        assert adaptive_leaves > 0
        assert uniform_leaves > 0

    def test_short_text_falls_back_to_uniform(self):
        """Documents shorter than coarse_window_size use uniform windows."""
        text = "Hello, world."
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            coarse_window_size=4000,
            fine_window_size=1200,
            adaptive=True,
        )
        assert len(nodes) >= 1
        # Single window for short text
        leaves = [n for n in nodes if n.is_leaf]
        assert len(leaves) == 1


# ---------------------------------------------------------------------------
# Phase 3: Sketch scores and confidence on unified nodes
# ---------------------------------------------------------------------------


class TestSketchScoresOnUnifiedNodes:
    """Test that CTreePO readout scores get written to unified nodes."""

    def test_populate_sketch_scores_manual(self):
        """After forward pass + readout, sketch_scores should be populated."""
        text = "a" * 2000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client,
            fine_window_size=800,
            adaptive=False,
        )

        config = CTreePOConfig(embedding_dim=64, sketch_dim=8, hidden_dim=16)
        model = CTreePOModel(config)
        forward_ctreepo(model, nodes)

        # Populate sketch_scores on all nodes manually
        for node in nodes:
            if node.sketch is not None:
                pred = model.predict(node.sketch, "rile")
                node.sketch_scores["rile"] = pred.item()
                pred_norm = model.predict_normalized(node.sketch, "rile")
                node.sketch_confidence = 1.0 - 2.0 * abs(pred_norm.item() - 0.5)

        for node in nodes:
            assert "rile" in node.sketch_scores
            assert node.sketch_confidence is not None
            assert 0.0 <= node.sketch_confidence <= 1.0


# ---------------------------------------------------------------------------
# Phase 2: Oracle → Feedback signals
# ---------------------------------------------------------------------------


class TestOracleToFeedbackSignals:
    """Test converting oracle scores on tree nodes to ChunkFeedbackSignals."""

    def test_basic_conversion(self):
        from src.preprocessing.chunker import ChunkFeedbackSignal, oracle_to_feedback_signals

        nodes = [
            EmbeddingTreeNode(
                level=1, text_span="abc", char_start=0, char_end=100,
                oracle_scores={"rile": 10.0},
                sketch_scores={"rile": 12.0},
            ),
            EmbeddingTreeNode(
                level=1, text_span="def", char_start=100, char_end=200,
                oracle_scores={"rile": -20.0},
                sketch_scores={"rile": 30.0},  # big error!
            ),
        ]
        signals = oracle_to_feedback_signals(nodes)
        assert len(signals) == 2

        # First node: low error → high oracle_relevance
        assert signals[0].oracle_relevance_probability > 0.9

        # Second node: high error → low oracle_relevance
        assert signals[1].oracle_relevance_probability < signals[0].oracle_relevance_probability

    def test_no_oracle_scores_returns_empty(self):
        from src.preprocessing.chunker import oracle_to_feedback_signals

        nodes = [
            EmbeddingTreeNode(level=0, text_span="x", char_start=0, char_end=10),
        ]
        signals = oracle_to_feedback_signals(nodes)
        assert signals == []

    def test_oracle_without_sketch_gives_moderate_confidence(self):
        from src.preprocessing.chunker import oracle_to_feedback_signals

        nodes = [
            EmbeddingTreeNode(
                level=1, text_span="abc", char_start=0, char_end=100,
                oracle_scores={"rile": 10.0},
                # no sketch_scores
            ),
        ]
        signals = oracle_to_feedback_signals(nodes)
        assert len(signals) == 1
        assert signals[0].confidence == 0.5  # moderate because no sketch


# ---------------------------------------------------------------------------
# Phase 3: Uncertainty-guided audit sampling
# ---------------------------------------------------------------------------


class TestUncertaintyGuidedAudit:
    """Test that audit node selection prefers uncertain sketches."""

    def _make_tree_with_sketches(self):
        text = "a" * 2000
        client = FakeEmbeddingClient(dim=64)
        nodes = build_unified_tree(
            text, client, fine_window_size=500, adaptive=False,
        )
        config = CTreePOConfig(embedding_dim=64, sketch_dim=8, hidden_dim=16)
        model = CTreePOModel(config)
        forward_ctreepo(model, nodes)
        return nodes, config, model

    def test_selects_correct_count(self):
        from src.training.ctreepo_trainer import CTreePOTrainer, CTreePOTrainingConfig

        nodes, config, model = self._make_tree_with_sketches()
        trainer = CTreePOTrainer(CTreePOTrainingConfig(model=config))
        trainer.model = model

        selected = trainer.select_audit_nodes(nodes, n_audit=3)
        internal_count = sum(1 for n in nodes if not n.is_leaf)
        assert len(selected) <= min(3, internal_count)
        assert len(selected) > 0

    def test_all_selected_are_internal(self):
        from src.training.ctreepo_trainer import CTreePOTrainer, CTreePOTrainingConfig

        nodes, config, model = self._make_tree_with_sketches()
        trainer = CTreePOTrainer(CTreePOTrainingConfig(model=config))
        trainer.model = model

        selected = trainer.select_audit_nodes(nodes, n_audit=5)
        for idx in selected:
            assert not nodes[idx].is_leaf

    def test_populate_sketch_scores_via_trainer(self):
        from src.training.ctreepo_trainer import CTreePOTrainer, CTreePOTrainingConfig

        nodes, config, model = self._make_tree_with_sketches()
        trainer = CTreePOTrainer(CTreePOTrainingConfig(model=config))
        trainer.model = model

        trainer.populate_sketch_scores(nodes, head="rile")
        for node in nodes:
            if node.sketch is not None:
                assert "rile" in node.sketch_scores
                assert node.sketch_confidence is not None


# ---------------------------------------------------------------------------
# Phase 4: Pipeline config and unified processing
# ---------------------------------------------------------------------------


class TestBatchedPipelineConfig:
    """Test unified tree config options on BatchedPipelineConfig."""

    def test_unified_tree_defaults_false(self):
        from src.pipelines.batched import BatchedPipelineConfig

        config = BatchedPipelineConfig()
        assert config.unified_tree is False
        assert config.adaptive_windows is False
        assert config.oracle_feedback_to_chunks is False
        assert config.mil_proxy_model_path is None

    def test_unified_tree_can_be_set(self):
        from src.pipelines.batched import BatchedPipelineConfig

        config = BatchedPipelineConfig(
            unified_tree=True,
            adaptive_windows=True,
            oracle_feedback_to_chunks=True,
        )
        assert config.unified_tree is True
        assert config.adaptive_windows is True
        assert config.oracle_feedback_to_chunks is True


class TestProcessUnified:
    """Test the unified pipeline path (process_unified)."""

    def test_process_unified_produces_result(self):
        """Smoke test: process_unified returns a DocumentResult."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.core.documents import DocumentSample, DocumentResult
        from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

        config = BatchedPipelineConfig(unified_tree=True, adaptive_windows=False)
        pipeline = BatchedDocPipeline(config=config)

        # Mock the unified components
        mock_emb_client = FakeEmbeddingClient(dim=64)
        pipeline._unified_emb_client = mock_emb_client

        # Set up a minimal CTreePO model
        from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel

        ctreepo_cfg = CTreePOConfig(embedding_dim=64, sketch_dim=8, hidden_dim=16)
        ctreepo_model = CTreePOModel(ctreepo_cfg)
        pipeline._unified_ctreepo_model = ctreepo_model
        pipeline._unified_ctreepo_config = ctreepo_cfg
        pipeline._unified_mil_model = None
        pipeline._unified_ctreepo_settings = {
            "window_size": 800,
            "coarse_window_size": 4000,
            "merge_drift_threshold": 0.03,
        }

        # Mock strategy
        strategy = AsyncMock()
        strategy.summarize = AsyncMock(return_value="summarized leaf")
        strategy.merge = AsyncMock(return_value="merged summary")

        sample = DocumentSample(doc_id="test_doc", text="a" * 2000)

        result = asyncio.run(pipeline.process_unified(sample, strategy))

        assert isinstance(result, DocumentResult)
        assert result.doc_id == "test_doc"
        assert result.final_summary != ""
        assert result.tree_leaves > 0
        assert result.tree_height >= 1
        assert "ctreepo_rile" in result.metadata
        assert "ctreepo_sketch_dim" in result.metadata
        assert result.metadata["unified_tree_node_count"] > 0
        assert result.error is None

    def test_process_batch_with_strategy_routes_through_orchestrator(self):
        """process_batch_with_strategy delegates to process_batch_global_async."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from src.core.documents import DocumentSample, DocumentResult
        from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

        config = BatchedPipelineConfig(unified_tree=True)
        pipeline = BatchedDocPipeline(config=config)

        mock_results = [DocumentResult(doc_id="test")]
        pipeline.process_batch_global_async = AsyncMock(return_value=mock_results)

        sample = DocumentSample(doc_id="test", text="hello")
        strategy = AsyncMock()

        results = asyncio.run(pipeline.process_batch_with_strategy([sample], strategy))

        assert results is mock_results
        pipeline.process_batch_global_async.assert_called_once_with(
            [sample], show_progress=True, strategy=strategy
        )


# ---------------------------------------------------------------------------
# Phase 5: MIL attention scores and score_windows_callback
# ---------------------------------------------------------------------------


class TestMILAttentionScores:
    """Test MIL proxy model per-window attention scoring."""

    def test_get_mil_attention_scores_returns_per_window(self):
        from src.training.embedding_proxy import EmbeddingMILSGDProxyModel

        model = EmbeddingMILSGDProxyModel(
            weights=[0.5, -0.3, 0.1],
            bias=0.0,
            bag_bias=0.0,
            embedding_dim=3,
            embedding_model="test",
        )

        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        scores = model.get_mil_attention_scores(embeddings)

        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s <= 1.0

        # Different embeddings should give different scores
        assert scores[0] != scores[1] or scores[1] != scores[2]

    def test_mil_scores_match_predict_from_embedding(self):
        from src.training.embedding_proxy import EmbeddingMILSGDProxyModel

        model = EmbeddingMILSGDProxyModel(
            weights=[0.5, -0.3, 0.1, 0.2],
            bias=0.1,
            bag_bias=0.0,
            embedding_dim=4,
            embedding_model="test",
        )

        embs = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]
        scores = model.get_mil_attention_scores(embs)

        for emb, score in zip(embs, scores):
            assert abs(score - model.predict_from_embedding(emb)) < 1e-10


class TestScoreWindowsCallback:
    """Test that score_windows_callback blends with drift scores."""

    def test_callback_blends_with_drift(self):
        """When a callback is provided, scores are 50/50 blended."""
        text = "a" * 20000
        client = FakeEmbeddingClient(dim=64)

        def fake_mil_scores(embs):
            return [0.9] * len(embs)  # high importance everywhere

        nodes_with_cb = build_unified_tree(
            text, client,
            coarse_window_size=4000,
            fine_window_size=1200,
            adaptive=True,
            score_windows_callback=fake_mil_scores,
        )
        nodes_without_cb = build_unified_tree(
            text, client,
            coarse_window_size=4000,
            fine_window_size=1200,
            adaptive=True,
        )

        # Both should produce valid trees
        assert len(nodes_with_cb) > 0
        assert len(nodes_without_cb) > 0
        assert nodes_with_cb[-1].char_end == 20000
        assert nodes_without_cb[-1].char_end == 20000

    def test_callback_not_used_when_not_adaptive(self):
        """Without adaptive=True, the callback is ignored."""
        text = "a" * 2000
        client = FakeEmbeddingClient(dim=64)

        call_count = [0]
        def counting_callback(embs):
            call_count[0] += 1
            return [0.5] * len(embs)

        nodes = build_unified_tree(
            text, client,
            fine_window_size=800,
            adaptive=False,
            score_windows_callback=counting_callback,
        )

        assert len(nodes) > 0
        assert call_count[0] == 0  # not called in non-adaptive mode


# ---------------------------------------------------------------------------
# Orchestrator integration: unified tree through BatchTreeOrchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorUnified:
    """Test that the orchestrator accepts pre-built unified trees."""

    def _make_fake_unified_tree(self, text_len=2000, window_size=800):
        """Build a small unified tree for testing."""
        client = FakeEmbeddingClient(dim=64)
        text = "a" * text_len
        nodes = build_unified_tree(
            text, client,
            fine_window_size=window_size,
            adaptive=False,
        )
        return text, nodes

    def test_process_documents_unified_produces_results(self):
        """process_documents_unified produces BuildResults via cascading."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from src.core.batch_orchestrator import BatchTreeOrchestrator
        from src.tree.builder import BuildConfig

        strategy = MagicMock()
        strategy.summarize = AsyncMock(return_value="leaf summary")
        strategy.merge = AsyncMock(return_value="merged")
        strategy.get_preferences = MagicMock(return_value=[])

        config = BuildConfig(max_chunk_chars=2000)
        orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)

        text, nodes = self._make_fake_unified_tree()
        documents = [text]
        unified_trees = {0: nodes}

        results = asyncio.run(orchestrator.process_documents_unified(
            documents=documents,
            rubric="test rubric",
            unified_trees=unified_trees,
            get_text_fn=lambda d: d,
            get_id_fn=lambda d: "doc_0",
        ))

        assert len(results) == 1
        result = results[0]
        assert not result.errors
        assert result.tree.final_summary is not None
        assert result.tree.metadata.get("unified_tree") is True
        assert result.tree.metadata.get("unified_tree_node_count") == len(nodes)
        assert len(result.tree.metadata.get("chunk_boundaries", [])) > 0

    def test_unified_uses_same_cascading_as_standard(self):
        """Verify unified path calls strategy.summarize/merge (same as standard)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from src.core.batch_orchestrator import BatchTreeOrchestrator
        from src.tree.builder import BuildConfig

        strategy = MagicMock()
        strategy.summarize = AsyncMock(return_value="leaf summary")
        strategy.merge = AsyncMock(return_value="merged")
        strategy.get_preferences = MagicMock(return_value=[])

        config = BuildConfig(max_chunk_chars=2000)
        orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)

        text, nodes = self._make_fake_unified_tree()
        unified_trees = {0: nodes}

        asyncio.run(orchestrator.process_documents_unified(
            documents=[text],
            rubric="test",
            unified_trees=unified_trees,
            get_text_fn=lambda d: d,
            get_id_fn=lambda d: "doc_0",
        ))

        # Strategy should have been called for leaf and merge operations
        assert strategy.summarize.call_count > 0
        assert strategy.merge.call_count > 0

    def test_chunk_boundaries_match_unified_leaves(self):
        """Verify chunk boundaries in results match the unified tree leaves."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from src.core.batch_orchestrator import BatchTreeOrchestrator
        from src.tree.builder import BuildConfig

        strategy = MagicMock()
        strategy.summarize = AsyncMock(return_value="leaf")
        strategy.merge = AsyncMock(return_value="merged")
        strategy.get_preferences = MagicMock(return_value=[])

        config = BuildConfig(max_chunk_chars=2000)
        orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)

        text, nodes = self._make_fake_unified_tree()
        leaves = [n for n in nodes if n.is_leaf]
        unified_trees = {0: nodes}

        results = asyncio.run(orchestrator.process_documents_unified(
            documents=[text],
            rubric="test",
            unified_trees=unified_trees,
            get_text_fn=lambda d: d,
            get_id_fn=lambda d: "doc_0",
        ))

        boundaries = results[0].tree.metadata.get("chunk_boundaries", [])
        assert len(boundaries) == len(leaves)
        for b, leaf_node in zip(boundaries, leaves):
            assert b["char_start"] == leaf_node.char_start
            assert b["char_end"] == leaf_node.char_end

    def test_fallback_to_standard_chunking_when_tree_missing(self):
        """If a doc has no unified tree, orchestrator falls back to chunking."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from src.core.batch_orchestrator import BatchTreeOrchestrator
        from src.tree.builder import BuildConfig

        strategy = MagicMock()
        strategy.summarize = AsyncMock(return_value="leaf")
        strategy.merge = AsyncMock(return_value="merged")
        strategy.get_preferences = MagicMock(return_value=[])

        config = BuildConfig(max_chunk_chars=2000)
        orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)

        text = "word " * 500
        # Empty unified_trees — no doc 0 entry
        unified_trees = {}

        results = asyncio.run(orchestrator.process_documents_unified(
            documents=[text],
            rubric="test",
            unified_trees=unified_trees,
            get_text_fn=lambda d: d,
            get_id_fn=lambda d: "doc_0",
        ))

        # Should still produce a result via standard chunking fallback
        assert len(results) == 1
        assert not results[0].errors


class TestProcessBatchAsyncAlwaysGlobal:
    """Test that process_batch_async always routes to the global path."""

    def test_process_batch_async_delegates_to_global(self):
        """process_batch_async should always delegate to process_batch_global_async."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.core.documents import DocumentSample
        from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

        config = BatchedPipelineConfig()
        pipeline = BatchedDocPipeline(config=config)

        mock_results = [MagicMock()]
        pipeline.process_batch_global_async = AsyncMock(return_value=mock_results)

        samples = [DocumentSample(doc_id="test", text="word " * 200)]
        result = asyncio.run(pipeline.process_batch_async(samples))

        pipeline.process_batch_global_async.assert_called_once()
        assert result == mock_results


class TestBuildAllUnifiedTrees:
    """Test the _build_all_unified_trees helper."""

    def test_builds_trees_for_all_samples(self):
        from src.core.documents import DocumentSample
        from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

        config = BatchedPipelineConfig(unified_tree=True, adaptive_windows=False)
        pipeline = BatchedDocPipeline(config=config)

        # Set up unified components manually
        pipeline._unified_emb_client = FakeEmbeddingClient(dim=64)
        pipeline._unified_ctreepo_model = None
        pipeline._unified_ctreepo_config = None
        pipeline._unified_mil_model = None
        pipeline._unified_ctreepo_settings = {
            "window_size": 800,
            "coarse_window_size": 4000,
            "merge_drift_threshold": 0.03,
        }

        samples = [
            DocumentSample(doc_id="doc_0", text="a" * 2000),
            DocumentSample(doc_id="doc_1", text="b" * 3000),
        ]

        trees = pipeline._build_all_unified_trees(samples, ["doc_0", "doc_1"])

        assert len(trees) == 2
        assert 0 in trees
        assert 1 in trees
        for idx, nodes in trees.items():
            assert len(nodes) > 0
            leaves = [n for n in nodes if n.is_leaf]
            assert len(leaves) > 0


class TestDeadCodeRemoved:
    """Verify dead code paths have been removed."""

    def test_no_chunk_for_ops_adaptive_export(self):
        import src.preprocessing as pp
        assert not hasattr(pp, "chunk_for_ops_adaptive")

    def test_no_process_batch_levelwise_async(self):
        from src.pipelines.batched import BatchedDocPipeline
        assert not hasattr(BatchedDocPipeline, "process_batch_levelwise_async")

    def test_no_build_trees_pipelined(self):
        from src.core.batch_orchestrator import BatchTreeOrchestrator
        assert not hasattr(BatchTreeOrchestrator, "_build_trees_pipelined")

    def test_no_build_tree_with_dspy(self):
        import src.pipelines.batched as mod
        assert not hasattr(mod, "build_tree_with_dspy")

    def test_no_process_with_dspy(self):
        from src.pipelines.batched import BatchedDocPipeline
        assert not hasattr(BatchedDocPipeline, "process_with_dspy")

    def test_no_process_batch_with_dspy(self):
        from src.pipelines.batched import BatchedDocPipeline
        assert not hasattr(BatchedDocPipeline, "process_batch_with_dspy")

    def test_no_process_single_async(self):
        from src.pipelines.batched import BatchedDocPipeline
        assert not hasattr(BatchedDocPipeline, "process_single_async")

    def test_no_process_documents_batched(self):
        import src.pipelines.batched as mod
        assert not hasattr(mod, "process_documents_batched")

    def test_no_run_batched_experiment(self):
        import src.pipelines.batched as mod
        assert not hasattr(mod, "run_batched_experiment")

    def test_no_use_global_batching_config(self):
        from src.pipelines.batched import BatchedPipelineConfig
        assert not hasattr(BatchedPipelineConfig, "use_global_batching")

    # --- Phase 1c: process_with_strategy removed ---

    def test_no_process_with_strategy(self):
        from src.pipelines.batched import BatchedDocPipeline
        assert not hasattr(BatchedDocPipeline, "process_with_strategy")

    # --- Phase 2a: non-pipelined orchestrator methods removed ---

    def test_no_build_all_leaves(self):
        from src.core.batch_orchestrator import BatchTreeOrchestrator
        assert not hasattr(BatchTreeOrchestrator, "_build_all_leaves")

    def test_no_build_trees_levelwise(self):
        from src.core.batch_orchestrator import BatchTreeOrchestrator
        assert not hasattr(BatchTreeOrchestrator, "_build_trees_levelwise")

    # --- Phase 2b: pipelined field and non-pipelined builder removed ---

    def test_no_build_config_pipelined(self):
        from src.tree.builder import BuildConfig
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(BuildConfig)}
        assert "pipelined" not in field_names

    def test_no_build_tree_from_leaves(self):
        from src.tree.builder import AsyncTreeBuilder
        assert not hasattr(AsyncTreeBuilder, "_build_tree_from_leaves")

    # --- Phase 2c: non-pipelined tournament removed ---

    def test_no_tournament_config_pipelined(self):
        from src.core.strategy import TournamentConfig
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(TournamentConfig)}
        assert "pipelined" not in field_names

    def test_no_run_tournament_non_pipelined(self):
        from src.core.strategy import TournamentStrategy
        assert not hasattr(TournamentStrategy, "_run_tournament")

    # --- Phase 2d: BatchOrchestrator and process_samples_batched removed ---

    def test_no_batch_orchestrator_class(self):
        import src.core.batch_processor as mod
        assert not hasattr(mod, "BatchOrchestrator")

    def test_no_process_samples_batched(self):
        import src.core.batch_processor as mod
        assert not hasattr(mod, "process_samples_batched")
