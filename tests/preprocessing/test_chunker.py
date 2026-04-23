"""
Tests for document chunking.
"""

import pytest
from pathlib import Path
from src.preprocessing.chunker import (
    AdaptiveChunkMemory,
    AdaptiveChunkingConfig,
    ChunkFeedbackSignal,
    Chunker,
    HonestChunkingPolicy,
    TextChunk,
    assign_honest_split,
    chunk_for_ops,
    feedback_from_prediction_errors,
)


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_create_chunk(self):
        """Basic chunk creation."""
        chunk = TextChunk(
            text="Hello world.",
            start_char=0,
            end_char=12,
            chunk_index=0
        )
        assert chunk.text == "Hello world."
        assert chunk.char_count == 12
        assert chunk.chunk_index == 0

    def test_metadata(self):
        """Metadata storage."""
        chunk = TextChunk(text="content", metadata={'source': 'test.txt'})
        assert chunk.metadata['source'] == 'test.txt'

    def test_repr(self):
        """Chunk repr is readable."""
        chunk = TextChunk(text="Short text", chunk_index=0)
        repr_str = repr(chunk)
        assert "TextChunk" in repr_str
        assert "0" in repr_str


class TestChunker:
    """Tests for Chunker class."""

    def test_chunk_empty_text(self):
        """Empty text returns empty list."""
        chunker = Chunker(max_tokens=500)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_creates_chunks(self, medium_text):
        """Chunker creates chunks from text."""
        chunker = Chunker(max_tokens=100)
        chunks = chunker.chunk(medium_text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.text) > 0

    def test_chunk_indices_sequential(self, long_text):
        """Chunk indices are sequential."""
        chunker = Chunker(max_tokens=100)
        chunks = chunker.chunk(long_text)
        if len(chunks) > 1:
            indices = [c.chunk_index for c in chunks]
            assert indices == list(range(len(chunks)))

    def test_chunk_file(self, temp_text_file):
        """Chunk a text file."""
        chunker = Chunker(max_tokens=500)
        chunks = chunker.chunk_file(temp_text_file)
        assert len(chunks) >= 1
        assert chunks[0].metadata['source_file'] == str(temp_text_file)

    def test_chunk_nonexistent_file(self):
        """Non-existent file raises."""
        chunker = Chunker(max_tokens=500)
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file(Path("/nonexistent/file.txt"))

    def test_iter_chunks(self, medium_text):
        """iter_chunks yields same as chunk."""
        chunker = Chunker(max_tokens=500)
        chunks_list = chunker.chunk(medium_text)
        chunks_iter = list(chunker.iter_chunks(medium_text))
        assert len(chunks_list) == len(chunks_iter)


class TestChunkForOps:
    """Tests for convenience function."""

    def test_axis_strategy_default(self, medium_text):
        """Axis strategy is the default for adaptive-ready chunking."""
        chunks = chunk_for_ops(medium_text, max_chars=200)
        assert len(chunks) >= 1

    def test_sentence_strategy(self, medium_text):
        """Sentence strategy remains available explicitly."""
        chunks = chunk_for_ops(medium_text, max_chars=200, strategy="sentence")
        assert len(chunks) >= 1

    def test_paragraph_strategy(self, medium_text):
        """Paragraph strategy works."""
        chunks = chunk_for_ops(medium_text, max_chars=500, strategy="paragraph")
        assert len(chunks) >= 1

    def test_invalid_strategy(self, short_text):
        """Invalid strategy falls back to default."""
        # Should not raise, just use default
        chunks = chunk_for_ops(short_text, strategy="unknown")
        assert len(chunks) >= 1

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert chunk_for_ops("") == []
        assert chunk_for_ops("   ") == []

    def test_token_budget_takes_precedence(self):
        """Token-budget chunking produces bounded leaves from one tokenization pass."""
        text = ("token budget chunking should respect exact token windows. " * 200).strip()
        chunks = chunk_for_ops(text, max_chars=8000, max_tokens=64, strategy="axis")
        assert len(chunks) > 1
        assert all(chunk.token_count <= 64 for chunk in chunks)
        assert all(chunk.metadata.get("token_budget") == 64 for chunk in chunks)

    def test_adaptive_chunking_metadata(self, long_text):
        """Adaptive chunking annotates chunks with policy metadata."""
        config = AdaptiveChunkingConfig(enabled=True, min_chars=80, max_chars=600)
        chunks = chunk_for_ops(long_text, max_chars=200, strategy="axis", adaptive_config=config)
        assert len(chunks) >= 1
        assert "adaptive_policy" in chunks[0].metadata
        assert "adaptive_target_chars" in chunks[0].metadata

    def test_feedback_signals_affect_adaptive_target(self, medium_text):
        """Boundary feedback can alter adaptive target size metadata."""
        baseline_cfg = AdaptiveChunkingConfig(enabled=True, min_chars=50, max_chars=400)
        base_chunks = chunk_for_ops(medium_text, max_chars=140, adaptive_config=baseline_cfg)
        assert len(base_chunks) >= 1
        base_target = base_chunks[0].metadata.get("adaptive_target_chars")

        signals = [
            ChunkFeedbackSignal(
                start_char=0,
                end_char=max(1, len(medium_text) // 2),
                low_info_probability=1.0,
                noise_probability=0.8,
                confidence=1.0,
            )
        ]
        adapt_chunks = chunk_for_ops(
            medium_text,
            max_chars=140,
            adaptive_config=baseline_cfg,
            feedback_signals=signals,
        )
        assert len(adapt_chunks) >= 1
        adapt_target = adapt_chunks[0].metadata.get("adaptive_target_chars")
        assert base_target is not None and adapt_target is not None
        assert adapt_target >= base_target


class TestAdaptiveFeedbackAndHonesty:
    """Tests for feedback construction and honest split controls."""

    def test_feedback_from_prediction_errors(self):
        """Prediction error converts to bounded low-info probabilities."""
        chunks = [
            TextChunk(text="a", start_char=0, end_char=5, chunk_index=0),
            TextChunk(text="b", start_char=5, end_char=10, chunk_index=1),
        ]
        signals = feedback_from_prediction_errors(
            chunks=chunks,
            predicted_values=[60.0, 40.0],
            target_values=[50.0, 40.0],
            scale_min=0.0,
            scale_max=100.0,
            confidences=[0.9, 0.5],
            honest_role="boundary",
        )
        assert len(signals) == 2
        assert 0.0 <= signals[0].low_info_probability <= 1.0
        assert signals[0].metadata.get("honest_role") == "boundary"
        assert signals[1].low_info_probability == pytest.approx(0.0)

    def test_assign_honest_split_deterministic(self):
        """Honest split assignment is deterministic for a sample ID."""
        policy = HonestChunkingPolicy(enabled=True, boundary_fraction=0.4, split_seed=123)
        split1 = assign_honest_split("doc_abc", policy)
        split2 = assign_honest_split("doc_abc", policy)
        assert split1 in {policy.boundary_role, policy.evaluation_role}
        assert split1 == split2

    def test_adaptive_memory_honest_role_filtering(self):
        """Memory returns only role-appropriate signals under honesty."""
        memory = AdaptiveChunkMemory()
        policy = HonestChunkingPolicy(enabled=True, boundary_fraction=0.5)
        doc_id = "doc1"

        boundary_signal = ChunkFeedbackSignal(
            start_char=0,
            end_char=10,
            low_info_probability=0.2,
            confidence=1.0,
        )
        eval_signal = ChunkFeedbackSignal(
            start_char=10,
            end_char=20,
            low_info_probability=0.8,
            confidence=1.0,
        )
        memory.update_signals(doc_id, [boundary_signal], honest_role=policy.boundary_role)
        memory.update_signals(doc_id, [eval_signal], honest_role=policy.evaluation_role)

        chunking_signals = memory.get_signals_for_chunking(doc_id, honest_policy=policy)
        evaluation_signals = memory.get_signals_for_evaluation(doc_id, honest_policy=policy)

        assert len(chunking_signals) == 1
        assert len(evaluation_signals) == 1
        assert chunking_signals[0].metadata.get("honest_role") == policy.boundary_role
        assert evaluation_signals[0].metadata.get("honest_role") == policy.evaluation_role
