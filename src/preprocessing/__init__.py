"""Document preprocessing and chunking."""

from src.preprocessing.chunker import (
    TextChunk,
    Chunker,
    AdaptiveChunkingConfig,
    AdaptiveChunkMemory,
    ChunkFeedbackSignal,
    HonestChunkingPolicy,
    assign_honest_split,
    feedback_from_prediction_errors,
    chunk_for_ops_token_budget,
    chunk_for_ops,
    chunk_for_ops_adaptive,
    chunk_text,
)

from src.preprocessing.adaptive_windows import (
    AxisWindow,
    uniform_axis_windows,
    adaptive_refine_windows,
    merge_adjacent_windows_by_embedding_drift,
)
from src.preprocessing.window_adapters import (
    AxisWindowAdapter,
    TextCharWindowAdapter,
    TextPageWindowAdapter,
    SequenceItemWindowAdapter,
    TimeSegmentWindowAdapter,
    build_window_adapter,
    build_adaptive_windows_for_sample,
)

from src.preprocessing.tokenizer import (
    TokenCounter,
    count_tokens,
    get_default_max_tokens,
)

__all__ = [
    # Chunking
    "TextChunk",
    "Chunker",
    "AdaptiveChunkingConfig",
    "AdaptiveChunkMemory",
    "ChunkFeedbackSignal",
    "HonestChunkingPolicy",
    "assign_honest_split",
    "feedback_from_prediction_errors",
    "chunk_for_ops_token_budget",
    "chunk_for_ops",
    "chunk_for_ops_adaptive",
    "chunk_text",
    "AxisWindow",
    "uniform_axis_windows",
    "adaptive_refine_windows",
    "merge_adjacent_windows_by_embedding_drift",
    "AxisWindowAdapter",
    "TextCharWindowAdapter",
    "TextPageWindowAdapter",
    "SequenceItemWindowAdapter",
    "TimeSegmentWindowAdapter",
    "build_window_adapter",
    "build_adaptive_windows_for_sample",
    # Token counting
    "TokenCounter",
    "count_tokens",
    "get_default_max_tokens",
]
