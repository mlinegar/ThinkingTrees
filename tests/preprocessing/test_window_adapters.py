"""Tests for modality-agnostic axis-window adapters."""

from src.preprocessing.adaptive_windows import (
    AxisWindow,
    merge_adjacent_windows_by_embedding_drift,
)
from src.preprocessing.window_adapters import (
    SequenceItemWindowAdapter,
    TextCharWindowAdapter,
    TextPageWindowAdapter,
    TimeSegmentWindowAdapter,
    build_adaptive_windows_for_sample,
    build_window_adapter,
)


def test_text_char_window_adapter_materializes_span():
    adapter = TextCharWindowAdapter()
    sample = "abcdef"
    window = AxisWindow(start=1, end=4, unit="char")
    assert adapter.total_extent(sample) == 6
    assert adapter.materialize(sample, window) == "bcd"


def test_text_page_window_adapter_materializes_pages():
    adapter = TextPageWindowAdapter()
    sample = {"pages": ["p0", "p1", "p2", "p3"]}
    window = AxisWindow(start=1, end=3, unit="page")
    assert adapter.total_extent(sample) == 4
    assert adapter.materialize(sample, window) == "p1\n\np2"


def test_sequence_item_window_adapter_materializes_items():
    adapter = SequenceItemWindowAdapter()
    sample = {"items": ["a", "b", "c", "d"]}
    window = AxisWindow(start=1, end=4, unit="item")
    assert adapter.total_extent(sample) == 4
    assert adapter.materialize(sample, window) == "b\nc\nd"


def test_time_segment_window_adapter_materializes_overlap():
    adapter = TimeSegmentWindowAdapter(include_timestamps=False)
    sample = {
        "segments": [
            {"start": 0, "end": 1000, "text": "hello"},
            {"start": 1000, "end": 2000, "text": "world"},
            {"start": 2500, "end": 3000, "text": "tail"},
        ]
    }
    window = AxisWindow(start=500, end=2100, unit="ms")
    assert adapter.total_extent(sample) == 3000
    assert adapter.materialize(sample, window) == "hello\nworld"


def test_build_window_adapter_factory():
    assert isinstance(build_window_adapter("text_char"), TextCharWindowAdapter)
    assert isinstance(build_window_adapter("text_page"), TextPageWindowAdapter)
    assert isinstance(build_window_adapter("sequence_item"), SequenceItemWindowAdapter)
    assert isinstance(build_window_adapter("time_segment"), TimeSegmentWindowAdapter)


def test_build_adaptive_windows_for_sample_with_text():
    sample = "x" * 2000
    adapter = TextCharWindowAdapter()

    def score(payloads, _windows):
        # Keep behavior deterministic and simple for test.
        return [0.5 for _ in payloads]

    windows = build_adaptive_windows_for_sample(
        sample=sample,
        adapter=adapter,
        score_materialized_windows=score,
        coarse_window_size=800,
        fine_window_size=200,
        max_windows=12,
    )
    assert windows
    assert all(window.unit == "char" for window in windows)


def test_merge_adjacent_windows_by_embedding_drift_merges_low_drift():
    windows = [
        AxisWindow(start=0, end=100, unit="char"),
        AxisWindow(start=100, end=200, unit="char"),
        AxisWindow(start=200, end=300, unit="char"),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.999, 0.001],  # Very close to first window.
        [0.0, 1.0],      # Far from second window.
    ]

    merged = merge_adjacent_windows_by_embedding_drift(
        windows,
        embeddings,
        max_cosine_distance=0.01,
        max_merged_width=250,
    )
    assert len(merged) == 2
    assert merged[0].start == 0
    assert merged[0].end == 200
    assert merged[0].metadata.get("merged_window_count") == 2


def test_merge_adjacent_windows_by_embedding_drift_respects_width_cap():
    windows = [
        AxisWindow(start=0, end=120, unit="char"),
        AxisWindow(start=120, end=240, unit="char"),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.999, 0.001],
    ]

    merged = merge_adjacent_windows_by_embedding_drift(
        windows,
        embeddings,
        max_cosine_distance=0.02,
        max_merged_width=150,
    )
    assert len(merged) == 2
