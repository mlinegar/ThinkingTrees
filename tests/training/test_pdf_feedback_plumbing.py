"""Tests for PDF-oriented adaptive feedback span mapping helpers."""

from src.core.documents import DocumentSample
from src.preprocessing.adaptive_windows import AxisWindow
from src.training.run_pipeline import (
    _sample_text_content,
    _window_char_span_from_sample,
)


def test_sample_text_content_uses_pages_when_text_empty():
    sample = DocumentSample(
        doc_id="doc",
        text="",
        pages=["first", "second"],
    )
    assert _sample_text_content(sample) == "first\n\nsecond"


def test_window_char_span_maps_page_windows_using_metadata_ranges():
    sample = DocumentSample(
        doc_id="doc",
        text="a\n\nb\n\nc",
        pages=["a", "b", "c"],
        metadata={
            "page_char_ranges": [[0, 1], [3, 4], [6, 7]],
            "axis_char_ranges": {"page": [[0, 1], [3, 4], [6, 7]]},
        },
    )
    span = _window_char_span_from_sample(sample, AxisWindow(start=1, end=3, unit="page"))
    assert span == (3, 7)


def test_window_char_span_char_windows_clamp_to_text():
    sample = DocumentSample(doc_id="doc", text="abcdef")
    span = _window_char_span_from_sample(sample, AxisWindow(start=-10, end=99, unit="char"))
    assert span == (0, 6)


def test_window_char_span_returns_none_without_axis_ranges():
    sample = DocumentSample(doc_id="doc", text="abcdef")
    span = _window_char_span_from_sample(sample, AxisWindow(start=0, end=2, unit="page"))
    assert span is None
