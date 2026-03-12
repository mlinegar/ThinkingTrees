"""Tests for PDF text parser utilities."""

from pathlib import Path

import pytest

from src.parsers.pdf_text import (
    PDFTextParser,
    _build_page_char_ranges,
    _build_page_parser_feedback,
)


def test_build_page_char_ranges_tracks_offsets():
    text, ranges = _build_page_char_ranges(
        ["alpha", "beta", "gamma"],
        page_joiner="\n\n",
    )
    assert text == "alpha\n\nbeta\n\ngamma"
    assert ranges == [(0, 5), (7, 11), (13, 18)]


def test_pdf_parser_uses_fallback_backend(tmp_path: Path):
    class DummyParser(PDFTextParser):
        def _extract_pages(self, path: Path, *, backend: str):  # type: ignore[override]
            if backend == "broken":
                raise RuntimeError("backend failed")
            return ["Page A", "Page B"]

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parser = DummyParser(backends=["broken", "ok"], page_joiner="\n\n")
    parsed = parser.parse_file(pdf_path)

    assert parsed.backend == "ok"
    assert parsed.pages == ["Page A", "Page B"]
    assert parsed.page_char_ranges == [(0, 6), (8, 14)]
    assert parsed.text == "Page A\n\nPage B"


def test_pdf_parser_raises_when_all_backends_fail(tmp_path: Path):
    class FailingParser(PDFTextParser):
        def _extract_pages(self, path: Path, *, backend: str):  # type: ignore[override]
            raise RuntimeError("always fails")

    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parser = FailingParser(backends=["one", "two"])
    with pytest.raises(RuntimeError):
        parser.parse_file(pdf_path)


def test_page_parser_feedback_prefers_ocr_or_visual_routing():
    pages = ["", "caption", "normal body text"]
    _, ranges = _build_page_char_ranges(pages, page_joiner="\n\n")
    feedback = _build_page_parser_feedback(
        pages,
        ranges,
        page_image_counts=[2, 1, 0],
        min_text_chars_for_visual_support=20,
    )

    assert feedback["strategy"] == "extraction_quality_routing"
    hints = feedback["axis_hints"]
    assert len(hints) == 2

    first = hints[0]
    assert first["source"] == "parser:pdf_needs_ocr"
    assert first["low_info_probability"] == 0.0
    assert first["noise_probability"] >= 0.9
    assert "ocr" in first["recommended_processors"]

    second = hints[1]
    assert second["source"] == "parser:pdf_visual_content"
    assert second["low_info_probability"] == 0.0
    assert "vision_embedding" in second["recommended_processors"]


def test_page_parser_feedback_includes_page_asset_refs():
    pages = ["", "caption"]
    _, ranges = _build_page_char_ranges(pages, page_joiner="\n\n")
    feedback = _build_page_parser_feedback(
        pages,
        ranges,
        page_image_counts=[1, 1],
        page_assets=[
            {"page_uri": "pdf://doc.pdf#page=1", "image_refs": ["xref:1"]},
            {"page_uri": "pdf://doc.pdf#page=2", "image_refs": ["xref:2"]},
        ],
        min_text_chars_for_visual_support=20,
    )
    hints = feedback["axis_hints"]
    assert hints[0]["page_asset_ref"] == "pdf://doc.pdf#page=1"
    assert hints[0]["page_image_refs"] == ["xref:1"]
