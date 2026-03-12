"""Tests for PDF dataset plugin."""

from pathlib import Path

from src.datasets.pdf import PDFDataset
from src.parsers.pdf_text import ParsedPDFDocument


class DummyPDFParser:
    """Deterministic parser stub for dataset tests."""

    def parse_file(self, path: Path) -> ParsedPDFDocument:
        name = path.stem
        if name == "empty":
            return ParsedPDFDocument(
                text="",
                pages=[],
                page_char_ranges=[],
                backend="dummy",
                metadata={"source_path": str(path), "page_count": 0},
            )
        if name == "routable_empty":
            return ParsedPDFDocument(
                text="",
                pages=[""],
                page_char_ranges=[(0, 0)],
                backend="dummy",
                metadata={
                    "source_path": str(path),
                    "page_count": 1,
                    "parser_feedback": {
                        "axis_hints": [
                            {
                                "axis_unit": "page",
                                "start": 0,
                                "end": 1,
                                "action": "ocr_first_then_vision_embedding",
                                "recommended_processors": ["ocr", "vision_embedding"],
                                "source": "parser:pdf_needs_ocr",
                            }
                        ]
                    },
                },
            )
        pages = [f"{name}-p1", f"{name}-p2"]
        text = f"{pages[0]}\n\n{pages[1]}"
        page_ranges = [(0, len(pages[0])), (len(pages[0]) + 2, len(text))]
        return ParsedPDFDocument(
            text=text,
            pages=pages,
            page_char_ranges=page_ranges,
            backend="dummy",
            metadata={"source_path": str(path), "page_count": 2},
        )


def test_pdf_dataset_loads_directory_with_page_metadata(tmp_path: Path):
    (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "doc2.pdf").write_bytes(b"%PDF-1.4\n")

    dataset = PDFDataset(path=str(tmp_path), recursive=False, require_text=True)
    samples = dataset.load_samples(parser=DummyPDFParser(), shuffle=False)

    assert len(samples) == 2
    sample = samples[0]
    assert sample.pages is not None and len(sample.pages) == 2
    assert sample.metadata.get("parser_backend") == "dummy"
    assert sample.metadata.get("axis_char_ranges", {}).get("page") == sample.metadata.get("page_char_ranges")


def test_pdf_dataset_filters_empty_text_when_required(tmp_path: Path):
    (tmp_path / "empty.pdf").write_bytes(b"%PDF-1.4\n")

    dataset = PDFDataset(path=str(tmp_path), recursive=False, require_text=True)
    samples = dataset.load_samples(parser=DummyPDFParser(), shuffle=False)

    assert samples == []


def test_pdf_dataset_single_file_doc_id_is_stem(tmp_path: Path):
    pdf_path = tmp_path / "manifesto_001.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    dataset = PDFDataset(path=str(pdf_path), require_text=True)
    samples = dataset.load_samples(parser=DummyPDFParser(), shuffle=False)

    assert len(samples) == 1
    assert samples[0].doc_id == "manifesto_001"


def test_pdf_dataset_keeps_routable_empty_text_sample(tmp_path: Path):
    (tmp_path / "routable_empty.pdf").write_bytes(b"%PDF-1.4\n")

    dataset = PDFDataset(path=str(tmp_path), recursive=False, require_text=True)
    samples = dataset.load_samples(parser=DummyPDFParser(), shuffle=False)

    assert len(samples) == 1
    assert samples[0].doc_id == "routable_empty"
