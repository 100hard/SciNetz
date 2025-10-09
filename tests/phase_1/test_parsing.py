from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.config import load_config
from backend.app.parsing.pipeline import ParseResult, ParsingPipeline

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
PDF_DIR = FIXTURES_DIR / "pdfs"
GOLDEN_DIR = FIXTURES_DIR / "golden" / "phase_1"


@pytest.fixture(scope="module")
def pipeline() -> ParsingPipeline:
    config = load_config()
    return ParsingPipeline(config=config)


def _load_golden(name: str) -> dict:
    with (GOLDEN_DIR / f"{name}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_docling_assets(result: ParseResult) -> None:
    """Skip tests when Docling models are unavailable in the environment."""

    for error in result.errors:
        lowered = error.lower()
        if "docling conversion error" in lowered and (
            "huggingface" in lowered
            or "localentrynotfounderror" in lowered
            or "cannot find the requested files in the local cache" in lowered
            or "403" in lowered
        ):
            pytest.skip(
                "Docling models unavailable – requires Hugging Face access for parsing tests."
            )


def test_parse_sample_transformer_matches_golden(tmp_path: Path, pipeline: ParsingPipeline) -> None:
    pdf_path = PDF_DIR / "sample_transformer.pdf"
    golden = _load_golden("sample_transformer")

    result = pipeline.parse_document(
        doc_id="sample_transformer",
        pdf_path=pdf_path,
        output_dir=tmp_path,
    )

    _require_docling_assets(result)
    assert not result.errors
    assert result.metadata.model_dump(exclude_none=False) == golden["metadata"]
    elements_dump = [element.model_dump() for element in result.elements]
    assert elements_dump == golden["elements"]


def test_parse_sample_graph_matches_golden(tmp_path: Path, pipeline: ParsingPipeline) -> None:
    pdf_path = PDF_DIR / "sample_graph.pdf"
    golden = _load_golden("sample_graph")

    result = pipeline.parse_document(
        doc_id="sample_graph",
        pdf_path=pdf_path,
        output_dir=tmp_path,
    )

    _require_docling_assets(result)
    assert not result.errors
    assert result.metadata.model_dump(exclude_none=False) == golden["metadata"]
    elements_dump = [element.model_dump() for element in result.elements]
    assert elements_dump == golden["elements"]


def test_parsed_elements_have_non_overlapping_offsets(tmp_path: Path, pipeline: ParsingPipeline) -> None:
    result = pipeline.parse_document(
        doc_id="sample_transformer",
        pdf_path=PDF_DIR / "sample_transformer.pdf",
        output_dir=tmp_path,
    )
    _require_docling_assets(result)
    last_end = 0
    for element in result.elements:
        assert element.start_char >= last_end
        assert element.end_char > element.start_char
        last_end = element.end_char


def test_malformed_pdf_returns_error(tmp_path: Path, pipeline: ParsingPipeline) -> None:
    result = pipeline.parse_document(
        doc_id="broken",
        pdf_path=PDF_DIR / "broken.pdf",
        output_dir=tmp_path,
    )
    assert result.errors
    assert not result.elements


def test_content_hash_stability(pipeline: ParsingPipeline, tmp_path: Path) -> None:
    first = pipeline.parse_document(
        doc_id="sample_graph",
        pdf_path=PDF_DIR / "sample_graph.pdf",
        output_dir=tmp_path,
    )
    _require_docling_assets(first)
    second = pipeline.parse_document(
        doc_id="sample_graph",
        pdf_path=PDF_DIR / "sample_graph.pdf",
        output_dir=tmp_path,
    )
    _require_docling_assets(second)
    first_hashes = [element.content_hash for element in first.elements]
    second_hashes = [element.content_hash for element in second.elements]
    assert first_hashes == second_hashes


def test_metadata_includes_title_and_year(tmp_path: Path, pipeline: ParsingPipeline) -> None:
    result = pipeline.parse_document(
        doc_id="sample_transformer",
        pdf_path=PDF_DIR / "sample_transformer.pdf",
        output_dir=tmp_path,
    )
    _require_docling_assets(result)
    assert result.metadata.title
    assert result.metadata.year == 2024
