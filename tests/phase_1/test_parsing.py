from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import LayoutOptions, ThreadedPdfPipelineOptions
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.models.layout_model import LayoutModel
from docling.models.table_structure_model import TableStructureModel

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
            or "missing safe tensors file" in lowered
        ):
            pytest.skip(
                "Docling models unavailable - requires Hugging Face access for parsing tests."
            )

def test_parsing_pipeline_configures_threaded_pipeline(monkeypatch) -> None:
    captured: Dict[str, object] = {}
    bootstrap: Dict[str, Path] = {}

    class FakeConverter:
        def __init__(self, *_args, **kwargs) -> None:
            captured["format_options"] = kwargs.get("format_options")

        def initialize_pipeline(self, doc_format) -> None:
            captured["warmup"] = doc_format

        def convert(self, *_args, **_kwargs):
            raise AssertionError("convert should not be invoked during configuration test")

    monkeypatch.setattr("backend.app.parsing.pipeline.DocumentConverter", FakeConverter)
    monkeypatch.setattr(
        ParsingPipeline,
        "_bootstrap_rapidocr_models",
        lambda self, cache_dir: bootstrap.setdefault("dir", cache_dir),
        raising=False,
    )
    config = load_config()
    pipeline = ParsingPipeline(config=config)
    format_options = captured.get("format_options")
    assert isinstance(format_options, dict)
    assert InputFormat.PDF in format_options
    pdf_option = format_options[InputFormat.PDF]
    assert pdf_option.pipeline_cls is ThreadedStandardPdfPipeline
    assert isinstance(pdf_option.pipeline_options, ThreadedPdfPipelineOptions)
    assert pdf_option.pipeline_options.ocr_options.backend == config.parsing.rapidocr.backend
    assert (
        pdf_option.pipeline_options.accelerator_options.device
        == config.parsing.accelerator.device
    )
    assert captured.get("warmup") == InputFormat.PDF
    assert bootstrap.get("dir") == pipeline._rapidocr_cache_dir




def test_parsing_pipeline_aligns_rapidocr_cache(monkeypatch, tmp_path: Path) -> None:
    class FakeConverter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def initialize_pipeline(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op
            return None

    monkeypatch.setattr("backend.app.parsing.pipeline.DocumentConverter", FakeConverter)
    config = load_config()
    docling_dir = tmp_path / "docling-artifacts"
    rapid_cache_dir = tmp_path / "rapid-cache"
    updated_config = config.model_copy(
        update={
            "parsing": config.parsing.model_copy(
                update={
                    "docling_artifacts_path": str(docling_dir),
                    "rapidocr": config.parsing.rapidocr.model_copy(
                        update={
                            "model_cache_dir": str(rapid_cache_dir),
                            "warmup_on_startup": False,
                        }
                    ),
                }
            )
        }
    )
    pipeline = ParsingPipeline(config=updated_config)
    rapid_home = Path(os.environ["RAPIDOCR_HOME"]).resolve()
    docling_rapid_dir = docling_dir / "RapidOcr"
    assert rapid_home == pipeline._rapidocr_cache_dir.resolve()
    assert docling_rapid_dir.exists()
    assert docling_rapid_dir.resolve() == pipeline._rapidocr_cache_dir.resolve()


def test_parsing_pipeline_downloads_layout_model_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeConverter:
        def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - test stub
            return

        def initialize_pipeline(self, *_args, **_kwargs) -> None:  # pragma: no cover - test stub
            return None

        def convert(self, *_args, **_kwargs) -> None:  # pragma: no cover - test stub
            raise AssertionError("convert should not be invoked during download bootstrap test")

    download_calls = {"layout": 0, "table": 0}

    def fake_download_layout_models(*, local_dir, layout_model_config, **_kwargs):
        download_calls["layout"] += 1
        repo_dir = Path(local_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        target = repo_dir / (layout_model_config.model_path or "model.safetensors")
        target.touch()
        return repo_dir

    def fake_download_table_models(*, local_dir, **_kwargs):
        download_calls["table"] += 1
        repo_dir = Path(local_dir)
        target_dir = repo_dir / TableStructureModel._model_path / "accurate"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "tm_config.json").write_text("{}", encoding="utf-8")
        return repo_dir

    monkeypatch.setattr("backend.app.parsing.pipeline.DocumentConverter", FakeConverter)
    monkeypatch.setattr(
        LayoutModel,
        "download_models",
        staticmethod(fake_download_layout_models),
    )
    monkeypatch.setattr(
        TableStructureModel,
        "download_models",
        staticmethod(fake_download_table_models),
    )
    monkeypatch.setattr(
        ParsingPipeline,
        "_bootstrap_rapidocr_models",
        lambda self, cache_dir: None,
        raising=False,
    )
    monkeypatch.setattr(
        ParsingPipeline,
        "_warmup_pdf_pipeline",
        lambda self: None,
        raising=False,
    )

    config = load_config()
    docling_dir = tmp_path / "docling-artifacts"
    updated_config = config.model_copy(
        update={
            "parsing": config.parsing.model_copy(
                update={
                    "docling_artifacts_path": str(docling_dir),
                    "rapidocr": config.parsing.rapidocr.model_copy(
                        update={
                            "model_cache_dir": str(tmp_path / "rapid-cache"),
                            "warmup_on_startup": False,
                        }
                    ),
                }
            )
        }
    )
    ParsingPipeline(config=updated_config)
    assert download_calls["layout"] == 1
    assert download_calls["table"] == 1
    layout_spec = LayoutOptions().model_spec
    expected_root = docling_dir / layout_spec.model_repo_folder
    assert expected_root.exists()
    table_root = (
        docling_dir
        / TableStructureModel._model_repo_folder
        / TableStructureModel._model_path
    )
    assert (table_root / "accurate" / "tm_config.json").exists()

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
