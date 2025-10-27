"""PDF parsing pipeline leveraging Docling output."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    ThreadedPdfPipelineOptions,
    LayoutOptions,
)
from docling.datamodel.document import ConversionResult
from docling.document_converter import (
    ConversionStatus,
    DocumentConverter,
    FormatOption,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from backend.app.config import AppConfig
from backend.app.contracts import PaperMetadata, ParsedElement

LOGGER = logging.getLogger(__name__)
HF_DISABLE_SYMLINKS_ENV = "HF_HUB_DISABLE_SYMLINKS"
_DEFAULT_SECTION = "_auto_"
_DOI_PATTERN = re.compile(r"10\.\d{4,}/\S+", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"20[1-2][0-9]")


class ParseResult(BaseModel):
    """Result produced by the parsing pipeline."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., min_length=1)
    metadata: PaperMetadata
    elements: List[ParsedElement] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    output_path: Optional[Path] = None


class ParsingPipeline:
    """Coordinate Docling conversion and downstream normalization."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the parsing pipeline.

        Args:
            config: Loaded application configuration.
        """
        self._config = config
        os.environ.setdefault(HF_DISABLE_SYMLINKS_ENV, "1")
        self._root_dir = Path(__file__).resolve().parents[2]
        self._artifacts_path = self._resolve_storage_path(config.parsing.docling_artifacts_path)
        self._rapidocr_cache_dir = self._prepare_rapidocr_cache(
            config.parsing.rapidocr.model_cache_dir
        )
        self._ensure_docling_layout_assets()
        os.environ["RAPIDOCR_HOME"] = str(self._rapidocr_cache_dir)
        self._converter_with_ocr = self._build_document_converter(use_ocr=True)
        self._converter_without_ocr: Optional[DocumentConverter] = None
        self._converter = self._converter_with_ocr
        self._ocr_available = True
        self._alias_lookup = self._build_alias_lookup(config.parsing.section_aliases)
        if config.parsing.rapidocr.warmup_on_startup:
            self._warmup_pdf_pipeline()

    @staticmethod
    def _build_alias_lookup(aliases: Dict[str, List[str]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for canonical, variations in aliases.items():
            canonical_lower = canonical.lower()
            mapping[canonical_lower] = canonical
            for variant in variations:
                mapping[variant.lower()] = canonical
        return mapping

    def _resolve_storage_path(self, path_value: str) -> Path:
        """Resolve a storage path relative to the repository root when needed.

        Args:
            path_value: Path string provided in configuration.

        Returns:
            Path: Absolute path guaranteed to exist on disk.
        """

        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (self._root_dir / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _resolve_optional_path(self, path_value: Optional[str]) -> Optional[Path]:
        """Resolve an optional storage path if provided in configuration.

        Args:
            path_value: Optional path string from configuration.

        Returns:
            Optional[Path]: Resolved path when supplied, otherwise None.
        """

        if not path_value:
            return None
        return self._resolve_storage_path(path_value)

    def _prepare_rapidocr_cache(self, configured_path: Optional[str]) -> Path:
        """Ensure Docling RapidOCR artifacts resolve to the desired cache directory.

        Args:
            configured_path: Optional path configured for RapidOCR model caching.

        Returns:
            Path: Directory that should be used as RAPIDOCR_HOME and by Docling.
        """

        target_dir = self._artifacts_path / "RapidOcr"
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not configured_path:
            target_dir.mkdir(parents=True, exist_ok=True)
            self._bootstrap_rapidocr_models(target_dir)
            return target_dir

        cache_dir = self._resolve_storage_path(configured_path)
        if cache_dir.resolve() == target_dir.resolve():
            self._bootstrap_rapidocr_models(target_dir)
            return target_dir

        cache_dir.mkdir(parents=True, exist_ok=True)
        if self._ensure_directory_alias(target_dir, cache_dir):
            LOGGER.info("RapidOCR cache directory set to %s", cache_dir)
            self._bootstrap_rapidocr_models(cache_dir)
            return cache_dir

        LOGGER.warning(
            "Falling back to Docling artifacts directory for RapidOCR cache due to linking failure"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap_rapidocr_models(target_dir)
        return target_dir

    def _ensure_directory_alias(self, alias: Path, target: Path) -> bool:
        """Link alias directory to target if possible, mirroring contents when needed."""

        try:
            if alias.exists() or alias.is_symlink():
                if alias.is_symlink():
                    try:
                        if alias.resolve(strict=True) == target.resolve():
                            return True
                    except FileNotFoundError:
                        alias.unlink(missing_ok=True)
                    else:
                        alias.unlink()
                else:
                    self._mirror_directory(alias, target)
                    shutil.rmtree(alias)
            alias.symlink_to(target, target_is_directory=True)
            return True
        except (OSError, NotImplementedError) as exc:
            LOGGER.debug(
                "Unable to create RapidOCR cache symlink %s -> %s: %s", alias, target, exc
            )
            return False

    @staticmethod
    def _mirror_directory(source: Path, destination: Path) -> None:
        """Copy directory contents from source into destination without clobbering existing files."""

        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target_path = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path)

    def _bootstrap_rapidocr_models(self, cache_dir: Path) -> None:
        """Seed RapidOCR cache directory with bundled models when available."""

        known_assets = {
            Path("torch/PP-OCRv4/det/ch_PP-OCRv4_det_infer.pth"): "ch_PP-OCRv4_det_infer.pth",
            Path("torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_infer.pth"): "ch_ptocr_mobile_v2.0_cls_infer.pth",
            Path("torch/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.pth"): "ch_PP-OCRv4_rec_infer.pth",
            Path("onnxruntime/PP-OCRv4/det/ch_PP-OCRv4_det_infer.onnx"): "ch_PP-OCRv4_det_infer.onnx",
            Path("onnxruntime/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"): "ch_ppocr_mobile_v2.0_cls_infer.onnx",
            Path("onnxruntime/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.onnx"): "ch_PP-OCRv4_rec_infer.onnx",
            Path("paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt"): "ppocr_keys_v1.txt",
            Path("paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocrv5_dict.txt"): "ppocrv5_dict.txt",
        }
        try:
            import rapidocr  # type: ignore
        except ImportError:
            LOGGER.debug("RapidOCR package not available for cache bootstrap")
            return

        models_dir = Path(rapidocr.__file__).resolve().parent / "models"
        if not models_dir.exists():
            LOGGER.debug("RapidOCR models directory missing at %s", models_dir)
            return

        for relative_path, filename in known_assets.items():
            destination = cache_dir / relative_path
            if destination.exists():
                continue
            source = models_dir / filename
            if not source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                continue
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            except OSError as exc:
                LOGGER.debug(
                    "Failed to copy RapidOCR asset %s -> %s: %s", source, destination, exc
                )

    def _ensure_docling_layout_assets(self) -> None:
        """Download Docling model weights into the configured artifacts directory."""

        layout_options = LayoutOptions()
        layout_spec = layout_options.model_spec
        expected_root = self._artifacts_path / layout_spec.model_repo_folder
        if layout_spec.model_path:
            expected_path = expected_root / layout_spec.model_path
        else:
            expected_path = expected_root
        if expected_path.exists():
            return

        try:
            from docling.models.layout_model import LayoutModel  # noqa: WPS433 - runtime import
        except ImportError as exc:
            LOGGER.warning("Docling layout model unavailable: %s", exc)
            return

        try:
            LayoutModel.download_models(
                local_dir=expected_root,
                progress=False,
                layout_model_config=layout_spec,
            )
        except Exception as exc:  # noqa: BLE001 - third-party errors for HF download
            LOGGER.warning("Failed to download Docling layout model assets: %s", exc)
        else:
            if expected_path.exists():
                LOGGER.info("Cached Docling layout model assets at %s", expected_path)
        try:
            from docling.models.table_structure_model import (  # noqa: WPS433 - runtime import
                TableStructureModel,
            )
        except ImportError as exc:
            LOGGER.warning("Docling table structure model unavailable: %s", exc)
            return

        table_repo = self._artifacts_path / TableStructureModel._model_repo_folder
        table_expected = table_repo / TableStructureModel._model_path
        if table_expected.exists():
            return

        try:
            TableStructureModel.download_models(
                local_dir=table_repo,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - third-party errors for HF download
            LOGGER.warning("Failed to download Docling table structure assets: %s", exc)
        else:
            if table_expected.exists():
                LOGGER.info("Cached Docling table structure assets at %s", table_expected)
    def _should_retry_without_ocr(self, exc: Exception) -> bool:
        """Identify RapidOCR-related failures that merit a no-OCR retry."""

        message = str(exc).lower()
        if isinstance(exc, (FileNotFoundError, ImportError, ModuleNotFoundError)):
            return "rapidocr" in message or "pp-ocrv4" in message
        return False

    def _retry_without_ocr(
        self, pdf_path: Path, result_errors: List[str]
    ) -> Optional[ConversionResult]:
        """Retry Docling conversion without OCR enabled."""

        if self._converter_without_ocr is None:
            self._converter_without_ocr = self._build_document_converter(use_ocr=False)
        self._converter = self._converter_without_ocr
        self._ocr_available = False
        try:
            return self._converter.convert(pdf_path)
        except Exception as fallback_exc:  # noqa: BLE001 - third-party exceptions
            LOGGER.exception(
                "Docling conversion failed without OCR for %s", pdf_path, exc_info=fallback_exc
            )
            result_errors.append(self._format_docling_error(fallback_exc))
            return None

    def _build_document_converter(self, *, use_ocr: bool) -> DocumentConverter:
        """Construct a Docling document converter with configured PDF options.

        Returns:
            DocumentConverter: Converter instance wired to the desired OCR backend.
        """

        parsing_cfg = self._config.parsing
        accelerator = AcceleratorOptions(
            device=parsing_cfg.accelerator.device,
            num_threads=parsing_cfg.accelerator.num_threads,
        )
        rapid_cfg = parsing_cfg.rapidocr
        rapid_kwargs = {
            "backend": rapid_cfg.backend,
            "text_score": rapid_cfg.text_score,
            "use_det": rapid_cfg.use_det,
            "use_cls": rapid_cfg.use_cls,
            "use_rec": rapid_cfg.use_rec,
            "det_model_path": rapid_cfg.det_model_path,
            "cls_model_path": rapid_cfg.cls_model_path,
            "rec_model_path": rapid_cfg.rec_model_path,
            "font_path": rapid_cfg.font_path,
        }
        rapid_options = RapidOcrOptions(
            **{key: value for key, value in rapid_kwargs.items() if value is not None}
        )
        pdf_options = self._build_pdf_options(
            threaded_cfg=parsing_cfg.threaded_pdf,
            rapid_options=rapid_options,
            accelerator=accelerator,
            use_ocr=use_ocr,
        )
        pipeline_cls = (
            ThreadedStandardPdfPipeline
            if parsing_cfg.threaded_pdf.enabled
            else StandardPdfPipeline
        )
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=pipeline_cls,
                backend=DoclingParseV4DocumentBackend,
                pipeline_options=pdf_options,
            )
        }
        return DocumentConverter(format_options=format_options)

    def _build_pdf_options(
        self,
        *,
        threaded_cfg: "ThreadedPDFConfig",
        rapid_options: RapidOcrOptions,
        accelerator: AcceleratorOptions,
        use_ocr: bool,
    ) -> PdfPipelineOptions:
        """Create PDF pipeline options respecting threaded configuration.

        Args:
            threaded_cfg: Threaded pipeline tuning parameters.
            rapid_options: RapidOCR configuration converted to Docling options.
            accelerator: Accelerator configuration for Docling models.

        Returns:
            PdfPipelineOptions: Options instance compatible with the selected pipeline.
        """

        if threaded_cfg.enabled:
            return ThreadedPdfPipelineOptions(
                artifacts_path=self._artifacts_path,
                accelerator_options=accelerator,
                ocr_options=rapid_options,
                do_ocr=use_ocr,
                ocr_batch_size=threaded_cfg.ocr_batch_size,
                layout_batch_size=threaded_cfg.layout_batch_size,
                table_batch_size=threaded_cfg.table_batch_size,
                batch_timeout_seconds=threaded_cfg.batch_timeout_seconds,
                queue_max_size=threaded_cfg.queue_max_size,
            )
        return PdfPipelineOptions(
            artifacts_path=self._artifacts_path,
            accelerator_options=accelerator,
            ocr_options=rapid_options,
            do_ocr=use_ocr,
        )

    def _warmup_pdf_pipeline(self) -> None:
        """Eagerly initialize the PDF pipeline so OCR assets are ready to use."""

        try:
            self._converter_with_ocr.initialize_pipeline(InputFormat.PDF)
        except Exception as exc:  # noqa: BLE001 - defensive warmup
            LOGGER.warning("RapidOCR warmup failed: %s", exc)
        else:
            LOGGER.info("RapidOCR pipeline pre-initialized successfully")

    def parse_document(
        self,
        doc_id: str,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
    ) -> ParseResult:
        """Parse a PDF and emit normalized elements.

        Args:
            doc_id: Document identifier used for downstream storage.
            pdf_path: Path to the PDF file on disk.
            output_dir: Optional override for where JSONL outputs are written.

        Returns:
            ParseResult: Structured parsing result with metadata, elements, and errors.
        """
        result_errors: List[str] = []
        metadata = PaperMetadata(doc_id=doc_id)
        elements: List[ParsedElement] = []
        output_path: Optional[Path] = None

        if self._ocr_available:
            self._converter = self._converter_with_ocr
        elif self._converter_without_ocr is not None:
            self._converter = self._converter_without_ocr

        try:
            conversion = self._converter.convert(pdf_path)
        except Exception as exc:  # noqa: BLE001 - third-party may raise any error
            if self._should_retry_without_ocr(exc):
                LOGGER.warning(
                    "RapidOCR unavailable for %s; retrying conversion without OCR.", pdf_path
                )
                conversion = self._retry_without_ocr(pdf_path, result_errors)
                if conversion is None:
                    return ParseResult(
                        doc_id=doc_id,
                        metadata=metadata,
                        elements=elements,
                        errors=result_errors,
                        output_path=output_path,
                    )
            else:
                LOGGER.exception("Docling conversion failed for %s", pdf_path)
                result_errors.append(self._format_docling_error(exc))
                return ParseResult(
                    doc_id=doc_id,
                    metadata=metadata,
                    elements=elements,
                    errors=result_errors,
                    output_path=output_path,
                )

        if conversion.status != ConversionStatus.SUCCESS:
            LOGGER.error(
                "Docling conversion returned status %s for %s", conversion.status, pdf_path
            )
            result_errors.append(f"docling status: {conversion.status}")
            return ParseResult(
                doc_id=doc_id,
                metadata=metadata,
                elements=elements,
                errors=result_errors,
                output_path=output_path,
            )

        export_dict = conversion.document.export_to_dict()
        text_items = self._filter_text_items(export_dict.get("texts", []))

        metadata = self._extract_metadata(doc_id, text_items)
        elements = self._build_elements(doc_id, text_items)

        try:
            output_path = self._write_jsonl(doc_id, metadata, elements, output_dir)
        except OSError as exc:
            LOGGER.exception("Failed to write parsed output for %s", doc_id)
            result_errors.append(f"write error: {exc}")

        return ParseResult(
            doc_id=doc_id,
            metadata=metadata,
            elements=elements,
            errors=result_errors,
            output_path=output_path,
        )

    @staticmethod
    def _format_docling_error(exc: Exception) -> str:
        """Create a normalized error message for Docling conversions.

        Args:
            exc: Exception raised during Docling conversion.

        Returns:
            str: Normalized error string for downstream test skips.
        """

        message = str(exc)
        lowercase = message.lower()
        network_indicators = (
            isinstance(exc, URLError)
            or "name or service not known" in lowercase
            or "temporary failure in name resolution" in lowercase
            or "getaddrinfo failed" in lowercase
        )
        if network_indicators:
            return f"docling conversion error: huggingface assets unavailable ({message})"
        return f"docling conversion error: {message}"

    def _filter_text_items(self, items: Iterable[dict]) -> List[dict]:
        """Filter Docling text items to the subset required for parsing.

        Args:
            items: Raw text items from Docling export.

        Returns:
            List[dict]: Filtered items retaining ordering and provenance.
        """
        filtered: List[dict] = []
        for item in items:
            text = (item.get("orig") or item.get("text") or "").strip()
            if not text:
                continue
            if item.get("label") not in {"text", "section_header"}:
                continue
            filtered.append({**item, "text": text})
        return filtered

    def _build_elements(self, doc_id: str, text_items: List[dict]) -> List[ParsedElement]:
        """Transform text items into immutable ParsedElement objects."""
        elements: List[ParsedElement] = []
        cursor = 0
        current_section = _DEFAULT_SECTION
        for idx, item in enumerate(text_items):
            label = item.get("label")
            text = item["text"]
            if label == "section_header":
                current_section = self._normalize_section(text)
                continue
            section_name = current_section
            content_hash = sha256(text.encode("utf-8")).hexdigest()
            start_char = cursor
            end_char = start_char + len(text)
            cursor = end_char
            element = ParsedElement(
                doc_id=doc_id,
                element_id=f"{doc_id}:{len(elements)}",
                section=section_name,
                content=text,
                content_hash=content_hash,
                start_char=start_char,
                end_char=end_char,
            )
            elements.append(element)
        return elements

    def _normalize_section(self, raw: str) -> str:
        """Normalize section headings using configured aliases and fuzzy matching."""
        cleaned = raw.strip()
        if not cleaned:
            return _DEFAULT_SECTION
        lowered = cleaned.lower()
        if lowered in self._alias_lookup:
            return self._alias_lookup[lowered]

        best_match: Tuple[Optional[str], float] = (None, 0.0)
        threshold = self._config.parsing.section_fuzzy_threshold * 100
        for alias, canonical in self._alias_lookup.items():
            score = fuzz.ratio(lowered, alias)
            if score >= threshold and score > best_match[1]:
                best_match = (canonical, score)
        return best_match[0] or _DEFAULT_SECTION

    def _extract_metadata(self, doc_id: str, text_items: List[dict]) -> PaperMetadata:
        """Derive PaperMetadata from parsed text items."""
        metadata = PaperMetadata(doc_id=doc_id)
        pages_limit = self._config.parsing.metadata_max_pages
        page_texts = [item for item in text_items if self._page_no(item) <= pages_limit]

        metadata = metadata.model_copy(update={
            "title": self._find_title(text_items),
            "authors": self._find_authors(page_texts),
            "year": self._find_year(page_texts),
            "venue": self._find_venue(page_texts),
            "doi": self._find_doi(page_texts),
        })
        return metadata

    @staticmethod
    def _page_no(item: dict) -> int:
        prov = item.get("prov") or []
        if prov and isinstance(prov, list):
            page = prov[0].get("page_no")
            if isinstance(page, int):
                return page
        return 1

    def _find_title(self, items: List[dict]) -> Optional[str]:
        for item in items:
            if item.get("label") == "section_header":
                normalized = self._normalize_section(item["text"])
                if normalized not in {"Abstract"}:
                    return item["text"].strip()
        return None

    def _find_authors(self, items: List[dict]) -> List[str]:
        names: List[str] = []
        collecting = False
        for item in items:
            if item.get("label") != "text":
                continue
            text = item["text"].strip()
            lowered = text.lower()
            if lowered.startswith("published") or lowered.startswith("doi"):
                if collecting:
                    break
                continue
            candidate = self._strip_after_known_tokens(text)
            new_names = [
                cleaned
                for part in re.split(r",| and ", candidate)
                if (cleaned := self._clean_author_candidate(part))
                and self._looks_like_name(cleaned)
            ]
            if new_names:
                collecting = True
                for name in new_names:
                    if name not in names:
                        names.append(name)
                continue
            if collecting:
                break
        return names

    def _strip_after_known_tokens(self, text: str) -> str:
        lowered = text.lower()
        cutoff = len(text)
        for venue in self._config.parsing.metadata_known_venues:
            idx = lowered.find(venue.lower())
            if idx != -1:
                cutoff = min(cutoff, idx)
        for token in ["proceedings", "conference", "published", "doi:"]:
            idx = lowered.find(token)
            if idx != -1:
                cutoff = min(cutoff, idx)
        # Stop before trailing years or numeric footnotes to avoid polluting author names.
        digit_match = re.search(r"\d", text)
        if digit_match:
            cutoff = min(cutoff, digit_match.start())
        return text[:cutoff].strip()

    @staticmethod
    def _clean_author_candidate(candidate: str) -> str:
        tokens = candidate.strip().split()
        cleaned: List[str] = []
        for token in tokens:
            normalized = token.strip(",")
            if not normalized:
                continue
            if normalized.isdigit():
                break
            if normalized.isupper() and len(normalized) > 1:
                break
            cleaned.append(normalized)
        return " ".join(cleaned)

    @staticmethod
    def _looks_like_name(candidate: str) -> bool:
        pattern = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$")
        return bool(pattern.match(candidate.strip()))

    def _find_year(self, items: List[dict]) -> Optional[int]:
        for item in items:
            match = _YEAR_PATTERN.search(item["text"])
            if match:
                year = int(match.group(0))
                if 2015 <= year <= 2025:
                    return year
        return None

    def _find_venue(self, items: List[dict]) -> Optional[str]:
        known = self._config.parsing.metadata_known_venues
        for item in items:
            lowered = item["text"].lower()
            for venue in known:
                if venue.lower() in lowered:
                    return venue
        return None

    def _find_doi(self, items: List[dict]) -> Optional[str]:
        for item in items:
            match = _DOI_PATTERN.search(item["text"])
            if match:
                return match.group(0)
        return None

    def _write_jsonl(
        self,
        doc_id: str,
        metadata: PaperMetadata,
        elements: List[ParsedElement],
        output_dir: Optional[Path],
    ) -> Path:
        """Persist results to JSONL for downstream phases."""
        target_dir = output_dir or (self._root_dir / "data" / "parsed")
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{doc_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"type": "paper_metadata", "data": metadata.model_dump(exclude_none=True)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            for element in elements:
                handle.write(
                    json.dumps(
                        {"type": "parsed_element", "data": element.model_dump()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return output_path

