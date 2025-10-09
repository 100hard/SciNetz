"""PDF parsing pipeline leveraging Docling output."""
from __future__ import annotations

import json
import logging
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from docling.document_converter import ConversionStatus, DocumentConverter
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
        self._converter = DocumentConverter()
        self._root_dir = Path(__file__).resolve().parents[2]
        self._alias_lookup = self._build_alias_lookup(config.parsing.section_aliases)

    @staticmethod
    def _build_alias_lookup(aliases: Dict[str, List[str]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for canonical, variations in aliases.items():
            canonical_lower = canonical.lower()
            mapping[canonical_lower] = canonical
            for variant in variations:
                mapping[variant.lower()] = canonical
        return mapping

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

        try:
            conversion = self._converter.convert(pdf_path)
        except Exception as exc:  # noqa: BLE001 - third-party may raise any error
            LOGGER.exception("Docling conversion failed for %s", pdf_path)
            result_errors.append(f"docling conversion error: {exc}")
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

