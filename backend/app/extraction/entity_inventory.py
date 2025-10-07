"""Entity inventory builder for extraction guidance."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from backend.app.config import AppConfig
from backend.app.contracts import ParsedElement

LOGGER = logging.getLogger(__name__)

_SPACY_MODEL = "en_core_web_sm"
_SCISPACY_MODEL = "en_core_sci_md"


@dataclass
class _Candidate:
    """Candidate entity record for ranking."""

    text: str
    score: int
    position: int


class EntityInventoryBuilder:
    """Construct candidate entity inventories for parsed chunks."""

    _BIOMEDICAL_TERMS: frozenset[str] = frozenset(
        {
            "cell",
            "cells",
            "protein",
            "proteins",
            "gene",
            "genes",
            "tumor",
            "tumors",
            "cancer",
            "dna",
            "rna",
            "enzyme",
            "cytokine",
        }
    )
    _STOPWORDS: frozenset[str] = frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "it",
            "its",
            "they",
            "them",
            "their",
            "this",
            "that",
            "these",
            "those",
            "we",
            "you",
            "he",
            "she",
            "him",
            "her",
            "i",
        }
    )
    _PRONOUNS: frozenset[str] = frozenset(
        {
            "it",
            "its",
            "they",
            "them",
            "their",
            "theirs",
            "we",
            "us",
            "our",
            "ours",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "you",
            "your",
            "yours",
            "i",
            "me",
            "my",
            "mine",
        }
    )

    def __init__(
        self,
        config: AppConfig,
        nlp_loader: Optional[Callable[[str], Callable[[str], object]]] = None,
    ) -> None:
        """Initialize the inventory builder.

        Args:
            config: Parsed application configuration.
            nlp_loader: Optional callable returning an NLP pipeline for a model name.
        """
        self._config = config
        self._nlp_loader = nlp_loader or self._default_nlp_loader
        self._pipelines: Dict[str, Callable[[str], object]] = {}

    def build_inventory(self, element: ParsedElement) -> List[str]:
        """Generate ranked entity candidates for a parsed element.

        Args:
            element: Parsed chunk produced by the parsing pipeline.

        Returns:
            List[str]: Ordered list of candidate entity mentions capped at fifty items.
        """
        if not element.content.strip():
            return []
        pipeline_key = self._select_pipeline_key(element.content)
        nlp = self._load_pipeline(pipeline_key)
        doc = nlp(element.content)
        candidates: Dict[str, _Candidate] = {}
        self._collect_named_entities(doc, element.content, candidates)
        self._collect_repeated_noun_chunks(doc, element.content, candidates)
        self._collect_proper_nouns(doc, element.content, candidates)
        if pipeline_key == _SCISPACY_MODEL:
            self._expand_abbreviations(element.content, candidates)
        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.score, candidate.position, candidate.text.lower()),
        )
        return [candidate.text for candidate in ordered[:50]]

    def _select_pipeline_key(self, text: str) -> str:
        """Determine which NLP pipeline should process the provided text.

        Args:
            text: Chunk content used to estimate biomedical vocabulary density.

        Returns:
            str: Model identifier for spaCy or scispaCy pipeline.
        """
        tokens = re.findall(r"[A-Za-z0-9-]+", text)
        if not tokens:
            return _SPACY_MODEL
        biomedical_tokens = sum(
            1 for token in tokens if token.lower() in self._BIOMEDICAL_TERMS
        )
        ratio = biomedical_tokens / max(len(tokens), 1)
        if ratio >= 0.2:
            return _SCISPACY_MODEL
        return _SPACY_MODEL

    def _load_pipeline(self, key: str) -> Callable[[str], object]:
        """Fetch or load an NLP pipeline for the given key.

        Args:
            key: Model identifier requested for text processing.

        Returns:
            Callable[[str], object]: Callable pipeline returning a document object.
        """
        if key in self._pipelines:
            return self._pipelines[key]
        try:
            pipeline = self._nlp_loader(key)
        except Exception:  # noqa: BLE001 - allow graceful degradation
            LOGGER.exception("Failed to load NLP pipeline %s", key)
            pipeline = self._fallback_pipeline()
        self._pipelines[key] = pipeline
        return pipeline

    def _collect_named_entities(
        self,
        doc: object,
        text: str,
        candidates: Dict[str, _Candidate],
    ) -> None:
        """Record named entities from the NLP document.

        Args:
            doc: NLP document returned from the pipeline.
            text: Original chunk text.
            candidates: Accumulator mapping of candidate entries.
        """
        for span in getattr(doc, "ents", []) or []:
            self._register_candidate(str(getattr(span, "text", "")).strip(), text, 3, candidates)

    def _collect_repeated_noun_chunks(
        self,
        doc: object,
        text: str,
        candidates: Dict[str, _Candidate],
    ) -> None:
        """Register noun chunks appearing multiple times within the chunk.

        Args:
            doc: NLP document returned from the pipeline.
            text: Original chunk text.
            candidates: Accumulator mapping of candidate entries.
        """
        noun_chunks = list(getattr(doc, "noun_chunks", []) or [])
        counts: Dict[str, int] = {}
        for span in noun_chunks:
            chunk = str(getattr(span, "text", "")).strip()
            if not chunk:
                continue
            key = chunk.lower()
            counts[key] = counts.get(key, 0) + 1
        for span in noun_chunks:
            chunk = str(getattr(span, "text", "")).strip()
            if not chunk:
                continue
            if counts.get(chunk.lower(), 0) < 2:
                continue
            self._register_candidate(chunk, text, 2, candidates)

    def _collect_proper_nouns(
        self,
        doc: object,
        text: str,
        candidates: Dict[str, _Candidate],
    ) -> None:
        """Add proper nouns as lower priority candidates.

        Args:
            doc: NLP document returned from the pipeline.
            text: Original chunk text.
            candidates: Accumulator mapping of candidate entries.
        """
        for token in doc:
            token_text = str(getattr(token, "text", "")).strip()
            if not token_text:
                continue
            pos = str(getattr(token, "pos_", ""))
            if pos.upper() == "PROPN" or (token_text[0].isupper() and token_text.lower() not in self._STOPWORDS):
                self._register_candidate(token_text, text, 1, candidates)

    def _expand_abbreviations(
        self,
        text: str,
        candidates: Dict[str, _Candidate],
    ) -> None:
        """Expand parenthetical abbreviations in biomedical contexts.

        Args:
            text: Original chunk text.
            candidates: Accumulator mapping of candidate entries.
        """
        pattern = re.compile(r"(?P<long>[A-Za-z][A-Za-z0-9\-\s]+?)\s*\((?P<abbr>[A-Za-z0-9\-]{2,})\)")
        for match in pattern.finditer(text):
            long_form = match.group("long").strip()
            abbreviation = match.group("abbr").strip()
            if long_form:
                self._register_candidate(long_form, text, 2, candidates)
            if abbreviation:
                self._register_candidate(abbreviation, text, 2, candidates)

    def _register_candidate(
        self,
        candidate_text: str,
        source_text: str,
        score: int,
        candidates: Dict[str, _Candidate],
    ) -> None:
        """Register a candidate mention with deduplication and ranking.

        Args:
            candidate_text: Proposed mention text.
            source_text: Chunk content where the mention must appear.
            score: Priority score where higher values rank earlier.
            candidates: Accumulator mapping of candidate entries.
        """
        cleaned = candidate_text.strip()
        if not cleaned:
            return
        normalized = cleaned.lower()
        if not self._is_valid_candidate(normalized, source_text):
            return
        position = self._find_position(cleaned, source_text)
        existing = candidates.get(normalized)
        if existing:
            if score > existing.score or (score == existing.score and position < existing.position):
                candidates[normalized] = _Candidate(text=cleaned, score=score, position=position)
            return
        candidates[normalized] = _Candidate(text=cleaned, score=score, position=position)

    def _is_valid_candidate(self, normalized: str, source_text: str) -> bool:
        """Validate whether a candidate satisfies filtering rules.

        Args:
            normalized: Lowercase representation of the candidate text.
            source_text: Chunk content where the mention must appear.

        Returns:
            bool: ``True`` when the candidate passes all filters.
        """
        if normalized in self._STOPWORDS or normalized in self._PRONOUNS:
            return False
        alnum = re.sub(r"[^A-Za-z0-9]", "", normalized)
        if len(alnum) <= 1:
            return False
        if normalized not in source_text.lower():
            return False
        return True

    @staticmethod
    def _find_position(candidate: str, source_text: str) -> int:
        """Locate the first index of the candidate within the source text.

        Args:
            candidate: Candidate text with original casing.
            source_text: Chunk content searched for the candidate.

        Returns:
            int: Character index of the first match or the content length when missing.
        """
        idx = source_text.find(candidate)
        if idx != -1:
            return idx
        lower_idx = source_text.lower().find(candidate.lower())
        if lower_idx != -1:
            return lower_idx
        return len(source_text)

    @staticmethod
    def _fallback_pipeline() -> Callable[[str], object]:
        """Provide a no-op NLP pipeline used when loading fails.

        Returns:
            Callable[[str], object]: Callable returning a minimal document.
        """
        def _blank(text: str) -> object:  # pragma: no cover - safety fallback
            return type(
                "FallbackDoc",
                (),
                {
                    "text": text,
                    "ents": [],
                    "noun_chunks": [],
                    "__iter__": lambda self: iter([]),
                },
            )()

        return _blank

    @staticmethod
    def _default_nlp_loader(model_name: str) -> Callable[[str], object]:
        """Load a spaCy pipeline for the requested model.

        Args:
            model_name: Name of the spaCy or scispaCy model to load.

        Returns:
            Callable[[str], object]: Callable pipeline that produces documents.

        Raises:
            RuntimeError: If spaCy is not installed in the environment.
        """
        try:
            import spacy
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("spaCy is required for entity inventory generation") from exc
        try:
            nlp = spacy.load(model_name)
        except OSError:
            LOGGER.warning("spaCy model %s missing; using blank English pipeline", model_name)
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        return nlp
