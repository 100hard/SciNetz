"""Two-pass triplet extraction pipeline for Phase 3."""
from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

from backend.app.config import AppConfig
from backend.app.contracts import Evidence, ParsedElement, TextSpan, Triplet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawLLMTriple:
    """Representation of a triple returned from the LLM pass."""

    subject_text: str
    relation_verbatim: str
    object_text: str
    supportive_sentence: str
    confidence: float


class LLMExtractor(ABC):
    """Interface for LLM-backed triple extraction adapters."""

    @abstractmethod
    def extract_triples(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
    ) -> Sequence[RawLLMTriple]:
        """Extract raw triples for a parsed element.

        Args:
            element: Parsed element describing the chunk to analyze.
            candidate_entities: Optional ordered list of candidate entity strings.
            max_triples: Maximum number of triples to return.

        Returns:
            Sequence[RawLLMTriple]: Raw triples emitted by the language model.
        """


class OpenAIExtractor(LLMExtractor):
    """Adapter placeholder for OpenAI-based extraction."""

    def __init__(self, *_: object, **__: object) -> None:
        """Initialize the extractor."""

    def extract_triples(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
    ) -> Sequence[RawLLMTriple]:
        """Invoke the OpenAI API for extraction.

        Args:
            element: Parsed element describing the chunk to analyze.
            candidate_entities: Optional ordered list of candidate entity strings.
            max_triples: Maximum number of triples to return.

        Returns:
            Sequence[RawLLMTriple]: Raw triples emitted by the language model.

        Raises:
            NotImplementedError: Always raised until the adapter is implemented.
        """

        raise NotImplementedError("OpenAI extractor requires API integration")


class TwoPassTripletExtractor:
    """Perform two-pass extraction with deterministic span linking."""

    def __init__(self, config: AppConfig, llm_extractor: LLMExtractor) -> None:
        """Create a new extraction pipeline.

        Args:
            config: Parsed application configuration.
            llm_extractor: Adapter responsible for producing raw triples.
        """

        self._config = config
        self._llm_extractor = llm_extractor
        self._threshold = config.extraction.fuzzy_match_threshold

    def extract_from_element(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
    ) -> List[Triplet]:
        """Extract validated triples from the provided element.

        Args:
            element: Parsed element to extract triples from.
            candidate_entities: Optional ordered list of suggested entity mentions.

        Returns:
            List[Triplet]: Triples that passed span validation and normalization.
        """

        max_triples = self._compute_max_triples(element.content)
        raw_triples = self._llm_extractor.extract_triples(
            element=element,
            candidate_entities=candidate_entities,
            max_triples=max_triples,
        )
        accepted: List[Triplet] = []
        for raw in raw_triples:
            try:
                normalized_relation, swap = normalize_relation(raw.relation_verbatim)
            except ValueError:
                LOGGER.info("Dropping triple with unmapped relation: %s", raw.relation_verbatim)
                continue
            subject_text = raw.subject_text.strip()
            object_text = raw.object_text.strip()
            sentence_text = raw.supportive_sentence.strip()
            if not subject_text or not object_text or not sentence_text:
                LOGGER.info("Dropping triple with empty fields: %s", raw)
                continue
            if not (0.0 <= raw.confidence <= 1.0):
                LOGGER.info("Dropping triple with invalid confidence %.3f", raw.confidence)
                continue
            if swap:
                subject_text, object_text = object_text, subject_text
            subject_span = self._find_span(element.content, subject_text)
            object_span = self._find_span(element.content, object_text)
            sentence_span = self._find_span(element.content, sentence_text)
            if not subject_span:
                LOGGER.info("Dropping triple; subject span not found: %s", subject_text)
                continue
            if not object_span:
                LOGGER.info("Dropping triple; object span not found: %s", object_text)
                continue
            if subject_span[0] < 0 or subject_span[1] > len(element.content):
                LOGGER.info("Dropping triple; subject span out of bounds: %s", subject_span)
                continue
            if object_span[0] < 0 or object_span[1] > len(element.content):
                LOGGER.info("Dropping triple; object span out of bounds: %s", object_span)
                continue
            if spans_overlap(subject_span, object_span):
                LOGGER.info("Dropping triple; overlapping subject/object spans: %s", raw)
                continue
            if not sentence_span:
                LOGGER.info("Dropping triple; sentence span not found: %s", sentence_text)
                continue
            if sentence_span[0] < 0 or sentence_span[1] > len(element.content):
                LOGGER.info("Dropping triple; sentence span out of bounds: %s", sentence_span)
                continue
            evidence = Evidence(
                element_id=element.element_id,
                doc_id=element.doc_id,
                text_span=TextSpan(start=sentence_span[0], end=sentence_span[1]),
                full_sentence=element.content[sentence_span[0] : sentence_span[1]],
            )
            triplet = Triplet(
                subject=subject_text,
                predicate=normalized_relation,
                object=object_text,
                confidence=raw.confidence,
                evidence=evidence,
                pipeline_version=self._config.pipeline.version,
            )
            accepted.append(triplet)
        return accepted

    def _compute_max_triples(self, content: str) -> int:
        """Compute the triple limit for an element based on token count.

        Args:
            content: Text content of the parsed element.

        Returns:
            int: Maximum number of triples allowed for the chunk.
        """

        tokens = re.findall(r"\w+", content)
        token_count = len(tokens)
        scaled = max(1, math.ceil(token_count / self._config.extraction.tokens_per_triple))
        sentence_count = max(1, len(re.findall(r"[^.!?]+[.!?]", content)))
        estimate = max(scaled, sentence_count)
        return min(self._config.extraction.max_triples_per_chunk_base, estimate)

    def _find_span(self, text: str, target: str) -> Optional[Tuple[int, int]]:
        """Locate the character span of the target text within the source.

        Args:
            text: Source text to search within.
            target: Target substring to locate.

        Returns:
            Optional[Tuple[int, int]]: Inclusive-exclusive span if found, otherwise None.
        """

        if not target:
            return None
        index = text.find(target)
        if index != -1:
            return index, index + len(target)
        lower_index = text.lower().find(target.lower())
        if lower_index != -1:
            return lower_index, lower_index + len(target)
        return self._fuzzy_find_span(text, target)

    def _fuzzy_find_span(self, text: str, target: str) -> Optional[Tuple[int, int]]:
        """Perform fuzzy matching to locate a target span.

        Args:
            text: Source text to search within.
            target: Target substring to locate approximately.

        Returns:
            Optional[Tuple[int, int]]: Inclusive-exclusive span if a close match is found.
        """

        threshold = self._threshold
        if threshold <= 0:
            return None
        normalized_target = target.strip()
        if not normalized_target:
            return None
        span_length = len(normalized_target)
        best_ratio = 0.0
        best_span: Optional[Tuple[int, int]] = None
        limit = len(text) - span_length
        if limit < 0:
            ratio = SequenceMatcher(None, text.lower(), normalized_target.lower()).ratio()
            if ratio >= threshold:
                return 0, len(text)
            return None
        for start in range(0, limit + 1):
            candidate = text[start : start + span_length]
            ratio = SequenceMatcher(None, candidate.lower(), normalized_target.lower()).ratio()
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, start + span_length)
        return best_span


def spans_overlap(first: Tuple[int, int], second: Tuple[int, int]) -> bool:
    """Return whether two spans overlap.

    Args:
        first: First character span.
        second: Second character span.

    Returns:
        bool: True if spans overlap, False otherwise.
    """

    return max(first[0], second[0]) < min(first[1], second[1])


def normalize_relation(relation_text: str) -> Tuple[str, bool]:
    """Normalize a relation phrase to the canonical predicate name.

    Args:
        relation_text: Relation phrase returned by the LLM.

    Returns:
        Tuple[str, bool]: Canonical predicate and whether subject/object should swap.

    Raises:
        ValueError: If the relation cannot be mapped to a canonical predicate.
    """

    cleaned = re.sub(r"[^a-z]+", " ", relation_text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for phrase, normalized, swap in _RELATION_PATTERNS:
        if phrase in cleaned:
            return normalized, swap
    raise ValueError(f"Unsupported relation phrase: {relation_text}")


_RELATION_PATTERNS: List[Tuple[str, str, bool]] = sorted(
    [
        ("is used by", "uses", True),
        ("was used by", "uses", True),
        ("uses", "uses", False),
        ("utilizes", "uses", False),
        ("employs", "uses", False),
        ("trained on", "trained-on", False),
        ("is trained on", "trained-on", False),
        ("was trained on", "trained-on", False),
        ("evaluated on", "evaluated-on", False),
        ("tested on", "evaluated-on", False),
        ("assessed on", "evaluated-on", False),
        ("compared to", "compared-to", False),
        ("compared with", "compared-to", False),
        ("outperforms", "outperforms", False),
        ("performs better than", "outperforms", False),
        ("increases", "increases", False),
        ("improves", "increases", False),
        ("boosts", "increases", False),
        ("decreases", "decreases", False),
        ("reduces", "decreases", False),
        ("lowers", "decreases", False),
        ("causes", "causes", False),
        ("results in", "causes", False),
        ("leads to", "causes", False),
        ("correlates with", "correlates-with", False),
        ("correlates to", "correlates-with", False),
        ("associated with", "correlates-with", False),
        ("defined as", "defined-as", False),
        ("is defined as", "defined-as", False),
        ("part of", "part-of", False),
        ("component of", "part-of", False),
        ("is a", "is-a", False),
        ("is an", "is-a", False),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)


__all__ = [
    "RawLLMTriple",
    "LLMExtractor",
    "OpenAIExtractor",
    "TwoPassTripletExtractor",
    "normalize_relation",
    "spans_overlap",
]

