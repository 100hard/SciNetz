"""Two-pass triplet extraction pipeline for Phase 3."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend.app.config import AppConfig, OpenAIConfig
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


@dataclass(frozen=True)
class _HTTPResponse:
    """Minimal HTTP response wrapper used by the OpenAI extractor."""

    status_code: int
    text: str
    body: Optional[Dict[str, Any]] = None


HTTPPostFn = Callable[[str, Dict[str, Any], Dict[str, str]], _HTTPResponse]


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
    """Adapter for invoking OpenAI chat completions to extract triples."""

    _SYSTEM_PROMPT = (
        "You are a structured information extraction service for research papers. "
        "Return relation triples grounded in the provided text."
    )

    def __init__(
        self,
        settings: OpenAIConfig,
        http_post: Optional[HTTPPostFn] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the OpenAI extractor.

        Args:
            settings: Configuration options for the OpenAI adapter.
            http_post: Optional callable used to perform HTTP POST requests.
            api_key: Optional override for the OpenAI API key; defaults to environment.

        Raises:
            RuntimeError: If the API key is missing.
        """

        self._settings = settings
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI API")
        self._api_key = resolved_key
        self._http_post = http_post or self._default_post
        self._retry_statuses = set(settings.retry_statuses)

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
        """

        payload = self._build_request(element, candidate_entities, max_triples)
        response_json = self._post_with_retries(payload)
        return self._parse_response(response_json, max_triples)

    def _build_request(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
    ) -> Dict[str, Any]:
        """Construct the JSON payload for the OpenAI chat completion request.

        Args:
            element: Parsed element describing the chunk to analyze.
            candidate_entities: Optional ordered list of candidate entity strings.
            max_triples: Maximum number of triples requested from the model.

        Returns:
            Dict[str, Any]: Payload that instructs the model to emit structured triples.
        """

        entity_hint = ""
        if candidate_entities:
            formatted_entities = "\n".join(f"- {entity}" for entity in candidate_entities)
            entity_hint = (
                "Candidate entities that may appear in the chunk:\n"
                f"{formatted_entities}\n\n"
            )
        user_prompt = (
            "Extract up to {limit} relation triples from the following research paper chunk. "
            "Return a JSON object with a `triples` array. Each triple must contain the fields "
            "`subject_text`, `relation_verbatim`, `object_text`, `supportive_sentence`, and `confidence` "
            "(a value between 0 and 1). Ensure all subjects, objects, and supportive sentences "
            "are copied verbatim from the text.\n\n"
            "Section: {section}\n"
            "Chunk:\n{chunk}\n\n"
            "{entity_hint}Only include triples that are directly supported by the text."
        ).format(
            limit=max_triples,
            section=element.section,
            chunk=element.content,
            entity_hint=entity_hint,
        )
        return {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

    def _post_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send the request to OpenAI with retry semantics for transient errors.

        Args:
            payload: JSON request body to send to the chat completions endpoint.

        Returns:
            Dict[str, Any]: Parsed JSON response from the API.

        Raises:
            RuntimeError: If the request ultimately fails or the response is invalid.
        """

        attempt = 0
        backoff = self._settings.backoff_initial_seconds
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        while True:
            attempt += 1
            response = self._http_post("/chat/completions", payload, headers)
            if response.status_code == 200:
                if response.body is not None:
                    return response.body
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError as exc:
                    LOGGER.error("OpenAI response was not valid JSON: %s", response.text)
                    raise RuntimeError("OpenAI response decoding failed") from exc
            if (
                response.status_code in self._retry_statuses
                and attempt <= self._settings.max_retries
            ):
                LOGGER.warning(
                    "OpenAI request failed with status %s; retrying in %.2f seconds",
                    response.status_code,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self._settings.backoff_max_seconds)
                continue
            LOGGER.error(
                "OpenAI request failed after %s attempts with status %s: %s",
                attempt,
                response.status_code,
                response.text,
            )
            raise RuntimeError("OpenAI request failed")

    def _default_post(
        self, path: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> _HTTPResponse:
        """Perform an HTTP POST using urllib for the given payload."""

        url = f"{self._settings.api_base.rstrip('/')}{path}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._settings.timeout_seconds) as response:
                body_bytes = response.read()
                text = body_bytes.decode("utf-8")
                try:
                    body = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    body = None
                return _HTTPResponse(status_code=response.getcode(), text=text, body=body)
        except urllib.error.HTTPError as exc:
            error_text = ""
            if exc.fp:
                try:
                    error_text = exc.fp.read().decode("utf-8")
                except Exception:  # pragma: no cover - defensive
                    error_text = str(exc)
            return _HTTPResponse(status_code=exc.code, text=error_text or str(exc))
        except urllib.error.URLError as exc:
            LOGGER.error("OpenAI network error: %s", exc)
            raise RuntimeError("OpenAI network error") from exc

    def _parse_response(
        self, response_json: Dict[str, Any], max_triples: int
    ) -> Sequence[RawLLMTriple]:
        """Convert the OpenAI chat completion response into raw triples.

        Args:
            response_json: Parsed JSON body from the OpenAI API.
            max_triples: Maximum number of triples requested for the chunk.

        Returns:
            Sequence[RawLLMTriple]: Parsed triples emitted by the model.
        """

        choices = response_json.get("choices")
        if not choices:
            LOGGER.error("OpenAI response missing choices field: %s", response_json)
            return []
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            LOGGER.error("OpenAI response missing content: %s", response_json)
            return []
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            LOGGER.error("OpenAI response content was not JSON: %s", content)
            return []
        triples_payload = payload.get("triples", [])
        if not isinstance(triples_payload, list):
            LOGGER.error("OpenAI response triples payload invalid: %s", triples_payload)
            return []
        triples: List[RawLLMTriple] = []
        for raw in triples_payload:
            if not isinstance(raw, dict):
                LOGGER.info("Skipping malformed triple payload: %s", raw)
                continue
            try:
                triples.append(
                    RawLLMTriple(
                        subject_text=str(raw["subject_text"]),
                        relation_verbatim=str(raw["relation_verbatim"]),
                        object_text=str(raw["object_text"]),
                        supportive_sentence=str(raw["supportive_sentence"]),
                        confidence=float(raw["confidence"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.info("Dropping triple due to missing fields: %s", exc)
        return triples[:max_triples]


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

