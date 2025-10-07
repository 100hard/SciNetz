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
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from urllib import error as urllib_error
from urllib import request as urllib_request

try:  # pragma: no cover - optional dependency
    import httpx as _httpx
except ModuleNotFoundError:  # pragma: no cover - tests rely on stub client instead
    _httpx = None

json_module = json

from backend.app.config import AppConfig, OpenAIConfig
from backend.app.contracts import Evidence, ParsedElement, TextSpan, Triplet

LOGGER = logging.getLogger(__name__)


@dataclass
class _SimpleHTTPResponse:
    """Minimal HTTP response wrapper with JSON helpers."""

    status_code: int
    _content: bytes

    def json(self) -> Any:
        """Parse the response body as JSON."""

        return json.loads(self.text)

    @property
    def text(self) -> str:
        """Return the decoded text body."""

        return self._content.decode("utf-8", errors="replace")


@runtime_checkable
class _HTTPClient(Protocol):
    """Protocol for minimal HTTP client used by the extractor."""

    def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, Any]) -> _SimpleHTTPResponse:
        """Send a POST request and return a simplified response."""

    def close(self) -> None:
        """Close any underlying resources."""


class _HTTPXClientWrapper:
    """Wrap an httpx.Client to satisfy the _HTTPClient protocol."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        if _httpx is None:  # pragma: no cover - defensive guard
            raise RuntimeError("httpx is not available")
        self._client = _httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, Any]) -> _SimpleHTTPResponse:
        response = self._client.post(path, headers=headers, json=json)
        return _SimpleHTTPResponse(status_code=response.status_code, _content=response.content)

    def close(self) -> None:
        self._client.close()


class _UrllibHTTPClient:
    """Fallback HTTP client implemented with urllib for environments without httpx."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, Any]) -> _SimpleHTTPResponse:
        url = f"{self._base_url}/{path.lstrip('/')}"
        payload = json_module.dumps(json).encode("utf-8")
        combined_headers = {"Content-Type": "application/json", **headers}
        request = urllib_request.Request(url, data=payload, headers=combined_headers, method="POST")
        try:
            with urllib_request.urlopen(request, timeout=self._timeout) as response:
                content = response.read()
                status = response.getcode() or 0
                return _SimpleHTTPResponse(status_code=status, _content=content)
        except urllib_error.HTTPError as exc:  # pragma: no cover - exercised via mocked clients in tests
            body = exc.read()
            return _SimpleHTTPResponse(status_code=exc.code, _content=body)

    def close(self) -> None:  # pragma: no cover - urllib has no persistent resources
        return None


@dataclass(frozen=True)
class RawLLMTriple:
    """Representation of a triple returned from the LLM pass."""

    subject_text: str
    relation_verbatim: str
    object_text: str
    supportive_sentence: str
    confidence: float


@dataclass(frozen=True)
class ExtractionResult:
    """Collection of validated triples and entity section statistics."""

    triplets: List[Triplet]
    section_distribution: Dict[str, Dict[str, int]]


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
    """Adapter that calls the OpenAI chat completions API for extraction."""

    _RESPONSE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "triplet_extraction_response",
            "schema": {
                "type": "object",
                "properties": {
                    "triples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject_text": {"type": "string"},
                                "relation_verbatim": {"type": "string"},
                                "object_text": {"type": "string"},
                                "supportive_sentence": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": [
                                "subject_text",
                                "relation_verbatim",
                                "object_text",
                                "supportive_sentence",
                                "confidence",
                            ],
                            "additionalProperties": False,
                        },
                        "default": [],
                    },
                    "prompt_version": {"type": "string"},
                },
                "required": ["triples"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        prompt_version: str = "phase3-v1",
        client: Optional[_HTTPClient] = None,
    ) -> None:
        """Initialize the extractor with OpenAI credentials and options.

        Args:
            api_key: Explicit API key; falls back to ``OPENAI_API_KEY`` env var.
            model: Model name to invoke for extraction.
            base_url: Base URL for the OpenAI REST API.
            timeout_seconds: Request timeout in seconds.
            prompt_version: Version identifier for the prompt template.
            client: Optional pre-configured HTTP client (used in tests).

        Raises:
            ValueError: If no API key is provided.
        """

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY")
        self._api_key = resolved_key
        self._model = model
        self._prompt_version = prompt_version
        if client is None:
            if _httpx is not None:
                self._client: _HTTPClient = _HTTPXClientWrapper(base_url, timeout_seconds)
            else:
                self._client = _UrllibHTTPClient(base_url, timeout_seconds)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._endpoint = "chat/completions"
        self._base_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def close(self) -> None:
        """Close the underlying HTTP client if owned by the adapter."""

        if getattr(self, "_owns_client", False):
            self._client.close()

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
            RuntimeError: If the OpenAI API returns an error or malformed response.
        """

        payload = self._build_payload(element, candidate_entities, max_triples)
        LOGGER.debug("Requesting OpenAI extraction for element %s", element.element_id)
        response = self._client.post(
            self._endpoint,
            headers=self._base_headers,
            json=payload,
        )
        if response.status_code >= 400:
            message = self._extract_error_message(response)
            LOGGER.error("OpenAI extraction failed: %s", message)
            raise RuntimeError(f"OpenAI extraction failed with status {response.status_code}: {message}")
        try:
            content = response.json()
        except json.JSONDecodeError as exc:
            LOGGER.error("OpenAI response was not valid JSON: %s", exc)
            raise RuntimeError("OpenAI response was not valid JSON") from exc
        return self._parse_response(content, max_triples)

    def _build_payload(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
    ) -> Dict[str, object]:
        """Construct the chat completion payload sent to OpenAI."""

        system_prompt = self._render_system_prompt(max_triples)
        user_prompt = self._render_user_prompt(element, candidate_entities)
        payload: Dict[str, object] = {
            "model": self._model,
            "temperature": 0.0,
            "max_tokens": 800,
            "response_format": self._RESPONSE_SCHEMA,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return payload

    def _render_system_prompt(self, max_triples: int) -> str:
        """Render the system prompt used for extraction."""

        return (
            "You are an expert scientific information extractor. "
            f"Follow prompt version {self._prompt_version}. "
            f"Extract factual relationships and return at most {max_triples} triples. "
            "Subjects and objects must be explicit entity mentions. "
            "Return full supportive sentences verbatim and include a confidence between 0.0 and 1.0. "
            "Normalize relations to the allowed set and convert passive voice to active voice."
        )

    def _render_user_prompt(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
    ) -> str:
        """Render the user prompt containing chunk content and context."""

        lines = [
            f"Document ID: {element.doc_id}",
            f"Section: {element.section}",
            "Chunk:",
            element.content,
        ]
        if candidate_entities:
            lines.append("Candidate entities (focus on these if relevant):")
            for entity in candidate_entities:
                lines.append(f"- {entity}")
        lines.append("Return JSON matching the requested schema.")
        return "\n".join(lines)

    def _parse_response(
        self,
        payload: Dict[str, object],
        max_triples: int,
    ) -> Sequence[RawLLMTriple]:
        """Parse the OpenAI response into ``RawLLMTriple`` objects."""

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            LOGGER.error("OpenAI response missing choices: %s", payload)
            raise RuntimeError("OpenAI response missing choices")
        message = choices[0].get("message")  # type: ignore[index]
        if not isinstance(message, dict) or "content" not in message:
            LOGGER.error("OpenAI response missing message content: %s", payload)
            raise RuntimeError("OpenAI response missing message content")
        content = message.get("content")
        if isinstance(content, list):
            content_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            response_text = "".join(content_parts)
        elif isinstance(content, str):
            response_text = content
        else:
            LOGGER.error("OpenAI response content was unexpected: %s", content)
            raise RuntimeError("OpenAI response content was unexpected")
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            LOGGER.error("OpenAI returned non-JSON content: %s", response_text)
            raise RuntimeError("OpenAI returned non-JSON content") from exc
        triples_payload = parsed.get("triples", [])
        if not isinstance(triples_payload, list):
            LOGGER.error("OpenAI response triples field malformed: %s", parsed)
            raise RuntimeError("OpenAI response triples field malformed")
        triples: List[RawLLMTriple] = []
        for item in triples_payload[:max_triples]:
            if not isinstance(item, dict):
                LOGGER.info("Skipping non-dict triple entry: %s", item)
                continue
            try:
                triple = RawLLMTriple(
                    subject_text=str(item["subject_text"]),
                    relation_verbatim=str(item["relation_verbatim"]),
                    object_text=str(item["object_text"]),
                    supportive_sentence=str(item["supportive_sentence"]),
                    confidence=float(item["confidence"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.info("Skipping malformed triple payload %s: %s", item, exc)
                continue
            triples.append(triple)
        return triples

    @staticmethod
    def _extract_error_message(response: _SimpleHTTPResponse) -> str:
        """Extract an error message from a failed OpenAI response."""

        try:
            payload = response.json()
        except json.JSONDecodeError:
            return response.text
        error = payload.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        return json.dumps(payload)


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

        triplets, _ = self._extract_internal(element, candidate_entities)
        return triplets

    def extract_with_metadata(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
    ) -> "ExtractionResult":
        """Extract triples and section distribution metadata for canonicalization."""

        triplets, section_distribution = self._extract_internal(element, candidate_entities)
        return ExtractionResult(triplets=triplets, section_distribution=section_distribution)

    def _extract_internal(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
    ) -> Tuple[List[Triplet], Dict[str, Dict[str, int]]]:
        """Run LLM extraction and span validation, returning metadata."""

        max_triples = self._compute_max_triples(element.content)
        raw_triples = self._llm_extractor.extract_triples(
            element=element,
            candidate_entities=candidate_entities,
            max_triples=max_triples,
        )
        accepted: List[Triplet] = []
        section_counts: Dict[str, Dict[str, int]] = {}
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
            self._update_section_counts(
                section_counts,
                element.section,
                subject_text,
                object_text,
            )
        section_distribution = {entity: dict(counts) for entity, counts in section_counts.items()}
        return accepted, section_distribution

    @staticmethod
    def _update_section_counts(
        section_counts: Dict[str, Dict[str, int]],
        section: str,
        subject_text: str,
        object_text: str,
    ) -> None:
        """Update section occurrence counts for subject and object entities."""

        for entity in (subject_text, object_text):
            entity_counts = section_counts.setdefault(entity, {})
            entity_counts[section] = entity_counts.get(section, 0) + 1

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
    "ExtractionResult",
    "LLMExtractor",
    "OpenAIExtractor",
    "TwoPassTripletExtractor",
    "normalize_relation",
    "spans_overlap",
]

