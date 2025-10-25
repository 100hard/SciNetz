"""Two-pass triplet extraction pipeline for Phase 3."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from functools import lru_cache
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)
from urllib import error as urllib_error
from urllib import request as urllib_request

try:  # pragma: no cover - optional dependency
    import httpx as _httpx
except ModuleNotFoundError:  # pragma: no cover - tests rely on stub client instead
    _httpx = None

json_module = json

from backend.app.config import AppConfig, OpenAIConfig, load_config
from backend.app.contracts import Evidence, PaperMetadata, ParsedElement, TextSpan, Triplet
from backend.app.extraction.domain import DomainRouter, ExtractionDomain

if TYPE_CHECKING:
    from backend.app.extraction.cache import LLMResponseCache, TokenBudgetCache

LOGGER = logging.getLogger(__name__)

_AMBIGUOUS_ENTITY_TYPES = {"other", "unknown", "misc", "miscellaneous"}


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
    subject_type: str
    relation_verbatim: str
    object_text: str
    object_type: str
    confidence: float
    supportive_sentence: Optional[str] = None


@dataclass(frozen=True)
class ExtractionResult:
    """Collection of validated triples and entity section statistics."""

    triplets: List[Triplet]
    section_distribution: Dict[str, Dict[str, int]]
    relation_verbatims: List[str] = field(default_factory=list)
    entity_type_votes: Dict[str, Dict[str, int]] = field(default_factory=dict)


class _TruncationError(RuntimeError):
    """Raised when the LLM response terminates before producing complete JSON."""


class LLMExtractor(ABC):
    """Interface for LLM-backed triple extraction adapters."""

    @abstractmethod
    def extract_triples(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
        *,
        domain: Optional[str] = None,
    ) -> Sequence[RawLLMTriple]:
        """Extract raw triples for a parsed element.

        Args:
            element: Parsed element describing the chunk to analyze.
            candidate_entities: Optional ordered list of candidate entity strings.
            max_triples: Maximum number of triples to return.
            domain: Optional domain hint guiding the extractor prompt.

        Returns:
            Sequence[RawLLMTriple]: Raw triples emitted by the language model.
        """


_HTTPPostCallable = Callable[[str, Dict[str, Any], Dict[str, str]], Any]



class OpenAIExtractor(LLMExtractor):
    """Adapter that calls the OpenAI chat completions API for extraction."""

    _ENDPOINT = "/chat/completions"
    _SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert scientific information extractor.\n"
        "Follow prompt version {prompt_version}. Extract factual relationships and return at most {max_triples} triples.\n"
        "Subjects and objects must be explicit entity mentions drawn directly from the provided text chunk. "
        "Allowed relations: {allowed_relations}. Set relation_verbatim to exactly one of these canonical relations; "
        "skip any triple whose predicate cannot be normalized to this inventory.\n"
        "Your goal is to extract specific, named scientific concepts. Prioritize them in this order:\n"
        "- Highest Priority: Concepts explicitly introduced or defined in the chunk (for example, \"we propose a new method called X\" or \"Y is defined as ...\"). Always capture these novel contributions.\n"
        "- High Priority: Well-known, named entities from the scientific field that appear in the chunk.\n"
        "Do not skip newly introduced concepts as long as the text gives them a concrete name.\n"
        "Rules:\n"
        "- CRITICAL: The subject and object must always be different. Reject any self-referential statement.\n"
        "- Use only the provided chunk; do not rely on external knowledge or assumptions.\n"
        "- Classify every subject and object using the allowed entity types: {allowed_entity_types}. "
        "If you cannot confidently assign one of these labels (for example, if the entity is ambiguous), Discard the triple instead of outputting Other or Unknown.\n"
        "- Avoid generic placeholders or conversational phrases such as our model, our strategy, this approach, the results, the score, the data, baseline accuracy, or similar vague references.\n"
        "- Output only specific, named scientific concepts that can be reused outside the immediate sentence.\n"
        "- Return full supportive sentences verbatim and include a confidence between 0.0 and 1.0.\n"
        "Examples of entity typing: Gradient descent algorithm -> Method; MNIST dataset -> Dataset; F1 score -> Metric; "
        "Image classification task -> Task; CRISPR protein -> Material.\n"
        "Avoid vague or generic language and convert passive voice to active voice when needed. "
        "Output must strictly adhere to the requested JSON schema."
    )
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
                                "subject_type": {"type": "string"},
                                "relation_verbatim": {"type": "string"},
                                "object_text": {"type": "string"},
                                "object_type": {"type": "string"},
                                "supportive_sentence": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": [
                                "subject_text",
                                "subject_type",
                                "relation_verbatim",
                                "object_text",
                                "object_type",
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

    _NON_JSON_ERROR = "OpenAI returned non-JSON content"

    def __init__(
        self,
        *,
        settings: OpenAIConfig,
        api_key: Optional[str] = None,
        client: Optional[_HTTPClient] = None,
        http_post: Optional[_HTTPPostCallable] = None,
        token_budget_per_triple: int,
        allowed_relations: Sequence[str],
        entity_types: Sequence[str],
        max_prompt_entities: int,
        response_cache: Optional["LLMResponseCache"] = None,
        token_cache: Optional["TokenBudgetCache"] = None,
    ) -> None:
        """Initialise the extractor with configuration and networking hooks."""

        if client is not None and http_post is not None:
            raise ValueError("Provide either a client or http_post, not both")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OpenAI API key must be provided via argument or OPENAI_API_KEY")
        self._settings = settings
        self._api_key = resolved_key
        self._http_post = http_post
        if token_budget_per_triple <= 0:
            raise ValueError("token_budget_per_triple must be positive")
        self._token_budget_per_triple = token_budget_per_triple
        if max_prompt_entities <= 0:
            raise ValueError("max_prompt_entities must be positive")
        self._max_prompt_entities = max_prompt_entities
        self._initial_multiplier = max(1.0, settings.initial_output_multiplier)
        self._token_multiplier_cache: Dict[str, int] = {}
        self._response_cache = response_cache
        self._token_cache = token_cache
        normalized_relations: List[str] = []
        for relation in allowed_relations:
            cleaned = str(relation).strip()
            if cleaned and cleaned not in normalized_relations:
                normalized_relations.append(cleaned)
        if not normalized_relations:
            raise ValueError("allowed_relations must include at least one value")
        self._allowed_relations = tuple(normalized_relations)
        self._allowed_relations_text = ", ".join(self._allowed_relations)
        normalized_entity_types: List[str] = []
        for entity_type in entity_types:
            cleaned_type = str(entity_type).strip()
            if cleaned_type and cleaned_type not in normalized_entity_types:
                normalized_entity_types.append(cleaned_type)
        if not normalized_entity_types:
            raise ValueError("entity_types must include at least one value")
        self._allowed_entity_types = tuple(normalized_entity_types)
        self._allowed_entity_types_text = ", ".join(self._allowed_entity_types)

        self._client: Optional[_HTTPClient]
        self._owns_client = False
        if http_post is None:
            if client is None:
                self._client = self._build_client(settings)
                self._owns_client = True
            else:
                self._client = client
        else:
            self._client = None

        self._base_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._retry_statuses = set(settings.retry_statuses)

    def _build_client(self, settings: OpenAIConfig) -> _HTTPClient:
        """Create an HTTP client for communicating with the OpenAI API."""

        if _httpx is not None:
            return _HTTPXClientWrapper(settings.api_base, settings.timeout_seconds)
        return _UrllibHTTPClient(settings.api_base, settings.timeout_seconds)

    def close(self) -> None:
        """Close underlying HTTP resources if this instance owns them."""

        if self._owns_client and self._client is not None:
            self._client.close()

    def extract_triples(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
        *,
        domain: Optional[str] = None,
    ) -> Sequence[RawLLMTriple]:
        """Invoke the OpenAI API and parse triples from the response."""

        prompt_entities = self._limit_entities(element, candidate_entities)
        normalized_entities: Optional[Tuple[str, ...]] = None
        if prompt_entities:
            normalized_entities = tuple(prompt_entities)
        response_cache_key: Optional[str] = None
        if self._response_cache is not None:
            response_cache_key = self._response_cache_key(
                element, normalized_entities, max_triples
            )
            cached_triples = self._response_cache.get(response_cache_key)
            if cached_triples is not None:
                LOGGER.debug("Using cached LLM triples for %s", element.element_id)
                return cached_triples[: max(1, max_triples)]
        attempt_budget = max(1, max_triples)
        token_multiplier = self._resolve_initial_multiplier(element, attempt_budget)
        bumped_token_budget = False
        last_error: Optional[Exception] = None
        while attempt_budget >= 1:
            cache_key = self._cache_key(element, attempt_budget)
            payload = self._build_payload(
                element,
                prompt_entities,
                attempt_budget,
                token_multiplier,
            )
            response_json = self._post_with_retries(payload)
            try:
                triples = self._parse_response(response_json, attempt_budget)
            except _TruncationError as exc:
                last_error = exc
                next_multiplier = self._next_token_multiplier(attempt_budget, token_multiplier)
                if next_multiplier is not None:
                    current_tokens = self._calculate_max_tokens(attempt_budget, token_multiplier)
                    next_tokens = self._calculate_max_tokens(attempt_budget, next_multiplier)
                    LOGGER.warning(
                        "OpenAI response truncated at %s tokens; retrying with %s tokens (budget %s)",
                        current_tokens,
                        next_tokens,
                        attempt_budget,
                    )
                    token_multiplier = next_multiplier
                    bumped_token_budget = True
                    continue
                if attempt_budget == 1:
                    raise
                reduced_budget = max(1, attempt_budget // 2)
                LOGGER.warning(
                    "OpenAI response truncated at %s triples with max token budget; retrying with budget %s",
                    attempt_budget,
                    reduced_budget,
                )
                attempt_budget = reduced_budget
                token_multiplier = self._resolve_initial_multiplier(element, attempt_budget)
                bumped_token_budget = False
                continue
            except RuntimeError as exc:
                if str(exc) != self._NON_JSON_ERROR or attempt_budget == 1:
                    raise
                LOGGER.warning(
                    "OpenAI response truncated at %s triples; retrying with budget %s",
                    attempt_budget,
                    max(1, attempt_budget // 2),
                )
                last_error = exc
                attempt_budget = max(1, attempt_budget // 2)
                token_multiplier = self._resolve_initial_multiplier(element, attempt_budget)
                bumped_token_budget = False
                continue
            if attempt_budget < max_triples:
                LOGGER.info(
                    "Successful extraction after reducing triple budget from %s to %s",
                    max_triples,
                    attempt_budget,
                )
            if bumped_token_budget:
                LOGGER.info(
                    "Successful extraction after increasing token budget multiplier to %s (triples %s)",
                    token_multiplier,
                    attempt_budget,
                )
            elif token_multiplier > 1:
                LOGGER.info(
                    "Successful extraction using configured token budget multiplier %s (triples %s)",
                    token_multiplier,
                    attempt_budget,
                )
            self._token_multiplier_cache[cache_key] = token_multiplier
            if self._token_cache is not None:
                self._token_cache.set(cache_key, token_multiplier)
            if attempt_budget != max_triples:
                original_key = self._cache_key(element, max_triples)
                self._token_multiplier_cache[original_key] = token_multiplier
                if self._token_cache is not None:
                    self._token_cache.set(original_key, token_multiplier)
            if self._response_cache is not None:
                attempt_response_key = self._response_cache_key(
                    element, normalized_entities, attempt_budget
                )
                self._response_cache.set(attempt_response_key, triples)
                if attempt_budget != max_triples and response_cache_key is not None:
                    self._response_cache.set(response_cache_key, triples)
            return triples
        raise last_error if last_error is not None else RuntimeError(self._NON_JSON_ERROR)

    def _build_payload(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
        token_multiplier: int,
    ) -> Dict[str, Any]:
        """Construct the chat completion payload sent to OpenAI."""

        system_prompt = self._render_system_prompt(max_triples)
        user_prompt = self._render_user_prompt(element, candidate_entities)
        payload: Dict[str, Any] = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._calculate_max_tokens(max_triples, token_multiplier),
            "response_format": self._RESPONSE_SCHEMA,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return payload

    def _limit_entities(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        """Trim candidate entities to respect configured prompt limits."""

        if not candidate_entities:
            return None
        cleaned: List[str] = []
        for entity in candidate_entities:
            if not isinstance(entity, str):
                continue
            stripped = entity.strip()
            if stripped:
                cleaned.append(stripped)
        if not cleaned:
            return None
        if len(cleaned) > self._max_prompt_entities:
            LOGGER.debug(
                "Trimming candidate entities from %s to %s for element %s",
                len(cleaned),
                self._max_prompt_entities,
                element.element_id,
            )
            return cleaned[: self._max_prompt_entities]
        return cleaned

    def _cache_key(self, element: ParsedElement, max_triples: int) -> str:
        """Generate a cache key for the element and triple budget."""

        budget = max(1, max_triples)
        content_hash = getattr(element, "content_hash", None)
        if not content_hash:
            content_hash = hashlib.sha256(element.content.encode("utf-8")).hexdigest()
        return f"{self._settings.model}:{self._settings.prompt_version}:{content_hash}:{budget}"

    def _response_cache_key(
        self,
        element: ParsedElement,
        prompt_entities: Optional[Tuple[str, ...]],
        max_triples: int,
    ) -> str:
        """Return a stable cache key for persisted LLM responses."""

        budget = str(max(1, max_triples))
        content_hash = getattr(element, "content_hash", None)
        if not content_hash:
            content_hash = hashlib.sha256(element.content.encode("utf-8")).hexdigest()
        parts: List[str] = [
            self._settings.model,
            self._settings.prompt_version,
            content_hash,
            budget,
            str(self._max_prompt_entities),
            self._allowed_entity_types_text,
        ]
        if prompt_entities:
            parts.extend(prompt_entities)
        digest_input = "||".join(parts)
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    def _resolve_initial_multiplier(self, element: ParsedElement, max_triples: int) -> int:
        """Determine the starting token multiplier for an extraction attempt."""

        cache_key = self._cache_key(element, max_triples)
        cached = self._token_multiplier_cache.get(cache_key)
        if cached is not None and cached >= 1:
            return cached
        if self._token_cache is not None:
            persisted = self._token_cache.get(cache_key)
            if persisted is not None and persisted >= 1:
                self._token_multiplier_cache[cache_key] = persisted
                return persisted
        multiplier = int(math.ceil(self._initial_multiplier))
        return max(1, multiplier)

    def _calculate_max_tokens(self, max_triples: int, token_multiplier: int) -> int:
        """Determine the token limit for a request based on triple budget."""

        base_tokens = max_triples * self._token_budget_per_triple * max(token_multiplier, 1)
        baseline = max(self._token_budget_per_triple, base_tokens)
        return min(self._settings.max_output_tokens, baseline)

    def _render_system_prompt(self, max_triples: int) -> str:
        """Render the system prompt used for extraction."""

        return self._SYSTEM_PROMPT_TEMPLATE.format(
            prompt_version=self._settings.prompt_version,
            max_triples=max_triples,
            allowed_relations=self._allowed_relations_text,
            allowed_entity_types=self._allowed_entity_types_text,
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
            lines.append("Candidate entities that may appear in the chunk:")
            for entity in candidate_entities:
                lines.append(f"- {entity}")
        lines.append("Return JSON matching the requested schema.")
        return "\n".join(lines)

    def _post_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send the payload to OpenAI with retry semantics for transient errors."""

        attempt = 0
        delay = self._settings.backoff_initial_seconds
        while True:
            start = time.perf_counter()
            try:
                response = self._dispatch_request(payload)
            except Exception as exc:
                elapsed = time.perf_counter() - start
                if not self._should_retry_exception(exc, attempt):
                    LOGGER.error(
                        "OpenAI extraction request failed after %.2fs: %s",
                        elapsed,
                        exc,
                    )
                    raise RuntimeError(f"OpenAI extraction request failed: {exc}") from exc
                attempt += 1
                LOGGER.warning(
                    "OpenAI extraction request raised %s after %.2fs; retrying (attempt %s)",
                    exc.__class__.__name__,
                    elapsed,
                    attempt,
                )
                sleep_seconds = min(delay, self._settings.backoff_max_seconds)
                time.sleep(sleep_seconds)
                delay = min(delay * 2, self._settings.backoff_max_seconds)
                continue
            elapsed = time.perf_counter() - start
            if response.status_code < 400:
                try:
                    response_payload = response.json()
                except json.JSONDecodeError as exc:
                    LOGGER.error(
                        "OpenAI response was not valid JSON after %.2fs: %s",
                        elapsed,
                        exc,
                    )
                    raise RuntimeError("OpenAI response was not valid JSON") from exc
                if attempt > 0:
                    LOGGER.info(
                        "OpenAI extraction succeeded after %s retries (elapsed %.2fs, status %s)",
                        attempt,
                        elapsed,
                        response.status_code,
                    )
                elif LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "OpenAI extraction completed in %.2fs (status %s)",
                        elapsed,
                        response.status_code,
                    )
                return response_payload
            if not self._should_retry(response.status_code, attempt):
                message = self._extract_error_message(response)
                LOGGER.error(
                    "OpenAI extraction failed with status %s after %.2fs: %s",
                    response.status_code,
                    elapsed,
                    message,
                )
                raise RuntimeError(
                    f"OpenAI extraction failed with status {response.status_code}: {message}"
                )
            attempt += 1
            LOGGER.warning(
                "OpenAI extraction received status %s after %.2fs; retrying (attempt %s)",
                response.status_code,
                elapsed,
                attempt,
            )
            sleep_seconds = min(delay, self._settings.backoff_max_seconds)
            time.sleep(sleep_seconds)
            delay = min(delay * 2, self._settings.backoff_max_seconds)

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """Return whether the request should be retried."""

        if status_code not in self._retry_statuses:
            return False
        return attempt < self._settings.max_retries

    def _should_retry_exception(self, exc: Exception, attempt: int) -> bool:
        """Return whether an exception warrants a retry."""

        if not self._is_timeout_exception(exc):
            return False
        return attempt < self._settings.max_retries

    def _is_timeout_exception(self, exc: Exception) -> bool:
        """Return whether the exception or its causes represent a timeout."""

        visited: set[int] = set()
        current: Optional[BaseException] = exc
        timeout_base = getattr(_httpx, "TimeoutException", None) if _httpx is not None else None
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, TimeoutError):
                return True
            if timeout_base is not None and isinstance(current, timeout_base):
                return True
            if "timeout" in current.__class__.__name__.lower():
                return True
            current = getattr(current, "__cause__", None)
        return False

    def _dispatch_request(self, payload: Dict[str, Any]) -> _SimpleHTTPResponse:
        """Dispatch the HTTP request via the configured client or hook."""

        headers = dict(self._base_headers)
        if self._http_post is not None:
            raw_response = self._http_post(self._ENDPOINT, payload, headers)
            return self._coerce_response(raw_response)
        if self._client is None:
            raise RuntimeError("No HTTP client configured for OpenAIExtractor")
        response = self._client.post(self._ENDPOINT, headers=headers, json=payload)
        if isinstance(response, _SimpleHTTPResponse):
            return response
        return self._coerce_response(response)

    def _coerce_response(self, response: Any) -> _SimpleHTTPResponse:
        """Convert various response types into ``_SimpleHTTPResponse``."""

        if isinstance(response, _SimpleHTTPResponse):
            return response
        status_code = int(getattr(response, "status_code", 0) or 0)
        if hasattr(response, "json") and callable(response.json):
            try:
                data = response.json()
            except Exception:  # pragma: no cover - defensive guard
                data = None
            else:
                if isinstance(data, (dict, list)):
                    return _SimpleHTTPResponse(
                        status_code=status_code,
                        _content=json.dumps(data).encode("utf-8"),
                    )
        if hasattr(response, "body"):
            body = getattr(response, "body")
            if isinstance(body, (bytes, bytearray)):
                return _SimpleHTTPResponse(status_code=status_code, _content=bytes(body))
            if isinstance(body, (dict, list)):
                return _SimpleHTTPResponse(
                    status_code=status_code,
                    _content=json.dumps(body).encode("utf-8"),
                )
            if body is not None:
                return _SimpleHTTPResponse(
                    status_code=status_code,
                    _content=str(body).encode("utf-8"),
                )
        text = getattr(response, "text", "")
        if isinstance(text, bytes):
            content = text
        elif isinstance(text, str):
            content = text.encode("utf-8")
        elif text:
            content = json.dumps(text).encode("utf-8")
        else:
            content = b"{}"
        return _SimpleHTTPResponse(status_code=status_code, _content=content)

    def _parse_response(
        self,
        payload: Dict[str, Any],
        max_triples: int,
    ) -> Sequence[RawLLMTriple]:
        """Parse the OpenAI response into ``RawLLMTriple`` objects."""

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            LOGGER.error("OpenAI response missing choices: %s", payload)
            raise RuntimeError("OpenAI response missing choices")
        choice = choices[0]  # type: ignore[index]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            LOGGER.warning("OpenAI response hit max tokens before completing JSON payload")
            raise _TruncationError("OpenAI response truncated by token limit")
        message = choice.get("message")
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
            supportive_sentence = item.get("supportive_sentence")
            if supportive_sentence is None:
                LOGGER.debug("LLM triple missing supportive sentence; attempting fallback: %s", item)
            try:
                triple = RawLLMTriple(
                    subject_text=str(item["subject_text"]),
                    subject_type=str(item["subject_type"]),
                    relation_verbatim=str(item["relation_verbatim"]),
                    object_text=str(item["object_text"]),
                    object_type=str(item["object_type"]),
                    supportive_sentence=(str(supportive_sentence) if supportive_sentence is not None else None),
                    confidence=float(item["confidence"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.info("Skipping malformed triple payload %s: %s", item, exc)
                continue
            triples.append(triple)
        return triples

    def _next_token_multiplier(self, max_triples: int, current_multiplier: int) -> Optional[int]:
        """Return the next token budget multiplier if additional headroom exists."""

        tentative_multiplier = max(current_multiplier, 1) * 2
        current_tokens = self._calculate_max_tokens(max_triples, current_multiplier)
        next_tokens = self._calculate_max_tokens(max_triples, tentative_multiplier)
        if next_tokens <= current_tokens:
            return None
        return tentative_multiplier
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


class DomainLLMExtractor(LLMExtractor):
    """LLM extractor that routes requests to domain-specific OpenAI prompts."""

    def __init__(
        self,
        *,
        config: AppConfig,
        api_key: str,
        token_budget_per_triple: int,
        allowed_relations: Sequence[str],
        max_prompt_entities: int,
        response_cache: Optional["LLMResponseCache"] = None,
        token_cache: Optional["TokenBudgetCache"] = None,
    ) -> None:
        self._default_domain = config.extraction.default_domain
        self._extractors: Dict[str, OpenAIExtractor] = {}
        base_settings = config.extraction.openai
        for domain in config.extraction.domains:
            settings = base_settings.model_copy(
                update={"prompt_version": domain.prompt_version or base_settings.prompt_version}
            )
            entity_types = (
                domain.normalized_entity_types or tuple(config.extraction.entity_types)
            )
            extractor = OpenAIExtractor(
                settings=settings,
                api_key=api_key,
                token_budget_per_triple=token_budget_per_triple,
                allowed_relations=allowed_relations,
                entity_types=entity_types,
                max_prompt_entities=max_prompt_entities,
                response_cache=response_cache,
                token_cache=token_cache,
            )
            self._extractors[domain.name] = extractor
        if self._default_domain not in self._extractors:
            fallback = OpenAIExtractor(
                settings=base_settings,
                api_key=api_key,
                token_budget_per_triple=token_budget_per_triple,
                allowed_relations=allowed_relations,
                entity_types=config.extraction.entity_types,
                max_prompt_entities=max_prompt_entities,
                response_cache=response_cache,
                token_cache=token_cache,
            )
            self._extractors[self._default_domain] = fallback

    def extract_triples(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
        *,
        domain: Optional[str] = None,
    ) -> Sequence[RawLLMTriple]:
        extractor = self._extractors.get(domain or self._default_domain)
        if extractor is None:
            extractor = self._extractors[self._default_domain]
        return extractor.extract_triples(
            element,
            candidate_entities,
            max_triples,
        )

    def close(self) -> None:
        """Close underlying HTTP resources for all managed extractors."""

        for extractor in self._extractors.values():
            extractor.close()


class TwoPassTripletExtractor:
    """Perform two-pass extraction with deterministic span linking."""

    def __init__(
        self,
        config: AppConfig,
        llm_extractor: LLMExtractor,
        domain_router: Optional[DomainRouter] = None,
    ) -> None:
        """Create a new extraction pipeline.

        Args:
            config: Parsed application configuration.
            llm_extractor: Adapter responsible for producing raw triples.
            domain_router: Optional resolver for domain-specific overrides.
        """

        self._config = config
        self._llm_extractor = llm_extractor
        self._domain_router = domain_router or DomainRouter(config.extraction)
        self._default_threshold = config.extraction.fuzzy_match_threshold
        normalized_types: Dict[str, str] = {}
        for entity_type in config.extraction.all_entity_types():
            cleaned = entity_type.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered not in normalized_types:
                normalized_types[lowered] = cleaned
        if not normalized_types:
            raise ValueError("config.extraction.entity_types must include at least one valid entry")
        self._entity_types = tuple(normalized_types.values())
        self._entity_type_lookup = normalized_types

    def extract_from_element(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        *,
        metadata: Optional[PaperMetadata] = None,
        domain: Optional[ExtractionDomain] = None,
    ) -> List[Triplet]:
        """Extract validated triples from the provided element.

        Args:
            element: Parsed element to extract triples from.
            candidate_entities: Optional ordered list of suggested entity mentions.
            metadata: Optional paper metadata influencing domain routing.
            domain: Optional pre-resolved extraction domain.

        Returns:
            List[Triplet]: Triples that passed span validation and normalization.
        """

        triplets, _, _, _ = self._extract_internal(
            element,
            candidate_entities,
            metadata=metadata,
            domain=domain,
        )
        return triplets

    def extract_with_metadata(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        *,
        metadata: Optional[PaperMetadata] = None,
        domain: Optional[ExtractionDomain] = None,
    ) -> "ExtractionResult":
        """Extract triples and section distribution metadata for canonicalization."""

        triplets, section_distribution, verbatims, type_votes = self._extract_internal(
            element,
            candidate_entities,
            metadata=metadata,
            domain=domain,
        )
        return ExtractionResult(
            triplets=triplets,
            section_distribution=section_distribution,
            relation_verbatims=verbatims,
            entity_type_votes=type_votes,
        )

    def _extract_internal(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        *,
        metadata: Optional[PaperMetadata],
        domain: Optional[ExtractionDomain],
    ) -> Tuple[
        List[Triplet],
        Dict[str, Dict[str, int]],
        List[str],
        Dict[str, Dict[str, int]],
    ]:
        """Run LLM extraction and span validation, returning metadata.

        Returns:
            Tuple[List[Triplet], Dict[str, Dict[str, int]], List[str], Dict[str, Dict[str, int]]]:
                Validated triplets, per-entity section counts, verbatim relations, and
                aggregated entity type votes derived from the LLM output.
        """

        context = domain or self._resolve_domain(metadata, element)
        domain_name = context.name if context is not None else None
        threshold_value = (
            context.fuzzy_match_threshold if context is not None else self._default_threshold
        )
        active_threshold = max(0.0, min(1.0, threshold_value))
        max_triples = self._compute_max_triples(element.content)
        raw_triples = self._llm_extractor.extract_triples(
            element=element,
            candidate_entities=candidate_entities,
            max_triples=max_triples,
            domain=domain_name,
        )
        accepted: List[Triplet] = []
        section_counts: Dict[str, Dict[str, int]] = {}
        relation_verbatims: List[str] = []
        type_votes: Dict[str, Dict[str, int]] = {}
        for raw in raw_triples:
            try:
                normalized_relation, swap = normalize_relation(raw.relation_verbatim)
            except ValueError:
                LOGGER.info("Dropping triple with unmapped relation: %s", raw.relation_verbatim)
                continue
            subject_text = raw.subject_text.strip()
            object_text = raw.object_text.strip()
            subject_type = self._normalize_entity_type(raw.subject_type)
            object_type = self._normalize_entity_type(raw.object_type)
            if not subject_text or not object_text:
                LOGGER.info("Dropping triple with empty subject/object: %s", raw)
                continue
            if self._is_ambiguous_type(subject_type):
                LOGGER.info("Dropping triple; ambiguous subject type: %s", raw.subject_type)
                continue
            if self._is_ambiguous_type(object_type):
                LOGGER.info("Dropping triple; ambiguous object type: %s", raw.object_type)
                continue
            sentence_text = self._resolve_supportive_sentence(
                raw,
                element,
                subject_text,
                object_text,
            )
            if not sentence_text:
                LOGGER.info("Dropping triple; supportive sentence unavailable: %s", raw)
                continue
            if not (0.0 <= raw.confidence <= 1.0):
                LOGGER.info("Dropping triple with invalid confidence %.3f", raw.confidence)
                continue
            if swap:
                subject_text, object_text = object_text, subject_text
                subject_type, object_type = object_type, subject_type
            if subject_text.casefold() == object_text.casefold():
                LOGGER.info("Dropping triple; subject and object identical: %s", subject_text)
                continue
            subject_span = self._find_span(
                element.content,
                subject_text,
                threshold=active_threshold,
            )
            object_span = self._find_span(
                element.content,
                object_text,
                threshold=active_threshold,
            )
            sentence_span = self._find_span(
                element.content,
                sentence_text,
                threshold=active_threshold,
            )
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
            relation_verbatims.append(raw.relation_verbatim)
            if subject_type:
                self._increment_type_vote(type_votes, subject_text, subject_type)
            elif raw.subject_type:
                LOGGER.debug("Unrecognized subject type '%s' for %s", raw.subject_type, subject_text)
            if object_type:
                self._increment_type_vote(type_votes, object_text, object_type)
            elif raw.object_type:
                LOGGER.debug("Unrecognized object type '%s' for %s", raw.object_type, object_text)
            self._update_section_counts(
                section_counts,
                element.section,
                subject_text,
                object_text,
            )
        section_distribution = {entity: dict(counts) for entity, counts in section_counts.items()}
        entity_type_votes = {entity: dict(counts) for entity, counts in type_votes.items()}
        return accepted, section_distribution, relation_verbatims, entity_type_votes

    def _resolve_domain(
        self,
        metadata: Optional[PaperMetadata],
        element: ParsedElement,
    ) -> Optional[ExtractionDomain]:
        """Determine the extraction domain for the provided context."""

        try:
            return self._domain_router.resolve(metadata=metadata, element=element)
        except Exception:  # noqa: BLE001 - fail soft while recording context
            LOGGER.exception("Domain resolution failed for %s", element.element_id)
            return None

    def _resolve_supportive_sentence(
        self,
        raw: RawLLMTriple,
        element: ParsedElement,
        subject_text: str,
        object_text: str,
    ) -> Optional[str]:
        """Resolve supportive sentence text for a raw triple.

        Args:
            raw: Raw triple returned by the LLM.
            element: Parsed element containing the source content.
            subject_text: Trimmed subject string.
            object_text: Trimmed object string.

        Returns:
            Optional[str]: Sentence containing both entities if available.
        """

        if raw.supportive_sentence:
            candidate = raw.supportive_sentence.strip()
            if candidate:
                return candidate
        return self._find_sentence_in_content(
            element.content,
            subject_text,
            object_text,
        )

    def _find_sentence_in_content(
        self,
        content: str,
        subject_text: str,
        object_text: str,
    ) -> Optional[str]:
        """Locate a sentence in the element content containing both entities.

        Args:
            content: Element text to search.
            subject_text: Trimmed subject string.
            object_text: Trimmed object string.

        Returns:
            Optional[str]: Matching sentence if found, otherwise None.
        """

        subject_lower = subject_text.lower()
        object_lower = object_text.lower()
        matches = list(re.finditer(r"[^.!?]+(?:[.!?]+|$)", content, flags=re.MULTILINE))
        for match in matches:
            sentence = match.group(0)
            if self._sentence_contains_entities(sentence, subject_lower, object_lower):
                return sentence.strip()
        return None

    @staticmethod
    def _sentence_contains_entities(
        sentence: str,
        subject_lower: str,
        object_lower: str,
    ) -> bool:
        """Return whether both entities appear in the candidate sentence.

        Args:
            sentence: Candidate sentence to inspect.
            subject_lower: Lowercase subject string.
            object_lower: Lowercase object string.

        Returns:
            bool: True when both entities are present, otherwise False.
        """

        normalized = sentence.lower()
        return subject_lower in normalized and object_lower in normalized

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

    def _normalize_entity_type(self, entity_type: Optional[str]) -> Optional[str]:
        """Normalize an entity type emitted by the LLM response."""

        if entity_type is None:
            return None
        cleaned = str(entity_type).strip().lower()
        if not cleaned:
            return None
        return self._entity_type_lookup.get(cleaned)

    @staticmethod
    def _is_ambiguous_type(normalized: Optional[str]) -> bool:
        if normalized is None:
            return True
        return normalized.strip().lower() in _AMBIGUOUS_ENTITY_TYPES

    @staticmethod
    def _increment_type_vote(
        accumulator: Dict[str, Dict[str, int]], entity: str, entity_type: str
    ) -> None:
        """Accumulate a vote for an entity type."""

        entity_votes = accumulator.setdefault(entity, {})
        entity_votes[entity_type] = entity_votes.get(entity_type, 0) + 1

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
        budget_cap = max(
            1,
            self._config.extraction.openai_max_output_tokens
            // self._config.extraction.tokens_per_triple,
        )
        return min(self._config.extraction.max_triples_per_chunk_base, estimate, budget_cap)

    def _find_span(
        self,
        text: str,
        target: str,
        *,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """Locate the character span of the target text within the source.

        Args:
            text: Source text to search within.
            target: Target substring to locate.
            threshold: Optional fuzzy matching threshold override.

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
        effective_threshold = (
            self._default_threshold if threshold is None else max(0.0, min(1.0, threshold))
        )
        return self._fuzzy_find_span(text, target, effective_threshold)

    def _fuzzy_find_span(
        self,
        text: str,
        target: str,
        threshold: float,
    ) -> Optional[Tuple[int, int]]:
        """Perform fuzzy matching to locate a target span.

        Args:
            text: Source text to search within.
            target: Target substring to locate approximately.
            threshold: Similarity ratio required to accept a match.

        Returns:
            Optional[Tuple[int, int]]: Inclusive-exclusive span if a close match is found.
        """

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


@lru_cache(maxsize=1)
def _default_relation_patterns() -> Tuple[Tuple[str, str, bool], ...]:
    """Return cached relation patterns derived from the configuration."""

    config = load_config()
    return tuple(config.relations.normalized_patterns())


def _relation_patterns(config: Optional[AppConfig]) -> Sequence[Tuple[str, str, bool]]:
    """Return relation normalization patterns for the provided configuration."""

    if config is not None:
        return config.relations.normalized_patterns()
    return _default_relation_patterns()


def normalize_relation(relation_text: str, *, config: Optional[AppConfig] = None) -> Tuple[str, bool]:
    """Normalize a relation phrase to the canonical predicate name.

    Args:
        relation_text: Relation phrase returned by the LLM.
        config: Optional configuration overriding the global defaults.

    Returns:
        Tuple[str, bool]: Canonical predicate and whether subject/object should swap.

    Raises:
        ValueError: If the relation cannot be mapped to a canonical predicate.
    """

    cleaned = re.sub(r"[^a-z]+", " ", relation_text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for phrase, normalized, swap in _relation_patterns(config):
        if phrase in cleaned:
            return normalized, swap
    raise ValueError(f"Unsupported relation phrase: {relation_text}")


__all__ = [
    "RawLLMTriple",
    "ExtractionResult",
    "LLMExtractor",
    "OpenAIExtractor",
    "TwoPassTripletExtractor",
    "normalize_relation",
    "spans_overlap",
]



