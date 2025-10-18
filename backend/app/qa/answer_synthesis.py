"""LLM-backed answer synthesis built on top of QA graph retrieval."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from backend.app.config import OpenAIConfig, QALLMConfig

LOGGER = logging.getLogger(__name__)

_ENDPOINT_CHAT_COMPLETIONS = "/chat/completions"


@dataclass(frozen=True)
class AnswerContextEdge:
    """Single edge used as evidence for answer synthesis."""

    subject: str
    predicate: str
    obj: str
    doc_id: str
    element_id: str
    confidence: float
    relation_verbatim: str
    sentence: Optional[str] = None


@dataclass(frozen=True)
class AnswerSynthesisRequest:
    """Payload describing the QA retrieval context."""

    question: str
    mode: str
    graph_paths: Sequence[Sequence[AnswerContextEdge]]
    neighbor_edges: Sequence[AnswerContextEdge]
    scope_documents: Sequence[str]
    pipeline_version: str


@dataclass(frozen=True)
class AnswerSynthesisResult:
    """Natural language answer produced by the LLM."""

    answer: str
    raw_response: Mapping[str, object]


class _HTTPClient(Protocol):
    """Minimal protocol for posting JSON payloads."""

    def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, object]) -> "_HTTPResponse":
        """Issue a POST request."""


@dataclass
class _HTTPResponse:
    """Simplified HTTP response wrapper."""

    status_code: int
    _content: bytes

    def json(self) -> Mapping[str, object]:
        """Decode the body as JSON."""

        text = self._content.decode("utf-8", errors="replace")
        return json.loads(text)


class _UrllibHTTPClient:
    """HTTP client implemented with urllib to avoid extra dependencies."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, object]) -> _HTTPResponse:
        """Send a POST request to the configured base URL."""

        url = f"{self._base_url}{path}"
        payload = json_module.dumps(json).encode("utf-8")
        merged_headers = {"Content-Type": "application/json", **headers}
        request = urllib_request.Request(url, data=payload, headers=merged_headers, method="POST")
        try:
            with urllib_request.urlopen(request, timeout=self._timeout) as response:
                content = response.read()
                status = response.getcode() or 0
                return _HTTPResponse(status_code=status, _content=content)
        except urllib_error.HTTPError as exc:  # pragma: no cover - error path exercised in tests
            content = exc.read()
            return _HTTPResponse(status_code=exc.code, _content=content)


json_module = json


class LLMAnswerSynthesizer:
    """Orchestrates LLM calls for grounded QA answers."""

    _SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert scientific assistant. Follow prompt version {prompt_version}.\n"
        "Only use the provided evidence when answering. Never invent facts or rely on external knowledge.\n"
        "Write concise, natural language answers. Include inline citations as [doc_id:element_id] for every claim.\n"
        "If the evidence is insufficient or conflicting, reply exactly with 'Insufficient evidence to answer.'\n"
        "Explain reasoning before listing citations and enumerate key supporting facts."
    )

    def __init__(
        self,
        *,
        llm_config: QALLMConfig,
        api_key: Optional[str],
        http_client: Optional[_HTTPClient] = None,
    ) -> None:
        self._config = llm_config
        self._settings: OpenAIConfig = llm_config.openai
        self._api_key = api_key or ""
        self._enabled = bool(llm_config.enabled)
        self._http_client = http_client or _UrllibHTTPClient(
            self._settings.api_base,
            self._settings.timeout_seconds,
        )

    @property
    def enabled(self) -> bool:
        """Return whether LLM synthesis is enabled in configuration."""

        return self._enabled

    def synthesize(self, request: AnswerSynthesisRequest) -> Optional[AnswerSynthesisResult]:
        """Produce a grounded natural language answer when enabled.

        Args:
            request: QA retrieval context including paths and neighbor evidence.

        Returns:
            Optional[AnswerSynthesisResult]: Synthesized answer or None when disabled or unavailable.
        """

        if not self._enabled:
            LOGGER.debug("QA LLM synthesis disabled via configuration")
            return None
        if not self._api_key:
            LOGGER.warning("QA LLM synthesis requested but no API key provided")
            return None

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(request)},
        ]
        payload: Dict[str, object] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
        }

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            response = self._http_client.post(
                _ENDPOINT_CHAT_COMPLETIONS,
                headers=headers,
                json=payload,
            )
        except Exception:  # noqa: BLE001 - external dependency failures should not crash QA
            LOGGER.exception("QA LLM synthesis request failed before reaching the API")
            return None

        if response.status_code >= 400:
            LOGGER.error("QA LLM synthesis failed with status %s", response.status_code)
            return None

        try:
            data = response.json()
        except Exception:  # noqa: BLE001 - invalid JSON
            LOGGER.exception("QA LLM synthesis response was not valid JSON")
            return None

        answer = _extract_answer_text(data)
        if answer is None:
            LOGGER.error("QA LLM synthesis response missing content")
            return None
        return AnswerSynthesisResult(answer=answer, raw_response=data)

    def _build_system_prompt(self) -> str:
        """Render the system prompt with the configured version identifier."""

        return self._SYSTEM_PROMPT_TEMPLATE.format(prompt_version=self._settings.prompt_version)

    def _build_user_prompt(self, request: AnswerSynthesisRequest) -> str:
        """Construct the user prompt describing evidence and constraints."""

        lines: List[str] = [
            f"Question: {request.question}",
            f"Mode: {request.mode}",
            f"pipeline_version: {request.pipeline_version}",
        ]
        if request.scope_documents:
            lines.append("Scope documents: " + ", ".join(request.scope_documents))
        else:
            lines.append("Scope documents: all available")

        if request.graph_paths:
            lines.append("Graph paths:")
            for path_index, path in enumerate(request.graph_paths, start=1):
                lines.append(f"  Path {path_index}:")
                if not path:
                    lines.append("    (no edges)")
                    continue
                for edge_index, edge in enumerate(path, start=1):
                    core = (
                        f"    Edge {edge_index}: {edge.subject} --{edge.predicate}--> {edge.obj} "
                        f"(doc_id: {edge.doc_id}, element_id: {edge.element_id}, confidence: {edge.confidence:.2f})"
                    )
                    lines.append(core)
                    lines.append(f"      relation_verbatim: {edge.relation_verbatim}")
                    if edge.sentence:
                        lines.append(f"      sentence: {edge.sentence}")
        else:
            lines.append("Graph paths: None")

        if request.neighbor_edges:
            lines.append("Neighbor edges:")
            for edge_index, edge in enumerate(request.neighbor_edges, start=1):
                core = (
                    f"  Neighbor {edge_index}: {edge.subject} --{edge.predicate}--> {edge.obj} "
                    f"(doc_id: {edge.doc_id}, element_id: {edge.element_id}, confidence: {edge.confidence:.2f})"
                )
                lines.append(core)
                lines.append(f"    relation_verbatim: {edge.relation_verbatim}")
                if edge.sentence:
                    lines.append(f"    sentence: {edge.sentence}")
        else:
            lines.append("Neighbor edges: None")

        lines.append("Instructions:")
        lines.append(
            "- Cite using [doc_id:element_id] tokens matching supplied evidence."
        )
        lines.append("- Do not introduce entities, facts, or documents that are not listed above.")
        lines.append("- If evidence is insufficient, respond with 'Insufficient evidence to answer.'")
        return "\n".join(lines)


def _extract_answer_text(payload: Mapping[str, object]) -> Optional[str]:
    """Safely extract the assistant message content from the OpenAI response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content.strip()
