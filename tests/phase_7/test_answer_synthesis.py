from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence

import pytest

from backend.app.config import QALLMConfig
from backend.app.qa.answer_synthesis import (
    AnswerContextEdge,
    AnswerSynthesisRequest,
    LLMAnswerSynthesizer,
)


class _StubHTTPClient:
    """Collects outgoing payloads and returns a canned response."""

    def __init__(
        self,
        response_text: str = "Mock answer with citations.",
        *,
        stream_chunks: Optional[Sequence[str]] = None,
    ) -> None:
        self.calls: List[Dict[str, object]] = []
        self.stream_calls: List[Dict[str, object]] = []
        self._response_text = response_text
        self._stream_chunks = (
            [chunk.encode("utf-8") for chunk in stream_chunks] if stream_chunks is not None else None
        )

    def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, object]) -> SimpleNamespace:
        self.calls.append({"path": path, "headers": headers, "json": json})
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": self._response_text,
                        }
                    }
                ]
            },
        )

    def post_stream(
        self,
        path: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, object],
    ) -> Iterator[bytes]:
        self.stream_calls.append({"path": path, "headers": headers, "json": json})
        if self._stream_chunks is None:
            raise RuntimeError("Streaming not configured for stub client.")
        return iter(self._stream_chunks)


def _qallm_config(**overrides: object) -> QALLMConfig:
    """Helper to build a QALLMConfig with sensible defaults for tests."""

    base = {
        "enabled": True,
        "provider": "openai",
        "openai_model": "gpt-4o-mini",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_timeout_seconds": 60,
        "openai_prompt_version": "qa-v1",
        "openai_max_retries": 2,
        "openai_temperature": 0.1,
        "openai_max_output_tokens": 600,
        "openai_initial_output_multiplier": 1.5,
        "openai_backoff_initial_seconds": 0.5,
        "openai_backoff_max_seconds": 4.0,
        "openai_retry_statuses": [429, 500, 502, 503, 504],
        "allow_fallback_without_evidence": True,
    }
    base.update(overrides)
    return QALLMConfig(**base)


def _sample_request() -> AnswerSynthesisRequest:
    edges = [
        AnswerContextEdge(
            subject="Transformer encoder",
            predicate="builds-upon",
            obj="Self-attention layers",
            doc_id="paper-1",
            element_id="sec-2",
            confidence=0.87,
            relation_verbatim="builds-upon",
            sentence="The transformer encoder builds upon self-attention layers to capture context.",
        )
    ]
    return AnswerSynthesisRequest(
        question="How does the transformer encoder relate to self-attention?",
        mode="direct",
        graph_paths=[edges],
        neighbor_edges=[],
        scope_documents=["paper-1"],
        pipeline_version="1.0.0",
        has_graph_evidence=True,
        allow_off_graph_answer=True,
    )


def test_disabled_synthesizer_returns_none() -> None:
    config = _qallm_config(enabled=False)
    synthesizer = LLMAnswerSynthesizer(
        llm_config=config,
        api_key="test-key",
        http_client=_StubHTTPClient(),
    )
    result = synthesizer.synthesize(_sample_request())
    assert result is None


def test_prompt_includes_graph_context_and_metadata() -> None:
    stub_client = _StubHTTPClient()
    synthesizer = LLMAnswerSynthesizer(
        llm_config=_qallm_config(),
        api_key="test-key",
        http_client=stub_client,
    )
    request = _sample_request()
    response = synthesizer.synthesize(request)

    assert response is not None
    assert response.answer == "Mock answer with citations."
    assert len(stub_client.calls) == 1

    payload = stub_client.calls[0]
    assert payload["path"] == "/chat/completions"
    headers = payload["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    body = payload["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 600

    messages = body["messages"]
    assert messages[0]["role"] == "system"
    assert "qa-v1" in messages[0]["content"]
    assert "When graph evidence is supplied" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert "Transformer encoder" in user_content
    assert "doc_id: paper-1" in user_content
    assert "pipeline_version: 1.0.0" in user_content


def test_streaming_emits_chunks_and_returns_final_answer() -> None:
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}'.rstrip() + "\n\n",
        'data: {"choices":[{"delta":{"content":" world"}}]}'.rstrip() + "\n\n",
        "data: [DONE]\n\n",
    ]
    stub_client = _StubHTTPClient(stream_chunks=chunks)
    synthesizer = LLMAnswerSynthesizer(
        llm_config=_qallm_config(),
        api_key="test-key",
        http_client=stub_client,
    )
    request = _sample_request()
    stream = synthesizer.stream(request)
    assert stream is not None
    collected: List[str] = []
    final_answer: Optional[str] = None
    try:
        while True:
            collected.append(next(stream))
    except StopIteration as exc:  # noqa: PERF203 - intentional control flow for generator result
        final_answer = exc.value

    assert collected == ["Hello", " world"]
    assert final_answer == "Hello world"
    assert stub_client.calls == []
    assert len(stub_client.stream_calls) == 1


def test_http_failure_returns_none_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClient(_StubHTTPClient):
        def post(self, path: str, *, headers: Dict[str, str], json: Dict[str, object]) -> SimpleNamespace:  # type: ignore[override]
            raise RuntimeError("boom")

    synthesizer = LLMAnswerSynthesizer(
        llm_config=_qallm_config(),
        api_key="test-key",
        http_client=FailingClient(),
    )
    result = synthesizer.synthesize(_sample_request())
    assert result is None
