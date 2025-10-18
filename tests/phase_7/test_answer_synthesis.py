from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

import pytest

from backend.app.config import QALLMConfig
from backend.app.qa.answer_synthesis import (
    AnswerContextEdge,
    AnswerSynthesisRequest,
    LLMAnswerSynthesizer,
)


class _StubHTTPClient:
    """Collects outgoing payloads and returns a canned response."""

    def __init__(self, response_text: str = "Mock answer with citations.") -> None:
        self.calls: List[Dict[str, object]] = []
        self._response_text = response_text

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
    assert "Only use the provided evidence" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert "Transformer encoder" in user_content
    assert "doc_id: paper-1" in user_content
    assert "pipeline_version: 1.0.0" in user_content


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
