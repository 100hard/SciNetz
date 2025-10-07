"""Tests for the OpenAIExtractor implementation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from backend.app.config import load_config
from backend.app.contracts import ParsedElement, Triplet
from backend.app.extraction import OpenAIExtractor, TwoPassTripletExtractor
import backend.app.extraction.triplet_extraction as extraction_module


@dataclass
class _FakeResponse:
    """Simplified HTTP response used for testing."""

    status_code: int
    text: str
    body: Optional[Dict[str, Any]] = None


class _FakeHTTPClient:
    """Deterministic HTTP client for unit tests."""

    def __init__(self, responses: List[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: List[Tuple[str, Dict[str, Any], Dict[str, str]]] = []

    def post(self, path: str, payload: Dict[str, Any], headers: Dict[str, str]) -> _FakeResponse:
        if not self._responses:
            raise AssertionError("No more fake responses configured")
        self.calls.append((path, payload, headers))
        return self._responses.pop(0)


@pytest.fixture(name="openai_settings")
def fixture_openai_settings():
    """Return the OpenAI configuration from the application config."""

    config = load_config()
    settings = config.extraction.openai
    assert settings is not None
    return settings


def test_openai_extractor_parses_successful_response(monkeypatch, openai_settings) -> None:
    """The extractor should parse a valid OpenAI response into raw triples."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    element = ParsedElement(
        doc_id="doc-42",
        element_id="doc-42:0",
        section="Methods",
        content="Model X uses Dataset Y to achieve strong accuracy.",
        content_hash="d" * 64,
        start_char=0,
        end_char=58,
    )
    response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "triples": [
                                {
                                    "subject_text": "Model X",
                                    "relation_verbatim": "uses",
                                    "object_text": "Dataset Y",
                                    "supportive_sentence": "Model X uses Dataset Y to achieve strong accuracy.",
                                    "confidence": 0.93,
                                }
                            ]
                        }
                    ),
                }
            }
        ]
    }
    client = _FakeHTTPClient(
        responses=[
            _FakeResponse(status_code=200, text=json.dumps(response_payload), body=response_payload)
        ]
    )
    extractor = OpenAIExtractor(
        settings=openai_settings,
        http_post=client.post,
        api_key="test-key",
    )

    result = extractor.extract_triples(
        element=element,
        candidate_entities=["Model X", "Dataset Y"],
        max_triples=2,
    )

    assert len(result) == 1
    triple = result[0]
    assert triple.subject_text == "Model X"
    assert triple.object_text == "Dataset Y"
    assert triple.relation_verbatim == "uses"
    assert triple.supportive_sentence.startswith("Model X uses")
    assert triple.confidence == pytest.approx(0.93)
    assert client.calls[0][0] == "/chat/completions"


def test_openai_extractor_retries_on_transient_failure(monkeypatch, openai_settings) -> None:
    """Retryable status codes should trigger exponential backoff before success."""

    updated_settings = openai_settings.model_copy(
        update={
            "max_retries": 1,
            "backoff_initial_seconds": 0.01,
            "backoff_max_seconds": 0.01,
            "retry_statuses": [429],
        }
    )
    success_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "triples": [
                                {
                                    "subject_text": "Subject",
                                    "relation_verbatim": "uses",
                                    "object_text": "Object",
                                    "supportive_sentence": "Subject uses Object.",
                                    "confidence": 0.88,
                                }
                            ]
                        }
                    ),
                }
            }
        ]
    }
    client = _FakeHTTPClient(
        responses=[
            _FakeResponse(status_code=429, text=json.dumps({"error": "Too Many"})),
            _FakeResponse(status_code=200, text=json.dumps(success_payload), body=success_payload),
        ]
    )
    sleep_calls: List[float] = []
    monkeypatch.setattr(extraction_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    extractor = OpenAIExtractor(settings=updated_settings, http_post=client.post, api_key="test-key")
    element = ParsedElement(
        doc_id="doc-99",
        element_id="doc-99:0",
        section="Results",
        content="Subject uses Object.",
        content_hash="e" * 64,
        start_char=0,
        end_char=20,
    )

    triples = extractor.extract_triples(element=element, candidate_entities=None, max_triples=1)

    assert len(client.calls) == 2
    assert sleep_calls == [pytest.approx(0.01)]
    assert len(triples) == 1


def test_openai_extractor_requires_api_key(monkeypatch, openai_settings) -> None:
    """Instantiating the extractor without an API key should fail fast."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        OpenAIExtractor(settings=openai_settings, http_post=_FakeHTTPClient([]).post)


def test_two_pass_pipeline_with_openai_extractor(monkeypatch, openai_settings) -> None:
    """TwoPassTripletExtractor should work end-to-end with the OpenAI adapter."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_config()
    fixture = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "triples": [
                                {
                                    "subject_text": "Subject",
                                    "relation_verbatim": "uses",
                                    "object_text": "Object",
                                    "supportive_sentence": "Subject uses Object.",
                                    "confidence": 0.7,
                                }
                            ]
                        }
                    ),
                }
            }
        ]
    }
    client = _FakeHTTPClient(
        responses=[
            _FakeResponse(status_code=200, text=json.dumps(fixture), body=fixture)
        ]
    )
    extractor = OpenAIExtractor(
        settings=openai_settings,
        http_post=client.post,
        api_key="test-key",
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)
    element = ParsedElement(
        doc_id="doc-77",
        element_id="doc-77:0",
        section="Discussion",
        content="Subject uses Object.",
        content_hash="f" * 64,
        start_char=0,
        end_char=20,
    )

    result = pipeline.extract_from_element(element, candidate_entities=None)

    assert len(result) == 1
    assert isinstance(result[0], Triplet)
    triple = result[0]
    assert triple.subject == "Subject"
    assert triple.object == "Object"
