"""Tests for the two-pass triplet extraction pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import pytest

from backend.app.config import load_config
from backend.app.contracts import ParsedElement
from backend.app.extraction.triplet_extraction import (
    ExtractionResult,
    LLMExtractor,
    OpenAIExtractor,
    RawLLMTriple,
    TwoPassTripletExtractor,
    normalize_relation,
)


@dataclass
class _FakeResponse:
    """Minimal response object emulating the extractor HTTP response."""

    status_code: int
    payload: dict

    def json(self) -> dict:
        """Return the stored payload."""

        return self.payload

    @property
    def text(self) -> str:
        """Return the payload serialized as JSON."""

        return json.dumps(self.payload)


class _FakeHTTPClient:
    """Deterministic HTTP client used to simulate OpenAI responses."""

    def __init__(self, handler: Callable[[str, dict, dict], _FakeResponse]) -> None:
        self._handler = handler
        self.closed = False

    def post(self, path: str, *, headers: dict, json: dict) -> _FakeResponse:
        return self._handler(path, headers, json)

    def close(self) -> None:
        self.closed = True


FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden" / "phase_3"


@pytest.fixture(name="config")
def fixture_config():
    """Load application configuration for extraction tests."""

    return load_config()


@dataclass(frozen=True)
class _StubExtractor(LLMExtractor):
    """LLM extractor that returns pre-seeded triples for tests."""

    triples: Sequence[RawLLMTriple]

    def extract_triples(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]],
        max_triples: int,
    ) -> Sequence[RawLLMTriple]:
        """Return the configured triples regardless of inputs."""

        return list(self.triples)[:max_triples]


def _element_from_fixture(payload: dict) -> ParsedElement:
    """Build a parsed element from a fixture payload."""

    return ParsedElement(**payload)


def _load_golden(name: str) -> dict:
    """Load a golden fixture from disk."""

    with (GOLDEN_DIR / f"{name}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_normalize_relation_rejects_unknown_relation() -> None:
    """Unknown relation phrases should raise a validation error."""

    with pytest.raises(ValueError):
        normalize_relation("collaborates with")


def test_triplet_with_missing_span_is_rejected(config) -> None:
    """Triples without resolvable spans should be dropped."""

    element = ParsedElement(
        doc_id="doc-1",
        element_id="doc-1:0",
        section="Results",
        content="Graph neural networks achieve strong accuracy.",
        content_hash="a" * 64,
        start_char=0,
        end_char=44,
    )
    extractor = _StubExtractor(
        triples=[
            RawLLMTriple(
                subject_text="Transformer",  # not present
                relation_verbatim="uses",
                object_text="attention",
                supportive_sentence="Graph neural networks achieve strong accuracy.",
                confidence=0.9,
            )
        ]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    triples = pipeline.extract_from_element(element, candidate_entities=None)

    assert triples == []


def test_passive_voice_flips_subject_and_object(config) -> None:
    """Passive voice relations should swap subject and object."""

    content = "The dataset is used by the model to improve accuracy."
    element = ParsedElement(
        doc_id="doc-2",
        element_id="doc-2:0",
        section="Methods",
        content=content,
        content_hash="b" * 64,
        start_char=0,
        end_char=len(content),
    )
    extractor = _StubExtractor(
        triples=[
            RawLLMTriple(
                subject_text="The dataset",
                relation_verbatim="is used by",
                object_text="the model",
                supportive_sentence=content,
                confidence=0.92,
            )
        ]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    triples = pipeline.extract_from_element(element, candidate_entities=None)

    assert len(triples) == 1
    triple = triples[0]
    assert triple.subject == "the model"
    assert triple.object == "The dataset"
    assert triple.predicate == "uses"


def test_golden_triplet_extraction_matches_fixture(config) -> None:
    """Two-pass extraction should reproduce the golden fixture output."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    triples_payload = [
        RawLLMTriple(**triple)
        for triple in fixture["llm_response"]["triples"]
    ]
    extractor = _StubExtractor(triples=triples_payload)
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    result = pipeline.extract_with_metadata(
        element,
        candidate_entities=fixture.get("candidate_entities"),
    )

    assert isinstance(result, ExtractionResult)
    assert [trip.model_dump() for trip in result.triplets] == fixture["expected"]
    assert result.section_distribution == fixture["expected_section_distribution"]


def test_extract_from_element_returns_triplets_only(config) -> None:
    """The legacy extract_from_element API should return only triplets."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    extractor = _StubExtractor(
        triples=[RawLLMTriple(**fixture["llm_response"]["triples"][0])]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    extracted = pipeline.extract_from_element(element, candidate_entities=None)

    assert isinstance(extracted, list)
    assert len(extracted) == 1


def test_openai_extractor_parses_valid_response(config) -> None:
    """The OpenAI extractor should convert a successful response into triples."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])

    settings = config.extraction.openai

    def handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        assert path == "/chat/completions"
        assert payload["model"] == settings.model
        assert payload["messages"][0]["role"] == "system"
        assert "at most" in payload["messages"][0]["content"]
        assert headers["Authorization"] == "Bearer test-key"
        assert "Candidate entities" in payload["messages"][1]["content"]
        response_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(fixture["llm_response"]),
                    }
                }
            ]
        }
        return _FakeResponse(status_code=200, payload=response_body)

    client = _FakeHTTPClient(handler)
    extractor = OpenAIExtractor(
        settings=settings,
        api_key="test-key",
        client=client,
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    result = pipeline.extract_with_metadata(
        element,
        candidate_entities=fixture.get("candidate_entities"),
    )

    assert [trip.model_dump() for trip in result.triplets] == fixture["expected"]
    assert result.section_distribution == fixture["expected_section_distribution"]


def test_openai_extractor_raises_on_error_response(config) -> None:
    """Non-successful OpenAI responses should raise a runtime error."""

    element = ParsedElement(
        doc_id="doc-err",
        element_id="doc-err:0",
        section="Intro",
        content="Sample text",
        content_hash="d" * 64,
        start_char=0,
        end_char=11,
    )

    settings = config.extraction.openai

    def handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        assert path == "/chat/completions"
        return _FakeResponse(status_code=500, payload={"error": {"message": "boom"}})

    client = _FakeHTTPClient(handler)
    extractor = OpenAIExtractor(
        settings=settings,
        api_key="test-key",
        client=client,
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    with pytest.raises(RuntimeError):
        pipeline.extract_from_element(element, candidate_entities=None)

