"""Tests for the two-pass triplet extraction pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pytest

from backend.app.config import load_config
from backend.app.contracts import ParsedElement
from backend.app.extraction.triplet_extraction import (
    LLMExtractor,
    RawLLMTriple,
    TwoPassTripletExtractor,
    normalize_relation,
)


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

    extracted = pipeline.extract_from_element(element, candidate_entities=fixture.get("candidate_entities"))

    assert [trip.model_dump() for trip in extracted] == fixture["expected"]

