"""Tests for the two-pass triplet extraction pipeline."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import pytest

from backend.app.config import load_config
from backend.app.contracts import ParsedElement
from backend.app.extraction.cache import LLMResponseCache, TokenBudgetCache
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


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("This paper introduces the concept of diffusion models.", "introduces-concept-of"),
        ("BERT extends the architecture of Transformer.", "extends-architecture-of"),
        ("We propose DiffusionFormer.", "proposes"),
        ("Our approach builds upon the ResNet backbone.", "builds-upon"),
        ("The experiment demonstrates that the inhibitor works.", "demonstrates"),
        ("Model A was benchmarked against Model B.", "benchmarked-against"),
        ("The pipeline depends on high-quality pretraining data.", "depends-on"),
        ("The hypothesis was validated by a double-blind study.", "validated-by"),
        ("Our system achieves state of the art on ImageNet.", "achieves-state-of-the-art-on"),
        ("The algorithm is implemented in PyTorch.", "implemented-in"),
        ("This procedure requires sterile equipment.", "requires"),
        ("ATP is required for the contraction response.", "requires-for"),
        ("Calcium influx enables neurotransmitter release.", "enables"),
        ("Secondary metabolites mediate the plant response.", "mediates"),
        ("Sample X is an instance of the MNIST dataset.", "instance-of"),
        ("The attention layer is a component of the encoder.", "component-of"),
        ("Transformer belongs to natural language processing.", "belongs-to"),
        ("The inhibitor suppresses the kinase and inhibits phosphorylation.", "inhibits"),
        ("cAMP activates protein kinase A.", "activates"),
        ("Protein X binds to DNA.", "binds-to"),
        ("Enzyme A catalyzes Reaction B.", "catalyzes"),
        ("Biomarker Y is present in liver tissue.", "present-in"),
        ("Protein complexes interact with the signaling pathway.", "interacts-with"),
        ("The detector observes gravitational waves.", "observes"),
        ("The interferometer detects the passing signal.", "detects"),
        ("The instrument measures magnetic flux.", "measures"),
        ("The parameter is inferred from spectral data.", "infers-from"),
        ("The collision results in high-energy photons.", "results-in"),
        ("Alloy Z is composed of iron and nickel.", "composed-of"),
        ("Nanotubes are synthesized from methane feedstock.", "synthesized-from"),
        ("Sodium chloride crystallizes as a cubic lattice.", "crystallizes-as"),
        ("Isotope ratios are derived from mass spectrometry.", "derived-from"),
    ],
)
def test_normalize_relation_accepts_new_relations(config, phrase: str, expected: str) -> None:
    """Newly added canonical relations should normalize successfully."""

    canonical, _ = normalize_relation(phrase, config=config)
    assert canonical == expected


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
                subject_type="Method",
                relation_verbatim="uses",
                object_text="attention",
                object_type="Method",
                supportive_sentence="Graph neural networks achieve strong accuracy.",
                confidence=0.9,
            )
        ]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    triples = pipeline.extract_from_element(element, candidate_entities=None)

    assert triples == []


def test_triplet_with_self_reference_is_rejected(config) -> None:
    """Triples whose subject and object are identical must be dropped."""

    content = "Gradient descent optimizes gradient descent for this task."
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
                subject_text="gradient descent",
                subject_type="Method",
                relation_verbatim="uses",
                object_text="gradient descent",
                object_type="Method",
                supportive_sentence=content,
                confidence=0.82,
            )
        ]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    triples = pipeline.extract_from_element(element, candidate_entities=None)

    assert triples == []


def test_triplet_with_ambiguous_type_is_rejected(config) -> None:
    """Triples should be skipped when an entity is reported with an ambiguous type."""

    content = "The MNIST dataset is evaluated using accuracy."
    element = ParsedElement(
        doc_id="doc-3",
        element_id="doc-3:0",
        section="Results",
        content=content,
        content_hash="c" * 64,
        start_char=0,
        end_char=len(content),
    )
    extractor = _StubExtractor(
        triples=[
            RawLLMTriple(
                subject_text="MNIST dataset",
                subject_type="Other",
                relation_verbatim="evaluated on",
                object_text="accuracy",
                object_type="Metric",
                supportive_sentence=content,
                confidence=0.76,
            )
        ]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    triples = pipeline.extract_from_element(element, candidate_entities=None)

    assert triples == []


def test_missing_supportive_sentence_falls_back_to_element_content(config) -> None:
    """Triples without supportive sentences should reuse the element sentence."""

    content = "Model A is compared to Model B in the study. Additional notes on results."
    element = ParsedElement(
        doc_id="doc-missing-support",
        element_id="doc-missing-support:0",
        section="Results",
        content=content,
        content_hash="c" * 64,
        start_char=0,
        end_char=len(content),
    )
    extractor = _StubExtractor(
        triples=[
            RawLLMTriple(
                subject_text="Model A",
                subject_type="Method",
                relation_verbatim="compared-to",
                object_text="Model B",
                object_type="Method",
                supportive_sentence=None,
                confidence=0.81,
            )
        ]
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    triples = pipeline.extract_from_element(element, candidate_entities=None)

    assert len(triples) == 1
    assert triples[0].evidence.full_sentence.strip() == "Model A is compared to Model B in the study."


def test_missing_supportive_sentence_is_dropped_when_no_sentence_found(config) -> None:
    """Triples remain skipped when entities never share a sentence."""

    content = "Model A improves accuracy substantially. Separately, Model B enhances recall."
    element = ParsedElement(
        doc_id="doc-missing-support",
        element_id="doc-missing-support:1",
        section="Discussion",
        content=content,
        content_hash="d" * 64,
        start_char=0,
        end_char=len(content),
    )
    extractor = _StubExtractor(
        triples=[
            RawLLMTriple(
                subject_text="Model A",
                subject_type="Method",
                relation_verbatim="compared-to",
                object_text="Model B",
                object_type="Method",
                supportive_sentence=None,
                confidence=0.8,
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
                subject_type="Dataset",
                relation_verbatim="is used by",
                object_text="the model",
                object_type="Method",
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
    assert result.relation_verbatims == [
        item["relation_verbatim"] for item in fixture["llm_response"]["triples"]
    ]
    assert result.entity_type_votes == {
        "The Reinforcement Learning agent": {"Method": 2},
        "the PPO algorithm": {"Method": 1},
        "baseline methods": {"Method": 1},
    }


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
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
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
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
    )
    pipeline = TwoPassTripletExtractor(config=config, llm_extractor=extractor)

    with pytest.raises(RuntimeError):
        pipeline.extract_from_element(element, candidate_entities=None)


def test_openai_extractor_limits_candidate_entities(config) -> None:
    """LLM prompt should include at most the configured number of candidate entities."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    settings = config.extraction.openai
    candidate_entities = [f"Entity {idx}" for idx in range(20)]
    observed_entities: list[str] = []

    def handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        assert path == "/chat/completions"
        user_lines = payload["messages"][1]["content"].splitlines()
        try:
            start = user_lines.index("Candidate entities that may appear in the chunk:") + 1
        except ValueError:  # pragma: no cover - defensive guard
            pytest.fail("Candidate entities section missing from user prompt")
        for line in user_lines[start:]:
            if not line.startswith("- "):
                break
            observed_entities.append(line)
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
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
    )

    triples = extractor.extract_triples(element, candidate_entities=candidate_entities, max_triples=5)

    assert len(observed_entities) == config.extraction.max_prompt_entities
    assert observed_entities[0] == "- Entity 0"
    assert observed_entities[-1] == f"- Entity {config.extraction.max_prompt_entities - 1}"
    assert len(triples) == len(fixture["llm_response"]["triples"])


def test_system_prompt_emphasizes_type_and_grounding_rules(config) -> None:
    """The system prompt should enforce grounding and entity typing constraints."""

    settings = config.extraction.openai

    client = _FakeHTTPClient(lambda path, headers, payload: _FakeResponse(status_code=200, payload={"choices": []}))
    extractor = OpenAIExtractor(
        settings=settings,
        api_key="test-key",
        client=client,
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
    )

    prompt = extractor._render_system_prompt(max_triples=5)  # type: ignore[attr-defined]

    assert "Subjects and objects must be different" in prompt
    assert "Use only the provided chunk" in prompt
    assert "Discard the triple" in prompt
    assert "Gradient descent" in prompt


def test_openai_extractor_uses_initial_token_multiplier(config) -> None:
    """OpenAI extractor should start with the configured token multiplier."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    settings = config.extraction.openai
    observed_tokens: list[int] = []

    def handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        observed_tokens.append(payload["max_tokens"])
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
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
    )

    extractor.extract_triples(element, candidate_entities=None, max_triples=3)

    expected_tokens = min(
        settings.max_output_tokens,
        config.extraction.tokens_per_triple
        * 3
        * math.ceil(settings.initial_output_multiplier),
    )
    assert observed_tokens == [expected_tokens]


def test_openai_extractor_logs_configured_multiplier_without_retry(caplog, config) -> None:
    """Extractor should not report a token increase when starting above 1."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    settings = config.extraction.openai

    def handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
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
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
    )

    with caplog.at_level(logging.INFO, logger="backend.app.extraction.triplet_extraction"):
        extractor.extract_triples(element, candidate_entities=None, max_triples=3)

    messages = [record.message for record in caplog.records]
    assert any(
        "using configured token budget multiplier" in message for message in messages
    )
    assert not any(
        "after increasing token budget multiplier" in message for message in messages
    )


def test_openai_extractor_uses_cached_response(tmp_path, config) -> None:
    """Repeated extraction should reuse cached LLM responses instead of reissuing requests."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    settings = config.extraction.openai

    cache_dir = tmp_path / "cache"
    response_cache_path = cache_dir / "responses.json"
    token_cache_path = cache_dir / "token.json"

    response_cache = LLMResponseCache(response_cache_path)
    token_cache = TokenBudgetCache(token_cache_path)

    def _payload() -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(fixture["llm_response"]),
                    }
                }
            ]
        }

    call_count = 0

    def first_handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        return _FakeResponse(status_code=200, payload=_payload())

    client_one = _FakeHTTPClient(first_handler)
    extractor_one = OpenAIExtractor(
        settings=settings,
        api_key="test-key",
        client=client_one,
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
        response_cache=response_cache,
        token_cache=token_cache,
    )

    triples_first = extractor_one.extract_triples(element, candidate_entities=None, max_triples=5)
    assert call_count == 1
    assert len(triples_first) == len(fixture["llm_response"]["triples"])

    # Reinstantiate caches to emulate a fresh process reading persisted files.
    response_cache_reloaded = LLMResponseCache(response_cache_path)
    token_cache_reloaded = TokenBudgetCache(token_cache_path)

    def second_handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        pytest.fail("LLM call should have been served from cache")

    client_two = _FakeHTTPClient(second_handler)
    extractor_two = OpenAIExtractor(
        settings=settings,
        api_key="test-key",
        client=client_two,
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
        response_cache=response_cache_reloaded,
        token_cache=token_cache_reloaded,
    )

    triples_second = extractor_two.extract_triples(element, candidate_entities=None, max_triples=5)
    assert len(triples_second) == len(triples_first)


def test_openai_system_prompt_discourages_generic_entities(config) -> None:
    """System prompt should instruct the model to avoid generic or vague nodes."""

    fixture = _load_golden("sample_chunk")
    element = _element_from_fixture(fixture["element"])
    settings = config.extraction.openai
    captured_prompt: dict[str, str] = {}

    def handler(path: str, headers: dict, payload: dict) -> _FakeResponse:
        captured_prompt["system"] = payload["messages"][0]["content"]
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
        token_budget_per_triple=config.extraction.tokens_per_triple,
        allowed_relations=config.relations.canonical_relation_names(),
        entity_types=config.extraction.entity_types,
        max_prompt_entities=config.extraction.max_prompt_entities,
    )

    extractor.extract_triples(element, candidate_entities=None, max_triples=3)

    assert "Do not extract overly generic or vague terms" in captured_prompt["system"]
