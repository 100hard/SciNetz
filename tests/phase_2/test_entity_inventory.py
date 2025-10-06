from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List

import pytest

from backend.app.config import load_config
from backend.app.contracts import ParsedElement
from backend.app.extraction.entity_inventory import EntityInventoryBuilder


@dataclass(frozen=True)
class _FakeSpan:
    text: str


@dataclass(frozen=True)
class _FakeToken:
    text: str
    pos_: str = "NOUN"
    is_stop: bool = False

    @property
    def lemma_(self) -> str:  # pragma: no cover - simple passthrough
        return self.text.lower()


class _FakeDoc:
    def __init__(
        self,
        text: str,
        ents: Iterable[_FakeSpan],
        noun_chunks: Iterable[_FakeSpan],
        tokens: Iterable[_FakeToken],
    ) -> None:
        self.text = text
        self._ents = list(ents)
        self._noun_chunks = list(noun_chunks)
        self._tokens = list(tokens)

    @property
    def ents(self) -> List[_FakeSpan]:
        return self._ents

    @property
    def noun_chunks(self) -> Iterable[_FakeSpan]:
        return iter(self._noun_chunks)

    @property
    def sents(self) -> Iterable[_FakeSpan]:
        return iter([_FakeSpan(self.text)])

    def __iter__(self):  # pragma: no cover - exercised in tests
        return iter(self._tokens)


def _tokenize(text: str, stopwords: set[str], proper_nouns: set[str]) -> List[_FakeToken]:
    tokens: List[_FakeToken] = []
    for raw in re.findall(r"[A-Za-z0-9-]+", text):
        lower = raw.lower()
        pos = "PROPN" if raw in proper_nouns else "NOUN"
        tokens.append(_FakeToken(text=raw, pos_=pos, is_stop=lower in stopwords))
    return tokens


@pytest.fixture(name="config")
def fixture_config():
    return load_config()


def test_inventory_prioritizes_entities_and_filters_pronouns(config) -> None:
    text = (
        "Neural Networks excel at pattern recognition. The reinforcement learning agent uses "
        "policy gradient method. Policy gradient method improves results when AlphaGo and "
        "DeepMind collaborate, but it should ignore pronouns."
    )
    stopwords = {
        "the",
        "at",
        "and",
        "but",
        "it",
        "should",
        "ignore",
    }
    proper_nouns = {"AlphaGo", "DeepMind"}

    def general_loader(model_name: str) -> Callable[[str], _FakeDoc]:
        assert model_name == "en_core_web_sm"

        def pipeline(content: str) -> _FakeDoc:
            tokens = _tokenize(content, stopwords, proper_nouns)
            ents = [_FakeSpan("Neural Networks"), _FakeSpan("policy gradient method")]
            noun_chunks = [
                _FakeSpan("policy gradient method"),
                _FakeSpan("policy gradient method"),
                _FakeSpan("pattern recognition"),
            ]
            return _FakeDoc(content, ents=ents, noun_chunks=noun_chunks, tokens=tokens)

        return pipeline

    builder = EntityInventoryBuilder(config, nlp_loader=general_loader)
    element = ParsedElement(
        doc_id="doc-1",
        element_id="doc-1:0",
        section="Introduction",
        content=text,
        content_hash="f" * 64,
        start_char=0,
        end_char=len(text),
    )

    inventory = builder.build_inventory(element)

    assert "Neural Networks" in inventory
    assert "policy gradient method" in inventory
    assert "AlphaGo" in inventory
    assert all(candidate in element.content for candidate in inventory)
    lower_inventory = {item.lower() for item in inventory}
    assert "it" not in lower_inventory
    assert "the" not in lower_inventory


def test_inventory_caps_results_to_fifty_candidates(config) -> None:
    base_text = " ".join(f"Entity{i} appears" for i in range(60))
    proper_nouns = {f"Entity{i}" for i in range(60)}
    stopwords: set[str] = set()

    def loader(_: str) -> Callable[[str], _FakeDoc]:
        def pipeline(content: str) -> _FakeDoc:
            tokens = _tokenize(content, stopwords, proper_nouns)
            noun_chunks = [_FakeSpan(token.text) for token in tokens]
            return _FakeDoc(content, ents=[], noun_chunks=noun_chunks, tokens=tokens)

        return pipeline

    builder = EntityInventoryBuilder(config, nlp_loader=loader)
    element = ParsedElement(
        doc_id="doc-2",
        element_id="doc-2:0",
        section="Results",
        content=base_text,
        content_hash="a" * 64,
        start_char=0,
        end_char=len(base_text),
    )

    inventory = builder.build_inventory(element)
    assert len(inventory) == 50
    assert all(candidate in element.content for candidate in inventory)


def test_biomedical_content_triggers_scispacy_and_expands_abbreviations(config) -> None:
    text = (
        "Tumor Necrosis Factor (TNF) activates immune response in cancer cells. "
        "The protein TNF influences cytokine production."
    )
    requests: list[str] = []
    stopwords = {"the", "in"}
    proper_nouns = {"TNF"}

    def loader(model_name: str) -> Callable[[str], _FakeDoc]:
        requests.append(model_name)

        def pipeline(content: str) -> _FakeDoc:
            tokens = _tokenize(content, stopwords, proper_nouns)
            ents = [_FakeSpan("TNF"), _FakeSpan("immune response")]
            noun_chunks = [
                _FakeSpan("Tumor Necrosis Factor"),
                _FakeSpan("immune response"),
                _FakeSpan("immune response"),
            ]
            return _FakeDoc(content, ents=ents, noun_chunks=noun_chunks, tokens=tokens)

        return pipeline

    builder = EntityInventoryBuilder(config, nlp_loader=loader)
    element = ParsedElement(
        doc_id="doc-3",
        element_id="doc-3:0",
        section="Discussion",
        content=text,
        content_hash="b" * 64,
        start_char=0,
        end_char=len(text),
    )

    inventory = builder.build_inventory(element)

    assert "en_core_sci_md" in requests
    assert "Tumor Necrosis Factor" in inventory
    assert "TNF" in inventory
    assert all(candidate in element.content for candidate in inventory)
