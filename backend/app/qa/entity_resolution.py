"""Entity extraction and resolution utilities for QA."""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from backend.app.canonicalization.entity_canonicalizer import EmbeddingBackend

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateNode:
    """Graph node candidate returned during entity resolution."""

    node_id: str
    name: str
    aliases: Sequence[str]
    times_seen: int
    section_distribution: Mapping[str, int]


@dataclass(frozen=True)
class ResolvedCandidate:
    """Candidate enriched with similarity metadata."""

    node_id: str
    name: str
    aliases: Sequence[str]
    times_seen: int
    section_distribution: Mapping[str, int]
    similarity: float


@dataclass(frozen=True)
class ResolvedEntity:
    """Entity mention resolved against the knowledge graph."""

    mention: str
    candidates: Sequence[ResolvedCandidate]


class QuestionEntityExtractor:
    """Extract entity mentions from natural language questions."""

    def __init__(self, nlp_loader: Optional[Callable[[str], object]] = None) -> None:
        self._nlp_loader = nlp_loader or self._default_nlp_loader
        self._nlp = self._initialize_nlp()

    def extract(self, question: str) -> List[str]:
        """Return unique entity mentions discovered in the supplied question."""

        if not question.strip():
            return []
        doc = self._nlp(question)
        results: List[str] = []
        seen: set[str] = set()
        stopwords = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
        for span in self._iter_spans(doc):
            text = span.text.strip()
            lowered = text.lower()
            if not text or lowered in seen or lowered in stopwords:
                continue
            seen.add(lowered)
            results.append(text)
        return results

    def _initialize_nlp(self) -> Callable[[str], object]:
        try:
            return self._nlp_loader("en_core_web_sm")
        except RuntimeError:
            LOGGER.warning("spaCy unavailable; falling back to rule-based extraction")
            return self._fallback_pipeline()

    @staticmethod
    def _iter_spans(doc: object) -> Iterable[object]:
        for span in getattr(doc, "ents", []):
            yield span
        for chunk in getattr(doc, "noun_chunks", []):
            yield chunk

    @staticmethod
    def _fallback_pipeline() -> Callable[[str], object]:
        def _blank(text: str) -> object:  # pragma: no cover - runtime safeguard
            tokens = [token for token in text.split() if token[:1].isupper()]
            return type(
                "FallbackDoc",
                (),
                {
                    "text": text,
                    "ents": [type("Span", (), {"text": token})() for token in tokens],
                    "noun_chunks": [],
                },
            )()

        return _blank

    @staticmethod
    def _default_nlp_loader(model_name: str) -> Callable[[str], object]:
        try:
            import spacy
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("spaCy is required for question analysis") from exc
        try:
            return spacy.load(model_name)
        except OSError:
            LOGGER.warning("spaCy model %s missing; using blank English pipeline", model_name)
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp


class EntityResolver:
    """Resolve question mentions into graph entities."""

    def __init__(
        self,
        *,
        repository: "QARepositoryProtocol",
        embedding_backend: EmbeddingBackend,
        similarity_threshold: float,
        max_candidates: int = 3,
        embedding_candidate_limit: int = 500,
    ) -> None:
        self._repository = repository
        self._embedding_backend = embedding_backend
        self._threshold = similarity_threshold
        self._max_candidates = max(1, max_candidates)
        self._candidate_limit = max(10, embedding_candidate_limit)
        self._embedding_cache: MutableMapping[str, np.ndarray] = {}

    def resolve(self, mention: str) -> ResolvedEntity:
        """Resolve the provided mention to graph candidates."""

        normalized = mention.strip()
        if not normalized:
            return ResolvedEntity(mention=mention, candidates=[])
        exact_matches = self._repository.fetch_nodes_by_exact_match(normalized)
        if exact_matches:
            candidates = [
                ResolvedCandidate(
                    node_id=node.node_id,
                    name=node.name,
                    aliases=list(node.aliases),
                    times_seen=node.times_seen,
                    section_distribution=dict(node.section_distribution),
                    similarity=1.0,
                )
                for node in exact_matches[: self._max_candidates]
            ]
            return ResolvedEntity(mention=mention, candidates=candidates)

        tokens = [token for token in _tokenize(normalized) if token]
        approximate = self._repository.fetch_candidates_for_mention(
            normalized,
            self._candidate_limit,
            tokens=tokens,
        )
        if not approximate:
            approximate = self._repository.fetch_candidate_nodes(self._candidate_limit)
        scored = self._score_candidates(normalized, approximate)
        scored.sort(key=lambda candidate: (-candidate.similarity, -candidate.times_seen, candidate.name))
        filtered = [candidate for candidate in scored if candidate.similarity >= self._threshold]
        return ResolvedEntity(mention=mention, candidates=filtered[: self._max_candidates])

    def _score_candidates(
        self, mention: str, candidates: Sequence[CandidateNode]
    ) -> List[ResolvedCandidate]:
        mention_embedding = self._embed(mention)
        results: List[ResolvedCandidate] = []
        for node in candidates:
            best_similarity = self._best_similarity(mention_embedding, node)
            results.append(
                ResolvedCandidate(
                    node_id=node.node_id,
                    name=node.name,
                    aliases=list(node.aliases),
                    times_seen=node.times_seen,
                    section_distribution=dict(node.section_distribution),
                    similarity=best_similarity,
                )
            )
        return results

    def _best_similarity(self, mention_embedding: np.ndarray, node: CandidateNode) -> float:
        scores = [self._compare_embeddings(mention_embedding, node.name)]
        for alias in node.aliases:
            scores.append(self._compare_embeddings(mention_embedding, alias))
        return max(scores)

    def _compare_embeddings(self, base_embedding: np.ndarray, text: str) -> float:
        candidate_embedding = self._embed(text)
        if base_embedding.size == 0 or candidate_embedding.size == 0:
            return 0.0
        return float(np.dot(base_embedding, candidate_embedding))

    def _embed(self, text: str) -> np.ndarray:
        normalized = _normalize_text(text)
        if normalized in self._embedding_cache:
            return self._embedding_cache[normalized]
        embedding = self._embedding_backend.embed(normalized)
        self._embedding_cache[normalized] = embedding
        return embedding


def _tokenize(text: str) -> List[str]:
    tokens = []
    for raw in text.split():
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if cleaned and len(cleaned) > 2:
            tokens.append(cleaned.lower())
    return tokens


@lru_cache(maxsize=1024)
def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    normalized = normalized.strip("\t\n\r ")
    normalized = normalized.strip("!\"#$%&'()*+, -./:;<=>?@[\\]^_`{|}~")
    return normalized.lower()


class QARepositoryProtocol:
    """Protocol describing required repository operations."""

    def fetch_nodes_by_exact_match(self, mention: str) -> Sequence[CandidateNode]:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch_candidates_for_mention(
        self, mention: str, limit: int, *, tokens: Sequence[str] | None = None
    ) -> Sequence[CandidateNode]:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch_candidate_nodes(self, limit: int) -> Sequence[CandidateNode]:  # pragma: no cover - interface
        raise NotImplementedError

    # Extended operations used by QAService; kept optional for lightweight stubs.
    def fetch_paths(
        self,
        *,
        start_id: str,
        end_id: str,
        max_hops: int,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[Mapping[str, object]]:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch_neighbors(
        self,
        node_id: str,
        *,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[Mapping[str, object]]:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch_document_edges(
        self,
        doc_id: str,
        *,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[Mapping[str, object]]:  # pragma: no cover - interface
        raise NotImplementedError
