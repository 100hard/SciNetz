"""Semantic relation matching utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from typing_extensions import Literal

import numpy as np

from backend.app.canonicalization.entity_canonicalizer import (
    E5EmbeddingBackend,
    EmbeddingBackend,
    HashingEmbeddingBackend,
)
from backend.app.config import RelationPatternConfig, RelationsConfig, RelationSemanticMatchConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PatternEmbedding:
    """Embedding metadata for a configured relation phrase."""

    canonical: str
    phrase: str
    swap: bool
    vector: np.ndarray


@dataclass(frozen=True)
class RelationSemanticMatch:
    """Outcome returned after comparing a candidate phrase to known patterns."""

    canonical: str
    swap: bool
    score: float
    matched_phrase: str
    candidate: str
    accepted: bool
    needs_review: bool
    acceptance_threshold: float
    review_threshold: Optional[float]


def _normalize(text: str) -> str:
    """Normalize text for embedding lookups.

    Args:
        text: Input phrase to normalize.

    Returns:
        str: Trimmed phrase preserving original casing.
    """

    return text.strip()


class SemanticRelationMatcher:
    """Match unseen relation phrases to canonical relations via embeddings."""

    def __init__(
        self,
        config: RelationsConfig,
        *,
        embedding_backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self._settings: RelationSemanticMatchConfig = config.semantic_matching
        self._enabled: bool = bool(self._settings.enabled and config.normalization_patterns)
        self._acceptance_threshold: float = self._settings.acceptance_threshold
        self._review_threshold: Optional[float] = self._settings.review_threshold
        if (
            self._review_threshold is not None
            and self._review_threshold > self._acceptance_threshold
        ):
            LOGGER.warning(
                (
                    "Relation review threshold %.3f exceeds acceptance threshold %.3f; "
                    "using acceptance threshold for review gating"
                ),
                self._review_threshold,
                self._acceptance_threshold,
            )
            self._review_threshold = self._acceptance_threshold
        self._patterns: List[_PatternEmbedding] = []
        self._backend: Optional[EmbeddingBackend] = None
        if not self._enabled:
            return
        self._backend = embedding_backend or self._create_default_backend(self._settings)
        if self._backend is None:
            LOGGER.warning("Semantic relation matching disabled: no embedding backend available")
            self._enabled = False
            return
        try:
            self._patterns = self._build_embeddings(config.normalization_patterns)
        except Exception:
            LOGGER.exception("Failed to build relation pattern embeddings; disabling matcher")
            self._enabled = False
            self._patterns = []

    @staticmethod
    def _create_default_backend(settings: RelationSemanticMatchConfig) -> Optional[EmbeddingBackend]:
        """Create the default embedding backend from configuration.

        Args:
            settings: Semantic matcher configuration block.

        Returns:
            Optional[EmbeddingBackend]: Instantiated backend or None if unavailable.
        """

        model_name = settings.embedding_model
        device = settings.embedding_device
        if model_name:
            if not E5EmbeddingBackend.is_available():
                LOGGER.warning(
                    "SentenceTransformer backend unavailable; semantic relation matching disabled",
                )
                return None
            return E5EmbeddingBackend(model_name=model_name, device=device)
        return HashingEmbeddingBackend()

    def _build_embeddings(
        self, patterns: Iterable[RelationPatternConfig]
    ) -> List[_PatternEmbedding]:
        """Embed configured relation phrases.

        Args:
            patterns: Relation normalization patterns from configuration.

        Returns:
            List[_PatternEmbedding]: Embedded representations for comparison.
        """

        backend = self._backend
        if backend is None:
            return []
        embeddings: List[_PatternEmbedding] = []
        for pattern in patterns:
            for phrase in pattern.phrases:
                normalized = _normalize(phrase)
                if not normalized:
                    continue
                try:
                    vector = backend.embed(normalized)
                except Exception:  # noqa: BLE001 - logged and skipped
                    LOGGER.exception("Failed to embed relation phrase '%s'", phrase)
                    continue
                if vector.size == 0:
                    continue
                embeddings.append(
                    _PatternEmbedding(
                        canonical=pattern.canonical,
                        phrase=normalized,
                        swap=pattern.swap,
                        vector=vector.astype(np.float32, copy=False),
                    )
                )
        if not embeddings:
            LOGGER.warning("Semantic relation matcher has no embeddings to compare against")
        return embeddings

    def match(self, phrase: str) -> Optional[RelationSemanticMatch]:
        """Match a candidate phrase against known relation patterns.

        Args:
            phrase: Verb phrase emitted by the extractor.

        Returns:
            Optional[RelationSemanticMatch]: Semantic comparison outcome when available.
        """

        if not self._enabled or not self._patterns:
            return None
        candidate = _normalize(phrase)
        if not candidate:
            return None
        backend = self._backend
        if backend is None:
            return None
        try:
            vector = backend.embed(candidate)
        except Exception:  # noqa: BLE001 - log and skip matching
            LOGGER.exception("Failed to embed relation candidate '%s'", phrase)
            return None
        if vector.size == 0:
            return None
        best_score = -1.0
        best_pattern: Optional[_PatternEmbedding] = None
        for pattern in self._patterns:
            score = float(np.dot(vector, pattern.vector))
            if score > best_score:
                best_score = score
                best_pattern = pattern
        if best_pattern is None:
            return None
        accepted = best_score >= self._acceptance_threshold
        review_threshold = self._review_threshold
        needs_review = False
        if not accepted and review_threshold is not None:
            needs_review = best_score >= review_threshold
        return RelationSemanticMatch(
            canonical=best_pattern.canonical,
            swap=best_pattern.swap,
            score=best_score,
            matched_phrase=best_pattern.phrase,
            candidate=candidate,
            accepted=accepted,
            needs_review=needs_review,
            acceptance_threshold=self._acceptance_threshold,
            review_threshold=review_threshold,
        )
