"""Semantic relevancy scoring utilities for graph edges."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend.app.canonicalization.entity_canonicalizer import (
    E5EmbeddingBackend,
    EmbeddingBackend,
    HashingEmbeddingBackend,
)
from backend.app.config import ObservabilityRelevancyConfig
from backend.app.contracts import Triplet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelevancySummary:
    """Aggregate relevancy statistics for a single paper."""

    count: int
    average: float
    minimum: float
    maximum: float


class SemanticRelevancyEvaluator:
    """Compute semantic relevancy scores for persisted triplets."""

    def __init__(
        self,
        config: ObservabilityRelevancyConfig,
        *,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self._config = config
        self._backend = backend or self._create_backend(config)
        self._cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _create_backend(config: ObservabilityRelevancyConfig) -> EmbeddingBackend:
        """Instantiate the embedding backend configured for relevancy scoring."""

        model = config.embedding_model.lower()
        if model == "hashing":
            return HashingEmbeddingBackend()
        try:
            return E5EmbeddingBackend(
                model_name=config.embedding_model,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
            )
        except Exception:  # pragma: no cover - fallback handled below
            LOGGER.exception(
                "Failed to initialise embedding backend '%s'; falling back to hashing",
                config.embedding_model,
            )
            return HashingEmbeddingBackend()

    def score_triplet(self, triplet: Triplet) -> Optional[float]:
        """Compute a relevancy score for a triplet based on its evidence sentence."""

        evidence = triplet.evidence.full_sentence
        if not evidence:
            return None
        subject = triplet.subject.strip()
        predicate = triplet.predicate.replace("-", " ").strip()
        obj = triplet.object.strip()
        if not subject or not predicate or not obj:
            return None
        composite = f"{subject} {predicate} {obj}"
        return self._cosine_similarity(composite, evidence)

    def summaries(self, values: Mapping[str, Sequence[float]]) -> Dict[str, RelevancySummary]:
        """Return aggregated statistics for each paper."""

        summaries: Dict[str, RelevancySummary] = {}
        for doc_id, scores in values.items():
            clean = [score for score in scores if score is not None]
            if not clean:
                continue
            count = len(clean)
            average = float(np.mean(clean))
            minimum = float(np.min(clean))
            maximum = float(np.max(clean))
            summaries[doc_id] = RelevancySummary(
                count=count,
                average=average,
                minimum=minimum,
                maximum=maximum,
            )
        return summaries

    def classify(self, score: float) -> str:
        """Return a qualitative label for the supplied score."""

        if score >= self._config.target:
            return "green"
        if score >= self._config.warning:
            return "yellow"
        return "red"

    def minimum_edges(self) -> int:
        """Return the configured minimum edge threshold for scoring."""

        return max(0, self._config.minimum_edges)

    def _cosine_similarity(self, left: str, right: str) -> Optional[float]:
        """Return the cosine similarity between two texts."""

        left_vec = self._embed(left)
        right_vec = self._embed(right)
        if left_vec.size == 0 or right_vec.size == 0:
            return None
        score = float(np.dot(left_vec, right_vec))
        return max(min(score, 1.0), -1.0)

    def _embed(self, text: str) -> np.ndarray:
        """Embed text using the configured backend with memoisation."""

        cached = self._cache.get(text)
        if cached is not None:
            return cached
        try:
            vector = self._backend.embed(text)
        except Exception:  # pragma: no cover - backend errors logged
            LOGGER.exception("Failed to embed text for relevancy scoring")
            vector = np.zeros(1, dtype=np.float32)
        if vector.size == 0:
            self._cache[text] = vector
            return vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            normalized = np.zeros(vector.size, dtype=np.float32)
        else:
            normalized = (vector / norm).astype(np.float32, copy=False)
        self._cache[text] = normalized
        return normalized

