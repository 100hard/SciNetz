"""Tests for semantic relation matching fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pytest

from backend.app.config import RelationPatternConfig, RelationsConfig, RelationSemanticMatchConfig


@dataclass
class _StubEmbeddingBackend:
    """Deterministic embedding backend for tests."""

    vectors: Dict[str, np.ndarray]

    def embed(self, text: str) -> np.ndarray:
        key = text.strip().lower()
        if key not in self.vectors:
            raise KeyError(f"missing embedding for {text}")
        vector = self.vectors[key]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


def _vector(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float32)


def test_semantic_matcher_returns_best_relation() -> None:
    """Closest known phrase should determine the canonical relation."""

    from backend.app.extraction.relation_semantics import SemanticRelationMatcher

    patterns = [
        RelationPatternConfig(canonical="trained-on", phrases=["trained on"], swap=False),
        RelationPatternConfig(canonical="implemented-in", phrases=["implemented in"], swap=False),
    ]
    config = RelationsConfig(
        normalization_patterns=patterns,
        semantic_matching=RelationSemanticMatchConfig(
            enabled=True,
            acceptance_threshold=0.9,
        ),
    )
    backend = _StubEmbeddingBackend(
        vectors={
            "trained on": _vector(1.0, 0.0, 0.0),
            "implemented in": _vector(0.0, 1.0, 0.0),
            "fine tuned on": _vector(0.96, 0.1, 0.0),
        }
    )
    matcher = SemanticRelationMatcher(config, embedding_backend=backend)

    result = matcher.match("fine tuned on")

    assert result is not None
    canonical, swap, score, matched_phrase = result
    assert canonical == "trained-on"
    assert swap is False
    expected = float(np.dot(backend.embed("fine tuned on"), backend.embed("trained on")))
    assert pytest.approx(score, rel=1e-6) == expected
    assert matched_phrase == "trained on"


def test_semantic_matcher_rejects_below_threshold() -> None:
    """Scores below threshold should return None."""

    from backend.app.extraction.relation_semantics import SemanticRelationMatcher

    patterns = [
        RelationPatternConfig(canonical="trained-on", phrases=["trained on"], swap=False),
    ]
    config = RelationsConfig(
        normalization_patterns=patterns,
        semantic_matching=RelationSemanticMatchConfig(
            enabled=True,
            acceptance_threshold=0.95,
        ),
    )
    backend = _StubEmbeddingBackend(
        vectors={
            "trained on": _vector(1.0, 0.0, 0.0),
            "lightly pre trained": _vector(0.7, 0.7, 0.0),
        }
    )
    matcher = SemanticRelationMatcher(config, embedding_backend=backend)

    assert matcher.match("lightly pre trained") is None
