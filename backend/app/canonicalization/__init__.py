"""Canonicalization package for entity merging logic."""

from .entity_canonicalizer import (
    CanonicalizationResult,
    E5EmbeddingBackend,
    EntityCandidate,
    EntityCanonicalizer,
    HashingEmbeddingBackend,
)
from .pipeline import CanonicalizationPipeline, EntityAggregator

__all__ = [
    "CanonicalizationResult",
    "EntityCandidate",
    "EntityCanonicalizer",
    "EntityAggregator",
    "CanonicalizationPipeline",
    "E5EmbeddingBackend",
    "HashingEmbeddingBackend",
]
