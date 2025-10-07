"""Canonicalization package for entity merging logic."""

from .entity_canonicalizer import (
    CanonicalizationResult,
    EntityCandidate,
    EntityCanonicalizer,
)
from .pipeline import CanonicalizationPipeline, EntityAggregator

__all__ = [
    "CanonicalizationResult",
    "EntityCandidate",
    "EntityCanonicalizer",
    "EntityAggregator",
    "CanonicalizationPipeline",
]
