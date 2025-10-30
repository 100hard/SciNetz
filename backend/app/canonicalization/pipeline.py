"""Entity canonicalization pipeline orchestration utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from backend.app.config import AppConfig, load_config
from backend.app.contracts import Triplet
from backend.app.extraction import ExtractionResult

from .entity_canonicalizer import (
    CanonicalizationResult,
    EntityCandidate,
    EntityCanonicalizer,
)

LOGGER = logging.getLogger(__name__)

TypeResolver = Callable[[str], Optional[str]]


@dataclass(slots=True)
class _AggregatedEntity:
    """Internal mutable structure storing aggregation statistics."""

    name: str
    section_counts: Dict[str, int] = field(default_factory=dict)
    doc_ids: Set[str] = field(default_factory=set)
    times_seen: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)

    def add_sections(self, sections: Mapping[str, int]) -> None:
        """Accumulate section occurrence counts for the entity."""

        for section, count in sections.items():
            if count <= 0:
                continue
            self.section_counts[section] = self.section_counts.get(section, 0) + count
            self.times_seen += count

    def add_doc(self, doc_id: str) -> None:
        """Record that the entity appeared in a document."""

        if doc_id:
            self.doc_ids.add(doc_id)

    def add_type(self, entity_type: str, weight: int = 1) -> None:
        """Accumulate type votes derived from contextual heuristics."""

        if weight <= 0:
            return
        self.type_counts[entity_type] = self.type_counts.get(entity_type, 0) + weight

    def to_candidate(self, default_type: str) -> EntityCandidate:
        """Convert the aggregation record into an :class:`EntityCandidate`."""

        if self.type_counts:
            entity_type = max(self.type_counts.items(), key=lambda item: (item[1], item[0]))[0]
        else:
            entity_type = default_type
        section_distribution = dict(sorted(self.section_counts.items()))
        times_seen = self.times_seen if self.times_seen > 0 else len(self.doc_ids)
        return EntityCandidate(
            name=self.name,
            type=entity_type,
            times_seen=times_seen,
            section_distribution=section_distribution,
            source_document_ids=sorted(self.doc_ids),
        )


class EntityAggregator:
    """Aggregate extraction outputs into canonicalization candidates."""

    def __init__(
        self,
        *,
        type_resolver: Optional[TypeResolver] = None,
        default_type: str = "Unknown",
    ) -> None:
        self._entities: Dict[str, _AggregatedEntity] = {}
        self._type_resolver = type_resolver
        self._default_type = default_type

    def ingest(self, extraction: ExtractionResult) -> None:
        """Ingest an extraction result and update aggregation stats."""

        for name, sections in extraction.section_distribution.items():
            entity = self._entities.setdefault(name, _AggregatedEntity(name=name))
            entity.add_sections(sections)
        touched: Set[str] = set()
        for triplet in extraction.triplets:
            touched.update((triplet.subject, triplet.object))
            self._register_triplet(triplet)
        for name, votes in extraction.entity_type_votes.items():
            entity = self._entities.setdefault(name, _AggregatedEntity(name=name))
            for entity_type, weight in votes.items():
                entity.add_type(entity_type, weight)
        for name in touched:
            entity = self._entities.setdefault(name, _AggregatedEntity(name=name))
            if not entity.type_counts:
                entity.add_type(self._resolve_type(name))

    def extend(self, extractions: Iterable[ExtractionResult]) -> None:
        """Ingest a collection of extraction results."""

        for extraction in extractions:
            self.ingest(extraction)

    def build_candidates(self) -> List[EntityCandidate]:
        """Return the aggregated entity candidates sorted by salience."""

        candidates = [entity.to_candidate(self._default_type) for entity in self._entities.values()]
        candidates.sort(key=lambda item: (-item.times_seen, item.name.lower()))
        return candidates

    def _register_triplet(self, triplet: Triplet) -> None:
        doc_id = triplet.evidence.doc_id
        for name in (triplet.subject, triplet.object):
            entity = self._entities.setdefault(name, _AggregatedEntity(name=name))
            entity.add_doc(doc_id)

    def _resolve_type(self, name: str) -> str:
        if self._type_resolver is None:
            return self._default_type
        try:
            resolved = self._type_resolver(name)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Type resolver failed for entity %s", name)
            return self._default_type
        if not resolved:
            return self._default_type
        return resolved


class CanonicalizationPipeline:
    """High-level orchestration for entity canonicalization."""

    def __init__(
        self,
        *,
        config: Optional[AppConfig] = None,
        canonicalizer: Optional[EntityCanonicalizer] = None,
        aggregator_factory: Optional[Callable[[], EntityAggregator]] = None,
        default_entity_type: str = "Unknown",
    ) -> None:
        self._config = config or load_config()
        self._default_entity_type = default_entity_type
        self._canonicalizer = canonicalizer or EntityCanonicalizer(self._config)
        if aggregator_factory is None:
            self._aggregator_factory = lambda: EntityAggregator(default_type=self._default_entity_type)
        else:
            self._aggregator_factory = aggregator_factory

        if self._config.canonicalization.preload_embeddings:
            self._launch_preload_thread()

    def run(self, extractions: Sequence[ExtractionResult]) -> CanonicalizationResult:
        """Canonicalize entities from a sequence of extraction outputs."""

        aggregator = self._aggregator_factory()
        aggregator.extend(extractions)
        candidates = aggregator.build_candidates()
        return self._canonicalizer.canonicalize(candidates)

    def preload_embeddings(self) -> None:
        """Warm the canonicalization embedding backend."""

        if hasattr(self._canonicalizer, "preload_embeddings"):
            self._canonicalizer.preload_embeddings()

    def _launch_preload_thread(self) -> None:
        """Start a background thread to warm the embedding backend."""

        from threading import Thread

        def _target() -> None:
            try:
                self.preload_embeddings()
            except Exception:  # noqa: BLE001 - warm-up issues should not block pipeline
                LOGGER.exception("Failed to preload canonicalization embeddings")

        Thread(
            target=_target,
            name="canonicalization-preload",
            daemon=True,
        ).start()

