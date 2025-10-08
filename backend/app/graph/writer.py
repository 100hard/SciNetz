"""Neo4j graph writer implementation for Phase 5."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional dependency for type checking
    from neo4j import Driver
except Exception:  # pragma: no cover - fallback when driver is unavailable
    Driver = Any  # type: ignore[misc,assignment]

from ..config import AppConfig, load_config
from ..contracts import Evidence, Node

LOGGER = logging.getLogger(__name__)


class GraphWriter:
    """Persist canonical nodes and edges into Neo4j with batching."""

    _ENTITY_QUERY = """
    UNWIND $entities AS entity
    MERGE (node:Entity {node_id: entity.node_id})
    WITH
        entity,
        node,
        coalesce(node.aliases, []) AS existing_aliases,
        coalesce(node.source_document_ids, []) AS existing_docs,
        coalesce(node.section_distribution, {}) AS existing_sections,
        coalesce(node.times_seen, 0) AS existing_times
    WITH
        entity,
        node,
        existing_aliases,
        existing_docs,
        existing_sections,
        existing_times,
        [doc IN entity.source_document_ids WHERE NOT doc IN existing_docs] AS new_docs,
        keys(existing_sections) + keys(entity.section_distribution) AS section_keys
    WITH
        entity,
        node,
        existing_aliases,
        existing_docs,
        existing_sections,
        existing_times,
        new_docs,
        reduce(
            acc = [],
            key IN section_keys |
                CASE
                    WHEN key IN acc THEN acc
                    ELSE acc + key
                END
        ) AS distinct_section_keys
    SET
        node.name = entity.name,
        node.type = entity.type,
        node.pipeline_version = entity.pipeline_version,
        node.aliases = reduce(
            acc = [],
            alias IN existing_aliases + entity.aliases |
                CASE
                    WHEN alias IN acc OR alias = entity.name THEN acc
                    ELSE acc + alias
                END
        ),
        node.source_document_ids = reduce(
            acc = [],
            doc IN existing_docs + entity.source_document_ids |
                CASE
                    WHEN doc IN acc THEN acc
                    ELSE acc + doc
                END
        ),
        node.section_distribution = CASE
            WHEN size(new_docs) = 0 THEN existing_sections
            ELSE reduce(
                acc = {},
                key IN distinct_section_keys |
                    acc + {
                        key: coalesce(existing_sections[key], 0) + coalesce(entity.section_distribution[key], 0)
                    }
            )
        END,
        node.times_seen = existing_times + CASE
            WHEN size(new_docs) = 0 THEN 0
            WHEN entity.times_seen > 0 THEN entity.times_seen
            ELSE size(new_docs)
        END
    RETURN node.node_id AS node_id
    """

    _EDGE_QUERY = """
    UNWIND $edges AS edge
    MATCH (src:Entity {node_id: edge.src_id})
    MATCH (dst:Entity {node_id: edge.dst_id})
    MERGE (src)-[rel:RELATION {
        relation_norm: edge.relation_norm,
        src_id: edge.src_id,
        dst_id: edge.dst_id
    }]->(dst)
    ON CREATE SET
        rel.relation_verbatim = edge.relation_verbatim,
        rel.evidence = edge.evidence,
        rel.confidence = edge.confidence,
        rel.pipeline_version = edge.pipeline_version,
        rel.times_seen = edge.times_seen,
        rel.created_at = edge.created_at,
        rel.attributes = edge.attributes,
        rel.conflicting = false
    ON MATCH SET
        rel.relation_verbatim = edge.relation_verbatim,
        rel.pipeline_version = edge.pipeline_version,
        rel.attributes = CASE
            WHEN edge.attributes_provided THEN edge.attributes
            ELSE rel.attributes
        END,
        rel.evidence = edge.evidence,
        rel.times_seen = coalesce(rel.times_seen, 0) + edge.times_seen,
        rel.confidence = CASE
            WHEN edge.confidence > rel.confidence THEN edge.confidence
            ELSE rel.confidence
        END
    WITH edge, rel, src, dst
    OPTIONAL MATCH (dst)-[reverse:RELATION {
        relation_norm: edge.relation_norm,
        src_id: edge.dst_id,
        dst_id: edge.src_id
    }]->(src)
    FOREACH (_ IN CASE
        WHEN edge.directional AND reverse IS NOT NULL THEN [1]
        ELSE []
    END |
        SET rel.conflicting = true
    )
    FOREACH (_ IN CASE
        WHEN edge.directional AND reverse IS NOT NULL THEN [1]
        ELSE []
    END |
        SET reverse.conflicting = true
    )
    """

    def __init__(
        self,
        driver: Driver,
        config: Optional[AppConfig] = None,
        *,
        entity_batch_size: int = 200,
        edge_batch_size: int = 500,
        max_retries: int = 1,
    ) -> None:
        """Create a graph writer using the provided Neo4j driver.

        Args:
            driver: Active Neo4j driver instance.
            config: Application configuration; if omitted the default config is loaded.
            entity_batch_size: Number of nodes to buffer before flushing.
            edge_batch_size: Number of edges to buffer before flushing.
            max_retries: Maximum number of retries when a transaction fails.

        Raises:
            ValueError: If the batch sizes are not positive.
        """

        if entity_batch_size <= 0:
            msg = "entity_batch_size must be positive"
            raise ValueError(msg)
        if edge_batch_size <= 0:
            msg = "edge_batch_size must be positive"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be zero or greater"
            raise ValueError(msg)

        self._driver = driver
        self._config = config or load_config()
        self._entity_batch_size = entity_batch_size
        self._edge_batch_size = edge_batch_size
        self._max_retries = max_retries
        self._pipeline_version = self._config.pipeline.version
        self._graph_config = self._config.graph
        self._entity_batch: List[Dict[str, Any]] = []
        self._edge_batch: List[Dict[str, Any]] = []

    def upsert_entity(self, node: Node) -> str:
        """Queue a canonical node for persistence.

        Args:
            node: Canonical node contract produced by the canonicalizer.

        Returns:
            str: The canonical node identifier.
        """

        payload = self._node_to_parameters(node)
        self._entity_batch.append(payload)
        if len(self._entity_batch) >= self._entity_batch_size:
            self._flush_entities()
        return node.node_id

    def upsert_edge(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation_norm: str,
        relation_verbatim: str,
        evidence: Evidence,
        confidence: float,
        attributes: Optional[Mapping[str, str]] = None,
        created_at: Optional[datetime] = None,
        times_seen: int = 1,
    ) -> None:
        """Queue an edge for persistence with validation.

        Args:
            src_id: Source canonical node identifier.
            dst_id: Destination canonical node identifier.
            relation_norm: Normalized relation label.
            relation_verbatim: Original relation text from extraction.
            evidence: Evidence metadata supporting the relation.
            confidence: Model confidence associated with the relation.
            attributes: Optional attribute map with additional context.
            created_at: Creation timestamp override; defaults to now if omitted.
            times_seen: Increment to apply to the edge frequency counter.

        Raises:
            ValueError: If the evidence is missing or the increment is non-positive.
        """

        if evidence is None:
            msg = "evidence is required when upserting an edge"
            raise ValueError(msg)
        if times_seen <= 0:
            msg = "times_seen increment must be positive"
            raise ValueError(msg)
        if not 0.0 <= confidence <= 1.0:
            msg = "confidence must be between 0.0 and 1.0"
            raise ValueError(msg)

        payload = self._edge_to_parameters(
            src_id=src_id,
            dst_id=dst_id,
            relation_norm=relation_norm,
            relation_verbatim=relation_verbatim,
            evidence=evidence,
            confidence=confidence,
            attributes=attributes,
            created_at=created_at or datetime.now(timezone.utc),
            times_seen=times_seen,
        )
        self._edge_batch.append(payload)
        if len(self._edge_batch) >= self._edge_batch_size:
            self._flush_edges()

    def flush(self) -> None:
        """Flush any buffered nodes or edges to Neo4j."""

        self._flush_entities()
        self._flush_edges()

    def _flush_entities(self) -> None:
        if not self._entity_batch:
            return
        batch = self._entity_batch
        self._entity_batch = []
        self._run_with_retry(self._write_entity_batch, batch)

    def _flush_edges(self) -> None:
        if not self._edge_batch:
            return
        batch = self._edge_batch
        self._edge_batch = []
        self._run_with_retry(self._write_edge_batch, batch)

    def _run_with_retry(
        self,
        callback: Callable[[Any, Iterable[MutableMapping[str, Any]]], None],
        payload: List[Dict[str, Any]],
    ) -> None:
        attempts = 0
        while True:
            try:
                with self._driver.session() as session:
                    session.execute_write(callback, payload)
                return
            except Exception:  # pragma: no cover - defensive retry
                LOGGER.exception("Neo4j write failed (attempt %s)", attempts + 1)
                attempts += 1
                if attempts > self._max_retries:
                    LOGGER.error(
                        "Dropping %s records after exhausting retries", len(payload)
                    )
                    return

    def _write_entity_batch(self, tx: Any, entities: Iterable[MutableMapping[str, Any]]) -> None:
        tx.run(self._ENTITY_QUERY, entities=list(entities))

    def _write_edge_batch(self, tx: Any, edges: Iterable[MutableMapping[str, Any]]) -> None:
        tx.run(self._EDGE_QUERY, edges=list(edges))

    def _node_to_parameters(self, node: Node) -> Dict[str, Any]:
        aliases = [alias for alias in node.aliases if alias and alias != node.name]
        alias_seen = set()
        deduped_aliases: List[str] = []
        for alias in aliases:
            if alias not in alias_seen:
                alias_seen.add(alias)
                deduped_aliases.append(alias)
        documents_seen = set()
        deduped_docs: List[str] = []
        for doc_id in node.source_document_ids:
            if doc_id not in documents_seen:
                documents_seen.add(doc_id)
                deduped_docs.append(doc_id)
        section_distribution = {
            section: count for section, count in node.section_distribution.items() if count > 0
        }
        return {
            "node_id": node.node_id,
            "name": node.name,
            "type": node.type,
            "aliases": deduped_aliases,
            "section_distribution": section_distribution,
            "times_seen": max(node.times_seen, 0),
            "source_document_ids": deduped_docs,
            "pipeline_version": self._pipeline_version,
        }

    def _edge_to_parameters(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation_norm: str,
        relation_verbatim: str,
        evidence: Evidence,
        confidence: float,
        attributes: Optional[Mapping[str, str]],
        created_at: datetime,
        times_seen: int,
    ) -> Dict[str, Any]:
        attribute_map = {key: str(value) for key, value in (attributes or {}).items()}
        evidence_payload = {
            "doc_id": evidence.doc_id,
            "element_id": evidence.element_id,
            "text_span": {
                "start": evidence.text_span.start,
                "end": evidence.text_span.end,
            },
        }
        if evidence.full_sentence is not None:
            evidence_payload["full_sentence"] = evidence.full_sentence
        return {
            "src_id": src_id,
            "dst_id": dst_id,
            "relation_norm": relation_norm,
            "relation_verbatim": relation_verbatim,
            "evidence": evidence_payload,
            "confidence": confidence,
            "attributes": attribute_map,
            "attributes_provided": attributes is not None,
            "created_at": created_at,
            "times_seen": times_seen,
            "pipeline_version": self._pipeline_version,
            "directional": self._graph_config.is_directional(relation_norm),
        }
