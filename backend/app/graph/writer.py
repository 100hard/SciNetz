"""Neo4j graph writer implementation for Phase 5."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency for type checking
    from neo4j import Driver
except Exception:  # pragma: no cover - fallback when driver is unavailable
    Driver = Any  # type: ignore[misc,assignment]

from ..config import AppConfig, load_config
from ..contracts import Evidence, Node
from .section_distribution import (
    decode_section_distribution,
    encode_section_distribution,
)

LOGGER = logging.getLogger(__name__)


class GraphWriter:
    """Persist canonical nodes and edges into Neo4j with batching."""

    _ENTITY_QUERY = """
    UNWIND $entities AS entity
    MERGE (node:Entity {node_id: entity.node_id})
    SET
        node.name = entity.name,
        node.type = entity.type,
        node.pipeline_version = entity.pipeline_version,
        node.aliases = entity.aliases,
        node.source_document_ids = entity.source_document_ids,
        node.section_distribution_keys = entity.section_distribution_keys,
        node.section_distribution_values = entity.section_distribution_values,
        node.section_distribution = null,
        node.times_seen = entity.times_seen
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
        entity_batch_size: Optional[int] = None,
        edge_batch_size: Optional[int] = None,
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

        resolved_config = config or load_config()
        resolved_entity_batch = (
            entity_batch_size
            if entity_batch_size is not None
            else resolved_config.graph.entity_batch_size
        )
        resolved_edge_batch = (
            edge_batch_size
            if edge_batch_size is not None
            else resolved_config.graph.edge_batch_size
        )
        if resolved_entity_batch <= 0:
            msg = "entity_batch_size must be positive"
            raise ValueError(msg)
        if resolved_edge_batch <= 0:
            msg = "edge_batch_size must be positive"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be zero or greater"
            raise ValueError(msg)

        self._driver = driver
        self._config = resolved_config
        self._entity_batch_size = resolved_entity_batch
        self._edge_batch_size = resolved_edge_batch
        self._max_retries = max_retries
        self._pipeline_version = self._config.pipeline.version
        self._graph_config = self._config.graph
        self._entity_batch: List[Dict[str, Any]] = []
        self._edge_batch: List[Dict[str, Any]] = []
        self._entity_label_exists: Optional[bool] = None

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
        prepared = self._prepare_entity_batch(batch)
        if not prepared:
            return
        self._run_with_retry(self._write_entity_batch, prepared)

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

    def _prepare_entity_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine staged entity payloads with existing graph state."""

        aggregated: Dict[str, Dict[str, Any]] = {}
        for payload in batch:
            node_id = payload["node_id"]
            existing = aggregated.get(node_id)
            if existing is None:
                aggregated[node_id] = dict(payload)
            else:
                self._merge_payload(existing, payload)

        node_ids = list(aggregated.keys())
        existing_records = self._load_existing_nodes(node_ids)

        prepared: List[Dict[str, Any]] = []
        for node_id, payload in aggregated.items():
            merged = self._merge_with_existing(payload, existing_records.get(node_id))
            prepared.append(merged)
        return prepared

    def _merge_payload(self, base: Dict[str, Any], addition: Dict[str, Any]) -> None:
        """Merge two staged payloads referring to the same node."""

        existing_aliases = list(base.get("aliases", []))
        additional_aliases = list(addition.get("aliases", []))
        existing_docs = list(base.get("source_document_ids", []))
        additional_docs = list(addition.get("source_document_ids", []))
        existing_doc_set = {doc for doc in existing_docs if doc}
        new_docs = [doc for doc in additional_docs if doc and doc not in existing_doc_set]
        base["aliases"] = self._merge_unique_list(existing_aliases, additional_aliases)
        base["source_document_ids"] = self._merge_unique_list(existing_docs, additional_docs)
        existing_sections = dict(base.get("section_distribution", {}))
        addition_sections = dict(addition.get("section_distribution", {}))
        if new_docs:
            base["section_distribution"] = self._merge_section_maps(
                existing_sections, addition_sections
            )
        else:
            merged_sections = dict(existing_sections)
            for section, count in addition_sections.items():
                merged_sections[section] = max(merged_sections.get(section, 0), count)
            base["section_distribution"] = merged_sections
        base_times = max(int(base.get("times_seen", 0)), 0)
        addition_times = max(int(addition.get("times_seen", 0)), 0)
        if new_docs:
            increment = max(addition_times, len(new_docs))
            base["times_seen"] = base_times + increment
        else:
            base["times_seen"] = max(base_times, addition_times)

    def _load_existing_nodes(self, node_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch existing node state from Neo4j for the supplied identifiers."""

        if not node_ids:
            return {}
        if not self._ensure_entity_label_exists():
            return {}

        def _fetch(tx: Any, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
            query = """
            MATCH (node:Entity)
            WHERE node.node_id IN $ids
            RETURN node.node_id AS node_id,
                   coalesce(node.aliases, []) AS aliases,
                   coalesce(node.source_document_ids, []) AS docs,
                   CASE
                       WHEN "section_distribution" IN keys(node)
                           THEN node.section_distribution
                       ELSE {}
                   END AS legacy_sections,
                   coalesce(node.section_distribution_keys, []) AS section_keys,
                   coalesce(node.section_distribution_values, []) AS section_values,
                   coalesce(node.times_seen, 0) AS times_seen
            """
            records = tx.run(query, ids=list(ids))
            result: Dict[str, Dict[str, Any]] = {}
            for record in records:
                keys = record["section_keys"]
                values = record["section_values"]
                legacy = record["legacy_sections"]
                result[record["node_id"]] = {
                    "aliases": list(record["aliases"] or []),
                    "docs": list(record["docs"] or []),
                    "sections": decode_section_distribution(legacy, keys, values),
                    "times_seen": int(record["times_seen"] or 0),
                }
            return result

        with self._driver.session() as session:
            execute_read = getattr(session, "execute_read", None)
            if callable(execute_read):
                return execute_read(_fetch, node_ids)
            store = getattr(session, "_store", None)
            if store is None:
                return {}
            result: Dict[str, Dict[str, Any]] = {}
            for node_id in node_ids:
                existing = getattr(store, "nodes", {}).get(node_id)  # type: ignore[attr-defined]
                if existing is None:
                    continue
                result[node_id] = {
                    "aliases": list(existing.aliases),
                    "docs": list(existing.source_document_ids),
                    "sections": dict(existing.section_distribution),
                    "times_seen": int(existing.times_seen),
                }
            return result

    def _merge_with_existing(
        self,
        payload: Dict[str, Any],
        existing: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge a staged payload with existing graph state."""

        aliases = self._merge_unique_list(existing.get("aliases", []) if existing else [], payload["aliases"])
        aliases = [alias for alias in aliases if alias != payload["name"]]
        docs = self._merge_unique_list(existing.get("docs", []) if existing else [], payload["source_document_ids"])
        existing_sections = existing.get("sections", {}) if existing else {}
        merged_sections = self._merge_section_maps(existing_sections, payload["section_distribution"])
        merged_sections = {
            section: count
            for section, count in merged_sections.items()
            if count > 0
        }

        existing_times = existing.get("times_seen", 0) if existing else 0
        existing_doc_list = existing.get("docs", []) if existing else []
        existing_doc_set = set(existing_doc_list)
        new_docs = [doc for doc in payload["source_document_ids"] if doc not in existing_doc_set]
        payload_times = max(int(payload.get("times_seen", 0)), 0)
        increment = payload_times if payload_times > 0 else len(new_docs)
        times_seen = existing_times + increment
        keys, values = encode_section_distribution(merged_sections)

        return {
            "node_id": payload["node_id"],
            "name": payload["name"],
            "type": payload["type"],
            "aliases": aliases,
            "section_distribution": merged_sections,
            "section_distribution_keys": list(keys),
            "section_distribution_values": list(values),
            "times_seen": times_seen,
            "source_document_ids": docs,
            "pipeline_version": payload["pipeline_version"],
        }

    @staticmethod
    def _merge_unique_list(existing: Sequence[str], addition: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in list(existing) + list(addition):
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _merge_section_maps(
        existing: Mapping[str, int],
        addition: Mapping[str, int],
    ) -> Dict[str, int]:
        merged: Dict[str, int] = dict(existing)
        for section, count in addition.items():
            value = merged.get(section, 0) + count
            merged[section] = value
        return merged

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
        keys, values = encode_section_distribution(section_distribution)
        return {
            "node_id": node.node_id,
            "name": node.name,
            "type": node.type,
            "aliases": deduped_aliases,
            "section_distribution": section_distribution,
            "section_distribution_keys": list(keys),
            "section_distribution_values": list(values),
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
        evidence_payload = self._serialize_evidence(evidence)
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

    @staticmethod
    def _serialize_evidence(evidence: Evidence) -> str:
        """Convert an evidence contract into a JSON payload for Neo4j.

        Args:
            evidence: Evidence metadata referencing the supporting text.

        Returns:
            str: JSON string containing only Neo4j-compatible primitive values.
        """

        payload: Dict[str, Any] = {
            "doc_id": evidence.doc_id,
            "element_id": evidence.element_id,
            "text_span_start": evidence.text_span.start,
            "text_span_end": evidence.text_span.end,
        }
        if evidence.full_sentence is not None:
            payload["full_sentence"] = evidence.full_sentence
        return json.dumps(payload, sort_keys=True)

    def _ensure_entity_label_exists(self) -> bool:
        if self._entity_label_exists:
            return True
        try:
            with self._driver.session() as session:
                execute_read = getattr(session, "execute_read", None)
                if not callable(execute_read):
                    self._entity_label_exists = True
                    return True

                def _check(tx: Any) -> bool:
                    query = (
                        "CALL db.labels() YIELD label "
                        "WHERE label = $target "
                        "RETURN 1 AS present "
                        "LIMIT 1"
                    )
                    record = tx.run(query, target="Entity").single()
                    return record is not None

                exists = bool(execute_read(_check))
        except Exception:
            LOGGER.exception("Failed to determine whether Entity label exists")
            self._entity_label_exists = False
            return False
        self._entity_label_exists = exists
        return exists
