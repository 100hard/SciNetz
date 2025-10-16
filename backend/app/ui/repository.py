"""Graph data access layer for UI visualisations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from neo4j import Driver, Record
from typing_extensions import Protocol

from backend.app.graph.section_distribution import decode_distribution_from_mapping


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphViewFilters:
    """Filter parameters applied when retrieving graph edges."""

    relations: Sequence[str]
    min_confidence: float
    sections: Sequence[str]
    include_co_mentions: bool
    papers: Sequence[str]
    limit: int


@dataclass(frozen=True)
class GraphNodeRecord:
    """Node payload returned from Neo4j queries."""

    node_id: str
    name: str
    type: Optional[str]
    aliases: Sequence[str]
    times_seen: int
    section_distribution: Mapping[str, int]
    source_document_ids: Sequence[str]


@dataclass(frozen=True)
class GraphEdgeRecord:
    """Edge payload linking two node records."""

    source: GraphNodeRecord
    target: GraphNodeRecord
    relation: Mapping[str, object]


class GraphViewRepositoryProtocol(Protocol):
    """Protocol describing repository methods for graph retrieval."""

    def fetch_edges(self, filters: GraphViewFilters) -> Sequence[GraphEdgeRecord]:
        """Return graph edges satisfying the provided filters."""

    def clear_graph(self) -> None:
        """Delete graph nodes and edges accessible to the UI."""


class Neo4jGraphViewRepository(GraphViewRepositoryProtocol):
    """Concrete repository issuing Cypher queries for graph views."""

    _QUERY = """
        MATCH (src)-[rel]->(dst)
        RETURN src, dst, rel
        LIMIT $limit
    """

    _CLEAR_QUERY = "MATCH (node:Entity) DETACH DELETE node"

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def fetch_edges(self, filters: GraphViewFilters) -> Sequence[GraphEdgeRecord]:
        params = {"limit": self._query_limit(filters)}
        records = self._run_query(params)
        edges = [self._record_to_edge(record) for record in records]
        edges = self._apply_confidence_filter(edges, filters.min_confidence)
        if filters.relations:
            edges = self._apply_relation_filter(edges, filters.relations)
        if not filters.include_co_mentions:
            edges = [edge for edge in edges if not self._is_co_mention(edge)]
        if filters.papers:
            edges = self._apply_paper_filter(edges, filters.papers)
        filtered = self._apply_section_filter(edges, filters.sections)
        filtered = self._sort_edges(filtered)
        limit = max(int(filters.limit), 0)
        return filtered[:limit] if limit else []

    @staticmethod
    def _query_limit(filters: GraphViewFilters) -> int:
        base_limit = max(int(filters.limit), 0)
        if not base_limit:
            return 0
        section_filter_active = any(section.strip() for section in filters.sections)
        multiplier = 3 if section_filter_active else 1
        if not filters.include_co_mentions:
            multiplier = max(multiplier, 2)
        if filters.papers:
            multiplier = max(multiplier, 3)
        if filters.relations:
            multiplier = max(multiplier, 2)
        expanded = base_limit * multiplier
        return min(expanded, 3000)

    @staticmethod
    def _apply_confidence_filter(
        edges: Sequence[GraphEdgeRecord], threshold: float
    ) -> List[GraphEdgeRecord]:
        if threshold <= 0:
            return list(edges)
        filtered: List[GraphEdgeRecord] = []
        for edge in edges:
            confidence = _as_float(edge.relation.get("confidence"))
            if confidence >= threshold:
                filtered.append(edge)
        return filtered

    @staticmethod
    def _apply_relation_filter(
        edges: Sequence[GraphEdgeRecord], relations: Sequence[str]
    ) -> List[GraphEdgeRecord]:
        allowed = {relation.strip().lower() for relation in relations if relation.strip()}
        if not allowed:
            return list(edges)
        filtered: List[GraphEdgeRecord] = []
        for edge in edges:
            relation_norm = str(edge.relation.get("relation_norm", "")).strip().lower()
            if relation_norm in allowed:
                filtered.append(edge)
        return filtered

    @staticmethod
    def _apply_section_filter(
        edges: Sequence[GraphEdgeRecord], sections: Sequence[str]
    ) -> List[GraphEdgeRecord]:
        trimmed = [section.strip() for section in sections if section.strip()]
        if not trimmed:
            return list(edges)
        allowed = {section for section in trimmed}
        filtered: List[GraphEdgeRecord] = []
        for edge in edges:
            relation_sections = set(
                Neo4jGraphViewRepository._extract_sections(edge.relation.get("attributes"))
            )
            if not relation_sections:
                relation_sections = Neo4jGraphViewRepository._sections_from_nodes(edge)
            else:
                relation_sections |= Neo4jGraphViewRepository._sections_from_nodes(edge)
            if any(section in allowed for section in relation_sections):
                filtered.append(edge)
        if not filtered and trimmed:
            return list(edges)
        return filtered

    @staticmethod
    def _sort_edges(edges: Sequence[GraphEdgeRecord]) -> List[GraphEdgeRecord]:
        def _sort_key(edge: GraphEdgeRecord) -> tuple:
            confidence = _as_float(edge.relation.get("confidence"))
            created_at = edge.relation.get("created_at")
            timestamp = 0.0
            if isinstance(created_at, datetime):
                timestamp = created_at.timestamp()
            elif isinstance(created_at, str):
                timestamp = _parse_datetime(created_at)
            return (-confidence, -timestamp)

        return sorted(edges, key=_sort_key)

    @staticmethod
    def _apply_paper_filter(
        edges: Sequence[GraphEdgeRecord], papers: Sequence[str]
    ) -> List[GraphEdgeRecord]:
        allowed = {paper.strip() for paper in papers if paper.strip()}
        if not allowed:
            return list(edges)
        filtered: List[GraphEdgeRecord] = []
        for edge in edges:
            if Neo4jGraphViewRepository._matches_papers(edge, allowed):
                filtered.append(edge)
        return filtered

    @staticmethod
    def _extract_sections(attributes: object) -> List[str]:
        sections: List[str] = []
        if isinstance(attributes, Mapping):
            sections.extend(
                Neo4jGraphViewRepository._normalise_section_values(
                    attributes.get("section")
                )
            )
            sections.extend(
                Neo4jGraphViewRepository._normalise_section_values(
                    attributes.get("sections")
                )
            )
        elif isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes)):
            for item in attributes:
                if isinstance(item, Mapping):
                    key = (item.get("key") or item.get("name") or "").lower()
                    if key in {"section", "sections"}:
                        sections.extend(
                            Neo4jGraphViewRepository._normalise_section_values(
                                item.get("value")
                            )
                        )
                elif isinstance(item, str):
                    sections.extend(
                        Neo4jGraphViewRepository._normalise_section_values(item)
                    )
        return [section for section in (section.strip() for section in sections) if section]
    @staticmethod
    def _sections_from_nodes(edge: GraphEdgeRecord) -> set[str]:
        sections: set[str] = set()
        for distribution in (edge.source.section_distribution, edge.target.section_distribution):
            if isinstance(distribution, Mapping):
                for section, count in distribution.items():
                    if count and section:
                        sections.add(str(section).strip())
        return sections

    @staticmethod
    def _normalise_section_values(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            collected: List[str] = []
            for item in value:
                if isinstance(item, str):
                    collected.extend(
                        Neo4jGraphViewRepository._normalise_section_values(item)
                    )
                elif item is not None:
                    collected.append(str(item).strip())
            return [section for section in collected if section]
        return [str(value).strip()]

    def _run_query(self, params: Mapping[str, object]) -> List[Record]:
        def _execute(tx):  # type: ignore[no-untyped-def]
            return list(tx.run(self._QUERY, **params))

        with self._driver.session() as session:
            return session.execute_read(_execute)

    def clear_graph(self) -> None:
        def _execute(tx):  # type: ignore[no-untyped-def]
            tx.run(self._CLEAR_QUERY)

        with self._driver.session() as session:
            session.execute_write(_execute)

    @staticmethod
    def _record_to_edge(record: Record) -> GraphEdgeRecord:
        source = Neo4jGraphViewRepository._to_node(record["src"])
        target = Neo4jGraphViewRepository._to_node(record["dst"])
        relation = Neo4jGraphViewRepository._to_relation(record["rel"])
        return GraphEdgeRecord(source=source, target=target, relation=relation)

    @staticmethod
    def _to_node(raw: Mapping[str, object]) -> GraphNodeRecord:
        if hasattr(raw, "items"):
            data = dict(raw.items())  # type: ignore[assignment]
        else:
            data = dict(raw)
        aliases = list(data.get("aliases", []) or [])
        distribution = decode_distribution_from_mapping(data)
        node_type = data.get("type")
        doc_ids_raw = data.get("source_document_ids", []) or []
        doc_ids = [str(doc_id) for doc_id in doc_ids_raw if str(doc_id).strip()]
        return GraphNodeRecord(
            node_id=str(data.get("node_id")),
            name=str(data.get("name")),
            type=str(node_type) if node_type is not None else None,
            aliases=[str(alias) for alias in aliases],
            times_seen=int(data.get("times_seen", 0) or 0),
            section_distribution=distribution,
            source_document_ids=doc_ids,
        )

    @staticmethod
    def _to_relation(raw: Mapping[str, object]) -> MutableMapping[str, object]:
        if hasattr(raw, "items"):
            items = raw.items()
        else:
            items = raw
        relation: MutableMapping[str, object] = {}
        for key, value in items:
            relation[str(key)] = value
        relation["attributes"] = Neo4jGraphViewRepository._attributes_from_payload(
            relation.get("attributes")
        )
        evidence = relation.get("evidence")
        if isinstance(evidence, Mapping):
            relation["evidence"] = Neo4jGraphViewRepository._normalise_evidence(evidence)
        elif isinstance(evidence, str):
            relation["evidence"] = Neo4jGraphViewRepository._evidence_from_json(evidence)
        return relation

    @staticmethod
    def _matches_papers(edge: GraphEdgeRecord, allowed: set[str]) -> bool:
        evidence = edge.relation.get("evidence")
        if isinstance(evidence, Mapping):
            doc_id = evidence.get("doc_id")
            if doc_id is not None and str(doc_id).strip() in allowed:
                return True
        elif isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            for entry in evidence:
                if isinstance(entry, Mapping):
                    doc_id = entry.get("doc_id") or entry.get("document_id")
                    if doc_id is not None and str(doc_id).strip() in allowed:
                        return True
        return False

    @staticmethod
    def _is_co_mention(edge: GraphEdgeRecord) -> bool:
        attributes = edge.relation.get("attributes")
        if isinstance(attributes, Mapping):
            value = attributes.get("method")
            if isinstance(value, str) and value.strip().lower() == "co-mention":
                return True
            if value is not None:
                return str(value).strip().lower() == "co-mention"
        elif isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes)):
            for item in attributes:
                if isinstance(item, Mapping):
                    key = (item.get("key") or item.get("name") or "").lower()
                    if key == "method":
                        value = item.get("value")
                        if isinstance(value, str):
                            return value.strip().lower() == "co-mention"
                        return str(value).strip().lower() == "co-mention"
                elif isinstance(item, str):
                    if item.strip().lower() == "co-mention":
                        return True
        return False

    @staticmethod
    def _normalise_evidence(evidence: Mapping[str, object]) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        doc_id = evidence.get("doc_id")
        if doc_id is not None:
            payload["doc_id"] = doc_id
        element_id = evidence.get("element_id")
        if element_id is not None:
            payload["element_id"] = element_id

        raw_span = evidence.get("text_span")
        if isinstance(raw_span, Mapping):
            start = int(raw_span.get("start", 0) or 0)
            end = int(raw_span.get("end", 0) or 0)
        else:
            start = int(evidence.get("text_span_start", 0) or 0)
            end = int(evidence.get("text_span_end", 0) or 0)
        payload["text_span"] = {"start": start, "end": end}

        if "full_sentence" in evidence and evidence["full_sentence"] is not None:
            payload["full_sentence"] = evidence["full_sentence"]
        return payload

    @staticmethod
    def _evidence_from_json(payload: str) -> Dict[str, object]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode evidence payload from JSON in UI repository")
            return {}
        if not isinstance(data, Mapping):
            LOGGER.warning("Decoded evidence payload is not a mapping in UI repository")
            return {}
        return Neo4jGraphViewRepository._normalise_evidence(data)

    @staticmethod
    def _attributes_from_payload(value: object) -> Dict[str, str]:
        if isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items() if k}
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to decode attributes payload from JSON in UI repository")
                return {}
            if isinstance(decoded, Mapping):
                return {str(k): str(v) for k, v in decoded.items() if k}
            LOGGER.warning("Decoded attributes payload is not a mapping in UI repository")
            return {}
        if value is not None:
            LOGGER.warning("Unexpected attributes payload type in UI repository: %s", type(value))
        return {}


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _parse_datetime(raw: str) -> float:
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0
