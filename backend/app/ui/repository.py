"""Graph data access layer for UI visualisations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from neo4j import Driver, Record
from typing_extensions import Protocol


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


class Neo4jGraphViewRepository(GraphViewRepositoryProtocol):
    """Concrete repository issuing Cypher queries for graph views."""

    _QUERY = """
        MATCH (src:Entity)-[rel:RELATION]->(dst:Entity)
        WITH
            src,
            dst,
            rel,
            CASE
                WHEN rel.attributes IS NULL THEN []
                WHEN rel.attributes['section'] IS NOT NULL THEN [rel.attributes['section']]
                WHEN rel.attributes['sections'] IS NOT NULL THEN split(rel.attributes['sections'], ',')
                ELSE []
            END AS relation_sections
        WHERE rel.confidence >= $min_confidence
          AND ($relations = [] OR rel.relation_norm IN $relations)
          AND ($include_co_mentions OR coalesce(rel.attributes['method'], '') <> 'co-mention')
          AND (
                $sections = []
                OR ANY(section IN relation_sections WHERE trim(section) IN $sections)
            )
          AND (
                $papers = []
                OR rel.evidence['doc_id'] IN $papers
            )
        RETURN src, dst, rel
        ORDER BY rel.confidence DESC, rel.created_at DESC
        LIMIT $limit
    """

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def fetch_edges(self, filters: GraphViewFilters) -> Sequence[GraphEdgeRecord]:
        params = {
            "min_confidence": filters.min_confidence,
            "relations": list(filters.relations),
            "sections": [section.strip() for section in filters.sections if section.strip()],
            "include_co_mentions": bool(filters.include_co_mentions),
            "papers": list(filters.papers),
            "limit": int(filters.limit),
        }
        records = self._run_query(params)
        return [self._record_to_edge(record) for record in records]

    def _run_query(self, params: Mapping[str, object]) -> List[Record]:
        def _execute(tx):  # type: ignore[no-untyped-def]
            return list(tx.run(self._QUERY, **params))

        with self._driver.session() as session:
            return session.execute_read(_execute)

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
        section_distribution = dict(data.get("section_distribution", {}) or {})
        node_type = data.get("type")
        return GraphNodeRecord(
            node_id=str(data.get("node_id")),
            name=str(data.get("name")),
            type=str(node_type) if node_type is not None else None,
            aliases=[str(alias) for alias in aliases],
            times_seen=int(data.get("times_seen", 0) or 0),
            section_distribution={str(key): int(value) for key, value in section_distribution.items()},
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
        attributes = relation.get("attributes")
        if isinstance(attributes, Mapping):
            relation["attributes"] = {str(k): str(v) for k, v in attributes.items()}
        evidence = relation.get("evidence")
        if isinstance(evidence, Mapping):
            relation["evidence"] = Neo4jGraphViewRepository._normalise_evidence(evidence)
        return relation

    @staticmethod
    def _normalise_evidence(evidence: Mapping[str, object]) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        for key, value in evidence.items():
            if key == "text_span" and isinstance(value, Mapping):
                payload[key] = {
                    "start": int(value.get("start", 0) or 0),
                    "end": int(value.get("end", 0) or 0),
                }
            else:
                payload[key] = value
        return payload

