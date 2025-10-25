"""Neo4j-backed repository for QA operations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency in tests
    from neo4j import Driver, Record
except Exception:  # pragma: no cover - fallback for typing when neo4j unavailable
    Driver = object  # type: ignore[assignment]
    Record = object  # type: ignore[assignment]

from backend.app.config import AppConfig, load_config
from backend.app.graph.section_distribution import (
    decode_distribution_from_mapping,
    decode_section_distribution,
)
from backend.app.qa.entity_resolution import CandidateNode, QARepositoryProtocol

LOGGER = logging.getLogger(__name__)

_DEFAULT_ALLOWED_RELATIONS: Sequence[str] = (
    "uses",
    "trained-on",
    "evaluated-on",
    "compared-to",
    "outperforms",
    "causes",
    "correlates-with",
)


@dataclass(frozen=True)
class PathRecord:
    """Representation of a path returned from the database."""

    nodes: Sequence[Mapping[str, object]]
    relationships: Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class NeighborRecord:
    """Single hop neighbor relation returned when no path exists."""

    source: Mapping[str, object]
    target: Mapping[str, object]
    relationship: Mapping[str, object]


class Neo4jQARepository(QARepositoryProtocol):
    """Repository issuing Cypher queries for QA retrieval."""

    def __init__(
        self,
        driver: Driver,
        *,
        config: Optional[AppConfig] = None,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> None:
        self._driver = driver
        if allowed_relations is not None:
            relations = list(allowed_relations)
        else:
            cfg = config or load_config()
            relations = cfg.relations.canonical_relation_names()
        if not relations:
            relations = list(_DEFAULT_ALLOWED_RELATIONS)
        self._allowed_relations = tuple(sorted({relation for relation in relations}))

    def fetch_nodes_by_exact_match(self, mention: str) -> Sequence[CandidateNode]:
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) = toLower($mention)
           OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) = toLower($mention))
        RETURN n.node_id AS node_id,
               n.name AS name,
               coalesce(n.aliases, []) AS aliases,
               coalesce(n.times_seen, 0) AS times_seen,
               coalesce(properties(n)['section_distribution'], {}) AS section_distribution,
               coalesce(n.section_distribution_keys, []) AS section_distribution_keys,
               coalesce(n.section_distribution_values, []) AS section_distribution_values
        ORDER BY times_seen DESC, name ASC
        LIMIT 5
        """
        records = self._run_read(query, {"mention": mention})
        return [self._record_to_candidate(record) for record in records]

    def fetch_candidates_for_mention(
        self,
        mention: str,
        limit: int,
        *,
        tokens: Sequence[str] | None = None,
    ) -> Sequence[CandidateNode]:
        lowered_tokens = [token.lower() for token in tokens or [] if token]
        params = {
            "mention": mention,
            "tokens": lowered_tokens,
            "limit": limit,
        }
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($mention)
           OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS toLower($mention))
           OR ($tokens <> [] AND (
                any(token IN $tokens WHERE toLower(n.name) CONTAINS token) OR
                any(alias IN coalesce(n.aliases, []) WHERE any(token IN $tokens WHERE toLower(alias) CONTAINS token))
           ))
        RETURN n.node_id AS node_id,
               n.name AS name,
               coalesce(n.aliases, []) AS aliases,
               coalesce(n.times_seen, 0) AS times_seen,
               coalesce(properties(n)['section_distribution'], {}) AS section_distribution,
               coalesce(n.section_distribution_keys, []) AS section_distribution_keys,
               coalesce(n.section_distribution_values, []) AS section_distribution_values,
               CASE
                   WHEN toLower(n.name) = toLower($mention) THEN 0
                   WHEN any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) = toLower($mention)) THEN 0
                   ELSE 1
               END AS match_priority
        ORDER BY match_priority ASC, times_seen DESC, name ASC
        LIMIT $limit
        """
        records = self._run_read(query, params)
        return [self._record_to_candidate(record) for record in records]

    def fetch_candidate_nodes(self, limit: int) -> Sequence[CandidateNode]:
        query = """
        MATCH (n:Entity)
        RETURN n.node_id AS node_id,
               n.name AS name,
               coalesce(n.aliases, []) AS aliases,
               coalesce(n.times_seen, 0) AS times_seen,
               coalesce(properties(n)['section_distribution'], {}) AS section_distribution,
               coalesce(n.section_distribution_keys, []) AS section_distribution_keys,
               coalesce(n.section_distribution_values, []) AS section_distribution_values
        ORDER BY times_seen DESC, name ASC
        LIMIT $limit
        """
        records = self._run_read(query, {"limit": limit})
        return [self._record_to_candidate(record) for record in records]

    def fetch_paths(
        self,
        *,
        start_id: str,
        end_id: str,
        max_hops: int,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[PathRecord]:
        relation_filter = tuple(allowed_relations) if allowed_relations else self._allowed_relations
        # Neo4j does not allow parameterizing the variable length in relationship patterns.
        # Clamp hops to a safe integer and embed as a literal in the pattern; still filter by confidence and relation set.
        hops_upper = max(1, int(max_hops))
        query = (
            "\n        MATCH path = (start:Entity {node_id: $start_id})-\n            "
            f"[rel:RELATION*1..{hops_upper}]->(end:Entity {{node_id: $end_id}})\n"
            "        WHERE ALL(r IN rel WHERE r.confidence >= $min_confidence AND r.relation_norm IN $allowed)\n"
            "        RETURN [node IN nodes(path) | node] AS nodes,\n"
            "               [relationship IN rel | relationship] AS relationships\n"
            "        LIMIT $limit\n        "
        )
        params = {
            "start_id": start_id,
            "end_id": end_id,
            "min_confidence": min_confidence,
            "allowed": list(relation_filter),
            "limit": limit,
        }
        records = self._run_read(query, params)
        return [self._record_to_path(record) for record in records]

    def fetch_neighbors(
        self,
        node_id: str,
        *,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[NeighborRecord]:
        relation_filter = tuple(allowed_relations) if allowed_relations else self._allowed_relations
        query = """
        MATCH (source:Entity {node_id: $node_id})-[rel:RELATION]->(target:Entity)
        WHERE rel.confidence >= $min_confidence AND rel.relation_norm IN $allowed
        RETURN source AS source, target AS target, rel AS relationship
        ORDER BY rel.confidence DESC, rel.created_at DESC
        LIMIT $limit
        """
        params = {
            "node_id": node_id,
            "min_confidence": min_confidence,
            "allowed": list(relation_filter),
            "limit": limit,
        }
        records = self._run_read(query, params)
        return [self._record_to_neighbor(record) for record in records]

    def _run_read(self, query: str, params: Mapping[str, object]) -> List[Record]:
        def _execute(tx):  # type: ignore[no-untyped-def]
            return list(tx.run(query, **params))

        with self._driver.session() as session:
            return session.execute_read(_execute)

    @staticmethod
    def _record_to_candidate(record: Record) -> CandidateNode:
        data = record.data()
        distribution = decode_section_distribution(
            data.get("section_distribution"),
            data.get("section_distribution_keys"),
            data.get("section_distribution_values"),
        )
        aliases = [str(alias) for alias in data.get("aliases") or []]
        return CandidateNode(
            node_id=str(data["node_id"]),
            name=str(data["name"]),
            aliases=aliases,
            times_seen=int(data.get("times_seen", 0) or 0),
            section_distribution=distribution,
        )

    @staticmethod
    def _record_to_path(record: Record) -> PathRecord:
        nodes = []
        for value in record["nodes"]:
            node_data = dict(_safe_map(value))
            node_data["section_distribution"] = decode_distribution_from_mapping(node_data)
            nodes.append(node_data)
        relationships = [dict(_safe_map(value)) for value in record["relationships"]]
        for relation in relationships:
            relation["evidence"] = _parse_evidence(relation.get("evidence"))
            relation["attributes"] = _parse_attributes(relation.get("attributes"))
        return PathRecord(nodes=nodes, relationships=relationships)

    @staticmethod
    def _record_to_neighbor(record: Record) -> NeighborRecord:
        source = dict(_safe_map(record["source"]))
        source["section_distribution"] = decode_distribution_from_mapping(source)
        target = dict(_safe_map(record["target"]))
        target["section_distribution"] = decode_distribution_from_mapping(target)
        relationship: MutableMapping[str, object] = dict(_safe_map(record["relationship"]))
        relationship["evidence"] = _parse_evidence(relationship.get("evidence"))
        relationship["attributes"] = _parse_attributes(relationship.get("attributes"))
        return NeighborRecord(source=source, target=target, relationship=relationship)


def _safe_map(value: object) -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        return value.items()
    if hasattr(value, "items"):
        return value.items()  # type: ignore[no-any-return]
    if hasattr(value, "_properties"):
        return value._properties.items()  # type: ignore[attr-defined,no-any-return]
    LOGGER.debug("Unexpected value type when converting Neo4j record: %s", type(value))
    return []


def _parse_evidence(value: object) -> Mapping[str, object]:
    """Decode relationship evidence into a mapping."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode evidence payload from JSON")
            return {}
        if isinstance(parsed, Mapping):
            return parsed
        LOGGER.warning("Evidence payload JSON did not decode into a mapping")
    elif value is not None:
        LOGGER.warning("Unexpected evidence payload type: %s", type(value))
    return {}


def _parse_attributes(value: object) -> Mapping[str, str]:
    """Decode relationship attributes into a string map."""

    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items() if k}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode attributes payload from JSON")
            return {}
        if isinstance(decoded, Mapping):
            return {str(k): str(v) for k, v in decoded.items() if k}
        LOGGER.warning("Attributes payload JSON did not decode into a mapping")
    elif value is not None:
        LOGGER.warning("Unexpected attributes payload type: %s", type(value))
    return {}
