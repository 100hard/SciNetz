"""Services powering the UI graph view responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from backend.app.ui.repository import GraphEdgeRecord, GraphNodeRecord, GraphViewFilters, GraphViewRepositoryProtocol


@dataclass(frozen=True)
class GraphNode:
    """Node payload returned to UI clients."""

    id: str
    label: str
    type: Optional[str]
    aliases: List[str]
    times_seen: int
    section_distribution: Dict[str, int]


@dataclass(frozen=True)
class GraphEdge:
    """Edge payload with evidence metadata for UI rendering."""

    id: str
    source: str
    target: str
    relation: str
    relation_verbatim: str
    confidence: float
    times_seen: int
    attributes: Dict[str, str]
    evidence: Dict[str, object]
    conflicting: bool
    created_at: Optional[str]


@dataclass(frozen=True)
class GraphView:
    """Container for a graph response including summary counts."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class GraphViewService:
    """Assemble graph nodes and edges from repository records."""

    def __init__(self, repository: GraphViewRepositoryProtocol, *, default_limit: int = 500) -> None:
        self._repository = repository
        self._default_limit = default_limit

    def fetch_graph(
        self,
        *,
        relations: Sequence[str],
        min_confidence: float,
        sections: Sequence[str],
        include_co_mentions: bool,
        papers: Sequence[str] = (),
        limit: Optional[int] = None,
    ) -> GraphView:
        """Fetch a graph using the supplied filters."""

        filters = GraphViewFilters(
            relations=relations,
            min_confidence=min_confidence,
            sections=sections,
            include_co_mentions=include_co_mentions,
            papers=papers,
            limit=limit or self._default_limit,
        )
        records = self._repository.fetch_edges(filters)
        node_map: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []
        for record in records:
            if record.source.node_id not in node_map:
                node_map[record.source.node_id] = self._node_from_record(record.source)
            if record.target.node_id not in node_map:
                node_map[record.target.node_id] = self._node_from_record(record.target)
            edges.append(self._edge_from_record(record))
        return GraphView(nodes=list(node_map.values()), edges=edges)

    @staticmethod
    def _node_from_record(record: GraphNodeRecord) -> GraphNode:
        return GraphNode(
            id=record.node_id,
            label=record.name,
            type=record.type,
            aliases=list(record.aliases),
            times_seen=record.times_seen,
            section_distribution=dict(record.section_distribution),
        )

    @staticmethod
    def _edge_from_record(record: GraphEdgeRecord) -> GraphEdge:
        relation = dict(record.relation)
        relation_norm = str(relation.get("relation_norm", ""))
        relation_verbatim = str(relation.get("relation_verbatim") or relation_norm)
        confidence = float(relation.get("confidence", 0.0) or 0.0)
        times_seen = int(relation.get("times_seen", 0) or 0)
        conflicting = bool(relation.get("conflicting", False))
        attributes = GraphViewService._to_str_map(relation.get("attributes"))
        evidence = GraphViewService._to_evidence(relation.get("evidence"))
        created_at = GraphViewService._to_iso8601(relation.get("created_at"))
        edge_id = GraphViewService._build_edge_id(
            record.source.node_id,
            record.target.node_id,
            relation_norm,
            created_at,
        )
        return GraphEdge(
            id=edge_id,
            source=record.source.node_id,
            target=record.target.node_id,
            relation=relation_norm,
            relation_verbatim=relation_verbatim,
            confidence=confidence,
            times_seen=times_seen,
            attributes=attributes,
            evidence=evidence,
            conflicting=conflicting,
            created_at=created_at,
        )

    @staticmethod
    def _to_str_map(value: object) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(val) for key, val in value.items()}

    @staticmethod
    def _to_evidence(value: object) -> Dict[str, object]:
        if not isinstance(value, Mapping):
            return {}
        doc_id = value.get("doc_id")
        element_id = value.get("element_id")
        raw_span = value.get("text_span")
        if isinstance(raw_span, Mapping):
            start = int(raw_span.get("start", 0) or 0)
            end = int(raw_span.get("end", 0) or 0)
        else:
            start = int(value.get("text_span_start", 0) or 0)
            end = int(value.get("text_span_end", 0) or 0)
        payload: Dict[str, object] = {
            "doc_id": doc_id,
            "element_id": element_id,
            "text_span": {"start": start, "end": end},
        }
        if value.get("full_sentence") is not None:
            payload["full_sentence"] = value["full_sentence"]
        return payload

    @staticmethod
    def _to_iso8601(value: object) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # pragma: no cover - defensive fallback
                pass
        if hasattr(value, "to_native"):
            native = value.to_native()  # type: ignore[call-arg]
            if hasattr(native, "isoformat"):
                return native.isoformat()
        return None

    @staticmethod
    def _build_edge_id(
        source_id: str,
        target_id: str,
        relation: str,
        created_at: Optional[str],
    ) -> str:
        suffix = created_at or "unknown"
        return f"{source_id}->{target_id}:{relation}:{suffix}"

