"""Unit tests for the graph view service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from backend.app.ui.repository import GraphEdgeRecord, GraphNodeRecord, GraphViewFilters, GraphViewRepositoryProtocol
from backend.app.ui.service import GraphEdge, GraphNode, GraphViewService


@dataclass
class _StubRepository(GraphViewRepositoryProtocol):
    edges: Sequence[GraphEdgeRecord]

    def fetch_edges(self, filters: GraphViewFilters):  # type: ignore[override]
        self.filters = filters  # type: ignore[attr-defined]
        return list(self.edges)


def _make_node(node_id: str, name: str) -> GraphNodeRecord:
    return GraphNodeRecord(
        node_id=node_id,
        name=name,
        type="Entity",
        aliases=[name.upper()],
        times_seen=3,
        section_distribution={"Results": 2},
    )


def test_graph_view_service_converts_records() -> None:
    source = _make_node("n1", "Alpha")
    target = _make_node("n2", "Beta")
    relation = {
        "relation_norm": "uses",
        "relation_verbatim": "uses",
        "confidence": 0.82,
        "times_seen": 2,
        "attributes": {"method": "llm", "section": "Results"},
        "evidence": {
            "doc_id": "doc1",
            "element_id": "el1",
            "text_span": {"start": 0, "end": 10},
            "full_sentence": "Alpha uses Beta.",
        },
        "conflicting": False,
    }
    record = GraphEdgeRecord(source=source, target=target, relation=relation)
    repo = _StubRepository(edges=[record])
    service = GraphViewService(repo)

    view = service.fetch_graph(
        relations=["uses"],
        min_confidence=0.5,
        sections=["Results"],
        include_co_mentions=False,
    )

    assert view.node_count == 2
    assert view.edge_count == 1
    node = view.nodes[0]
    assert isinstance(node, GraphNode)
    assert node.label == "Alpha"
    edge = view.edges[0]
    assert isinstance(edge, GraphEdge)
    assert edge.relation == "uses"
    assert edge.attributes["method"] == "llm"
    assert edge.evidence["doc_id"] == "doc1"


def test_graph_view_service_respects_limit() -> None:
    source = _make_node("n1", "Alpha")
    target = _make_node("n2", "Beta")
    relation = {"relation_norm": "uses", "confidence": 0.7, "times_seen": 1}
    repo = _StubRepository(edges=[GraphEdgeRecord(source=source, target=target, relation=relation)])
    service = GraphViewService(repo, default_limit=10)

    service.fetch_graph(
        relations=["uses"],
        min_confidence=0.5,
        sections=["Results"],
        include_co_mentions=True,
        limit=5,
    )

    assert repo.filters.limit == 5
