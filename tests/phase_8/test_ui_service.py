"""Unit tests for the graph view service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from backend.app.ui.repository import GraphEdgeRecord, GraphNodeRecord, GraphViewFilters, GraphViewRepositoryProtocol
from backend.app.ui.service import GraphEdge, GraphNode, GraphViewService


@dataclass
class _StubRepository(GraphViewRepositoryProtocol):
    edges: Sequence[GraphEdgeRecord]
    cleared: bool = False

    def fetch_edges(self, filters: GraphViewFilters):  # type: ignore[override]
        self.filters = filters  # type: ignore[attr-defined]
        return list(self.edges)

    def clear_graph(self) -> None:  # type: ignore[override]
        self.cleared = True


def _make_node(node_id: str, name: str) -> GraphNodeRecord:
    return GraphNodeRecord(
        node_id=node_id,
        name=name,
        type="Entity",
        aliases=[name.upper()],
        times_seen=3,
        section_distribution={"Results": 2},
        source_document_ids=[f"{node_id}-doc"],
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
    assert isinstance(node.x, float)
    assert isinstance(node.y, float)
    edge = view.edges[0]
    assert isinstance(edge, GraphEdge)
    assert edge.relation == "uses"
    assert edge.attributes["method"] == "llm"
    assert edge.evidence["doc_id"] == "doc1"
    assert node.importance is not None
    assert 0.0 <= node.importance <= 1.0
    assert node.layout_ring is not None


def test_graph_view_service_decodes_attribute_strings() -> None:
    source = _make_node("n1", "Alpha")
    target = _make_node("n2", "Beta")
    relation = {
        "relation_norm": "uses",
        "confidence": 0.75,
        "times_seen": 1,
        "attributes": '{"section":"Results","method":"llm"}',
    }
    repo = _StubRepository(edges=[GraphEdgeRecord(source=source, target=target, relation=relation)])
    service = GraphViewService(repo)

    view = service.fetch_graph(
        relations=["uses"],
        min_confidence=0.0,
        sections=[],
        include_co_mentions=True,
    )

    assert view.edges[0].attributes == {"method": "llm", "section": "Results"}


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


def test_graph_view_service_clears_repository() -> None:
    repo = _StubRepository(edges=[])
    service = GraphViewService(repo)

    service.clear_graph()

    assert repo.cleared is True
