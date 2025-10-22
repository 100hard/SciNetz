"""Graph view service RBAC tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pytest

from backend.app.ui.repository import (
    GraphEdgeRecord,
    GraphNodeRecord,
    GraphViewFilters,
    GraphViewRepositoryProtocol,
)
from backend.app.ui.service import GraphView, GraphViewService


@dataclass
class _StubRepository(GraphViewRepositoryProtocol):
    """Repository returning a static graph for testing."""

    edge: GraphEdgeRecord

    def fetch_edges(
        self, filters: GraphViewFilters, allowed_papers: Optional[Sequence[str]] = None
    ) -> Sequence[GraphEdgeRecord]:
        return [self.edge]

    def clear_graph(self) -> None:  # pragma: no cover - unused in tests
        return None


def _build_stub_edge(doc_id: str) -> GraphEdgeRecord:
    node = GraphNodeRecord(
        node_id="n1",
        name="Node 1",
        type=None,
        aliases=[],
        times_seen=1,
        section_distribution={},
        source_document_ids=[doc_id],
    )
    relation = {
        "relation_norm": "related",
        "confidence": 0.9,
        "evidence": {"doc_id": doc_id, "text_span": {"start": 0, "end": 5}},
    }
    return GraphEdgeRecord(source=node, target=node, relation=relation)


def test_fetch_graph_rejects_unauthorised_papers() -> None:
    repository = _StubRepository(edge=_build_stub_edge("paper-allowed"))
    service = GraphViewService(repository, default_limit=5)

    authorised_view = service.fetch_graph(
        relations=[],
        min_confidence=0.0,
        sections=[],
        include_co_mentions=True,
        allowed_papers=["paper-allowed"],
    )
    assert isinstance(authorised_view, GraphView)
    assert authorised_view.edge_count == 1

    empty_view = service.fetch_graph(
        relations=[],
        min_confidence=0.0,
        sections=[],
        include_co_mentions=True,
        allowed_papers=[],
    )
    assert empty_view.edge_count == 0

    with pytest.raises(PermissionError):
        service.fetch_graph(
            relations=[],
            min_confidence=0.0,
            sections=[],
            include_co_mentions=True,
            papers=["paper-denied"],
            allowed_papers=["paper-allowed"],
        )
