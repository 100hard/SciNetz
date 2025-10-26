from __future__ import annotations

import types
from typing import Mapping, Sequence

from backend.app.qa.repository import Neo4jQARepository


def _make_repository(default_relations: Sequence[str] | None = ("related-to",)) -> Neo4jQARepository:
    repo = Neo4jQARepository.__new__(Neo4jQARepository)  # type: ignore[call-arg]
    repo._driver = None  # type: ignore[attr-defined]
    repo._allowed_relations = tuple(default_relations) if default_relations else None  # type: ignore[attr-defined]
    repo._captured_query = None  # type: ignore[attr-defined]
    repo._captured_params = None  # type: ignore[attr-defined]

    def _capture(self: Neo4jQARepository, query: str, params: Mapping[str, object]):
        self._captured_query = query
        self._captured_params = params
        return []

    repo._run_read = types.MethodType(_capture, repo)  # type: ignore[attr-defined]
    return repo


def test_fetch_neighbors_omits_relation_filter_for_empty_selection() -> None:
    repo = _make_repository()
    repo.fetch_neighbors("node-1", min_confidence=0.5, limit=5, allowed_relations=())

    captured_query = repo._captured_query  # type: ignore[attr-defined]
    captured_params = repo._captured_params  # type: ignore[attr-defined]

    assert captured_query is not None
    assert "rel.relation_norm IN $allowed" not in captured_query
    assert "allowed" not in captured_params


def test_fetch_neighbors_uses_default_relation_filter() -> None:
    repo = _make_repository(("rel-a", "rel-b"))
    repo.fetch_neighbors("node-1", min_confidence=0.5, limit=5, allowed_relations=None)

    captured_query = repo._captured_query  # type: ignore[attr-defined]
    captured_params = repo._captured_params  # type: ignore[attr-defined]

    assert captured_query is not None
    assert "rel.relation_norm IN $allowed" in captured_query
    assert captured_params.get("allowed") == ["rel-a", "rel-b"]


def test_fetch_document_edges_supports_unbounded_relations() -> None:
    repo = _make_repository()
    repo.fetch_document_edges("doc-1", min_confidence=0.5, limit=5, allowed_relations=())

    captured_query = repo._captured_query  # type: ignore[attr-defined]
    captured_params = repo._captured_params  # type: ignore[attr-defined]

    assert captured_query is not None
    assert "rel.relation_norm IN $allowed" not in captured_query
    assert "allowed" not in captured_params
