"""Integration tests for the Neo4j graph view repository."""

from __future__ import annotations

from typing import Optional

import contextlib

import pytest

from backend.app.ui.repository import GraphViewFilters, Neo4jGraphViewRepository

try:  # pragma: no cover - optional dependency in CI
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - allows test skipping when driver missing
    GraphDatabase = None  # type: ignore[assignment]

try:  # pragma: no cover - testcontainers optional
    from testcontainers.neo4j import Neo4jContainer
except Exception:  # pragma: no cover
    Neo4jContainer = None  # type: ignore[assignment]


def _docker_daemon_available() -> bool:
    try:
        import docker
    except Exception:  # pragma: no cover - docker client optional
        return False
    try:
        client = docker.from_env()
    except Exception:  # pragma: no cover - environment missing docker socket
        return False
    try:
        client.ping()
        return True
    except Exception:  # pragma: no cover - ping failed
        return False
    finally:
        with contextlib.suppress(Exception):
            client.close()


DOCKER_AVAILABLE = _docker_daemon_available()


@pytest.mark.skipif(
    Neo4jContainer is None or GraphDatabase is None or not DOCKER_AVAILABLE,
    reason="neo4j driver and testcontainers are required for integration tests",
)
def test_repository_applies_filters() -> None:
    container: Optional[Neo4jContainer] = None
    try:
        container = Neo4jContainer("neo4j:5.21").with_env("NEO4J_AUTH", "neo4j/test")
        container.start()
        uri = container.get_connection_url()
        assert GraphDatabase is not None
        driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))
        with driver.session() as session:
            session.run(
                """
                CREATE (a:Entity {node_id: 'n1', name: 'Alpha', aliases: ['Alpha'],
                                   section_distribution: {Results: 2}, times_seen: 3})
                CREATE (b:Entity {node_id: 'n2', name: 'Beta', aliases: ['Beta'],
                                   section_distribution: {Results: 1}, times_seen: 2})
                CREATE (a)-[:RELATION {
                    relation_norm: 'uses',
                    relation_verbatim: 'uses',
                    confidence: 0.9,
                    times_seen: 1,
                    attributes: {method: 'llm', section: 'Results'},
                    evidence: {doc_id: 'doc1', element_id: 'el1', text_span: {start: 0, end: 10}}
                }]->(b)
                CREATE (a)-[:RELATION {
                    relation_norm: 'uses',
                    relation_verbatim: 'uses',
                    confidence: 0.3,
                    times_seen: 1,
                    attributes: {method: 'llm', section: 'Background'},
                    evidence: {doc_id: 'doc1', element_id: 'el2', text_span: {start: 0, end: 10}}
                }]->(b)
                """
            )
        repository = Neo4jGraphViewRepository(driver)
        filters = GraphViewFilters(
            relations=["uses"],
            min_confidence=0.5,
            sections=["Results"],
            include_co_mentions=False,
            papers=[],
            limit=10,
        )
        edges = repository.fetch_edges(filters)
        assert len(edges) == 1
        relation = edges[0].relation
        assert relation["confidence"] >= 0.5
        assert relation["attributes"]["section"] == "Results"
    finally:
        if container is not None:
            container.stop()
