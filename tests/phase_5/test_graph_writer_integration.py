"""Integration tests for the Neo4j graph writer."""
from __future__ import annotations

from time import perf_counter
from typing import Iterable, List, Tuple

import pytest
try:  # pragma: no cover - optional dependency for integration environment
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - skip when neo4j driver missing or incompatible
    GraphDatabase = None  # type: ignore[assignment]

from backend.app.contracts import Evidence, Node, TextSpan
from backend.app.graph import GraphWriter

try:  # pragma: no cover - optional dependency for integration environment
    from testcontainers.neo4j import Neo4jContainer
except ModuleNotFoundError:  # pragma: no cover - skip when testcontainers missing
    Neo4jContainer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for integration environment
    from docker.errors import DockerException
except ModuleNotFoundError:  # pragma: no cover - skip when docker not installed
    DockerException = RuntimeError  # type: ignore[assignment]


@pytest.mark.skipif(
    Neo4jContainer is None or GraphDatabase is None,
    reason="neo4j driver and testcontainers are required for integration tests",
)
def test_graph_writer_persists_entities_and_edges() -> None:
    """Ensure the graph writer can persist entities and edges into Neo4j."""

    container = None
    driver = None

    try:
        container = Neo4jContainer("neo4j:5.21").with_env("NEO4J_AUTH", "neo4j/test")
        container.start()
    except DockerException as exc:  # pragma: no cover - environment without Docker
        pytest.skip(f"Docker is not available: {exc}")
    except Exception as exc:  # pragma: no cover - surface unexpected startup issues
        pytest.skip(f"Unable to start Neo4j test container: {exc}")

    try:
        uri = container.get_connection_url()
        driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))
        writer = GraphWriter(driver=driver, entity_batch_size=100, edge_batch_size=200)

        node_ids = _write_entities(writer, count=60)

        span = TextSpan(start=0, end=50)
        edge_pairs = _generate_edge_pairs(node_ids, desired=1000)

        start = perf_counter()
        for index, (src_id, dst_id) in enumerate(edge_pairs):
            writer.upsert_edge(
                src_id=src_id,
                dst_id=dst_id,
                relation_norm="related-to",
                relation_verbatim="related to",
                evidence=Evidence(
                    element_id=f"edge-{index}",
                    text_span=span,
                    doc_id=f"edge-doc-{index % 10}",
                    full_sentence="Entity relation extracted from integration test.",
                ),
                confidence=0.5,
            )
        writer.flush()
        duration = perf_counter() - start
        assert duration < 10.0

        conflict_evidence = Evidence(
            element_id="edge-conflict",
            text_span=span,
            doc_id="conflict-doc",
            full_sentence="Model A outperforms Model B.",
        )
        writer.upsert_edge(
            src_id=node_ids[0],
            dst_id=node_ids[1],
            relation_norm="outperforms",
            relation_verbatim="outperforms",
            evidence=conflict_evidence,
            confidence=0.8,
            times_seen=2,
        )
        writer.upsert_edge(
            src_id=node_ids[1],
            dst_id=node_ids[0],
            relation_norm="outperforms",
            relation_verbatim="beats",
            evidence=conflict_evidence,
            confidence=0.7,
        )
        writer.flush()

        with driver.session() as session:
            entity_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
            relation_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            conflict_count = (
                session.run("MATCH ()-[r {conflicting: true}]->() RETURN count(r) AS count").single()["count"]
            )

        assert entity_count == len(node_ids)
        assert relation_count == len(edge_pairs) + 2
        assert conflict_count == 2
    finally:
        if driver is not None:
            driver.close()
        if container is not None:
            container.stop()


@pytest.mark.skipif(
    Neo4jContainer is None or GraphDatabase is None,
    reason="neo4j driver and testcontainers are required for integration tests",
)
def test_entity_section_distribution_merges_counts() -> None:
    """Ensure section distributions accumulate when new documents are added."""

    container = None
    driver = None

    try:
        container = Neo4jContainer("neo4j:5.21").with_env("NEO4J_AUTH", "neo4j/test")
        container.start()
    except DockerException as exc:  # pragma: no cover - environment without Docker
        pytest.skip(f"Docker is not available: {exc}")
    except Exception as exc:  # pragma: no cover - surface unexpected startup issues
        pytest.skip(f"Unable to start Neo4j test container: {exc}")

    try:
        uri = container.get_connection_url()
        driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))
        writer = GraphWriter(driver=driver, entity_batch_size=10, edge_batch_size=10)

        node_id = "entity-merge-test"
        writer.upsert_entity(
            Node(
                node_id=node_id,
                name="Transformer Models",
                type="Model",
                aliases=["Transformers"],
                section_distribution={"Methods": 1},
                times_seen=1,
                source_document_ids=["doc-initial"],
            )
        )
        writer.flush()

        writer.upsert_entity(
            Node(
                node_id=node_id,
                name="Transformer Models",
                type="Model",
                aliases=["Transformers"],
                section_distribution={"Methods": 2, "Results": 1},
                times_seen=3,
                source_document_ids=["doc-followup"],
            )
        )
        writer.flush()

        with driver.session() as session:
            record = session.run(
                """
                MATCH (n:Entity {node_id: $node_id})
                RETURN n.section_distribution AS sections,
                       n.section_distribution_keys AS section_keys,
                       n.section_distribution_values AS section_values,
                       n.times_seen AS times_seen
                """,
                node_id=node_id,
            ).single()

        assert record is not None
        sections = decode_section_distribution(
            record["sections"],
            record["section_keys"],
            record["section_values"],
        )
        assert sections == {"Methods": 3, "Results": 1}
        assert record["times_seen"] == 4
    finally:
        if driver is not None:
            driver.close()
        if container is not None:
            container.stop()


@pytest.mark.skipif(
    Neo4jContainer is None or GraphDatabase is None,
    reason="neo4j driver and testcontainers are required for integration tests",
)
def test_entity_update_without_new_documents_preserves_counts() -> None:
    """Ensure repeated uploads of the same document do not inflate counts."""

    container = None
    driver = None

    try:
        container = Neo4jContainer("neo4j:5.21").with_env("NEO4J_AUTH", "neo4j/test")
        container.start()
    except DockerException as exc:  # pragma: no cover - environment without Docker
        pytest.skip(f"Docker is not available: {exc}")
    except Exception as exc:  # pragma: no cover - surface unexpected startup issues
        pytest.skip(f"Unable to start Neo4j test container: {exc}")

    try:
        uri = container.get_connection_url()
        driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))
        writer = GraphWriter(driver=driver, entity_batch_size=10, edge_batch_size=10)

        node_id = "entity-repeat-test"
        original = Node(
            node_id=node_id,
            name="Self-Supervised Pipeline",
            type="Method",
            aliases=["SSP"],
            section_distribution={"Methods": 2},
            times_seen=2,
            source_document_ids=["doc-static"],
        )
        writer.upsert_entity(original)
        writer.flush()

        duplicate = Node(
            node_id=node_id,
            name="Self-Supervised Pipeline",
            type="Method",
            aliases=["SSP", "Self-Supervised Pipeline"],
            section_distribution={"Methods": 5, "Results": 3},
            times_seen=10,
            source_document_ids=["doc-static"],
        )
        writer.upsert_entity(duplicate)
        writer.flush()

        with driver.session() as session:
            record = session.run(
                """
                MATCH (n:Entity {node_id: $node_id})
                RETURN n.section_distribution AS sections,
                       n.section_distribution_keys AS section_keys,
                       n.section_distribution_values AS section_values,
                       n.times_seen AS times_seen,
                       n.source_document_ids AS docs
                """,
                node_id=node_id,
            ).single()

        assert record is not None
        sections = decode_section_distribution(
            record["sections"],
            record["section_keys"],
            record["section_values"],
        )
        assert sections == {"Methods": 2}
        assert record["times_seen"] == 2
        assert record["docs"] == ["doc-static"]
    finally:
        if driver is not None:
            driver.close()
        if container is not None:
            container.stop()


def _write_entities(writer: GraphWriter, count: int) -> List[str]:
    """Seed Neo4j with a collection of canonical nodes."""

    node_ids: List[str] = []
    for index in range(count):
        node_id = f"node-{index}"
        node_ids.append(node_id)
        writer.upsert_entity(
            Node(
                node_id=node_id,
                name=f"Entity {index}",
                type="Concept",
                aliases=[f"E{index}"],
                section_distribution={"Methods": 1},
                times_seen=1,
                source_document_ids=[f"doc-{index}"],
            )
        )
    writer.flush()
    return node_ids


def _generate_edge_pairs(node_ids: Iterable[str], desired: int) -> List[Tuple[str, str]]:
    """Generate a list of unique directed edge pairs for testing."""

    pairs: List[Tuple[str, str]] = []
    node_list = list(node_ids)
    for src_index, src_id in enumerate(node_list):
        for dst_index, dst_id in enumerate(node_list):
            if src_index == dst_index:
                continue
            pairs.append((src_id, dst_id))
            if len(pairs) >= desired:
                return pairs
    return pairs
