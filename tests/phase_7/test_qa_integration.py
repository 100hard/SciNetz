from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Sequence

import pytest

from backend.app.canonicalization.entity_canonicalizer import HashingEmbeddingBackend
from backend.app.config import load_config
from backend.app.qa import AnswerMode, QAService
from backend.app.qa.repository import Neo4jQARepository

try:  # pragma: no cover - optional dependencies for integration
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - skip when driver unavailable
    GraphDatabase = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependencies for integration
    from testcontainers.neo4j import Neo4jContainer
except ModuleNotFoundError:  # pragma: no cover - skip when testcontainers missing
    Neo4jContainer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for integration
    from docker.errors import DockerException
except ModuleNotFoundError:  # pragma: no cover - skip when docker missing
    DockerException = RuntimeError  # type: ignore[assignment]


class StubExtractor:
    """Deterministic extractor returning known entity names."""

    def __init__(self, known_entities: Sequence[str]) -> None:
        self._known = list(known_entities)

    def extract(self, question: str) -> List[str]:
        lower = question.lower()
        return [name for name in self._known if name.lower() in lower]


@pytest.mark.skipif(
    Neo4jContainer is None or GraphDatabase is None,
    reason="neo4j driver and testcontainers are required for integration tests",
)
def test_qa_service_answers_and_provides_context() -> None:
    container = None
    driver = None

    try:
        container = Neo4jContainer("neo4j:5.21").with_env("NEO4J_AUTH", "neo4j/test")
        container.start()
    except DockerException as exc:  # pragma: no cover - docker not available
        pytest.skip(f"Docker is not available: {exc}")
    except Exception as exc:  # pragma: no cover - startup failure
        pytest.skip(f"Unable to start Neo4j test container: {exc}")

    try:
        uri = container.get_connection_url()
        driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))
        _seed_graph(driver)

        config = load_config()
        repository = Neo4jQARepository(driver, config=config)
        extractor = StubExtractor(
            ["Model Alpha", "Model Beta", "Dataset Delta", "Benchmark Gamma", "Model Zeta"]
        )
        qa_service = QAService(
            config=config,
            repository=repository,
            embedding_backend=HashingEmbeddingBackend(),
            extractor=extractor,  # type: ignore[arg-type]
        )

        direct = qa_service.answer("Does Model Alpha outperform Model Beta?")
        assert direct.mode == AnswerMode.DIRECT
        assert direct.paths
        assert any(edge.evidence.doc_id for edge in direct.paths[0].edges)

        multi_hop = qa_service.answer("How is Model Alpha related to Benchmark Gamma?")
        assert multi_hop.mode == AnswerMode.DIRECT
        assert multi_hop.paths
        assert len(multi_hop.paths[0].edges) == 2

        insufficient = qa_service.answer("Does Model Beta cause Benchmark Gamma?")
        assert insufficient.mode == AnswerMode.INSUFFICIENT
        assert not insufficient.paths
        assert insufficient.fallback_edges
    finally:
        if driver is not None:
            driver.close()
        if container is not None:
            container.stop()


def _seed_graph(driver) -> None:
    now = datetime.now(timezone.utc)
    config = load_config()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(
            """
            CREATE (alpha:Entity {
                node_id: 'alpha',
                name: 'Model Alpha',
                aliases: ['Alpha Model'],
                times_seen: 5,
                section_distribution: {Results: 3, Methods: 2}
            })
            CREATE (beta:Entity {
                node_id: 'beta',
                name: 'Model Beta',
                aliases: ['Model B'],
                times_seen: 4,
                section_distribution: {Results: 2}
            })
            CREATE (delta:Entity {
                node_id: 'delta',
                name: 'Dataset Delta',
                aliases: ['Delta Data'],
                times_seen: 3,
                section_distribution: {Methods: 2}
            })
            CREATE (gamma:Entity {
                node_id: 'gamma',
                name: 'Benchmark Gamma',
                aliases: ['Gamma Benchmark'],
                times_seen: 2,
                section_distribution: {Results: 1}
            })
            CREATE (zeta:Entity {
                node_id: 'zeta',
                name: 'Model Zeta',
                aliases: [],
                times_seen: 1,
                section_distribution: {Discussion: 1}
            })
            """
        )
        session.run(
            """
            MATCH (alpha:Entity {node_id: 'alpha'}), (beta:Entity {node_id: 'beta'})
            CREATE (alpha)-[:RELATION {
                relation_norm: 'outperforms',
                relation_verbatim: 'outperforms',
                confidence: 0.92,
                pipeline_version: $version,
                created_at: $now,
                times_seen: 2,
                conflicting: false,
                evidence: {
                    doc_id: 'doc-alpha-beta',
                    element_id: 'alpha:0',
                    text_span: {start: 0, end: 42},
                    full_sentence: 'Model Alpha outperforms Model Beta.'
                },
                attributes: {method: 'llm'}
            }]->(beta)
            """,
            version=config.pipeline.version,
            now=now,
        )
        session.run(
            """
            MATCH (alpha:Entity {node_id: 'alpha'}), (delta:Entity {node_id: 'delta'})
            CREATE (alpha)-[:RELATION {
                relation_norm: 'uses',
                relation_verbatim: 'uses',
                confidence: 0.88,
                pipeline_version: $version,
                created_at: $now,
                times_seen: 1,
                conflicting: false,
                evidence: {
                    doc_id: 'doc-alpha-delta',
                    element_id: 'alpha:1',
                    text_span: {start: 0, end: 35},
                    full_sentence: 'Model Alpha uses Dataset Delta.'
                },
                attributes: {method: 'llm'}
            }]->(delta)
            """,
            version=config.pipeline.version,
            now=now,
        )
        session.run(
            """
            MATCH (delta:Entity {node_id: 'delta'}), (gamma:Entity {node_id: 'gamma'})
            CREATE (delta)-[:RELATION {
                relation_norm: 'evaluated-on',
                relation_verbatim: 'evaluated on',
                confidence: 0.86,
                pipeline_version: $version,
                created_at: $now,
                times_seen: 1,
                conflicting: false,
                evidence: {
                    doc_id: 'doc-delta-gamma',
                    element_id: 'delta:2',
                    text_span: {start: 0, end: 44},
                    full_sentence: 'Dataset Delta is evaluated on Benchmark Gamma.'
                },
                attributes: {method: 'llm'}
            }]->(gamma)
            """,
            version=config.pipeline.version,
            now=now,
        )
        session.run(
            """
            MATCH (beta:Entity {node_id: 'beta'}), (zeta:Entity {node_id: 'zeta'})
            CREATE (beta)-[:RELATION {
                relation_norm: 'compared-to',
                relation_verbatim: 'compared to',
                confidence: 0.75,
                pipeline_version: $version,
                created_at: $now,
                times_seen: 1,
                conflicting: false,
                evidence: {
                    doc_id: 'doc-beta-zeta',
                    element_id: 'beta:3',
                    text_span: {start: 0, end: 39},
                    full_sentence: 'Model Beta is compared to Model Zeta.'
                },
                attributes: {method: 'llm'}
            }]->(zeta)
            """,
            version=config.pipeline.version,
            now=now,
        )
