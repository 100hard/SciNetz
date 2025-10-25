from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

try:  # pragma: no cover - optional dependency for integration environment
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - skip when neo4j driver unavailable
    GraphDatabase = None  # type: ignore[assignment]

from backend.app.canonicalization import (
    CanonicalizationPipeline,
    EntityCanonicalizer,
    HashingEmbeddingBackend,
)
from backend.app.config import load_config
from backend.app.contracts import Evidence, PaperMetadata, ParsedElement, TextSpan, Triplet
from backend.app.extraction import ExtractionResult
from backend.app.graph import GraphWriter
from backend.app.main import create_app
from backend.app.orchestration import ExtractionOrchestrator
from backend.app.orchestration.orchestrator import ProcessedChunkStore
from backend.app.parsing.pipeline import ParseResult

try:  # pragma: no cover - optional dependency for integration environment
    from testcontainers.neo4j import Neo4jContainer
except ModuleNotFoundError:  # pragma: no cover - skip when testcontainers missing
    Neo4jContainer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for integration environment
    from docker.errors import DockerException
except ModuleNotFoundError:  # pragma: no cover - skip when docker not installed
    DockerException = RuntimeError  # type: ignore[assignment]


class _StaticParsingPipeline:
    """Return pre-seeded parse results keyed by document identifier."""

    def __init__(self, results: Dict[str, ParseResult]) -> None:
        self._results = results

    def parse_document(self, doc_id: str, pdf_path: Path) -> ParseResult:
        return self._results[doc_id]


class _StaticTripletExtractor:
    """Triplet extractor that returns canned extraction results."""

    def __init__(self, outputs: Dict[str, ExtractionResult]) -> None:
        self._outputs = outputs
        self.calls: List[str] = []

    def extract_with_metadata(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Iterable[str]],
        *,
        metadata: Optional[PaperMetadata] = None,
        domain: Optional[object] = None,
    ) -> ExtractionResult:
        self.calls.append(element.element_id)
        return self._outputs[element.element_id]


class _StaticInventoryBuilder:
    """Inventory builder that returns predetermined entity lists."""

    def __init__(self, outputs: Dict[str, List[str]]) -> None:
        self._outputs = outputs

    def build_inventory(self, element: ParsedElement, domain: Optional[str] = None) -> List[str]:
        return list(self._outputs.get(element.element_id, []))


def _element(doc_id: str, element_id: str, section: str, content: str) -> ParsedElement:
    content_hash = hashlib.sha256(f"{doc_id}:{element_id}:{content}".encode("utf-8")).hexdigest()
    return ParsedElement(
        doc_id=doc_id,
        element_id=element_id,
        section=section,
        content=content,
        content_hash=content_hash,
        start_char=0,
        end_char=len(content),
    )


def _triplet(subject: str, predicate: str, obj: str, element: ParsedElement, version: str) -> Triplet:
    evidence = Evidence(
        element_id=element.element_id,
        doc_id=element.doc_id,
        text_span=TextSpan(start=0, end=len(element.content)),
        full_sentence=element.content,
    )
    return Triplet(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=0.85,
        evidence=evidence,
        pipeline_version=version,
    )


@pytest.mark.skipif(
    Neo4jContainer is None or GraphDatabase is None,
    reason="neo4j driver and testcontainers are required for integration tests",
)
def test_extraction_endpoint_runs_pipeline_end_to_end(tmp_path: Path) -> None:
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

        base_config = load_config()
        graph_config = base_config.graph.model_copy(
            update={"uri": uri, "username": "neo4j", "password": "test"}
        )
        root_dir = Path(__file__).resolve().parents[2]
        upload_rel = os.path.relpath(tmp_path / "uploads", root_dir)
        registry_rel = os.path.relpath(tmp_path / "registry.json", root_dir)
        ui_config = base_config.ui.model_copy(
            update={"upload_dir": upload_rel, "paper_registry_path": registry_rel}
        )
        config = base_config.model_copy(update={"graph": graph_config, "ui": ui_config})

        doc_a = "paper-alpha"
        doc_b = "paper-beta"
        element_a = _element(doc_a, f"{doc_a}:0", "Methods", "Model X uses dataset Y.")
        element_b = _element(doc_b, f"{doc_b}:0", "Results", "Model X outperforms baseline.")
        metadata_a = PaperMetadata(doc_id=doc_a, title="Alpha Paper")
        metadata_b = PaperMetadata(doc_id=doc_b, title="Beta Paper")
        parse_a = ParseResult(doc_id=doc_a, metadata=metadata_a, elements=[element_a], errors=[])
        parse_b = ParseResult(doc_id=doc_b, metadata=metadata_b, elements=[element_b], errors=[])

        parsing = _StaticParsingPipeline({doc_a: parse_a, doc_b: parse_b})
        triplet_a = _triplet("Model X", "uses", "dataset Y", element_a, config.pipeline.version)
        triplet_b = _triplet("Model X", "outperforms", "baseline", element_b, config.pipeline.version)
        extractor = _StaticTripletExtractor(
            {
                element_a.element_id: ExtractionResult(
                    triplets=[triplet_a],
                    section_distribution={
                        "Model X": {"Methods": 1},
                        "dataset Y": {"Methods": 1},
                    },
                    relation_verbatims=["uses"],
                ),
                element_b.element_id: ExtractionResult(
                    triplets=[triplet_b],
                    section_distribution={
                        "Model X": {"Results": 1},
                        "baseline": {"Results": 1},
                    },
                    relation_verbatims=["outperforms"],
                ),
            }
        )
        inventory = _StaticInventoryBuilder({})
        canonicalizer = EntityCanonicalizer(
            config,
            embedding_backend=HashingEmbeddingBackend(),
            embedding_dir=tmp_path / "embeddings",
            report_dir=tmp_path / "reports",
        )
        canonicalization = CanonicalizationPipeline(config=config, canonicalizer=canonicalizer)
        chunk_store = ProcessedChunkStore(tmp_path / "processed.json")
        graph_writer = GraphWriter(
            driver=driver,
            config=config,
            entity_batch_size=config.graph.entity_batch_size,
            edge_batch_size=config.graph.edge_batch_size,
        )
        orchestrator = ExtractionOrchestrator(
            config=config,
            parsing_pipeline=parsing,
            inventory_builder=inventory,
            triplet_extractor=extractor,
            canonicalization=canonicalization,
            graph_writer=graph_writer,
            chunk_store=chunk_store,
        )

        pdf_a = tmp_path / "paper_a.pdf"
        pdf_b = tmp_path / "paper_b.pdf"
        pdf_a.write_text("dummy")
        pdf_b.write_text("dummy")

        with patch("backend.app.main.GoogleTokenVerifier") as verifier_ctor:
            verifier_ctor.return_value = object()
            app = create_app(config=config, orchestrator=orchestrator)

        with TestClient(app) as client:
            response_a = client.post(
                f"/api/extract/{doc_a}", json={"pdf_path": str(pdf_a.resolve())}
            )
            response_b = client.post(
                f"/api/extract/{doc_b}", json={"pdf_path": str(pdf_b.resolve())}
            )
            rerun_a = client.post(
                f"/api/extract/{doc_a}", json={"pdf_path": str(pdf_a.resolve())}
            )

        assert response_a.status_code == 200
        assert response_b.status_code == 200
        assert rerun_a.status_code == 200

        payload_a = response_a.json()
        payload_b = response_b.json()
        payload_rerun = rerun_a.json()

        assert payload_a["processed_chunks"] == 1
        assert payload_b["processed_chunks"] == 1
        assert payload_rerun["processed_chunks"] == 0
        assert payload_a["edges_written"] >= 1
        assert payload_b["edges_written"] >= 1
        assert payload_rerun["skipped_chunks"] >= 1

        with driver.session() as session:
            node_count = session.run(
                "MATCH (n:Entity) RETURN count(n) AS count"
            ).single()["count"]
            edge_count = session.run(
                "MATCH ()-[r:RELATION]->() RETURN count(r) AS count"
            ).single()["count"]
            missing_evidence = session.run(
                "MATCH ()-[r:RELATION]->() WHERE r.evidence IS NULL RETURN count(r) AS count"
            ).single()["count"]

        assert node_count > 0
        assert edge_count > 0
        assert missing_evidence == 0

    finally:
        if driver is not None:
            driver.close()
        if container is not None:
            container.stop()
