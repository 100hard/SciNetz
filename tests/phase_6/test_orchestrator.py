"""Tests for the Phase 6 extraction orchestrator."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pytest

from backend.app.canonicalization import (
    CanonicalizationPipeline,
    EntityCanonicalizer,
    HashingEmbeddingBackend,
)
from backend.app.config import AppConfig, load_config
from backend.app.contracts import Evidence, PaperMetadata, ParsedElement, TextSpan, Triplet
from backend.app.extraction import ExtractionResult
from backend.app.orchestration import ExtractionOrchestrator
from backend.app.orchestration.orchestrator import OrchestrationResult, ProcessedChunkStore
from backend.app.parsing.pipeline import ParseResult


class StubParsingPipeline:
    """Parsing pipeline that returns a predefined result."""

    def __init__(self, result: ParseResult) -> None:
        self.result = result
        self.calls: int = 0

    def parse_document(self, doc_id: str, pdf_path: Path) -> ParseResult:
        self.calls += 1
        return self.result


class StubTripletExtractor:
    """Extractor that returns pre-seeded extraction results."""

    def __init__(
        self,
        outputs: Dict[str, ExtractionResult],
        failures: Optional[Iterable[str]] = None,
    ) -> None:
        self._outputs = outputs
        self._failures = set(failures or [])
        self.calls: List[str] = []

    def extract_with_metadata(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Iterable[str]],
    ) -> ExtractionResult:
        self.calls.append(element.element_id)
        if element.element_id in self._failures:
            raise RuntimeError(f"LLM failure for {element.element_id}")
        return self._outputs[element.element_id]


class StubInventoryBuilder:
    """Inventory builder that returns scripted entity lists."""

    def __init__(self, outputs: Dict[str, List[str]]) -> None:
        self._outputs = outputs
        self.calls: List[str] = []

    def build_inventory(self, element: ParsedElement) -> List[str]:
        self.calls.append(element.element_id)
        return list(self._outputs.get(element.element_id, []))


class StubGraphWriter:
    """Graph writer that records nodes and edges in-memory."""

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def upsert_entity(self, node) -> str:  # pragma: no cover - simple data container
        self.nodes.append(node)
        return node.node_id

    def upsert_edge(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation_norm: str,
        relation_verbatim: str,
        evidence: Evidence,
        confidence: float,
        attributes: Optional[Dict[str, str]] = None,
        created_at=None,
        times_seen: int = 1,
    ) -> None:
        self.edges.append(
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "relation_norm": relation_norm,
                "relation_verbatim": relation_verbatim,
                "evidence": evidence,
                "confidence": confidence,
                "attributes": attributes or {},
                "times_seen": times_seen,
            }
        )

    def flush(self) -> None:  # pragma: no cover - interface compatibility
        return None


@pytest.fixture(name="config")
def fixture_config() -> AppConfig:
    return load_config()


def _parsed_element(doc_id: str, element_id: str, section: str, content: str) -> ParsedElement:
    content_hash = hashlib.sha256(element_id.encode("utf-8")).hexdigest()
    return ParsedElement(
        doc_id=doc_id,
        element_id=element_id,
        section=section,
        content=content,
        content_hash=content_hash,
        start_char=0,
        end_char=len(content),
    )


def _triplet(subject: str, predicate: str, obj: str, element: ParsedElement) -> Triplet:
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
        confidence=0.9,
        evidence=evidence,
        pipeline_version="1.0.0",
    )


def test_orchestrator_runs_end_to_end(config: AppConfig, tmp_path: Path) -> None:
    doc_id = "paper-1"
    element_a = _parsed_element(doc_id, f"{doc_id}:0", "Methods", "Model A uses data set X.")
    element_b = _parsed_element(doc_id, f"{doc_id}:1", "Results", "Model A outperforms Model B.")
    metadata = PaperMetadata(doc_id=doc_id, title="Test Paper")
    parse_result = ParseResult(doc_id=doc_id, metadata=metadata, elements=[element_a, element_b], errors=[])
    parsing = StubParsingPipeline(parse_result)

    triplet_a = _triplet("Model A", "uses", "data set X", element_a)
    triplet_b = _triplet("Model A", "outperforms", "Model B", element_b)
    extraction_results = {
        element_a.element_id: ExtractionResult(
            triplets=[triplet_a],
            section_distribution={"Model A": {"Methods": 1}, "data set X": {"Methods": 1}},
            relation_verbatims=["uses"],
        ),
        element_b.element_id: ExtractionResult(
            triplets=[triplet_b],
            section_distribution={"Model A": {"Results": 1}, "Model B": {"Results": 1}},
            relation_verbatims=["outperforms"],
        ),
    }
    extractor = StubTripletExtractor(extraction_results)
    inventory = StubInventoryBuilder({})
    graph_writer = StubGraphWriter()

    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=HashingEmbeddingBackend(),
        embedding_dir=tmp_path / "embeddings",
        report_dir=tmp_path / "reports",
    )
    canonicalization = CanonicalizationPipeline(config=config, canonicalizer=canonicalizer)
    chunk_store = ProcessedChunkStore(tmp_path / "processed.json")
    orchestrator = ExtractionOrchestrator(
        config=config,
        parsing_pipeline=parsing,
        inventory_builder=inventory,
        triplet_extractor=extractor,
        canonicalization=canonicalization,
        graph_writer=graph_writer,
        chunk_store=chunk_store,
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_text("placeholder")

    result = orchestrator.run(paper_id=doc_id, pdf_path=pdf_path)

    assert isinstance(result, OrchestrationResult)
    assert result.processed_chunks == 2
    assert result.skipped_chunks == 0
    assert result.errors == []
    assert len(graph_writer.nodes) == 3  # Model A, data set X, Model B
    assert len(graph_writer.edges) == 2
    assert chunk_store.should_process(doc_id, element_a.content_hash, config.pipeline.version) is False


def test_orchestrator_is_idempotent(config: AppConfig, tmp_path: Path) -> None:
    doc_id = "paper-2"
    element = _parsed_element(doc_id, f"{doc_id}:0", "Results", "System Y surpasses baseline.")
    metadata = PaperMetadata(doc_id=doc_id)
    parse_result = ParseResult(doc_id=doc_id, metadata=metadata, elements=[element], errors=[])
    parsing = StubParsingPipeline(parse_result)

    triplet = _triplet("System Y", "outperforms", "baseline", element)
    extraction = ExtractionResult(
        triplets=[triplet],
        section_distribution={"System Y": {"Results": 1}, "baseline": {"Results": 1}},
        relation_verbatims=["outperforms"],
    )
    extractor = StubTripletExtractor({element.element_id: extraction})
    inventory = StubInventoryBuilder({})
    graph_writer = StubGraphWriter()
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=HashingEmbeddingBackend(),
        embedding_dir=tmp_path / "embeddings",
        report_dir=tmp_path / "reports",
    )
    canonicalization = CanonicalizationPipeline(config=config, canonicalizer=canonicalizer)
    chunk_store = ProcessedChunkStore(tmp_path / "processed.json")
    orchestrator = ExtractionOrchestrator(
        config=config,
        parsing_pipeline=parsing,
        inventory_builder=inventory,
        triplet_extractor=extractor,
        canonicalization=canonicalization,
        graph_writer=graph_writer,
        chunk_store=chunk_store,
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_text("placeholder")

    first = orchestrator.run(paper_id=doc_id, pdf_path=pdf_path)
    second = orchestrator.run(paper_id=doc_id, pdf_path=pdf_path)

    assert first.processed_chunks == 1
    assert second.processed_chunks == 0
    assert second.skipped_chunks == 1
    assert extractor.calls == [element.element_id]
    assert len(graph_writer.edges) == 1

def test_orchestrator_force_reprocesses_when_requested(config: AppConfig, tmp_path: Path) -> None:
    doc_id = "paper-2"
    element = _parsed_element(doc_id, f"{doc_id}:0", "Results", "System Y surpasses baseline.")
    metadata = PaperMetadata(doc_id=doc_id)
    parse_result = ParseResult(doc_id=doc_id, metadata=metadata, elements=[element], errors=[])
    parsing = StubParsingPipeline(parse_result)

    triplet = _triplet("System Y", "outperforms", "baseline", element)
    extraction = ExtractionResult(
        triplets=[triplet],
        section_distribution={"System Y": {"Results": 1}, "baseline": {"Results": 1}},
        relation_verbatims=["outperforms"],
    )
    extractor = StubTripletExtractor({element.element_id: extraction})
    inventory = StubInventoryBuilder({})
    graph_writer = StubGraphWriter()
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=HashingEmbeddingBackend(),
        embedding_dir=tmp_path / "embeddings",
        report_dir=tmp_path / "reports",
    )
    canonicalization = CanonicalizationPipeline(config=config, canonicalizer=canonicalizer)
    chunk_store = ProcessedChunkStore(tmp_path / "processed.json")
    orchestrator = ExtractionOrchestrator(
        config=config,
        parsing_pipeline=parsing,
        inventory_builder=inventory,
        triplet_extractor=extractor,
        canonicalization=canonicalization,
        graph_writer=graph_writer,
        chunk_store=chunk_store,
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_text("placeholder")

    orchestrator.run(paper_id=doc_id, pdf_path=pdf_path)
    rerun = orchestrator.run(paper_id=doc_id, pdf_path=pdf_path, force=True)

    assert rerun.processed_chunks == 1
    assert extractor.calls == [element.element_id, element.element_id]
    assert len(graph_writer.edges) == 2
