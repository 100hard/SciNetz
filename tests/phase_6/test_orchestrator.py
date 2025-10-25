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
from backend.app.orchestration.orchestrator import (
    BatchExtractionInput,
    BatchOrchestrationResult,
    OrchestrationResult,
    ProcessedChunkStore,
)
from backend.app.parsing.pipeline import ParseResult


class StubParsingPipeline:
    """Parsing pipeline that returns a predefined result."""

    def __init__(self, result: ParseResult) -> None:
        self.result = result
        self.calls: int = 0

    def parse_document(self, doc_id: str, pdf_path: Path) -> ParseResult:
        self.calls += 1
        return self.result


class MappingParsingPipeline:
    """Parsing pipeline that returns results from a mapping keyed by doc id."""

    def __init__(self, results: Dict[str, ParseResult]) -> None:
        self._results = results
        self.calls: List[str] = []

    def parse_document(self, doc_id: str, pdf_path: Path) -> ParseResult:
        self.calls.append(doc_id)
        return self._results[doc_id]


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
        self.metadata_calls: Dict[str, Optional[PaperMetadata]] = {}
        self.domain_calls: Dict[str, Optional[object]] = {}

    def extract_with_metadata(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Iterable[str]],
        *,
        metadata: Optional[PaperMetadata] = None,
        domain: Optional[object] = None,
    ) -> ExtractionResult:
        self.calls.append(element.element_id)
        self.metadata_calls[element.element_id] = metadata
        self.domain_calls[element.element_id] = domain
        if element.element_id in self._failures:
            raise RuntimeError(f"LLM failure for {element.element_id}")
        return self._outputs[element.element_id]


class StubInventoryBuilder:
    """Inventory builder that returns scripted entity lists."""

    def __init__(self, outputs: Dict[str, List[str]]) -> None:
        self._outputs = outputs
        self.calls: List[str] = []
        self.domains: Dict[str, Optional[str]] = {}

    def build_inventory(self, element: ParsedElement, domain: Optional[str] = None) -> List[str]:
        self.calls.append(element.element_id)
        self.domains[element.element_id] = domain
        return list(self._outputs.get(element.element_id, []))


class StubGraphWriter:
    """Graph writer that records nodes and edges in-memory."""

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.deprecation_calls: List[str] = []

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

    def deprecate_edges(self, doc_id: str) -> int:  # pragma: no cover - simple counter
        self.deprecation_calls.append(doc_id)
        return 1


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


def _triplet(
    subject: str,
    predicate: str,
    obj: str,
    element: ParsedElement,
    *,
    pipeline_version: str = "1.0.0",
) -> Triplet:
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
        pipeline_version=pipeline_version,
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


def test_orchestrator_routes_biology_domain(config: AppConfig, tmp_path: Path) -> None:
    config = config.model_copy(
        update={
            "extraction": config.extraction.model_copy(update={"use_entity_inventory": True})
        }
    )
    doc_id = "bio-paper"
    element = _parsed_element(
        doc_id,
        f"{doc_id}:0",
        "Results",
        "Cas9 nuclease binds to the guide RNA scaffold in human cells.",
    )
    metadata = PaperMetadata(
        doc_id=doc_id,
        title="Programmable RNA editing with CRISPR-Cas9",
        venue="Cell",
    )
    parse_result = ParseResult(doc_id=doc_id, metadata=metadata, elements=[element], errors=[])
    parsing = StubParsingPipeline(parse_result)

    triplet = _triplet("Cas9 nuclease", "binds-to", "the guide RNA scaffold", element)
    extraction = ExtractionResult(
        triplets=[triplet],
        section_distribution={
            "Cas9 nuclease": {"Results": 1},
            "the guide RNA scaffold": {"Results": 1},
        },
        relation_verbatims=["binds to"],
    )
    extractor = StubTripletExtractor({element.element_id: extraction})
    inventory = StubInventoryBuilder({element.element_id: ["Cas9 nuclease", "guide RNA scaffold"]})
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

    pdf_path = tmp_path / "bio-paper.pdf"
    pdf_path.write_text("placeholder")

    result = orchestrator.run(paper_id=doc_id, pdf_path=pdf_path)

    assert result.processed_chunks == 1
    assert inventory.domains[element.element_id] == "biology"
    assert extractor.metadata_calls[element.element_id] == metadata
    assert len(graph_writer.edges) == 1
    assert graph_writer.edges[0]["relation_norm"] == "binds-to"


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


def test_pipeline_version_upgrade_deprecates_previous_edges(
    config: AppConfig, tmp_path: Path
) -> None:
    doc_id = "paper-version"
    element = _parsed_element(doc_id, f"{doc_id}:0", "Discussion", "Model R relates to Model S.")
    metadata = PaperMetadata(doc_id=doc_id)
    parse_result = ParseResult(doc_id=doc_id, metadata=metadata, elements=[element], errors=[])
    parsing_v1 = StubParsingPipeline(parse_result)

    triplet_v1 = _triplet(
        "Model R",
        "related-to",
        "Model S",
        element,
        pipeline_version=config.pipeline.version,
    )
    extraction_v1 = ExtractionResult(
        triplets=[triplet_v1],
        section_distribution={"Model R": {"Discussion": 1}, "Model S": {"Discussion": 1}},
        relation_verbatims=["related to"],
    )
    extractor_v1 = StubTripletExtractor({element.element_id: extraction_v1})
    inventory = StubInventoryBuilder({})
    graph_writer_v1 = StubGraphWriter()
    canonicalizer_v1 = EntityCanonicalizer(
        config,
        embedding_backend=HashingEmbeddingBackend(),
        embedding_dir=tmp_path / "embeddings_v1",
        report_dir=tmp_path / "reports_v1",
    )
    canonicalization_v1 = CanonicalizationPipeline(config=config, canonicalizer=canonicalizer_v1)
    store_path = tmp_path / "processed.json"
    chunk_store_v1 = ProcessedChunkStore(store_path)
    orchestrator_v1 = ExtractionOrchestrator(
        config=config,
        parsing_pipeline=parsing_v1,
        inventory_builder=inventory,
        triplet_extractor=extractor_v1,
        canonicalization=canonicalization_v1,
        graph_writer=graph_writer_v1,
        chunk_store=chunk_store_v1,
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_text("placeholder")

    orchestrator_v1.run(paper_id=doc_id, pdf_path=pdf_path)

    upgraded_pipeline = config.pipeline.model_copy(update={"version": "1.1.0"})
    upgraded_config = config.model_copy(update={"pipeline": upgraded_pipeline})
    parsing_v2 = StubParsingPipeline(parse_result)
    triplet_v2 = _triplet(
        "Model R",
        "related-to",
        "Model S",
        element,
        pipeline_version="1.1.0",
    )
    extraction_v2 = ExtractionResult(
        triplets=[triplet_v2],
        section_distribution={"Model R": {"Discussion": 1}, "Model S": {"Discussion": 1}},
        relation_verbatims=["related to"],
    )
    extractor_v2 = StubTripletExtractor({element.element_id: extraction_v2})
    graph_writer_v2 = StubGraphWriter()
    canonicalizer_v2 = EntityCanonicalizer(
        upgraded_config,
        embedding_backend=HashingEmbeddingBackend(),
        embedding_dir=tmp_path / "embeddings_v2",
        report_dir=tmp_path / "reports_v2",
    )
    canonicalization_v2 = CanonicalizationPipeline(
        config=upgraded_config, canonicalizer=canonicalizer_v2
    )
    chunk_store_v2 = ProcessedChunkStore(store_path)
    orchestrator_v2 = ExtractionOrchestrator(
        config=upgraded_config,
        parsing_pipeline=parsing_v2,
        inventory_builder=inventory,
        triplet_extractor=extractor_v2,
        canonicalization=canonicalization_v2,
        graph_writer=graph_writer_v2,
        chunk_store=chunk_store_v2,
    )

    rerun = orchestrator_v2.run(paper_id=doc_id, pdf_path=pdf_path)

    assert rerun.processed_chunks == 1
    assert graph_writer_v2.deprecation_calls == [doc_id]
    assert (
        chunk_store_v2.should_process(doc_id, element.content_hash, upgraded_config.pipeline.version)
        is False
    )


def test_run_batch_merges_cross_papers(config: AppConfig, tmp_path: Path) -> None:
    doc_a = "paper-a"
    doc_b = "paper-b"
    element_a = _parsed_element(doc_a, f"{doc_a}:0", "Methods", "Model A uses dataset X.")
    element_b = _parsed_element(doc_b, f"{doc_b}:0", "Evaluation", "Model A evaluates dataset Y.")
    metadata_a = PaperMetadata(doc_id=doc_a, title="Paper A")
    metadata_b = PaperMetadata(doc_id=doc_b, title="Paper B")
    parse_result_a = ParseResult(doc_id=doc_a, metadata=metadata_a, elements=[element_a], errors=[])
    parse_result_b = ParseResult(doc_id=doc_b, metadata=metadata_b, elements=[element_b], errors=[])
    parsing = MappingParsingPipeline({doc_a: parse_result_a, doc_b: parse_result_b})

    triplet_a = _triplet("Model A", "uses", "dataset X", element_a)
    triplet_b = _triplet("Model A", "evaluates", "dataset Y", element_b)
    extraction_a = ExtractionResult(
        triplets=[triplet_a],
        section_distribution={"Model A": {"Methods": 1}, "dataset X": {"Methods": 1}},
        relation_verbatims=["uses"],
    )
    extraction_b = ExtractionResult(
        triplets=[triplet_b],
        section_distribution={"Model A": {"Evaluation": 1}, "dataset Y": {"Evaluation": 1}},
        relation_verbatims=["evaluates"],
    )
    extractor = StubTripletExtractor(
        {
            element_a.element_id: extraction_a,
            element_b.element_id: extraction_b,
        }
    )
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

    pdf_a = tmp_path / "paper_a.pdf"
    pdf_b = tmp_path / "paper_b.pdf"
    pdf_a.write_text("placeholder a")
    pdf_b.write_text("placeholder b")

    batch_result = orchestrator.run_batch(
        [
            BatchExtractionInput(paper_id=doc_a, pdf_path=pdf_a),
            BatchExtractionInput(paper_id=doc_b, pdf_path=pdf_b),
        ]
    )

    assert isinstance(batch_result, BatchOrchestrationResult)
    assert batch_result.total_processed_chunks == 2
    assert batch_result.total_edges_written == 2
    assert batch_result.total_nodes_written == 3
    assert set(batch_result.documents.keys()) == {doc_a, doc_b}

    per_doc_a = batch_result.documents[doc_a]
    per_doc_b = batch_result.documents[doc_b]

    assert per_doc_a.processed_chunks == 1
    assert per_doc_b.processed_chunks == 1
    assert per_doc_a.nodes_written == 2
    assert per_doc_b.nodes_written == 2
    assert per_doc_a.edges_written == 1
    assert per_doc_b.edges_written == 1

    assert len(graph_writer.nodes) == 3
    model_nodes = [node for node in graph_writer.nodes if node.name == "Model A"]
    assert len(model_nodes) == 1
    assert set(model_nodes[0].source_document_ids) == {doc_a, doc_b}


def test_co_mention_fallback_creates_hidden_edges(
    config: AppConfig, tmp_path: Path
) -> None:
    doc_id = "paper-co"
    content = (
        "Model Z collaborates with Model Q in experiments. "
        "Furthermore, Model Q and Model Z appear together in discussions."
    )
    element = _parsed_element(doc_id, f"{doc_id}:0", "Discussion", content)
    metadata = PaperMetadata(doc_id=doc_id, title="Fallback Paper")
    parse_result = ParseResult(doc_id=doc_id, metadata=metadata, elements=[element], errors=[])
    parsing = StubParsingPipeline(parse_result)

    extractor = StubTripletExtractor({}, failures=[element.element_id])
    inventory = StubInventoryBuilder({element.element_id: ["Model Z", "Model Q"]})
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

    assert result.processed_chunks == 1
    assert result.co_mention_edges == 1
    assert result.edges_written == 1
    assert len(graph_writer.edges) == 1
    co_edge = graph_writer.edges[0]
    assert co_edge["relation_norm"] == "correlates-with"
    assert co_edge["relation_verbatim"] == "co-mentioned"
    assert co_edge["attributes"].get("method") == "co-mention"
    assert co_edge["attributes"].get("hidden") == "true"
    assert co_edge["confidence"] == pytest.approx(config.co_mention.confidence)
    assert co_edge["times_seen"] == 2
    assert any("LLM failure" in error for error in result.errors)
    assert (
        chunk_store.should_process(doc_id, element.content_hash, config.pipeline.version)
        is False
    )
