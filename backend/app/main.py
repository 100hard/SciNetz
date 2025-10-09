"""FastAPI application factory for SciNets backend."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.app.canonicalization import CanonicalizationPipeline
from backend.app.config import AppConfig, load_config
from backend.app.extraction import EntityInventoryBuilder, OpenAIExtractor, TwoPassTripletExtractor
from backend.app.graph import GraphWriter
from backend.app.orchestration import ExtractionOrchestrator, OrchestrationResult
from backend.app.orchestration.orchestrator import ProcessedChunkStore
from backend.app.parsing import ParsingPipeline
from backend.app.qa import Neo4jQARepository, QAService
from backend.app.ui import (
    GraphView,
    GraphViewService,
    Neo4jGraphViewRepository,
    PaperRecord,
    PaperRegistry,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency for runtime only
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - driver optional for tests
    GraphDatabase = None  # type: ignore[misc,assignment]


class ExtractionRequest(BaseModel):
    """Request payload for the extraction endpoint."""

    pdf_path: str = Field(..., min_length=1, description="Absolute path to the PDF file")


class QARequest(BaseModel):
    """Question answering request payload."""

    question: str = Field(..., min_length=1, description="Natural language question to answer")


class PaperSummary(BaseModel):
    """Serialized representation of a paper registry entry."""

    paper_id: str
    filename: str
    status: str
    uploaded_at: str
    updated_at: str
    metadata: Optional[Dict[str, object]] = None
    errors: List[str] = Field(default_factory=list)
    nodes_written: int = 0
    edges_written: int = 0
    co_mention_edges: int = 0


class GraphNodePayload(BaseModel):
    """Node description returned for UI graph rendering."""

    id: str
    label: str
    type: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    times_seen: int
    section_distribution: Dict[str, int] = Field(default_factory=dict)


class GraphEdgePayload(BaseModel):
    """Edge description including evidence metadata."""

    id: str
    source: str
    target: str
    relation: str
    relation_verbatim: str
    confidence: float
    times_seen: int
    attributes: Dict[str, str] = Field(default_factory=dict)
    evidence: Dict[str, object] = Field(default_factory=dict)
    conflicting: bool = False
    created_at: Optional[str] = None


class GraphResponse(BaseModel):
    """Graph payload consumed by the frontend."""

    nodes: List[GraphNodePayload]
    edges: List[GraphEdgePayload]
    node_count: int
    edge_count: int


class UISettingsResponse(BaseModel):
    """UI configuration defaults served to the frontend."""

    graph_defaults: Dict[str, object]


def create_app(
    config: AppConfig | None = None,
    orchestrator: Optional[ExtractionOrchestrator] = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        config: Optional pre-loaded configuration. If omitted, the default
            configuration defined in config.yaml is used.
        orchestrator: Optional pipeline orchestrator. When omitted the factory
            attempts to create one using default dependencies. If creation
            fails, the extraction endpoint will return ``503``.

    Returns:
        FastAPI: Configured FastAPI application.
    """

    resolved_config = config or load_config()
    app = FastAPI(title="SciNets API", version=resolved_config.pipeline.version)

    orchestrator_instance = orchestrator
    neo4j_driver = None
    if orchestrator_instance is None:
        orchestrator_instance, neo4j_driver = _build_default_orchestrator(resolved_config)
    app.state.orchestrator = orchestrator_instance
    app.state.neo4j_driver = neo4j_driver
    qa_service = _build_default_qa_service(resolved_config, neo4j_driver)
    app.state.qa_service = qa_service
    root_dir = Path(__file__).resolve().parents[2]
    upload_dir = (root_dir / resolved_config.ui.upload_dir).resolve()
    registry_path = (root_dir / resolved_config.ui.paper_registry_path).resolve()
    paper_registry = PaperRegistry(registry_path)
    app.state.paper_registry = paper_registry
    app.state.upload_dir = upload_dir
    app.state.ui_config = resolved_config.ui
    graph_view_service = _build_graph_view_service(neo4j_driver)
    app.state.graph_view_service = graph_view_service

    @app.get("/health", tags=["system"], summary="Service health probe")
    def health() -> dict[str, str]:
        """Return service health information."""

        return {"status": "ok", "pipeline_version": resolved_config.pipeline.version}

    @app.get("/api/ui/settings", tags=["ui"], summary="UI configuration defaults")
    def ui_settings() -> UISettingsResponse:
        """Return UI defaults sourced from the configuration file."""

        graph_defaults = resolved_config.ui.graph_defaults
        payload = {
            "relations": list(graph_defaults.relations),
            "min_confidence": graph_defaults.min_confidence,
            "sections": list(graph_defaults.sections),
            "show_co_mentions": graph_defaults.show_co_mentions,
            "layout": graph_defaults.layout,
        }
        return UISettingsResponse(graph_defaults=payload)

    @app.post("/api/extract/{paper_id}", tags=["extraction"], summary="Run extraction pipeline")
    def extract(paper_id: str, request: ExtractionRequest) -> dict[str, object]:
        """Run the extraction orchestrator for the supplied paper."""

        orchestrator_obj = getattr(app.state, "orchestrator", None)
        if orchestrator_obj is None:
            raise HTTPException(status_code=503, detail="Extraction orchestrator unavailable")
        pdf_path = Path(request.pdf_path)
        if not pdf_path.exists():
            raise HTTPException(status_code=400, detail="PDF path does not exist")
        result: OrchestrationResult = orchestrator_obj.run(
            paper_id=paper_id,
            pdf_path=pdf_path,
        )
        response = {
            "doc_id": result.doc_id,
            "metadata": result.metadata.model_dump(),
            "processed_chunks": result.processed_chunks,
            "skipped_chunks": result.skipped_chunks,
            "nodes_written": result.nodes_written,
            "edges_written": result.edges_written,
            "co_mention_edges": result.co_mention_edges,
            "errors": result.errors,
        }
        return response

    @app.post(
        "/api/ui/upload",
        tags=["ui"],
        summary="Upload a PDF for extraction",
    )
    async def upload_paper(
        file: UploadFile = File(..., description="PDF file to upload"),
        paper_id: Optional[str] = Form(default=None, description="Optional paper identifier"),
    ) -> PaperSummary:
        """Persist an uploaded PDF and register it for processing."""

        registry = _require_registry(app)
        upload_dir: Path = getattr(app.state, "upload_dir")
        candidate_id = paper_id or (file.filename or "paper")
        derived_id = _derive_paper_id(candidate_id, registry)
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / f"{derived_id}.pdf"
        content = await file.read()
        pdf_path.write_bytes(content)
        record = registry.register_upload(derived_id, file.filename or pdf_path.name, pdf_path)
        return _record_to_summary(record)

    @app.get(
        "/api/ui/papers",
        tags=["ui"],
        summary="List uploaded papers and their processing status",
    )
    def list_papers() -> List[PaperSummary]:
        """Return registered papers sorted by upload time."""

        registry = _require_registry(app)
        return [_record_to_summary(record) for record in registry.list_records()]

    @app.post(
        "/api/ui/papers/{paper_id}/extract",
        tags=["ui"],
        summary="Run extraction for an uploaded paper",
    )
    def extract_uploaded(paper_id: str) -> dict[str, object]:
        """Execute the extraction pipeline for a previously uploaded PDF."""

        registry = _require_registry(app)
        record = registry.get(paper_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        orchestrator_obj = getattr(app.state, "orchestrator", None)
        if orchestrator_obj is None:
            raise HTTPException(status_code=503, detail="Extraction orchestrator unavailable")
        registry.mark_processing(paper_id)
        try:
            result: OrchestrationResult = orchestrator_obj.run(
                paper_id=paper_id,
                pdf_path=record.pdf_path,
            )
        except Exception as exc:  # noqa: BLE001 - propagate failure to caller
            registry.mark_failed(paper_id, [str(exc)])
            raise HTTPException(status_code=500, detail="Extraction failed") from exc
        if result.errors:
            registry.mark_complete(
                paper_id,
                metadata=result.metadata,
                nodes_written=result.nodes_written,
                edges_written=result.edges_written,
                co_mention_edges=result.co_mention_edges,
                errors=result.errors,
            )
        else:
            registry.mark_complete(
                paper_id,
                metadata=result.metadata,
                nodes_written=result.nodes_written,
                edges_written=result.edges_written,
                co_mention_edges=result.co_mention_edges,
            )
        response = {
            "doc_id": result.doc_id,
            "metadata": result.metadata.model_dump(),
            "processed_chunks": result.processed_chunks,
            "skipped_chunks": result.skipped_chunks,
            "nodes_written": result.nodes_written,
            "edges_written": result.edges_written,
            "co_mention_edges": result.co_mention_edges,
            "errors": result.errors,
        }
        return response

    @app.get(
        "/api/ui/graph",
        tags=["ui"],
        summary="Fetch a filtered graph view",
    )
    def graph_view(
        relations: Optional[str] = Query(None, description="Comma-separated relation filters"),
        min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
        sections: Optional[str] = Query(None, description="Comma-separated section filters"),
        include_co_mentions: Optional[bool] = Query(None, description="Whether to include co-mention edges"),
        papers: Optional[str] = Query(None, description="Comma-separated paper identifiers"),
        limit: int = Query(500, ge=1, le=2000),
    ) -> GraphResponse:
        """Return nodes and edges satisfying the provided filters."""

        service = _require_graph_service(app)
        defaults = getattr(app.state, "ui_config").graph_defaults
        relation_list = _parse_csv(relations) or list(defaults.relations)
        section_list = _parse_csv(sections) or list(defaults.sections)
        paper_list = _parse_csv(papers)
        min_conf = min_confidence if min_confidence is not None else defaults.min_confidence
        include = (
            include_co_mentions if include_co_mentions is not None else defaults.show_co_mentions
        )
        view = service.fetch_graph(
            relations=relation_list,
            min_confidence=min_conf,
            sections=section_list,
            include_co_mentions=include,
            papers=paper_list,
            limit=limit,
        )
        return _graph_response_from_view(view)

    @app.post("/api/qa/ask", tags=["qa"], summary="Answer question using the knowledge graph")
    def ask(request: QARequest) -> dict[str, object]:
        """Answer a user question using the graph-first QA pipeline."""

        qa_service = getattr(app.state, "qa_service", None)
        if qa_service is None:
            raise HTTPException(status_code=503, detail="QA service unavailable")
        result = qa_service.answer(request.question)
        return result.model_dump()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - network resource cleanup
        driver = getattr(app.state, "neo4j_driver", None)
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001 - defensive close
                LOGGER.exception("Failed to close Neo4j driver")

    return app


def _build_default_orchestrator(
    config: AppConfig,
) -> Tuple[Optional[ExtractionOrchestrator], Optional[object]]:
    """Construct the default orchestrator if dependencies are available."""

    try:
        parsing = ParsingPipeline(config)
        inventory = EntityInventoryBuilder(config)
        llm_extractor = _create_llm_extractor(config)
        if llm_extractor is None:
            LOGGER.warning("LLM extractor unavailable; extraction endpoint disabled")
            return None, None
        triplet_extractor = TwoPassTripletExtractor(config=config, llm_extractor=llm_extractor)
        canonicalization = CanonicalizationPipeline(config=config)
        graph_writer, driver = _create_graph_writer(config)
        if graph_writer is None:
            LOGGER.warning("Graph writer unavailable; extraction endpoint disabled")
            return None, None
        store_path = Path(__file__).resolve().parents[2] / "data" / "pipeline" / "processed_chunks.json"
        chunk_store = ProcessedChunkStore(store_path)
        orchestrator = ExtractionOrchestrator(
            config=config,
            parsing_pipeline=parsing,
            inventory_builder=inventory,
            triplet_extractor=triplet_extractor,
            canonicalization=canonicalization,
            graph_writer=graph_writer,
            chunk_store=chunk_store,
        )
        return orchestrator, driver
    except Exception:  # noqa: BLE001 - safeguard during startup
        LOGGER.exception("Failed to initialize extraction orchestrator")
        return None, None


def _build_default_qa_service(config: AppConfig, driver: Optional[object]) -> Optional[QAService]:
    """Instantiate the QA service when a Neo4j driver is available."""

    if driver is None:
        return None
    try:
        repository = Neo4jQARepository(driver, config=config)  # type: ignore[arg-type]
        return QAService(config=config, repository=repository)
    except Exception:  # noqa: BLE001 - QA should fail softly
        LOGGER.exception("Failed to initialize QA service")
        return None


def _build_graph_view_service(driver: Optional[object]) -> Optional[GraphViewService]:
    """Construct the graph view service when a Neo4j driver is available."""

    if driver is None:
        return None
    try:
        repository = Neo4jGraphViewRepository(driver)  # type: ignore[arg-type]
        return GraphViewService(repository)
    except Exception:  # noqa: BLE001 - view service is optional
        LOGGER.exception("Failed to initialize graph view service")
        return None


def _require_registry(app: FastAPI) -> PaperRegistry:
    registry = getattr(app.state, "paper_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Paper registry unavailable")
    return registry


def _require_graph_service(app: FastAPI) -> GraphViewService:
    service = getattr(app.state, "graph_view_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Graph view service unavailable")
    return service


def _record_to_summary(record: PaperRecord) -> PaperSummary:
    return PaperSummary(**record.to_dict())


def _graph_response_from_view(view: GraphView) -> GraphResponse:
    nodes = [
        GraphNodePayload(
            id=node.id,
            label=node.label,
            type=node.type,
            aliases=list(node.aliases),
            times_seen=node.times_seen,
            section_distribution=dict(node.section_distribution),
        )
        for node in view.nodes
    ]
    edges = [
        GraphEdgePayload(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            relation=edge.relation,
            relation_verbatim=edge.relation_verbatim,
            confidence=edge.confidence,
            times_seen=edge.times_seen,
            attributes=dict(edge.attributes),
            evidence=dict(edge.evidence),
            conflicting=edge.conflicting,
            created_at=edge.created_at,
        )
        for edge in view.edges
    ]
    return GraphResponse(nodes=nodes, edges=edges, node_count=view.node_count, edge_count=view.edge_count)


def _parse_csv(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    values = [part.strip() for part in raw.split(",")]
    return [value for value in values if value]


def _derive_paper_id(candidate: str, registry: PaperRegistry) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", candidate.lower()).strip("-")
    if not base:
        base = f"paper-{uuid4().hex[:8]}"
    unique = base
    counter = 1
    while registry.get(unique) is not None:
        unique = f"{base}-{counter}"
        counter += 1
    return unique


def _create_llm_extractor(config: AppConfig) -> Optional[OpenAIExtractor]:
    """Instantiate the default OpenAI extractor if credentials are present."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAIExtractor(settings=config.extraction.openai, api_key=api_key)
    except Exception:  # noqa: BLE001 - dependency failures should disable extractor
        LOGGER.exception("Failed to initialize OpenAI extractor")
        return None


def _create_graph_writer(
    config: AppConfig,
) -> Tuple[Optional[GraphWriter], Optional[object]]:
    """Create the graph writer backed by a Neo4j driver if available."""

    if GraphDatabase is None:
        return None, None
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and password):
        return None, None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception:  # noqa: BLE001 - connection issues
        LOGGER.exception("Failed to connect to Neo4j driver")
        return None, None
    writer = GraphWriter(driver=driver, config=config)
    return writer, driver
