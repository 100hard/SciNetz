"""FastAPI application factory for SciNets backend."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.app.canonicalization import CanonicalizationPipeline
from backend.app.config import AppConfig, load_config
from backend.app.extraction import EntityInventoryBuilder, OpenAIExtractor, TwoPassTripletExtractor
from backend.app.graph import GraphWriter
from backend.app.orchestration import ExtractionOrchestrator, OrchestrationResult
from backend.app.orchestration.orchestrator import ProcessedChunkStore
from backend.app.parsing import ParsingPipeline

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency for runtime only
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - driver optional for tests
    GraphDatabase = None  # type: ignore[misc,assignment]


class ExtractionRequest(BaseModel):
    """Request payload for the extraction endpoint."""

    pdf_path: str = Field(..., min_length=1, description="Absolute path to the PDF file")


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

    @app.get("/health", tags=["system"], summary="Service health probe")
    def health() -> dict[str, str]:
        """Return service health information."""

        return {"status": "ok", "pipeline_version": resolved_config.pipeline.version}

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
