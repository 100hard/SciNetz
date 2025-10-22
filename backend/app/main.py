"""FastAPI application factory for SciNets backend."""
from __future__ import annotations

import base64
import binascii
import contextlib
import io
import json
import logging
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from backend.app.auth.models import Base as AuthBase
from backend.app.auth.router import router as auth_router
from backend.app.auth.utils import EmailDispatcher, JWTManager
from backend.app.canonicalization import CanonicalizationPipeline
from backend.app.config import AppConfig, load_config
from backend.app.export.bundle import ExportBundleBuilder, GraphViewExportProvider
from backend.app.export.metadata import JSONShareMetadataRepository
from backend.app.export.models import ShareExportFilters, ShareExportRequest, ShareExportResponse
from backend.app.export.service import ExportSizeLimitError, ShareExportService
from backend.app.export.storage import S3BundleStorage
from backend.app.export.token import ExpiredTokenError, InvalidTokenError, ShareTokenManager
from backend.app.export.viewer import render_share_html
from backend.app.extraction import (
    EntityInventoryBuilder,
    LLMResponseCache,
    OpenAIExtractor,
    TokenBudgetCache,
    TwoPassTripletExtractor,
)
from backend.app.graph import GraphWriter
from backend.app.orchestration import ExtractionOrchestrator, OrchestrationResult
from backend.app.orchestration.orchestrator import ProcessedChunkStore
from backend.app.parsing import ParsingPipeline
from backend.app.qa import LLMAnswerSynthesizer, Neo4jQARepository, QAService
from backend.app.ui import (
    GraphView,
    GraphViewService,
    Neo4jGraphViewRepository,
    PaperRecord,
    PaperRegistry,
    PaperStatus,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency for runtime only
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - driver optional for tests
    GraphDatabase = None  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency for runtime only
    from minio import Minio
except Exception:  # pragma: no cover - Minio optional for tests
    Minio = None  # type: ignore[misc,assignment]


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


class UploadPaperRequest(BaseModel):
    """Request payload for uploading a paper via the UI."""

    filename: str = Field(..., min_length=1, description="Original filename supplied by the user")
    content_base64: str = Field(..., min_length=1, description="Base64-encoded PDF content")
    paper_id: Optional[str] = Field(default=None, description="Optional caller-provided identifier")


class GraphNodePayload(BaseModel):
    """Node description returned for UI graph rendering."""

    id: str
    label: str
    type: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    times_seen: int
    section_distribution: Dict[str, int] = Field(default_factory=dict)
    source_document_ids: List[str] = Field(default_factory=list)


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


class ShareExportAPIRequest(BaseModel):
    """Request payload for generating a shareable export link via the API."""

    filters: ShareExportFilters
    include_snippets: bool = True
    truncate_snippets: bool = False
    requested_by: str = Field(..., min_length=1)


class ShareLinkResolution(BaseModel):
    """Response payload for resolving a shareable export link."""

    download_url: str
    expires_at: Optional[datetime]
    pipeline_version: str
    bundle_size_mb: float
    warning: bool = False


class ShareLinkRevokeRequest(BaseModel):
    """Request payload when revoking a share link."""

    revoked_by: str = Field("system", min_length=1)


class UISettingsResponse(BaseModel):
    """UI configuration defaults served to the frontend."""

    graph_defaults: Dict[str, object]
    qa: Dict[str, object]


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

    auth_engine: AsyncEngine = create_async_engine(
        resolved_config.auth.database_url, future=True
    )
    auth_session_factory = async_sessionmaker(auth_engine, expire_on_commit=False)
    app.state.auth_engine = auth_engine
    app.state.auth_session_factory = auth_session_factory
    app.state.auth_config = resolved_config.auth
    app.state.jwt_manager = JWTManager(resolved_config.auth.jwt)
    app.state.email_dispatcher = EmailDispatcher(
        resolved_config.auth.smtp, resolved_config.auth.verification
    )

    @app.on_event("startup")
    async def _init_auth_schema() -> None:
        async with auth_engine.begin() as connection:
            await connection.run_sync(AuthBase.metadata.create_all)

    @app.on_event("shutdown")
    async def _dispose_auth_engine() -> None:
        await auth_engine.dispose()

    allowed_origins = resolved_config.ui.allowed_origins
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    orchestrator_instance = orchestrator
    neo4j_driver = None
    if orchestrator_instance is None:
        orchestrator_instance, neo4j_driver = _build_default_orchestrator(resolved_config)
    if neo4j_driver is None:
        neo4j_driver = _create_neo4j_driver(resolved_config)
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
    (
        share_export_service,
        share_metadata_repo,
        share_token_manager,
        share_storage_client,
        share_storage_reader,
    ) = _build_share_export_service(resolved_config, graph_view_service, root_dir)
    app.state.share_export_service = share_export_service
    app.state.share_metadata_repository = share_metadata_repo
    app.state.share_token_manager = share_token_manager
    app.state.share_storage_client = share_storage_client
    app.state.share_storage_reader = share_storage_reader

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
        qa_llm = getattr(resolved_config.qa, "llm", None)
        qa_settings = {
            "llm_enabled": bool(getattr(qa_llm, "enabled", False)),
            "llm_provider": getattr(qa_llm, "provider", None),
        }
        return UISettingsResponse(graph_defaults=payload, qa=qa_settings)

    @app.post(
        "/api/export/share",
        tags=["export"],
        summary="Create shareable export link",
        response_model=ShareExportResponse,
    )
    def create_share_link(payload: ShareExportAPIRequest) -> ShareExportResponse:
        """Generate a shareable export link for the provided filters."""

        service: Optional[ShareExportService] = getattr(app.state, "share_export_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Share export service unavailable")
        request = ShareExportRequest(
            filters=payload.filters,
            include_snippets=payload.include_snippets,
            truncate_snippets=payload.truncate_snippets,
            requested_by=payload.requested_by,
            pipeline_version=resolved_config.pipeline.version,
        )
        try:
            response = service.create_share(request)
        except ExportSizeLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to create share export link")
            raise HTTPException(status_code=500, detail="Unable to create share link") from exc
        return response

    @app.get(
        "/api/export/share/{token}",
        tags=["export"],
        summary="Resolve shareable export link",
        response_model=ShareLinkResolution,
        responses={200: {"content": {"text/html": {"description": "Rendered shared graph view"}}}},
    )
    def resolve_share_link(
        token: str,
        request: Request,
        format: Optional[str] = Query(default=None),
    ):
        """Return a share link resolution payload or rendered HTML view."""

        token_manager: Optional[ShareTokenManager] = getattr(app.state, "share_token_manager", None)
        metadata_repo = getattr(app.state, "share_metadata_repository", None)
        storage_client = getattr(app.state, "share_storage_client", None)
        storage_reader = getattr(app.state, "share_storage_reader", storage_client)
        if not (token_manager and metadata_repo and storage_client):
            raise HTTPException(status_code=503, detail="Share export service unavailable")
        try:
            decoded = token_manager.decode(token)
        except ExpiredTokenError as exc:
            raise HTTPException(status_code=410, detail="Share link expired") from exc
        except InvalidTokenError as exc:
            raise HTTPException(status_code=403, detail="Invalid share token") from exc

        record = metadata_repo.fetch(decoded.metadata_id)
        if not record:
            raise HTTPException(status_code=404, detail="Share link not found")
        revoked_at_raw = record.get("revoked_at")
        if revoked_at_raw:
            raise HTTPException(status_code=410, detail="Share link revoked")
        record_expires_at: Optional[datetime] = None
        expires_at_raw = record.get("expires_at")
        if isinstance(expires_at_raw, datetime):
            record_expires_at = expires_at_raw
        elif isinstance(expires_at_raw, str) and expires_at_raw:
            try:
                record_expires_at = datetime.fromisoformat(expires_at_raw)
            except ValueError:
                LOGGER.warning("Invalid expires_at on share metadata %s", decoded.metadata_id)
                record_expires_at = None
        effective_expires_at = record_expires_at or decoded.expires_at
        now_utc = datetime.now(timezone.utc)
        if effective_expires_at is not None and now_utc >= effective_expires_at:
            raise HTTPException(status_code=410, detail="Share link expired")
        bucket = resolved_config.export.storage.bucket
        object_key = str(record.get("object_key"))
        if not object_key:
            raise HTTPException(status_code=500, detail="Share metadata missing object key")
        try:
            download_url = storage_client.presigned_get_object(
                bucket,
                object_key,
                expires=timedelta(minutes=resolved_config.export.signed_url_ttl_minutes),
            )
        except Exception as exc:  # pragma: no cover - storage failures logged
            LOGGER.exception("Failed to issue presigned download URL")
            raise HTTPException(status_code=500, detail="Unable to generate download URL") from exc

        prefers_html = _prefers_html_response(request, format=format)
        if prefers_html:
            try:
                graph_payload = _load_graph_payload(storage_reader, bucket, object_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Failed to render shared graph view")
                raise HTTPException(status_code=500, detail="Unable to render share view") from exc
            html_content = render_share_html(
                graph_payload,
                download_url=download_url,
                expires_at=effective_expires_at,
            )
            csp_header = (
                "default-src 'none'; "
                "script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; "
                "img-src data:; "
                "font-src data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'none'; "
                "form-action 'none'"
            )
            return HTMLResponse(
                content=html_content,
                headers={
                    "Content-Security-Policy": csp_header,
                    "X-Content-Type-Options": "nosniff",
                },
            )

        size_bytes = float(record.get("size_bytes", 0) or 0)
        pipeline_version = str(record.get("pipeline_version") or resolved_config.pipeline.version)
        return ShareLinkResolution(
            download_url=download_url,
            expires_at=effective_expires_at,
            pipeline_version=pipeline_version,
            bundle_size_mb=size_bytes / 1_000_000 if size_bytes else 0.0,
            warning=bool(record.get("warning", False)),
        )

    @app.post(
        "/api/export/share/{metadata_id}/revoke",
        tags=["export"],
        summary="Revoke a shareable export link",
    )
    def revoke_share_link(metadata_id: str, payload: ShareLinkRevokeRequest) -> dict[str, object]:
        """Revoke an existing share link, preventing further downloads."""

        service: Optional[ShareExportService] = getattr(app.state, "share_export_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Share export service unavailable")
        try:
            updated = service.revoke_share(metadata_id, revoked_by=payload.revoked_by)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Share metadata not found") from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to revoke share link")
            raise HTTPException(status_code=500, detail="Unable to revoke share link") from exc
        return {"metadata_id": metadata_id, "revoked": True, "previously_revoked": not updated}

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
    def upload_paper(request: UploadPaperRequest) -> PaperSummary:
        """Persist an uploaded PDF and register it for processing."""

        registry = _require_registry(app)
        upload_dir: Path = getattr(app.state, "upload_dir")
        original_name = request.paper_id or request.filename or "paper"
        candidate_id = _derive_paper_id(original_name, registry)
        try:
            content = base64.b64decode(request.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:  # pragma: no cover - validation guard
            raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / f"{candidate_id}.pdf"
        pdf_path.write_bytes(content)
        record = registry.register_upload(candidate_id, request.filename or pdf_path.name, pdf_path)
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
        force_reprocess = record.status in {PaperStatus.COMPLETE, PaperStatus.FAILED}
        orchestrator_obj = getattr(app.state, "orchestrator", None)
        if orchestrator_obj is None:
            raise HTTPException(status_code=503, detail="Extraction orchestrator unavailable")
        registry.mark_processing(paper_id)
        try:
            result: OrchestrationResult = orchestrator_obj.run(
                paper_id=paper_id,
                pdf_path=record.pdf_path,
                force=force_reprocess,
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

    @app.post(
        "/api/ui/graph/clear",
        tags=["ui"],
        summary="Remove graph data from the Neo4j store",
    )
    def clear_graph() -> dict[str, str]:
        """Delete nodes and edges so subsequent UI fetches return an empty graph."""

        service = _require_graph_service(app)
        try:
            service.clear_graph()
        except Exception as exc:  # noqa: BLE001 - surface failure without hiding details
            LOGGER.exception("Failed to clear graph state")
            raise HTTPException(status_code=500, detail="Unable to clear graph") from exc
        return {"status": "cleared"}

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

    app.include_router(auth_router)

    return app


def _build_share_export_service(
    config: AppConfig,
    graph_service: GraphViewService,
    root_dir: Path,
) -> Tuple[
    Optional[ShareExportService],
    Optional[JSONShareMetadataRepository],
    Optional[ShareTokenManager],
    Optional[object],
    Optional[object],
]:
    """Construct the share export service if storage credentials are present."""

    if Minio is None:
        LOGGER.warning("MinIO dependency unavailable; disabling share export")
        return (None, None, None, None, None)

    endpoint_raw = os.getenv("EXPORT_STORAGE_ENDPOINT")
    access_key = os.getenv("EXPORT_STORAGE_ACCESS_KEY")
    secret_key = os.getenv("EXPORT_STORAGE_SECRET_KEY")
    token_secret = os.getenv("EXPORT_TOKEN_SECRET")
    if not all([endpoint_raw, access_key, secret_key, token_secret]):
        LOGGER.warning("Missing share export credentials; set EXPORT_STORAGE_* env vars to enable")
        return (None, None, None, None, None)
    secure_default = os.getenv("EXPORT_STORAGE_SECURE", "false").strip().lower() == "true"

    try:
        internal_endpoint, internal_secure = _normalise_endpoint(endpoint_raw, secure_default)
    except ValueError:
        LOGGER.warning("Invalid export storage endpoint '%s'; disabling share export", endpoint_raw)
        return (None, None, None, None, None)

    try:
        client = Minio(
            internal_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=internal_secure,
            region=config.export.storage.region,
        )
        bucket = config.export.storage.bucket
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    except Exception:  # pragma: no cover - external dependency handling
        LOGGER.exception("Failed to initialise share export storage; disabling feature")
        return (None, None, None, None, None)

    public_endpoint_raw = (
        os.getenv("EXPORT_STORAGE_PUBLIC_ENDPOINT")
        or config.export.storage.public_endpoint
        or endpoint_raw
    )
    try:
        public_endpoint, public_secure = _normalise_endpoint(public_endpoint_raw, internal_secure)
    except ValueError:
        LOGGER.warning(
            "Invalid public export storage endpoint '%s'; defaulting to internal endpoint",
            public_endpoint_raw,
        )
        public_endpoint, public_secure = internal_endpoint, internal_secure

    try:
        public_client = Minio(
            public_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=public_secure,
            region=config.export.storage.region,
        )
        public_client._region_map[config.export.storage.bucket] = config.export.storage.region  # type: ignore[attr-defined]
    except Exception:
        LOGGER.exception("Failed to initialise public presign client; disabling share export")
        return (None, None, None, None, None)

    metadata_path = (root_dir / "data" / "export" / "share_metadata.json").resolve()
    bundle_dir = (root_dir / "data" / "export" / "bundles").resolve()
    metadata_repo = JSONShareMetadataRepository(metadata_path)
    token_manager = ShareTokenManager(secret_key=token_secret)
    bundle_builder = ExportBundleBuilder(
        graph_provider=GraphViewExportProvider(graph_service),
        output_dir=bundle_dir,
        pipeline_version=config.pipeline.version,
        snippet_truncate_length=config.export.snippet_truncate_length,
    )
    storage = S3BundleStorage(
        client=client,
        bucket=config.export.storage.bucket,
        prefix=config.export.storage.prefix,
    )
    service = ShareExportService(
        bundle_builder=bundle_builder,
        storage=storage,
        metadata_repository=metadata_repo,
        token_manager=token_manager,
        config=config.export,
        clock=lambda: datetime.now(timezone.utc),
    )
    return service, metadata_repo, token_manager, public_client, client


def _normalise_endpoint(raw_endpoint: str, secure_default: bool) -> Tuple[str, bool]:
    """Parse an endpoint string into host:port and secure flag."""

    if not raw_endpoint:
        raise ValueError("Endpoint cannot be empty")
    candidate = raw_endpoint.strip()
    if "://" not in candidate:
        scheme = "https" if secure_default else "http"
        candidate = f"{scheme}://{candidate}"
    parsed = urlparse(candidate)
    host = parsed.netloc or parsed.path
    if not host:
        raise ValueError(f"Invalid endpoint '{raw_endpoint}'")
    secure = parsed.scheme == "https"
    return host, secure


def _build_default_orchestrator(
    config: AppConfig,
) -> Tuple[Optional[ExtractionOrchestrator], Optional[object]]:
    """Construct the default orchestrator if dependencies are available."""

    try:
        driver = _create_neo4j_driver(config)
        if driver is None:
            LOGGER.warning("Graph writer unavailable; extraction endpoint disabled")
            return None, None
        parsing = ParsingPipeline(config)
        inventory = EntityInventoryBuilder(config)
        llm_extractor = _create_llm_extractor(config)
        if llm_extractor is None:
            LOGGER.warning("LLM extractor unavailable; extraction endpoint disabled")
            return None, driver
        triplet_extractor = TwoPassTripletExtractor(config=config, llm_extractor=llm_extractor)
        canonicalization = CanonicalizationPipeline(config=config)
        graph_writer = GraphWriter(
            driver=driver,
            config=config,
            entity_batch_size=config.graph.entity_batch_size,
            edge_batch_size=config.graph.edge_batch_size,
        )
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
        return None, driver


def _build_default_qa_service(config: AppConfig, driver: Optional[object]) -> Optional[QAService]:
    """Instantiate the QA service when a Neo4j driver is available."""

    if driver is None:
        return None
    try:
        repository = Neo4jQARepository(driver, config=config)  # type: ignore[arg-type]
        synthesizer = _build_qa_synthesizer(config)
        return QAService(
            config=config,
            repository=repository,
            answer_synthesizer=synthesizer,
        )
    except Exception:  # noqa: BLE001 - QA should fail softly
        LOGGER.exception("Failed to initialize QA service")
        return None


def _build_qa_synthesizer(config: AppConfig) -> Optional[LLMAnswerSynthesizer]:
    """Construct the optional QA LLM synthesizer based on configuration."""

    llm_config = getattr(config.qa, "llm", None)
    if llm_config is None or not llm_config.enabled:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        return LLMAnswerSynthesizer(
            llm_config=llm_config,
            api_key=api_key,
        )
    except Exception:  # noqa: BLE001 - failure should not disable QA entirely
        LOGGER.exception("Failed to initialize QA LLM synthesizer")
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
            source_document_ids=list(node.source_document_ids),
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
    cache_root = Path(config.extraction.cache_dir)
    if not cache_root.is_absolute():
        cache_root = Path(__file__).resolve().parents[2] / cache_root
    try:
        response_cache = LLMResponseCache(
            cache_root / config.extraction.response_cache_filename
        )
        token_cache = TokenBudgetCache(
            cache_root / config.extraction.token_cache_filename
        )
        return OpenAIExtractor(
            settings=config.extraction.openai,
            api_key=api_key,
            token_budget_per_triple=config.extraction.tokens_per_triple,
            allowed_relations=config.relations.canonical_relation_names(),
            entity_types=config.extraction.entity_types,
            max_prompt_entities=config.extraction.max_prompt_entities,
            response_cache=response_cache,
            token_cache=token_cache,
        )
    except Exception:  # noqa: BLE001 - dependency failures should disable extractor
        LOGGER.exception("Failed to initialize OpenAI extractor")
        return None


def _prefers_html_response(request: Request, *, format: Optional[str]) -> bool:
    """Determine if the caller prefers an HTML response."""

    if format:
        lowered = format.lower()
        if lowered == "html":
            return True
        if lowered == "json":
            return False
    accept = request.headers.get("accept", "")
    accept_lower = accept.lower()
    if "text/html" in accept_lower:
        return True
    if "application/json" in accept_lower and "text/html" not in accept_lower:
        return False
    return False


def _load_graph_payload(storage_client: object, bucket: str, object_key: str) -> Dict[str, object]:
    """Fetch and decode the stored graph payload from object storage."""

    if not hasattr(storage_client, "get_object"):
        raise RuntimeError("Storage client does not support get_object")
    response = storage_client.get_object(bucket, object_key)
    try:
        raw_bytes = response.read()
    finally:
        with contextlib.suppress(Exception):
            response.close()
        with contextlib.suppress(Exception):
            response.release_conn()
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as archive:
            graph_bytes = archive.read("graph.json")
    except KeyError as exc:
        raise RuntimeError("graph.json missing from export bundle") from exc
    decoded: Dict[str, object] = json.loads(graph_bytes.decode("utf-8"))
    return decoded


def _create_neo4j_driver(config: AppConfig) -> Optional[object]:
    """Create a Neo4j driver using configured connection details."""

    if GraphDatabase is None:
        return None
    uri = os.getenv("NEO4J_URI") or config.graph.uri
    user = os.getenv("NEO4J_USER") or config.graph.username
    password = os.getenv("NEO4J_PASSWORD") or config.graph.password
    if not (uri and user and password):
        LOGGER.warning("Neo4j connection details missing; graph-dependent features disabled")
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception:  # noqa: BLE001 - connection issues
        LOGGER.exception("Failed to connect to Neo4j driver")
        return None
    return driver
