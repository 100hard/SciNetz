from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from fastapi.testclient import TestClient

from backend.app.config import load_config
from backend.app.export import (
    ExportBundle,
    ExportSizeExceeded,
    ExportSizeWarning,
    GraphExportService,
)
from backend.app.main import create_app
from backend.app.ui.service import GraphEdge, GraphNode, GraphView


class _StubOrchestrator:
    """Placeholder orchestrator to satisfy the app factory."""

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


class _StubGraphViewService:
    """Graph service stub returning a fixed view and recording filters."""

    def __init__(self, view: GraphView) -> None:
        self._view = view
        self.last_request: Dict[str, Any] | None = None

    def fetch_graph(
        self,
        *,
        relations: Sequence[str],
        min_confidence: float,
        sections: Sequence[str],
        include_co_mentions: bool,
        papers: Sequence[str],
        limit: int,
    ) -> GraphView:
        self.last_request = {
            "relations": list(relations),
            "min_confidence": min_confidence,
            "sections": list(sections),
            "include_co_mentions": include_co_mentions,
            "papers": list(papers),
            "limit": limit,
        }
        return self._view

    def clear_graph(self) -> None:  # pragma: no cover - not used
        raise NotImplementedError


class _WarningExportService:
    def generate_bundle(self, *args: Any, **kwargs: Any) -> ExportBundle:
        raise ExportSizeWarning("Export is large. Options:")


class _ExceededExportService:
    def generate_bundle(self, *args: Any, **kwargs: Any) -> ExportBundle:
        raise ExportSizeExceeded("Export too large. Please apply stricter filters or paginate by paper.")


def _build_view() -> GraphView:
    node_a = GraphNode(
        id="node-a",
        label="Model Alpha",
        type="model",
        aliases=["Alpha"],
        times_seen=3,
        section_distribution={"Results": 2},
        source_document_ids=["paper-1"],
    )
    node_b = GraphNode(
        id="node-b",
        label="Dataset Beta",
        type="dataset",
        aliases=["Beta"],
        times_seen=1,
        section_distribution={"Results": 1},
        source_document_ids=["paper-2"],
    )
    edge = GraphEdge(
        id="edge-1",
        source=node_a.id,
        target=node_b.id,
        relation="uses",
        relation_verbatim="uses",
        confidence=0.9,
        times_seen=1,
        attributes={"section": "Results"},
        evidence={
            "doc_id": "paper-1",
            "element_id": "el-1",
            "text_span": {"start": 10, "end": 60},
            "full_sentence": "Model Alpha uses Dataset Beta to improve accuracy.",
        },
        conflicting=False,
        created_at="2024-01-01T00:00:00Z",
    )
    return GraphView(nodes=[node_a, node_b], edges=[edge])


def _build_test_app(tmp_path: Path) -> Tuple[TestClient, _StubGraphViewService]:
    config = load_config()
    ui_config = config.ui.model_copy(
        update={
            "upload_dir": str(tmp_path / "uploads"),
            "paper_registry_path": str(tmp_path / "registry.json"),
        }
    )
    graph_config = config.graph.model_copy(update={"uri": None, "username": None, "password": None})
    app_config = config.model_copy(update={"ui": ui_config, "graph": graph_config})
    app = create_app(config=app_config, orchestrator=_StubOrchestrator())

    view_service = _StubGraphViewService(_build_view())
    template_path = Path(__file__).resolve().parents[2] / "export" / "scinets_view.html"
    export_service = GraphExportService(
        graph_service=view_service,
        export_config=config.export,
        pipeline_version=config.pipeline.version,
        template_path=template_path,
    )

    app.state.graph_view_service = view_service
    app.state.export_service = export_service

    return TestClient(app), view_service


def test_export_endpoint_returns_zip_bundle(tmp_path: Path) -> None:
    client, view_service = _build_test_app(tmp_path)

    response = client.get(
        "/api/export/html",
        params={
            "relations": "uses",
            "min_confidence": 0.6,
            "sections": "Results",
            "include_snippets": True,
            "truncate_snippets": False,
            "papers": "paper-1",
            "limit": 250,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "filename=" in response.headers["content-disposition"].lower()
    assert view_service.last_request is not None
    assert view_service.last_request["relations"] == ["uses"]
    assert view_service.last_request["min_confidence"] == 0.6
    assert view_service.last_request["limit"] == 250

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    html_name = next(name for name in archive.namelist() if name.endswith("index.html"))
    graph_json_name = next(name for name in archive.namelist() if name.endswith("graph.json"))
    html = archive.read(html_name).decode("utf-8")
    assert "Evidence inspector" in html
    graph_payload = json.loads(archive.read(graph_json_name))
    assert graph_payload["nodes"][0]["label"] == "Model Alpha"


def test_export_endpoint_returns_warning_on_large_size(tmp_path: Path) -> None:
    client, _ = _build_test_app(tmp_path)
    client.app.state.export_service = _WarningExportService()

    response = client.get("/api/export/html")

    assert response.status_code == 400
    assert "Export is large" in response.json()["detail"]


def test_export_endpoint_errors_when_exceeding_limit(tmp_path: Path) -> None:
    client, _ = _build_test_app(tmp_path)
    client.app.state.export_service = _ExceededExportService()

    response = client.get("/api/export/html")

    assert response.status_code == 413
    assert "Export too large" in response.json()["detail"]
