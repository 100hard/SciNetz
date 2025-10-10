"""Tests for the paper upload endpoint using base64 payloads."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app.config import load_config
from backend.app.main import PaperSummary, create_app


class _StubOrchestrator:
    """Minimal orchestrator stub to satisfy the application factory."""

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not exercised
        raise NotImplementedError


def _build_test_app(tmp_path: Path) -> TestClient:
    """Create an application instance with temporary storage paths."""

    config = load_config()
    ui_config = config.ui.model_copy(
        update={
            "upload_dir": str(tmp_path / "uploads"),
            "paper_registry_path": str(tmp_path / "registry.json"),
        }
    )
    graph_config = config.graph.model_copy(
        update={"uri": None, "username": None, "password": None}
    )
    app_config = config.model_copy(update={"ui": ui_config, "graph": graph_config})
    app = create_app(config=app_config, orchestrator=_StubOrchestrator())
    return TestClient(app)


def test_upload_endpoint_persists_pdf(tmp_path: Path) -> None:
    """Uploading a PDF stores it on disk and updates the registry."""

    client = _build_test_app(tmp_path)
    payload = {
        "filename": "sample.pdf",
        "content_base64": base64.b64encode(b"%PDF-1.4 sample").decode("utf-8"),
    }

    response = client.post("/api/ui/upload", json=payload)
    assert response.status_code == 200

    summary = PaperSummary.model_validate(response.json())
    stored_pdf = tmp_path / "uploads" / f"{summary.paper_id}.pdf"
    assert stored_pdf.exists()
    assert stored_pdf.read_bytes() == b"%PDF-1.4 sample"

    registry_path = tmp_path / "registry.json"
    assert registry_path.exists()
    registry_content = registry_path.read_text(encoding="utf-8")
    assert summary.paper_id in registry_content


def test_upload_endpoint_rejects_invalid_base64(tmp_path: Path) -> None:
    """Invalid base64 payloads return a 400 error."""

    client = _build_test_app(tmp_path)
    payload = {"filename": "bad.pdf", "content_base64": "not-base64"}

    response = client.post("/api/ui/upload", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid base64 payload"


def test_upload_endpoint_handles_cors_preflight(tmp_path: Path) -> None:
    """CORS preflight requests receive the appropriate response headers."""

    client = _build_test_app(tmp_path)
    response = client.options(
        "/api/ui/upload",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
