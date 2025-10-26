"""Integration tests covering the UI extraction endpoint background execution."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.config import AppConfig, load_config
from backend.app.contracts import PaperMetadata
from backend.app.main import create_app
from backend.app.orchestration import OrchestrationResult
from backend.app.ui import PaperRegistry, PaperStatus


class _StubOrchestrator:
    """Orchestrator that records calls and returns a canned result."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Path, bool]] = []

    def run(self, *, paper_id: str, pdf_path: Path, force: bool) -> OrchestrationResult:
        self.calls.append((paper_id, pdf_path, force))
        metadata = PaperMetadata(doc_id=paper_id, title=f"Paper {paper_id}")
        return OrchestrationResult(
            doc_id=paper_id,
            metadata=metadata,
            processed_chunks=1,
            skipped_chunks=0,
            nodes_written=2,
            edges_written=1,
            co_mention_edges=0,
            errors=[],
        )


def _configure_app(tmp_path: Path) -> AppConfig:
    base_config = load_config()
    root_dir = Path(__file__).resolve().parents[2]
    upload_dir = tmp_path / "uploads"
    registry_path = tmp_path / "registry.json"
    upload_rel = os.path.relpath(upload_dir, root_dir)
    registry_rel = os.path.relpath(registry_path, root_dir)
    ui_config = base_config.ui.model_copy(
        update={
            "upload_dir": upload_rel,
            "paper_registry_path": registry_rel,
            "extraction_worker_count": 1,
        }
    )
    auth_config = base_config.auth.model_copy(
        update={
            "database_url": "sqlite+aiosqlite:///:memory:",
        }
    )
    return base_config.model_copy(update={"ui": ui_config, "auth": auth_config})


def test_ui_extraction_runs_in_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the UI extraction endpoint enqueues work and completes asynchronously."""

    for env_var in (
        "EXPORT_STORAGE_ENDPOINT",
        "EXPORT_STORAGE_ACCESS_KEY",
        "EXPORT_STORAGE_SECRET_KEY",
        "EXPORT_TOKEN_SECRET",
    ):
        monkeypatch.delenv(env_var, raising=False)

    config = _configure_app(tmp_path)
    orchestrator = _StubOrchestrator()
    with patch("backend.app.main.GoogleTokenVerifier") as verifier_ctor, patch(
        "backend.app.main._create_neo4j_driver", return_value=None
    ), patch(
        "backend.app.main._build_share_export_service",
        return_value=(None, None, None, None, None),
    ):
        verifier_ctor.return_value = object()
        app = create_app(config=config, orchestrator=orchestrator)

    pdf_bytes = base64.b64encode(b"%PDF-1.4\n%fake\n").decode("ascii")

    with TestClient(app) as client:
        upload_response = client.post(
            "/api/ui/upload",
            json={"filename": "paper.pdf", "content_base64": pdf_bytes},
        )
        assert upload_response.status_code == 200
        upload_payload = upload_response.json()
        paper_id = upload_payload["paper_id"]

        extract_response = client.post(f"/api/ui/papers/{paper_id}/extract")
        assert extract_response.status_code == 202
        extract_payload = extract_response.json()
        assert extract_payload["paper_id"] == paper_id
        assert extract_payload["status"] == PaperStatus.PROCESSING.value
        assert extract_payload.get("queued") is True

        registry: PaperRegistry = app.state.paper_registry
        deadline = time.time() + 5.0
        while time.time() < deadline:
            record = registry.get(paper_id)
            if record and record.status == PaperStatus.COMPLETE:
                assert record.metadata is not None
                assert record.metadata.title == f"Paper {paper_id}"
                break
            time.sleep(0.05)
        else:  # pragma: no cover - safety net in CI
            pytest.fail("Extraction did not complete within timeout")

    assert orchestrator.calls, "Background orchestrator was not invoked"
    call_paper_id, call_pdf_path, force_flag = orchestrator.calls[0]
    assert call_paper_id == paper_id
    assert call_pdf_path.exists()
    assert force_flag is False
