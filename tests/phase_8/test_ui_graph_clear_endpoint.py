"""Tests covering the graph clear UI endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from fastapi.testclient import TestClient

from backend.app.config import load_config
from backend.app.main import create_app


class _StubOrchestrator:
    """Minimal orchestrator stub to satisfy the application factory."""

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not exercised
        raise NotImplementedError


class _StubGraphViewService:
    """Test double capturing clear_graph invocations."""

    def __init__(self) -> None:
        self.cleared = False

    def fetch_graph(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError

    def clear_graph(self) -> None:
        self.cleared = True


def _build_test_app(tmp_path: Path) -> Tuple[TestClient, _StubGraphViewService]:
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
    service = _StubGraphViewService()
    app.state.graph_view_service = service
    return TestClient(app), service


def test_clear_graph_endpoint_invokes_service(tmp_path: Path) -> None:
    client, service = _build_test_app(tmp_path)

    response = client.post("/api/ui/graph/clear")

    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}
    assert service.cleared is True
