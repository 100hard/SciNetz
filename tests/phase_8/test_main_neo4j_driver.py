from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
import pytest
from neo4j.exceptions import ServiceUnavailable

from backend.app.config import load_config
from backend.app import main


class _DummyDriver:
    """Simple stand-in for the Neo4j driver object."""

    def __init__(self) -> None:
        self.verified = False
        self.executed = False
        self.closed = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def execute_query(self, query: str) -> None:  # noqa: D401 - mimic driver method
        self.executed = True

    def close(self) -> None:
        self.closed = True


class _RecordingGraphDatabase:
    """Test double that records the connection parameters."""

    def __init__(self) -> None:
        self.calls: Dict[str, Any] = {}

    def driver(self, uri: str, auth: tuple[str, str]) -> _DummyDriver:
        self.calls["uri"] = uri
        self.calls["auth"] = auth
        return _DummyDriver()


def _clear_neo4j_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)


def test_create_neo4j_driver_prefers_config_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuration credentials are used when environment variables are absent."""

    config = load_config()
    assert config.graph.uri is not None
    assert config.graph.username is not None
    assert config.graph.password is not None

    _clear_neo4j_env(monkeypatch)
    recording = _RecordingGraphDatabase()
    monkeypatch.setattr(main, "GraphDatabase", recording)

    driver = main._create_neo4j_driver(config)

    assert isinstance(driver, _DummyDriver)
    assert driver.verified is True
    assert recording.calls["uri"] == config.graph.uri
    assert recording.calls["auth"] == (config.graph.username, config.graph.password)
    assert driver.executed is True


def test_create_neo4j_driver_returns_none_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driver creation fails gracefully when no credentials are configured."""

    config = load_config()
    graph_config = config.graph.model_copy(update={"uri": None, "username": None, "password": None})
    app_config = config.model_copy(update={"graph": graph_config})

    _clear_neo4j_env(monkeypatch)
    calls: Dict[str, Any] = {}

    class _FailingGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> None:
            calls["uri"] = uri
            raise AssertionError("driver should not be called when credentials are missing")

    monkeypatch.setattr(main, "GraphDatabase", _FailingGraphDatabase)

    driver = main._create_neo4j_driver(app_config)

    assert driver is None
    assert "uri" not in calls


def test_create_neo4j_driver_falls_back_when_routing_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A routing URI gracefully falls back to a direct bolt connection."""

    config = load_config()
    graph_config = config.graph.model_copy(update={"uri": "neo4j://example.com:7687"})
    app_config = config.model_copy(update={"graph": graph_config})

    _clear_neo4j_env(monkeypatch)

    class _FailingDriver(_DummyDriver):
        def execute_query(self, query: str) -> None:  # noqa: D401 - override
            super().execute_query(query)
            raise ServiceUnavailable("Unable to retrieve routing information")

    class _RoutingGraphDatabase:
        def __init__(self) -> None:
            self.calls: List[str] = []
            self.first_driver = _FailingDriver()

        def driver(self, uri: str, auth: tuple[str, str]) -> _DummyDriver:
            self.calls.append(uri)
            if len(self.calls) == 1:
                return self.first_driver
            return _DummyDriver()

    recording = _RoutingGraphDatabase()
    monkeypatch.setattr(main, "GraphDatabase", recording)

    driver = main._create_neo4j_driver(app_config)

    assert isinstance(driver, _DummyDriver)
    assert driver is not recording.first_driver
    assert driver.verified is True
    assert driver.executed is True
    assert recording.calls == [
        "neo4j://example.com:7687",
        "bolt://example.com:7687",
    ]
    assert recording.first_driver.closed is True


def test_require_graph_service_rebuilds_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The graph view service is recreated when a driver becomes available."""

    app = FastAPI()
    config = load_config()
    app.state.app_config = config
    app.state.graph_view_service = None
    app.state.neo4j_driver = None

    created: Dict[str, Any] = {}

    def _fake_create_driver(received_config: Any) -> _DummyDriver:
        created["config"] = received_config
        return _DummyDriver()

    def _fake_build_service(driver: Any) -> str:
        created["driver"] = driver
        return "service"

    monkeypatch.setattr(main, "_create_neo4j_driver", _fake_create_driver)
    monkeypatch.setattr(main, "_build_graph_view_service", _fake_build_service)

    service = main._require_graph_service(app)

    assert service == "service"
    assert app.state.graph_view_service == "service"
    assert app.state.neo4j_driver is created["driver"]
    assert created["config"] is config
