from __future__ import annotations

from typing import Any, Dict

import pytest

from backend.app.config import load_config
from backend.app import main


class _DummyDriver:
    """Simple stand-in for the Neo4j driver object."""


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
    assert recording.calls["uri"] == config.graph.uri
    assert recording.calls["auth"] == (config.graph.username, config.graph.password)


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
