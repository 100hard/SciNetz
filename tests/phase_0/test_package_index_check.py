"""Unit tests for the package index availability checker."""

from __future__ import annotations

import urllib.error

import pytest

from scripts.check_package_index import main, verify_package_index


def test_verify_package_index_accepts_valid_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verifier should succeed when the index responds with <400."""

    class _MockResponse:
        def __init__(self) -> None:
            self.status = 200

        def __enter__(self) -> "_MockResponse":
            return self

        def __exit__(self, *args: object) -> None:  # noqa: D401 - context manager contract
            return None

    def _mock_urlopen(request: object, timeout: float) -> _MockResponse:  # noqa: ARG001
        return _MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen)

    verify_package_index("https://example.com/simple")


def test_verify_package_index_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response with status >=400 should trigger a RuntimeError."""

    class _MockResponse:
        def __init__(self) -> None:
            self.status = 503

        def __enter__(self) -> "_MockResponse":
            return self

        def __exit__(self, *args: object) -> None:  # noqa: D401 - context manager contract
            return None

    def _mock_urlopen(request: object, timeout: float) -> _MockResponse:  # noqa: ARG001
        return _MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen)

    with pytest.raises(RuntimeError):
        verify_package_index("https://example.com/simple")


def test_verify_package_index_handles_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network failures should surface as RuntimeError for caller context."""

    def _raising_urlopen(request: object, timeout: float) -> None:  # noqa: ARG001
        raise urllib.error.URLError("forbidden")

    monkeypatch.setattr("urllib.request.urlopen", _raising_urlopen)

    with pytest.raises(RuntimeError):
        verify_package_index("https://example.com/simple")


def test_main_reports_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exit code should be 1 with an informative message on failure."""

    def _mock_verify(url: str, timeout: float = 5.0) -> None:  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.check_package_index.verify_package_index", _mock_verify)

    exit_code = main(["--url", "https://example.com/simple"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unable to reach the Python package index" in captured.err
    assert "boom" in captured.err


def test_main_succeeds(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Successful verification should yield a zero exit code and no stderr output."""

    def _mock_verify(url: str, timeout: float = 5.0) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr("scripts.check_package_index.verify_package_index", _mock_verify)

    exit_code = main(["--url", "https://example.com/simple"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
