"""Tests for the API health check helper."""
from __future__ import annotations

import httpx

from backend.app.utils.api_health import APIHealthResult, check_api_health


def test_check_api_health_success() -> None:
    """A 200 response should yield a successful result with payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    result = check_api_health("http://example.com", client=client)

    client.close()

    assert result == APIHealthResult(
        ok=True,
        status_code=200,
        detail="API health check succeeded",
        latency_ms=result.latency_ms,
        payload={"status": "ok"},
    )
    assert result.latency_ms is not None


def test_check_api_health_failure_status() -> None:
    """A non-200 response should be treated as a failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    result = check_api_health("http://example.com", client=client)

    client.close()

    assert not result.ok
    assert result.status_code == 503
    assert "Health endpoint returned" in result.detail


def test_check_api_health_network_error() -> None:
    """Network errors should yield a failure with explanatory detail."""

    class ErrorTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
            raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=ErrorTransport())

    result = check_api_health("http://example.com", client=client)

    client.close()

    assert not result.ok
    assert result.status_code is None
    assert "connection refused" in result.detail
