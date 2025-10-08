"""Helpers for verifying API connectivity via the health endpoint."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class APIHealthResult:
    """Structured information about an API health probe result."""

    ok: bool
    status_code: Optional[int]
    detail: str
    latency_ms: Optional[float]
    payload: Optional[dict[str, Any]]


def check_api_health(
    base_url: str,
    *,
    timeout: float = 5.0,
    client: Optional[httpx.Client] = None,
) -> APIHealthResult:
    """Ping the API health endpoint and return a structured result.

    Args:
        base_url: Base URL where the API is hosted (e.g. ``"http://localhost:8000"``).
        timeout: Request timeout in seconds when creating an internal client.
        client: Optional pre-configured ``httpx.Client`` (useful for testing).

    Returns:
        APIHealthResult: Structured outcome describing whether the API responded
            successfully and any additional context about failures.
    """

    url = f"{base_url.rstrip('/')}/health"
    should_close = client is None
    session = client or httpx.Client(timeout=timeout)
    start_time = time.monotonic()

    try:
        response = session.get(url)
        latency_ms = (time.monotonic() - start_time) * 1000
        if response.status_code == httpx.codes.OK:
            try:
                payload = response.json()
            except ValueError:
                payload = None
                logger.warning("Health endpoint returned non-JSON payload", extra={"url": url})
            logger.info(
                "API health check succeeded",
                extra={"url": url, "status_code": response.status_code, "latency_ms": latency_ms},
            )
            return APIHealthResult(
                ok=True,
                status_code=response.status_code,
                detail="API health check succeeded",
                latency_ms=latency_ms,
                payload=payload,
            )

        logger.warning(
            "API health check failed with status",
            extra={
                "url": url,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "response_text": response.text,
            },
        )
        return APIHealthResult(
            ok=False,
            status_code=response.status_code,
            detail=f"Health endpoint returned {response.status_code}",
            latency_ms=latency_ms,
            payload=None,
        )
    except httpx.HTTPError as exc:  # pragma: no cover - network failures are environment dependent
        latency_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "API health check request raised an error",
            extra={"url": url, "latency_ms": latency_ms, "error": str(exc)},
        )
        return APIHealthResult(
            ok=False,
            status_code=None,
            detail=f"Request to {url} failed: {exc}",
            latency_ms=latency_ms,
            payload=None,
        )
    finally:
        if should_close:
            session.close()


__all__ = ["APIHealthResult", "check_api_health"]
