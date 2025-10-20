from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.app.export.token import (
    ExpiredTokenError,
    InvalidTokenError,
    ShareTokenManager,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def test_share_token_round_trip() -> None:
    manager = ShareTokenManager(secret_key="super-secret", clock=_utc_now)
    issued = manager.issue(metadata_id="bundle-123", ttl_hours=2)

    assert issued.metadata_id == "bundle-123"
    assert issued.expires_at > _utc_now()

    decoded = manager.decode(issued.token)
    assert decoded.metadata_id == "bundle-123"
    assert decoded.expires_at == issued.expires_at


def test_share_token_rejects_tampering() -> None:
    manager = ShareTokenManager(secret_key="another-secret", clock=_utc_now)
    issued = manager.issue(metadata_id="bundle-456", ttl_hours=1)

    with pytest.raises(InvalidTokenError):
        manager.decode(issued.token + "corrupted")


def test_share_token_expired() -> None:
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock = lambda: now  # noqa: E731
    manager = ShareTokenManager(secret_key="key", clock=clock)

    issued = manager.issue(metadata_id="bundle-789", ttl_hours=1)

    future = now + timedelta(hours=2)
    with pytest.raises(ExpiredTokenError):
        manager.decode(issued.token, current_time=future)


def test_share_token_permanent_links() -> None:
    manager = ShareTokenManager(secret_key="permanent-secret", clock=_utc_now)

    issued = manager.issue(metadata_id="bundle-permanent", ttl_hours=None)

    assert issued.metadata_id == "bundle-permanent"
    assert issued.expires_at is None

    decoded = manager.decode(issued.token)
    assert decoded.metadata_id == "bundle-permanent"
    assert decoded.expires_at is None
