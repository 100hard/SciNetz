from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from backend.app.config import ExportConfig, ExportStorageConfig
from backend.app.export.models import ShareExportFilters, ShareExportRequest, StoredBundle
from backend.app.export.service import (
    ExportSizeLimitError,
    ShareExportResponse,
    ShareExportService,
)
from backend.app.export.token import ShareTokenManager


@dataclass
class _FakeBundle:
    archive_path: Path
    size_bytes: int
    sha256: str = "deadbeef"
    manifest: dict = None


class _FakeBundleBuilder:
    def __init__(self, bundle: _FakeBundle) -> None:
        self.bundle = bundle
        self.called_with: Optional[ShareExportRequest] = None

    def build(self, request: ShareExportRequest) -> _FakeBundle:
        self.called_with = request
        return self.bundle


class _FakeStorage:
    def __init__(self) -> None:
        self.stored: Optional[_FakeBundle] = None

    def store(self, bundle: _FakeBundle) -> StoredBundle:
        self.stored = bundle
        return StoredBundle(
            object_key="exports/export.zip",
            size_bytes=bundle.size_bytes,
            sha256=bundle.sha256,
        )


class _FakeMetadataRepo:
    def __init__(self) -> None:
        self.persisted = []

    def persist(self, metadata) -> None:  # pragma: no cover - interface placeholder
        self.persisted.append(metadata)

    def fetch(self, metadata_id: str):  # pragma: no cover - interface placeholder
        for entry in self.persisted:
            if entry.get("metadata_id") == metadata_id:
                return entry
        return None

    def update(self, metadata_id: str, changes) -> None:  # pragma: no cover - interface placeholder
        for index, entry in enumerate(self.persisted):
            if entry.get("metadata_id") == metadata_id:
                updated = dict(entry)
                updated.update(changes)
                self.persisted[index] = updated
                return
        raise KeyError(metadata_id)


def _config(link_ttl_hours: Optional[int] = 24) -> ExportConfig:
    storage = ExportStorageConfig(
        bucket="scinets-test-exports",
        region="us-east-1",
        prefix="exports",
    )
    return ExportConfig(
        max_bundle_mb=5,
        warn_bundle_mb=3,
        snippet_truncate_length=200,
        link_ttl_hours=link_ttl_hours,
        signed_url_ttl_minutes=10,
        storage=storage,
    )


def _request(size_bytes: int = 1_000_000) -> ShareExportRequest:
    filters = ShareExportFilters(
        min_confidence=0.5,
        relations=["uses"],
        sections=["Results"],
        papers=["paper-1"],
        include_co_mentions=False,
    )
    return ShareExportRequest(
        filters=filters,
        include_snippets=True,
        truncate_snippets=False,
        requested_by="tests",
        pipeline_version="1.0.0",
        estimated_size_bytes=size_bytes,
    )


def test_guardrail_warns_when_threshold_exceeded(tmp_path) -> None:
    bundle = _FakeBundle(
        archive_path=tmp_path / "bundle.zip",
        size_bytes=3_500_000,
        manifest={"pipeline_version": "1.0.0"},
    )
    service = ShareExportService(
        bundle_builder=_FakeBundleBuilder(bundle),
        storage=_FakeStorage(),
        metadata_repository=_FakeMetadataRepo(),
        token_manager=ShareTokenManager(secret_key="secret"),
        config=_config(),
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    response = service.create_share(_request(size_bytes=bundle.size_bytes))

    assert isinstance(response, ShareExportResponse)
    assert response.warning is True
    assert response.bundle_size_mb == pytest.approx(bundle.size_bytes / 1_000_000, rel=1e-6)
    assert response.token
    assert response.metadata_id


def test_guardrail_block_when_exceeds_max(tmp_path) -> None:
    bundle = _FakeBundle(
        archive_path=tmp_path / "bundle.zip",
        size_bytes=7_000_000,
        manifest={"pipeline_version": "1.0.0"},
    )
    service = ShareExportService(
        bundle_builder=_FakeBundleBuilder(bundle),
        storage=_FakeStorage(),
        metadata_repository=_FakeMetadataRepo(),
        token_manager=ShareTokenManager(secret_key="secret"),
        config=_config(),
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(ExportSizeLimitError):
        service.create_share(_request(size_bytes=bundle.size_bytes))


def test_token_expiry_respects_config(tmp_path) -> None:
    now = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    bundle = _FakeBundle(
        archive_path=tmp_path / "bundle.zip",
        size_bytes=2_000_000,
        manifest={"pipeline_version": "1.0.0"},
    )
    manager = ShareTokenManager(secret_key="secret", clock=lambda: now)
    service = ShareExportService(
        bundle_builder=_FakeBundleBuilder(bundle),
        storage=_FakeStorage(),
        metadata_repository=_FakeMetadataRepo(),
        token_manager=manager,
        config=_config(),
        clock=lambda: now,
    )

    response = service.create_share(_request(size_bytes=bundle.size_bytes))

    config = _config()
    assert config.link_ttl_hours is not None
    expected_expiry = now + timedelta(hours=config.link_ttl_hours)
    assert response.expires_at == expected_expiry
    decoded = manager.decode(response.token, current_time=now)
    assert decoded.expires_at == expected_expiry


def test_revoke_share_sets_metadata(tmp_path) -> None:
    bundle = _FakeBundle(
        archive_path=tmp_path / "bundle.zip",
        size_bytes=2_000_000,
        manifest={"pipeline_version": "1.0.0"},
    )
    repo = _FakeMetadataRepo()
    service = ShareExportService(
        bundle_builder=_FakeBundleBuilder(bundle),
        storage=_FakeStorage(),
        metadata_repository=repo,
        token_manager=ShareTokenManager(secret_key="secret"),
        config=_config(),
        clock=lambda: datetime(2025, 2, 2, tzinfo=timezone.utc),
    )

    response = service.create_share(_request(size_bytes=bundle.size_bytes))
    metadata_id = response.metadata_id
    assert repo.fetch(metadata_id).get("revoked_at") is None

    updated = service.revoke_share(metadata_id, revoked_by="tests")

    assert updated is True
    record = repo.fetch(metadata_id)
    assert record["revoked_by"] == "tests"
    assert record["revoked_at"].endswith("+00:00")
    # Second revocation is a no-op
    updated_again = service.revoke_share(metadata_id, revoked_by="tests")
    assert updated_again is False


def test_permanent_links_do_not_set_expiry(tmp_path) -> None:
    bundle = _FakeBundle(
        archive_path=tmp_path / "bundle.zip",
        size_bytes=2_000_000,
        manifest={"pipeline_version": "1.0.0"},
    )
    repo = _FakeMetadataRepo()
    now = datetime(2025, 3, 3, tzinfo=timezone.utc)
    manager = ShareTokenManager(secret_key="secret", clock=lambda: now)
    service = ShareExportService(
        bundle_builder=_FakeBundleBuilder(bundle),
        storage=_FakeStorage(),
        metadata_repository=repo,
        token_manager=manager,
        config=_config(link_ttl_hours=None),
        clock=lambda: now,
    )

    response = service.create_share(_request(size_bytes=bundle.size_bytes))

    assert response.expires_at is None
    decoded = manager.decode(response.token, current_time=now)
    assert decoded.expires_at is None

    stored = repo.fetch(response.metadata_id)
    assert stored is not None
    assert "expires_at" not in stored or stored["expires_at"] is None
