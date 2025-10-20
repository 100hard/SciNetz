from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi.testclient import TestClient

from backend.app.config import load_config
from backend.app.export.models import ShareExportResponse
from backend.app.export.token import ShareTokenManager
from backend.app.main import create_app


class _InMemoryMetadataRepo:
    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, object]] = {}

    def persist(self, record) -> None:
        metadata_id = str(record["metadata_id"])
        self._records[metadata_id] = dict(record)

    def fetch(self, metadata_id: str) -> Optional[Dict[str, object]]:
        if metadata_id not in self._records:
            return None
        return dict(self._records[metadata_id])

    def update(self, metadata_id: str, changes) -> None:
        if metadata_id not in self._records:
            raise KeyError(metadata_id)
        updated = dict(self._records[metadata_id])
        updated.update(changes)
        self._records[metadata_id] = updated


class _InMemoryObject:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def release_conn(self) -> None:  # pragma: no cover - compatibility shim
        return None


class _StubStorageClient:
    def __init__(self) -> None:
        self._archive_bytes: Optional[bytes] = None

    def presigned_get_object(self, bucket: str, object_key: str, expires: timedelta) -> str:
        ttl = int(expires.total_seconds())
        return f"https://downloads.example/{bucket}/{object_key}?ttl={ttl}"

    def get_object(self, bucket: str, object_key: str) -> _InMemoryObject:
        if self._archive_bytes is None:
            raise RuntimeError("Archive content not set")
        return _InMemoryObject(self._archive_bytes)

    def set_archive(self, archive_bytes: bytes) -> None:
        self._archive_bytes = archive_bytes


class _StubShareService:
    def __init__(
        self,
        token_manager: ShareTokenManager,
        repo: _InMemoryMetadataRepo,
        *,
        clock,
        ttl_hours: Optional[int] = 24,
    ) -> None:
        self._token_manager = token_manager
        self._repo = repo
        self._clock = clock
        self._ttl_hours = ttl_hours

    def create_share(self, request) -> ShareExportResponse:  # type: ignore[override]
        metadata_id = "meta-001"
        issued = self._token_manager.issue(metadata_id=metadata_id, ttl_hours=self._ttl_hours)
        created_at = self._clock()
        record = {
            "metadata_id": metadata_id,
            "object_key": "exports/export.zip",
            "sha256": "abc123",
            "pipeline_version": request.pipeline_version,
            "filters": request.filters.to_dict(),
            "created_at": created_at.isoformat(),
            "requested_by": request.requested_by,
            "size_bytes": 1_536_000,
            "warning": False,
        }
        if issued.expires_at is not None:
            record["expires_at"] = issued.expires_at.isoformat()
        self._repo.persist(record)
        return ShareExportResponse(
            token=issued.token,
            expires_at=issued.expires_at,
            bundle_size_mb=1.536,
            warning=False,
            pipeline_version=request.pipeline_version,
            metadata_id=metadata_id,
        )

    def revoke_share(self, metadata_id: str, *, revoked_by: str) -> bool:
        record = self._repo.fetch(metadata_id)
        if record is None:
            raise KeyError(metadata_id)
        if record.get("revoked_at"):
            return False
        revoked_at = self._clock().isoformat()
        self._repo.update(metadata_id, {"revoked_at": revoked_at, "revoked_by": revoked_by})
        return True


def _build_test_client(
    *,
    ttl_hours: Optional[int] = 24,
) -> tuple[TestClient, ShareTokenManager, _InMemoryMetadataRepo, _StubStorageClient]:
    config = load_config()
    app = create_app(config=config)
    token_manager = ShareTokenManager(secret_key="secret-key")
    metadata_repo = _InMemoryMetadataRepo()
    storage_client = _StubStorageClient()
    app.state.share_export_service = _StubShareService(
        token_manager,
        metadata_repo,
        clock=lambda: datetime.now(timezone.utc),
        ttl_hours=ttl_hours,
    )
    app.state.share_token_manager = token_manager
    app.state.share_metadata_repository = metadata_repo
    app.state.share_storage_client = storage_client
    app.state.share_storage_reader = storage_client
    client = TestClient(app)
    return client, token_manager, metadata_repo, storage_client


def _make_graph_archive() -> bytes:
    payload = {
        "nodes": [
            {
                "id": "n1",
                "label": "Node A",
                "type": "Method",
                "aliases": ["A"],
                "times_seen": 2,
                "section_distribution": {"Results": 1},
                "source_document_ids": ["doc-1"],
            }
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n1",
                "target": "n1",
                "relation": "uses",
                "relation_verbatim": "uses",
                "confidence": 0.9,
                "times_seen": 1,
                "attributes": {"section": "Results"},
                "evidence": {"doc_id": "doc-1"},
                "conflicting": False,
            }
        ],
        "node_count": 1,
        "edge_count": 1,
        "pipeline_version": "1.0.0",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("graph.json", json.dumps(payload).encode("utf-8"))
        archive.writestr("scinets_view.html", "<html></html>")
        archive.writestr("README_export.md", "# Export")
        archive.writestr("manifest.json", json.dumps({"pipeline_version": "1.0.0"}))
    return buffer.getvalue()


def test_share_link_create_resolve_and_revoke_flow() -> None:
    client, _, metadata_repo, storage = _build_test_client()

    payload = {
        "filters": {
            "min_confidence": 0.5,
            "relations": ["uses"],
            "sections": ["Results"],
            "papers": [],
            "include_co_mentions": False,
        },
        "include_snippets": True,
        "truncate_snippets": False,
        "requested_by": "tests",
    }
    create_response = client.post("/api/export/share", json=payload)
    assert create_response.status_code == 200
    created = create_response.json()
    token = created["token"]
    metadata_id = created["metadata_id"]
    assert metadata_id == "meta-001"
    stored = metadata_repo.fetch(metadata_id)
    assert stored is not None

    storage.set_archive(_make_graph_archive())

    resolve_before = client.get(
        f"/api/export/share/{token}",
        headers={"accept": "application/json"},
    )
    assert resolve_before.status_code == 200
    resolved = resolve_before.json()
    assert resolved["download_url"].startswith("https://downloads.example/")

    revoke_response = client.post(f"/api/export/share/{metadata_id}/revoke", json={"revoked_by": "tester"})
    assert revoke_response.status_code == 200
    revoked_record = metadata_repo.fetch(metadata_id)
    assert revoked_record is not None and revoked_record["revoked_by"] == "tester"

    resolve_after = client.get(f"/api/export/share/{token}", headers={"accept": "application/json"})
    assert resolve_after.status_code == 410
    assert resolve_after.json()["detail"] == "Share link revoked"


def test_share_link_html_rendering() -> None:
    client, _, _, storage = _build_test_client()
    payload = {
        "filters": {
            "min_confidence": 0.5,
            "relations": ["uses"],
            "sections": ["Results"],
            "papers": [],
            "include_co_mentions": False,
        },
        "include_snippets": True,
        "truncate_snippets": False,
        "requested_by": "tests",
    }
    create_response = client.post("/api/export/share", json=payload)
    token = create_response.json()["token"]
    storage.set_archive(_make_graph_archive())

    html_response = client.get(
        f"/api/export/share/{token}",
        headers={"accept": "text/html"},
    )
    assert html_response.status_code == 200
    assert "text/html" in html_response.headers["content-type"]
    body = html_response.text
    assert "SciNets Shared Graph" in body
    assert "Download bundle" in body


def test_share_link_json_response_handles_permanent() -> None:
    client, _, metadata_repo, storage = _build_test_client(ttl_hours=None)
    payload = {
        "filters": {
            "min_confidence": 0.5,
            "relations": ["uses"],
            "sections": ["Results"],
            "papers": [],
            "include_co_mentions": False,
        },
        "include_snippets": True,
        "truncate_snippets": False,
        "requested_by": "tests",
    }
    create_response = client.post("/api/export/share", json=payload)
    data = create_response.json()
    assert data["expires_at"] is None
    metadata_id = data["metadata_id"]
    stored = metadata_repo.fetch(metadata_id)
    assert stored is not None
    assert "expires_at" not in stored or stored["expires_at"] is None

    storage.set_archive(_make_graph_archive())
    resolve_response = client.get(f"/api/export/share/{data['token']}")
    assert resolve_response.status_code == 200
    resolved = resolve_response.json()
    assert resolved["expires_at"] is None

