from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest
from testcontainers.minio import MinioContainer

from backend.app.config import ExportConfig, ExportStorageConfig
from backend.app.export.bundle import ExportBundleBuilder
from backend.app.export.models import ShareExportFilters, ShareExportRequest
from backend.app.export.service import ShareExportResponse, ShareExportService
from backend.app.export.storage import S3BundleStorage
from backend.app.export.token import ShareTokenManager
from backend.app.ui.service import GraphEdge, GraphNode, GraphView

docker = pytest.importorskip("docker")


def _require_docker() -> None:
    client: docker.DockerClient | None = None
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Docker daemon not available: {exc}", allow_module_level=True)
    finally:
        if client is not None:
            client.close()


_require_docker()


@dataclass
class _StaticGraphProvider:
    graph: GraphView

    def fetch(self, filters: ShareExportFilters) -> GraphView:  # pragma: no cover - simple pass-through
        return self.graph


class _InMemoryMetadataRepository:
    def __init__(self) -> None:
        self.saved: List[dict] = []

    def persist(self, metadata: dict) -> None:
        self.saved.append(metadata)

    def fetch(self, metadata_id: str):
        for entry in self.saved:
            if entry.get("metadata_id") == metadata_id:
                return entry
        return None

    def update(self, metadata_id: str, changes: dict) -> None:
        for index, entry in enumerate(self.saved):
            if entry.get("metadata_id") == metadata_id:
                merged = dict(entry)
                merged.update(changes)
                self.saved[index] = merged
                return
        raise KeyError(metadata_id)


def _graph() -> GraphView:
    node = GraphNode(
        id="n1",
        label="Transformer Model",
        type="Method",
        aliases=["Transformers"],
        times_seen=3,
        section_distribution={"Results": 2},
        source_document_ids=["doc-1"],
    )
    edge = GraphEdge(
        id="e1",
        source="n1",
        target="n1",
        relation="uses",
        relation_verbatim="uses",
        confidence=0.92,
        times_seen=1,
        attributes={"section": "Results"},
        evidence={
            "doc_id": "doc-1",
            "element_id": "el-1",
            "text_span": {"start": 10, "end": 42},
            "full_sentence": "The Transformer model uses self-attention.",
        },
        conflicting=False,
        created_at="2025-05-05T12:00:00Z",
    )
    return GraphView(nodes=[node], edges=[edge])


def _config(bucket: str, prefix: str) -> ExportConfig:
    storage = ExportStorageConfig(bucket=bucket, region="us-east-1", prefix=prefix)
    return ExportConfig(
        max_bundle_mb=10,
        warn_bundle_mb=5,
        snippet_truncate_length=200,
        link_ttl_hours=24,
        signed_url_ttl_minutes=5,
        storage=storage,
    )


@pytest.mark.integration
def test_share_flow_with_minio(tmp_path: Path) -> None:
    graph = _graph()
    provider = _StaticGraphProvider(graph=graph)
    now = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)

    with MinioContainer("quay.io/minio/minio:RELEASE.2024-08-03T04-33-23Z") as container:
        client = container.get_client()
        bucket_name = "scinets-test-exports"
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        bundle_builder = ExportBundleBuilder(
            graph_provider=provider,
            output_dir=tmp_path,
            pipeline_version="1.0.0",
            clock=lambda: now,
        )
        storage = S3BundleStorage(
            client=client,
            bucket=bucket_name,
            prefix="exports",
        )
        metadata_repo = _InMemoryMetadataRepository()
        service = ShareExportService(
            bundle_builder=bundle_builder,
            storage=storage,
            metadata_repository=metadata_repo,
            token_manager=ShareTokenManager(secret_key="secret", clock=lambda: now),
            config=_config(bucket=bucket_name, prefix="exports"),
            clock=lambda: now,
        )

        request = ShareExportRequest(
            filters=ShareExportFilters(
                min_confidence=0.5,
                relations=["uses"],
                sections=["Results"],
                papers=["doc-1"],
                include_co_mentions=False,
            ),
            include_snippets=True,
            truncate_snippets=False,
            requested_by="integration-tests",
            pipeline_version="1.0.0",
        )

        response = service.create_share(request)

        assert isinstance(response, ShareExportResponse)
        assert response.token
        assert response.warning is False
        assert response.bundle_size_mb > 0
        assert response.pipeline_version == "1.0.0"
        assert response.metadata_id

        assert metadata_repo.saved, "Metadata should be persisted"
        saved_entry = metadata_repo.saved[0]
        object_key = saved_entry["object_key"]

        downloaded = tmp_path / "downloaded.zip"
        client.fget_object(bucket_name, object_key, str(downloaded))
        assert downloaded.exists()

        with zipfile.ZipFile(downloaded, "r") as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names
            assert "graph.json" in names
            assert "graph.graphml" in names
            assert "README_export.md" in names
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["pipeline_version"] == "1.0.0"
            assert manifest["files"]["graph.json"]["sha256"]
            graph_json = json.loads(archive.read("graph.json"))
            assert len(graph_json["nodes"]) == 1
            assert graph_json["nodes"][0]["label"] == "Transformer Model"
            assert len(graph_json["edges"]) == 1

        service.revoke_share(response.metadata_id, revoked_by="integration-tests")
        revoked_entry = metadata_repo.fetch(response.metadata_id)
        assert revoked_entry["revoked_by"] == "integration-tests"
        assert revoked_entry["revoked_at"].endswith("+00:00")
