"""Object storage interactions for export bundles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.app.export.models import ExportBundle, StoredBundle

LOGGER = logging.getLogger(__name__)


class BundleStorageProtocol(Protocol):
    """Protocol describing bundle storage behaviour."""

    def store(self, bundle: ExportBundle) -> StoredBundle:
        """Persist the given bundle and return storage metadata."""


@dataclass
class S3BundleStorage:
    """Upload bundles to an S3-compatible object store."""

    client: object
    bucket: str
    prefix: str

    def store(self, bundle: ExportBundle) -> StoredBundle:
        """Upload the bundle archive to object storage."""

        object_key = self._object_key(bundle.archive_path.name)
        try:
            self.client.fput_object(
                self.bucket,
                object_key,
                str(bundle.archive_path),
            )
        except Exception as exc:  # pragma: no cover - storage failures logged
            LOGGER.exception("Failed to upload export bundle to storage")
            raise
        return StoredBundle(
            object_key=object_key,
            size_bytes=bundle.size_bytes,
            sha256=bundle.sha256,
        )

    def _object_key(self, filename: str) -> str:
        prefix = self.prefix.strip("/")
        if not prefix:
            return filename
        return f"{prefix}/{filename}"
