"""Service responsible for creating shareable export links."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from backend.app.config import ExportConfig
from backend.app.export.models import (
    ShareExportRequest,
    ShareExportResponse,
    ShareMetadataRecord,
    ShareMetadataRepositoryProtocol,
)
from backend.app.export.storage import BundleStorageProtocol
from backend.app.export.token import ShareTokenManager

LOGGER = logging.getLogger(__name__)


class ExportSizeLimitError(RuntimeError):
    """Raised when the bundle exceeds configured size limits."""


class ShareExportService:
    """Coordinate bundle generation, storage, metadata, and token issuance."""

    def __init__(
        self,
        *,
        bundle_builder,
        storage: BundleStorageProtocol,
        metadata_repository: ShareMetadataRepositoryProtocol,
        token_manager: ShareTokenManager,
        config: ExportConfig,
        clock: Callable[[], datetime],
    ) -> None:
        self._bundle_builder = bundle_builder
        self._storage = storage
        self._metadata_repository = metadata_repository
        self._token_manager = token_manager
        self._config = config
        self._clock = clock

    def create_share(self, request: ShareExportRequest) -> ShareExportResponse:
        """Create a shareable link for the provided export request."""

        bundle = self._bundle_builder.build(request)
        size_mb = bundle.size_bytes / 1_000_000
        warning = size_mb > self._config.warn_bundle_mb
        if size_mb > self._config.max_bundle_mb:
            msg = (
                f"Bundle size {size_mb:.2f} MB exceeds limit "
                f"{self._config.max_bundle_mb} MB"
            )
            LOGGER.warning(msg)
            raise ExportSizeLimitError(msg)

        stored = self._storage.store(bundle)
        metadata_id = self._metadata_id()
        issued_token = self._token_manager.issue(
            metadata_id=metadata_id,
            ttl_hours=self._config.link_ttl_hours,
        )
        created_at = self._ensure_timezone(self._clock())
        record = ShareMetadataRecord(
            metadata_id=metadata_id,
            object_key=stored.object_key,
            sha256=stored.sha256,
            pipeline_version=str(bundle.manifest.get("pipeline_version")),
            filters=request.filters.to_dict(),
            created_at=created_at,
            expires_at=issued_token.expires_at,
            requested_by=request.requested_by,
            size_bytes=stored.size_bytes,
            warning=warning,
        )
        try:
            self._metadata_repository.persist(record.as_dict())
        except Exception:  # pragma: no cover - metadata persistence failure logged
            LOGGER.exception("Failed to persist share metadata for %s", metadata_id)
            raise

        response = ShareExportResponse(
            token=issued_token.token,
            expires_at=issued_token.expires_at,
            bundle_size_mb=size_mb,
            warning=warning,
            pipeline_version=record.pipeline_version,
            metadata_id=metadata_id,
        )
        return response

    def revoke_share(self, metadata_id: str, *, revoked_by: str) -> bool:
        """Mark an existing share link as revoked.

        Args:
            metadata_id: Identifier of the metadata record to revoke.
            revoked_by: User or system identifier performing the revocation.

        Returns:
            bool: ``True`` if the record was updated, ``False`` if already revoked.

        Raises:
            KeyError: If the metadata record cannot be found.
        """

        record = self._metadata_repository.fetch(metadata_id)
        if record is None:
            raise KeyError(f"Share metadata {metadata_id} not found")
        if record.get("revoked_at"):
            return False
        revoked_at = self._ensure_timezone(self._clock())
        changes = {
            "revoked_at": revoked_at.isoformat(),
            "revoked_by": revoked_by,
        }
        self._metadata_repository.update(metadata_id, changes)
        return True

    @staticmethod
    def _metadata_id() -> str:
        return uuid4().hex

    @staticmethod
    def _ensure_timezone(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
