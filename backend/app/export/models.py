"""Data models supporting shareable export links."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    """Base class for immutable Pydantic models."""

    model_config = ConfigDict(frozen=True)


class ShareExportFilters(_FrozenModel):
    """Filters applied when querying the graph for export."""

    min_confidence: float = Field(..., ge=0.0, le=1.0)
    relations: tuple[str, ...] = Field(default_factory=tuple)
    sections: tuple[str, ...] = Field(default_factory=tuple)
    papers: tuple[str, ...] = Field(default_factory=tuple)
    include_co_mentions: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the filters."""

        return {
            "min_confidence": self.min_confidence,
            "relations": list(self.relations),
            "sections": list(self.sections),
            "papers": list(self.papers),
            "include_co_mentions": self.include_co_mentions,
        }


class ShareExportRequest(_FrozenModel):
    """Request payload for generating a shareable export bundle."""

    filters: ShareExportFilters
    include_snippets: bool = True
    truncate_snippets: bool = False
    requested_by: str = Field(..., min_length=1)
    pipeline_version: str = Field(..., min_length=1)
    estimated_size_bytes: Optional[int] = Field(default=None, ge=0)
    allowed_papers: Optional[tuple[str, ...]] = Field(default=None)
    run_id: Optional[str] = Field(default=None, min_length=1)


class ShareExportResponse(_FrozenModel):
    """Response returned after creating a shareable link."""

    token: str
    expires_at: Optional[datetime]
    bundle_size_mb: float
    warning: bool
    pipeline_version: str
    metadata_id: str


@dataclass(frozen=True)
class ExportBundle:
    """Bundle contents prepared for upload."""

    archive_path: Path
    size_bytes: int
    sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class StoredBundle:
    """Details of a bundle persisted to object storage."""

    object_key: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ShareMetadataRecord:
    """Metadata persisted for issued share tokens."""

    metadata_id: str
    object_key: str
    sha256: str
    pipeline_version: str
    filters: Mapping[str, object]
    created_at: datetime
    expires_at: Optional[datetime]
    requested_by: str
    size_bytes: int
    warning: bool
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    run_id: Optional[str] = None
    first_download_at: Optional[datetime] = None
    first_download_latency_seconds: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        """Convert record to a dictionary for persistence."""

        payload: Dict[str, object] = asdict(self)
        payload["filters"] = dict(self.filters)
        payload["created_at"] = self.created_at.isoformat()
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at.isoformat()
        else:
            payload.pop("expires_at", None)
        if self.revoked_at is not None:
            payload["revoked_at"] = self.revoked_at.isoformat()
        else:
            payload.pop("revoked_at", None)
        if self.revoked_by is None:
            payload.pop("revoked_by", None)
        if self.run_id is None:
            payload.pop("run_id", None)
        if self.first_download_at is not None:
            payload["first_download_at"] = self.first_download_at.isoformat()
        else:
            payload.pop("first_download_at", None)
        if self.first_download_latency_seconds is None:
            payload.pop("first_download_latency_seconds", None)
        return payload


class ShareMetadataRepositoryProtocol:
    """Protocol for persisting share metadata records."""

    def persist(self, record: Mapping[str, object]) -> None:
        """Persist metadata associated with a share link."""

    def fetch(self, metadata_id: str) -> Optional[Mapping[str, object]]:
        """Retrieve share metadata by identifier."""

    def update(self, metadata_id: str, changes: Mapping[str, object]) -> None:
        """Apply partial updates to an existing metadata record."""
