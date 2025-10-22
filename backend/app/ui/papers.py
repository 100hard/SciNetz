"""Paper registry utilities for tracking uploads and processing status."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional, Sequence

from backend.app.auth.enums import UserRole
from backend.app.contracts import PaperMetadata


class PaperStatus(str, Enum):
    """Discrete lifecycle states for uploaded papers."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PaperRecord:
    """Metadata describing a tracked paper upload."""

    paper_id: str
    filename: str
    pdf_path: Path
    status: PaperStatus
    uploaded_at: datetime
    updated_at: datetime
    metadata: Optional[PaperMetadata] = None
    errors: List[str] = field(default_factory=list)
    nodes_written: int = 0
    edges_written: int = 0
    co_mention_edges: int = 0
    owner_id: Optional[str] = None
    shared_with: List[str] = field(default_factory=list)
    is_public: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable payload for API responses."""

        payload: Dict[str, object] = {
            "paper_id": self.paper_id,
            "filename": self.filename,
            "status": self.status.value,
            "uploaded_at": self.uploaded_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "errors": list(self.errors),
            "nodes_written": self.nodes_written,
            "edges_written": self.edges_written,
            "co_mention_edges": self.co_mention_edges,
            "owner_id": self.owner_id,
            "shared_with": list(self.shared_with),
            "is_public": self.is_public,
        }
        if self.metadata is not None:
            payload["metadata"] = self.metadata.model_dump()
        return payload

    def is_accessible_by(self, user_id: str, role: UserRole) -> bool:
        """Return True if the supplied user can access the record."""

        if role is UserRole.ADMIN:
            return True
        if self.is_public:
            return True
        if self.owner_id is None:
            return False
        if self.owner_id == user_id:
            return True
        return user_id in self.shared_with

    def with_status(
        self,
        status: PaperStatus,
        *,
        metadata: Optional[PaperMetadata] = None,
        errors: Optional[Iterable[str]] = None,
        nodes_written: Optional[int] = None,
        edges_written: Optional[int] = None,
        co_mention_edges: Optional[int] = None,
        ) -> "PaperRecord":
        """Create a copy with updated status and derived fields."""

        updated = datetime.now(timezone.utc)
        return PaperRecord(
            paper_id=self.paper_id,
            filename=self.filename,
            pdf_path=self.pdf_path,
            status=status,
            uploaded_at=self.uploaded_at,
            updated_at=updated,
            metadata=metadata if metadata is not None else self.metadata,
            errors=list(errors or self.errors),
            nodes_written=nodes_written if nodes_written is not None else self.nodes_written,
            edges_written=edges_written if edges_written is not None else self.edges_written,
            co_mention_edges=(
                co_mention_edges if co_mention_edges is not None else self.co_mention_edges
            ),
            owner_id=self.owner_id,
            shared_with=list(self.shared_with),
            is_public=self.is_public,
        )


class PaperRegistry:
    """Persistent registry tracking uploaded paper lifecycle."""

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path
        self._lock = RLock()
        self._records = self._load()

    def register_upload(
        self,
        paper_id: str,
        filename: str,
        pdf_path: Path,
        owner_id: str,
        *,
        shared_with: Optional[Iterable[str]] = None,
        is_public: bool = False,
    ) -> PaperRecord:
        """Record a new upload and persist the registry."""

        now = datetime.now(timezone.utc)
        normalized_owner = owner_id.strip()
        if not normalized_owner:
            raise ValueError("Owner ID cannot be empty")
        record = PaperRecord(
            paper_id=paper_id,
            filename=filename,
            pdf_path=pdf_path,
            status=PaperStatus.UPLOADED,
            uploaded_at=now,
            updated_at=now,
            owner_id=normalized_owner,
            shared_with=self._normalise_user_ids(shared_with),
            is_public=bool(is_public),
        )
        with self._lock:
            self._records[paper_id] = record
            self._persist()
        return record

    def mark_processing(self, paper_id: str) -> Optional[PaperRecord]:
        """Update a record to the processing state if it exists."""

        return self._update(paper_id, PaperStatus.PROCESSING)

    def mark_complete(
        self,
        paper_id: str,
        *,
        metadata: PaperMetadata,
        nodes_written: int,
        edges_written: int,
        co_mention_edges: int,
        errors: Optional[Iterable[str]] = None,
    ) -> Optional[PaperRecord]:
        """Mark a paper as complete with extracted metadata and stats."""

        return self._update(
            paper_id,
            PaperStatus.COMPLETE,
            metadata=metadata,
            errors=errors,
            nodes_written=nodes_written,
            edges_written=edges_written,
            co_mention_edges=co_mention_edges,
        )

    def mark_failed(self, paper_id: str, errors: Iterable[str]) -> Optional[PaperRecord]:
        """Mark a paper as failed with captured errors."""

        return self._update(paper_id, PaperStatus.FAILED, errors=list(errors))

    def get(self, paper_id: str) -> Optional[PaperRecord]:
        """Return the record for the supplied paper identifier."""

        with self._lock:
            return self._records.get(paper_id)

    def get_for_user(self, paper_id: str, user_id: str, role: UserRole) -> Optional[PaperRecord]:
        """Return a paper when the user is authorised to access it."""

        with self._lock:
            record = self._records.get(paper_id)
            if record is None:
                return None
            return record if record.is_accessible_by(user_id, role) else None

    def list_records(self) -> List[PaperRecord]:
        """Return all records ordered by upload time (most recent first)."""

        with self._lock:
            return sorted(self._records.values(), key=lambda rec: rec.uploaded_at, reverse=True)

    def list_records_for_user(self, user_id: str, role: UserRole) -> List[PaperRecord]:
        """Return accessible records for the supplied user."""

        with self._lock:
            if role is UserRole.ADMIN:
                candidates = self._records.values()
            else:
                candidates = [
                    record
                    for record in self._records.values()
                    if record.is_accessible_by(user_id, role)
                ]
            return sorted(candidates, key=lambda rec: rec.uploaded_at, reverse=True)

    def accessible_paper_ids(self, user_id: str, role: UserRole) -> List[str]:
        """Return identifiers for papers the user may access."""

        return [record.paper_id for record in self.list_records_for_user(user_id, role)]

    def update_access(
        self,
        paper_id: str,
        *,
        shared_with: Optional[Iterable[str]] = None,
        is_public: Optional[bool] = None,
    ) -> Optional[PaperRecord]:
        """Update the access control lists for the supplied paper."""

        with self._lock:
            record = self._records.get(paper_id)
            if record is None:
                return None
            new_shared = (
                self._normalise_user_ids(shared_with)
                if shared_with is not None
                else list(record.shared_with)
            )
            new_public = bool(is_public) if is_public is not None else record.is_public
            updated = replace(
                record,
                updated_at=datetime.now(timezone.utc),
                shared_with=new_shared,
                is_public=new_public,
            )
            self._records[paper_id] = updated
            self._persist()
            return updated

    def _update(
        self,
        paper_id: str,
        status: PaperStatus,
        *,
        metadata: Optional[PaperMetadata] = None,
        errors: Optional[Iterable[str]] = None,
        nodes_written: Optional[int] = None,
        edges_written: Optional[int] = None,
        co_mention_edges: Optional[int] = None,
    ) -> Optional[PaperRecord]:
        with self._lock:
            current = self._records.get(paper_id)
            if current is None:
                return None
            updated = current.with_status(
                status,
                metadata=metadata,
                errors=errors,
                nodes_written=nodes_written,
                edges_written=edges_written,
                co_mention_edges=co_mention_edges,
            )
            self._records[paper_id] = updated
            self._persist()
            return updated

    @staticmethod
    def _normalise_user_ids(user_ids: Optional[Iterable[str]]) -> List[str]:
        if user_ids is None:
            return []
        if isinstance(user_ids, (str, bytes)):
            candidates = [str(user_ids)]
        else:
            candidates = [str(candidate) for candidate in user_ids]
        seen: List[str] = []
        for candidate in candidates:
            trimmed = candidate.strip()
            if not trimmed or trimmed in seen:
                continue
            seen.append(trimmed)
        return seen

    def _load(self) -> Dict[str, PaperRecord]:
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        records: Dict[str, PaperRecord] = {}
        for paper_id, payload in data.items():
            record = self._deserialize_record(str(paper_id), payload)
            if record is not None:
                records[record.paper_id] = record
        return records

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {
            paper_id: self._serialise_record(record)
            for paper_id, record in self._records.items()
        }
        self._path.write_text(json.dumps(serialised, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _serialise_record(record: PaperRecord) -> Dict[str, object]:
        payload = record.to_dict()
        payload["pdf_path"] = str(record.pdf_path)
        return payload

    @staticmethod
    def _deserialize_record(paper_id: str, payload: object) -> Optional[PaperRecord]:
        if not isinstance(payload, dict):
            return None
        filename = payload.get("filename")
        pdf_path = payload.get("pdf_path")
        status = payload.get("status")
        uploaded_at = payload.get("uploaded_at")
        updated_at = payload.get("updated_at")
        if not all(isinstance(value, str) for value in (filename, pdf_path, status, uploaded_at, updated_at)):
            return None
        try:
            parsed_status = PaperStatus(status)
        except ValueError:
            return None
        try:
            uploaded = datetime.fromisoformat(uploaded_at)
            updated = datetime.fromisoformat(updated_at)
        except ValueError:
            uploaded = datetime.now(timezone.utc)
            updated = uploaded
        metadata_payload = payload.get("metadata")
        metadata = None
        if isinstance(metadata_payload, dict):
            try:
                metadata = PaperMetadata(**metadata_payload)
            except Exception:
                metadata = None
        errors_raw = payload.get("errors")
        errors: List[str] = []
        if isinstance(errors_raw, list):
            errors = [str(item) for item in errors_raw]
        nodes_written = int(payload.get("nodes_written", 0) or 0)
        edges_written = int(payload.get("edges_written", 0) or 0)
        co_mention_edges = int(payload.get("co_mention_edges", 0) or 0)
        owner_raw = payload.get("owner_id")
        owner_id: Optional[str]
        if isinstance(owner_raw, str):
            owner_id = owner_raw.strip() or None
        elif owner_raw is None:
            owner_id = None
        else:
            owner_id = str(owner_raw).strip() or None
        shared_raw = payload.get("shared_with")
        shared_with: List[str]
        if isinstance(shared_raw, list):
            shared_with = PaperRegistry._normalise_user_ids(shared_raw)
        elif isinstance(shared_raw, (str, bytes)):
            shared_with = PaperRegistry._normalise_user_ids([shared_raw])
        else:
            shared_with = []
        is_public = bool(payload.get("is_public", False))
        return PaperRecord(
            paper_id=paper_id,
            filename=filename,
            pdf_path=Path(pdf_path),
            status=parsed_status,
            uploaded_at=uploaded,
            updated_at=updated,
            metadata=metadata,
            errors=errors,
            nodes_written=nodes_written,
            edges_written=edges_written,
            co_mention_edges=co_mention_edges,
            owner_id=owner_id,
            shared_with=shared_with,
            is_public=is_public,
        )

