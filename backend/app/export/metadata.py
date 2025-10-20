"""Persistence helpers for shareable export metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, MutableSequence

from backend.app.export.models import ShareMetadataRepositoryProtocol

LOGGER = logging.getLogger(__name__)


class JSONShareMetadataRepository(ShareMetadataRepositoryProtocol):
    """Persist share metadata records to a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def persist(self, record: Mapping[str, object]) -> None:
        """Append the record to the JSON repository."""

        try:
            existing = self._load()
            existing.append(dict(record))
            self._path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to persist share metadata to %s", self._path)
            raise

    def fetch(self, metadata_id: str) -> Mapping[str, object] | None:
        """Return metadata entry for the provided identifier."""

        for entry in self._load():
            if str(entry.get("metadata_id")) == metadata_id:
                return dict(entry)
        return None

    def update(self, metadata_id: str, changes: Mapping[str, object]) -> None:
        """Apply partial updates to a metadata entry and persist to disk."""

        data = self._load()
        updated = False
        for index, entry in enumerate(data):
            if str(entry.get("metadata_id")) == metadata_id:
                merged = dict(entry)
                merged.update(changes)
                data[index] = merged
                updated = True
                break
        if not updated:
            raise KeyError(f"Share metadata {metadata_id} not found")
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> MutableSequence[Mapping[str, object]]:
        if not self._path.exists():
            return []
        try:
            content = self._path.read_text(encoding="utf-8")
            if not content.strip():
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return list(data)
            LOGGER.warning("Unexpected metadata repository format at %s", self._path)
            return []
        except json.JSONDecodeError:
            LOGGER.warning("Corrupt metadata repository at %s; resetting file", self._path)
            return []
