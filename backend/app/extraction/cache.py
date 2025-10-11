"""Persistent caches for extraction performance optimizations."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from backend.app.extraction.triplet_extraction import RawLLMTriple

LOGGER = logging.getLogger(__name__)


class _JSONCacheBase:
    """Lightweight JSON-backed cache with basic concurrency protection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed to load cache from %s; starting fresh", self._path)
            return {}
        if isinstance(payload, dict):
            return payload
        LOGGER.warning("Cache payload at %s was not a mapping; starting fresh", self._path)
        return {}

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.parent / f"{self._path.name}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, sort_keys=True)
        tmp_path.replace(self._path)


class TokenBudgetCache(_JSONCacheBase):
    """Persist token multiplier adjustments across extractor instances."""

    def get(self, key: str) -> Optional[int]:
        """Return a cached token multiplier for the supplied key."""

        with self._lock:
            value = self._data.get(key)
            if isinstance(value, int) and value >= 1:
                return value
            return None

    def set(self, key: str, multiplier: int) -> None:
        """Persist a token multiplier for subsequent runs."""

        if multiplier < 1:
            return
        with self._lock:
            current = self._data.get(key)
            if isinstance(current, int) and current == multiplier:
                return
            self._data[key] = int(multiplier)
            self._write()


class LLMResponseCache(_JSONCacheBase):
    """Store raw LLM responses for reuse when chunks are unchanged."""

    def get(self, key: str) -> Optional[List[RawLLMTriple]]:
        """Return cached triples if available."""

        with self._lock:
            payload = self._data.get(key)
            if not isinstance(payload, list):
                return None
            triples: List[RawLLMTriple] = []
            for item in payload:
                if not isinstance(item, dict):
                    LOGGER.debug("Skipping malformed cached triple entry for key %s", key)
                    return None
                try:
                    triples.append(RawLLMTriple(**item))
                except (TypeError, ValueError):
                    LOGGER.debug("Failed to deserialize cached triple for key %s", key)
                    return None
            return list(triples)

    def set(self, key: str, triples: Sequence[RawLLMTriple]) -> None:
        """Persist triples emitted by the LLM for future reuse."""

        serializable = [asdict(triple) for triple in triples]
        with self._lock:
            self._data[key] = serializable
            self._write()

    def prune(self, keys_to_keep: Iterable[str]) -> None:
        """Retain only the provided keys (best-effort)."""

        with self._lock:
            keep = set(keys_to_keep)
            removed = [key for key in list(self._data) if key not in keep]
            if not removed:
                return
            for key in removed:
                self._data.pop(key, None)
            self._write()


__all__ = ["LLMResponseCache", "TokenBudgetCache"]
