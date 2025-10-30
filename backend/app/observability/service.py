"""Utilities for persisting pipeline observability artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence
from uuid import uuid4

from backend.app.config import ObservabilityConfig

LOGGER = logging.getLogger(__name__)


def _ensure_timezone(value: datetime) -> datetime:
    """Return a timezone-aware datetime normalised to UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalise_payload(value: object) -> object:
    """Convert payload values into JSON serialisable primitives."""

    if isinstance(value, datetime):
        return _ensure_timezone(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise_payload(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_payload(item) for item in value]
    return value


@dataclass
class ObservabilityRun:
    """Mutable context capturing metrics for a single pipeline run."""

    _service: "ObservabilityService"
    run_id: str
    pipeline_version: str
    papers: Sequence[str]
    started_at: datetime
    documents: Dict[str, Dict[str, object]] = field(default_factory=dict)
    phase_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, object] = field(default_factory=dict)
    batch_errors: list[str] = field(default_factory=list)
    total_seconds: Optional[float] = None
    _completed: bool = field(default=False, init=False)

    def record_document(self, doc_id: str, metrics: Mapping[str, object], errors: Sequence[str] | None = None) -> None:
        """Merge per-document metrics and errors into the run manifest."""

        entry = self.documents.setdefault(doc_id, {"metrics": {}, "errors": []})
        entry_metrics: MutableMapping[str, object] = entry["metrics"]  # type: ignore[assignment]
        for key, value in metrics.items():
            entry_metrics[key] = value
        if errors:
            entry_errors: list[str] = entry["errors"]  # type: ignore[assignment]
            for error in errors:
                cleaned = str(error).strip()
                if not cleaned:
                    continue
                if cleaned not in entry_errors:
                    entry_errors.append(cleaned)

    def record_phase_metrics(self, phase: str, metrics: Mapping[str, float]) -> None:
        """Attach phase-level metrics (e.g. timings, counts) to the run manifest."""

        bucket = self.phase_metrics.setdefault(phase, {})
        for key, value in metrics.items():
            bucket[key] = float(value)

    def record_summary(self, values: Mapping[str, object]) -> None:
        """Persist aggregate metrics that summarise the run."""

        for key, value in values.items():
            self.summary[key] = value

    def record_batch_errors(self, errors: Iterable[str]) -> None:
        """Attach batch-level errors to the manifest, deduplicating entries."""

        merged = list(self.batch_errors)
        for error in errors:
            cleaned = str(error).strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        self.batch_errors = merged

    def set_total_seconds(self, duration: float) -> None:
        """Record the total orchestration wall-clock time."""

        if duration < 0:
            return
        self.total_seconds = duration

    def finalize(self) -> None:
        """Persist the run manifest to disk if not already written."""

        if self._completed:
            return
        completed_at = self._service.now()
        payload = {
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "papers": sorted({paper for paper in self.papers if paper}),
            "started_at": self.started_at,
            "completed_at": completed_at,
            "documents": self.documents,
            "phase_metrics": self.phase_metrics,
            "summary": self.summary,
            "batch_errors": self.batch_errors,
        }
        if self.total_seconds is not None:
            payload["total_seconds"] = self.total_seconds
        try:
            self._service._append_run_manifest(payload)
        except Exception:  # pragma: no cover - persistence failures should not crash callers
            LOGGER.exception("Failed to persist observability run manifest for %s", self.run_id)
        self._completed = True


class ObservabilityService:
    """Service responsible for persisting observability artifacts to disk."""

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        root_dir: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        base_root = root_dir or Path(__file__).resolve().parents[2]
        self._root = self._resolve_root(base_root, config.root_dir)
        self._run_manifests_path = self._root / config.run_manifests_filename
        self._export_events_path = self._root / config.export_events_filename
        self._qa_metrics_path = self._root / config.qa_metrics_filename
        self._kpi_history_path = self._root / config.kpi_history_filename
        for path in (
            self._run_manifests_path,
            self._export_events_path,
            self._qa_metrics_path,
            self._kpi_history_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

    def now(self) -> datetime:
        """Return the current timestamp in UTC."""

        return _ensure_timezone(self._clock())

    def start_run(self, *, papers: Sequence[str], pipeline_version: str) -> ObservabilityRun:
        """Create a new run context for the supplied paper batch."""

        run_id = uuid4().hex
        started_at = self.now()
        return ObservabilityRun(
            _service=self,
            run_id=run_id,
            pipeline_version=pipeline_version,
            papers=tuple(papers),
            started_at=started_at,
        )

    def record_export_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        """Append an export lifecycle event to the audit log."""

        enriched = dict(payload)
        enriched["event_type"] = event_type
        enriched.setdefault("timestamp", self.now())
        self._write_json_line(self._export_events_path, enriched)

    def record_qa_metrics(self, payload: Mapping[str, object]) -> None:
        """Persist QA latency and evidence metrics for dashboard visualisation."""

        enriched = dict(payload)
        enriched.setdefault("timestamp", self.now())
        self._write_json_line(self._qa_metrics_path, enriched)

    def persist_kpi_snapshot(self, payload: Mapping[str, object]) -> None:
        """Append a KPI snapshot to the historical trend log."""

        enriched = dict(payload)
        enriched.setdefault("timestamp", self.now())
        self._write_json_line(self._kpi_history_path, enriched)

    def _append_run_manifest(self, payload: Mapping[str, object]) -> None:
        self._write_json_line(self._run_manifests_path, payload)

    def _write_json_line(self, path: Path, payload: Mapping[str, object]) -> None:
        normalised = _normalise_payload(dict(payload))
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalised, sort_keys=True))
                handle.write("\n")
        except Exception:  # pragma: no cover - writing failures logged
            LOGGER.exception("Failed to write observability payload to %s", path)

    @staticmethod
    def _resolve_root(base_root: Path, configured: str) -> Path:
        """Resolve the configured observability directory relative to the repo root."""

        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = (base_root / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
