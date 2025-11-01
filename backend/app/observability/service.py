"""Utilities for persisting pipeline observability artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence
from typing_extensions import Literal
from uuid import uuid4

from backend.app.config import (
    ObservabilityAutoAuditConfig,
    ObservabilityConfig,
    ObservabilityRelevancyConfig,
)

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
            self._service._finalize_run_manifest(payload)
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
        relation_auto_accepts_filename: Optional[str] = None,
        relation_review_queue_filename: Optional[str] = None,
    ) -> None:
        self._config = config
        self._quality = config.quality
        self._auto_audit: ObservabilityAutoAuditConfig = config.auto_audit
        self._relevancy_config = config.relevancy
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        base_root = root_dir or Path(__file__).resolve().parents[2]
        self._root = self._resolve_root(base_root, config.root_dir)
        self._run_manifests_path = self._root / config.run_manifests_filename
        self._export_events_path = self._root / config.export_events_filename
        self._qa_metrics_path = self._root / config.qa_metrics_filename
        self._kpi_history_path = self._root / config.kpi_history_filename
        self._audit_results_path = self._root / config.audit_results_filename
        self._semantic_drift_path = self._root / config.semantic_drift_filename
        self._quality_alerts_path = self._root / config.quality_alerts_filename
        self._relevancy_metrics_path = self._root / config.relevancy_metrics_filename
        self._relation_auto_accepts_path: Optional[Path] = None
        self._relation_review_queue_path: Optional[Path] = None
        if relation_auto_accepts_filename:
            self._relation_auto_accepts_path = self._root / relation_auto_accepts_filename
        if relation_review_queue_filename:
            self._relation_review_queue_path = self._root / relation_review_queue_filename
        for path in (
            self._run_manifests_path,
            self._export_events_path,
            self._qa_metrics_path,
            self._kpi_history_path,
            self._audit_results_path,
            self._semantic_drift_path,
            self._quality_alerts_path,
            self._relevancy_metrics_path,
            *(p for p in (self._relation_auto_accepts_path, self._relation_review_queue_path) if p),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

    def artifact_path(
        self,
        artifact: Literal[
            "run_manifests",
            "export_events",
            "qa_metrics",
            "kpi_history",
            "audit_results",
            "semantic_drift",
            "quality_alerts",
            "relevancy_metrics",
        ],
    ) -> Path:
        """Return the filesystem path for a named observability artifact."""

        mapping = {
            "run_manifests": self._run_manifests_path,
            "export_events": self._export_events_path,
            "qa_metrics": self._qa_metrics_path,
            "kpi_history": self._kpi_history_path,
            "audit_results": self._audit_results_path,
            "semantic_drift": self._semantic_drift_path,
            "quality_alerts": self._quality_alerts_path,
            "relevancy_metrics": self._relevancy_metrics_path,
        }
        return mapping[artifact]

    @property
    def auto_audit(self) -> ObservabilityAutoAuditConfig:
        """Return the configured automated audit settings."""

        return self._auto_audit

    @property
    def relevancy(self) -> ObservabilityRelevancyConfig:
        """Return the configured semantic relevancy settings."""

        return self._relevancy_config

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

    def record_relevancy_metrics(self, payload: Mapping[str, object]) -> None:
        """Persist semantic relevancy metrics for dashboard visualisation."""

        enriched = dict(payload)
        enriched.setdefault("timestamp", self.now())
        self._write_json_line(self._relevancy_metrics_path, enriched)

    def persist_kpi_snapshot(self, payload: Mapping[str, object]) -> None:
        """Append a KPI snapshot to the historical trend log."""

        enriched = dict(payload)
        enriched.setdefault("timestamp", self.now())
        self._write_json_line(self._kpi_history_path, enriched)

    def record_edge_audit(
        self,
        *,
        run_id: str,
        paper_id: str,
        reviewer: str,
        total_edges: int,
        correct_edges: int,
        duplicate_edges: int,
        failure_reasons: Mapping[str, int] | None = None,
    ) -> None:
        """Persist manual edge audit results and update associated KPIs."""

        if total_edges <= 0:
            LOGGER.warning("Received audit for %s with no edges to score", paper_id)
            total_edges = 0
        reasons: Dict[str, int] = {}
        if failure_reasons:
            for key, value in failure_reasons.items():
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    reasons[str(key)] = count
        payload = {
            "run_id": run_id,
            "paper_id": paper_id,
            "reviewer": reviewer,
            "total_edges": total_edges,
            "correct_edges": max(0, correct_edges),
            "duplicate_edges": max(0, duplicate_edges),
            "failure_reasons": reasons,
            "timestamp": self.now(),
        }
        self._write_json_line(self._audit_results_path, payload)

        correct_ratio = self._safe_ratio(payload["correct_edges"], total_edges)
        duplicate_ratio = self._safe_ratio(payload["duplicate_edges"], total_edges)

        noise_status = self._quality_status(
            correct_ratio,
            warning=self._quality.noise_control_warning,
            target=self._quality.noise_control_target,
            higher_is_better=True,
        )
        duplicate_status = self._quality_status(
            duplicate_ratio,
            warning=self._quality.duplicate_rate_warning,
            target=self._quality.duplicate_rate_target,
            higher_is_better=False,
        )

        kpi_context = {
            "run_id": run_id,
            "paper_id": paper_id,
            "reviewer": reviewer,
            "sample_size": total_edges,
            "source": "edge_audit",
        }
        self.persist_kpi_snapshot(
            {
                "metric": "noise_control",
                "value": correct_ratio,
                "status": noise_status,
                **kpi_context,
            }
        )
        self.persist_kpi_snapshot(
            {
                "metric": "duplicate_rate",
                "value": duplicate_ratio,
                "status": duplicate_status,
                **kpi_context,
            }
        )

        if noise_status != "green":
            threshold = (
                self._quality.noise_control_target
                if noise_status == "red"
                else self._quality.noise_control_warning
            )
            self.record_quality_alert(
                level="critical" if noise_status == "red" else "warning",
                metric="noise_control",
                current_value=correct_ratio,
                threshold=threshold,
                context={
                    "run_id": run_id,
                    "paper_id": paper_id,
                    "reviewer": reviewer,
                    "sample_size": total_edges,
                    "duplicate_edges": payload["duplicate_edges"],
                    "correct_edges": payload["correct_edges"],
                },
            )

        if duplicate_status != "green":
            threshold = (
                self._quality.duplicate_rate_target
                if duplicate_status == "red"
                else self._quality.duplicate_rate_warning
            )
            self.record_quality_alert(
                level="critical" if duplicate_status == "red" else "warning",
                metric="duplicate_rate",
                current_value=duplicate_ratio,
                threshold=threshold,
                context={
                    "run_id": run_id,
                    "paper_id": paper_id,
                    "reviewer": reviewer,
                    "sample_size": total_edges,
                    "duplicate_edges": payload["duplicate_edges"],
                    "correct_edges": payload["correct_edges"],
                },
            )

    def record_quality_alert(
        self,
        *,
        level: str,
        metric: str,
        current_value: float,
        threshold: float,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Persist a quality alert with diagnostic context for triage."""

        payload: Dict[str, object] = {
            "level": level,
            "metric": metric,
            "current_value": current_value,
            "threshold": threshold,
            "timestamp": self.now(),
        }
        if context:
            for key, value in context.items():
                payload[str(key)] = value
        self._write_json_line(self._quality_alerts_path, payload)

    def _finalize_run_manifest(self, payload: Mapping[str, object]) -> None:
        self._record_run_kpis(payload)
        self._spot_semantic_drift(payload)
        self._append_run_manifest(payload)

    def _append_run_manifest(self, payload: Mapping[str, object]) -> None:
        self._write_json_line(self._run_manifests_path, payload)

    def _record_run_kpis(self, payload: Mapping[str, object]) -> None:
        documents = payload.get("documents")
        if not isinstance(documents, Mapping):
            return

        attempted_total = 0.0
        accepted_total = 0.0
        processed_total = 0.0
        skipped_total = 0.0
        doc_ids: set[str] = set()

        for doc_id, document in documents.items():
            if not isinstance(document, Mapping):
                continue
            doc_ids.add(str(doc_id))
            metrics = document.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            attempted_total += self._coerce_float(metrics.get("attempted_triples"))
            accepted_total += self._coerce_float(metrics.get("accepted_triples"))
            processed_total += self._coerce_float(metrics.get("processed_chunks"))
            skipped_total += self._coerce_float(metrics.get("skipped_chunks"))

        context = {
            "run_id": payload.get("run_id"),
            "pipeline_version": payload.get("pipeline_version"),
            "documents": sorted(doc_ids),
            "source": "run_manifest",
        }

        if attempted_total > 0:
            acceptance_ratio = self._safe_ratio(accepted_total, attempted_total)
            acceptance_status = self._quality_status(
                acceptance_ratio,
                warning=self._quality.acceptance_rate_warning,
                target=self._quality.acceptance_rate_target,
                higher_is_better=True,
            )
            self.persist_kpi_snapshot(
                {
                    "metric": "extraction_acceptance_rate",
                    "value": acceptance_ratio,
                    "status": acceptance_status,
                    "attempted_triples": attempted_total,
                    "accepted_triples": accepted_total,
                    **context,
                }
            )

        total_chunks = processed_total + skipped_total
        if total_chunks > 0:
            success_ratio = self._safe_ratio(processed_total, total_chunks)
            success_status = self._quality_status(
                success_ratio,
                warning=self._quality.pipeline_success_warning,
                target=self._quality.pipeline_success_target,
                higher_is_better=True,
            )
            self.persist_kpi_snapshot(
                {
                    "metric": "pipeline_success_rate",
                    "value": success_ratio,
                    "status": success_status,
                    "processed_chunks": processed_total,
                    "skipped_chunks": skipped_total,
                    **context,
                }
            )

    def _write_json_line(self, path: Path, payload: Mapping[str, object]) -> None:
        normalised = _normalise_payload(dict(payload))
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalised, sort_keys=True))
                handle.write("\n")
        except Exception:  # pragma: no cover - writing failures logged
            LOGGER.exception("Failed to write observability payload to %s", path)

    def record_relation_auto_accept(self, payload: Mapping[str, object]) -> None:
        """Persist a semantic matcher acceptance event."""

        if self._relation_auto_accepts_path is None:
            return
        self._write_json_line(self._relation_auto_accepts_path, payload)

    def enqueue_relation_review(self, payload: Mapping[str, object]) -> None:
        """Persist a relation phrase that requires manual review."""

        if self._relation_review_queue_path is None:
            return
        self._write_json_line(self._relation_review_queue_path, payload)

    @staticmethod
    def _resolve_root(base_root: Path, configured: str) -> Path:
        """Resolve the configured observability directory relative to the repo root."""

        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = (base_root / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _spot_semantic_drift(self, payload: Mapping[str, object]) -> None:
        documents = payload.get("documents")
        if not isinstance(documents, Mapping):
            return
        run_id = str(payload.get("run_id", ""))
        pipeline_version = str(payload.get("pipeline_version", ""))
        for doc_id, doc_payload in documents.items():
            if not isinstance(doc_payload, Mapping):
                continue
            metrics = doc_payload.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            previous = self._load_previous_document_metrics(str(doc_id))
            if previous is None:
                continue
            prev_metrics = previous.get("metrics", {})
            current_accepted = self._coerce_float(metrics.get("accepted_triples"))
            previous_accepted = self._coerce_float(prev_metrics.get("accepted_triples"))
            accepted_drop_ratio = 0.0
            drop_triggered = False
            if previous_accepted > 0:
                accepted_drop_ratio = max(
                    0.0,
                    (previous_accepted - current_accepted) / previous_accepted,
                )
                drop_triggered = (
                    accepted_drop_ratio >= self._quality.semantic_drift_drop_threshold
                )

            current_relations = self._coerce_relation_types(
                metrics.get("relation_types")
            )
            previous_relations = self._coerce_relation_types(
                prev_metrics.get("relation_types")
            )
            new_relations = sorted(current_relations - previous_relations)
            relation_triggered = (
                len(new_relations) >= self._quality.semantic_drift_relation_threshold
            )

            if not drop_triggered and not relation_triggered:
                continue

            drift_payload: Dict[str, object] = {
                "run_id": run_id,
                "paper_id": str(doc_id),
                "pipeline_version": pipeline_version,
                "previous_run_id": previous.get("run_id"),
                "previous_completed_at": previous.get("completed_at"),
                "accepted_drop_ratio": accepted_drop_ratio,
                "accepted_triples": metrics.get("accepted_triples"),
                "previous_accepted_triples": prev_metrics.get("accepted_triples"),
                "new_relation_types": new_relations,
                "timestamp": self.now(),
            }
            self._write_json_line(self._semantic_drift_path, drift_payload)

            if drop_triggered:
                self.persist_kpi_snapshot(
                    {
                        "metric": "semantic_drift",
                        "value": accepted_drop_ratio,
                        "status": "critical",
                        "run_id": run_id,
                        "paper_id": str(doc_id),
                        "baseline_run_id": previous.get("run_id"),
                        "source": "run_diff",
                    }
                )
                self.record_quality_alert(
                    level="critical",
                    metric="semantic_drift",
                    current_value=accepted_drop_ratio,
                    threshold=self._quality.semantic_drift_drop_threshold,
                    context={
                        "run_id": run_id,
                        "paper_id": str(doc_id),
                        "baseline_run_id": previous.get("run_id"),
                        "accepted_triples": metrics.get("accepted_triples"),
                        "previous_accepted_triples": prev_metrics.get("accepted_triples"),
                    },
                )

            if relation_triggered:
                self.persist_kpi_snapshot(
                    {
                        "metric": "relation_type_growth",
                        "value": len(new_relations),
                        "status": "warning",
                        "run_id": run_id,
                        "paper_id": str(doc_id),
                        "baseline_run_id": previous.get("run_id"),
                        "source": "run_diff",
                    }
                )
                self.record_quality_alert(
                    level="warning",
                    metric="relation_type_growth",
                    current_value=len(new_relations),
                    threshold=self._quality.semantic_drift_relation_threshold,
                    context={
                        "run_id": run_id,
                        "paper_id": str(doc_id),
                        "baseline_run_id": previous.get("run_id"),
                        "new_relation_types": new_relations,
                    },
                )

    def _load_previous_document_metrics(self, doc_id: str) -> Optional[Dict[str, object]]:
        if not self._run_manifests_path.exists():
            return None
        latest: Optional[Dict[str, object]] = None
        try:
            with self._run_manifests_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = line.strip()
                    if not record:
                        continue
                    try:
                        entry = json.loads(record)
                    except json.JSONDecodeError:
                        continue
                    documents = entry.get("documents")
                    if not isinstance(documents, Mapping):
                        continue
                    doc_payload = documents.get(doc_id)
                    if not isinstance(doc_payload, Mapping):
                        continue
                    metrics = doc_payload.get("metrics")
                    if not isinstance(metrics, Mapping):
                        metrics = {}
                    latest = {
                        "run_id": entry.get("run_id"),
                        "completed_at": entry.get("completed_at"),
                        "metrics": metrics,
                    }
        except OSError:
            LOGGER.exception("Unable to read previous run manifests for %s", doc_id)
            return None
        return latest

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _quality_status(
        value: float,
        *,
        warning: float,
        target: float,
        higher_is_better: bool,
    ) -> str:
        if higher_is_better:
            if value >= warning:
                return "green"
            if value >= target:
                return "amber"
            return "red"
        if value <= warning:
            return "green"
        if value <= target:
            return "amber"
        return "red"

    @staticmethod
    def _coerce_relation_types(value: object) -> set[str]:
        relations: set[str] = set()
        if value is None:
            return relations
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                relations.add(cleaned)
            return relations
        if isinstance(value, Mapping):
            iterable = value.keys()
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            iterable = value
        else:
            iterable = []
        for item in iterable:
            cleaned = str(item).strip()
            if cleaned:
                relations.add(cleaned)
        return relations

    @staticmethod
    def _coerce_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
