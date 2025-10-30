"""Dashboard helpers for rendering observability data."""

from __future__ import annotations

import html
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from typing_extensions import Protocol

from backend.app.observability.service import ObservabilityService


class PaperRegistryReader(Protocol):
    """Protocol describing the PaperRegistry read operations used by the dashboard."""

    def list_records(self) -> Sequence["PaperRecord"]:  # pragma: no cover - protocol definition
        """Return available paper records."""


@dataclass(frozen=True)
class QueueEntry:
    """Summary of a paper awaiting or undergoing processing."""

    paper_id: str
    filename: str
    status: str
    updated_at: Optional[datetime]


@dataclass(frozen=True)
class QueueSnapshot:
    """Aggregated queue counts and active entries."""

    totals: Dict[str, int]
    active: List[QueueEntry]


@dataclass(frozen=True)
class RunDocumentSummary:
    """Per-document metrics captured in a run manifest."""

    paper_id: str
    metrics: Dict[str, object]
    errors: List[str]


@dataclass(frozen=True)
class RunSummary:
    """Aggregated representation of a pipeline run."""

    run_id: str
    pipeline_version: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_seconds: Optional[float]
    documents: List[RunDocumentSummary]
    summary: Dict[str, object]
    phase_metrics: Dict[str, Dict[str, float]]
    batch_errors: List[str]

    def document_count(self) -> int:
        """Return the number of documents processed in the run."""

        return len(self.documents)

    def aggregate_metric(self, name: str) -> float:
        """Return the sum of a numeric metric across documents."""

        total = 0.0
        for document in self.documents:
            value = document.metrics.get(name)
            if isinstance(value, (int, float)):
                total += float(value)
        return total


@dataclass(frozen=True)
class KPIHistoryPoint:
    """Historical KPI reading for charting trends."""

    timestamp: Optional[datetime]
    value: Optional[float]
    status: Optional[str]


@dataclass(frozen=True)
class KPIStatus:
    """Latest KPI status with historical context."""

    metric: str
    latest_value: Optional[float]
    latest_status: Optional[str]
    last_updated: Optional[datetime]
    history: List[KPIHistoryPoint]


@dataclass(frozen=True)
class AlertRecord:
    """Alert entry emitted by the quality monitoring subsystem."""

    level: str
    metric: str
    current_value: Optional[float]
    threshold: Optional[float]
    timestamp: Optional[datetime]
    context: Dict[str, object]


@dataclass(frozen=True)
class ExportSummary:
    """Aggregated export lifecycle metrics for dashboard display."""

    created: int
    downloads: int
    revocations: int
    warnings: int
    first_download_p50: Optional[float]
    first_download_p95: Optional[float]
    pending_first_downloads: int
    total_bundle_mb: float


@dataclass(frozen=True)
class QASummary:
    """Aggregated QA latency and fallback statistics."""

    total_records: int
    resolution_p50: Optional[float]
    resolution_p95: Optional[float]
    total_latency_p50: Optional[float]
    total_latency_p95: Optional[float]
    fallback_queries: int
    zero_path_queries: int


@dataclass(frozen=True)
class SoftFailureSummary:
    """Counts of degradations surfaced on the dashboard."""

    co_mention_edges: float
    fallback_queries: int
    zero_path_queries: int


@dataclass(frozen=True)
class RejectionBreakdown:
    """Breakdown of rejection reasons captured during extraction."""

    reason: str
    count: int
    percent: Optional[float]


@dataclass(frozen=True)
class RelationCoverage:
    """Count of relation verbatim values encountered in a run."""

    relation: str
    occurrences: int


@dataclass(frozen=True)
class AuditFinding:
    """Single manual audit entry summarised for dashboard output."""

    run_id: str
    paper_id: str
    reviewer: str
    total_edges: int
    correct_ratio: Optional[float]
    duplicate_ratio: Optional[float]
    failure_reasons: Dict[str, int]
    timestamp: Optional[datetime]


@dataclass(frozen=True)
class AuditSummary:
    """Aggregated manual audit telemetry."""

    total_reviews: int
    total_edges: int
    average_correct_ratio: Optional[float]
    average_duplicate_ratio: Optional[float]
    last_reviewed_at: Optional[datetime]
    reviewers: List[str]
    recent_findings: List[AuditFinding]


@dataclass(frozen=True)
class GraphQualitySummary:
    """Knowledge-graph quality metrics derived from recent runs and audits."""

    run_id: Optional[str]
    pipeline_version: Optional[str]
    document_count: int
    attempted_triples: int
    accepted_triples: int
    rejected_triples: int
    acceptance_rate: Optional[float]
    nodes_written: int
    edges_written: int
    edges_per_node: Optional[float]
    rejection_breakdown: List[RejectionBreakdown]
    relation_coverage: List[RelationCoverage]
    audit_summary: AuditSummary
    semantic_drift_events: int


@dataclass(frozen=True)
class DashboardSnapshot:
    """Complete payload rendered in the HTML dashboard."""

    generated_at: datetime
    queue: QueueSnapshot
    runs: List[RunSummary]
    kpis: List[KPIStatus]
    alerts: List[AlertRecord]
    exports: ExportSummary
    qa: QASummary
    soft_failures: SoftFailureSummary
    quality: GraphQualitySummary
    recent_errors: List[str]


class ObservabilityDashboard:
    """Generate and render the observability dashboard."""

    def __init__(
        self,
        service: ObservabilityService,
        *,
        paper_registry: Optional[PaperRegistryReader] = None,
    ) -> None:
        self._service = service
        self._registry = paper_registry

    def snapshot(
        self,
        *,
        run_limit: int = 5,
        history_limit: int = 12,
        alert_limit: int = 20,
        error_limit: int = 50,
    ) -> DashboardSnapshot:
        """Collect a snapshot of observability data for rendering."""

        queue = self._build_queue_snapshot()
        runs = self._load_recent_runs(run_limit)
        kpis = self._load_kpi_status(history_limit)
        alerts = self._load_alerts(alert_limit)
        exports = self._summarise_exports()
        qa_summary = self._summarise_qa()
        soft_failures = self._collect_soft_failures(runs, qa_summary)
        quality_summary = self._summarise_quality(runs)
        recent_errors = self._collect_recent_errors(runs, error_limit)
        return DashboardSnapshot(
            generated_at=datetime.now(timezone.utc),
            queue=queue,
            runs=runs,
            kpis=kpis,
            alerts=alerts,
            exports=exports,
            qa=qa_summary,
            soft_failures=soft_failures,
            quality=quality_summary,
            recent_errors=recent_errors,
        )

    def render_html(self, snapshot: DashboardSnapshot) -> str:
        """Render the snapshot as a standalone HTML document."""

        def fmt_datetime(value: Optional[datetime]) -> str:
            if value is None:
                return "&mdash;"
            return html.escape(value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

        def fmt_float(value: Optional[float], *, digits: int = 2) -> str:
            if value is None:
                return "&mdash;"
            return f"{value:.{digits}f}"

        def fmt_status_class(status: Optional[str]) -> str:
            mapping = {
                "green": "status-green",
                "amber": "status-amber",
                "yellow": "status-amber",
                "warning": "status-amber",
                "red": "status-red",
                "critical": "status-red",
            }
            return mapping.get((status or "").lower(), "")

        def fmt_percent(value: Optional[float], *, digits: int = 1) -> str:
            if value is None:
                return "&mdash;"
            return f"{value * 100:.{digits}f}%"

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<title>SciNetz Observability Dashboard</title>",
            "<style>",
            "body { font-family: 'Inter', 'Segoe UI', sans-serif; margin: 1.5rem; background: #f9fafb; color: #0f172a; }",
            "h1, h2, h3 { color: #0f172a; }",
            "section { margin-bottom: 2rem; }",
            "table { border-collapse: collapse; width: 100%; background: #fff; }",
            "th, td { border: 1px solid #e2e8f0; padding: 0.5rem; text-align: left; vertical-align: top; }",
            "th { background: #f1f5f9; font-weight: 600; }",
            "tbody tr:nth-child(even) { background: #f8fafc; }",
            ".status-green { background: #d1fae5; color: #065f46; padding: 0.2rem 0.5rem; border-radius: 0.4rem; display: inline-block; }",
            ".status-amber { background: #fef3c7; color: #92400e; padding: 0.2rem 0.5rem; border-radius: 0.4rem; display: inline-block; }",
            ".status-red { background: #fee2e2; color: #b91c1c; padding: 0.2rem 0.5rem; border-radius: 0.4rem; display: inline-block; }",
            ".badge { font-size: 0.85rem; padding: 0.1rem 0.6rem; border-radius: 999px; background: #e2e8f0; display: inline-block; margin-right: 0.4rem; }",
            "details { margin-top: 0.25rem; }",
            "summary { cursor: pointer; font-weight: 600; }",
            "ul { margin: 0.25rem 0 0.75rem 1.25rem; }",
            "footer { font-size: 0.85rem; color: #475569; margin-top: 2rem; }",
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            "<h1>SciNetz Observability Dashboard</h1>",
            f"<p>Generated at <strong>{fmt_datetime(snapshot.generated_at)}</strong></p>",
            "</header>",
        ]

        html_parts.append("<section><h2>Extraction Queue</h2>")
        html_parts.append("<table><thead><tr><th>Status</th><th>Count</th></tr></thead><tbody>")
        for status, count in sorted(snapshot.queue.totals.items()):
            html_parts.append(
                f"<tr><td>{html.escape(status.title())}</td><td>{count}</td></tr>"
            )
        html_parts.append("</tbody></table>")
        if snapshot.queue.active:
            html_parts.append("<details><summary>Active uploads / processing</summary><ul>")
            for entry in snapshot.queue.active[:10]:
                updated = fmt_datetime(entry.updated_at)
                html_parts.append(
                    "<li>"
                    f"<span class=\"badge\">{html.escape(entry.status.upper())}</span>"
                    f"{html.escape(entry.paper_id)} — {html.escape(entry.filename)} "
                    f"<em>(updated {updated})</em>"
                    "</li>"
                )
            html_parts.append("</ul></details>")
        html_parts.append("</section>")

        html_parts.append("<section><h2>Recent Runs</h2>")
        html_parts.append(
            "<table><thead><tr><th>Run ID</th><th>Version</th><th>Started</th><th>Completed</th><th>Duration (s)</th><th>Documents</th><th>Accepted Triples</th><th>Rejected Triples</th><th>Edges Written</th><th>Co-mentions</th><th>Errors</th></tr></thead><tbody>"
        )
        for run in snapshot.runs:
            accepted = fmt_float(run.aggregate_metric("accepted_triples"), digits=0)
            rejected = fmt_float(run.aggregate_metric("rejected_triples"), digits=0)
            edges = fmt_float(run.aggregate_metric("edges_written"), digits=0)
            co_mentions = fmt_float(run.aggregate_metric("co_mention_edges"), digits=0)
            errors = len(run.batch_errors) + sum(len(doc.errors) for doc in run.documents)
            html_parts.append(
                "<tr>"
                f"<td>{html.escape(run.run_id)}</td>"
                f"<td>{html.escape(run.pipeline_version)}</td>"
                f"<td>{fmt_datetime(run.started_at)}</td>"
                f"<td>{fmt_datetime(run.completed_at)}</td>"
                f"<td>{fmt_float(run.total_seconds)}</td>"
                f"<td>{run.document_count()}</td>"
                f"<td>{accepted}</td>"
                f"<td>{rejected}</td>"
                f"<td>{edges}</td>"
                f"<td>{co_mentions}</td>"
                f"<td>{errors}</td>"
                "</tr>"
            )
            if run.documents:
                html_parts.append("<tr><td colspan=\"11\">")
                html_parts.append("<details><summary>Per-paper metrics</summary><ul>")
                for document in run.documents:
                    metrics_parts = []
                    for key in (
                        "parsed_elements",
                        "processed_chunks",
                        "attempted_triples",
                        "accepted_triples",
                        "rejected_triples",
                        "nodes_written",
                        "edges_written",
                        "co_mention_edges",
                    ):
                        value = document.metrics.get(key)
                        if value is None:
                            continue
                        metrics_parts.append(f"{key.replace('_', ' ')}: {value}")
                    metrics_html = html.escape(", ".join(metrics_parts) or "No metrics")
                    error_list = "".join(
                        f"<li>{html.escape(err)}</li>" for err in document.errors
                    ) or "<li>No errors recorded</li>"
                    html_parts.append(
                        "<li>"
                        f"<strong>{html.escape(document.paper_id)}</strong> — {metrics_html}"
                        f"<ul>{error_list}</ul>"
                        "</li>"
                    )
                html_parts.append("</ul></details>")
                html_parts.append("</td></tr>")
        if not snapshot.runs:
            html_parts.append("<tr><td colspan=\"11\">No run manifests recorded yet.</td></tr>")
        html_parts.append("</tbody></table></section>")

        quality = snapshot.quality
        html_parts.append("<section><h2>Knowledge Graph Quality</h2>")
        html_parts.append("<table><tbody>")
        html_parts.append(
            f"<tr><th>Latest run</th><td>{html.escape(quality.run_id or '—')}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Pipeline version</th><td>{html.escape(quality.pipeline_version or '—')}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Documents scored</th><td>{quality.document_count}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Attempted triples</th><td>{quality.attempted_triples}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Accepted triples</th><td>{quality.accepted_triples}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Acceptance rate</th><td>{fmt_percent(quality.acceptance_rate)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Rejected triples</th><td>{quality.rejected_triples}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Nodes written</th><td>{quality.nodes_written}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Edges written</th><td>{quality.edges_written}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Edges per node</th><td>{fmt_float(quality.edges_per_node)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Semantic drift events</th><td>{quality.semantic_drift_events}</td></tr>"
        )
        if quality.rejection_breakdown:
            items: List[str] = []
            for item in quality.rejection_breakdown:
                percent_html = (
                    f" ({fmt_percent(item.percent)})"
                    if item.percent is not None
                    else ""
                )
                items.append(
                    f"<li>{html.escape(item.reason)} — {item.count}{percent_html}</li>"
                )
            html_parts.append(
                "<tr><th>Rejection reasons</th><td><ul>" + "".join(items) + "</ul></td></tr>"
            )
        if quality.relation_coverage:
            relations_html = "".join(
                f"<li>{html.escape(item.relation)} — {item.occurrences}</li>"
                for item in quality.relation_coverage
            )
            html_parts.append(
                "<tr><th>Relation coverage</th><td><ul>" + relations_html + "</ul></td></tr>"
            )
        html_parts.append("</tbody></table>")

        audit = quality.audit_summary
        html_parts.append("<h3>Manual edge audits</h3>")
        html_parts.append("<table><tbody>")
        html_parts.append(
            f"<tr><th>Total reviews</th><td>{audit.total_reviews}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Total edges sampled</th><td>{audit.total_edges}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Average correctness</th><td>{fmt_percent(audit.average_correct_ratio)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Average duplicate rate</th><td>{fmt_percent(audit.average_duplicate_ratio)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Last reviewed</th><td>{fmt_datetime(audit.last_reviewed_at)}</td></tr>"
        )
        reviewers = ", ".join(html.escape(name) for name in audit.reviewers) or "—"
        html_parts.append(
            f"<tr><th>Reviewers</th><td>{reviewers}</td></tr>"
        )
        if audit.recent_findings:
            findings_html: List[str] = []
            for finding in audit.recent_findings:
                timestamp = fmt_datetime(finding.timestamp)
                ratios: List[str] = []
                if finding.correct_ratio is not None:
                    ratios.append(
                        f"correct={fmt_percent(finding.correct_ratio)}"
                    )
                if finding.duplicate_ratio is not None:
                    ratios.append(
                        f"duplicates={fmt_percent(finding.duplicate_ratio)}"
                    )
                context_parts = [
                    f"paper={html.escape(finding.paper_id)}",
                    f"run={html.escape(finding.run_id)}",
                    f"edges={finding.total_edges}",
                ]
                if ratios:
                    context_parts.append(" / ".join(ratios))
                if finding.failure_reasons:
                    failures = ", ".join(
                        f"{html.escape(reason)}={count}"
                        for reason, count in sorted(finding.failure_reasons.items())
                    )
                    context_parts.append(f"failures: {failures}")
                context = "; ".join(context_parts)
                findings_html.append(
                    f"<li><strong>{html.escape(finding.reviewer)}</strong> at {timestamp} — {context}</li>"
                )
            html_parts.append(
                "<tr><th>Recent findings</th><td><ul>" + "".join(findings_html) + "</ul></td></tr>"
            )
        html_parts.append("</tbody></table>")
        html_parts.append("</section>")

        html_parts.append("<section><h2>KPI Status</h2>")
        html_parts.append(
            "<table><thead><tr><th>Metric</th><th>Latest Value</th><th>Status</th><th>Last Updated</th><th>Recent History</th></tr></thead><tbody>"
        )
        for entry in snapshot.kpis:
            history_items = []
            for point in entry.history:
                history_items.append(
                    f"<li>{fmt_datetime(point.timestamp)} — {fmt_float(point.value)} ({html.escape((point.status or '').title() or 'Unknown')})</li>"
                )
            if history_items:
                history_html = "<ul>" + "".join(history_items) + "</ul>"
            else:
                history_html = "<em>No historical data</em>"
            status_class = fmt_status_class(entry.latest_status)
            status_label = html.escape(entry.latest_status.title()) if entry.latest_status else "Unknown"
            status_html = f"<span class=\"{status_class}\">{status_label}</span>" if status_class else status_label
            html_parts.append(
                "<tr>"
                f"<td>{html.escape(entry.metric)}</td>"
                f"<td>{fmt_float(entry.latest_value)}</td>"
                f"<td>{status_html}</td>"
                f"<td>{fmt_datetime(entry.last_updated)}</td>"
                f"<td>{history_html}</td>"
                "</tr>"
            )
        if not snapshot.kpis:
            html_parts.append("<tr><td colspan=\"5\">No KPI snapshots recorded yet.</td></tr>")
        html_parts.append("</tbody></table></section>")

        html_parts.append("<section><h2>Export Lifecycle</h2>")
        html_parts.append("<table><tbody>")
        html_parts.append(
            f"<tr><th>Share links created</th><td>{snapshot.exports.created}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Downloads</th><td>{snapshot.exports.downloads}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Pending first downloads</th><td>{snapshot.exports.pending_first_downloads}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Revocations</th><td>{snapshot.exports.revocations}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Warnings issued</th><td>{snapshot.exports.warnings}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>First-download latency p50</th><td>{fmt_float(snapshot.exports.first_download_p50)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>First-download latency p95</th><td>{fmt_float(snapshot.exports.first_download_p95)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Total bundle size (MB)</th><td>{fmt_float(snapshot.exports.total_bundle_mb)}</td></tr>"
        )
        html_parts.append("</tbody></table></section>")

        html_parts.append("<section><h2>QA Performance & Soft Failures</h2>")
        html_parts.append("<table><tbody>")
        html_parts.append(
            f"<tr><th>QA records</th><td>{snapshot.qa.total_records}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Resolution latency p50</th><td>{fmt_float(snapshot.qa.resolution_p50)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Resolution latency p95</th><td>{fmt_float(snapshot.qa.resolution_p95)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Total latency p50</th><td>{fmt_float(snapshot.qa.total_latency_p50)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Total latency p95</th><td>{fmt_float(snapshot.qa.total_latency_p95)}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Fallback queries</th><td>{snapshot.soft_failures.fallback_queries}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Zero-path queries</th><td>{snapshot.soft_failures.zero_path_queries}</td></tr>"
        )
        html_parts.append(
            f"<tr><th>Co-mention edges created</th><td>{fmt_float(snapshot.soft_failures.co_mention_edges, digits=0)}</td></tr>"
        )
        html_parts.append("</tbody></table></section>")

        html_parts.append("<section><h2>Recent Alerts</h2>")
        if snapshot.alerts:
            html_parts.append("<table><thead><tr><th>Timestamp</th><th>Level</th><th>Metric</th><th>Value</th><th>Threshold</th><th>Context</th></tr></thead><tbody>")
            for alert in snapshot.alerts:
                level_class = fmt_status_class(alert.level)
                level_label = html.escape(alert.level.title())
                level_html = (
                    f"<span class=\"{level_class}\">{level_label}</span>"
                    if level_class
                    else level_label
                )
                context_items = [
                    f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
                    for key, value in sorted(alert.context.items())
                ]
                context_html = "<ul>" + "".join(context_items) + "</ul>" if context_items else "<em>No context</em>"
                html_parts.append(
                    "<tr>"
                    f"<td>{fmt_datetime(alert.timestamp)}</td>"
                    f"<td>{level_html}</td>"
                    f"<td>{html.escape(alert.metric)}</td>"
                    f"<td>{fmt_float(alert.current_value)}</td>"
                    f"<td>{fmt_float(alert.threshold)}</td>"
                    f"<td>{context_html}</td>"
                    "</tr>"
                )
            html_parts.append("</tbody></table>")
        else:
            html_parts.append("<p>No alerts recorded.</p>")
        html_parts.append("</section>")

        html_parts.append("<section><h2>Recent Errors</h2>")
        if snapshot.recent_errors:
            html_parts.append("<ul>")
            for error in snapshot.recent_errors:
                html_parts.append(f"<li>{html.escape(error)}</li>")
            html_parts.append("</ul>")
        else:
            html_parts.append("<p>No errors recorded.</p>")
        html_parts.append("</section>")

        html_parts.append(
            "<footer>Observability data is sourced from JSONL artifacts stored on disk. Refresh the page to see the latest metrics.</footer>"
        )
        html_parts.append("</body></html>")
        return "".join(html_parts)

    def _build_queue_snapshot(self) -> QueueSnapshot:
        totals: Dict[str, int] = {}
        active: List[QueueEntry] = []
        if self._registry is None:
            return QueueSnapshot(totals={}, active=[])
        try:
            records = list(self._registry.list_records())
        except Exception:  # pragma: no cover - registry errors treated as no data
            return QueueSnapshot(totals={}, active=[])
        counter: Counter[str] = Counter()
        for record in records:
            status = getattr(record, "status", None)
            if status is None:
                continue
            status_value = getattr(status, "value", None) or str(status)
            counter[status_value] += 1
            if status_value in {"uploaded", "processing"}:
                updated_at = getattr(record, "updated_at", None)
                if isinstance(updated_at, datetime):
                    updated = _ensure_timezone(updated_at)
                else:
                    updated = None
                active.append(
                    QueueEntry(
                        paper_id=str(getattr(record, "paper_id", "")),
                        filename=str(getattr(record, "filename", "")),
                        status=status_value,
                        updated_at=updated,
                    )
                )
        totals = dict(sorted(counter.items()))
        active.sort(key=lambda entry: entry.updated_at or datetime.fromtimestamp(0, tz=timezone.utc), reverse=True)
        return QueueSnapshot(totals=totals, active=active)

    def _load_recent_runs(self, limit: int) -> List[RunSummary]:
        path = self._service.artifact_path("run_manifests")
        entries = _load_jsonl(path, limit)
        runs: List[RunSummary] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            documents_payload = entry.get("documents")
            documents: List[RunDocumentSummary] = []
            if isinstance(documents_payload, Mapping):
                for paper_id, payload in documents_payload.items():
                    if not isinstance(payload, Mapping):
                        continue
                    metrics = {}
                    raw_metrics = payload.get("metrics")
                    if isinstance(raw_metrics, Mapping):
                        metrics = {str(key): value for key, value in raw_metrics.items()}
                    errors_raw = payload.get("errors")
                    errors: List[str] = []
                    if isinstance(errors_raw, Iterable) and not isinstance(errors_raw, (str, bytes)):
                        errors = [str(item) for item in errors_raw if str(item).strip()]
                    documents.append(
                        RunDocumentSummary(
                            paper_id=str(paper_id),
                            metrics=metrics,
                            errors=errors,
                        )
                    )
            summary_payload = entry.get("summary")
            summary: Dict[str, object] = {}
            if isinstance(summary_payload, Mapping):
                summary = {str(key): value for key, value in summary_payload.items()}
            phase_payload = entry.get("phase_metrics")
            phase_metrics: Dict[str, Dict[str, float]] = {}
            if isinstance(phase_payload, Mapping):
                for phase, metrics in phase_payload.items():
                    if not isinstance(metrics, Mapping):
                        continue
                    phase_metrics[str(phase)] = {
                        str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))
                    }
            errors_payload = entry.get("batch_errors")
            batch_errors: List[str] = []
            if isinstance(errors_payload, Iterable) and not isinstance(errors_payload, (str, bytes)):
                batch_errors = [str(item) for item in errors_payload if str(item).strip()]
            runs.append(
                RunSummary(
                    run_id=str(entry.get("run_id", "unknown")),
                    pipeline_version=str(entry.get("pipeline_version", "")),
                    started_at=_parse_datetime(entry.get("started_at")),
                    completed_at=_parse_datetime(entry.get("completed_at")),
                    total_seconds=_coerce_float(entry.get("total_seconds")),
                    documents=documents,
                    summary=summary,
                    phase_metrics=phase_metrics,
                    batch_errors=batch_errors,
                )
            )
        runs.sort(key=lambda run: run.completed_at or datetime.fromtimestamp(0, tz=timezone.utc), reverse=True)
        return runs

    def _load_kpi_status(self, history_limit: int) -> List[KPIStatus]:
        path = self._service.artifact_path("kpi_history")
        entries = _load_jsonl(path, None)
        buckets: Dict[str, List[KPIHistoryPoint]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            metric = str(entry.get("metric")) if entry.get("metric") is not None else None
            if not metric:
                continue
            point = KPIHistoryPoint(
                timestamp=_parse_datetime(entry.get("timestamp")),
                value=_coerce_float(entry.get("value")),
                status=str(entry.get("status")) if entry.get("status") else None,
            )
            buckets.setdefault(metric, []).append(point)
        results: List[KPIStatus] = []
        for metric, points in buckets.items():
            points.sort(key=lambda item: item.timestamp or datetime.fromtimestamp(0, tz=timezone.utc))
            latest = points[-1]
            history = points[-history_limit:]
            results.append(
                KPIStatus(
                    metric=metric,
                    latest_value=latest.value,
                    latest_status=latest.status,
                    last_updated=latest.timestamp,
                    history=history,
                )
            )
        results.sort(key=lambda item: item.metric)
        return results

    def _load_alerts(self, limit: int) -> List[AlertRecord]:
        path = self._service.artifact_path("quality_alerts")
        entries = _load_jsonl(path, limit)
        alerts: List[AlertRecord] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            context: Dict[str, object] = {}
            for key, value in entry.items():
                if key in {"level", "metric", "current_value", "threshold", "timestamp"}:
                    continue
                context[str(key)] = value
            alerts.append(
                AlertRecord(
                    level=str(entry.get("level", "")),
                    metric=str(entry.get("metric", "")),
                    current_value=_coerce_float(entry.get("current_value")),
                    threshold=_coerce_float(entry.get("threshold")),
                    timestamp=_parse_datetime(entry.get("timestamp")),
                    context=context,
                )
            )
        alerts.sort(key=lambda alert: alert.timestamp or datetime.fromtimestamp(0, tz=timezone.utc), reverse=True)
        return alerts

    def _summarise_exports(self) -> ExportSummary:
        path = self._service.artifact_path("export_events")
        entries = _load_jsonl(path, None)
        created = 0
        downloads = 0
        revocations = 0
        warnings = 0
        total_bundle_mb = 0.0
        first_latencies: List[float] = []
        created_ids: Dict[str, bool] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            event_type = str(entry.get("event_type", ""))
            metadata_id = str(entry.get("metadata_id", ""))
            if event_type == "create":
                created += 1
                created_ids[metadata_id] = False
                size_bytes = _coerce_float(entry.get("bundle_size_bytes"))
                total_bundle_mb += size_bytes / 1_000_000
                if bool(entry.get("warning")):
                    warnings += 1
            elif event_type == "download":
                downloads += 1
                first_flag = bool(entry.get("first_download"))
                if first_flag and metadata_id:
                    created_ids[metadata_id] = True
                latency = _coerce_float(entry.get("latency_seconds"))
                if first_flag and latency is not None:
                    first_latencies.append(latency)
            elif event_type == "revoke":
                revocations += 1
        pending_first = sum(1 for value in created_ids.values() if not value)
        return ExportSummary(
            created=created,
            downloads=downloads,
            revocations=revocations,
            warnings=warnings,
            first_download_p50=_percentile(first_latencies, 0.5),
            first_download_p95=_percentile(first_latencies, 0.95),
            pending_first_downloads=pending_first,
            total_bundle_mb=total_bundle_mb,
        )

    def _summarise_qa(self) -> QASummary:
        path = self._service.artifact_path("qa_metrics")
        entries = _load_jsonl(path, None)
        resolution: List[float] = []
        total_latency: List[float] = []
        fallback_queries = 0
        zero_path_queries = 0
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            res = _coerce_float(entry.get("resolution_seconds"))
            if res is not None:
                resolution.append(res)
            total = _coerce_float(entry.get("total_seconds"))
            if total is not None:
                total_latency.append(total)
            fallback_edges = _coerce_float(entry.get("fallback_edges")) or 0.0
            if fallback_edges > 0:
                fallback_queries += 1
            paths = _coerce_float(entry.get("paths")) or 0.0
            if paths <= 0:
                zero_path_queries += 1
        return QASummary(
            total_records=len(resolution) if resolution or total_latency else len(entries),
            resolution_p50=_percentile(resolution, 0.5),
            resolution_p95=_percentile(resolution, 0.95),
            total_latency_p50=_percentile(total_latency, 0.5),
            total_latency_p95=_percentile(total_latency, 0.95),
            fallback_queries=fallback_queries,
            zero_path_queries=zero_path_queries,
        )

    def _collect_soft_failures(
        self,
        runs: Sequence[RunSummary],
        qa_summary: QASummary,
    ) -> SoftFailureSummary:
        co_mentions = sum(run.aggregate_metric("co_mention_edges") for run in runs)
        return SoftFailureSummary(
            co_mention_edges=co_mentions,
            fallback_queries=qa_summary.fallback_queries,
            zero_path_queries=qa_summary.zero_path_queries,
        )

    def _collect_recent_errors(self, runs: Sequence[RunSummary], limit: int) -> List[str]:
        errors: List[str] = []
        for run in runs:
            for err in run.batch_errors:
                errors.append(f"{run.run_id}: {err}")
            for document in run.documents:
                for err in document.errors:
                    errors.append(f"{document.paper_id}: {err}")
        deduped: List[str] = []
        seen: set[str] = set()
        for err in errors:
            if err in seen:
                continue
            seen.add(err)
            deduped.append(err)
            if len(deduped) >= limit:
                break
        return deduped

    def _summarise_quality(self, runs: Sequence[RunSummary]) -> GraphQualitySummary:
        audit_summary = self._summarise_audits()
        drift_events = self._count_semantic_drift()
        if not runs:
            return GraphQualitySummary(
                run_id=None,
                pipeline_version=None,
                document_count=0,
                attempted_triples=0,
                accepted_triples=0,
                rejected_triples=0,
                acceptance_rate=None,
                nodes_written=0,
                edges_written=0,
                edges_per_node=None,
                rejection_breakdown=[],
                relation_coverage=[],
                audit_summary=audit_summary,
                semantic_drift_events=drift_events,
            )

        latest = runs[0]
        attempted = int(latest.aggregate_metric("attempted_triples"))
        accepted = int(latest.aggregate_metric("accepted_triples"))
        rejected = int(latest.aggregate_metric("rejected_triples"))
        nodes = int(latest.aggregate_metric("nodes_written"))
        edges = int(latest.aggregate_metric("edges_written"))
        acceptance_rate = accepted / attempted if attempted > 0 else None
        edges_per_node = edges / nodes if nodes > 0 else None

        rejection_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()
        for document in latest.documents:
            metrics = document.metrics
            reasons_payload = metrics.get("rejection_reasons")
            if isinstance(reasons_payload, Mapping):
                for reason, value in reasons_payload.items():
                    count_value = _coerce_float(value)
                    if count_value and count_value > 0:
                        rejection_counter[str(reason)] += int(count_value)
            relation_payload = metrics.get("relation_types")
            relation_iterable: Iterable[object]
            if isinstance(relation_payload, Mapping):
                relation_iterable = relation_payload.keys()
            elif isinstance(relation_payload, Iterable) and not isinstance(
                relation_payload, (str, bytes)
            ):
                relation_iterable = relation_payload
            else:
                relation_iterable = []
            for relation in relation_iterable:
                relation_name = str(relation).strip()
                if relation_name:
                    relation_counter[relation_name] += 1

        rejection_breakdown: List[RejectionBreakdown] = []
        for reason, count in rejection_counter.most_common():
            percent = (count / rejected) if rejected > 0 else None
            rejection_breakdown.append(
                RejectionBreakdown(reason=reason, count=count, percent=percent)
            )

        relation_coverage: List[RelationCoverage] = [
            RelationCoverage(relation=relation, occurrences=count)
            for relation, count in relation_counter.most_common(10)
        ]

        return GraphQualitySummary(
            run_id=latest.run_id,
            pipeline_version=latest.pipeline_version,
            document_count=latest.document_count(),
            attempted_triples=attempted,
            accepted_triples=accepted,
            rejected_triples=rejected,
            acceptance_rate=acceptance_rate,
            nodes_written=nodes,
            edges_written=edges,
            edges_per_node=edges_per_node,
            rejection_breakdown=rejection_breakdown,
            relation_coverage=relation_coverage,
            audit_summary=audit_summary,
            semantic_drift_events=drift_events,
        )

    def _summarise_audits(self, limit: int = 5) -> AuditSummary:
        path = self._service.artifact_path("audit_results")
        entries = _load_jsonl(path, None)
        total_reviews = 0
        total_edges = 0
        correct_sum = 0.0
        duplicate_sum = 0.0
        correct_count = 0
        duplicate_count = 0
        last_reviewed: Optional[datetime] = None
        reviewers: set[str] = set()
        findings: List[AuditFinding] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            total_reviews += 1
            total = int(_coerce_float(entry.get("total_edges")) or 0.0)
            total_edges += total
            correct_edges = int(_coerce_float(entry.get("correct_edges")) or 0.0)
            duplicate_edges = int(_coerce_float(entry.get("duplicate_edges")) or 0.0)
            correct_ratio = None
            duplicate_ratio = None
            if total > 0:
                correct_ratio = correct_edges / total
                duplicate_ratio = duplicate_edges / total
                correct_sum += correct_ratio
                duplicate_sum += duplicate_ratio
                correct_count += 1
                duplicate_count += 1
            reviewer = str(entry.get("reviewer", "")).strip()
            if reviewer:
                reviewers.add(reviewer)
            timestamp = _parse_datetime(entry.get("timestamp"))
            if timestamp and (last_reviewed is None or timestamp > last_reviewed):
                last_reviewed = timestamp
            failure_reasons: Dict[str, int] = {}
            reasons_payload = entry.get("failure_reasons")
            if isinstance(reasons_payload, Mapping):
                for reason, value in reasons_payload.items():
                    count = int(_coerce_float(value) or 0.0)
                    if count > 0:
                        failure_reasons[str(reason)] = count
            findings.append(
                AuditFinding(
                    run_id=str(entry.get("run_id", "")),
                    paper_id=str(entry.get("paper_id", "")),
                    reviewer=reviewer or "Unknown",
                    total_edges=total,
                    correct_ratio=correct_ratio,
                    duplicate_ratio=duplicate_ratio,
                    failure_reasons=failure_reasons,
                    timestamp=timestamp,
                )
            )
        findings.sort(
            key=lambda item: item.timestamp or datetime.fromtimestamp(0, tz=timezone.utc),
            reverse=True,
        )
        average_correct = (
            correct_sum / correct_count if correct_count > 0 else None
        )
        average_duplicate = (
            duplicate_sum / duplicate_count if duplicate_count > 0 else None
        )
        return AuditSummary(
            total_reviews=total_reviews,
            total_edges=total_edges,
            average_correct_ratio=average_correct,
            average_duplicate_ratio=average_duplicate,
            last_reviewed_at=last_reviewed,
            reviewers=sorted(reviewers),
            recent_findings=findings[:limit],
        )

    def _count_semantic_drift(self) -> int:
        path = self._service.artifact_path("semantic_drift")
        entries = _load_jsonl(path, None)
        return len(entries)


def _load_jsonl(path: Path, limit: Optional[int]) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    buffer: deque[Dict[str, object]]
    if limit is not None and limit > 0:
        buffer = deque(maxlen=limit)
    else:
        buffer = deque()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = line.strip()
                if not record:
                    continue
                try:
                    payload = json.loads(record)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    buffer.append(payload)
    except OSError:  # pragma: no cover - file access errors treated as empty
        return []
    return list(buffer)


def _parse_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _ensure_timezone(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return _ensure_timezone(parsed)
    return None


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if isinstance(value, str) and value.strip():
            return float(value)
    except ValueError:
        return None
    return None


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    clamped = min(max(quantile, 0.0), 1.0)
    index = (len(sorted_values) - 1) * clamped
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * fraction

