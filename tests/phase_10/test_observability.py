from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from backend.app.config import ObservabilityConfig, ObservabilityQualityConfig
from backend.app.observability import ObservabilityDashboard, ObservabilityService


@dataclass(frozen=True)
class _StubStatus:
    value: str


@dataclass(frozen=True)
class _StubRecord:
    paper_id: str
    filename: str
    status: _StubStatus
    updated_at: datetime


class _StubRegistry:
    def __init__(self, records: Sequence[_StubRecord]) -> None:
        self._records = list(records)

    def list_records(self) -> Sequence[_StubRecord]:
        return list(self._records)


def _config() -> ObservabilityConfig:
    return ObservabilityConfig(
        root_dir="observability",
        run_manifests_filename="runs.jsonl",
        export_events_filename="events.jsonl",
        qa_metrics_filename="qa.jsonl",
        kpi_history_filename="kpi.jsonl",
        audit_results_filename="audits.jsonl",
        semantic_drift_filename="drift.jsonl",
        quality_alerts_filename="alerts.jsonl",
        quality=ObservabilityQualityConfig(
            noise_control_target=0.85,
            noise_control_warning=0.9,
            duplicate_rate_target=0.1,
            duplicate_rate_warning=0.05,
            semantic_drift_drop_threshold=0.25,
            semantic_drift_relation_threshold=2,
        ),
    )


def test_observability_run_persists_manifest(tmp_path) -> None:
    clock_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    service = ObservabilityService(_config(), root_dir=tmp_path, clock=lambda: clock_time)

    run = service.start_run(papers=["paper-1", "paper-2"], pipeline_version="1.0.0")
    run.record_document(
        "paper-1",
        {
            "parsed_elements": 5,
            "processed_chunks": 3,
        },
        errors=["parse-error"],
    )
    run.record_phase_metrics("parsing", {"seconds": 1.23})
    run.record_summary({"documents": 2})
    run.record_batch_errors(["parse-error"])
    run.set_total_seconds(2.5)
    run.finalize()

    manifest_path = tmp_path / "observability" / "runs.jsonl"
    content = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    entry = json.loads(content[0])
    assert entry["pipeline_version"] == "1.0.0"
    assert entry["summary"]["documents"] == 2
    doc_entry = entry["documents"]["paper-1"]
    assert doc_entry["metrics"]["parsed_elements"] == 5
    assert doc_entry["errors"] == ["parse-error"]
    assert entry["phase_metrics"]["parsing"]["seconds"] == 1.23
    assert entry["total_seconds"] == 2.5


def test_observability_event_logs(tmp_path) -> None:
    clock_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    service = ObservabilityService(_config(), root_dir=tmp_path, clock=lambda: clock_time)

    service.record_export_event(
        "create",
        {"metadata_id": "meta-123", "bundle_size_bytes": 1024},
    )
    service.record_qa_metrics(
        {
            "question": "What is observed?",
            "mode": "direct",
            "resolution_seconds": 0.5,
        }
    )
    service.persist_kpi_snapshot({"metric": "faithfulness", "value": 0.95})

    events_path = tmp_path / "observability" / "events.jsonl"
    qa_path = tmp_path / "observability" / "qa.jsonl"
    kpi_path = tmp_path / "observability" / "kpi.jsonl"

    export_entry = json.loads(events_path.read_text(encoding="utf-8").strip())
    assert export_entry["event_type"] == "create"
    assert export_entry["metadata_id"] == "meta-123"

    qa_entry = json.loads(qa_path.read_text(encoding="utf-8").strip())
    assert qa_entry["question"] == "What is observed?"
    assert qa_entry["mode"] == "direct"

    kpi_entry = json.loads(kpi_path.read_text(encoding="utf-8").strip())
    assert kpi_entry["metric"] == "faithfulness"
    assert kpi_entry["value"] == 0.95


def test_edge_audit_updates_kpis_and_alerts(tmp_path) -> None:
    clock_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    service = ObservabilityService(_config(), root_dir=tmp_path, clock=lambda: clock_time)

    service.record_edge_audit(
        run_id="run-1",
        paper_id="paper-42",
        reviewer="auditor",
        total_edges=20,
        correct_edges=16,
        duplicate_edges=4,
        failure_reasons={"direction": 2, "hallucination": 0},
    )

    base = tmp_path / "observability"
    audit_entry = json.loads((base / "audits.jsonl").read_text(encoding="utf-8").strip())
    assert audit_entry["paper_id"] == "paper-42"
    assert audit_entry["correct_edges"] == 16
    assert audit_entry["duplicate_edges"] == 4
    assert audit_entry["failure_reasons"] == {"direction": 2}

    kpi_path = base / "kpi.jsonl"
    kpi_lines = [json.loads(line) for line in kpi_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {entry["metric"] for entry in kpi_lines} == {"noise_control", "duplicate_rate"}
    noise_entry = next(entry for entry in kpi_lines if entry["metric"] == "noise_control")
    duplicate_entry = next(entry for entry in kpi_lines if entry["metric"] == "duplicate_rate")
    assert noise_entry["status"] == "red"
    assert duplicate_entry["status"] == "red"

    alerts_path = base / "alerts.jsonl"
    alert_lines = [json.loads(line) for line in alerts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(alert_lines) == 2
    assert {entry["metric"] for entry in alert_lines} == {"noise_control", "duplicate_rate"}
    assert {entry["level"] for entry in alert_lines} == {"critical"}


def test_semantic_drift_detection_emits_alert(tmp_path) -> None:
    service = ObservabilityService(
        _config(),
        root_dir=tmp_path,
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    run_one = service.start_run(papers=["paper-42"], pipeline_version="1.0.0")
    run_one.record_document(
        "paper-42",
        {"accepted_triples": 40, "relation_types": ["uses", "defines"]},
    )
    run_one.finalize()

    run_two = service.start_run(papers=["paper-42"], pipeline_version="1.0.0")
    run_two.record_document(
        "paper-42",
        {
            "accepted_triples": 20,
            "relation_types": ["uses", "defines", "causes", "prevents"],
        },
    )
    run_two.finalize()

    base = tmp_path / "observability"
    drift_lines = [json.loads(line) for line in (base / "drift.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(drift_lines) == 1
    drift_entry = drift_lines[0]
    assert drift_entry["paper_id"] == "paper-42"
    assert drift_entry["accepted_drop_ratio"] >= 0.25
    assert set(drift_entry["new_relation_types"]) == {"causes", "prevents"}

    alerts_lines = [json.loads(line) for line in (base / "alerts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = [entry["metric"] for entry in alerts_lines]
    assert "semantic_drift" in metrics
    assert "relation_type_growth" in metrics


def test_observability_artifact_path_resolves(tmp_path) -> None:
    service = ObservabilityService(
        _config(),
        root_dir=tmp_path,
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    runs_path = service.artifact_path("run_manifests")
    alerts_path = service.artifact_path("quality_alerts")

    assert runs_path == (tmp_path / "observability" / "runs.jsonl")
    assert alerts_path == (tmp_path / "observability" / "alerts.jsonl")


def test_dashboard_snapshot_and_render(tmp_path) -> None:
    clock_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    service = ObservabilityService(_config(), root_dir=tmp_path, clock=lambda: clock_time)

    run = service.start_run(papers=["paper-1"], pipeline_version="1.2.3")
    run.record_document(
        "paper-1",
        {
            "parsed_elements": 10,
            "processed_chunks": 5,
            "attempted_triples": 12,
            "accepted_triples": 10,
            "rejected_triples": 2,
            "nodes_written": 4,
            "edges_written": 8,
            "co_mention_edges": 2,
        },
        errors=["parse warning"],
    )
    run.record_summary({"documents": 1, "co_mention_edges": 2})
    run.finalize()

    service.record_qa_metrics(
        {
            "question": "Q1",
            "mode": "direct",
            "resolution_seconds": 0.5,
            "total_seconds": 1.2,
            "paths": 1,
            "fallback_edges": 0,
        }
    )
    service.record_qa_metrics(
        {
            "question": "Q2",
            "mode": "direct",
            "resolution_seconds": 1.0,
            "total_seconds": 2.0,
            "paths": 0,
            "fallback_edges": 2,
        }
    )
    service.persist_kpi_snapshot(
        {"metric": "noise_control", "value": 0.9, "status": "green", "timestamp": clock_time.isoformat()}
    )
    service.persist_kpi_snapshot(
        {"metric": "noise_control", "value": 0.8, "status": "amber", "timestamp": clock_time.isoformat()}
    )
    service.record_quality_alert(
        level="warning",
        metric="noise_control",
        current_value=0.8,
        threshold=0.85,
        context={"run_id": "run-demo"},
    )
    service.record_export_event(
        "create",
        {
            "metadata_id": "meta-1",
            "bundle_size_bytes": 2_000_000,
            "warning": False,
        },
    )
    service.record_export_event(
        "download",
        {
            "metadata_id": "meta-1",
            "first_download": True,
            "latency_seconds": 3.0,
        },
    )

    registry = _StubRegistry(
        [
            _StubRecord(
                paper_id="paper-1",
                filename="demo.pdf",
                status=_StubStatus("uploaded"),
                updated_at=clock_time,
            ),
            _StubRecord(
                paper_id="paper-2",
                filename="second.pdf",
                status=_StubStatus("processing"),
                updated_at=clock_time,
            ),
        ]
    )

    dashboard = ObservabilityDashboard(service, paper_registry=registry)
    snapshot = dashboard.snapshot()

    assert snapshot.queue.totals["processing"] == 1
    assert snapshot.queue.totals["uploaded"] == 1
    assert snapshot.exports.created == 1
    assert snapshot.exports.downloads == 1
    assert snapshot.exports.pending_first_downloads == 0
    assert snapshot.qa.fallback_queries == 1
    assert snapshot.soft_failures.co_mention_edges == 2
    assert snapshot.kpis and snapshot.kpis[0].metric == "noise_control"
    assert snapshot.alerts and snapshot.alerts[0].metric == "noise_control"
    assert any("parse warning" in error for error in snapshot.recent_errors)

    html_content = dashboard.render_html(snapshot)
    assert "SciNetz Observability Dashboard" in html_content
    assert snapshot.runs[0].run_id in html_content
    assert "Extraction Queue" in html_content
