from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.app.config import ObservabilityConfig
from backend.app.observability import ObservabilityService


def _config() -> ObservabilityConfig:
    return ObservabilityConfig(
        root_dir="observability",
        run_manifests_filename="runs.jsonl",
        export_events_filename="events.jsonl",
        qa_metrics_filename="qa.jsonl",
        kpi_history_filename="kpi.jsonl",
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
