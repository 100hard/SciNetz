"""Unit tests covering the paper registry lifecycle."""

from __future__ import annotations

from pathlib import Path

from backend.app.contracts import PaperMetadata
from backend.app.ui import PaperRegistry, PaperStatus


def test_paper_registry_tracks_status_transitions(tmp_path: Path) -> None:
    registry_path = tmp_path / "papers.json"
    registry = PaperRegistry(registry_path)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    record = registry.register_upload("paper-1", "doc.pdf", pdf_path, owner_id="user-1")
    assert record.status is PaperStatus.UPLOADED
    assert record.owner_id == "user-1"
    assert record.shared_with == []
    assert not record.is_public

    in_progress = registry.mark_processing("paper-1")
    assert in_progress is not None
    assert in_progress.status is PaperStatus.PROCESSING

    metadata = PaperMetadata(doc_id="paper-1", title="Sample")
    completed = registry.mark_complete(
        "paper-1",
        metadata=metadata,
        nodes_written=3,
        edges_written=5,
        co_mention_edges=1,
    )
    assert completed is not None
    assert completed.status is PaperStatus.COMPLETE
    assert completed.metadata == metadata
    assert completed.nodes_written == 3
    assert completed.edges_written == 5
    assert completed.co_mention_edges == 1

    stored = registry.get("paper-1")
    assert stored is not None
    assert stored.status is PaperStatus.COMPLETE


def test_paper_registry_records_failures(tmp_path: Path) -> None:
    registry = PaperRegistry(tmp_path / "papers.json")
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    registry.register_upload("paper-2", "doc.pdf", pdf_path, owner_id="owner-2")
    failed = registry.mark_failed("paper-2", ["boom"])
    assert failed is not None
    assert failed.status is PaperStatus.FAILED
    assert failed.errors == ["boom"]

    # failure state persists after reload
    reloaded = PaperRegistry(tmp_path / "papers.json")
    record = reloaded.get("paper-2")
    assert record is not None
    assert record.status is PaperStatus.FAILED
    assert record.errors == ["boom"]
