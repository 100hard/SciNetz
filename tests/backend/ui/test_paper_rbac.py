"""Tests covering paper registry access controls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.auth.enums import UserRole
from backend.app.ui import PaperRegistry, PaperStatus


def test_legacy_records_default_to_private(tmp_path: Path) -> None:
    registry_path = tmp_path / "legacy.json"
    pdf_path = tmp_path / "legacy.pdf"
    pdf_path.write_text("legacy", encoding="utf-8")
    timestamp = datetime.now(timezone.utc).isoformat()
    legacy_payload = {
        "paper-legacy": {
            "filename": "legacy.pdf",
            "pdf_path": str(pdf_path),
            "status": PaperStatus.COMPLETE.value,
            "uploaded_at": timestamp,
            "updated_at": timestamp,
            "nodes_written": 1,
            "edges_written": 1,
            "co_mention_edges": 0,
        }
    }
    registry_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    registry = PaperRegistry(registry_path)
    record = registry.get("paper-legacy")
    assert record is not None
    assert record.owner_id is None
    assert not record.is_public
    assert record.shared_with == []
    assert not record.is_accessible_by("user-1", UserRole.USER)
    assert not record.is_accessible_by("admin", UserRole.USER)


def test_registry_access_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "papers.json"
    registry = PaperRegistry(registry_path)
    pdf_one = tmp_path / "one.pdf"
    pdf_one.write_text("one", encoding="utf-8")
    pdf_two = tmp_path / "two.pdf"
    pdf_two.write_text("two", encoding="utf-8")

    registry.register_upload("paper-1", "one.pdf", pdf_one, owner_id="owner-1")
    shared_record = registry.register_upload(
        "paper-2",
        "two.pdf",
        pdf_two,
        owner_id="owner-2",
        shared_with=["collaborator", "collaborator", "other"],
    )
    assert shared_record.shared_with == ["collaborator", "other"]

    updated = registry.update_access(
        "paper-2", shared_with=["collaborator", "helper"], is_public=True
    )
    assert updated is not None
    assert updated.is_public
    assert set(updated.shared_with) == {"collaborator", "helper"}

    owner_view = registry.list_records_for_user("owner-1", UserRole.USER)
    assert {record.paper_id for record in owner_view} == {"paper-1", "paper-2"}

    collaborator_view = registry.list_records_for_user("helper", UserRole.USER)
    assert [record.paper_id for record in collaborator_view] == ["paper-2"]

    assert registry.get_for_user("paper-1", "helper", UserRole.USER) is None
    assert registry.get_for_user("paper-2", "helper", UserRole.USER) is not None

    assert registry.accessible_paper_ids("helper", UserRole.USER) == ["paper-2"]
