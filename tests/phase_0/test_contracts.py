from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from backend.app.contracts import (
    Edge,
    Evidence,
    Node,
    PaperMetadata,
    ParsedElement,
    TextSpan,
    Triplet,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "sample_payload.json"


@pytest.fixture()
def sample_payload() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_parsed_element_contract(sample_payload: dict) -> None:
    raw_element = sample_payload["elements"][0]
    element = ParsedElement(**raw_element)
    assert element.content_hash
    assert element.end_char > element.start_char


def test_paper_metadata_required_fields(sample_payload: dict) -> None:
    metadata = PaperMetadata(**sample_payload["paper_metadata"])
    assert metadata.doc_id == "doc123"
    assert "Doe, Jane" in metadata.authors


def test_triplet_includes_pipeline_version(sample_payload: dict) -> None:
    triplet = Triplet(**sample_payload["triplets"][0])
    assert triplet.pipeline_version == "1.0.0"
    assert isinstance(triplet.evidence.text_span, TextSpan)
    assert triplet.evidence.full_sentence is not None


def test_node_section_distribution_preserved(sample_payload: dict) -> None:
    node = Node(**sample_payload["nodes"][0])
    assert node.section_distribution["Introduction"] == 1


def test_edge_contract_includes_temporal_fields(sample_payload: dict) -> None:
    edge = Edge(**sample_payload["edges"][0])
    assert edge.pipeline_version == "1.0.0"
    assert edge.created_at == datetime.fromisoformat("2025-01-01T00:00:00+00:00")
    assert isinstance(edge.evidence, Evidence)
