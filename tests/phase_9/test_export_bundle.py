from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.app.export.bundle import ExportBundleBuilder
from backend.app.export.models import ShareExportFilters, ShareExportRequest
from backend.app.ui.service import GraphEdge, GraphNode, GraphView


@dataclass
class _StaticGraphProvider:
    graph: GraphView

    def fetch(self, _filters: ShareExportFilters) -> GraphView:
        return self.graph


def _share_request() -> ShareExportRequest:
    return ShareExportRequest(
        filters=ShareExportFilters(
            min_confidence=0.4,
            relations=("uses",),
            sections=("Results",),
            papers=("doc-42",),
            include_co_mentions=False,
        ),
        include_snippets=True,
        truncate_snippets=False,
        requested_by="tests",
        pipeline_version="1.0.0",
    )


def test_bundle_preserves_graph_order(tmp_path: Path) -> None:
    node_primary = GraphNode(
        id="zeta-method",
        label="Zeta Method",
        type="method",
        aliases=["Zeta"],
        times_seen=5,
        section_distribution={"Results": 3},
        source_document_ids=["doc-42"],
    )
    node_secondary = GraphNode(
        id="alpha-dataset",
        label="Alpha Dataset",
        type="dataset",
        aliases=["AlphaData"],
        times_seen=2,
        section_distribution={"Methods": 1},
        source_document_ids=["doc-24"],
    )
    edge_primary = GraphEdge(
        id="edge-b",
        source=node_primary.id,
        target=node_secondary.id,
        relation="uses",
        relation_verbatim="uses",
        confidence=0.91,
        times_seen=2,
        attributes={"section": "Results"},
        evidence={
            "doc_id": "doc-42",
            "element_id": "element-1",
            "text_span": {"start": 10, "end": 40},
            "full_sentence": "Zeta Method uses Alpha Dataset for evaluation.",
        },
        conflicting=False,
        created_at="2025-01-01T00:00:00Z",
    )
    edge_secondary = GraphEdge(
        id="edge-a",
        source=node_secondary.id,
        target=node_primary.id,
        relation="supports",
        relation_verbatim="supports",
        confidence=0.75,
        times_seen=1,
        attributes={"section": "Discussion"},
        evidence={
            "doc_id": "doc-24",
            "element_id": "element-2",
            "text_span": {"start": 5, "end": 32},
            "full_sentence": "Alpha Dataset supports conclusions about Zeta Method.",
        },
        conflicting=False,
        created_at="2025-01-02T00:00:00Z",
    )
    graph = GraphView(
        nodes=[node_primary, node_secondary],
        edges=[edge_primary, edge_secondary],
    )
    builder = ExportBundleBuilder(
        graph_provider=_StaticGraphProvider(graph=graph),
        output_dir=tmp_path,
        pipeline_version="1.0.0",
        clock=lambda: datetime(2025, 5, 5, tzinfo=timezone.utc),
    )

    bundle = builder.build(_share_request())

    with zipfile.ZipFile(bundle.archive_path, "r") as archive:
        graph_json = json.loads(archive.read("graph.json"))

    assert [entry["id"] for entry in graph_json["nodes"][:2]] == [
        node_primary.id,
        node_secondary.id,
    ]
    assert [entry["id"] for entry in graph_json["edges"][:2]] == [
        edge_primary.id,
        edge_secondary.id,
    ]
