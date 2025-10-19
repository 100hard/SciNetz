from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import networkx as nx
import pytest

from backend.app.config import ExportConfig
from backend.app.export import (
    ExportOptions,
    ExportRequest,
    ExportSizeExceeded,
    ExportSizeWarning,
    GraphExportService,
)
from backend.app.ui.service import GraphEdge, GraphNode, GraphView


@dataclass
class _StubGraphViewService:
    """In-memory stub returning a fixed graph view."""

    view: GraphView

    def fetch_graph(
        self,
        *,
        relations: Sequence[str],
        min_confidence: float,
        sections: Sequence[str],
        include_co_mentions: bool,
        papers: Sequence[str],
        limit: int,
    ) -> GraphView:
        self.last_request = {
            "relations": relations,
            "min_confidence": min_confidence,
            "sections": sections,
            "include_co_mentions": include_co_mentions,
            "papers": papers,
            "limit": limit,
        }
        return self.view


def _build_view(snippet: str) -> GraphView:
    node_a = GraphNode(
        id="node-a",
        label="Model Alpha",
        type="model",
        aliases=["Alpha"],
        times_seen=3,
        section_distribution={"Results": 2, "Methods": 1},
        source_document_ids=["paper-1"],
    )
    node_b = GraphNode(
        id="node-b",
        label="Dataset Beta",
        type="dataset",
        aliases=["Beta"],
        times_seen=1,
        section_distribution={"Results": 1},
        source_document_ids=["paper-1"],
    )
    evidence: Dict[str, Any] = {
        "doc_id": "paper-1",
        "element_id": "el-123",
        "text_span": {"start": 5, "end": 120},
        "full_sentence": snippet,
    }
    edge = GraphEdge(
        id="edge-1",
        source=node_a.id,
        target=node_b.id,
        relation="uses",
        relation_verbatim="uses",
        confidence=0.82,
        times_seen=2,
        attributes={"section": "Results"},
        evidence=evidence,
        conflicting=False,
        created_at="2024-01-01T00:00:00Z",
    )
    return GraphView(nodes=[node_a, node_b], edges=[edge])


def _make_service(snippet: str, config: ExportConfig) -> GraphExportService:
    view = _build_view(snippet)
    stub = _StubGraphViewService(view=view)
    template_path = Path(__file__).resolve().parents[2] / "export" / "scinets_view.html"
    service = GraphExportService(
        graph_service=stub,
        export_config=config,
        pipeline_version="9.0.0",
        template_path=template_path,
    )
    return service


def _make_request() -> ExportRequest:
    return ExportRequest(
        relations=["uses"],
        min_confidence=0.5,
        sections=["Results"],
        include_co_mentions=False,
        papers=["paper-1"],
        limit=100,
    )


def test_generate_bundle_creates_expected_artifacts() -> None:
    snippet = "Model Alpha uses Dataset Beta to improve results." * 4
    config = ExportConfig(max_size_mb=5, warn_threshold_mb=3, snippet_truncate_length=40)
    service = _make_service(snippet, config)

    bundle = service.generate_bundle(
        _make_request(), ExportOptions(include_snippets=True, truncate_snippets=True)
    )

    archive = zipfile.ZipFile(io.BytesIO(bundle.archive))
    names = set(archive.namelist())
    index_name = next(name for name in names if name.endswith("index.html"))
    graph_name = next(name for name in names if name.endswith("graph.json"))
    graphml_name = next(name for name in names if name.endswith("graph.graphml"))
    readme_name = next(name for name in names if name.endswith("README_export.md"))
    full_snippets_name = next(name for name in names if name.endswith("full_snippets.json"))

    html = archive.read(index_name).decode("utf-8")
    assert "cytoscape.min.js" in html
    graph = json.loads(archive.read(graph_name))
    edge_payload = graph["edges"][0]
    assert edge_payload["evidence"]["snippet_truncated"] is True
    assert edge_payload["evidence"]["full_sentence"].endswith("[truncated]")

    full_snippets = json.loads(archive.read(full_snippets_name))
    assert full_snippets["edge-1"]["full_sentence"].startswith("Model Alpha")

    graphml_bytes = archive.read(graphml_name)
    network = nx.read_graphml(io.BytesIO(graphml_bytes))
    assert len(network.nodes) == 2
    assert len(network.edges) == 1

    readme = archive.read(readme_name).decode("utf-8")
    assert "Pipeline version: 9.0.0" in readme
    assert bundle.metadata["snippet_strategy"] == "truncated"


def test_generate_bundle_warns_when_estimate_exceeds_threshold() -> None:
    snippet = "Y" * 1_500_000
    config = ExportConfig(max_size_mb=5, warn_threshold_mb=1, snippet_truncate_length=40)
    service = _make_service(snippet, config)

    with pytest.raises(ExportSizeWarning):
        service.generate_bundle(
            _make_request(), ExportOptions(include_snippets=True, truncate_snippets=False)
        )


def test_generate_bundle_errors_when_exceeding_maximum_size() -> None:
    snippet = "X" * 2_200_000
    config = ExportConfig(max_size_mb=1, warn_threshold_mb=1, snippet_truncate_length=40)
    service = _make_service(snippet, config)

    with pytest.raises(ExportSizeExceeded):
        service.generate_bundle(
            _make_request(), ExportOptions(include_snippets=True, truncate_snippets=False)
        )


def test_generate_bundle_can_strip_snippets() -> None:
    snippet = "Model Alpha uses Dataset Beta."
    config = ExportConfig(max_size_mb=5, warn_threshold_mb=3, snippet_truncate_length=40)
    service = _make_service(snippet, config)

    bundle = service.generate_bundle(
        _make_request(), ExportOptions(include_snippets=False, truncate_snippets=False)
    )

    edge = bundle.graph_data["edges"][0]
    assert "full_sentence" not in edge["evidence"]
    assert edge["evidence"]["text_span"] == {"start": 5, "end": 120}
    assert bundle.metadata["snippet_strategy"] == "identifiers-only"
