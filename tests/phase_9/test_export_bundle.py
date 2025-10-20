import json
import zipfile
from dataclasses import dataclass

from backend.app.export.bundle import ExportBundleBuilder
from backend.app.export.models import ShareExportFilters, ShareExportRequest
from backend.app.ui.service import GraphEdge, GraphNode, GraphView


@dataclass
class _StaticGraphProvider:
    graph: GraphView

    def fetch(self, filters: ShareExportFilters) -> GraphView:  # pragma: no cover - passthrough
        return self.graph


def _graph() -> GraphView:
    nodes = [
        GraphNode(
            id="node-b",
            label="Second Node",
            type="method",
            aliases=["Node Beta"],
            times_seen=5,
            section_distribution={"Results": 3},
            source_document_ids=["doc-2"],
        ),
        GraphNode(
            id="node-a",
            label="First Node",
            type="dataset",
            aliases=["Node Alpha"],
            times_seen=8,
            section_distribution={"Methods": 4},
            source_document_ids=["doc-1"],
        ),
        GraphNode(
            id="node-c",
            label="Third Node",
            type="metric",
            aliases=["Node Gamma"],
            times_seen=2,
            section_distribution={"Abstract": 1},
            source_document_ids=["doc-3"],
        ),
    ]
    edges = [
        GraphEdge(
            id="edge-2",
            source="node-b",
            target="node-a",
            relation="evaluates",
            relation_verbatim="evaluates",
            confidence=0.91,
            times_seen=3,
            attributes={"section": "Results"},
            evidence={
                "doc_id": "doc-2",
                "element_id": "el-2",
                "text_span": {"start": 4, "end": 42},
                "full_sentence": "Node B evaluates Node A.",
            },
            conflicting=False,
            created_at="2024-04-01T12:00:00Z",
        ),
        GraphEdge(
            id="edge-1",
            source="node-a",
            target="node-c",
            relation="reports",
            relation_verbatim="reports",
            confidence=0.83,
            times_seen=2,
            attributes={"section": "Discussion"},
            evidence={
                "doc_id": "doc-1",
                "element_id": "el-1",
                "text_span": {"start": 10, "end": 64},
                "full_sentence": "Node A reports findings about Node C.",
            },
            conflicting=False,
            created_at="2024-03-31T10:00:00Z",
        ),
    ]
    return GraphView(nodes=nodes, edges=edges)


def _request() -> ShareExportRequest:
    return ShareExportRequest(
        filters=ShareExportFilters(
            min_confidence=0.5,
            relations=("evaluates", "reports"),
            sections=("Results",),
            papers=(),
            include_co_mentions=False,
        ),
        include_snippets=True,
        truncate_snippets=False,
        requested_by="tests",
        pipeline_version="test-suite",
    )


def test_bundle_preserves_graph_iteration_order(tmp_path):
    graph = _graph()
    provider = _StaticGraphProvider(graph=graph)
    builder = ExportBundleBuilder(
        graph_provider=provider,
        output_dir=tmp_path,
        pipeline_version="test-suite",
    )

    bundle = builder.build(_request())

    with zipfile.ZipFile(bundle.archive_path, "r") as archive:
        graph_json = json.loads(archive.read("graph.json"))

    node_ids = [node["id"] for node in graph_json["nodes"]]
    edge_ids = [edge["id"] for edge in graph_json["edges"]]

    assert node_ids[:3] == ["node-b", "node-a", "node-c"]
    assert edge_ids[:2] == ["edge-2", "edge-1"]
