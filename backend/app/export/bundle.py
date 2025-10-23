"""Bundle builder for shareable export links."""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence

from backend.app.export.models import ExportBundle, ShareExportFilters, ShareExportRequest
from backend.app.export.viewer import VISUALIZATION_NODE_LIMIT, render_share_html
from backend.app.ui.service import GraphEdge, GraphNode, GraphView, GraphViewService

LOGGER = logging.getLogger(__name__)


class GraphProviderProtocol(Protocol):
    """Protocol describing a graph provider for exports."""

    def fetch(
        self, filters: ShareExportFilters, *, allowed_papers: Optional[Sequence[str]] = None
    ) -> GraphView:
        """Return a graph for the provided filters."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GraphViewExportProvider(GraphProviderProtocol):
    """Graph provider powered by the existing UI graph service."""

    def __init__(self, service: GraphViewService) -> None:
        self._service = service

    def fetch(
        self, filters: ShareExportFilters, *, allowed_papers: Optional[Sequence[str]] = None
    ) -> GraphView:
        kwargs: Dict[str, object] = {}
        if allowed_papers is not None:
            kwargs["allowed_papers"] = allowed_papers
        view = self._service.fetch_graph(
            relations=filters.relations,
            min_confidence=filters.min_confidence,
            sections=filters.sections,
            include_co_mentions=filters.include_co_mentions,
            papers=filters.papers,
            **kwargs,
        )
        return view


class ExportBundleBuilder:
    """Construct deterministic export bundles ready for storage."""

    def __init__(
        self,
        *,
        graph_provider: GraphProviderProtocol,
        output_dir: Path,
        pipeline_version: str,
        clock: Callable[[], datetime] = _utc_now,
        snippet_truncate_length: int = 200,
    ) -> None:
        self._graph_provider = graph_provider
        self._output_dir = output_dir
        self._pipeline_version = pipeline_version
        self._clock = clock
        self._snippet_truncate_length = snippet_truncate_length
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, request: ShareExportRequest) -> ExportBundle:
        """Build an export bundle for the given request."""

        fetch_kwargs = {}
        if request.allowed_papers is not None:
            fetch_kwargs["allowed_papers"] = request.allowed_papers
        graph = self._graph_provider.fetch(request.filters, **fetch_kwargs)
        created_at = self._clock().astimezone(timezone.utc)
        sanitized = self._serialise_graph(graph, request)
        files: Dict[str, bytes] = {
            "graph.json": json.dumps(sanitized, indent=2, sort_keys=True).encode("utf-8"),
            "graph.graphml": self._render_graphml(graph).encode("utf-8"),
            "scinets_view.html": render_share_html(sanitized).encode("utf-8"),
            "README_export.md": self._render_readme(request, created_at).encode("utf-8"),
        }
        file_metadata = {
            name: {
                "sha256": sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in files.items()
        }
        manifest = {
            "pipeline_version": self._pipeline_version,
            "created_at": created_at.isoformat(),
            "filters": request.filters.to_dict(),
            "include_snippets": request.include_snippets,
            "truncate_snippets": request.truncate_snippets,
            "files": file_metadata,
        }
        files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        archive_name = f"export_{int(created_at.timestamp())}.zip"
        archive_path = self._output_dir / archive_name
        self._write_zip(archive_path, files)
        archive_bytes = archive_path.read_bytes()
        archive_sha = sha256(archive_bytes).hexdigest()
        bundle = ExportBundle(
            archive_path=archive_path,
            size_bytes=len(archive_bytes),
            sha256=archive_sha,
            manifest=manifest,
        )
        return bundle

    def _serialise_graph(self, graph: GraphView, request: ShareExportRequest) -> Dict[str, object]:
        nodes = [self._normalise_node(node) for node in graph.nodes]
        edges = [self._normalise_edge(edge, request) for edge in graph.edges]
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "pipeline_version": self._pipeline_version,
            "visualization_limit": VISUALIZATION_NODE_LIMIT,
        }

    @staticmethod
    def _normalise_node(node: GraphNode) -> Dict[str, object]:
        payload = asdict(node)
        payload["aliases"] = list(node.aliases)
        payload["section_distribution"] = dict(node.section_distribution)
        payload["source_document_ids"] = list(node.source_document_ids)
        return payload

    def _normalise_edge(self, edge: GraphEdge, request: ShareExportRequest) -> Dict[str, object]:
        payload = asdict(edge)
        payload["attributes"] = dict(edge.attributes)
        evidence = dict(edge.evidence or {})
        if not request.include_snippets:
            evidence.pop("full_sentence", None)
        elif request.truncate_snippets and isinstance(evidence.get("full_sentence"), str):
            full_sentence = evidence["full_sentence"]
            truncated = self._truncate(full_sentence)
            evidence["full_sentence"] = truncated
        payload["evidence"] = evidence
        payload["conflicting"] = bool(edge.conflicting)
        return payload

    def _truncate(self, text: str) -> str:
        limit = max(self._snippet_truncate_length, 1)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "... [truncated]"

    @staticmethod
    def _render_graphml(graph: GraphView) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">",
            '  <graph edgedefault="directed">',
        ]
        for node in sorted(graph.nodes, key=lambda n: n.id):
            lines.append(f'    <node id="{node.id}">')
            lines.append(f'      <data key="label">{ExportBundleBuilder._escape(node.label)}</data>')
            if node.type:
                lines.append(f'      <data key="type">{ExportBundleBuilder._escape(node.type)}</data>')
            lines.append("    </node>")
        for edge in sorted(graph.edges, key=lambda e: e.id):
            lines.append(
                f'    <edge id="{edge.id}" source="{edge.source}" target="{edge.target}">'
            )
            lines.append(
                f'      <data key="relation">{ExportBundleBuilder._escape(edge.relation)}</data>'
            )
            lines.append(
                f'      <data key="confidence">{edge.confidence:.4f}</data>'
            )
            lines.append("    </edge>")
        lines.append("  </graph>")
        lines.append("</graphml>")
        return "\n".join(lines)

    def _render_readme(self, request: ShareExportRequest, created_at: datetime) -> str:
        filters = json.dumps(request.filters.to_dict(), indent=2, sort_keys=True)
        return f"""# SciNets Export Bundle

- Generated at: {created_at.isoformat()}
- Pipeline version: {self._pipeline_version}
- Include snippets: {request.include_snippets}
- Truncate snippets: {request.truncate_snippets}

## Applied filters

```json
{filters}
```
"""

    @staticmethod
    def _escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _write_zip(path: Path, files: Mapping[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files.keys()):
                archive.writestr(name, files[name], compress_type=zipfile.ZIP_DEFLATED)
