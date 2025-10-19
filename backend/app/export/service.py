"""Services for generating standalone HTML exports of the knowledge graph."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import networkx as nx

from backend.app.config import ExportConfig
from backend.app.ui.service import GraphEdge, GraphNode, GraphView, GraphViewService


class ExportSizeWarning(RuntimeError):
    """Raised when an export exceeds the configured warning threshold."""


class ExportSizeExceeded(RuntimeError):
    """Raised when an export exceeds the configured maximum size."""


@dataclass(frozen=True)
class ExportRequest:
    """Filter parameters supplied for generating an export."""

    relations: Sequence[str]
    min_confidence: float
    sections: Sequence[str]
    include_co_mentions: bool
    papers: Sequence[str]
    limit: int


@dataclass(frozen=True)
class ExportOptions:
    """Export behaviour flags controlling evidence handling."""

    include_snippets: bool
    truncate_snippets: bool


@dataclass(frozen=True)
class ExportBundle:
    """Container holding the generated export artefacts."""

    filename: str
    archive: bytes
    graph_data: Dict[str, object]
    metadata: Dict[str, object]


class GraphExportService:
    """Create standalone HTML bundles for filtered graph views."""

    def __init__(
        self,
        *,
        graph_service: GraphViewService,
        export_config: ExportConfig,
        pipeline_version: str,
        template_path: Path,
    ) -> None:
        self._graph_service = graph_service
        self._pipeline_version = pipeline_version
        self._template = template_path.read_text(encoding="utf-8")
        self._snippet_limit = int(export_config.snippet_truncate_length)
        self._warn_threshold = float(export_config.warn_threshold_mb)
        self._max_size = float(export_config.max_size_mb)

    def generate_bundle(self, request: ExportRequest, options: ExportOptions) -> ExportBundle:
        """Generate a zipped HTML export matching the provided filters.

        Args:
            request: Graph filters mirroring the UI API.
            options: Controls for snippet inclusion and truncation.

        Returns:
            ExportBundle: Packaged archive bytes plus metadata.

        Raises:
            ExportSizeWarning: When the estimated payload exceeds the warning
                threshold.
            ExportSizeExceeded: When the estimated payload exceeds the maximum
                allowed size.
        """

        view = self._graph_service.fetch_graph(
            relations=request.relations,
            min_confidence=request.min_confidence,
            sections=request.sections,
            include_co_mentions=request.include_co_mentions,
            papers=request.papers,
            limit=request.limit,
        )
        nodes_payload = [self._node_to_payload(node) for node in view.nodes]
        full_snippets: Dict[str, Dict[str, object]] = {}
        edges_payload = [
            self._edge_to_payload(edge, options, full_snippets) for edge in view.edges
        ]
        estimated_size = self._estimate_size_mb(nodes_payload, edges_payload)
        if estimated_size > self._max_size:
            msg = (
                "Export too large. Please apply stricter filters or paginate by paper. "
                f"(estimated {estimated_size:.2f} MB, limit {self._max_size:.0f} MB)"
            )
            raise ExportSizeExceeded(msg)
        if estimated_size > self._warn_threshold:
            msg = (
                "Export is large. Options:\n"
                f"  - Truncate snippets (first {self._snippet_limit} chars)\n"
                "  - Exclude snippets (just IDs)\n"
                "  - Filter to fewer papers"
            )
            raise ExportSizeWarning(msg)

        generated_at = datetime.now(tz=timezone.utc)
        metadata = self._build_metadata(
            view=view,
            request=request,
            options=options,
            estimated_size=estimated_size,
            generated_at=generated_at,
            full_snippets=full_snippets,
        )
        graph_data: Dict[str, object] = {
            "nodes": nodes_payload,
            "edges": edges_payload,
            "metadata": metadata,
        }
        graph_json = json.dumps(graph_data, ensure_ascii=False, indent=2)
        graphml = self._build_graphml(nodes_payload, edges_payload, metadata)
        readme = self._build_readme(metadata, bool(full_snippets))
        html = self._render_html(graph_json, metadata)
        archive_name = f"scinets_export_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
        archive_bytes = self._build_archive(
            archive_name=archive_name,
            html=html,
            graph_json=graph_json,
            graphml=graphml,
            readme=readme,
            full_snippets=full_snippets,
        )
        return ExportBundle(
            filename=f"{archive_name}.zip",
            archive=archive_bytes,
            graph_data=graph_data,
            metadata=metadata,
        )

    def _estimate_size_mb(
        self, nodes: Sequence[Mapping[str, object]], edges: Sequence[Mapping[str, object]]
    ) -> float:
        nodes_len = len(json.dumps(list(nodes), ensure_ascii=False))
        edges_len = len(json.dumps(list(edges), ensure_ascii=False))
        return (nodes_len + edges_len) / 1_000_000

    @staticmethod
    def _node_to_payload(node: GraphNode) -> Dict[str, object]:
        return {
            "id": node.id,
            "label": node.label,
            "type": node.type,
            "aliases": list(node.aliases),
            "times_seen": node.times_seen,
            "section_distribution": dict(node.section_distribution),
            "source_document_ids": list(node.source_document_ids),
        }

    def _edge_to_payload(
        self,
        edge: GraphEdge,
        options: ExportOptions,
        full_snippets: MutableMapping[str, Dict[str, object]],
    ) -> Dict[str, object]:
        evidence = self._normalise_evidence(edge.evidence)
        if options.include_snippets:
            snippet = evidence.get("full_sentence")
            if isinstance(snippet, str):
                if options.truncate_snippets and len(snippet) > self._snippet_limit:
                    truncated = snippet[: self._snippet_limit].rstrip()
                    evidence["full_sentence"] = f"{truncated}... [truncated]"
                    evidence["snippet_truncated"] = True
                    full_snippets[edge.id] = {
                        "edge_id": edge.id,
                        "doc_id": evidence.get("doc_id"),
                        "element_id": evidence.get("element_id"),
                        "full_sentence": snippet,
                    }
                else:
                    evidence["snippet_truncated"] = False
        else:
            evidence.pop("full_sentence", None)
        payload = {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "relation_verbatim": edge.relation_verbatim,
            "confidence": edge.confidence,
            "times_seen": edge.times_seen,
            "attributes": dict(edge.attributes),
            "evidence": evidence,
            "conflicting": edge.conflicting,
            "created_at": edge.created_at,
        }
        return payload

    @staticmethod
    def _normalise_evidence(evidence: Mapping[str, object]) -> Dict[str, object]:
        doc_id = evidence.get("doc_id")
        element_id = evidence.get("element_id")
        raw_span = evidence.get("text_span")
        start = 0
        end = 0
        if isinstance(raw_span, Mapping):
            start = int(raw_span.get("start", 0) or 0)
            end = int(raw_span.get("end", 0) or 0)
        else:
            start = int(evidence.get("text_span_start", 0) or 0)
            end = int(evidence.get("text_span_end", 0) or 0)
        payload: Dict[str, object] = {
            "doc_id": doc_id,
            "element_id": element_id,
            "text_span": {"start": start, "end": end},
        }
        if "full_sentence" in evidence and evidence["full_sentence"] is not None:
            payload["full_sentence"] = str(evidence["full_sentence"])
        return payload

    def _build_metadata(
        self,
        *,
        view: GraphView,
        request: ExportRequest,
        options: ExportOptions,
        estimated_size: float,
        generated_at: datetime,
        full_snippets: Mapping[str, Mapping[str, object]],
    ) -> Dict[str, object]:
        snippet_strategy = "full"
        if not options.include_snippets:
            snippet_strategy = "identifiers-only"
        elif options.truncate_snippets:
            snippet_strategy = "truncated"
        metadata: Dict[str, object] = {
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            "pipeline_version": self._pipeline_version,
            "node_count": view.node_count,
            "edge_count": view.edge_count,
            "estimated_size_mb": round(estimated_size, 3),
            "size_warn_threshold_mb": self._warn_threshold,
            "size_max_mb": self._max_size,
            "filters": {
                "relations": list(request.relations),
                "min_confidence": request.min_confidence,
                "sections": list(request.sections),
                "include_co_mentions": request.include_co_mentions,
                "papers": list(request.papers),
                "limit": request.limit,
            },
            "snippet_strategy": snippet_strategy,
            "snippet_truncate_length": self._snippet_limit,
            "includes_full_snippets_file": bool(full_snippets),
        }
        return metadata

    def _build_graphml(
        self,
        nodes: Sequence[Mapping[str, object]],
        edges: Sequence[Mapping[str, object]],
        metadata: Mapping[str, object],
    ) -> bytes:
        graph = nx.MultiDiGraph()
        graph.graph["pipeline_version"] = metadata.get("pipeline_version")
        graph.graph["generated_at"] = metadata.get("generated_at")
        for node in nodes:
            graph.add_node(
                node["id"],
                label=node.get("label"),
                type=node.get("type"),
                times_seen=node.get("times_seen"),
                aliases=json.dumps(node.get("aliases", []), ensure_ascii=False),
                section_distribution=json.dumps(
                    node.get("section_distribution", {}), ensure_ascii=False
                ),
                source_document_ids=json.dumps(
                    node.get("source_document_ids", []), ensure_ascii=False
                ),
            )
        for edge in edges:
            evidence_json = json.dumps(edge.get("evidence", {}), ensure_ascii=False)
            attributes_json = json.dumps(edge.get("attributes", {}), ensure_ascii=False)
            graph.add_edge(
                edge.get("source"),
                edge.get("target"),
                key=edge.get("id"),
                relation=edge.get("relation"),
                relation_verbatim=edge.get("relation_verbatim"),
                confidence=edge.get("confidence"),
                times_seen=edge.get("times_seen"),
                conflicting=edge.get("conflicting"),
                created_at=edge.get("created_at"),
                evidence=evidence_json,
                attributes=attributes_json,
            )
        buffer = io.BytesIO()
        nx.write_graphml(graph, buffer)
        return buffer.getvalue()

    def _build_readme(self, metadata: Mapping[str, object], has_full_snippets: bool) -> str:
        lines = [
            "# SciNets Graph Export",
            "",
            "This bundle contains a standalone graph visualisation that can be viewed",
            "offline. Use the controls in `index.html` to explore the relationships",
            "between extracted entities.",
            "",
            f"Generated at: {metadata.get('generated_at')}",
            f"Pipeline version: {metadata.get('pipeline_version')}",
            "",
            "## Filters",
            "",
        ]
        filters = metadata.get("filters", {})
        lines.extend(
            [
                f"- Relations: {', '.join(filters.get('relations', [])) or 'All'}",
                f"- Minimum confidence: {filters.get('min_confidence')}",
                f"- Sections: {', '.join(filters.get('sections', [])) or 'All'}",
                f"- Include co-mentions: {filters.get('include_co_mentions')}",
                f"- Papers: {', '.join(filters.get('papers', [])) or 'All'}",
                f"- Result limit: {filters.get('limit')}",
            ]
        )
        lines.extend(
            [
                "",
                "## Files",
                "",
                "- `index.html`: Standalone interactive visualisation",
                "- `graph.json`: Raw nodes and edges",
                "- `graph.graphml`: NetworkX compatible graph representation",
                "- `README_export.md`: This file",
            ]
        )
        if has_full_snippets:
            lines.append("- `full_snippets.json`: Full evidence sentences for truncated snippets")
        lines.extend(
            [
                "",
                "Snippet strategy: " + str(metadata.get("snippet_strategy")),
                "Estimated export size: " + str(metadata.get("estimated_size_mb")) + " MB",
                "",
                "Open `index.html` in a modern browser to explore the graph.",
            ]
        )
        return "\n".join(lines)

    def _render_html(self, graph_json: str, metadata: Mapping[str, object]) -> str:
        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
        return (
            self._template.replace("{{GRAPH_JSON}}", graph_json)
            .replace("{{METADATA_JSON}}", metadata_json)
        )

    def _build_archive(
        self,
        *,
        archive_name: str,
        html: str,
        graph_json: str,
        graphml: bytes,
        readme: str,
        full_snippets: Mapping[str, Mapping[str, object]],
    ) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{archive_name}/index.html", html)
            zf.writestr(f"{archive_name}/graph.json", graph_json)
            zf.writestr(f"{archive_name}/graph.graphml", graphml)
            zf.writestr(f"{archive_name}/README_export.md", readme)
            if full_snippets:
                zf.writestr(
                    f"{archive_name}/full_snippets.json",
                    json.dumps(full_snippets, ensure_ascii=False, indent=2),
                )
        return buffer.getvalue()
