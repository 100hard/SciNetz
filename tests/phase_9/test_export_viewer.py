from __future__ import annotations

import json

from backend.app.export.viewer import VISUALIZATION_NODE_LIMIT, render_share_html


def _extract_graph_payload(html: str) -> dict[str, object]:
    marker = "const GRAPH_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index("const DOWNLOAD_URL", start)
    json_blob = html[start:end].rsplit(";", 1)[0].strip()
    return json.loads(json_blob)


def test_render_share_html_injects_default_limit_when_missing() -> None:
    html = render_share_html({"nodes": [], "edges": [], "node_count": 0, "edge_count": 0})
    payload = _extract_graph_payload(html)
    assert payload["visualization_limit"] == VISUALIZATION_NODE_LIMIT

    constant_index = html.index("const VISUALIZATION_NODE_LIMIT")
    configured_index = html.index("const configuredLimit")
    assert constant_index < configured_index


def test_render_share_html_respects_positive_limit_values() -> None:
    html = render_share_html(
        {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "visualization_limit": "128",
        }
    )
    payload = _extract_graph_payload(html)
    assert payload["visualization_limit"] == 128


def test_render_share_html_ignores_invalid_limit_values() -> None:
    html = render_share_html(
        {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "visualization_limit": -5,
        }
    )
    payload = _extract_graph_payload(html)
    assert payload["visualization_limit"] == VISUALIZATION_NODE_LIMIT
