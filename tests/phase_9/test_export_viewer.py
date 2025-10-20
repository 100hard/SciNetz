from __future__ import annotations

import json
from datetime import datetime

from backend.app.export.viewer import VISUALIZATION_NODE_LIMIT, render_share_html


def _extract_graph_payload(html: str) -> dict[str, object]:
    marker = "const GRAPH_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    json_blob = html[start:end].strip()
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


def test_render_share_html_uses_full_width_graph_container() -> None:
    html = render_share_html({"nodes": [], "edges": [], "node_count": 0, "edge_count": 0})

    assert "#graph {\n        position: relative;\n        width: 100%;" in html
    assert '<section class="graph-summary">' in html
    assert "<aside" not in html


def test_render_share_html_uses_light_background_theme() -> None:
    html = render_share_html({"nodes": [], "edges": [], "node_count": 0, "edge_count": 0})

    assert "background: #f8fafc;" in html
    assert "#020617" not in html


def test_render_share_html_includes_zoom_wheel_handler() -> None:
    html = render_share_html({"nodes": [], "edges": [], "node_count": 0, "edge_count": 0})

    assert 'canvas.addEventListener("wheel"' in html
    assert "const ZOOM_LIMITS" in html


def test_render_share_html_includes_pointer_pan_handlers() -> None:
    html = render_share_html({"nodes": [], "edges": [], "node_count": 0, "edge_count": 0})

    assert 'canvas.addEventListener("pointerdown"' in html
    assert 'window.addEventListener("pointermove"' in html


def test_render_share_html_omits_download_bundle_link() -> None:
    html = render_share_html({"nodes": [], "edges": [], "node_count": 0, "edge_count": 0})

    assert "Download bundle" not in html
    assert "download-link" not in html


def test_render_share_html_marks_expiry_metadata() -> None:
    html = render_share_html(
        {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
        },
        expires_at=datetime.fromisoformat("2024-10-21T12:00:00"),
    )

    assert 'data-meta="expires"' in html


def test_render_share_html_escapes_script_breakout_sequences() -> None:
    html = render_share_html(
        {
            "nodes": [
                {
                    "id": "node-1",
                    "label": "Node with </script> sequence",
                    "type": "method",
                    "times_seen": 1,
                    "section_distribution": {},
                }
            ],
            "edges": [
                {
                    "id": "edge-1",
                    "source": "node-1",
                    "target": "node-1",
                    "relation": "related-to",
                    "confidence": 0.9,
                    "evidence": {"full_sentence": "Example with </script> and \u2028 lines"},
                }
            ],
            "node_count": 1,
            "edge_count": 1,
        }
    )

    assert "</script>" not in html.split("const GRAPH_DATA = ", 1)[1].split(";", 1)[0]
    assert "\\u2028" in html
