"""Utilities for rendering shareable export HTML pages."""

from __future__ import annotations

import html
import json
from typing import Final
from datetime import datetime
from typing import Mapping, MutableMapping


VISUALIZATION_NODE_LIMIT: Final[int] = 200


def _escape_script_value(value: str) -> str:
    """Escape a JSON string so it is safe for inline ``<script>`` embedding.

    Args:
        value: Raw JSON string produced by ``json.dumps``.

    Returns:
        The escaped string that will not prematurely close the surrounding script
        tag and preserves line separator characters.
    """

    return (
        value.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_share_html(
    graph_data: Mapping[str, object],
    *,
    download_url: str | None = None,
    expires_at: datetime | None = None,
) -> str:
    """Render an interactive HTML view for a shared graph export."""

    normalised_graph = dict(graph_data)
    _ = download_url  # Preserve signature for backward compatibility.
    limit_value = normalised_graph.get("visualization_limit")
    limit: int
    try:
        limit_candidate = int(limit_value) if limit_value is not None else VISUALIZATION_NODE_LIMIT
    except (TypeError, ValueError):
        limit = VISUALIZATION_NODE_LIMIT
    else:
        limit = limit_candidate if limit_candidate > 0 else VISUALIZATION_NODE_LIMIT
    normalised_graph["visualization_limit"] = limit

    payload = _escape_script_value(
        json.dumps(normalised_graph, separators=(",", ":"), ensure_ascii=False)
    )
    bundle_info: MutableMapping[str, str] = {
        "Pipeline version": str(normalised_graph.get("pipeline_version", "unknown")),
        "Nodes": str(normalised_graph.get("node_count", 0)),
        "Edges": str(normalised_graph.get("edge_count", 0)),
    }
    if expires_at is not None:
        bundle_info["Link expires"] = expires_at.isoformat()
    else:
        bundle_info["Link expires"] = "Never"

    def _format_bundle_item(key: str, value: str) -> str:
        data_attr = " data-meta=\"expires\"" if key.lower() == "link expires" else ""
        return (
            f"<li{data_attr}><strong>{html.escape(key)}:</strong> {html.escape(value)}</li>"
        )

    bundle_lines = "\n            ".join(
        _format_bundle_item(key, value) for key, value in bundle_info.items()
    )
    download_link_markup = ""
    if download_url:
        escaped_url = html.escape(download_url, quote=True)
        download_link_markup = (
            "<div class=\"download-actions\">"
            f"<a id=\"download-bundle\" class=\"download-bundle\" href=\"{escaped_url}\" download>"
            "Download bundle (.zip)</a>"
            "</div>"
        )

    expires_js = (
        _escape_script_value(json.dumps(expires_at.isoformat(), ensure_ascii=False))
        if expires_at
        else "null"
    )

    html_template = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'self';" />
    <title>SciNets Shared Graph</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <style>
      :root {{
        color-scheme: light dark;
      }}
      body {{
        margin: 0;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #0f172a;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        background: #f8fafc;
      }}
      header {{
        padding: 1.5rem 2rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        background: #ffffff;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
      }}
      header h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 600;
      }}
      header p {{
        margin: 0;
        color: rgba(15, 23, 42, 0.65);
      }}
      .download-actions {{
        display: inline-flex;
      }}
      .download-bundle {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        border-radius: 9999px;
        border: 1px solid rgba(37, 99, 235, 0.28);
        background: #2563eb;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }}
      .download-bundle:hover {{
        transform: translateY(-1px);
        box-shadow: 0 18px 30px rgba(37, 99, 235, 0.35);
      }}
      .download-bundle:focus {{
        outline: 3px solid rgba(37, 99, 235, 0.35);
        outline-offset: 2px;
      }}
      main {{
        flex: 1;
        padding: 1.5rem;
      }}
      #graph {{
        position: relative;
        width: 100%;
        min-height: 640px;
        background: #ffffff;
        border-radius: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
        overflow: hidden;
        display: flex;
        align-items: stretch;
      }}
      #graph-canvas {{
        position: relative;
        flex: 1;
        min-height: inherit;
      }}
      #graph canvas {{
        position: absolute;
        inset: 0;
      }}
      .graph-summary {{
        padding: 0 1.5rem 2rem;
        display: grid;
        gap: 1.5rem;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }}
      .graph-summary .bundle-card {{
        background: #ffffff;
        border-radius: 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 1.1rem;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
      }}
      .graph-summary h2 {{
        margin: 0 0 0.75rem;
        font-size: 1.05rem;
      }}
      .graph-summary ul {{
        list-style: none;
        margin: 0;
        padding: 0;
      }}
      .graph-summary li {{
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        font-size: 0.95rem;
      }}
      .graph-summary li:last-child {{
        border-bottom: none;
      }}
      #details {{
        background: rgba(148, 163, 184, 0.12);
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        font-size: 0.9rem;
        line-height: 1.5;
      }}
      #details pre {{
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
      }}
      @media (max-width: 1080px) {{
        main {{
          padding: 1rem;
        }}
        #graph {{
          min-height: 560px;
        }}
        .graph-summary {{
          padding: 0 1rem 1.5rem;
        }}
      }}
      @media (max-width: 768px) {{
        #graph {{
          min-height: 540px;
        }}
        .graph-summary {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    <script>
      const GRAPH_DATA = __GRAPH_DATA__;
    </script>
  </head>
  <body>
    <header>
      <h1>SciNets Shared Graph</h1>
      <p>Interactive snapshot exported from the SciNets knowledge graph.</p>
      __DOWNLOAD_LINK__
    </header>
    <main>
      <section id="graph"></section>
    </main>
    <section class="graph-summary">
      <div class="bundle-card">
        <h2>Graph details</h2>
        <ul id="bundle-info">
          __BUNDLE_LINES__
        </ul>
      </div>
      <section id="details">
        <h3>Details</h3>
        <p>Select a node or edge to see contextual information.</p>
      </section>
    </section>
    <script>
      (function () {{
        const VISUALIZATION_NODE_LIMIT = __NODE_LIMIT__;
        const graphData = GRAPH_DATA || {{}};
        const expiresAt = __EXPIRES_AT__;
        if (expiresAt) {{
          const expiryItem = document.querySelector("li[data-meta='expires']");
          if (!expiryItem) {{
            const list = document.getElementById("bundle-info");
            if (list) {{
              const item = document.createElement("li");
              item.dataset.meta = "expires";
              item.innerHTML = "<strong>Link expires:</strong> " + expiresAt;
              list.appendChild(item);
            }}
          }}
        }}

        const container = document.getElementById("graph");
        if (!container) {{
          return;
        }}

        const nodesRaw = Array.isArray(graphData.nodes) ? graphData.nodes : [];
        const edgesRaw = Array.isArray(graphData.edges) ? graphData.edges : [];
        const totalNodeCount = nodesRaw.length;
        const totalEdgeCount = edgesRaw.length;
        const configuredLimit = Number.isFinite(Number(graphData.visualization_limit))
          ? Math.max(1, Math.floor(Number(graphData.visualization_limit)))
          : VISUALIZATION_NODE_LIMIT;
        const nodeLimit = totalNodeCount > 0 ? Math.min(configuredLimit, totalNodeCount) : 0;
        const nodesForLayout = nodesRaw.slice(0, nodeLimit);
        if (nodesForLayout.length === 0) {{
          container.innerHTML = "<p style='padding:1rem;'>No nodes available in this export.</p>";
          return;
        }}

        const NODE_STROKE_COLOR = "#0f172a";
        const NODE_LABEL_TEXT_COLOR = "#0f172a";
        const NODE_LABEL_FONT_SIZE = 13;
        const NODE_LABEL_FONT_WEIGHT = 700;
        const NODE_LABEL_LINE_HEIGHT = 16;
        const NODE_LABEL_VERTICAL_PADDING = 10;
        const MIN_NODE_RADIUS = 30;
        const NODE_RADIUS_SCALE = 1.35;
        const EDGE_MIN_STROKE_WIDTH = 1.6;
        const EDGE_MAX_STROKE_WIDTH = 3.6;
        const EDGE_MIN_OPACITY = 0.55;
        const EDGE_MAX_OPACITY = 0.95;
        const EDGE_LABEL_FONT_SIZE = 10;
        const EDGE_LABEL_FONT_WEIGHT = 600;
        const EDGE_LABEL_MIN_WIDTH = 72;
        const EDGE_LABEL_MAX_WIDTH = 220;
        const EDGE_LABEL_HORIZONTAL_PADDING = 14;
        const EDGE_LABEL_RECT_HEIGHT = 26;
        const EDGE_DARKEN_BASE = "#0f172a";
        const EDGE_LABEL_LIGHTEN_TARGET = "#f8fafc";
        const LAYOUT_AREA_SCALE = 0.74;
        const LAYOUT_ATTRACTION_STRENGTH = 0.044;
        const LAYOUT_REPULSION_STRENGTH = 0.32;
        const LAYOUT_CROSS_COMPONENT_PULL = 0.032;
        const LAYOUT_ANCHOR_GRAVITY = 0.024;
        const LAYOUT_CENTER_GRAVITY = 0.03;
        const LAYOUT_TEMPERATURE_DIVISOR = 2.45;
        const LAYOUT_COOLING_FACTOR = 0.89;

        const TYPE_COLOR_MAP = {{
          method: "#2563eb",
          methods: "#2563eb",
          dataset: "#16a34a",
          datasets: "#16a34a",
          metric: "#7c3aed",
          metrics: "#7c3aed",
          task: "#ea580c",
          tasks: "#ea580c",
          entity: "#0ea5e9",
        }};

        const RELATION_COLOR_MAP = {{
          evaluates: "#2563eb",
          uses: "#0ea5e9",
          reports: "#16a34a",
          improves: "#7c3aed",
          predicts: "#f97316",
          extends: "#db2777",
          "builds-upon": "#8b5cf6",
          "defined-as": "#f59e0b",
          "similar-to": "#0ea5e9",
          "related-to": "#14b8a6",
          "achieves-state-of-the-art-on": "#db2777",
        }};

        const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

        const hashColor = (value) => {{
          if (!value) {{
            return "hsl(198, 83%, 64%)";
          }}
          let hash = 0;
          for (let index = 0; index < value.length; index += 1) {{
            hash = (hash * 31 + value.charCodeAt(index)) % 360;
          }}
          const hue = Math.abs(hash);
          return `hsl(${{hue}}, 65%, 60%)`;
        }};

        const parseHexColor = (value) => {{
          const hex = value.replace("#", "").trim();
          if (hex.length === 3) {{
            const r = Number.parseInt(hex[0] + hex[0], 16);
            const g = Number.parseInt(hex[1] + hex[1], 16);
            const b = Number.parseInt(hex[2] + hex[2], 16);
            if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {{
              return null;
            }}
            return {{ r, g, b }};
          }}
          if (hex.length === 6) {{
            const r = Number.parseInt(hex.slice(0, 2), 16);
            const g = Number.parseInt(hex.slice(2, 4), 16);
            const b = Number.parseInt(hex.slice(4, 6), 16);
            if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {{
              return null;
            }}
            return {{ r, g, b }};
          }}
          return null;
        }};

        const parseRgbColor = (value) => {{
          const match = value.match(/rgba?\\(\\s*([\\d.]+)\\s*,\\s*([\\d.]+)\\s*,\\s*([\\d.]+)/i);
          if (!match) {{
            return null;
          }}
          const r = Math.min(255, Math.max(0, Number.parseFloat(match[1])));
          const g = Math.min(255, Math.max(0, Number.parseFloat(match[2])));
          const b = Math.min(255, Math.max(0, Number.parseFloat(match[3])));
          if ([r, g, b].some((channel) => Number.isNaN(channel))) {{
            return null;
          }}
          return {{ r, g, b }};
        }};

        const parseHslColor = (value) => {{
          const match = value.match(/hsla?\\(\\s*([\\d.+-]+)(deg|rad|turn)?\\s*,\\s*([\\d.+-]+)%\\s*,\\s*([\\d.+-]+)%/i);
          if (!match) {{
            return null;
          }}
          let hue = Number.parseFloat(match[1]);
          if (Number.isNaN(hue)) {{
            return null;
          }}
          const unit = (match[2] || "deg").toLowerCase();
          if (unit === "rad") {{
            hue = (hue * 180) / Math.PI;
          }} else if (unit === "turn") {{
            hue *= 360;
          }}
          const saturation = Math.min(100, Math.max(0, Number.parseFloat(match[3]))) / 100;
          const lightness = Math.min(100, Math.max(0, Number.parseFloat(match[4]))) / 100;
          if ([saturation, lightness].some((component) => Number.isNaN(component))) {{
            return null;
          }}
          const h = (((hue % 360) + 360) % 360) / 360;
          const q = lightness < 0.5 ? lightness * (1 + saturation) : lightness + saturation - lightness * saturation;
          const p = 2 * lightness - q;
          const hueToRgb = (t) => {{
            let temp = t;
            if (temp < 0) {{
              temp += 1;
            }}
            if (temp > 1) {{
              temp -= 1;
            }}
            if (temp < 1 / 6) {{
              return p + (q - p) * 6 * temp;
            }}
            if (temp < 1 / 2) {{
              return q;
            }}
            if (temp < 2 / 3) {{
              return p + (q - p) * (2 / 3 - temp) * 6;
            }}
            return p;
          }};
          const r = Math.round(hueToRgb(h + 1 / 3) * 255);
          const g = Math.round(hueToRgb(h) * 255);
          const b = Math.round(hueToRgb(h - 1 / 3) * 255);
          return {{ r, g, b }};
        }};

        const toRgbColor = (value) => {{
          const trimmed = (value || "").trim();
          const lower = trimmed.toLowerCase();
          if (lower.startsWith("#")) {{
            return parseHexColor(trimmed);
          }}
          if (lower.startsWith("rgb")) {{
            return parseRgbColor(trimmed);
          }}
          if (lower.startsWith("hsl")) {{
            return parseHslColor(trimmed);
          }}
          return null;
        }};

        const channelToLinear = (channel) => {{
          const normalized = channel / 255;
          if (normalized <= 0.03928) {{
            return normalized / 12.92;
          }}
          return ((normalized + 0.055) / 1.055) ** 2.4;
        }};

        const getRelativeLuminance = ({{
          r,
          g,
          b,
        }}) => {{
          const red = channelToLinear(r);
          const green = channelToLinear(g);
          const blue = channelToLinear(b);
          return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
        }};

        const getContrastingLabelColors = (fill) => {{
          const rgb = toRgbColor(fill);
          if (!rgb) {{
            return {{ color: NODE_LABEL_TEXT_COLOR, outline: "rgba(255, 255, 255, 0.78)" }};
          }}
          const luminance = getRelativeLuminance(rgb);
          if (luminance > 0.65) {{
            return {{ color: NODE_LABEL_TEXT_COLOR, outline: "rgba(15, 23, 42, 0.3)" }};
          }}
          return {{ color: NODE_LABEL_TEXT_COLOR, outline: "rgba(255, 255, 255, 0.78)" }};
        }};

        const createSeededGenerator = (seed) => {{
          let value = 0;
          for (let index = 0; index < seed.length; index += 1) {{
            value = (value * 31 + seed.charCodeAt(index)) >>> 0;
          }}
          if (value === 0) {{
            value = 0x9e3779b9;
          }}
          return () => {{
            value ^= value << 13;
            value ^= value >>> 17;
            value ^= value << 5;
            value >>>= 0;
            return (value & 0x0fffffff) / 0x10000000;
          }};
        }};

        const getNodeFill = (type) => {{
          if (!type) {{
            return hashColor(null);
          }}
          const key = type.toString().trim().toLowerCase();
          return TYPE_COLOR_MAP[key] || hashColor(type);
        }};

        const getNodeImportance = (node) => {{
          const provided = typeof node.importance === "number" ? node.importance : null;
          if (provided !== null && Number.isFinite(provided) && provided > 0) {{
            return provided;
          }}
          const sectionTotal = Object.values(node.section_distribution || {{}}).reduce(
            (accumulator, value) => accumulator + value,
            0,
          );
          const base = Math.max(node.times_seen || 0, 0);
          const fallback = base + sectionTotal;
          return Math.max(fallback, 1);
        }};

        const formatNodeLabel = (label) => {{
          const text = (label || "").toString();
          const words = text.split(/\\s+/).filter(Boolean);
          if (!words.length) {{
            return ["Unknown"];
          }}
          const lines = [];
          let current = "";
          const maxLength = 14;
          for (const word of words) {{
            const candidate = current ? `${{current}} ${{word}}` : word;
            if (candidate.length <= maxLength) {{
              current = candidate;
              continue;
            }}
            if (current) {{
              lines.push(current);
            }}
            if (word.length > maxLength) {{
              lines.push(`${{word.slice(0, maxLength - 1)}}…`);
              current = "";
            }} else {{
              current = word;
            }}
            if (lines.length === 2) {{
              break;
            }}
          }}
          if (lines.length < 2 && current) {{
            lines.push(current);
          }}
          if (lines.length === 0) {{
            lines.push(words[0]);
          }}
          return lines.slice(0, 3);
        }};

        const calculateNodeRadius = (node, degree, labelLines) => {{
          const importance = getNodeImportance(node);
          const importanceContribution = Math.log10(importance + 1) * 11;
          const degreeContribution = Math.sqrt(Math.max(degree, 1)) * 3.5;
          const longestLine = labelLines.reduce((acc, line) => Math.max(acc, line.length), 0);
          const approxCharWidth = NODE_LABEL_FONT_SIZE * 0.68;
          const horizontalRadius = longestLine > 0 ? (longestLine * approxCharWidth) / 2 : 0;
          const verticalRadius = (labelLines.length * NODE_LABEL_LINE_HEIGHT) / 2 + NODE_LABEL_VERTICAL_PADDING;
          const minimum = Math.max(MIN_NODE_RADIUS, horizontalRadius, verticalRadius);
          return (minimum + importanceContribution + degreeContribution) * NODE_RADIUS_SCALE;
        }};

        const getRelationColor = (relation) => {{
          const key = (relation || "").toString().trim().toLowerCase();
          return RELATION_COLOR_MAP[key] || "rgba(15, 23, 42, 0.6)";
        }};

        const estimateCollisionRadius = (node, degree) => {{
          const labelLines = formatNodeLabel(node.label || node.id || "");
          const radius = calculateNodeRadius(node, degree, labelLines);
          return radius + 6;
        }};

        const rgbToCss = (color) => `rgb(${{Math.round(color.r)}}, ${{Math.round(color.g)}}, ${{Math.round(color.b)}})`;

        const blendColors = (baseColor, mixColor, amount) => {{
          const base = toRgbColor(baseColor);
          const mix = toRgbColor(mixColor);
          if (!base || !mix) {{
            return null;
          }}
          const ratio = clamp(amount, 0, 1);
          const blended = {{
            r: base.r * (1 - ratio) + mix.r * ratio,
            g: base.g * (1 - ratio) + mix.g * ratio,
            b: base.b * (1 - ratio) + mix.b * ratio,
          }};
          return rgbToCss(blended);
        }};

        const drawRoundedRectPath = (context, x, y, width, height, radius) => {{
          const clampedRadius = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
          context.beginPath();
          context.moveTo(x + clampedRadius, y);
          context.lineTo(x + width - clampedRadius, y);
          context.quadraticCurveTo(x + width, y, x + width, y + clampedRadius);
          context.lineTo(x + width, y + height - clampedRadius);
          context.quadraticCurveTo(x + width, y + height, x + width - clampedRadius, y + height);
          context.lineTo(x + clampedRadius, y + height);
          context.quadraticCurveTo(x, y + height, x, y + height - clampedRadius);
          context.lineTo(x, y + clampedRadius);
          context.quadraticCurveTo(x, y, x + clampedRadius, y);
          context.closePath();
        }};

        const getEdgeStrokeColor = (baseColor, confidence) => {{
          const normalized = clamp(confidence, 0, 1);
          const blendAmount = 0.25 + normalized * 0.45;
          return blendColors(baseColor, EDGE_DARKEN_BASE, blendAmount) || baseColor;
        }};

        const getEdgeStrokeWidth = (confidence) => {{
          const normalized = clamp(confidence, 0, 1);
          const emphasis = Math.sqrt(normalized);
          return EDGE_MIN_STROKE_WIDTH + (EDGE_MAX_STROKE_WIDTH - EDGE_MIN_STROKE_WIDTH) * emphasis;
        }};

        const getEdgeStrokeOpacity = (confidence) => {{
          const normalized = clamp(confidence, 0, 1);
          return EDGE_MIN_OPACITY + (EDGE_MAX_OPACITY - EDGE_MIN_OPACITY) * Math.pow(normalized, 0.65);
        }};

        const getEdgeMarkerOpacity = (confidence) => {{
          const normalized = clamp(confidence, 0, 1);
          return EDGE_MIN_OPACITY + (EDGE_MAX_OPACITY - EDGE_MIN_OPACITY) * Math.pow(normalized, 0.5);
        }};

        const getEdgeLabelColor = (strokeColor) => {{
          return blendColors(strokeColor, EDGE_LABEL_LIGHTEN_TARGET, 0.32) || strokeColor;
        }};

        const resolveCollisions = (nodes) => {{
          const padding = 16;
          const epsilon = 0.0001;
          const iterations = 8;
          for (let iteration = 0; iteration < iterations; iteration += 1) {{
            let moved = false;
            for (let i = 0; i < nodes.length; i += 1) {{
              for (let j = i + 1; j < nodes.length; j += 1) {{
                const nodeA = nodes[i];
                const nodeB = nodes[j];
                let dx = nodeA.x - nodeB.x;
                let dy = nodeA.y - nodeB.y;
                let distance = Math.sqrt(dx * dx + dy * dy);
                const minDistance =
                  estimateCollisionRadius(nodeA.node, nodeA.degree) +
                  estimateCollisionRadius(nodeB.node, nodeB.degree) +
                  padding;
                if (distance >= minDistance) {{
                  continue;
                }}
                let nx;
                let ny;
                if (distance < epsilon) {{
                  const angle = ((i + 1) * 0.318309886 + (j + 1) * 0.127323954) * Math.PI * 2;
                  nx = Math.cos(angle);
                  ny = Math.sin(angle);
                  distance = epsilon;
                }} else {{
                  nx = dx / distance;
                  ny = dy / distance;
                }}
                const overlap = minDistance - distance;
                const shift = overlap / 2;
                nodeA.x += nx * shift;
                nodeA.y += ny * shift;
                nodeB.x -= nx * shift;
                nodeB.y -= ny * shift;
                moved = true;
              }}
            }}
            if (!moved) {{
              break;
            }}
          }}
        }};

        const runForceLayout = (nodes, edges, width, height) => {{
          if (!nodes.length) {{
            return {{
              nodes: [],
              edges: [],
              bounds: {{
                minX: 0,
                maxX: width,
                minY: 0,
                maxY: height,
                width: Math.max(width, 1),
                height: Math.max(height, 1),
              }},
              densityScale: 1,
            }};
          }}

          const centerX = width / 2;
          const centerY = height / 2;
          const allowedIds = new Set(nodes.map((node) => node.id));
          const filteredEdges = edges.filter(
            (edge) => allowedIds.has(edge.source) && allowedIds.has(edge.target),
          );

          const adjacency = new Map();
          for (const node of nodes) {{
            adjacency.set(node.id, new Set());
          }}
          for (const edge of filteredEdges) {{
            const sourceSet = adjacency.get(edge.source);
            if (sourceSet) {{
              sourceSet.add(edge.target);
            }}
            const targetSet = adjacency.get(edge.target);
            if (targetSet) {{
              targetSet.add(edge.source);
            }}
          }}

          const componentByNode = new Map();
          const componentSizes = [];
          for (const node of nodes) {{
            if (componentByNode.has(node.id)) {{
              continue;
            }}
            const queue = [node.id];
            const componentId = componentSizes.length;
            componentByNode.set(node.id, componentId);
            let size = 0;
            while (queue.length) {{
              const current = queue.pop();
              if (!current) {{
                continue;
              }}
              size += 1;
              const neighbours = adjacency.get(current);
              if (!neighbours) {{
                continue;
              }}
              neighbours.forEach((neighbour) => {{
                if (!componentByNode.has(neighbour)) {{
                  componentByNode.set(neighbour, componentId);
                  queue.push(neighbour);
                }}
              }});
            }}
            componentSizes.push(size);
          }}

          if (!componentSizes.length) {{
            componentSizes.push(nodes.length);
          }}

          const componentAnchors = new Map();
          const componentSpreadBase = Math.max(nodes.length, 1);
          const minDimension = Math.min(width, height);
          const goldenAngle = Math.PI * (3 - Math.sqrt(5));
          for (let index = 0; index < componentSizes.length; index += 1) {{
            const seeded = createSeededGenerator(`component-${{index}}`);
            const weight = componentSizes[index] / componentSpreadBase;
            const angle = index === 0 ? 0 : goldenAngle * index + seeded() * 0.18;
            const radialStep = minDimension * (0.03 + weight * 0.028);
            const distance =
              index === 0
                ? 0
                : Math.min(
                    minDimension * 0.05,
                    Math.sqrt(index + 1) * radialStep * 0.4 + seeded() * minDimension * 0.008,
                  );
            const spread = minDimension * (0.14 + weight * 0.075 + seeded() * 0.024);
            const noise = (seeded() - 0.5) * minDimension * 0.006;
            componentAnchors.set(index, {{
              x: centerX + Math.cos(angle) * distance,
              y: centerY + Math.sin(angle) * distance,
              spread,
              noise,
            }});
          }}

          const simulationNodes = nodes.map((node, index) => {{
            const componentId = componentByNode.has(node.id) ? componentByNode.get(node.id) : 0;
            const anchor = componentAnchors.get(componentId) || null;
            const seeded = createSeededGenerator(`${{node.id}}-${{index}}`);
            const angle = seeded() * Math.PI * 2;
            const spread = anchor ? anchor.spread : minDimension * 0.46;
            const radius = spread * (0.18 + seeded() * 0.52);
            const jitterMagnitude = spread * 0.08;
            const jitterX = (seeded() - 0.5) * 2 * jitterMagnitude;
            const jitterY = (seeded() - 0.5) * 2 * jitterMagnitude;
            const noise = anchor ? anchor.noise : 0;
            const baseAnchorX = (anchor ? anchor.x : centerX) + noise;
            const baseAnchorY = (anchor ? anchor.y : centerY) - noise;
            return {{
              node,
              componentId,
              anchorX: baseAnchorX,
              anchorY: baseAnchorY,
              x: baseAnchorX + Math.cos(angle) * radius + jitterX,
              y: baseAnchorY + Math.sin(angle) * radius + jitterY,
              dx: 0,
              dy: 0,
            }};
          }});

          const nodeIndex = new Map(simulationNodes.map((entry) => [entry.node.id, entry]));

          const iterations = Math.min(560, 220 + simulationNodes.length * 3);
          const area = width * height * LAYOUT_AREA_SCALE;
          const k = Math.sqrt(area / simulationNodes.length);
          let temperature = Math.max(width, height) / LAYOUT_TEMPERATURE_DIVISOR;
          const epsilon = 0.0001;

          for (let iteration = 0; iteration < iterations; iteration += 1) {{
            for (const node of simulationNodes) {{
              node.dx = 0;
              node.dy = 0;
            }}

            for (let i = 0; i < simulationNodes.length; i += 1) {{
              const nodeA = simulationNodes[i];
              for (let j = i + 1; j < simulationNodes.length; j += 1) {{
                const nodeB = simulationNodes[j];
                const dx = nodeA.x - nodeB.x;
                const dy = nodeA.y - nodeB.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || epsilon;
                const force = (LAYOUT_REPULSION_STRENGTH * k * k) / distance;
                const offsetX = (dx / distance) * force;
                const offsetY = (dy / distance) * force;
                nodeA.dx += offsetX;
                nodeA.dy += offsetY;
                nodeB.dx -= offsetX;
                nodeB.dy -= offsetY;
              }}
            }}

            for (const edge of filteredEdges) {{
              const source = nodeIndex.get(edge.source);
              const target = nodeIndex.get(edge.target);
              if (!source || !target) {{
                continue;
              }}
              const dx = source.x - target.x;
              const dy = source.y - target.y;
              const distance = Math.sqrt(dx * dx + dy * dy) || epsilon;
              const force = (LAYOUT_ATTRACTION_STRENGTH * distance * distance) / k;
              const offsetX = (dx / distance) * force;
              const offsetY = (dy / distance) * force;
              source.dx -= offsetX;
              source.dy -= offsetY;
              target.dx += offsetX;
              target.dy += offsetY;
            }}

            for (const node of simulationNodes) {{
              const toAnchorX = node.x - node.anchorX;
              const toAnchorY = node.y - node.anchorY;
              node.dx -= toAnchorX * (LAYOUT_ANCHOR_GRAVITY * 0.55);
              node.dy -= toAnchorY * (LAYOUT_ANCHOR_GRAVITY * 0.55);
              const toCenterX = node.x - centerX;
              const toCenterY = node.y - centerY;
              node.dx -= toCenterX * LAYOUT_CENTER_GRAVITY;
              node.dy -= toCenterY * LAYOUT_CENTER_GRAVITY;
              if (node.componentId !== 0) {{
                const blendedAnchorX = (node.anchorX + centerX) / 2;
                const blendedAnchorY = (node.anchorY + centerY) / 2;
                node.dx -= (node.x - blendedAnchorX) * LAYOUT_CROSS_COMPONENT_PULL;
                node.dy -= (node.y - blendedAnchorY) * LAYOUT_CROSS_COMPONENT_PULL;
              }}
              const displacement = Math.sqrt(node.dx * node.dx + node.dy * node.dy) || epsilon;
              const limited = Math.min(displacement, temperature);
              node.x += (node.dx / displacement) * limited;
              node.y += (node.dy / displacement) * limited;
            }}

            temperature *= LAYOUT_COOLING_FACTOR;
            if (temperature < 0.2) {{
              break;
            }}
          }}

          const positionedNodes = simulationNodes.map((node, index) => {{
            const jitterGenerator = createSeededGenerator(`post-${{node.node.id}}-${{index}}`);
            const jitterScale = Math.max(width, height) * 0.004;
            const jitterX = (jitterGenerator() - 0.5) * 2 * jitterScale;
            const jitterY = (jitterGenerator() - 0.5) * 2 * jitterScale;
            node.x += jitterX;
            node.y += jitterY;
            const neighbours = adjacency.get(node.node.id);
            return {{
              node: node.node,
              x: node.x,
              y: node.y,
              degree: neighbours ? neighbours.size : 0,
              componentId: node.componentId,
              anchorX: node.anchorX,
              anchorY: node.anchorY,
            }};
          }});

          resolveCollisions(positionedNodes);
          let minX = Number.POSITIVE_INFINITY;
          let maxX = Number.NEGATIVE_INFINITY;
          let minY = Number.POSITIVE_INFINITY;
          let maxY = Number.NEGATIVE_INFINITY;
          for (const entry of positionedNodes) {{
            if (entry.x < minX) minX = entry.x;
            if (entry.x > maxX) maxX = entry.x;
            if (entry.y < minY) minY = entry.y;
            if (entry.y > maxY) maxY = entry.y;
          }}
          if (!Number.isFinite(minX)) minX = 0;
          if (!Number.isFinite(maxX)) maxX = width;
          if (!Number.isFinite(minY)) minY = 0;
          if (!Number.isFinite(maxY)) maxY = height;
          const bounds = {{
            minX,
            maxX,
            minY,
            maxY,
          }};

          const positionedNodeIndex = new Map(positionedNodes.map((entry) => [entry.node.id, entry]));
          const positionedEdges = filteredEdges.map((edge) => {{
            const source = positionedNodeIndex.get(edge.source);
            const target = positionedNodeIndex.get(edge.target);
            if (!source || !target) {{
              throw new Error("Graph layout attempted to render an edge without positioned nodes");
            }}
            return {{
              id: edge.id,
              source,
              target,
              relation: edge.relation,
              confidence: Number(edge.confidence || 0),
              data: edge,
            }};
          }});

          return {{
            nodes: positionedNodes,
            edges: positionedEdges,
            bounds: {{
              minX: bounds.minX,
              maxX: bounds.maxX,
              minY: bounds.minY,
              maxY: bounds.maxY,
            }},
            densityScale: 1,
          }};
        }};

        const canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        container.innerHTML = "";
        container.appendChild(canvas);

        const ctx = canvas.getContext("2d");
        if (!ctx) {{
          container.innerHTML = "<p style='padding:1rem;'>Canvas rendering unavailable in this browser.</p>";
          return;
        }}

        const ratio = window.devicePixelRatio || 1;
        const baseWidth = Math.max(container.clientWidth || 960, 320);
        const baseHeight = Math.max(container.clientHeight || 640, 320);
        const layout = runForceLayout(nodesForLayout, edgesRaw, baseWidth, baseHeight);
        const ZOOM_LIMITS = { min: 0.5, max: 3 };
        const ZOOM_SENSITIVITY = 0.0025;
        const PAN_MOVE_THRESHOLD = 4;

        const layoutMargin = 30;
        const minX = layout.bounds.minX - layoutMargin;
        const maxX = layout.bounds.maxX + layoutMargin;
        const minY = layout.bounds.minY - layoutMargin;
        const maxY = layout.bounds.maxY + layoutMargin;
        const viewWidth = Math.max(maxX - minX, 1);
        const viewHeight = Math.max(maxY - minY, 1);

        const translatedNodes = layout.nodes.map((entry) => ({
          ...entry,
          x: entry.x - minX,
          y: entry.y - minY,
          anchorX: entry.anchorX - minX,
          anchorY: entry.anchorY - minY,
        }));

        const nodes = translatedNodes.map((entry) => {{
          const labelLines = formatNodeLabel(entry.node.label || entry.node.id || "Unknown");
          const fill = getNodeFill(entry.node.type);
          const {{ color: labelColor, outline: labelOutline }} = getContrastingLabelColors(fill);
          const radius = calculateNodeRadius(entry.node, entry.degree, labelLines);
          return {{
            id: entry.node.id,
            data: entry.node,
            graphX: entry.x,
            graphY: entry.y,
            degree: entry.degree,
            componentId: entry.componentId,
            radius,
            labelLines,
            fill,
            labelColor,
            labelOutline,
          }};
        }});

        const nodeIndex = new Map(nodes.map((node) => [node.id, node]));
        const edges = layout.edges
          .map((edge) => {{
            const source = nodeIndex.get(edge.source.node.id);
            const target = nodeIndex.get(edge.target.node.id);
            if (!source || !target) {{
              return null;
            }}
            const confidence = Number(edge.confidence || 0);
            const strokeBase = getRelationColor(edge.relation);
            const stroke = getEdgeStrokeColor(strokeBase, confidence);
            return {{
              id: edge.id,
              data: edge.data || null,
              source,
              target,
              relation: edge.relation,
              confidence,
              length: Math.hypot(
                target.graphX - source.graphX,
                target.graphY - source.graphY,
              ),
              relationLabel:
                (edge.data && (edge.data.relation_verbatim || edge.data.relation)) || edge.relation,
              stroke,
              strokeWidth: getEdgeStrokeWidth(confidence),
              strokeOpacity: getEdgeStrokeOpacity(confidence),
              markerOpacity: getEdgeMarkerOpacity(confidence),
              labelColor: getEdgeLabelColor(stroke),
            }};
          }})
          .filter(Boolean);

        const displayedNodeCount = nodes.length;
        const displayedEdgeCount = edges.length;
        const truncatedNodes = totalNodeCount > displayedNodeCount;
        const truncatedEdges = totalEdgeCount > displayedEdgeCount;

        const details = document.getElementById("details");
        const formatJSON = (value) => JSON.stringify(value, null, 2);
        let selectedNode = null;

        const appendTruncationNotice = () => {
          if (!details || (!truncatedNodes && !truncatedEdges)) {
            return;
          }
          const notice = document.createElement("p");
          notice.style.marginTop = "0.75rem";
          notice.style.fontSize = "0.85rem";
          notice.style.color = "rgba(148, 163, 184, 0.8)";
          const nodeSummary = `${displayedNodeCount} of ${totalNodeCount}`;
          const edgeSummary = `${displayedEdgeCount} of ${totalEdgeCount}`;
          notice.textContent =
            `Showing first ${nodeSummary} nodes and ${edgeSummary} edges in the interactive view.`;
          details.appendChild(notice);
        };

        const setDefaultDetails = () => {
          if (!details) {
            return;
          }
          details.innerHTML =
            "<h3>Details</h3><p>Select a node or edge to see contextual information.</p>";
          appendTruncationNotice();
        };

        setDefaultDetails();

        function resizeCanvas() {{
          const rect = container.getBoundingClientRect();
          const width = Math.max(rect.width, 200);
          const height = Math.max(rect.height, 200);
          canvas.width = width * ratio;
          canvas.height = height * ratio;
          canvas.style.width = width + "px";
          canvas.style.height = height + "px";
          draw();
        }}

        let interactionScale = 1;
        let interactionOffsetX = 0;
        let interactionOffsetY = 0;
        let currentScale = 1;
        let pointerActive = false;
        let pointerMoved = false;
        let pointerStartX = 0;
        let pointerStartY = 0;
        let pointerStartOffsetX = 0;
        let pointerStartOffsetY = 0;

        canvas.style.cursor = "grab";

        function getCanvasMetrics() {{
          const width = canvas.width / ratio;
          const height = canvas.height / ratio;
          const fitScale = Math.min(width / viewWidth, height / viewHeight);
          const baseCenterX = width / 2;
          const baseCenterY = height / 2;
          return {{ width, height, fitScale, baseCenterX, baseCenterY }};
        }}

        function draw() {{
          const {{ width, height, fitScale, baseCenterX, baseCenterY }} = getCanvasMetrics();
          const scale = fitScale * interactionScale;
          const offsetX = interactionOffsetX;
          const offsetY = interactionOffsetY;
          currentScale = scale;
          for (const node of nodes) {{
            const centeredX = node.graphX - viewWidth / 2;
            const centeredY = node.graphY - viewHeight / 2;
            node.screenX = centeredX * scale + baseCenterX + offsetX;
            node.screenY = centeredY * scale + baseCenterY + offsetY;
            node.screenRadius = Math.max(4, node.radius * scale);
          }}
          ctx.save();
          ctx.scale(ratio, ratio);
          ctx.clearRect(0, 0, width, height);
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          for (const edge of edges) {{
            const sourceX = edge.source.screenX;
            const sourceY = edge.source.screenY;
            const targetX = edge.target.screenX;
            const targetY = edge.target.screenY;
            ctx.globalAlpha = edge.strokeOpacity;
            ctx.strokeStyle = edge.stroke;
            ctx.lineWidth = Math.max(0.75, edge.strokeWidth * scale);
            ctx.beginPath();
            ctx.moveTo(sourceX, sourceY);
            ctx.lineTo(targetX, targetY);
            ctx.stroke();
          }}
          ctx.globalAlpha = 1;
          const labelScale = Math.max(scale, 0.08);
          for (const edge of edges) {{
            const label = edge.relationLabel || "";
            if (!label) {{
              continue;
            }}
            const sourceX = edge.source.screenX;
            const sourceY = edge.source.screenY;
            const targetX = edge.target.screenX;
            const targetY = edge.target.screenY;
            const midX = (sourceX + targetX) / 2;
            const midY = (sourceY + targetY) / 2;
            const angle = Math.atan2(targetY - sourceY, targetX - sourceX);
            const flipped = angle > Math.PI / 2 || angle < -Math.PI / 2;
            const fontSize = Math.max(EDGE_LABEL_FONT_SIZE * labelScale, 5);
            const padding = EDGE_LABEL_HORIZONTAL_PADDING * labelScale;
            const minWidth = EDGE_LABEL_MIN_WIDTH * labelScale;
            const maxWidth = EDGE_LABEL_MAX_WIDTH * labelScale;
            const rectHeight = EDGE_LABEL_RECT_HEIGHT * labelScale;
            ctx.save();
            ctx.font = `${{EDGE_LABEL_FONT_WEIGHT}} ${{fontSize}}px "Inter", system-ui, sans-serif`;
            const textWidth = ctx.measureText(label).width + padding;
            const rectWidth = clamp(textWidth, minWidth, maxWidth);
            ctx.translate(midX, midY);
            ctx.rotate(angle);
            if (flipped) {{
              ctx.rotate(Math.PI);
            }}
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const cornerRadius = Math.min(rectHeight / 2, 8 * labelScale);
            ctx.fillStyle = "rgba(248, 250, 252, 0.98)";
            drawRoundedRectPath(ctx, -rectWidth / 2, -rectHeight / 2, rectWidth, rectHeight, cornerRadius);
            ctx.fill();
            ctx.strokeStyle = edge.stroke;
            ctx.globalAlpha = Math.min(1, edge.strokeOpacity + 0.12);
            ctx.lineWidth = Math.max(0.5, 0.9 * labelScale);
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.strokeStyle = "rgba(15, 23, 42, 0.08)";
            ctx.lineWidth = Math.max(0.4, 0.6 * labelScale);
            ctx.fillStyle = edge.labelColor || edge.stroke || "#0f172a";
            ctx.strokeText(label, 0, 0);
            ctx.fillText(label, 0, 0);
            ctx.restore();
          }}
          for (const node of nodes) {{
            const nodeX = node.screenX;
            const nodeY = node.screenY;
            ctx.beginPath();
            ctx.fillStyle = node.fill;
            ctx.strokeStyle = NODE_STROKE_COLOR;
            const strokeWidth = selectedNode && selectedNode.id === node.id ? 3.2 : 2;
            ctx.lineWidth = Math.max(0.8, strokeWidth * labelScale);
            ctx.arc(nodeX, nodeY, node.screenRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }}
          const nodeFontSize = Math.max(NODE_LABEL_FONT_SIZE * labelScale, 4.5);
          const nodeLineHeight = NODE_LABEL_LINE_HEIGHT * labelScale;
          ctx.font = `${{NODE_LABEL_FONT_WEIGHT}} ${{nodeFontSize}}px "Inter", system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          for (const node of nodes) {{
            const nodeX = node.screenX;
            const nodeY = node.screenY;
            const labelLines = Array.isArray(node.labelLines) && node.labelLines.length ? node.labelLines : [
              node.data.label || node.id || "",
            ];
            ctx.fillStyle = node.labelColor || NODE_LABEL_TEXT_COLOR;
            ctx.strokeStyle = node.labelOutline || "rgba(255, 255, 255, 0.75)";
            const outlineWidth = Math.max(0.6, 1.2 * labelScale);
            ctx.lineWidth = outlineWidth;
            for (let index = 0; index < labelLines.length; index += 1) {{
              const line = labelLines[index];
              if (!line) {{
                continue;
              }}
              const offset = (index - (labelLines.length - 1) / 2) * nodeLineHeight;
              const textY = nodeY + offset;
              ctx.strokeText(line, nodeX, textY);
              ctx.fillText(line, nodeX, textY);
            }}
          }}
          ctx.restore();
        }}

        function pickNode(clientX, clientY) {{
          const rect = canvas.getBoundingClientRect();
          if (!rect.width || !rect.height) {{
            return null;
          }}
          const width = canvas.width / ratio;
          const height = canvas.height / ratio;
          const x = ((clientX - rect.left) / rect.width) * width;
          const y = ((clientY - rect.top) / rect.height) * height;
          let nearest = null;
          let minDistance = Number.POSITIVE_INFINITY;
          for (const node of nodes) {{
            const dx = node.screenX - x;
            const dy = node.screenY - y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance <= node.screenRadius) {{
              return node;
            }}
            const threshold = node.screenRadius + 20;
            if (distance < threshold && distance < minDistance) {{
              nearest = node;
              minDistance = distance;
            }}
          }}
          return nearest;
        }}

        function pointToSegmentDistance(px, py, x1, y1, x2, y2) {{
          const dx = x2 - x1;
          const dy = y2 - y1;
          if (dx === 0 && dy === 0) {{
            return Math.hypot(px - x1, py - y1);
          }}
          const t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
          const clampedT = Math.max(0, Math.min(1, t));
          const projX = x1 + clampedT * dx;
          const projY = y1 + clampedT * dy;
          return Math.hypot(px - projX, py - projY);
        }}

        function pickEdge(clientX, clientY) {{
          const rect = canvas.getBoundingClientRect();
          if (!rect.width || !rect.height) {{
            return null;
          }}
          const width = canvas.width / ratio;
          const height = canvas.height / ratio;
          const x = ((clientX - rect.left) / rect.width) * width;
          const y = ((clientY - rect.top) / rect.height) * height;
          let closest = null;
          let minDist = Math.max(12, currentScale * 18);
          for (const edge of edges) {{
            const x1 = edge.source.screenX;
            const y1 = edge.source.screenY;
            const x2 = edge.target.screenX;
            const y2 = edge.target.screenY;
            const dist = pointToSegmentDistance(x, y, x1, y1, x2, y2);
            if (dist < minDist) {{
              minDist = dist;
              closest = edge;
            }}
          }}
          return closest;
        }}

        function handleGraphSelection(clientX, clientY) {{
          const node = pickNode(clientX, clientY);
          if (node) {{
            showNodeDetails(node);
            return;
          }}
          const edge = pickEdge(clientX, clientY);
          if (edge) {{
            showEdgeDetails(edge);
            return;
          }}
          selectedNode = null;
          draw();
          setDefaultDetails();
        }}

        function showNodeDetails(node) {{
          selectedNode = node;
          draw();
          const aliases = (node.data.aliases || []).join(", ") || "-";
          const sections = formatJSON(node.data.section_distribution || {{}});
          details.innerHTML = "<h3>Node</h3><pre>" +
            "ID: " + node.data.id + "\\n" +
            "Label: " + node.data.label + "\\n" +
            "Type: " + (node.data.type || "Unknown") + "\\n" +
            "Times seen: " + (node.data.times_seen || 0) + "\\n" +
            "Aliases: " + aliases + "\\n" +
            "Sections:\\n" + sections +
            "</pre>";
          appendTruncationNotice();
        }}

        function showEdgeDetails(edge) {{
          selectedNode = null;
          draw();
          const evidence = formatJSON((edge.data && edge.data.evidence) || {{}});
          const relationCanonical =
            edge.data && edge.data.relation ? edge.data.relation : edge.relation;
          const relationLabel =
            edge.relationLabel ||
            (edge.data && (edge.data.relation_verbatim || edge.data.relation)) ||
            relationCanonical ||
            "-";
          const relationDisplay =
            relationCanonical && relationCanonical !== relationLabel
              ? `${{relationLabel}} (${{relationCanonical}})`
              : relationLabel;
          const confidenceValue =
            typeof edge.confidence === "number"
              ? edge.confidence
              : Number((edge.data && edge.data.confidence) || 0);
          details.innerHTML = "<h3>Edge</h3><pre>" +
            "Relation: " + relationDisplay + "\\n" +
            "Confidence: " + Number.parseFloat(confidenceValue || 0).toFixed(3) + "\\n" +
            "Source -> Target: " + (edge.source.data.label || edge.source.data.id) + " -> " +
            (edge.target.data.label || edge.target.data.id) + "\\n" +
            "Evidence:\\n" + evidence +
            "</pre>";
          appendTruncationNotice();
        }}

        function handleWheel(event) {{
          event.preventDefault();
          const {{ width, height, fitScale, baseCenterX, baseCenterY }} = getCanvasMetrics();
          const rect = canvas.getBoundingClientRect();
          const mouseX = ((event.clientX - rect.left) / rect.width) * width;
          const mouseY = ((event.clientY - rect.top) / rect.height) * height;
          const totalScaleBefore = fitScale * interactionScale;
          if (totalScaleBefore <= 0) {{
            return;
          }}
          const delta = Math.max(-120, Math.min(120, event.deltaY));
          const zoomMultiplier = Math.exp(-delta * ZOOM_SENSITIVITY);
          interactionScale = Math.min(
            ZOOM_LIMITS.max,
            Math.max(ZOOM_LIMITS.min, interactionScale * zoomMultiplier),
          );
          const totalScaleAfter = fitScale * interactionScale;
          const centeredX = (mouseX - baseCenterX - interactionOffsetX) / totalScaleBefore;
          const centeredY = (mouseY - baseCenterY - interactionOffsetY) / totalScaleBefore;
          interactionOffsetX = mouseX - baseCenterX - centeredX * totalScaleAfter;
          interactionOffsetY = mouseY - baseCenterY - centeredY * totalScaleAfter;
          draw();
        }}

        function handlePointerDown(event) {{
          if (event.button !== undefined && event.button !== 0) {{
            return;
          }}
          event.preventDefault();
          pointerActive = true;
          pointerMoved = false;
          pointerStartX = event.clientX;
          pointerStartY = event.clientY;
          pointerStartOffsetX = interactionOffsetX;
          pointerStartOffsetY = interactionOffsetY;
          canvas.style.cursor = "grabbing";
          if (canvas.setPointerCapture) {{
            try {{
              canvas.setPointerCapture(event.pointerId);
            }} catch (error) {{}}
          }}
        }}

        function handlePointerMove(event) {{
          if (!pointerActive) {{
            return;
          }}
          const deltaX = event.clientX - pointerStartX;
          const deltaY = event.clientY - pointerStartY;
          if (
            !pointerMoved &&
            (Math.abs(deltaX) >= PAN_MOVE_THRESHOLD || Math.abs(deltaY) >= PAN_MOVE_THRESHOLD)
          ) {{
            pointerMoved = true;
          }}
          if (pointerMoved) {{
            interactionOffsetX = pointerStartOffsetX + deltaX;
            interactionOffsetY = pointerStartOffsetY + deltaY;
            draw();
          }}
        }}

        function handlePointerEnd(event) {{
          if (!pointerActive) {{
            return;
          }}
          if (canvas.releasePointerCapture) {{
            try {{
              canvas.releasePointerCapture(event.pointerId);
            }} catch (error) {{}}
          }}
          pointerActive = false;
          canvas.style.cursor = "grab";
          if (!pointerMoved) {{
            handleGraphSelection(event.clientX, event.clientY);
          }}
        }}

        canvas.addEventListener("wheel", handleWheel, {{ passive: false }});
        canvas.addEventListener("pointerdown", handlePointerDown);
        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", handlePointerEnd);
        window.addEventListener("pointercancel", handlePointerEnd);

        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();
        draw();
      })();
    </script>
  </body>
</html>
"""
    html_template = html_template.replace("{{", "{").replace("}}", "}")
    html_content = (
        html_template.replace("__GRAPH_DATA__", payload)
        .replace("__EXPIRES_AT__", expires_js)
        .replace("__BUNDLE_LINES__", bundle_lines)
        .replace("__DOWNLOAD_LINK__", download_link_markup)
        .replace("__NODE_LIMIT__", str(VISUALIZATION_NODE_LIMIT))
    )
    return html_content
