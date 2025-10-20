"""Utilities for rendering shareable export HTML pages."""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Mapping, MutableMapping


def render_share_html(
    graph_data: Mapping[str, object],
    *,
    download_url: str | None = None,
    expires_at: datetime | None = None,
) -> str:
    """Render an interactive HTML view for a shared graph export."""

    payload = json.dumps(graph_data, separators=(",", ":"))
    bundle_info: MutableMapping[str, str] = {
        "Pipeline version": str(graph_data.get("pipeline_version", "unknown")),
        "Nodes": str(graph_data.get("node_count", 0)),
        "Edges": str(graph_data.get("edge_count", 0)),
    }
    if expires_at is not None:
        bundle_info["Link expires"] = expires_at.isoformat()
    else:
        bundle_info["Link expires"] = "Never"

    bundle_lines = "\n            ".join(
        f"<li><strong>{html.escape(key)}:</strong> {html.escape(value)}</li>"
        for key, value in bundle_info.items()
    )
    download_js = json.dumps(download_url) if download_url is not None else "null"
    expires_js = json.dumps(expires_at.isoformat()) if expires_at else "null"

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
        background-color: #0b1120;
        color: #f8fafc;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
      }}
      header {{
        padding: 1.5rem 2rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
      }}
      header h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 600;
      }}
      header p {{
        margin: 0.5rem 0 0;
        color: rgba(226, 232, 240, 0.75);
      }}
      main {{
        flex: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        padding: 1.5rem;
      }}
      #graph {{
        flex: 1 1 600px;
        min-height: 70vh;
        background: rgba(15, 23, 42, 0.85);
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 24px 48px rgba(2, 6, 23, 0.45);
      }}
      aside {{
        width: 320px;
        max-width: 100%;
        background: rgba(15, 23, 42, 0.85);
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 1.25rem;
        box-shadow: 0 20px 40px rgba(2, 6, 23, 0.4);
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }}
      aside ul {{
        list-style: none;
        margin: 0;
        padding: 0;
      }}
      aside li {{
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        font-size: 0.95rem;
      }}
      aside li:last-child {{
        border-bottom: none;
      }}
      .download {{
        display: none;
        justify-content: center;
        align-items: center;
        padding: 0.6rem 1rem;
        border-radius: 9999px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        text-decoration: none;
        color: #fff;
        font-weight: 600;
        box-shadow: 0 18px 30px rgba(99, 102, 241, 0.35);
      }}
      #details {{
        background: rgba(30, 41, 59, 0.65);
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        font-size: 0.9rem;
        line-height: 1.5;
      }}
      #details pre {{
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
      }}
      @media (max-width: 768px) {{
        aside {{
          width: 100%;
        }}
        #graph {{
          min-height: 60vh;
        }}
      }}
    </style>
    <script>
      const GRAPH_DATA = __GRAPH_DATA__;
      const DOWNLOAD_URL = __DOWNLOAD_URL__;
      const EXPIRES_AT = __EXPIRES_AT__;
    </script>
  </head>
  <body>
    <header>
      <h1>SciNets Shared Graph</h1>
      <p>Interactive snapshot exported from the SciNets knowledge graph.</p>
    </header>
    <main>
      <aside>
        <div>
          <h2>Bundle details</h2>
          <ul>
            __BUNDLE_LINES__
          </ul>
        </div>
        <a id="download-link" class="download" rel="noopener" href="#" style="display:none;">
          Download bundle (.zip)
        </a>
        <section id="details">
          <h3>Details</h3>
          <p>Select a node or edge to see contextual information.</p>
        </section>
      </aside>
      <section id="graph"></section>
    </main>
    <script>
      (function () {{
        const graphData = GRAPH_DATA || {{}};
        const downloadUrl = DOWNLOAD_URL;
        const expiresAt = EXPIRES_AT;
        const downloadLink = document.getElementById("download-link");
        if (downloadLink && downloadUrl) {{
          downloadLink.href = downloadUrl;
          downloadLink.style.display = "inline-flex";
        }}
        if (expiresAt) {{
          const expiryItem = document.querySelector("li[data-meta='expires']");
          if (!expiryItem) {{
            const list = document.querySelector("aside ul");
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
        if (nodesRaw.length === 0) {{
          container.innerHTML = "<p style='padding:1rem;'>No nodes available in this export.</p>";
          return;
        }}

        const NODE_STROKE_COLOR = "#0f172a";
        const NODE_LABEL_LIGHT_COLOR = "#f8fafc";
        const NODE_LABEL_DARK_COLOR = "#0f172a";
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
        const LAYOUT_DENSITY_TARGET = 0.24;
        const LAYOUT_DENSITY_MAX_SCALE = 2.25;

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
            return {{ color: NODE_LABEL_LIGHT_COLOR, outline: "rgba(15, 23, 42, 0.45)" }};
          }}
          const luminance = getRelativeLuminance(rgb);
          if (luminance > 0.55) {{
            return {{ color: NODE_LABEL_DARK_COLOR, outline: "rgba(255, 255, 255, 0.7)" }};
          }}
          return {{ color: NODE_LABEL_LIGHT_COLOR, outline: "rgba(15, 23, 42, 0.5)" }};
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

        const computeBounds = (nodes) => {{
          let minX = Number.POSITIVE_INFINITY;
          let maxX = Number.NEGATIVE_INFINITY;
          let minY = Number.POSITIVE_INFINITY;
          let maxY = Number.NEGATIVE_INFINITY;
          for (const entry of nodes) {{
            if (entry.x < minX) minX = entry.x;
            if (entry.x > maxX) maxX = entry.x;
            if (entry.y < minY) minY = entry.y;
            if (entry.y > maxY) maxY = entry.y;
          }}
          if (!Number.isFinite(minX)) {{
            minX = 0;
          }}
          if (!Number.isFinite(maxX)) {{
            maxX = 0;
          }}
          if (!Number.isFinite(minY)) {{
            minY = 0;
          }}
          if (!Number.isFinite(maxY)) {{
            maxY = 0;
          }}
          return {{
            minX,
            maxX,
            minY,
            maxY,
            width: Math.max(maxX - minX, 1),
            height: Math.max(maxY - minY, 1),
          }};
        }};

        const expandLayoutIfDense = (nodes) => {{
          if (!nodes.length) {{
            return computeBounds(nodes);
          }}
          const bounds = computeBounds(nodes);
          const area = bounds.width * bounds.height;
          if (!Number.isFinite(area) || area <= 0) {{
            return bounds;
          }}
          const totalNodeArea = nodes.reduce((accumulator, entry) => {{
            const radius = estimateCollisionRadius(entry.node, entry.degree);
            return accumulator + Math.PI * radius * radius;
          }}, 0);
          if (!Number.isFinite(totalNodeArea) || totalNodeArea <= 0) {{
            return bounds;
          }}
          const density = totalNodeArea / area;
          if (density <= LAYOUT_DENSITY_TARGET) {{
            return bounds;
          }}
          const scale = Math.min(
            LAYOUT_DENSITY_MAX_SCALE,
            Math.sqrt(density / LAYOUT_DENSITY_TARGET),
          );
          if (!Number.isFinite(scale) || scale <= 1) {{
            return bounds;
          }}
          const centerX = (bounds.minX + bounds.maxX) / 2;
          const centerY = (bounds.minY + bounds.maxY) / 2;
          nodes.forEach((entry) => {{
            entry.x = centerX + (entry.x - centerX) * scale;
            entry.y = centerY + (entry.y - centerY) * scale;
          }});
          return computeBounds(nodes);
        }};

        const runForceLayout = (nodes, edges, width, height) => {{
          if (!nodes.length) {{
            return {{
              nodes: [],
              edges: [],
              bounds: {{ minX: 0, maxX: width, minY: 0, maxY: height }},
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
            }};
          }});

          resolveCollisions(positionedNodes);
          const bounds = expandLayoutIfDense(positionedNodes);

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
        const layout = runForceLayout(nodesRaw, edgesRaw, baseWidth, baseHeight);

        const layoutMargin = 30;
        const minX = layout.bounds.minX - layoutMargin;
        const maxX = layout.bounds.maxX + layoutMargin;
        const minY = layout.bounds.minY - layoutMargin;
        const maxY = layout.bounds.maxY + layoutMargin;
        const graphWidth = Math.max(maxX - minX, 1);
        const graphHeight = Math.max(maxY - minY, 1);
        const graphCenterX = graphWidth / 2;
        const graphCenterY = graphHeight / 2;

        const nodes = layout.nodes.map((entry) => {{
          const x = entry.x - minX;
          const y = entry.y - minY;
          const labelLines = formatNodeLabel(entry.node.label || entry.node.id || "Unknown");
          const fill = getNodeFill(entry.node.type);
          const {{ color: labelColor, outline: labelOutline }} = getContrastingLabelColors(fill);
          const radius = calculateNodeRadius(entry.node, entry.degree, labelLines);
          return {{
            id: entry.node.id,
            data: entry.node,
            x,
            y,
            layoutX: graphWidth ? x / graphWidth : 0.5,
            layoutY: graphHeight ? y / graphHeight : 0.5,
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

        const details = document.getElementById("details");
        const formatJSON = (value) => JSON.stringify(value, null, 2);
        let selectedNode = null;

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

        function draw() {{
          const width = canvas.width / ratio;
          const height = canvas.height / ratio;
          ctx.save();
          ctx.scale(ratio, ratio);
          ctx.clearRect(0, 0, width, height);
          ctx.lineCap = "round";
          for (const edge of edges) {{
            const sourceX = edge.source.layoutX * width;
            const sourceY = edge.source.layoutY * height;
            const targetX = edge.target.layoutX * width;
            const targetY = edge.target.layoutY * height;
            ctx.globalAlpha = edge.strokeOpacity;
            ctx.strokeStyle = edge.stroke;
            ctx.lineWidth = edge.strokeWidth;
            ctx.beginPath();
            ctx.moveTo(sourceX, sourceY);
            ctx.lineTo(targetX, targetY);
            ctx.stroke();
          }}
          ctx.globalAlpha = 1;
          for (const node of nodes) {{
            const nodeX = node.layoutX * width;
            const nodeY = node.layoutY * height;
            ctx.beginPath();
            ctx.fillStyle = node.fill;
            ctx.strokeStyle = NODE_STROKE_COLOR;
            ctx.lineWidth = selectedNode && selectedNode.id === node.id ? 3 : 1.8;
            ctx.arc(nodeX, nodeY, node.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }}
          ctx.font = "11px Inter, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          for (const node of nodes) {{
            const label = node.data.label || node.id;
            const nodeX = node.layoutX * width;
            const nodeY = node.layoutY * height;
            const labelWidth = Math.max(48, ctx.measureText(label).width + 12);
            ctx.save();
            ctx.fillStyle = "rgba(2, 6, 23, 0.75)";
            ctx.fillRect(nodeX - labelWidth / 2, nodeY - node.radius - 24, labelWidth, 18);
            ctx.fillStyle = "#f8fafc";
            ctx.fillText(label, nodeX, nodeY - node.radius - 15);
            ctx.restore();
          }}
          ctx.restore();
        }}

        function pickNode(clientX, clientY) {{
          const rect = canvas.getBoundingClientRect();
          const x = (clientX - rect.left) / rect.width;
          const y = (clientY - rect.top) / rect.height;
          let nearest = null;
          let minDist = 0.05;
          for (const node of nodes) {{
            const dx = node.layoutX - x;
            const dy = node.layoutY - y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < minDist) {{
              minDist = dist;
              nearest = node;
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
          const width = canvas.width / ratio;
          const height = canvas.height / ratio;
          const x = ((clientX - rect.left) / rect.width) * width;
          const y = ((clientY - rect.top) / rect.height) * height;
          let closest = null;
          let minDist = 12;
          for (const edge of edges) {{
            const x1 = edge.source.layoutX * width;
            const y1 = edge.source.layoutY * height;
            const x2 = edge.target.layoutX * width;
            const y2 = edge.target.layoutY * height;
            const dist = pointToSegmentDistance(x, y, x1, y1, x2, y2);
            if (dist < minDist) {{
              minDist = dist;
              closest = edge;
            }}
          }}
          return closest;
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
        }}

        canvas.addEventListener("click", (event) => {{
          const node = pickNode(event.clientX, event.clientY);
          if (node) {{
            showNodeDetails(node);
            return;
          }}
          const edge = pickEdge(event.clientX, event.clientY);
          if (edge) {{
            showEdgeDetails(edge);
            return;
          }}
          selectedNode = null;
          draw();
          details.innerHTML = "<h3>Details</h3><p>Select a node or edge to see contextual information.</p>";
        }});

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
        .replace("__DOWNLOAD_URL__", download_js)
        .replace("__EXPIRES_AT__", expires_js)
        .replace("__BUNDLE_LINES__", bundle_lines)
    )
    return html_content
