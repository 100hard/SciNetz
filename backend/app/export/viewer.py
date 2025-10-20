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
        const nodes = nodesRaw.map((node, index) => ({
          data: node,
          id: node.id,
          x: Math.cos(index) * 60 + Math.random() * 20,
          y: Math.sin(index) * 60 + Math.random() * 20,
          vx: 0,
          vy: 0,
        }));

        const nodeIndex = new Map();
        nodes.forEach((node, idx) => nodeIndex.set(node.id, idx));

        const edges = edgesRaw
          .map((edge) => {{
            const sourceIndex = nodeIndex.get(edge.source);
            const targetIndex = nodeIndex.get(edge.target);
            if (sourceIndex === undefined || targetIndex === undefined) {{
              return null;
            }}
            return {{
              data: edge,
              source: nodes[sourceIndex],
              target: nodes[targetIndex],
              weight: Math.max(0.1, Number(edge.confidence) || 0.5),
            }};
          }})
          .filter(Boolean);

        const iterations = Math.min(600, 120 + nodes.length * 10);
        const repulsion = 480;
        const springLength = 90;
        const springStrength = 0.012;
        const damping = 0.85;

        for (let i = 0; i < iterations; i += 1) {{
          for (let j = 0; j < nodes.length; j += 1) {{
            const nodeA = nodes[j];
            for (let k = j + 1; k < nodes.length; k += 1) {{
              const nodeB = nodes[k];
              let dx = nodeA.x - nodeB.x;
              let dy = nodeA.y - nodeB.y;
              const distSq = dx * dx + dy * dy + 0.01;
              const force = repulsion / distSq;
              const distance = Math.sqrt(distSq);
              dx /= distance;
              dy /= distance;
              nodeA.vx += dx * force;
              nodeA.vy += dy * force;
              nodeB.vx -= dx * force;
              nodeB.vy -= dy * force;
            }}
          }}

          for (const edge of edges) {{
            const nodeA = edge.source;
            const nodeB = edge.target;
            let dx = nodeB.x - nodeA.x;
            let dy = nodeB.y - nodeA.y;
            const distance = Math.max(0.01, Math.sqrt(dx * dx + dy * dy));
            const displacement = distance - springLength;
            const force = springStrength * displacement;
            dx /= distance;
            dy /= distance;
            nodeA.vx += dx * force;
            nodeA.vy += dy * force;
            nodeB.vx -= dx * force;
            nodeB.vy -= dy * force;
          }}

          for (const node of nodes) {{
            node.vx *= damping;
            node.vy *= damping;
            node.x += node.vx;
            node.y += node.vy;
          }}
        }}

        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const node of nodes) {{
          if (node.x < minX) minX = node.x;
          if (node.x > maxX) maxX = node.x;
          if (node.y < minY) minY = node.y;
          if (node.y > maxY) maxY = node.y;
        }}
        const rangeX = Math.max(maxX - minX, 1);
        const rangeY = Math.max(maxY - minY, 1);
        nodes.forEach((node) => {{
          node.layoutX = (node.x - minX) / rangeX;
          node.layoutY = (node.y - minY) / rangeY;
        }});

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
          ctx.globalAlpha = 0.6;
          ctx.strokeStyle = "#94a3b8";
          ctx.lineWidth = 1.5;
          for (const edge of edges) {{
            const sourceX = edge.source.layoutX * width;
            const sourceY = edge.source.layoutY * height;
            const targetX = edge.target.layoutX * width;
            const targetY = edge.target.layoutY * height;
            ctx.beginPath();
            ctx.moveTo(sourceX, sourceY);
            ctx.lineTo(targetX, targetY);
            ctx.stroke();
          }}
          ctx.globalAlpha = 1;
          for (const node of nodes) {{
            const nodeX = node.layoutX * width;
            const nodeY = node.layoutY * height;
            const radius = 10 + Math.min(15, (node.data.times_seen || 0) * 1.5);
            ctx.beginPath();
            ctx.fillStyle = selectedNode && selectedNode.id === node.id ? "#22d3ee" : "#60a5fa";
            ctx.strokeStyle = "#1d4ed8";
            ctx.lineWidth = selectedNode && selectedNode.id === node.id ? 3 : 1.5;
            ctx.arc(nodeX, nodeY, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }}
          ctx.fillStyle = "#0b1120";
          ctx.font = "11px Inter, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          for (const node of nodes) {{
            const label = node.data.label || node.id;
            const nodeX = node.layoutX * width;
            const nodeY = node.layoutY * height;
            const labelWidth = ctx.measureText(label).width + 10;
            ctx.save();
            ctx.fillStyle = "rgba(2, 6, 23, 0.75)";
            ctx.fillRect(nodeX - labelWidth / 2, nodeY - 24, labelWidth, 18);
            ctx.fillStyle = "#f8fafc";
            ctx.fillText(label, nodeX, nodeY - 15);
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
          const evidence = formatJSON(edge.data.evidence || {{}});
          details.innerHTML = "<h3>Edge</h3><pre>" +
            "Relation: " + (edge.data.relation || edge.data.relation_verbatim || "-") + "\\n" +
            "Confidence: " + Number.parseFloat(edge.data.confidence || 0).toFixed(3) + "\\n" +
            "Source -> Target: " + edge.source.data.label + " -> " + edge.target.data.label + "\\n" +
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
