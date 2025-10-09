"use client";

import dynamic from "next/dynamic";
import type cytoscape from "cytoscape";
import { useEffect, useMemo, useState } from "react";

import type { GraphEdge, GraphNode } from "../lib/types";

const CytoscapeComponent = dynamic(() => import("react-cytoscapejs"), { ssr: false });

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  layout: string;
  onSelectEdge: (edge: GraphEdge) => void;
}

export function GraphCanvas({ nodes, edges, layout, onSelectEdge }: GraphCanvasProps) {
  const [cyInstance, setCyInstance] = useState<cytoscape.Core | null>(null);

  const elements = useMemo(() => {
    const nodeElements = nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.label,
        degree: node.aliases.length,
        times: node.times_seen
      }
    }));
    const edgeElements = edges.map((edge) => ({
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: `${edge.relation} (${edge.confidence.toFixed(2)})`
      },
      classes: edge.conflicting ? "conflicting" : edge.attributes.hidden === "true" ? "hidden" : ""
    }));
    return [...nodeElements, ...edgeElements];
  }, [nodes, edges]);

  useEffect(() => {
    if (!cyInstance) {
      return;
    }
    const handler = (event: cytoscape.EventObject) => {
      const id = event.target.id();
      const edge = edges.find((candidate) => candidate.id === id);
      if (edge) {
        onSelectEdge(edge);
      }
    };
    cyInstance.on("tap", "edge", handler);
    return () => {
      cyInstance.off("tap", "edge", handler);
    };
  }, [cyInstance, edges, onSelectEdge]);

  const cytoscapeLayout = useMemo(() => ({
    name: layout === "cose-bilkent" ? "cose-bilkent" : "fcose",
    animate: false,
    fit: true,
    padding: 40
  }), [layout]);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900">
      <CytoscapeComponent
        elements={elements}
        style={{ width: "100%", height: "420px" }}
        layout={cytoscapeLayout}
        cy={setCyInstance}
        stylesheet={stylesheet}
      />
    </div>
  );
}

const stylesheet: cytoscape.Stylesheet[] = [
  {
    selector: "node",
    style: {
      "background-color": "#38bdf8",
      label: "data(label)",
      color: "#0f172a",
      "font-size": 12,
      "text-valign": "center",
      "text-halign": "center",
      width: (ele) => 30 + Math.min(40, ele.data("times") * 4),
      height: (ele) => 30 + Math.min(40, ele.data("times") * 4)
    }
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "line-color": "#94a3b8",
      "target-arrow-color": "#94a3b8",
      "target-arrow-shape": "triangle",
      label: "data(label)",
      color: "#cbd5f5",
      "font-size": 10,
      "curve-style": "bezier"
    }
  },
  {
    selector: "edge.conflicting",
    style: {
      "line-color": "#f97316",
      "target-arrow-color": "#f97316"
    }
  },
  {
    selector: "edge.hidden",
    style: {
      opacity: 0.35,
      "line-style": "dashed"
    }
  }
];
