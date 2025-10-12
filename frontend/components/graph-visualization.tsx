"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { GraphEdge, GraphNode } from "./graph-explorer";

export const GRAPH_VISUALIZATION_NODE_LIMIT = 200;

type Dimensions = {
  width: number;
  height: number;
};

type PositionedNode = {
  node: GraphNode;
  x: number;
  y: number;
};

type PositionedEdge = {
  id: string;
  source: PositionedNode;
  target: PositionedNode;
};

const DEFAULT_HEIGHT = 420;

const hashColor = (value: string | null | undefined): string => {
  if (!value) {
    return "hsl(var(--primary))";
  }
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) % 360;
  }
  const hue = Math.abs(hash);
  return `hsl(${hue}, 65%, 55%)`;
};

const truncateLabel = (label: string): string => {
  if (label.length <= 26) {
    return label;
  }
  return `${label.slice(0, 23)}…`;
};

const GraphVisualization = ({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState<Dimensions>({ width: 0, height: DEFAULT_HEIGHT });

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const entry = entries[0];
      setDimensions({
        width: entry.contentRect.width,
        height: Math.max(entry.contentRect.height, DEFAULT_HEIGHT),
      });
    });
    observer.observe(element);
    setDimensions({
      width: element.clientWidth,
      height: Math.max(element.clientHeight, DEFAULT_HEIGHT),
    });
    return () => {
      observer.disconnect();
    };
  }, []);

  const { positionedNodes, positionedEdges, viewBox } = useMemo(() => {
    const limitedNodes = nodes.slice(0, GRAPH_VISUALIZATION_NODE_LIMIT);
    const width = Math.max(dimensions.width, 320);
    const height = Math.max(dimensions.height, DEFAULT_HEIGHT);
    if (!limitedNodes.length) {
      return { positionedNodes: [] as PositionedNode[], positionedEdges: [] as PositionedEdge[], viewBox: `0 0 ${width} ${height}` };
    }
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.max(Math.min(width, height) / 2 - 60, Math.min(width, height) * 0.3);
    const arrangedNodes: PositionedNode[] = limitedNodes.map((node, index) => {
      const angle = (2 * Math.PI * index) / limitedNodes.length;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      return { node, x, y };
    });
    const nodeIndex = new Map(arrangedNodes.map((item) => [item.node.id, item]));
    const allowedIds = new Set(limitedNodes.map((node) => node.id));
    const arrangedEdges: PositionedEdge[] = edges
      .filter((edge) => allowedIds.has(edge.source) && allowedIds.has(edge.target))
      .map((edge) => {
        const source = nodeIndex.get(edge.source);
        const target = nodeIndex.get(edge.target);
        if (!source || !target) {
          throw new Error("Graph layout attempted to render an edge without positioned nodes");
        }
        return {
          id: edge.id,
          source,
          target,
        };
      });
    return {
      positionedNodes: arrangedNodes,
      positionedEdges: arrangedEdges,
      viewBox: `0 0 ${width} ${height}`,
    };
  }, [dimensions.height, dimensions.width, edges, nodes]);

  return (
    <div ref={containerRef} className="relative h-[420px] w-full overflow-hidden rounded-md border border-border bg-background">
      <svg className="h-full w-full" viewBox={viewBox} role="img" aria-label="Knowledge graph preview">
        <defs>
          <marker
            id="graph-arrow"
            markerWidth="8"
            markerHeight="8"
            refX="8"
            refY="4"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L8,4 L0,8" fill="hsl(var(--muted-foreground))" />
          </marker>
        </defs>
        <g stroke="hsl(var(--muted-foreground))" strokeWidth="1" strokeOpacity="0.35">
          {positionedEdges.map((edge) => (
            <line
              key={edge.id}
              x1={edge.source.x}
              y1={edge.source.y}
              x2={edge.target.x}
              y2={edge.target.y}
              markerEnd="url(#graph-arrow)"
            />
          ))}
        </g>
        <g>
          {positionedNodes.map((entry) => (
            <g key={entry.node.id} transform={`translate(${entry.x}, ${entry.y})`}>
              <circle
                r={12}
                fill={hashColor(entry.node.type ?? null)}
                stroke="hsl(var(--card))"
                strokeWidth={1.5}
                opacity={0.9}
              >
                <title>
                  {entry.node.label}
                  {entry.node.type ? `\nType: ${entry.node.type}` : ""}
                  {`\nTimes seen: ${entry.node.times_seen}`}
                  {`\nConfidence-weighted edges: ${entry.node.section_distribution ? Object.values(entry.node.section_distribution).reduce((acc, value) => acc + value, 0) : 0}`}
                </title>
              </circle>
              <text
                x={0}
                y={22}
                textAnchor="middle"
                fontSize="10"
                fill="hsl(var(--muted-foreground))"
              >
                {truncateLabel(entry.node.label)}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
};

export default GraphVisualization;
