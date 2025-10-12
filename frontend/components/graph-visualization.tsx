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
  relation: string;
  confidence: number;
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
  return `hsl(${hue}, 70%, 58%)`;
};

type SimulationNode = {
  node: GraphNode;
  x: number;
  y: number;
  dx: number;
  dy: number;
};

type LayoutResult = {
  nodes: PositionedNode[];
  edges: PositionedEdge[];
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
};

const runForceLayout = (
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number,
): LayoutResult => {
  if (!nodes.length) {
    return {
      nodes: [],
      edges: [],
      bounds: { minX: 0, maxX: width, minY: 0, maxY: height },
    };
  }

  const centerX = width / 2;
  const centerY = height / 2;
  const allowedIds = new Set(nodes.map((node) => node.id));
  const filteredEdges = edges.filter((edge) => allowedIds.has(edge.source) && allowedIds.has(edge.target));

  const simulationNodes: SimulationNode[] = nodes.map((node) => ({
    node,
    x: centerX + (Math.random() - 0.5) * width * 0.7,
    y: centerY + (Math.random() - 0.5) * height * 0.7,
    dx: 0,
    dy: 0,
  }));

  const nodeIndex = new Map(simulationNodes.map((entry) => [entry.node.id, entry]));

  const iterations = Math.min(300, 100 + simulationNodes.length * 4);
  const area = width * height;
  const k = Math.sqrt(area / simulationNodes.length);
  let temperature = Math.min(width, height) / 2;
  const coolingFactor = 0.92;
  const gravity = 0.08;
  const epsilon = 0.0001;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (const node of simulationNodes) {
      node.dx = 0;
      node.dy = 0;
    }

    for (let i = 0; i < simulationNodes.length; i += 1) {
      const nodeA = simulationNodes[i];
      for (let j = i + 1; j < simulationNodes.length; j += 1) {
        const nodeB = simulationNodes[j];
        const dx = nodeA.x - nodeB.x;
        const dy = nodeA.y - nodeB.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || epsilon;
        const force = (k * k) / distance;
        const offsetX = (dx / distance) * force;
        const offsetY = (dy / distance) * force;
        nodeA.dx += offsetX;
        nodeA.dy += offsetY;
        nodeB.dx -= offsetX;
        nodeB.dy -= offsetY;
      }
    }

    for (const edge of filteredEdges) {
      const source = nodeIndex.get(edge.source);
      const target = nodeIndex.get(edge.target);
      if (!source || !target) {
        continue;
      }
      const dx = source.x - target.x;
      const dy = source.y - target.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || epsilon;
      const force = (distance * distance) / k;
      const offsetX = (dx / distance) * force;
      const offsetY = (dy / distance) * force;
      source.dx -= offsetX;
      source.dy -= offsetY;
      target.dx += offsetX;
      target.dy += offsetY;
    }

    for (const node of simulationNodes) {
      const toCenterX = node.x - centerX;
      const toCenterY = node.y - centerY;
      node.dx -= toCenterX * gravity;
      node.dy -= toCenterY * gravity;

      const displacement = Math.sqrt(node.dx * node.dx + node.dy * node.dy) || epsilon;
      const limited = Math.min(displacement, temperature);
      node.x += (node.dx / displacement) * limited;
      node.y += (node.dy / displacement) * limited;

      node.x = Math.min(width, Math.max(0, node.x));
      node.y = Math.min(height, Math.max(0, node.y));
    }

    temperature *= coolingFactor;
    if (temperature < 1) {
      break;
    }
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  const positionedNodes: PositionedNode[] = simulationNodes.map((node) => {
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x);
    maxY = Math.max(maxY, node.y);
    return { node: node.node, x: node.x, y: node.y };
  });

  const positionedNodeIndex = new Map(positionedNodes.map((entry) => [entry.node.id, entry]));
  const positionedEdges: PositionedEdge[] = filteredEdges.map((edge) => {
    const source = positionedNodeIndex.get(edge.source);
    const target = positionedNodeIndex.get(edge.target);
    if (!source || !target) {
      throw new Error("Graph layout attempted to render an edge without positioned nodes");
    }
    return {
      id: edge.id,
      source,
      target,
      relation: edge.relation,
      confidence: edge.confidence,
    };
  });

  return {
    nodes: positionedNodes,
    edges: positionedEdges,
    bounds: { minX, maxX, minY, maxY },
  };
};

const truncateLabel = (label: string): string => {
  if (label.length <= 26) {
    return label;
  }
  return `${label.slice(0, 23)}…`;
};

const getNodeRadius = (node: GraphNode): number => {
  const baseRadius = 18;
  const scaled = Math.log10(Math.max(node.times_seen, 1));
  return baseRadius + scaled * 6;
};

const getEdgeStroke = (confidence: number): number => 1.2 + confidence * 1.4;

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

    const layout = runForceLayout(limitedNodes, edges, width, height);
    const margin = 80;
    const minX = Math.min(layout.bounds.minX, 0) - margin;
    const maxX = Math.max(layout.bounds.maxX, width) + margin;
    const minY = Math.min(layout.bounds.minY, 0) - margin;
    const maxY = Math.max(layout.bounds.maxY, height) + margin;

    const viewWidth = maxX - minX;
    const viewHeight = maxY - minY;

    const translatedNodes = layout.nodes.map((node) => ({
      ...node,
      x: node.x - minX,
      y: node.y - minY,
    }));

    const translatedNodeIndex = new Map(translatedNodes.map((entry) => [entry.node.id, entry]));
    const translatedEdges = layout.edges.map((edge) => {
      const source = translatedNodeIndex.get(edge.source.node.id);
      const target = translatedNodeIndex.get(edge.target.node.id);
      if (!source || !target) {
        throw new Error("Translated graph layout attempted to render an edge without positioned nodes");
      }
      return {
        ...edge,
        source,
        target,
      };
    });

    return {
      positionedNodes: translatedNodes,
      positionedEdges: translatedEdges,
      viewBox: `0 0 ${viewWidth} ${viewHeight}`,
    };
  }, [dimensions.height, dimensions.width, edges, nodes]);

  return (
    <div
      ref={containerRef}
      className="relative h-[420px] w-full overflow-hidden rounded-md border border-border bg-gradient-to-br from-background via-background/70 to-background"
    >
      <svg className="h-full w-full" viewBox={viewBox} role="img" aria-label="Knowledge graph preview">
        <defs>
          <marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8" fill="hsl(var(--muted-foreground))" />
          </marker>
          <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="1" stdDeviation="1.4" floodColor="rgba(0,0,0,0.25)" />
          </filter>
        </defs>
        <g stroke="hsl(var(--muted-foreground))" strokeOpacity="0.45">
          {positionedEdges.map((edge) => {
            const midX = (edge.source.x + edge.target.x) / 2;
            const midY = (edge.source.y + edge.target.y) / 2;
            const rawAngle = (Math.atan2(edge.target.y - edge.source.y, edge.target.x - edge.source.x) * 180) / Math.PI;
            const flipped = rawAngle > 90 || rawAngle < -90;
            const angle = flipped ? rawAngle + 180 : rawAngle;
            const label = edge.relation;
            const labelWidth = Math.min(140, Math.max(56, label.length * 6));
            return (
              <g key={edge.id}>
                <line
                  x1={edge.source.x}
                  y1={edge.source.y}
                  x2={edge.target.x}
                  y2={edge.target.y}
                  markerEnd="url(#graph-arrow)"
                  strokeWidth={getEdgeStroke(edge.confidence)}
                />
                <g transform={`translate(${midX}, ${midY}) rotate(${angle})`}>
                  <rect
                    x={-labelWidth / 2}
                    y={-10}
                    width={labelWidth}
                    height={20}
                    rx={6}
                    fill="hsl(var(--background))"
                    opacity={0.85}
                  />
                  <text
                    textAnchor="middle"
                    fontSize="9"
                    fill="hsl(var(--foreground))"
                    transform={flipped ? "scale(-1, -1)" : undefined}
                  >
                    {label}
                  </text>
                </g>
              </g>
            );
          })}
        </g>
        <g>
          {positionedNodes.map((entry) => (
            <g key={entry.node.id} transform={`translate(${entry.x}, ${entry.y})`}>
              <circle
                r={getNodeRadius(entry.node)}
                fill={hashColor(entry.node.type ?? null)}
                stroke="hsl(var(--card))"
                strokeWidth={2}
                opacity={0.95}
                filter="url(#node-shadow)"
              >
                <title>
                  {entry.node.label}
                  {entry.node.type ? `\nType: ${entry.node.type}` : ""}
                  {`\nTimes seen: ${entry.node.times_seen}`}
                  {`\nConfidence-weighted edges: ${
                    entry.node.section_distribution
                      ? Object.values(entry.node.section_distribution).reduce((acc, value) => acc + value, 0)
                      : 0
                  }`}
                </title>
              </circle>
              <text x={0} y={getNodeRadius(entry.node) + 16} textAnchor="middle" fontSize="11" fill="hsl(var(--foreground))">
                {truncateLabel(entry.node.label)}
              </text>
              {entry.node.type ? (
                <text
                  x={0}
                  y={getNodeRadius(entry.node) + 30}
                  textAnchor="middle"
                  fontSize="9"
                  fill="hsl(var(--muted-foreground))"
                >
                  {entry.node.type}
                </text>
              ) : null}
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
};

export default GraphVisualization;
