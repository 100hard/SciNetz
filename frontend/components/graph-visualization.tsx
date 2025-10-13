"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
const MIN_SCALE = 0.35;
const MAX_SCALE = 4.2;

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
  componentId: number;
  anchorX: number;
  anchorY: number;
  anchorRadius: number;
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

type GraphTransform = {
  scale: number;
  x: number;
  y: number;
};

const createSeededGenerator = (seed: string): (() => number) => {
  let value = 0;
  for (let index = 0; index < seed.length; index += 1) {
    value = (value * 31 + seed.charCodeAt(index)) >>> 0;
  }
  if (value === 0) {
    value = 0x9e3779b9;
  }
  return () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    value >>>= 0;
    return (value & 0xfffffff) / 0x10000000;
  };
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

  const maxDimension = Math.max(width, height);

  const adjacency = new Map<string, Set<string>>();
  for (const node of nodes) {
    adjacency.set(node.id, new Set());
  }
  for (const edge of filteredEdges) {
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }

  const componentByNode = new Map<string, number>();
  const componentSizes: number[] = [];
  for (const node of nodes) {
    if (componentByNode.has(node.id)) {
      continue;
    }
    const queue: string[] = [node.id];
    const componentId = componentSizes.length;
    componentByNode.set(node.id, componentId);
    let size = 0;
    while (queue.length) {
      const current = queue.pop();
      if (!current) {
        continue;
      }
      size += 1;
      const neighbours = adjacency.get(current);
      if (!neighbours) {
        continue;
      }
      for (const neighbour of neighbours) {
        if (!componentByNode.has(neighbour)) {
          componentByNode.set(neighbour, componentId);
          queue.push(neighbour);
        }
      }
    }
    componentSizes.push(size);
  }

  if (!componentSizes.length) {
    componentSizes.push(nodes.length);
  }

  const componentAnchors = new Map<
    number,
    { x: number; y: number; radius: number }
  >();
  const componentSpreadBase = Math.max(nodes.length, 1);
  for (let index = 0; index < componentSizes.length; index += 1) {
    const seeded = createSeededGenerator(`component-${index}`);
    const weight = componentSizes[index] / componentSpreadBase;
    const angle = seeded() * Math.PI * 2;
    const distance = maxDimension * (0.18 + seeded() * 0.22 + weight * 0.12);
    const radius = maxDimension * (0.38 + weight * 0.24 + seeded() * 0.1);
    componentAnchors.set(index, {
      x: centerX + Math.cos(angle) * distance,
      y: centerY + Math.sin(angle) * distance,
      radius,
    });
  }
  const simulationNodes: SimulationNode[] = nodes.map((node, index) => {
    const componentId = componentByNode.get(node.id) ?? 0;
    const anchor = componentAnchors.get(componentId);
    const seeded = createSeededGenerator(`${node.id}-${index}`);
    const angle = seeded() * Math.PI * 2;
    const radius = maxDimension * (0.22 + seeded() * 0.24);
    const jitterMagnitude = maxDimension * 0.04;
    const jitterX = (seeded() - 0.5) * 2 * jitterMagnitude;
    const jitterY = (seeded() - 0.5) * 2 * jitterMagnitude;
    return {
      node,
      componentId,
      anchorX: anchor?.x ?? centerX,
      anchorY: anchor?.y ?? centerY,
      anchorRadius: anchor?.radius ?? maxDimension * 0.6,
      x: (anchor?.x ?? centerX) + Math.cos(angle) * radius + jitterX,
      y: (anchor?.y ?? centerY) + Math.sin(angle) * radius + jitterY,
      dx: 0,
      dy: 0,
    };
  });

  const nodeIndex = new Map(simulationNodes.map((entry) => [entry.node.id, entry]));

  const iterations = Math.min(480, 160 + simulationNodes.length * 4);
  const area = width * height;
  const k = Math.sqrt(area / simulationNodes.length);
  let temperature = maxDimension / 1.3;
  const coolingFactor = 0.92;
  const gravity = 0.045;
  const repulsionStrength = 1.05;
  const attractionStrength = 0.055;
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
        const force = (repulsionStrength * k * k) / distance;
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
      const force = (attractionStrength * distance * distance) / k;
      const offsetX = (dx / distance) * force;
      const offsetY = (dy / distance) * force;
      source.dx -= offsetX;
      source.dy -= offsetY;
      target.dx += offsetX;
      target.dy += offsetY;
    }

    for (const node of simulationNodes) {
      const toAnchorX = node.x - node.anchorX;
      const toAnchorY = node.y - node.anchorY;
      node.dx -= toAnchorX * gravity;
      node.dy -= toAnchorY * gravity;

      const displacement = Math.sqrt(node.dx * node.dx + node.dy * node.dy) || epsilon;
      const limited = Math.min(displacement, temperature);
      node.x += (node.dx / displacement) * limited;
      node.y += (node.dy / displacement) * limited;

      const distanceToAnchor = Math.sqrt((node.x - node.anchorX) ** 2 + (node.y - node.anchorY) ** 2);
      const maxDistance = node.anchorRadius;
      if (distanceToAnchor > maxDistance) {
        const scale = maxDistance / distanceToAnchor;
        node.x = node.anchorX + (node.x - node.anchorX) * scale;
        node.y = node.anchorY + (node.y - node.anchorY) * scale;
      }
    }

    temperature *= coolingFactor;
    if (temperature < 0.4) {
      break;
    }
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  const positionedNodes: PositionedNode[] = simulationNodes.map((node, index) => {
    const jitterGenerator = createSeededGenerator(`post-${node.node.id}-${index}`);
    const jitterScale = maxDimension * 0.012;
    const jitterX = (jitterGenerator() - 0.5) * 2 * jitterScale;
    const jitterY = (jitterGenerator() - 0.5) * 2 * jitterScale;
    node.x += jitterX;
    node.y += jitterY;
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

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const GraphVisualization = ({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dimensions, setDimensions] = useState<Dimensions>({ width: 0, height: DEFAULT_HEIGHT });
  const [transform, setTransform] = useState<GraphTransform>({ scale: 1, x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panState = useRef({
    isActive: false,
    pointerId: null as number | null,
    startClientX: 0,
    startClientY: 0,
    originX: 0,
    originY: 0,
  });

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

  const { positionedNodes, positionedEdges, viewBox, viewWidth, viewHeight } = useMemo(() => {
    const limitedNodes = nodes.slice(0, GRAPH_VISUALIZATION_NODE_LIMIT);
    const width = Math.max(dimensions.width, 320);
    const height = Math.max(dimensions.height, DEFAULT_HEIGHT);
    if (!limitedNodes.length) {
      return {
        positionedNodes: [] as PositionedNode[],
        positionedEdges: [] as PositionedEdge[],
        viewBox: `0 0 ${width} ${height}`,
        viewWidth: width,
        viewHeight: height,
      };
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
      viewWidth,
      viewHeight,
    };
  }, [dimensions.height, dimensions.width, edges, nodes]);

  useEffect(() => {
    setTransform({ scale: 1, x: 0, y: 0 });
  }, [positionedNodes.length, positionedEdges.length]);

  const handleWheel = useCallback(
    (event: React.WheelEvent<SVGSVGElement>) => {
      if (!svgRef.current || viewWidth === 0 || viewHeight === 0) {
        return;
      }
      event.preventDefault();
      const svgBounds = svgRef.current.getBoundingClientRect();
      const pointer = {
        x: ((event.clientX - svgBounds.left) / svgBounds.width) * viewWidth,
        y: ((event.clientY - svgBounds.top) / svgBounds.height) * viewHeight,
      };
      const zoomFactor = event.deltaY < 0 ? 1.12 : 0.88;
      setTransform((current) => {
        const nextScale = clamp(current.scale * zoomFactor, MIN_SCALE, MAX_SCALE);
        const delta = current.scale - nextScale;
        return {
          scale: nextScale,
          x: current.x + delta * pointer.x,
          y: current.y + delta * pointer.y,
        };
      });
    },
    [viewHeight, viewWidth],
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!svgRef.current || viewWidth === 0 || viewHeight === 0) {
        return;
      }
      svgRef.current.setPointerCapture(event.pointerId);
      panState.current = {
        isActive: true,
        pointerId: event.pointerId,
        startClientX: event.clientX,
        startClientY: event.clientY,
        originX: transform.x,
        originY: transform.y,
      };
      setIsPanning(true);
    },
    [transform.x, transform.y, viewHeight, viewWidth],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!svgRef.current || !panState.current.isActive) {
        return;
      }
      const svgBounds = svgRef.current.getBoundingClientRect();
      const unitX = viewWidth / svgBounds.width;
      const unitY = viewHeight / svgBounds.height;
      const deltaX = (event.clientX - panState.current.startClientX) * unitX;
      const deltaY = (event.clientY - panState.current.startClientY) * unitY;
      setTransform((current) => ({
        ...current,
        x: panState.current.originX + deltaX,
        y: panState.current.originY + deltaY,
      }));
    },
    [viewHeight, viewWidth],
  );

  const endPan = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    if (svgRef.current && panState.current.pointerId !== null) {
      try {
        svgRef.current.releasePointerCapture(panState.current.pointerId);
      } catch (error) {
        console.error("Failed to release pointer capture", error);
      }
    }
    panState.current = {
      isActive: false,
      pointerId: null,
      startClientX: 0,
      startClientY: 0,
      originX: 0,
      originY: 0,
    };
    setIsPanning(false);
  }, []);

  const handleDoubleClick = useCallback(() => {
    setTransform({ scale: 1, x: 0, y: 0 });
  }, []);

  const zoomCentered = useCallback(
    (factor: number) => {
      if (viewWidth === 0 || viewHeight === 0) {
        return;
      }
      const center = { x: viewWidth / 2, y: viewHeight / 2 };
      setTransform((current) => {
        const nextScale = clamp(current.scale * factor, MIN_SCALE, MAX_SCALE);
        const delta = current.scale - nextScale;
        return {
          scale: nextScale,
          x: current.x + delta * center.x,
          y: current.y + delta * center.y,
        };
      });
    },
    [viewHeight, viewWidth],
  );

  const handleZoomIn = useCallback(() => {
    zoomCentered(1.12);
  }, [zoomCentered]);

  const handleZoomOut = useCallback(() => {
    zoomCentered(0.88);
  }, [zoomCentered]);

  return (
    <div
      ref={containerRef}
      className="relative h-[420px] w-full overflow-hidden rounded-md border border-border bg-gradient-to-br from-background via-background/70 to-background"
    >
      <div className="pointer-events-none absolute left-3 top-3 flex gap-2">
        <button
          type="button"
          onClick={handleZoomIn}
          className="pointer-events-auto rounded-md border border-border bg-background/80 px-2 py-1 text-xs font-medium shadow-sm transition hover:bg-background"
        >
          Zoom in
        </button>
        <button
          type="button"
          onClick={handleZoomOut}
          className="pointer-events-auto rounded-md border border-border bg-background/80 px-2 py-1 text-xs font-medium shadow-sm transition hover:bg-background"
        >
          Zoom out
        </button>
        <button
          type="button"
          onClick={handleDoubleClick}
          className="pointer-events-auto rounded-md border border-border bg-background/80 px-2 py-1 text-xs font-medium shadow-sm transition hover:bg-background"
        >
          Reset view
        </button>
      </div>
      <svg
        ref={svgRef}
        className={`h-full w-full ${isPanning ? "cursor-grabbing" : "cursor-grab"}`}
        viewBox={viewBox}
        role="img"
        aria-label="Knowledge graph preview"
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endPan}
        onPointerLeave={endPan}
        onPointerCancel={endPan}
        onDoubleClick={handleDoubleClick}
        style={{ touchAction: "none" }}
      >
        <defs>
          <marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8" fill="hsl(var(--muted-foreground))" />
          </marker>
          <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="1" stdDeviation="1.4" floodColor="rgba(0,0,0,0.25)" />
          </filter>
        </defs>
        <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
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
        </g>
      </svg>
    </div>
  );
};

export default GraphVisualization;
