"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";

import type { GraphEdge, GraphNode } from "./graph-explorer";

type GraphVisualizationProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  isFullscreen?: boolean;
};

const DEFAULT_HEIGHT = 420;
const CROSS_PAPER_KEYS = ["cross_paper", "cross-paper", "crossPaper"];

const TYPE_COLOR_MAP: Record<string, string> = {
  method: "#2563eb",
  methods: "#2563eb",
  dataset: "#16a34a",
  datasets: "#16a34a",
  metric: "#7c3aed",
  metrics: "#7c3aed",
  task: "#ea580c",
  tasks: "#ea580c",
  entity: "#0ea5e9",
};

const RELATION_COLOR_MAP: Record<string, string> = {
  evaluates: "#2563eb",
  uses: "#0ea5e9",
  reports: "#16a34a",
  improves: "#7c3aed",
  predicts: "#f97316",
  extends: "#db2777",
};

const EDGE_CROSS_PAPER_COLOR = "#fbbf24";

const RING_LABELS: Record<number, { title: string; description: string }> = {
  0: { title: "Core Hypothesis", description: "Primary method or idea" },
  1: { title: "Supporting Methodology", description: "Key mechanisms & techniques" },
  2: { title: "Building Blocks", description: "Components, parameters, reagents" },
  3: { title: "Data & Inputs", description: "Datasets, corpora, experimental setups" },
  4: { title: "Evidence & Metrics", description: "Results, metrics, observations" },
  5: { title: "Context", description: "Background or related concepts" },
};
const RING_ORDER = [0, 1, 2, 3, 4, 5];

const resolveRingAccent = (ring?: number | null): string | undefined => {
  switch (ring) {
    case 0:
      return "#facc15";
    case 1:
      return "#fb7185";
    case 2:
      return "#38bdf8";
    case 3:
      return "#34d399";
    case 4:
      return "#a855f7";
    default:
      return undefined;
  }
};

const describeEdge = (
  edge: GraphEdge,
  labelMap: Map<string, string>,
): { text: string; color: string } => {
  const source = labelMap.get(edge.source) ?? edge.source;
  const target = labelMap.get(edge.target) ?? edge.target;
  const relation = edge.relation_verbatim || edge.relation;
  const color = resolveEdgeColor(edge);
  return {
    text: `${source} ${relation} ${target}`,
    color,
  };
};

const hashToUnit = (value: string, salt: string): number => {
  const input = `${salt}:${value}`;
  let hash = 0;
  for (let index = 0; index < input.length; index += 1) {
    hash = (hash * 31 + input.charCodeAt(index)) >>> 0;
  }
  return (hash % 10000) / 10000;
};

const computeWebLayout = (nodes: GraphNode[]): Map<string, { x: number; y: number }> => {
  const positions = new Map<string, { x: number; y: number }>();
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const baseRadius = 0.16;
  const ringSpacing = 0.12;
  const radiusJitter = 0.12;
  const angleJitter = Math.PI / 6;

  const ringBuckets = new Map<number, GraphNode[]>();
  nodes.forEach((node) => {
    const ring = node.layout_ring ?? 5;
    if (!ringBuckets.has(ring)) {
      ringBuckets.set(ring, []);
    }
    ringBuckets.get(ring)?.push(node);
  });

  const sortedRings = Array.from(ringBuckets.entries()).sort((a, b) => a[0] - b[0]);
  sortedRings.forEach(([ring, ringNodes]) => {
    const ringOffset = hashToUnit(String(ring), "ring-offset") * Math.PI * 2;
    ringNodes
      .slice()
      .sort((a, b) => a.id.localeCompare(b.id))
      .forEach((node, index) => {
        const baseAngle = ringOffset + index * goldenAngle;
        const jitteredAngle =
          baseAngle + (hashToUnit(node.id, "angle") - 0.5) * 2 * angleJitter;
        const jitteredRadius =
          baseRadius +
          ringSpacing * ring +
          (hashToUnit(node.id, "radius") - 0.5) * 2 * radiusJitter;
        const x = Math.cos(jitteredAngle) * jitteredRadius;
        const y = Math.sin(jitteredAngle) * jitteredRadius;
        positions.set(node.id, { x, y });
      });
  });

  return positions;
};

const resolveNodeColor = (node: GraphNode): string => {
  const type = node.type?.trim().toLowerCase();
  if (type && TYPE_COLOR_MAP[type]) {
    return TYPE_COLOR_MAP[type];
  }
  if (node.section_distribution) {
    const sections = Object.keys(node.section_distribution);
    if (sections.length === 1) {
      const mapped = TYPE_COLOR_MAP[sections[0].toLowerCase()];
      if (mapped) {
        return mapped;
      }
    }
  }
  return "#2563eb";
};

const isCrossPaperEdge = (edge: GraphEdge): boolean => {
  const attributes = edge.attributes;
  if (!attributes) {
    return false;
  }
  return CROSS_PAPER_KEYS.some((key) => {
    const value = attributes[key];
    if (value === undefined || value === null) {
      return false;
    }
    const normalised = String(value).trim().toLowerCase();
    return normalised === "true" || normalised === "1" || normalised === "yes";
  });
};

const resolveEdgeColor = (edge: GraphEdge): string => {
  if (isCrossPaperEdge(edge)) {
    return EDGE_CROSS_PAPER_COLOR;
  }
  const relation = edge.relation.trim().toLowerCase();
  return RELATION_COLOR_MAP[relation] ?? "rgba(15, 23, 42, 0.6)";
};

const computeNodeSize = (node: GraphNode): number => {
  const importance = typeof node.importance === "number" ? node.importance : Math.min(1, Math.log2(node.times_seen + 1) / 6);
  const ring = node.layout_ring ?? 5;
  const ringBoost = ring === 0 ? 1.9 : ring === 1 ? 1.5 : ring === 2 ? 1.25 : ring === 3 ? 1.1 : ring === 4 ? 1.0 : 0.9;
  return Math.max(6, 12 * importance * ringBoost);
};

const computeEdgeSize = (edge: GraphEdge): number => {
  const base = Math.log2(edge.times_seen + 1);
  return Math.max(0.8, base);
};

const hasPositiveDimensions = (element: HTMLElement | null): boolean => {
  if (!element) {
    return false;
  }
  return element.offsetWidth > 0 && element.offsetHeight > 0;
};

type FocusState = {
  nodeId: string;
  neighborIds: string[];
  edges: GraphEdge[];
};

const GraphVisualization = ({ nodes, edges, isFullscreen = false }: GraphVisualizationProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sigmaRef = useRef<InstanceType<typeof Sigma> | null>(null);
  const graphRef = useRef<InstanceType<typeof Graph> | null>(null);
  const nodesRef = useRef<GraphNode[]>(nodes);
  const edgesRef = useRef<GraphEdge[]>(edges);
  const pendingRefreshRef = useRef(false);
  const pendingCameraResetRef = useRef(false);
  const [focusState, setFocusState] = useState<FocusState | null>(null);
  const nodeLabelMap = useMemo(() => {
    const map = new Map<string, string>();
    nodes.forEach((node) => map.set(node.id, node.label));
    return map;
  }, [nodes]);

  useEffect(() => {
    nodesRef.current = nodes;
    edgesRef.current = edges;
  }, [nodes, edges]);

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const observer = new ResizeObserver(() => {
      const sigma = sigmaRef.current;
      const currentContainer = containerRef.current;
      if (!sigma || !hasPositiveDimensions(currentContainer)) {
        return;
      }

      sigma.refresh();
      pendingRefreshRef.current = false;

      if (pendingCameraResetRef.current && nodesRef.current.length > 0) {
        sigma.getCamera().animatedReset({ duration: 300 });
        pendingCameraResetRef.current = false;
      }
    });

    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const graph = new Graph({ type: "directed", multi: true });
    graphRef.current = graph;

    const sigma = new Sigma(graph, container, {
      allowInvalidContainer: false,
      renderLabels: true,
      defaultEdgeType: "arrow",
      labelDensity: 1,
      minCameraRatio: 0.05,
      maxCameraRatio: 50,
    });

    sigmaRef.current = sigma;

    return () => {
      sigma.kill();
      graph.clear();
      graphRef.current = null;
      sigmaRef.current = null;
    };
  }, []);

  useEffect(() => {
    const sigma = sigmaRef.current;
    if (!sigma) {
      return;
    }
    const effectSigma = sigma;
    const effectContainer = containerRef.current;
    const highlightedNodes = focusState ? new Set([focusState.nodeId, ...focusState.neighborIds]) : null;
    const highlightedEdgeIds = focusState ? new Set(focusState.edges.map((edge) => edge.id)) : null;

    sigma.setSetting("nodeReducer", (node, data) => {
      if (!highlightedNodes) {
        return data;
      }
      if (highlightedNodes.has(node)) {
        return data;
      }
      return {
        ...data,
        color: "#cbd5f5",
        opacity: 0.2,
      };
    });

    sigma.setSetting("edgeReducer", (edge, data) => {
      if (!highlightedEdgeIds) {
        return data;
      }
      if (highlightedEdgeIds.has(edge)) {
        return data;
      }
      return {
        ...data,
        color: "rgba(148, 163, 184, 0.25)",
      };
    });

    if (hasPositiveDimensions(effectContainer)) {
      sigma.refresh();
      pendingRefreshRef.current = false;
    } else {
      pendingRefreshRef.current = true;
    }
    return () => {
      if (sigmaRef.current !== effectSigma) {
        return;
      }
      sigma.setSetting("nodeReducer", null);
      sigma.setSetting("edgeReducer", null);
      if (hasPositiveDimensions(effectContainer)) {
        sigma.refresh();
        pendingRefreshRef.current = false;
      } else {
        pendingRefreshRef.current = true;
      }
    };
  }, [focusState]);

  useEffect(() => {
    const sigma = sigmaRef.current;
    if (!sigma) {
      return;
    }
    const handleEnterNode = ({ node }: { node: string }) => {
      const graph = sigma.getGraph();
      const neighborIds = graph.neighbors(node);
      const relatedEdges = edgesRef.current
        .filter((edge) => edge.source === node || edge.target === node)
        .sort((a, b) => b.times_seen - a.times_seen)
        .slice(0, 6);
      setFocusState({
        nodeId: node,
        neighborIds,
        edges: relatedEdges,
      });
    };
    const handleLeaveNode = () => {
      setFocusState(null);
    };
    sigma.on("enterNode", handleEnterNode);
    sigma.on("leaveNode", handleLeaveNode);
    return () => {
      if (sigmaRef.current !== sigma) {
        return;
      }
      sigma.off("enterNode", handleEnterNode);
      sigma.off("leaveNode", handleLeaveNode);
    };
  }, []);

  useEffect(() => {
    const graph = graphRef.current;
    const sigma = sigmaRef.current;
    if (!graph || !sigma) {
      return;
    }

    graph.clear();

    const hasCoordinates = nodes.every((node) => typeof node.x === "number" && typeof node.y === "number");

    const seededPositions = hasCoordinates ? null : computeWebLayout(nodes);

    nodes.forEach((node) => {
      const seeded = seededPositions?.get(node.id);
      const x = hasCoordinates ? node.x : seeded?.x ?? hashToUnit(node.id, "x") * 2 - 1;
      const y = hasCoordinates ? node.y : seeded?.y ?? hashToUnit(node.id, "y") * 2 - 1;
      const nodeType = typeof node.type === "string" && node.type.trim().length > 0 ? node.type.trim() : "default";
      const normalisedType = nodeType.toLowerCase();
      const displayType = normalisedType.length > 0 ? normalisedType : "default";

      const accent = resolveRingAccent(node.layout_ring);
      graph.addNode(node.id, {
        label: node.label,
        x,
        y,
        size: computeNodeSize(node),
        color: resolveNodeColor(node),
        category: displayType,
        borderColor: accent ?? "rgba(15, 23, 42, 0.08)",
        borderSize: accent ? (node.layout_ring === 0 ? 3 : 1.5) : 1,
        zIndex: node.layout_ring === 0 ? 10 : 1,
      });
    });

    edges.forEach((edge) => {
      if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) {
        return;
      }
      const color = resolveEdgeColor(edge);
      graph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
        size: computeEdgeSize(edge),
        color,
        label: edge.relation_verbatim || edge.relation,
        type: "arrow",
      });
    });

    const shouldRunLayout = graph.order > 1 && graph.size > 0;
    if (shouldRunLayout) {
      try {
        const iterations = Math.min(600, Math.max(240, Math.round(nodes.length * 4)));
        const inferred = forceAtlas2.inferSettings(graph);
        forceAtlas2.assign(graph, {
          iterations,
          settings: {
            ...inferred,
            adjustSizes: true,
            scalingRatio: 10,
            gravity: 1.6,
            strongGravityMode: true,
            linLogMode: true,
            barnesHutOptimize: true,
            barnesHutTheta: 0.6,
            outboundAttractionDistribution: true,
            slowDown: 2.1,
          },
        });
      } catch (error) {
        // eslint-disable-next-line no-console -- surfaced in dev tools only
        console.error("forceAtlas2 layout failed", error);
      }
    }

    if (!hasPositiveDimensions(containerRef.current)) {
      pendingRefreshRef.current = true;
      pendingCameraResetRef.current = nodes.length > 0;
      return;
    }

    sigma.refresh();
    pendingRefreshRef.current = false;

    if (nodes.length > 0) {
      sigma.getCamera().animatedReset({ duration: 300 });
    }
    pendingCameraResetRef.current = false;
  }, [nodes, edges]);

  const containerHeight = isFullscreen ? "calc(100vh - 10rem)" : `${DEFAULT_HEIGHT}px`;
  const focusNode = focusState ? nodes.find((node) => node.id === focusState.nodeId) ?? null : null;
  const focusRingLabel = focusNode ? RING_LABELS[focusNode.layout_ring ?? 5] : null;

  return (
    <div className="relative w-full" style={{ height: containerHeight }}>
      <div ref={containerRef} className="h-full w-full rounded-lg border border-border bg-card shadow-sm" />
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-4">
        {focusNode ? (
          <div className="pointer-events-auto max-w-sm rounded-lg border border-slate-200 bg-white/95 p-4 text-sm text-slate-800 shadow-lg backdrop-blur-sm">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Focused concept</p>
            <h3 className="text-lg font-bold leading-tight text-slate-900">
              {focusNode.label}
            </h3>
            {focusRingLabel ? (
              <p className="mt-1 text-xs text-slate-500">
                {focusRingLabel.title} · {focusRingLabel.description}
              </p>
            ) : null}
            <ul className="mt-4 space-y-3">
              {focusState?.edges.map((edge) => {
                const narrative = describeEdge(edge, nodeLabelMap);
                const confidenceText =
                  typeof edge.confidence === "number" && edge.confidence > 0
                    ? `${(edge.confidence * 100).toFixed(0)}% confidence`
                    : "evidence only";
                return (
                  <li key={edge.id} className="flex gap-3">
                    <span
                      className="mt-1 h-2 w-2 flex-shrink-0 rounded-full"
                      style={{ backgroundColor: narrative.color }}
                    />
                    <div>
                      <p className="text-xs font-medium text-slate-800">{narrative.text}</p>
                      <p className="text-[11px] text-slate-500">
                        {confidenceText} · seen {edge.times_seen}×
                      </p>
                    </div>
                  </li>
                );
              })}
              {focusState?.edges.length === 0 ? (
                <li className="text-xs text-slate-500">No direct relations captured yet.</li>
              ) : null}
            </ul>
          </div>
        ) : (
          <div className="pointer-events-none max-w-sm rounded-lg border border-dashed border-slate-300 bg-white/80 p-3 text-xs text-slate-600 shadow-sm backdrop-blur-sm">
            Hover a node to see how it links hypotheses, components, and evidence.
          </div>
        )}
        <div className="pointer-events-auto self-end rounded-lg border border-slate-200 bg-white/90 p-3 text-xs text-slate-700 shadow">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">Legend</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {RING_ORDER.map((ring) => {
              const entry = RING_LABELS[ring];
              if (!entry) {
                return null;
              }
              const accent = resolveRingAccent(ring);
              return (
                <div key={ring} className="flex items-start gap-2">
                  <span
                    className="mt-1 h-2.5 w-2.5 flex-shrink-0 rounded-full border"
                    style={{
                      backgroundColor: accent ?? "#e2e8f0",
                      borderColor: accent ? accent : "#cbd5f5",
                    }}
                  />
                  <div>
                    <p className="text-xs font-semibold text-slate-800">{entry.title}</p>
                    <p className="text-[11px] text-slate-500">{entry.description}</p>
                  </div>
                </div>
              );
            })}
            <div className="flex items-start gap-2">
              <span
                className="mt-1 h-2.5 w-2.5 flex-shrink-0 rounded-full border border-amber-400 bg-amber-300/70"
                title="Cross-paper link"
              />
              <div>
                <p className="text-xs font-semibold text-slate-800">Cross-paper evidence</p>
                <p className="text-[11px] text-slate-500">Shared concept linking multiple publications</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphVisualization;
