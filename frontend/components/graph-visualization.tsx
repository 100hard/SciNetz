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
  degree: number;
  componentId: number;
  anchorX: number;
  anchorY: number;
};

type PositionedEdge = {
  id: string;
  source: PositionedNode;
  target: PositionedNode;
  relation: string;
  confidence: number;
};

type StyledNode = PositionedNode & {
  radius: number;
  fill: string;
  labelLines: string[];
  labelColor: string;
  labelOutline: string;
  strokeColor: string;
  strokeWidth: number;
};

type StyledEdge = PositionedEdge & {
  stroke: string;
  markerKey: string;
  labelColor: string;
  strokeOpacity: number;
};

type ComponentBackground = {
  id: number;
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  fill: string;
  stroke: string;
  label: string;
};

const DEFAULT_HEIGHT = 420;
const MIN_SCALE = 0.35;
const MAX_SCALE = 4.2;
const NODE_STROKE_COLOR = "#0f172a";
const NODE_LABEL_LIGHT_COLOR = "#f8fafc";
const NODE_LABEL_DARK_COLOR = "#0f172a";
const NODE_LABEL_FONT_SIZE = 13;
const NODE_LABEL_FONT_WEIGHT = 700;
const NODE_LABEL_LINE_HEIGHT = 16;
const NODE_LABEL_VERTICAL_PADDING = 10;
const MIN_NODE_RADIUS = 30;
const NODE_RADIUS_SCALE = 1.35;
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

const hashColor = (value: string | null | undefined): string => {
  if (!value) {
    return "hsl(var(--primary))";
  }
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) % 360;
  }
  const hue = Math.abs(hash);
  return `hsl(${hue}, 65%, 65%)`;
};

type RgbColor = { r: number; g: number; b: number };

const parseHexColor = (value: string): RgbColor | null => {
  const hex = value.replace("#", "").trim();
  if (hex.length === 3) {
    const r = Number.parseInt(hex[0] + hex[0], 16);
    const g = Number.parseInt(hex[1] + hex[1], 16);
    const b = Number.parseInt(hex[2] + hex[2], 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
      return null;
    }
    return { r, g, b };
  }
  if (hex.length === 6) {
    const r = Number.parseInt(hex.slice(0, 2), 16);
    const g = Number.parseInt(hex.slice(2, 4), 16);
    const b = Number.parseInt(hex.slice(4, 6), 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
      return null;
    }
    return { r, g, b };
  }
  return null;
};

const parseRgbColor = (value: string): RgbColor | null => {
  const match = value.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
  if (!match) {
    return null;
  }
  const r = Math.min(255, Math.max(0, Number.parseFloat(match[1])));
  const g = Math.min(255, Math.max(0, Number.parseFloat(match[2])));
  const b = Math.min(255, Math.max(0, Number.parseFloat(match[3])));
  if ([r, g, b].some((channel) => Number.isNaN(channel))) {
    return null;
  }
  return { r, g, b };
};

const parseHslColor = (value: string): RgbColor | null => {
  const match = value.match(/hsla?\(\s*([\d.+-]+)(deg|rad|turn)?\s*,\s*([\d.+-]+)%\s*,\s*([\d.+-]+)%/i);
  if (!match) {
    return null;
  }
  let hue = Number.parseFloat(match[1]);
  if (Number.isNaN(hue)) {
    return null;
  }
  const unit = (match[2] ?? "deg").toLowerCase();
  if (unit === "rad") {
    hue = (hue * 180) / Math.PI;
  } else if (unit === "turn") {
    hue *= 360;
  }
  const saturation = Math.min(100, Math.max(0, Number.parseFloat(match[3]))) / 100;
  const lightness = Math.min(100, Math.max(0, Number.parseFloat(match[4]))) / 100;
  if ([saturation, lightness].some((component) => Number.isNaN(component))) {
    return null;
  }
  const h = (((hue % 360) + 360) % 360) / 360;
  const q = lightness < 0.5 ? lightness * (1 + saturation) : lightness + saturation - lightness * saturation;
  const p = 2 * lightness - q;
  const hueToRgb = (t: number) => {
    let temp = t;
    if (temp < 0) {
      temp += 1;
    }
    if (temp > 1) {
      temp -= 1;
    }
    if (temp < 1 / 6) {
      return p + (q - p) * 6 * temp;
    }
    if (temp < 1 / 2) {
      return q;
    }
    if (temp < 2 / 3) {
      return p + (q - p) * (2 / 3 - temp) * 6;
    }
    return p;
  };
  const r = Math.round(hueToRgb(h + 1 / 3) * 255);
  const g = Math.round(hueToRgb(h) * 255);
  const b = Math.round(hueToRgb(h - 1 / 3) * 255);
  return { r, g, b };
};

const toRgbColor = (value: string): RgbColor | null => {
  const trimmed = value.trim();
  const lower = trimmed.toLowerCase();
  if (lower.startsWith("#")) {
    return parseHexColor(trimmed);
  }
  if (lower.startsWith("rgb")) {
    return parseRgbColor(trimmed);
  }
  if (lower.startsWith("hsl")) {
    return parseHslColor(trimmed);
  }
  return null;
};

const channelToLinear = (channel: number): number => {
  const normalized = channel / 255;
  if (normalized <= 0.03928) {
    return normalized / 12.92;
  }
  return ((normalized + 0.055) / 1.055) ** 2.4;
};

const getRelativeLuminance = ({ r, g, b }: RgbColor): number => {
  const red = channelToLinear(r);
  const green = channelToLinear(g);
  const blue = channelToLinear(b);
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
};

const getContrastingLabelColors = (fill: string): { color: string; outline: string } => {
  const rgb = toRgbColor(fill);
  if (!rgb) {
    return { color: NODE_LABEL_LIGHT_COLOR, outline: "rgba(15, 23, 42, 0.45)" };
  }
  const luminance = getRelativeLuminance(rgb);
  if (luminance > 0.55) {
    return { color: NODE_LABEL_DARK_COLOR, outline: "rgba(255, 255, 255, 0.7)" };
  }
  return { color: NODE_LABEL_LIGHT_COLOR, outline: "rgba(15, 23, 42, 0.5)" };
};

type SimulationNode = {
  node: GraphNode;
  componentId: number;
  anchorX: number;
  anchorY: number;
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

  const componentAnchors = new Map<number, { x: number; y: number; spread: number; noise: number }>();
  const componentSpreadBase = Math.max(nodes.length, 1);
  const minDimension = Math.min(width, height);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let index = 0; index < componentSizes.length; index += 1) {
    const seeded = createSeededGenerator(`component-${index}`);
    const weight = componentSizes[index] / componentSpreadBase;
    const angle = index === 0 ? 0 : goldenAngle * index + seeded() * 0.16;
    const radialStep = minDimension * (0.035 + weight * 0.032);
    const distance =
      index === 0
        ? 0
        : Math.min(
            minDimension * 0.095,
            Math.sqrt(index + 1) * radialStep + seeded() * minDimension * 0.01,
          );
    const spread = minDimension * (0.11 + weight * 0.085 + seeded() * 0.02);
    const noise = (seeded() - 0.5) * minDimension * 0.01;
    componentAnchors.set(index, {
      x: centerX + Math.cos(angle) * distance,
      y: centerY + Math.sin(angle) * distance,
      spread,
      noise,
    });
  }
  const simulationNodes: SimulationNode[] = nodes.map((node, index) => {
    const componentId = componentByNode.get(node.id) ?? 0;
    const anchor = componentAnchors.get(componentId);
    const seeded = createSeededGenerator(`${node.id}-${index}`);
    const angle = seeded() * Math.PI * 2;
    const spread = anchor?.spread ?? minDimension * 0.4;
    const radius = spread * (0.12 + seeded() * 0.4);
    const jitterMagnitude = spread * 0.05;
    const jitterX = (seeded() - 0.5) * 2 * jitterMagnitude;
    const jitterY = (seeded() - 0.5) * 2 * jitterMagnitude;
    const noise = anchor?.noise ?? 0;
    return {
      node,
      componentId,
      anchorX: (anchor?.x ?? centerX) + noise,
      anchorY: (anchor?.y ?? centerY) - noise,
      x: (anchor?.x ?? centerX) + Math.cos(angle) * radius + jitterX + noise,
      y: (anchor?.y ?? centerY) + Math.sin(angle) * radius + jitterY - noise,
      dx: 0,
      dy: 0,
    };
  });

  const nodeIndex = new Map(simulationNodes.map((entry) => [entry.node.id, entry]));

  const iterations = Math.min(560, 220 + simulationNodes.length * 3);
  const area = width * height;
  const k = Math.sqrt(area / simulationNodes.length);
  let temperature = maxDimension / 1.25;
  const coolingFactor = 0.94;
  const gravity = 0.028;
  const centerGravity = 0.032;
  const repulsionStrength = 0.9;
  const attractionStrength = 0.068;
  const crossComponentPull = 0.064;
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
      const toCenterX = node.x - centerX;
      const toCenterY = node.y - centerY;
      node.dx -= toCenterX * centerGravity;
      node.dy -= toCenterY * centerGravity;
      if (node.componentId !== 0) {
        const blendedAnchorX = (node.anchorX + centerX) / 2;
        const blendedAnchorY = (node.anchorY + centerY) / 2;
        node.dx -= (node.x - blendedAnchorX) * crossComponentPull;
        node.dy -= (node.y - blendedAnchorY) * crossComponentPull;
      }

      const displacement = Math.sqrt(node.dx * node.dx + node.dy * node.dy) || epsilon;
      const limited = Math.min(displacement, temperature);
      node.x += (node.dx / displacement) * limited;
      node.y += (node.dy / displacement) * limited;
    }

    temperature *= coolingFactor;
    if (temperature < 0.4) {
      break;
    }
  }

  const positionedNodes: PositionedNode[] = simulationNodes.map((node, index) => {
    const jitterGenerator = createSeededGenerator(`post-${node.node.id}-${index}`);
    const jitterScale = maxDimension * 0.012;
    const jitterX = (jitterGenerator() - 0.5) * 2 * jitterScale;
    const jitterY = (jitterGenerator() - 0.5) * 2 * jitterScale;
    node.x += jitterX;
    node.y += jitterY;
    const neighbours = adjacency.get(node.node.id);
    return {
      node: node.node,
      x: node.x,
      y: node.y,
      degree: neighbours?.size ?? 0,
      componentId: node.componentId,
      anchorX: node.anchorX,
      anchorY: node.anchorY,
    };
  });

  resolveCollisions(positionedNodes);

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const entry of positionedNodes) {
    minX = Math.min(minX, entry.x);
    minY = Math.min(minY, entry.y);
    maxX = Math.max(maxX, entry.x);
    maxY = Math.max(maxY, entry.y);
  }

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

const formatNodeLabel = (label: string): string[] => {
  const words = label.split(/\s+/).filter(Boolean);
  if (!words.length) {
    return ["Unknown"];
  }
  const lines: string[] = [];
  let current = "";
  const maxLength = 14;
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxLength) {
      current = candidate;
      continue;
    }
    if (current) {
      lines.push(current);
    }
    if (word.length > maxLength) {
      lines.push(`${word.slice(0, maxLength - 1)}…`);
      current = "";
    } else {
      current = word;
    }
    if (lines.length === 2) {
      break;
    }
  }
  if (lines.length < 2 && current) {
    lines.push(current);
  }
  if (lines.length === 0) {
    lines.push(words[0]);
  }
  return lines.slice(0, 2);
};

const getNodeFill = (type: string | null | undefined): string => {
  if (!type) {
    return hashColor(null);
  }
  const key = type.trim().toLowerCase();
  return TYPE_COLOR_MAP[key] ?? hashColor(type);
};

const getNodeImportance = (node: GraphNode): number => {
  const provided = typeof node.importance === "number" ? node.importance : null;
  if (provided !== null && Number.isFinite(provided) && provided > 0) {
    return provided;
  }
  const sectionTotal = Object.values(node.section_distribution ?? {}).reduce(
    (accumulator, value) => accumulator + value,
    0,
  );
  const base = Math.max(node.times_seen, 0);
  const fallback = base + sectionTotal;
  return Math.max(fallback, 1);
};

const calculateNodeRadius = (node: GraphNode, degree: number, labelLines: string[]): number => {
  const importance = getNodeImportance(node);
  const importanceContribution = Math.log10(importance + 1) * 11;
  const degreeContribution = Math.sqrt(Math.max(degree, 1)) * 3.5;
  const longestLine = labelLines.reduce((acc, line) => Math.max(acc, line.length), 0);
  const approxCharWidth = NODE_LABEL_FONT_SIZE * 0.68;
  const horizontalRadius = longestLine > 0 ? (longestLine * approxCharWidth) / 2 : 0;
  const verticalRadius = (labelLines.length * NODE_LABEL_LINE_HEIGHT) / 2 + NODE_LABEL_VERTICAL_PADDING;
  const minimum = Math.max(MIN_NODE_RADIUS, horizontalRadius, verticalRadius);
  return (minimum + importanceContribution + degreeContribution) * NODE_RADIUS_SCALE;
};

const getRelationColor = (relation: string): string => {
  const key = relation.trim().toLowerCase();
  return RELATION_COLOR_MAP[key] ?? "rgba(15, 23, 42, 0.6)";
};

const estimateCollisionRadius = (node: GraphNode, degree: number): number => {
  const labelLines = formatNodeLabel(node.label);
  const radius = calculateNodeRadius(node, degree, labelLines);
  return radius + 6;
};

const getComponentColors = (componentId: number): { fill: string; stroke: string } => {
  const seeded = createSeededGenerator(`component-color-${componentId}`);
  const hue = Math.floor(seeded() * 360);
  const saturation = 58 + seeded() * 12;
  const lightness = 68 + seeded() * 10;
  const strokeLightness = Math.max(38, lightness - 22);
  const strokeSaturation = Math.max(38, saturation - 18);
  return {
    fill: `hsla(${hue}, ${saturation.toFixed(1)}%, ${lightness.toFixed(1)}%, 0.22)`,
    stroke: `hsla(${hue}, ${strokeSaturation.toFixed(1)}%, ${strokeLightness.toFixed(1)}%, 0.55)`,
  };
};

const resolveCollisions = (nodes: PositionedNode[]): void => {
  const padding = 16;
  const epsilon = 0.0001;
  const iterations = 8;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let moved = false;
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const nodeA = nodes[i];
        const nodeB = nodes[j];
        let dx = nodeA.x - nodeB.x;
        let dy = nodeA.y - nodeB.y;
        let distance = Math.sqrt(dx * dx + dy * dy);
        const minDistance =
          estimateCollisionRadius(nodeA.node, nodeA.degree) +
          estimateCollisionRadius(nodeB.node, nodeB.degree) +
          padding;
        if (distance >= minDistance) {
          continue;
        }
        let nx: number;
        let ny: number;
        if (distance < epsilon) {
          const angle = ((i + 1) * 0.318309886 + (j + 1) * 0.127323954) * Math.PI * 2;
          nx = Math.cos(angle);
          ny = Math.sin(angle);
          distance = epsilon;
        } else {
          nx = dx / distance;
          ny = dy / distance;
        }
        const overlap = minDistance - distance;
        const shift = overlap / 2;
        nodeA.x += nx * shift;
        nodeA.y += ny * shift;
        nodeB.x -= nx * shift;
        nodeB.y -= ny * shift;
        moved = true;
      }
    }
    if (!moved) {
      break;
    }
  }
};

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

  const { positionedNodes, positionedEdges, componentBackgrounds, viewBox, viewWidth, viewHeight } = useMemo(() => {
    const limitedNodes = nodes.slice(0, GRAPH_VISUALIZATION_NODE_LIMIT);
    const width = Math.max(dimensions.width, 320);
    const height = Math.max(dimensions.height, DEFAULT_HEIGHT);
    if (!limitedNodes.length) {
      return {
        positionedNodes: [] as StyledNode[],
        positionedEdges: [] as StyledEdge[],
        componentBackgrounds: [] as ComponentBackground[],
        viewBox: `0 0 ${width} ${height}`,
        viewWidth: width,
        viewHeight: height,
      };
    }

    const layout = runForceLayout(limitedNodes, edges, width, height);
    const margin = 30;
    const minX = layout.bounds.minX - margin;
    const maxX = layout.bounds.maxX + margin;
    const minY = layout.bounds.minY - margin;
    const maxY = layout.bounds.maxY + margin;

    const viewWidth = maxX - minX;
    const viewHeight = maxY - minY;

    const translatedNodes = layout.nodes.map((node) => ({
      ...node,
      x: node.x - minX,
      y: node.y - minY,
      anchorX: node.anchorX - minX,
      anchorY: node.anchorY - minY,
    }));

    const styledNodes: StyledNode[] = translatedNodes.map((entry) => {
      const labelLines = formatNodeLabel(entry.node.label);
      const fill = getNodeFill(entry.node.type ?? null);
      const { color: labelColor, outline: labelOutline } = getContrastingLabelColors(fill);
      return {
        ...entry,
        labelLines,
        radius: calculateNodeRadius(entry.node, entry.degree, labelLines),
        fill,
        labelColor,
        labelOutline,
        strokeColor: NODE_STROKE_COLOR,
        strokeWidth: 3,
      };
    });

    const styledNodeIndex = new Map(styledNodes.map((entry) => [entry.node.id, entry]));
    const styledEdges: StyledEdge[] = layout.edges.map((edge) => {
      const source = styledNodeIndex.get(edge.source.node.id);
      const target = styledNodeIndex.get(edge.target.node.id);
      if (!source || !target) {
        throw new Error("Graph layout attempted to render an edge without positioned nodes");
      }
      const stroke = getRelationColor(edge.relation);
      const opacity = 0.25 + Math.min(Math.max(edge.confidence, 0), 1) * 0.45;
      return {
        ...edge,
        source,
        target,
        stroke,
        markerKey: stroke,
        labelColor: stroke,
        strokeOpacity: opacity,
      };
    });

    const componentMap = new Map<number, StyledNode[]>();
    styledNodes.forEach((entry) => {
      const group = componentMap.get(entry.componentId);
      if (group) {
        group.push(entry);
        return;
      }
      componentMap.set(entry.componentId, [entry]);
    });

    const componentBackgrounds: ComponentBackground[] = Array.from(componentMap.entries()).map(
      ([componentId, componentNodes]) => {
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        const typeCounts = new Map<string, number>();
        componentNodes.forEach((node) => {
          minX = Math.min(minX, node.x - node.radius);
          maxX = Math.max(maxX, node.x + node.radius);
          minY = Math.min(minY, node.y - node.radius);
          maxY = Math.max(maxY, node.y + node.radius);
          const type = node.node.type?.toLowerCase();
          if (!type) {
            return;
          }
          typeCounts.set(type, (typeCounts.get(type) ?? 0) + 1);
        });
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const paddingX = Math.max(28, Math.sqrt(componentNodes.length) * 14);
        const paddingY = Math.max(24, Math.sqrt(componentNodes.length) * 12);
        const rx = Math.max((maxX - minX) / 2 + paddingX, 42);
        const ry = Math.max((maxY - minY) / 2 + paddingY, 42);
        let dominantType: string | null = null;
        let dominantCount = 0;
        typeCounts.forEach((count, type) => {
          if (count > dominantCount) {
            dominantType = type;
            dominantCount = count;
          }
        });
        const { fill, stroke } = getComponentColors(componentId);
        const labelBase = `${componentNodes.length} node${componentNodes.length === 1 ? "" : "s"}`;
        const label = dominantType ? `${labelBase} · ${dominantType}` : labelBase;
        return {
          id: componentId,
          cx,
          cy,
          rx,
          ry,
          fill,
          stroke,
          label,
        };
      },
    );

    return {
      positionedNodes: styledNodes,
      positionedEdges: styledEdges,
      componentBackgrounds,
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
        if (nextScale === current.scale) {
          return current;
        }
        const pointerGraphX = (pointer.x - current.x) / current.scale;
        const pointerGraphY = (pointer.y - current.y) / current.scale;
        return {
          scale: nextScale,
          x: pointer.x - pointerGraphX * nextScale,
          y: pointer.y - pointerGraphY * nextScale,
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
        if (nextScale === current.scale) {
          return current;
        }
        const centerGraphX = (center.x - current.x) / current.scale;
        const centerGraphY = (center.y - current.y) / current.scale;
        return {
          scale: nextScale,
          x: center.x - centerGraphX * nextScale,
          y: center.y - centerGraphY * nextScale,
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

  const edgeMarkerMap = useMemo(() => {
    const map = new Map<string, string>();
    positionedEdges.forEach((edge) => {
      if (!map.has(edge.markerKey)) {
        map.set(edge.markerKey, `graph-arrow-${map.size}`);
      }
    });
    return map;
  }, [positionedEdges]);

  const markerEntries = useMemo(() => Array.from(edgeMarkerMap.entries()), [edgeMarkerMap]);

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[420px] w-full overflow-hidden rounded-md border border-border bg-gradient-to-br from-background via-background/70 to-background"
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
          {markerEntries.map(([color, id]) => (
            <marker key={id} id={id} markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L8,4 L0,8" fill={color} fillOpacity="0.85" />
            </marker>
          ))}
          <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="1" stdDeviation="1.4" floodColor="rgba(0,0,0,0.25)" />
          </filter>
        </defs>
        <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
          <g>
            {componentBackgrounds.map((background) => (
              <g key={`component-${background.id}`}>
                <ellipse
                  cx={background.cx}
                  cy={background.cy}
                  rx={background.rx}
                  ry={background.ry}
                  fill={background.fill}
                  stroke={background.stroke}
                  strokeWidth={1.6}
                  strokeDasharray="12 10"
                />
                <text
                  x={background.cx}
                  y={background.cy - background.ry + 22}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight={600}
                  fill="rgba(15, 23, 42, 0.74)"
                  paintOrder="stroke"
                  stroke="rgba(248, 250, 252, 0.92)"
                  strokeWidth={2.1}
                >
                  {background.label}
                </text>
              </g>
            ))}
          </g>
          <g>
            {positionedEdges.map((edge) => {
              const midX = (edge.source.x + edge.target.x) / 2;
              const midY = (edge.source.y + edge.target.y) / 2;
              const rawAngle = (Math.atan2(edge.target.y - edge.source.y, edge.target.x - edge.source.x) * 180) / Math.PI;
              const flipped = rawAngle > 90 || rawAngle < -90;
              const angle = flipped ? rawAngle + 180 : rawAngle;
              const label = edge.relation;
              const labelWidth = Math.min(160, Math.max(56, label.length * 6));
              const markerId = edgeMarkerMap.get(edge.markerKey);
              return (
                <g key={edge.id}>
                  <line
                    x1={edge.source.x}
                    y1={edge.source.y}
                    x2={edge.target.x}
                    y2={edge.target.y}
                    markerEnd={markerId ? `url(#${markerId})` : undefined}
                    stroke={edge.stroke}
                    strokeWidth={1.2}
                    strokeOpacity={edge.strokeOpacity}
                  />
                  <g transform={`translate(${midX}, ${midY}) rotate(${angle})`}>
                    <rect
                      x={-labelWidth / 2}
                      y={-10}
                      width={labelWidth}
                      height={20}
                      rx={6}
                      fill="rgba(248, 250, 252, 0.9)"
                    />
                    <text
                      textAnchor="middle"
                      fontSize="9"
                      fill={edge.labelColor}
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
            {positionedNodes.map((entry) => {
              const importanceScore = getNodeImportance(entry.node);
              const confidenceWeightedEdges = entry.node.section_distribution
                ? Object.values(entry.node.section_distribution).reduce((acc, value) => acc + value, 0)
                : 0;
              const tooltipLines = [
                entry.node.label,
                entry.node.type ? `Type: ${entry.node.type}` : null,
                `Times seen: ${entry.node.times_seen}`,
                `Importance score: ${importanceScore.toFixed(2)}`,
                `Confidence-weighted edges: ${confidenceWeightedEdges}`,
              ]
                .filter((line): line is string => Boolean(line))
                .join("\n");
              return (
                <g key={entry.node.id} transform={`translate(${entry.x}, ${entry.y})`}>
                  <circle
                    r={entry.radius}
                    fill={entry.fill}
                    stroke={entry.strokeColor}
                    strokeWidth={entry.strokeWidth}
                    opacity={0.92}
                    filter="url(#node-shadow)"
                  >
                    <title>{tooltipLines}</title>
                  </circle>
                  {entry.labelLines.map((line, index) => {
                    const offset = (index - (entry.labelLines.length - 1) / 2) * NODE_LABEL_LINE_HEIGHT;
                    return (
                      <text
                        key={`${entry.node.id}-label-${index}`}
                        x={0}
                        y={offset + NODE_LABEL_FONT_SIZE / 3}
                        textAnchor="middle"
                        fontSize={NODE_LABEL_FONT_SIZE}
                        fontWeight={NODE_LABEL_FONT_WEIGHT}
                        fill={entry.labelColor}
                        paintOrder="stroke"
                        stroke={entry.labelOutline}
                        strokeWidth={1.2}
                        letterSpacing="0.25px"
                      >
                        {line}
                      </text>
                    );
                  })}
                </g>
              );
            })}
          </g>
        </g>
      </svg>
    </div>
  );
};

export default GraphVisualization;
