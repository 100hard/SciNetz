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
  strokeWidth: number;
  markerColor: string;
  markerOpacity: number;
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

type GraphVisualizationProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  showComponentBackgrounds?: boolean;
  isFullscreen?: boolean;
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
const LAYOUT_AREA_SCALE = 0.74;
const LAYOUT_ATTRACTION_STRENGTH = 0.044;
const LAYOUT_REPULSION_STRENGTH = 0.32;
const LAYOUT_CROSS_COMPONENT_PULL = 0.032;
const LAYOUT_ANCHOR_GRAVITY = 0.024;
const LAYOUT_CENTER_GRAVITY = 0.03;
const LAYOUT_TEMPERATURE_DIVISOR = 2.45;
const LAYOUT_COOLING_FACTOR = 0.89;
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

type CachedPosition = {
  x: number;
  y: number;
  anchorX: number;
  anchorY: number;
  componentId: number;
};

type LayoutResult = {
  nodes: PositionedNode[];
  edges: PositionedEdge[];
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  positions: Map<string, CachedPosition>;
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
  previousPositions: Map<string, CachedPosition>,
): LayoutResult => {
  if (!nodes.length) {
    return {
      nodes: [],
      edges: [],
      bounds: { minX: 0, maxX: width, minY: 0, maxY: height },
      positions: new Map(),
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
    const spread = anchor?.spread ?? minDimension * 0.46;
    const radius = spread * (0.18 + seeded() * 0.52);
    const jitterMagnitude = spread * 0.08;
    const jitterX = (seeded() - 0.5) * 2 * jitterMagnitude;
    const jitterY = (seeded() - 0.5) * 2 * jitterMagnitude;
    const noise = anchor?.noise ?? 0;
    const previous = previousPositions.get(node.id);
    const baseAnchorX = (anchor?.x ?? centerX) + noise;
    const baseAnchorY = (anchor?.y ?? centerY) - noise;
    const anchorX = previous ? previous.anchorX : baseAnchorX;
    const anchorY = previous ? previous.anchorY : baseAnchorY;
    const startX = previous ? previous.x : baseAnchorX + Math.cos(angle) * radius + jitterX;
    const startY = previous ? previous.y : baseAnchorY + Math.sin(angle) * radius + jitterY;
    return {
      node,
      componentId,
      anchorX,
      anchorY,
      x: startX,
      y: startY,
      dx: 0,
      dy: 0,
    };
  });

  const nodeIndex = new Map(simulationNodes.map((entry) => [entry.node.id, entry]));

  const iterations = Math.min(560, 220 + simulationNodes.length * 3);
  const area = width * height * LAYOUT_AREA_SCALE;
  const k = Math.sqrt(area / simulationNodes.length);
  let temperature = maxDimension / LAYOUT_TEMPERATURE_DIVISOR;
  const coolingFactor = LAYOUT_COOLING_FACTOR;
  const gravity = LAYOUT_ANCHOR_GRAVITY;
  const centerGravity = LAYOUT_CENTER_GRAVITY;
  const repulsionStrength = LAYOUT_REPULSION_STRENGTH;
  const attractionStrength = LAYOUT_ATTRACTION_STRENGTH;
  const crossComponentPull = LAYOUT_CROSS_COMPONENT_PULL;
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
      node.dx -= toAnchorX * (gravity * 0.55);
      node.dy -= toAnchorY * (gravity * 0.55);
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
      node.dx = 0;
      node.dy = 0;
    }

    temperature *= coolingFactor;
    if (temperature < 0.2) {
      break;
    }
  }

  const positionedNodes: PositionedNode[] = simulationNodes.map((node, index) => {
    const jitterGenerator = createSeededGenerator(`post-${node.node.id}-${index}`);
    const jitterScale = maxDimension * 0.004;
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

  const positionCache = new Map<string, CachedPosition>();
  for (const entry of positionedNodes) {
    positionCache.set(entry.node.id, {
      x: entry.x,
      y: entry.y,
      anchorX: entry.anchorX,
      anchorY: entry.anchorY,
      componentId: entry.componentId,
    });
  }

  return {
    nodes: positionedNodes,
    edges: positionedEdges,
    bounds: { minX, maxX, minY, maxY },
    positions: positionCache,
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

const getDocumentColors = (docId: string): { fill: string; stroke: string } => {
  const normalized = docId.trim();
  const seeded = createSeededGenerator(`paper-color-${normalized}`);
  const baseHue = Math.floor(seeded() * 360);
  const saturation = 62 + seeded() * 10;
  const lightness = 70 + seeded() * 8;
  const strokeLightness = Math.max(42, lightness - 18);
  const strokeSaturation = Math.max(40, saturation - 14);
  return {
    fill: `hsla(${baseHue}, ${saturation.toFixed(1)}%, ${lightness.toFixed(1)}%, 0.26)`,
    stroke: `hsla(${baseHue}, ${strokeSaturation.toFixed(1)}%, ${strokeLightness.toFixed(1)}%, 0.6)`,
  };
};

const getComponentColors = (componentId: number, dominantDocId: string | null): { fill: string; stroke: string } => {
  if (dominantDocId) {
    return getDocumentColors(dominantDocId);
  }
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
const EDGE_LABEL_GAP_MARGIN = 18;
const EDGE_LABEL_BASE_OFFSET = 6;
const EDGE_LABEL_SHORTAGE_MULTIPLIER = 0.55;
const EDGE_LABEL_MAX_OFFSET = 140;
const EDGE_LABEL_HORIZONTAL_STABILITY_THRESHOLD = 0.12;
const EDGE_DARKEN_BASE = "#0f172a";
const EDGE_LABEL_LIGHTEN_TARGET = "#f8fafc";

const rgbToCss = (color: RgbColor): string => `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;

const blendColors = (baseColor: string, mixColor: string, amount: number): string | null => {
  const base = toRgbColor(baseColor);
  const mix = toRgbColor(mixColor);
  if (!base || !mix) {
    return null;
  }
  const ratio = clamp(amount, 0, 1);
  const blended: RgbColor = {
    r: base.r * (1 - ratio) + mix.r * ratio,
    g: base.g * (1 - ratio) + mix.g * ratio,
    b: base.b * (1 - ratio) + mix.b * ratio,
  };
  return rgbToCss(blended);
};

const getDeterministicEdgeLabelSide = (edgeId: string): number => {
  if (!edgeId) {
    return 1;
  }
  let hash = 0;
  for (let index = 0; index < edgeId.length; index += 1) {
    hash = (hash * 31 + edgeId.charCodeAt(index)) >>> 0;
  }
  return (hash & 1) === 0 ? 1 : -1;
};

const getEdgeStrokeColor = (baseColor: string, confidence: number): string => {
  const normalized = clamp(confidence, 0, 1);
  const blendAmount = 0.25 + normalized * 0.45;
  return blendColors(baseColor, EDGE_DARKEN_BASE, blendAmount) ?? baseColor;
};

const getEdgeStrokeWidth = (confidence: number): number => {
  const normalized = clamp(confidence, 0, 1);
  const emphasis = Math.sqrt(normalized);
  return EDGE_MIN_STROKE_WIDTH + (EDGE_MAX_STROKE_WIDTH - EDGE_MIN_STROKE_WIDTH) * emphasis;
};

const getEdgeStrokeOpacity = (confidence: number): number => {
  const normalized = clamp(confidence, 0, 1);
  return EDGE_MIN_OPACITY + (EDGE_MAX_OPACITY - EDGE_MIN_OPACITY) * Math.pow(normalized, 0.65);
};

const getEdgeMarkerOpacity = (confidence: number): number => {
  const normalized = clamp(confidence, 0, 1);
  return EDGE_MIN_OPACITY + (EDGE_MAX_OPACITY - EDGE_MIN_OPACITY) * Math.pow(normalized, 0.5);
};

const getEdgeLabelColor = (strokeColor: string): string => {
  return blendColors(strokeColor, EDGE_LABEL_LIGHTEN_TARGET, 0.32) ?? strokeColor;
};

const DIMENSION_UPDATE_EPSILON = 0.5;
const LAYOUT_DIMENSION_RECOMPUTE_THRESHOLD = 120;

type LayoutCache = {
  nodeSignature: string;
  edgeSignature: string;
  width: number;
  height: number;
  positions: Map<string, CachedPosition>;
};

const createNodeSignature = (nodes: GraphNode[]): string =>
  nodes
    .map((node) => node.id)
    .filter(Boolean)
    .sort()
    .join("|");

const createEdgeSignature = (edges: GraphEdge[]): string =>
  edges
    .map((edge) => `${edge.source}->${edge.target}:${edge.id}`)
    .sort()
    .join("|");

const GraphVisualization = ({
  nodes,
  edges,
  showComponentBackgrounds = true,
  isFullscreen = false,
}: GraphVisualizationProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const previousPositionsRef = useRef<Map<string, CachedPosition>>(new Map());
  const layoutCacheRef = useRef<LayoutCache | null>(null);
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

    const applyDimensions = (width: number, height: number) => {
      const nextWidth = Math.max(width, 0);
      const nextHeight = Math.max(height, DEFAULT_HEIGHT);
      setDimensions((current) => {
        const widthChanged = Math.abs(current.width - nextWidth) > DIMENSION_UPDATE_EPSILON;
        const heightChanged = Math.abs(current.height - nextHeight) > DIMENSION_UPDATE_EPSILON;
        if (!widthChanged && !heightChanged) {
          return current;
        }
        return { width: nextWidth, height: nextHeight };
      });
    };

    applyDimensions(element.clientWidth, element.clientHeight);

    let frame: number | null = null;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const entry = entries[0];
      const { width, height } = entry.contentRect;
      if (frame !== null && typeof window !== "undefined") {
        window.cancelAnimationFrame(frame);
        frame = null;
      }
      const schedule = () => {
        applyDimensions(width, height);
        frame = null;
      };
      if (typeof window !== "undefined") {
        frame = window.requestAnimationFrame(schedule);
      } else {
        schedule();
      }
    });

    observer.observe(element);

    return () => {
      if (typeof window !== "undefined" && frame !== null) {
        window.cancelAnimationFrame(frame);
      }
      observer.disconnect();
    };
  }, []);

  const { positionedNodes, positionedEdges, componentBackgrounds, viewBox, viewWidth, viewHeight } = useMemo(() => {
    const limitedNodes = nodes.slice(0, GRAPH_VISUALIZATION_NODE_LIMIT);
    const width = Math.max(dimensions.width, 320);
    const height = Math.max(dimensions.height, DEFAULT_HEIGHT);

    if (!limitedNodes.length) {
      layoutCacheRef.current = null;
      previousPositionsRef.current = new Map();
      return {
        positionedNodes: [] as StyledNode[],
        positionedEdges: [] as StyledEdge[],
        componentBackgrounds: [] as ComponentBackground[],
        viewBox: `0 0 ${width} ${height}`,
        viewWidth: width,
        viewHeight: height,
      };
    }

    const allowedIds = new Set(limitedNodes.map((node) => node.id));
    const relevantEdges = edges.filter((edge) => allowedIds.has(edge.source) && allowedIds.has(edge.target));
    const nodeSignature = createNodeSignature(limitedNodes);
    const edgeSignature = createEdgeSignature(relevantEdges);

    const cache = layoutCacheRef.current;
    const widthDelta = cache ? Math.abs(cache.width - width) : Number.POSITIVE_INFINITY;
    const heightDelta = cache ? Math.abs(cache.height - height) : Number.POSITIVE_INFINITY;

    let positions: Map<string, CachedPosition>;
    let usingCache = false;

    if (
      cache &&
      cache.nodeSignature === nodeSignature &&
      cache.edgeSignature === edgeSignature &&
      widthDelta < LAYOUT_DIMENSION_RECOMPUTE_THRESHOLD &&
      heightDelta < LAYOUT_DIMENSION_RECOMPUTE_THRESHOLD
    ) {
      const missingNode = limitedNodes.some((node) => !cache.positions.has(node.id));
      if (!missingNode) {
        positions = cache.positions;
        usingCache = true;
      }
    }

    if (!usingCache) {
      const layout = runForceLayout(limitedNodes, edges, width, height, previousPositionsRef.current);
      positions = layout.positions;
      layoutCacheRef.current = {
        nodeSignature,
        edgeSignature,
        width,
        height,
        positions,
      };
    } else {
      const cachedWidth = cache?.width ?? width;
      const cachedHeight = cache?.height ?? height;
      layoutCacheRef.current = {
        nodeSignature,
        edgeSignature,
        width: cachedWidth,
        height: cachedHeight,
        positions,
      };
    }

    previousPositionsRef.current = positions;

    const adjacencyCounts = new Map<string, number>();
    relevantEdges.forEach((edge) => {
      adjacencyCounts.set(edge.source, (adjacencyCounts.get(edge.source) ?? 0) + 1);
      adjacencyCounts.set(edge.target, (adjacencyCounts.get(edge.target) ?? 0) + 1);
    });

    const positionedNodesRaw: PositionedNode[] = [];
    let missingPosition = false;
    for (const node of limitedNodes) {
      const cached = positions.get(node.id);
      if (!cached) {
        missingPosition = true;
        break;
      }
      positionedNodesRaw.push({
        node,
        x: cached.x,
        y: cached.y,
        degree: adjacencyCounts.get(node.id) ?? 0,
        componentId: cached.componentId,
        anchorX: cached.anchorX,
        anchorY: cached.anchorY,
      });
    }

    if (missingPosition) {
      const layout = runForceLayout(limitedNodes, edges, width, height, new Map());
      positions = layout.positions;
      layoutCacheRef.current = {
        nodeSignature,
        edgeSignature,
        width,
        height,
        positions,
      };
      previousPositionsRef.current = positions;
      positionedNodesRaw.length = 0;
      for (const node of limitedNodes) {
        const cached = positions.get(node.id);
        if (!cached) {
          continue;
        }
        positionedNodesRaw.push({
          node,
          x: cached.x,
          y: cached.y,
          degree: adjacencyCounts.get(node.id) ?? 0,
          componentId: cached.componentId,
          anchorX: cached.anchorX,
          anchorY: cached.anchorY,
        });
      }
    }

    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    positionedNodesRaw.forEach((entry) => {
      minX = Math.min(minX, entry.x);
      maxX = Math.max(maxX, entry.x);
      minY = Math.min(minY, entry.y);
      maxY = Math.max(maxY, entry.y);
    });

    if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
      minX = 0;
      maxX = width;
      minY = 0;
      maxY = height;
    }

    const margin = 30;
    const minXWithMargin = minX - margin;
    const maxXWithMargin = maxX + margin;
    const minYWithMargin = minY - margin;
    const maxYWithMargin = maxY + margin;

    const viewWidth = Math.max(maxXWithMargin - minXWithMargin, 1);
    const viewHeight = Math.max(maxYWithMargin - minYWithMargin, 1);

    const translatedNodes = positionedNodesRaw.map((entry) => ({
      ...entry,
      x: entry.x - minXWithMargin,
      y: entry.y - minYWithMargin,
      anchorX: entry.anchorX - minXWithMargin,
      anchorY: entry.anchorY - minYWithMargin,
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
    const styledEdges: StyledEdge[] = relevantEdges.map((edge) => {
      const source = styledNodeIndex.get(edge.source);
      const target = styledNodeIndex.get(edge.target);
      if (!source || !target) {
        throw new Error("Graph layout attempted to render an edge without positioned nodes");
      }
      const baseStroke = getRelationColor(edge.relation);
      const stroke = getEdgeStrokeColor(baseStroke, edge.confidence);
      const strokeWidth = getEdgeStrokeWidth(edge.confidence);
      const strokeOpacity = getEdgeStrokeOpacity(edge.confidence);
      const markerOpacity = getEdgeMarkerOpacity(edge.confidence);
      const labelColor = getEdgeLabelColor(stroke);
      const markerKey = `${stroke}-${markerOpacity.toFixed(2)}`;
      return {
        id: edge.id,
        source,
        target,
        relation: edge.relation,
        confidence: edge.confidence,
        stroke,
        markerKey,
        markerColor: stroke,
        markerOpacity,
        labelColor,
        strokeOpacity,
        strokeWidth,
      };
    });

    let componentBackgrounds: ComponentBackground[] = [];
    if (showComponentBackgrounds) {
      const componentMap = new Map<number, StyledNode[]>();
      styledNodes.forEach((entry) => {
        const group = componentMap.get(entry.componentId);
        if (group) {
          group.push(entry);
          return;
        }
        componentMap.set(entry.componentId, [entry]);
      });

      componentBackgrounds = Array.from(componentMap.entries()).map(([componentId, componentNodes]) => {
        let minComponentX = Number.POSITIVE_INFINITY;
        let maxComponentX = Number.NEGATIVE_INFINITY;
        let minComponentY = Number.POSITIVE_INFINITY;
        let maxComponentY = Number.NEGATIVE_INFINITY;
        const typeCounts = new Map<string, number>();
        const docCounts = new Map<string, number>();
        componentNodes.forEach((node) => {
          minComponentX = Math.min(minComponentX, node.x - node.radius);
          maxComponentX = Math.max(maxComponentX, node.x + node.radius);
          minComponentY = Math.min(minComponentY, node.y - node.radius);
          maxComponentY = Math.max(maxComponentY, node.y + node.radius);
          const type = node.node.type?.toLowerCase();
          if (type) {
            typeCounts.set(type, (typeCounts.get(type) ?? 0) + 1);
          }
          const docs = Array.isArray(node.node.source_document_ids) ? node.node.source_document_ids : [];
          docs.forEach((docId) => {
            const trimmed = docId.trim();
            if (trimmed) {
              docCounts.set(trimmed, (docCounts.get(trimmed) ?? 0) + 1);
            }
          });
        });
        const cx = (minComponentX + maxComponentX) / 2;
        const cy = (minComponentY + maxComponentY) / 2;
        const paddingX = Math.max(28, Math.sqrt(componentNodes.length) * 14);
        const paddingY = Math.max(24, Math.sqrt(componentNodes.length) * 12);
        const rx = Math.max((maxComponentX - minComponentX) / 2 + paddingX, 42);
        const ry = Math.max((maxComponentY - minComponentY) / 2 + paddingY, 42);
        let dominantType: string | null = null;
        let dominantCount = 0;
        typeCounts.forEach((count, type) => {
          if (count > dominantCount) {
            dominantType = type;
            dominantCount = count;
          }
        });
        let dominantDoc: string | null = null;
        let dominantDocCount = 0;
        docCounts.forEach((count, docId) => {
          if (
            dominantDoc === null ||
            count > dominantDocCount ||
            (count === dominantDocCount && docId < dominantDoc)
          ) {
            dominantDoc = docId;
            dominantDocCount = count;
          }
        });
        const { fill, stroke } = getComponentColors(componentId, dominantDoc);
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
      });
    }

    return {
      positionedNodes: styledNodes,
      positionedEdges: styledEdges,
      componentBackgrounds,
      viewBox: `0 0 ${viewWidth} ${viewHeight}`,
      viewWidth,
      viewHeight,
    };
  }, [dimensions.height, dimensions.width, edges, nodes, showComponentBackgrounds]);

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

  const markerDefinitions = useMemo(() => {
    const definitions: Array<{ key: string; id: string; color: string; opacity: number }> = [];
    const indexByKey = new Map<string, number>();
    positionedEdges.forEach((edge) => {
      if (indexByKey.has(edge.markerKey)) {
        return;
      }
      const id = `graph-arrow-${definitions.length}`;
      indexByKey.set(edge.markerKey, definitions.length);
      definitions.push({
        key: edge.markerKey,
        id,
        color: edge.markerColor,
        opacity: edge.markerOpacity,
      });
    });
    return definitions;
  }, [positionedEdges]);

  const markerIdByKey = useMemo(() => {
    const map = new Map<string, string>();
    markerDefinitions.forEach((entry) => {
      map.set(entry.key, entry.id);
    });
    return map;
  }, [markerDefinitions]);

  const containerBaseClass =
    "relative h-full w-full overflow-hidden rounded-md border border-border bg-gradient-to-br from-background via-background/70 to-background";
  const containerClassName = isFullscreen ? containerBaseClass : `${containerBaseClass} min-h-[260px]`;

  return (
    <div ref={containerRef} className={containerClassName}>
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
          {markerDefinitions.map((definition) => (
            <marker
              key={definition.key}
              id={definition.id}
              markerWidth="8"
              markerHeight="8"
              refX="8"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L8,4 L0,8" fill={definition.color} fillOpacity={definition.opacity} />
            </marker>
          ))}
          <filter id="edge-label-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="0.6" stdDeviation="0.8" floodColor="rgba(15, 23, 42, 0.22)" />
          </filter>
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
              const approxLabelWidth =
                label.length * EDGE_LABEL_FONT_SIZE * 0.62 + EDGE_LABEL_HORIZONTAL_PADDING * 2;
              const labelWidth = Math.min(
                EDGE_LABEL_MAX_WIDTH,
                Math.max(EDGE_LABEL_MIN_WIDTH, approxLabelWidth),
              );
              const dx = edge.target.x - edge.source.x;
              const dy = edge.target.y - edge.source.y;
              const distance = Math.sqrt(dx * dx + dy * dy);
              const safeDistance = distance > 0 ? distance : 0.0001;
              let normalX = -dy / safeDistance;
              let normalY = dx / safeDistance;
              const normalLength = Math.sqrt(normalX * normalX + normalY * normalY);
              if (normalLength > 0) {
                normalX /= normalLength;
                normalY /= normalLength;
              }
              let orientationMultiplier: number;
              if (normalY < -EDGE_LABEL_HORIZONTAL_STABILITY_THRESHOLD) {
                orientationMultiplier = 1;
              } else if (normalY > EDGE_LABEL_HORIZONTAL_STABILITY_THRESHOLD) {
                orientationMultiplier = -1;
              } else {
                orientationMultiplier = getDeterministicEdgeLabelSide(edge.id);
              }
              const alongSpacing = safeDistance - edge.source.radius - edge.target.radius;
              const shortage = labelWidth + EDGE_LABEL_GAP_MARGIN - alongSpacing;
              const shortageContribution = Math.max(0, shortage) * EDGE_LABEL_SHORTAGE_MULTIPLIER;
              const offsetMagnitude = clamp(
                EDGE_LABEL_BASE_OFFSET + shortageContribution,
                EDGE_LABEL_BASE_OFFSET,
                EDGE_LABEL_MAX_OFFSET,
              );
              const offsetX = normalX * offsetMagnitude * orientationMultiplier;
              const offsetY = normalY * offsetMagnitude * orientationMultiplier;
              const labelX = midX + offsetX;
              const labelY = midY + offsetY;
              const markerId = markerIdByKey.get(edge.markerKey);
              return (
                <g key={edge.id}>
                  <line
                    x1={edge.source.x}
                    y1={edge.source.y}
                    x2={edge.target.x}
                    y2={edge.target.y}
                    markerEnd={markerId ? `url(#${markerId})` : undefined}
                    stroke={edge.stroke}
                    strokeWidth={edge.strokeWidth}
                    strokeOpacity={edge.strokeOpacity}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <g transform={`translate(${labelX}, ${labelY}) rotate(${angle})`}>
                    <rect
                      x={-labelWidth / 2}
                      y={-EDGE_LABEL_RECT_HEIGHT / 2}
                      width={labelWidth}
                      height={EDGE_LABEL_RECT_HEIGHT}
                      rx={8}
                      fill="rgba(248, 250, 252, 0.98)"
                      stroke={edge.stroke}
                      strokeWidth={0.9}
                      strokeOpacity={Math.min(1, edge.strokeOpacity + 0.12)}
                      filter="url(#edge-label-shadow)"
                    />
                    <text
                      textAnchor="middle"
                      fontSize={EDGE_LABEL_FONT_SIZE}
                      fontWeight={EDGE_LABEL_FONT_WEIGHT}
                      fill={edge.labelColor}
                      transform={flipped ? "scale(-1, -1)" : undefined}
                      paintOrder="stroke"
                      stroke="rgba(15, 23, 42, 0.08)"
                      strokeWidth={0.6}
                      letterSpacing="0.3px"
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
