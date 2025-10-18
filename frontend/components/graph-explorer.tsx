"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Eraser, Filter, Loader2, Maximize2, Minimize2, RefreshCw } from "lucide-react";

import GraphVisualization, { GRAPH_VISUALIZATION_NODE_LIMIT } from "./graph-visualization";
import apiClient, { extractErrorMessage } from "../lib/http";

type GraphDefaults = {
  relations: string[];
  min_confidence: number;
  sections: string[];
  show_co_mentions: boolean;
  layout: string;
};

type UiSettingsResponse = {
  graph_defaults: GraphDefaults;
  qa?: {
    llm_enabled: boolean;
    llm_provider?: string | null;
  };
};

export type GraphNode = {
  id: string;
  label: string;
  type?: string | null;
  aliases: string[];
  times_seen: number;
  importance?: number | null;
  section_distribution: Record<string, number>;
  source_document_ids: string[];
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  relation: string;
  relation_verbatim: string;
  confidence: number;
  times_seen: number;
  attributes: Record<string, string>;
  evidence: Record<string, unknown>;
  conflicting: boolean;
  created_at?: string | null;
};

type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
};

type PartitionedGraph = {
  id: string | null;
  label: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

const formatTimestamp = (value?: string | null) => {
  if (!value) {
    return "—";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
};

const formatSectionDistribution = (distribution: Record<string, number>) => {
  const entries = Object.entries(distribution);
  if (entries.length === 0) {
    return "—";
  }
  return entries
    .map(([key, value]) => `${key}: ${value}`)
    .sort()
    .join(" · ");
};

const stringify = (value: Record<string, unknown>) => {
  const entries = Object.entries(value);
  if (!entries.length) {
    return "—";
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch (error) {
    return entries
      .map(([key, val]) => `${key}: ${String(val)}`)
      .join("\n");
  }
};

const AUTO_FETCH_STORAGE_KEY = "graphAutoFetchEnabled";

const GraphExplorer = () => {
  const [defaults, setDefaults] = useState<GraphDefaults | null>(null);
  const [selectedRelations, setSelectedRelations] = useState<string[]>([]);
  const [selectedSections, setSelectedSections] = useState<string[]>([]);
  const [includeCoMentions, setIncludeCoMentions] = useState(true);
  const [minConfidence, setMinConfidence] = useState(0.0);
  const [limit, setLimit] = useState(500);
  const [paperFilter, setPaperFilter] = useState("");
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetchingSettings, setIsFetchingSettings] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoFetchEnabled, setAutoFetchEnabled] = useState<boolean | null>(null);
  const [isClearing, setIsClearing] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const skipNextAutoFetchRef = useRef(false);
  const visualizationContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.sessionStorage.getItem(AUTO_FETCH_STORAGE_KEY);
    if (stored === "true") {
      setAutoFetchEnabled(true);
      return;
    }
    if (stored === "false") {
      setAutoFetchEnabled(false);
      return;
    }
    setAutoFetchEnabled(true);
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const handleChange = () => {
      const target = visualizationContainerRef.current;
      setIsFullscreen(document.fullscreenElement === target);
    };
    document.addEventListener("fullscreenchange", handleChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleChange);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || autoFetchEnabled === null) {
      return;
    }
    window.sessionStorage.setItem(
      AUTO_FETCH_STORAGE_KEY,
      autoFetchEnabled ? "true" : "false",
    );
  }, [autoFetchEnabled]);

  useEffect(() => {
    const loadSettings = async () => {
      setIsFetchingSettings(true);
      try {
        const { data } = await apiClient.get<UiSettingsResponse>("/api/ui/settings");
        setDefaults(data.graph_defaults);
        setSelectedRelations(data.graph_defaults.relations);
        setSelectedSections(data.graph_defaults.sections);
        setIncludeCoMentions(data.graph_defaults.show_co_mentions);
        setMinConfidence(data.graph_defaults.min_confidence);
        setLimit(500);
      } catch (err) {
        const message = extractErrorMessage(err, "Unable to load UI settings.");
        setError(message);
      } finally {
        setIsFetchingSettings(false);
      }
    };

    void loadSettings();
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (typeof document === "undefined") {
      return;
    }
    const target = visualizationContainerRef.current;
    if (!target) {
      return;
    }
    if (document.fullscreenElement === target) {
      void document.exitFullscreen().catch(() => {});
      return;
    }
    if (!document.fullscreenElement) {
      void target.requestFullscreen().catch(() => {});
      return;
    }
    void document
      .exitFullscreen()
      .then(() => {
        void target.requestFullscreen().catch(() => {});
      })
      .catch(() => {});
  }, []);

  const sectionClassName = isFullscreen
    ? "fixed inset-0 z-50 flex flex-col overflow-hidden bg-card"
    : "rounded-lg border bg-card p-6 shadow-sm";
  const contentClassName = isFullscreen ? "flex h-full flex-col gap-4" : "space-y-4";
  const headerClassName = `flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between ${isFullscreen ? "px-6 pt-6" : ""}`;
  const controlsClassName = "flex flex-col items-start gap-2 sm:items-end";
  const graphContainerClass = isFullscreen ? "flex-1 px-6 pb-6 min-h-0" : undefined;

  const fetchGraph = useCallback(async () => {
    setIsLoading(true);
    try {
      const params: Record<string, unknown> = {
        relations: selectedRelations.join(","),
        sections: selectedSections.join(","),
        min_confidence: minConfidence,
        include_co_mentions: includeCoMentions,
        limit,
      };
      const trimmedPapers = paperFilter.trim();
      if (trimmedPapers) {
        params.papers = trimmedPapers;
      }
      const { data } = await apiClient.get<GraphResponse>("/api/ui/graph", { params });
      setGraph(data);
      setError(null);
    } catch (err) {
      setGraph(null);
      const message = extractErrorMessage(err, "Unable to fetch graph data.");
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [includeCoMentions, limit, minConfidence, paperFilter, selectedRelations, selectedSections]);

  useEffect(() => {
    if (!defaults) {
      return;
    }
    if (autoFetchEnabled !== true) {
      return;
    }

    if (skipNextAutoFetchRef.current) {
      skipNextAutoFetchRef.current = false;
      return;
    }

    void fetchGraph();
  }, [autoFetchEnabled, defaults, fetchGraph]);

  const relationOptions = defaults?.relations ?? [];
  const sectionOptions = defaults?.sections ?? [];

  const summary = useMemo(() => {
    if (!graph) {
      return null;
    }
    return {
      nodes: graph.node_count,
      edges: graph.edge_count,
    };
  }, [graph]);

  const partitionedGraphs = useMemo<PartitionedGraph[]>(() => {
    if (!graph) {
      return [];
    }

    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const nodeDocs = new Map<string, Set<string>>();
    const docEntries = new Map<
      string,
      { nodes: GraphNode[]; nodeIds: Set<string>; edges: GraphEdge[] }
    >();
    const doclessNodeIds = new Set<string>();
    const doclessEdges: GraphEdge[] = [];

    for (const node of graph.nodes) {
      const docs = Array.isArray(node.source_document_ids)
        ? node.source_document_ids
            .map((docId) => docId.trim())
            .filter((docId) => docId.length > 0)
        : [];
      if (docs.length === 0) {
        doclessNodeIds.add(node.id);
        continue;
      }
      const uniqueDocs = Array.from(new Set(docs));
      const docSet = new Set(uniqueDocs);
      nodeDocs.set(node.id, docSet);
      for (const docId of docSet) {
        let entry = docEntries.get(docId);
        if (!entry) {
          entry = { nodes: [], nodeIds: new Set(), edges: [] };
          docEntries.set(docId, entry);
        }
        if (!entry.nodeIds.has(node.id)) {
          entry.nodeIds.add(node.id);
          entry.nodes.push(node);
        }
      }
    }

    for (const edge of graph.edges) {
      const sourceDocs = nodeDocs.get(edge.source) ?? new Set<string>();
      const targetDocs = nodeDocs.get(edge.target) ?? new Set<string>();
      const sharedDocs: string[] = [];
      sourceDocs.forEach((docId) => {
        if (targetDocs.has(docId)) {
          sharedDocs.push(docId);
        }
      });
      if (sharedDocs.length > 0) {
        for (const docId of sharedDocs) {
          const entry = docEntries.get(docId);
          if (!entry) {
            continue;
          }
          entry.edges.push(edge);
        }
        continue;
      }
      doclessEdges.push(edge);
      doclessNodeIds.add(edge.source);
      doclessNodeIds.add(edge.target);
    }

    const results: PartitionedGraph[] = Array.from(docEntries.entries())
      .map(([docId, entry]) => ({
        id: docId,
        label: docId,
        nodes: entry.nodes,
        edges: entry.edges,
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    if (doclessNodeIds.size > 0) {
      const nodes = Array.from(doclessNodeIds)
        .map((nodeId) => nodeById.get(nodeId))
        .filter((node): node is GraphNode => Boolean(node));
      if (nodes.length > 0) {
        const nodeIdSet = new Set(nodes.map((node) => node.id));
        const edges = doclessEdges.filter(
          (edge) => nodeIdSet.has(edge.source) && nodeIdSet.has(edge.target),
        );
        results.push({
          id: null,
          label: "Unattributed entities",
          nodes,
          edges,
        });
      }
    }

    return results;
  }, [graph]);

  const toggleSelection = (value: string, current: string[], setFn: (next: string[]) => void) => {
    if (current.includes(value)) {
      setFn(current.filter((item) => item !== value));
    } else {
      setFn([...current, value]);
    }
  };

  const handleClearGraph = useCallback(async () => {
    if (isClearing) {
      return;
    }
    setIsClearing(true);
    setError(null);
    try {
      await apiClient.post("/api/ui/graph/clear");
      setGraph(null);
      setAutoFetchEnabled(false);
    } catch (err) {
      const message = extractErrorMessage(err, "Unable to clear graph.");
      setError(message);
    } finally {
      setIsClearing(false);
    }
  }, [isClearing]);

  const handleRefreshGraph = useCallback(() => {
    if (autoFetchEnabled !== true) {
      skipNextAutoFetchRef.current = true;
      setAutoFetchEnabled(true);
    }

    void fetchGraph();
  }, [autoFetchEnabled, fetchGraph]);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border bg-card p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <Filter className="h-5 w-5 text-muted-foreground" />
          <div>
            <h2 className="text-lg font-semibold text-foreground">Filters</h2>
            <p className="text-sm text-muted-foreground">Configure the graph query before fetching data.</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={handleClearGraph}
              disabled={isClearing || isFetchingSettings}
              className={`inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm font-medium text-foreground shadow-sm transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${isClearing ? "opacity-80" : "hover:bg-muted/40"} disabled:cursor-not-allowed disabled:opacity-60`}
            >
              {isClearing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eraser className="h-4 w-4" />} Clear graph
            </button>
            <button
              type="button"
              onClick={handleRefreshGraph}
              disabled={isLoading || isFetchingSettings}
              className={`inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground shadow-sm transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${isLoading ? "opacity-80" : "hover:bg-primary/90"} disabled:cursor-not-allowed disabled:opacity-60`}
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />} Refresh graph
            </button>
          </div>
        </div>

        <div className="mt-6 grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">Relations</p>
            <div className="flex flex-wrap gap-2">
              {relationOptions.map((relation) => {
                const checked = selectedRelations.includes(relation);
                return (
                  <button
                    key={relation}
                    type="button"
                    onClick={() => toggleSelection(relation, selectedRelations, setSelectedRelations)}
                    className={`rounded-full border px-3 py-1 text-xs font-medium capitalize transition ${checked ? "border-primary bg-primary/10 text-primary" : "border-border bg-background text-muted-foreground hover:bg-muted/40"}`}
                  >
                    {relation.replace(/_/g, " ")}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">Sections</p>
            <div className="flex flex-wrap gap-2">
              {sectionOptions.map((section) => {
                const checked = selectedSections.includes(section);
                return (
                  <button
                    key={section}
                    type="button"
                    onClick={() => toggleSelection(section, selectedSections, setSelectedSections)}
                    className={`rounded-full border px-3 py-1 text-xs font-medium transition ${checked ? "border-primary bg-primary/10 text-primary" : "border-border bg-background text-muted-foreground hover:bg-muted/40"}`}
                  >
                    {section}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-2">
            <label className="flex flex-col gap-2 text-sm">
              <span className="font-medium text-foreground">Minimum confidence</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={minConfidence}
                onChange={(event) => setMinConfidence(Number(event.target.value))}
                className="w-32 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
            </label>
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <input
                type="checkbox"
                checked={includeCoMentions}
                onChange={(event) => setIncludeCoMentions(event.target.checked)}
                className="h-4 w-4 rounded border border-input text-primary focus:ring-primary"
              />
              Include co-mention edges
            </label>
            <label className="flex flex-col gap-2 text-sm">
              <span className="font-medium text-foreground">Result limit</span>
              <input
                type="number"
                min={50}
                max={2000}
                step={50}
                value={limit}
                onChange={(event) => setLimit(Number(event.target.value))}
                className="w-32 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
            </label>
            <label className="flex flex-col gap-2 text-sm">
              <span className="font-medium text-foreground">Paper filter</span>
              <input
                type="text"
                value={paperFilter}
                onChange={(event) => setPaperFilter(event.target.value)}
                placeholder="Comma-separated IDs"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
            </label>
          </div>
        </div>
      </section>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <div>
            <p className="font-medium">Unable to load graph data</p>
            <p className="text-xs text-destructive/80">{error}</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Fetching graph view…
        </div>
      )}

      {summary && (
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-lg border bg-card p-5 shadow-sm">
            <p className="text-sm font-medium text-muted-foreground">Nodes</p>
            <p className="mt-2 text-2xl font-semibold text-foreground">{summary.nodes}</p>
          </div>
          <div className="rounded-lg border bg-card p-5 shadow-sm">
            <p className="text-sm font-medium text-muted-foreground">Edges</p>
            <p className="mt-2 text-2xl font-semibold text-foreground">{summary.edges}</p>
          </div>
        </section>
      )}

      {graph && (
        <Fragment>
          <section ref={visualizationContainerRef} className={sectionClassName}>
            <div className={contentClassName}>
              <div className={headerClassName}>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Graph preview</h3>
                  <p className="text-sm text-muted-foreground">
                    Lightweight layout for visually exploring the current filters.
                  </p>
                </div>
                <div className={controlsClassName}>
                  {graph.nodes.length > GRAPH_VISUALIZATION_NODE_LIMIT && (
                    <p className="text-xs text-muted-foreground">
                      Showing first {GRAPH_VISUALIZATION_NODE_LIMIT} nodes out of {graph.nodes.length}.
                    </p>
                  )}
                  <button
                    type="button"
                    onClick={toggleFullscreen}
                    className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm font-medium text-foreground transition hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2"
                  >
                    {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    {isFullscreen ? "Exit full screen" : "Full screen"}
                  </button>
                </div>
              </div>
              {graph.nodes.length === 0 ? (
                <p className={`text-sm text-muted-foreground ${isFullscreen ? "px-6" : ""}`}>
                  No nodes found for the selected filters.
                </p>
              ) : (
                <div className={graphContainerClass}>
                  <GraphVisualization
                    nodes={graph.nodes}
                    edges={graph.edges}
                    showComponentBackgrounds={false}
                  />
                </div>
              )}
            </div>
          </section>

          <div className="grid gap-6 xl:grid-cols-2">
            <section className="space-y-3 rounded-lg border bg-card p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-foreground">Nodes</h3>
              {graph.nodes.length === 0 ? (
                <p className="text-sm text-muted-foreground">No nodes found for the selected filters.</p>
              ) : (
              <div className="overflow-auto">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium">Label</th>
                      <th className="px-4 py-3 text-left font-medium">Type</th>
                      <th className="px-4 py-3 text-left font-medium">Aliases</th>
                      <th className="px-4 py-3 text-left font-medium">Times seen</th>
                      <th className="px-4 py-3 text-left font-medium">Sections</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-background">
                    {graph.nodes.map((node) => (
                      <tr key={node.id} className="hover:bg-muted/40">
                        <td className="px-4 py-3 text-sm text-foreground">
                          <p className="font-medium">{node.label}</p>
                          <p className="text-xs text-muted-foreground">{node.id}</p>
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">{node.type ?? "—"}</td>
                        <td className="px-4 py-3 text-xs text-muted-foreground">{node.aliases.join(", ") || "—"}</td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">{node.times_seen}</td>
                        <td className="px-4 py-3 text-xs text-muted-foreground">{formatSectionDistribution(node.section_distribution)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="space-y-3 rounded-lg border bg-card p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-foreground">Edges</h3>
            {graph.edges.length === 0 ? (
              <p className="text-sm text-muted-foreground">No edges found for the selected filters.</p>
            ) : (
              <div className="overflow-auto">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium">Relation</th>
                      <th className="px-4 py-3 text-left font-medium">Endpoints</th>
                      <th className="px-4 py-3 text-left font-medium">Confidence</th>
                      <th className="px-4 py-3 text-left font-medium">Evidence</th>
                      <th className="px-4 py-3 text-left font-medium">Attributes</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-background">
                    {graph.edges.map((edge) => (
                      <tr key={edge.id} className="align-top hover:bg-muted/40">
                        <td className="px-4 py-3 text-sm text-foreground">
                          <p className="font-medium">{edge.relation_verbatim || edge.relation}</p>
                          <p className="text-xs text-muted-foreground">{formatTimestamp(edge.created_at)}</p>
                          {edge.conflicting ? (
                            <p className="mt-1 text-[11px] font-semibold uppercase tracking-wide text-rose-600">Conflicting</p>
                          ) : null}
                        </td>
                        <td className="px-4 py-3 text-xs text-muted-foreground">
                          <p>{edge.source}</p>
                          <p>{edge.target}</p>
                          <p className="mt-1">Times seen: {edge.times_seen}</p>
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">{(edge.confidence * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-xs text-muted-foreground whitespace-pre-wrap">{stringify(edge.evidence)}</td>
                        <td className="px-4 py-3 text-xs text-muted-foreground whitespace-pre-wrap">{stringify(edge.attributes)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            </section>
          </div>
        </Fragment>
      )}
    </div>
  );
};

export default GraphExplorer;
