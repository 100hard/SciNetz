"use client";

import { useEffect, useMemo, useState } from "react";

import { GraphCanvas } from "../components/GraphCanvas";
import { EvidenceDrawer } from "../components/EvidenceDrawer";
import { FiltersPanel } from "../components/FiltersPanel";
import { PaperList } from "../components/PaperList";
import { QAPanel } from "../components/QAPanel";
import { UploadPanel } from "../components/UploadPanel";
import { extractPaper, fetchGraph, fetchSettings, listPapers, type GraphFilters } from "../lib/api";
import type { GraphEdge, GraphResponse, PaperSummary, QAPathEdge, UISettings } from "../lib/types";

export default function DashboardPage() {
  const [settings, setSettings] = useState<UISettings | null>(null);
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [filters, setFilters] = useState<GraphFilters | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [extracting, setExtracting] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  useEffect(() => {
    fetchSettings()
      .then((config) => {
        setSettings(config);
        const defaults: GraphFilters = {
          relations: config.graph_defaults.relations,
          min_confidence: config.graph_defaults.min_confidence,
          sections: config.graph_defaults.sections,
          include_co_mentions: config.graph_defaults.show_co_mentions,
          papers: []
        };
        setFilters(defaults);
      })
      .catch((err) => setGraphError(err instanceof Error ? err.message : String(err)));
    listPapers()
      .then(setPapers)
      .catch((err) => setGraphError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (!filters) {
      return;
    }
    setLoadingGraph(true);
    setGraphError(null);
    fetchGraph(filters)
      .then(setGraph)
      .catch((err) => setGraphError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoadingGraph(false));
  }, [filters]);

  const selectedEdge = useMemo<GraphEdge | null>(() => {
    if (!graph || !selectedEdgeId) {
      return null;
    }
    return graph.edges.find((edge) => edge.id === selectedEdgeId) ?? null;
  }, [graph, selectedEdgeId]);

  const relationOptions = useMemo(() => {
    if (!settings) {
      return [];
    }
    const canonical = settings.graph_defaults.relations;
    const derived = graph?.edges.map((edge) => edge.relation) ?? [];
    return Array.from(new Set([...canonical, ...derived])).sort();
  }, [settings, graph]);

  const sectionOptions = useMemo(() => {
    if (!graph) {
      return settings?.graph_defaults.sections ?? [];
    }
    const sections = new Set<string>();
    graph.nodes.forEach((node) => {
      Object.keys(node.section_distribution).forEach((section) => sections.add(section));
    });
    return Array.from(sections);
  }, [graph, settings]);

  const paperOptions = useMemo(() => papers.map((paper) => paper.paper_id), [papers]);

  const handleUploaded = (summary: PaperSummary) => {
    setPapers((current) => [summary, ...current.filter((item) => item.paper_id !== summary.paper_id)]);
  };

  const handleExtract = async (paperId: string) => {
    setExtracting(paperId);
    try {
      await extractPaper(paperId);
      const refreshed = await listPapers();
      setPapers(refreshed);
    } catch (err) {
      setGraphError(err instanceof Error ? err.message : String(err));
    } finally {
      setExtracting(null);
    }
  };

  const handleRefreshPapers = async () => {
    const refreshed = await listPapers();
    setPapers(refreshed);
  };

  const handleFiltersChange = (next: GraphFilters) => {
    setFilters(next);
  };

  const handleEdgeSelect = (edge: GraphEdge) => {
    setSelectedEdgeId(edge.id);
  };

  const handleHighlightEdge = (edge: QAPathEdge) => {
    if (!graph) {
      return;
    }
    const match = graph.edges.find(
      (candidate) =>
        candidate.source === edge.src_id &&
        candidate.target === edge.dst_id &&
        candidate.relation === edge.relation
    );
    if (match) {
      setSelectedEdgeId(match.id);
    }
  };

  return (
    <div className="grid gap-6">
      <section className="grid gap-4 md:grid-cols-2">
        <UploadPanel onUploaded={handleUploaded} />
        <QAPanel onHighlightEdge={handleHighlightEdge} />
      </section>
      <PaperList papers={papers} onExtract={handleExtract} extracting={extracting} onRefresh={handleRefreshPapers} />
      {filters && (
        <FiltersPanel
          filters={filters}
          relations={relationOptions}
          sections={sectionOptions}
          papers={paperOptions}
          onChange={handleFiltersChange}
        />
      )}
      <section className="grid gap-4 md:grid-cols-[2fr,1fr]">
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <span>
              {loadingGraph
                ? "Loading graph…"
                : graph
                  ? `${graph.node_count} ${graph.node_count === 1 ? "node" : "nodes"} · ${graph.edge_count} ${graph.edge_count === 1 ? "edge" : "edges"}`
                  : "Graph unavailable"}
            </span>
            {graphError && <span className="text-rose-400">{graphError}</span>}
          </div>
          {graph && filters ? (
            <GraphCanvas
              nodes={graph.nodes}
              edges={graph.edges}
              layout={settings?.graph_defaults.layout ?? "fcose"}
              onSelectEdge={handleEdgeSelect}
            />
          ) : (
            <div className="rounded-lg border border-dashed border-slate-800 bg-slate-900/50 p-6 text-sm text-slate-400">
              Configure filters to load the graph.
            </div>
          )}
        </div>
        <EvidenceDrawer edge={selectedEdge} onClose={() => setSelectedEdgeId(null)} />
      </section>
    </div>
  );
}
