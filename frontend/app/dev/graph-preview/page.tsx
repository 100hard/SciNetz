import { notFound } from "next/navigation";

import GraphVisualization from "../../../components/graph-visualization";
import type { GraphEdge, GraphNode } from "../../../components/graph-explorer";

const sampleNodes: GraphNode[] = [
  {
    id: "n1",
    label: "Protein diffusion transformers",
    type: "method",
    aliases: ["PDT"],
    times_seen: 12,
    section_distribution: {
      Abstract: 3,
      Results: 4,
      Discussion: 2,
    },
  },
  {
    id: "n2",
    label: "Ligand docking dataset",
    type: "dataset",
    aliases: ["LDD"],
    times_seen: 8,
    section_distribution: {
      Methods: 5,
      Appendix: 2,
    },
  },
  {
    id: "n3",
    label: "Binding affinity metric",
    type: "metric",
    aliases: ["BAM"],
    times_seen: 6,
    section_distribution: {
      Results: 3,
      Evaluation: 2,
    },
  },
  {
    id: "n4",
    label: "Baseline CNN",
    type: "method",
    aliases: ["Baseline"],
    times_seen: 5,
    section_distribution: {
      Methods: 2,
      Results: 1,
    },
  },
  {
    id: "n5",
    label: "Enzyme classification task",
    type: "task",
    aliases: ["Enzyme task"],
    times_seen: 7,
    section_distribution: {
      Introduction: 2,
      Results: 3,
    },
  },
  {
    id: "n6",
    label: "CryoEM benchmark",
    type: "dataset",
    aliases: ["CryoEM"],
    times_seen: 3,
    section_distribution: {
      Methods: 1,
      Results: 1,
    },
  },
];

const sampleEdges: GraphEdge[] = [
  {
    id: "e1",
    source: "n1",
    target: "n2",
    relation: "uses",
    relation_verbatim: "uses",
    confidence: 0.92,
    times_seen: 5,
    attributes: {},
    evidence: { sentences: ["The PDT uses the ligand docking dataset for training."] },
    conflicting: false,
    created_at: null,
  },
  {
    id: "e2",
    source: "n1",
    target: "n3",
    relation: "evaluates",
    relation_verbatim: "evaluates",
    confidence: 0.88,
    times_seen: 4,
    attributes: {},
    evidence: { sentences: ["We evaluate PDT with the binding affinity metric."] },
    conflicting: false,
    created_at: null,
  },
  {
    id: "e3",
    source: "n4",
    target: "n3",
    relation: "evaluates",
    relation_verbatim: "evaluates",
    confidence: 0.81,
    times_seen: 3,
    attributes: {},
    evidence: { sentences: ["Baseline CNN evaluates the binding affinity metric as well."] },
    conflicting: false,
    created_at: null,
  },
  {
    id: "e4",
    source: "n1",
    target: "n5",
    relation: "improves",
    relation_verbatim: "improves",
    confidence: 0.9,
    times_seen: 6,
    attributes: {},
    evidence: { sentences: ["PDT improves enzyme classification accuracy."] },
    conflicting: false,
    created_at: null,
  },
  {
    id: "e5",
    source: "n6",
    target: "n5",
    relation: "reports",
    relation_verbatim: "reports",
    confidence: 0.76,
    times_seen: 2,
    attributes: {},
    evidence: { sentences: ["CryoEM benchmark reports enzyme task metrics."] },
    conflicting: false,
    created_at: null,
  },
];

const GraphPreviewPage = () => {
  if (process.env.NODE_ENV === "production") {
    notFound();
  }

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-8">
      <header className="space-y-2">
        <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Development preview
        </p>
        <h1 className="text-2xl font-bold text-foreground">Graph visualization sample</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          This page renders the graph visualization component with deterministic sample data so UI changes can be inspected
          without connecting to Neo4j.
        </p>
      </header>
      <section className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <GraphVisualization nodes={sampleNodes} edges={sampleEdges} />
      </section>
    </main>
  );
};

export default GraphPreviewPage;
