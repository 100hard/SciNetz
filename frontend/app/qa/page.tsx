"use client";

import dynamic from "next/dynamic";
import { MessageSquare } from "lucide-react";

const QaPanel = dynamic(() => import("../../components/qa-panel"), { ssr: false });

const QaPage = () => {
  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Graph QA</p>
        <h1 className="flex items-center gap-2 text-2xl font-semibold text-foreground">
          <MessageSquare className="h-6 w-6 text-primary" /> Ask the knowledge graph
        </h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Pose natural language questions and review evidence-backed answers assembled from the canonical knowledge graph.
        </p>
      </header>

      <QaPanel />
    </div>
  );
};

export default QaPage;

