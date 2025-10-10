"use client";

import type { GraphEdge } from "../lib/types";

interface EvidenceDrawerProps {
  edge: GraphEdge | null;
  onClose: () => void;
}

export function EvidenceDrawer({ edge, onClose }: EvidenceDrawerProps) {
  if (!edge) {
    return null;
  }
  const evidence = edge.evidence ?? {};
  const sentence = typeof evidence.full_sentence === "string" ? evidence.full_sentence : undefined;
  return (
    <aside className="rounded-lg border border-slate-800 bg-slate-900 p-4 shadow-lg">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-sky-300">Evidence</h3>
        <button
          type="button"
          onClick={onClose}
          className="text-xs text-slate-400 hover:text-slate-200"
        >
          Close
        </button>
      </div>
      <dl className="mt-3 space-y-2 text-sm">
        <div>
          <dt className="text-slate-400">Relation</dt>
          <dd className="font-semibold text-slate-100">{edge.relation_verbatim}</dd>
        </div>
        <div>
          <dt className="text-slate-400">Confidence</dt>
          <dd className="font-semibold text-slate-100">{edge.confidence.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="text-slate-400">Method</dt>
          <dd className="font-semibold text-slate-100">{edge.attributes.method ?? "llm"}</dd>
        </div>
        <div>
          <dt className="text-slate-400">Evidence sentence</dt>
          <dd className="rounded bg-slate-800 p-3 text-slate-100">
            {sentence ?? "Sentence unavailable"}
          </dd>
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs text-slate-400">
          <div>
            <span className="block font-medium text-slate-200">Document</span>
            <span>{String(evidence.doc_id ?? "unknown")}</span>
          </div>
          <div>
            <span className="block font-medium text-slate-200">Element</span>
            <span>{String(evidence.element_id ?? "unknown")}</span>
          </div>
        </div>
      </dl>
    </aside>
  );
}
