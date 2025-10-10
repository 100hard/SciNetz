"use client";

import clsx from "clsx";

import type { PaperSummary } from "../lib/types";

interface PaperListProps {
  papers: PaperSummary[];
  onExtract: (paperId: string) => void;
  extracting?: string | null;
  onRefresh?: () => void;
}

export function PaperList({ papers, onExtract, extracting, onRefresh }: PaperListProps) {
  if (!papers.length) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-sm text-slate-400">
        Upload a PDF to see its processing status.
      </div>
    );
  }
  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
      <table className="min-w-full divide-y divide-slate-800 text-sm">
        <thead className="bg-slate-900/80">
          <tr>
            <th className="px-4 py-3 text-left font-semibold text-slate-300">Paper</th>
            <th className="px-4 py-3 text-left font-semibold text-slate-300">Status</th>
            <th className="px-4 py-3 text-left font-semibold text-slate-300">Metadata</th>
            <th className="px-4 py-3 text-left font-semibold text-slate-300">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {papers.map((paper) => (
            <tr key={paper.paper_id} className="hover:bg-slate-800/40">
              <td className="px-4 py-3">
                <div className="flex flex-col">
                  <span className="font-medium text-slate-100">{paper.paper_id}</span>
                  <span className="text-xs text-slate-400">{paper.filename}</span>
                </div>
              </td>
              <td className="px-4 py-3">
                <StatusBadge status={paper.status} />
                {paper.errors.length > 0 && (
                  <p className="mt-1 max-w-xs text-xs text-rose-400">{paper.errors[0]}</p>
                )}
              </td>
              <td className="px-4 py-3">
                {paper.metadata?.title ? (
                  <div className="flex flex-col">
                    <span className="text-slate-100">{paper.metadata.title as string}</span>
                    {paper.metadata.year && (
                      <span className="text-xs text-slate-400">{paper.metadata.year as number}</span>
                    )}
                  </div>
                ) : (
                  <span className="text-xs text-slate-500">Pending metadata</span>
                )}
              </td>
              <td className="px-4 py-3">
                <button
                  type="button"
                  onClick={() => onExtract(paper.paper_id)}
                  disabled={paper.status === "processing" || extracting === paper.paper_id}
                  className="rounded bg-sky-500 px-3 py-1 text-xs font-semibold text-slate-950 disabled:cursor-not-allowed disabled:bg-sky-800"
                >
                  {extracting === paper.paper_id ? "Extracting…" : "Run extraction"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex justify-end border-t border-slate-800 bg-slate-900/70 px-4 py-3 text-xs text-slate-500">
        <button type="button" onClick={onRefresh} className="underline-offset-2 hover:underline">
          Refresh list
        </button>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const intent =
    status === "complete"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
      : status === "failed"
        ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
        : "bg-slate-700/40 text-slate-200 border-slate-600/50";
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2 py-1 text-xs font-semibold uppercase tracking-wide",
        intent
      )}
    >
      {status}
    </span>
  );
}
