"use client";

import { useState } from "react";

import { askQuestion } from "../lib/api";
import type { QAPathEdge, QAResponsePayload } from "../lib/types";

interface QAPanelProps {
  onHighlightEdge?: (edge: QAPathEdge) => void;
}

export function QAPanel({ onHighlightEdge }: QAPanelProps) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QAResponsePayload | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!question.trim()) {
      setError("Ask a question about the graph");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await askQuestion(question);
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to answer";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-3 rounded-lg border border-slate-800 bg-slate-900 p-4">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <label className="text-sm font-semibold text-slate-200" htmlFor="qa-input">
          Ask the graph
        </label>
        <textarea
          id="qa-input"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={3}
          placeholder="What model outperforms Alpha on the CIFAR dataset?"
          className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
        />
        {error && <p className="text-xs text-rose-400">{error}</p>}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="rounded bg-violet-500 px-4 py-2 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:bg-violet-800"
          >
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </form>
      {result && (
        <div className="space-y-4 text-sm">
          <div>
            <h3 className="font-semibold text-slate-200">Summary</h3>
            <p className="mt-1 text-slate-300">{result.summary}</p>
            <span className="mt-1 inline-flex rounded-full border border-slate-700 px-2 py-0.5 text-xs uppercase tracking-wide text-slate-400">
              {result.mode}
            </span>
          </div>
          {result.paths.length > 0 && (
            <div>
              <h4 className="font-semibold text-slate-200">Supporting paths</h4>
              <ul className="mt-2 space-y-2">
                {result.paths.map((path, idx) => (
                  <li key={idx} className="rounded border border-slate-800 bg-slate-950 p-3">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span>Score: {path.score.toFixed(2)}</span>
                      <span>Confidence: {path.confidence_product.toFixed(2)}</span>
                    </div>
                    <ol className="mt-2 space-y-1 text-sm">
                      {path.edges.map((edge, edgeIdx) => (
                        <li key={edgeIdx} className="flex items-start justify-between gap-3">
                          <span>
                            <strong className="text-slate-100">{edge.src_name}</strong>
                            <span className="text-slate-400"> {edge.relation}</span>
                            <strong className="text-slate-100"> {edge.dst_name}</strong>
                          </span>
                          <button
                            type="button"
                            onClick={() => onHighlightEdge?.(edge)}
                            className="text-xs text-sky-300 hover:text-sky-100"
                          >
                            Highlight
                          </button>
                        </li>
                      ))}
                    </ol>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.fallback_edges.length > 0 && (
            <div>
              <h4 className="font-semibold text-slate-200">Related findings</h4>
              <ul className="mt-2 space-y-2">
                {result.fallback_edges.map((edge, idx) => (
                  <li key={idx} className="rounded border border-slate-800 bg-slate-950 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <span>
                        <strong className="text-slate-100">{edge.src_name}</strong>
                        <span className="text-slate-400"> {edge.relation}</span>
                        <strong className="text-slate-100"> {edge.dst_name}</strong>
                      </span>
                      <button
                        type="button"
                        onClick={() => onHighlightEdge?.(edge)}
                        className="text-xs text-sky-300 hover:text-sky-100"
                      >
                        Highlight
                      </button>
                    </div>
                    {edge.evidence.full_sentence && (
                      <p className="mt-2 text-xs text-slate-300">{edge.evidence.full_sentence}</p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
