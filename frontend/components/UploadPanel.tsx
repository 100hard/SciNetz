"use client";

import { useState } from "react";

import { uploadPaper } from "../lib/api";
import type { PaperSummary } from "../lib/types";

interface UploadPanelProps {
  onUploaded: (paper: PaperSummary) => void;
}

export function UploadPanel({ onUploaded }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [paperId, setPaperId] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setError("Select a PDF before uploading");
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      const summary = await uploadPaper(file, paperId.trim() || undefined);
      onUploaded(summary);
      setFile(null);
      setPaperId("");
      (event.target as HTMLFormElement).reset();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 rounded-lg border border-slate-800 bg-slate-900 p-4 shadow"
    >
      <h2 className="text-base font-semibold text-sky-300">Upload paper</h2>
      <p className="text-sm text-slate-400">
        Drop a PDF to ingest it through the full pipeline. Provide an optional custom identifier if
        you need to align with external systems.
      </p>
      <label className="flex flex-col gap-2 text-sm">
        <span className="font-medium text-slate-200">Choose PDF</span>
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => {
            const selected = event.target.files?.[0];
            setFile(selected ?? null);
          }}
          className="rounded border border-slate-700 bg-slate-950 px-3 py-2"
        />
      </label>
      <label className="flex flex-col gap-2 text-sm">
        <span className="font-medium text-slate-200">Paper ID (optional)</span>
        <input
          type="text"
          value={paperId}
          onChange={(event) => setPaperId(event.target.value)}
          placeholder="auto-generated if left blank"
          className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
        />
      </label>
      {error && <p className="text-sm text-rose-400">{error}</p>}
      <div className="flex items-center justify-end gap-3">
        <button
          type="submit"
          disabled={isSubmitting}
          className="rounded bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:bg-sky-800"
        >
          {isSubmitting ? "Uploading…" : "Upload"}
        </button>
      </div>
    </form>
  );
}
