"use client";

import { useMemo } from "react";

import type { GraphFilters } from "../lib/api";

interface FiltersPanelProps {
  filters: GraphFilters;
  relations: string[];
  sections: string[];
  papers: string[];
  onChange: (filters: GraphFilters) => void;
}

export function FiltersPanel({ filters, relations, sections, papers, onChange }: FiltersPanelProps) {
  const sectionOptions = useMemo(() => {
    const canonical = ["Intro", "Methods", "Results", "Discussion"];
    const custom = sections.filter((section) => !canonical.includes(section));
    return [...canonical, ...custom];
  }, [sections]);

  const handleRelationToggle = (relation: string) => {
    const active = filters.relations.includes(relation)
      ? filters.relations.filter((item) => item !== relation)
      : [...filters.relations, relation];
    onChange({ ...filters, relations: active });
  };

  const handleSectionToggle = (section: string) => {
    if (section === "All") {
      onChange({ ...filters, sections });
      return;
    }
    const next = filters.sections.includes(section)
      ? filters.sections.filter((item) => item !== section)
      : [...filters.sections, section];
    onChange({ ...filters, sections: next });
  };

  const handleConfidence = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number.parseFloat(event.target.value);
    onChange({ ...filters, min_confidence: Number.isFinite(value) ? value : filters.min_confidence });
  };

  const handlePaperToggle = (paperId: string) => {
    const next = filters.papers.includes(paperId)
      ? filters.papers.filter((item) => item !== paperId)
      : [...filters.papers, paperId];
    onChange({ ...filters, papers: next });
  };

  return (
    <div className="grid gap-4 rounded-lg border border-slate-800 bg-slate-900 p-4 text-sm">
      <div>
        <h3 className="text-sm font-semibold text-slate-200">Relations</h3>
        <div className="mt-2 flex flex-wrap gap-2">
          {relations.map((relation) => {
            const active = filters.relations.includes(relation);
            return (
              <button
                key={relation}
                type="button"
                onClick={() => handleRelationToggle(relation)}
                className={`rounded-full border px-3 py-1 text-xs font-semibold ${active ? "border-sky-500 bg-sky-500/20 text-sky-200" : "border-slate-700 text-slate-400"}`}
              >
                {relation}
              </button>
            );
          })}
        </div>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-200">Minimum confidence</h3>
        <div className="mt-2 flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={filters.min_confidence}
            onChange={handleConfidence}
            className="w-full"
          />
          <span className="w-12 text-right text-xs text-slate-300">
            {filters.min_confidence.toFixed(2)}
          </span>
        </div>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-200">Sections</h3>
        <div className="mt-2 flex flex-wrap gap-2">
          {sectionOptions.map((section) => {
            const active = filters.sections.includes(section);
            return (
              <button
                key={section}
                type="button"
                onClick={() => handleSectionToggle(section)}
                className={`rounded-full border px-3 py-1 text-xs font-semibold ${active ? "border-emerald-500 bg-emerald-500/20 text-emerald-200" : "border-slate-700 text-slate-400"}`}
              >
                {section}
              </button>
            );
          })}
          <button
            type="button"
            onClick={() => handleSectionToggle("All")}
            className="rounded-full border border-slate-700 px-3 py-1 text-xs font-semibold text-slate-400"
          >
            All
          </button>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm text-slate-200">
          <input
            type="checkbox"
            checked={filters.include_co_mentions}
            onChange={(event) =>
              onChange({ ...filters, include_co_mentions: event.target.checked })
            }
          />
          Show co-mention edges
        </label>
        <div className="flex flex-wrap gap-2 text-xs">
          {papers.map((paper) => {
            const active = filters.papers.includes(paper);
            return (
              <button
                key={paper}
                type="button"
                onClick={() => handlePaperToggle(paper)}
                className={`rounded border px-2 py-1 ${active ? "border-violet-500 bg-violet-500/20 text-violet-200" : "border-slate-700 text-slate-400"}`}
              >
                {paper}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
