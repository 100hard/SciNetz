"use client";

import { FormEvent, useCallback, useMemo, useState } from "react";
import axios from "axios";
import { AlertCircle, FileText, Loader2, Search } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_RESULTS = 8;

type SemanticSearchResult = {
  paper_id: string;
  section_id?: string | null;
  section_title?: string | null;
  snippet: string;
  score: number;
  page_number?: number | null;
  char_start?: number | null;
  char_end?: number | null;
};

const getErrorMessage = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string" && detail.trim().length > 0) {
      return detail;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error while performing semantic search.";
};

const formatScore = (value: number) => {
  const normalized = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
  const percentage = Math.round(((normalized + 1) / 2) * 100);
  return `${percentage}% match`;
};

const formatOffsets = (start?: number | null, end?: number | null) => {
  if (typeof start === "number" && typeof end === "number") {
    return `${start.toLocaleString()} – ${end.toLocaleString()}`;
  }
  return null;
};

const truncateId = (value: string) => {
  if (value.length <= 12) {
    return value;
  }
  return `${value.slice(0, 6)}…${value.slice(-4)}`;
};

const SemanticSearch = () => {
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasResults = results.length > 0;

  const hint = useMemo(() => {
    if (isSearching) {
      return `Searching for “${activeQuery}”...`;
    }
    if (!activeQuery) {
      return "Ask a question about your corpus to surface relevant sections.";
    }
    if (error) {
      return null;
    }
    if (!hasResults) {
      return `No results for “${activeQuery}”. Try refining your question.`;
    }
    return `Top matches for “${activeQuery}”.`;
  }, [activeQuery, error, hasResults, isSearching]);

  const performSearch = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      setActiveQuery(trimmed);

      if (!trimmed) {
        setResults([]);
        setError(null);
        return;
      }

      setIsSearching(true);
      try {
        const response = await axios.get<SemanticSearchResult[]>(
          `${API_BASE_URL}/api/search/similarity`,
          {
            params: { q: trimmed, limit: MAX_RESULTS },
          }
        );
        setResults(response.data);
        setError(null);
      } catch (err) {
        setResults([]);
        setError(getErrorMessage(err));
      } finally {
        setIsSearching(false);
      }
    },
    []
  );

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      void performSearch(query);
    },
    [performSearch, query]
  );

  const handleReset = useCallback(() => {
    setQuery("");
    setActiveQuery("");
    setResults([]);
    setError(null);
  }, []);

  return (
    <section className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Search className="h-4 w-4" />
          </span>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Semantic search</h2>
            {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about findings, methods, or datasets..."
            className="w-full rounded-md border border-input bg-background px-3 py-2 pl-9 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
          />
        </div>
        <div className="flex gap-2">
          <button
            type="submit"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-70"
            disabled={isSearching}
          >
            {isSearching ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Searching
              </span>
            ) : (
              "Search"
            )}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-70"
            disabled={isSearching}
          >
            Clear
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-4 flex items-start gap-2 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      <div className="mt-5 space-y-4">
        {isSearching && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Fetching semantic matches…
          </div>
        )}

        {!isSearching && hasResults && (
          <ul className="space-y-4">
            {results.map((item, index) => {
              const offset = formatOffsets(item.char_start, item.char_end);
              const key = item.section_id ?? `${item.paper_id}-${index}`;
              return (
                <li
                  key={key}
                  className="rounded-md border bg-background/60 p-4 shadow-sm transition hover:bg-background"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                      <FileText className="h-4 w-4 text-primary" />
                      <span>{item.section_title?.trim() || "Untitled section"}</span>
                    </div>
                    <span className="inline-flex items-center rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-primary">
                      {formatScore(item.score)}
                    </span>
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{item.snippet}</p>
                  <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
                    {item.page_number != null && (
                      <div className="flex items-center gap-1">
                        <dt className="font-medium text-foreground">Page</dt>
                        <dd>{item.page_number}</dd>
                      </div>
                    )}
                    {offset && (
                      <div className="flex items-center gap-1">
                        <dt className="font-medium text-foreground">Offsets</dt>
                        <dd>{offset}</dd>
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <dt className="font-medium text-foreground">Paper</dt>
                      <dd className="font-mono text-[11px] text-muted-foreground/80">
                        {truncateId(item.paper_id)}
                      </dd>
                    </div>
                  </dl>
                </li>
              );
            })}
          </ul>
        )}

        {!isSearching && !hasResults && !error && activeQuery && (
          <p className="text-sm text-muted-foreground">
            No semantic matches yet. Consider broadening your question or trying alternative terminology.
          </p>
        )}

        {!isSearching && !activeQuery && !error && (
          <p className="text-sm text-muted-foreground">
            Tip: Ask targeted questions like “How do graph neural networks model protein folding?”
          </p>
        )}
      </div>
    </section>
  );
};

export default SemanticSearch;
