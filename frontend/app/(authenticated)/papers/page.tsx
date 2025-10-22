"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  FileText,
  Loader2,
  RefreshCw,
  Search,
  User,
} from "lucide-react";

import { toast } from "sonner";

import apiClient, { extractErrorMessage } from "../../lib/http";

type PaperMetadata = {
  doc_id: string;
  title?: string | null;
  authors?: string[];
  year?: number | null;
  venue?: string | null;
  doi?: string | null;
};

type PaperSummary = {
  paper_id: string;
  filename: string;
  status: string;
  uploaded_at: string;
  updated_at: string;
  metadata?: PaperMetadata | null;
  errors: string[];
  nodes_written: number;
  edges_written: number;
  co_mention_edges: number;
};

type ExtractionResponse = {
  doc_id: string;
  nodes_written: number;
  edges_written: number;
  co_mention_edges: number;
  errors?: string[];
  metadata?: PaperMetadata;
  processed_chunks?: number;
  skipped_chunks?: number;
};

const POLL_INTERVAL_MS = 12000;
const ACTIVE_STATUSES = new Set(["uploaded", "processing"]);

const STATUS_STYLES: Record<string, string> = {
  complete: "border-emerald-200 bg-emerald-50 text-emerald-700",
  uploaded: "border-amber-200 bg-amber-50 text-amber-700",
  processing: "border-sky-200 bg-sky-50 text-sky-700",
  failed: "border-rose-200 bg-rose-50 text-rose-700",
};

const formatDate = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
};

const toStatusLabel = (status: string) =>
  status
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1).toLowerCase())
    .join(" ") || "Unknown";

const StatusBadge = ({ status }: { status: string }) => {
  const normalized = status.toLowerCase();
  const variant = STATUS_STYLES[normalized] ?? "border-border bg-muted/40 text-muted-foreground";
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium capitalize transition ${variant}`}>
      {toStatusLabel(status)}
    </span>
  );
};

const summariseMetadata = (metadata?: PaperMetadata | null) => {
  if (!metadata) {
    return "Metadata pending";
  }
  const pieces: string[] = [];
  if (metadata.title) {
    pieces.push(metadata.title);
  }
  if (metadata.authors && metadata.authors.length > 0) {
    pieces.push(metadata.authors.join(", "));
  }
  if (metadata.venue || metadata.year) {
    pieces.push([metadata.venue, metadata.year].filter(Boolean).join(" · "));
  }
  if (metadata.doi) {
    pieces.push(metadata.doi);
  }
  return pieces.filter(Boolean).join("\n");
};

export default function PapersPage() {
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingExtraction, setPendingExtraction] = useState<Record<string, boolean>>({});
  const [titleQuery, setTitleQuery] = useState("");
  const [authorQuery, setAuthorQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchPapers = useCallback(
    async (mode: "loading" | "refresh" = "refresh") => {
      if (mode === "loading") {
        setIsLoading(true);
      } else {
        setIsRefreshing(true);
      }

      try {
        const response = await apiClient.get<PaperSummary[]>("/api/ui/papers");
        setPapers(response.data);
        setLastUpdated(new Date());
        setError(null);
      } catch (err) {
        const message = extractErrorMessage(err, "Unexpected error while fetching papers.");
        setError(message);
      } finally {
        if (mode === "loading") {
          setIsLoading(false);
        } else {
          setIsRefreshing(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    void fetchPapers("loading");
  }, [fetchPapers]);

  const shouldPoll = useMemo(
    () => papers.some((paper) => ACTIVE_STATUSES.has(paper.status.toLowerCase())),
    [papers],
  );

  useEffect(() => {
    if (!shouldPoll) {
      return undefined;
    }

    const interval = window.setInterval(() => {
      void fetchPapers("refresh");
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(interval);
  }, [fetchPapers, shouldPoll]);

  useEffect(() => {
    const handleFocus = () => {
      void fetchPapers("refresh");
    };

    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        void fetchPapers("refresh");
      }
    };

    window.addEventListener("focus", handleFocus);
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      window.removeEventListener("focus", handleFocus);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [fetchPapers]);

  const filteredPapers = useMemo(() => {
    const normalizedTitle = titleQuery.trim().toLowerCase();
    const normalizedAuthor = authorQuery.trim().toLowerCase();
    const normalizedStatus = statusFilter.toLowerCase();

    return papers.filter((paper) => {
      const metadataSummary = summariseMetadata(paper.metadata).toLowerCase();
      const matchesTitle =
        !normalizedTitle ||
        paper.filename.toLowerCase().includes(normalizedTitle) ||
        metadataSummary.includes(normalizedTitle);
      const matchesAuthor =
        !normalizedAuthor || metadataSummary.includes(normalizedAuthor);
      const matchesStatus =
        normalizedStatus === "all" || paper.status.toLowerCase() === normalizedStatus;

      return matchesTitle && matchesAuthor && matchesStatus;
    });
  }, [papers, titleQuery, authorQuery, statusFilter]);

  const summary = useMemo(() => {
    const pendingIds = new Set(Object.keys(pendingExtraction));
    const deriveStatus = (paper: PaperSummary) =>
      (pendingIds.has(paper.paper_id) ? "processing" : paper.status).toLowerCase();

    const total = papers.length;
    const complete = papers.filter((paper) => deriveStatus(paper) === "complete").length;
    const failed = papers.filter((paper) => deriveStatus(paper) === "failed").length;
    const inFlight = papers.filter((paper) => ACTIVE_STATUSES.has(deriveStatus(paper))).length;

    return [
      { label: "Total papers", value: total },
      { label: "Complete", value: complete },
      { label: "In progress", value: inFlight },
      { label: "Failed", value: failed },
    ];
  }, [papers, pendingExtraction]);

  const handleRefresh = useCallback(() => {
    void fetchPapers("refresh");
  }, [fetchPapers]);

  const triggerExtraction = useCallback(
    async (paper: PaperSummary) => {
      const paperId = paper.paper_id;
      setPendingExtraction((prev) => ({ ...prev, [paperId]: true }));
      try {
        const response = await apiClient.post<ExtractionResponse>(
          `/api/ui/papers/${encodeURIComponent(paperId)}/extract`,
        );
        toast.success("Extraction complete", {
          description: `${paper.filename} processed (${response.data.nodes_written ?? 0} nodes, ${response.data.edges_written ?? 0} edges).`,
        });
        await fetchPapers("refresh");
      } catch (err) {
        const message = extractErrorMessage(err, "Extraction request failed.");
        toast.error("Extraction failed", { description: message });
      } finally {
        setPendingExtraction((prev) => {
          const next = { ...prev };
          delete next[paperId];
          return next;
        });
      }
    },
    [fetchPapers],
  );

  const handleResetFilters = () => {
    setTitleQuery("");
    setAuthorQuery("");
    setStatusFilter("all");
  };

  const lastUpdatedLabel = lastUpdated ? `Last updated ${lastUpdated.toLocaleTimeString()}` : "Awaiting first sync";

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Papers</p>
        <h1 className="text-2xl font-semibold text-foreground">Monitor ingestion progress</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Search, filter, and monitor uploaded research papers as they progress through the parsing pipeline.
        </p>
      </div>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {summary.map((item) => (
          <div
            key={item.label}
            className="fade-in-up rounded-lg border bg-card p-5 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md"
          >
            <p className="text-sm font-medium text-muted-foreground">{item.label}</p>
            <p className="mt-3 text-2xl font-semibold text-foreground">
              {isLoading ? <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /> : item.value}
            </p>
          </div>
        ))}
      </section>

      <section className="space-y-4 rounded-lg border bg-card p-6 shadow-sm">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="search"
              value={titleQuery}
              onChange={(event) => setTitleQuery(event.target.value)}
              placeholder="Search by filename or title"
              className="h-10 w-60 rounded-md border border-input bg-background px-3 py-2 pl-9 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
            />
          </div>

          <div className="relative">
            <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="search"
              value={authorQuery}
              onChange={(event) => setAuthorQuery(event.target.value)}
              placeholder="Filter by author or venue"
              className="h-10 w-56 rounded-md border border-input bg-background px-3 py-2 pl-9 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
            />
          </div>

          <select
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value)}
            className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground shadow-sm transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
            aria-label="Filter by status"
          >
            <option value="all">All statuses</option>
            <option value="complete">Complete</option>
            <option value="uploaded">Uploaded</option>
            <option value="processing">Processing</option>
            <option value="failed">Failed</option>
          </select>

          <button
            type="button"
            onClick={handleResetFilters}
            className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm font-medium text-foreground shadow-sm transition hover:bg-muted"
          >
            Reset filters
          </button>

          <div className="ml-auto flex items-center gap-3 text-sm text-muted-foreground">
            <span>{lastUpdatedLabel}</span>
            <button
              type="button"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className={`inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground shadow-sm transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${isRefreshing ? "opacity-80" : "hover:bg-primary/90"} disabled:cursor-not-allowed disabled:opacity-60`}
            >
              {isRefreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />} Refresh
            </button>
          </div>
        </div>

        {error ? (
          <div className="fade-in flex items-start gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            <div>
              <p className="font-medium">Unable to load papers</p>
              <p className="text-xs text-destructive/80">{error}</p>
            </div>
          </div>
        ) : null}

        <div className="overflow-hidden rounded-md border border-border">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th scope="col" className="px-4 py-3 text-left font-medium">
                  Filename
                </th>
                <th scope="col" className="px-4 py-3 text-left font-medium">
                  Metadata
                </th>
                <th scope="col" className="px-4 py-3 text-left font-medium">
                  Status
                </th>
                <th scope="col" className="px-4 py-3 text-left font-medium">
                  Uploaded
                </th>
                <th scope="col" className="px-4 py-3 text-left font-medium">
                  Updated
                </th>
                <th scope="col" className="px-4 py-3 text-right font-medium">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-background">
              {isLoading
                ? Array.from({ length: 5 }).map((_, index) => (
                    <tr key={index} className="animate-pulse">
                      <td className="px-4 py-4">
                        <div className="h-4 w-3/4 rounded bg-muted" />
                        <div className="mt-2 h-3 w-1/3 rounded bg-muted/60" />
                      </td>
                      <td className="px-4 py-4">
                        <div className="h-4 w-2/3 rounded bg-muted" />
                      </td>
                      <td className="px-4 py-4">
                        <div className="h-6 w-20 rounded-full bg-muted" />
                      </td>
                      <td className="px-4 py-4">
                        <div className="h-4 w-24 rounded bg-muted" />
                      </td>
                      <td className="px-4 py-4">
                        <div className="h-4 w-24 rounded bg-muted" />
                      </td>
                      <td className="px-4 py-4">
                        <div className="ml-auto h-8 w-24 rounded bg-muted" />
                      </td>
                    </tr>
                  ))
                : filteredPapers.length > 0
                ? filteredPapers.map((paper) => {
                    const isPending = Boolean(pendingExtraction[paper.paper_id]);
                    const derivedStatus = isPending ? "processing" : paper.status;
                    const metadataSummary = summariseMetadata(paper.metadata);
                    return (
                      <tr key={paper.paper_id} className="fade-in transition-colors duration-200 hover:bg-muted/40">
                        <td className="px-4 py-4">
                          <div className="font-medium text-foreground">{paper.filename}</div>
                          <div className="mt-1 text-xs text-muted-foreground">Paper ID: {paper.paper_id}</div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            Nodes: {paper.nodes_written} · Edges: {paper.edges_written} · Co-mentions: {paper.co_mention_edges}
                          </div>
                          {paper.errors.length > 0 ? (
                            <div className="mt-2 rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
                              <p className="font-semibold text-foreground">Pipeline warnings</p>
                              <ul className="list-inside list-disc space-y-1">
                                {paper.errors.map((item) => (
                                  <li key={item}>{item}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                        </td>
                        <td className="px-4 py-4 whitespace-pre-wrap text-sm text-muted-foreground">
                          {metadataSummary}
                        </td>
                        <td className="px-4 py-4">
                          <StatusBadge status={derivedStatus} />
                        </td>
                        <td className="px-4 py-4 text-sm text-muted-foreground">{formatDate(paper.uploaded_at)}</td>
                        <td className="px-4 py-4 text-sm text-muted-foreground">{formatDate(paper.updated_at)}</td>
                        <td className="px-4 py-4">
                          <div className="flex justify-end">
                            <button
                              type="button"
                              onClick={() => void triggerExtraction(paper)}
                              disabled={
                                pendingExtraction[paper.paper_id] ||
                                paper.status.toLowerCase() === "processing"
                              }
                              className={`inline-flex items-center gap-2 rounded-md border border-input px-3 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${
                                pendingExtraction[paper.paper_id] || paper.status.toLowerCase() === "processing"
                                  ? "cursor-not-allowed border-muted bg-muted text-muted-foreground opacity-60"
                                  : "bg-background text-foreground hover:bg-muted"
                              }`}
                            >
                              {pendingExtraction[paper.paper_id] ? (
                                <>
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                  Running…
                                </>
                              ) : (
                                <>
                                  <RefreshCw className="h-4 w-4" />
                                  {paper.status.toLowerCase() === "complete" ? "Re-run extraction" : "Start extraction"}
                                </>
                              )}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                : (
                    <tr className="fade-in-up">
                      <td colSpan={6} className="px-6 py-12">
                        <div className="flex flex-col items-center gap-3 text-center text-sm text-muted-foreground">
                          <FileText className="h-8 w-8 text-muted-foreground" />
                          <div>
                            <p className="font-medium text-foreground">No papers match the current filters.</p>
                            <p className="text-xs text-muted-foreground">
                              Adjust the search fields or refresh to fetch the latest uploads.
                            </p>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
            </tbody>
          </table>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <p>
            Showing <span className="font-semibold text-foreground">{filteredPapers.length}</span> of
            <span className="font-semibold text-foreground"> {papers.length}</span> papers
          </p>
          <p>Polling every {Math.round(POLL_INTERVAL_MS / 1000)}s for status updates</p>
        </div>
      </section>
    </div>
  );
}
