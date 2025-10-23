"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, HelpCircle, Loader2, MessageSquare } from "lucide-react";

import apiClient, { extractErrorMessage } from "../lib/http";

type EvidenceModel = {
  doc_id: string;
  element_id: string;
  text_span: { start: number; end: number };
  full_sentence?: string | null;
};

type PathEdgeModel = {
  src_id: string;
  src_name: string;
  dst_id: string;
  dst_name: string;
  relation: string;
  relation_verbatim: string;
  confidence: number;
  created_at: string;
  conflicting: boolean;
  evidence: EvidenceModel;
  attributes: Record<string, string>;
};

type PathModel = {
  edges: PathEdgeModel[];
  confidence_product: number;
  section_score: number;
  score: number;
  latest_timestamp: string;
};

type CandidateModel = {
  node_id: string;
  name: string;
  aliases: string[];
  times_seen: number;
  section_distribution: Record<string, number>;
  similarity: number;
  selected: boolean;
};

type ResolvedEntityModel = {
  mention: string;
  candidates: CandidateModel[];
};

type QAResponse = {
  mode: "direct" | "insufficient" | "conflicting";
  summary: string;
  resolved_entities: ResolvedEntityModel[];
  paths: PathModel[];
  fallback_edges: PathEdgeModel[];
  llm_answer?: string | null;
};

type ChatTurn = {
  id: string;
  question: string;
  response: QAResponse;
  createdAt: string;
};

type UiSettingsResponse = {
  qa?: {
    llm_enabled: boolean;
    llm_provider?: string | null;
  };
};

const MODE_LABELS: Record<QAResponse["mode"], string> = {
  direct: "Answer",
  insufficient: "Insufficient evidence",
  conflicting: "Conflicting evidence",
};

const formatTimestamp = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "—";
  }
  return parsed.toLocaleString();
};

const confidenceToPercent = (value: number) => `${Math.round(value * 100)}%`;

const renderAttributes = (attributes: Record<string, string>) => {
  const entries = Object.entries(attributes);
  if (entries.length === 0) {
    return null;
  }
  return (
    <dl className="grid gap-1 text-xs text-muted-foreground">
      {entries.map(([key, val]) => (
        <div key={key} className="flex gap-2">
          <dt className="font-medium text-foreground">{key}</dt>
          <dd>{val}</dd>
        </div>
      ))}
    </dl>
  );
};

const CandidateList = ({ candidates }: { candidates: CandidateModel[] }) => {
  if (!candidates.length) {
    return <p className="text-xs italic text-muted-foreground">No entities resolved.</p>;
  }

  return (
    <ul className="space-y-2 text-xs text-muted-foreground">
      {candidates.map((candidate) => (
        <li
          key={candidate.node_id}
          className={`rounded-md border px-3 py-2 ${candidate.selected ? "border-primary/60 bg-primary/5" : "border-border bg-background"}`}
        >
          <div className="flex items-center justify-between gap-3">
            <p className="font-medium text-foreground">{candidate.name}</p>
            <span className="rounded-full bg-muted px-2 py-0.5 font-medium text-muted-foreground">
              {confidenceToPercent(candidate.similarity)} match
            </span>
          </div>
          {candidate.aliases.length > 0 ? (
            <p className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground">
              Aliases: {candidate.aliases.join(", ")}
            </p>
          ) : null}
          <p className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground">
            Times seen: {candidate.times_seen}
          </p>
        </li>
      ))}
    </ul>
  );
};

const EdgeList = ({ edges, title }: { edges: PathEdgeModel[]; title: string }) => {
  if (!edges.length) {
    return null;
  }

  return (
    <section className="space-y-3 rounded-lg border bg-card p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      <ul className="space-y-3 text-xs text-muted-foreground">
        {edges.map((edge, index) => (
          <li key={`${edge.src_id}-${edge.dst_id}-${index}`} className="rounded-md border border-border bg-background p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
              <span>
                {edge.src_name} → {edge.dst_name}
              </span>
              <span>
                {edge.relation_verbatim} · {confidenceToPercent(edge.confidence)} · {formatTimestamp(edge.created_at)}
              </span>
            </div>
            <p className="mt-2 text-sm text-foreground">{edge.evidence.full_sentence ?? "Evidence excerpt unavailable."}</p>
            <p className="mt-2 text-[11px] text-muted-foreground">
              Doc: {edge.evidence.doc_id} · Element: {edge.evidence.element_id} · Offsets {edge.evidence.text_span.start}–
              {edge.evidence.text_span.end}
            </p>
            {edge.conflicting ? (
              <p className="mt-2 text-[11px] font-semibold uppercase tracking-wide text-rose-600">
                Conflicting evidence detected
              </p>
            ) : null}
            {renderAttributes(edge.attributes)}
          </li>
        ))}
      </ul>
    </section>
  );
};

const PathList = ({ paths }: { paths: PathModel[] }) => {
  if (!paths.length) {
    return null;
  }

  return (
    <div className="space-y-4">
      {paths.map((path, index) => (
        <div key={index} className="space-y-3 rounded-lg border bg-card p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
            <span>Path #{index + 1}</span>
            <span>
              Score {confidenceToPercent(path.score)} · Confidence product {confidenceToPercent(path.confidence_product)} ·
              Latest {formatTimestamp(path.latest_timestamp)}
            </span>
          </div>
          <EdgeList edges={path.edges} title="Reasoning steps" />
        </div>
      ))}
    </div>
  );
};

const QaPanel = () => {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [qaSettings, setQaSettings] = useState<{ llmEnabled: boolean | null; provider: string | null }>({
    llmEnabled: null,
    provider: null,
  });
  const [settingsError, setSettingsError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const { data } = await apiClient.get<UiSettingsResponse>("/api/ui/settings");
        if (!mounted) {
          return;
        }
        setQaSettings({
          llmEnabled: data.qa ? Boolean(data.qa.llm_enabled) : false,
          provider: data.qa?.llm_provider ?? null,
        });
      } catch (err) {
        if (!mounted) {
          return;
        }
        setQaSettings({ llmEnabled: false, provider: null });
        setSettingsError(extractErrorMessage(err, "Unable to load QA settings; defaulting to graph summaries."));
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const settingsLoaded = qaSettings.llmEnabled !== null;
  const isLlmEnabled = qaSettings.llmEnabled ?? false;

  const hint = useMemo(() => {
    if (!settingsLoaded) {
      return "Loading QA settings...";
    }
    if (!isLlmEnabled) {
      return "LLM synthesis disabled; responses will include graph summaries only.";
    }
    if (isLoading && pendingQuestion) {
      return `Answering "${pendingQuestion}"...`;
    }
    if (history.length === 0) {
      return "Ask a question about the knowledge graph to surface relevant evidence.";
    }
    if (error) {
      return null;
    }
    const latest = history[history.length - 1];
    return `Latest results for "${latest.question}".`;
  }, [error, history, isLlmEnabled, isLoading, pendingQuestion, settingsLoaded]);

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = question.trim();

      if (!trimmed) {
        setError("Please enter a question before asking.");
        return;
      }
      if (!settingsLoaded) {
        setError("QA settings are still loading. Please try again in a moment.");
        return;
      }

      setPendingQuestion(trimmed);
      setIsLoading(true);
      setError(null);
      try {
        const { data } = await apiClient.post<QAResponse>("/api/qa/ask", { question: trimmed });
        const id =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? crypto.randomUUID()
            : `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const turn: ChatTurn = {
          id,
          question: trimmed,
          response: data,
          createdAt: new Date().toISOString(),
        };
        setHistory((prev) => [...prev, turn]);
        setQuestion("");
      } catch (err) {
        const message = extractErrorMessage(err, "Unable to answer the question right now.");
        setError(message);
      } finally {
        setPendingQuestion(null);
        setIsLoading(false);
      }
    }, [question, settingsLoaded]);

  const handleClear = () => {
    setQuestion("");
    setHistory([]);
    setPendingQuestion(null);
    setError(null);
  };

  return (
    <section className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
            <MessageSquare className="h-4 w-4" />
          </span>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Graph QA</h2>
            {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <HelpCircle className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask how concepts are connected or what evidence links entities"
            className="w-full rounded-md border border-input bg-background px-3 py-2 pl-9 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
          />
        </div>
        <div className="flex gap-2">
          <button
            type="submit"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-70"
            disabled={isLoading || !question.trim()}
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Answering
              </span>
            ) : (
              "Ask"
            )}
          </button>
          <button
            type="button"
            onClick={handleClear}
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-70"
            disabled={isLoading || (history.length === 0 && !question)}
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

      <div className="mt-6 space-y-6">
        {history.length === 0 ? (
          <div className="rounded-lg border bg-background p-6 text-sm text-muted-foreground shadow-sm">
            {isLoading ? (
              <span className="flex items-center gap-2 text-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Generating the answer...
              </span>
            ) : (
              <p>
                No questions asked yet. Try queries like{" "}
                <span className="font-medium text-foreground">&ldquo;How does Method A compare to Method B?&rdquo;</span> or{" "}
                <span className="font-medium text-foreground">&ldquo;What evidence links Dataset X to Metric Y?&rdquo;</span>
              </p>
            )}
          </div>
        ) : (
          history
            .slice()
            .reverse()
            .map((turn) => {
              const llmAnswer = turn.response.llm_answer?.trim();
              return (
                <article key={turn.id} className="space-y-4 rounded-lg border bg-card/70 p-5 shadow-sm">
                  <header className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">Question</p>
                      <p className="text-sm font-semibold text-foreground">{turn.question}</p>
                    </div>
                    <span className="text-xs text-muted-foreground">{formatTimestamp(turn.createdAt)}</span>
                  </header>

                  <div className="rounded-md border border-primary/30 bg-primary/5 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                      <span className="font-semibold text-foreground">{MODE_LABELS[turn.response.mode]}</span>
                      <span>{turn.response.paths.length} reasoning paths</span>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                      {llmAnswer || turn.response.summary}
                    </p>
                    {llmAnswer ? (
                      <p className="mt-3 text-xs text-muted-foreground">Graph summary: {turn.response.summary}</p>
                    ) : null}
                  </div>

                  <details className="rounded-md border border-border/60 bg-background p-4">
                    <summary className="cursor-pointer text-sm font-semibold text-foreground">
                      Evidence &amp; supporting details
                    </summary>
                    <div className="mt-4 space-y-6 text-sm">
                      <section className="space-y-3">
                        <h4 className="text-sm font-semibold text-foreground">Resolved entities</h4>
                        {turn.response.resolved_entities.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No entities were resolved from the question.</p>
                        ) : (
                          <div className="grid gap-4 md:grid-cols-2">
                            {turn.response.resolved_entities.map((entity) => (
                              <div key={entity.mention} className="space-y-2 rounded-lg border bg-card p-4 shadow-sm">
                                <p className="text-sm font-semibold text-foreground">Mention: {entity.mention}</p>
                                <CandidateList candidates={entity.candidates} />
                              </div>
                            ))}
                          </div>
                        )}
                      </section>

                      <section className="space-y-3">
                        <h4 className="text-sm font-semibold text-foreground">Reasoning paths</h4>
                        {turn.response.paths.length > 0 ? (
                          <PathList paths={turn.response.paths} />
                        ) : (
                          <p className="text-xs text-muted-foreground">No multi-hop paths discovered.</p>
                        )}
                      </section>

                      <section className="space-y-3">
                        <h4 className="text-sm font-semibold text-foreground">Fallback evidence</h4>
                        {turn.response.fallback_edges.length > 0 ? (
                          <EdgeList edges={turn.response.fallback_edges} title="Related findings" />
                        ) : (
                          <p className="text-xs text-muted-foreground">No fallback evidence returned.</p>
                        )}
                      </section>
                    </div>
                  </details>
                </article>
              );
            })
        )}
      </div>
    </section>
  );
};

export default QaPanel;
