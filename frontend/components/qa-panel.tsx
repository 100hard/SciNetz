"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, HelpCircle, Loader2, MessageSquare } from "lucide-react";

import apiClient, { buildApiUrl, extractErrorMessage } from "@/lib/http";

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

type QAStreamEvent =
  | { type: "classification"; payload: { intent: string; document_ids: string[] } }
  | { type: "entities"; payload: ResolvedEntityModel[] }
  | { type: "paths"; payload: PathModel[] }
  | { type: "fallback"; payload: PathEdgeModel[] }
  | { type: "llm_delta"; payload: string }
  | { type: "llm_answer"; payload: string | null }
  | { type: "final"; payload: QAResponse }
  | { type: string; payload: unknown };

type ChatTurn = {
  id: string;
  question: string;
  response: QAResponse;
  createdAt: string;
  pending?: boolean;
  classification?: { intent: string; document_ids: string[] } | null;
};

type UiSettingsResponse = {
  qa?: {
    llm_enabled: boolean;
    llm_provider?: string | null;
  };
  polling?: {
    active_interval_seconds?: number;
    idle_interval_seconds?: number;
  };
};

const MODE_LABELS: Record<QAResponse["mode"], string> = {
  direct: "Answer",
  insufficient: "Insufficient evidence",
  conflicting: "Conflicting evidence",
};

const HISTORY_STORAGE_KEY = "scinetz.qa.history";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isChatTurn = (value: unknown): value is ChatTurn => {
  if (!isRecord(value)) {
    return false;
  }
  if (typeof value.id !== "string" || typeof value.question !== "string" || typeof value.createdAt !== "string") {
    return false;
  }
  const response = value.response;
  if (!isRecord(response)) {
    return false;
  }
  if (typeof response.mode !== "string" || typeof response.summary !== "string") {
    return false;
  }
  if (!Array.isArray(response.resolved_entities) || !Array.isArray(response.paths) || !Array.isArray(response.fallback_edges)) {
    return false;
  }
  return true;
};

const parseStoredHistory = (raw: string | null): ChatTurn[] => {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    const turns = parsed.filter(isChatTurn);
    return turns;
  } catch (err) {
    console.warn("Failed to parse stored QA history", err);
    return [];
  }
};

const createInitialResponse = (): QAResponse => ({
  mode: "insufficient",
  summary: "Processing answer...",
  resolved_entities: [],
  paths: [],
  fallback_edges: [],
  llm_answer: null,
});

const normalizeClassification = (payload: unknown): { intent: string; document_ids: string[] } | null => {
  if (!isRecord(payload)) {
    return null;
  }
  const intent = typeof payload.intent === "string" ? payload.intent : "factoid";
  const documentIds = Array.isArray(payload.document_ids)
    ? payload.document_ids.map((value) => String(value))
    : [];
  return { intent, document_ids: documentIds };
};

const coerceResolvedEntities = (payload: unknown): ResolvedEntityModel[] => {
  if (!Array.isArray(payload)) {
    return [];
  }
  return payload as ResolvedEntityModel[];
};

const coercePaths = (payload: unknown): PathModel[] => {
  if (!Array.isArray(payload)) {
    return [];
  }
  return payload as PathModel[];
};

const coerceEdges = (payload: unknown): PathEdgeModel[] => {
  if (!Array.isArray(payload)) {
    return [];
  }
  return payload as PathEdgeModel[];
};

const coerceResponse = (payload: unknown, fallback: QAResponse): QAResponse => {
  if (!isRecord(payload)) {
    return fallback;
  }
  try {
    const response = payload as QAResponse;
    return {
      mode: response.mode,
      summary: response.summary,
      resolved_entities: response.resolved_entities ?? [],
      paths: response.paths ?? [],
      fallback_edges: response.fallback_edges ?? [],
      llm_answer: response.llm_answer ?? null,
    };
  } catch (err) {
    console.warn("Failed to coerce QA response from stream event", err);
    return fallback;
  }
};

const displayAnswerText = (response: QAResponse): string => {
  const trimmed = response.llm_answer?.trim();
  if (!trimmed) {
    return response.summary;
  }
  const normalized = trimmed.toLowerCase();
  if (normalized === "insufficient evidence to answer." && response.fallback_edges.length > 0) {
    return response.summary;
  }
  return trimmed;
};

const applyStreamEventToTurn = (turn: ChatTurn, event: QAStreamEvent): ChatTurn => {
  const response: QAResponse = { ...turn.response };
  const updated: ChatTurn = { ...turn, response };

  switch (event.type) {
    case "classification": {
      updated.classification = normalizeClassification(event.payload);
      updated.pending = true;
      break;
    }
    case "entities": {
      response.resolved_entities = coerceResolvedEntities(event.payload);
      updated.pending = true;
      break;
    }
    case "paths": {
      response.paths = coercePaths(event.payload);
      updated.pending = true;
      break;
    }
    case "fallback": {
      response.fallback_edges = coerceEdges(event.payload);
      updated.pending = true;
      break;
    }
    case "llm_delta": {
      if (typeof event.payload === "string") {
        response.llm_answer = (response.llm_answer ?? "") + event.payload;
      } else if (event.payload !== null && event.payload !== undefined) {
        response.llm_answer = (response.llm_answer ?? "") + String(event.payload);
      }
      updated.pending = true;
      break;
    }
    case "llm_answer": {
      const payload = event.payload;
      if (typeof payload === "string") {
        response.llm_answer = payload;
      } else if (payload === null) {
        response.llm_answer = null;
      } else {
        response.llm_answer = String(payload);
      }
      updated.pending = true;
      break;
    }
    case "final": {
      const finalResponse = coerceResponse(event.payload, response);
      updated.response = finalResponse;
      updated.pending = false;
      break;
    }
    default: {
      break;
    }
  }

  return updated;
};

const formatTimestamp = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "--";
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
                {edge.src_name} {"->"} {edge.dst_name}
              </span>
              <span>
                {edge.relation_verbatim} {" | "} {confidenceToPercent(edge.confidence)} {" | "} {formatTimestamp(edge.created_at)}
              </span>
            </div>
            <p className="mt-2 text-sm text-foreground">{edge.evidence.full_sentence ?? "Evidence excerpt unavailable."}</p>
            <p className="mt-2 text-[11px] text-muted-foreground">
              Doc: {edge.evidence.doc_id}
              {" | "}Element: {edge.evidence.element_id}
              {" | "}Offsets {edge.evidence.text_span.start}
              {"-"}
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
              Score {confidenceToPercent(path.score)}
              {" | "}Confidence product {confidenceToPercent(path.confidence_product)}
              {" | "}Latest {formatTimestamp(path.latest_timestamp)}
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
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [qaSettings, setQaSettings] = useState<{ llmEnabled: boolean | null; provider: string | null }>({
    llmEnabled: null,
    provider: null,
  });
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const applyStreamEvent = useCallback(
    (turnId: string, event: QAStreamEvent) => {
      setHistory((prev) => prev.map((turn) => (turn.id === turnId ? applyStreamEventToTurn(turn, event) : turn)));
    },
    [],
  );

  const streamAnswer = useCallback(
    async (prompt: string, turnId: string) => {
      const response = await fetch(buildApiUrl("/api/qa/ask?stream=1"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        credentials: "include",
        body: JSON.stringify({ question: prompt }),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Streaming request failed with status ${response.status}`);
      }
      const body = response.body;
      if (!body) {
        throw new Error("Streaming response missing body.");
      }
      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;
      let readerCancelled = false;

      const processBuffer = () => {
        let finalReceived = false;
        while (true) {
          const boundary = buffer.indexOf("\n\n");
          if (boundary === -1) {
            break;
          }
          const raw = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          if (!raw.trim()) {
            continue;
          }
          let eventType = "message";
          const dataLines: string[] = [];
          for (const line of raw.split("\n")) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }
          if (dataLines.length === 0) {
            continue;
          }
          const dataText = dataLines.join("");
          try {
            const parsed = JSON.parse(dataText);
            const type = typeof parsed.type === "string" ? parsed.type : eventType;
            const payload = Object.prototype.hasOwnProperty.call(parsed, "payload") ? parsed.payload : parsed;
            const evt: QAStreamEvent = { type, payload };
            applyStreamEvent(turnId, evt);
            if (type === "final") {
              finalReceived = true;
              setPendingQuestion(null);
            }
          } catch (err) {
            console.warn("Failed to parse QA stream event", err, raw);
          }
        }
        return finalReceived;
      };

      try {
        while (!completed) {
          const { value, done } = await reader.read();
          if (value) {
            buffer += decoder.decode(value, { stream: !done });
            if (processBuffer()) {
              completed = true;
              await reader.cancel().catch(() => undefined);
              readerCancelled = true;
              break;
            }
          }
          if (done) {
            buffer += decoder.decode(new Uint8Array(), { stream: false });
            completed = processBuffer() || completed;
            break;
          }
        }
      } finally {
        if (!readerCancelled) {
          await reader.cancel().catch(() => undefined);
        }
      }

      if (!completed) {
        throw new Error("Stream ended before completion.");
      }
    },
    [applyStreamEvent, setPendingQuestion],
  );

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

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    const restored = parseStoredHistory(stored);
    if (restored.length > 0) {
      setHistory(restored);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const completed = history.filter((turn) => !turn.pending);
    if (completed.length === 0) {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
      return;
    }
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(completed));
    } catch (err) {
      console.warn("Failed to persist QA history", err);
    }
  }, [history]);

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
    if (latest?.pending) {
      return `Streaming answer for "${latest.question}"...`;
    }
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

      const id =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const skeleton: ChatTurn = {
        id,
        question: trimmed,
        response: createInitialResponse(),
        createdAt: new Date().toISOString(),
        pending: true,
        classification: null,
      };

      setPendingQuestion(trimmed);
      setIsLoading(true);
      setError(null);
      setHistory((prev) => [...prev, skeleton]);
      setSelectedTurnId(id);
      setQuestion("");

      try {
        await streamAnswer(trimmed, id);
      } catch (streamErr) {
        console.warn("QA streaming failed; attempting fallback request", streamErr);
        try {
          const { data } = await apiClient.post<QAResponse>("/api/qa/ask", { question: trimmed });
          setHistory((prev) =>
            prev.map((turn) => (turn.id === id ? { ...turn, response: data, pending: false } : turn)),
          );
        } catch (err) {
          const message = extractErrorMessage(err, "Unable to answer the question right now.");
          setError(message);
          setHistory((prev) => prev.filter((turn) => turn.id !== id));
        }
      } finally {
        setPendingQuestion(null);
        setIsLoading(false);
      }
    },
    [question, settingsLoaded, streamAnswer],
  );

  const handleClear = () => {
    setQuestion("");
    setHistory([]);
    setPendingQuestion(null);
    setError(null);
    setSelectedTurnId(null);
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    }
  };

  const selectedTurn = useMemo(() => {
    if (history.length === 0) {
      return null;
    }
    if (selectedTurnId) {
      const match = history.find((turn) => turn.id === selectedTurnId);
      if (match) {
        return match;
      }
    }
    return history[history.length - 1];
  }, [history, selectedTurnId]);

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
          <div className="grid gap-6 lg:grid-cols-[minmax(0,240px)_1fr]">
            <aside className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-foreground">History</h3>
                <span className="text-xs text-muted-foreground">
                  {history.length} {history.length === 1 ? "question" : "questions"}
                </span>
              </div>
              <ul className="space-y-2">
                {history
                  .slice()
                  .reverse()
                  .map((turn) => {
                    const basePreview = displayAnswerText(turn.response);
                    const answerPreview = (turn.pending ? "Processing answer..." : basePreview).replace(/\s+/g, " ");
                    const snippet =
                      answerPreview.length > 140 ? `${answerPreview.slice(0, 137).trimEnd()}...` : answerPreview;
                    const activeId = selectedTurn?.id ?? null;
                    const isActive = activeId === turn.id;
                    const statusLabel = turn.pending ? "Processing" : MODE_LABELS[turn.response.mode];
                    return (
                      <li key={turn.id}>
                        <button
                          type="button"
                          onClick={() => setSelectedTurnId(turn.id)}
                          className={`w-full rounded-md border px-3 py-3 text-left text-sm transition ${
                            isActive
                              ? "border-primary/60 bg-primary/10 text-foreground"
                              : "border-border bg-background text-foreground hover:border-primary/40 hover:bg-primary/5"
                          }`}
                        >
                          <p className="font-semibold">{turn.question}</p>
                          <p className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                            {statusLabel} | {formatTimestamp(turn.createdAt)}{turn.classification?.intent ? ` | ${turn.classification.intent}` : ""}
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">{snippet}</p>
                        </button>
                      </li>
                    );
                  })}
              </ul>
            </aside>

            <div>
              {selectedTurn ? (
                <article className="space-y-4 rounded-lg border bg-card/70 p-5 shadow-sm">
                  <header className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">Question</p>
                      <p className="text-sm font-semibold text-foreground">{selectedTurn.question}</p>
                    </div>
                    <span className="text-xs text-muted-foreground">{formatTimestamp(selectedTurn.createdAt)}</span>
                  </header>

                  <div className="rounded-md border border-primary/30 bg-primary/5 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                      <span className="font-semibold text-foreground">
                        {selectedTurn.pending ? "Processing" : MODE_LABELS[selectedTurn.response.mode]}
                      </span>
                      <span>{selectedTurn.response.paths.length} reasoning paths</span>
                    </div>
                    {selectedTurn.classification?.intent ? (
                      <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                        Intent: {selectedTurn.classification.intent}
                      </p>
                    ) : null}
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                      {selectedTurn.pending
                        ? "Processing answer..."
                        : displayAnswerText(selectedTurn.response)}
                    </p>
                    {selectedTurn.pending ? (
                      <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Streaming graph evidence...
                      </div>
                    ) : selectedTurn.response.llm_answer ? (
                      <p className="mt-3 text-xs text-muted-foreground">
                        Graph summary: {selectedTurn.response.summary}
                      </p>
                    ) : null}
                  </div>

                  <details className="rounded-md border border-border/60 bg-background p-4">
                    <summary className="cursor-pointer text-sm font-semibold text-foreground">
                      Evidence &amp; supporting details
                    </summary>
                    <div className="mt-4 space-y-6 text-sm">
                      <section className="space-y-3">
                        <h4 className="text-sm font-semibold text-foreground">Resolved entities</h4>
                        {selectedTurn.response.resolved_entities.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No entities were resolved from the question.</p>
                        ) : (
                          <div className="grid gap-4 md:grid-cols-2">
                            {selectedTurn.response.resolved_entities.map((entity) => (
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
                        {selectedTurn.response.paths.length > 0 ? (
                          <PathList paths={selectedTurn.response.paths} />
                        ) : (
                          <p className="text-xs text-muted-foreground">No multi-hop paths discovered.</p>
                        )}
                      </section>

                      <section className="space-y-3">
                        <h4 className="text-sm font-semibold text-foreground">Fallback evidence</h4>
                        {selectedTurn.response.fallback_edges.length > 0 ? (
                          <EdgeList edges={selectedTurn.response.fallback_edges} title="Related findings" />
                        ) : (
                          <p className="text-xs text-muted-foreground">No fallback evidence returned.</p>
                        )}
                      </section>
                    </div>
                  </details>
                </article>
              ) : (
                <div className="rounded-lg border bg-background p-6 text-sm text-muted-foreground shadow-sm">
                  Select a question from the history to review its answer and supporting evidence.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default QaPanel;
