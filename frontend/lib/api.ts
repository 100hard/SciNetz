import { GraphResponse, PaperSummary, QAResponsePayload, UISettings } from "./types";

export interface GraphFilters {
  relations: string[];
  min_confidence: number;
  sections: string[];
  include_co_mentions: boolean;
  papers: string[];
  limit?: number;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchSettings(): Promise<UISettings> {
  const response = await fetch("/api/ui/settings");
  return handleResponse<UISettings>(response);
}

export async function listPapers(): Promise<PaperSummary[]> {
  const response = await fetch("/api/ui/papers");
  return handleResponse<PaperSummary[]>(response);
}

export async function uploadPaper(file: File, paperId?: string): Promise<PaperSummary> {
  const formData = new FormData();
  formData.append("file", file);
  if (paperId) {
    formData.append("paper_id", paperId);
  }
  const response = await fetch("/api/ui/upload", {
    method: "POST",
    body: formData
  });
  return handleResponse<PaperSummary>(response);
}

export async function extractPaper(paperId: string): Promise<Record<string, unknown>> {
  const response = await fetch(`/api/ui/papers/${encodeURIComponent(paperId)}/extract`, {
    method: "POST"
  });
  return handleResponse<Record<string, unknown>>(response);
}

export async function fetchGraph(filters: GraphFilters): Promise<GraphResponse> {
  const params = new URLSearchParams();
  if (filters.relations.length) {
    params.set("relations", filters.relations.join(","));
  }
  if (filters.sections.length) {
    params.set("sections", filters.sections.join(","));
  }
  if (filters.papers.length) {
    params.set("papers", filters.papers.join(","));
  }
  params.set("min_confidence", filters.min_confidence.toString());
  params.set("include_co_mentions", String(filters.include_co_mentions));
  if (filters.limit !== undefined) {
    params.set("limit", filters.limit.toString());
  }
  const response = await fetch(`/api/ui/graph?${params.toString()}`);
  return handleResponse<GraphResponse>(response);
}

export async function askQuestion(question: string): Promise<QAResponsePayload> {
  const response = await fetch("/api/qa/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  return handleResponse<QAResponsePayload>(response);
}
