import { GraphResponse, PaperSummary, QAResponsePayload, UISettings } from "./types";

function buildUrl(path: string): string {
  const base = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");
  if (base && base.length > 0) {
    return `${base}${path}`;
  }
  return path;
}

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
    const contentType = response.headers.get("content-type");
    const bodyText = await response.text();
    if (contentType?.includes("application/json") && bodyText) {
      try {
        const payload = JSON.parse(bodyText) as unknown;
        const detail =
          typeof payload === "string"
            ? payload
            : typeof (payload as { detail?: unknown })?.detail === "string"
              ? ((payload as { detail?: string }).detail as string)
              : undefined;
        if (detail) {
          throw new Error(detail);
        }
      } catch (error) {
        if (error instanceof Error && !(error instanceof SyntaxError)) {
          throw error;
        }
      }
    }
    const sanitized = bodyText.startsWith("<!DOCTYPE") ? undefined : bodyText.trim();
    throw new Error(sanitized || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchSettings(): Promise<UISettings> {
  const response = await fetch(buildUrl("/api/ui/settings"));
  return handleResponse<UISettings>(response);
}

export async function listPapers(): Promise<PaperSummary[]> {
  const response = await fetch(buildUrl("/api/ui/papers"));
  return handleResponse<PaperSummary[]>(response);
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Unexpected file reader result"));
        return;
      }
      const commaIndex = result.indexOf(",");
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

export async function uploadPaper(file: File, paperId?: string): Promise<PaperSummary> {
  const contentBase64 = await fileToBase64(file);
  const payload: Record<string, unknown> = {
    filename: file.name,
    content_base64: contentBase64
  };
  if (paperId) {
    payload.paper_id = paperId;
  }
  const response = await fetch(buildUrl("/api/ui/upload"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return handleResponse<PaperSummary>(response);
}

export async function extractPaper(paperId: string): Promise<Record<string, unknown>> {
  const response = await fetch(buildUrl(`/api/ui/papers/${encodeURIComponent(paperId)}/extract`), {
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
  const response = await fetch(buildUrl(`/api/ui/graph?${params.toString()}`));
  return handleResponse<GraphResponse>(response);
}

export async function askQuestion(question: string): Promise<QAResponsePayload> {
  const response = await fetch(buildUrl("/api/qa/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  return handleResponse<QAResponsePayload>(response);
}
