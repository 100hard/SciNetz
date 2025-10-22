"use client";

import { type FormEventHandler, useCallback, useMemo, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { toast } from "sonner";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Info,
  Loader2,
  UploadCloud,
} from "lucide-react";

import apiClient, { extractErrorMessage } from "../../lib/http";

const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 800;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const readFileAsBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Unable to read file as base64"));
        return;
      }
      const [, base64 = ""] = result.split(",");
      resolve(base64);
    };
    reader.onerror = () => {
      reject(reader.error ?? new Error("Unable to read file"));
    };
    reader.readAsDataURL(file);
  });

const formatDate = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "—";
  }
  return parsed.toLocaleString();
};

const formatMetadata = (metadata?: PaperMetadata | null) => {
  if (!metadata) {
    return null;
  }
  const parts: string[] = [];
  if (metadata.title) {
    parts.push(metadata.title);
  }
  if (metadata.authors && metadata.authors.length > 0) {
    parts.push(metadata.authors.join(", "));
  }
  const venuePieces: string[] = [];
  if (metadata.venue) {
    venuePieces.push(metadata.venue);
  }
  if (metadata.year) {
    venuePieces.push(String(metadata.year));
  }
  if (venuePieces.length > 0) {
    parts.push(venuePieces.join(" · "));
  }
  if (metadata.doi) {
    parts.push(metadata.doi);
  }
  return parts.join("\n");
};

const toUploadPayload = async (
  file: File,
  filenameOverride: string,
  paperId?: string,
): Promise<UploadPaperRequest> => {
  const content = await readFileAsBase64(file);
  const payload: UploadPaperRequest = {
    filename: filenameOverride || file.name,
    content_base64: content,
  };
  const trimmedId = paperId?.trim();
  if (trimmedId) {
    payload.paper_id = trimmedId;
  }
  return payload;
};

const createInitialFormState = () => ({
  filename: "",
  paperId: "",
});

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

type UploadPaperRequest = {
  filename: string;
  content_base64: string;
  paper_id?: string;
};

export default function IngestionPage() {
  const [formState, setFormState] = useState(createInitialFormState);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastUploadedPaper, setLastUploadedPaper] = useState<PaperSummary | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return;
    const nextFile = acceptedFiles[0];
    setFile(nextFile);
    setLastUploadedPaper(null);
    setProgress(0);
    setFormState((prev) => ({
      ...prev,
      filename: prev.filename || nextFile.name.replace(/\.pdf$/i, ""),
    }));
  }, []);

  const onDropRejected = useCallback((rejections: FileRejection[]) => {
    rejections.forEach((rejection) => {
      const { file: rejectedFile, errors } = rejection;
      errors.forEach((err) => {
        if (err.code === "file-invalid-type") {
          toast.error(`${rejectedFile.name} is not a PDF.`);
        } else if (err.code === "file-too-large") {
          toast.error(`${rejectedFile.name} exceeds the allowed size.`);
        } else {
          toast.error(err.message);
        }
      });
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    maxFiles: 1,
  });

  const derivedFileName = useMemo(() => file?.name ?? "No file selected", [file]);

  const canSubmit = useMemo(() => Boolean(file) && !isUploading, [file, isUploading]);

  const resetForm = useCallback(() => {
    setFormState(createInitialFormState());
    setFile(null);
    setProgress(0);
  }, []);

  const handleUpload = useCallback(async () => {
    if (!file) {
      toast.error("Please select a PDF before uploading.");
      return;
    }

    setIsUploading(true);
    setProgress(0);

    const buildPayload = () => toUploadPayload(file, formState.filename.trim() || file.name, formState.paperId);

    let attempt = 0;

    try {
      for (attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
          const payload = await buildPayload();
          const response = await apiClient.post<PaperSummary>("/api/ui/upload", payload, {
            onUploadProgress: (event) => {
              if (!event.total) {
                setProgress((prev) => (prev < 90 ? prev + 5 : prev));
                return;
              }
              const percentage = Math.round((event.loaded / event.total) * 100);
              setProgress(percentage);
            },
          });

          setProgress(100);
          setLastUploadedPaper(response.data);
          toast.success("Upload successful", {
            description: `${response.data.filename} queued for processing.`,
          });
          resetForm();
          return;
        } catch (error) {
          if (attempt < MAX_RETRIES) {
            const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1);
            toast.warning(`Upload failed (attempt ${attempt}). Retrying in ${Math.round(delay / 1000)}s...`);
            await sleep(delay);
            setProgress(0);
            continue;
          }
          throw error;
        }
      }
    } catch (error) {
      const message = extractErrorMessage(error, "Unexpected error during upload.");
      toast.error("Upload failed", { description: message });
    } finally {
      setIsUploading(false);
      setProgress((prev) => (prev === 100 ? 100 : 0));
    }
  }, [file, formState.filename, formState.paperId, resetForm]);

  const handleSubmit: FormEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();
    if (!isUploading) {
      void handleUpload();
    }
  };

  const lastUploadMetadata = formatMetadata(lastUploadedPaper?.metadata ?? undefined);

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Ingestion</p>
        <h1 className="mt-1 text-2xl font-semibold text-foreground">Upload research papers</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Drop PDFs or browse to add them to the parsing pipeline. Provide an optional identifier to keep uploads aligned with downstream processing.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <form onSubmit={handleSubmit} className="space-y-6 rounded-lg border bg-card p-6 shadow-sm">
          <div
            {...getRootProps()}
            className={`fade-in-up rounded-lg border-2 border-dashed p-8 transition-all duration-300 ${isDragActive ? "border-primary bg-primary/5" : "border-border bg-muted/40"}`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center text-center">
              <UploadCloud className="h-12 w-12 text-primary" />
              <p className="mt-4 text-sm font-medium text-foreground">
                {isDragActive ? "Drop the PDF to upload" : "Drag & drop your PDF here"}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">Only .pdf files are supported for this release.</p>
              <span className="mt-4 inline-flex items-center gap-2 rounded-md border border-primary bg-primary/10 px-4 py-2 text-xs font-medium uppercase tracking-wide text-primary">
                <FileText className="h-4 w-4" />
                {derivedFileName}
              </span>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-2 text-sm">
              <span className="font-medium text-foreground">Filename override</span>
              <input
                type="text"
                value={formState.filename}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    filename: event.target.value,
                  }))
                }
                placeholder="Optional friendly name"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
              <p className="text-xs text-muted-foreground">Defaults to the original filename when left blank.</p>
            </label>
            <label className="space-y-2 text-sm">
              <span className="font-medium text-foreground">Custom paper ID</span>
              <input
                type="text"
                value={formState.paperId}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    paperId: event.target.value,
                  }))
                }
                placeholder="Optional identifier"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
              <p className="text-xs text-muted-foreground">Leave blank to auto-generate from the filename.</p>
            </label>
          </div>

          {isUploading && (
            <div className="fade-in space-y-2">
              <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
                <span className="inline-flex items-center gap-2 text-foreground">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" /> Uploading
                </span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 rounded-full bg-muted">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{ width: `${Math.min(progress, 100)}%` }}
                  aria-valuenow={progress}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={!canSubmit}
              className={`inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${canSubmit ? "hover:bg-primary/90" : "cursor-not-allowed opacity-60"}`}
            >
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Uploading...
                </>
              ) : (
                <>
                  <UploadCloud className="h-4 w-4" /> Queue upload
                </>
              )}
            </button>
            <button
              type="button"
              onClick={resetForm}
              disabled={isUploading}
              className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground shadow-sm transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
            >
              Reset
            </button>
          </div>
        </form>

        <div className="space-y-6">
          <div className="fade-in-up rounded-lg border bg-card p-6 shadow-sm">
            <div className="flex items-start gap-3">
              <Info className="mt-1 h-5 w-5 text-primary" />
              <div className="space-y-2 text-sm text-muted-foreground">
                <p className="text-sm font-semibold text-foreground">Upload guidelines</p>
                <ul className="list-inside list-disc space-y-1">
                  <li>PDFs up to 50MB are supported for this milestone.</li>
                  <li>Use custom IDs to align uploads with pipeline orchestration.</li>
                  <li>Uploads trigger parsing automatically once the file is stored.</li>
                </ul>
              </div>
            </div>
          </div>

          {lastUploadedPaper ? (
            <div className="fade-in-up rounded-lg border bg-card p-6 shadow-sm transition-all duration-300">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="mt-1 h-5 w-5 text-emerald-500" />
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p className="text-sm font-semibold text-foreground">Last upload</p>
                  <div className="space-y-1">
                    <p className="font-medium text-foreground">{lastUploadedPaper.filename}</p>
                    <p className="text-xs text-muted-foreground">Paper ID: {lastUploadedPaper.paper_id}</p>
                    <p className="text-xs text-muted-foreground">
                      Status: <span className="font-medium text-foreground">{lastUploadedPaper.status}</span>
                    </p>
                    <p className="text-xs text-muted-foreground">Uploaded {formatDate(lastUploadedPaper.uploaded_at)}</p>
                    <p className="text-xs text-muted-foreground">Updated {formatDate(lastUploadedPaper.updated_at)}</p>
                    {lastUploadMetadata ? (
                      <pre className="whitespace-pre-wrap text-xs text-muted-foreground">{lastUploadMetadata}</pre>
                    ) : (
                      <p className="text-xs italic text-muted-foreground">Metadata pending extraction</p>
                    )}
                    {lastUploadedPaper.errors.length > 0 ? (
                      <div className="rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
                        <p className="font-semibold text-foreground">Pipeline warnings</p>
                        <ul className="list-inside list-disc space-y-1">
                          {lastUploadedPaper.errors.map((error) => (
                            <li key={error}>{error}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    <p className="text-xs text-muted-foreground">
                      Nodes written: <span className="font-medium text-foreground">{lastUploadedPaper.nodes_written}</span>
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Edges written: <span className="font-medium text-foreground">{lastUploadedPaper.edges_written}</span>
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Co-mention edges: <span className="font-medium text-foreground">{lastUploadedPaper.co_mention_edges}</span>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="fade-in-up rounded-lg border bg-card p-6 shadow-sm text-sm text-muted-foreground">
              <div className="flex items-start gap-3">
                <AlertCircle className="mt-1 h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="font-semibold text-foreground">No uploads yet</p>
                  <p>Select a PDF and submit the form to see progress here.</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
