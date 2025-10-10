"use client";

import { type FormEventHandler, useCallback, useMemo, useState } from "react";
import { useDropzone, FileRejection } from "react-dropzone";
import axios from "axios";
import { toast } from "sonner";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Info,
  Loader2,
  UploadCloud,
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 800;

const defaultMetadata = {
  title: "",
  authors: "",
  venue: "",
  year: "",
};

type PaperResponse = {
  id: string;
  title: string;
  authors?: string | null;
  venue?: string | null;
  year?: number | null;
  status: string;
  file_name?: string | null;
  file_size?: number | null;
  created_at: string;
  updated_at: string;
};

const formatBytes = (bytes?: number | null) => {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, index);
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
};

const getErrorMessage = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data;
    if (data && typeof data === "object" && "detail" in data) {
      const detail = (data as { detail?: unknown }).detail;
      if (typeof detail === "string") {
        return detail;
      }
    }
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Unexpected error occurred";
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const createDefaultMetadata = () => ({ ...defaultMetadata });

export default function IngestionPage() {
  const [metadata, setMetadata] = useState(createDefaultMetadata);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastUploadedPaper, setLastUploadedPaper] = useState<PaperResponse | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (!acceptedFiles.length) return;
      const nextFile = acceptedFiles[0];
      setFile(nextFile);
      setLastUploadedPaper(null);
      setProgress(0);

      setMetadata((prev) => ({
        ...prev,
        title: prev.title || nextFile.name.replace(/\.pdf$/i, ""),
      }));
    },
    []
  );

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
    setMetadata(createDefaultMetadata());
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

    const buildFormData = () => {
      const data = new FormData();
      data.append("file", file);
      if (metadata.title.trim()) data.append("title", metadata.title.trim());
      if (metadata.authors.trim()) data.append("authors", metadata.authors.trim());
      if (metadata.venue.trim()) data.append("venue", metadata.venue.trim());
      if (metadata.year.trim()) data.append("year", metadata.year.trim());
      return data;
    };

    let attempt = 0;

    try {
      for (attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
          const response = await axios.post<PaperResponse>(
            `${API_BASE_URL}/api/papers/upload`,
            buildFormData(),
            {
              headers: { "Content-Type": "multipart/form-data" },
              onUploadProgress: (event) => {
                if (!event.total) {
                  setProgress((prev) => (prev < 90 ? prev + 5 : prev));
                  return;
                }
                const percentage = Math.round((event.loaded / event.total) * 100);
                setProgress(percentage);
              },
            }
          );

          setProgress(100);
          setLastUploadedPaper(response.data);
          toast.success("Upload successful", {
            description: `${response.data.title} queued for parsing.`,
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
      const message = getErrorMessage(error);
      toast.error("Upload failed", { description: message });
    } finally {
      setIsUploading(false);
      setProgress((prev) => (prev === 100 ? 100 : 0));
    }
  }, [file, metadata, resetForm]);

  const handleSubmit: FormEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();
    if (!isUploading) {
      void handleUpload();
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Ingestion</p>
        <h1 className="mt-1 text-2xl font-semibold text-foreground">Upload research papers</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Drop PDFs or browse to add them to the parsing pipeline. Provide metadata to improve searchability and downstream
          entity linking.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <form onSubmit={handleSubmit} className="space-y-6 rounded-lg border bg-card p-6 shadow-sm">
          <div {...getRootProps()} className={`fade-in-up rounded-lg border-2 border-dashed p-8 transition-all duration-300 ${
            isDragActive ? "border-primary bg-primary/5" : "border-border bg-muted/40"
          }`}
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
              <span className="font-medium text-foreground">Title</span>
              <input
                type="text"
                value={metadata.title}
                onChange={(event) => setMetadata((prev) => ({ ...prev, title: event.target.value }))}
                placeholder="e.g. Neural Graph Discovery in Scientific Corpora"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
            </label>
            <label className="space-y-2 text-sm">
              <span className="font-medium text-foreground">Authors</span>
              <input
                type="text"
                value={metadata.authors}
                onChange={(event) => setMetadata((prev) => ({ ...prev, authors: event.target.value }))}
                placeholder="Separate authors with commas"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
            </label>
            <label className="space-y-2 text-sm">
              <span className="font-medium text-foreground">Venue</span>
              <input
                type="text"
                value={metadata.venue}
                onChange={(event) => setMetadata((prev) => ({ ...prev, venue: event.target.value }))}
                placeholder="Conference or journal"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
            </label>
            <label className="space-y-2 text-sm">
              <span className="font-medium text-foreground">Year</span>
              <input
                type="number"
                inputMode="numeric"
                min={1800}
                max={2100}
                value={metadata.year}
                onChange={(event) => setMetadata((prev) => ({ ...prev, year: event.target.value }))}
                placeholder="2024"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
              />
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
              className={`inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${
                canSubmit ? "hover:bg-primary/90" : "cursor-not-allowed opacity-60"
              }`}
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
                  <li>Populate metadata to improve downstream entity linking.</li>
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
                    <p className="font-medium text-foreground">{lastUploadedPaper.title}</p>
                    {lastUploadedPaper.authors ? (
                      <p>{lastUploadedPaper.authors}</p>
                    ) : (
                      <p className="italic">No authors provided</p>
                    )}
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">
                      {lastUploadedPaper.venue ? `${lastUploadedPaper.venue} · ` : ""}
                      {lastUploadedPaper.year ?? "Year not set"}
                    </div>
                    {lastUploadedPaper.file_name && (
                      <p className="text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">File:</span> {lastUploadedPaper.file_name}
                        {lastUploadedPaper.file_size ? ` · ${formatBytes(lastUploadedPaper.file_size)}` : ""}
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      Status: <span className="font-medium text-foreground">{lastUploadedPaper.status}</span>
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