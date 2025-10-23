import axios, { AxiosError } from "axios";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

const sanitizeBaseUrl = (value: string): string => value.replace(/\/$/, "");

const resolveApiBaseUrl = (): string => {
  const raw = process.env.NEXT_PUBLIC_API_URL;
  if (typeof raw === "string" && raw.trim().length > 0) {
    const sanitized = sanitizeBaseUrl(raw.trim());
    if (sanitized.length > 0) {
      return sanitized;
    }
  }
  return sanitizeBaseUrl(DEFAULT_API_BASE_URL);
};

const API_BASE_URL = resolveApiBaseUrl();

export const apiClient = axios.create({
  baseURL: API_BASE_URL || undefined,
  timeout: 60000,
});

export const buildApiUrl = (path: string): string => {
  if (!path.startsWith("/")) {
    return `${API_BASE_URL}/${path}`;
  }
  return `${API_BASE_URL}${path}`;
};

export const extractErrorMessage = (error: unknown, fallback: string): string => {
  if (axios.isAxiosError(error)) {
    const detail = (error.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  return fallback;
};

export type HttpError = AxiosError;

export default apiClient;
