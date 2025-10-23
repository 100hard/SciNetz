"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  ExternalLink,
  Loader2,
  RefreshCw,
  SlidersHorizontal,
} from "lucide-react";

import apiClient, { extractErrorMessage } from "@/lib/http";

const AUTO_FETCH_STORAGE_KEY = "graphAutoFetchEnabled";

type GraphDefaults = {
  relations: string[];
  min_confidence: number;
  sections: string[];
  show_co_mentions: boolean;
  layout: string;
};

type UiSettingsResponse = {
  graph_defaults: GraphDefaults;
};

type LoadingMode = "initial" | "refresh";

const SettingsPage = () => {
  const [settings, setSettings] = useState<UiSettingsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [autoFetchEnabled, setAutoFetchEnabled] = useState<boolean | null>(null);

  const loadUiSettings = useCallback(async (mode: LoadingMode = "refresh") => {
    if (mode === "initial") {
      setIsLoading(true);
      setError(null);
    } else {
      setIsRefreshing(true);
    }

    try {
      const { data } = await apiClient.get<UiSettingsResponse>("/api/ui/settings");
      setSettings(data);
      setError(null);
    } catch (err) {
      const message = extractErrorMessage(err, "Unable to load UI settings.");
      setError(message);
    } finally {
      if (mode === "initial") {
        setIsLoading(false);
      } else {
        setIsRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    void loadUiSettings("initial");
  }, [loadUiSettings]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.sessionStorage.getItem(AUTO_FETCH_STORAGE_KEY);
    if (stored === "true") {
      setAutoFetchEnabled(true);
      return;
    }
    if (stored === "false") {
      setAutoFetchEnabled(false);
      return;
    }
    setAutoFetchEnabled(true);
  }, []);

  const updateAutoFetchPreference = useCallback((value: boolean) => {
    setAutoFetchEnabled(value);
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(AUTO_FETCH_STORAGE_KEY, value ? "true" : "false");
    }
  }, []);

  const handleToggleAutoFetch = useCallback(() => {
    const resolved = autoFetchEnabled ?? true;
    updateAutoFetchPreference(!resolved);
  }, [autoFetchEnabled, updateAutoFetchPreference]);

  const handleResetPreferences = useCallback(() => {
    if (typeof window !== "undefined") {
      window.sessionStorage.removeItem(AUTO_FETCH_STORAGE_KEY);
    }
    setAutoFetchEnabled(true);
  }, []);

  const handleRefresh = useCallback(() => {
    void loadUiSettings("refresh");
  }, [loadUiSettings]);

  const graphDefaults = settings?.graph_defaults;

  const relationPreview = useMemo(() => {
    if (!graphDefaults) {
      return [];
    }
    return graphDefaults.relations.slice(0, 10);
  }, [graphDefaults]);

  const hiddenRelationCount = useMemo(() => {
    if (!graphDefaults) {
      return 0;
    }
    return Math.max(graphDefaults.relations.length - relationPreview.length, 0);
  }, [graphDefaults, relationPreview.length]);

  const resolvedAutoFetch = autoFetchEnabled ?? true;
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Settings</p>
        <h1 className="text-2xl font-semibold text-foreground">Configure SciNets</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Review application defaults, manage graph exploration preferences, and understand which features are available in your environment.
        </p>
      </div>

      {error ? (
        <div className="flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-semibold text-foreground">Unable to fetch current settings.</p>
            <p className="mt-1 text-xs text-rose-700">{error}</p>
            <button
              type="button"
              onClick={handleRefresh}
              className="mt-3 inline-flex items-center gap-2 rounded-md border border-rose-300 bg-rose-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-rose-500/40 focus:ring-offset-2"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Retry
            </button>
          </div>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-lg border bg-card p-6 shadow-sm">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Graph explorer defaults</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                These values come directly from the backend configuration and are used whenever the graph view loads.
              </p>
            </div>
            <button
              type="button"
              onClick={handleRefresh}
              className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-1.5 text-xs font-medium text-foreground shadow-sm transition hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isLoading || isRefreshing}
            >
              {isRefreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Refresh
            </button>
          </div>

          {isLoading ? (
            <div className="mt-6 space-y-3">
              <div className="h-4 w-2/3 rounded bg-muted" />
              <div className="grid gap-3 md:grid-cols-2">
                <div className="h-16 rounded-md border border-dashed border-muted" />
                <div className="h-16 rounded-md border border-dashed border-muted" />
              </div>
              <div className="h-10 rounded-md border border-dashed border-muted" />
            </div>
          ) : graphDefaults ? (
            <div className="mt-6 space-y-4 text-sm text-muted-foreground">
              <dl className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-md border border-border bg-background/40 p-4">
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground/80">Minimum confidence</dt>
                  <dd className="mt-1 text-lg font-semibold text-foreground">
                    {graphDefaults.min_confidence.toFixed(2)}
                  </dd>
                </div>
                <div className="rounded-md border border-border bg-background/40 p-4">
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground/80">Sections included</dt>
                  <dd className="mt-1 text-sm text-foreground">{graphDefaults.sections.join(" · ") || "All sections"}</dd>
                </div>
                <div className="rounded-md border border-border bg-background/40 p-4">
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground/80">Co-mention edges</dt>
                  <dd className="mt-1 text-sm text-foreground">
                    {graphDefaults.show_co_mentions ? "Visible by default" : "Hidden by default"}
                  </dd>
                </div>
                <div className="rounded-md border border-border bg-background/40 p-4">
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground/80">Layout</dt>
                  <dd className="mt-1 text-sm text-foreground uppercase">{graphDefaults.layout || "Auto"}</dd>
                </div>
              </dl>
              <div>
                <p className="text-xs uppercase tracking-wide text-muted-foreground/80">Default relations</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {relationPreview.map((relation) => (
                    <span
                      key={relation}
                      className="rounded-full border border-border bg-background px-3 py-1 text-xs font-medium capitalize text-foreground"
                    >
                      {relation.replace(/_/g, " ")}
                    </span>
                  ))}
                  {hiddenRelationCount > 0 ? (
                    <span className="rounded-full border border-dashed border-border px-3 py-1 text-xs font-medium text-muted-foreground">
                      +{hiddenRelationCount} more
                    </span>
                  ) : null}
                </div>
              </div>
              <Link
                href="/graph"
                className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-xs font-medium text-foreground transition hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2"
              >
                Launch graph explorer
                <ExternalLink className="h-3.5 w-3.5" />
              </Link>
            </div>
          ) : (
            <p className="mt-6 text-sm text-muted-foreground">Graph defaults are not available yet.</p>
          )}
        </section>

        <section className="rounded-lg border bg-card p-6 shadow-sm lg:col-span-2">
          <div className="flex items-start gap-3">
            <SlidersHorizontal className="h-5 w-5 text-primary" />
            <div>
              <h2 className="text-lg font-semibold text-foreground">Workspace preferences</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Set client-side preferences that tailor how the explorer behaves for your browser. These settings are stored locally and can be reset at any time.
              </p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-md border border-border bg-background/40 p-4">
              <p className="text-sm font-medium text-foreground">Auto-fetch graph data</p>
              <p className="mt-1 text-xs text-muted-foreground">
                When enabled, the graph view automatically loads data as soon as defaults are available. Disable to control when queries run.
              </p>
              <button
                type="button"
                onClick={handleToggleAutoFetch}
                className={`mt-3 inline-flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 ${
                  resolvedAutoFetch
                    ? "border border-primary bg-primary text-primary-foreground hover:bg-primary/90"
                    : "border border-input bg-background text-foreground hover:bg-muted"
                }`}
              >
                {resolvedAutoFetch ? "Auto-fetch enabled" : "Enable auto-fetch"}
              </button>
            </div>
            <div className="rounded-md border border-border bg-background/40 p-4">
              <p className="text-sm font-medium text-foreground">Reset local preferences</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Clears graph auto-fetch choices and reverts to project defaults. Useful if behaviour seems out of sync with configuration.
              </p>
              <button
                type="button"
                onClick={handleResetPreferences}
                className="mt-3 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-xs font-medium text-foreground transition hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2"
              >
                Reset preferences
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default SettingsPage;
