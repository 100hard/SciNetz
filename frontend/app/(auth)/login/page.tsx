"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";

import { useAuth } from "../../../components/auth-provider";
import apiClient, { extractErrorMessage } from "@/lib/http";

type GoogleCredentialResponse = {
  credential?: string;
};

type GoogleConfigResponse = {
  client_ids: string[];
};

declare global {
  interface Window {
    google?: {
      accounts?: {
        id?: {
          initialize: (config: {
            client_id: string;
            callback: (response: GoogleCredentialResponse) => void;
          }) => void;
          renderButton: (element: HTMLElement, options: Record<string, unknown>) => void;
          prompt: () => void;
        };
      };
    };
    NEXT_PUBLIC_GOOGLE_CLIENT_ID?: string;
  }
}

const LoginPage = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { status, loginWithGoogle } = useAuth();
  const [isSubmitting, setSubmitting] = useState(false);
  const buttonContainerRef = useRef<HTMLDivElement | null>(null);
  const scriptInitializedRef = useRef(false);
  const scriptLoadingRef = useRef(false);

  // ✅ Resolve Google Client ID (supports Docker + Local)
  const initialClientId = useMemo(() => {
    let raw = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

    if (typeof window !== "undefined") {
      const runtimeId = window.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
      if ((!raw || raw === "undefined") && runtimeId) raw = runtimeId;

      if ((!raw || raw === "undefined") && !runtimeId) {
        const meta = document.querySelector('meta[name="google-client-id"]');
        if (meta) raw = meta.getAttribute("content") ?? "";
      }
    }

    console.log("✅ Resolved Google Client ID:", raw);
    return typeof raw === "string" ? raw.trim() : "";
  }, []);

  const [googleClientId, setGoogleClientId] = useState(initialClientId);
  const [configState, setConfigState] = useState<"loading" | "ready" | "error">(
    initialClientId ? "ready" : "loading"
  );
  const [configError, setConfigError] = useState<string | null>(null);
  const [isGoogleButtonRendered, setGoogleButtonRendered] = useState(false);

  const rawNextParam = searchParams?.get("next") ?? "/";
  const nextParam = rawNextParam.startsWith("/") ? rawNextParam : "/";

  // 1️⃣ Redirect if already authenticated
  useEffect(() => {
    if (status === "authenticated") {
      router.replace(nextParam || "/");
    }
  }, [nextParam, router, status]);

  // 2️⃣ Load or resolve Google Client ID
  useEffect(() => {
    let isMounted = true;

    if (initialClientId) {
      setGoogleClientId(initialClientId);
      setConfigError(null);
      setConfigState("ready");
      return () => {
        isMounted = false;
      };
    }

    const loadGoogleConfig = async () => {
      setConfigState("loading");
      try {
        const { data } = await apiClient.get<GoogleConfigResponse>("/api/auth/google/config");
        if (!isMounted) return;
        const resolvedId =
          data.client_ids.map((id) => id.trim()).find((id) => id.length > 0) ?? "";
        if (resolvedId) {
          setGoogleClientId(resolvedId);
          setConfigError(null);
          setConfigState("ready");
        } else {
          setGoogleClientId("");
          setConfigError(
            "Google Sign-In is not configured. Add a client ID to auth.google.client_ids or set NEXT_PUBLIC_GOOGLE_CLIENT_ID."
          );
          setConfigState("error");
        }
      } catch (error) {
        if (!isMounted) return;
        setGoogleClientId("");
        setConfigError(
          extractErrorMessage(error, "Unable to load Google Sign-In configuration.")
        );
        setConfigState("error");
      }
    };

    void loadGoogleConfig();

    return () => {
      isMounted = false;
    };
  }, [initialClientId]);

  // 3️⃣ Handle Google credential response
  const handleCredential = useCallback(
    (response: GoogleCredentialResponse) => {
      const credential = response.credential;
      if (!credential) {
        toast.error("Google authentication did not return a credential. Please try again.");
        return;
      }
      setSubmitting(true);
      void (async () => {
        try {
          const result = await loginWithGoogle(credential);
          toast.success(result.message);
          router.replace(nextParam || "/");
        } catch (error) {
          toast.error(extractErrorMessage(error, "Unable to sign in with Google."));
        } finally {
          setSubmitting(false);
        }
      })();
    },
    [loginWithGoogle, nextParam, router]
  );

  // 4️⃣ Google Sign-In button rendering and script initialization
  useEffect(() => {
    if (typeof window === "undefined") return;

    if (!googleClientId) {
      console.warn("⏳ Waiting for Google Client ID to become available...");
      if (buttonContainerRef.current) buttonContainerRef.current.innerHTML = "";
      scriptInitializedRef.current = false;
      setGoogleButtonRendered(false);
      return;
    }

    const initializeButton = () => {
      if (scriptInitializedRef.current) return;

      const googleAccounts = window.google?.accounts?.id;
      if (!googleAccounts || !buttonContainerRef.current) {
        console.warn("⚠️ Google accounts object not yet available, retrying...");
        setTimeout(initializeButton, 300);
        return;
      }

      try {
        console.log("🚀 Initializing Google Sign-In with ID:", googleClientId);
        buttonContainerRef.current.innerHTML = "";
        googleAccounts.initialize({
          client_id: googleClientId,
          callback: handleCredential,
        });
        googleAccounts.renderButton(buttonContainerRef.current, {
          type: "standard",
          theme: "outline",
          size: "large",
          text: "continue_with",
          shape: "pill",
        });
        googleAccounts.prompt();
        setGoogleButtonRendered(true);
        scriptInitializedRef.current = true;
      } catch (error) {
        console.error("❌ Unable to render Google Sign-In button:", error);
        setGoogleButtonRendered(false);
        toast.error("Unable to initialize Google Sign-In. Please try again.");
      }
    };

    const ensureScript = () => {
      if (window.google?.accounts?.id) {
        initializeButton();
        return;
      }
      if (scriptLoadingRef.current) return;

      const script = document.createElement("script");
      script.src = "https://accounts.google.com/gsi/client";
      script.async = true;
      script.defer = true;
      script.onload = initializeButton;
      script.onerror = () => toast.error("Failed to load Google authentication script.");
      document.head.appendChild(script);
      scriptLoadingRef.current = true;
    };

    if (document.readyState === "complete") {
      ensureScript();
    } else {
      window.addEventListener("load", ensureScript);
      return () => window.removeEventListener("load", ensureScript);
    }
  }, [googleClientId, handleCredential]);

  // Fallback manual click handler
  const isConfigReady = configState === "ready" && Boolean(googleClientId);
  const isConfigLoading = configState === "loading";
  const isDisabled = isSubmitting || status === "loading" || !isConfigReady;

  const handleFallbackClick = useCallback(() => {
    if (isDisabled) return;

    if (typeof window === "undefined") return;

    const googleAccounts = window.google?.accounts?.id;
    if (!googleAccounts) {
      toast.error("Google authentication is still loading. Please try again in a moment.");
      return;
    }

    try {
      googleAccounts.prompt();
    } catch (error) {
      console.error("Unable to trigger Google Sign-In prompt", error);
      toast.error("Unable to open Google Sign-In. Please refresh and try again.");
    }
  }, [isDisabled]);

  return (
    <div className="w-full max-w-md space-y-6">
      <div className="space-y-2 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-primary">SciNets</p>
        <h1 className="text-2xl font-semibold text-foreground">Welcome back</h1>
        <p className="text-sm text-muted-foreground">
          Sign in with your Google account to access the dashboard.
        </p>
      </div>
      <div className="space-y-6 rounded-lg border bg-card p-8 shadow-xl">
        {isConfigReady ? (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Continue with Google to securely authenticate without a password.
            </p>
            <div className={`flex justify-center ${isDisabled ? "pointer-events-none opacity-75" : ""}`}>
              <div
                ref={buttonContainerRef}
                className={isGoogleButtonRendered ? "" : "hidden"}
                aria-hidden={!isGoogleButtonRendered}
              />
              {!isGoogleButtonRendered && (
                <button
                  type="button"
                  onClick={handleFallbackClick}
                  disabled={isDisabled}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-input bg-background px-5 py-2 text-sm font-medium text-foreground shadow-sm transition hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  Continue with Google
                </button>
              )}
            </div>
            {isSubmitting && <p className="text-xs text-muted-foreground">Signing you in…</p>}
          </div>
        ) : (
          <div className="space-y-2 text-center">
            {isConfigLoading ? (
              <p className="text-sm text-muted-foreground">Loading Google Sign-In configuration…</p>
            ) : (
              <p className="text-sm text-muted-foreground">
                {configError ??
                  "Google Sign-In is not configured. Add a client ID to auth.google.client_ids or set NEXT_PUBLIC_GOOGLE_CLIENT_ID."}
              </p>
            )}
          </div>
        )}
        <p className="text-center text-sm text-muted-foreground">
          Account creation happens automatically the first time you continue with Google.
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
