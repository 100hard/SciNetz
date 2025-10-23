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
          initialize: (config: { client_id: string; callback: (response: GoogleCredentialResponse) => void }) => void;
          renderButton: (element: HTMLElement, options: Record<string, unknown>) => void;
          prompt: () => void;
        };
      };
    };
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
  const initialClientId = useMemo(() => {
    const raw = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
    return typeof raw === "string" ? raw.trim() : "";
  }, []);
  const [googleClientId, setGoogleClientId] = useState(initialClientId);
  const [configState, setConfigState] = useState<"loading" | "ready" | "error">(
    initialClientId ? "ready" : "loading",
  );
  const [configError, setConfigError] = useState<string | null>(null);

  const rawNextParam = searchParams?.get("next") ?? "/";
  const nextParam = rawNextParam.startsWith("/") ? rawNextParam : "/";

  useEffect(() => {
    if (status === "authenticated") {
      router.replace(nextParam || "/");
    }
  }, [nextParam, router, status]);

  useEffect(() => {
    if (initialClientId) {
      return;
    }

    let isMounted = true;

    const loadGoogleConfig = async () => {
      setConfigState("loading");
      try {
        const { data } = await apiClient.get<GoogleConfigResponse>("/api/auth/google/config");
        if (!isMounted) {
          return;
        }
        const resolvedId = data.client_ids.map((id) => id.trim()).find((id) => id.length > 0) ?? "";
        if (resolvedId) {
          setGoogleClientId(resolvedId);
          setConfigError(null);
          setConfigState("ready");
        } else {
          setGoogleClientId("");
          setConfigError(
            "Google Sign-In is not configured. Add a client ID to auth.google.client_ids or set NEXT_PUBLIC_GOOGLE_CLIENT_ID.",
          );
          setConfigState("error");
        }
      } catch (error) {
        if (!isMounted) {
          return;
        }
        setGoogleClientId("");
        setConfigError(extractErrorMessage(error, "Unable to load Google Sign-In configuration."));
        setConfigState("error");
      }
    };

    void loadGoogleConfig();

    return () => {
      isMounted = false;
    };
  }, [initialClientId]);

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
    [loginWithGoogle, nextParam, router],
  );

  useEffect(() => {
    if (!googleClientId || typeof window === "undefined") {
      return;
    }

    const initializeButton = () => {
      if (scriptInitializedRef.current) {
        return;
      }
      const googleAccounts = window.google?.accounts?.id;
      if (!googleAccounts || !buttonContainerRef.current) {
        return;
      }
      googleAccounts.initialize({ client_id: googleClientId, callback: handleCredential });
      googleAccounts.renderButton(buttonContainerRef.current, {
        type: "standard",
        theme: "outline",
        size: "large",
        text: "continue_with",
        shape: "pill",
      });
      googleAccounts.prompt();
      scriptInitializedRef.current = true;
    };

    if (window.google?.accounts?.id) {
      initializeButton();
      return;
    }

    if (scriptLoadingRef.current) {
      return;
    }

    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = initializeButton;
    script.onerror = () => {
      toast.error("Unable to load Google authentication. Please try again later.");
    };
    document.head.appendChild(script);
    scriptLoadingRef.current = true;

    return () => {
      script.onload = null;
      script.onerror = null;
    };
  }, [googleClientId, handleCredential]);

  const isConfigReady = configState === "ready" && Boolean(googleClientId);
  const isConfigLoading = configState === "loading";
  const isDisabled = isSubmitting || status === "loading" || !isConfigReady;

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
              <div ref={buttonContainerRef} />
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
