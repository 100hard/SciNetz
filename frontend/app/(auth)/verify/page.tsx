"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import apiClient, { extractErrorMessage } from "@/lib/http";
import type { VerificationResponse } from "../../../types/auth";

type VerificationState = "idle" | "loading" | "success" | "error";

const VerifyPage = () => {
  const searchParams = useSearchParams();
  const token = searchParams?.get("token") ?? "";
  const [status, setStatus] = useState<VerificationState>("idle");
  const [message, setMessage] = useState<string>("Follow the verification link sent to your email.");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("Verification token is missing or has already been used.");
      return;
    }

    let cancelled = false;
    const verify = async () => {
      setStatus("loading");
      setMessage("Validating your verification link...");
      try {
        const { data } = await apiClient.get<VerificationResponse>("/api/auth/verify", {
          params: { token },
        });
        if (cancelled) {
          return;
        }
        setStatus("success");
        setMessage(data.message);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setStatus("error");
        setMessage(extractErrorMessage(error, "We couldn't verify your account. Please request a new link."));
      }
    };

    void verify();

    return () => {
      cancelled = true;
    };
  }, [token]);

  const statusLabel = useMemo(() => {
    if (status === "loading") {
      return "Verifying";
    }
    if (status === "success") {
      return "Verification successful";
    }
    if (status === "error") {
      return "Verification failed";
    }
    return "Awaiting verification";
  }, [status]);

  return (
    <div className="w-full max-w-md space-y-6 text-center">
      <div className="rounded-lg border bg-card p-8 shadow-xl">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-primary">SciNets</p>
        <h1 className="mt-4 text-2xl font-semibold text-foreground">{statusLabel}</h1>
        <p className="mt-3 text-sm text-muted-foreground">{message}</p>
        <div className="mt-6 flex justify-center">
          {status === "loading" ? (
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" aria-label="Loading" />
          ) : (
            <Link
              href="/login"
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:shadow focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2"
            >
              Back to login
            </Link>
          )}
        </div>
      </div>
    </div>
  );
};

export default VerifyPage;
