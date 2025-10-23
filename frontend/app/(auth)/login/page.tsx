"use client";

import type { FormEvent } from "react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";

import { useAuth } from "../../../components/auth-provider";
import { extractErrorMessage } from "@/lib/http";

const LoginPage = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { status, login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setSubmitting] = useState(false);

  const rawNextParam = searchParams?.get("next") ?? "/";
  const nextParam = rawNextParam.startsWith("/") ? rawNextParam : "/";
  const registered = searchParams?.get("registered");

  useEffect(() => {
    if (registered) {
      toast.success("Account created. Check your email for a verification link.");
    }
  }, [registered]);

  useEffect(() => {
    if (status === "authenticated") {
      router.replace(nextParam || "/");
    }
  }, [nextParam, router, status]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      const response = await login(email.trim().toLowerCase(), password);
      toast.success(response.message);
      router.replace(nextParam || "/");
    } catch (error) {
      toast.error(extractErrorMessage(error, "Unable to log in."));
    } finally {
      setSubmitting(false);
    }
  };

  const isDisabled = isSubmitting || status === "loading";

  return (
    <div className="w-full max-w-md space-y-6">
      <div className="space-y-2 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-primary">SciNets</p>
        <h1 className="text-2xl font-semibold text-foreground">Welcome back</h1>
        <p className="text-sm text-muted-foreground">Sign in with your credentials to access the dashboard.</p>
      </div>
      <div className="rounded-lg border bg-card p-8 shadow-xl">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <label className="flex flex-col gap-1 text-left">
              <span className="text-sm font-medium text-foreground">Email</span>
              <input
                required
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
                placeholder="you@example.com"
              />
            </label>
            <label className="flex flex-col gap-1 text-left">
              <span className="text-sm font-medium text-foreground">Password</span>
              <input
                required
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
                placeholder="••••••••"
                minLength={8}
              />
            </label>
          </div>
          <button
            type="submit"
            disabled={isDisabled}
            className={`inline-flex w-full items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`}
          >
            {isSubmitting ? "Signing in..." : "Sign in"}
          </button>
        </form>
        <p className="mt-6 text-center text-sm text-muted-foreground">
          Don&apos;t have an account?{" "}
          <Link href="/register" className="font-medium text-primary hover:underline">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
