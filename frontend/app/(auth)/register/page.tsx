"use client";

import Link from "next/link";

const RegisterPage = () => {
  return (
    <div className="w-full max-w-md space-y-6 text-center">
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-primary">SciNets</p>
        <h1 className="text-2xl font-semibold text-foreground">Sign up with Google</h1>
        <p className="text-sm text-muted-foreground">
          New accounts are created automatically the first time you continue with Google.
        </p>
      </div>
      <div className="space-y-4 rounded-lg border bg-card p-8 shadow-xl">
        <p className="text-sm text-muted-foreground">
          To get started, head to the sign-in page and use the <span className="font-medium text-primary">Continue with Google</span>{" "}
          option. We&apos;ll create your profile as soon as Google confirms your identity—no passwords required.
        </p>
        <Link
          href="/login"
          className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2"
        >
          Go to sign-in
        </Link>
      </div>
    </div>
  );
};

export default RegisterPage;
