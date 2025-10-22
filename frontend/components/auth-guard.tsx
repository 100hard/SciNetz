"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { useAuth } from "./auth-provider";

const AuthGuard = ({ children }: { children: ReactNode }) => {
  const { status } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (status === "unauthenticated") {
      const query = searchParams?.toString();
      const nextPath = pathname ? `${pathname}${query ? `?${query}` : ""}` : "";
      const redirectTarget = nextPath ? `?next=${encodeURIComponent(nextPath)}` : "";
      router.replace(`/login${redirectTarget}`);
    }
  }, [pathname, router, searchParams, status]);

  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div
          aria-label="Loading"
          className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"
        />
      </div>
    );
  }

  if (status !== "authenticated") {
    return null;
  }

  return <>{children}</>;
};

export default AuthGuard;
