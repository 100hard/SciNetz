"use client";

import { Bell, LogOut, Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo } from "react";
import { toast } from "sonner";

import { useAuth } from "./auth-provider";
import { extractErrorMessage } from "../lib/http";

const Header = () => {
  const router = useRouter();
  const { user, logout } = useAuth();

  const initials = useMemo(() => {
    if (!user?.email) {
      return "SN";
    }
    return user.email
      .split("@")[0]
      .split(/[._-]/)
      .filter(Boolean)
      .map((segment) => segment[0]?.toUpperCase())
      .join("")
      .slice(0, 2) || user.email[0]?.toUpperCase() || "SN";
  }, [user?.email]);

  const handleLogout = async () => {
    try {
      await logout();
      toast.success("Signed out successfully.");
      router.replace("/login");
    } catch (error) {
      toast.error(extractErrorMessage(error, "Unable to sign out."));
    }
  };

  return (
    <header className="flex items-center border-b bg-card/80 px-6 py-4 backdrop-blur supports-[backdrop-filter]:bg-card/60">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Dashboard</p>
        <h1 className="text-lg font-semibold text-foreground">SciNets Overview</h1>
      </div>
      <div className="ml-auto flex items-center gap-4">
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search papers, datasets, people..."
            className="w-64 rounded-md border border-input bg-background px-3 py-2 pl-9 text-sm shadow-sm outline-none transition focus:border-transparent focus:ring-2 focus:ring-primary/40"
          />
        </div>
        <button
          type="button"
          className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-input bg-background text-muted-foreground transition hover:text-foreground"
          aria-label="View notifications"
        >
          <Bell className="h-4 w-4" />
        </button>
        <div className="flex items-center gap-3">
          <div className="hidden text-right sm:block">
            <p className="text-sm font-medium text-foreground">{user?.email ?? "Account"}</p>
            <p className="text-xs text-muted-foreground">
              {user?.is_verified ? "Verified" : "Awaiting verification"}
            </p>
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold uppercase text-primary">
            {initials}
          </div>
          <button
            type="button"
            onClick={handleLogout}
            className="inline-flex h-9 items-center gap-2 rounded-md border border-input bg-background px-3 text-sm font-medium text-muted-foreground transition hover:border-primary/40 hover:text-foreground"
          >
            <LogOut className="h-4 w-4" />
            <span className="hidden sm:inline">Sign out</span>
            <span className="sr-only">Sign out</span>
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
