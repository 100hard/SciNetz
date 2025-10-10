import { Bell, Search } from "lucide-react";

const Header = () => {
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
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold uppercase text-primary">
          SN
        </div>
      </div>
    </header>
  );
};

export default Header;
