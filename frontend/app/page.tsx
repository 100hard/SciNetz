import {
  Activity,
  Database,
  FileText,
  LineChart,
  Users,
} from "lucide-react";

import SemanticSearch from "../components/semantic-search";

const stats = [
  {
    title: "Knowledge Graph Nodes",
    value: "18,245",
    change: "+3.4% vs last week",
    icon: Database,
  },
  {
    title: "Published Papers",
    value: "612",
    change: "+28 new",
    icon: FileText,
  },
  {
    title: "Collaborating Researchers",
    value: "134",
    change: "+12 active this month",
    icon: Users,
  },
  {
    title: "Pipeline Health",
    value: "Stable",
    change: "Last run 2 hours ago",
    icon: Activity,
  },
];

const activityFeed = [
  {
    title: "Ingestion completed",
    description: "ArXiv computer vision papers batch",
    time: "10 minutes ago",
  },
  {
    title: "Dataset validated",
    description: "Protein folding simulation set",
    time: "45 minutes ago",
  },
  {
    title: "New collaborator joined",
    description: "Dr. Li Wei added to Quantum Materials project",
    time: "2 hours ago",
  },
];

export default function Home() {
  return (
    <div className="space-y-6">
      <SemanticSearch />

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.title}
              className="rounded-lg border bg-card p-5 shadow-sm transition hover:shadow-md"
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">{stat.title}</p>
                  <p className="mt-2 text-2xl font-semibold text-foreground">{stat.value}</p>
                </div>
                <span className="rounded-md bg-primary/10 p-2 text-primary">
                  <Icon className="h-5 w-5" />
                </span>
              </div>
              <p className="mt-4 text-xs font-medium uppercase tracking-wide text-primary">
                {stat.change}
              </p>
            </div>
          );
        })}
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Ingestion Pipeline</h2>
              <p className="text-sm text-muted-foreground">Track the latest crawl and enrichment jobs.</p>
            </div>
            <LineChart className="h-5 w-5 text-muted-foreground" />
          </div>
          <div className="mt-6 space-y-5">
            <div>
              <div className="flex items-center justify-between text-sm font-medium text-foreground">
                <span>Metadata aggregation</span>
                <span className="text-muted-foreground">82%</span>
              </div>
              <div className="mt-2 h-2 rounded-full bg-muted">
                <div className="h-2 rounded-full bg-primary" style={{ width: "82%" }} />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between text-sm font-medium text-foreground">
                <span>Entity resolution</span>
                <span className="text-muted-foreground">64%</span>
              </div>
              <div className="mt-2 h-2 rounded-full bg-muted">
                <div className="h-2 rounded-full bg-primary/70" style={{ width: "64%" }} />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between text-sm font-medium text-foreground">
                <span>Graph sync</span>
                <span className="text-muted-foreground">Next run: 03:00 UTC</span>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Automated refresh scheduled overnight to capture newly published research.
              </p>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="rounded-lg border bg-card p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground">Recent activity</h2>
            <ul className="mt-4 space-y-4">
              {activityFeed.map((item) => (
                <li key={item.title} className="border-l-2 border-primary/30 pl-4">
                  <p className="text-sm font-semibold text-foreground">{item.title}</p>
                  <p className="text-sm text-muted-foreground">{item.description}</p>
                  <p className="text-xs text-muted-foreground">{item.time}</p>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg border bg-card p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground">Action items</h2>
            <ul className="mt-4 space-y-3 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-400" aria-hidden />
                Review entity merges proposed for AI Safety papers.
              </li>
              <li className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-400" aria-hidden />
                Approve dataset sync for Materials Project updates.
              </li>
              <li className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-sky-400" aria-hidden />
                Invite collaborators from the National Lab workspace.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
