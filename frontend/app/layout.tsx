import "./globals.css";

export const metadata = {
  title: "SciNets Graph Explorer",
  description: "Explore research knowledge graphs with evidence-backed insights"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100">
        <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
            <h1 className="text-lg font-semibold tracking-wide">SciNets Knowledge Graph</h1>
            <span className="text-sm text-slate-400">Phase 8 · Graph + Evidence</span>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
