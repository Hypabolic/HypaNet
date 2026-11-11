import { useState } from "react";

type Source = { id: string; title: string };

export function ResearchPanel({ sources, running, error }: { sources: Source[]; running: boolean; error?: string }) {
  const [query, setQuery] = useState("");
  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Deep research</p>
        <h1 className="text-xl font-semibold">Ask your library</h1>
        <p className="text-sm text-muted-foreground">Answers cite the documents they draw on.</p>
      </header>

      <div className="space-y-2">
        <label htmlFor="q" className="text-sm font-medium">Question</label>
        <textarea id="q" placeholder="Ask a question…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <p className="text-xs text-muted-foreground">Your question.</p>
        <button disabled={running}>{running ? "Researching…" : "Start research"}</button>
        <p className="text-xs text-muted-foreground">Deep research can take up to ten minutes.</p>
      </div>

      <section>
        <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Overview</p>
        <h2 className="text-lg font-medium">Sources</h2>
        <p className="text-sm text-muted-foreground">The sources used for this answer.</p>
        {sources.length === 0 ? (
          <div>
            <h3 className="font-medium">No sources yet</h3>
            <p className="text-sm text-muted-foreground">Upload a document to start building your library.</p>
          </div>
        ) : (
          <ul>
            {sources.map((s) => (
              <li key={s.id}>{s.title}</li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="text-lg font-medium">Report</h2>
        <p className="text-sm text-muted-foreground">Reports are saved to your library and can be shared by link.</p>
        <div className="flex gap-2">
          <button>Export</button>
          <button title="Export">PDF</button>
          <button title="Copy a link anyone can open">Share</button>
        </div>
      </section>

      {error && (
        <div role="alert">
          <h3 className="font-medium">Research failed</h3>
          <p className="text-sm">The research could not be completed.</p>
          <p className="text-sm">{error}</p>
        </div>
      )}

      <footer className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
        <span>All systems operational</span>
        <span className="font-mono">v0.9.2</span>
      </footer>
    </div>
  );
}
