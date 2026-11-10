export function Sync({ sync }: { sync: { phase: string; file: string; count: number; error?: string } }) {
  return (
    <p className="text-xs">
      <span className="font-medium">{sync.file}</span>
      {sync.phase === "sending" && " · sending…"}
      {sync.phase === "checking" && " · checking…"}
      {sync.phase === "done" && ` · saved · ${sync.count} waypoints`}
      {sync.phase === "failed" && ` · ${sync.error ?? "failed"}`}
    </p>
  );
}
