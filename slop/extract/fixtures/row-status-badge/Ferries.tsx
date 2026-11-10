function SailingPill({ state }: { state: string }) {
  if (state === "late") {
    return <span className="inline-flex items-center gap-1 text-amber-600">Running late</span>;
  }
  return <span className="inline-flex items-center gap-1 text-emerald-600">On time</span>;
}

export function Ferries({ rows }: { rows: { id: string; route: string; berth: string; full: boolean }[] }) {
  return (
    <ul className="divide-y">
      {rows.map((r) => (
        <li key={r.id} className="flex items-center gap-3 py-2">
          <span className="font-medium">{r.route}</span>
          <span className="text-sm text-stone-500">{r.berth}</span>
          {r.full && <span className="text-xs text-rose-600">Full</span>}
          <SailingPill state="late" />
        </li>
      ))}
    </ul>
  );
}
