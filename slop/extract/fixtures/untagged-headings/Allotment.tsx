export function Allotment({ plots, older }: { plots: string[]; older: string[] }) {
  return (
    <div className="flex">
      <aside className="w-56">
        <p className="text-xs text-muted-foreground">Pinned</p>
        <ul>
          {plots.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
        <p className="text-xs text-muted-foreground">Older</p>
        <div className="flex flex-col">
          {older.map((p) => (
            <button key={p} type="button">{p}</button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">Synced a minute ago.</p>
      </aside>
      <main>
        <div className="text-2xl font-bold">Plot 14</div>
        <div className="text-lg font-semibold">Watering rota</div>
        <p>Each holder waters on their own day; the tap is shut after dusk.</p>
        <div className="plot-name">Compost bays</div>
        <p>Turn the left bay first.</p>
        <div className="text-lg font-semibold">Gates lock at nine on weekdays.</div>
        <p>Ask the secretary for a key.</p>
        <div className="text-lg font-medium">Seed swap</div>
        <p>Bring labelled envelopes.</p>
        <div className="text-xl font-semibold">Contact the committee</div>
      </main>
    </div>
  );
}
