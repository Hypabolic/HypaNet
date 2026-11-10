export function UsageCard({ minutes }: { minutes: number }) {
  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <span className="text-[11px] uppercase tracking-wider text-zinc-500">Minutes logged</span>
        <div className="text-4xl font-bold">1,204</div>
      </div>
      <div>
        <span className="uppercase tracking-wide">Streak</span>
        <div className="text-3xl font-bold">{minutes}</div>
      </div>
    </div>
  );
}
