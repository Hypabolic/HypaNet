export function Hud({ ammo, wave, objective, health }: Props) {
  return (
    <div className="pointer-events-none fixed inset-0 font-mono">
      <div className="absolute left-4 top-4 text-xs uppercase tracking-widest">
        <div>Wave {wave}</div>
        <div className={health < 30 ? "text-red-500" : "text-white"}>HP {health}</div>
      </div>
      <div className="absolute right-4 top-4 text-right">
        <div className="text-[10px] uppercase tracking-[0.25em] text-zinc-400">Objective</div>
        <div>{objective}</div>
      </div>
      <div className="absolute bottom-4 right-4 text-3xl">{ammo}</div>
    </div>
  );
}
