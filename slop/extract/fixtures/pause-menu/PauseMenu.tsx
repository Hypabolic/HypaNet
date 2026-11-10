export function PauseMenu({ onResume }: { onResume: () => void }) {
  return (
    <div className="fixed inset-0 grid place-items-center bg-black/70">
      <div className="w-80 space-y-6 text-center">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.3em] text-amber-400">OPERATION ON HOLD</p>
          <h1 className="text-5xl font-black">PAUSED</h1>
          <p className="text-sm text-zinc-400">Resume recaptures the mouse.</p>
        </div>
        <div className="flex flex-col gap-2">
          <button onClick={onResume}>RESUME</button>
          <button>SETTINGS</button>
          <button>CONTROLS</button>
          <button>RESTART MISSION</button>
        </div>
      </div>
    </div>
  );
}
