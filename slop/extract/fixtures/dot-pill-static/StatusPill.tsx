export function StatusPill() {
  return (
    <header className="flex items-center justify-between">
      <h1 className="text-xl font-semibold">Ferry timetable</h1>
      <span className="inline-flex items-center gap-1.5 rounded-full bg-green-100 px-2.5 py-0.5 text-xs text-green-800">
        <span className="size-1.5 rounded-full bg-green-500" />
        Live
      </span>
    </header>
  );
}
