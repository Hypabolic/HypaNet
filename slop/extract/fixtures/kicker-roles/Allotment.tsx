export function Allotment({ plot }: { plot: { bed: string; crop: string; sown: string; water: string } }) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold">Greenhouse</h2>
        <p className="text-[11px] uppercase tracking-wide">A gardener's ledger</p>
      </div>
      <div className="rounded-lg border p-4">
        <p className="text-[10px] uppercase tracking-widest">Plot record</p>
        <dl>
          <dt>Bed</dt>
          <dd>{plot.bed}</dd>
          <dt>Crop</dt>
          <dd>{plot.crop}</dd>
        </dl>
      </div>
      <div>
        <p className="text-[10px] uppercase tracking-widest">Season</p>
        <h3 className="text-xl">Spring sowing</h3>
      </div>
      <div className="p-2">
        <p className="text-[10px] uppercase tracking-wide">Watering</p>
        <button type="button">Log a can</button>
      </div>
    </div>
  );
}
