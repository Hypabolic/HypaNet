export function SeedPackets({ n, sownAt, savedAt, organic, phase, error, dirty, grower }: {
  n: number; sownAt?: string; savedAt: string; organic: boolean; phase: string; error?: string; dirty: boolean; grower?: string;
}) {
  return (
    <ul>
      <li>{n} packet{n === 1 ? "" : "s"}</li>
      <li>
        {sownAt ? `Sown ${sownAt}` : `Saved ${savedAt}`}
        {organic ? " · Organic" : ""}
      </li>
      <li>
        <span className="font-medium">Order</span>
        {phase === "packing" && " · packing"}
        {phase === "failed" && ` · ${error ?? "not sent"}`}
      </li>
      <li>
        Draft{dirty && <em> (unsaved)</em>}
      </li>
      <li>Grown by {grower ?? ""}</li>
    </ul>
  );
}
