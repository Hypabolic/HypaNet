export function StampBook({ stamps, swaps }: { stamps: string[]; swaps: string[] }) {
  return (
    <section>
      <h2>Your stamps</h2>
      {stamps.length === 0 ? (
        <p className="text-muted-foreground">No stamps yet</p>
      ) : (
        <ul>
          {stamps.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      )}
      <h2>Swaps</h2>
      {!swaps.length && <p>Swaps you agree with other collectors appear here once both sides confirm.</p>}
    </section>
  );
}
