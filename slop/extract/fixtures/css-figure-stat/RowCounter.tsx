export function RowCounter({ rows }: { rows: number }) {
  return (
    <section className="counter-card">
      <p className="tally">{rows}</p>
      <p>Rows knitted this week</p>
      <p className="tally-fluid">86%</p>
      <p>Of the sleeve finished</p>
      <p className="note">12</p>
      <p>Stitches dropped</p>
    </section>
  );
}
