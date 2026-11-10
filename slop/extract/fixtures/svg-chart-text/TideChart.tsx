const HOURS = ["06:00", "12:00", "18:00"];

export function TideChart({ rows }: { rows: { port: string; heights: number[] }[] }) {
  return (
    <figure>
      <svg viewBox="0 0 300 120">
        <text className="chart-axis" x={0} y={10}>Height (m)</text>
        {HOURS.map((h, i) => (
          <text key={h} x={40 + i * 80} y={115}>{h}</text>
        ))}
        {rows.map((r, y) => (
          <g key={r.port}>
            <text x={0} y={30 + y * 20}>{r.port}</text>
            {r.heights.map((v, x) => (
              <text key={x} x={40 + x * 80} y={30 + y * 20}>{v.toFixed(1)}</text>
            ))}
          </g>
        ))}
        <text className="tide-value" x={250} y={10}>4.2</text>
      </svg>
      <figcaption>Predicted heights at three harbours.</figcaption>
    </figure>
  );
}

export function Glyph() {
  return (
    <svg viewBox="0 0 10 10">
      <path d="M0 0L10 10" />
    </svg>
  );
}
