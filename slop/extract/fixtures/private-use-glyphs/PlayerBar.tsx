export function PlayerBar({ track }: { track: string }) {
  return (
    <div>
      <button type="button"> Play</button>
      <button type="button" aria-label="Skip">
        <span className="material-icons"></span>
      </button>
      <p>{"Shuffle \uE001 on"}</p>
      <p>Queue  {track}</p>
      <span title={`Repeat \uE000 one`}>Repeat</span>
    </div>
  );
}
