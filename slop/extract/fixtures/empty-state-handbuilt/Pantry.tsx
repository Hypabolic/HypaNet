export function Pantry({ jars, loading }: { jars: { id: string; label: string }[]; loading: boolean }) {
  if (jars.length === 0 && loading) return null;
  return (
    <section>
      <h2 className="text-lg font-semibold">Pantry shelf</h2>
      {loading ? (
        <p className="text-sm text-stone-500">Counting jars…</p>
      ) : jars.length === 0 ? (
        <div className="rounded-xl border border-dashed p-8 text-center">
          <p className="text-sm font-medium">No jars on the shelf</p>
          <p className="text-xs text-stone-500">Label a jar to start tracking what is left in it.</p>
        </div>
      ) : (
        <ul>
          {jars.map((j) => (
            <li key={j.id}>{j.label}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function Spices({ spices }: { spices: string[] }) {
  if (!spices.length) {
    return (
      <div>
        <p>Spice rack is bare</p>
        <p>Add a spice and it shows up here with its best-before date.</p>
      </div>
    );
  }
  return <p>{spices.length} spices</p>;
}
