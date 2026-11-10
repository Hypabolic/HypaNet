export function Intro() {
  return (
    <section>
      <div className="space-y-3">
        <p className="text-sm font-medium text-indigo-500">Version 4 is here</p>
        <h2 className="text-3xl font-semibold">A calmer way to run your book club</h2>
        <p>Rotate hosts, track who has finished, and keep the snack rota honest.</p>
      </div>
      <div className="mt-10 space-y-4">
        <h3 className="text-xl font-semibold">Hosting</h3>
        <p>Each month the app suggests whoever hosted least recently.</p>
        <p>You can swap with a friend in two taps if a date clashes.</p>
      </div>
    </section>
  );
}
