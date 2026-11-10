export function Features() {
  return (
    <section className="py-16">
      <span className="text-xs uppercase tracking-widest text-slate-500">Features</span>
      <h2 className="text-3xl font-bold">Everything a repair café needs</h2>
      <div className="mt-10 grid grid-cols-3 gap-8">
        <div>
          <h3 className="font-semibold">Ticket queue</h3>
          <p className="text-slate-600">Visitors take a number at the door and see their place on the screen.</p>
        </div>
        <div>
          <h3 className="font-semibold">Parts shelf</h3>
          <p className="text-slate-600">Volunteers log spare fuses, belts and buttons as they arrive.</p>
        </div>
        <div>
          <h3 className="font-semibold">Fix log</h3>
          <p className="text-slate-600">Every mended toaster is counted towards the yearly landfill tally.</p>
        </div>
      </div>
    </section>
  );
}
