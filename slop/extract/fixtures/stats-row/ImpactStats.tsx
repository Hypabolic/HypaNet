export function ImpactStats() {
  return (
    <section>
      <h2 className="text-2xl font-bold">Last season on the river</h2>
      <dl className="mt-8 grid grid-cols-3 gap-6">
        <div>
          <dt className="text-sm text-muted-foreground">Litter bags filled</dt>
          <dd className="text-3xl font-semibold">312</dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Volunteer hours</dt>
          <dd className="text-3xl font-semibold">2.4k</dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Water quality</dt>
          <dd className="text-3xl font-semibold">87%</dd>
        </div>
      </dl>
    </section>
  );
}
