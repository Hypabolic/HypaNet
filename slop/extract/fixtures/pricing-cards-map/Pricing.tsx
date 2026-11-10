const tiers = [
  { name: "Seedling", price: 0, blurb: "One plot, one grower." },
  { name: "Orchard", price: 6, blurb: "Up to five plots and shared notes." },
  { name: "Estate", price: 14, blurb: "Unlimited plots for community gardens." },
];

export function Pricing() {
  return (
    <section className="py-20">
      <div className="text-center">
        <p className="text-sm font-semibold uppercase tracking-wide text-lime-700">Pricing</p>
        <h2 className="text-4xl font-bold">Pick the size of your patch</h2>
      </div>
      <div className="mt-12 grid gap-6 md:grid-cols-3">
        {tiers.map((tier) => (
          <div key={tier.name} className="rounded-2xl border p-8">
            <h3 className="text-lg font-semibold">{tier.name}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{tier.blurb}</p>
            <p className="mt-6">
              <span className="text-4xl font-bold">£{tier.price}</span>
              <span className="text-sm">/month</span>
            </p>
            <button className="mt-8 w-full rounded-md bg-lime-700 py-2 text-white">Choose {tier.name}</button>
          </div>
        ))}
      </div>
    </section>
  );
}
