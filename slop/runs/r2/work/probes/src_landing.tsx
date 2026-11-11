export default function Landing() {
  const plans = [
    { name: "Starter", price: "£0", blurb: "For trying things out.", features: ["3 projects", "Community support"] },
    { name: "Team", price: "£12", blurb: "For small teams that ship weekly.", features: ["Unlimited projects", "Priority support", "SSO"] },
  ];
  return (
    <main>
      <header className="flex items-center justify-between">
        <a href="/" className="font-bold">Tally</a>
        <nav className="flex gap-4">
          <a href="#features">Features</a>
          <a href="#pricing">Pricing</a>
          <a href="/login">Sign in</a>
        </nav>
      </header>

      <section className="py-24 text-center">
        <span className="rounded-full border px-3 py-1 text-xs">New: recurring invoices</span>
        <h1 className="text-5xl font-bold">Invoices that send themselves</h1>
        <p className="text-lg text-muted-foreground">Set a schedule once and Tally sends, chases and reconciles every invoice.</p>
        <div className="flex justify-center gap-3">
          <button className="btn-primary">Start free trial</button>
          <a href="#demo">Watch the demo</a>
        </div>
        <p className="text-xs text-muted-foreground">No card needed. Cancel any time.</p>
      </section>

      <section id="features" className="py-16">
        <p className="text-xs uppercase tracking-widest text-muted-foreground">Features</p>
        <h2 className="text-3xl font-semibold">Everything you need to get paid</h2>
        <p className="text-muted-foreground">All the features you need, in one place.</p>
        <div className="grid grid-cols-3 gap-6">
          <div className="rounded-lg border p-6">
            <h3 className="font-medium">Automatic reminders</h3>
            <p className="text-sm text-muted-foreground">Reminders go out 3, 7 and 14 days after the due date.</p>
          </div>
          <div className="rounded-lg border p-6">
            <h3 className="font-medium">Bank sync</h3>
            <p className="text-sm text-muted-foreground">Sync with your bank.</p>
          </div>
          <div className="rounded-lg border p-6">
            <h3 className="font-medium">Multi-currency</h3>
            <p className="text-sm text-muted-foreground">Invoice in 30 currencies and get paid in your own.</p>
          </div>
        </div>
      </section>

      <section id="pricing" className="py-16">
        <p className="text-xs uppercase tracking-widest text-muted-foreground">Pricing</p>
        <h2 className="text-3xl font-semibold">Simple pricing</h2>
        <p className="text-muted-foreground">Pricing that's clear and easy to understand.</p>
        <div className="grid grid-cols-2 gap-6">
          {plans.map((p) => (
            <div key={p.name} className="rounded-lg border p-6">
              <h3 className="font-medium">{p.name}</h3>
              <p className="text-3xl font-bold">{p.price}</p>
              <p className="text-sm text-muted-foreground">{p.blurb}</p>
              <ul>{p.features.map((f) => <li key={f}>{f}</li>)}</ul>
              <button>Choose {p.name}</button>
            </div>
          ))}
        </div>
        <p className="text-sm text-muted-foreground">Prices exclude VAT. Switch plans whenever you like.</p>
      </section>

      <section className="py-16">
        <p className="text-xs uppercase tracking-widest text-muted-foreground">FAQ</p>
        <h2 className="text-3xl font-semibold">Frequently asked questions</h2>
        <h3 className="font-medium">Can I cancel any time?</h3>
        <p className="text-muted-foreground">Yes. Your plan stays active until the end of the billing period.</p>
        <h3 className="font-medium">Do you store card details?</h3>
        <p className="text-muted-foreground">No. Payments are handled by Stripe.</p>
      </section>

      <section className="py-16 text-center">
        <h2 className="text-3xl font-semibold">Ready to get paid on time?</h2>
        <button className="btn-primary">Start free trial</button>
        <p className="text-xs text-muted-foreground">No card needed. Cancel any time.</p>
      </section>

      <footer className="border-t py-8 text-sm">
        <p>© 2026 Tally Ltd. All rights reserved.</p>
        <a href="/privacy">Privacy</a>
        <a href="/terms">Terms</a>
      </footer>
    </main>
  );
}
