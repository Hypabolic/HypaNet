import { QuoteCard } from "./QuoteCard";

const quotes = [
  { title: "Our bread order arrives before the ovens are hot", description: "Bakery on Mill Lane", featured: true },
  { title: "Invoices reconcile themselves on a Friday", description: "Greengrocer, Harbour Row", featured: false },
  { title: "I stopped chasing deliveries by phone", description: "Café by the lock", featured: false },
  { title: "This quote is never shown", description: "Fourth customer" },
];

export function Quotes() {
  return (
    <section>
      <h2 className="text-2xl font-semibold">Kitchens that switched</h2>
      <div className="grid gap-4 md:grid-cols-3">
        {quotes.slice(0, 3).map((q) => (
          <div key={q.title}>
            {q.featured ? <span className="rounded-full border px-2 text-xs">Featured</span> : null}
            <QuoteCard {...q} />
          </div>
        ))}
      </div>
    </section>
  );
}
