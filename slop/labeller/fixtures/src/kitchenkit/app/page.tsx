import Link from "next/link";

const features = [
  { n: "01 /", title: "Fridge scan", body: "Snap the shelves and get a list of what you have." },
  { n: "02 /", title: "Week planner", body: "Seven dinners built from that list and your tastes." },
  { n: "03 /", title: "Shop once", body: "One shopping list, grouped by aisle." },
];

export default function Home() {
  return (
    <main>
      <section className="hero">
        <span className="inline-flex items-center gap-2 rounded-full px-3 text-xs">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          Now in beta
        </span>
        <h1>Plan a week of dinners in ten minutes</h1>
        <p className="text-lg text-muted-foreground">Kitchenkit turns what is in your fridge into a plan and a list.</p>
        <Link href="/signup" className="btn">Start free trial</Link>
        <p className="text-xs">No card needed</p>
      </section>

      <section id="features">
        <p className="text-xs uppercase tracking-widest">FEATURES</p>
        <h2>Tools for small kitchens</h2>
        <p>The three tools below cover planning, shopping and cooking.</p>
        <div className="grid grid-cols-3 gap-6">
          <article>
            <span className="rounded-full text-xs">● New</span>
            <span className="font-mono">01 /</span>
            <h3>Fridge scan</h3>
            <p>Snap the shelves and get a list of what you have.</p>
          </article>
          <article>
            <span className="rounded-full text-xs">● New</span>
            <span className="font-mono">02 /</span>
            <h3>Week planner</h3>
            <p>Seven dinners built from that list and your tastes.</p>
          </article>
          <article>
            <span className="rounded-full text-xs">● New</span>
            <span className="font-mono">03 /</span>
            <h3>Shop once</h3>
            <p>One shopping list, grouped by aisle.</p>
          </article>
        </div>
      </section>

      <section className="cta">
        <h2>Ready to cook?</h2>
        <Link href="/signup" className="btn">Start free trial</Link>
        <p className="text-sm">That is the whole of Kitchenkit: scan, plan, shop.</p>
      </section>

      <footer className="sticky bottom-0">
        <Link href="/signup">Start free trial</Link>
      </footer>
    </main>
  );
}
