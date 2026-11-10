import { Compass, Lamp, Tent } from "lucide-react";

export default function Features() {
  const features = [
    { icon: Compass, kicker: "routes", title: "Paths drawn by locals", body: "Every trail is walked by someone who lives nearby.", badge: "New" },
    { icon: Tent, kicker: "camps", title: "Pitches you can book", body: "Reserve a pitch the night before, not a season ahead." },
    { icon: Lamp, kicker: "kit", title: "Lamps on loan", body: "Borrow a head torch at any hut and drop it at the next.", badge: "" },
  ] satisfies { icon: unknown; kicker: string; title: string; body: string; badge?: string }[];

  return (
    <section className="py-16">
      <h2 className="text-4xl font-bold">Walk further with less</h2>
      <div className="mt-10 grid gap-6 md:grid-cols-3">
        {features.map(({ icon: Icon, kicker, title, body, badge }) => (
          <div key={title} className="rounded-xl border p-6">
            <Icon className="h-5 w-5" />
            <p className="text-xs uppercase tracking-widest text-stone-500">{kicker.toUpperCase()}</p>
            <h3 className="mt-2 text-lg font-semibold">{title}</h3>
            {badge && <span className="rounded-full bg-amber-100 px-2 text-xs">{badge}</span>}
            <p className="mt-2 text-sm text-muted-foreground">{body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
