import Link from "next/link";
import { ArrowRight } from "lucide-react";

export default function Hero() {
  return (
    <section className="mx-auto max-w-5xl px-6 py-24">
      <div className="max-w-2xl">
        <p className="text-xs font-semibold uppercase tracking-widest text-emerald-600">Field notes for allotment growers</p>
        <h1 className="mt-4 text-5xl font-bold tracking-tight">Plan every bed before the frost lifts</h1>
        <p className="mt-6 text-lg text-muted-foreground">
          Sketch your plot, log what went in the ground, and get a nudge when the
          seedlings need potting on.
        </p>
        <div className="mt-8 flex gap-3">
          <Link href="/start" className="rounded-md bg-emerald-600 px-4 py-2 text-white">
            Start a plot plan <ArrowRight className="size-4" />
          </Link>
          <Link href="/tour">Take the two-minute tour →</Link>
        </div>
      </div>
    </section>
  );
}
