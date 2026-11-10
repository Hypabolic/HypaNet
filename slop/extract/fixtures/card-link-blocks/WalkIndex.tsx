import Link from "next/link";

export function WalkIndex({ walks }: { walks: { slug: string; name: string; summary: string; miles: number }[] }) {
  return (
    <main>
      <h1>Coastal walks</h1>
      <nav>
        <Link href="/walks">
          <div>All walks</div>
        </Link>
        <a href="/maps">Maps</a>
      </nav>
      {walks.map((w) => (
        <Link key={w.slug} href={`/walks/${w.slug}`} className="block rounded-lg border p-4">
          <h2>{w.name}</h2>
          <p className="text-sm text-muted-foreground">{w.summary}</p>
          <div className="text-xs">{w.miles} miles</div>
          <span>Open route</span>
        </Link>
      ))}
      <a href="/shop" className="block">
        <p>Printed maps</p>
        <p>Waterproof, folded to pocket size</p>
      </a>
      <a href="/about">About the <strong>trail society</strong></a>
    </main>
  );
}
