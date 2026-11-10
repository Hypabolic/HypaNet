import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="border-b">
      <div className="flex h-14 items-center justify-between">
        <Link href="/" className="font-bold">Tidewell</Link>
        <nav className="flex gap-6 text-sm">
          <Link href="/charts">Charts</Link>
          <Link href="/moorings">Moorings</Link>
          <Link href="/pricing">Pricing</Link>
        </nav>
        <button className="rounded-md border px-3 py-1.5 text-sm">Sign in</button>
      </div>
    </header>
  );
}
