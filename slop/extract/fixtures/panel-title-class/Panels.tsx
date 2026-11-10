import Link from "next/link";

const links = [
  { label: "Charts", href: "/charts" },
  { label: "Tides", href: "/tides" },
];

export function Panels({ open }: { open: boolean }) {
  return (
    <div>
      <header className="border-b">
        <Link href="/" className="font-semibold">Quay</Link>
        {open && (
          <ul>
            {links.map((l) => (
              <li key={l.href}>
                <Link href={l.href}>{l.label}</Link>
              </li>
            ))}
          </ul>
        )}
      </header>
      <section className="panel">
        <div className="panel__title">Swell forecast</div>
        <p>Long-period swell from the west through Thursday.</p>
      </section>
      <section className="widget">
        <div className="widget-header">Wind</div>
        <p>Force four, backing south-west.</p>
      </section>
    </div>
  );
}
