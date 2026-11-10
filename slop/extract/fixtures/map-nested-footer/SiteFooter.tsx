const columns = [
  {
    heading: "Workshop",
    links: [
      { label: "Timetable", href: "/timetable" },
      { label: "Tool library", href: "/tools" },
    ],
  },
  {
    heading: "Members",
    links: [
      { label: "Join the guild", href: "/join" },
      { label: "Renew", href: "/renew" },
      { label: "Lockers", href: "/lockers" },
    ],
  },
];

const legal = [{ key: "footer.legal.terms" }, { key: "footer.legal.privacy" }];

export function SiteFooter({ t }: { t: (k: string) => string }) {
  return (
    <footer className="border-t py-12">
      <div className="grid grid-cols-2 gap-8">
        {columns.map((col) => (
          <div key={col.heading}>
            <h4 className="font-medium">{col.heading}</h4>
            <ul className="mt-3 space-y-1">
              {col.links.map((link) => (
                <li key={link.href}>
                  <a href={link.href} aria-label={`Go to ${link.label}`}>
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <p className="mt-8 text-xs">
        {legal.map((item) => (
          <a key={item.key} href="#">{t(item.key)}</a>
        ))}
      </p>
    </footer>
  );
}
