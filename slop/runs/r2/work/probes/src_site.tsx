const roles = [
  { period: "2021 — Present", title: "Chief Technology Officer", org: "Northwind Labs", note: "Platform, security and data teams" },
  { period: "2016 — 2021", title: "Head of Engineering", org: "Harbour Systems", note: "Payments infrastructure" },
];

const projects = [
  { name: "Lantern", blurb: "A static analyser for infrastructure code.", tags: ["Rust", "Security"] },
  { name: "Quire", blurb: "Notes that sync without a server.", tags: ["TypeScript", "CRDT"] },
];

export function Experience() {
  return (
    <section id="experience" className="py-16">
      <h2 className="text-sm uppercase tracking-widest text-muted-foreground">Experience</h2>
      <p className="mt-2 text-lg">Engineering leadership across platform, security, data, and payments.</p>
      <ol className="mt-8 space-y-8">
        {roles.map((r) => (
          <li key={r.title}>
            <p className="font-mono text-xs text-muted-foreground">{r.period}</p>
            <h3 className="font-medium">{r.title}</h3>
            <p className="text-sm">{r.org}</p>
            <p className="text-sm text-muted-foreground">{r.note}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function Projects() {
  return (
    <section id="projects" className="py-16">
      <p className="text-xs uppercase tracking-widest text-muted-foreground">Selected work</p>
      <h2 className="text-2xl font-semibold">Projects</h2>
      <p className="mt-2 text-muted-foreground">A selection of my projects.</p>
      <div className="mt-8 grid grid-cols-2 gap-6">
        {projects.map((p) => (
          <article key={p.name} className="rounded-lg border p-6">
            <h3 className="font-medium">{p.name}</h3>
            <p className="text-sm text-muted-foreground">{p.blurb}</p>
            <div className="mt-3 flex gap-2">
              {p.tags.map((t) => (
                <span key={t} className="rounded-full border px-2 py-0.5 text-xs">{t}</span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export function About() {
  return (
    <section id="about" className="py-16">
      <p className="text-xs uppercase tracking-widest text-muted-foreground">About</p>
      <h2 className="text-2xl font-semibold">About me</h2>
      <p className="mt-4">I build platforms for regulated industries and write about what goes wrong.</p>
      <p className="mt-4">Outside work I restore old sailing dinghies.</p>
      <div className="mt-6 flex items-center gap-2 text-sm">
        <span className="h-2 w-2 rounded-full bg-green-500" />
        <span>Available for work</span>
      </div>
      <a href="mailto:hello@example.com" className="mt-6 inline-block underline">Get in touch</a>
    </section>
  );
}
