const PERKS = [
  "Sharpening booked within two days",
  `Loan knives while yours are away`,
  "Free re-edge every spring",
] as const;

export function Perks() {
  return (
    <section>
      <h2 className="text-3xl font-semibold">What membership covers</h2>
      <ul className="mt-6 space-y-2">
        {PERKS.map((perk, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-emerald-600">✓</span>
            {perk}
          </li>
        ))}
      </ul>
    </section>
  );
}
