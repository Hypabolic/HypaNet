const STEPS = [
  { title: "Send us the frame", minutes: 5, detail: "pack it in the box we post", photo: "/frame.jpg" },
  { title: "We true the wheels", minutes: 40, detail: "spokes tensioned by hand", photo: "/wheel.jpg" },
  { title: "Ride it home", minutes: 0, detail: "collect from the shop or by courier", photo: "/ride.jpg" },
];

export function HowItWorks() {
  return (
    <section>
      <h2 className="text-3xl font-bold">From box to bike</h2>
      <ol className="mt-8 grid gap-4 md:grid-cols-3">
        {STEPS.map((step, index) => {
          const { title, minutes } = step;
          return (
            <li key={title} className="rounded-lg border p-5">
              <img src={step.photo} alt={title} />
              <span className="text-xs text-muted-foreground">{`Stage ${index + 1}`}</span>
              <h3 className="font-semibold">{title}</h3>
              <p className="text-sm">{minutes ? `${minutes} min · ${step.detail}` : `No wait · ${step.detail}`}</p>
              <a href="/stages" className="text-sm underline">Read the details</a>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
