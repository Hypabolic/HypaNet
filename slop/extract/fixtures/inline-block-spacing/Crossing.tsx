export function Crossing({ next }: { next: { title: string } }) {
  return (
    <div>
      <p>Over the bridge<small className="block">sleeps the ferryman</small></p>
      <a href="/next" className="block">
        <span className="block text-sm">Next crossing</span>
        <span>{next.title}</span>
      </a>
      <p>Low<strong>tide</strong> at noon</p>
    </div>
  );
}
