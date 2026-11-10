export default function Newsletter({ sent }) {
  if (sent) return <p className="text-green-700">Thanks, check your inbox.</p>;
  return (
    <form className="flex gap-2">
      <input type="email" placeholder="you@example.com" aria-label="Email address" />
      <button type="submit">Subscribe</button>
    </form>
  );
}
