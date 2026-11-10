export default function OrdersPage() {
  const { data, error, isLoading } = useOrders();
  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Fetching your orders…</p>;
  }
  if (error) {
    return (
      <div role="alert">
        <h2>We could not reach the till</h2>
        <p>Check your connection and try again.</p>
      </div>
    );
  }
  return (
    <main>
      <h1>Orders</h1>
      <ul>
        {data.map((o) => (
          <li key={o.id}>{o.reference}</li>
        ))}
      </ul>
    </main>
  );
}
