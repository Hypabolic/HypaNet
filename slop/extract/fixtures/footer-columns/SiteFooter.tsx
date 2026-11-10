export function SiteFooter() {
  return (
    <footer className="border-t py-12">
      <div className="grid grid-cols-2 gap-8">
        <div>
          <h4 className="font-semibold">Product</h4>
          <ul>
            <li><a href="/tide-tables">Tide tables</a></li>
            <li><a href="/alerts">Swell alerts</a></li>
          </ul>
        </div>
        <div>
          <h4 className="font-semibold">Company</h4>
          <ul>
            <li><a href="/about">About us</a></li>
            <li><a href="/press">Press kit</a></li>
          </ul>
        </div>
      </div>
      <p className="mt-8 text-xs text-muted-foreground">© 2026 Tidewell Ltd. Registered in Cornwall.</p>
    </footer>
  );
}
