import { RefreshCw } from "lucide-react";
import { Layout } from "../Layout";

export function StatusPage({ services, updatedAt, refresh }: Props) {
  return (
    <Layout>
      <nav className="tabs">
        <a href="/">Overview</a>
        <a href="/status" aria-current="page">Status</a>
        <a href="/billing">Billing</a>
      </nav>
      <header>
        <h1>Status</h1>
        <p className="text-muted-foreground">Checks run every minute from three regions.</p>
        <span className="rounded-full text-xs">● Live</span>
        <button onClick={refresh} title="Refresh now" aria-label="Refresh service status">
          <RefreshCw className="h-4 w-4" />
        </button>
      </header>
      <ul className="legend">
        <li><span className="dot bg-green-500" />Healthy</li>
        <li><span className="dot bg-amber-500" />Degraded</li>
        <li><span className="dot bg-red-500" />Down</li>
      </ul>
      <table>
        <thead>
          <tr><th>Service</th><th>State</th><th>Uptime (30 days)</th></tr>
        </thead>
        <tbody>
          {services.map((s) => (
            <tr key={s.id}>
              <td>{s.name}</td>
              <td><span className={dotClass(s.state)} />{s.stateLabel}</td>
              <td>{s.uptime}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs">Last updated {updatedAt}. All times are UTC.</p>
      <p className="text-xs">The table above lists every service with its state and uptime.</p>
    </Layout>
  );
}
