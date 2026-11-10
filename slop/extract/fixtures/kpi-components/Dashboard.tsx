import { StatCard } from "@/components/stat-card";

export function Dashboard({ stats }: Props) {
  return (
    <div className="grid grid-cols-3 gap-4">
      <StatCard label="Open tickets" value={stats.open} />
      <StatCard label="Median reply" value="4h 12m" />
      <StatCard label="Satisfaction" value="96%" />
    </div>
  );
}
