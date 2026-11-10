import { cn } from "@/lib/utils";

export function ConnectionRow({ online, label }: { online: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={cn("h-2 w-2 rounded-full", online ? "bg-emerald-500" : "bg-zinc-400")} />
      <span className="text-sm">{online ? "Connected" : "Offline"}</span>
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  );
}
