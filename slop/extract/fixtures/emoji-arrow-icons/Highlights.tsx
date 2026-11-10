import { Search, ChevronRight } from "lucide-react";

export function Highlights() {
  return (
    <div className="space-y-6">
      <h3>✨ Smart shelf search</h3>
      <p className="flex items-center gap-2">
        <Search className="size-4" />
        Find any book by the colour of its spine.
      </p>
      <a href="/guide" className="inline-flex items-center">
        Read the cataloguing guide
        <ChevronRight className="size-4" />
      </a>
      <p>🎉 Your branch reached 10,000 loans this year</p>
    </div>
  );
}
