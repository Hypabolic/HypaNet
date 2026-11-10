import { Kbd } from "@/components/kbd";

export function Shortcuts() {
  return (
    <div className="grid grid-cols-2 gap-2">
      <span>Search the catalogue</span>
      <Kbd>⌘ K</Kbd>
      <span>New loan</span>
      <Kbd>N</Kbd>
    </div>
  );
}
