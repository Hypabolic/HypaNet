import { Trash, Pencil } from "lucide-react";

export function Toolbar() {
  return (
    <div className="flex gap-2">
      <IconButton icon={<Trash />} label="Bin this sketch" />
      <ToolButton icon={Pencil} label="Redraw the outline" />
      <ToolButton icon={Pencil} label="Ink it">Ink</ToolButton>
      <ActionButton label="Publish the map" />
    </div>
  );
}
