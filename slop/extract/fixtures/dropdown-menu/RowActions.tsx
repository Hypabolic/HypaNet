import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { MoreHorizontal } from "lucide-react";

export function RowActions() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger aria-label="Open actions">
        <MoreHorizontal />
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuLabel>Shipment</DropdownMenuLabel>
        <DropdownMenuItem>Print label</DropdownMenuItem>
        <DropdownMenuItem>Mark as collected</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
