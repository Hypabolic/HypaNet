import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Bold, Italic } from "lucide-react";

export function Toolbar() {
  return (
    <div className="flex gap-1" role="toolbar">
      <Tooltip>
        <TooltipTrigger asChild>
          <button aria-label="Bold"><Bold className="size-4" /></button>
        </TooltipTrigger>
        <TooltipContent>Bold (Ctrl+B)</TooltipContent>
      </Tooltip>
      <button title="Italic text" aria-label="Italic"><Italic className="size-4" /></button>
      <img src="/stamp.png" alt="Post office stamp with a puffin" />
      <a href="/help" data-tooltip="Opens the style guide in a new tab">Style guide</a>
    </div>
  );
}
