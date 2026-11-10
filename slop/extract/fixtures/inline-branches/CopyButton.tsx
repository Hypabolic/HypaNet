import { Check, Copy } from "lucide-react";

export function CopyButton({ copied, count }: Props) {
  return (
    <div className="flex gap-2">
      <button className="inline-flex items-center gap-1">
        {copied ? (
          <>
            <Check className="size-3" />
            Copied
          </>
        ) : (
          <>
            <Copy className="size-3" />
            Copy invite link
          </>
        )}
      </button>
      <p>
        {count === 1 ? <span>One seat left</span> : <span>{count} seats left</span>} on this coach
      </p>
    </div>
  );
}
