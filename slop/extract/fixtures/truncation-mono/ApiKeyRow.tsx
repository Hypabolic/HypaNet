export function ApiKeyRow({ keyName, prefix }: Props) {
  return (
    <div className="flex items-center gap-4">
      <span className="max-w-40 truncate" title={keyName}>{keyName}</span>
      <code className="text-xs">{prefix}••••</code>
      <span className="font-mono text-sm">Created 3 Nov</span>
      <p className="line-clamp-2">Used by the nightly export to push stock levels to the warehouse partner.</p>
      <p>Press <kbd>Esc</kbd> to close</p>
    </div>
  );
}
