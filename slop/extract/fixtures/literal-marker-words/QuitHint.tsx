export function QuitHint({ platform, name }: { platform: string; name: string }) {
  return (
    <div className="space-y-2">
      <p>Press ALT and F4 to quit, {name}.</p>
      <p>{platform === "mac" ? "Use the VAR key" : "Use the ALT key"}</p>
      <span title="VAR means variance here">VAR</span>
    </div>
  );
}
