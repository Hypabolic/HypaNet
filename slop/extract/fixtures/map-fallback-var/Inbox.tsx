import { loadFolders } from "./api";

const items = ["Drafts", "Sent", "Archive"];
let labels = ["Receipts", "Travel", "Family"];
const extra = [...items, "Spam"];
const folders = loadFolders();
const pinned = ["Starred"];
pinned.push("Follow up");
const SHORT: Record<string, string> = { low: "Can wait", high: "Today" };

function FolderList({ items }: { items: { name: string; count: number }[] }) {
  return (
    <ul>
      {items.map((f) => (
        <li key={f.name}>
          <span>{f.name}</span>
          <span className="text-xs text-muted-foreground">{f.count} unread</span>
        </li>
      ))}
    </ul>
  );
}

export function Inbox() {
  const recent: string[] = [];
  for (const f of ["Sent", "Drafts"]) recent.push(f);
  return (
    <aside>
      <h2>Mailboxes</h2>
      <FolderList items={folders} />
      <ul>
        {labels.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
      <ul>
        {extra.map((x) => (
          <li key={x}>{x}</li>
        ))}
      </ul>
      <ul>
        {pinned.map((p) => (
          <li key={p}>{p}</li>
        ))}
        {recent.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
      <p>
        {(["low", "high"] as const).map((k) => (
          <span key={k}>{SHORT[k]}</span>
        ))}
      </p>
    </aside>
  );
}
