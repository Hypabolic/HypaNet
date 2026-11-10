import { useTranslation } from "react-i18next";

const PAGE_TITLE = "Tide tables";
const INTRO = `Heights are in metres above chart datum.`;
const COPY = {
  empty: "No readings for this harbour",
  footer: { note: "Times are local." },
} as const;
const STATE_LABEL: Record<string, string> = { rising: "Rising", falling: "Falling", slack: "Slack water" };
const SAVE_KEY = "tides.saveHarbour";
let banner = "Spring tides this week";
const PREFIX = "Next high water";
const prefixed = PREFIX + " at";

export function TideTable({ state, harbour, rows }: { state: string; harbour: string; rows: string[] }) {
  const { t } = useTranslation();
  const subtitle = "Tap a row for the hourly curve";
  const greeting = `Welcome to ${harbour}`;
  return (
    <section>
      <h1>{PAGE_TITLE}</h1>
      <p className="text-muted-foreground">{INTRO}</p>
      <p>{subtitle}</p>
      <span className="rounded-full px-2 text-xs">{STATE_LABEL[state]}</span>
      <span>{STATE_LABEL["slack"]}</span>
      <p>{greeting}</p>
      <p>{banner}</p>
      <p>{prefixed}</p>
      {rows.length === 0 && <p>{COPY.empty}</p>}
      <button type="button">{t(SAVE_KEY)}</button>
      <footer>
        <small>{COPY.footer.note}</small>
      </footer>
      <Shadow PAGE_TITLE="Harbour" />
      <h2 title={PAGE_TITLE}>{PAGE_TITLE}</h2>
    </section>
  );
}

function Shadow({ PAGE_TITLE }: { PAGE_TITLE: string }) {
  return <p>{PAGE_TITLE}</p>;
}
