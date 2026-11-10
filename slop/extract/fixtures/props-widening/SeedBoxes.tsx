import { PriceTable, FeatureList, TagRow, Steps } from "./parts";

export function PlansPage() {
  return (
    <main>
      <h1>Pick a seed box</h1>
      <PriceTable
        headers={["Box", "Monthly price", "Packets"]}
        rows={[
          ["Starter", "£4", "6"],
          ["Grower", "£9", "14"],
        ]}
      />
      <FeatureList
        items={[
          { id: "sow", icon: "sprout", feature: "Sowing calendar by post", subLabel: "Every season", href: "/calendar" },
          { id: "swap", icon: "swap", feature: "Swap spare packets", subLabel: "Members only", variant: "muted" },
        ]}
      />
      <TagRow tags={["Heirloom", "Organic", "pollinator-friendly", "https://example.com/x"]} />
      <Steps steps={[{ title: "Choose a box", snippet: "Start small and add later." }, { title: "Sow", description: "Follow the card." }]} />
      <Chart series={["north-bed", "south-bed"]} colours={["#3a7", "#a73"]} />
    </main>
  );
}
