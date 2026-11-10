import { Stack, Tabs } from "expo-router";
import { DataTable, Legend, Toggle } from "./parts";

const MODES = [
  {
    id: "unit",
    label: "Units",
    options: [
      { value: "mm", label: "Millimetres" },
      { value: "in", label: "Inches" },
    ],
  },
];

export function RainfallFigure({ rows }: { rows: string[][] }) {
  return (
    <figure>
      <Legend
        items={[
          { tone: "sky", shape: "bar", label: "Wettest year" },
          { tone: "sand", shape: "dot", label: "Driest year" },
        ]}
      />
      <Toggle controls={MODES} />
      <DataTable
        heading="Monthly totals"
        columns={[
          { head: "Month" },
          { head: "Rainfall", numeric: true },
          { head: "Rain days", numeric: true },
        ]}
        rows={rows}
      />
      <Gallery tiles={[{ caption: "Gauge at the allotment", src: "/gauge.jpg" }]} />
    </figure>
  );
}

export function GaugeTabs() {
  return (
    <Tabs>
      <Tabs.Screen name="today" options={{ title: "Today", tabBarBadge: 3 }} />
      <Tabs.Screen name="history" options={{ title: "History" }} />
      <Stack.Screen options={{ title: "Gauge details" }} />
    </Tabs>
  );
}
