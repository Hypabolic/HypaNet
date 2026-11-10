import { Badge } from "@/components/ui/badge";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function Releases() {
  return (
    <section className="space-y-6">
      <h2 className="text-2xl font-semibold">This spring's additions</h2>
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <Badge>New</Badge>
            <CardTitle>Seed swap board</CardTitle>
            <CardDescription>Post spare packets and claim a neighbour's.</CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <Badge>New</Badge>
            <CardTitle>Rain gauge sync</CardTitle>
            <CardDescription>Pull readings from a paired gauge every hour.</CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <Badge>New</Badge>
            <CardTitle>Shared tool shed</CardTitle>
            <CardDescription>See who borrowed the long loppers.</CardDescription>
          </CardHeader>
        </Card>
      </div>
    </section>
  );
}
