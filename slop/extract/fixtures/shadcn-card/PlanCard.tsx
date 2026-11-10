import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

export function PlanCard() {
  return (
    <Card>
      <CardHeader>
        <Badge variant="secondary">Most picked</Badge>
        <CardTitle>Workshop</CardTitle>
        <CardDescription>For studios with up to eight benches.</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-4xl font-bold">£19</p>
        <p className="text-sm text-muted-foreground">per bench, billed monthly</p>
      </CardContent>
      <CardFooter>
        <Button className="w-full">Choose Workshop</Button>
      </CardFooter>
    </Card>
  );
}
