import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function ScheduleTabs() {
  return (
    <Tabs defaultValue="week">
      <TabsList>
        <TabsTrigger value="week">This week</TabsTrigger>
        <TabsTrigger value="month">This month</TabsTrigger>
      </TabsList>
      <TabsContent value="week">
        <p>Three sessions booked, one waiting for a coach.</p>
      </TabsContent>
      <TabsContent value="month">
        <p>Twelve sessions across four venues.</p>
      </TabsContent>
    </Tabs>
  );
}
