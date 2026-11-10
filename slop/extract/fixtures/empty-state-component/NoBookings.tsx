import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/empty-state";

export function NoBookings() {
  return (
    <EmptyState
      title="No bookings yet"
      description="Bookings appear here once a guest confirms a slot."
      action={<Button>Share booking link</Button>}
    />
  );
}
