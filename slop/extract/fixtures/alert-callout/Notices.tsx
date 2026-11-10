import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function Notices() {
  return (
    <div className="space-y-4">
      <Alert>
        <AlertTitle>Planned maintenance</AlertTitle>
        <AlertDescription>The booking system is read-only on Sunday morning.</AlertDescription>
      </Alert>
      <Alert variant="destructive">
        <AlertTitle>Payment failed</AlertTitle>
        <AlertDescription>Your card was declined by the bank.</AlertDescription>
      </Alert>
      <div role="alert" className="text-sm">The court you chose is already taken.</div>
    </div>
  );
}
