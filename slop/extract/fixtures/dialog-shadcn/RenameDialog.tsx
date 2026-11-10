import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function RenameDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline">Rename hive</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename hive</DialogTitle>
          <DialogDescription>Inspection notes keep the old name in their history.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-2">
          <Label htmlFor="hive">Hive name</Label>
          <Input id="hive" defaultValue="Hive 3" placeholder="e.g. Orchard east" />
        </div>
        <DialogFooter>
          <Button variant="ghost">Cancel</Button>
          <Button>Save name</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
