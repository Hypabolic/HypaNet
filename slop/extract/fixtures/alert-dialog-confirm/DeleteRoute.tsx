import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export function DeleteRoute({ open, onDelete }: Props) {
  function quickDelete() {
    if (window.confirm("Delete this cycling route and its 14 waypoints?")) onDelete();
  }
  return (
    <AlertDialog open={open}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete route?</AlertDialogTitle>
          <AlertDialogDescription>This also removes the route from two shared rides.</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Keep route</AlertDialogCancel>
          <AlertDialogAction onClick={onDelete}>Delete route</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
