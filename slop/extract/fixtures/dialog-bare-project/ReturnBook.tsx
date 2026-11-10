import { Dialog, DialogTitle } from "../ui/dialog";

export function ReturnBook({ open, title, onClose }: { open: boolean; title: string; onClose: () => void }) {
  return (
    <div>
      <button type="button">Return a book</button>
      <Dialog open={open} onClose={onClose}>
        <DialogTitle>Return {title}?</DialogTitle>
        <p>Late fees stop counting from today.</p>
        <button type="button" onClick={onClose}>Return it</button>
      </Dialog>
    </div>
  );
}
