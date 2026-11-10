import { ShoppingBag } from "lucide-react";

export function CartButton({ count }: { count: number }) {
  return (
    <button className="relative">
      <ShoppingBag aria-hidden="true" />
      <span className="sr-only">Basket, {count} items</span>
      <span className="absolute -right-1 -top-1 rounded-full bg-rose-600 px-1.5 text-[10px] text-white">{count}</span>
    </button>
  );
}
