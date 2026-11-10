import { toast } from "sonner";

export function useSaveRecipe() {
  return async function save(recipe: Recipe) {
    try {
      await api.put(recipe);
      toast.success("Recipe saved to your box");
    } catch (e) {
      toast.error("Could not save the recipe", { description: "The kitchen tablet is offline." });
    }
  };
}
