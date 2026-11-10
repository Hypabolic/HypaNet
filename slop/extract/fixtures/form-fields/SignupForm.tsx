export function SignupForm({ errors }: Props) {
  return (
    <form className="space-y-6">
      <h2 className="text-lg font-semibold">Create a club account</h2>
      <div>
        <label htmlFor="club">Club name</label>
        <input id="club" placeholder="Harbour Rowing Club" />
        <p className="text-sm text-muted-foreground">Shown on fixture lists and receipts.</p>
      </div>
      <div>
        <label htmlFor="email">Secretary email</label>
        <input id="email" type="email" />
        {errors.email && <p className="text-sm text-destructive">Enter an email address like name@example.com</p>}
      </div>
      <label className="flex items-center gap-2">
        <input type="checkbox" /> Send me the monthly fixtures digest
      </label>
      <textarea defaultValue="We train on Tuesdays." />
      <input type="submit" value="Create account" />
    </form>
  );
}
