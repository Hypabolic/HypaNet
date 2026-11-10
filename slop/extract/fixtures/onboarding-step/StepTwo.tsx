export function StepTwo() {
  return (
    <div className="mx-auto max-w-md">
      <p className="text-xs text-muted-foreground">Step 2 of 4</p>
      <h1 className="text-2xl font-semibold">Tell us about your kitchen</h1>
      <div className="mt-6 space-y-2">
        <label htmlFor="covers">Covers per night</label>
        <input id="covers" type="number" />
        <p className="text-xs text-muted-foreground">We use this to size your prep lists.</p>
      </div>
      <div className="mt-8 flex justify-between">
        <button>Back</button>
        <button>Continue</button>
      </div>
    </div>
  );
}
