export function KilnStatus({ done, total, firing, warn, eta }: { done: number; total: number; firing: boolean; warn: boolean; eta: string }) {
  return (
    <div>
      <p className="text-sm">
        {firing ? (
          <>
            <span>{done} of {total} stages complete</span>
            <span>· {eta} left</span>
          </>
        ) : (
          <span>Kiln idle</span>
        )}
      </p>
      <p className="text-sm">
        {firing ? (
          <>
            <span>Glaze firing</span>
            {warn && <span className="text-amber-600"> · check the vents</span>}
          </>
        ) : (
          "Ready to load"
        )}
      </p>
      <p>
        {firing ? <span>Heating</span> : <span>Cooling</span>}
        <span>· shelf 2</span>
      </p>
    </div>
  );
}
