export function KilnStatus({ firing, n }: { firing: boolean; n: number }) {
  return (
    <div>
      <p>
        {firing ? (
          <>
            <span>
              {n} stages complete
            </span>
            <span>Cooling next</span>
          </>
        ) : (
          <span>Idle</span>
        )}
      </p>
      <p>
        <span>
          {n} stages complete
        </span>
        <span>Cooling next</span>
      </p>
    </div>
  );
}
