export function Voyage({ trip, crew }: { trip: { portOfCall: string; etaHours: number }; crew: string[] }) {
  return (
    <div>
      <h2>{trip.portOfCall}</h2>
      <p>Arriving in {trip.etaHours} hours with {crew.length} crew</p>
      <p>{t("voyage.weatherWarningShort")}</p>
      <p>Crew: {crew.join(", ")}</p>
    </div>
  );
}
