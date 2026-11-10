export function Timetable({ rows }: Props) {
  return (
    <table className="w-full text-sm">
      <caption>Departures from Platform 2</caption>
      <thead>
        <tr>
          <th>Destination</th>
          <th>Due</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Penzance</td>
          <td>14:05</td>
        </tr>
        <tr>
          <td>Plymouth</td>
          <td>
            <span className="rounded-full bg-amber-100 px-2 text-xs">Delayed</span>
          </td>
        </tr>
      </tbody>
    </table>
  );
}
