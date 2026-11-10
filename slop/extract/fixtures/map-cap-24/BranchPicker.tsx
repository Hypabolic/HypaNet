const BRANCHES = [
  "Abbotsley",
  "Barrowden",
  "Caxby",
  "Dunholme",
  "Eskdale",
  "Fairlop",
  "Gorran",
  "Hesketh",
  "Islip",
  "Jevington",
  "Kelmarsh",
  "Lundy",
  "Marhamchurch",
  "Nettlecombe",
  "Oakhanger",
  "Pettistree",
  "Quenby",
  "Rufforth",
  "Stonesfield",
  "Tilshead",
  "Upwey",
  "Varteg",
  "Walberswick",
  "Yarlington",
  "Zeals",
  "Ashwell",
];

export function BranchPicker() {
  return (
    <label>
      Collect from
      <select name="branch">
        {BRANCHES.map((b) => (
          <option key={b} value={b}>
            {b}
          </option>
        ))}
      </select>
    </label>
  );
}
