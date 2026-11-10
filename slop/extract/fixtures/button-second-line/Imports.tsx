export function Imports({ busy }: { busy: boolean }) {
  return (
    <div className="flex flex-col gap-2">
      <button type="button" className="flex items-center gap-3">
        <div>
          <p className="text-xs font-medium">Attach a chart</p>
          <p className="text-[10px] text-stone-400">GPX, KML or a photo</p>
        </div>
      </button>
      <button type="button">
        {busy ? <span>Saving the route</span> : <span>Save the route</span>}
      </button>
      <button type="button">Plain button</button>
    </div>
  );
}
