export function Upload({ uploadError, failed, warningNote }: { uploadError: string | null; failed: boolean; warningNote: string }) {
  return (
    <div className="space-y-2">
      <p className="text-sm">Drop a tide table here.</p>
      {failed && <p className="text-sm text-red-600">The table could not be read. Check the file and try again.</p>}
      {uploadError && <p className="text-xs font-mono">{uploadError}</p>}
      <p className="text-xs text-destructive">Files over 4 MB are refused.</p>
      <p className="text-xs text-amber-700">{warningNote}</p>
      <p className="text-sm text-red-600">● Live</p>
    </div>
  );
}
