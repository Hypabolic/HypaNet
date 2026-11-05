# WSP-B audit adjudication (PREREGISTRATION §10)

Session dd5e5940-2e7c-4007-a3b8-05927e29a4c9 (resumed), attempt wsp-b.20261105T205149Z, freeze f654d2c9 (registry
revision 1, D1). Audit: `wsp-b.stream.audit.json` (0 violations, 0 denials, 0 flags, 2 file findings; ledger
40/40 ok, 1,439,522 of 2,000,000 tokens, at most 2 edit states per text set).

Adjudicated by the operator (Claude, for Matthew) on 2026-11-06, before any readout of the edit.

## output/wsp-b/scratch/copy_libs.py:3: import of shutil

The script copies four of Fable's own WSP-A scenario libraries (`mk1.py`, `mk2.py`, `scen_e.py`, `scen_f.py`) with
`shutil.copyfile`, from `output/wsp-a/scratch/` (read-only) to `output/wsp-b/scratch/`. It also re-saves six of
its own request and meta JSON files. Every source and destination is inside the author dir and is Fable's own
work. Nothing is read from or written outside it. It ran under `./py` (tool result: "copied mk1.py …").

**Finding: rule breach with no effect on the evaluation.** The run stands. **Guard gap:** `shutil` is in
`V04_FORBIDDEN_MODULES`, whose comment says the guard refuses these at run time too, but this import was not
refused. Check `hn_guard` before WSP-F.

## output/wsp-b/scratch/scen_r.py:11: getattr() with a computed attribute name

`getattr(r, "win_" + cls)(v)` dispatches to the window builders (`win_prod`, `win_reco`, `win_stal`, `win_osc`)
of Fable's own scenario class `R` (from its own `mk3.py`). It touches no interpreter internals and no kernel or
`wsp` object.

**Finding: benign.**
