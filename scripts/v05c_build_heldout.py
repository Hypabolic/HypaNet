"""Generate the v0.5 Phase C held-out evaluation sets and write their manifest (``agent_loop_lab.v05.phase_c_data``).

    uv run python scripts/v05c_build_heldout.py
    uv run python scripts/v05c_build_heldout.py --data-root DIR --manifest PATH [--ood-per-family N]
                                                [--indist-sessions N]                          (dry run)

Default (official): writes ``datasets/v0_5c/heldout_ood/`` (``synth_ood``, master seed 20261110, 200 sessions per
family, all 10 families; every session re-timed by its per-session clock offset, ``phase_c_data.retime_session``)
and ``datasets/v0_5c/heldout_indist/`` (standard generator, master seed 20261111, 4,000 sessions), each as
``<set>.{events,oracle,raw}.jsonl.gz`` (gitignored by ``datasets/**/*.jsonl.gz``), and
``experiments/v0_5/phase_c/heldout/HELDOUT_MANIFEST.json`` (committed): every file's sha256, the window rows'
sha256, the token encoding's sha256, counts, seeds, the re-timing rule, the family rule and its 6/4 split, the
generator / renderer / tokeniser source sha256 and the disjointness checks. The seeds and sizes are fixed (no flag changes them in an
official run). The run is reproducible byte for byte (deterministic gzip, no timestamps).

Every generated session id is checked to be outside the reserved seed namespaces (20261104 sealed; 20261107 /
20261108 the OOD diagnosis; 20261109 the OOD disclosure, every candidate of it; ids computed from the seed formulas,
no data file is opened) and the two sets are checked to be disjoint. If
``experiments/v0_5/phase_c/DISCLOSURE_OOD_PRIVATE.json`` exists, its disclosed sessions are checked too.

Refuses to overwrite any existing data file or manifest. A dry run (``--data-root`` and ``--manifest``, both
required, neither inside the official locations) may use smaller sizes and writes ``"official": false``.
Run this BEFORE the Phase C launch; the held-out sets are never handed over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover - uv run puts the package on the path
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v05 import phase_c_data as P  # noqa: E402


def _inside(p: Path, root: Path) -> bool:
    p, root = p.resolve(), root.resolve()
    return p == root or root in p.parents


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, help="dry run: where the data files go")
    ap.add_argument("--manifest", type=Path, help="dry run: where the manifest goes")
    ap.add_argument("--ood-per-family", type=int, help="dry run only: OOD sessions per family")
    ap.add_argument("--indist-sessions", type=int, help="dry run only: in-distribution sessions")
    args = ap.parse_args(argv)
    dry = args.data_root is not None or args.manifest is not None
    if dry and (args.data_root is None or args.manifest is None):
        ap.error("a dry run needs both --data-root and --manifest")
    if not dry and (args.ood_per_family is not None or args.indist_sessions is not None):
        ap.error("--ood-per-family / --indist-sessions are dry-run options (the official sizes are fixed)")
    data_root = (args.data_root or P.DATA_ROOT).resolve()
    manifest = (args.manifest or P.MANIFEST_PATH).resolve()
    if dry:
        for p, what in ((data_root, "--data-root"), (manifest, "--manifest")):
            for root in (P.DATA_ROOT, P.MANIFEST_PATH.parent, REPO / "datasets" / "v0_2", REPO / "datasets" / "v0_3"):
                if _inside(p, root):
                    ap.error(f"{what} of a dry run may not be {root} or inside it")
    for n, flag in ((args.ood_per_family, "--ood-per-family"), (args.indist_sessions, "--indist-sessions")):
        if n is not None and n < 1:
            ap.error(f"{flag} must be >= 1")
    if manifest.exists():
        print(f"refusing: {manifest} exists", file=sys.stderr)
        return 2
    existing = [str(p) for name in P.SET_NAMES for p in P.set_paths(data_root, name).values() if p.exists()]
    if existing:
        print(f"refusing: {', '.join(existing)} exist(s)", file=sys.stderr)
        return 2
    t0 = time.time()
    kwargs = {}
    if args.ood_per_family is not None:
        kwargs["ood_per_family"] = args.ood_per_family
    if args.indist_sessions is not None:
        kwargs["indist_sessions"] = args.indist_sessions
    try:
        man = P.build_heldout(data_root, official=not dry, disclosure_private=P.DISCLOSURE_OOD_PRIVATE,
                              progress=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True), **kwargs)
    except (P.DataError, P.SeedError) as exc:
        print(f"ERROR: {exc} (data files removed, no manifest written)", file=sys.stderr)
        return 1
    if not dry:
        problems = P.manifest_problems(man, official=True)
        if problems:  # pragma: no cover - the builder uses the pre-registered constants
            print(f"ERROR: manifest problems {problems}", file=sys.stderr)
            return 1
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    for name in P.SET_NAMES:
        s = man["sets"][name]
        print(f"{name}: {s['sessions']} sessions, {s['windows']} windows, rows_sha256 {s['rows_sha256']}")
    print(f"seen families {man['family_rule']['seen']}; unseen {man['family_rule']['unseen']}")
    print(f"wrote {manifest} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
