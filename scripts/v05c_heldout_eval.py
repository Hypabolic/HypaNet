"""v0.5 Phase C one-time held-out evaluation and its val dry run (``agent_loop_lab.v05.phase_c_eval``).

    uv run python scripts/v05c_heldout_eval.py --dry-run-val --report-dir DIR [--phase-c-artifact PATH]
                                               [--limit-sessions N] [--workers W] [--force]
    uv run python scripts/v05c_heldout_eval.py --heldout --confirm-one-time [--workers W]

Columns: v0.5 Phase C (``weight_program/v0_5/phase-c/artifact.dlw``), v0.5 B, v0.5 r1, v0.3 A, v0.3 B, v0.3 C-all
seed0, and the prior always-PRODUCTIVE (reference). Pre-registration: ``experiments/v0_5/phase_c/heldout/README.md``.

**Dry run** (``--dry-run-val``): val only (``excluding_disclosed`` primary, ``all_val``); never reads the held-out
data or the attempt record. The report goes to ``DIR/v05c_heldout_dryrun_val.json`` with ``"official": false``
(``DIR`` may not be ``artifacts/reports/`` or inside it). Before the Phase C build exists, ``--phase-c-artifact PATH``
puts any ``.dlw`` (e.g. B's, ``weight_program/v0_5/phase-b/artifact.dlw``) in the Phase C column; the report then
says ``"phase_c_stand_in": true`` and its first note is a STAND-IN warning. On full val the r1 and B columns must
reproduce their val reports (fatal), and the v0.3 columns are compared with ``v03_sealed.json``'s val blocks.

**Held-out** (``--heldout --confirm-one-time``): runs once. Refuses if ``artifacts/reports/v05c_heldout.json`` or
the attempt record ``v05c_heldout.started.json`` exists (no ``--force``), or if the pre-registration, the held-out
manifest or the OOD disclosure's private record is missing, untracked or differs from HEAD. Order: columns (Phase
C's base must be B's artefact), manifest checks and held-out file hashes (nothing decoded), val with all checks,
the attempt record, then each held-out set is decoded, checked against the manifest (rows and encoding hashes)
and the disclosure (disjoint sessions), scored in memory, and the report is written atomically to
``artifacts/reports/v05c_heldout.json``. Metrics are printed only after that.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover - uv run puts the package on the path
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent_loop_lab.v03 import sealed as S  # noqa: E402
from agent_loop_lab.v05 import phase_c_data as P  # noqa: E402
from agent_loop_lab.v05 import phase_c_eval as E  # noqa: E402

MAX_WORKERS = 16


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-val", action="store_true", help="val only; unofficial report in --report-dir")
    mode.add_argument("--heldout", action="store_true", help="the one-time official held-out run (needs --confirm-one-time)")
    ap.add_argument("--confirm-one-time", action="store_true", help="required with --heldout")
    ap.add_argument("--report-dir", type=Path, help="dry run only: where the unofficial report goes")
    ap.add_argument("--phase-c-artifact", type=Path, help="dry run only: a stand-in .dlw for the Phase C column")
    ap.add_argument("--limit-sessions", type=int, help="dry run only: first N val sessions (skips the full-val checks)")
    ap.add_argument("--force", action="store_true", help="dry run only: overwrite an existing dry-run report")
    ap.add_argument("--workers", type=int, default=S.DEFAULT_WORKERS,
                    help=f"network worker processes, {S.TORCH_THREADS} torch threads each (default {S.DEFAULT_WORKERS})")
    args = ap.parse_args(argv)
    if args.heldout:
        if not args.confirm_one_time:
            ap.error("--heldout needs --confirm-one-time (the held-out evaluation runs once)")
        for flag, val in (("--report-dir", args.report_dir), ("--phase-c-artifact", args.phase_c_artifact),
                          ("--limit-sessions", args.limit_sessions), ("--force", args.force or None)):
            if val is not None:
                ap.error(f"{flag} is not allowed with --heldout")
    else:
        if args.confirm_one_time:
            ap.error("--confirm-one-time belongs to --heldout")
        if args.report_dir is None:
            ap.error("--dry-run-val needs --report-dir")
        problem = E.dry_run_dir_problem(args.report_dir)
        if problem:
            ap.error(problem)
        if args.limit_sessions is not None and args.limit_sessions < 1:
            ap.error("--limit-sessions must be >= 1")
    if not 0 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 0..{MAX_WORKERS} (0: in this process)")
    return args


def print_summary(report: dict[str, Any]) -> None:
    if report["phase_c_stand_in"]:
        print("NOTE: the 'v0.5 Phase C network' column is a STAND-IN (dry run)")
    for split, pops in report["summary"].items():
        for pop, t in pops.items():
            print(f"== {split} [{pop}]")
            for name, v in t["columns"].items():
                print(f"  {name:28s} F1 {v['macro_f1'][0]:.4f} {v['macro_f1'][1]}  FA {v['false_alarm_rate'][0]:.4f}")
            for name, d in t["paired"].items():
                f1, fa = d["macro_f1"], d["false_alarm_rate"]
                tag = "  [primary]" if (split, pop) == E.PRIMARY_POPULATION and name in E.PRIMARY_PAIRS else ""
                print(f"  {name:52s} dF1 {f1[0]:+.4f} {f1[1]}  dFA {fa[0]:+.4f} {fa[1]}{tag}")


def run(args: argparse.Namespace) -> int:
    official = bool(args.heldout)
    if official:
        reasons = E.refuse_reasons()
        if reasons:
            for r in reasons:
                print(f"refusing: {r}", file=sys.stderr)
            return 2
        out_path = E.HELDOUT_REPORT
    else:
        out_path = args.report_dir.resolve() / E.DRY_REPORT_NAME
        if out_path.exists() and not args.force:
            print(f"refusing: {out_path} exists (pass --force to overwrite a dry-run report)", file=sys.stderr)
            return 2
    torch.set_num_threads(S.TORCH_THREADS)
    meta = {"started_utc": _now(), "git": S.git_state(),
            "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                            "platform": platform.platform(), "torch_threads": S.TORCH_THREADS, "workers": args.workers}}
    _say("checking columns: v0.5 Phase C, B, r1, v0.3 A/B archives, v0.3 C-all seed0")
    cols = E.load_eval_columns(None if official else args.phase_c_artifact)
    if cols.phase_c_stand_in:
        _say(f"WARNING: Phase C column is a STAND-IN: {cols.info[E.C]['artifact_path']}")
    kwargs = dict(heldout=official, official=official, report_path=out_path,
                  marker_path=E.HELDOUT_MARKER if official else None, workers=args.workers,
                  val_limit=None if official else args.limit_sessions, progress=_say, meta=meta)
    try:
        if args.workers > 0:
            with S.make_pool(args.workers) as pool:
                report = E.evaluate(cols, pool=pool, **kwargs)
        else:
            report = E.evaluate(cols, **kwargs)
    except E.StageError as exc:
        print(f"{exc}. No held-out metric was written or printed.", file=sys.stderr)
        cause = exc.__cause__
        if isinstance(cause, (S.CheckFailed, S.SealedAccessError, P.DataError)):
            print(f"check: {cause}", file=sys.stderr)
        else:
            traceback.print_exception(cause, limit=8)
        return 3
    _say(f"wrote {out_path}")
    print_summary(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (S.CheckFailed, P.DataError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
