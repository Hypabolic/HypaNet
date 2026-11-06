"""v0.5 validation scoring of a run's network (``agent_loop_lab.v05.scoring``).

    uv run python scripts/v05_score_validation.py RUN [--force] [--workers W]

RUN names ``experiments/v0_5/runs/RUN/`` (e.g. ``r1``). Its ``artifact.dlw`` (checked against
``artifact.sha256``; a ``run`` in the artefact metadata must be RUN) is scored as a network only,
on the **validation** split only, with v0.3's windows, gate, populations (``excluding_disclosed``,
primary, and ``all_val``) and metrics (macro-F1 with the v0.3 session bootstrap, per-class F1,
confusion matrix, false-alarm rate), beside the archived v0.3 Phase A network on the same draws
(its blocks must reproduce ``artifacts/reports/v03_val_v03-phase-a.json``) and the two networks'
argmax agreement. The split is hard-coded; no option, argument or environment variable changes it.

Writes ``artifacts/reports/v05_val_<RUN>.json`` and refuses to overwrite a report unless given
``--force``. The report is for the adjudicator only: a v0.5 author gets no feedback of any kind.

``--artifact PATH --report-dir DIR`` (both together, for tests and dry runs): score any ``.dlw``
instead (checked against ``artifact.sha256`` beside it when there is one). The split is still val.
The report goes to ``DIR``, which may not be ``artifacts/reports/``, and is marked
``"official": false``. ``--skip-v03-a`` (dry runs only) leaves out the v0.3 A network.

``--workers W`` runs the networks in W spawn worker processes of 2 torch threads each (default 8,
as the v0.3 sealed run); ``--workers 0`` runs them in this process (the v0.3 val script's path).
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

import torch  # noqa: E402

from agent_loop_lab.v05 import scoring  # noqa: E402

REPORT_DIR = scoring.REPORT_DIR


def _sha_file_beside(artifact: Path) -> Path | None:
    for cand in (artifact.parent / "artifact.sha256", artifact.with_suffix(".sha256")):
        if cand.is_file():
            try:
                scoring.read_sha256_file(cand, artifact.name)
            except scoring.ScoringError:
                continue
            return cand
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run name under experiments/v0_5/runs/ (e.g. r1)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing report")
    ap.add_argument("--workers", type=int, default=scoring.DEFAULT_WORKERS,
                    help="network worker processes (0 = in process)")
    ap.add_argument("--artifact", type=Path, help="tests / dry runs only: .dlw to score (split stays val)")
    ap.add_argument("--report-dir", type=Path, help="with --artifact: where the (unofficial) report goes")
    ap.add_argument("--skip-v03-a", action="store_true", help="dry runs only: leave out the v0.3 A network")
    args = ap.parse_args(argv)
    if (args.artifact is None) != (args.report_dir is None):
        ap.error("--artifact and --report-dir go together")
    official = args.artifact is None
    if official and args.skip_v03_a:
        ap.error("--skip-v03-a is for dry runs only (an official report always compares with v0.3 A)")
    if args.workers < 0:
        ap.error("--workers must be >= 0")
    try:
        scoring.check_run_name(args.run)
    except scoring.ScoringError as exc:
        ap.error(str(exc))
    if official:
        try:
            d = scoring.run_dir(args.run)
        except FileNotFoundError as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 2
        artifact, sha_file, report_dir = d / "artifact.dlw", d / "artifact.sha256", REPORT_DIR
    else:
        report_dir = args.report_dir.resolve()
        if report_dir == REPORT_DIR.resolve():
            ap.error("--report-dir may not be artifacts/reports/ (official reports come from experiments/v0_5/runs/ only)")
        artifact = args.artifact.resolve()
        if not artifact.is_file():
            ap.error(f"{artifact} is not a file")
        sha_file = _sha_file_beside(artifact)
    out = report_dir / scoring.REPORT_NAME.format(run=args.run)
    if out.exists() and not args.force:
        print(f"refusing: {out} exists (pass --force to overwrite)", file=sys.stderr)
        return 2
    torch.set_num_threads(scoring.TORCH_THREADS)
    t0 = time.time()
    rep = scoring.report(args.run, artifact=artifact, sha_file=sha_file, official=official,
                         include_v03_a=not args.skip_v03_a, workers=args.workers)
    report_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(f"{args.run}: {time.time() - t0:.1f}s -> {out}")
    lg = rep["network_logits"]
    print(f" logits: {lg['events']} events, {lg['nonfinite_events']} non-finite, {lg['tied_top_events']} tied top, "
          f"{lg['all_equal_events']} all equal; nonzero parameters {rep['nonzero_parameters']['nonzero']}")
    repro = rep["v03_phase_a_reproduction"]
    if repro is not None and not repro["reproduced"]:
        print(f" WARNING: the v0.3 A column differs from its official val report at {repro['first_difference']}",
              file=sys.stderr)
    for pop in scoring.POPULATIONS:
        s = rep["summary"][pop]
        x = s["network"]
        print(f" [{pop}] sessions {s['sessions']} scored windows {s['scored_windows']}")
        print(f"  network  macro-F1 {x['macro_f1']['value']:.4f} {x['macro_f1']['ci95']}"
              f"  false-alarm {x['false_alarm_rate']['value']:.4f} {x['false_alarm_rate']['ci95']}")
        if scoring.A_BLOCK in s:
            a = s[scoring.A_BLOCK]
            print(f"  v0.3 A   macro-F1 {a['macro_f1']['value']:.4f} {a['macro_f1']['ci95']}")
            g = s[scoring.AGREE_BLOCK]["all"]
            print(f"  agreement with v0.3 A (gate-passing windows) {g['value']:.4f} {g['ci95']}")
        for name, d in s["paired_macro_f1"].items():
            print(f"  {name:40s} {d['value']:+.4f} {d['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
