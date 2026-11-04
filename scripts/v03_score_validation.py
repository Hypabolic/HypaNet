"""v0.3 validation scoring of an archived phase (spec v0.3 §10, E11; experiments/v0_3/phase_b/README.md §1).

    uv run python scripts/v03_score_validation.py PHASE [--force]

PHASE is ``v03-phase-a`` or ``v03-phase-b``. The archived ``artifact.dlw`` (checked against
``artifact.sha256`` and its metadata) and ``reference_policy.py`` are loaded from
``weight_program/v0_3/phase-a/`` or ``.../phase-b/``, never from the author directory, and scored on
the **validation** split only. The split is hard-coded (``scoring.SPLIT``); no option, argument
or environment variable changes it. The sealed test and OOD sets are scored once, at the times E7
sets, by a separate script.

Writes ``artifacts/reports/v03_val_<PHASE>.json``, with both populations, ``excluding_disclosed``
(primary) and ``all_val``. The script refuses to overwrite a report unless given ``--force``. The
reports are for the adjudicator only: no aggregate validation metric is ever given to the weight
author.

``--weight-dir DIR --report-dir DIR`` (both together, for tests and dry runs only): score the
archive in ``DIR`` instead. The split is still val. The report goes to ``--report-dir``, which
may not be ``artifacts/reports/``, and is marked ``"official": false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover - uv run puts the package on the path
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent_loop_lab.eval import metrics  # noqa: E402
from agent_loop_lab.v03 import artifact as v03_artifact  # noqa: E402
from agent_loop_lab.v03 import scoring  # noqa: E402

REPORT_DIR = REPO / "artifacts" / "reports"
REPORT_NAME = "v03_val_{phase}.json"
PRIVATE_DISCLOSURE = REPO / "experiments" / "v0_3" / "DISCLOSURE_PRIVATE.json"  # v03_build_disclosure.py


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_private_disclosure(disclosed: frozenset[str], path: Path = PRIVATE_DISCLOSURE) -> bool | None:
    """True if the recorded disclosure covers exactly ``disclosed``; None if none is recorded yet."""
    if not path.is_file():
        return None
    rec = json.loads(path.read_text())
    recorded = frozenset(w["session_id"] for w in rec["windows"])
    if recorded != disclosed:
        raise ValueError(f"{path}: disclosed sessions differ from the README §2 rule")
    return True


def score_phase(phase: str, data: scoring.ValData, *, weight_dir: Path | None = None,
                n_resamples: int = metrics.BOOTSTRAP_RESAMPLES, progress=print) -> dict[str, Any]:
    d = scoring.phase_dir(phase, weight_dir)
    model, meta, pol, shas = scoring.load_phase(phase, d)
    t0 = time.time()
    net, latency = scoring.network_predictions(model, data)
    progress(f"  network: {len(data.sessions)} sessions in {time.time() - t0:.1f}s")
    t1 = time.time()
    ref = scoring.policy_predictions(pol, data, progress=lambda m: progress("  " + m))
    t_pol = time.time() - t1
    progress(f"  reference_policy: {int((ref != -1).sum())} calls in {t_pol:.1f}s")
    gate = np.array([r.gate_passed for r in data.rows])
    if np.any(net[gate] < 0) or np.any(ref[gate] == -1):
        raise RuntimeError("a gate-passing window has no prediction")
    disclosed = scoring.disclosed_sessions(data.rows)
    matches = _check_private_disclosure(disclosed) if weight_dir is None else None
    t2 = time.time()
    body = scoring.score(data, net, ref, n_resamples=n_resamples, disclosed=disclosed)
    progress(f"  metrics + bootstrap: {time.time() - t2:.1f}s")
    body["disclosed_sessions"]["matches_DISCLOSURE_PRIVATE"] = matches
    rel = (lambda p: str(p.relative_to(REPO))) if weight_dir is None else str
    return {
        "report": "v03_val",
        "spec_version": "0.3",
        "phase": phase,
        "split": scoring.SPLIT,
        "official": weight_dir is None,
        "inputs": {
            "archive_dir": rel(d),
            **shas,
            "artifact_metadata": meta,
            "dataset": data.inputs["cache_key"],
            "scoring_code_sha256": {
                "scripts/v03_score_validation.py": _sha256_file(Path(__file__).resolve()),
                "src/agent_loop_lab/v03/scoring.py": _sha256_file(Path(scoring.__file__).resolve()),
                "src/agent_loop_lab/eval/metrics.py": _sha256_file(Path(metrics.__file__).resolve()),
            },
        },
        "network_latency": latency,
        "reference_policy_wall_s": t_pol,
        "nonzero_parameters": v03_artifact.sparsity(model)["overall"],
        "policy_failure_rule": (
            "a reference_policy call that times out (0.5 s CPU), raises or returns a non-int / out-of-range value "
            "is a failure: an H2 disagreement, and in H1 a wrong prediction recorded as the lowest class index "
            "that is neither the network class nor (scored windows) the oracle class"
        ),
        **body,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=sorted(scoring.PHASE_DIRS))
    ap.add_argument("--force", action="store_true", help="overwrite an existing report")
    ap.add_argument("--weight-dir", type=Path, help="tests / dry runs only: archive directory to score (split stays val)")
    ap.add_argument("--report-dir", type=Path, help="with --weight-dir: where the (unofficial) report goes")
    args = ap.parse_args(argv)
    if (args.weight_dir is None) != (args.report_dir is None):
        ap.error("--weight-dir and --report-dir go together")
    report_dir = REPORT_DIR
    if args.weight_dir is not None:
        report_dir = args.report_dir.resolve()
        if report_dir == REPORT_DIR.resolve():
            ap.error("--report-dir may not be artifacts/reports/ (official reports come from the archive only)")
    torch.set_num_threads(scoring.TORCH_THREADS)
    scoring.phase_dir(args.phase, args.weight_dir)
    out = report_dir / REPORT_NAME.format(phase=args.phase)
    if out.exists() and not args.force:
        print(f"refusing: {out} exists (pass --force to overwrite)", file=sys.stderr)
        return 2
    t0 = time.time()
    data = scoring.load_val()
    print(f"{scoring.SPLIT}: {len(data.sessions)} sessions, {len(data.rows)} windows loaded in {time.time() - t0:.1f}s")
    t1 = time.time()
    rep = score_phase(args.phase, data, weight_dir=args.weight_dir)
    rep["runtime_s"] = {"load": t1 - t0, "score": time.time() - t1}
    report_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(f"{args.phase}: {time.time() - t1:.1f}s -> {out}")
    for pop in ("excluding_disclosed", "all_val"):
        s = rep["summary"][pop]
        print(f" [{pop}] sessions {s['sessions']} scored windows {s['scored_windows']} policy failures {s['policy_failures']}")
        for m in ("network", "reference_policy"):
            x = s[m]
            print(f"  {m:16s} macro-F1 {x['macro_f1']['value']:.4f} {x['macro_f1']['ci95']}"
                  f"  false-alarm {x['false_alarm_rate']['value']:.4f} {x['false_alarm_rate']['ci95']}")
        h = s["h2_fidelity_all"]
        print(f"  h2 fidelity      {h['value']:.4f} {h['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
