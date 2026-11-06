"""v0.5 paired validation comparison and one-time sealed evaluation (``agent_loop_lab.v05.sealed``).

    uv run python scripts/v05_sealed_eval.py --dry-run-val --report-dir DIR [--limit-sessions N] [--workers W]
    uv run python scripts/v05_sealed_eval.py --sealed --confirm-one-time [--workers W]

Columns: the v0.5 r1 network (``experiments/v0_5/runs/r1/artifact.dlw``) and the v0.5 Phase B network
(``weight_program/v0_5/phase-b/artifact.dlw``), beside every v0.3 column of the v0.3 sealed run
(A and B networks and reference policies, every Phase C model, the two priors), all on the same
bootstrap draws. Paired differences (macro-F1 and false-alarm rate, shared session bootstrap):
v0.5 B - v0.3 B, v0.5 B - v0.3 A, v0.5 B - v0.3 C-all, v0.5 B - v0.5 r1, v0.5 r1 - v0.3 A.

**Dry run** (``--dry-run-val``): val (``excluding_disclosed`` primary, ``all_val``) and the unsealed
counterfactual set, with the code path of the sealed run. It never opens test or OOD and never
reads or writes the attempt record. The report goes to ``DIR/v05_sealed_dryrun_val.json`` (``DIR``
may not be ``artifacts/reports/``) and says ``"official": false``. On full val it checks that the
v0.5 r1 and B blocks reproduce their val reports and the v0.3 A/B blocks reproduce
``artifacts/reports/v03_val_*.json`` exactly and that every Phase C model's early-stopping macro-F1
equals its training log (v0.3's check; a failure stops the run), and records whether every
v0.3 column reproduces ``artifacts/reports/v03_sealed.json``. ``--limit-sessions N`` keeps the
first N sessions of val and counterfactual and skips the full-val checks.

**Sealed** (``--sealed --confirm-one-time``): runs once. It refuses if
``artifacts/reports/v05_sealed.json`` or the attempt record ``v05_sealed.started.json`` exists
(there is no ``--force``), if ``v03_sealed.json`` is missing, or if the pre-registration
``experiments/v0_5/sealed/README.md`` is missing, untracked or differs from HEAD. Order: every
input hash (the v0.3 columns, code, data files and the sealed files' registered hashes, all equal
to those of the v0.3 sealed run; the v0.5 artefacts and records, B's base equal to r1); val and counterfactual with all checks; the attempt
record; then the sealed files are hashed, decoded, identity-checked against v0.2 and against the
window rows of the v0.3 sealed run, scored in memory, and the report is written atomically to
``artifacts/reports/v05_sealed.json``. Metrics and the reproduction status of the v0.3 columns are
printed only after that.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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

from agent_loop_lab.v03 import scoring  # noqa: E402
from agent_loop_lab.v03 import sealed as S  # noqa: E402
from agent_loop_lab.v04 import scoring as v04_scoring  # noqa: E402
from agent_loop_lab.v05 import sealed as V  # noqa: E402

DRY_REPORT_NAME = "v05_sealed_dryrun_val.json"
MAX_WORKERS = 16


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-val", action="store_true", help="val + counterfactual only; unofficial report in --report-dir")
    mode.add_argument("--sealed", action="store_true", help="the one-time official sealed run (needs --confirm-one-time)")
    ap.add_argument("--confirm-one-time", action="store_true", help="required with --sealed")
    ap.add_argument("--report-dir", type=Path, help="dry run only: where the unofficial report goes")
    ap.add_argument("--limit-sessions", type=int, help="dry run only: first N sessions of val and counterfactual")
    ap.add_argument("--no-counterfactual", action="store_true", help="dry run only: skip the counterfactual set")
    ap.add_argument("--force", action="store_true", help="dry run only: overwrite an existing dry-run report")
    ap.add_argument("--workers", type=int, default=S.DEFAULT_WORKERS,
                    help=f"network worker processes, {S.TORCH_THREADS} torch threads each (default {S.DEFAULT_WORKERS})")
    args = ap.parse_args(argv)
    if args.sealed:
        if not args.confirm_one_time:
            ap.error("--sealed needs --confirm-one-time (the sealed evaluation runs once)")
        for flag, val in (("--report-dir", args.report_dir), ("--limit-sessions", args.limit_sessions),
                          ("--no-counterfactual", args.no_counterfactual or None), ("--force", args.force or None)):
            if val is not None:
                ap.error(f"{flag} is not allowed with --sealed")
    else:
        if args.confirm_one_time:
            ap.error("--confirm-one-time belongs to --sealed")
        if args.report_dir is None:
            ap.error("--dry-run-val needs --report-dir")
        if args.report_dir.resolve() == S.REPORT_DIR.resolve():
            ap.error("--report-dir may not be artifacts/reports/ (official reports come from the sealed run only)")
        if args.limit_sessions is not None and args.limit_sessions < 1:
            ap.error("--limit-sessions must be >= 1")
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 1..{MAX_WORKERS}")
    return args


def _print_summary(summary: dict[str, Any], repro: dict[str, Any]) -> None:
    for split, pops in summary.items():
        for pop, t in pops.items():
            print(f"== {split} [{pop}]")
            for name, v in t["columns"].items():
                if isinstance(v.get("macro_f1"), list):
                    fid = f"  fid {v['h2_fidelity_all'][0]:.4f}" if "h2_fidelity_all" in v else ""
                    print(f"  {name:34s} F1 {v['macro_f1'][0]:.4f} {v['macro_f1'][1]}  FA {v['false_alarm_rate'][0]:.4f}{fid}")
                else:
                    m = v["macro_f1"]
                    sd = "n/a" if m["sd"] is None else f"{m['sd']:.4f}"
                    print(f"  {name:34s} F1 {m['mean']:.4f} ± {sd}")
            for name, d in t["paired"].items():
                f1, fa = d["macro_f1"], d["false_alarm_rate"]
                print(f"  {name:40s} dF1 {f1[0]:+.4f} {f1[1]}  dFA {fa[0]:+.4f} {fa[1]}")
    for split, r in repro.items():
        if not r.get("compared"):
            print(f"v0.3 columns vs v03_sealed.json [{split}]: not compared ({r.get('reason')})")
        elif r["reproduced"]:
            print(f"v0.3 columns vs v03_sealed.json [{split}]: reproduced ({r['items_compared']} items)")
        else:
            print(f"WARNING: v0.3 columns vs v03_sealed.json [{split}]: NOT reproduced: {r['not_reproduced']}",
                  file=sys.stderr)
            for k, d in r["first_difference"].items():
                print(f"  {k}: first difference at {d}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    official = bool(args.sealed)
    limit = None if official else args.limit_sessions
    do_cf = official or not args.no_counterfactual
    if official:
        reasons = V.refuse_reasons()
        if reasons:
            for r in reasons:
                print(f"refusing: {r}", file=sys.stderr)
            return 2
        out_path = V.SEALED_REPORT
    else:
        out_path = args.report_dir.resolve() / DRY_REPORT_NAME
        if out_path.exists() and not args.force:
            print(f"refusing: {out_path} exists (pass --force to overwrite a dry-run report)", file=sys.stderr)
            return 2
    torch.set_num_threads(S.TORCH_THREADS)
    started = _now()
    t_start = time.time()
    runtime: dict[str, float] = {}
    git = S.git_state()
    previous = V.previous_attempts() if official else []

    # 1. Inputs (no sealed file is opened).
    _say("checking inputs: v0.3 archives and Phase C runs, v0.5 artefacts, manifests, unsealed data files")
    cols = S.load_columns()
    v05 = V.load_v05_columns()
    data_sha = S.verify_unsealed_inputs(("dev", "val", "counterfactual") if do_cf else ("dev", "val"))
    v02 = json.loads(S.V02_REPORT.read_text())
    v02_sha = S.sha256_file(S.V02_REPORT)
    sealed_records = S.verify_sealed_hash_records(v02)
    v03_rep = json.loads(V.V03_SEALED_REPORT.read_text())
    v03_rep_sha = S.sha256_file(V.V03_SEALED_REPORT)
    checks: dict[str, Any] = {"identity_v02": {}, "prior_confirmation": {}, "v03_sealed_reproduction": {}}
    checks["v03_sealed_inputs"] = V.check_against_v03_sealed_inputs(v03_rep, cols, data_sha, sealed_records)
    prior = S.dev_prior()
    if prior["class_counts"] != v02["dev_prior"]["class_counts"] or prior["majority_index"] != v02["dev_prior"]["majority_index"]:
        raise S.CheckFailed("dev prior differs from m4_sealed_v02.json dev_prior")
    if S.canon(prior) != S.canon(v03_rep["dev_prior"]):
        raise S.CheckFailed("dev prior differs from v03_sealed.json dev_prior")
    code_sha = V.code_hashes()
    disclosed = S.private_disclosed()
    checks["v05_disclosure"] = V.check_v05_disclosure(disclosed)
    runtime["inputs"] = time.time() - t_start
    _say(f"inputs ok: {len(cols.nets)} v0.3 networks, {len(v05.specs)} v0.5 networks, {len(cols.policies)} policies "
         f"({runtime['inputs']:.0f}s)")

    splits: dict[str, Any] = {}
    logits: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    with S.make_pool(args.workers) as pool:
        # 2. Val.
        t = time.time()
        sd_val = S.build_split("val", limit=limit)
        _say(f"val: {len(sd_val.data.sessions)} sessions, {len(sd_val.data.rows)} windows ({time.time() - t:.0f}s)")
        if limit is None:
            if scoring.disclosed_sessions(sd_val.data.rows) != disclosed:
                raise S.CheckFailed("DISCLOSURE_PRIVATE.json differs from the Phase B README §2 rule")
            if S.rows_sha256(sd_val.data.rows) != v04_scoring.VAL_ROWS_SHA256:
                raise S.CheckFailed("val rows differ from the v0.3 sealed record (v04.scoring.VAL_ROWS_SHA256)")
            checks["disclosure"] = {"matches_rule": True}
        else:
            checks["disclosure"] = {"matches_rule": "skipped (limited dry run)"}
        checks["disclosure"] |= {"count": len(disclosed), "source": "experiments/v0_3/DISCLOSURE_PRIVATE.json",
                                 "sha256_of_sorted_ids": scoring._session_set_sha256(disclosed)}
        checks["identity_v02"]["val"] = S.identity_check(sd_val, v02, limit=limit)
        _say("val: identity with v0.2 rows ok")
        sp_val, logits["val"] = V.predict_all(sd_val, cols, v05, prior, pool, workers=args.workers, progress=_say)
        t = time.time()
        sc_val = V.score_all(sd_val, sp_val, cols, disclosed)
        checks["scoring_path"], bodies = S.scoring_path_check(sd_val, sp_val, sc_val, disclosed)
        if limit is None:
            checks["official_val_reports"] = S.official_val_check(bodies, cols, disclosed)
            checks["v05_val_reports"] = V.v05_val_check(sd_val, sp_val, sc_val, v05, disclosed)
            checks["phase_c_es"] = S.phase_c_es_check(sd_val, sp_val, cols, disclosed)
            checks["prior_confirmation"]["val"] = S.prior_confirmation(sc_val, v02, "val", "all_val")
            checks["v03_sealed_reproduction"]["val"] = V.compare_with_v03_sealed("val", sc_val, v03_rep["splits"].get("val"), cols)
            V.caveat_pairs(sc_val, checks["v03_sealed_reproduction"]["val"])
        else:
            for k in ("official_val_reports", "v05_val_reports", "phase_c_es"):
                checks[k] = "skipped (limited dry run)"
            checks["v03_sealed_reproduction"]["val"] = {"compared": False, "reason": "skipped (limited dry run)"}
        _say(f"val: scored and checked ({time.time() - t:.0f}s)")
        splits["val"] = {**sc_val, "latency": sp_val.latency, "policy_wall_s": sp_val.policy_wall_s}
        summary["val"] = V.summary(sc_val, cols)
        lat_sample_ids, lat_sample = S.latency_sample(sd_val.data)
        del sd_val, sp_val, bodies

        # 3. Counterfactual (unsealed; v0.3's builder only).
        if do_cf:
            t = time.time()
            sd_cf = S.build_split("counterfactual", limit=limit)
            _say(f"counterfactual: {len(sd_cf.data.sessions)} sessions, {len(sd_cf.data.rows)} windows, "
                 f"{len({m['pair_id'] for m in sd_cf.meta.values()})} pairs ({time.time() - t:.0f}s)")
            checks["identity_v02"]["counterfactual"] = S.identity_check(sd_cf, v02, limit=limit)
            sp_cf, logits["counterfactual"] = V.predict_all(sd_cf, cols, v05, prior, pool, workers=args.workers,
                                                            progress=_say)
            splits["counterfactual"] = {**S.counterfactual_scores(sd_cf, sp_cf), "latency": sp_cf.latency,
                                        "policy_wall_s": sp_cf.policy_wall_s}
            checks["v03_sealed_reproduction"]["counterfactual"] = (
                V.compare_counterfactual_with_v03_sealed(splits["counterfactual"], v03_rep["splits"].get("counterfactual"), cols)
                if limit is None else {"compared": False, "reason": "skipped (limited dry run)"})
            del sd_cf, sp_cf
        # CPU latency with nothing else running (val sample, idle pool; v0.3 README §5).
        t = time.time()
        specs = [*v05.specs, *cols.nets]
        latency_isolated = {"split": "val", "sessions": len(lat_sample),
                            "session_ids_sha256": S.hashlib.sha256("\n".join(lat_sample_ids).encode()).hexdigest(),
                            "choice": f"{S.LATENCY_SESSIONS} val sessions evenly spaced in file order",
                            "columns": S.isolated_latency(specs, lat_sample)}
        runtime["latency_isolated"] = time.time() - t
        _say(f"isolated latency: {len(specs)} networks x {len(lat_sample)} sessions ({runtime['latency_isolated']:.0f}s)")
        runtime["unsealed"] = time.time() - t_start - runtime["inputs"]
        json.dumps([splits, checks, logits], allow_nan=False)  # fail before the attempt record if not serialisable

        attempt_info: dict[str, Any] = {}
        sealed_sha: dict[str, str] = {}
        if official:
            # 4. Attempt record, then the sealed splits.
            record = {"started_utc": started, "sealed_opened_utc": _now(), "git": git,
                      "attempt_number": len(previous) + 1, "previous_attempts": previous,
                      "inputs": {"v03_columns": cols.inputs, "v05_columns": v05.inputs},
                      "data_sha256": data_sha, "sealed_hash_records": sealed_records, "code_sha256": code_sha,
                      "v02_report_sha256": v02_sha, "v03_sealed_report_sha256": v03_rep_sha,
                      "workers": args.workers, "pre_registration": V.pre_registration()}
            attempt = S.write_attempt_record(record, V.SEALED_MARKER)
            attempt_info = {"marker": V._rel(V.SEALED_MARKER), "number": len(previous) + 1,
                            "previous_attempts": previous}
            _say(f"attempt record written: {attempt_info['marker']} (no sealed file opened before this)")
            t_sealed = time.time()
            try:
                sealed_sha = S.verify_sealed_files(attempt)
                if sealed_sha != sealed_records:
                    raise S.CheckFailed("sealed files' hashes differ from their registered hashes")
                _say("sealed files match their manifests")
                built = {}
                for split in S.SEALED_SPLITS:
                    t = time.time()
                    built[split] = S.build_split(split, attempt=attempt)
                    _say(f"{split}: {len(built[split].data.sessions)} sessions, {len(built[split].data.rows)} windows "
                         f"({time.time() - t:.0f}s)")
                    checks["identity_v02"][split] = S.identity_check(built[split], v02, attempt=attempt)
                    want = v03_rep["checks"]["identity_v02"][split]["rows_sha256"]
                    if checks["identity_v02"][split]["rows_sha256"] != want:
                        raise S.CheckFailed(f"{split}: window rows differ from those of the v0.3 sealed run")
                    checks["identity_v02"][split]["matches_v03_sealed_rows"] = True
                    _say(f"{split}: identity with v0.2 and the v0.3 sealed rows ok")
                for split in S.SEALED_SPLITS:
                    sd = built.pop(split)
                    sp, logits[split] = V.predict_all(sd, cols, v05, prior, pool, workers=args.workers, progress=_say)
                    t = time.time()
                    sc = V.score_all(sd, sp, cols, frozenset())
                    checks["prior_confirmation"][split] = S.prior_confirmation(sc, v02, split, "all")
                    checks["v03_sealed_reproduction"][split] = V.compare_with_v03_sealed(
                        split, sc, v03_rep["splits"].get(split), cols)
                    V.caveat_pairs(sc, checks["v03_sealed_reproduction"][split])
                    _say(f"{split}: scored ({time.time() - t:.0f}s)")
                    splits[split] = {**sc, "latency": sp.latency, "policy_wall_s": sp.policy_wall_s}
                    summary[split] = V.summary(sc, cols)
                    del sd, sp
            except BaseException as exc:
                print(f"sealed stage stopped ({type(exc).__name__}); the attempt record stays as a spent attempt. "
                      f"No sealed metric was written or printed.", file=sys.stderr)
                if isinstance(exc, (S.CheckFailed, S.SealedAccessError)):
                    print(f"check: {exc}", file=sys.stderr)
                else:
                    traceback.print_exc(limit=8)
                return 3
            runtime["sealed"] = time.time() - t_sealed

    repro = checks["v03_sealed_reproduction"]
    not_repro = {s: r["not_reproduced"] for s, r in repro.items() if r.get("compared") and not r["reproduced"]}
    order = [s for s in ("test", "ood", "val", "counterfactual") if s in splits]
    notes = [
        ("val is reference only: v0.3 Phase C early-stopped on 500 val sessions; the 200 disclosed sessions are val "
         "sessions, and v0.5 Phase B was revised with 200 labelled windows from them (v0.3's disclosure)"),
        ("v0.3 sealed results were already published (v03_sealed.json); the v0.3 columns here are re-scored on this "
         "machine and compared with that report block by block (checks.v03_sealed_reproduction)"),
        "v0.5 networks have no reference policy, so no H2 fidelity; paired differences are overall, not per family",
        "Phase C budgets 2048 and all have one seed; their sd is null",
    ]
    if not_repro:
        notes.insert(0, "WARNING: v0.3 columns that did NOT reproduce v03_sealed.json on this machine: "
                        + "; ".join(f"{s}: {', '.join(v)}" for s, v in not_repro.items())
                        + ". Paired differences that use them carry a caveat.")
    report = {
        "report": "v05_sealed" if official else "v05_sealed_dryrun_val",
        "official": official,
        "spec_version": "0.5",
        "pre_registration": V.pre_registration(),
        "authority": ("adjudicator request 2026-11-06: paired val comparison of v0.5 Phase B against v0.3, and a "
                      "one-time evaluation of the v0.5 networks on v0.3's sealed test and OOD sets"),
        "protocol": "v0.3 sealed protocol (experiments/v0_3/sealed/README.md), code reused by import",
        "started_utc": started,
        "finished_utc": _now(),
        "attempt": attempt_info or None,
        "limit_sessions": limit,
        "git": git,
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "torch_threads": S.TORCH_THREADS, "workers": args.workers},
        "inputs": {"v03_columns": cols.inputs, "v05_columns": v05.inputs, "data_sha256": {**data_sha, **sealed_sha},
                   "sealed_hash_records": sealed_records, "code_sha256": code_sha, "v02_report_sha256": v02_sha,
                   "v03_sealed_report_sha256": v03_rep_sha,
                   "disclosure_private_sha256": S.sha256_file(S.PRIVATE_DISCLOSURE)},
        "dev_prior": prior,
        "columns": {**v05.info, **cols.info},
        "phase_c_groups": cols.groups,
        "fidelity_pairs": {k: list(v) for k, v in cols.fid_pairs.items()},
        "pairs": {k: list(v) for k, v in V.PAIRS.items()},
        "paired_definition": ("metric(minuend) - metric(subtrahend) over the shared session-bootstrap draws of the "
                              "population (overall): macro_f1 by v04.scoring.paired_macro_f1, false_alarm_rate by the "
                              "same computation (v05.sealed.paired_metric)"),
        "populations": {"test": "all sessions", "ood": "all sessions", **scoring.POPULATIONS},
        "v03_sealed_reproduced": (not not_repro) if any(r.get("compared") for r in repro.values()) else None,
        "checks": checks,
        "summary": {s: summary[s] for s in order if s in summary},
        "splits": {s: splits[s] for s in order},
        "v05_logits": logits,
        "latency_isolated": latency_isolated,
        "notes": notes,
        "runtime_s": {**runtime, "total": time.time() - t_start},
    }
    S.write_json_atomic(out_path, report)
    _say(f"wrote {out_path}")
    _print_summary(report["summary"], repro)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except S.CheckFailed as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
