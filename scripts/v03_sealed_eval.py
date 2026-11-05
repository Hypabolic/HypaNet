"""v0.3 one-time sealed evaluation (spec v0.3 §10, E7, §14 V4; experiments/v0_3/sealed/README.md).

    uv run python scripts/v03_sealed_eval.py --dry-run-val --report-dir DIR [--limit-sessions N] [--workers W]
    uv run python scripts/v03_sealed_eval.py --sealed --confirm-one-time [--workers W]

**Dry run** (``--dry-run-val``): scores every v0.3 column on val (``excluding_disclosed`` primary,
``all_val``) and on the unsealed counterfactual set, with the code path of the sealed run. It never
opens test or OOD and never reads or writes the attempt record. The report goes to
``DIR/v03_sealed_dryrun_val.json`` (``DIR`` may not be ``artifacts/reports/``) and says
``"official": false``. On full val it also checks that the v0.3 A/B blocks reproduce the official
``artifacts/reports/v03_val_*.json`` exactly, and that each Phase C model reproduces its training
log's early-stopping macro-F1. ``--limit-sessions N`` (dry run only) keeps the first N sessions.

**Sealed** (``--sealed --confirm-one-time``): runs once. It refuses if
``artifacts/reports/v03_sealed.json`` or the attempt record ``v03_sealed.started.json`` exists
(there is no ``--force``). Order: every input hash; val and counterfactual with all checks; the
attempt record; then the sealed files are hashed, decoded, identity-checked against v0.2, scored
in memory, and the report is written atomically. Metrics are printed only after that.
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

DRY_REPORT_NAME = "v03_sealed_dryrun_val.json"
AUTHOR_PATHS = ("experiments/v0_3/SESSION_ID", "experiments/v0_3/logs/")
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


def _print_summary(summary: dict[str, Any]) -> None:
    for split, pops in summary.items():
        for pop, t in pops.items():
            print(f"== {split} [{pop}]")
            for name, v in t.items():
                if isinstance(v.get("macro_f1"), list):
                    fid = f"  fid {v['h2_fidelity_all'][0]:.4f}" if "h2_fidelity_all" in v else ""
                    print(f"  {name:34s} F1 {v['macro_f1'][0]:.4f} {v['macro_f1'][1]}  FA {v['false_alarm_rate'][0]:.4f}{fid}")
                else:
                    m = v["macro_f1"]
                    sd = "n/a" if m["sd"] is None else f"{m['sd']:.4f}"
                    print(f"  {name:34s} F1 {m['mean']:.4f} ± {sd}")


def run(args: argparse.Namespace) -> int:
    official = bool(args.sealed)
    limit = None if official else args.limit_sessions
    do_cf = official or not args.no_counterfactual
    if official:
        reasons = S.refuse_reasons()
        if reasons:
            for r in reasons:
                print(f"refusing: {r}", file=sys.stderr)
            return 2
        out_path = S.SEALED_REPORT
    else:
        out_path = args.report_dir.resolve() / DRY_REPORT_NAME
        if out_path.exists() and not args.force:
            print(f"refusing: {out_path} exists (pass --force to overwrite a dry-run report)", file=sys.stderr)
            return 2
    torch.set_num_threads(S.TORCH_THREADS)
    started = _now()
    t_start = time.time()
    runtime: dict[str, float] = {}
    git = S.git_state(AUTHOR_PATHS)
    previous = S.previous_attempts() if official else []

    # 1. Inputs (no sealed file is opened).
    _say("checking inputs: archives, Phase C runs, manifests, unsealed data files")
    cols = S.load_columns()
    data_sha = S.verify_unsealed_inputs(("dev", "val", "counterfactual") if do_cf else ("dev", "val"))
    v02 = json.loads(S.V02_REPORT.read_text())
    v02_sha = S.sha256_file(S.V02_REPORT)
    sealed_records = S.verify_sealed_hash_records(v02)
    prior = S.dev_prior()
    if prior["class_counts"] != v02["dev_prior"]["class_counts"] or prior["majority_index"] != v02["dev_prior"]["majority_index"]:
        raise S.CheckFailed("dev prior differs from m4_sealed_v02.json dev_prior")
    code_sha = S.code_hashes()
    disclosed = S.private_disclosed()
    runtime["inputs"] = time.time() - t_start
    _say(f"inputs ok: {len(cols.nets)} networks, {len(cols.policies)} policies ({runtime['inputs']:.0f}s)")

    splits: dict[str, Any] = {}
    checks: dict[str, Any] = {"identity_v02": {}, "prior_confirmation": {}}
    summary: dict[str, Any] = {}
    with S.make_pool(args.workers) as pool:
        # 2. Val.
        t = time.time()
        sd_val = S.build_split("val", limit=limit)
        _say(f"val: {len(sd_val.data.sessions)} sessions, {len(sd_val.data.rows)} windows ({time.time() - t:.0f}s)")
        if limit is None:
            if scoring.disclosed_sessions(sd_val.data.rows) != disclosed:
                raise S.CheckFailed("DISCLOSURE_PRIVATE.json differs from the Phase B README §2 rule")
            checks["disclosure"] = {"matches_rule": True}
        else:
            checks["disclosure"] = {"matches_rule": "skipped (limited dry run)"}
        checks["disclosure"] |= {"count": len(disclosed), "source": "experiments/v0_3/DISCLOSURE_PRIVATE.json",
                                 "sha256_of_sorted_ids": scoring._session_set_sha256(disclosed)}
        checks["identity_v02"]["val"] = S.identity_check(sd_val, v02, limit=limit)
        _say("val: identity with v0.2 rows ok")
        sp_val = S.predict_split(sd_val, cols, prior, pool, workers=args.workers, progress=_say)
        t = time.time()
        sc_val = S.score_split(sd_val, sp_val, cols, disclosed)
        checks["scoring_path"], bodies = S.scoring_path_check(sd_val, sp_val, sc_val, disclosed)
        if limit is None:
            checks["official_val_reports"] = S.official_val_check(bodies, cols, disclosed)
            checks["phase_c_es"] = S.phase_c_es_check(sd_val, sp_val, cols, disclosed)
            checks["prior_confirmation"]["val"] = S.prior_confirmation(sc_val, v02, "val", "all_val")
        else:
            for k in ("official_val_reports", "phase_c_es"):
                checks[k] = "skipped (limited dry run)"
        _say(f"val: scored and checked ({time.time() - t:.0f}s)")
        splits["val"] = {**sc_val, "latency": sp_val.latency, "policy_wall_s": sp_val.policy_wall_s}
        summary["val"] = {pop: S.summary_table(sc, cols) for pop, sc in sc_val["populations"].items()}
        lat_sample_ids, lat_sample = S.latency_sample(sd_val.data)
        del sd_val, sp_val, bodies

        # 3. Counterfactual (unsealed).
        if do_cf:
            t = time.time()
            sd_cf = S.build_split("counterfactual", limit=limit)
            _say(f"counterfactual: {len(sd_cf.data.sessions)} sessions, {len(sd_cf.data.rows)} windows, "
                 f"{len({m['pair_id'] for m in sd_cf.meta.values()})} pairs ({time.time() - t:.0f}s)")
            checks["identity_v02"]["counterfactual"] = S.identity_check(sd_cf, v02, limit=limit)
            sp_cf = S.predict_split(sd_cf, cols, prior, pool, workers=args.workers, progress=_say)
            splits["counterfactual"] = {**S.counterfactual_scores(sd_cf, sp_cf), "latency": sp_cf.latency,
                                        "policy_wall_s": sp_cf.policy_wall_s}
            del sd_cf, sp_cf
        # CPU latency with nothing else running (val sample, idle pool; README §5).
        t = time.time()
        latency_isolated = {"split": "val", "sessions": len(lat_sample),
                            "session_ids_sha256": S.hashlib.sha256("\n".join(lat_sample_ids).encode()).hexdigest(),
                            "choice": f"{S.LATENCY_SESSIONS} val sessions evenly spaced in file order",
                            "columns": S.isolated_latency(cols.nets, lat_sample)}
        runtime["latency_isolated"] = time.time() - t
        _say(f"isolated latency: {len(cols.nets)} networks x {len(lat_sample)} sessions ({runtime['latency_isolated']:.0f}s)")
        runtime["unsealed"] = time.time() - t_start - runtime["inputs"]
        json.dumps([splits, checks], allow_nan=False)  # fail before the attempt record if not serialisable

        attempt_info: dict[str, Any] = {}
        sealed_sha: dict[str, str] = {}
        if official:
            # 4. Attempt record, then the sealed splits.
            record = {"started_utc": started, "sealed_opened_utc": _now(), "git": git,
                      "attempt_number": len(previous) + 1, "previous_attempts": previous,
                      "inputs": cols.inputs, "data_sha256": data_sha, "code_sha256": code_sha,
                      "v02_report_sha256": v02_sha, "workers": args.workers,
                      "pre_registration": S.PRE_REGISTRATION}
            attempt = S.write_attempt_record(record)
            attempt_info = {"marker": str(S.SEALED_MARKER.relative_to(REPO)), "number": len(previous) + 1,
                            "previous_attempts": previous}
            _say(f"attempt record written: {attempt_info['marker']} (no sealed file opened before this)")
            t_sealed = time.time()
            try:
                sealed_sha = S.verify_sealed_files(attempt)
                _say("sealed files match their manifests")
                built = {}
                for split in S.SEALED_SPLITS:
                    t = time.time()
                    built[split] = S.build_split(split, attempt=attempt)
                    _say(f"{split}: {len(built[split].data.sessions)} sessions, {len(built[split].data.rows)} windows "
                         f"({time.time() - t:.0f}s)")
                    checks["identity_v02"][split] = S.identity_check(built[split], v02, attempt=attempt)
                    _say(f"{split}: identity with v0.2 ok")
                for split in S.SEALED_SPLITS:
                    sd = built.pop(split)
                    sp = S.predict_split(sd, cols, prior, pool, workers=args.workers, progress=_say)
                    t = time.time()
                    sc = S.score_split(sd, sp, cols, frozenset())
                    _say(f"{split}: scored ({time.time() - t:.0f}s)")
                    checks["prior_confirmation"][split] = S.prior_confirmation(sc, v02, split, "all")
                    splits[split] = {**sc, "latency": sp.latency, "policy_wall_s": sp.policy_wall_s}
                    summary[split] = {pop: S.summary_table(x, cols) for pop, x in sc["populations"].items()}
                    del sd, sp
            except BaseException as exc:
                print(f"sealed stage stopped ({type(exc).__name__}); the attempt record stays as a spent attempt "
                      f"(README §6). No sealed metric was written or printed.", file=sys.stderr)
                if isinstance(exc, (S.CheckFailed, S.SealedAccessError)):
                    print(f"check: {exc}", file=sys.stderr)
                else:
                    traceback.print_exc(limit=8)
                return 3
            runtime["sealed"] = time.time() - t_sealed

    order = [s for s in ("test", "ood", "val", "counterfactual") if s in splits]
    report = {
        "report": "v03_sealed" if official else "v03_sealed_dryrun_val",
        "official": official,
        "spec_version": "0.3",
        "pre_registration": S.PRE_REGISTRATION,
        "authority": "spec v0.3 E7: v0.3 sealed scoring runs once, after the v0.3 Phase B archive",
        "started_utc": started,
        "finished_utc": _now(),
        "attempt": attempt_info or None,
        "limit_sessions": limit,
        "git": git,
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "torch_threads": S.TORCH_THREADS, "workers": args.workers},
        "inputs": {"columns": cols.inputs, "data_sha256": {**data_sha, **sealed_sha},
                   "sealed_hash_records": sealed_records, "code_sha256": code_sha,
                   "v02_report_sha256": v02_sha,
                   "disclosure_private_sha256": S.sha256_file(S.PRIVATE_DISCLOSURE)},
        "dev_prior": prior,
        "columns": cols.info,
        "phase_c_groups": cols.groups,
        "fidelity_pairs": {k: list(v) for k, v in cols.fid_pairs.items()},
        "populations": {"test": "all sessions", "ood": "all sessions", **scoring.POPULATIONS},
        "policy_failure_rule": ("a reference_policy call that times out (0.5 s CPU), raises or returns a non-int / "
                                "out-of-range value is a failure: an H2 disagreement, and in H1 a wrong prediction "
                                "recorded as the lowest class index that is neither the network class nor (scored "
                                "windows) the oracle class"),
        "checks": checks,
        "summary": {s: summary[s] for s in order if s in summary},
        "splits": {s: splits[s] for s in order},
        "latency_isolated": latency_isolated,
        "v02_cited": S.cite_v02(v02),
        "notes": [
            "val is reference only: Phase C early-stopped on 500 val sessions, and the 200 Phase B disclosed sessions are val sessions",
            "Phase C budgets 2048 and all have one seed; their sd is null",
            "columns without a reference policy (Phase C, priors) have no H2 fidelity",
            "no random-window H2 sampler: spec v0.3 r1 measures H2 on benchmark windows only",
        ],
        "runtime_s": {**runtime, "total": time.time() - t_start},
    }
    S.write_json_atomic(out_path, report)
    _say(f"wrote {out_path}")
    _print_summary(report["summary"])
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
