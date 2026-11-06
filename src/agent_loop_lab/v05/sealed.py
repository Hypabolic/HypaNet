"""v0.5 paired validation comparison and one-time sealed evaluation of the v0.5 networks. Harness only.

``scripts/v05_sealed_eval.py`` drives this module. It scores the two v0.5 networks (r1, written with
no tools, and Phase B, r1 revised with 200 labelled windows) beside every v0.3 column, on v0.3's
validation, counterfactual, sealed test and sealed OOD sets.

Reuse, not re-implementation. The v0.3 columns go through v0.3's sealed code unchanged, and the
v0.5 networks are added to it:

* inputs, data and access: ``v03.sealed.load_columns`` (A/B archives, every Phase C run),
  ``verify_unsealed_inputs``, ``verify_sealed_hash_records``, ``build_split`` (val through
  ``v03.scoring.load_val``; counterfactual, test and OOD only through v0.3's builder, and test /
  OOD only with the token of this run's attempt record), ``identity_check`` against v0.2,
  ``verify_sealed_files``, ``write_attempt_record`` (here with the v0.5 marker path);
* v0.3 predictions: ``v03.sealed.predict_split`` as is (networks in the pool, then each reference
  policy alone). The v0.5 networks then run in the same idle pool through
  ``v05.scoring.network_predictions`` (the path of the v0.5 val reports, with its label-free logit
  diagnostics), so no policy call shares the CPU with network work;
* metrics: ``v03.sealed.score_split`` (``eval.m4.score_columns``, the v0.2/v0.3 session bootstrap:
  10,000 resamples, seed 20261104, per-family seeds) over every column at once, so all columns
  share the same draws; the v0.3 val checks ``scoring_path_check``, ``official_val_check``,
  ``phase_c_es_check`` and ``prior_confirmation``;
* paired differences: ``v04.scoring.paired_macro_f1`` (the v0.4/v0.5 method). The false-alarm
  difference is the same computation with ``false_alarm_rate`` in place of ``macro_f1``
  (:func:`paired_metric`; for ``macro_f1`` it equals ``paired_macro_f1`` exactly).

**Checks.** On full val, before any sealed file is opened (each failure stops the run):

* v0.3's own checks (disclosure rule, identity with v0.2, ``scoring.score`` = ``m4.score_columns``,
  the A/B blocks = ``artifacts/reports/v03_val_*.json``, each Phase C model's early-stopping
  macro-F1 = its training log);
* each v0.5 network's blocks reproduce its val report exactly (:func:`v05_val_check`): population,
  overall and per-family network blocks (macro-F1 and CI, false alarm, confusion, ...) in both
  populations, its paired macro-F1 rows, the v0.3 A blocks of that report, the artefact sha256 and
  non-zero count, the disclosed-session hash and the val encoding's cache key.

Before that, also fatal: the v0.3 columns, code files, data files and registered sealed hashes equal
those of ``v03_sealed.json``; v0.5 B's recorded base artefact is r1; and, for ``--sealed``, the
pre-registration ``experiments/v0_5/sealed/README.md`` is committed and unchanged.

Recorded, not fatal (a CPU float difference in another network's argmax would show here):
the reproduction of every v0.3 column, group,
H2 fidelity, population, gate and policy-failure block of ``artifacts/reports/v03_sealed.json`` on
val, counterfactual, test and OOD (:func:`compare_with_v03_sealed`). A paired difference that uses a
v0.3 column which did not reproduce carries a caveat.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..eval import m4
from ..eval import metrics as M
from ..eval.collisions import WindowRow
from ..v03 import sealed as S
from ..v04 import scoring as v04_scoring
from . import scoring as v05_scoring

__all__ = [
    "SEALED_REPORT", "SEALED_MARKER", "V03_SEALED_REPORT", "V05_NETWORKS", "PAIRS", "V05Network", "V05Columns",
    "pre_registration_problem", "refuse_reasons", "previous_attempts", "load_v05_columns", "check_v05_disclosure", "check_against_v03_sealed_inputs",
    "predict_all", "paired_metric", "paired_differences", "score_all", "v05_val_check", "compare_with_v03_sealed",
    "compare_counterfactual_with_v03_sealed", "caveat_pairs", "summary", "code_hashes",
]

REPO = S.REPO
REPORT_DIR = S.REPORT_DIR
SEALED_REPORT = REPORT_DIR / "v05_sealed.json"
SEALED_MARKER = REPORT_DIR / "v05_sealed.started.json"
PREVIOUS_ATTEMPT_GLOB = "v05_sealed.attempt*.json"
V03_SEALED_REPORT = S.SEALED_REPORT  # artifacts/reports/v03_sealed.json (read only)
V05B_DISCLOSURE = REPO / "experiments" / "v0_5" / "phase_b" / "DISCLOSURE_PRIVATE.json"
PRE_REGISTRATION = "experiments/v0_5/sealed/README.md"
POPULATIONS = ("excluding_disclosed", "all_val")

R1 = "v0.5 r1 network"
V05B = "v0.5 B network"
V03A, V03B, V03C_ALL = "v0.3 A network", "v0.3 B network", "v0.3 C-all seed0"
PRIOR = S.PRIOR_COLUMNS[1]  # "prior always-PRODUCTIVE"
PAIRS: dict[str, tuple[str, str]] = {v04_scoring.pair_name(a, b): (a, b) for a, b in (
    (V05B, V03B), (V05B, V03A), (V05B, V03C_ALL), (V05B, R1), (R1, V03A))}
PAIRED_METRICS = ("macro_f1", "false_alarm_rate")


@dataclasses.dataclass(frozen=True)
class V05Network:
    column: str
    weight_dir: Path
    val_report: Path
    run: str | None  # official r1: the artefact metadata must name this run
    phase: str | None  # Phase B: the artefact metadata must name this phase
    labelled: dict[str, Any]


V05_NETWORKS = (
    V05Network(R1, REPO / "experiments" / "v0_5" / "runs" / "r1", REPORT_DIR / "v05_val_r1.json", "r1", None,
               {"labelled_examples": 0, "revisions": 0, "tools": False}),
    V05Network(V05B, REPO / "weight_program" / "v0_5" / "phase-b",
               REPO / "experiments" / "v0_5" / "phase_b" / "val_report" / "v05_val_v05b.json", None, "v05-phase-b",
               {"labelled_examples": 200, "revisions": 1, "tools": True, "base": R1}),
)


# ---------------------------------------------------------------------------
# One-time rule
# ---------------------------------------------------------------------------


def pre_registration_problem(path: Path | None = None) -> str | None:
    """Why the pre-registration does not fix the protocol (``None``: it exists, is tracked by git and equals HEAD)."""
    import subprocess

    path = REPO / PRE_REGISTRATION if path is None else Path(path)
    if not path.is_file():
        return f"{path} missing; the sealed protocol must be pre-registered and committed first"

    def git(*a: str) -> int:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True).returncode

    try:
        rel = str(path.resolve().relative_to(REPO))
    except ValueError:
        return f"{path} is outside the repository; the pre-registration must be committed"
    if git("ls-files", "--error-unmatch", "--", rel) != 0:
        return f"{rel} is not committed"
    if git("diff", "--quiet", "HEAD", "--", rel) != 0:
        return f"{rel} differs from HEAD; commit the pre-registration first"
    return None


def refuse_reasons(report: Path | None = None, marker: Path | None = None,
                   v03_report: Path | None = None, pre_registration: Path | None = None) -> list[str]:
    """Why a sealed run may not start (empty: it may)."""
    report = SEALED_REPORT if report is None else report
    marker = SEALED_MARKER if marker is None else marker
    v03_report = V03_SEALED_REPORT if v03_report is None else v03_report
    out = [f"{p} exists; the v0.5 sealed evaluation runs once" for p in (report, marker) if p.exists()]
    if not v03_report.is_file():
        out.append(f"{v03_report} missing; the v0.3 columns are checked against it")
    problem = pre_registration_problem(pre_registration)
    if problem is not None:
        out.append(problem)
    return out


def previous_attempts(report_dir: Path | None = None) -> list[dict[str, Any]]:
    """Earlier attempt records, renamed ``v05_sealed.attempt<N>.json`` by the adjudicator."""
    report_dir = SEALED_MARKER.parent if report_dir is None else report_dir
    return [{"file": p.name, "sha256": S.sha256_file(p), "content": json.loads(p.read_text())}
            for p in sorted(report_dir.glob(PREVIOUS_ATTEMPT_GLOB))]


# ---------------------------------------------------------------------------
# Columns and inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class V05Columns:
    specs: list[S.NetSpec]
    info: dict[str, dict[str, Any]]
    inputs: dict[str, Any]
    val_reports: dict[str, Path]


def load_v05_columns(networks: Sequence[V05Network] = V05_NETWORKS) -> V05Columns:
    """sha-check and load each v0.5 artefact (``v05.scoring.load_run_artifact``) and tie it to its records."""
    specs: list[S.NetSpec] = []
    info: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    reports: dict[str, Path] = {}
    for n in networks:
        dlw, sha_file = n.weight_dir / "artifact.dlw", n.weight_dir / "artifact.sha256"
        if not sha_file.is_file():
            raise S.CheckFailed(f"{n.column}: {sha_file} missing")
        try:
            spec, art = v05_scoring.load_run_artifact(dlw, sha_file, run=n.run)
            apply_sha = v05_scoring.check_apply_report(n.weight_dir, art["artifact_sha256"])
        except (v05_scoring.ScoringError, FileNotFoundError) as exc:
            raise S.CheckFailed(f"{n.column}: {exc}") from None
        sha = art["artifact_sha256"]
        rec: dict[str, Any] = {"artifact_sha256": sha, "apply_report_artifact_sha256": apply_sha}
        if n.phase is not None:
            if art["artifact_metadata"].get("phase") != n.phase:
                raise S.CheckFailed(f"{n.column}: artefact metadata phase is not {n.phase}")
            build = n.weight_dir / "BUILD_RECORD.json"
            if not build.is_file() or json.loads(build.read_text()).get("artifact_sha256") != sha:
                raise S.CheckFailed(f"{n.column}: BUILD_RECORD.json artifact_sha256 differs from the artefact")
            rec["build_record_sha256"] = S.sha256_file(build)
        val = json.loads(n.val_report.read_text())
        if val["inputs"]["artifact"]["artifact_sha256"] != sha:
            raise S.CheckFailed(f"{n.column}: {n.val_report.name} scored another artefact")
        rec |= {"val_report": _rel(n.val_report), "val_report_sha256": S.sha256_file(n.val_report),
                "val_report_official": val["official"]}
        specs.append(dataclasses.replace(spec, name=n.column))
        info[n.column] = {"kind": "network", **n.labelled, "weight_dir": _rel(n.weight_dir),
                          "artifact_metadata": art["artifact_metadata"],
                          "nonzero_parameters": art["nonzero_parameters"], **rec}
        inputs[n.column] = rec
        reports[n.column] = n.val_report
    for n in networks:
        base = n.labelled.get("base")
        if base is None:
            continue
        if base not in info:
            raise S.CheckFailed(f"{n.column}: its base {base} is not loaded, so its lineage cannot be checked")
        want = info[base]["artifact_sha256"]
        build_path = n.weight_dir / "BUILD_RECORD.json"
        build = json.loads(build_path.read_text()) if build_path.is_file() else {}
        got = {"artifact_metadata": info[n.column]["artifact_metadata"].get("base_artifact_sha256"),
               "BUILD_RECORD.json": build.get("base_artifact_sha256")}
        bad = [k for k, v in got.items() if v != want]
        if bad:
            raise S.CheckFailed(f"{n.column}: base_artifact_sha256 in {', '.join(bad)} is not the {base} artefact")
        info[n.column]["base_artifact_sha256"] = inputs[n.column]["base_artifact_sha256"] = want
    return V05Columns(specs, info, inputs, reports)


def check_v05_disclosure(v03_disclosed: frozenset[str], path: Path = V05B_DISCLOSURE) -> dict[str, Any]:
    """v0.5 Phase B was shown v0.3's 200 disclosed windows: its record's sessions must be v0.3's set."""
    rec = json.loads(Path(path).read_text())
    ids = frozenset(w["session_id"] for w in rec["windows"])
    if ids != v03_disclosed:
        raise S.CheckFailed(f"{_rel(path)}: disclosed sessions differ from experiments/v0_3/DISCLOSURE_PRIVATE.json")
    return {"source": _rel(path), "sha256": S.sha256_file(path), "windows": len(rec["windows"]),
            "same_sessions_as_v03": True}


def check_against_v03_sealed_inputs(v03_rep: Mapping[str, Any], cols: S.Columns, data_sha: Mapping[str, str],
                                    sealed_records: Mapping[str, str]) -> dict[str, Any]:
    """The v0.3 columns, the v0.3 code files, the unsealed data files and the sealed files' registered hashes
    must be exactly those of the v0.3 sealed run (no sealed file is read)."""
    checks = {"columns": (v03_rep["inputs"]["columns"], cols.inputs),
              "code_sha256": (v03_rep["inputs"]["code_sha256"], S.code_hashes()),
              "sealed_hash_records": (v03_rep["inputs"]["sealed_hash_records"], sealed_records),
              "unsealed_data_sha256": ({k: v03_rep["inputs"]["data_sha256"].get(k) for k in data_sha}, dict(data_sha))}
    for what, (a, b) in checks.items():
        d = S.first_difference(S.canon(a), S.canon(b))
        if d is not None:
            raise S.CheckFailed(f"{what} differ from v03_sealed.json inputs at {d}")
    return {"match": True, "compared": sorted(checks)}


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def predict_all(sd: S.SplitData, cols: S.Columns, v05: V05Columns, prior: Mapping[str, Any], pool: Any, *,
                workers: int, progress: Callable[[str], None] | None = None,
                evaluate: Callable[..., list[int | None]] | None = None) -> tuple[S.SplitPredictions, dict[str, Any]]:
    """The v0.3 columns through ``v03.sealed.predict_split`` unchanged, then the v0.5 networks in the same
    (idle) pool. Returns the merged predictions (v0.5 columns first) and the v0.5 logit diagnostics."""
    say = progress or (lambda m: None)
    sp = S.predict_split(sd, cols, prior, pool, workers=workers, progress=progress, evaluate=evaluate)
    res = v05_scoring.network_predictions(v05.specs, sd.data, pool, workers=workers,
                                          progress=lambda m: say(f"{sd.split} v0.5 {m}"))
    gate = np.array([r.gate_passed for r in sd.data.rows], dtype=bool)
    preds: dict[str, np.ndarray] = {}
    latency: dict[str, Any] = {}
    logits: dict[str, Any] = {}
    for spec in v05.specs:
        r = res[spec.name]
        if np.any(r.pred[gate] < 0):
            raise RuntimeError(f"{sd.split}: a gate-passing window has no {spec.name} prediction")
        preds[spec.name], latency[spec.name], logits[spec.name] = r.pred, r.latency, {**r.logits, "split": sd.split}
    merged = S.SplitPredictions({**preds, **sp.preds}, sp.raw_ref, {**latency, **sp.latency}, sp.policy_wall_s)
    return merged, logits


# ---------------------------------------------------------------------------
# Paired differences
# ---------------------------------------------------------------------------


def paired_metric(rows: Sequence[WindowRow], preds: Mapping[str, np.ndarray], pairs: Mapping[str, tuple[str, str]],
                  metric: str, *, n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED
                  ) -> dict[str, Any]:
    """``v04.scoring.paired_macro_f1`` for any metric of ``metrics.metrics_from_counts``: metric(a) - metric(b)
    and the 95% percentile interval of that difference over the shared session-bootstrap draws (overall)."""
    if not pairs:
        return {}
    rows = list(rows)
    columns = list(dict.fromkeys(c for ab in pairs.values() for c in ab))
    unknown = [c for c in columns if c not in preds]
    if unknown:
        raise S.CheckFailed(f"paired comparison names unknown columns: {unknown}")
    ids, _, sidx = M.session_index(rows)
    n_s = len(ids)
    counts = np.stack([m4.single_counts(rows, preds[c], sidx, n_s) for c in columns], axis=1)
    flat = counts.reshape(n_s, -1)
    k = len(columns)

    def fn(c: np.ndarray) -> dict[str, np.ndarray]:
        return {metric: M.metrics_from_counts(c.reshape(c.shape[:-1] + (k, M.MODEL_COLUMNS)))[metric]}

    point = fn(flat.sum(axis=0).astype(np.float64))[metric]
    dist = M.bootstrap_distribution(flat, n_resamples, seed, metrics=fn)[metric]
    j = {c: i for i, c in enumerate(columns)}
    out: dict[str, Any] = {}
    for name, (a, b) in pairs.items():
        v = float(point[j[a]] - point[j[b]])
        ci, undefined = M._ci(dist[:, j[a]] - dist[:, j[b]])
        out[name] = {"minuend": a, "subtrahend": b, "value": v if np.isfinite(v) else None, "ci95": ci,
                     "ci95_above_0": None if ci is None else bool(ci[0] > 0)}
        if undefined:
            out[name]["undefined_resamples"] = undefined
    return out


def paired_differences(rows: Sequence[WindowRow], preds: Mapping[str, np.ndarray],
                       pairs: Mapping[str, tuple[str, str]] = PAIRS, *,
                       n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:
    """Macro-F1 (``v04.scoring.paired_macro_f1``) and false-alarm (:func:`paired_metric`) differences."""
    return {"macro_f1": v04_scoring.paired_macro_f1(rows, preds, pairs, n_resamples=n_resamples, seed=seed),
            "false_alarm_rate": paired_metric(rows, preds, pairs, "false_alarm_rate", n_resamples=n_resamples,
                                              seed=seed)}


def score_all(sd: S.SplitData, sp: S.SplitPredictions, cols: S.Columns, disclosed: frozenset[str], *,
              pairs: Mapping[str, tuple[str, str]] = PAIRS, n_resamples: int = M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """``v03.sealed.score_split`` over every column, plus the paired differences per population."""
    sc = S.score_split(sd, sp, cols, disclosed, n_resamples=n_resamples)
    rows = sd.data.rows
    for pop, mask in S.population_masks(sd, disclosed).items():
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        sc["populations"][pop]["paired_differences"] = paired_differences(
            sub, {k: v[idx] for k, v in sp.preds.items()}, pairs, n_resamples=n_resamples)
    return sc


# ---------------------------------------------------------------------------
# Val reproduction of the v0.5 reports (fatal)
# ---------------------------------------------------------------------------


def _v05_report_pairs(column: str) -> dict[str, tuple[str, str]]:
    return {v04_scoring.pair_name(v05_scoring.NET_COLUMN, V03A): (column, V03A),
            v04_scoring.pair_name(v05_scoring.NET_COLUMN, PRIOR): (column, PRIOR)}


def v05_val_check(sd: S.SplitData, sp: S.SplitPredictions, scored: Mapping[str, Any], v05: V05Columns,
                  disclosed: frozenset[str], *, reports: Mapping[str, Path] | None = None,
                  n_resamples: int = M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Each v0.5 column must reproduce its val report exactly (full val only); raises CheckFailed."""
    reports = v05.val_reports if reports is None else reports
    rows = sd.data.rows
    masks = S.population_masks(sd, disclosed)
    out: dict[str, Any] = {}
    for column, path in reports.items():
        rep = json.loads(Path(path).read_text())
        pairs: list[tuple[str, Any, Any]] = [
            ("disclosed_sessions/sha256", rep["disclosed_sessions"]["sha256_of_sorted_ids"],
             S.scoring._session_set_sha256(disclosed)),
            ("disclosed_sessions/count", rep["disclosed_sessions"]["count"], len(disclosed)),
            ("artifact_sha256", rep["inputs"]["artifact"]["artifact_sha256"], v05.info[column]["artifact_sha256"]),
            ("nonzero_parameters", rep["nonzero_parameters"], v05.info[column]["nonzero_parameters"]),
            ("val_rows_sha256", rep["inputs"]["val_rows_sha256"], S.rows_sha256(rows)),
        ]
        if sd.data.inputs.get("cache_key") is not None:
            pairs.append(("dataset", rep["inputs"]["dataset"], sd.data.inputs["cache_key"]))
        for pop in POPULATIONS:
            r, sc = rep[pop], scored["populations"][pop]
            if set(r["per_family"]) != set(sc["per_family"]):
                raise S.CheckFailed(f"val {column} {pop}: families differ from {Path(path).name}")
            pairs.append((f"{pop}/population", r["population"], sc["population"]))
            for blk in ("overall", *[f"per_family/{f}" for f in sorted(r["per_family"])]):
                rb = r["overall"] if blk == "overall" else r["per_family"][blk.split("/", 1)[1]]
                sb = sc["overall"] if blk == "overall" else sc["per_family"][blk.split("/", 1)[1]]
                pairs += [(f"{pop}/{blk}/sessions", rb["sessions"], sb["sessions"]),
                          (f"{pop}/{blk}/network", rb["network"], sb["columns"][column]),
                          (f"{pop}/{blk}/v03_phase_a_network", rb[v05_scoring.A_BLOCK], sb["columns"][V03A])]
            summ = rep["summary"][pop]["network"]
            mine = sc["overall"]["columns"][column]
            for k in summ:
                pairs.append((f"summary/{pop}/network/{k}", summ[k],
                              {"value": mine[k]["value"], "ci95": mine[k]["ci95"]}))
            idx = np.flatnonzero(masks[pop])
            sub = [rows[i] for i in idx.tolist()]
            got = v04_scoring.paired_macro_f1(sub, {c: sp.preds[c][idx] for c in (column, V03A, PRIOR)},
                                              _v05_report_pairs(column), n_resamples=n_resamples)
            for name, want in r["paired_macro_f1"].items():
                g = got[name]
                pairs.append((f"{pop}/paired_macro_f1/{name}",
                              {k: want[k] for k in ("value", "ci95", "ci95_above_0")},
                              {k: g[k] for k in ("value", "ci95", "ci95_above_0")}))
        for what, a, b in pairs:
            d = S.first_difference(S.canon(a), S.canon(b))
            if d is not None:
                raise S.CheckFailed(f"val {column}: differs from {Path(path).name} at {what}{d}")
        out[column] = {"match": True, "report": _rel(Path(path)), "report_sha256": S.sha256_file(Path(path)),
                       "blocks_compared": len(pairs),
                       "compared": ("disclosed sessions, artefact sha256, non-zero parameters, val rows sha256, "
                                    "dataset cache key, population, overall and per-family network and v0.3 A "
                                    "blocks (both populations), summary, paired macro-F1 rows")}
    return out


# ---------------------------------------------------------------------------
# Reproduction of v03_sealed.json (recorded, not fatal)
# ---------------------------------------------------------------------------


def _note_diff(diffs: dict[str, str | None], key: str, a: Any, b: Any, where: str) -> None:
    d = S.first_difference(S.canon(a), S.canon(b))
    if d is not None and diffs.get(key) is None:
        diffs[key] = f"{where}{d}"


def compare_with_v03_sealed(split: str, scored: Mapping[str, Any], recorded_split: Mapping[str, Any] | None,
                            cols: S.Columns) -> dict[str, Any]:
    """Every v0.3 column, group, H2 fidelity, population, bootstrap, gate and policy-failure block of one split
    against ``v03_sealed.json``: per item the path of the first difference (``None`` when equal)."""
    if recorded_split is None or "populations" not in recorded_split:
        return {"compared": False, "reason": f"v03_sealed.json has no scored {split} split"}
    columns = S.column_order(cols)
    diffs: dict[str, str | None] = {c: None for c in columns}
    diffs |= {f"group {g}": None for g in cols.groups} | {f"h2_fidelity {f}": None for f in cols.fid_pairs}
    diffs |= {"population": None, "bootstrap": None, "gate": None, "policy_failures": None}
    _note_diff(diffs, "gate", recorded_split["gate"], scored["gate"], f"{split} gate")
    if set(recorded_split["populations"]) != set(scored["populations"]):
        raise S.CheckFailed(f"{split}: populations differ from v03_sealed.json")
    for pop, rec in recorded_split["populations"].items():
        mine = scored["populations"][pop]
        if set(rec["per_family"]) != set(mine["per_family"]):
            raise S.CheckFailed(f"{split} {pop}: families differ from v03_sealed.json")
        for key in ("population", "bootstrap", "policy_failures"):
            _note_diff(diffs, key, rec[key], mine[key], f"{split} {pop} {key}")
        blocks = [("overall", rec["overall"], mine["overall"])]
        blocks += [(f"per_family/{f}", rec["per_family"][f], mine["per_family"][f]) for f in sorted(rec["per_family"])]
        for where, a, b in blocks:
            tag = f"{split} {pop} {where}"
            _note_diff(diffs, "sessions", a["sessions"], b["sessions"], tag)
            for c in columns:
                if c not in a["columns"]:
                    raise S.CheckFailed(f"{tag}: {c} is not in v03_sealed.json")
                _note_diff(diffs, c, a["columns"][c], b["columns"][c], tag)
            for g in cols.groups:
                _note_diff(diffs, f"group {g}", a["groups"][g], b["groups"][g], tag)
            for f in cols.fid_pairs:
                _note_diff(diffs, f"h2_fidelity {f}", a["h2_fidelity"][f], b["h2_fidelity"][f], tag)
    bad = sorted(k for k, v in diffs.items() if v is not None)
    return {"compared": True, "reproduced": not bad, "not_reproduced": bad,
            "first_difference": {k: v for k, v in diffs.items() if v is not None}, "items_compared": len(diffs)}


def compare_counterfactual_with_v03_sealed(scored: Mapping[str, Any], recorded: Mapping[str, Any] | None,
                                           cols: S.Columns) -> dict[str, Any]:
    if recorded is None or "counterfactual_consistency" not in recorded:
        return {"compared": False, "reason": "v03_sealed.json has no counterfactual split"}
    rc, mc = recorded["counterfactual_consistency"], scored["counterfactual_consistency"]
    items = [(key, recorded[key], scored[key]) for key in ("gate", "policy_failures")]
    items += [(key, rc[key], mc.get(key)) for key in rc if key != "columns"]
    for c in S.column_order(cols):
        if c not in rc["columns"]:
            raise S.CheckFailed(f"counterfactual: {c} is not in v03_sealed.json")
        items.append((c, rc["columns"][c], mc["columns"][c]))
    diffs: dict[str, str | None] = {}
    for key, a, b in items:
        _note_diff(diffs, key, a, b, "counterfactual ")
    bad = sorted(diffs)
    return {"compared": True, "reproduced": not bad, "not_reproduced": bad, "first_difference": diffs,
            "items_compared": len(items)}


def caveat_pairs(scored: Mapping[str, Any], repro: Mapping[str, Any]) -> None:
    """Mark every paired difference that uses a v0.3 column that did not reproduce ``v03_sealed.json``."""
    bad = set(repro.get("not_reproduced") or ())
    for pop in scored["populations"].values():
        for per_metric in pop.get("paired_differences", {}).values():
            for d in per_metric.values():
                hit = sorted({d["minuend"], d["subtrahend"]} & bad)
                if hit:
                    d["caveat"] = f"{', '.join(hit)} did not reproduce its v03_sealed.json blocks on this split"


# ---------------------------------------------------------------------------
# Report pieces
# ---------------------------------------------------------------------------


def summary(scored: Mapping[str, Any], cols: S.Columns) -> dict[str, Any]:
    """Per population: ``v03.sealed.summary_table`` over every column, and the paired differences."""
    out: dict[str, Any] = {}
    for pop, sc in scored["populations"].items():
        paired = {name: {m: [sc["paired_differences"][m][name]["value"], sc["paired_differences"][m][name]["ci95"]]
                         for m in PAIRED_METRICS}
                  for name in sc["paired_differences"]["macro_f1"]}
        out[pop] = {"columns": S.summary_table(sc, cols), "paired": paired}
    return out


CODE_FILES = (
    "scripts/v05_sealed_eval.py",
    "src/agent_loop_lab/v05/sealed.py",
    "src/agent_loop_lab/v05/scoring.py",
    "src/agent_loop_lab/v04/scoring.py",
    "scripts/v05_score_validation.py",
)


def code_hashes() -> dict[str, str | None]:
    """v0.3's code files, the v0.5 ones and the pre-registration (``None`` while it does not exist;
    ``--sealed`` refuses then)."""
    return {**S.code_hashes(), **{p: S.sha256_file(REPO / p) for p in CODE_FILES},
            PRE_REGISTRATION: pre_registration()["sha256"]}


def pre_registration() -> dict[str, Any]:
    p = REPO / PRE_REGISTRATION
    return {"path": PRE_REGISTRATION, "sha256": S.sha256_file(p) if p.is_file() else None}


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
