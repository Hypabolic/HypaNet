"""v0.5 Phase C held-out evaluation (one time) and its val dry run. Harness only.

``scripts/v05c_heldout_eval.py`` drives this module; the pre-registration is
``experiments/v0_5/phase_c/heldout/README.md``. The sealed v0.2/v0.3 test and OOD sets are not used.

**Sets.** The held-out OOD set (``synth_ood``, master seed 20261110, 200 sessions per family, all 10 families),
reported for ALL families, the 6 SEEN families (those the OOD disclosure drew from) and the 4 UNSEEN families;
the held-out in-distribution set (standard generator, master seed 20261111, 4,000 sessions); and val
(``excluding_disclosed``, primary as in v0.3, and ``all_val``). Held-out data comes only through
``v05.phase_c_data.load_set`` (file hashes, window-row hash and encoding hash against the committed manifest, and
only under a written attempt record).

**Columns.** v0.5 Phase C, v0.5 B, v0.5 r1, v0.3 A, v0.3 B, v0.3 C-all seed0 (networks), plus the prior
always-PRODUCTIVE (reference). r1 and B are loaded and tied to their records by ``v05.sealed.load_v05_columns``;
v0.3 A/B by ``v03.scoring.load_phase``; C-all by ``v03.sealed.check_phase_c_run``; Phase C by
:func:`load_phase_c` (sha256 file, ``apply.json``, ``BUILD_RECORD.json``, metadata phase ``v05-phase-c``, base =
B's artefact). A dry run may replace the Phase C artefact by any ``.dlw`` (``stand_in``: clearly marked).

**Predictions** (``v05.scoring.network_predictions``: one causal forward per session, argmax at each window's last
EOE, 2 torch threads per worker) and **metrics** (``eval.m4.score_columns``: the v0.2/v0.3 session bootstrap,
10,000 resamples, seed 20261104, per-family seeds, every column on the same draws), overall and per family, per
population. **Paired differences** (``v05.sealed.paired_differences``: macro-F1 by ``v04.scoring.paired_macro_f1``,
false-alarm rate by the same computation), per population.

**Checks on full val** (fatal unless noted): the val rows hash to v0.3's sealed record; the 200 disclosed sessions
follow v0.3's rule and equal ``experiments/v0_3/DISCLOSURE_PRIVATE.json``; v0.5 B's disclosure record names the
same sessions; the r1 and B columns (and the v0.3 A blocks inside their reports) reproduce their val reports
exactly (``v05.sealed.v05_val_check``). Recorded, not fatal: the v0.3 A, B and C-all val blocks against
``artifacts/reports/v03_sealed.json`` (a CPU float difference would show here; pairs using such a column get a
caveat).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..eval import m4
from ..eval import metrics as M
from ..v03 import scoring as v03_scoring
from ..v03 import sealed as S
from ..v04 import scoring as v04_scoring
from . import phase_c_data as P
from . import scoring as v05_scoring
from . import sealed as V

__all__ = [
    "HELDOUT_REPORT", "HELDOUT_MARKER", "PRE_REGISTRATION", "MANIFEST", "PHASE_C_DIR", "PHASE_C_PHASE",
    "COLUMN_ORDER", "PAIRS", "PRIMARY_PAIRS", "EvalColumns", "committed_problem", "refuse_reasons",
    "dry_run_dir_problem", "load_phase_c", "load_eval_columns", "predict", "population_masks", "score_set",
    "val_checks", "evaluate", "prelaunch_problems",
]

REPO = S.REPO
REPORT_DIR = S.REPORT_DIR
HELDOUT_REPORT = REPORT_DIR / "v05c_heldout.json"
HELDOUT_MARKER = REPORT_DIR / "v05c_heldout.started.json"
PREVIOUS_ATTEMPT_GLOB = "v05c_heldout.attempt*.json"
DRY_REPORT_NAME = "v05c_heldout_dryrun_val.json"
PRE_REGISTRATION = "experiments/v0_5/phase_c/heldout/README.md"
MANIFEST = "experiments/v0_5/phase_c/heldout/HELDOUT_MANIFEST.json"
DISCLOSURE_OOD_PRIVATE = "experiments/v0_5/phase_c/DISCLOSURE_OOD_PRIVATE.json"
PHASE_C_DIR = REPO / "weight_program" / "v0_5" / "phase-c"
PHASE_C_PHASE = "v05-phase-c"
V03_SEALED_REPORT = V.V03_SEALED_REPORT  # read only: the v0.3 val blocks (no sealed data file is opened)

C = "v0.5 Phase C network"
B, R1 = V.V05B, V.R1
V03A, V03B, V03C_ALL = V.V03A, V.V03B, V.V03C_ALL
PRIOR = V.PRIOR
COLUMN_ORDER = (C, B, R1, V03A, V03B, V03C_ALL, PRIOR)
_PAIR_LIST = ((C, B), (C, V03B), (C, R1), (C, V03A), (C, V03C_ALL), (B, V03B))
PAIRS: dict[str, tuple[str, str]] = {v04_scoring.pair_name(a, b): (a, b) for a, b in _PAIR_LIST}
PRIMARY_PAIRS = tuple(v04_scoring.pair_name(a, b) for a, b in _PAIR_LIST[:2])
PRIMARY_POPULATION = ("heldout_ood", "unseen")
# Pre-registered order of the comparisons (README §6): primary first.
COMPARISON_ORDER = (("heldout_ood", "unseen"), ("heldout_ood", "seen"), ("heldout_ood", "all"),
                    ("heldout_indist", "all"), ("val", "excluding_disclosed"))
V03_COLUMNS = (V03A, V03B, V03C_ALL)


class StageError(RuntimeError):
    """The held-out stage stopped after the attempt record was written (the attempt is spent)."""


# ---------------------------------------------------------------------------
# One-time rule
# ---------------------------------------------------------------------------


def committed_problem(rel: str | Path, what: str) -> str | None:
    """Why ``rel`` is not committed unchanged (``None``: it exists, is tracked and equals HEAD)."""
    path = Path(rel) if Path(rel).is_absolute() else REPO / rel
    if not path.is_file():
        return f"{path} missing; the {what} must exist and be committed first"
    try:
        r = str(path.resolve().relative_to(REPO))
    except ValueError:
        return f"{path} is outside the repository; the {what} must be committed"

    def git(*a: str) -> int:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True).returncode

    if git("ls-files", "--error-unmatch", "--", r) != 0:
        return f"{r} is not committed ({what})"
    if git("diff", "--quiet", "HEAD", "--", r) != 0:
        return f"{r} differs from HEAD; commit the {what} first"
    return None


def _committed_problems(pre_registration: str | Path, manifest: str | Path, disclosure_private: str | Path,
                        code_files: Sequence[str | Path]) -> list[str]:
    items = [(pre_registration, "pre-registration"), (manifest, "held-out manifest"),
             (disclosure_private, "OOD disclosure private record")]
    items += [(f, "scoring code") for f in code_files]
    return [p for p in (committed_problem(rel, what) for rel, what in items) if p is not None]


def refuse_reasons(report: Path | None = None, marker: Path | None = None, *, pre_registration: str | Path = PRE_REGISTRATION,
                   manifest: str | Path = MANIFEST, disclosure_private: str | Path = DISCLOSURE_OOD_PRIVATE,
                   code_files: Sequence[str | Path] = P.SCORING_CODE_FILES) -> list[str]:
    """Why the one-time held-out run may not start (empty: it may): the report or attempt record exists, or the
    pre-registration, the manifest, the OOD disclosure private record or the scoring code is not committed
    unchanged. (That the scoring code also equals what the manifest pinned before launch is checked by
    ``P.manifest_problems(official=True)`` inside :func:`evaluate`.)"""
    report = HELDOUT_REPORT if report is None else Path(report)
    marker = HELDOUT_MARKER if marker is None else Path(marker)
    out = [f"{p} exists; the v0.5 Phase C held-out evaluation runs once" for p in (report, marker) if p.exists()]
    return out + _committed_problems(pre_registration, manifest, disclosure_private, code_files)


def prelaunch_problems(*, pre_registration: str | Path = PRE_REGISTRATION, manifest: str | Path = MANIFEST,
                       disclosure_private: str | Path = DISCLOSURE_OOD_PRIVATE,
                       code_files: Sequence[str | Path] = P.SCORING_CODE_FILES) -> list[str]:
    """Why the real Phase C author session may not be launched yet (empty: it may). The pre-registration, exactly
    ``HELDOUT_MANIFEST.json``, the OOD disclosure private record and the scoring code must be committed unchanged,
    and the manifest must pass ``P.manifest_problems(official=True)`` (official, pre-registered seeds and sizes,
    and the scoring code byte for byte what it pinned). Run by scripts/v05c_launch.sh in real mode."""
    out = _committed_problems(pre_registration, manifest, disclosure_private, code_files)
    path = Path(manifest) if Path(manifest).is_absolute() else REPO / manifest
    if path.is_file():
        try:
            man = json.loads(path.read_text())
        except ValueError as exc:
            return out + [f"{path}: not JSON ({exc})"]
        out += [f"held-out manifest: {p}" for p in P.manifest_problems(man, official=True)]
    return out


def _main(argv: Sequence[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["prelaunch-check"]:
        print("usage: python -m agent_loop_lab.v05.phase_c_eval prelaunch-check", file=sys.stderr)
        return 2
    problems = prelaunch_problems()
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


def dry_run_dir_problem(report_dir: Path) -> str | None:
    d, official = Path(report_dir).resolve(), REPORT_DIR.resolve()
    if d == official or official in d.parents:
        return "--report-dir may not be artifacts/reports/ or inside it (official reports come from the held-out run only)"
    return None


def previous_attempts(report_dir: Path | None = None) -> list[dict[str, Any]]:
    report_dir = HELDOUT_MARKER.parent if report_dir is None else report_dir
    return [{"file": p.name, "sha256": S.sha256_file(p), "content": json.loads(p.read_text())}
            for p in sorted(Path(report_dir).glob(PREVIOUS_ATTEMPT_GLOB))]


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalColumns:
    specs: list[S.NetSpec]  # COLUMN_ORDER without the prior
    info: dict[str, dict[str, Any]]
    inputs: dict[str, Any]
    v05: V.V05Columns  # r1 and B (with their val reports)
    phase_c_stand_in: bool


def load_phase_c(weight_dir: Path = PHASE_C_DIR, *, base_sha256: str) -> tuple[S.NetSpec, dict[str, Any]]:
    """The Phase C network, checked against ``artifact.sha256``, ``apply.json``, ``BUILD_RECORD.json``, its
    metadata phase and its base (B's artefact, in the metadata and the build record)."""
    weight_dir = Path(weight_dir)
    dlw, sha_file = weight_dir / "artifact.dlw", weight_dir / "artifact.sha256"
    if not dlw.is_file() or not sha_file.is_file():
        raise S.CheckFailed(f"{C}: {dlw} or {sha_file} missing (the Phase C build has not been archived)")
    try:
        spec, art = v05_scoring.load_run_artifact(dlw, sha_file)
        apply_sha = v05_scoring.check_apply_report(weight_dir, art["artifact_sha256"])
    except (v05_scoring.ScoringError, FileNotFoundError) as exc:
        raise S.CheckFailed(f"{C}: {exc}") from None
    sha = art["artifact_sha256"]
    meta = art["artifact_metadata"]
    if meta.get("phase") != PHASE_C_PHASE:
        raise S.CheckFailed(f"{C}: artefact metadata phase is not {PHASE_C_PHASE}")
    build_path = weight_dir / "BUILD_RECORD.json"
    build = json.loads(build_path.read_text()) if build_path.is_file() else {}
    if build.get("artifact_sha256") != sha:
        raise S.CheckFailed(f"{C}: BUILD_RECORD.json artifact_sha256 differs from the artefact")
    bad = [k for k, v in (("artifact metadata", meta.get("base_artifact_sha256")),
                          ("BUILD_RECORD.json", build.get("base_artifact_sha256"))) if v != base_sha256]
    if bad:
        raise S.CheckFailed(f"{C}: base_artifact_sha256 in {', '.join(bad)} is not v0.5 B's artefact")
    info = {"kind": "network", "labelled_examples": 400, "revisions": 2, "tools": True, "base": B,
            "weight_dir": V._rel(weight_dir), "artifact_sha256": sha, "apply_report_artifact_sha256": apply_sha,
            "build_record_sha256": S.sha256_file(build_path), "base_artifact_sha256": base_sha256,
            "artifact_metadata": meta, "nonzero_parameters": art["nonzero_parameters"], "stand_in": False}
    return dataclasses.replace(spec, name=C), info


def load_stand_in(path: Path) -> tuple[S.NetSpec, dict[str, Any]]:
    """Dry runs only: any ``.dlw`` in the Phase C column (checked against an ``artifact.sha256`` beside it, if any)."""
    path = Path(path)
    sha = path.parent / "artifact.sha256"
    try:
        spec, art = v05_scoring.load_run_artifact(path, sha if sha.is_file() else None)
    except (v05_scoring.ScoringError, FileNotFoundError) as exc:
        raise S.CheckFailed(f"{C} stand-in: {exc}") from None
    info = {"kind": "network", "stand_in": True,
            "WARNING": ("STAND-IN: this column is NOT the Phase C network; a dry run put the artefact below in "
                        "the Phase C column"),
            "artifact_path": V._rel(path), "artifact_sha256": art["artifact_sha256"],
            "artifact_metadata": art["artifact_metadata"], "nonzero_parameters": art["nonzero_parameters"]}
    return dataclasses.replace(spec, name=C), info


def load_eval_columns(phase_c_artifact: Path | None = None, *, phase_c_dir: Path = PHASE_C_DIR) -> EvalColumns:
    """Every column, checked. ``phase_c_artifact`` (dry runs only) puts a stand-in in the Phase C column."""
    v05 = V.load_v05_columns()
    by_name = {s.name: s for s in v05.specs}
    b_sha = v05.info[B]["artifact_sha256"]
    if b_sha != P.B_SHA256:
        raise S.CheckFailed(f"{B}: artefact {b_sha} is not the pinned Phase B artefact")
    if phase_c_artifact is None:
        c_spec, c_info = load_phase_c(phase_c_dir, base_sha256=b_sha)
    else:
        c_spec, c_info = load_stand_in(phase_c_artifact)
    specs = [c_spec, by_name[B], by_name[R1]]
    info: dict[str, dict[str, Any]] = {C: c_info, B: v05.info[B], R1: v05.info[R1]}
    inputs: dict[str, Any] = {C: {k: c_info[k] for k in c_info if k.endswith("sha256") or k == "stand_in"},
                              B: v05.inputs[B], R1: v05.inputs[R1]}
    for short, phase in S.PHASES:
        d = v03_scoring.phase_dir(phase)
        model, meta, _pol, shas = v03_scoring.load_phase(phase, d)
        name = f"{short} network"
        specs.append(S.NetSpec(name, "artifact", str(d / "artifact.dlw"), shas["artifact_sha256"]))
        from ..v03 import artifact as v03_artifact

        info[name] = {"kind": "network", **S.LABELLED[phase], **shas, "archive_dir": V._rel(d),
                      "nonzero_parameters": v03_artifact.sparsity(model)["overall"]}
        inputs[name] = {**shas, "packet_sha256": meta.get("packet_sha256")}
    spec, _log, inf = S.check_phase_c_run("all", 0)
    if spec.name != V03C_ALL:
        raise S.CheckFailed(f"C-all column is named {spec.name!r}, expected {V03C_ALL!r}")
    specs.append(spec)
    info[V03C_ALL] = inf
    inputs[V03C_ALL] = {"artifact_sha256": inf["artifact_sha256"], "training_log_sha256": inf["training_log_sha256"]}
    info[PRIOR] = {"kind": "prior", "definition": v05_scoring.DEFINITIONS[PRIOR], "reference_only": True}
    if [s.name for s in specs] + [PRIOR] != list(COLUMN_ORDER):
        raise S.CheckFailed("column order differs from COLUMN_ORDER")
    return EvalColumns(specs, info, inputs, v05, phase_c_artifact is not None)


# ---------------------------------------------------------------------------
# Predictions and scoring
# ---------------------------------------------------------------------------


def predict(sd: S.SplitData, cols: EvalColumns, pool: Any = None, *, workers: int = 0,
            progress: Callable[[str], None] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Class per row for every column (prior included), latency and label-free logit diagnostics."""
    say = progress or (lambda m: None)
    res = v05_scoring.network_predictions(cols.specs, sd.data, pool, workers=workers,
                                          progress=lambda m: say(f"{sd.split} {m}"))
    gate = np.array([r.gate_passed for r in sd.data.rows], dtype=bool)
    preds: dict[str, np.ndarray] = {}
    latency: dict[str, Any] = {}
    logits: dict[str, Any] = {}
    for spec in cols.specs:
        r = res[spec.name]
        if np.any(r.pred[gate] < 0):
            raise RuntimeError(f"{sd.split}: a gate-passing window has no {spec.name} prediction")
        preds[spec.name], latency[spec.name], logits[spec.name] = r.pred, r.latency, r.logits
    preds[PRIOR] = np.where(gate, 0, -1).astype(np.int64)
    return {c: preds[c] for c in COLUMN_ORDER}, latency, logits


def population_masks(sd: S.SplitData, disclosed: frozenset[str] = frozenset()) -> dict[str, np.ndarray]:
    """val: ``excluding_disclosed`` and ``all_val``; held-out OOD: ``unseen``, ``seen``, ``all``; held-out
    in-distribution: ``all``."""
    rows = sd.data.rows
    if sd.split == "val":
        return S.population_masks(sd, disclosed)
    allm = np.ones(len(rows), dtype=bool)
    if sd.split == "heldout_ood":
        unseen = {P.ood_row_family(f) for f in P.UNSEEN_FAMILIES}
        seen = {P.ood_row_family(f) for f in P.SEEN_FAMILIES}
        fam = np.array([r.family for r in rows], dtype=object)
        m_unseen = np.isin(fam, sorted(unseen))
        m_seen = np.isin(fam, sorted(seen))
        if not np.all(m_unseen ^ m_seen):
            raise S.CheckFailed("heldout_ood: a window belongs to neither or both of the seen / unseen families")
        return {"unseen": m_unseen, "seen": m_seen, "all": allm}
    if sd.split == "heldout_indist":
        return {"all": allm}
    raise ValueError(f"unknown set {sd.split!r}")


def score_set(sd: S.SplitData, preds: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray], *,
              pairs: Mapping[str, tuple[str, str]] = PAIRS, n_resamples: int = M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """``m4.score_columns`` (overall and per family) and the paired differences, per population."""
    rows = sd.data.rows
    out: dict[str, Any] = {"gate": m4.gate_report(rows), "populations": {}}
    for pop, mask in masks.items():
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        sp = {k: np.asarray(v)[idx] for k, v in preds.items()}
        sc = m4.score_columns(sub, sp, {}, {}, n_resamples=n_resamples)
        sc["paired_differences"] = V.paired_differences(sub, sp, pairs, n_resamples=n_resamples)
        out["populations"][pop] = sc
    return out


def summary(scored: Mapping[str, Any]) -> dict[str, Any]:
    return V.summary(scored, None)


# ---------------------------------------------------------------------------
# Val checks
# ---------------------------------------------------------------------------


def compare_v03_val_blocks(scored: Mapping[str, Any], v03_rep: Mapping[str, Any]) -> dict[str, Any]:
    """The v0.3 columns' val blocks (overall and per family, both populations) against ``v03_sealed.json``
    (recorded, not fatal)."""
    rec = v03_rep.get("splits", {}).get("val")
    if rec is None or "populations" not in rec:
        return {"compared": False, "reason": "v03_sealed.json has no scored val split"}
    diffs: dict[str, str | None] = {c: None for c in V03_COLUMNS}
    for pop, mine in scored["populations"].items():
        theirs = rec["populations"].get(pop)
        if theirs is None:
            raise S.CheckFailed(f"val: population {pop} is not in v03_sealed.json")
        blocks = [("overall", theirs["overall"], mine["overall"])]
        blocks += [(f"per_family/{f}", theirs["per_family"][f], mine["per_family"].get(f)) for f in sorted(theirs["per_family"])]
        for where, a, b in blocks:
            if b is None:
                raise S.CheckFailed(f"val {pop}: family {where} missing")
            for c in V03_COLUMNS:
                V._note_diff(diffs, c, a["columns"][c], b["columns"][c], f"val {pop} {where}")
    bad = sorted(k for k, v in diffs.items() if v is not None)
    return {"compared": True, "reproduced": not bad, "not_reproduced": bad,
            "first_difference": {k: v for k, v in diffs.items() if v is not None}}


def caveat_pairs(scored: Mapping[str, Any], not_reproduced: Sequence[str]) -> None:
    bad = set(not_reproduced)
    for pop in scored["populations"].values():
        for per_metric in pop["paired_differences"].values():
            for d in per_metric.values():
                hit = sorted({d["minuend"], d["subtrahend"]} & bad)
                if hit:
                    d["caveat"] = f"{', '.join(hit)} did not reproduce its v03_sealed.json val blocks"


def val_checks(sd: S.SplitData, preds: Mapping[str, np.ndarray], scored: Mapping[str, Any], cols: EvalColumns,
               disclosed: frozenset[str], *, n_resamples: int = M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Full val only: rows, disclosure and the v0.5 reports (fatal), the v0.3 blocks (recorded)."""
    out: dict[str, Any] = {}
    if v03_scoring.disclosed_sessions(sd.data.rows) != disclosed:
        raise S.CheckFailed("experiments/v0_3/DISCLOSURE_PRIVATE.json differs from the v0.3 disclosure rule")
    if S.rows_sha256(sd.data.rows) != v04_scoring.VAL_ROWS_SHA256:
        raise S.CheckFailed("val rows differ from the v0.3 sealed record (v04.scoring.VAL_ROWS_SHA256)")
    out["disclosure"] = {"matches_rule": True, "count": len(disclosed),
                         "sha256_of_sorted_ids": v03_scoring._session_set_sha256(disclosed),
                         "val_rows_sha256": v04_scoring.VAL_ROWS_SHA256}
    out["v05b_disclosure"] = V.check_v05_disclosure(disclosed)
    sp = S.SplitPredictions(dict(preds), {}, {}, {})
    out["v05_val_reports"] = V.v05_val_check(sd, sp, scored, cols.v05, disclosed, n_resamples=n_resamples)
    if V03_SEALED_REPORT.is_file():
        v03_rep = json.loads(V03_SEALED_REPORT.read_text())
        out["v03_val_blocks"] = compare_v03_val_blocks(scored, v03_rep)
        out["v03_val_blocks"]["report_sha256"] = S.sha256_file(V03_SEALED_REPORT)
    else:
        out["v03_val_blocks"] = {"compared": False, "reason": f"{V._rel(V03_SEALED_REPORT)} missing"}
    return out


# ---------------------------------------------------------------------------
# Held-out inputs
# ---------------------------------------------------------------------------


def check_disclosure_private(path: Path) -> tuple[frozenset[str], dict[str, Any]]:
    """The OOD disclosure's private record: master seed, families (seen only) and its session ids."""
    rec = json.loads(Path(path).read_text())
    if rec.get("master_seed") != P.DISCLOSURE_SEED:
        raise S.CheckFailed(f"{V._rel(path)}: master seed {rec.get('master_seed')} != {P.DISCLOSURE_SEED}")
    fams = {w["family"] for w in rec["windows"]}
    if not fams <= set(P.SEEN_FAMILIES):
        raise S.CheckFailed(f"{V._rel(path)}: disclosed families {sorted(fams - set(P.SEEN_FAMILIES))} are not seen families")
    ids = frozenset(w["session_id"] for w in rec["windows"])
    return ids, {"path": V._rel(path), "sha256": S.sha256_file(Path(path)), "windows": len(rec["windows"]),
                 "families_subset_of_seen": True}


CODE_FILES = (*P.SCORING_CODE_FILES, PRE_REGISTRATION, MANIFEST)


def code_hashes() -> dict[str, str | None]:
    return {p: (S.sha256_file(REPO / p) if (REPO / p).is_file() else None) for p in CODE_FILES}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def evaluate(cols: EvalColumns, *, heldout: bool, official: bool, report_path: Path, marker_path: Path | None = None,
             data_root: Path = P.DATA_ROOT, manifest_path: Path = REPO / MANIFEST,
             disclosure_private: Path = REPO / DISCLOSURE_OOD_PRIVATE, pool: Any = None, workers: int = 0,
             val_limit: int | None = None, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
             progress: Callable[[str], None] | None = None, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score val (and, with ``heldout``, the held-out sets after writing the attempt record) and write the report
    atomically to ``report_path``. Returns the report.

    Order with ``heldout``: manifest checks and file hashes (no data decoded), the OOD disclosure record, val with
    its checks, the attempt record at ``marker_path`` (created exclusively), then each held-out set is decoded,
    checked against the manifest and the disclosure, predicted and scored. A failure after the attempt record
    raises :class:`StageError` and leaves the record (a spent attempt)."""
    import time

    say = progress or (lambda m: None)
    t0 = time.time()
    if official and (cols.phase_c_stand_in or not heldout):
        raise ValueError("an official report needs the held-out sets and the real Phase C artefact")
    checks: dict[str, Any] = {}
    inputs: dict[str, Any] = {"columns": cols.inputs, "code_sha256": code_hashes()}
    man: dict[str, Any] | None = None
    reserved_extra: frozenset[str] = frozenset()
    if heldout:
        if marker_path is None:
            raise ValueError("a held-out run needs marker_path")
        for f in (Path(report_path), Path(marker_path)):
            if f.exists():
                raise S.CheckFailed(f"{f} exists; the held-out evaluation runs once")
        man = json.loads(Path(manifest_path).read_text())
        problems = P.manifest_problems(man, official=official)
        if problems:
            raise S.CheckFailed("held-out manifest: " + "; ".join(problems))
        file_sha = {name: P.verify_set_files(data_root, name, man["sets"][name]) for name in P.SET_NAMES}
        inputs["heldout_manifest"] = {"path": V._rel(Path(manifest_path)), "sha256": S.sha256_file(Path(manifest_path)),
                                      "files_sha256": file_sha}
        if Path(disclosure_private).is_file():
            reserved_extra, checks["ood_disclosure"] = check_disclosure_private(Path(disclosure_private))
        elif official:
            raise S.CheckFailed(f"{V._rel(Path(disclosure_private))} missing")
        else:
            checks["ood_disclosure"] = {"checked": False, "reason": "no disclosure record (dry run)"}
        say("held-out manifest and file hashes ok (no held-out data decoded)")

    # Val.
    disclosed = S.private_disclosed()
    sets: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    logits: dict[str, Any] = {}
    sd_val = S.build_split("val", limit=val_limit)
    say(f"val: {len(sd_val.data.sessions)} sessions, {len(sd_val.data.rows)} windows")
    preds, latency["val"], logits["val"] = predict(sd_val, cols, pool, workers=workers, progress=say)
    sc_val = score_set(sd_val, preds, population_masks(sd_val, disclosed), n_resamples=n_resamples)
    if val_limit is None:
        checks["val"] = val_checks(sd_val, preds, sc_val, cols, disclosed, n_resamples=n_resamples)
        blk = checks["val"]["v03_val_blocks"]
        if blk.get("compared") and not blk["reproduced"]:
            caveat_pairs(sc_val, blk["not_reproduced"])
    else:
        checks["val"] = "skipped (limited dry run)"
    sets["val"] = sc_val
    say("val: scored and checked")
    del sd_val, preds

    attempt_info: dict[str, Any] | None = None
    if heldout:
        assert man is not None and marker_path is not None
        previous = previous_attempts(Path(marker_path).parent)
        record = {"started_utc": (meta or {}).get("started_utc"), "git": S.git_state(),
                  "attempt_number": len(previous) + 1, "previous_attempts": previous, "inputs": inputs,
                  "official": official}
        attempt = S.write_attempt_record(record, Path(marker_path))
        attempt_info = {"marker": V._rel(Path(marker_path)), "number": len(previous) + 1,
                        "previous_attempts": previous}
        say(f"attempt record written: {attempt_info['marker']} (no held-out data decoded before this)")
        try:
            reserved = P.reserved_session_ids(disclosure_private=None)
            if reserved_extra:
                reserved = {**reserved, "OOD disclosure windows (private record)": reserved_extra}
            checks["disjointness"] = {}
            for name in P.SET_NAMES:
                sd = P.load_set(data_root, name, man["sets"][name], attempt=attempt)
                say(f"{name}: {len(sd.data.sessions)} sessions, {len(sd.data.rows)} windows (manifest hashes ok)")
                checks["disjointness"][name] = P.check_disjoint(sd.data.session_ids, name, reserved)
                reserved = {**reserved, f"held-out set {name}": frozenset(sd.data.session_ids)}
                preds, latency[name], logits[name] = predict(sd, cols, pool, workers=workers, progress=say)
                sets[name] = score_set(sd, preds, population_masks(sd), n_resamples=n_resamples)
                say(f"{name}: scored")
                del sd, preds
        except BaseException as exc:
            raise StageError(f"held-out stage stopped ({type(exc).__name__}: {exc}); the attempt record stays as a "
                             "spent attempt") from exc

    order = [s for s in ("heldout_ood", "heldout_indist", "val") if s in sets]
    notes = [
        "val is reference only: v0.3 Phase C early-stopped on 500 val sessions; the 200 disclosed sessions (v0.3's "
        "disclosure, shown to v0.5 Phase B and again to Phase C) are excluded from the primary val population",
        "v0.5 networks have no reference policy (no H2 fidelity); paired differences are overall, not per family",
        "prior always-PRODUCTIVE is a reference column, not compared",
    ]
    if cols.phase_c_stand_in:
        notes.insert(0, "WARNING: STAND-IN: the 'v0.5 Phase C network' column is NOT the Phase C network (dry run; "
                        f"artefact {cols.info[C]['artifact_path']}, sha256 {cols.info[C]['artifact_sha256']})")
    report = {
        "report": "v05c_heldout" if official else ("v05c_heldout_unofficial" if heldout else "v05c_heldout_dryrun_val"),
        "official": official,
        "spec_version": "0.5",
        "phase_c_stand_in": cols.phase_c_stand_in,
        "pre_registration": {"path": PRE_REGISTRATION, "sha256": code_hashes()[PRE_REGISTRATION]},
        "authority": ("adjudicator decision 2026-11-07: v0.5 Phase C (a new Fable session revises v0.5 B with v0.3's "
                      "200 val windows and 200 OOD windows from 6 of the 10 OOD families), evaluated once on fresh "
                      "held-out sets; the sealed v0.2/v0.3 sets are not used"),
        "attempt": attempt_info,
        "val_limit_sessions": val_limit,
        "bootstrap_resamples": n_resamples,
        "inputs": inputs,
        "columns": {c: cols.info[c] for c in COLUMN_ORDER},
        "column_order": list(COLUMN_ORDER),
        "pairs": {k: list(v) for k, v in PAIRS.items()},
        "primary": {"pairs": list(PRIMARY_PAIRS), "set": PRIMARY_POPULATION[0], "population": PRIMARY_POPULATION[1]},
        "comparison_order": [list(x) for x in COMPARISON_ORDER],
        "family_split": {"seen": list(P.SEEN_FAMILIES), "unseen": list(P.UNSEEN_FAMILIES), "rule": P.FAMILY_RULE_TEXT},
        "populations": {"heldout_ood": {"unseen": "sessions of the 4 unseen families (primary)",
                                        "seen": "sessions of the 6 families the OOD disclosure drew from",
                                        "all": "all 10 families"},
                        "heldout_indist": {"all": "all sessions"}, "val": dict(v03_scoring.POPULATIONS)},
        "paired_definition": ("metric(minuend) - metric(subtrahend) over the shared session-bootstrap draws of the "
                              "population (overall): macro_f1 by v04.scoring.paired_macro_f1, false_alarm_rate by the "
                              "same computation (v05.sealed.paired_metric)"),
        "checks": checks,
        "summary": {s: summary(sets[s]) for s in order},
        "sets": {s: {**sets[s], "latency": latency[s]} for s in order},
        "logits": logits,
        "notes": notes,
        "meta": dict(meta or {}),
        "runtime_s": time.time() - t0,
    }
    S.write_json_atomic(Path(report_path), report)
    return report


if __name__ == "__main__":
    raise SystemExit(_main())
