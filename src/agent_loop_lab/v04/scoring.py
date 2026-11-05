"""v0.4 validation scoring bridge: Qwen per-window logits -> the unchanged v0.3 metrics. Harness only.

The Qwen environment writes one ``.npz`` per column (``hypaqwen.run_readout``): ``label_logits``
``[rows, 4]``, ``logsumexp``, ``session_ids`` and ``row_pos`` in the export's row order. This
module runs in the frozen v0.3 environment and:

* rebuilds the val window rows from ``datasets/v0_4/val.labels.npz``, and checks them against the
  ``rows_sha256`` the v0.3 sealed run recorded for val, so every Qwen column is scored on exactly
  the v0.3 windows;
* checks each column's session ids and row positions against those rows, and refuses non-finite
  logits. The class is the first-index argmax over the four label-word logits, the tie rule of
  v0.3 ``model.predict``;
* adds the two constant priors (``prior dev-majority`` from the dev prior the v0.3 sealed run
  recorded, and ``prior always-PRODUCTIVE``) and, when given, the v0.3 columns re-scored on val
  (:mod:`.v03_val`), so every column of a report shares the same bootstrap draws. A v0.3 columns
  file is not taken on its own say-so: its columns and the priors are re-scored on the full val
  rows and must equal ``v03_sealed.json`` exactly (``v03_val.reproduction_check``), on a subset
  report too; the v0.3 columns the file does not hold (not reproduced, PREREGISTRATION §1) are
  cited from ``v03_sealed.json`` on full val;
* scores with ``eval.m4.score_columns`` (as v0.3 scored Phase C), with no fidelity pairs. H2 has
  no meaning without a reference policy, and Q5 is not tested in v0.4;
* reports both v0.3 populations: ``excluding_disclosed`` (primary) and ``all_val``, and in each the
  paired macro-F1 differences (``paired_macro_f1``): the difference of two columns over the same
  bootstrap draws ``score_columns`` uses, with its 95% percentile interval. The §9 pairs are built
  in (:func:`default_pairs`, each when both its columns are present); extra pairs add to them.

**Coverage** (PREREGISTRATION §3.5). A column whose ``meta["subset"]`` is set (``run_readout
--sessions``) is accepted only when the subset file's sha256 equals that record and the column's
session ids are exactly the file's ids in export order. The default subset file is the quick-look
subset ``datasets/v0_4/VAL_QUICK.txt``. Every Qwen column of one report must cover the same
sessions (a mix of full and subset columns, or of two subsets, is refused); the window rows, the
priors and the v0.3 columns are then restricted to those sessions. The report records the subset
(``file``, ``sha256``, ``sessions``).

It refuses unless the fresh-test ``MANIFEST_fresh_test.json`` and ``REGISTRATION.json`` are committed
unchanged and the manifest cites that registration (:func:`fresh_test_committed`). It touches only
those two committed files: the fresh-test data files are not opened, listed or hashed, and no
sealed set is scored in v0.4 (PREREGISTRATION §3.4). Val numbers are never shown to the weight
author.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..eval import m4
from ..eval import metrics as M
from ..eval.collisions import WindowRow
from ..schema import CLASS_NAMES, OracleLabel
from ..v03 import scoring as v03_scoring
from ..v03 import sealed as v03_sealed
from . import export, fresh_test

__all__ = [
    "SPLIT", "VAL_ROWS_SHA256", "VAL_QUICK", "VAL_QUICK_SHA256", "PRIOR_COLUMNS", "ScoringError",
    "rows_from_labels", "load_rows", "checked_disclosed", "read_subset", "restrict", "column_meta", "coverage",
    "load_column", "recorded_dev_prior", "recorded_val_blocks", "prior_columns", "load_v03_columns", "V03_PAIRS",
    "default_pairs", "paired_macro_f1", "diagnostics", "score", "build_report", "fresh_test_committed", "report",
]

SPLIT = "val"
# artifacts/reports/v03_sealed.json checks.identity_v02.val.rows_sha256 (87,176 windows, 4,004 sessions).
VAL_ROWS_SHA256 = "cdf6cce88e28499a8c513d0dd23126812db60590f7f09570a28a138a9e3196bc"
# The quick-look subset (scripts/v04_val_quick_subset.py; PREREGISTRATION §3.5), fixed before any edit existed.
VAL_QUICK = export.V04_DATA / "VAL_QUICK.txt"
VAL_QUICK_SHA256 = "b77f74150ecc19edf51ed57b5bc77f9e9eacd51b1992c0467ab4a4bce6cfd556"
PRIOR_COLUMNS = v03_sealed.PRIOR_COLUMNS  # ("prior dev-majority", "prior always-PRODUCTIVE")
BASE_COLUMN = "Qwen-0"
# The §9 comparisons with columns other than Qwen-0 (PREREGISTRATION §9; exact column names).
V03_A, V03_B = "v0.3 A network", "v0.3 B network"
V03_PAIRS: tuple[tuple[str, str], ...] = (
    ("WSP-B", V03_B),  # Q2 (primary)
    ("WSP-A", V03_A),
    (BASE_COLUMN, V03_A),  # reported with Q1: the value of hand-building
    ("WSP-F", "WSP-B"),
)
_LABELS = tuple(OracleLabel)
H2_NOT_APPLICABLE = {"applicable": False, "reason": "no reference policy for a Qwen column; H2 has no v0.4 counterpart"}
VISIBILITY = "harness only: never shown to the weight author (PREREGISTRATION §3.2)"
QUICK_LOOK_NOTE = ("quick look (PREREGISTRATION §3.5): reported beside, never instead of, the full val result; "
                   "the full-val excluding_disclosed numbers are the result of record")


class ScoringError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Rows, coverage, columns
# ---------------------------------------------------------------------------


def rows_from_labels(lab: Mapping[str, np.ndarray]) -> list[WindowRow]:
    """Window rows from a labels export (``export.labels_path``), in its row order."""
    ids, fams = [str(s) for s in lab["session_ids"]], [str(f) for f in lab["families"]]
    return [
        WindowRow(session_id=ids[s], family=fams[s], split=SPLIT, endpoint=int(e), label=_LABELS[int(l)],
                  gate_passed=bool(g), vector=None)
        for s, e, l, g in zip(lab["row_session"].tolist(), lab["row_endpoint"].tolist(), lab["row_label"].tolist(),
                              lab["row_gate"].tolist())
    ]


def load_rows(data_dir: Path = export.V04_DATA) -> tuple[list[WindowRow], dict[str, np.ndarray]]:
    """The val window rows from the export, checked against the v0.3 val rows hash."""
    M.check_split(SPLIT)
    with np.load(export.labels_path(SPLIT, data_dir)) as z:
        lab = {k: z[k] for k in z.files}
    rows = rows_from_labels(lab)
    got = v03_sealed.rows_sha256(rows)
    if got != VAL_ROWS_SHA256:
        raise ScoringError(f"val rows sha256 {got} != v0.3 sealed record {VAL_ROWS_SHA256}")
    return rows, lab


def checked_disclosed(rows: Sequence[WindowRow]) -> frozenset[str]:
    """The 200 Phase B disclosed sessions by the v0.3 rule on the full val rows, checked against the private record."""
    disclosed = v03_scoring.disclosed_sessions(rows)
    if disclosed != v03_sealed.private_disclosed():
        raise ScoringError("disclosed sessions differ from experiments/v0_3/DISCLOSURE_PRIVATE.json")
    return disclosed


def read_subset(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Session ids of a subset file (one per line, as ``hypaqwen.run_readout.read_sessions``) and its record."""
    raw = Path(path).read_bytes()
    ids = [s for s in raw.decode().split("\n") if s]
    if len(set(ids)) != len(ids):
        raise ScoringError(f"{path}: duplicate session ids")
    return ids, {"file": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest(), "sessions": len(ids)}


def restrict(rows: Sequence[WindowRow], lab: Mapping[str, np.ndarray], ids: Sequence[str]
             ) -> tuple[list[WindowRow], dict[str, np.ndarray], np.ndarray]:
    """``(rows, lab, row_mask)`` restricted to the sessions ``ids``, in export order."""
    all_ids = [str(s) for s in lab["session_ids"]]
    missing = set(ids) - set(all_ids)
    if missing:
        raise ScoringError(f"{len(missing)} subset sessions are not in {SPLIT} (e.g. {sorted(missing)[0]})")
    keep = set(ids)
    sess_mask = np.array([s in keep for s in all_ids], dtype=bool)
    row_session = np.asarray(lab["row_session"])
    row_mask = sess_mask[row_session]
    new_index = np.cumsum(sess_mask) - 1
    sub = {
        "session_ids": np.asarray(lab["session_ids"])[sess_mask],
        "families": np.asarray(lab["families"])[sess_mask],
        "row_session": new_index[row_session[row_mask]],
    }
    for k in ("row_endpoint", "row_label", "row_gate", "row_pos"):
        sub[k] = np.asarray(lab[k])[row_mask]
    idx = np.flatnonzero(row_mask).tolist()
    return [rows[i] for i in idx], sub, row_mask


def column_meta(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def coverage(metas: Mapping[str, Mapping[str, Any]], subset_file: Path | None = None) -> dict[str, Any] | None:
    """The sessions every column covers: ``None`` for full val, else the subset (ids and record).

    Refuses a mix (full and subset columns, or two subsets), a subset whose file has another sha256,
    and ``subset_file`` with full-val columns."""
    if not metas:
        raise ScoringError("no columns")
    shas = {name: None if m.get("subset") is None else (m["subset"].get("sha256") or "<no sha256>")
            for name, m in metas.items()}
    if len(set(shas.values())) > 1:
        raise ScoringError(f"columns cover different sessions (subset sha256 per column: {shas}); "
                           "all columns of one report must cover the same sessions")
    sha = next(iter(shas.values()))
    if sha is None:
        if subset_file is not None:
            raise ScoringError(f"--subset {subset_file} given, but every column is a full {SPLIT} readout")
        return None
    path = VAL_QUICK if subset_file is None else Path(subset_file)
    ids, info = read_subset(path)
    if info["sha256"] != sha:
        raise ScoringError(f"the columns' subset sha256 {sha} != {path.name} sha256 {info['sha256']}")
    for name, m in metas.items():
        if m["subset"].get("sessions") not in (None, len(ids)):
            raise ScoringError(f"{name}: subset record has {m['subset'].get('sessions')} sessions, {path.name} {len(ids)}")
    return {**info, "ids": ids, "val_quick": info["sha256"] == VAL_QUICK_SHA256}


def load_column(path: Path, lab: Mapping[str, np.ndarray], *, subset_sha256: str | None = None
                ) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """``(pred, meta, arrays)`` for one readout file, aligned to ``lab`` or refused.

    With ``subset_sha256`` None the column must be a full val readout; otherwise its
    ``meta["subset"]["sha256"]`` must equal it, and ``lab`` is the export restricted to the subset
    (:func:`restrict`), so its session ids must be exactly the subset's, in export order."""
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(str(arr.pop("meta")))
    if meta.get("split") != SPLIT or meta.get("limit") is not None:
        raise ScoringError(f"{path.name}: not a complete {SPLIT} run (split={meta.get('split')}, limit={meta.get('limit')})")
    got_subset = meta.get("subset")
    if subset_sha256 is None and got_subset is not None:
        raise ScoringError(f"{path.name}: a subset readout ({got_subset.get('file')}) in a full {SPLIT} report")
    if subset_sha256 is not None and (got_subset is None or got_subset.get("sha256") != subset_sha256):
        raise ScoringError(f"{path.name}: subset {got_subset} is not the report's subset (sha256 {subset_sha256})")
    if list(arr["session_ids"]) != list(lab["session_ids"]):
        raise ScoringError(f"{path.name}: session ids differ from the export (order or count)")
    if "row_pos" not in arr or not np.array_equal(arr["row_pos"], lab["row_pos"]):
        raise ScoringError(f"{path.name}: row positions differ from the export")
    logits = arr["label_logits"]
    if logits.shape != (len(lab["row_pos"]), len(CLASS_NAMES)):
        raise ScoringError(f"{path.name}: label_logits shape {logits.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(arr["logsumexp"]).all():
        raise ScoringError(f"{path.name}: non-finite logits")
    return np.argmax(logits, axis=1).astype(np.int64), meta, arr


def recorded_dev_prior(path: Path | None = None) -> dict[str, Any]:
    """The dev prior the v0.3 sealed run recorded (``artifacts/reports/v03_sealed.json`` ``dev_prior``)."""
    return json.loads(Path(v03_sealed.SEALED_REPORT if path is None else path).read_text())["dev_prior"]


def recorded_val_blocks(path: Path | None = None) -> dict[str, Any]:
    """``splits.val.populations`` of the v0.3 sealed report (both populations' val blocks)."""
    return json.loads(Path(v03_sealed.SEALED_REPORT if path is None else path).read_text())["splits"]["val"]["populations"]


def prior_columns(rows: Sequence[WindowRow], majority_index: int) -> dict[str, np.ndarray]:
    """The two constant priors as v0.3 built them (``v03.sealed.predict_split``): a class on gate-passing windows."""
    gate = np.array([r.gate_passed for r in rows], dtype=bool)
    return {PRIOR_COLUMNS[0]: np.where(gate, int(majority_index), -1).astype(np.int64),
            PRIOR_COLUMNS[1]: np.where(gate, 0, -1).astype(np.int64)}


def load_v03_columns(path: Path, rows: Sequence[WindowRow], disclosed: frozenset[str], majority_index: int, *,
                     recorded: Mapping[str, Any] | None = None, n_resamples: int = M.BOOTSTRAP_RESAMPLES
                     ) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, Any]]:
    """``(preds, groups, record)`` from a :mod:`.v03_val` file, aligned to the full val ``rows`` and
    re-checked: its columns and the priors are re-scored on ``rows`` (both populations) and must equal
    ``recorded`` (default :func:`recorded_val_blocks`) exactly, whatever the file says about itself."""
    from . import v03_val

    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        pred, names = z["pred"], [str(n) for n in z["names"]]
    if meta.get("format") != v03_val.FORMAT:
        raise ScoringError(f"{Path(path).name}: format {meta.get('format')!r}, not {v03_val.FORMAT}")
    if meta.get("val_rows_sha256") != VAL_ROWS_SHA256:
        raise ScoringError(f"{Path(path).name}: computed on other val rows ({meta.get('val_rows_sha256')})")
    if pred.shape != (len(names), len(rows)):
        raise ScoringError(f"{Path(path).name}: pred shape {pred.shape}, expected ({len(names)}, {len(rows)})")
    reserved = sorted(set(names) & set(PRIOR_COLUMNS))
    if reserved:
        raise ScoringError(f"{Path(path).name}: holds prior columns {reserved}; the bridge builds the priors")
    preds = {n: pred[j].astype(np.int64) for j, n in enumerate(names)}
    groups = {g: list(m) for g, m in meta["groups"].items()}
    stray = sorted({m for ms in groups.values() for m in ms} - set(names))
    if stray:
        raise ScoringError(f"{Path(path).name}: groups name columns the file does not hold: {stray}")
    recorded = recorded_val_blocks() if recorded is None else recorded
    try:
        check = v03_val.reproduction_check(rows, {**preds, **prior_columns(rows, majority_index)}, groups, disclosed,
                                           recorded, n_resamples=n_resamples)
    except v03_val.ReproductionError as exc:
        raise ScoringError(f"{Path(path).name}: does not reproduce the v0.3 val blocks: {exc}") from None
    rec_cols = set(recorded["excluding_disclosed"]["overall"]["columns"])
    rec_groups = set(recorded["excluding_disclosed"]["overall"].get("groups", {}))
    missing = sorted(c for c in rec_cols - set(names) - set(PRIOR_COLUMNS) if c.endswith(" network") or " seed" in c)
    record = {"file": Path(path).name, "sha256": _sha256_file(Path(path)), "columns": names,
              "reproduction_recorded": meta.get("reproduction"), "reproduction_rechecked": check,
              "not_reproduced": meta.get("not_reproduced", {}), "groups_not_reproduced": meta.get("groups_not_reproduced", {}),
              "cited_not_paired": missing, "groups_cited_not_paired": sorted(rec_groups - set(groups))}
    return preds, groups, record


def cited_v03(recorded: Mapping[str, Any], columns: Sequence[str], groups: Sequence[str]) -> dict[str, Any]:
    """The v0.3 val blocks of columns and groups the report does not re-score (cited, no paired CI)."""
    out: dict[str, Any] = {}
    for pop in ("excluding_disclosed", "all_val"):
        ov = recorded[pop]["overall"]
        out[pop] = {"columns": {c: ov["columns"][c] for c in columns},
                    "groups": {g: ov["groups"][g] for g in groups}}
    return out


# ---------------------------------------------------------------------------
# Paired differences
# ---------------------------------------------------------------------------


def pair_name(a: str, b: str) -> str:
    return f"{a} - {b}"


def default_pairs(qwen_columns: Sequence[str], other_columns: Sequence[str] = ()) -> dict[str, tuple[str, str]]:
    """The §9 comparisons, each when both its columns are present: every Qwen column minus Qwen-0,
    Qwen-0 minus the dev-majority prior (Q1), and :data:`V03_PAIRS` (WSP-B − v0.3 B network for Q2,
    WSP-A − v0.3 A network, Qwen-0 − v0.3 A network, WSP-F − WSP-B)."""
    qwen = list(qwen_columns)
    present = set(qwen) | set(other_columns)
    out: dict[str, tuple[str, str]] = {}
    if BASE_COLUMN in qwen:
        out.update({pair_name(c, BASE_COLUMN): (c, BASE_COLUMN) for c in qwen if c != BASE_COLUMN})
        if PRIOR_COLUMNS[0] in present:
            out[pair_name(BASE_COLUMN, PRIOR_COLUMNS[0])] = (BASE_COLUMN, PRIOR_COLUMNS[0])
    for a, b in V03_PAIRS:
        if a in qwen and b in present:
            out[pair_name(a, b)] = (a, b)
    return out


def _macro_f1_draws(rows: Sequence[WindowRow], preds: Mapping[str, np.ndarray], columns: Sequence[str], *,
                    n_resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Point macro-F1 ``(k,)`` and its bootstrap draws ``(n_resamples, k)`` for ``columns``, from the session
    counts and draws ``m4.score_columns`` uses (same session order, ``metrics.bootstrap_distribution``, same seed)."""
    rows = list(rows)
    ids, _, sidx = M.session_index(rows)
    n_s = len(ids)
    counts = np.stack([m4.single_counts(rows, preds[c], sidx, n_s) for c in columns], axis=1)
    flat = counts.reshape(n_s, -1)
    k = len(columns)

    def fn(c: np.ndarray) -> dict[str, np.ndarray]:
        return {"macro_f1": M.metrics_from_counts(c.reshape(c.shape[:-1] + (k, M.MODEL_COLUMNS)))["macro_f1"]}

    point = fn(flat.sum(axis=0).astype(np.float64))["macro_f1"]
    dist = M.bootstrap_distribution(flat, n_resamples, seed, metrics=fn)["macro_f1"]
    return point, dist


def paired_macro_f1(rows: Sequence[WindowRow], preds: Mapping[str, np.ndarray], pairs: Mapping[str, tuple[str, str]],
                    *, n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:
    """For each pair ``(a, b)``: macro-F1(a) − macro-F1(b) and the 95% percentile interval of that
    difference over the shared session-bootstrap draws (overall, not per family)."""
    if not pairs:
        return {}
    columns = list(dict.fromkeys(c for ab in pairs.values() for c in ab))
    unknown = [c for c in columns if c not in preds]
    if unknown:
        raise ScoringError(f"paired comparison names unknown columns: {unknown}")
    point, dist = _macro_f1_draws(rows, preds, columns, n_resamples=n_resamples, seed=seed)
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


# ---------------------------------------------------------------------------
# Scoring and report
# ---------------------------------------------------------------------------


def diagnostics(arr: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Label-free readout diagnostics: label-word probability mass and the top-2 logit margin."""
    lg, lse = arr["label_logits"].astype(np.float64), arr["logsumexp"].astype(np.float64)
    m = lg.max(axis=1, keepdims=True)
    mass = np.exp((m[:, 0] + np.log(np.exp(lg - m).sum(axis=1))) - lse)
    top2 = np.sort(lg, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    q = (5, 25, 50, 75, 95)
    return {
        "label_word_mass": {f"p{p}": float(np.percentile(mass, p)) for p in q},
        "top2_margin": {f"p{p}": float(np.percentile(margin, p)) for p in q},
        "margin_below_1e-3": int((margin < 1e-3).sum()),
    }


def score(rows: Sequence[WindowRow], preds: Mapping[str, np.ndarray], groups: Mapping[str, Sequence[str]] | None = None,
          *, pairs: Mapping[str, tuple[str, str]] | None = None, disclosed: frozenset[str] | None = None,
          n_resamples: int = M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Both populations' ``m4.score_columns`` and paired differences. ``disclosed`` defaults to the
    v0.3 rule on ``rows`` (full val only); a subset report passes the set found on the full val rows."""
    if disclosed is None:
        disclosed = checked_disclosed(rows)
    excl = np.array([r.session_id not in disclosed for r in rows], dtype=bool)
    present = {r.session_id for r in rows} & disclosed
    out: dict[str, Any] = {
        "disclosed_sessions": {"count": len(disclosed),
                               "sha256_of_sorted_ids": hashlib.sha256("\n".join(sorted(disclosed)).encode()).hexdigest(),
                               "in_scored_sessions": len(present)},
        "populations": dict(v03_scoring.POPULATIONS),
    }
    for name, mask in (("excluding_disclosed", excl), ("all_val", np.ones(len(rows), dtype=bool))):
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        sub_preds = {c: np.asarray(p)[idx] for c, p in preds.items()}
        block = m4.score_columns(sub, sub_preds, fidelity_pairs={}, groups=groups, n_resamples=n_resamples)
        if pairs:
            block["paired_macro_f1"] = paired_macro_f1(sub, sub_preds, pairs, n_resamples=n_resamples)
        out[name] = block
    return out


def build_report(arm: str, columns: Mapping[str, Path], rows: Sequence[WindowRow], lab: Mapping[str, np.ndarray],
                 disclosed: frozenset[str], groups: Mapping[str, Sequence[str]] | None = None, *,
                 v03_columns: Path | None = None, subset_file: Path | None = None,
                 pairs: Mapping[str, tuple[str, str]] | None = None, majority_index: int | None = None,
                 v03_recorded: Mapping[str, Any] | None = None,
                 n_resamples: int = M.BOOTSTRAP_RESAMPLES, official: bool = True) -> dict[str, Any]:
    """The report body from already-checked full val ``rows``/``lab`` and the disclosed set (:func:`report`).
    ``pairs`` are added to :func:`default_pairs`, never instead of them."""
    metas = {name: column_meta(p) for name, p in columns.items()}
    sub = coverage(metas, subset_file)
    if sub is None:
        rows_s, lab_s, row_mask = list(rows), lab, None
    else:
        rows_s, lab_s, row_mask = restrict(rows, lab, sub["ids"])
    preds: dict[str, np.ndarray] = {}
    inputs, diag = {}, {}
    for name, path in columns.items():
        pred, meta, arr = load_column(path, lab_s, subset_sha256=None if sub is None else sub["sha256"])
        preds[name] = pred
        inputs[name] = {"file": Path(path).name, "sha256": _sha256_file(Path(path)), "meta": meta}
        diag[name] = diagnostics(arr)
    qwen = list(preds)
    if majority_index is None:
        majority_index = int(recorded_dev_prior()["majority_index"])
    extra = prior_columns(rows_s, majority_index)
    all_groups = {k: list(v) for k, v in (groups or {}).items()}
    v03_record, v03_cited = None, None
    if v03_columns is not None:
        recorded = recorded_val_blocks() if v03_recorded is None else v03_recorded
        v3, v3_groups, v03_record = load_v03_columns(v03_columns, rows, disclosed, majority_index, recorded=recorded,
                                                     n_resamples=n_resamples)
        extra.update({k: (v if row_mask is None else v[row_mask]) for k, v in v3.items()})
        for g, m in v3_groups.items():
            all_groups.setdefault(g, m)
        if sub is None and (v03_record["cited_not_paired"] or v03_record["groups_cited_not_paired"]):
            v03_cited = cited_v03(recorded, v03_record["cited_not_paired"], v03_record["groups_cited_not_paired"])
    clash = sorted(set(qwen) & set(extra))
    if clash:
        raise ScoringError(f"column names {clash} are reserved for the priors and the v0.3 columns")
    preds.update(extra)
    pairs = {**default_pairs(qwen, list(extra)), **dict(pairs or {})}
    body = score(rows_s, preds, all_groups or None, pairs=pairs, disclosed=disclosed, n_resamples=n_resamples)
    subset = None if sub is None else {k: sub[k] for k in ("file", "sha256", "sessions")}
    kind = "v04_val" if sub is None else ("v04_val_quick" if sub["val_quick"] else "v04_val_subset")
    return {
        "report": kind,
        "spec_version": "0.4 (plan draft)",
        "arm": arm,
        "split": SPLIT,
        "official": official,
        "visibility": VISIBILITY,
        "coverage": "full val" if sub is None else f"subset {sub['file']} ({sub['sessions']} sessions)",
        "subset": subset,
        **({"quick_look": QUICK_LOOK_NOTE} if sub is not None else {}),
        "inputs": {
            "columns": inputs,
            "qwen_columns": qwen,
            "prior_columns": {"names": list(PRIOR_COLUMNS), "majority_index": int(majority_index),
                              "source": "artifacts/reports/v03_sealed.json dev_prior"},
            "v03_columns": v03_record,
            "val_rows_sha256": VAL_ROWS_SHA256,
        },
        "paired_comparisons": {k: list(v) for k, v in pairs.items()},
        "definitions": {**M.DEFINITIONS, "h2_fidelity": H2_NOT_APPLICABLE,
                        "class": "first-index argmax over the four label-word logits",
                        "paired_macro_f1": ("macro-F1(minuend) - macro-F1(subtrahend) over the shared session-bootstrap "
                                            "draws of score_columns; 95% percentile interval; overall only")},
        "readout_diagnostics": diag,
        **({"v03_cited": {"source": "artifacts/reports/v03_sealed.json splits.val (not reproduced on this Mac; "
                                    "cited beside, no paired CI; PREREGISTRATION §1)", **v03_cited}}
           if v03_cited is not None else {}),
        **body,
    }


def fresh_test_committed() -> dict[str, Any]:
    """Refuse unless the fresh-test manifest and registration are committed unchanged at HEAD and the
    manifest cites that registration. Reads only those two committed JSON files: the fresh-test data
    files are not opened, listed or hashed (PREREGISTRATION §3.4)."""
    for path in (fresh_test.MANIFEST, fresh_test.REGISTRATION):
        if not path.is_file() or not fresh_test.committed_unchanged(path):
            raise ScoringError(f"{path.relative_to(fresh_test.REPO)} must be committed unchanged before any v0.4 scoring")
    man = json.loads(fresh_test.MANIFEST.read_text(encoding="utf-8"))
    reg_sha = _sha256_file(fresh_test.REGISTRATION)
    if man["registration"]["sha256"] != reg_sha:
        raise ScoringError("REGISTRATION.json differs from the one the fresh-test manifest cites")
    return {"manifest_sha256": _sha256_file(fresh_test.MANIFEST), "registration_sha256": reg_sha,
            "sessions": man["sessions"],
            "status": "sealed: data files not opened, listed or hashed, not scored in v0.4 (PREREGISTRATION §3.4)"}


def report(arm: str, columns: Mapping[str, Path], groups: Mapping[str, Sequence[str]] | None = None, *,
           v03_columns: Path | None = None, subset_file: Path | None = None,
           pairs: Mapping[str, tuple[str, str]] | None = None, registry: Mapping[str, Any] | None = None,
           n_resamples: int = M.BOOTSTRAP_RESAMPLES, official: bool = True) -> dict[str, Any]:
    """The report on the real val export. ``registry`` (the freeze record the script checked) goes into ``inputs``."""
    fresh = fresh_test_committed()
    rows, lab = load_rows()
    disclosed = checked_disclosed(rows)
    rep = build_report(arm, columns, rows, lab, disclosed, groups, v03_columns=v03_columns, subset_file=subset_file,
                       pairs=pairs, n_resamples=n_resamples, official=official)
    rep["inputs"].update({
        "export_manifest_sha256": _sha256_file(export.V04_DATA / "EXPORT_MANIFEST.json"),
        "fresh_test": fresh,
        "registry": None if registry is None else dict(registry),
        "scoring_code_sha256": {Path(f).name: _sha256_file(Path(f))
                                for f in (__file__, m4.__file__, M.__file__, _v03_val_file())},
    })
    return rep


def _v03_val_file() -> str:
    from . import v03_val

    return v03_val.__file__
