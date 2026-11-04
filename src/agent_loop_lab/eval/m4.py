"""M4 multi-column scoring (spec v0.2 §10.1; experiments/m4/README.md §3).

Extends the M3 metric code (:mod:`.metrics`) to many predictors on one split
with **shared bootstrap draws**: every column of a split is scored from the
same resampled sessions (the draws depend only on the number of sessions and
the seed). Populations and metric definitions are those of :mod:`.metrics`.

This module never opens a file except through :func:`load_split`, which the
M4 scripts call with an explicit split name.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import CLASS_NAMES, ClassLabel, OracleLabel
from . import metrics as M
from .collisions import WindowRow, session_windows

__all__ = [
    "LOADABLE_SPLITS",
    "load_split",
    "single_counts",
    "fidelity_counts",
    "score_columns",
    "gate_report",
    "counterfactual_meta",
    "counterfactual_consistency",
]

LOADABLE_SPLITS = frozenset({"dev", "val", "test", "counterfactual", "ood"})
N = len(ClassLabel)
SAFE = {int(c) for c in M.SAFE_CLASSES}
LOOP = {int(c) for c in M.LOOP_CLASSES}
SUMMARY_METRICS = ("macro_f1", "accuracy", "false_alarm_rate") + tuple(f"recall.{c}" for c in CLASS_NAMES)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_split(dataset_dir: str | Path, split: str) -> tuple[list[WindowRow], dict[str, Any]]:
    """Window rows of one split and, per session, the oracle record's counterfactual metadata."""
    if split not in LOADABLE_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "ood":
        from ..datasets.ood import iter_ood_sessions

        sessions = iter_ood_sessions(dataset_dir)
    else:
        from ..datasets.build import iter_sessions

        sessions = iter_sessions(dataset_dir, split)
    rows: list[WindowRow] = []
    meta: dict[str, Any] = {}
    for events, record in sessions:
        rows.extend(session_windows(events, record, split))
        cf = record.latent_trace.get("counterfactual") if isinstance(record.latent_trace, Mapping) else None
        if cf is not None:
            meta[record.session_id] = {**cf, "labels": {int(k): OracleLabel(v) for k, v in record.window_labels.items()}}
    return rows, meta


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def single_counts(rows: Sequence[WindowRow], pred: np.ndarray, sidx: np.ndarray, n_sessions: int) -> np.ndarray:
    """``(n_sessions, MODEL_COLUMNS)`` counts of one predictor (metrics column layout)."""
    pred = M._check_predictions(rows, pred, "column")
    out = np.zeros((n_sessions, M.MODEL_COLUMNS), dtype=np.int64)
    for r, s, p in zip(rows, sidx.tolist(), pred.tolist()):
        if not r.gate_passed:
            continue
        lab = r.label
        if lab.is_scored:
            out[s, int(lab.to_class()) * N + p] += 1
        elif lab is OracleLabel.INSUFFICIENT_AMBIGUOUS:
            out[s, M._AMB.start + p] += 1
        elif lab is OracleLabel.INSUFFICIENT_INSTRUMENTATION:
            out[s, M._INSTR.start + p] += 1
    return out


def fidelity_counts(rows: Sequence[WindowRow], a: np.ndarray, b: np.ndarray, sidx: np.ndarray, n_sessions: int) -> np.ndarray:
    """``(n_sessions, 4)``: agree_all, total_all, agree_scored, total_scored."""
    out = np.zeros((n_sessions, M.FIDELITY_COLUMNS), dtype=np.int64)
    for r, s, pa, pb in zip(rows, sidx.tolist(), a.tolist(), b.tolist()):
        if not r.gate_passed:
            continue
        out[s, 1] += 1
        out[s, 0] += pa == pb
        if r.label.is_scored:
            out[s, 3] += 1
            out[s, 2] += pa == pb
    return out


# ---------------------------------------------------------------------------
# Scoring many columns with shared draws
# ---------------------------------------------------------------------------


def _metrics_fn(n_cols: int, n_fid: int):
    k_cols = n_cols * M.MODEL_COLUMNS

    def fn(c: np.ndarray) -> dict[str, np.ndarray]:
        lead = c.shape[:-1]
        out: dict[str, np.ndarray] = {}
        mc = M.metrics_from_counts(c[..., :k_cols].reshape(lead + (n_cols, M.MODEL_COLUMNS)))
        for k, v in mc.items():
            out[f"m:{k}"] = v
        if n_fid:
            fc = M.fidelity_from_counts(c[..., k_cols:].reshape(lead + (n_fid, M.FIDELITY_COLUMNS)))
            for k, v in fc.items():
                out[f"f:{k}"] = v
        return out

    return fn


def _col_view(d: Mapping[str, np.ndarray], tag: str, j: int, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}:{k[len(tag) + 1:]}": v[..., j] for k, v in d.items() if k.startswith(tag + ":")}


def _finite(v: float) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


def _score_block(
    counts: np.ndarray,
    fid: np.ndarray | None,
    columns: Sequence[str],
    fid_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    n_s, n_cols = counts.shape[0], counts.shape[1]
    n_fid = 0 if fid is None else fid.shape[1]
    flat = counts.reshape(n_s, -1)
    if n_fid:
        flat = np.concatenate([flat, fid.reshape(n_s, -1)], axis=1)
    fn = _metrics_fn(n_cols, n_fid)
    point = fn(flat.sum(axis=0).astype(np.float64))
    dist = M.bootstrap_distribution(flat, n_resamples, seed, metrics=fn)
    out_cols = {}
    for j, name in enumerate(columns):
        out_cols[name] = M._model_block(_col_view(point, "m", j, "x"), _col_view(dist, "m", j, "x"), "x")
    out_fid = {}
    for j, name in enumerate(fid_names):
        out_fid[name] = M._fidelity_block(_col_view(point, "f", j, "h2_fidelity"), _col_view(dist, "f", j, "h2_fidelity"))
    out_groups = {}
    idx = {c: j for j, c in enumerate(columns)}
    for gname, members in groups.items():
        js = [idx[c] for c in members]
        g: dict[str, Any] = {"columns": list(members)}
        for m in SUMMARY_METRICS:
            vals = np.array([point[f"m:{m}"][j] for j in js], dtype=np.float64)
            seed_mean = np.mean(dist[f"m:{m}"][:, js], axis=1) if len(js) else np.array([])
            g[m] = {
                "per_seed": [None if not np.isfinite(v) else float(v) for v in vals],
                "mean": _finite(np.mean(vals)),
                "sd": _finite(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "seed_mean_ci95": M._ci(seed_mean)[0],
            }
        out_groups[gname] = g
    return {"sessions": int(n_s), "columns": out_cols, "h2_fidelity": out_fid, "groups": out_groups}


def score_columns(
    rows: Sequence[WindowRow],
    preds: Mapping[str, np.ndarray],
    fidelity_pairs: Mapping[str, tuple[str, str]] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    *,
    n_resamples: int = M.BOOTSTRAP_RESAMPLES,
    seed: int = M.BOOTSTRAP_SEED,
    per_family: bool = True,
) -> dict[str, Any]:
    """§10.1 metrics for every column of ``preds`` on ``rows``, overall and per family.

    ``fidelity_pairs`` maps a name to ``(column_a, column_b)`` whose argmax
    agreement is reported (H2). ``groups`` maps a name (Phase C budget) to
    the columns whose mean ± sd and seed-mean CI are reported.
    """
    rows = list(rows)
    fidelity_pairs = dict(fidelity_pairs or {})
    groups = dict(groups or {})
    columns = list(preds)
    ids, families, sidx = M.session_index(rows)
    n_s = len(ids)
    counts = np.stack([single_counts(rows, preds[c], sidx, n_s) for c in columns], axis=1)
    fid_names = list(fidelity_pairs)
    fid = (
        np.stack([fidelity_counts(rows, np.asarray(preds[a]), np.asarray(preds[b]), sidx, n_s)
                  for a, b in fidelity_pairs.values()], axis=1)
        if fid_names else None
    )
    overall = _score_block(counts, fid, columns, fid_names, groups, n_resamples, seed)
    fams: dict[str, Any] = {}
    if per_family:
        fam_arr = np.array(families, dtype=object)
        for fam in sorted(set(families)):
            mask = fam_arr == fam
            fams[fam] = _score_block(counts[mask], None if fid is None else fid[mask], columns, fid_names,
                                     groups, n_resamples, M.family_seed(seed, fam))
    return {
        "bootstrap": {
            "unit": "session",
            "resamples": n_resamples,
            "seed": seed,
            "chunk": M._CHUNK,
            "shared_draws": "every column of a split uses the same resampled sessions",
        },
        "population": M._population(rows, n_s),
        "overall": overall,
        "per_family": fams,
    }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def gate_report(rows: Sequence[WindowRow]) -> dict[str, Any]:
    """Gate coverage and agreement with the oracle's instrumentation flag."""
    t = Counter()
    for r in rows:
        t[(r.gate_passed, r.label is OracleLabel.INSUFFICIENT_INSTRUMENTATION)] += 1
    n = len(rows)
    passed = t[(True, False)] + t[(True, True)]
    agree = t[(False, True)] + t[(True, False)]
    return {
        "windows": n,
        "gate_passed": passed,
        "coverage": passed / n if n else None,
        "rejected_and_instrumentation": t[(False, True)],
        "rejected_not_instrumentation": t[(False, False)],
        "passed_and_instrumentation": t[(True, True)],
        "passed_not_instrumentation": t[(True, False)],
        "agreement": agree / n if n else None,
        "definition": "agreement = (rejected & label INSUFFICIENT_INSTRUMENTATION) + (passed & other label), over all windows",
    }


# ---------------------------------------------------------------------------
# Counterfactual consistency (README §3.5)
# ---------------------------------------------------------------------------


def counterfactual_meta(meta: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """pair_id -> {"kind", "a": session_id, "b": session_id, "contrast": [...], labels}."""
    pairs: dict[str, dict[str, Any]] = {}
    for sid, cf in meta.items():
        p = pairs.setdefault(cf["pair_id"], {"kind": cf["kind"]})
        if p["kind"] != cf["kind"]:
            raise ValueError(f"pair {cf['pair_id']}: kinds differ")
        if cf["branch"] in p:
            raise ValueError(f"pair {cf['pair_id']}: two sessions of branch {cf['branch']}")
        p[cf["branch"]] = sid
        p[f"labels_{cf['branch']}"] = cf["labels"]
        p.setdefault("contrast", list(cf["contrast_endpoints"]))
        if list(cf["contrast_endpoints"]) != p["contrast"]:
            raise ValueError(f"pair {cf['pair_id']}: contrast endpoints differ between branches")
    for pid, p in pairs.items():
        if "a" not in p or "b" not in p:
            raise ValueError(f"pair {pid} is missing a branch")
    return pairs


def _ci_over(values: np.ndarray, n_resamples: int, seed: int) -> list[float] | None:
    """Bootstrap CI of the mean of per-pair values (pairs resampled)."""
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        k = min(M._CHUNK, n_resamples - done)
        idx = rng.integers(0, values.size, size=(k, values.size))
        means[done:done + k] = values[idx].mean(axis=1)
        done += k
    return M._ci(means)[0]


def _ratio_ci(num: np.ndarray, den: np.ndarray, n_resamples: int, seed: int) -> list[float] | None:
    """Bootstrap CI of sum(num)/sum(den) over pairs."""
    if den.size == 0 or den.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        k = min(M._CHUNK, n_resamples - done)
        idx = rng.integers(0, den.size, size=(k, den.size))
        d = den[idx].sum(axis=1)
        vals[done:done + k] = np.where(d > 0, num[idx].sum(axis=1) / np.maximum(d, 1), np.nan)
        done += k
    return M._ci(vals)[0]


def counterfactual_consistency(
    rows: Sequence[WindowRow],
    meta: Mapping[str, Any],
    preds: Mapping[str, np.ndarray],
    *,
    n_resamples: int = M.BOOTSTRAP_RESAMPLES,
    seed: int = M.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """README §3.5 measures for every column."""
    from ..synth.counterfactual import INTENDED

    pairs = counterfactual_meta(meta)
    where = {(r.session_id, r.endpoint): i for i, r in enumerate(rows)}
    pids = sorted(pairs)
    no_contrast = [p for p in pids if not pairs[p]["contrast"]]
    # Per pair: list of (row index a, row index b) at gate-passing contrast endpoints.
    units: dict[str, list[tuple[int, int, int, int]]] = {}
    gate_excluded = 0
    for pid in pids:
        p = pairs[pid]
        benign, degenerate = INTENDED[p["kind"]]
        lst = []
        for e in p["contrast"]:
            ia, ib = where.get((p["a"], e)), where.get((p["b"], e))
            if ia is None or ib is None:
                raise ValueError(f"pair {pid}: contrast endpoint {e} has no window")
            la, lb = rows[ia].label, rows[ib].label
            if la not in benign or lb is not degenerate:
                raise ValueError(f"pair {pid}@{e}: labels {la}/{lb} are not a contrast")
            if not (rows[ia].gate_passed and rows[ib].gate_passed):
                gate_excluded += 1
                continue
            lst.append((ia, ib, int(la.to_class()), int(lb.to_class())))
        if lst:
            units[pid] = lst
    scored_pids = sorted(units)
    kinds = sorted({pairs[p]["kind"] for p in pids})

    def measure(pred: np.ndarray, subset: list[str]) -> dict[str, Any]:
        res: dict[str, Any] = {"pairs": len(subset), "endpoints": sum(len(units[p]) for p in subset)}
        for mname in ("directional", "strict"):
            num = np.zeros(len(subset))
            den = np.zeros(len(subset))
            all_ok = np.zeros(len(subset))
            any_ok = np.zeros(len(subset))
            for k, pid in enumerate(subset):
                oks = []
                for ia, ib, ta, tb in units[pid]:
                    pa, pb = int(pred[ia]), int(pred[ib])
                    if mname == "directional":
                        oks.append(pa in SAFE and pb in LOOP)
                    else:
                        oks.append(pa == ta and pb == tb)
                num[k], den[k] = sum(oks), len(oks)
                all_ok[k], any_ok[k] = float(all(oks)), float(any(oks))
            res[mname] = {
                "endpoint_rate": float(num.sum() / den.sum()) if den.sum() else None,
                "endpoint_rate_ci95": _ratio_ci(num, den, n_resamples, seed),
                "pairs_all": float(all_ok.mean()) if subset else None,
                "pairs_all_ci95": _ci_over(all_ok, n_resamples, seed),
                "pairs_any": float(any_ok.mean()) if subset else None,
                "pairs_any_ci95": _ci_over(any_ok, n_resamples, seed),
            }
        return res

    columns = {}
    for name, pred in preds.items():
        pred = np.asarray(pred)
        col = {"overall": measure(pred, scored_pids)}
        for kind in kinds:
            col[kind] = measure(pred, [p for p in scored_pids if pairs[p]["kind"] == kind])
        columns[name] = col
    return {
        "definitions": {
            "unit": "contrast endpoint: a window endpoint present in both branches where a's label is benign and b's is the degenerate label (synth.counterfactual.INTENDED)",
            "directional": "pred(a) in {PRODUCTIVE, RECOVERY} and pred(b) in {STALLED, OSCILLATING}",
            "strict": "pred(a) == oracle label of a and pred(b) == oracle label of b",
            "endpoint_rate": "consistent contrast endpoints / gate-passing contrast endpoints (pooled)",
            "pairs_all": "fraction of pairs consistent at every gate-passing contrast endpoint",
            "pairs_any": "fraction of pairs consistent at at least one gate-passing contrast endpoint",
            "bootstrap": f"pairs resampled with replacement, {n_resamples} resamples, seed {seed}",
        },
        "pairs_total": len(pids),
        "pairs_without_contrast_endpoints": len(no_contrast),
        "contrast_endpoints_gate_excluded": gate_excluded,
        "pairs_scored": len(scored_pids),
        "pairs_scored_per_kind": {k: sum(pairs[p]["kind"] == k for p in scored_pids) for k in kinds},
        "columns": columns,
    }
