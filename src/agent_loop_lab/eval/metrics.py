"""M3 validation scoring (spec v0.2 §10.1; experiments/m3/README.md §1).

Pure metric code over window rows (:class:`~agent_loop_lab.eval.collisions.WindowRow`)
and per-window class predictions. ``scripts/score_validation.py`` loads the
archived artefacts and writes the reports; this module never loads a model.

Split. :func:`load_rows` reads only the splits in :data:`ALLOWED_SPLITS`
(``{"val"}``). The check runs before any dataset module is imported or any file
is opened. There is no override. The sealed test and OOD sets are scored in M4
by a separate script.

Populations (each also written into every report under ``definitions``):

* *scored* windows pass the runtime gate **and** carry a scored oracle label
  (PRODUCTIVE / RECOVERY / STALLED / OSCILLATING). Macro-F1, per-class
  precision / recall / F1, accuracy, the false-loop alarm rate and the
  confusion matrix are computed on these only;
* *oracle-ambiguous* windows pass the gate and carry INSUFFICIENT_AMBIGUOUS.
  Only the class distribution of the predictions is reported (not scored);
* gate-passing INSUFFICIENT_INSTRUMENTATION windows get the same distribution
  (they indicate gate / oracle disagreement);
* gate-rejected windows have no input vector, so nothing is predicted. They
  are counted per oracle label under ``population``.

Metric conventions (match ``collisions._bayes_from_counts``):

* confusion matrix: rows are the oracle label, columns the predicted class,
  both in ``ClassLabel`` order;
* precision of a class = TP / predicted; 0 when the class has support but is
  never predicted; undefined (null) when it has neither;
* recall = TP / support, undefined when support is 0;
* F1 = 2PR / (P + R), 0 when P + R = 0, undefined when support is 0;
* macro-F1 = mean F1 over the canonical classes with support > 0;
* false-loop alarm rate = scored windows with oracle label PRODUCTIVE or
  RECOVERY predicted STALLED or OSCILLATING / scored PRODUCTIVE + RECOVERY
  windows;
* H2 fidelity = argmax agreement between the network and ``reference_policy``
  over all gate-passing windows (and, separately, the scored ones). Labels are
  not used.

Bootstrap. Sessions are resampled with replacement (``n`` of ``n``),
:data:`BOOTSTRAP_RESAMPLES` times, from ``numpy.random.default_rng(seed)``.
Every metric of a report block is recomputed from the summed per-session counts
of each resample, so all metrics of a block share the same draws. The 95%
interval is the 2.5th and 97.5th percentile (numpy's default linear method) of
the resamples on which the metric is defined; the number of resamples on which
it is undefined is reported when non-zero. Per-family intervals resample the
family's own sessions with the seed :func:`family_seed`.

Everything is computed from integer counts in a fixed order, so a report does
not depend on the order rows are supplied in beyond the session order, which
fixes the bootstrap draws. Outputs are plain JSON-serialisable dicts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from ..schema import CLASS_NAMES, ClassLabel, OracleLabel
from .collisions import WindowRow

__all__ = [
    "ALLOWED_SPLITS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DEFINITIONS",
    "check_split",
    "load_rows",
    "session_index",
    "session_counts",
    "metrics_from_counts",
    "bootstrap_indices",
    "bootstrap_distribution",
    "family_seed",
    "score_report",
]

ALLOWED_SPLITS: frozenset[str] = frozenset({"val"})
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261104
CI_PERCENTILES = (2.5, 97.5)
_CHUNK = 250

N_CLASSES = len(ClassLabel)
SAFE_CLASSES = (ClassLabel.PRODUCTIVE, ClassLabel.RECOVERY)
LOOP_CLASSES = (ClassLabel.STALLED, ClassLabel.OSCILLATING)

# Per-session count columns of one predictor.
_CM = slice(0, N_CLASSES * N_CLASSES)  # confusion cell (true * 4 + pred), scored windows
_AMB = slice(16, 20)  # predicted-class counts on gate-passing INSUFFICIENT_AMBIGUOUS windows
_INSTR = slice(20, 24)  # predicted-class counts on gate-passing INSUFFICIENT_INSTRUMENTATION windows
MODEL_COLUMNS = 24
# Fidelity columns: agree_all, total_all, agree_scored, total_scored.
FIDELITY_COLUMNS = 4

DEFINITIONS: dict[str, str] = {
    "split": "validation only; the scoring code refuses every other split before opening any file",
    "scored_population": "windows that pass the runtime gate and carry a scored oracle label (PRODUCTIVE, RECOVERY, STALLED, OSCILLATING)",
    "confusion_matrix": "rows = oracle label, columns = predicted class, both in ClassLabel order (PRODUCTIVE, RECOVERY, STALLED, OSCILLATING); counts of scored windows",
    "precision": "TP / windows predicted as the class; 0 if the class has support but is never predicted; null if it has neither",
    "recall": "TP / support; null if support is 0",
    "f1": "2PR / (P + R); 0 if P + R = 0; null if support is 0",
    "macro_f1": "mean F1 over the canonical classes with support > 0 in the scored population",
    "accuracy": "correct scored windows / scored windows",
    "false_alarm_rate": "scored windows with oracle label PRODUCTIVE or RECOVERY predicted STALLED or OSCILLATING / scored windows with oracle label PRODUCTIVE or RECOVERY",
    "h2_fidelity": "argmax agreement between the network and reference_policy; 'all' = every gate-passing validation window (any oracle label), 'scored' = the scored population; no labels are used. Part (b) of spec §10.1 (random valid-window sampler) is not part of this validation report",
    "ambiguous_predictions": "class distribution of predictions on gate-passing INSUFFICIENT_AMBIGUOUS windows (reported, not scored)",
    "instrumentation_predictions": "class distribution of predictions on gate-passing INSUFFICIENT_INSTRUMENTATION windows (gate/oracle disagreement; reported, not scored)",
    "gate_rejected": "windows the runtime gate rejects; they have no input vector and receive no prediction; excluded from every metric",
    "bootstrap": "sessions resampled with replacement (n of n); every metric recomputed from summed per-session counts; 95% CI = 2.5th/97.5th percentile (numpy linear) over resamples where the metric is defined",
    "per_family_bootstrap": "each family resamples its own sessions, seed = int.from_bytes(sha256(f'{seed}:family:{family}').digest()[:8], 'big')",
}


# ---------------------------------------------------------------------------
# Loading (val only)
# ---------------------------------------------------------------------------


def check_split(split: str) -> str:
    """Return ``split`` if it may be scored here; raise ``ValueError`` otherwise."""
    if not isinstance(split, str) or split not in ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} may not be scored by the M3 harness (allowed: {sorted(ALLOWED_SPLITS)})")
    return split


def load_rows(out_dir: str | Path, split: str = "val") -> list[WindowRow]:
    """Window rows of one allowed split (gate + encode via ``session_windows``).

    The split check runs first: nothing is imported from the dataset package and
    no file is opened for a refused split.
    """
    check_split(split)
    from ..datasets.build import iter_sessions  # imported only after the check
    from .collisions import session_windows

    rows: list[WindowRow] = []
    for events, record in iter_sessions(out_dir, split):
        rows.extend(session_windows(events, record, split))
    return rows


# ---------------------------------------------------------------------------
# Per-session counts
# ---------------------------------------------------------------------------


def session_index(rows: Sequence[WindowRow]) -> tuple[list[str], list[str], np.ndarray]:
    """Session ids (first-appearance order), their families, and each row's session index."""
    ids: dict[str, int] = {}
    families: list[str] = []
    idx = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        j = ids.get(r.session_id)
        if j is None:
            j = ids[r.session_id] = len(families)
            families.append(r.family)
        elif families[j] != r.family:
            raise ValueError(f"session {r.session_id!r} has rows of two families")
        idx[i] = j
    return list(ids), families, idx


def _check_predictions(rows: Sequence[WindowRow], pred: np.ndarray, name: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    if pred.shape != (len(rows),):
        raise ValueError(f"{name}: expected {len(rows)} predictions, got shape {pred.shape}")
    for r, p in zip(rows, pred.tolist()):
        if r.gate_passed and not 0 <= p < N_CLASSES:
            raise ValueError(f"{name}: gate-passing window {r.session_id}@{r.endpoint} has prediction {p}")
    return pred


def session_counts(
    rows: Sequence[WindowRow], net_pred: np.ndarray, ref_pred: np.ndarray, n_sessions: int, sidx: np.ndarray
) -> np.ndarray:
    """``(n_sessions, 2 * MODEL_COLUMNS + FIDELITY_COLUMNS)`` integer counts.

    Columns: network block, reference-policy block, fidelity block. Predictions
    of gate-rejected rows are ignored (use -1).
    """
    net_pred = _check_predictions(rows, net_pred, "network")
    ref_pred = _check_predictions(rows, ref_pred, "reference_policy")
    out = np.zeros((n_sessions, 2 * MODEL_COLUMNS + FIDELITY_COLUMNS), dtype=np.int64)
    fid = 2 * MODEL_COLUMNS
    for r, s, pn, pr in zip(rows, sidx.tolist(), net_pred.tolist(), ref_pred.tolist()):
        if not r.gate_passed:
            continue
        out[s, fid + 1] += 1
        out[s, fid] += pn == pr
        lab = r.label
        if lab.is_scored:
            t = int(lab.to_class())
            out[s, t * N_CLASSES + pn] += 1
            out[s, MODEL_COLUMNS + t * N_CLASSES + pr] += 1
            out[s, fid + 3] += 1
            out[s, fid + 2] += pn == pr
        elif lab is OracleLabel.INSUFFICIENT_AMBIGUOUS:
            out[s, _AMB.start + pn] += 1
            out[s, MODEL_COLUMNS + _AMB.start + pr] += 1
        elif lab is OracleLabel.INSUFFICIENT_INSTRUMENTATION:
            out[s, _INSTR.start + pn] += 1
            out[s, MODEL_COLUMNS + _INSTR.start + pr] += 1
    return out


# ---------------------------------------------------------------------------
# Metrics from counts (vectorised over any leading axes)
# ---------------------------------------------------------------------------


def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def metrics_from_counts(c: np.ndarray) -> dict[str, np.ndarray]:
    """§10.1 metrics of one predictor from summed counts of shape ``(..., MODEL_COLUMNS)``.

    Undefined values are NaN. Keys are flat names (``precision.STALLED``,
    ``confusion.STALLED.PRODUCTIVE`` = oracle STALLED predicted PRODUCTIVE, ...).
    """
    c = np.asarray(c, dtype=np.float64)
    lead = c.shape[:-1]
    cm = c[..., _CM].reshape(lead + (N_CLASSES, N_CLASSES))
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    precision = np.where(predicted > 0, _div(tp, predicted), np.where(support > 0, 0.0, np.nan))
    recall = _div(tp, support)
    psum = precision + recall
    f1 = np.where(support > 0, np.where(psum > 0, _div(2 * precision * recall, psum), 0.0), np.nan)
    has = support > 0
    n_has = has.sum(axis=-1)
    macro = _div(np.where(has, f1, 0.0).sum(axis=-1), n_has)
    total = support.sum(axis=-1)
    accuracy = _div(tp.sum(axis=-1), total)
    safe = [int(k) for k in SAFE_CLASSES]
    loop = [int(k) for k in LOOP_CLASSES]
    fa_num = cm[..., safe, :][..., :, loop].sum(axis=(-2, -1))
    fa_den = support[..., safe].sum(axis=-1)
    out: dict[str, np.ndarray] = {
        "scored_windows": total,
        "macro_f1": macro,
        "accuracy": accuracy,
        "false_alarm_rate": _div(fa_num, fa_den),
        "false_alarm_windows": fa_num,
        "false_alarm_denominator": fa_den,
    }
    for k, name in enumerate(CLASS_NAMES):
        out[f"support.{name}"] = support[..., k]
        out[f"predicted.{name}"] = predicted[..., k]
        out[f"precision.{name}"] = precision[..., k]
        out[f"recall.{name}"] = recall[..., k]
        out[f"f1.{name}"] = f1[..., k]
        for j, pname in enumerate(CLASS_NAMES):
            out[f"confusion.{name}.{pname}"] = cm[..., k, j]
    for tag, sl in (("ambiguous", _AMB), ("instrumentation", _INSTR)):
        d = c[..., sl]
        n = d.sum(axis=-1)
        out[f"{tag}.windows"] = n
        for k, name in enumerate(CLASS_NAMES):
            out[f"{tag}.count.{name}"] = d[..., k]
            out[f"{tag}.fraction.{name}"] = _div(d[..., k], n)
    return out


def fidelity_from_counts(c: np.ndarray) -> dict[str, np.ndarray]:
    c = np.asarray(c, dtype=np.float64)
    return {
        "all.agree": c[..., 0],
        "all.windows": c[..., 1],
        "all.agreement": _div(c[..., 0], c[..., 1]),
        "scored.agree": c[..., 2],
        "scored.windows": c[..., 3],
        "scored.agreement": _div(c[..., 2], c[..., 3]),
    }


def _all_metrics(c: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for prefix, block in (
        ("network", metrics_from_counts(c[..., :MODEL_COLUMNS])),
        ("reference_policy", metrics_from_counts(c[..., MODEL_COLUMNS : 2 * MODEL_COLUMNS])),
        ("h2_fidelity", fidelity_from_counts(c[..., 2 * MODEL_COLUMNS :])),
    ):
        for k, v in block.items():
            out[f"{prefix}:{k}"] = v
    return out


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------


def bootstrap_indices(n_sessions: int, n_resamples: int, seed: int, chunk: int = _CHUNK) -> Iterator[np.ndarray]:
    """Resampled session indices, ``(k, n_sessions)`` per chunk, ``n_resamples`` rows in total.

    Row ``i`` is resample ``i``; the draws are ``default_rng(seed).integers(0, n, size=(k, n))``
    chunk after chunk, so they depend only on ``(n_sessions, n_resamples, seed, chunk)``.
    """
    if n_sessions <= 0:
        raise ValueError("no sessions to resample")
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_resamples:
        k = min(chunk, n_resamples - done)
        yield rng.integers(0, n_sessions, size=(k, n_sessions))
        done += k


def bootstrap_distribution(
    counts: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    metrics: Callable[[np.ndarray], Mapping[str, np.ndarray]] = _all_metrics,
) -> dict[str, np.ndarray]:
    """Metric values on each resample: name -> array of shape ``(n_resamples,)``."""
    counts = np.asarray(counts, dtype=np.float64)  # integer counts < 2**53: sums are exact
    n = counts.shape[0]
    parts: dict[str, list[np.ndarray]] = {}
    for idx in bootstrap_indices(n, n_resamples, seed):
        w = np.stack([np.bincount(row, minlength=n) for row in idx]).astype(np.float64)
        for k, v in metrics(w @ counts).items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def family_seed(seed: int, family: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:family:{family}".encode()).digest()[:8], "big")


def _num(v: float) -> float | int | None:
    if not np.isfinite(v):
        return None
    f = float(v)
    return int(f) if f.is_integer() and abs(f) < 2**53 else f


def _ci(values: np.ndarray) -> tuple[list[float] | None, int]:
    ok = values[np.isfinite(values)]
    undefined = int(values.size - ok.size)
    if not ok.size:
        return None, undefined
    lo, hi = np.percentile(ok, CI_PERCENTILES)
    return [float(lo), float(hi)], undefined


def _stat(point: np.ndarray, dist: np.ndarray) -> dict[str, Any]:
    ci, undefined = _ci(dist)
    v = float(point)
    out: dict[str, Any] = {"value": None if not np.isfinite(v) else v, "ci95": ci}
    if undefined:
        out["undefined_resamples"] = undefined
    return out


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _model_block(point: Mapping[str, np.ndarray], dist: Mapping[str, np.ndarray], prefix: str) -> dict[str, Any]:
    def s(name: str) -> dict[str, Any]:
        return _stat(point[f"{prefix}:{name}"], dist[f"{prefix}:{name}"])

    def n(name: str) -> float | int | None:
        return _num(point[f"{prefix}:{name}"])

    block: dict[str, Any] = {
        "scored_windows": n("scored_windows"),
        "macro_f1": s("macro_f1"),
        "accuracy": s("accuracy"),
        "false_alarm_rate": {
            **s("false_alarm_rate"),
            "windows": n("false_alarm_windows"),
            "denominator": n("false_alarm_denominator"),
        },
        "per_class": {
            name: {
                "support": n(f"support.{name}"),
                "predicted": n(f"predicted.{name}"),
                "correct": n(f"confusion.{name}.{name}"),
                "precision": s(f"precision.{name}"),
                "recall": s(f"recall.{name}"),
                "f1": s(f"f1.{name}"),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": {
            "orientation": "rows = oracle label, columns = predicted class",
            "classes": list(CLASS_NAMES),
            "counts": [[n(f"confusion.{t}.{p}") for p in CLASS_NAMES] for t in CLASS_NAMES],
            "ci95": [[_ci(dist[f"{prefix}:confusion.{t}.{p}"])[0] for p in CLASS_NAMES] for t in CLASS_NAMES],
        },
    }
    for tag in ("ambiguous", "instrumentation"):
        block[f"{tag}_predictions"] = {
            "windows": n(f"{tag}.windows"),
            "counts": {name: n(f"{tag}.count.{name}") for name in CLASS_NAMES},
            "fraction": {name: s(f"{tag}.fraction.{name}") for name in CLASS_NAMES},
        }
    return block


def _fidelity_block(point: Mapping[str, np.ndarray], dist: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for part in ("all", "scored"):
        out[part] = {
            "agreement": _stat(point[f"h2_fidelity:{part}.agreement"], dist[f"h2_fidelity:{part}.agreement"]),
            "agree": _num(point[f"h2_fidelity:{part}.agree"]),
            "windows": _num(point[f"h2_fidelity:{part}.windows"]),
        }
    return out


def _blocks(counts: np.ndarray, n_resamples: int, seed: int) -> dict[str, Any]:
    point = _all_metrics(counts.sum(axis=0).astype(np.float64))
    dist = bootstrap_distribution(counts, n_resamples, seed)
    return {
        "sessions": int(counts.shape[0]),
        "network": _model_block(point, dist, "network"),
        "reference_policy": _model_block(point, dist, "reference_policy"),
        "h2_fidelity": _fidelity_block(point, dist),
    }


def _population(rows: Sequence[WindowRow], n_sessions: int) -> dict[str, Any]:
    passed = {lab.value: 0 for lab in OracleLabel}
    rejected = {lab.value: 0 for lab in OracleLabel}
    for r in rows:
        (passed if r.gate_passed else rejected)[r.label.value] += 1
    scored = sum(passed[c] for c in CLASS_NAMES)
    return {
        "sessions": n_sessions,
        "windows_total": len(rows),
        "gate_passed": sum(passed.values()),
        "gate_rejected": sum(rejected.values()),
        "gate_passed_by_oracle_label": passed,
        "gate_rejected_by_oracle_label": rejected,
        "scored_windows": scored,
        "scored_label_gate_rejected": sum(rejected[c] for c in CLASS_NAMES),
    }


def score_report(
    rows: Sequence[WindowRow],
    net_pred: np.ndarray,
    ref_pred: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Overall and per-family §10.1 metrics, H2 fidelity and ambiguous-window distributions.

    ``net_pred`` / ``ref_pred`` hold one class index per row (-1 or anything
    for gate-rejected rows, which are ignored).
    """
    rows = list(rows)
    for r in rows:
        if r.split is not None:
            check_split(r.split)
    ids, families, sidx = session_index(rows)
    counts = session_counts(rows, net_pred, ref_pred, len(ids), sidx)
    fam_arr = np.array(families, dtype=object)
    per_family = {}
    for fam in sorted(set(families)):
        mask = fam_arr == fam
        per_family[fam] = _blocks(counts[mask], n_resamples, family_seed(seed, fam))
    overall = _blocks(counts, n_resamples, seed)
    return {
        "definitions": dict(DEFINITIONS),
        "bootstrap": {
            "unit": "session",
            "resamples": n_resamples,
            "seed": seed,
            "rng": "numpy.random.default_rng(seed).integers(0, n_sessions, size=(chunk, n_sessions))",
            "chunk": _CHUNK,
            "ci": f"percentiles {list(CI_PERCENTILES)} (numpy linear) over resamples where the metric is defined",
        },
        "population": _population(rows, len(ids)),
        "overall": overall,
        "per_family": per_family,
    }
