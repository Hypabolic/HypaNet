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
* scores with ``eval.m4.score_columns`` (as v0.3 scored Phase C), with no fidelity pairs. H2 has
  no meaning without a reference policy, and Q5 replaces it;
* reports both v0.3 populations: ``excluding_disclosed`` (primary) and ``all_val``.

It refuses unless the v0.4 fresh test manifest is committed (``fresh_test.require_sealed``),
because scoring is the first time a Qwen output meets a label.
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

__all__ = ["SPLIT", "VAL_ROWS_SHA256", "load_rows", "load_column", "diagnostics", "score", "report"]

SPLIT = "val"
# artifacts/reports/v03_sealed.json checks.identity_v02.val.rows_sha256 (87,176 windows, 4,004 sessions).
VAL_ROWS_SHA256 = "cdf6cce88e28499a8c513d0dd23126812db60590f7f09570a28a138a9e3196bc"
_LABELS = tuple(OracleLabel)
H2_NOT_APPLICABLE = {"applicable": False, "reason": "no reference policy for a Qwen column; plan v0.4 Q5 replaces H2"}


class ScoringError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(data_dir: Path = export.V04_DATA) -> tuple[list[WindowRow], dict[str, np.ndarray]]:
    """The val window rows from the export, checked against the v0.3 val rows hash."""
    M.check_split(SPLIT)
    with np.load(export.labels_path(SPLIT, data_dir)) as z:
        lab = {k: z[k] for k in z.files}
    ids, fams = [str(s) for s in lab["session_ids"]], [str(f) for f in lab["families"]]
    rows = [
        WindowRow(session_id=ids[s], family=fams[s], split=SPLIT, endpoint=int(e), label=_LABELS[int(l)],
                  gate_passed=bool(g), vector=None)
        for s, e, l, g in zip(lab["row_session"].tolist(), lab["row_endpoint"].tolist(), lab["row_label"].tolist(),
                              lab["row_gate"].tolist())
    ]
    got = v03_sealed.rows_sha256(rows)
    if got != VAL_ROWS_SHA256:
        raise ScoringError(f"val rows sha256 {got} != v0.3 sealed record {VAL_ROWS_SHA256}")
    return rows, lab


def load_column(path: Path, lab: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """``(pred, meta, arrays)`` for one readout file, aligned to ``lab`` or refused."""
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(str(arr.pop("meta")))
    if meta.get("split") != SPLIT or meta.get("limit") is not None:
        raise ScoringError(f"{path.name}: not a full {SPLIT} run (split={meta.get('split')}, limit={meta.get('limit')})")
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
          *, n_resamples: int = M.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    disclosed = v03_scoring.disclosed_sessions(rows)
    if disclosed != v03_sealed.private_disclosed():
        raise ScoringError("disclosed sessions differ from experiments/v0_3/DISCLOSURE_PRIVATE.json")
    excl = np.array([r.session_id not in disclosed for r in rows], dtype=bool)
    out: dict[str, Any] = {
        "disclosed_sessions": {"count": len(disclosed),
                               "sha256_of_sorted_ids": hashlib.sha256("\n".join(sorted(disclosed)).encode()).hexdigest()},
        "populations": dict(v03_scoring.POPULATIONS),
    }
    for name, mask in (("excluding_disclosed", excl), ("all_val", np.ones(len(rows), dtype=bool))):
        idx = np.flatnonzero(mask)
        sub = [rows[i] for i in idx.tolist()]
        out[name] = m4.score_columns(sub, {c: np.asarray(p)[idx] for c, p in preds.items()}, fidelity_pairs={},
                                     groups=groups, n_resamples=n_resamples)
    return out


def report(arm: str, columns: Mapping[str, Path], groups: Mapping[str, Sequence[str]] | None = None, *,
           n_resamples: int = M.BOOTSTRAP_RESAMPLES, official: bool = True) -> dict[str, Any]:
    fresh = fresh_test.require_sealed()
    rows, lab = load_rows()
    preds, inputs, diag = {}, {}, {}
    for name, path in columns.items():
        pred, meta, arr = load_column(path, lab)
        preds[name] = pred
        inputs[name] = {"file": path.name, "sha256": _sha256_file(path), "meta": meta}
        diag[name] = diagnostics(arr)
    body = score(rows, preds, groups, n_resamples=n_resamples)
    return {
        "report": "v04_val",
        "spec_version": "0.4 (plan draft)",
        "arm": arm,
        "split": SPLIT,
        "official": official,
        "inputs": {
            "columns": inputs,
            "val_rows_sha256": VAL_ROWS_SHA256,
            "export_manifest_sha256": _sha256_file(export.V04_DATA / "EXPORT_MANIFEST.json"),
            "fresh_test_manifest_sha256": _sha256_file(fresh_test.MANIFEST),
            "fresh_test_sessions": fresh["sessions"],
            "scoring_code_sha256": {Path(f).name: _sha256_file(Path(f)) for f in (__file__, m4.__file__, M.__file__)},
        },
        "definitions": {**M.DEFINITIONS, "h2_fidelity": H2_NOT_APPLICABLE,
                        "class": "first-index argmax over the four label-word logits"},
        "readout_diagnostics": diag,
        **body,
    }
