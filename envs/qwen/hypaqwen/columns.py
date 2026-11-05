"""Shared plumbing of the probe and LoRA recipes: tasks, training windows, macro-F1, column files.

* **Tasks.** dev and val come from the export through :mod:`hypaqwen.data`. A sealed split is
  exported by the sealed evaluation itself and passed as an explicit ``--tasks`` file of the same
  format (:func:`iter_task_file`); nothing here opens a sealed path on its own.
* **Training windows** (§4): the gate-passing windows with a scored oracle label. The export codes
  ``row_label`` as the ``OracleLabel`` index, whose first four entries are the classes in
  ``CLASS_NAMES`` order; ``INSUFFICIENT_*`` labels have indices >= 4. :func:`session_rows` groups
  the label rows by session and checks them against the task file's ``window_pos``.
* **Macro-F1** with the ``eval.metrics.metrics_from_counts`` conventions (``agent_loop_lab`` is not
  importable here): per class, precision is 0 when nothing is predicted but the class has support,
  F1 is 0 when precision + recall is 0, and the macro mean runs over the classes with support.
* **Column files** in ``hypaqwen.run_readout``'s format, the one ``agent_loop_lab.v04.scoring``
  reads: ``label_logits [rows, 4]``, ``logsumexp [rows]``, ``session_ids [sessions]``,
  ``row_pos [rows]`` (export row order) and a JSON ``meta`` with ``split`` and ``limit``.
* **Cross-entropy** (:func:`cross_entropy_rows`) as ``-Σ onehot · log_softmax``, elementwise:
  ``nll_loss`` has no deterministic CUDA kernel and refuses under §3.1's
  ``use_deterministic_algorithms(True)``. The value is the same.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import MODEL_ID, MODEL_REVISION, V04_DATA
from . import data as qdata
from .data import Task
from .interface import CLASS_NAMES

__all__ = ["N_CLASSES", "SessionRows", "iter_split_tasks", "iter_task_file", "session_rows", "confusion",
           "macro_f1", "cross_entropy_rows", "write_column", "model_meta", "sha256_file", "sha256_bytes"]

N_CLASSES = len(CLASS_NAMES)


def cross_entropy_rows(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row cross-entropy ``[n]`` of ``logits [n, k]`` at class indices ``target [n]``."""
    k = logits.shape[-1]
    onehot = (target.reshape(-1, 1) == torch.arange(k, device=logits.device)).to(logits.dtype)
    return -(onehot * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_task_file(path: Path) -> Iterator[Task]:
    """A task export file (``{split}.tasks.jsonl.gz`` format), named explicitly."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield Task(d["session_id"], d["blocks"], d["window_pos"])


def iter_split_tasks(split: str, *, tasks: Path | None = None, data_dir: Path = V04_DATA) -> Iterator[Task]:
    """dev / val from the export; any other split only from an explicit ``tasks`` file."""
    if tasks is not None:
        return iter_task_file(tasks)
    return qdata.iter_tasks(split, data_dir)


@dataclass(frozen=True)
class SessionRows:
    row_pos: np.ndarray  # int, every window of the session, in export order
    label: np.ndarray  # int, OracleLabel index
    gate: np.ndarray  # bool

    @property
    def train(self) -> np.ndarray:
        """Mask of the training windows: gate-passing with a scored label (§4)."""
        return self.gate & (self.label < N_CLASSES)


def session_rows(labels: Mapping[str, np.ndarray]) -> dict[str, SessionRows]:
    """Label rows grouped by session id (rows are grouped by session in the export)."""
    sids = [str(s) for s in labels["session_ids"]]
    r_sess = labels["row_session"].astype(np.int64)
    if np.any(np.diff(r_sess) < 0):
        raise ValueError("label rows are not grouped by session")
    bounds = np.searchsorted(r_sess, np.arange(len(sids) + 1))
    pos, lab, gate = labels["row_pos"], labels["row_label"].astype(np.int64), labels["row_gate"].astype(bool)
    return {sid: SessionRows(pos[a:b].astype(np.int64), lab[a:b], gate[a:b])
            for sid, a, b in zip(sids, bounds[:-1].tolist(), bounds[1:].tolist())}


def confusion(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> np.ndarray:
    """``cm[true, pred]`` counts over the four classes."""
    t, p = np.asarray(y_true, dtype=np.int64).reshape(-1), np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred differ in length")
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    np.add.at(cm, (t, p), 1.0)
    return cm


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 with the ``metrics_from_counts`` conventions (NaN when no class has support)."""
    cm = confusion(y_true, y_pred)
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diagonal(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.where(predicted > 0, predicted, 1), 0.0)
        recall = tp / np.where(support > 0, support, 1)
        psum = precision + recall
        f1 = np.where(psum > 0, 2 * precision * recall / np.where(psum > 0, psum, 1), 0.0)
    has = support > 0
    return float(f1[has].sum() / has.sum()) if has.any() else float("nan")


def model_meta() -> dict[str, Any]:
    """The model and interface fields of ``run_readout``'s meta."""
    from . import interface as iface_mod
    from . import model as qmodel

    return {
        "attn_implementation": qmodel.ATTN_IMPLEMENTATION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": qmodel.FILE_SHA256},
        "interface_sha256": sha256_file(Path(iface_mod.__file__)),
        "interface": {"name": iface_mod.FROZEN.name, "spec_sha256": iface_mod.spec_sha256()},
        "platform": platform.platform(), "python": platform.python_version(),
    }


def write_column(path: Path, *, label_logits: np.ndarray, logsumexp: np.ndarray, session_ids: Sequence[str],
                 row_pos: np.ndarray, meta: Mapping[str, Any]) -> None:
    """One column file, refused unless aligned and finite (what the scoring bridge checks again)."""
    if "split" not in meta or "limit" not in meta:
        raise ValueError("meta needs 'split' and 'limit'")
    ll = np.asarray(label_logits)
    if ll.ndim != 2 or ll.shape[1] != N_CLASSES or ll.shape[0] != len(row_pos) or len(logsumexp) != len(row_pos):
        raise ValueError(f"label_logits {ll.shape}, logsumexp {len(logsumexp)}, row_pos {len(row_pos)} do not align")
    if not np.isfinite(ll).all() or not np.isfinite(logsumexp).all():
        raise ValueError("non-finite logits")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, label_logits=ll, logsumexp=np.asarray(logsumexp), session_ids=np.array(list(session_ids)),
             row_pos=np.asarray(row_pos, dtype=np.int32), meta=json.dumps(dict(meta)))
    tmp.replace(path)
