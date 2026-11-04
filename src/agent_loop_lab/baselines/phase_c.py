"""Phase C trainer: supervised MLP of the frozen topology (spec v0.2 §9.5; experiments/m4/README.md §1).

Pure training code over in-memory arrays. ``scripts/m4_train_phase_c.py``
loads the dev and val windows and writes the artefacts. Never imported by a
weight-build path.

Protocol (pre-registered, README §1):

* ``torch.manual_seed(seed)`` then ``DoomLoopMLP()``: PyTorch default init;
* cross-entropy (mean, unweighted), Adam(lr=1e-3, defaults), batch 256,
  epoch order ``torch.randperm(n, generator=torch.Generator().manual_seed(seed))``
  (one generator per run), last partial batch kept;
* after every epoch: val macro-F1 (``metrics.metrics_from_counts``); a strictly
  higher value is a new best; stop after 5 epochs without one or at 50; keep best.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ..model import DoomLoopMLP
from ..schema import INPUT_DIM, ClassLabel

__all__ = [
    "BUDGET_KEY",
    "BUDGETS",
    "SEEDS",
    "LR",
    "BATCH_SIZE",
    "MAX_EPOCHS",
    "PATIENCE",
    "budget_key",
    "budget_order",
    "budget_subset",
    "TrainResult",
    "macro_f1",
    "train_run",
    "windows_to_arrays",
]

BUDGET_KEY = "20261105:m4-budget:{session_id}"
BUDGETS: tuple[int | str, ...] = (32, 128, 512, 2048, "all")
SEEDS: tuple[int, ...] = (0, 1, 2)
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5
N_CLASSES = len(ClassLabel)


# ---------------------------------------------------------------------------
# Budget subsets
# ---------------------------------------------------------------------------


def budget_key(session_id: str) -> str:
    return hashlib.sha256(BUDGET_KEY.format(session_id=session_id).encode("utf-8")).hexdigest()


def budget_order(session_ids: Iterable[str]) -> list[str]:
    """Dev session ids sorted ascending by :func:`budget_key` (hex string order)."""
    ids = list(session_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session ids")
    return sorted(ids, key=budget_key)


def budget_subset(session_ids: Iterable[str], budget: int | str) -> list[str]:
    """The first ``budget`` sessions of :func:`budget_order` (``"all"`` = every session)."""
    order = budget_order(session_ids)
    if budget == "all":
        return order
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"budget must be a positive int or 'all', got {budget!r}")
    if budget > len(order):
        raise ValueError(f"budget {budget} > {len(order)} sessions")
    return order[:budget]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def windows_to_arrays(rows: Sequence[Any], sessions: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(x float32 (n, 120), y int64 (n,), session_ids)`` of gate-passing, scored rows.

    ``rows`` are :class:`~agent_loop_lab.eval.collisions.WindowRow`; when
    ``sessions`` is given, only rows of those sessions are kept. Row order is kept.
    """
    xs: list[bytes] = []
    ys: list[int] = []
    sids: list[str] = []
    for r in rows:
        if not (r.gate_passed and r.label.is_scored):
            continue
        if sessions is not None and r.session_id not in sessions:
            continue
        xs.append(r.vector)
        ys.append(int(r.label.to_class()))
        sids.append(r.session_id)
    x = np.frombuffer(b"".join(xs), dtype=np.float32).reshape(len(xs), INPUT_DIM).copy() if xs else np.zeros((0, INPUT_DIM), np.float32)
    return x, np.asarray(ys, dtype=np.int64), sids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 with the ``eval.metrics.metrics_from_counts`` conventions."""
    from ..eval.metrics import MODEL_COLUMNS, metrics_from_counts

    c = np.zeros(MODEL_COLUMNS, dtype=np.float64)
    np.add.at(c, np.asarray(y_true, np.int64) * N_CLASSES + np.asarray(y_pred, np.int64), 1.0)
    return float(metrics_from_counts(c)["macro_f1"])


def _predict(model: DoomLoopMLP, x: torch.Tensor, chunk: int = 65536) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            out.append(torch.argmax(model(x[i : i + chunk]), dim=-1))
    return torch.cat(out).numpy() if out else np.zeros(0, np.int64)


@dataclass
class TrainResult:
    model: DoomLoopMLP
    budget: int | str
    seed: int
    train_windows: int
    train_class_counts: dict[str, int]
    epochs_run: int
    best_epoch: int
    best_val_macro_f1: float
    stopped: str
    history: list[dict[str, float]] = field(default_factory=list)

    def log(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "seed": self.seed,
            "train_windows": self.train_windows,
            "train_class_counts": self.train_class_counts,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "stopped": self.stopped,
            "history": self.history,
        }


def train_run(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    budget: int | str = "?",
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> TrainResult:
    """One Phase C run (README §1). Returns the best-val-macro-F1 model."""
    if x_train.ndim != 2 or x_train.shape[1] != INPUT_DIM or x_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"bad training arrays {x_train.shape} / {y_train.shape}")
    if x_train.shape[0] == 0:
        raise ValueError("no training windows")
    torch.manual_seed(seed)
    model = DoomLoopMLP().to(dtype=torch.float32, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64))
    xv = torch.from_numpy(np.ascontiguousarray(x_val, dtype=np.float32))
    yv = np.asarray(y_val, dtype=np.int64)
    n = xt.shape[0]

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    since = 0
    history: list[dict[str, float]] = []
    stopped = "max_epochs"
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]
        model.eval()
        f1 = macro_f1(yv, _predict(model, xv))
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_epoch, since = f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= patience:
                stopped = f"patience ({patience} epochs without improvement)"
                break
    assert best_state is not None
    final = DoomLoopMLP()
    final.load_state_dict(best_state)
    for p in final.parameters():
        p.requires_grad_(False)
    final.eval()
    counts = np.bincount(y_train, minlength=N_CLASSES)
    return TrainResult(
        model=final,
        budget=budget,
        seed=seed,
        train_windows=int(n),
        train_class_counts={c.name: int(counts[int(c)]) for c in ClassLabel},
        epochs_run=epoch,
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        stopped=stopped,
        history=history,
    )
