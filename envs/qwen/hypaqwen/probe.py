"""Qwen-probe: a linear 4-class head on frozen Qwen states (PREREGISTRATION §4, §7.2).

Three steps, each a subcommand:

    python -m hypaqwen.probe features --split dev --out DEV.npz --device cuda
    python -m hypaqwen.probe fit --budget 512 --features DEV.npz --out runs/probe-512 --device cuda
    python -m hypaqwen.probe fit --budget 200w --features DEV.npz --val-features VAL.npz --out runs/probe-200w
    python -m hypaqwen.probe predict --model runs/probe-512 --features VAL.npz --out cols/probe-512.npz

**Features.** ``hidden[24]`` as the WSP kernel defines it: the residual stream after the last
decoder block, the input of the final norm (pre-norm, not ``last_hidden_state``), at the last
suffix token of each window, fp32. It is taken with a forward pre-hook on the final norm during
the standard readout pass (:class:`FeatureReader`, a :class:`~hypaqwen.readout.Verbaliser`, one
pass per session with the cache branched after each window, as ``wsp.server._Capture`` does), so
the probe reads exactly the states the verbaliser reads out. Every window of every session is kept
in export row order; the same pass also stores the Qwen-0 label logits. A sealed split comes only
as an explicit ``--tasks`` file written by the sealed evaluation.

**Training windows.** The budget sessions' gate-passing windows with a scored oracle label
(``columns.SessionRows.train``); ``INSUFFICIENT_*`` and gate-rejected windows contribute nothing.
Budgets come from ``BUDGETS.json`` (:mod:`hypaqwen.budgets`). ``200w`` uses the 200 disclosed val
windows (val features and val labels; the disclosure rule is re-checked on the labels, the only val
labels this module reads), CV inside the 200.

**Fit** (§4, exactly). Per-feature mean and population standard deviation (floor 1e-6) of the
training windows (inside CV, of the training folds). Multinomial logistic regression over the
classes present, ``C · Σ_i w_{y_i} · CE_i + ½‖W‖²`` with balanced weights ``n / (K · n_c)`` and an
unpenalised intercept, float64, full-batch L-BFGS with strong-Wolfe line search from zero, stopped
at gradient max-norm 1e-6 or 2,000 iterations (``torch.optim.LBFGS``: ``tolerance_change=0`` so no
other test stops it, history 100, evaluations unbounded in practice). ``C`` from {1e-4 .. 10} by
5-fold CV grouped by session: fold = rank of ``sha256("20261105:v04-probe-fold:{sid}")`` among the
budget's sessions, mod 5; score = macro-F1 (``metrics_from_counts`` conventions) of the pooled
out-of-fold predictions; ties to the smaller C. Then a refit on the whole subset with that C.

**Predict.** A column file in ``run_readout``'s format: ``label_logits`` are the probe's four logits
(float64; a class absent from the budget has no row and gets the finite logit -1e30, so it is never
the argmax), ``logsumexp`` is the logsumexp of those four logits, and ``meta["readout"]`` is
``"probe"``. The bridge's first-index argmax is the probe's class.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import V04_DATA
from . import backend
from . import budgets as B
from . import data as qdata
from .columns import (N_CLASSES, cross_entropy_rows, iter_split_tasks, macro_f1, model_meta, session_rows,
                      sha256_file, write_column)
from .readout import Verbaliser

__all__ = [
    "FEATURES_FORMAT", "MODEL_FORMAT", "FOLD_KEY", "N_FOLDS", "C_GRID", "STD_FLOOR", "GTOL", "MAX_ITER", "HISTORY",
    "ABSENT_LOGIT", "FeatureReader", "extract_features", "load_features", "TrainSet", "dev_train_set",
    "window_train_set", "fold_key", "folds", "standardise", "class_weights", "LogReg", "fit_logreg",
    "cross_validate", "Probe", "fit_probe", "probe_logits", "main",
]

FEATURES_FORMAT = "v04-probe-features/1"
MODEL_FORMAT = "v04-probe/1"
FOLD_KEY = "20261105:v04-probe-fold:{session_id}"
N_FOLDS = 5
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
STD_FLOOR = 1e-6
GTOL = 1e-6
MAX_ITER = 2000
HISTORY = 100
ABSENT_LOGIT = -1e30
_F64 = torch.float64


# ---------------------------------------------------------------------------
# Features (hidden[24], pre-norm, at the readout position)
# ---------------------------------------------------------------------------


class FeatureReader(Verbaliser):
    """The verbaliser readout plus the final norm's input at each window's readout position."""

    def __init__(self, model, iface):
        super().__init__(model, iface)
        self.n_layers = int(model.config.num_hidden_layers)
        self._pre: torch.Tensor | None = None
        self._resid: list[torch.Tensor] = []

    def _hook(self, _module, args) -> None:
        self._pre = args[0][0, -1].detach().float().clone()

    @contextlib.contextmanager
    def hooks(self) -> Iterator[None]:
        h = self.model.model.norm.register_forward_pre_hook(self._hook)
        try:
            yield
        finally:
            h.remove()

    def _read(self, out, logits4, lse, hid) -> None:
        super()._read(out, logits4, lse, hid)  # right after the suffix run: _pre is its last position
        self._resid.append(self._pre)

    def features(self, block_ids, window_pos):
        """``(WindowReadout, hidden[n_layers] [windows, d])`` for one session."""
        self._resid = []
        with self.hooks():
            r = self.session(block_ids, window_pos)
        d = self.model.config.hidden_size
        return r, (torch.stack(self._resid).cpu() if self._resid else torch.empty(0, d))


def extract_features(reader: FeatureReader, tok, tasks, *, progress=None) -> dict[str, Any]:
    """Arrays of a features file for ``tasks`` (every window, export row order)."""
    feats, logits, lse, sids, r_sess, r_pos = [], [], [], [], [], []
    n_tok, t0 = 0, time.monotonic()
    for n, task in enumerate(tasks):
        r, h = reader.features(reader.iface.block_ids(tok, task.blocks), task.window_pos)
        if r.pos != list(task.window_pos):
            raise ValueError(f"{task.session_id}: window_pos not sorted and unique")
        feats.append(h.numpy()), logits.append(r.label_logits.numpy()), lse.append(r.logsumexp.numpy())
        r_sess.extend([len(sids)] * len(r.pos)), r_pos.extend(r.pos)
        sids.append(task.session_id)
        n_tok += r.tokens
        if progress is not None and (n + 1) % 50 == 0:
            progress(f"{n + 1} sessions, {n_tok} tokens, {n_tok / (time.monotonic() - t0):.0f} tok/s")
    d = reader.model.config.hidden_size
    return {
        "features": np.concatenate(feats) if feats else np.zeros((0, d), np.float32),
        "label_logits": np.concatenate(logits) if logits else np.zeros((0, N_CLASSES), np.float32),
        "logsumexp": np.concatenate(lse) if lse else np.zeros(0, np.float32),
        "session_ids": np.array(sids), "row_session": np.array(r_sess, dtype=np.int32),
        "row_pos": np.array(r_pos, dtype=np.int32), "tokens": n_tok, "seconds": time.monotonic() - t0,
    }


@dataclass
class Features:
    features: np.ndarray
    session_ids: list[str]
    row_session: np.ndarray
    row_pos: np.ndarray
    meta: dict[str, Any]
    sha256: str
    _slices: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.any(np.diff(self.row_session) < 0):
            raise ValueError("feature rows are not grouped by session")
        b = np.searchsorted(self.row_session, np.arange(len(self.session_ids) + 1))
        self._slices = {s: (int(b[i]), int(b[i + 1])) for i, s in enumerate(self.session_ids)}

    def rows(self, session_id: str) -> slice:
        if session_id not in self._slices:
            raise KeyError(f"{session_id}: not in the features file ({self.meta.get('split')})")
        return slice(*self._slices[session_id])


def load_features(path: Path) -> Features:
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(str(arr["meta"]))
    if meta.get("format") != FEATURES_FORMAT:
        raise ValueError(f"{path}: not a {FEATURES_FORMAT} file")
    return Features(arr["features"], [str(s) for s in arr["session_ids"]], arr["row_session"].astype(np.int64),
                    arr["row_pos"].astype(np.int64), meta, sha256_file(path))


# ---------------------------------------------------------------------------
# Training sets
# ---------------------------------------------------------------------------


@dataclass
class TrainSet:
    X: np.ndarray  # [n, d] float32
    y: np.ndarray  # [n] class index
    groups: list[str]  # session id per row
    sessions: list[str]  # the budget's sessions (the fold rank runs over these)

    @property
    def class_counts(self) -> dict[str, int]:
        from .interface import CLASS_NAMES

        c = np.bincount(self.y, minlength=N_CLASSES)
        return {name: int(c[i]) for i, name in enumerate(CLASS_NAMES)}


def dev_train_set(feats: Features, labels: Mapping[str, np.ndarray], session_ids: Sequence[str]) -> TrainSet:
    """Training windows of ``session_ids`` (a dev budget, in its order)."""
    rows = session_rows(labels)
    X, y, groups = [], [], []
    for sid in session_ids:
        fr, lr = feats.rows(sid), rows[sid]
        if not np.array_equal(feats.row_pos[fr], lr.row_pos):
            raise ValueError(f"{sid}: feature rows differ from the label rows")
        m = lr.train
        X.append(feats.features[fr][m]), y.append(lr.label[m])
        groups.extend([sid] * int(m.sum()))
    return TrainSet(np.concatenate(X), np.concatenate(y).astype(np.int64), groups, list(session_ids))


def window_train_set(feats: Features, labels: Mapping[str, np.ndarray], windows: Sequence[B.Window]) -> TrainSet:
    """One training window per entry (the 200 disclosed val windows of ``200w``)."""
    rows = session_rows(labels)
    X, y, groups = [], [], []
    for w in windows:
        fr, lr = feats.rows(w.session_id), rows[w.session_id]
        if not np.array_equal(feats.row_pos[fr], lr.row_pos):
            raise ValueError(f"{w.session_id}: feature rows differ from the label rows")
        hit = np.flatnonzero(lr.row_pos == w.row_pos)
        if hit.size != 1 or not lr.train[hit[0]]:
            raise ValueError(f"{w.session_id}@{w.row_pos}: not one gate-passing scored window")
        X.append(feats.features[fr][hit[0]]), y.append(int(lr.label[hit[0]]))
        groups.append(w.session_id)
    return TrainSet(np.stack(X), np.asarray(y, np.int64), groups, [w.session_id for w in windows])


# ---------------------------------------------------------------------------
# Fit (§4)
# ---------------------------------------------------------------------------


def fold_key(session_id: str) -> str:
    return hashlib.sha256(FOLD_KEY.format(session_id=session_id).encode("utf-8")).hexdigest()


def folds(session_ids: Sequence[str], n_folds: int = N_FOLDS) -> dict[str, int]:
    """Fold of each session: its rank by :func:`fold_key` among ``session_ids``, mod ``n_folds``."""
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("duplicate session ids")
    return {s: r % n_folds for r, s in enumerate(sorted(session_ids, key=fold_key))}


def standardise(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean and population std (floor :data:`STD_FLOOR`), float64."""
    X = X.to(_F64)
    mean = X.mean(dim=0)
    std = X.std(dim=0, correction=0).clamp_min(STD_FLOOR)
    return mean, std


def class_weights(y_idx: np.ndarray, k: int) -> np.ndarray:
    """Balanced weights ``n / (k · n_c)`` over the ``k`` classes present (indices ``0..k-1``)."""
    n_c = np.bincount(y_idx, minlength=k).astype(np.float64)
    if np.any(n_c == 0):
        raise ValueError("a class index has no window")
    return len(y_idx) / (k * n_c)


@dataclass
class LogReg:
    classes: np.ndarray  # present class indices, ascending
    W: torch.Tensor  # [k, d] float64
    b: torch.Tensor  # [k]
    info: dict[str, Any]

    def logits4(self, Z: torch.Tensor) -> torch.Tensor:
        """``[n, 4]`` logits; an absent class gets :data:`ABSENT_LOGIT`."""
        out = torch.full((Z.shape[0], N_CLASSES), ABSENT_LOGIT, dtype=_F64, device=Z.device)
        out[:, torch.as_tensor(self.classes, device=Z.device)] = Z.to(_F64) @ self.W.T + self.b
        return out

    def predict(self, Z: torch.Tensor) -> np.ndarray:
        return np.argmax(self.logits4(Z).cpu().numpy(), axis=1)  # first-index tie rule


def _objective(Z, yi, sw, W, b, C):
    return C * (sw * cross_entropy_rows(Z @ W.T + b, yi)).sum() + 0.5 * (W * W).sum()


def fit_logreg(Z: torch.Tensor, y: np.ndarray, C: float, *, max_iter: int = MAX_ITER, gtol: float = GTOL) -> LogReg:
    """Weighted multinomial logistic regression on standardised ``Z`` (float64), L-BFGS from zero."""
    Z = Z.to(_F64)
    classes = np.unique(y)
    k, d = len(classes), Z.shape[1]
    W = torch.zeros(k, d, dtype=_F64, device=Z.device, requires_grad=True)
    b = torch.zeros(k, dtype=_F64, device=Z.device, requires_grad=True)
    if k < 2:  # one class: every window is that class, the optimum is W = 0
        return LogReg(classes, W.detach(), b.detach(), {"C": C, "classes": classes.tolist(), "iterations": 0,
                                                        "grad_max": 0.0, "objective": 0.0, "converged": True})
    y_idx = np.searchsorted(classes, y)
    yi = torch.as_tensor(y_idx, dtype=torch.long, device=Z.device)
    sw = torch.as_tensor(class_weights(y_idx, k), dtype=_F64, device=Z.device)[yi]
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, max_eval=25 * max_iter, tolerance_grad=gtol,
                            tolerance_change=0.0, history_size=HISTORY, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _objective(Z, yi, sw, W, b, C)
        loss.backward()
        return loss

    opt.step(closure)
    obj = closure()
    g = torch.cat([W.grad.reshape(-1), b.grad.reshape(-1)])
    gmax = float(g.abs().max())
    info = {"C": C, "classes": classes.tolist(), "iterations": int(opt.state[W].get("n_iter", 0)),
            "evaluations": int(opt.state[W].get("func_evals", 0)), "grad_max": gmax, "objective": float(obj.detach()),
            "converged": gmax <= gtol}
    return LogReg(classes, W.detach(), b.detach(), info)


def _fit_standardised(X: torch.Tensor, y: np.ndarray, C: float):
    mean, std = standardise(X)
    return mean, std, fit_logreg((X.to(_F64) - mean) / std, y, C)


def cross_validate(ts: TrainSet, *, grid: Sequence[float] = C_GRID, device: str = "cpu",
                   fit=None) -> tuple[float, list[dict[str, Any]]]:
    """``(chosen C, per-C table)``: pooled out-of-fold macro-F1, ties to the smaller C."""
    fit = fit or _fit_standardised
    fold_of = folds(ts.sessions)
    fidx = np.array([fold_of[g] for g in ts.groups], dtype=np.int64)
    X = torch.as_tensor(ts.X, device=device)
    table, best_c, best = [], None, None
    for C in sorted(grid):
        oof = np.full(len(ts.y), -1, dtype=np.int64)
        fits = []
        for f in range(N_FOLDS):
            va = fidx == f
            if not va.any():
                continue
            tr = ~va
            if not tr.any():
                raise ValueError("a fold holds every training window")
            mean, std, lr = fit(X[torch.as_tensor(tr, device=device)], ts.y[tr], C)
            Zv = (X[torch.as_tensor(va, device=device)].to(_F64) - mean) / std
            oof[va] = lr.predict(Zv)
            fits.append({"fold": f, "train_windows": int(tr.sum()), "val_windows": int(va.sum()), **lr.info})
        score = macro_f1(ts.y, oof)
        table.append({"C": C, "oof_macro_f1": score, "folds": fits})
        if best is None or score > best:  # strictly better: ties keep the smaller C
            best_c, best = C, score
    return best_c, table


@dataclass
class Probe:
    mean: torch.Tensor
    std: torch.Tensor
    model: LogReg

    def logits4(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        X = torch.as_tensor(X).to(device=self.mean.device, dtype=_F64)
        return self.model.logits4((X - self.mean) / self.std).cpu().numpy()

    def save(self, path: Path, extra: Mapping[str, Any]) -> None:
        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy(), classes=self.model.classes,
                 W=self.model.W.cpu().numpy(), b=self.model.b.cpu().numpy(),
                 meta=json.dumps({"format": MODEL_FORMAT, **extra}))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> tuple["Probe", dict[str, Any]]:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != MODEL_FORMAT:
                raise ValueError(f"{path}: not a {MODEL_FORMAT} file")
            t = {k: torch.as_tensor(z[k], device=device) for k in ("mean", "std", "W", "b")}
            classes = z["classes"].astype(np.int64)
        return cls(t["mean"], t["std"], LogReg(classes, t["W"], t["b"], {})), meta


def fit_probe(ts: TrainSet, *, device: str = "cpu", grid: Sequence[float] = C_GRID) -> tuple[Probe, dict[str, Any]]:
    """CV for C, then the refit on the whole training set."""
    c, table = cross_validate(ts, grid=grid, device=device)
    mean, std, lr = _fit_standardised(torch.as_tensor(ts.X, device=device), ts.y, c)
    log = {"chosen_C": c, "cv": table, "refit": lr.info, "train_windows": len(ts.y),
           "train_sessions_with_windows": len(set(ts.groups)), "budget_sessions": len(ts.sessions),
           "train_class_counts": ts.class_counts}
    return Probe(mean, std, lr), log


def probe_logits(probe: Probe, X: np.ndarray, chunk: int = 65536) -> np.ndarray:
    return np.concatenate([probe.logits4(X[i: i + chunk]) for i in range(0, len(X), chunk)]) if len(X) \
        else np.zeros((0, N_CLASSES))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_features(a) -> int:
    from . import model as qmodel
    from .interface import build_interface

    env = backend.configure(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    tasks = iter_split_tasks(a.split, tasks=a.tasks, data_dir=a.data_dir)
    subset = None
    if a.budget:
        if a.split != "dev" or a.budget not in B.SESSION_BUDGETS:
            raise SystemExit("--budget selects a dev session budget")
        subset = set(B.load(a.budgets).sessions(a.budget))
        tasks = (t for t in tasks if t.session_id in subset)
    if a.limit is not None:
        tasks = itertools.islice(tasks, a.limit)
    tok, model = qmodel.load(a.device, "fp32")
    reader = FeatureReader(model, build_interface(tok))
    with backend.compute(a.device), torch.no_grad():
        arr = extract_features(reader, tok, tasks, progress=_log)
    meta = {"format": FEATURES_FORMAT, "split": a.split, "limit": a.limit, "budget": a.budget,
            "tasks_file": str(a.tasks) if a.tasks else None,
            "tasks_sha256": sha256_file(a.tasks) if a.tasks else None,
            "hidden": {"layer": reader.n_layers, "what": "input of the final norm (pre-norm residual)",
                       "position": "last suffix token"},
            "device": a.device, "dtype": "fp32", "environment": env, **model_meta(),
            "sessions": len(arr["session_ids"]), "windows": int(len(arr["row_pos"])), "tokens": arr["tokens"],
            "seconds": round(arr["seconds"], 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, features=arr["features"], label_logits=arr["label_logits"], logsumexp=arr["logsumexp"],
             session_ids=arr["session_ids"], row_session=arr["row_session"], row_pos=arr["row_pos"],
             meta=json.dumps(meta))
    _log(json.dumps({k: meta[k] for k in ("split", "sessions", "windows", "tokens", "seconds")}))
    return 0


def train_set_for(budget: str, b: B.Budgets, feats: Features, val_feats: Features | None, data_dir: Path) -> TrainSet:
    """The §4 / §7.2 training set of ``budget`` (dev labels; val labels only for ``200w``)."""
    if budget == "200w":
        if val_feats is None or val_feats.meta.get("split") != "val":
            raise ValueError("200w needs --val-features from the val split")
        labels = qdata.load_labels("val", data_dir)
        B.check_disclosure(b, labels)
        return window_train_set(val_feats, labels, b.disclosed_200w)
    if feats.meta.get("split") != "dev":
        raise ValueError(f"budget {budget} trains on dev features, not {feats.meta.get('split')!r}")
    labels = qdata.load_labels("dev", data_dir)
    B.check_dev_ids(b, [str(s) for s in labels["session_ids"]])
    return dev_train_set(feats, labels, b.sessions(budget))


def cmd_fit(a) -> int:
    if a.budget not in B.BUDGET_NAMES:
        raise SystemExit(f"--budget must be one of {B.BUDGET_NAMES}")
    env = backend.configure(a.device)
    b = B.load(a.budgets)
    feats = load_features(a.features) if a.features else None
    val_feats = load_features(a.val_features) if a.val_features else None
    if feats is None and a.budget != "200w":
        raise SystemExit("--features (dev) is required")
    t0 = time.monotonic()
    ts = train_set_for(a.budget, b, feats, val_feats, a.data_dir)
    probe, log = fit_probe(ts, device=a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    src = val_feats if a.budget == "200w" else feats
    extra = {"budget": a.budget, "chosen_C": log["chosen_C"], "features_sha256": src.sha256,
             "features_split": src.meta["split"], "hidden": src.meta.get("hidden"),
             "budgets_file_sha256": b.file_sha256}
    probe.save(a.out / "probe.npz", extra)
    log = {"recipe": "PREREGISTRATION §4", "budget": a.budget, **extra,
           "protocol": {"C_grid": list(C_GRID), "folds": N_FOLDS, "fold_key": FOLD_KEY, "std_floor": STD_FLOOR,
                        "gtol_max_norm": GTOL, "max_iter": MAX_ITER, "lbfgs_history": HISTORY,
                        "line_search": "strong_wolfe", "dtype": "float64", "class_weight": "balanced n/(k*n_c)",
                        "tie_rule": "smaller C"},
           "probe_sha256": sha256_file(a.out / "probe.npz"), "environment": env,
           "seconds": round(time.monotonic() - t0, 1), **log}
    (a.out / "probe_log.json").write_text(json.dumps(log, indent=1) + "\n")
    _log(json.dumps({"budget": a.budget, "chosen_C": log["chosen_C"], "train_windows": log["train_windows"],
                     "refit_converged": log["refit"]["converged"]}))
    return 0


def cmd_predict(a) -> int:
    path = a.model / "probe.npz" if a.model.is_dir() else a.model
    probe, pmeta = Probe.load(path, device=a.device)
    with np.load(a.features) as z:
        X = z["features"]
        sids, row_pos = z["session_ids"], z["row_pos"]
        fmeta = json.loads(str(z["meta"]))
    if fmeta.get("format") != FEATURES_FORMAT:
        raise SystemExit(f"{a.features}: not a {FEATURES_FORMAT} file")
    if fmeta.get("hidden") != pmeta.get("hidden"):
        raise SystemExit("the features' hidden layer differs from the probe's")
    logits = probe_logits(probe, X)
    lse = torch.logsumexp(torch.as_tensor(logits), dim=1).numpy()
    meta = {**{k: fmeta[k] for k in ("split", "limit", "device", "dtype", "attn_implementation", "model",
                                     "interface_sha256", "interface") if k in fmeta},
            "readout": "probe", "logsumexp": "logsumexp of the four probe logits",
            "probe": {"budget": pmeta.get("budget"), "chosen_C": pmeta.get("chosen_C"), "sha256": sha256_file(path)},
            "features_sha256": sha256_file(a.features), "sessions": len(sids)}
    write_column(a.out, label_logits=logits, logsumexp=lse, session_ids=[str(s) for s in sids], row_pos=row_pos,
                 meta=meta)
    _log(json.dumps({"out": str(a.out), "rows": int(len(row_pos)), "split": meta["split"]}))
    return 0


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypaqwen.probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="hidden[24] at every window of a split")
    f.add_argument("--split", required=True)
    f.add_argument("--tasks", type=Path, default=None, help="explicit task file (a split other than dev/val)")
    f.add_argument("--budget", default=None, help="only this dev budget's sessions")
    f.add_argument("--budgets", type=Path, default=B.PATH)
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--device", default=_default_device())
    f.add_argument("--threads", type=int, default=None)
    f.add_argument("--data-dir", type=Path, default=V04_DATA)
    f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("fit", help="§4 fit of one budget")
    t.add_argument("--budget", required=True)
    t.add_argument("--features", type=Path, default=None)
    t.add_argument("--val-features", type=Path, default=None)
    t.add_argument("--budgets", type=Path, default=B.PATH)
    t.add_argument("--data-dir", type=Path, default=V04_DATA)
    t.add_argument("--device", default=_default_device())
    t.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict", help="a column file from a fitted probe")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"features": cmd_features, "fit": cmd_fit, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
