"""v0.3 Phase C trainer: the v0.3 transformer trained on dev labels (spec v0.3 §9.5, E8).

The pre-registration is ``experiments/v0_3/phase_c/README.md``; this module implements it.
``scripts/v03_train_phase_c.py`` runs it and writes the artefacts. Harness only: never imported by
a weight-build path (runner, packet, guards, ``weight_program/``).

Splits: training reads **dev** only (:func:`load_dev`, which refuses every other split); early
stopping reads the cached **val** encoding through ``v03.scoring.load_val``. Test, OOD and
counterfactual are never opened.

Protocol (README §1-§5), in short:

* :func:`init_model`: ``torch.manual_seed(seed)``, ``DoomLoopTransformer()`` (draws nothing), then
  in ``modules()`` order ``nn.Linear.reset_parameters`` for every Linear and ``normal_(0, 0.02)``
  for every Embedding; ``rel_bias`` stays zero;
* budget subsets: first *k* dev ids by ascending ``sha256(BUDGET_KEY)``; sessions without a
  training position leave the pool;
* loss: pooled mean cross-entropy over the batch's training positions (gate-passing, scored),
  computed by a per-session loop with gradient accumulation; Adam(lr=3e-4), clip 1.0;
  batches of 8, ``randperm`` from a per-run generator seeded with the run seed;
* early stopping on a fixed 500-session val subset (disclosed sessions excluded first), evaluated
  when ``epoch % ceil(cap / 10) == 0`` or at the cap, patience 3 evaluations, keep best.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from ..schema import CLASS_NAMES, ClassLabel, OracleLabel
from ..v03 import infer, renderer, tokenizer
from ..v03.model import DoomLoopTransformer, predict

__all__ = [
    "BUDGET_KEY",
    "ES_KEY",
    "BUDGETS",
    "SEEDS",
    "EPOCH_CAPS",
    "SCHEDULE",
    "LR",
    "BATCH_SESSIONS",
    "GRAD_CLIP",
    "PATIENCE_EVALS",
    "ES_SESSIONS",
    "MAX_ES_EVALS",
    "EMB_STD",
    "THREADS",
    "TRAIN_SPLIT",
    "budget_key",
    "budget_order",
    "budget_subset",
    "es_key",
    "es_session_ids",
    "es_eval_epochs",
    "init_model",
    "TrainSession",
    "DevData",
    "load_dev",
    "train_sessions",
    "EsSet",
    "es_set",
    "es_macro_f1",
    "batch_step",
    "TrainResult",
    "train_run",
]

BUDGET_KEY = "20261105:v03-phase-c-budget:{session_id}"
ES_KEY = "20261105:v03-phase-c-es:{session_id}"
BUDGETS: tuple[int | str, ...] = (32, 128, 512, 2048, "all")
SEEDS: dict[int | str, tuple[int, ...]] = {32: (0, 1, 2), 128: (0, 1, 2), 512: (0, 1, 2), 2048: (0,), "all": (0,)}
EPOCH_CAPS: dict[int | str, int] = {32: 60, 128: 30, 512: 12, 2048: 4, "all": 3}
SCHEDULE: tuple[tuple[int | str, int], ...] = tuple((b, s) for b in BUDGETS for s in SEEDS[b])
LR = 3e-4
BATCH_SESSIONS = 8
GRAD_CLIP = 1.0
PATIENCE_EVALS = 3
ES_SESSIONS = 500
MAX_ES_EVALS = 10
EMB_STD = 0.02
THREADS = 2
TRAIN_SPLIT = "dev"  # the only split load_dev reads

REPO = Path(__file__).resolve().parents[3]
V03_DATA = REPO / "datasets" / "v0_3"
V02_DATA = REPO / "datasets" / "v0_2"
DEV_CACHE_PATH = V03_DATA / f"{TRAIN_SPLIT}.encoded.pt"
DEV_CACHE_FORMAT = "v03-dev-encoded/1"
_LABELS = tuple(OracleLabel)
assert tuple(l.name for l in _LABELS[: len(CLASS_NAMES)]) == CLASS_NAMES  # label index == class index when scored
assert tuple(c.name for c in ClassLabel) == CLASS_NAMES


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    """sha256 of the newline-joined ids in the given order."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Budget subsets and the ES set (README §3, §5)
# ---------------------------------------------------------------------------


def budget_key(session_id: str) -> str:
    return hashlib.sha256(BUDGET_KEY.format(session_id=session_id).encode("utf-8")).hexdigest()


def budget_order(session_ids: Iterable[str]) -> list[str]:
    ids = list(session_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session ids")
    return sorted(ids, key=budget_key)


def budget_subset(session_ids: Iterable[str], budget: int | str) -> list[str]:
    """The first ``budget`` dev ids in :func:`budget_order` (``"all"`` = every id)."""
    order = budget_order(session_ids)
    if budget == "all":
        return order
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"budget must be a positive int or 'all', got {budget!r}")
    if budget > len(order):
        raise ValueError(f"budget {budget} > {len(order)} sessions")
    return order[:budget]


def es_key(session_id: str) -> str:
    return hashlib.sha256(ES_KEY.format(session_id=session_id).encode("utf-8")).hexdigest()


def es_session_ids(val_ids: Iterable[str], disclosed: Iterable[str], n: int = ES_SESSIONS) -> list[str]:
    """Val ids by ascending :func:`es_key`, disclosed sessions removed first, then the first ``n``."""
    ids = list(val_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session ids")
    drop = set(disclosed)
    order = [s for s in sorted(ids, key=es_key) if s not in drop]
    if len(order) < n:
        raise ValueError(f"only {len(order)} non-disclosed val sessions, need {n}")
    return order[:n]


def es_eval_epochs(cap: int) -> list[int]:
    """Epochs after which ES is evaluated: ``e % ceil(cap / 10) == 0`` or ``e == cap``."""
    if cap < 1:
        raise ValueError("cap must be >= 1")
    every = math.ceil(cap / MAX_ES_EVALS)
    return [e for e in range(1, cap + 1) if e % every == 0 or e == cap]


# ---------------------------------------------------------------------------
# Initialisation (README §1)
# ---------------------------------------------------------------------------


def init_model(seed: int) -> DoomLoopTransformer:
    """Seeded standard init of the frozen architecture (the frozen class itself constructs zeros)."""
    torch.manual_seed(seed)
    model = DoomLoopTransformer()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.Linear.reset_parameters(m)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=EMB_STD)
        for layer in model.layers:
            layer.attn.rel_bias.zero_()
    return model.to(dtype=torch.float32, device="cpu").train()


# ---------------------------------------------------------------------------
# Dev data (README §2)
# ---------------------------------------------------------------------------


@dataclass
class TrainSession:
    session_id: str
    tokens: torch.Tensor  # long [n]
    event_ord: torch.Tensor
    offset: torch.Tensor
    positions: torch.Tensor  # long [k]: token index (EOE of the window's last block) per training position
    targets: torch.Tensor  # long [k]: class index


@dataclass
class DevData:
    session_ids: list[str]
    families: list[str]
    payload: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = self.payload
        lengths = p["lengths"].tolist()
        n_ev = p["n_events"].tolist()
        self._tok_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._ev_start = np.concatenate([[0], np.cumsum(n_ev)]).astype(np.int64)
        r_sess = p["row_session"].numpy().astype(np.int64)
        if np.any(np.diff(r_sess) < 0):
            raise ValueError("window rows are not grouped by session")
        self._row_session = r_sess
        self._index = {sid: i for i, sid in enumerate(self.session_ids)}

    def index(self, session_id: str) -> int:
        return self._index[session_id]

    def session(self, s: int) -> TrainSession:
        """Tensors and training positions (gate-passing, scored windows) of session ``s``."""
        p = self.payload
        a, b = int(self._tok_start[s]), int(self._tok_start[s + 1])
        ea = int(self._ev_start[s])
        ra, rb = np.searchsorted(self._row_session, [s, s + 1])
        pos = p["row_pos"][ra:rb].long()
        lab = p["row_label"][ra:rb].long()
        keep = p["row_gate"][ra:rb] & (lab < len(CLASS_NAMES))
        eoe = p["eoe_index"][ea + pos[keep]].long()
        return TrainSession(
            session_id=self.session_ids[s],
            tokens=p["tokens"][a:b].long(),
            event_ord=p["event_ord"][a:b].long(),
            offset=p["offset"][a:b].long(),
            positions=eoe,
            targets=lab[keep],
        )

    def lengths(self) -> np.ndarray:
        return self.payload["lengths"].numpy()


def _dev_input_hashes() -> dict[str, Any]:
    s = TRAIN_SPLIT
    return {
        "renderer_sha256": _sha256_file(Path(renderer.__file__)),
        "tokenizer_sha256": _sha256_file(Path(tokenizer.__file__)),
        f"datasets/v0_3/{s}.raw.jsonl.gz": _sha256_file(V03_DATA / f"{s}.raw.jsonl.gz"),
        f"datasets/v0_2/standard/{s}.events.jsonl.gz": _sha256_file(V02_DATA / "standard" / f"{s}.events.jsonl.gz"),
        f"datasets/v0_2/standard/{s}.oracle.jsonl.gz": _sha256_file(V02_DATA / "standard" / f"{s}.oracle.jsonl.gz"),
        "datasets/v0_3/MANIFEST_v0_3.json": _sha256_file(V03_DATA / "MANIFEST_v0_3.json"),
    }


def _check_manifest(hashes: dict[str, Any]) -> None:
    man = json.loads((V03_DATA / "MANIFEST_v0_3.json").read_text())
    for key, mkey in (("renderer_sha256", "renderer_source_sha256"), ("tokenizer_sha256", "tokeniser_source_sha256")):
        if man.get(mkey) != hashes[key]:
            raise ValueError(f"{key} {hashes[key]} differs from MANIFEST_v0_3.json {mkey} {man.get(mkey)}")
    raw = f"{TRAIN_SPLIT}.raw.jsonl.gz"
    want = man.get("files", {}).get(raw, {}).get("sha256")
    if want != hashes[f"datasets/v0_3/{raw}"]:
        raise ValueError(f"datasets/v0_3/{raw} sha256 differs from MANIFEST_v0_3.json ({want})")


def _check_train_split(split: str) -> None:
    if split != TRAIN_SPLIT:
        raise ValueError(f"Phase C training may read only the {TRAIN_SPLIT!r} split, not {split!r}")


def _build_dev(limit: int | None, split: str = TRAIN_SPLIT) -> dict[str, Any]:
    _check_train_split(split)
    from ..datasets.build import iter_sessions
    from ..v03.build_raw import alignment_errors, iter_raw
    from ..windows import gate, validate_sequence, window_endpoints

    ids: list[str] = []
    fams: list[str] = []
    tok, eo, off, eoe, lengths, n_events = [], [], [], [], [], []
    r_sess, r_pos, r_end, r_lab, r_gate = [], [], [], [], []
    pairs = zip(iter_raw(V03_DATA, split), iter_sessions(V02_DATA, split), strict=True)
    for n, ((sid, raw), (events, record)) in enumerate(pairs):
        if limit is not None and n >= limit:
            break
        if sid != record.session_id or len(raw) != len(events):
            raise ValueError(f"{split} line {n + 1}: raw/events/oracle misaligned for {sid!r}")
        for k, (r, e) in enumerate(zip(raw, events)):
            errs = alignment_errors(r, e)
            if errs:
                raise ValueError(f"{sid} event {k}: raw op misaligned: " + "; ".join(errs))
        blocks = renderer.render_session([r.to_dict() for r in raw], sid)
        enc = tokenizer.encode_session(blocks)
        if len(enc.eoe_index) != len(events):
            raise ValueError(f"{sid}: {len(enc.eoe_index)} EOE tokens for {len(events)} events")
        s = len(ids)
        ids.append(sid)
        fams.append(record.scenario_family)
        tok.extend(enc.tokens), eo.extend(enc.event_ord), off.extend(enc.offset), eoe.extend(enc.eoe_index)
        lengths.append(len(enc.tokens))
        n_events.append(len(enc.eoe_index))
        validate_sequence(events)
        for pos in window_endpoints(events):
            endpoint = events[pos].sequence_index
            try:
                label = OracleLabel(record.window_labels[endpoint])
            except KeyError:
                raise ValueError(f"{sid}: no oracle label for window endpoint {endpoint}") from None
            r_sess.append(s), r_pos.append(pos), r_end.append(endpoint)
            r_lab.append(_LABELS.index(label)), r_gate.append(gate(events[: pos + 1]).passed)
    if max(tok, default=0) >= 2**15 or max(eo, default=0) >= 2**15:
        raise ValueError("token id or event ordinal does not fit int16")
    return {
        "session_ids": ids,
        "families": fams,
        "tokens": torch.tensor(tok, dtype=torch.int16),
        "event_ord": torch.tensor(eo, dtype=torch.int16),
        "offset": torch.tensor(off, dtype=torch.int16),
        "eoe_index": torch.tensor(eoe, dtype=torch.int32),
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "n_events": torch.tensor(n_events, dtype=torch.int64),
        "row_session": torch.tensor(r_sess, dtype=torch.int32),
        "row_pos": torch.tensor(r_pos, dtype=torch.int32),
        "row_endpoint": torch.tensor(r_end, dtype=torch.int32),
        "row_label": torch.tensor(r_lab, dtype=torch.int8),
        "row_gate": torch.tensor(r_gate, dtype=torch.bool),
    }


def load_dev(*, cache: bool = True, limit: int | None = None, cache_path: Path | None = None) -> DevData:
    """Rendered and tokenised dev sessions with window labels and the v0.2 gate (cached)."""
    hashes = _dev_input_hashes()
    _check_manifest(hashes)
    key = {"format": DEV_CACHE_FORMAT, "split": TRAIN_SPLIT, **hashes}
    path = cache_path or DEV_CACHE_PATH
    use_cache = cache and limit is None
    payload = None
    built = False
    if use_cache and path.is_file():
        try:
            stored = torch.load(path, weights_only=True)
        except Exception:  # noqa: BLE001 - a broken cache is rebuilt
            stored = None
        if isinstance(stored, dict) and stored.get("key") == key:
            payload = stored["payload"]
    if payload is None:
        payload = _build_dev(limit)
        built = True
        if use_cache:
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"key": key, "payload": payload}, tmp)
            tmp.replace(path)
    data = DevData(list(payload["session_ids"]), list(payload["families"]), payload)
    data.inputs = {"cache_key": key, "limit": limit, "built_now": built}
    return data


def train_sessions(dev: DevData, ids: Sequence[str]) -> list[TrainSession]:
    """Training pool for ``ids`` (in that order): sessions with at least one training position."""
    out = []
    for sid in ids:
        ts = dev.session(dev.index(sid))
        if ts.positions.numel():
            out.append(ts)
    return out


# ---------------------------------------------------------------------------
# Early-stopping set and metric (README §5)
# ---------------------------------------------------------------------------


@dataclass
class EsSet:
    session_ids: list[str]
    encs: list[Any]  # EncodedSession-like (tokens, event_ord, offset, eoe_index)
    row_pos: list[np.ndarray]  # per session: block positions of its gate-passing scored windows
    y_true: np.ndarray  # concatenated targets in session order

    @property
    def windows(self) -> int:
        return int(self.y_true.shape[0])


def es_set(val: Any, n: int = ES_SESSIONS, disclosed: Iterable[str] | None = None) -> EsSet:
    """The fixed ES subset of ``val`` (a ``scoring.ValData``)."""
    from ..v03 import scoring

    if disclosed is None:
        disclosed = scoring.disclosed_sessions(val.rows)
    ids = es_session_ids(val.session_ids, disclosed, n)
    index = {sid: i for i, sid in enumerate(val.session_ids)}
    encs, row_pos, ys = [], [], []
    for sid in ids:
        s = index[sid]
        rp, yy = [], []
        for i in val.rows_of(s).tolist():
            r = val.rows[i]
            if r.gate_passed and r.label.is_scored:
                rp.append(int(val.row_pos[i]))
                yy.append(int(r.label.to_class()))
        encs.append(val.sessions[s])
        row_pos.append(np.asarray(rp, dtype=np.int64))
        ys.extend(yy)
    return EsSet(ids, encs, row_pos, np.asarray(ys, dtype=np.int64))


def es_macro_f1(model: DoomLoopTransformer, es: EsSet) -> float:
    """Macro-F1 (``metrics_from_counts`` conventions) of argmax at ``EOE_pos`` over the ES windows."""
    from .phase_c import macro_f1

    preds = []
    for enc, rp in zip(es.encs, es.row_pos):
        if rp.size == 0:
            continue
        p = predict(infer.eoe_logits(model, enc)).numpy()
        preds.append(p[rp])
    y_pred = np.concatenate(preds) if preds else np.zeros(0, np.int64)
    return macro_f1(es.y_true, y_pred)


# ---------------------------------------------------------------------------
# Training (README §4)
# ---------------------------------------------------------------------------


def batch_step(
    model: DoomLoopTransformer,
    opt: torch.optim.Optimizer,
    batch: Sequence[TrainSession],
    *,
    clip: float = GRAD_CLIP,
    timings: list[tuple[int, float]] | None = None,
) -> tuple[float, int]:
    """One optimiser step on the pooled mean cross-entropy of ``batch``. Returns (loss, positions)."""
    total = sum(int(ts.positions.numel()) for ts in batch)
    if total == 0:
        raise ValueError("batch without training positions")
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for ts in batch:
        t0 = time.perf_counter()
        logits = model(ts.tokens, ts.event_ord, ts.offset)
        loss = nn.functional.cross_entropy(logits[ts.positions], ts.targets, reduction="sum") / total
        loss.backward()
        loss_sum += float(loss.detach())
        if timings is not None:
            timings.append((int(ts.tokens.shape[0]), time.perf_counter() - t0))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
    opt.step()
    return loss_sum, total


@dataclass
class TrainResult:
    state_dict: dict[str, torch.Tensor]
    seed: int
    cap: int
    pool_sessions: int
    train_positions: int
    train_class_counts: dict[str, int]
    epochs_run: int
    steps: int
    best_epoch: int
    best_es_macro_f1: float
    stopped: str
    epochs: list[dict[str, Any]] = field(default_factory=list)
    es_evaluations: list[dict[str, Any]] = field(default_factory=list)
    session_timings: list[tuple[int, float]] | None = None

    def log(self) -> dict[str, Any]:
        out = {
            "seed": self.seed,
            "epoch_cap": self.cap,
            "pool_sessions": self.pool_sessions,
            "train_positions": self.train_positions,
            "train_class_counts": self.train_class_counts,
            "epochs_run": self.epochs_run,
            "steps": self.steps,
            "best_epoch": self.best_epoch,
            "best_es_macro_f1": self.best_es_macro_f1,
            "stopped": self.stopped,
            "epochs": self.epochs,
            "es_evaluations": self.es_evaluations,
        }
        if self.session_timings is not None:
            out["session_timings_tokens_seconds"] = [[n, round(t, 6)] for n, t in self.session_timings]
        return out


def train_run(
    pool: Sequence[TrainSession],
    evaluate: Callable[[DoomLoopTransformer], float],
    *,
    seed: int,
    cap: int,
    lr: float = LR,
    batch_sessions: int = BATCH_SESSIONS,
    clip: float = GRAD_CLIP,
    patience: int = PATIENCE_EVALS,
    record_session_times: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TrainResult:
    """One Phase C run (README §4-§5). ``evaluate(model)`` returns the ES macro-F1."""
    if not pool:
        raise ValueError("empty training pool")
    model = init_model(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eval_at = set(es_eval_epochs(cap))
    n = len(pool)
    best_f1, best_epoch, best_state, since = -1.0, 0, None, 0
    epochs: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    timings: list[tuple[int, float]] | None = [] if record_session_times else None
    stopped = "epoch_cap"
    steps = 0
    epoch = 0
    for epoch in range(1, cap + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen).tolist()
        loss_w, pos_total, ep_steps = 0.0, 0, 0
        for i in range(0, n, batch_sessions):
            batch = [pool[j] for j in perm[i : i + batch_sessions]]
            loss, k = batch_step(model, opt, batch, clip=clip, timings=timings)
            loss_w += loss * k
            pos_total += k
            ep_steps += 1
        steps += ep_steps
        rec: dict[str, Any] = {"epoch": epoch, "train_loss": loss_w / pos_total, "steps": ep_steps,
                               "train_wall_s": time.perf_counter() - t0}
        epochs.append(rec)
        msg = f"epoch {epoch}/{cap}: loss {rec['train_loss']:.4f}, {ep_steps} steps, {rec['train_wall_s']:.1f}s"
        if epoch in eval_at:
            model.eval()
            t1 = time.perf_counter()
            f1 = float(evaluate(model))
            ev = {"epoch": epoch, "es_macro_f1": f1, "wall_s": time.perf_counter() - t1}
            evals.append(ev)
            msg += f"; ES macro-F1 {f1:.4f} ({ev['wall_s']:.1f}s)"
            if f1 > best_f1:
                best_f1, best_epoch, since = f1, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since += 1
            if progress is not None:
                progress(msg)
            if since >= patience:
                stopped = f"patience ({patience} evaluations without improvement)"
                break
        elif progress is not None:
            progress(msg)
    assert best_state is not None  # the cap epoch is always evaluated
    counts = np.bincount(torch.cat([ts.targets for ts in pool]).numpy(), minlength=len(CLASS_NAMES))
    return TrainResult(
        state_dict=best_state,
        seed=seed,
        cap=cap,
        pool_sessions=n,
        train_positions=int(counts.sum()),
        train_class_counts={c: int(counts[i]) for i, c in enumerate(CLASS_NAMES)},
        epochs_run=epoch,
        steps=steps,
        best_epoch=best_epoch,
        best_es_macro_f1=best_f1,
        stopped=stopped,
        epochs=epochs,
        es_evaluations=evals,
        session_timings=timings,
    )
