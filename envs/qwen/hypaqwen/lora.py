"""Qwen-LoRA: the one fixed LoRA recipe (PREREGISTRATION §5, §7.2), with its readout.

    python -m hypaqwen.lora train --budget 512 --seed 0 --out runs/lora-512-s0 --device cuda
    python -m hypaqwen.lora readout --run runs/lora-512-s0 --split val --out cols/lora-512-s0.npz --device cuda

**Adapters** (``peft`` 0.21.2): rank 16, alpha 32 (scale 2), dropout 0.05 on the adapter input, bias
none, standard init (A Kaiming-uniform, B zero) drawn after ``torch.manual_seed(seed)``, on the CPU
copy of the model before it moves to the device, so the initial adapters do not depend on the GPU.
Targets (:func:`target_modules`): ``self_attn.{q,k,v,o}_proj`` of the 6 full-attention layers,
``linear_attn.{in_proj_qkv, in_proj_z, out_proj}`` of the 18 Gated DeltaNet layers and
``mlp.{gate,up,down}_proj`` of all 24, named in full so nothing else can match; 10,223,616
trainable parameters (checked at every attach on the real configuration).

**Loss.** Cross-entropy over the four label-word logits (the frozen verbaliser: tied label rows
times the final-norm output at the suffix's last token) at every training window (§4), pooled mean
over the batch's windows, unweighted; one backward per session with gradient accumulation (v0.3
Phase C). The logits are those of the separate prompt ``prefix + blocks[0..t] + suffix``, computed
by branching the cache as the readout does, but on :class:`FnCache`: ``transformers``' cache
updates the DeltaNet conv and recurrent states in place (``copy_``), which autograd refuses (or
would corrupt) when a state is reused, so :class:`FnCache` stores every update out of place and
:meth:`FnCache.fork` hands the suffix run a shallow copy. The main stream then continues from the
untouched state and gradients flow through every state into earlier blocks. Its layers also report
``record_past``, which keeps ``Qwen3_5GatedDeltaNet`` off the in-place one-token decode path. The
test checks the branched logits, loss and gradients against separate prompts.

**Optimiser and schedule.** AdamW (lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0), global
norm clip 1.0, 8 sessions per step with the last partial batch kept, one ``torch.Generator`` per run
seeded with the seed, ``randperm`` per epoch; sessions without a training window leave the pool.
The learning rate of optimiser step ``s`` (1-based) of ``S`` planned steps, warm-up ``W =
ceil(0.05·S)``: ``lr·s/W`` for ``s <= W``, then ``lr·½(1 + cos(π(s − W)/(S − W)))``, which is 0 at
``s = S`` (:func:`lr_at`). ``cap(b) = min(60, max(3, ceil(6144 / b)))``; planned steps ``cap ·
ceil(pool / 8)``. Precision: fp32 master weights, ``torch.autocast("cuda", bfloat16)`` for forward
and backward on CUDA under §3.1's settings; fp32 without autocast elsewhere.

**Early stopping** (v0.3 Phase C §5): macro-F1 of the verbaliser argmax over the gate-passing scored
windows of the 500-session val ES set (``BUDGETS.json`` ``es_val``), after epoch ``e`` when ``e %
ceil(cap / 10) == 0`` or ``e == cap``, patience 3 evaluations, strictly better is the new best, the
best adapters kept. The ES readout is the standard :class:`~hypaqwen.readout.Verbaliser` on the
unmerged model in eval mode, fp32 (no autocast). **200w** has no early stopping: cap 31, final state.
Early stopping only truncates the schedule.

**Merge and readout.** The adapters are merged into the fp32 weights as ``W + 2·(B @ A)`` computed
in fp32 (:func:`merge_adapter`, the arithmetic of ``peft``'s ``merge``). A run directory holds
``adapter.safetensors`` (the kept adapters), ``train_log.json`` (protocol constants, counts,
per-epoch loss, ES evaluations, the adapter's sha256 and the merged weights' sha256 on the
training device, environment) and nothing that depends on recomputation. ``readout`` rebuilds the
merged model from the pinned checkpoint and the adapter, checks the merged sha256 when it runs on
the training device type, and writes a column file with the standard verbaliser, as every
verbaliser arm.

Data: dev for training, the val ES set for early stopping, the 200 disclosed val windows for 200w.
No other split is read by training code; ``readout`` reads only the split it is given.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer

from . import V04_DATA
from . import backend
from . import budgets as B
from . import data as qdata
from .columns import (N_CLASSES, cross_entropy_rows, iter_split_tasks, macro_f1, model_meta, session_rows,
                      sha256_file, write_column)
from .interface import CLASS_NAMES, Interface
from .readout import Verbaliser

__all__ = [
    "RUN_FORMAT", "RANK", "ALPHA", "SCALE", "DROPOUT", "LR", "BETAS", "EPS", "WEIGHT_DECAY", "WARMUP_FRACTION",
    "GRAD_CLIP", "BATCH_SESSIONS", "CAP_PASSES", "CAP_MIN", "CAP_MAX", "PATIENCE_EVALS", "MAX_ES_EVALS",
    "TRAINABLE_PARAMETERS", "TEXT_PARAMETERS", "FULL_ATTENTION_LAYERS", "target_modules", "epoch_cap",
    "planned_steps", "warmup_steps", "lr_at", "es_eval_epochs", "attach_lora", "trainable_parameters", "FnCache",
    "branched_label_logits", "separate_label_logits", "Item", "Divergence", "batch_step", "TrainResult",
    "train_run", "es_evaluator", "adapter_state", "merge_adapter", "weights_sha256", "save_run", "load_merged",
    "run_training", "readout_tasks", "main",
]

RUN_FORMAT = "v04-lora-run/1"
RANK, ALPHA, DROPOUT = 16, 32, 0.05
SCALE = ALPHA / RANK  # 2.0
LR, BETAS, EPS, WEIGHT_DECAY = 2e-4, (0.9, 0.999), 1e-8, 0.0
WARMUP_FRACTION = 0.05
GRAD_CLIP = 1.0
BATCH_SESSIONS = 8
CAP_PASSES, CAP_MIN, CAP_MAX = 6144, 3, 60
PATIENCE_EVALS, MAX_ES_EVALS = 3, 10
TRAINABLE_PARAMETERS = 10_223_616
TEXT_PARAMETERS = 752_393_024
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
DELTANET_PROJ = ("in_proj_qkv", "in_proj_z", "out_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
ADAPTER_FILE, LOG_FILE = "adapter.safetensors", "train_log.json"


# ---------------------------------------------------------------------------
# Protocol numbers (§5)
# ---------------------------------------------------------------------------


def target_modules(config) -> list[str]:
    """Full module names of the adapted linears, in layer order."""
    names = []
    for i, kind in enumerate(config.layer_types):
        p = f"model.layers.{i}."
        if kind == "full_attention":
            names += [p + "self_attn." + m for m in ATTN_PROJ]
        elif kind == "linear_attention":
            names += [p + "linear_attn." + m for m in DELTANET_PROJ]
        else:
            raise ValueError(f"layer {i}: unexpected layer type {kind!r}")
        names += [p + "mlp." + m for m in MLP_PROJ]
    return names


def epoch_cap(b: int) -> int:
    """``min(60, max(3, ceil(6144 / b)))``; ``b`` = budget sessions (200 for 200w)."""
    if b < 1:
        raise ValueError("budget must be >= 1")
    return min(CAP_MAX, max(CAP_MIN, math.ceil(CAP_PASSES / b)))


def planned_steps(cap: int, pool: int) -> int:
    return cap * math.ceil(pool / BATCH_SESSIONS)


def warmup_steps(planned: int) -> int:
    return max(1, math.ceil(WARMUP_FRACTION * planned))


def lr_at(step: int, planned: int, base: float = LR) -> float:
    """Learning rate of optimiser step ``step`` (1-based) of ``planned``: linear warm-up, cosine to 0."""
    if not 1 <= step <= planned:
        raise ValueError(f"step {step} outside 1..{planned}")
    w = warmup_steps(planned)
    if step <= w:
        return base * step / w
    return base * 0.5 * (1.0 + math.cos(math.pi * (step - w) / (planned - w)))


def es_eval_epochs(cap: int) -> list[int]:
    """Epochs after which ES is evaluated: ``e % ceil(cap / 10) == 0`` or ``e == cap``."""
    if cap < 1:
        raise ValueError("cap must be >= 1")
    every = math.ceil(cap / MAX_ES_EVALS)
    return [e for e in range(1, cap + 1) if e % every == 0 or e == cap]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _lora_config(config):
    from peft import LoraConfig

    return LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none",
                      target_modules=target_modules(config), init_lora_weights=True)


def attach_lora(model, seed: int, *, check_count: bool = True):
    """``peft`` adapters on ``model`` (freezes everything else), init drawn after ``manual_seed(seed)``."""
    from peft import get_peft_model

    torch.manual_seed(seed)
    pm = get_peft_model(model, _lora_config(model.config))
    n = trainable_parameters(pm)
    if check_count and n != TRAINABLE_PARAMETERS:
        raise ValueError(f"{n} trainable parameters, §5 has {TRAINABLE_PARAMETERS}")
    return pm


def trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_state(peft_model) -> dict[str, torch.Tensor]:
    """``{module}.lora_A`` [r, in] and ``{module}.lora_B`` [out, r], fp32 CPU copies."""
    from peft.tuners.lora import LoraLayer

    out = {}
    for name, mod in peft_model.get_base_model().named_modules():
        if isinstance(mod, LoraLayer) and "default" in mod.lora_A:
            out[f"{name}.lora_A"] = mod.lora_A["default"].weight.detach().to("cpu", torch.float32, copy=True)
            out[f"{name}.lora_B"] = mod.lora_B["default"].weight.detach().to("cpu", torch.float32, copy=True)
    return out


@torch.no_grad()
def merge_adapter(model, adapter: Mapping[str, torch.Tensor], scale: float = SCALE) -> None:
    """In place on a plain model: ``W <- W + scale · (B @ A)`` in fp32 for every adapted linear."""
    names = sorted({k.rsplit(".", 1)[0] for k in adapter})
    want = target_modules(model.config)
    if set(names) != set(want) or len(adapter) != 2 * len(want):
        raise ValueError("adapter modules differ from the §5 targets")
    mods = dict(model.named_modules())
    for name in want:
        lin = mods[name]
        if lin.weight.dtype != torch.float32:
            raise ValueError(f"{name}: merge needs fp32 weights, got {lin.weight.dtype}")
        a = adapter[name + ".lora_A"].to(lin.weight.device, torch.float32)
        b = adapter[name + ".lora_B"].to(lin.weight.device, torch.float32)
        if a.shape != (RANK, lin.in_features) or b.shape != (lin.out_features, RANK):
            raise ValueError(f"{name}: adapter shapes {tuple(a.shape)} {tuple(b.shape)}")
        lin.weight.add_((b @ a) * scale)


def weights_sha256(model) -> str:
    """sha256 over the state dict in name order: name, NUL, the tensor's bytes."""
    import hashlib

    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode() + b"\0")
        h.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def _tensors_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode() + b"\0")
        h.update(state[name].contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Branched, differentiable label logits
# ---------------------------------------------------------------------------


class _FnLinearLayer(LinearAttentionLayer):
    """A linear-attention cache layer whose updates never write into a tensor autograd may hold."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.record_past = True  # Cache.is_recording_past: no in-place causal_conv1d_update decode path

    def update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None,
                          **kwargs) -> torch.Tensor:
        if not self.is_conv_states_initialized[state_idx]:
            self.dtype, self.device = conv_states.dtype, conv_states.device
            self.conv_kernel_size[state_idx] = conv_kernel_size or conv_states.shape[-1]
            self.is_conv_states_initialized[state_idx] = True
        if self.has_previous_state[state_idx]:
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        else:  # the first piece: no left context (causal_conv1d_fn pads with zeros, as the original's padding)
            full = conv_states
            self.has_previous_state[state_idx] = True
        self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx]:]
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs) -> torch.Tensor:
        self.recurrent_states[state_idx] = recurrent_states
        self.is_recurrent_states_initialized[state_idx] = True
        return recurrent_states

    def fork(self) -> "_FnLinearLayer":
        new = copy.copy(self)
        for attr in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                     "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
            setattr(new, attr, dict(getattr(self, attr)))
        return new


class FnCache(Cache):
    """Qwen3.5 cache with out-of-place updates; :meth:`fork` is a cheap, independent branch."""

    def __init__(self, config):
        layers = []
        for i, kind in enumerate(config.layer_types):
            if kind == "linear_attention":
                layers.append(_FnLinearLayer())
            elif kind == "full_attention":
                layers.append(DynamicLayer())
            else:
                raise ValueError(f"layer {i}: unexpected layer type {kind!r}")
        super().__init__(layers=layers)

    def fork(self) -> "FnCache":
        new = copy.copy(self)
        new.layers = [layer.fork() if isinstance(layer, _FnLinearLayer) else copy.copy(layer) for layer in self.layers]
        return new


def _label_rows(model, iface: Interface) -> torch.Tensor:
    emb = model.get_input_embeddings().weight
    if model.get_output_embeddings().weight.data_ptr() != emb.data_ptr():
        raise ValueError("expected tied input and output embeddings")
    return emb.index_select(0, torch.tensor(iface.label_ids, dtype=torch.long, device=emb.device))


def _ids(ids: Sequence[int], device) -> torch.Tensor:
    return torch.tensor([list(ids)], dtype=torch.long, device=device)


def branched_label_logits(model, iface: Interface, block_ids: Sequence[Sequence[int]],
                          windows: Sequence[int]) -> torch.Tensor:
    """``[len(windows), 4]`` label logits, windows ascending, one pass with a forked cache per window."""
    want = sorted(set(int(w) for w in windows))
    if not want or list(windows) != want or want[0] < 0 or want[-1] >= len(block_ids):
        raise ValueError("windows must be ascending, unique and inside the session")
    inner, rows = model.model, _label_rows(model, iface)
    device = rows.device
    suffix = _ids(iface.suffix_ids, device)
    cache = FnCache(model.config)
    pending, out, wset = list(iface.prefix_ids), [], set(want)
    for k, ids in enumerate(block_ids[: want[-1] + 1]):
        pending.extend(ids)
        if k not in wset:
            continue  # blocks between windows run as one chunk
        inner(input_ids=_ids(pending, device), past_key_values=cache, use_cache=True)
        pending = []
        h = inner(input_ids=suffix, past_key_values=cache.fork(), use_cache=True).last_hidden_state[0, -1]
        out.append(rows @ h)
    return torch.stack(out)


def separate_label_logits(model, iface: Interface, block_ids: Sequence[Sequence[int]],
                          windows: Sequence[int]) -> torch.Tensor:
    """The reference: each window as its own prompt ``prefix + blocks[0..t] + suffix``, no cache."""
    inner, rows = model.model, _label_rows(model, iface)
    out = []
    for t in windows:
        h = inner(input_ids=_ids(iface.window_prompt(block_ids, t), rows.device), use_cache=False)
        out.append(rows @ h.last_hidden_state[0, -1])
    return torch.stack(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One pool session: its blocks up to its last training window, the windows and their targets."""

    session_id: str
    tokens: np.ndarray  # int32, the blocks' token ids concatenated
    bounds: np.ndarray  # int64 [n_blocks + 1]
    windows: list[int]
    targets: torch.Tensor  # long [len(windows)]

    @classmethod
    def build(cls, session_id: str, block_ids: Sequence[Sequence[int]], windows: Sequence[int],
              targets: Sequence[int]) -> "Item":
        windows = [int(w) for w in windows]
        if not windows or windows != sorted(set(windows)) or len(targets) != len(windows):
            raise ValueError(f"{session_id}: windows must be non-empty, ascending, unique, one target each")
        if any(not 0 <= int(t) < N_CLASSES for t in targets):
            raise ValueError(f"{session_id}: target outside the four classes")
        kept = block_ids[: windows[-1] + 1]
        bounds = np.concatenate([[0], np.cumsum([len(b) for b in kept])]).astype(np.int64)
        tokens = np.fromiter(itertools.chain.from_iterable(kept), dtype=np.int32, count=int(bounds[-1]))
        return cls(session_id, tokens, bounds, windows, torch.tensor([int(t) for t in targets], dtype=torch.long))

    def block_ids(self) -> list[list[int]]:
        return [self.tokens[a:b].tolist() for a, b in zip(self.bounds[:-1], self.bounds[1:])]


class Divergence(RuntimeError):
    """A non-finite loss or gradient norm: the run is reported as failed (PREREGISTRATION §10)."""


def batch_step(model, opt, batch: Sequence[Item], iface: Interface, *, device: str, lr: float,
               params: Sequence[torch.Tensor]) -> tuple[float, int]:
    """One optimiser step on the pooled mean cross-entropy of ``batch``. Returns (loss, windows)."""
    total = sum(len(it.windows) for it in batch)
    if total == 0:
        raise ValueError("batch without training windows")
    hf = model.get_base_model()
    opt.zero_grad(set_to_none=True)
    for g in opt.param_groups:
        g["lr"] = lr
    loss_sum = 0.0
    for it in batch:
        with backend.autocast(device):
            logits = branched_label_logits(hf, iface, it.block_ids(), it.windows)
        loss = cross_entropy_rows(logits.float(), it.targets.to(logits.device)).sum() / total
        if not torch.isfinite(loss):
            raise Divergence(f"non-finite loss on {it.session_id}")
        loss.backward()
        loss_sum += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP)
    if not torch.isfinite(norm):
        raise Divergence("non-finite gradient norm")
    opt.step()
    return loss_sum, total


@dataclass
class TrainResult:
    adapter: dict[str, torch.Tensor]
    seed: int
    cap: int
    planned_steps: int
    pool_sessions: int
    train_windows: int
    train_class_counts: dict[str, int]
    epochs_run: int
    steps: int
    kept_epoch: int
    best_es_macro_f1: float | None
    stopped: str
    epochs: list[dict[str, Any]] = field(default_factory=list)
    es_evaluations: list[dict[str, Any]] = field(default_factory=list)

    def log(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "adapter"}


def train_run(model, pool: Sequence[Item], iface: Interface, *, seed: int, cap: int, device: str,
              evaluate: Callable[[Any], float] | None, progress: Callable[[str], None] | None = None) -> TrainResult:
    """One §5 run on a ``peft`` model already carrying the seed's adapters. ``evaluate=None``: no ES (200w)."""
    if not pool:
        raise ValueError("empty training pool")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)
    n = len(pool)
    planned = planned_steps(cap, n)
    eval_at = set(es_eval_epochs(cap)) if evaluate is not None else set()
    best_f1, best_epoch, best, since = None, 0, None, 0
    epochs, evals = [], []
    stopped, steps, epoch = "epoch_cap", 0, 0
    for epoch in range(1, cap + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen).tolist()
        loss_w, win_total, ep_steps = 0.0, 0, 0
        for i in range(0, n, BATCH_SESSIONS):
            steps += 1
            loss, k = batch_step(model, opt, [pool[j] for j in perm[i: i + BATCH_SESSIONS]], iface, device=device,
                                 lr=lr_at(steps, planned), params=params)
            loss_w += loss * k
            win_total += k
            ep_steps += 1
        rec = {"epoch": epoch, "train_loss": loss_w / win_total, "steps": ep_steps, "lr_last": lr_at(steps, planned),
               "train_wall_s": round(time.perf_counter() - t0, 3)}
        epochs.append(rec)
        msg = f"epoch {epoch}/{cap}: loss {rec['train_loss']:.4f}, {ep_steps} steps, {rec['train_wall_s']:.1f}s"
        if epoch in eval_at:
            model.eval()
            t1 = time.perf_counter()
            f1 = float(evaluate(model))
            evals.append({"epoch": epoch, "es_macro_f1": f1, "wall_s": round(time.perf_counter() - t1, 3)})
            msg += f"; ES macro-F1 {f1:.4f}"
            if best_f1 is None or f1 > best_f1:  # strictly better is the new best
                best_f1, best_epoch, since, best = f1, epoch, 0, adapter_state(model)
            else:
                since += 1
        if progress is not None:
            progress(msg)
        if evaluate is not None and since >= PATIENCE_EVALS:
            stopped = f"patience ({PATIENCE_EVALS} evaluations without improvement)"
            break
    if evaluate is None:
        best, best_epoch = adapter_state(model), epoch
    assert best is not None  # the cap epoch is always evaluated
    counts = np.bincount(torch.cat([it.targets for it in pool]).numpy(), minlength=N_CLASSES)
    return TrainResult(adapter=best, seed=seed, cap=cap, planned_steps=planned, pool_sessions=n,
                       train_windows=int(counts.sum()),
                       train_class_counts={c: int(counts[i]) for i, c in enumerate(CLASS_NAMES)},
                       epochs_run=epoch, steps=steps, kept_epoch=best_epoch, best_es_macro_f1=best_f1,
                       stopped=stopped if evaluate is not None else "no early stopping (200w): final state",
                       epochs=epochs, es_evaluations=evals)


def es_evaluator(items: Sequence[Item], iface: Interface, device: str) -> Callable[[Any], float]:
    """ES macro-F1: the standard verbaliser (fp32, eval mode) on the unmerged model, first-index argmax."""
    y_true = torch.cat([it.targets for it in items]).numpy()

    def evaluate(peft_model) -> float:
        verb = Verbaliser(peft_model.get_base_model(), iface)
        preds = []
        with backend.compute(device):
            for it in items:
                r = verb.session(it.block_ids(), it.windows)
                preds.append(np.argmax(r.label_logits.numpy(), axis=1))
        return macro_f1(y_true, np.concatenate(preds))

    return evaluate


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


def save_run(out: Path, result: TrainResult, info: Mapping[str, Any]) -> dict[str, Any]:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: result.adapter[k].contiguous() for k in sorted(result.adapter)}, str(out / ADAPTER_FILE),
              metadata={"format": RUN_FORMAT, "rank": str(RANK), "alpha": str(ALPHA)})
    log = {"format": RUN_FORMAT, "status": "ok", **info, **result.log(),
           "adapter_tensors_sha256": _tensors_sha256(result.adapter),
           "adapter_file_sha256": sha256_file(out / ADAPTER_FILE)}
    (out / LOG_FILE).write_text(json.dumps(log, indent=1, default=float) + "\n")
    return log


def _read_adapter(run: Path, log: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    adapter = load_file(str(run / ADAPTER_FILE))
    if _tensors_sha256(adapter) != log["adapter_tensors_sha256"]:
        raise ValueError(f"{run / ADAPTER_FILE}: adapter sha256 differs from the training log")
    return adapter


def load_merged(run: Path, device: str, loader: Callable[[str, str], tuple[Any, Any]]):
    """``(tok, merged model, log, merged sha256, matches the training record)``."""
    log = json.loads((run / LOG_FILE).read_text())
    if log.get("format") != RUN_FORMAT or log.get("status") != "ok":
        raise ValueError(f"{run}: not a finished {RUN_FORMAT} run")
    adapter = _read_adapter(run, log)
    tok, model = loader(device, "fp32")
    merge_adapter(model, adapter)
    sha = weights_sha256(model)
    same_kind = torch.device(device).type == torch.device(log["merged"]["device"]).type
    if same_kind and sha != log["merged"]["sha256"]:
        raise ValueError(f"merged sha256 {sha} != training record {log['merged']['sha256']} on {device}")
    return tok, model, log, sha, sha == log["merged"]["sha256"]


@dataclass
class TrainInputs:
    pool: list[Item]
    es: list[Item] | None  # None: no early stopping (200w)
    budget_size: int
    provenance: dict[str, Any]


def run_training(inputs: TrainInputs, *, budget: str, seed: int, device: str, out: Path,
                 loader: Callable[[str, str], tuple[Any, Any]], iface_of: Callable[[Any], Interface],
                 cap: int | None = None, check_count: bool = True, environment: Mapping[str, Any] | None = None,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Attach, train, save the adapter, merge on a fresh copy and record its sha256. ``cap`` overrides tests only."""
    if (out / LOG_FILE).exists():
        raise FileExistsError(f"{out / LOG_FILE} exists; a run directory is never reused")
    cap = epoch_cap(inputs.budget_size) if cap is None else cap
    tok, base = loader("cpu", "fp32")
    iface = iface_of(tok)
    model = attach_lora(base, seed, check_count=check_count).to(device)
    evaluate = es_evaluator(inputs.es, iface, device) if inputs.es is not None else None
    info = {"budget": budget, "budget_size": inputs.budget_size, "seed": seed, "device": device,
            "protocol": protocol(), "provenance": inputs.provenance, "environment": dict(environment or {}),
            "trainable_parameters": trainable_parameters(model),
            "es_sessions": len(inputs.es) if inputs.es is not None else 0,
            "es_windows": sum(len(it.windows) for it in inputs.es) if inputs.es is not None else 0}
    t0 = time.monotonic()
    try:
        with backend.compute(device):
            result = train_run(model, inputs.pool, iface, seed=seed, cap=cap, device=device, evaluate=evaluate,
                               progress=progress)
    except Divergence as e:
        out.mkdir(parents=True, exist_ok=True)
        (out / LOG_FILE).write_text(json.dumps({"format": RUN_FORMAT, "status": f"failed: {e}", **info}, indent=1))
        raise
    info["train_seconds"] = round(time.monotonic() - t0, 1)
    del model, base
    _, fresh = loader(device, "fp32")
    merge_adapter(fresh, result.adapter)
    info["merged"] = {"device": device, "sha256": weights_sha256(fresh), "formula": "W + 2*(B @ A), fp32"}
    return save_run(out, result, info)


def protocol() -> dict[str, Any]:
    return {"recipe": "PREREGISTRATION §5", "rank": RANK, "alpha": ALPHA, "scale": SCALE, "dropout": DROPOUT,
            "lr": LR, "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION, "warmup": "ceil(0.05*planned), linear from lr/W at step 1",
            "decay": "cosine to 0 at the planned end", "grad_clip": GRAD_CLIP, "batch_sessions": BATCH_SESSIONS,
            "epoch_cap": f"min({CAP_MAX}, max({CAP_MIN}, ceil({CAP_PASSES}/b)))", "patience_evals": PATIENCE_EVALS,
            "es_every": "ceil(cap/10)", "precision": "fp32 master; bf16 autocast on CUDA only",
            "loss": "CE over the four label-word logits, pooled mean over the batch's windows",
            "targets": "self_attn.{q,k,v,o}_proj (full attention), linear_attn.{in_proj_qkv,in_proj_z,out_proj} "
                       "(Gated DeltaNet), mlp.{gate,up,down}_proj (all)"}


# ---------------------------------------------------------------------------
# Data (dev budgets, 200w, the ES set)
# ---------------------------------------------------------------------------


def _tasks_by_id(split: str, ids: Iterable[str], data_dir: Path) -> dict[str, Any]:
    want = set(ids)
    got = {t.session_id: t for t in qdata.iter_tasks(split, data_dir) if t.session_id in want}
    missing = want - set(got)
    if missing:
        raise ValueError(f"{len(missing)} {split} sessions missing from the export, e.g. {sorted(missing)[:3]}")
    return got


def session_items(tok, iface: Interface, split: str, ids: Sequence[str], labels: Mapping[str, np.ndarray],
                  data_dir: Path) -> list[Item]:
    """Items of ``ids`` in that order, each with all its training windows; sessions without one are left out."""
    rows, tasks, out = session_rows(labels), _tasks_by_id(split, ids, data_dir), []
    for sid in ids:
        task, r = tasks[sid], rows[sid]
        if list(task.window_pos) != r.row_pos.tolist():
            raise ValueError(f"{sid}: task windows differ from the label rows")
        m = r.train
        if m.any():
            out.append(Item.build(sid, iface.block_ids(tok, task.blocks), r.row_pos[m].tolist(), r.label[m].tolist()))
    return out


def window_items(tok, iface: Interface, windows: Sequence[B.Window], labels: Mapping[str, np.ndarray],
                 data_dir: Path) -> list[Item]:
    """One item per disclosed val window (200w), in key order."""
    rows, tasks = session_rows(labels), _tasks_by_id("val", [w.session_id for w in windows], data_dir)
    out = []
    for w in windows:
        r = rows[w.session_id]
        hit = np.flatnonzero(r.row_pos == w.row_pos)
        if hit.size != 1 or not r.train[hit[0]]:
            raise ValueError(f"{w.session_id}@{w.row_pos}: not one gate-passing scored window")
        out.append(Item.build(w.session_id, iface.block_ids(tok, tasks[w.session_id].blocks), [w.row_pos],
                              [int(r.label[hit[0]])]))
    return out


def build_inputs(tok, iface: Interface, budget: str, b: B.Budgets, data_dir: Path) -> TrainInputs:
    val_labels = qdata.load_labels("val", data_dir)
    prov = {"budgets_file_sha256": b.file_sha256, "budgets_sha256": dict(b.sha256),
            "val_labels_sha256": sha256_file(data_dir / "val.labels.npz"),
            "val_tasks_sha256": sha256_file(data_dir / "val.tasks.jsonl.gz")}
    B.check_disclosure(b, val_labels)  # both paths: the ES set is defined with the disclosed sessions removed (§5)
    if budget == "200w":
        return TrainInputs(window_items(tok, iface, b.disclosed_200w, val_labels, data_dir), None, b.size(budget), prov)
    dev_labels = qdata.load_labels("dev", data_dir)
    B.check_dev_ids(b, [str(s) for s in dev_labels["session_ids"]])
    B.check_val_ids(b, [str(s) for s in val_labels["session_ids"]])
    prov.update(dev_labels_sha256=sha256_file(data_dir / "dev.labels.npz"),
                dev_tasks_sha256=sha256_file(data_dir / "dev.tasks.jsonl.gz"))
    pool = session_items(tok, iface, "dev", b.sessions(budget), dev_labels, data_dir)
    es = session_items(tok, iface, "val", b.es_val, val_labels, data_dir)
    return TrainInputs(pool, es, b.size(budget), prov)


# ---------------------------------------------------------------------------
# Readout and CLI
# ---------------------------------------------------------------------------


def readout_tasks(model, tok, iface: Interface, tasks, *, progress=None) -> dict[str, Any]:
    """``run_readout``'s loop: the standard verbaliser over every window of ``tasks``."""
    verb = Verbaliser(model, iface)
    logits, lse, sids, row_pos, n_tok, t0 = [], [], [], [], 0, time.monotonic()
    for n, task in enumerate(tasks):
        r = verb.session(iface.block_ids(tok, task.blocks), task.window_pos)
        logits.append(r.label_logits.numpy()), lse.append(r.logsumexp.numpy())
        sids.append(task.session_id)
        row_pos.extend(r.pos)
        n_tok += r.tokens
        if progress is not None and (n + 1) % 50 == 0:
            progress(f"{n + 1} sessions, {n_tok} tokens, {n_tok / (time.monotonic() - t0):.0f} tok/s")
    return {"label_logits": np.concatenate(logits) if logits else np.zeros((0, N_CLASSES), np.float32),
            "logsumexp": np.concatenate(lse) if lse else np.zeros(0, np.float32), "session_ids": sids,
            "row_pos": np.array(row_pos, dtype=np.int32), "tokens": n_tok, "seconds": time.monotonic() - t0}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pinned_loader(device: str, dtype: str):
    from . import model as qmodel

    return qmodel.load(device, dtype)


def _frozen_iface(tok) -> Interface:
    from .interface import build_interface

    return build_interface(tok)


def cmd_train(a) -> int:
    if a.budget not in B.BUDGET_NAMES:
        raise SystemExit(f"--budget must be one of {B.BUDGET_NAMES}")
    if a.seed not in B.LORA_SEEDS[a.budget]:
        raise SystemExit(f"seed {a.seed} is not registered for {a.budget}: {B.LORA_SEEDS[a.budget]}")
    env = backend.configure(a.device, deterministic=a.deterministic)
    if a.threads:
        torch.set_num_threads(a.threads)
    b = B.load(a.budgets)
    from transformers import AutoTokenizer

    from . import model as qmodel

    tok = AutoTokenizer.from_pretrained(qmodel.verify_files())
    inputs = build_inputs(tok, _frozen_iface(tok), a.budget, b, a.data_dir)
    _log(f"{a.budget} seed {a.seed}: pool {len(inputs.pool)} sessions, cap {epoch_cap(inputs.budget_size)}, "
         f"ES {'none' if inputs.es is None else len(inputs.es)}")
    try:
        log = run_training(inputs, budget=a.budget, seed=a.seed, device=a.device, out=a.out, loader=_pinned_loader,
                           iface_of=_frozen_iface, environment=env, progress=_log)
    except Divergence as e:
        _log(f"FAILED: {e}")
        return 2
    _log(json.dumps({k: log[k] for k in ("budget", "seed", "epochs_run", "steps", "kept_epoch", "best_es_macro_f1",
                                         "stopped")}))
    return 0


def cmd_readout(a) -> int:
    env = backend.configure(a.device, deterministic=a.deterministic)
    tok, model, log, sha, match = load_merged(a.run, a.device, _pinned_loader)
    iface = _frozen_iface(tok)
    tasks = iter_split_tasks(a.split, tasks=a.tasks, data_dir=a.data_dir)
    if a.limit is not None:
        tasks = itertools.islice(tasks, a.limit)
    with backend.compute(a.device), torch.no_grad():
        arr = readout_tasks(model, tok, iface, tasks, progress=_log)
    meta = {"split": a.split, "limit": a.limit, "device": a.device, "dtype": "fp32", "environment": env,
            **model_meta(), "readout": "verbaliser", "arm": f"lora-{log['budget']}", "seed": log["seed"],
            "lora_run": {"train_log_sha256": sha256_file(a.run / LOG_FILE),
                         "adapter_file_sha256": log["adapter_file_sha256"], "merged_sha256": sha,
                         "merged_sha256_matches_training": match, "training_device": log["merged"]["device"]},
            "tasks_file": str(a.tasks) if a.tasks else None, "sessions": len(arr["session_ids"]),
            "tokens": arr["tokens"], "seconds": round(arr["seconds"], 1)}
    write_column(a.out, label_logits=arr["label_logits"], logsumexp=arr["logsumexp"], session_ids=arr["session_ids"],
                 row_pos=arr["row_pos"], meta=meta)
    _log(json.dumps({k: meta[k] for k in ("split", "arm", "seed", "sessions", "tokens", "seconds")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypaqwen.lora")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="one §5 run")
    t.add_argument("--budget", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--deterministic", action="store_true",
                   help="deterministic algorithms on MPS too (always on CUDA and CPU)")
    t.add_argument("--threads", type=int, default=None)
    t.add_argument("--budgets", type=Path, default=B.PATH)
    t.add_argument("--data-dir", type=Path, default=V04_DATA)
    r = sub.add_parser("readout", help="a column file from a finished run (merged, standard verbaliser)")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--split", required=True)
    r.add_argument("--tasks", type=Path, default=None, help="explicit task file (a split other than dev/val)")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--deterministic", action="store_true")
    r.add_argument("--data-dir", type=Path, default=V04_DATA)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"train": cmd_train, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
