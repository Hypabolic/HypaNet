"""WSP model server (docs/V04_WSP_KERNEL.md §3): weights, edit state, token meter, edit journal, ledger.

Harness-owned and outside the author sandbox. It never executes author code. Clients reach it over
two Unix domain sockets in ``AUTHOR_DIR/.kernel/``:

* ``kernel.sock`` for ``./py`` scripts: static metadata, ``load``/``load_original``, ``save_edit``,
  ``save_label_rows``, ``drop_edit``, ``edited``, ``edit_hash``, ``budget``. No forward pass.
* ``fwd.sock`` for the ``./fwd`` tool only: ``forward`` (and, in WSP-B, ``forward_disclosed``),
  ``budget``, ``edit_hash``. The client guard permits ``./py`` only the kernel socket, so a script
  cannot run the model (§1 principle 2) even by writing frames by hand.

One connection is served at a time. A malformed frame, or any failure while serving one
connection, drops that connection only; the server keeps serving.

Names are the checkpoint's (``model.language_model.*``, 320 tensors). Edits are fp32 and are copied
into the model's own parameters in place (``param.data.copy_``), so the tied embedding and lm_head
stay one tensor and ``hypaqwen.readout.Verbaliser`` reads edited label rows live. Pristine copies
are kept (on CPU) for edited tensors only; ``original`` and ``both`` forwards swap them in and
restore the edits afterwards, even on error.

The edit state is a function of the effective weights: saving a tensor equal to its original
removes that tensor's edit, so the edit hash never depends on history. The embedding's edit is its
four label rows (``[4, d]``), stored and hashed as such.

Persistence (all outside the author dir; a continuation reloads both):

* ``LOG_DIR/<phase>/journal/``: ``journal.jsonl`` plus one safetensors file per save, written after
  each save or drop and replayed (with sha256 and edit-hash checks) on start.
* ``LOG_DIR/<phase>.ledger.jsonl``: one line per forward request, ok or refused. It is also the
  meter's record: tokens used and requests are re-read from it on start. Each line records, per
  session, the sha256 of exactly what ran (the token ids of the blocks and the windows), the
  windows, a short hash per block (for near-duplicate audits) and, when windows are disclosed, the
  session's overlap with them.
* ``LOG_DIR/<phase>.kernel.jsonl``: one line per kernel op (load, save, drop, metadata), with the
  connection id, tensor name and bytes. It is also the weight-load quota's record.

Rules added on top of the contract's text (docs/V04_WSP_KERNEL.md; pre-registered here, to be
folded into the contract at G1):

* §2 scalar multiples: an edit is refused as a scalar multiple when it is dominated by a rescaling
  of the original, not only when it is exactly ``c * W``: with ``new = c W + R``,
  ``c = <new, W> / <W, W>``, it is refused when ``|c - 1| > SCALE_C_TOL`` and
  ``||R|| < SCALE_RESID_TAU * ||(c - 1) W||`` (``c = 0`` covers zeroing a tensor). For the label
  rows the same test also runs on the row-centred block and on the changed rows when two or more
  changed, which refuses a common rescaling of the rows with or without a shared offset.
* §3 forward: a session whose text overlaps a WSP-B disclosed window (an identical window prefix,
  or a share of the window's blocks or 16-character shingles, overall or in its last four blocks,
  of at least DISCLOSED_OVERLAP_REFUSE) is refused, so ``forward`` cannot return the activations
  ``forward_disclosed`` withholds. Blocks after a session's last window are refused (they never run).
* §3 forward_disclosed: at most DISCLOSED_MAX_EDIT_HASHES distinct non-empty edit states may be
  evaluated on the disclosed windows per phase (``original`` is not limited).
* §3 kernel ops: ``load``/``load_original`` of layer tensors (everything except the embedding and
  the final norm) draw on a per-phase quota of distinct tensor bytes, LOAD_QUOTA_FRACTION of the
  layer weights by default. Reloading a tensor is free; the quota exists so that a ``./py`` script
  (or a series of them caching tensors under ``output/``) cannot assemble the whole model and run
  its own unmetered forward passes. A per-run quota would not do this: tensors can be cached to
  ``output/<phase>/`` and re-read by the next run.

CLI: ``python -m hypaqwen.wsp.server --phase P --author-dir D --log-dir L --device mps --budget 2000000``
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import select
import signal
import sys
import threading
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .. import MODEL_ID, MODEL_REVISION
from ..interface import CLASS_NAMES, Interface, build_interface
from ..readout import Verbaliser
from . import edit_state as ES
from . import protocol as P
from .edit_state import (EDIT_HASH_DOMAIN, EMPTY_EDIT_HASH, SCALE_C_TOL, SCALE_RESID_TAU,  # noqa: F401 (re-exported)
                         label_rows_rescaled, scalar_multiple, tensor_sha256)

PUBLIC_PREFIX = "model.language_model."  # checkpoint names, as the author sees them
INTERNAL_PREFIX = "model."  # Qwen3_5ForCausalLM parameter names
EMBED = PUBLIC_PREFIX + "embed_tokens.weight"
FINAL_NORM = PUBLIC_PREFIX + "norm.weight"
REFUSED_PREFIXES = ("mtp.", "model.visual.")

PHASE_BUDGET = 2_000_000  # tokens per phase (§3)
REQUEST_CAP = 50_000  # tokens per request
RANK_RTOL = 1e-6  # delta_rank counts singular values > RANK_RTOL * ||dW||_2
# Disclosed-window overlap (WSP-B). Calibrated on 200 random val windows against 400 other val
# sessions: the largest unrelated 16-gram containment was 0.24 (whole window) and 0.34 (last four
# blocks), and no unrelated session shared a single whole block. A pasted or lightly edited window
# scores near 1.
SHINGLE_CHARS = 16
TAIL_BLOCKS = 4
DISCLOSED_OVERLAP_REFUSE = 0.6
DISCLOSED_MAX_EDIT_HASHES = 2  # WSP-B sees the disclosed windows "once": the WSP-A edit and one revision
LOAD_QUOTA_FRACTION = 0.5  # of the layer tensors' bytes, distinct tensors, per phase


KERNEL_OPS = frozenset({"names", "shapes", "arch", "load", "load_original", "load_label_rows", "save_edit",
                        "save_label_rows", "drop_edit", "edited", "edit_hash", "budget", "load_quota"})
FWD_OPS = frozenset({"forward", "forward_disclosed", "budget", "edit_hash"})
ROLE_OPS = {"kernel": KERNEL_OPS, "fwd": FWD_OPS}

RequestError = P.RequestError


class EditRefused(ValueError):
    """A save outside the edit scope (§2). The edit state is unchanged."""


class BudgetRefused(RuntimeError):
    """A forward request over the per-request cap or the phase budget. Nothing ran or was charged."""


class QuotaRefused(RuntimeError):
    """A weight load over the phase's distinct-tensor quota. Nothing was sent."""


@dataclass(frozen=True)
class EditReceipt:
    """What one save changed, relative to the original tensor (§3).

    For the label rows (``label_rows=True``) every count and norm is over the ``[4, d]`` block.
    ``bf16_changed_entries`` counts entries whose bf16 rounding changes (§1.4); ``edit_hash`` is the
    edit-state hash after the save.
    """

    name: str
    changed_entries: int
    changed_rows: int
    changed_cols: int
    rel_frobenius: float
    delta_rank: int
    sha256: str
    label_rows: bool
    bf16_changed_entries: int
    edit_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DisclosedWindow:
    """One WSP-B disclosed window: the session's blocks and the block index that ends the window."""

    blocks: tuple[str, ...]
    pos: int


@dataclass
class ForwardResponse:
    result: dict  # JSON-able: logits, ids, deltas, call id, budget
    arrays: dict[str, torch.Tensor]  # requested activations, "<model>.s<i>.<hidden|mlp_act>.L<layer>" -> [n_windows, d]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def session_sha256(block_ids: Sequence[Sequence[int]], windows: Sequence[int]) -> str:
    """sha256 of exactly what a session runs: its blocks' token ids (as the Verbaliser sees them) and windows."""
    payload = {"blocks": [list(map(int, b)) for b in block_ids], "windows": list(map(int, windows))}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def block_sha16(ids: Sequence[int]) -> str:
    """A short per-block hash of the token ids, for near-duplicate audits in the ledger."""
    return hashlib.sha256(json.dumps(list(map(int, ids)), separators=(",", ":")).encode()).hexdigest()[:16]


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()


def _shingles(text: str) -> frozenset[int]:
    """Hashed SHINGLE_CHARS-character shingles of normalised text (hashes are per-process; never stored)."""
    k = SHINGLE_CHARS
    return frozenset(hash(text[i:i + k]) for i in range(max(1, len(text) - k + 1)))


def _containment(d: frozenset, r: frozenset) -> float:
    return len(d & r) / len(d) if d else 0.0


@dataclass(frozen=True)
class _DisclosedIndex:
    """What ``forward`` compares request texts against: one entry per disclosed window."""

    prefix: dict[str, str]  # sha256 of the normalised window prefix -> window id
    blocks: dict[str, frozenset[str]]  # window id -> normalised block texts
    whole: dict[str, frozenset[int]]  # window id -> shingles of the whole window
    tail: dict[str, frozenset[int]]  # window id -> shingles of its last TAIL_BLOCKS blocks

    @staticmethod
    def _prefix_key(blocks: Sequence[str]) -> str:
        return hashlib.sha256("\x1e".join(blocks).encode()).hexdigest()

    @classmethod
    def build(cls, disclosed: Mapping[str, "DisclosedWindow"]) -> "_DisclosedIndex":
        prefix, blocks, whole, tail = {}, {}, {}, {}
        for wid, w in disclosed.items():
            nb = [_norm_text(b) for b in w.blocks[: w.pos + 1]]
            prefix[cls._prefix_key(nb)] = wid
            blocks[wid] = frozenset(nb)
            whole[wid] = _shingles(" ".join(nb))
            tail[wid] = _shingles(" ".join(nb[-TAIL_BLOCKS:]))
        return cls(prefix, blocks, whole, tail)

    def overlap(self, blocks: Sequence[str], windows: Sequence[int]) -> dict:
        """The session's largest overlap with any disclosed window: ``{window, score, exact, ...}``."""
        nb = [_norm_text(b) for b in blocks]
        exact = next((self.prefix[k] for k in (self._prefix_key(nb[: w + 1]) for w in windows) if k in self.prefix),
                     None)
        mine_blocks, mine = frozenset(nb), _shingles(" ".join(nb))
        best = {"window": None, "score": 0.0, "blocks": 0.0, "whole": 0.0, "tail": 0.0}
        for wid in self.blocks:
            parts = {"blocks": len(self.blocks[wid] & mine_blocks) / len(self.blocks[wid]),
                     "whole": _containment(self.whole[wid], mine), "tail": _containment(self.tail[wid], mine)}
            score = max(parts.values())
            if score > best["score"]:
                best = {"window": wid, "score": score, **parts}
        if exact is not None:
            best = {**best, "window": exact, "score": 1.0}
        rounded = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}
        return {**rounded, "exact": exact is not None}


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)  # 1-D -> [n, 1]; conv1d [C, 1, K] -> [C, K]


def change_stats(new: torch.Tensor, orig: torch.Tensor) -> dict:
    """Entry/row/column counts, relative Frobenius norm, rank and bf16 count of ``new - orig`` (CPU fp32)."""
    changed = _as_2d(new != orig)
    d = new.double() - orig.double()  # exact for fp32 operands
    d_norm = torch.linalg.vector_norm(d).item()
    w_norm = torch.linalg.vector_norm(orig.double()).item()
    rank = 0
    if d_norm > 0:
        s = torch.linalg.svdvals(_as_2d(d))
        rank = int((s > RANK_RTOL * s[0]).sum())
    return {
        "changed_entries": int(changed.sum()),
        "changed_rows": int(changed.any(1).sum()),
        "changed_cols": int(changed.any(0).sum()),
        "rel_frobenius": d_norm / w_norm if w_norm > 0 else (float("inf") if d_norm > 0 else 0.0),
        "delta_rank": rank,
        "bf16_changed_entries": int((new.bfloat16() != orig.bfloat16()).sum()),
    }


class _Capture(Verbaliser):
    """The Verbaliser, plus full-vocabulary argmax/top-k ids, MLP activations and the pre-norm final residual.

    ``mlp_act[L]`` is the input of layer L's ``down_proj`` (``act_fn(gate) * up``) and ``hidden[n_layers]``
    the input of the final norm, both at the readout position. Hooks record the last position of every
    model call; ``_read`` (called right after each suffix run) keeps the values of that call.
    """

    def __init__(self, model, iface: Interface, *, hidden: Sequence[int], mlp_act: Sequence[int], topk: int,
                 n_layers: int):
        super().__init__(model, iface, hidden_layers=[layer for layer in hidden if layer < n_layers])
        self.final = n_layers in hidden
        self.n_layers = n_layers
        self.mlp_layers = tuple(mlp_act)
        self.topk = topk
        self._last: dict[Any, torch.Tensor] = {}
        self._reset()

    def _reset(self) -> None:
        self.argmax: list[int] = []
        self.top: list[list[int]] = []
        self.mlp: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.mlp_layers}
        self.resid: list[torch.Tensor] = []

    def _keeper(self, key):
        def hook(_module, args):
            self._last[key] = args[0][0, -1].detach().float().clone()
        return hook

    @contextlib.contextmanager
    def hooks(self) -> Iterator[None]:
        inner = self.model.model
        handles = [inner.layers[layer].mlp.down_proj.register_forward_pre_hook(self._keeper(("mlp", layer)))
                   for layer in self.mlp_layers]
        if self.final:
            handles.append(inner.norm.register_forward_pre_hook(self._keeper("final")))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _read(self, out, logits4, lse, hid) -> None:
        super()._read(out, logits4, lse, hid)
        emb = self.model.get_input_embeddings().weight
        full = (emb @ out.last_hidden_state[0, -1]).float()
        self.argmax.append(int(full.argmax()))
        if self.topk:
            self.top.append(full.topk(self.topk).indices.tolist())
        for layer in self.mlp_layers:
            self.mlp[layer].append(self._last[("mlp", layer)])
        if self.final:
            self.resid.append(self._last["final"])

    def run(self, block_ids: Sequence[Sequence[int]], windows: Sequence[int]) -> dict:
        self._reset()
        r = self.session(block_ids, windows)
        hidden = dict(r.hidden)
        if self.final:
            hidden[self.n_layers] = torch.stack(self.resid).cpu()
        return {
            "label_logits": r.label_logits, "logsumexp": r.logsumexp, "argmax": list(self.argmax),
            "topk": list(self.top), "hidden": hidden,
            "mlp_act": {layer: torch.stack(v).cpu() for layer, v in self.mlp.items()}, "tokens": r.tokens,
        }


class ModelServer:
    """In-process server state and ops. ``serve`` exposes it over the two sockets."""

    def __init__(self, model, tok, *, phase: str, log_dir: Path | str, budget: int = PHASE_BUDGET,
                 request_cap: int = REQUEST_CAP, disclosed: Mapping[str, DisclosedWindow] | None = None,
                 iface: Interface | None = None, disclosed_max_edits: int = DISCLOSED_MAX_EDIT_HASHES,
                 load_quota_bytes: int | None = None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", phase):
            raise ValueError(f"bad phase name {phase!r}")
        self.model, self.tok, self.phase = model, tok, phase
        self.iface = iface or build_interface(tok)
        self.n_layers = int(model.config.num_hidden_layers)
        self.budget_tokens, self.request_cap = int(budget), int(request_cap)
        self.disclosed = dict(disclosed) if disclosed is not None else None  # None: forward_disclosed disabled
        self._disclosed_index = _DisclosedIndex.build(self.disclosed) if self.disclosed else None
        self.disclosed_max_edits = int(disclosed_max_edits)
        log_dir = Path(log_dir)
        self.journal_dir = log_dir / phase / "journal"
        self.ledger_path = log_dir / f"{phase}.ledger.jsonl"
        self.kernel_ledger_path = log_dir / f"{phase}.kernel.jsonl"

        self._params: dict[str, torch.nn.Parameter] = {}
        for n, p in model.named_parameters():  # tied lm_head.weight is not listed separately
            if not n.startswith(INTERNAL_PREFIX) or p.dtype != torch.float32:
                raise ValueError(f"unexpected parameter {n} ({p.dtype}); load the model with dtype fp32")
            self._params[PUBLIC_PREFIX + n[len(INTERNAL_PREFIX):]] = p
        self.names = tuple(self._params)
        self._quota_bytes = {n: p.numel() * 4 for n, p in self._params.items() if n not in (EMBED, FINAL_NORM)}
        self.load_quota_bytes = int(load_quota_bytes) if load_quota_bytes is not None \
            else int(LOAD_QUOTA_FRACTION * sum(self._quota_bytes.values()))
        self._max_frame_bytes = max(p.numel() for p in self._params.values()) * 4 + (1 << 20)
        self._conn_seq = 0
        self._label_index = torch.tensor(list(self.iface.label_ids), dtype=torch.long)
        self._edits: dict[str, torch.Tensor] = {}  # CPU fp32; EMBED -> its [4, d] label rows
        self._edit_sha: dict[str, str] = {}
        self._pristine: dict[str, torch.Tensor] = {}
        self._lock = threading.RLock()
        self._journal_seq = 0
        self._disclosed_hashes: set[str] = set()  # non-empty edit states run on the disclosed windows
        self._tokens_used, self._requests, self._calls = self._replay_ledger()
        self._loaded: set[str] = set()  # quota-counted tensors sent to a client this phase
        self._kernel_seq = self._replay_kernel_ledger()
        self._replay_journal()

    # ------------------------------------------------------------ static metadata

    def shapes(self) -> dict[str, list[int]]:
        return {n: list(p.shape) for n, p in self._params.items()}

    def arch(self) -> dict:
        cfg = self.model.config
        keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size", "layer_types",
                "num_attention_heads", "num_key_value_heads", "head_dim", "attn_output_gate",
                "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                "linear_conv_kernel_dim", "full_attention_interval", "rms_norm_eps", "tie_word_embeddings",
                "hidden_act")
        return {
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "float32",
            **{k: getattr(cfg, k) for k in keys if hasattr(cfg, k)},
            "label_ids": list(self.iface.label_ids), "class_names": list(CLASS_NAMES),
            "embed_tensor": EMBED, "final_norm_tensor": FINAL_NORM,
            "final_norm": "zero-centred RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)",
            "hidden_layers": f"hidden[L], L in 0..{self.n_layers}: the residual stream after L decoder blocks "
                             f"(0 = token embeddings, {self.n_layers} = before the final norm)",
            "mlp_act_layers": f"mlp_act[L], L in 0..{self.n_layers - 1}: act_fn(gate_proj x) * up_proj x, "
                              "the input of down_proj",
            "edit_scope": f"{PUBLIC_PREFIX}* except {FINAL_NORM} (refused) and {EMBED} (label rows only)",
            "load_quota_bytes": self.load_quota_bytes,
            "load_quota": "load/load_original of distinct tensors other than the embedding and the final norm, "
                          "per phase; reloading a tensor is free",
        }

    # ------------------------------------------------------------ reading weights

    def _known(self, name: Any) -> str:
        if not isinstance(name, str) or name not in self._params:
            raise RequestError(f"unknown tensor {name!r} (mtp.* and model.visual.* are not loaded)")
        return name

    def load(self, name: str) -> torch.Tensor:
        """The current tensor: the edit, else the original (a CPU fp32 copy)."""
        with self._lock:
            return self._params[self._known(name)].detach().to("cpu", copy=True)

    def load_original(self, name: str) -> torch.Tensor:
        with self._lock:
            name = self._known(name)
            t = self._params[name].detach().to("cpu", copy=True)
            if name in self._pristine:
                if name == EMBED:
                    t[self._label_index] = self._pristine[name]
                else:
                    t = self._pristine[name].clone()
            return t

    def load_label_rows(self, original: bool = False) -> torch.Tensor:
        """The four label rows of the tied embedding, ``[4, d]`` in CLASS_NAMES order."""
        with self._lock:
            if original and EMBED in self._pristine:
                return self._pristine[EMBED].clone()
            return self._params[EMBED].detach()[self._label_index.to(self._params[EMBED].device)].cpu()

    def edited(self) -> list[str]:
        return sorted(self._edits)

    def edit_hash(self) -> str:
        return self._state_hash(self._edit_sha)

    @staticmethod
    def _state_hash(shas: Mapping[str, str]) -> str:
        return ES.edit_state_hash(shas)

    # ------------------------------------------------------------ edits (§2)

    def _check_scope(self, name: Any) -> str:
        if not isinstance(name, str):
            raise EditRefused(f"tensor name must be a string, not {type(name).__name__}")
        if name.startswith(REFUSED_PREFIXES):
            raise EditRefused(f"{name}: mtp.* and model.visual.* are outside the edit scope")
        if name == FINAL_NORM:
            raise EditRefused(f"{name}: the final norm is refused (scaling it is a global temperature knob)")
        if name not in self._params:
            raise EditRefused(f"{name}: not a {PUBLIC_PREFIX}* tensor")
        return name

    @staticmethod
    def _check_tensor(what: str, t: Any, shape: Sequence[int]) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise EditRefused(f"{what}: expected a torch.Tensor, not {type(t).__name__}")
        if t.dtype != torch.float32:
            raise EditRefused(f"{what}: dtype {t.dtype}, expected torch.float32")
        if tuple(t.shape) != tuple(shape):
            raise EditRefused(f"{what}: shape {tuple(t.shape)}, expected {tuple(shape)}")
        t = t.detach().to("cpu", copy=True).contiguous()
        if not bool(torch.isfinite(t).all()):
            raise EditRefused(f"{what}: non-finite values")
        return t

    def _original_of(self, name: str) -> torch.Tensor:
        """The original of an edit-state entry (the label rows for EMBED)."""
        if name in self._pristine:
            return self._pristine[name]
        if name == EMBED:
            return self.load_label_rows()
        return self.load(name)

    def _write_param(self, name: str, t: torch.Tensor) -> None:
        p = self._params[name]
        with torch.no_grad():
            if name == EMBED:
                p.data.index_copy_(0, self._label_index.to(p.device), t.to(p.device))
            else:
                p.data.copy_(t)

    def _apply(self, name: str, new: torch.Tensor, *, journal: bool) -> torch.Tensor:
        """Set ``name``'s edit to ``new`` (validated, CPU); returns the original. Equal-to-original drops it."""
        orig = self._original_of(name)
        if torch.equal(new, orig):
            if name in self._edits:
                self._drop(name, journal=journal)
            return orig
        why = ES.rescaling(name, new, orig)
        if why is not None:
            raise EditRefused(f"{name}: {why}; |c - 1| > {SCALE_C_TOL} and the rest is under {SCALE_RESID_TAU} of "
                              "the rescaling")
        sha = tensor_sha256(new)
        if journal:
            self._journal("save", name, self._state_hash({**self._edit_sha, name: sha}), tensor=new, sha=sha)
        self._pristine.setdefault(name, orig.clone())
        self._edits[name], self._edit_sha[name] = new, sha
        self._write_param(name, new)
        return orig

    def _drop(self, name: str, *, journal: bool) -> None:
        if journal:
            self._journal("drop", name, self._state_hash({k: v for k, v in self._edit_sha.items() if k != name}))
        self._write_param(name, self._pristine[name])
        del self._edits[name], self._edit_sha[name], self._pristine[name]

    def _receipt(self, name: str, new: torch.Tensor, orig: torch.Tensor) -> EditReceipt:
        return EditReceipt(name=name, **change_stats(new, orig), sha256=tensor_sha256(new),
                           label_rows=name == EMBED, edit_hash=self.edit_hash())

    def save_edit(self, name: str, tensor: torch.Tensor) -> EditReceipt:
        """Validate (§2) and apply a whole-tensor edit. The embedding accepts changes to label rows only."""
        with self._lock:
            name = self._check_scope(name)
            t = self._check_tensor(name, tensor, self._params[name].shape)
            if name == EMBED:
                cur = self._params[EMBED].detach().cpu()
                rows = (t != cur).any(1)
                rows[self._label_index] = False
                if bool(rows.any()):
                    raise EditRefused(f"{EMBED}: {int(rows.sum())} rows other than the 4 label rows differ "
                                      "(only the label rows may be edited, through save_label_rows)")
                return self.save_label_rows(t[self._label_index])
            orig = self._apply(name, t, journal=True)
            return self._receipt(name, t, orig)

    def save_label_rows(self, rows: torch.Tensor) -> EditReceipt:
        """Write the four label-word rows of the tied embedding (``[4, d]``, CLASS_NAMES order)."""
        with self._lock:
            t = self._check_tensor(f"{EMBED} label rows", rows, (len(self.iface.label_ids), self._params[EMBED].shape[1]))
            orig = self._apply(EMBED, t, journal=True)
            return self._receipt(EMBED, t, orig)

    def drop_edit(self, name: str) -> bool:
        """Remove one edit (restoring the original); False if the tensor was not edited."""
        with self._lock:
            name = self._known(name)
            if name not in self._edits:
                return False
            self._drop(name, journal=True)
            return True

    def release(self) -> None:
        """Write the original weights back into the model without touching the journal (shutdown, tests)."""
        with self._lock:
            for name in list(self._edits):
                self._write_param(name, self._pristine[name])
            self._edits.clear()
            self._edit_sha.clear()
            self._pristine.clear()

    @contextlib.contextmanager
    def _weights(self, kind: str) -> Iterator[None]:
        """Run with the edited weights (as stored) or temporarily with the pristine ones."""
        if kind == "edited" or not self._edits:
            yield
            return
        try:
            for name in self._edits:
                self._write_param(name, self._pristine[name])
            yield
        finally:
            for name, t in self._edits.items():
                self._write_param(name, t)

    # ------------------------------------------------------------ journal

    def _journal(self, op: str, name: str, edit_hash: str, *, tensor: torch.Tensor | None = None,
                 sha: str | None = None) -> None:
        from safetensors.torch import save_file

        self.journal_dir.mkdir(parents=True, exist_ok=True)
        seq = self._journal_seq + 1
        entry = {"seq": seq, "utc": _utc(), "op": op, "name": name, "edit_hash": edit_hash}
        if tensor is not None:
            fname = f"{seq:06d}.safetensors"
            tmp = self.journal_dir / (fname + ".tmp")
            save_file({"tensor": tensor}, str(tmp))
            os.replace(tmp, self.journal_dir / fname)
            entry.update(file=fname, sha256=sha)
        with open(self.journal_dir / "journal.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._journal_seq = seq

    def _replay_journal(self) -> None:
        from safetensors.torch import load_file

        path = self.journal_dir / "journal.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["seq"] != self._journal_seq + 1:
                raise ValueError(f"{path}: journal sequence broken at {e['seq']}")
            name = e["name"]
            if e["op"] == "save":
                t = load_file(str(self.journal_dir / e["file"]))["tensor"]
                if tensor_sha256(t) != e["sha256"]:
                    raise ValueError(f"{path}: {e['file']} does not match its sha256")
                shape = (len(self.iface.label_ids), self._params[EMBED].shape[1]) if name == EMBED \
                    else self._params[self._check_scope(name)].shape
                self._apply(name, self._check_tensor(name, t, shape), journal=False)
            elif e["op"] == "drop":
                self._drop(name, journal=False)
            else:
                raise ValueError(f"{path}: unknown journal op {e['op']!r}")
            self._journal_seq = e["seq"]
            if self.edit_hash() != e["edit_hash"]:
                raise ValueError(f"{path}: replay diverged at seq {e['seq']}")

    # ------------------------------------------------------------ meter and ledger

    def budget(self) -> dict:
        return {"tokens_used": self._tokens_used, "tokens_left": max(0, self.budget_tokens - self._tokens_used),
                "requests": self._requests}

    def _replay_ledger(self) -> tuple[int, int, int]:
        used = requests = calls = 0
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                calls += 1
                if e["call_id"] != calls:
                    raise ValueError(f"{self.ledger_path}: call ids broken at line {calls}")
                if e["status"] == "ok":
                    used += e["tokens"]
                    requests += 1
                    if e.get("op") == "forward_disclosed" and e.get("model") in ("edited", "both") \
                            and e["edit_hash"] != EMPTY_EDIT_HASH:
                        self._disclosed_hashes.add(e["edit_hash"])
                if e["cumulative_tokens"] != used:
                    raise ValueError(f"{self.ledger_path}: cumulative tokens inconsistent at call {calls}")
        return used, requests, calls

    def _ledger(self, entry: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._calls = entry["call_id"]

    def _replay_kernel_ledger(self) -> int:
        seq = 0
        if self.kernel_ledger_path.exists():
            for line in self.kernel_ledger_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                seq += 1
                if e["seq"] != seq:
                    raise ValueError(f"{self.kernel_ledger_path}: sequence broken at line {seq}")
                if e["status"] == "ok" and e.get("quota"):
                    self._loaded.add(e["name"])
        return seq

    def _kernel_log(self, entry: dict) -> None:
        self.kernel_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"seq": self._kernel_seq + 1, "utc": _utc(), **entry}
        with open(self.kernel_ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._kernel_seq = entry["seq"]

    def load_quota(self) -> dict:
        used = sum(self._quota_bytes[n] for n in self._loaded)
        return {"bytes_used": used, "bytes_left": max(0, self.load_quota_bytes - used),
                "bytes_quota": self.load_quota_bytes, "tensors": len(self._loaded)}

    def client_load(self, name: Any, *, original: bool = False) -> tuple[torch.Tensor, bool]:
        """``load``/``load_original`` for a client: draws on the phase quota. Returns ``(tensor, counted)``."""
        with self._lock:
            name = self._known(name)
            counted = name in self._quota_bytes
            if counted and name not in self._loaded:
                left = self.load_quota()["bytes_left"]
                if self._quota_bytes[name] > left:
                    raise QuotaRefused(f"{name} ({self._quota_bytes[name]} bytes) is over the phase's weight-load "
                                       f"quota: {left} of {self.load_quota_bytes} bytes of distinct layer tensors "
                                       "left. Tensors already loaded can be loaded again.")
            t = self.load_original(name) if original else self.load(name)
            if counted:
                self._loaded.add(name)
            return t, counted

    def _check_cost(self, cost: int) -> None:
        if cost > self.request_cap:
            raise BudgetRefused(f"request costs {cost} tokens, over the per-request cap of {self.request_cap}")
        if self._tokens_used + cost > self.budget_tokens:
            raise BudgetRefused(f"request costs {cost} tokens; {self.budget_tokens - self._tokens_used} of the "
                                f"phase budget of {self.budget_tokens} are left")

    def session_cost(self, block_ids: Sequence[Sequence[int]], windows: Sequence[int]) -> int:
        """Tokens the Verbaliser runs: prefix, blocks through the last window, one suffix per window."""
        return (len(self.iface.prefix_ids) + sum(len(b) for b in block_ids[: windows[-1] + 1])
                + len(self.iface.suffix_ids) * len(windows))

    def _metered(self, op: str, kinds: Sequence[str], sessions: list[tuple[str, list, list[int]]],
                 ret: dict, info: dict) -> dict[str, list[dict]]:
        """Check the cost, run every session under each weight kind, charge on completion, write the ledger."""
        call_id = self._calls + 1
        base = {"call_id": call_id, "op": op, "edit_hash": self.edit_hash(), **info,
                **self._session_record(sessions),
                "layers": {"hidden": ret["hidden"], "mlp_act": ret["mlp_act"]}, "topk": ret["topk"]}
        cost = len(kinds) * sum(self.session_cost(b, w) for _, b, w in sessions)
        try:
            self._check_cost(cost)
            runs = {}
            for kind in kinds:
                with self._weights(kind):
                    cap = _Capture(self.model, self.iface, n_layers=self.n_layers, **ret)
                    with cap.hooks():
                        runs[kind] = [cap.run(b, w) for _, b, w in sessions]
        except Exception as e:
            status = "refused" if isinstance(e, (BudgetRefused, RequestError)) else "error"
            self._ledger({**base, "utc": _utc(), "status": status, "tokens": 0,
                          "cumulative_tokens": self._tokens_used, "error": f"{type(e).__name__}: {e}"})
            raise
        tokens = sum(r["tokens"] for rs in runs.values() for r in rs)
        if tokens != cost:  # the run happened, so charge what ran; the mismatch is a server bug, logged for audit
            base["estimate"] = cost
        self._tokens_used += tokens
        self._requests += 1
        self._ledger({**base, "utc": _utc(), "status": "ok", "tokens": tokens,
                      "cumulative_tokens": self._tokens_used})
        return runs

    @staticmethod
    def _session_record(sessions: Sequence[tuple[str, list, list[int]]]) -> dict:
        """Ledger fields for what each session runs: its hash (token ids + windows), windows, per-block hashes."""
        return {"session_sha256": [session_sha256(b, w) for _, b, w in sessions],
                "windows": [list(w) for _, _, w in sessions],
                "block_sha256": [[block_sha16(x) for x in b] for _, b, _ in sessions]}

    def _refuse(self, op: str, err: Exception, info: dict) -> None:
        self._ledger({"call_id": self._calls + 1, "op": op, "utc": _utc(), "status": "refused", "tokens": 0,
                      "cumulative_tokens": self._tokens_used, "edit_hash": self.edit_hash(), **info,
                      "error": f"{type(err).__name__}: {err}"})

    # ------------------------------------------------------------ forward (./fwd only)

    def forward(self, request: Any, *, request_path: str | None = None) -> ForwardResponse:
        """Run a ``./fwd`` request (schema: ``protocol.validate_forward_request``), metered and ledgered.

        With disclosed windows (WSP-B), a session overlapping one of them is refused (see the module
        docstring), and every session's overlap is ledgered as ``disclosed_overlap``.
        """
        with self._lock:
            info: dict[str, Any] = {"request_path": request_path}
            try:
                req = P.validate_forward_request(request, self.n_layers)
            except RequestError as e:
                self._refuse("forward", e, info)
                raise
            model, ret = req["model"], req["return"]
            info["model"] = model
            sessions = [(s["id"], self.iface.block_ids(self.tok, s["blocks"]), s["windows"]) for s in req["sessions"]]
            if self._disclosed_index is not None:
                overlaps = [self._disclosed_index.overlap(s["blocks"], s["windows"]) for s in req["sessions"]]
                info["disclosed_overlap"] = overlaps
                hits = [(i, o) for i, o in enumerate(overlaps) if o["exact"] or o["score"] >= DISCLOSED_OVERLAP_REFUSE]
                if hits:
                    i, o = hits[0]
                    why = "an identical window prefix" if o["exact"] else \
                        f"overlap {o['score']:.2f} >= {DISCLOSED_OVERLAP_REFUSE}"
                    err = RequestError(
                        f"sessions[{i}] ({req['sessions'][i]['id']!r}) overlaps disclosed window {o['window']} "
                        f"({why}); disclosed windows can only be read through forward_disclosed (label logits only)")
                    self._refuse("forward", err, {**info, **self._session_record(sessions)})
                    raise err
            kinds = ("edited", "original") if model == "both" else (model,)
            runs = self._metered("forward", kinds, sessions, ret, info)
            result = self._result_head(model, runs)
            arrays: dict[str, torch.Tensor] = {}
            for i, (sid, _, windows) in enumerate(sessions):
                s: dict[str, Any] = {"id": sid, "windows": windows}
                for kind in kinds:
                    r = runs[kind][i]
                    s[kind] = {"label_logits": r["label_logits"].tolist(), "logsumexp": r["logsumexp"].tolist(),
                               "label_argmax": r["label_logits"].argmax(1).tolist(), "argmax": r["argmax"]}
                    if ret["topk"]:
                        s[kind]["topk"] = r["topk"]
                    for what in ("hidden", "mlp_act"):
                        for layer, a in r[what].items():
                            arrays[f"{kind}.s{i}.{what}.L{layer}"] = a
                if model == "both":
                    e, o = runs["edited"][i], runs["original"][i]
                    s["delta"] = {
                        "label_logits": (e["label_logits"] - o["label_logits"]).tolist(),
                        "logsumexp": (e["logsumexp"] - o["logsumexp"]).tolist(),
                        "label_argmax_flips": int((e["label_logits"].argmax(1) != o["label_logits"].argmax(1)).sum()),
                        "argmax_flips": sum(a != b for a, b in zip(e["argmax"], o["argmax"], strict=True)),
                    }
                result["sessions"].append(s)
            return ForwardResponse(result, arrays)

    def forward_disclosed(self, window_ids: Any, model: Any) -> ForwardResponse:
        """WSP-B: label logits and label argmax only on the disclosed windows (no activations or top-k).

        At most ``disclosed_max_edits`` distinct non-empty edit states can be run here per phase
        (``model="original"`` is not limited); the count is re-read from the ledger on start.
        """
        with self._lock:
            info: dict[str, Any] = {"window_ids": window_ids if isinstance(window_ids, list) else None}
            try:
                if self.disclosed is None:
                    raise RequestError("forward_disclosed is not enabled in this phase")
                req = P.validate_disclosed_request({"disclosed_windows": window_ids, "model": model})
                unknown = [w for w in req["disclosed_windows"] if w not in self.disclosed]
                if unknown:
                    raise RequestError(f"unknown disclosed window ids {unknown[:5]}")
                info["model"] = req["model"]
                h = self.edit_hash()
                new_state = req["model"] != "original" and h != EMPTY_EDIT_HASH and h not in self._disclosed_hashes
                info["disclosed_edit_hashes"] = len(self._disclosed_hashes) + int(new_state)
                if new_state and len(self._disclosed_hashes) >= self.disclosed_max_edits:
                    raise BudgetRefused(
                        f"forward_disclosed has already run {len(self._disclosed_hashes)} distinct edit states, the "
                        f"limit for this phase ({self.disclosed_max_edits}); this edit state ({h[:12]}) is not one "
                        "of them")
            except (RequestError, BudgetRefused) as e:
                self._refuse("forward_disclosed", e, info)
                raise
            model = req["model"]
            sessions = []
            for wid in req["disclosed_windows"]:
                w = self.disclosed[wid]
                sessions.append((wid, self.iface.block_ids(self.tok, list(w.blocks[: w.pos + 1])), [w.pos]))
            kinds = ("edited", "original") if model == "both" else (model,)
            runs = self._metered("forward_disclosed", kinds, sessions, {"hidden": [], "mlp_act": [], "topk": 0}, info)
            if new_state:
                self._disclosed_hashes.add(h)
            result = self._result_head(model, runs)
            for i, (wid, *_rest) in enumerate(sessions):
                s: dict[str, Any] = {"id": wid}
                for kind in kinds:
                    ll = runs[kind][i]["label_logits"][0]
                    s[kind] = {"label_logits": ll.tolist(), "label_argmax": int(ll.argmax())}
                result["sessions"].append(s)
            return ForwardResponse(result, {})

    def _result_head(self, model: str, runs: Mapping[str, list[dict]]) -> dict:
        return {"call_id": self._calls, "phase": self.phase, "model": model, "edit_hash": self.edit_hash(),
                "tokens": sum(r["tokens"] for rs in runs.values() for r in rs), "budget": self.budget(),
                "class_names": list(CLASS_NAMES), "sessions": []}

    # ------------------------------------------------------------ wire

    def handle(self, header: dict, tensors: Mapping[str, np.ndarray], role: str,
               conn: int | None = None) -> tuple[dict, dict[str, np.ndarray]]:
        """One request frame -> one response frame. Errors come back as typed error headers.

        Every op other than the forward ops (which have the forward ledger) gets a kernel-ledger
        line: connection id, role, op, tensor name, bytes in and out, status.
        """
        op = header.get("op")
        args = header.get("args") or {}
        entry: dict[str, Any] = {"conn": conn, "role": role, "op": op if isinstance(op, str) else repr(op)[:64]}
        if isinstance(args, dict) and isinstance(args.get("name"), str):
            entry["name"] = args["name"][:256]
        entry["bytes_in"] = int(sum(t.nbytes for t in tensors.values()))
        out: dict[str, torch.Tensor] = {}
        try:
            if not isinstance(op, str) or op not in ROLE_OPS[role]:
                raise RequestError(f"op {op!r} is not available on the {role} endpoint")
            if not isinstance(args, dict):
                raise RequestError("args must be an object")
            result, out = self._dispatch(op, args, tensors, entry)
            response = {"ok": True, "result": result}
            entry["status"] = "ok"
        except (EditRefused, BudgetRefused, RequestError, QuotaRefused) as e:
            response = P.error_header(type(e).__name__, str(e))
            entry.update(status="refused", error=f"{type(e).__name__}: {e}"[:500])
            out = {}
        except Exception as e:  # keep serving; the client sees the failure
            response = P.error_header("InternalError", f"{type(e).__name__}: {e}")
            entry.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
            out = {}
        arrays = {k: v.detach().cpu().numpy() for k, v in out.items()}
        if role == "kernel" or op not in ("forward", "forward_disclosed"):  # refused forward attempts too
            entry["bytes_out"] = int(sum(a.nbytes for a in arrays.values()))
            self._kernel_log(entry)
        return response, arrays

    def _tensor_arg(self, tensors: Mapping[str, np.ndarray], key: str) -> torch.Tensor:
        if key not in tensors:
            raise RequestError(f"missing tensor {key!r}")
        return torch.from_numpy(tensors[key])

    def _dispatch(self, op: str, args: dict, tensors: Mapping[str, np.ndarray],
                  note: dict | None = None) -> tuple[Any, dict[str, torch.Tensor]]:
        note = {} if note is None else note
        if op == "names":
            return list(self.names), {}
        if op == "shapes":
            return self.shapes(), {}
        if op == "arch":
            return self.arch(), {}
        if op in ("load", "load_original"):
            t, counted = self.client_load(args.get("name"), original=op == "load_original")
            note.update(quota=counted, quota_used=self.load_quota()["bytes_used"])
            return None, {"tensor": t}
        if op == "load_label_rows":
            return None, {"rows": self.load_label_rows(bool(args.get("original", False)))}
        if op == "save_edit":
            return self.save_edit(args.get("name"), self._tensor_arg(tensors, "tensor")).to_dict(), {}
        if op == "save_label_rows":
            return self.save_label_rows(self._tensor_arg(tensors, "rows")).to_dict(), {}
        if op == "drop_edit":
            return {"dropped": self.drop_edit(args.get("name")), "edit_hash": self.edit_hash()}, {}
        if op == "edited":
            return self.edited(), {}
        if op == "edit_hash":
            return self.edit_hash(), {}
        if op == "budget":
            return self.budget(), {}
        if op == "load_quota":
            return self.load_quota(), {}
        if op == "forward":
            path = args.get("request_path")
            if path is not None and (not isinstance(path, str) or len(path) > 4096):
                raise RequestError("request_path must be a string")
            r = self.forward(args.get("request"), request_path=path)
            return r.result, r.arrays
        if op == "forward_disclosed":
            r = self.forward_disclosed(args.get("window_ids"), args.get("model"))
            return r.result, r.arrays
        raise RequestError(f"unknown op {op!r}")

    def _serve_connection(self, conn, role: str, conn_id: int | None = None) -> None:
        while True:
            try:
                header, tensors = P.recv_msg(conn, max_total_bytes=self._max_frame_bytes)
            except EOFError:
                return
            except (P.ProtocolError, OSError):
                return  # a broken frame: drop the connection
            out_header, out = self.handle(header, tensors, role, conn_id)
            P.send_msg(conn, out_header, out)

    def serve(self, kernel_path: Path | str, fwd_path: Path | str, *, stop: threading.Event | None = None,
              on_ready: Callable[[], None] | None = None, poll: float = 0.2) -> None:
        """Serve both endpoints, one connection at a time, until ``stop`` is set (checked between connections).

        Whatever goes wrong inside one connection (a hostile frame, a client that vanishes, a bug)
        ends that connection only; it is reported on stderr and the server keeps serving.
        """
        listeners = {}
        try:
            for path, role in ((kernel_path, "kernel"), (fwd_path, "fwd")):
                listeners[P.listen(path)] = (role, Path(path))
            if on_ready:
                on_ready()
            while not (stop and stop.is_set()):
                ready, _, _ = select.select(list(listeners), [], [], poll)
                if not ready:
                    continue
                try:
                    conn, _ = ready[0].accept()
                except OSError:
                    continue
                self._conn_seq += 1
                role = listeners[ready[0]][0]
                with conn:
                    try:
                        self._serve_connection(conn, role, self._conn_seq)
                    except Exception as e:  # noqa: BLE001  a connection never takes the server down
                        print(f"wsp server: dropped {role} connection {self._conn_seq}: {type(e).__name__}: {e}",
                              file=sys.stderr, flush=True)
        finally:
            for s, (_, path) in listeners.items():
                s.close()
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()


def read_disclosed(path: Path) -> dict[str, DisclosedWindow]:
    """``{"windows": [{"id", "blocks", "pos"}, ...]}`` (the read-only disclosure copy for WSP-B)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for w in d["windows"]:
        if w["id"] in out or not 0 <= w["pos"] < len(w["blocks"]):
            raise ValueError(f"{path}: bad disclosed window {w['id']!r}")
        out[w["id"]] = DisclosedWindow(tuple(w["blocks"]), int(w["pos"]))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypaqwen.wsp.server", description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=True)
    ap.add_argument("--author-dir", type=Path, required=True, help="the sockets go in AUTHOR_DIR/.kernel/")
    ap.add_argument("--log-dir", type=Path, required=True, help="journal and ledger (outside the author dir)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--budget", type=int, default=PHASE_BUDGET)
    ap.add_argument("--request-cap", type=int, default=REQUEST_CAP)
    ap.add_argument("--disclosed", type=Path, help="WSP-B: the disclosure copy; enables forward_disclosed")
    ap.add_argument("--disclosed-max-edits", type=int, default=DISCLOSED_MAX_EDIT_HASHES,
                    help="distinct edit states forward_disclosed may run per phase")
    ap.add_argument("--load-quota-bytes", type=int, default=None,
                    help="distinct layer-tensor bytes clients may load per phase "
                         f"(default {LOAD_QUOTA_FRACTION} of them)")
    args = ap.parse_args(argv)
    author, logs = args.author_dir.resolve(), args.log_dir.resolve()
    if logs == author or author in logs.parents:
        ap.error("--log-dir must be outside the author dir")

    from .. import model as qmodel

    tok, model = qmodel.load(args.device, "fp32")
    server = ModelServer(model, tok, phase=args.phase, log_dir=logs, budget=args.budget,
                         request_cap=args.request_cap,
                         disclosed=read_disclosed(args.disclosed) if args.disclosed else None,
                         disclosed_max_edits=args.disclosed_max_edits, load_quota_bytes=args.load_quota_bytes)
    kernel_dir = author / ".kernel"
    kernel_dir.mkdir(mode=0o700, exist_ok=True)

    def _exit(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _exit)

    def ready() -> None:
        print(json.dumps({"ready": True, "kernel": str(kernel_dir / "kernel.sock"), "fwd": str(kernel_dir / "fwd.sock"),
                          "phase": args.phase, "edit_hash": server.edit_hash(), "budget": server.budget()}),
              flush=True)

    try:
        server.serve(kernel_dir / "kernel.sock", kernel_dir / "fwd.sock", on_ready=ready)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
