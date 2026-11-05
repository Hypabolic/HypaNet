"""WSP author client (docs/V04_WSP_KERNEL.md §4), copied into the author dir as ``tools/hn_kernel/client.py``.

Imports only the standard library, numpy and torch (plus, optionally, ``tokenizers`` for ``tok``),
never ``transformers`` or ``hypaqwen``: it runs in the author's guarded ``./py`` interpreter. It
never loads a model and has no ``forward``; forward passes are the separate ``./fwd`` tool.

    from wsp import load, load_original, save_edit, save_label_rows, drop_edit, edited, edit_hash, budget, \\
        NAMES, SHAPES, ARCH, LABEL_IDS, CLASS_NAMES, tok

The module-level functions use one connection to the kernel socket, opened on first use:
``$HN_KERNEL_SOCKET``, else ``AUTHOR_DIR/.kernel/kernel.sock``. ``NAMES``, ``SHAPES``, ``ARCH``,
``LABEL_IDS`` and ``CLASS_NAMES`` are fetched from the server on first access. Tensor names are the
checkpoint's (``model.language_model.*``).

The weight-only helpers (``svd``, ``head_slice``, ``top_writers``, ``project_out``, ``rank1``,
``zero_head``, ``label_lens``, ``dla``, ``diff_means``, ``control_direction``) are plain torch. None
of them saves anything; pass their results to ``save_edit`` yourself.
"""

from __future__ import annotations

import atexit
import contextlib  # noqa: F401  (protocol's imports, loaded before protocol itself)
import json  # noqa: F401
import math  # noqa: F401
import os
import socket  # noqa: F401
import struct  # noqa: F401
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch


def _sibling(name: str):
    """``<this dir>/<name>.py`` loaded by explicit path, never through sys.path (shared via sys.modules).

    The ``./py`` wrapper imports this module before entering the guard (for ``init_tok``), so it
    must not make anything in an author-writable directory importable: it never touches sys.path.
    """
    import importlib.util

    key = f"hn_kernel_{name}"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return mod


if __package__:
    from . import protocol as _P
else:  # loaded by path as tools/hn_kernel/client.py
    _P = _sibling("protocol")

SOCKET_ENV = "HN_KERNEL_SOCKET"
TOKENIZER_ENV = "HN_TOKENIZER"
AUTHOR_ENV = "HN_AUTHOR_DIR"
# Set by the launcher; else the install layout AUTHOR_DIR/tools/hn_kernel/client.py.
AUTHOR_DIR = Path(os.environ.get(AUTHOR_ENV) or Path(__file__).resolve().parent.parent.parent).resolve()
PREFIX = "model.language_model."
EMBED = PREFIX + "embed_tokens.weight"
FINAL_NORM = PREFIX + "norm.weight"


class KernelError(RuntimeError):
    """The server refused or failed a request; the edit state and the meter are unchanged."""


class EditRefused(KernelError):
    pass


class BudgetRefused(KernelError):
    pass


class RequestError(KernelError):
    pass


class QuotaRefused(KernelError):
    """The phase's weight-load quota is used up (server §3): no more distinct layer tensors can be loaded."""


_ERRORS = {"EditRefused": EditRefused, "BudgetRefused": BudgetRefused, "RequestError": RequestError,
           "QuotaRefused": QuotaRefused}


class EditReceipt(NamedTuple):
    """What one save changed (server §3). Label-row receipts count over the ``[4, d]`` block."""

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


def _wire(t: Any, what: str) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if not isinstance(t, torch.Tensor):
        raise EditRefused(f"{what}: expected a torch.Tensor, not {type(t).__name__}")
    if t.dtype != torch.float32:
        raise EditRefused(f"{what}: dtype {t.dtype}, expected torch.float32 (edits are fp32; cast explicitly)")
    return t.detach().cpu().contiguous().numpy()


class KernelClient:
    """One connection to the kernel socket. The server serves one connection at a time."""

    def __init__(self, path: Path | str | None = None, timeout: float | None = None):
        self.path = Path(path or os.environ.get(SOCKET_ENV) or AUTHOR_DIR / ".kernel" / "kernel.sock")
        self._sock = _P.connect(self.path, timeout)

    def close(self) -> None:
        self._sock.close()

    def call(self, op: str, tensors: dict[str, np.ndarray] | None = None, **args) -> tuple[Any, dict[str, np.ndarray]]:
        _P.send_msg(self._sock, {"op": op, "args": args}, tensors)
        header, out = _P.recv_msg(self._sock)
        if not header.get("ok"):
            err = header.get("error") or {}
            raise _ERRORS.get(err.get("type"), KernelError)(err.get("message", "unknown error"))
        return header.get("result"), out

    def names(self) -> tuple[str, ...]:
        return tuple(self.call("names")[0])

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {k: tuple(v) for k, v in self.call("shapes")[0].items()}

    def arch(self) -> dict:
        return self.call("arch")[0]

    def load(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.call("load", name=name)[1]["tensor"])

    def load_original(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.call("load_original", name=name)[1]["tensor"])

    def label_rows(self, original: bool = False) -> torch.Tensor:
        return torch.from_numpy(self.call("load_label_rows", original=original)[1]["rows"])

    def save_edit(self, name: str, tensor: torch.Tensor) -> EditReceipt:
        return EditReceipt(**self.call("save_edit", {"tensor": _wire(tensor, name)}, name=name)[0])

    def save_label_rows(self, rows: torch.Tensor) -> EditReceipt:
        return EditReceipt(**self.call("save_label_rows", {"rows": _wire(rows, "label rows")})[0])

    def drop_edit(self, name: str) -> bool:
        return bool(self.call("drop_edit", name=name)[0]["dropped"])

    def edited(self) -> list[str]:
        return list(self.call("edited")[0])

    def edit_hash(self) -> str:
        return self.call("edit_hash")[0]

    def budget(self) -> dict:
        return self.call("budget")[0]

    def load_quota(self) -> dict:
        return self.call("load_quota")[0]


# ---------------------------------------------------------------- module-level API

_client: KernelClient | None = None
_cache: dict[str, Any] = {}


def connect(path: Path | str | None = None) -> KernelClient:
    """(Re)open the module's connection, e.g. to a socket other than the default.

    The server's metadata is fetched again; ``tok`` is kept: it comes from the packet, and the ``./py``
    launcher builds it before the guard, which refuses the tokenizer's file read afterwards.
    """
    global _client
    if _client is not None:
        _client.close()
    _client = KernelClient(path)
    for k in [k for k in _cache if k != "tok"]:
        del _cache[k]
    return _client


def _c() -> KernelClient:
    return _client if _client is not None else connect()


@atexit.register
def _close() -> None:
    if _client is not None:
        _client.close()


def init_tok(path: Path | str | None = None):
    """The packet's ``tokenizers.Tokenizer`` (``$HN_TOKENIZER``, else ``AUTHOR_DIR/packet/qwen/tokenizer.json``).

    None if ``tokenizers`` is not installed or the file is missing. The ``./py`` wrapper calls this
    before entering the guard, which forbids reading other files afterwards.
    """
    p = Path(path or os.environ.get(TOKENIZER_ENV) or AUTHOR_DIR / "packet" / "qwen" / "tokenizer.json")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(str(p)) if p.is_file() else None


_LAZY = {
    "NAMES": lambda: _c().names(),
    "SHAPES": lambda: _c().shapes(),
    "ARCH": lambda: _c().arch(),
    "LABEL_IDS": lambda: tuple(__getattr__("ARCH")["label_ids"]),
    "CLASS_NAMES": lambda: tuple(__getattr__("ARCH")["class_names"]),
    "tok": init_tok,
}


def __getattr__(name: str) -> Any:  # PEP 562: metadata is fetched on first access
    if name in _LAZY:
        if name not in _cache:
            _cache[name] = _LAZY[name]()
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load(name: str) -> torch.Tensor:
    """The current tensor: your edit if there is one, else the original (fp32, CPU)."""
    return _c().load(name)


def load_original(name: str) -> torch.Tensor:
    return _c().load_original(name)


def label_rows(original: bool = False) -> torch.Tensor:
    """The four label-word rows of the tied embedding, ``[4, d]`` in CLASS_NAMES order."""
    return _c().label_rows(original)


def save_edit(name: str, tensor: torch.Tensor) -> EditReceipt:
    """Replace a whole tensor. Refused outside the edit scope; equal to the original drops the edit."""
    return _c().save_edit(name, tensor)


def save_label_rows(rows: torch.Tensor) -> EditReceipt:
    return _c().save_label_rows(rows)


def drop_edit(name: str) -> bool:
    return _c().drop_edit(name)


def edited() -> list[str]:
    return _c().edited()


def edit_hash() -> str:
    return _c().edit_hash()


def budget() -> dict:
    """``{tokens_used, tokens_left, requests}`` of the phase's forward-pass budget."""
    return _c().budget()


def load_quota() -> dict:
    """``{bytes_used, bytes_left, bytes_quota, tensors}`` of the phase's weight-load quota.

    ``load``/``load_original`` of a layer tensor (anything but the embedding and the final norm)
    you have not loaded before in this phase draws on it; loading it again is free.
    """
    return _c().load_quota()


# ---------------------------------------------------------------- weight-only helpers


class HeadSlice(NamedTuple):
    """One head's block of a weight matrix: ``weight == load(name).narrow(dim, start, stop - start)``."""

    name: str
    dim: int
    start: int
    stop: int
    weight: torch.Tensor


def _f32(x: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).float()


def _2d(w: torch.Tensor) -> torch.Tensor:
    return w.reshape(w.shape[0], -1)


def svd(name: str, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-``k`` ``U [m, k]``, ``S [k]``, ``Vh [k, n]`` of the current tensor (as ``[shape[0], -1]``), in float64."""
    u, s, vh = torch.linalg.svd(_2d(load(name)).double(), full_matrices=False)
    return u[:, :k].float(), s[:k].float(), vh[:k].float()


def _head_table(layer: int, head: int) -> dict[str, tuple[str, int, int, int]]:
    a = __getattr__("ARCH")
    if a["layer_types"][layer] == "full_attention":
        nh, nkv, hd = a["num_attention_heads"], a["num_key_value_heads"], a["head_dim"]
        if not 0 <= head < nh:
            raise ValueError(f"layer {layer} has query heads 0..{nh - 1}")
        kv = head // (nh // nkv)  # GQA: query head -> its shared key/value head
        step = 2 * hd if a.get("attn_output_gate") else hd  # q_proj rows per head: [q (hd), gate (hd)]
        t = {"q": ("self_attn.q_proj.weight", 0, head * step, head * step + hd),
             "k": ("self_attn.k_proj.weight", 0, kv * hd, (kv + 1) * hd),
             "v": ("self_attn.v_proj.weight", 0, kv * hd, (kv + 1) * hd),
             "o": ("self_attn.o_proj.weight", 1, head * hd, (head + 1) * hd)}
        if a.get("attn_output_gate"):
            t["gate"] = ("self_attn.q_proj.weight", 0, head * step + hd, (head + 1) * step)
        return t
    nk, nv = a["linear_num_key_heads"], a["linear_num_value_heads"]
    dk, dv = a["linear_key_head_dim"], a["linear_value_head_dim"]
    if not 0 <= head < nv:
        raise ValueError(f"layer {layer} has value heads 0..{nv - 1}")
    kh, kd = head // (nv // nk), nk * dk  # value head -> its key head; in_proj_qkv rows are [q | k | v]
    qkv = "linear_attn.in_proj_qkv.weight"
    return {"q": (qkv, 0, kh * dk, (kh + 1) * dk),
            "k": (qkv, 0, kd + kh * dk, kd + (kh + 1) * dk),
            "v": (qkv, 0, 2 * kd + head * dv, 2 * kd + (head + 1) * dv),
            "z": ("linear_attn.in_proj_z.weight", 0, head * dv, (head + 1) * dv),
            "b": ("linear_attn.in_proj_b.weight", 0, head, head + 1),
            "a": ("linear_attn.in_proj_a.weight", 0, head, head + 1),
            "o": ("linear_attn.out_proj.weight", 1, head * dv, (head + 1) * dv)}


def head_slice(layer: int, head: int, kind: str, W: torch.Tensor | None = None) -> HeadSlice:
    """One head's rows (``q``/``k``/``v``/``gate``/``z``/``a``/``b``) or output columns (``o``).

    Full-attention layers: ``head`` is a query head (``k``/``v`` give its shared KV head). Gated
    DeltaNet layers: ``head`` is a value head (``q``/``k`` give its key head). ``W`` defaults to
    ``load(name)``; pass a tensor you are editing to slice that instead.
    """
    table = _head_table(layer, head)
    if kind not in table:
        raise ValueError(f"kind must be one of {sorted(table)} for layer {layer}")
    rest, dim, start, stop = table[kind]
    name = f"{PREFIX}layers.{layer}.{rest}"
    W = load(name) if W is None else W
    return HeadSlice(name, dim, start, stop, W.narrow(dim, start, stop - start).clone())


def zero_head(layer: int, head: int) -> tuple[str, torch.Tensor]:
    """``(name, W')``: the layer's output projection with this head's input columns zeroed."""
    s = head_slice(layer, head, "o")
    W = load(s.name)
    W.narrow(1, s.start, s.stop - s.start).zero_()
    return s.name, W


def top_writers(name: str, direction: Any, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` units whose weight vectors most align with ``direction``: ``(indices, signed scores)``, by |score|.

    If ``direction`` lives in the output space (``shape[0]``: ``o_proj``, ``down_proj``, ``out_proj``)
    the units are columns (what each input unit writes); otherwise in the input space (``shape[1]``,
    e.g. embedding rows) they are rows.
    """
    W, d = _2d(load(name)), _f32(direction).reshape(-1)
    if W.shape[0] == d.numel():
        scores = W.T @ d
    elif W.shape[1] == d.numel():
        scores = W @ d
    else:
        raise ValueError(f"{name} {tuple(W.shape)} has no dimension of size {d.numel()}")
    idx = scores.abs().topk(min(k, scores.numel())).indices
    return idx, scores[idx]


def project_out(W: torch.Tensor, u: Any, cols: Sequence[int] | None = None, alpha: float = 1.0) -> torch.Tensor:
    """``W - alpha * u (mask * (u^T W))`` with ``u`` normalised to unit length (``u`` in the row space).

    ``cols`` limits the projection to those columns; ``alpha=1`` removes the ``u`` component exactly.
    """
    W = _f32(W)
    u = _f32(u).reshape(-1)
    u = u / torch.linalg.vector_norm(u)
    coef = u @ _2d(W)
    if cols is not None:
        mask = torch.zeros_like(coef)
        mask[list(cols)] = 1.0
        coef = coef * mask
    return W - alpha * torch.outer(u, coef).reshape(W.shape)


def rank1(W: torch.Tensor, a: Any, b: Any) -> torch.Tensor:
    """``W + a b^T``."""
    W = _f32(W)
    return W + torch.outer(_f32(a).reshape(-1), _f32(b).reshape(-1)).reshape(W.shape)


def _final_norm() -> tuple[torch.Tensor, float]:
    if "_final_norm" not in _cache:
        _cache["_final_norm"] = (load(FINAL_NORM), float(__getattr__("ARCH")["rms_norm_eps"]))
    return _cache["_final_norm"]


def label_lens(h: Any, original: bool = False) -> torch.Tensor:
    """The four label logits a residual ``h [..., d]`` would give: final ``(1 + w)`` RMSNorm, then the label rows.

    For ``./fwd`` hidden states (``hidden.L<n>.npy``); at the last layer this reproduces the readout.
    """
    w, eps = _final_norm()
    h = _f32(h)
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)
    return x @ label_rows(original).T


def dla(vec: Any, resid: Any = None, original: bool = False) -> torch.Tensor:
    """Direct label attribution of a residual contribution ``vec [..., d]``: ``(vec * (1 + w)) / rms(resid) @ rows^T``.

    ``resid`` is the full final residual the contribution sits in (its RMS freezes the norm's scale);
    without it the RMS factor is 1.
    """
    w, eps = _final_norm()
    x = _f32(vec) * (1.0 + w)
    if resid is not None:
        r = _f32(resid)
        x = x * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
    return x @ label_rows(original).T


def diff_means(a: Any, b: Any) -> torch.Tensor:
    """``mean(a, 0) - mean(b, 0)``."""
    return _f32(a).mean(0) - _f32(b).mean(0)


# Where the guard's control_normal lives: the install layout (tools/hn_kernel/ next to tools/hn_guard/),
# then the source layout (author/client.py next to author/hn_guard/).
_GUARD_KEYS = ("hn_guard.guards_v04", "hypaqwen.wsp.author.hn_guard.guards_v04", "hn_kernel_guards_v04")


def _control_normal():
    """The guard's ``control_normal`` (the only RNG either guard allows; ``torch.Generator`` is refused).

    Reuses the guard module already loaded (the ``./py`` wrapper installs it before the script runs);
    else loads ``hn_guard/guards_v04.py`` by explicit path, never through sys.path. Loading it does not
    install anything.
    """
    for key in _GUARD_KEYS:
        mod = sys.modules.get(key)
        if mod is not None and hasattr(mod, "control_normal"):
            return mod.control_normal
    if __package__:
        from .hn_guard.guards_v04 import control_normal

        return control_normal
    import importlib.util

    here = Path(__file__).resolve().parent
    path = here.parent / "hn_guard" / "guards_v04.py"
    if not path.is_file():
        path = here / "hn_guard" / "guards_v04.py"
    key = _GUARD_KEYS[-1]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[key]
        raise
    return mod.control_normal


def control_direction(seed: int, like: Any) -> torch.Tensor:
    """A seeded Gaussian tensor with ``like``'s shape and Frobenius norm: the only randomness allowed.

    Built on the guard's ``control_normal(seed, shape)`` (SHAKE-256, bit-identical everywhere), so it
    runs under KERNEL_CLIENT and BUILD_EDIT alike.
    """
    like = _f32(like)
    v = _control_normal()(int(seed), tuple(like.shape)).to(torch.float32)
    return v * (torch.linalg.vector_norm(like) / torch.linalg.vector_norm(v))


__all__ = [
    "load", "load_original", "label_rows", "save_edit", "save_label_rows", "drop_edit", "edited", "edit_hash",
    "budget", "load_quota", "connect", "init_tok", "KernelClient", "KernelError", "EditRefused", "BudgetRefused",
    "RequestError", "QuotaRefused",
    "EditReceipt", "HeadSlice", "svd", "head_slice", "top_writers", "project_out", "rank1", "zero_head",
    "label_lens", "dla", "diff_means", "control_direction",
    "NAMES", "SHAPES", "ARCH", "LABEL_IDS", "CLASS_NAMES", "tok",
]
