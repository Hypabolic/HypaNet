"""The WSP edit-state hash (docs/V04_WSP_KERNEL.md §3), in one place for the server and the build runner.

The model server stamps every ledger line and journal entry with this hash; the build runner
records it in ``BUILD_RECORD.json`` and checks it against the journal, so an archived edit can be
linked to the ledger lines that forward-tested it.

* **Canonical form.** ``{name: fp32 tensor}`` of the edited tensors only. The embedding
  (``model.language_model.embed_tokens.weight``) is held as its four label-word rows, ``[4, d]`` in
  ``LABEL_IDS`` order, exactly as ``save_label_rows`` stores it (:func:`canonical_edit`).
* **Tensor hash.** sha256 of the little-endian fp32 C-order bytes (:func:`tensor_sha256`).
* **State hash.** ``sha256(EDIT_HASH_DOMAIN + concat over sorted names of name utf-8 + b"\\n" +
  tensor digest (32 raw bytes))``; the empty state hashes to :data:`EMPTY_EDIT_HASH`
  (:func:`edit_state_hash`).

The §2 **rescaling refusal** (:func:`rescaling`) lives here too: the model server applies it to
every ``save_edit`` and the build runner to every returned tensor, so an edit the server accepted
is never refused at build time, or the reverse.
"""

from __future__ import annotations

import hashlib
import re
from typing import Mapping, Sequence

import numpy as np
import torch

__all__ = ["EDIT_HASH_DOMAIN", "EMPTY_EDIT_HASH", "EMBED", "HASH_RULE", "SCALE_C_TOL", "SCALE_RESID_TAU",
           "RESCALING_RULE", "ZERO_CENTRED_NORM_RE", "tensor_sha256", "edit_state_hash", "canonical_edit",
           "canonical_edit_hash", "scalar_multiple", "label_rows_rescaled", "is_zero_centred_norm", "rescaling"]

EDIT_HASH_DOMAIN = b"hn-wsp-edit-state/1\n"
EMPTY_EDIT_HASH = hashlib.sha256(EDIT_HASH_DOMAIN).hexdigest()
EMBED = "model.language_model.embed_tokens.weight"
SCALE_C_TOL = 1e-3  # |c - 1| at or below this is never treated as a rescaling
SCALE_RESID_TAU = 0.5  # new = c W + R is a rescaling when ||R|| < tau * ||(c - 1) W||
# Qwen3_5RMSNorm weights below the final norm: the layer computes x * (1 + w), so 1 + w is the gain.
ZERO_CENTRED_NORM_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.(input_layernorm|post_attention_layernorm|self_attn\.q_norm|"
    r"self_attn\.k_norm)\.weight$")
RESCALING_RULE = (f"with new = c W + R, c = <new, W>/<W, W>: refused when |c - 1| > {SCALE_C_TOL} and "
                  f"||R|| < {SCALE_RESID_TAU} ||(c - 1) W||; for the zero-centred (1 + w) norms also on the gain "
                  "1 + w; for the label rows also on the row-centred block and on the changed rows (2 or more)")
HASH_RULE = ("sha256(b'hn-wsp-edit-state/1\\n' + for each name in sorted order: name utf-8 + b'\\n' + "
             "sha256 digest of the little-endian fp32 C-order bytes); the embedding as its [4, d] label rows "
             "in LABEL_IDS order (the model server's edit hash, hypaqwen.wsp.edit_state)")


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of the little-endian fp32 C-order bytes of ``t`` (which must be float32)."""
    if t.dtype != torch.float32:
        raise TypeError(f"edit tensors are float32, not {t.dtype}")
    a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
    return hashlib.sha256(memoryview(a.reshape(-1)).cast("B")).hexdigest()


def edit_state_hash(tensor_sha256s: Mapping[str, str]) -> str:
    """The state hash from each canonical tensor's sha256 (hex)."""
    h = hashlib.sha256(EDIT_HASH_DOMAIN)
    for name in sorted(tensor_sha256s):
        h.update(name.encode("utf-8") + b"\n" + bytes.fromhex(tensor_sha256s[name]))
    return h.hexdigest()


def canonical_edit(edits: Mapping[str, torch.Tensor], label_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """The canonical form: a full-size embedding is cut to its label rows; a ``[4, d]`` one is kept."""
    out: dict[str, torch.Tensor] = {}
    for name, t in edits.items():
        if name == EMBED and t.dim() == 2 and t.shape[0] != len(label_ids):
            t = t[list(label_ids)]
        out[name] = t
    return out


def canonical_edit_hash(edits: Mapping[str, torch.Tensor], label_ids: Sequence[int]) -> str:
    return edit_state_hash({k: tensor_sha256(v) for k, v in canonical_edit(edits, label_ids).items()})


def scalar_multiple(new: torch.Tensor, orig: torch.Tensor) -> float | None:
    """``c`` if ``new`` is dominated by a rescaling of ``orig``, else None (§2).

    ``new = c * orig + R`` with ``c = <new, orig> / <orig, orig>``. It is a rescaling when
    ``|c - 1| > SCALE_C_TOL`` and ``||R|| < SCALE_RESID_TAU * ||(c - 1) * orig||``: an exact multiple,
    a multiple plus a small perturbation, or a multiple with a few entries left alone (``c = 0``
    is zeroing the tensor). Edits whose change is mostly not along ``orig`` pass.
    """
    n = new.double().reshape(-1)
    w = orig.double().reshape(-1)
    ww = torch.dot(w, w)
    if ww == 0:
        return None
    c = (torch.dot(n, w) / ww).item()
    if abs(c - 1.0) <= SCALE_C_TOL:
        return None
    resid = torch.linalg.vector_norm(n - c * w).item()
    if resid < SCALE_RESID_TAU * abs(c - 1.0) * ww.sqrt().item():
        return c
    return None


def label_rows_rescaled(new: torch.Tensor, orig: torch.Tensor) -> float | None:
    """``c`` if the label rows are a common rescaling of the originals, with or without a shared offset.

    The four-way label softmax depends only on the row-centred block, so ``c * rows + 1 b^T`` is
    the same temperature knob as ``c * rows``; both tests run. A common rescaling of two or more of
    the rows (the others left alone) is refused too. Rescaling a single row is an edit to that
    class's readout direction and passes.
    """
    c = scalar_multiple(new, orig)
    if c is None:
        c = scalar_multiple(new - new.mean(0, keepdim=True), orig - orig.mean(0, keepdim=True))
    changed = (new != orig).any(1)
    if c is None and int(changed.sum()) >= 2:
        c = scalar_multiple(new[changed], orig[changed])
    return c


def is_zero_centred_norm(name: str) -> bool:
    return ZERO_CENTRED_NORM_RE.match(name) is not None


def rescaling(name: str, new: torch.Tensor, orig: torch.Tensor) -> str | None:
    """Why ``new`` is refused as a rescaling of ``orig`` (§2), or None. For :data:`EMBED`, ``new`` and
    ``orig`` are the ``[4, d]`` label rows."""
    if name == EMBED:
        c = label_rows_rescaled(new, orig)
        return None if c is None else f"the label rows are a common rescaling, a scalar multiple ({c:.6g} x the original)"
    c = scalar_multiple(new, orig)
    if c is not None:
        return f"the edit is dominated by a whole-tensor scalar multiple ({c:.6g} x the original)"
    if is_zero_centred_norm(name):
        c = scalar_multiple(new.double() + 1.0, orig.double() + 1.0)
        if c is not None:
            return f"the edit rescales the norm's effective gain 1 + w ({c:.6g} x)"
    return None
