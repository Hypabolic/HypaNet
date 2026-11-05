"""Verbaliser readout (plan v0.4, Task interface, items 3-5).

One pass per session. After each block that ends a window, the cache (full-attention KV and the
Gated DeltaNet conv and recurrent states) is snapshotted, the suffix is run on top of it, the four
label-word logits are read at the suffix's last token, and the cache is restored. The label after
block ``t`` therefore never sees block ``t + 1``, and the result equals a separate prompt per window
(:func:`reference_window`) up to floating-point reassociation in the chunked DeltaNet kernel.

``transformers``' own ``Cache.crop`` cannot do this: it rolls back conv states but not the DeltaNet
recurrent state, so the snapshot is taken here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from .interface import Interface


@dataclass
class WindowReadout:
    pos: list[int]
    label_logits: torch.Tensor  # [n_windows, 4], float32, CLASS_NAMES order
    logsumexp: torch.Tensor  # [n_windows], full-vocabulary log-normaliser at the readout position
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [n_windows, d] at the readout position
    tokens: int = 0  # tokens run through the model for this session (prefix + blocks + every suffix)


def _snapshot(cache) -> list:
    snap = []
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            snap.append(("lin", layer.conv_states[0].clone(), layer.recurrent_states[0].clone()))
        else:
            snap.append(("kv", layer.get_seq_length()))
    return snap


def _restore(cache, snap: list) -> None:
    for layer, s in zip(cache.layers, snap, strict=True):
        if s[0] == "lin":
            layer.conv_states[0].copy_(s[1])
            layer.recurrent_states[0].copy_(s[2])
        else:
            n = s[1]
            layer.keys = layer.keys[..., :n, :]
            layer.values = layer.values[..., :n, :]


class Verbaliser:
    def __init__(self, model, iface: Interface, *, hidden_layers: Sequence[int] = ()):
        self.model = model
        self.iface = iface
        self.hidden_layers = tuple(hidden_layers)
        self.device = next(model.parameters()).device
        self._label_index = torch.tensor(list(iface.label_ids), dtype=torch.long, device=self.device)
        self._check_tied()

    def _check_tied(self) -> torch.Tensor:
        emb = self.model.get_input_embeddings().weight
        if self.model.get_output_embeddings().weight.data_ptr() != emb.data_ptr():
            raise ValueError("expected tied input and output embeddings")
        return emb

    def _run(self, ids: Sequence[int], cache, *, hidden: bool = False):
        x = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        return self.model.model(input_ids=x, past_key_values=cache, use_cache=True,
                                output_hidden_states=hidden and bool(self.hidden_layers))

    def _read(self, out, logits4, lse, hid) -> None:
        # Read the (tied) embedding at read time, so an edit to the label-word rows reaches the readout.
        emb = self._check_tied()  # [V, d]
        h = out.last_hidden_state[0, -1]  # final norm applied
        logits4.append((emb.index_select(0, self._label_index) @ h).float())
        lse.append(torch.logsumexp((emb @ h).float(), dim=0))
        for layer in self.hidden_layers:
            hid[layer].append(out.hidden_states[layer][0, -1].float())

    @torch.no_grad()
    def session(self, block_ids: Sequence[Sequence[int]], window_pos: Sequence[int]) -> WindowReadout:
        want = set(window_pos)
        if any(p < 0 or p >= len(block_ids) for p in want):
            raise ValueError("window position outside the session")
        logits4: list[torch.Tensor] = []
        lse: list[torch.Tensor] = []
        hid: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.hidden_layers}
        cache = None
        pending = list(self.iface.prefix_ids)
        n_tok = 0
        for k, ids in enumerate(block_ids):
            pending.extend(ids)
            if k not in want:
                continue  # consecutive blocks with no window are run as one chunk
            out = self._run(pending, cache)
            cache = out.past_key_values
            n_tok += len(pending)
            pending = []
            snap = _snapshot(cache)
            out = self._run(self.iface.suffix_ids, cache, hidden=True)
            n_tok += len(self.iface.suffix_ids)
            self._read(out, logits4, lse, hid)
            _restore(cache, snap)
        return WindowReadout(
            pos=sorted(want),
            label_logits=torch.stack(logits4).cpu() if logits4 else torch.empty(0, 4),
            logsumexp=torch.stack(lse).cpu() if lse else torch.empty(0),
            hidden={layer: torch.stack(v).cpu() for layer, v in hid.items() if v},
            tokens=n_tok,
        )

    @torch.no_grad()
    def reference_window(self, block_ids: Sequence[Sequence[int]], t: int) -> WindowReadout:
        """The window ending at block ``t`` as its own prompt, in one forward pass and a fresh cache."""
        ids = self.iface.window_prompt(block_ids, t)
        logits4: list[torch.Tensor] = []
        lse: list[torch.Tensor] = []
        hid: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.hidden_layers}
        out = self._run(ids, None, hidden=True)
        self._read(out, logits4, lse, hid)
        return WindowReadout([t], torch.stack(logits4).cpu(), torch.stack(lse).cpu(),
                             {layer: torch.stack(v).cpu() for layer, v in hid.items()}, len(ids))
