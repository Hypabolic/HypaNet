"""Run the v0.3 transformer over encoded sessions (harness side, not packet).

An *encoded session* is anything with the attributes of the tokeniser's
``Encoded`` record: ``tokens``, ``event_ord``, ``offset``, ``eoe_index`` (and
``surface``, unused here). The prediction for event ``t`` (0-based block index)
is the argmax at ``eoe_index[t]`` (spec v0.3 §2.2, §6).
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

import torch

from .model import DoomLoopTransformer, predict


class EncodedLike(Protocol):
    tokens: Sequence[int]
    event_ord: Sequence[int]
    offset: Sequence[int]
    eoe_index: Sequence[int]


def _tensors(enc: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.as_tensor(list(enc.tokens), dtype=torch.long)
    event_ord = torch.as_tensor(list(enc.event_ord), dtype=torch.long)
    offset = torch.as_tensor(list(enc.offset), dtype=torch.long)
    eoe = torch.as_tensor(list(enc.eoe_index), dtype=torch.long)
    n = tokens.shape[0]
    if eoe.numel() and (int(eoe.min()) < 0 or int(eoe.max()) >= n):
        raise ValueError(f"eoe_index outside [0, {n})")
    return tokens, event_ord, offset, eoe


def eoe_logits(model: DoomLoopTransformer, enc: Any) -> torch.Tensor:
    """Logits ``[num_events, 4]`` read at each event's ``EOE`` token."""
    tokens, event_ord, offset, eoe = _tensors(enc)
    with torch.no_grad():
        logits = model(tokens, event_ord, offset)
    return logits[eoe]


def eoe_predictions(model: DoomLoopTransformer, enc: Any) -> list[int]:
    """Predicted class index per event."""
    return predict(eoe_logits(model, enc)).tolist()


def run_sessions(model: DoomLoopTransformer, encs: Iterable[Any], batch_size: int = 1) -> list[torch.Tensor]:
    """``eoe_logits`` for many sessions, in input order.

    ``batch_size == 1`` (the default) runs one causal forward per session,
    which is the reference path. Larger batches pad sessions and call
    ``forward_batch``; results agree to float32 rounding, not bit for bit.
    """
    encs = list(encs)
    if batch_size <= 1:
        return [eoe_logits(model, e) for e in encs]
    out: list[torch.Tensor] = []
    for start in range(0, len(encs), batch_size):
        chunk = [_tensors(e) for e in encs[start : start + batch_size]]
        n = max(t[0].shape[0] for t in chunk)
        bsz = len(chunk)
        tok = torch.zeros(bsz, n, dtype=torch.long)
        eo = torch.zeros(bsz, n, dtype=torch.long)
        off = torch.zeros(bsz, n, dtype=torch.long)
        mask = torch.zeros(bsz, n, dtype=torch.bool)
        for i, (t, e, o, _) in enumerate(chunk):
            k = t.shape[0]
            tok[i, :k], eo[i, :k], off[i, :k], mask[i, :k] = t, e, o, True
        with torch.no_grad():
            logits = model.forward_batch(tok, eo, off, mask)
        out.extend(logits[i][chunk[i][3]] for i in range(bsz))
    return out


def encode_blocks(blocks: Sequence[str]) -> Any:
    """Tokenise rendered event blocks with the frozen v0.3 tokeniser."""
    try:
        from . import tokenizer  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - until tokenizer.py lands
        raise RuntimeError("agent_loop_lab.v03.tokenizer is not available yet") from exc
    return tokenizer.encode_session(list(blocks))
