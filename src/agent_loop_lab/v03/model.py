"""Immutable target transformer (spec v0.3 §2.2).

This file ships in the authoring packet as a top-level module (``import
model``). It therefore imports only ``torch`` and the standard library and has
no relative imports.

Architecture (no normalisation, no dropout, no absolute token position)::

    h0 = E[tokens] + P[offset]
    for each of 4 layers:
        q, k, v = h W_q^T + b_q, h W_k^T + b_k, h W_v^T + b_v   (8 heads x 16)
        score[h,i,j] = q_{h,i} . k_{h,j} / 4 + B[h, c(i,j)]      (j <= i)
        c(i,j) = R            if key j is BOS (event_ord_j == 0)
               = min(event_ord_i - event_ord_j, R - 1) otherwise
        h = h + concat_h(softmax_j(score) v_h) W_o^T + b_o
        h = h + W_2 ReLU(W_1 h + b_1) + b_2
    logits = W_out h + b_out                                       (4 classes)

Parameter names (``state_dict`` order) are listed in :data:`PARAM_SHAPES`.
Class order is ``[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]``; argmax ties
go to the first index, so all-zero logits mean PRODUCTIVE.

Implementation note (does not change the maths): the forward pass runs over
fixed-size blocks of ``BLOCK`` query rows, and query block ``b`` always sees
keys ``[0, (b + 1) * BLOCK)`` with not-yet-valid keys masked. Every tensor
operation that touches a given row therefore has a shape that depends only on
that row's block index, not on the session length, so appending events leaves
every earlier position's logits bit-for-bit unchanged (spec §11 causal check;
tested). It also bounds peak memory to ``H x BLOCK x n`` scores per layer.
Query rows of the same event share one bias row, which is gathered once per
row group and copied to the rows exactly.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn

# -- frozen constants (must match tokenizer.py) --------------------------------

V = 10240  # vocabulary size
D = 128  # residual width d
L = 4  # layers
H = 8  # heads
DH = 16  # head width d_h
M = 512  # MLP width m
R = 64  # event-distance buckets 0..R-1; column R is the BOS column
O = 192  # within-event offsets
N_CLASSES = 4

PAD = 0
BOS = 1
EOE = 2

CLASS_NAMES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")

MASKED = R + 1  # internal bias-table row holding -inf (padding mask)
QK_SCALE = 0.25  # 1 / sqrt(d_h), exact in binary floating point
BLOCK = 128  # query rows per block (implementation detail, see module docstring)

assert H * DH == D


def _param_shapes() -> "OrderedDict[str, tuple[int, ...]]":
    shapes: "OrderedDict[str, tuple[int, ...]]" = OrderedDict()
    shapes["tok_emb.weight"] = (V, D)
    shapes["off_emb.weight"] = (O, D)
    for i in range(L):
        p = f"layers.{i}."
        # nn.Module lists a module's own parameters before its submodules'.
        shapes[p + "attn.rel_bias"] = (H, R + 1)
        for name in ("q", "k", "v", "o"):
            shapes[p + f"attn.{name}.weight"] = (D, D)
            shapes[p + f"attn.{name}.bias"] = (D,)
        shapes[p + "mlp.fc1.weight"] = (M, D)
        shapes[p + "mlp.fc1.bias"] = (M,)
        shapes[p + "mlp.fc2.weight"] = (D, M)
        shapes[p + "mlp.fc2.bias"] = (D,)
    shapes["head.weight"] = (N_CLASSES, D)
    shapes["head.bias"] = (N_CLASSES,)
    return shapes


PARAM_SHAPES = _param_shapes()
PARAM_COUNT = sum(math.prod(s) for s in PARAM_SHAPES.values())
assert PARAM_COUNT == 2_128_932, PARAM_COUNT


# -- zero-initialised building blocks ------------------------------------------
# Construction never draws a random number: reset_parameters zero-fills
# instead of calling uniform_/normal_ (spec §2.2 initial state is all zero).


class _ZeroLinear(nn.Linear):
    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()


class _ZeroEmbedding(nn.Embedding):
    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = _ZeroLinear(D, D, bias=True)
        self.k = _ZeroLinear(D, D, bias=True)
        self.v = _ZeroLinear(D, D, bias=True)
        self.o = _ZeroLinear(D, D, bias=True)
        self.rel_bias = nn.Parameter(torch.zeros(H, R + 1))


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = _ZeroLinear(D, M, bias=True)
        self.fc2 = _ZeroLinear(M, D, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(h)))


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()
        self.mlp = MLP()


def _split_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, n, D] -> [B, H, n, DH]."""
    b, n, _ = x.shape
    return x.reshape(b, n, H, DH).transpose(1, 2)


class DoomLoopTransformer(nn.Module):
    """The fixed v0.3 target model. Every parameter starts at exactly 0.0."""

    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = _ZeroEmbedding(V, D)
        self.off_emb = _ZeroEmbedding(O, D)
        self.layers = nn.ModuleList(Layer() for _ in range(L))
        self.head = _ZeroLinear(D, N_CLASSES, bias=True)

    # -- public API -------------------------------------------------------------

    def forward(self, tokens: torch.Tensor, event_ord: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Logits ``[n, 4]`` at every position of ONE session (1-D LongTensors)."""
        tokens, event_ord, offset = _as_session(tokens, event_ord, offset)
        n = tokens.shape[0]
        valid = torch.ones(1, n, dtype=torch.bool)
        logits, _ = self._run(tokens[None], event_ord[None], offset[None], valid, None)
        return logits[0]

    def forward_batch(
        self,
        tokens: torch.Tensor,
        event_ord: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Logits ``[B, n, 4]`` for a right-padded batch of sessions.

        ``mask[b, i]`` is True for real tokens; each row's real tokens must form
        a prefix. Logits at padded positions are set to 0. Autograd-safe (no
        in-place writes into saved tensors), so it also serves batched training.
        """
        tokens, event_ord, offset, mask = _as_batch(tokens, event_ord, offset, mask)
        logits, _ = self._run(tokens, event_ord, offset, mask, None)
        return logits.masked_fill(~mask[..., None], 0.0)

    def activations(
        self,
        tokens: torch.Tensor,
        event_ord: torch.Tensor,
        offset: torch.Tensor,
        query_positions: "list[int] | None" = None,
    ) -> "dict[str, object]":
        """Intermediate values of ONE session for the inspector.

        Returns ``embed`` ``[n, D]``; ``resid_attn`` and ``resid_mlp``: lists
        (one per layer) of ``[n, D]`` residuals after the attention and MLP
        updates; ``attn``: list (one per layer) of ``[H, len(query_positions),
        n]`` attention weights for the requested query rows (all rows when
        ``query_positions`` is None, which is ``4 x H x n x n`` floats: keep
        ``n`` small); ``query_positions``; and ``logits`` ``[n, 4]``.
        """
        tokens, event_ord, offset = _as_session(tokens, event_ord, offset)
        n = tokens.shape[0]
        if query_positions is None:
            query_positions = list(range(n))
        qp = [int(p) for p in query_positions]
        for p in qp:
            if not 0 <= p < n:
                raise ValueError(f"query position {p} outside [0, {n})")
        collect: dict = {"query_positions": qp}
        valid = torch.ones(1, n, dtype=torch.bool)
        logits, collect = self._run(tokens[None], event_ord[None], offset[None], valid, collect)
        collect["logits"] = logits[0]
        return collect

    # -- core -------------------------------------------------------------------

    def _run(
        self,
        tokens: torch.Tensor,
        event_ord: torch.Tensor,
        offset: torch.Tensor,
        valid: torch.Tensor,
        collect: "dict | None",
    ) -> "tuple[torch.Tensor, dict | None]":
        bsz, n = tokens.shape
        nb = max(1, -(-n // BLOCK))
        npad = nb * BLOCK
        if npad != n:
            pad = npad - n
            tokens = torch.cat([tokens, torch.full((bsz, pad), PAD, dtype=torch.long)], dim=1)
            event_ord = torch.cat([event_ord, torch.zeros(bsz, pad, dtype=torch.long)], dim=1)
            offset = torch.cat([offset, torch.zeros(bsz, pad, dtype=torch.long)], dim=1)
            valid = torch.cat([valid, torch.zeros(bsz, pad, dtype=torch.bool)], dim=1)

        is_bos = event_ord == 0  # [B, npad]; padded keys are masked anyway
        upper = torch.ones(BLOCK, BLOCK, dtype=torch.bool).triu(1)  # j > i inside a block

        # Bias-table indices, shared by all layers. Within a block, query rows of
        # the same event share one bias row, so indices are built per row group
        # (a run of equal event_ord) and expanded to rows by an exact lookup.
        # Index MASKED selects a -inf entry (padded keys); j > i inside the
        # diagonal block is masked separately with ``upper``.
        groups: list[torch.Tensor] = []  # [B*H*BLOCK] (batch, head, row) -> bias row
        gidx: list[torch.Tensor] = []  # [B, U, e] group -> table row
        for b in range(nb):
            s, e = b * BLOCK, (b + 1) * BLOCK
            eo_q = event_ord[:, s:e]
            change = torch.ones_like(eo_q, dtype=torch.long)
            change[:, 1:] = (eo_q[:, 1:] != eo_q[:, :-1]).long()
            change[:, 0] = 0
            grp = torch.cumsum(change, dim=1)  # [B, BLOCK]
            u = int(grp.max()) + 1
            geo = torch.zeros(bsz, u, dtype=torch.long).scatter_(1, grp, eo_q)
            dist = geo[:, :, None] - event_ord[:, None, :e]  # [B, U, e]
            c = dist.clamp(0, R - 1).masked_fill(is_bos[:, None, :e], R)
            c = c.masked_fill(~valid[:, None, :e], MASKED)
            # Row of the flattened [B*H*U, e] bias table for each (batch, head, query row).
            bh = torch.arange(bsz * H, dtype=torch.long).view(bsz, H, 1)
            groups.append((bh * u + grp[:, None, :]).reshape(-1))
            gidx.append(c)

        hs = [self.tok_emb(tokens[:, b * BLOCK : (b + 1) * BLOCK]) + self.off_emb(offset[:, b * BLOCK : (b + 1) * BLOCK]) for b in range(nb)]
        if collect is not None:
            collect["embed"] = torch.cat(hs, dim=1)[0, :n].detach().clone()
            collect["resid_attn"] = []
            collect["resid_mlp"] = []
            collect["attn"] = []
            qp = collect["query_positions"]
            qpt = torch.tensor(qp, dtype=torch.long)

        # With autograd live (training), keys/values are concatenated afresh per
        # block; otherwise they are written into preallocated buffers.
        grad_live = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        neg_inf = float("-inf")
        neg_inf_col = torch.full((H, 1), neg_inf)
        for layer in self.layers:
            attn = layer.attn
            table = torch.cat([attn.rel_bias, neg_inf_col], dim=1).t().contiguous()  # [R + 2, H]
            qs: list[torch.Tensor] = []
            ks: list[torch.Tensor] = []
            vs: list[torch.Tensor] = []
            if not grad_live:
                kbuf = torch.empty(bsz * H, npad, DH)
                vbuf = torch.empty(bsz * H, npad, DH)
            for b in range(nb):
                hb = hs[b]
                qs.append((_split_heads(attn.q(hb)) * QK_SCALE).reshape(bsz * H, BLOCK, DH))
                kb = _split_heads(attn.k(hb)).reshape(bsz * H, BLOCK, DH)
                vb = _split_heads(attn.v(hb)).reshape(bsz * H, BLOCK, DH)
                if grad_live:
                    ks.append(kb)
                    vs.append(vb)
                else:
                    kbuf[:, b * BLOCK : (b + 1) * BLOCK] = kb
                    vbuf[:, b * BLOCK : (b + 1) * BLOCK] = vb
            if collect is not None:
                amap = torch.zeros(H, len(qp), npad)
            new_hs: list[torch.Tensor] = []
            for b in range(nb):
                s, e = b * BLOCK, (b + 1) * BLOCK
                if grad_live:
                    kb = torch.cat(ks[: b + 1], dim=1)  # [B*H, e, DH]
                    vb = torch.cat(vs[: b + 1], dim=1)
                else:
                    kb = kbuf[:, :e]
                    vb = vbuf[:, :e]
                bias_g = torch.embedding(table, gidx[b]).permute(0, 3, 1, 2).reshape(-1, e)  # [B*H*U, e]
                bias = bias_g.index_select(0, groups[b]).view(bsz * H, BLOCK, e)
                score = torch.bmm(qs[b], kb.transpose(1, 2)).add_(bias)
                score[:, :, s:e].masked_fill_(upper, neg_inf)
                a = torch.softmax(score, dim=-1)  # [B*H, BLOCK, e]
                ctx = torch.bmm(a, vb).view(bsz, H, BLOCK, DH).transpose(1, 2).reshape(bsz, BLOCK, D)
                hb = hs[b] + attn.o(ctx)
                new_hs.append(hb)
                if collect is not None:
                    sel = ((qpt >= s) & (qpt < e)).nonzero().flatten()
                    if sel.numel():
                        amap[:, sel, :e] = a.view(bsz, H, BLOCK, e)[0][:, qpt[sel] - s, :].detach()
            if collect is not None:
                collect["attn"].append(amap[:, :, :n].clone())
                collect["resid_attn"].append(torch.cat(new_hs, dim=1)[0, :n].detach().clone())
            hs = [hb + layer.mlp(hb) for hb in new_hs]
            if collect is not None:
                collect["resid_mlp"].append(torch.cat(hs, dim=1)[0, :n].detach().clone())

        logits = torch.cat([self.head(hb) for hb in hs], dim=1)[:, :n]
        return logits, collect


# -- input checks ---------------------------------------------------------------


def _check_rows(tokens: torch.Tensor, event_ord: torch.Tensor, offset: torch.Tensor, where: str) -> None:
    n = tokens.shape[0]
    if n == 0:
        raise ValueError(f"{where}: empty session")
    if int(tokens[0]) != BOS or int(event_ord[0]) != 0:
        raise ValueError(f"{where}: position 0 must be BOS with event_ord 0")
    if n > 1:
        if int(event_ord[1:].min()) < 1:
            raise ValueError(f"{where}: event_ord must be >= 1 after BOS")
        if bool((event_ord[1:] < event_ord[:-1]).any()):
            raise ValueError(f"{where}: event_ord must be non-decreasing")
    if int(tokens.min()) < 0 or int(tokens.max()) >= V:
        raise ValueError(f"{where}: token id outside [0, {V})")
    if int(offset.min()) < 0 or int(offset.max()) >= O:
        raise ValueError(f"{where}: offset outside [0, {O})")


def _as_long(x: object, name: str) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise ValueError(f"{name}: integer tensor required, got {t.dtype}")
    return t.to(torch.long)


def _as_session(tokens: object, event_ord: object, offset: object) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    t, e, o = _as_long(tokens, "tokens"), _as_long(event_ord, "event_ord"), _as_long(offset, "offset")
    if t.ndim != 1 or e.shape != t.shape or o.shape != t.shape:
        raise ValueError(f"expected three equal 1-D tensors, got {tuple(t.shape)}, {tuple(e.shape)}, {tuple(o.shape)}")
    _check_rows(t, e, o, "session")
    return t, e, o


def _as_batch(
    tokens: object, event_ord: object, offset: object, mask: object
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    t, e, o = _as_long(tokens, "tokens"), _as_long(event_ord, "event_ord"), _as_long(offset, "offset")
    m = torch.as_tensor(mask).to(torch.bool)
    if t.ndim != 2 or e.shape != t.shape or o.shape != t.shape or m.shape != t.shape:
        raise ValueError("expected four equal 2-D tensors [B, n]")
    lengths = m.sum(dim=1)
    for b in range(t.shape[0]):
        n = int(lengths[b])
        if not bool(m[b, :n].all()):
            raise ValueError(f"batch row {b}: real tokens must form a prefix")
        _check_rows(t[b, :n], e[b, :n], o[b, :n], f"batch row {b}")
    # Padded slots get neutral, in-range values.
    t = t.masked_fill(~m, PAD)
    e = e.masked_fill(~m, 0)
    o = o.masked_fill(~m, 0)
    return t, e, o, m


# -- helpers --------------------------------------------------------------------


def zero_model(requires_grad: bool = False) -> DoomLoopTransformer:
    """Model with every parameter exactly 0.0 (spec §2.2 initial state).

    Draws no random numbers. The direct-weight path uses ``requires_grad=False``.
    """
    model = DoomLoopTransformer()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
            p.requires_grad_(requires_grad)
    return model.to(dtype=torch.float32, device="cpu").eval()


def predict(logits: torch.Tensor) -> torch.Tensor:
    """argmax over the class dimension; ties go to the first maximal index."""
    return torch.argmax(logits, dim=-1)
