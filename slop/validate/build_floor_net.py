"""Build the §12.3 interpreter test network: the exact-stem floor written into the slop transformer by hand.

Harness code, kept out of the author kit. ``validate/floor.py`` states the floor; this network computes the same
calls for every pair of nodes at most :data:`RANGE` (7) nodes apart in reading order, and points with layer 4,
head 0. Pairs further apart are out of its reach (the residue code below has 16 values).

Residual layout (128 dims; ``_Dims`` names them). Embeddings write: 14 bits of the token id (+1/-1) for every word
id (512 and up), ``is_word``, ``is_eon``, ``is_bos``, ``const`` (offsets and BOS), ``ctx`` (context-only roles and
``sr_only``), ``prec`` (LABELLING.md §5 rank, 0-8), ``is_label``, ``in_ctrl``, and the cluster phase
(cos, sin of 2 pi k / 64).

- Layer 1, head 0 counts the EONs before each token (``x`` = number of earlier nodes, to within 2e-5): BOS has bias
  20 and value 0, every earlier EON bias 0 and value e^20. MLP 1 turns ``x`` into ``R``, the one-hot of the node
  number mod 16 (exact, from ReLU steps), gates the id bits to zero on tokens that do not take part, writes ``pw``
  (a word of a participating node) and lifts the precedence of a ``label`` with ``in_control`` to 8.
- Layer 2, head 0 (presence): each word token attends to the tokens with the same id (score 40 x matching bits) in
  the same cluster (cluster phase x 32768) among the nodes 1-7 away, or to BOS (score between a match and a miss);
  its value is the key's ``R``, so the output is positive at the residues of the nodes that hold the same stem.
  Head 1 (existence) does the same over every participating word in the same cluster. MLP 2 binarises both
  (``P``, ``Ex``), writes ``Q = pw - P`` and the layer-4 key ``KE = (8192 + 600 prec) R`` on EON tokens, and clears
  the id bits.
- Layer 3, head 0 averages ``P`` over the node's own words at its EON (``F[r]`` = share of the node's words found in
  node r, so ``F[r] = 1`` means this node's stems are a subset of node r's). Head 1 (missing) attends from the EON to
  the words of nearby nodes whose stem is not in this node (``Q[r_self] = 1``); its value ``R`` marks the nodes that
  have such a word. MLP 3 writes ``N[r] = Sub + Sup`` with ``Sub = [F = 1]`` and ``Sup = Ex and no missing word and
  not Sub``, so ``N[r] = 1`` exactly when the two stem sets are equal or nested.
- Layer 4, head 0 (pointer head) attends from each EON over the EONs of the node itself and the nodes 1-7 away:
  score ``(N + R_self) . KE - 40 d`` = 8192 + 600 prec for the node itself and for each nested partner, minus 40 per
  node of signed distance, so the winner is the partner with the highest precedence, earlier first, unless the node
  itself outranks them all. Its value is ``R``; MLP 4 writes ``flag = sum relu(o - R_self)``, which is 1 when the
  winner is another node. The output head gives KEEP 0.5 and ECHO ``flag``.

    uv run python validate/build_floor_net.py OUT_DIR
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui import model as mdl  # noqa: E402
from ui import tokenizer as tk  # noqa: E402
from validate.floor import PRECEDENCE  # noqa: E402

RANGE = 7  # largest node distance the network compares
NRES = 16  # residues of the node number
NEG = -1e9
LAM = 40.0  # presence: score per matching id bit pair (14 bits)
AC = 32768.0  # cluster phase scale
LBIG = 100.0  # existence, aggregation and missing heads
LC = 1000.0  # count head: non-EON keys score -LC
BCNT = 20.0  # count head: BOS bias, and value e^20 on EONs
KBIN = 1000.0  # binarising slope
KMISS = 1e5  # weight of "a missing word" in the Sup unit
A = 8192.0  # layer-4 key: base
BETA = 600.0  # layer-4 key: per precedence step
GAMMA = 40.0  # layer-4 bias: per node of signed distance
POINTER_HEAD = (4, 0)

# A small vocabulary of common UI stems; every other stem uses its hash bucket. The floor only compares ids.
VOCAB = ["save", "set", "notif", "project", "account", "delet", "cancel", "search", "team", "continu"]


class _Dims:
    BITS = list(range(14))
    IS_WORD, IS_EON, IS_BOS, CONST, CTX, PREC, IS_LABEL, IN_CTRL = 14, 15, 16, 17, 18, 19, 20, 21
    CL = [22, 23]
    X = 24
    R = list(range(25, 41))
    PW = 41
    P = list(range(42, 58))  # raw presence, then binary P; cleared after layer 3; then the layer-4 output
    Q = list(range(58, 74))
    EX = list(range(74, 90))
    KE = list(range(90, 106))
    F = list(range(106, 122))  # F, then N
    MI = BITS + [122, 123]  # the id bits are cleared by MLP 2
    FLAG = 124


def _bias_row(rule) -> np.ndarray:
    row = np.full(mdl.NB, NEG)
    for c in range(mdl.NB - 1):
        v = rule(c - mdl.MAX_DIST)
        if v is not None:
            row[c] = v
    return row


def build_tensors() -> "dict[str, np.ndarray]":
    T = {k: np.zeros(s, dtype=np.float64) for k, s in mdl.TENSOR_SHAPES.items()}
    d = _Dims

    # Embeddings.
    E = T["tok_emb.weight"]
    for tid in range(tk.VOCAB_BASE, tk.V):
        for b in range(14):
            E[tid, d.BITS[b]] = 1.0 if (tid >> b) & 1 else -1.0
        E[tid, d.IS_WORD] = 1.0
    E[tk.EON, d.IS_EON] = 1.0
    E[tk.BOS, d.IS_BOS] = 1.0
    E[tk.BOS, d.CONST] = 1.0
    T["off_emb.weight"][:, d.CONST] = 1.0
    for role, rid in tk.ROLE_IDS.items():
        T["role_emb.weight"][rid, d.PREC] = PRECEDENCE[role]
        if role in tk.CONTEXT_ROLES:
            T["role_emb.weight"][rid, d.CTX] = 1.0
        if role == "label":
            T["role_emb.weight"][rid, d.IS_LABEL] = 1.0
    T["flag_emb.weight"][tk.FLAG_IDS["sr_only"], d.CTX] = 1.0
    T["flag_emb.weight"][tk.FLAG_IDS["in_control"], d.IN_CTRL] = 1.0
    for c in range(mdl.N_CLUSTERS):
        T["cluster_emb.weight"][c, d.CL[0]] = math.cos(2 * math.pi * c / mdl.N_CLUSTERS)
        T["cluster_emb.weight"][c, d.CL[1]] = math.sin(2 * math.pi * c / mdl.N_CLUSTERS)

    def layer(i: int) -> dict:
        p = f"layers.{i}."
        return {n: T[p + n] for n in (
            "attn.rel_bias", "attn.q.weight", "attn.q.bias", "attn.k.weight", "attn.k.bias", "attn.v.weight",
            "attn.v.bias", "attn.o.weight", "attn.o.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight",
            "mlp.fc2.bias")}

    def near(dist: int) -> float | None:
        return 0.0 if 1 <= abs(dist) <= RANGE else None

    # Layer 1: count earlier nodes; MLP 1 builds R, gated bits, pw and the label precedence.
    t = layer(0)
    t["attn.q.bias"][0] = 4 * LC
    t["attn.k.weight"][0, d.IS_EON] = 1.0
    t["attn.k.weight"][0, d.IS_BOS] = 1.0
    t["attn.k.weight"][0, d.CONST] = -1.0
    t["attn.v.weight"][0, d.IS_EON] = math.exp(BCNT)
    t["attn.o.weight"][d.X, 0] = 1.0
    t["attn.rel_bias"][0] = _bias_row(lambda dist: 0.0 if dist < 0 else None)
    t["attn.rel_bias"][0, mdl.BOS_COL] = BCNT
    W1, b1, W2 = t["mlp.fc1.weight"], t["mlp.fc1.bias"], t["mlp.fc2.weight"]
    u = 0
    steps = []  # (unit list, coefficient list) giving s_m for m = 0..64
    W1[u, d.CONST], W1[u, d.IS_BOS] = 1.0, -1.0  # s_0 = 1 except at BOS
    steps.append([(u, 1.0)])
    u += 1
    for m in range(1, 65):
        W1[u, d.X], b1[u] = 1.0, -m + 0.75
        W1[u + 1, d.X], b1[u + 1] = 1.0, -m + 0.25
        steps.append([(u, 2.0), (u + 1, -2.0)])
        u += 2
    for m, parts in enumerate(steps):
        for r in range(NRES):
            coef = (1.0 if m <= 63 and m % NRES == r else 0.0) - (1.0 if m >= 1 and (m - 1) % NRES == r else 0.0)
            if coef:
                for unit, c in parts:
                    W2[d.R[r], unit] += coef * c
    for b in range(14):
        bit = d.BITS[b]
        for sign in (1.0, -1.0):
            W1[u, bit], W1[u, d.IS_WORD], W1[u, d.CTX], b1[u] = sign, 2.0, -2.0, -2.0
            W2[bit, u] = sign
            u += 1
            W1[u, bit] = sign
            W2[bit, u] = -sign
            u += 1
    W1[u, d.IS_WORD], W1[u, d.CTX] = 1.0, -1.0
    W2[d.PW, u] = 1.0
    u += 1
    W1[u, d.IS_LABEL], W1[u, d.IN_CTRL], b1[u] = 1.0, 1.0, -1.0
    W2[d.PREC, u] = 8.0 - PRECEDENCE["label"]
    u += 1
    assert u <= mdl.M

    # Layer 2: presence (head 0) and existence (head 1); MLP 2 binarises, writes Q and KE, clears the bits.
    t = layer(1)
    Wq, Wk, Wv, Wo, bq = t["attn.q.weight"], t["attn.k.weight"], t["attn.v.weight"], t["attn.o.weight"], t["attn.q.bias"]
    for b in range(14):
        Wq[b, d.BITS[b]] = 4 * LAM
        Wk[b, d.BITS[b]] = 1.0
    for j in range(2):
        Wq[14 + j, d.CL[j]] = 4 * AC
        Wk[14 + j, d.CL[j]] = 1.0
    for r in range(NRES):
        Wv[r, d.R[r]] = 1.0
        Wo[d.P[r], r] = 1.0
    t["attn.rel_bias"][0] = _bias_row(near)
    t["attn.rel_bias"][0, mdl.BOS_COL] = 13 * LAM + AC
    bq[16] = 4 * LBIG
    Wk[16, d.PW] = 1.0
    for j in range(2):
        Wq[17 + j, d.CL[j]] = 4 * AC
        Wk[17 + j, d.CL[j]] = 1.0
    for r in range(NRES):
        Wv[16 + r, d.R[r]] = 1.0
        Wo[d.EX[r], 16 + r] = 1.0
    t["attn.rel_bias"][1] = _bias_row(near)
    t["attn.rel_bias"][1, mdl.BOS_COL] = LBIG / 2 + AC
    W1, b1, W2 = t["mlp.fc1.weight"], t["mlp.fc1.bias"], t["mlp.fc2.weight"]
    u = 0
    upw = u
    W1[u, d.PW] = 1.0
    u += 1
    for r in range(NRES):
        p1, p2, p0 = u, u + 1, u + 2
        W1[p1, d.P[r]] = KBIN
        W1[p2, d.P[r]], b1[p2] = KBIN, -1.0
        W1[p0, d.P[r]] = 1.0
        W2[d.P[r], p1], W2[d.P[r], p2], W2[d.P[r], p0] = 1.0, -1.0, -1.0
        W2[d.Q[r], p1], W2[d.Q[r], p2], W2[d.Q[r], upw] = -1.0, 1.0, 1.0
        u += 3
        e1, e2, e0 = u, u + 1, u + 2
        W1[e1, d.EX[r]] = KBIN
        W1[e2, d.EX[r]], b1[e2] = KBIN, -1.0
        W1[e0, d.EX[r]] = 1.0
        W2[d.EX[r], e1], W2[d.EX[r], e2], W2[d.EX[r], e0] = 1.0, -1.0, -1.0
        u += 3
        k1, k2 = u, u + 1
        W1[k1, d.R[r]], W1[k1, d.IS_EON], b1[k1] = 1.0, 1.0, -1.0
        W1[k2, d.PREC], W1[k2, d.R[r]], W1[k2, d.IS_EON], b1[k2] = 1.0, 8.0, 8.0, -16.0
        W2[d.KE[r], k1], W2[d.KE[r], k2] = A, BETA
        u += 2
    for b in range(14):
        for sign in (1.0, -1.0):
            W1[u, d.BITS[b]] = sign
            W2[d.BITS[b], u] = -sign
            u += 1
    assert u <= mdl.M

    # Layer 3: aggregation over the node's own words (head 0) and missing words (head 1); MLP 3 writes N.
    t = layer(2)
    Wq, Wk, Wv, Wo, bq = t["attn.q.weight"], t["attn.k.weight"], t["attn.v.weight"], t["attn.o.weight"], t["attn.q.bias"]
    bq[0] = 4 * LBIG
    Wk[0, d.PW] = 1.0
    for r in range(NRES):
        Wv[r, d.P[r]] = 1.0
        Wo[d.F[r], r] = 1.0
    t["attn.rel_bias"][0] = _bias_row(lambda dist: 0.0 if dist == 0 else None)
    t["attn.rel_bias"][0, mdl.BOS_COL] = LBIG / 2
    for r in range(NRES):
        Wq[16 + r, d.R[r]] = 4 * LBIG
        Wk[16 + r, d.Q[r]] = 1.0
        Wv[16 + r, d.R[r]] = 1.0
        Wo[d.MI[r], 16 + r] = 1.0
    t["attn.rel_bias"][1] = _bias_row(near)
    t["attn.rel_bias"][1, mdl.BOS_COL] = LBIG / 2
    W1, b1, W2 = t["mlp.fc1.weight"], t["mlp.fc1.bias"], t["mlp.fc2.weight"]
    u = 0
    for r in range(NRES):
        s1, s2, z1, z2, f0, pc = range(u, u + 6)
        W1[s1, d.F[r]], b1[s1] = 62.0, -61.0
        W1[s2, d.F[r]], b1[s2] = 62.0, -62.0
        for z, off in ((z1, 0.0), (z2, -1.0)):
            W1[z, d.EX[r]], W1[z, d.MI[r]], W1[z, d.F[r]], b1[z] = 31.0, -KMISS, -31.0, off
        W1[f0, d.F[r]] = 1.0
        W1[pc, d.P[r]] = 1.0
        W2[d.F[r], s1], W2[d.F[r], s2], W2[d.F[r], z1], W2[d.F[r], z2], W2[d.F[r], f0] = 1.0, -1.0, 1.0, -1.0, -1.0
        W2[d.P[r], pc] = -1.0
        u += 6
    assert u <= mdl.M

    # Layer 4: the pointer head and the flag.
    t = layer(3)
    Wq, Wk, Wv, Wo = t["attn.q.weight"], t["attn.k.weight"], t["attn.v.weight"], t["attn.o.weight"]
    for r in range(NRES):
        Wq[r, d.F[r]] = 4.0
        Wq[r, d.R[r]] = 4.0
        Wk[r, d.KE[r]] = 1.0
        Wv[r, d.R[r]] = 1.0
        Wo[d.P[r], r] = 1.0
    t["attn.rel_bias"][0] = _bias_row(lambda dist: -GAMMA * dist if abs(dist) <= RANGE else None)
    W1, W2 = t["mlp.fc1.weight"], t["mlp.fc2.weight"]
    for r in range(NRES):
        W1[r, d.P[r]], W1[r, d.R[r]] = 1.0, -1.0
        W2[d.FLAG, r] = 1.0

    T["head.weight"][1, d.FLAG] = 1.0
    T["head.bias"][0] = 0.5
    return {k: v.astype(np.float32) for k, v in T.items()}


def build(out_dir: str | Path) -> str:
    return mdl.save_artifact(out_dir, build_tensors(), VOCAB, POINTER_HEAD)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: build_floor_net.py OUT_DIR", file=sys.stderr)
        return 2
    sha = build(argv[0])
    print(f"{argv[0]}: floor test network, artefact sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
