"""Hand-written weights for the v0.3 doom-loop transformer, Phase B revision.

Every parameter is an explicit constant or a closed-form rule; the circuit implements
reference_policy.py and is documented head by head in CIRCUITS.md.  Scores are q.k/4 + B, so
query weights carry a factor 4 and comments quote scores directly.  Head dim 15 is a key gate
(-M per unmet condition, +M*is_bos), so the BOS key scores exactly its bias entry: BOS is the
sink or anchor of every head.
"""

import math

import torch

import model
import tokenizer

# ---- residual-subspace allocation (see CIRCUITS.md) --------------------------------------------
BOS, EOE, NLAT, HASHED, GT, NONRES, DIGIT, NUM = 0, 1, 2, 3, 4, 5, 6, 7
BITS = 8  # 8..20: 13 identity bits of every non-reserved token id
ISSTAT, SOK, SFAIL = 21, 22, 23
TINSP, TVAL, TMUT, TREVERT, THTTP, TPOLL = 24, 25, 26, 27, 28, 29
CODE = 30  # 30..43: 7 frequencies x (cos, sin) of the within-block offset
OFFLIN, OFF7, OFFGE10 = 44, 45, 46
BTINSP, BTVAL, BTMUT, BTREVERT, BTHTTP, BTPOLL = 47, 48, 49, 50, 51, 52
BSOK, BSFAIL = 53, 54
PBITS = 55  # 55..67: identity bits of the previous token
PHASHED, PGT, PISSTAT, HASHUNK = 68, 69, 70, 71
IBITS = 72  # 72..84: identity bits of the block's intent token (gathered into the EOE)
VM = 85
REVMATCH, PV, EXISTS, PSOK, PSFAIL, NEAREST, OUTFAIL, ANCH, EXISTS7 = 86, 87, 88, 89, 90, 91, 92, 93, 94
NOTIDST, CNT3, CNT5, REVTOK, NOOP, REDINSP = 95, 96, 97, 98, 99, 100
PPV, PEX7, PPSFAIL, PNEAR, MAXPV, NVEX, NVNOTST, REVBLK = 101, 102, 103, 104, 105, 106, 107, 108
SI, INPOL, RESOLV, IMPROVED = 109, 110, 111, 112
AVGSI, AVGREV, AVGIMP, ANCH8, ACT, REC4 = 113, 114, 115, 116, 117, 118
V2, I5, PROG, RECB = 119, 120, 121, 122

NFREQ = 7
M = 1.0e5  # gate magnitude
KAPPA = 4000.0  # "latest offset wins" slope (per unit of offset/192)
LAM = 15.0  # nearest-block slope per event distance
A_OFF = 30.0  # offset-code match scale: exact 7*A, worst other 5.27*A, sink 6.15*A
A_BITS = 20.0  # hunk-id match scale (13 bits): match 13*A, other <= 11*A, sink 12*A
A_INT = 800.0  # intent match scale (13 bits + 2 class bits): same key 15*A, other <= 13*A
LAMP = 12.0  # passed-count slope of the max-count head (counts are capped at 100)
B_HASH = 6000.0  # intent gather: bonus of a hashed token over any other non-reserved token
KAP2 = 3000.0  # intent gather: earliest offset wins (15.6 per offset)
KSTEP = 40.0  # threshold sharpness of count / comparison units
KGATE = 1.0e4  # conjunction gate of MLP units
DH = model.DH
GATE = 15  # head dim used as the key gate

STATUS_WORDS = ("ok", "error", "FAILED", "ready", "pending")


def _freqs():
    return [math.pi / (1.5 ** i) for i in range(NFREQ)]


def _code(o):
    out = []
    for w in _freqs():
        out.append(math.cos(w * o))
        out.append(math.sin(w * o))
    return out


def _rot(delta):
    """Linear map R with R code(o) = code(o + delta): list of (row, col, value)."""
    out = []
    for i, w in enumerate(_freqs()):
        c, s = math.cos(w * delta), math.sin(w * delta)
        out.append((2 * i, 2 * i, c))
        out.append((2 * i, 2 * i + 1, -s))
        out.append((2 * i + 1, 2 * i, s))
        out.append((2 * i + 1, 2 * i + 1, c))
    return out


def build() -> model.DoomLoopTransformer:
    m = model.zero_model()
    with torch.no_grad():
        E = m.tok_emb.weight
        P = m.off_emb.weight

        # ---- token embedding -----------------------------------------------------------------
        E[tokenizer.BOS, BOS] = 1.0
        E[tokenizer.EOE, EOE] = 1.0
        E[tokenizer.NL_AT, NLAT] = 1.0
        nonres = torch.arange(128, tokenizer.V)
        E[nonres, NONRES] = 1.0
        pat = torch.remainder(nonres - tokenizer.HASH_BASE, tokenizer.HASH_BUCKETS)
        for b in range(13):
            E[nonres, BITS + b] = ((pat >> b) & 1).to(torch.float32) * 2.0 - 1.0
        hashed = torch.arange(tokenizer.HASH_BASE, tokenizer.V)
        E[hashed, HASHED] = 1.0
        E[tokenizer.word_id("h0"), HASHED] = 0.0
        E[tokenizer.PUNCT_IDS[">"], GT] = 1.0
        digits = torch.arange(tokenizer.DIGIT_BASE, tokenizer.DIGIT_BASE + 1110)
        E[digits, DIGIT] = 1.0
        d1 = torch.arange(10)
        E[tokenizer.DIGIT_BASE + d1, NUM] = d1.to(torch.float32)
        d2 = torch.arange(100)
        E[tokenizer.DIGIT_BASE + 10 + d2, NUM] = d2.to(torch.float32)
        E[tokenizer.DIGIT_BASE + 110 + torch.arange(1000), NUM] = 100.0
        for w in STATUS_WORDS:
            E[tokenizer.RESERVED_IDS[w], ISSTAT] = 1.0
        for w in ("ok", "ready"):
            E[tokenizer.RESERVED_IDS[w], SOK] = 1.0
        for w in ("error", "pending", "FAILED"):
            E[tokenizer.RESERVED_IDS[w], SFAIL] = 1.0
        for w in ("read_file", "search", "list_dir", "plan", "ask", "yield"):
            E[tokenizer.RESERVED_IDS[w], TINSP] = 1.0
        for w in ("run_tests", "build", "lint"):
            E[tokenizer.RESERVED_IDS[w], TVAL] = 1.0
        for w in ("edit_file", "write_file", "revert"):
            E[tokenizer.RESERVED_IDS[w], TMUT] = 1.0
        E[tokenizer.RESERVED_IDS["revert"], TREVERT] = 1.0
        E[tokenizer.RESERVED_IDS["http_call"], THTTP] = 1.0
        E[tokenizer.RESERVED_IDS["poll_status"], TPOLL] = 1.0

        # ---- offset embedding ----------------------------------------------------------------
        for o in range(model.O):
            code = _code(o)
            for i in range(2 * NFREQ):
                P[o, CODE + i] = code[i]
            P[o, OFFLIN] = o / 192.0
            if o >= 10:
                P[o, OFFGE10] = 1.0
        P[7, OFF7] = 1.0

        # ---- helpers -------------------------------------------------------------------------
        def layer(l):
            return m.layers[l]

        def wq(l, h, hd, dim, w):
            layer(l).attn.q.weight[h * DH + hd, dim] = w

        def bq(l, h, hd, w):
            layer(l).attn.q.bias[h * DH + hd] = w

        def wk(l, h, hd, dim, w):
            layer(l).attn.k.weight[h * DH + hd, dim] = w

        def bk(l, h, hd, w):
            layer(l).attn.k.bias[h * DH + hd] = w

        def wv(l, h, hd, dim, w):
            layer(l).attn.v.weight[h * DH + hd, dim] = w

        def wo(l, h, hd, dim, w):
            layer(l).attn.o.weight[dim, h * DH + hd] = w

        def gate(l, h, conds, any_of=()):
            """Key gate: -M per unmet condition in `conds`; `any_of` is one more condition met
            when at least one of its (mutually exclusive) flags is 1."""
            bq(l, h, GATE, 4.0)
            n = len(conds) + (1 if any_of else 0)
            for dim in conds:
                wk(l, h, GATE, dim, M)
            for dim in any_of:
                wk(l, h, GATE, dim, M)
            bk(l, h, GATE, -n * M)
            wk(l, h, GATE, BOS, n * M)

        def rel(l, h, same=None, near=None, far=None, sink=0.0, upto=7):
            """Bias row: same-block entry, near(c) for c = 1..upto, far for c > upto, BOS column."""
            rb = layer(l).attn.rel_bias
            rb[h, :] = -M
            if same is not None:
                rb[h, 0] = same
            if near is not None:
                for c in range(1, upto + 1):
                    rb[h, c] = near(c)
            if far is not None:
                for c in range(upto + 1, model.R):
                    rb[h, c] = far
            rb[h, model.R] = sink

        def code_match(l, h, delta, scale):
            """q = 4*scale*code(query); k = R_delta code(key), BOS key nulled: score =
            scale * K(o_q - o_k - delta), exact hit 7*scale."""
            for i in range(2 * NFREQ):
                wq(l, h, i, CODE + i, 4.0 * scale)
            c0 = _code(0)
            rc0 = [0.0] * (2 * NFREQ)
            for r, c, v in _rot(delta):
                wk(l, h, r, CODE + c, v)
                rc0[r] += v * c0[c]
            for i in range(2 * NFREQ):
                wk(l, h, i, BOS, -rc0[i])

        def intent_match(l, h, scale, with_class=True):
            """13 intent bits (plus 2 tool-class bits) on both sides: same key scores 15*scale
            (13*scale without the class bits)."""
            for b in range(13):
                wq(l, h, b, IBITS + b, 4.0 * scale)
                wk(l, h, b, IBITS + b, 1.0)
            if not with_class:
                return
            cls1 = {BTINSP: 1.0, BTMUT: 1.0, BTVAL: -1.0, BTHTTP: -1.0, BTPOLL: -1.0}
            cls2 = {BTINSP: 1.0, BTMUT: -1.0, BTVAL: 1.0, BTHTTP: -1.0, BTPOLL: -1.0}
            for hd, cls in ((13, cls1), (14, cls2)):
                for dim, w in cls.items():
                    wq(l, h, hd, dim, 4.0 * scale * w)
                    wk(l, h, hd, dim, w)

        def nearest_same_key(l, h, values):
            """Nearest earlier block with the same intent and class (whole prefix); `values`
            maps source dims of that block to destination dims of the query's block."""
            intent_match(l, h, A_INT)
            gate(l, h, (EOE,))
            rel(l, h, same=-M, near=lambda c: -LAM * c, far=None, sink=SINK_INT, upto=model.R - 1)
            for hd, (src, dst) in enumerate(values):
                wv(l, h, hd, src, 1.0)
                wo(l, h, hd, dst, 1.0)

        unit_counter = [0] * model.L

        def unit(l, inputs, bias, outputs):
            u = unit_counter[l]
            unit_counter[l] += 1
            for dim, w in inputs.items():
                layer(l).mlp.fc1.weight[u, dim] = w
            layer(l).mlp.fc1.bias[u] = bias
            for dim, w in outputs.items():
                layer(l).mlp.fc2.weight[dim, u] = w

        def clip_unit(l, inputs, bias, outputs):
            """clip(x, 0, 1) = relu(x) - relu(x - 1) for x = inputs.h + bias."""
            unit(l, inputs, bias, outputs)
            unit(l, inputs, bias - 1.0, {d: -w for d, w in outputs.items()})

        def gated_clip(l, cond, n_cond, lin, bias, outputs):
            """clip(lin.h + bias) when every flag in `cond` (weights +1 / -1) sums to n_cond,
            else 0: KGATE * (cond.h - n_cond) is added to the argument."""
            inputs = {d: w * KGATE for d, w in cond.items()}
            for d, w in lin.items():
                inputs[d] = inputs.get(d, 0.0) + w
            clip_unit(l, inputs, bias - n_cond * KGATE, outputs)

        SINK_OFF = (7.0 + 5.3) / 2.0 * A_OFF
        SINK_INT = 10700.0  # nearest same-key: match >= 12000 - 945, other <= 10400
        SINK_INT7 = 11200.0  # same-key within 7: match 12000, other <= 10400
        SINK_MAXPV = 10200.0  # max count: same id >= 13*A = 10400, other <= 11*A + 100*LAMP = 10000

        # ================================ layer 0 (within the block) ============================
        # H0.0: copy the tool flags of the token at offset 7 to every token of the block
        gate(0, 0, (OFF7,))
        rel(0, 0, same=0.0, sink=-M / 2)
        for i, (src, dst) in enumerate(((TINSP, BTINSP), (TVAL, BTVAL), (TMUT, BTMUT), (TREVERT, BTREVERT),
                                        (THTTP, BTHTTP), (TPOLL, BTPOLL))):
            wv(0, 0, i, src, 1.0)
            wo(0, 0, i, dst, 1.0)
        # H0.1: the last status word before me (largest offset wins) -> block status
        gate(0, 1, (ISSTAT,))
        bq(0, 1, 14, 4.0)
        wk(0, 1, 14, OFFLIN, KAPPA)
        rel(0, 1, same=0.0, sink=-M / 2)
        for i, (src, dst) in enumerate(((SOK, BSOK), (SFAIL, BSFAIL))):
            wv(0, 1, i, src, 1.0)
            wo(0, 1, i, dst, 1.0)
        # H0.2: the token exactly one before me: its identity bits and flags
        code_match(0, 2, 1, A_OFF)
        rel(0, 2, same=0.0, sink=SINK_OFF)
        for b in range(13):
            wv(0, 2, b, BITS + b, 1.0)
            wo(0, 2, b, PBITS + b, 1.0)
        for hd, (src, dst) in enumerate(((HASHED, PHASHED), (GT, PGT), (ISSTAT, PISSTAT)), start=13):
            wv(0, 2, hd, src, 1.0)
            wo(0, 2, hd, dst, 1.0)
        # H0.3: any hunk-line marker before me in my block -> HASHUNK
        gate(0, 3, (NLAT,))
        rel(0, 3, same=0.0, sink=-M / 2)
        wv(0, 3, 0, NLAT, 1.0)
        wo(0, 3, 0, HASHUNK, 1.0)
        # H0.4: intent token = first hashed token at offset >= 10, else first other non-reserved
        gate(0, 4, (NONRES, OFFGE10))
        bq(0, 4, 0, 4.0 * B_HASH)
        wk(0, 4, 0, HASHED, 1.0)
        bq(0, 4, 1, -4.0 * KAP2)
        wk(0, 4, 1, OFFLIN, 1.0)
        rel(0, 4, same=0.0, sink=-M / 2)
        for b in range(13):
            wv(0, 4, b, BITS + b, 1.0)
            wo(0, 4, b, IBITS + b, 1.0)
        # MLP 0: VM = clip(val + has_hunk), eligibility for the nearest validation/hunk lookup
        clip_unit(0, {BTVAL: 1.0, HASHUNK: 1.0}, 0.0, {VM: 1.0})

        # ================================ layer 1 ==============================================
        # H1.0: my id bits against the h_old ids carried by '>' tokens of blocks 1..7 back
        for b in range(13):
            wq(1, 0, b, BITS + b, 4.0 * A_BITS)
            wk(1, 0, b, PBITS + b, 1.0)
        gate(1, 0, (PHASHED, GT))
        rel(1, 0, same=-M, near=lambda c: 0.0, far=-M, sink=12.0 * A_BITS)
        wv(1, 0, 0, GT, 1.0)
        wo(1, 0, 0, REVMATCH, 1.0)
        # H1.1: the last digit group that follows a status word -> PV (passed count, <= 100)
        gate(1, 1, (DIGIT, PISSTAT))
        bq(1, 1, 14, 4.0)
        wk(1, 1, 14, OFFLIN, KAPPA)
        rel(1, 1, same=0.0, sink=-M / 2)
        wv(1, 1, 0, NUM, 1.0)
        wo(1, 1, 0, PV, 1.0)
        # H1.2: the nearest earlier block with the same intent and class: exists, its status
        nearest_same_key(1, 2, ((EOE, EXISTS), (BSOK, PSOK), (BSFAIL, PSFAIL)))
        # H1.3: a same-key block within 1..7 back exists -> EXISTS7
        intent_match(1, 3, A_INT)
        gate(1, 3, (EOE,))
        rel(1, 3, same=-M, near=lambda c: 0.0, far=-M, sink=SINK_INT7)
        wv(1, 3, 0, EOE, 1.0)
        wo(1, 3, 0, EXISTS7, 1.0)
        # H1.4: count of earlier same-key failing blocks (BOS anchor) -> OUTFAIL, ANCH
        intent_match(1, 4, A_INT)
        gate(1, 4, (EOE,))
        rel(1, 4, same=-M, near=lambda c: 0.0, far=None, sink=15.0 * A_INT, upto=model.R - 1)
        wv(1, 4, 0, BSFAIL, 1.0)
        wo(1, 4, 0, OUTFAIL, 1.0)
        wv(1, 4, 1, BOS, 1.0)
        wo(1, 4, 1, ANCH, 1.0)
        # H1.5: nearest earlier block that is a validation (+1) or shows a hunk line (-1)
        gate(1, 5, (EOE, VM))
        rel(1, 5, same=-M, near=lambda c: -LAM * c, far=None, sink=-64.0 * LAM, upto=model.R - 1)
        wv(1, 5, 0, BTVAL, 1.0)
        wv(1, 5, 0, HASHUNK, -1.0)
        wo(1, 5, 0, NEAREST, 1.0)
        # MLP 1
        unit(1, {BSOK: 1.0, PSOK: -1.0}, 0.0, {NOTIDST: 1.0})
        unit(1, {PSOK: 1.0, BSOK: -1.0}, 0.0, {NOTIDST: 1.0})
        unit(1, {BSFAIL: 1.0, PSFAIL: -1.0}, 0.0, {NOTIDST: 1.0})
        unit(1, {PSFAIL: 1.0, BSFAIL: -1.0}, 0.0, {NOTIDST: 1.0})
        clip_unit(1, {ANCH: 3.5 * KSTEP, OUTFAIL: -KSTEP}, 0.0, {CNT3: 1.0})
        clip_unit(1, {ANCH: 5.5 * KSTEP, OUTFAIL: -KSTEP}, 0.0, {CNT5: 1.0})
        unit(1, {REVMATCH: 1.0, PGT: 1.0, HASHED: 1.0}, -2.0, {REVTOK: 1.0})
        unit(1, {BTMUT: 1.0, HASHUNK: -1.0, BSFAIL: -1.0}, 0.0, {NOOP: 1.0})
        unit(1, {BTINSP: 1.0, BSOK: 1.0, EXISTS: 1.0}, -2.0, {REDINSP: 1.0})

        # ================================ layer 2 ==============================================
        # H2.0: the nearest earlier same-key block again: its count and layer-1 flags
        nearest_same_key(2, 0, ((PV, PPV), (EXISTS7, PEX7), (PSFAIL, PPSFAIL), (NEAREST, PNEAR)))
        # H2.1: the largest passed count among earlier blocks with the same intent id -> MAXPV
        intent_match(2, 1, A_INT, with_class=False)
        bq(2, 1, 13, 4.0 * LAMP)
        wk(2, 1, 13, PV, 1.0)
        gate(2, 1, (EOE,))
        rel(2, 1, same=-M, near=lambda c: 0.0, far=None, sink=SINK_MAXPV, upto=model.R - 1)
        wv(2, 1, 0, PV, 1.0)
        wo(2, 1, 0, MAXPV, 1.0)
        # H2.2: the nearest earlier validation block (any suite): exists, same status as its own
        # predecessor?
        gate(2, 2, (EOE, BTVAL))
        rel(2, 2, same=-M, near=lambda c: -LAM * c, far=None, sink=-64.0 * LAM, upto=model.R - 1)
        for hd, (src, dst) in enumerate(((EXISTS, NVEX), (NOTIDST, NVNOTST))):
            wv(2, 2, hd, src, 1.0)
            wo(2, 2, hd, dst, 1.0)
        # H2.3: any reverting h_new token in my block -> REVBLK
        gate(2, 3, (REVTOK,))
        rel(2, 3, same=0.0, sink=-M / 2)
        wv(2, 3, 0, REVTOK, 1.0)
        wo(2, 3, 0, REVBLK, 1.0)
        # MLP 2: block effect flags (the SI terms are mutually exclusive)
        better = {PV: KSTEP, MAXPV: -KSTEP}
        gated_clip(2, {BSFAIL: 1.0, EXISTS: 1.0}, 2, better, -0.5 * KSTEP, {IMPROVED: 1.0})
        gated_clip(2, {BSOK: 1.0, EXISTS: 1.0, PSFAIL: -1.0}, 2, better, -0.5 * KSTEP, {IMPROVED: 1.0})
        clip_unit(2, {REVBLK: KSTEP}, -KSTEP / 2, {SI: 1.0})
        unit(2, {BTREVERT: 1.0, REVBLK: -1.0}, 0.0, {SI: 1.0})
        unit(2, {NOOP: 1.0}, 0.0, {SI: 1.0})
        unit(2, {REDINSP: 1.0}, 0.0, {SI: 1.0})
        unit(2, {BTMUT: 1.0, HASHUNK: 1.0, BSFAIL: -1.0, NVEX: 1.0, NVNOTST: -1.0, REVBLK: -1.0}, -2.0, {SI: 1.0})
        # stagnant validation: same status as the previous same-key block, no new high, and not
        # itself a reversion (a block with reverting hunk lines is counted once, by REVBLK)
        gated_clip(2, {BTVAL: 1.0, EXISTS: 1.0, NOTIDST: -1.0, REVBLK: -1.0}, 2, {MAXPV: KSTEP, PV: -KSTEP},
                   0.5 * KSTEP, {SI: 1.0})
        # in-policy validation retry (at least as many passes as the predecessor, fewer than 4
        # earlier same-key failures): it is not stall-ineffective
        rv = {BTVAL: 1.0, BSFAIL: 1.0, EXISTS7: 1.0, PSFAIL: 1.0, NEAREST: 1.0, CNT3: 1.0, REVBLK: -1.0}
        gated_clip(2, rv, 6, {PV: KSTEP, PPV: -KSTEP}, 0.5 * KSTEP, {INPOL: 1.0, SI: -1.0})
        # network / inspection retries
        unit(2, {BTHTTP: 1.0, BTINSP: 1.0, BSFAIL: 1.0, EXISTS7: 1.0, PSFAIL: 1.0, CNT3: 1.0}, -4.0, {INPOL: 1.0})
        unit(2, {BTHTTP: 1.0, BTINSP: 1.0, BSFAIL: 1.0, EXISTS7: 1.0, PSFAIL: 1.0, CNT3: -1.0}, -3.0, {SI: 1.0})
        unit(2, {BTPOLL: 1.0, BSFAIL: 1.0, EXISTS7: 1.0, PSFAIL: 1.0, CNT5: 1.0}, -4.0, {INPOL: 1.0})
        unit(2, {BTPOLL: 1.0, BSFAIL: 1.0, EXISTS7: 1.0, PSFAIL: 1.0, CNT5: -1.0}, -3.0, {SI: 1.0})
        # resolutions
        unit(2, {BTVAL: 1.0, BSOK: 1.0, PSFAIL: 1.0, PEX7: 1.0, PPSFAIL: 1.0, PNEAR: 1.0}, -5.0, {RESOLV: 1.0})
        unit(2, {BTHTTP: 1.0, BTPOLL: 1.0, BTINSP: 1.0, BSOK: 1.0, PSFAIL: 1.0}, -2.0, {RESOLV: 1.0})

        # ================================ layer 3 ==============================================
        # H3.0: average over the EOEs 0..7 back with BOS as anchor
        gate(3, 0, (EOE,))
        rel(3, 0, same=0.0, near=lambda c: 0.0, far=-M, sink=0.0)
        for hd, (src, dst) in enumerate(((SI, AVGSI), (REVBLK, AVGREV), (IMPROVED, AVGIMP), (BOS, ANCH8))):
            wv(3, 0, hd, src, 1.0)
            wo(3, 0, hd, dst, 1.0)
        # H3.1: the nearest failing or resolving block (whole prefix): is it an in-policy retry?
        gate(3, 1, (EOE,), any_of=(BSFAIL, RESOLV))
        rel(3, 1, same=0.0, near=lambda c: -LAM * c, far=None, sink=-64.0 * LAM, upto=model.R - 1)
        wv(3, 1, 0, INPOL, 1.0)
        wo(3, 1, 0, ACT, 1.0)
        # H3.2: a resolving block among the last four -> REC4
        gate(3, 2, (EOE, RESOLV))
        rel(3, 2, same=0.0, near=lambda c: 0.0 if c <= 3 else -M, far=-M, sink=-M / 2)
        wv(3, 2, 0, RESOLV, 1.0)
        wo(3, 2, 0, REC4, 1.0)
        # MLP 3: decision bits
        clip_unit(3, {AVGREV: KSTEP, ANCH8: -1.5 * KSTEP}, 0.0, {V2: 1.0})
        clip_unit(3, {AVGSI: KSTEP, ANCH8: -4.5 * KSTEP}, 0.0, {I5: 1.0})
        clip_unit(3, {AVGIMP: KSTEP, ANCH8: -0.5 * KSTEP}, 0.0, {PROG: 1.0})
        clip_unit(3, {ACT: 1.0, REC4: 1.0}, 0.0, {RECB: 1.0})

        # ================================ head =================================================
        W, b = m.head.weight, m.head.bias
        W[3, V2] = 60.0
        W[3, PROG] = -100.0
        b[3] = -5.0
        W[2, I5] = 50.0
        W[2, SI] = 50.0
        W[2, ACT] = -100.0
        W[2, PROG] = -100.0
        b[2] = -80.0
        W[1, RECB] = 15.0
        b[1] = -5.0
    return m
