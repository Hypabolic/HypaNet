"""Hand-written weights for the v0.3 doom-loop transformer (Phase A, zero labels).

Every parameter is an explicit constant or a closed-form rule; the circuit implements
reference_policy.py and is documented head by head in CIRCUITS.md.  Scores are q.k/4 + B, so
query weights carry a factor 4 and the comments quote scores directly.  Head dim 15 is a key
gate (-M per unmet condition, +M*is_bos), so the BOS key scores exactly its bias entry: BOS is
the sink or anchor of every head.
"""

import math

import torch

import model
import tokenizer

# ---- residual-subspace allocation (see CIRCUITS.md) --------------------------------------------
BOS, EOE, NLAT, HASHED, H0, PASSEDW, DIGIT, NUM = 0, 1, 2, 3, 4, 5, 6, 7
BITS = 8  # 8..20: 13 sign bits of the hashed word id
ISSTAT, SOK, SFAIL, SFAILED = 21, 22, 23, 24
TINSP, TVAL, TMUT, TREVERT, THTTP, TPOLL, TSEARCH = 25, 26, 27, 28, 29, 30, 31
CODE = 32  # 32..45: 7 frequencies x (cos, sin) of the within-block offset
OFFLIN, OFF7, OFF10, OFF13, OFF14 = 46, 47, 48, 49, 50
HOLD, HNEW = 51, 52
BTINSP, BTVAL, BTMUT, BTREVERT, BTHTTP, BTPOLL, BTSEARCH = 53, 54, 55, 56, 57, 58, 59
BSOK, BSFAIL, BSFAILED = 60, 61, 62
HASHUNK, HASPASSED = 63, 64
CODEP = 65  # 65..78: offset code of (passed count position), fetched into the EOE
SLOT, VM = 79, 80
REVMATCH, PV, NEAREST = 81, 82, 83
REVTOK, REDVAL = 84, 85
OUTALL, OUTFAIL, ANCH2, REVBLK = 86, 87, 88, 89
SI, INPOL, RESOLV, KILL = 90, 91, 92, 110
AVGSI, AVGREV, ANCH8, NEAR4 = 93, 94, 95, 96
LPV, LFAILED, LOK, FPV, FFAILED, FOK = 98, 99, 100, 101, 102, 103
V2, I5, A4, PROGN, PROGF, SIT = 104, 105, 106, 107, 108, 109

NFREQ = 7
M = 1.0e4  # gate magnitude
KAPPA = 4000.0  # "latest offset wins" slope (per unit of offset/192)
LAM = 20.0  # nearest / farthest block slope per event distance
A_OFF = 30.0  # offset-code match scale: exact 7*A, worst other 5.27*A (d=162), sink 6.15*A
A_BITS = 20.0  # id-bit match scale (13 bits): match 13*A, other <= 11*A, sink 12*A
A_INT = 20.0  # intent match scale (13 bits + 2 class bits): match 15*A, other <= 13*A
KSTEP = 40.0  # threshold sharpness of count units
KGATE = 1000.0  # conjunction gate of MLP units
DH = model.DH
GATE = 15  # head dim used as the key gate

TOOL_WORDS = ("read_file", "search", "list_dir", "edit_file", "write_file", "revert", "run_tests",
              "build", "lint", "http_call", "poll_status", "plan", "ask", "yield")
STATUS_WORDS = ("ok", "error", "FAILED", "ready", "pending")


def _freqs():
    return [math.pi / (1.5 ** i) for i in range(NFREQ)]


def _code(o):
    """Offset code of offset o (14 values)."""
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
        hashed = torch.arange(tokenizer.HASH_BASE, tokenizer.V)
        E[hashed, HASHED] = 1.0
        for b in range(13):
            E[hashed, BITS + b] = (((hashed - tokenizer.HASH_BASE) >> b) & 1).to(torch.float32) * 2.0 - 1.0
        E[tokenizer.word_id("h0"), H0] = 1.0
        E[tokenizer.RESERVED_IDS["passed"], PASSEDW] = 1.0
        digits = torch.arange(tokenizer.DIGIT_BASE, tokenizer.DIGIT_BASE + 1110)
        E[digits, DIGIT] = 1.0
        d1 = torch.arange(10)
        E[tokenizer.DIGIT_BASE + d1, NUM] = d1.to(torch.float32) / 100.0
        d2 = torch.arange(100)
        E[tokenizer.DIGIT_BASE + 10 + d2, NUM] = d2.to(torch.float32) / 100.0
        d3 = torch.arange(1000)
        E[tokenizer.DIGIT_BASE + 110 + d3, NUM] = d3.to(torch.float32) / 100.0
        for w in STATUS_WORDS:
            E[tokenizer.RESERVED_IDS[w], ISSTAT] = 1.0
        for w in ("ok", "ready"):
            E[tokenizer.RESERVED_IDS[w], SOK] = 1.0
        for w in ("error", "pending"):
            E[tokenizer.RESERVED_IDS[w], SFAIL] = 1.0
        E[tokenizer.RESERVED_IDS["FAILED"], SFAILED] = 1.0
        for w in ("read_file", "search", "list_dir"):
            E[tokenizer.RESERVED_IDS[w], TINSP] = 1.0
        for w in ("run_tests", "build", "lint"):
            E[tokenizer.RESERVED_IDS[w], TVAL] = 1.0
        for w in ("edit_file", "write_file", "revert"):
            E[tokenizer.RESERVED_IDS[w], TMUT] = 1.0
        E[tokenizer.RESERVED_IDS["revert"], TREVERT] = 1.0
        E[tokenizer.RESERVED_IDS["http_call"], THTTP] = 1.0
        E[tokenizer.RESERVED_IDS["poll_status"], TPOLL] = 1.0
        E[tokenizer.RESERVED_IDS["search"], TSEARCH] = 1.0

        # ---- offset embedding ----------------------------------------------------------------
        for o in range(model.O):
            code = _code(o)
            for i in range(2 * NFREQ):
                P[o, CODE + i] = code[i]
            P[o, OFFLIN] = o / 192.0
        P[7, OFF7] = 1.0
        P[10, OFF10] = 1.0
        P[13, OFF13] = 1.0
        P[14, OFF14] = 1.0

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

        def gate(l, h, conds, extra_neg=()):
            """Key gate: -M per unmet condition (flag dims that must be 1), -M per dim in
            extra_neg that must be 0; BOS scores exactly 0 on the gate."""
            bq(l, h, GATE, 4.0)
            n = len(conds)
            for dim in conds:
                wk(l, h, GATE, dim, M)
            for dim in extra_neg:
                wk(l, h, GATE, dim, -M)
            bk(l, h, GATE, -n * M)
            wk(l, h, GATE, BOS, n * M)

        def rel(l, h, same=None, far=None, near=None, sink=0.0):
            """Event-distance bias row: same-block entry, entries 1..7, 8..63, and the BOS column."""
            rb = layer(l).attn.rel_bias
            rb[h, :] = -M
            if same is not None:
                rb[h, 0] = same
            if near is not None:
                for c in range(1, 8):
                    rb[h, c] = near(c)
            if far is not None:
                for c in range(8, model.R):
                    rb[h, c] = far
            rb[h, model.R] = sink

        def code_match(l, h, qdim, delta, scale):
            """q = 4*scale*code(query) ; k = R_delta code(key) with the BOS key nulled.
            Score = scale * K(o_q - o_k - delta); exact hit = 7*scale."""
            for i in range(2 * NFREQ):
                wq(l, h, i, qdim + i, 4.0 * scale)
            c0 = _code(0)
            for r, c, v in _rot(delta):
                wk(l, h, r, CODE + c, v)
            rc0 = [0.0] * (2 * NFREQ)
            for r, c, v in _rot(delta):
                rc0[r] += v * c0[c]
            for i in range(2 * NFREQ):
                wk(l, h, i, BOS, -rc0[i])

        def bits_match(l, h, scale):
            for b in range(13):
                wq(l, h, b, BITS + b, 4.0 * scale)
                wk(l, h, b, BITS + b, 1.0)

        unit_counter = [0] * model.L

        def unit(l, inputs, bias, outputs):
            """One ReLU unit: inputs {dim: w}, bias, outputs {dim: w}."""
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

        SINK_OFF = (7.0 + 5.3) / 2.0 * A_OFF  # between an exact offset hit (7A) and any miss (<=5.27A)

        # ================================ layer 0 ==============================================
        # H0.0 / H0.1: "is there a hunk-line marker exactly 2 / 4 tokens before me" -> HOLD / HNEW
        for h, delta, out in ((0, 2, HOLD), (1, 4, HNEW)):
            code_match(0, h, CODE, delta, A_OFF)
            gate(0, h, (NLAT,))
            rel(0, h, same=0.0, sink=SINK_OFF)
            wv(0, h, 0, NLAT, 1.0)
            wo(0, h, 0, out, 1.0)
        # H0.2: copy the tool flags of the token at offset 7 to every token of the block
        gate(0, 2, (OFF7,))
        rel(0, 2, same=0.0, sink=-M / 2)
        for i, (src, dst) in enumerate(((TINSP, BTINSP), (TVAL, BTVAL), (TMUT, BTMUT), (TREVERT, BTREVERT),
                                        (THTTP, BTHTTP), (TPOLL, BTPOLL), (TSEARCH, BTSEARCH))):
            wv(0, 2, i, src, 1.0)
            wo(0, 2, i, dst, 1.0)
        # H0.3: the last status word before me (largest offset wins) -> block status
        gate(0, 3, (ISSTAT,))
        bq(0, 3, 14, 4.0)
        wk(0, 3, 14, OFFLIN, KAPPA)
        rel(0, 3, same=0.0, sink=-M / 2)
        for i, (src, dst) in enumerate(((SOK, BSOK), (SFAIL, BSFAIL), (SFAILED, BSFAILED))):
            wv(0, 3, i, src, 1.0)
            wo(0, 3, i, dst, 1.0)
        # H0.4: the last 'passed' word: fetch code(offset - 3) (the passed count's position)
        gate(0, 4, (PASSEDW,))
        bq(0, 4, 14, 4.0)
        wk(0, 4, 14, OFFLIN, KAPPA)
        rel(0, 4, same=0.0, sink=-M / 2)
        c0 = _code(0)
        rc0 = [0.0] * (2 * NFREQ)
        for r, c, v in _rot(-3):
            wv(0, 4, r, CODE + c, v)
            rc0[r] += v * c0[c]
        for i in range(2 * NFREQ):
            wv(0, 4, i, BOS, -rc0[i])
            wo(0, 4, i, CODEP + i, 1.0)
        wv(0, 4, 14, PASSEDW, 1.0)
        wo(0, 4, 14, HASPASSED, 1.0)
        # H0.5: any hunk-line marker before me in my block -> HASHUNK
        gate(0, 5, (NLAT,))
        rel(0, 5, same=0.0, sink=-M / 2)
        wv(0, 5, 0, NLAT, 1.0)
        wo(0, 5, 0, HASHUNK, 1.0)
        # MLP 0: intent slot = off10 & ~http & ~search | off13 & http | off14 & search
        unit(0, {OFF10: 1.0, BTHTTP: -1.0, BTSEARCH: -1.0}, 0.0, {SLOT: 1.0})
        unit(0, {OFF13: 1.0, BTHTTP: 1.0}, -1.0, {SLOT: 1.0})
        unit(0, {OFF14: 1.0, BTSEARCH: 1.0}, -1.0, {SLOT: 1.0})
        # VM = clip(val + has_hunk): eligible for the nearest-validation/hunk lookup
        clip_unit(0, {BTVAL: 1.0, HASHUNK: 1.0}, 0.0, {VM: 1.0})

        # ================================ layer 1 ==============================================
        # H1.0: my id bits against the h_old ids of blocks 1..7 back -> REVMATCH
        bits_match(1, 0, A_BITS)
        gate(1, 0, (HOLD, HASHED), extra_neg=(H0,))
        rel(1, 0, same=-M, near=lambda c: 0.0, far=-M, sink=12.0 * A_BITS)
        wv(1, 0, 0, HOLD, 1.0)
        wo(1, 0, 0, REVMATCH, 1.0)
        # H1.1: gather the intent token's id bits into the EOE (and every later token)
        gate(1, 1, (SLOT,))
        rel(1, 1, same=0.0, sink=-M / 2)
        for b in range(13):
            wv(1, 1, b, BITS + b, 1.0)
            wo(1, 1, b, BITS + b, 1.0)
        # H1.2: the digit token at the fetched position (passed count) -> PV
        code_match(1, 2, CODEP, 0, A_OFF)
        gate(1, 2, (DIGIT,))
        rel(1, 2, same=0.0, sink=SINK_OFF)
        wv(1, 2, 0, NUM, 1.0)
        wo(1, 2, 0, PV, 1.0)
        # H1.3: nearest earlier block that is a validation (+1) or shows a hunk line (-1)
        gate(1, 3, (EOE, VM))
        rel(1, 3, same=-M, near=lambda c: -LAM * c, far=-M, sink=-8.0 * LAM)
        wv(1, 3, 0, BTVAL, 1.0)
        wv(1, 3, 0, HASHUNK, -1.0)
        wo(1, 3, 0, NEAREST, 1.0)
        # MLP 1
        unit(1, {REVMATCH: 1.0, HNEW: 1.0, HASHED: 1.0}, -2.0, {REVTOK: 1.0})
        unit(1, {BTVAL: 2.0, NEAREST: 2.0}, -3.0, {REDVAL: 1.0})

        # ================================ layer 2 ==============================================
        # H2.0: count earlier EOEs (1..7 back) with the same intent (id bits + tool class)
        bits_match(2, 0, A_INT)
        cls1 = {BTINSP: 1.0, BTMUT: 1.0, BTVAL: -1.0, BTHTTP: -1.0, BTPOLL: -1.0}
        cls2 = {BTINSP: 1.0, BTMUT: -1.0, BTVAL: 1.0, BTHTTP: -1.0, BTPOLL: -1.0}
        for hd, cls in ((13, cls1), (14, cls2)):
            for dim, w in cls.items():
                wq(2, 0, hd, dim, 4.0 * A_INT * w)
                wk(2, 0, hd, dim, w)
        gate(2, 0, (EOE,))
        rel(2, 0, same=-M, near=lambda c: 0.0, far=-M, sink=15.0 * A_INT)
        wv(2, 0, 0, EOE, 1.0)
        wo(2, 0, 0, OUTALL, 1.0)
        wv(2, 0, 1, BSFAIL, 1.0)
        wo(2, 0, 1, OUTFAIL, 1.0)
        wv(2, 0, 2, BOS, 1.0)
        wo(2, 0, 2, ANCH2, 1.0)
        # H2.1: any REVTOK token in my block -> REVBLK
        gate(2, 1, (REVTOK,))
        rel(2, 1, same=0.0, sink=-M / 2)
        wv(2, 1, 0, REVTOK, 1.0)
        wo(2, 1, 0, REVBLK, 1.0)
        # MLP 2: per-block effect flags (mutually exclusive terms summed into SI / INPOL)
        clip_unit(2, {REVBLK: KSTEP}, -KSTEP / 2, {SI: 1.0})
        unit(2, {BTREVERT: 1.0, REVBLK: -1.0}, 0.0, {SI: 1.0})
        unit(2, {BTMUT: 1.0, HASHUNK: -1.0, BSFAIL: -1.0}, 0.0, {SI: 1.0})
        unit(2, {REDVAL: 1.0}, 0.0, {SI: 1.0})
        clip_unit(2, {OUTALL: KSTEP, OUTFAIL: -KSTEP, ANCH2: -0.5 * KSTEP, BTINSP: KGATE}, -KGATE, {SI: 1.0})
        clip_unit(2, {OUTFAIL: KSTEP, ANCH2: -3.5 * KSTEP, BSFAIL: KGATE, BTPOLL: -KGATE}, -KGATE, {SI: 1.0, KILL: 1.0})
        clip_unit(2, {OUTFAIL: KSTEP, ANCH2: -4.5 * KSTEP, BSFAIL: KGATE, BTPOLL: KGATE}, -2 * KGATE, {SI: 1.0, KILL: 1.0})
        clip_unit(2, {OUTFAIL: -KSTEP, ANCH2: 3.5 * KSTEP, BSFAIL: KGATE, BTPOLL: -KGATE}, -KGATE, {INPOL: 1.0})
        clip_unit(2, {OUTFAIL: -KSTEP, ANCH2: 4.5 * KSTEP, BSFAIL: KGATE, BTPOLL: KGATE}, -2 * KGATE, {INPOL: 1.0})
        clip_unit(2, {OUTFAIL: KSTEP, ANCH2: -0.5 * KSTEP, BSOK: KGATE}, -KGATE, {RESOLV: 1.0})

        # ================================ layer 3 ==============================================
        # H3.0: average over the EOEs 0..7 back with BOS as anchor -> AVGSI, AVGREV, ANCH8
        gate(3, 0, (EOE,))
        rel(3, 0, same=0.0, near=lambda c: 0.0, far=-M, sink=0.0)
        for hd, src, dst in ((0, SI, AVGSI), (1, REVBLK, AVGREV), (2, BOS, ANCH8)):
            wv(3, 0, hd, src, 1.0)
            wo(3, 0, hd, dst, 1.0)
        # H3.1: the nearest EOE 0..3 back that is an in-policy failure / resolution (+1) or an
        # out-of-policy retry (-1); INPOL, RESOLV and KILL are mutually exclusive so their sum is
        # the eligibility flag -> NEAR4 in {+1, -1, 0}
        gate(3, 1, (EOE, INPOL))
        wk(3, 1, GATE, RESOLV, M)
        wk(3, 1, GATE, KILL, M)
        rel(3, 1, same=0.0, near=lambda c: -LAM * c if c <= 3 else -M, far=-M, sink=-4.0 * LAM)
        wv(3, 1, 0, INPOL, 1.0)
        wv(3, 1, 0, RESOLV, 1.0)
        wv(3, 1, 0, KILL, -1.0)
        wo(3, 1, 0, NEAR4, 1.0)
        # H3.2 / H3.3: latest / earliest validation block in the window
        for h, near, sink, dsts in ((2, lambda c: -LAM * c, -8.0 * LAM, (LPV, LFAILED, LOK)),
                                    (3, lambda c: LAM * (c + 1), 0.0, (FPV, FFAILED, FOK))):
            gate(3, h, (EOE, HASPASSED))
            rel(3, h, same=near(0), near=near, far=-M, sink=sink)
            for hd, src, dst in ((0, PV, dsts[0]), (1, BSFAILED, dsts[1]), (2, BSOK, dsts[2])):
                wv(3, h, hd, src, 1.0)
                wo(3, h, hd, dst, 1.0)
        # MLP 3: decision bits
        clip_unit(3, {AVGREV: KSTEP, ANCH8: -1.5 * KSTEP}, 0.0, {V2: 1.0})
        clip_unit(3, {AVGSI: KSTEP, ANCH8: -4.5 * KSTEP}, 0.0, {I5: 1.0})
        clip_unit(3, {NEAR4: KSTEP}, -0.5 * KSTEP, {A4: 1.0})
        clip_unit(3, {LPV: 400.0, FPV: -400.0}, -2.0, {PROGN: 1.0})
        unit(3, {FFAILED: 1.0, LOK: 1.0}, -1.0, {PROGF: 1.0})
        clip_unit(3, {SI: 1.0}, 0.0, {SIT: 1.0})

        # ================================ head =================================================
        W, b = m.head.weight, m.head.bias
        W[3, V2] = 60.0
        W[3, PROGN] = -100.0
        W[3, PROGF] = -100.0
        b[3] = -5.0
        W[2, I5] = 50.0
        W[2, SIT] = 50.0
        W[2, INPOL] = -100.0
        W[2, PROGN] = -100.0
        W[2, PROGF] = -100.0
        b[2] = -80.0
        W[1, A4] = 15.0
        b[1] = -5.0
    return m
