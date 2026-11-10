#!/usr/bin/env python3
"""Builds the hand-written slop network: vocab.txt, artifact.npz, meta.json (and directions.json for ./inspect).

    ./python3 work/build.py

Reads only work/lexicon/*.txt (hand-written word lists) and kit/lib. Every number below is a literal chosen by
reasoning (see the comment beside it and work/MODEL.md), a formula, or a construction from those lists.

Mechanism (details in MODEL.md):
  L1 attention  token-level concept matching against the nodes just before / after (code match, BOS as threshold);
                reverse matches for the pointer; per-node sums of lexical classes and of the concept codes at EON.
  L1 MLP        per token: "novel" (informative and not covered), role-gated anchor hits, matched code;
                per EON: static booleans, squared length of the code sum.
  L2 attention  per-node sums of the token features; duplicate head (EON -> earlier EON with the nearest bag of
                concepts, subtracts its code sum).
  L2 MLP        EON booleans from the sums; L1 distance of the duplicate comparison.
  L3 attention  the pointer head (attention pattern only).
  L3 MLP        rule units (AND of booleans with vetoes) -> class evidence.
  L4 MLP        caps on the class evidence.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "kit" / "lib"))
import model as mdl  # noqa: E402
import tokenizer as tk  # noqa: E402

LEX = WORK / "lexicon"

# ---------------------------------------------------------------------------------------------------------------
# 1. Lexicon -> vocabulary and per-stem attributes
# ---------------------------------------------------------------------------------------------------------------


def read_words(name):
    return (LEX / name).read_text(encoding="utf-8").split()


def stem_of(word):
    toks = tk.tokenize_text(word).tokens
    if len(toks) != 1 or toks[0].kind != "word":
        raise SystemExit(f"lexicon word {word!r} does not tokenise to one word")
    return toks[0].stem


CLASS_FILES = {  # file -> attribute name
    "function.txt": "function",
    "generic.txt": "generic",
    "semi.txt": "semi",
    "deictic.txt": "deic",
    "uinoun.txt": "uin",
    "action.txt": "act",
    "gesture.txt": "act uin",  # scroll, swipe: an action that needs no other interface noun to be narration
    "promo.txt": "promo",
    "narrverb.txt": "narr",
    "genre.txt": "genre",
    "status.txt": "status",
    "summary.txt": "summ",
    "unit.txt": "unit",
    "dem.txt": "dem",
    "timeword.txt": "timeword",
}

# Number words share the concept code of the digit token ("Three days before" = "3 days before").
# "one" is left out: its stem is "on", the preposition.
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12,
                "second": 2, "third": 3, "fourth": 4, "fifth": 5}  # ordinals ("a second check" = "two-step")
COMPASS = ["n", "s", "e", "w"]  # the letter after a coordinate ("45.6442° N")

vocab = []  # stems in first-seen order
attrs = {}  # stem -> set of attribute names
group_of = {}  # stem -> first stem of its synonym group
numword_of = {}  # stem -> number
notes = []


def add_stem(st):
    if st not in attrs:
        attrs[st] = set()
        vocab.append(st)
    return st


for fname, attr in CLASS_FILES.items():
    for w in read_words(fname):
        attrs[add_stem(stem_of(w))].update(attr.split())

for line in (LEX / "groups.txt").read_text(encoding="utf-8").splitlines():
    ws = line.split()
    if not ws:
        continue
    head = None
    for w in ws:
        st = add_stem(stem_of(w))
        if head is None:
            head = group_of.get(st, st)
        if st in group_of and group_of[st] != head:
            notes.append(f"group conflict: {w!r} ({st}) already in group {group_of[st]!r}, not moved to {head!r}")
            continue
        group_of[st] = head

for w, n in NUMBER_WORDS.items():
    st = add_stem(stem_of(w))
    attrs[st].add("numword")
    numword_of[st] = n
for w in COMPASS:
    attrs[add_stem(stem_of(w))].add("coord")

vocab_id = {st: tk.VOCAB_BASE + n for n, st in enumerate(vocab)}
assert len(vocab) <= tk.VOCAB_MAX

# ---------------------------------------------------------------------------------------------------------------
# 2. Residual allocation
# ---------------------------------------------------------------------------------------------------------------

NAMES = []


def alloc(*names):
    out = []
    for n in names:
        out.append(len(NAMES))
        NAMES.append(n)
    return out if len(out) > 1 else out[0]


NCODE = 15
# tokens: +-1 concept code (0 for function words). EON: the node's code sum after L1, and after L2 the
# difference between it and the code sum of the best earlier candidate copy (0 = same bag of concepts).
CODE = [alloc(f"code{i}") for i in range(NCODE)]
ONE, IS_EON, IS_BOS, FIRST = alloc("one", "is_eon", "is_bos", "first")
# static lexical dims: a per-token value at tokens, the per-node sum at EON (written by the L1 count head)
WI, MT, WN, WX = alloc("wi", "mt", "wn|b_xs", "wx")
B_XS = WN  # at EON (where wn is never summed): the node has information besides status words
NUM, VAR, DEIC, UIN, ACT, NARR, STATUS, SUMM, DEM, PROMO, COORD, UNIT = alloc(
    "num", "var", "deic", "uin", "act", "narr", "status", "summ", "dem", "promo", "coord", "unit")
STATIC_COUNTED = [WI, MT, WX, NUM, VAR, DEIC, UIN, ACT, NARR, STATUS, SUMM, DEM, PROMO, COORD, UNIT]
# match outputs (tokens only)
M_P, M_PH, M_PL, M_PC, M_N, M_NC, M_N1H, M_G = alloc("m_p|b_novp2", "m_ph|b_novx1", "m_pl|b_word", "m_pc", "m_n",
                                                    "m_nc", "m_n1h", "m_g")
B_WORD = M_PL  # at EON: the node has a coded word that is not a number
B_NOVP2 = M_P  # at EON (match outputs are 0 there): more than one informative word is left over
B_NOVX1 = M_PH  # at EON: a narrating line names something new (narration novelty above one half-weight word)
# matched code: at a token, its code (12 bits) if the token found its concept in a node this node could be
# restating, else 0; at EON the sum over the node (the pointer's query: "the concepts I repeat")
NMC = 12
MC = [alloc(f"mc{i}") for i in range(NMC)]
# token features computed by the L1 MLP; per-node sums at EON (L2 count head)
NOV_P, NOV_N, NOV_X, HIT_P, HIT_N, HIT_NC, HIT_NC0 = alloc(
    "nov_p", "nov_n", "nov_x", "hit_p", "hit_n", "hit_nc", "hit_nc0")
DYN_COUNTED = [NOV_P, NOV_N, NOV_X, HIT_P, HIT_N, HIT_NC, HIT_NC0]
# role groups (role embedding; on every token of the node and its EON)
(G_EYEBROW, G_SUB, G_HELP, G_PH, G_TIP, G_BODY, G_LI, G_SBODY, G_LABEL, G_H1, G_H2, G_LINK, G_BADGE, R_BUTTON,
 R_UNFL, R_ERR) = alloc("g_eyebrow", "g_sub", "g_help", "g_ph", "g_tip", "g_body", "g_li", "g_sbody", "g_label",
                        "g_h1", "g_h2", "g_link", "g_badge", "r_button", "r_unfl", "r_err")
# flags
(F_DYN, F_STF, F_COND, F_NAVTAB, F_PILL, F_SR, F_LAST, F_CARD, F_CTL, F_TRK, F_SMALL, F_STATE, F_REP) = alloc(
    "f_dyn", "f_stf", "f_cond", "f_navtab", "f_pill", "f_sr", "f_last", "f_card", "f_ctl", "f_trk", "f_small",
    "f_state", "f_rep")
# EON booleans, static (L1 MLP)
(B_VAR, B_WX, B_WX2, B_STATUS, B_COORD, B_DEIC, B_UINNUM, B_UIN, B_UIN2, B_ACT, B_XP, B_DEM, B_NUM, B_SUMM,
 B_MT, B_MT2, B_WI2U, B_WIHI, B_PILLISH, B_WI1, B_SUMM2, B_WI3) = alloc(
    "b_var", "b_wx", "b_wx2", "b_status", "b_coord", "b_deic", "b_uinnum", "b_uin", "b_uin2", "b_act", "b_xp",
    "b_dem", "b_num", "b_summ", "b_mt", "b_mt2", "b_wi2u", "b_wihi", "b_pillish", "b_wi1", "b_summ2|rf",
    "b_wi3|rb")
# At tokens (where booleans are never written) three of these dims carry pointer flags:
RF = B_SUMM2  # this token's concept is repeated by a lower-rank or duplicate-class node later on the screen
RB = B_WI3  # ... by an eyebrow, tooltip or narrating line just before it
F_ANY = B_COORD  # either of the two (clamped to 1)
# EON, dynamic (L2 MLP)
(B_HITP, B_HITN, B_HITNC, B_HITNC0, B_NOVP, B_NOVN, B_NOVN2, B_NOVX2, B_NOVX3, B_NCTX, DL1) = alloc(
    "b_hitp", "b_hitn", "b_hitnc", "b_hitnc0", "b_novp", "b_novn", "b_novn2", "b_novx2", "b_novx3", "b_nctx", "dl1")
O_ECHO, O_NARR, O_DECOR = alloc("o_echo", "o_narr", "o_decor")
NORM2 = alloc("norm2")  # EON: squared length of the node's code sum over the duplicate head's bits
assert len(NAMES) <= mdl.D, len(NAMES)

T = mdl.zero_tensors()
E = T["tok_emb.weight"]

# ---------------------------------------------------------------------------------------------------------------
# 3. Token embedding
# ---------------------------------------------------------------------------------------------------------------


def code_vec(cid):
    """A 15-bit +-1 code, distinct for every code id below 16384.

    The first 14 bits are a bijective scramble of the id (odd multiplier mod 2^14, so neighbouring ids do not get
    neighbouring codes); the 15th is their parity. The matching heads compare the first 14 bits (any two different
    codes differ in at least one of them); the duplicate comparison uses all 15.
    """
    assert 0 <= cid < 16384
    x = (cid * 5167 + 1234) % 16384
    bits = [(x >> i) & 1 for i in range(14)]
    bits.append(sum(bits) & 1)
    return np.array([1.0 if b else -1.0 for b in bits], dtype=np.float64)


def set_code(tid, cid):
    E[tid, CODE] = code_vec(cid)
    E[tid, MT] = 1.0


# specials
E[tk.EON, IS_EON] = 1.0
E[tk.BOS, IS_BOS] = 1.0
# VAR: interpolated runtime text is always new information, and vetoes every rule. ALT also vetoes (K14).
E[tk.VAR, [VAR, WI, WX, WN]] = 1.0
E[tk.ALT, VAR] = 1.0
# glyphs: only SYMBOL matters (the degree sign of a decorative coordinate)
E[tk.GLYPH_IDS["SYMBOL"], COORD] = 1.0
# numbers 0-99: matchable (the same number in two nodes is the same fact), informative when unmatched,
# never counted against narration ("Choose from 4 options below").
for n in range(100):
    tid = tk.NUMBER_BASE + n
    set_code(tid, tid)
    E[tid, [NUM, WI, WX]] = 1.0
# YEAR, NUM, DEC stand for many different values, so they never match anything and are always informative.
for tid in (tk.YEAR, tk.NUM, tk.DEC):
    E[tid, [NUM, WI, WX]] = 1.0
E[tk.DEC, COORD] = 1.0  # the body of a coordinate
# hash buckets: an unknown stem is a content word: own code, fully informative.
for tid in range(tk.HASH_BASE, tk.V):
    set_code(tid, tid)
    E[tid, [WI, WN, WX]] = 1.0
# vocabulary
for st in vocab:
    tid = vocab_id[st]
    a = attrs[st]
    for attr, dim in (("deic", DEIC), ("uin", UIN), ("act", ACT), ("narr", NARR), ("status", STATUS),
                      ("summ", SUMM), ("dem", DEM), ("promo", PROMO), ("coord", COORD), ("unit", UNIT)):
        if attr in a:
            E[tid, dim] = 1.0
    if "function" in a:
        continue  # no code, no weight: ignored by matching and by novelty
    cid = tk.NUMBER_BASE + numword_of[st] if "numword" in a else vocab_id[group_of.get(st, st)]
    set_code(tid, cid)
    if a & {"generic", "deic", "summ"}:
        wi = 0.0  # filler: restating with these adds nothing (a summary cue is not itself information)
    elif "semi" in a:
        wi = 0.5  # half weight: alone it does not make a line informative, with a second one it does
    else:
        wi = 1.0
    if "numword" in a:
        E[tid, NUM] = 1.0
    E[tid, WI] = wi
    # narration novelty: words that only point at the interface do not count as new information
    E[tid, WN] = 0.0 if a & {"uin", "act", "narr", "numword"} else wi
    # class-external information: genre, status and compass words do not count (pure-genre / pure-status rules)
    E[tid, WX] = 0.0 if a & {"genre", "status", "coord"} else wi
    if "timeword" in a:
        # "today", "now": filler inside a sentence (wi = 0), but as a kicker on their own they are a date
        # category (K1: a sidebar group "TODAY"), so they count as class-external information
        E[tid, WX] = 1.0

# ---------------------------------------------------------------------------------------------------------------
# 4. Offset, role and flag embeddings
# ---------------------------------------------------------------------------------------------------------------

T["off_emb.weight"][:, ONE] = 1.0  # a constant 1 at every position except BOS
T["off_emb.weight"][0, FIRST] = 1.0  # the node's first token

ROLE_DIMS = {
    "h1": [G_H1], "dialog_title": [G_H1], "empty_title": [G_H1],
    "h2": [G_H2], "h3": [G_H2], "h4": [G_H2], "h5": [G_H2], "h6": [G_H2],
    "eyebrow": [G_EYEBROW], "subtitle": [G_SUB], "helper": [G_HELP], "caption": [G_HELP],
    "placeholder": [G_PH], "tooltip": [G_TIP], "body": [G_BODY], "list_item": [G_LI],
    "empty_body": [G_SBODY], "dialog_body": [G_SBODY], "error": [G_SBODY, R_ERR],
    "label": [G_LABEL], "link": [G_LINK], "badge": [G_BADGE], "button": [R_BUTTON, R_UNFL],
}
for role, rid in tk.ROLE_IDS.items():
    for d in ROLE_DIMS.get(role, [R_UNFL]):  # every other role is never flagged
        T["role_emb.weight"][rid, d] = 1.0

FLAG_DIMS = {"dynamic": F_DYN, "stateful": F_STF, "conditional": F_COND, "in_nav": F_NAVTAB, "in_table": F_NAVTAB,
             "pill": F_PILL, "dot": F_PILL, "sr_only": F_SR, "section_last": F_LAST, "in_card": F_CARD,
             "in_control": F_CTL, "tracked": F_TRK, "small": F_SMALL, "in_state": F_STATE, "repeated": F_REP}
for flag, d in FLAG_DIMS.items():
    T["flag_emb.weight"][tk.FLAG_IDS[flag], d] = 1.0

# key-side role classes (sums of role dims)
K_HEAD = [G_H1, G_H2]
K_HEADE = [G_H1, G_H2, R_ERR]  # headings, and an error line (the first of two error lines acts as the title)
K_CTRL = [R_BUTTON, G_LINK]
K_LOC = [G_H1, G_H2, G_LABEL, R_BUTTON, G_LINK, G_BODY, R_ERR]  # ranks 1-3: what a lower-rank node can restate
LOWRANK = [G_SUB, G_HELP, G_PH, G_TIP, G_BODY, G_LI, G_SBODY]
NARR_ROLES = [G_SUB, G_HELP, G_SBODY, G_BODY]

# ---------------------------------------------------------------------------------------------------------------
# 5. Builders
# ---------------------------------------------------------------------------------------------------------------

G = 100.0  # gate / veto size: larger than any count (a node has at most 31 tokens)
OFF = -200.0  # attention bias for a distance a matching / count head must not look at (scores there are < 100)
BIG_OFF = -5000.0  # the same for heads whose query-key product is a code-sum dot product (unbounded)
COUNT_W = 1.0e-4  # per-token attention weight of a count head (BOS takes the rest), undone by 1e4 in W_o


class Layer:
    def __init__(self, li):
        self.p = f"layers.{li}."
        self.units = 0

    def q(self, head, hd, dim, val):
        T[self.p + "attn.q.weight"][16 * head + hd, dim] += val

    def qb(self, head, hd, val):
        T[self.p + "attn.q.bias"][16 * head + hd] += val

    def k(self, head, hd, dim, val):
        T[self.p + "attn.k.weight"][16 * head + hd, dim] += val

    def v(self, head, hd, dim, val):
        T[self.p + "attn.v.weight"][16 * head + hd, dim] += val

    def o(self, dim, head, hd, val):
        T[self.p + "attn.o.weight"][dim, 16 * head + hd] += val

    def bias(self, head, cols, val):
        T[self.p + "attn.rel_bias"][head, cols] = val

    def unit(self, win, bias, wout):
        u = self.units
        assert u < mdl.M
        for d, c in win.items():
            T[self.p + "mlp.fc1.weight"][u, d] += c
        T[self.p + "mlp.fc1.bias"][u] = bias
        for d, c in wout.items():
            T[self.p + "mlp.fc2.weight"][d, u] += c
        self.units += 1


def dist_cols(lo, hi):
    """Bias columns for signed node distances lo..hi (63 is the same node)."""
    return list(range(lo + 63, hi + 64))


MATCH_BITS = 14
BIT = 6.0  # score per agreeing code bit (minus the same per disagreeing bit)
FULL = BIT * MATCH_BITS  # 84: identical codes
THETA = 79.0  # BOS: 5 below a full match, 7 above the best non-match (one bit differs: 72), 4 above that + bonus
ANCHOR_BONUS = 3.0  # an anchor (heading, label, control) wins a tie against a body key holding the same word


def match_head(L, head, cols, elig, values, state_pair=False):
    """Token-level concept matching.

    score(query token, key token) = 6 * (#agreeing - #disagreeing of 14 code bits) + eligibility(key)
                                    + bias(distance).
    An identical code scores 84; a different code at most 72 (75 with the anchor bonus); BOS scores 79. So a token
    attends to a matching eligible key when there is one (weight >= 0.993) and to BOS otherwise (leak <= 0.018 per
    one-bit neighbour). Function words, EON and BOS have a zero code and always fall to BOS.
    `elig` is in score units. With `state_pair`, a query outside a state cluster cannot match a key inside one
    (LABELLING 2a: separately mounted states are not visible together).
    """
    a = math.sqrt(4.0 * BIT)
    for i in range(MATCH_BITS):
        L.q(head, i, CODE[i], a)
        L.k(head, i, CODE[i], a)
    if state_pair:
        L.qb(head, 14, 1.0)
        L.q(head, 14, F_STATE, -1.0)  # 1 when the query's node is not in a state cluster
        L.k(head, 14, F_STATE, -240.0)  # -60 for a key inside one
    L.qb(head, 15, 1.0)
    for d, c in elig.items():
        L.k(head, 15, d, 4.0 * c)
    L.bias(head, slice(None), OFF)
    L.bias(head, cols, 0.0)
    L.bias(head, mdl.BOS_COL, THETA)
    for hd, (reads, write) in enumerate(values):
        for d in reads:
            L.v(head, hd, d, 1.0)
        L.o(write, head, hd, 1.0)


def elig(allowed, bonus=(), banned=()):
    e = {ONE: -60.0}
    for d in allowed:
        e[d] = e.get(d, 0.0) + 60.0
    for d in bonus:
        e[d] = e.get(d, 0.0) + ANCHOR_BONUS
    for d in banned:
        e[d] = e.get(d, 0.0) - 60.0
    return e


def count_head(L, head, pairs):
    """EON sums a feature over its own node's tokens: weight 1e-4 per token, the rest on BOS (whose value is 0),
    multiplied back by 1e4 on the way out (error below 0.4% for 31 tokens). Token and BOS queries go to BOS only;
    EON positions are never keys."""
    L.qb(head, 0, 1.0)
    L.k(head, 0, IS_EON, -240.0)  # -60: an EON is not summed
    L.qb(head, 1, 4.0)
    L.q(head, 1, IS_EON, -4.0)  # 4 * (1 - is_eon)
    L.k(head, 1, ONE, -200.0)  # non-EON queries see -200 on every key but BOS
    L.bias(head, slice(None), OFF)
    L.bias(head, 63, math.log(COUNT_W))
    L.bias(head, mdl.BOS_COL, 0.0)
    assert len(pairs) <= 16
    for hd, (read, write) in enumerate(pairs):
        L.v(head, hd, read, 1.0)
        L.o(write, head, hd, 1.0 / COUNT_W)


def tok_unit(L, win, bias, wout):
    """A token-level MLP unit: switched off at EON (where the same dims hold per-node sums)."""
    w = dict(win)
    w[IS_EON] = w.get(IS_EON, 0.0) - G
    L.unit(w, bias, wout)


def ge(L, x, lo, hi, out):
    """EON boolean: 0 when x <= lo, 1 when x >= hi (linear between), from two ReLUs. x is {dim: coef}."""
    s = 1.0 / (hi - lo)
    w = {d: c * s for d, c in x.items()}
    w[IS_EON] = w.get(IS_EON, 0.0) + G
    L.unit(w, -lo * s - G, {out: 1.0})
    L.unit(w, -lo * s - 1.0 - G, {out: -1.0})


def rule(L, pos, need, neg, outs, soft=()):
    """Rule unit: relu(sum(pos) - (need - 1) - G * sum(neg) - sum(soft)): 1 when `need` positive booleans hold,
    no veto is set and the soft terms are 0."""
    w = {}
    for d in pos:
        w[d] = w.get(d, 0.0) + 1.0
    for d in neg:
        w[d] = w.get(d, 0.0) - G
    for d in soft:
        w[d] = w.get(d, 0.0) - 1.0
    L.unit(w, -(need - 1.0), outs)


L1, L2, L3, L4 = Layer(0), Layer(1), Layer(2), Layer(3)

# ---------------------------------------------------------------------------------------------------------------
# 6. Layer 1 attention: matching and static counts
# ---------------------------------------------------------------------------------------------------------------

# H0 previous nodes (-4..-1): is this token's concept already given by a rank 1-3 node just before?
# -4 reaches a card heading from its feature list; dynamic and sr-only nodes are never keys.
match_head(L1, 0, dist_cols(-4, -1), elig(K_LOC, bonus=K_HEADE + [G_LABEL] + K_CTRL, banned=[F_SR, F_DYN]),
           [([ONE], M_P), (K_HEADE, M_PH), ([G_LABEL], M_PL), (K_CTRL, M_PC)], state_pair=True)
# H1 next nodes (+1..+3): what a tooltip or a narrating line names below it.
match_head(L1, 1, dist_cols(1, 3), elig(K_LOC, bonus=K_HEADE + [G_LABEL] + K_CTRL, banned=[F_SR, F_DYN]),
           [([ONE], M_N), (K_CTRL, M_NC)], state_pair=True)
# H2 the heading directly after (+1): what an eyebrow can restate.
match_head(L1, 2, dist_cols(1, 1), elig(K_HEAD, banned=[F_SR, F_DYN]), [([ONE], M_N1H)])
# H3 any earlier heading, button or text line that could hold an earlier copy (used only for the pointer query)
match_head(L1, 3, dist_cols(-63, -1), elig(K_HEAD + [R_BUTTON, G_HELP, G_SUB],
                                           banned=[F_SR, F_DYN, F_COND, F_NAVTAB]), [([ONE], M_G)])
# H4 / H7 reverse matches, used only as pointer keys: is this token repeated by a node that could be restating it?
# H4: any later lower-rank or duplicate-class node; H7: an eyebrow, tooltip or narrating line in the 3 nodes before.
match_head(L1, 4, dist_cols(1, 63), elig(LOWRANK + [G_LABEL, G_H1, G_H2, G_LINK, G_BADGE], banned=[F_SR, F_DYN]),
           [([ONE], RF)])
match_head(L1, 7, dist_cols(-3, -1), elig([G_EYEBROW, G_TIP] + NARR_ROLES, banned=[F_SR, F_DYN]), [([ONE], RB)])
# H5 per-node sums of the static lexical classes; H6 the node's code sum (written into the code dims of its EON)
count_head(L1, 5, [(d, d) for d in STATIC_COUNTED])
count_head(L1, 6, [(CODE[i], CODE[i]) for i in range(NCODE)])

# ---------------------------------------------------------------------------------------------------------------
# 7. Layer 1 MLP: token features, static EON booleans
# ---------------------------------------------------------------------------------------------------------------

# novel = informative weight, cancelled when the concept is covered (m ~ 1; -2*m also cancels a half-split match)
tok_unit(L1, {WI: 1.0, M_P: -2.0}, 0.0, {NOV_P: 1.0})
tok_unit(L1, {WI: 1.0, M_N: -2.0}, 0.0, {NOV_N: 1.0})
tok_unit(L1, {WN: 1.0, M_P: -2.0, M_N: -2.0}, 0.0, {NOV_X: 1.0})
# a tooltip must repeat its control word for word: filler words count too (wi + (mt - wi) = every coded word)
tok_unit(L1, {MT: 1.0, WI: -1.0, G_TIP: 1.0, M_P: -2.0}, -1.0, {NOV_P: 1.0})
tok_unit(L1, {MT: 1.0, WI: -1.0, G_TIP: 1.0, M_N: -2.0}, -1.0, {NOV_N: 1.0})


def hit_unit(roles, ms, out, extra=None, need=1.0):
    """1 when a token of a node in `roles` matched a key of the listed anchor classes (numbers never count)."""
    w = {NUM: -1.0}
    for d in roles:
        w[d] = 1.0
    for d in ms:
        w[d] = 1.0
    if extra:
        w.update(extra)
    tok_unit(L1, w, -need, {out: 1.0})


hit_unit([G_SUB, G_PH, G_BODY, G_HELP], [M_PH, M_PL], HIT_P)  # restates the heading or label before it
hit_unit([G_TIP], [M_PC, M_PL], HIT_P)  # tooltip: the control before it
hit_unit([G_LI, G_SBODY, G_LABEL], [M_PH], HIT_P)  # list line, state body, card label: the heading
hit_unit([G_EYEBROW], [M_N1H], HIT_N)  # eyebrow: the heading directly after it
hit_unit([G_TIP], [M_NC], HIT_N)  # tooltip: the control after it
hit_unit(NARR_ROLES, [M_N], HIT_NC)  # narration naming a control, label or heading below
hit_unit([G_SUB], [M_NC], HIT_NC0, extra={FIRST: 1.0}, need=2.0)  # ... as its first word ("Resume does ...")

# matched code: mc_i = code_i when the token is matched through a gate that fits its node's role, else 0.
# relu(s * code_i + 3 * m (+ 3 * role - 3) - 3) is 1 only for code_i = s and m = 1; two gates open: 2 * code.
MC_GATES = [  # (gate, roles it applies to, a match that takes precedence)
    (M_P, LOWRANK + [G_LABEL], None),  # a lower-rank node's word found in the nodes just before (not an eyebrow's)
    (M_N1H, [G_EYEBROW], None),  # an eyebrow's word in the heading after it
    (M_NC, [G_TIP] + NARR_ROLES, None),  # a tooltip's or a narrating line's word in a control below
    # a repeated title / call to action: its earlier copy. Not for helper / caption / subtitle: any word of theirs
    # that also occurs in some far heading would join the query and drown the label they restate (a repeated
    # fact, R14 text, therefore gets no target).
    (M_G, [G_H1, G_H2, G_LINK], None),
]
for i in range(NMC):
    for sgn in (1.0, -1.0):
        for gate, roles, minus in MC_GATES:
            w = {CODE[i]: sgn, gate: 3.0}
            w.update({d: 3.0 for d in roles})
            if minus is not None:
                w[minus] = -3.0
            tok_unit(L1, w, -6.0, {MC[i]: sgn})

# pointer gate: the token is repeated by some node that could be restating it (rf or rb), clamped to 1
tok_unit(L1, {RF: 1.0, RB: 1.0}, 0.0, {F_ANY: 1.0})
tok_unit(L1, {RF: 1.0, RB: 1.0}, -1.0, {F_ANY: -1.0})

# static EON booleans. Counts are integers (x0.997), half weights 0.5: thresholds sit between the values.
ge(L1, {VAR: 1.0}, 0.4, 0.8, B_VAR)  # any VAR or ALT marker
ge(L1, {WX: 1.0}, 0.3, 0.45, B_WX)  # any information outside genre / status / compass words (0.5 counts)
ge(L1, {WX: 1.0}, 1.3, 1.45, B_WX2)  # more than one such
ge(L1, {WI: 1.0, STATUS: -1.0}, 0.3, 0.45, B_XS)  # any informative word that is not a status word
ge(L1, {STATUS: 1.0}, 0.4, 0.8, B_STATUS)
ge(L1, {COORD: 1.0}, 2.4, 2.8, B_COORD)  # decimal + degree sign + compass letter
ge(L1, {DEIC: 1.0}, 0.4, 0.8, B_DEIC)
ge(L1, {UIN: 1.0, NUM: 1.0}, 0.4, 0.8, B_UINNUM)
ge(L1, {UIN: 1.0}, 0.4, 0.8, B_UIN)
ge(L1, {UIN: 1.0, NARR: 1.0}, 1.4, 1.8, B_UIN2)
ge(L1, {ACT: 1.0}, 0.4, 0.8, B_ACT)
ge(L1, {WI: 1.0, PROMO: -1.0}, 0.3, 0.45, B_XP)  # any informative word that is not a promo-pill word
ge(L1, {DEM: 1.0}, 0.4, 0.8, B_DEM)
ge(L1, {NUM: 1.0}, 0.4, 0.8, B_NUM)
ge(L1, {SUMM: 1.0}, 0.4, 0.8, B_SUMM)
ge(L1, {MT: 1.0}, 0.4, 0.8, B_MT)
ge(L1, {MT: 1.0}, 1.4, 1.8, B_MT2)
ge(L1, {MT: 1.0, NUM: -1.0}, 0.4, 0.8, B_WORD)  # a coded word that is not a number ("12" as a heading is a figure)
ge(L1, {WI: 1.0, UNIT: -1.0}, 1.4, 1.8, B_WI2U)  # two informative tokens that are not units / qualifiers (K13)
ge(L1, {WI: 1.0}, 1.3, 1.45, B_WIHI)  # more than one informative token
ge(L1, {WI: 1.0}, 0.3, 0.45, B_WI1)  # any informative token (a half-weight word counts)
ge(L1, {WI: 1.0}, 2.3, 2.45, B_WI3)  # more than two
ge(L1, {SUMM: 1.0}, 1.4, 1.8, B_SUMM2)  # two summary cues ("That covers ...")
ge(L1, {G_BADGE: 1.0, F_PILL: 1.0}, 0.4, 0.8, B_PILLISH)  # a badge, or any node drawn as a pill / with a dot

# squared length of the code sum over the duplicate head's bits (EON). Each entry x is an integer, and
# x^2 = |x| + 2 * sum_{k>=1} relu(|x| - k) exactly for |x| <= 11 (linear beyond: only very long nodes).
DUP_BITS = 10
for i in range(DUP_BITS):
    for sgn in (1.0, -1.0):
        for kk in range(0, 11):
            L1.unit({CODE[i]: sgn, IS_EON: G}, -kk - G, {NORM2: 1.0 if kk == 0 else 2.0})

# ---------------------------------------------------------------------------------------------------------------
# 8. Layer 2 attention: per-node sums, duplicate head, pointer head
# ---------------------------------------------------------------------------------------------------------------

count_head(L2, 0, [(d, d) for d in DYN_COUNTED])

# -- duplicate head (head 1): EON of A -> EON of the earlier eligible node B whose code sum is nearest to A's.
# score = 3 * (cs_A . cs_B - |cs_B|^2 / 2) over 10 bits = 1.5 * (|cs_A|^2 - |cs_A - cs_B|^2): an exact copy is
# the maximum, any other node is lower by 1.5 * |difference|^2 (15 for one differing word), so the softmax is sharp.
# The value subtracts cs_B from A's code dims: they end at 0 exactly when the two bags of concepts are equal
# (a copy split over several equal earlier copies still sums to 0). No eligible node: BOS, nothing subtracted.
DUP = 1
DUP_CLASSES = [  # (head dim, query roles, key roles, may the key be a repeated sibling item?)
    # the same badge on sibling cards (R11); a card label equal to its card's heading (R5): both sit inside
    # repeated cards, so repeated keys are allowed here
    (10, [G_BADGE, G_LABEL], [G_BADGE] + K_HEAD, True),
    (11, [G_H1, G_H2], K_HEAD, False),  # a heading repeats an earlier heading (R9)
    (13, [G_LINK], [R_BUTTON], False),  # a link repeats an earlier button (R14: the primary call to action again)
    (14, [G_HELP, G_SUB], [G_HELP, G_SUB, G_BODY], False),  # a text line repeats an earlier one (R14)
]
count_head(L2, 2, [(d, d) for d in MC])  # the node's matched-code sum (pointer query)
for H in (DUP,):
    for i in range(DUP_BITS):
        L2.q(H, i, CODE[i], math.sqrt(12.0))
        L2.k(H, i, CODE[i], math.sqrt(12.0))
    for hd, qroles, kroles, rep_ok in DUP_CLASSES:
        for d in qroles:
            L2.q(H, hd, d, 1.0)
        # The code-sum dot product is unbounded (up to 3 * 10 * words), so bans here are 1000s, not 60s.
        L2.k(H, hd, ONE, -4.0 * 2000.0)  # -2000 unless ...
        L2.k(H, hd, IS_EON, 4.0 * 1000.0)  # ... the key is an EON (+1000) ...
        for d in kroles:
            L2.k(H, hd, d, 4.0 * 1000.0)  # ... of an eligible role (+1000)
        # never another branch, runtime text, sr-only, nav or table copies (K5, 2a), nor (except for badges) a
        # repeated sibling item: two list entries with the same title are data, not an echo (K5)
        for d in (F_COND, F_DYN, F_SR, F_NAVTAB) + (() if rep_ok else (F_REP,)):
            L2.k(H, hd, d, -4.0 * 1000.0)
        L2.k(H, hd, NORM2, -4.0 * 1.5)  # - |cs_B|^2 / 2 (x3): with the dot product this is -|cs_A - cs_B|^2 / 2
    L2.q(H, 12, G_H1, 1.0)
    L2.k(H, 12, G_H2, -4.0 * 1000.0)  # a main heading only echoes another main heading (rank, LABELLING 5)
    L2.qb(H, 15, 4.0)
    L2.q(H, 15, IS_EON, -4.0)
    L2.k(H, 15, ONE, -8000.0)  # token queries go to BOS: their codes are not touched
    L2.bias(H, slice(None), BIG_OFF)
    L2.bias(H, dist_cols(-63, -1), 0.0)  # earlier nodes only: the first copy is kept (K6)
    L2.bias(H, mdl.BOS_COL, -300.0)  # below any plausible eligible candidate, far above a banned one
for i in range(NCODE):
    L2.v(DUP, i, CODE[i], 1.0)
    L2.o(CODE[i], DUP, i, -1.0)

# ---------------------------------------------------------------------------------------------------------------
# 9. Layer 2 MLP: dynamic EON booleans, duplicate distance
# ---------------------------------------------------------------------------------------------------------------

ge(L2, {HIT_P: 1.0}, 0.5, 0.9, B_HITP)
ge(L2, {HIT_N: 1.0}, 0.5, 0.9, B_HITN)
ge(L2, {HIT_NC: 1.0}, 0.5, 0.9, B_HITNC)
ge(L2, {HIT_NC0: 1.0}, 0.5, 0.9, B_HITNC0)
# novelty: 0 and 0.5 (one half-weight word) are "nothing new"; one content word (>= 0.9 after leaks) is new
ge(L2, {NOV_P: 1.0}, 0.6, 0.85, B_NOVP)
ge(L2, {NOV_N: 1.0}, 0.6, 0.85, B_NOVN)
ge(L2, {NOV_P: 1.0}, 1.6, 1.85, B_NOVP2)
ge(L2, {NOV_N: 1.0}, 1.1, 1.35, B_NOVN2)  # an eyebrow may carry one extra word ("OPERATION on hold")
# narration may name one thing ("This page lists your vehicles"); a content word plus a half-weight word (1.5) is
# already more than that ("Fill out the form below and we'll get back to you within 24 hours")
ge(L2, {NOV_X: 1.0}, 1.1, 1.35, B_NOVX2)
ge(L2, {NOV_X: 1.0}, 2.6, 2.85, B_NOVX3)
ge(L2, {NOV_X: 1.0}, 0.6, 0.85, B_NOVX1)
ge(L2, {UIN: 1.0, DEIC: 1.0, HIT_NC: 1.0}, 0.5, 0.9, B_NCTX)  # an interface noun, "below", or a button's name
# L1 distance between this node's code sum and its candidate copy's (code dims of the EON after the dup head)
for i in range(NCODE):
    L2.unit({CODE[i]: 1.0, IS_EON: G}, -G, {DL1: 1.0})
    L2.unit({CODE[i]: -1.0, IS_EON: G}, -G, {DL1: 1.0})

# ---------------------------------------------------------------------------------------------------------------
# 10. Layer 3 MLP: rules
# ---------------------------------------------------------------------------------------------------------------

# Evidence a fired rule adds to its class. Three levels, by how much the rule has checked, not by scores:
R_EXACT = 3.0  # word for word: the same bag of concepts as another node, or a tooltip fully inside its control
R_HIT = 2.5  # a word shared with the anchor and nothing informative left over
R_LEX = 2.0  # lexicon only (genre eyebrow, status pill, narration words, filler line): no second node involved
# The head bias of -0.5 leaves margins of +2.5, +2.0, +1.5.
VETO = [B_VAR, F_DYN]  # runtime text or alternative branches are never flagged
NARR = {O_NARR: R_LEX, O_ECHO: -1.0}  # narration wording makes NARRATE the class even when an echo rule fires too

# ECHO, local
rule(L3, [G_EYEBROW, B_HITN], 2, VETO + [B_NOVN2], {O_ECHO: R_HIT})  # R1
rule(L3, [G_SUB, G_HELP, G_PH, B_HITP], 2, VETO + [B_NOVP], {O_ECHO: R_HIT})  # R3 R5 R8
rule(L3, [G_TIP, B_HITP], 2, VETO + [B_NOVP], {O_ECHO: R_EXACT})  # R5 tooltip after its control
rule(L3, [G_TIP, B_HITN], 2, VETO + [B_NOVN], {O_ECHO: R_EXACT})  # R5 tooltip before its control
rule(L3, [G_BODY, F_SMALL, B_HITP], 3, VETO + [B_NOVP], {O_ECHO: R_HIT})  # R5 stat-card note
rule(L3, [G_LI, F_CARD, B_HITP], 3, VETO + [B_NOVP], {O_ECHO: R_HIT})  # R4 plan-card line
rule(L3, [G_SBODY, B_HITP], 2, VETO + [B_NOVP, F_CTL], {O_ECHO: R_HIT})  # R4 (not an inline field error: K12)
rule(L3, [G_SUB, B_MT2], 2, VETO + [B_WI1], {O_ECHO: R_LEX})  # R3 a subtitle of filler words only says nothing
# Near miss (ranking only, never a call): a shared word and exactly one informative word left over. 0.25 stays
# under the 0.5 a call needs; it orders KEEP nodes for the precision-recall curve (margin -0.25 instead of -0.5).
rule(L3, [G_SUB, G_HELP, G_PH, G_SBODY, B_HITP], 2, VETO + [B_NOVP2, F_CTL], {O_ECHO: 0.25})
# ECHO, same bag of concepts as an earlier node (DL1 = 0)
rule(L3, [G_H1, G_H2, B_WORD], 2, VETO, {O_ECHO: R_EXACT}, soft=[DL1])  # R9 (two equal figures are not a title)
# R5 a card label equal to the card heading just before it (a word must also match a heading within 4 nodes:
# a field label far below a heading that happens to use the same words is not an echo)
rule(L3, [G_LABEL, B_HITP], 2, VETO + [F_CTL, F_NAVTAB], {O_ECHO: R_EXACT}, soft=[DL1])
rule(L3, [G_LINK, B_MT2], 2, VETO + [F_COND, F_NAVTAB], {O_ECHO: R_EXACT}, soft=[DL1])  # R14 call to action
rule(L3, [G_HELP, G_SUB, B_WI2U], 2, VETO + [F_COND, F_REP], {O_ECHO: R_EXACT}, soft=[DL1])  # R14 fact
# ECHO, closing line (R17): last body line of a section, a summary word, at most one informative word
rule(L3, [G_BODY, G_LI, F_LAST, B_SUMM], 3, VETO + [B_WIHI], {O_ECHO: R_LEX})
rule(L3, [G_BODY, G_LI, F_LAST, B_SUMM2, B_WIHI], 4, VETO + [B_WI3], {O_ECHO: R_LEX})  # two cues: two words
# DECOR
# R2 eyebrow / kicker-styled label made of genre and filler words only (at least one coded word: "For you" stays)
rule(L3, [G_EYEBROW, B_MT], 2, VETO + [B_WX], {O_DECOR: R_LEX})
rule(L3, [G_LABEL, F_TRK, B_MT], 3, VETO + [B_WX, F_CTL, F_NAVTAB], {O_DECOR: R_LEX})  # not a nav group label
# R11 static status. Not on repeated sibling items (a status there usually differs between siblings: K2), not on
# list lines (dotted list lines are a legend: K2).
rule(L3, [B_PILLISH, B_STATUS], 2,
     VETO + [B_XS, F_STF, F_COND, F_NAVTAB, F_REP, R_UNFL, G_LINK, G_H1, G_H2, G_LI], {O_DECOR: R_LEX})
# R11 the same promo pill ("New", "Fresh", "Popular") repeated on sibling cards. Only promo words: a repeated
# tag ("TypeScript" on two project cards) is data that differs between siblings (K2).
rule(L3, [G_BADGE, B_MT], 2, VETO + [F_STF, F_NAVTAB, B_XP], {O_DECOR: R_EXACT}, soft=[DL1])
rule(L3, [G_HELP, G_EYEBROW, B_COORD], 2, VETO + [B_WX2], {O_DECOR: R_LEX})  # R2 decorative coordinate
# NARRATE
rule(L3, NARR_ROLES + [B_DEIC, B_UINNUM], 3, VETO + [B_NOVX2], NARR)  # R16 "four options below"
rule(L3, NARR_ROLES + [B_DEIC, B_HITNC], 3, VETO + [B_NOVX2, B_UINNUM], NARR)  # "enter your email below"
# "Below you'll find our pricing." under "Pricing": a layout word and nothing new at all (LABELLING 3: "below
# you'll find"). Stricter novelty than the other narration rules because nothing else anchors it.
rule(L3, NARR_ROLES + [B_DEIC], 2, VETO + [B_NOVX1, B_UINNUM, B_HITNC], NARR)
rule(L3, NARR_ROLES + [B_DEM, B_UIN2], 3, VETO + [B_NOVX2], NARR)  # R16 "this page lists ..."
rule(L3, NARR_ROLES + [B_DEM, B_NUM, B_UIN], 4, VETO + [B_NOVX2], NARR)  # R16 "here are three numbers"
rule(L3, NARR_ROLES + [B_ACT, B_NCTX], 3, VETO + [B_NOVX2], NARR)  # R6 "click the button below"
rule(L3, [G_HELP, B_ACT, B_HITP], 3, VETO + [B_NOVX2], NARR)  # R6 "toggle <label>"
rule(L3, [G_SUB, B_HITNC0], 2, VETO + [B_NOVX3, B_NUM], NARR)  # R6 "<Button> does ..."

# ---------------------------------------------------------------------------------------------------------------
# 11. Layer 3 attention: the pointer head (attention pattern only; it writes nothing)
# ---------------------------------------------------------------------------------------------------------------
# score(EON of A, key token u) =
#     2 * (matched-code sum of A . code of u, 11 bits)   22 for each concept of A that u carries; an unrelated
#                                                         token gets noise of sd 6.6 * sqrt(matched concepts)
#   - 24 unless u is flagged as repeated by some node that could be restating it (rf / rb): with three matched
#     concepts the code noise (sd 11.5) reaches 3 sd = 35 somewhere on a 300-token screen, which beat the true
#     token (22 + 3) before this gate; flagged tokens are few. (A reward for flagged tokens was tried first: a
#     token with both flags then outscored the true target.)
#   + 3 when u is in an anchor node (heading, label, control, error title)
#   + 6 when u has the role an earlier copy of A would have (heading for a heading / card label, button for a link)
#   + 3 * (A has an action verb or a layout word) * (u is in a button)   narration points at the control
#   - 0.05 * |distance| (capped at -0.6), and -1 more for later nodes.
#     Distance and direction only break ties: attention is summed per node, so a far node holding two of A's
#     concepts (2 e^s) must still beat a near node holding one (e^(s + 0.6) < 2 e^s), and a button below that A
#     names with two words must beat the title above that shares one of them. (A larger "later" penalty and a
#     larger button bonus were tried; with the code noise they sent helpers to far buttons.)
# EON keys are excluded (their code dims hold other things). BOS = 7: when A matched nothing every score is exact
# (no code term) and at most 3 + 3, so there is no target; a narrating line that names no control has none.
PTR = 0
PTR_BITS = 11
for i in range(PTR_BITS):
    L3.q(PTR, i, MC[i], math.sqrt(8.0))
    L3.k(PTR, i, CODE[i], math.sqrt(8.0))
L3.qb(PTR, 11, 1.0)
for d in (G_H1, G_H2, G_LABEL, R_BUTTON, G_LINK, R_ERR):
    L3.k(PTR, 11, d, 12.0)
L3.k(PTR, 11, IS_EON, -4000.0)
L3.q(PTR, 12, B_ACT, 1.0)
L3.q(PTR, 12, B_DEIC, 1.0)  # "the button below" names a control as much as "click" does
L3.k(PTR, 12, R_BUTTON, 12.0)
for d in (G_H1, G_H2, G_LABEL, G_LINK):
    L3.q(PTR, 13, d, 1.0)
for d in K_HEAD + [R_BUTTON]:
    L3.k(PTR, 13, d, 24.0)
# gate: a key token that no candidate restater repeats loses 24. An eyebrow's target comes after it, so for an
# eyebrow only "repeated just before" (rb) counts; for every other role either flag does.
for d in (G_SUB, G_HELP, G_PH, G_TIP, G_BODY, G_LI, G_SBODY, G_LABEL, G_H1, G_H2, G_LINK, G_BADGE):
    L3.q(PTR, 14, d, 1.0)
L3.k(PTR, 14, F_ANY, 96.0)
L3.k(PTR, 14, ONE, -96.0)
L3.q(PTR, 15, G_EYEBROW, 1.0)
L3.k(PTR, 15, RB, 96.0)
L3.k(PTR, 15, ONE, -96.0)
for d in range(-63, 64):
    L3.bias(PTR, d + 63, max(-0.6, -0.05 * abs(d)) - (1.0 if d > 0 else 0.0))
L3.bias(PTR, 63, BIG_OFF)  # never the node itself
L3.bias(PTR, mdl.BOS_COL, 7.0)

# ---------------------------------------------------------------------------------------------------------------
# 12. Layer 4 MLP: caps. Two rules firing on one node are not twice the evidence: each class is capped at the
#     level of its strongest kind of rule (3 for ECHO and DECOR, 2 for NARRATE, whose rules are all lexical).
# ---------------------------------------------------------------------------------------------------------------
for dim, cap in ((O_ECHO, R_EXACT), (O_DECOR, R_EXACT), (O_NARR, R_LEX)):
    L4.unit({dim: 1.0}, -cap, {dim: -1.0})

# ---------------------------------------------------------------------------------------------------------------
# 13. Output head
# ---------------------------------------------------------------------------------------------------------------

T["head.weight"][1, O_ECHO] = 1.0
T["head.weight"][2, O_NARR] = 1.0
T["head.weight"][3, O_DECOR] = 1.0
T["head.bias"][1:] = -0.5  # a rule must fire to at least a quarter before a node is flagged

sha = mdl.save_artifact(WORK, T, vocab, (3, PTR))
# for ./inspect: a short list of EON features (the full name -> dim table goes to dims.json)
SHOW = ["wi", "mt", "wx", "nov_p", "nov_n", "nov_x", "hit_p", "hit_n", "hit_nc", "dl1", "o_echo", "o_narr", "o_decor"]
(WORK / "directions.json").write_text(json.dumps({n: [NAMES.index(n)] for n in SHOW}) + "\n")
(WORK / "dims.json").write_text(json.dumps({n: i for i, n in enumerate(NAMES)}, indent=0) + "\n")
nz = sum(int(np.count_nonzero(v)) for v in T.values())
print(f"vocab {len(vocab)} stems, {len(NAMES)} residual dims, units {[L.units for L in (L1, L2, L3, L4)]}, "
      f"non-zero params {nz}, artefact {sha[:12]}")
for n in notes:
    print("note:", n)
