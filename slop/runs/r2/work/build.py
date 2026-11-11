#!/usr/bin/env python3
"""Builds the hand-written slop network (round 2): vocab.txt, artifact.npz, meta.json, plus directions.json and
dims.json for ./inspect.

    ./python3 work/build.py

Reads work/lexicon/*.txt (hand-written word lists), work/thesaurus_groups.tsv (written by work/select_thesaurus.py
from resources/thesaurus by the rule stated there and in MODEL.md) and kit/lib. It never reads examples/. Every
number below is a literal chosen by reasoning (see the comment beside it and work/MODEL.md), a formula, or a
construction from those lists.

Mechanism (details in MODEL.md):
  L1 attention  token-level concept matching against the nodes just before / after (code match, BOS as threshold),
                with a separate head for the node's own anchor (the two nodes before); reverse matches for the
                pointer; per-node sum of the concept codes at EON.
  L1 MLP        per token: "novel" (informative and not covered by the own anchor; scope words in a confirmation
                body; an imperative lead), anchor hits split by heading / label, matched code; per EON: squared
                length of the code sum.
  L2 attention  per-node sums of the token features and of the lexical classes; duplicate head (EON -> earlier
                EON with the nearest bag of stems, subtracts its code sum); "anywhere" matching (the concept
                occurs in some other node).
  L2 MLP        EON booleans from the sums; L1 distance of the duplicate comparison; per-token novelty against the
                whole window.
  L3 attention  per-node sum of the window novelty; "named control" head (a control just below whose whole label
                the narrating lines before it repeat).
  L3 MLP        rule units (AND of booleans with vetoes) -> class evidence.
  L4 attention  the pointer head (attention pattern only; it writes nothing).
  L4 MLP        caps on the layer-3 class evidence; the two rules that need the window novelty (summary line,
                enumerating intro).
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
LAY = tk.LAYOUTS[2]  # id layout 2: vocabulary 512..16895, hash buckets 16896..25087

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


CLASS_FILES = {  # file -> attribute names
    "function.txt": "function",  # no concept code, no weight: invisible to matching and novelty
    "particle.txt": "particle",  # in / out / up / off ...: a code (so "Sign in" is not "Sign up") but never a hit
    "marker.txt": "marker",  # consequence / condition / time / audience markers: always weight 1
    "scope.txt": "scope",  # all / every / everything: weight 0, but 1 inside a confirmation body
    "frame.txt": "frame",  # restating frames and filler: a code, weight 0
    "semi.txt": "semi",  # weight 0.5: one alone is not news, two are
    "ilead.txt": "il",  # low-weight verbs that open an instruction ("Pick a playlist.")
    "deictic.txt": "deic",
    "uinoun.txt": "uin",
    "action.txt": "act",  # gestures and control mechanics (click, tap, press, toggle): a whole narration cue
    "actionweak.txt": "actw",  # verbs that are only sometimes about a control (choose, pick, turn, switch): half
    "actionpart.txt": "actp",  # "on" / "off": a quarter each ("switch ... on or off" is a whole cue)
    "gesture.txt": "act uin",  # scroll, swipe: an action that needs no other interface noun to be narration
    "promo.txt": "promo",
    "narrverb.txt": "narr",
    # padding of a narrating line: "when you're ready", "once you are happy / done" (codex ui_padding_phrases),
    # "to continue", "to proceed", "to get started" (codex ui_narration_verbs, outcome narration) and the same
    # purpose tails "to finish", "to begin", "to go ahead". No weight in the narration novelty only; everywhere
    # else these words keep their weight and their synonym group.
    "narrpad.txt": "npad",
    "status.txt": "status",
    # summary cues, weighted (see SUMM in section 3): loose cues of a closing line 1; a summary frame 4
    # ("... at a glance", "in one place", "an overview / a summary ..."); parts of a frame: "everything" 2 +
    # "about" 2, "together" 3 + "all" 1
    "summary.txt": "summ1",
    "summarystrict.txt": "summ4",
    "summaryhalf.txt": "summ2",
    "summarymost.txt": "summ3",
    "unit.txt": "unit",
    "dem.txt": "dem",
    "timeword.txt": "timeword",
    "aux.txt": "aux",  # clause markers: auxiliaries, modals, relative pronouns (a noun-phrase list has none)
    "hid.txt": "hid",  # hidden-input cues (hold, shift, long-press, shortcut): never narration (K4)
    "navgenre.txt": "tiny",  # contents / menu / navigation: a heading inside navigation made of these says nothing
    "meta.txt": "meta1",  # decorative meta-strip words (index, sector, lat, gmt ...)
    "metahalf.txt": "metah",
    "kick.txt": "kick1",  # a section-genre kicker on its own (features, overview, about ...)
    "kickhalf.txt": "kickh",  # half of a two-word genre kicker (how + works, our + team, key + facts)
}
GROUP_FILES = ["groups_actions.txt", "groups_objects.txt", "groups_states.txt", "groups_more.txt",
               "groups_general.txt"]  # earlier wins

# Number words share the concept code of the digit token ("Three days before" = "3 days before").
# "one" is left out: its stem is "on", the particle.
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12,
                "second": 2, "third": 3, "fourth": 4, "fifth": 5}  # ordinals ("a second check" = "two-step")
# Compass letters after a coordinate. n / w are rarely anything else after a decimal; s / e also read as units
# ("2.5 s"), so they weigh half as much in the meta-strip sum (see META below).
COMPASS = {"n": 0.5, "w": 0.5, "s": 0.25, "e": 0.25}
LETTERS = "abcdefghijklmnopqrstuvwxyz"  # a static pill holding one letter is a monogram (R11)

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

for fname in GROUP_FILES:
    for line in (LEX / fname).read_text(encoding="utf-8").splitlines():
        ws = line.split()
        if not ws:
            continue
        head = None
        for w in ws:
            st = add_stem(stem_of(w))
            if head is None:
                head = group_of.get(st, st)
            if st in group_of and group_of[st] != head:
                notes.append(f"group conflict: {w!r} ({st}) stays in {group_of[st]!r}, not moved to {head!r}")
                continue
            group_of[st] = head

for w, n in NUMBER_WORDS.items():
    st = add_stem(stem_of(w))
    attrs[st].add("numword")
    numword_of[st] = n
for w in LETTERS:
    attrs[add_stem(stem_of(w))].add("tiny")
for w in COMPASS:
    attrs[add_stem(stem_of(w))].add("compass")

# Thesaurus-derived entries (work/select_thesaurus.py; rule in MODEL.md section 2): "stem <tab> head <tab> via".
# A stem joins the group of `head`; it takes the weight class (frame / semi) of `via`, the hand-lexicon member it
# is a strict synonym of, and no other class. Stems already in the hand lexicon are never touched.
HAND = set(attrs)
THES = WORK / "thesaurus_groups.tsv"
n_thes = 0
if THES.exists():
    for line in THES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        st, head, via = line.split("\t")
        if st in HAND:
            continue
        add_stem(st)
        group_of[st] = group_of.get(head, head)
        if head not in attrs:
            add_stem(head)
            group_of[head] = group_of.get(head, head)
        if via in HAND:
            attrs[st].update(attrs[via] & {"frame", "semi"})
        n_thes += 1

vocab_id = {st: LAY.vocab_base + n for n, st in enumerate(vocab)}
assert len(vocab) <= LAY.vocab_max, len(vocab)

# ---------------------------------------------------------------------------------------------------------------
# 2. Residual allocation
# ---------------------------------------------------------------------------------------------------------------
# Three kinds of dimension:
#   both  a per-token value at tokens and the per-node sum (or the same constant) at the node's EON
#   tok   only meaningful at tokens     } these two kinds share physical dimensions: every MLP unit is gated to
#   eon   only meaningful at EON        } the kind of position it is meant for, and attention readers of a tok /
#                                         eon dimension only ever look at keys or queries of that kind.

NAMES = {}  # physical dim -> name ("tokname|eonname" when shared)
_next = [0]
_tok, _eon = [], []


def both(*names):
    out = []
    for n in names:
        NAMES[_next[0]] = n
        out.append(_next[0])
        _next[0] += 1
    return out if len(out) > 1 else out[0]


class Lazy:
    """A tok-only or eon-only dimension; its physical index is fixed by finish_alloc()."""

    def __init__(self, name, kind):
        self.name, self.kind, self.i = name, kind, None
        (_tok if kind == "tok" else _eon).append(self)

    def __index__(self):
        assert self.i is not None
        return self.i

    def __hash__(self):
        return hash(self.__index__())

    def __eq__(self, other):
        return int(self) == int(other)


def tok(*names):
    out = [Lazy(n, "tok") for n in names]
    return out if len(out) > 1 else out[0]


def eon(*names):
    out = [Lazy(n, "eon") for n in names]
    return out if len(out) > 1 else out[0]


def finish_alloc():
    base = _next[0]
    for j in range(max(len(_tok), len(_eon))):
        t = _tok[j] if j < len(_tok) else None
        e = _eon[j] if j < len(_eon) else None
        for x in (t, e):
            if x is not None:
                x.i = base + j
        NAMES[base + j] = "|".join(x.name for x in (t, e) if x is not None)
    _next[0] = base + max(len(_tok), len(_eon))
    assert _next[0] <= mdl.D, _next[0]


NCODE = 15
# tokens: +-1 concept code (0 for function words). EON: the node's code sum after L1, and after L2 the
# difference between it and the code sum of the best earlier candidate copy (0 = same bag of concepts).
CODE = [both(f"code{i}") for i in range(NCODE)]
ONE, IS_EON, FIRST = both("one", "is_eon", "first")
# static lexical dims: a per-token value at tokens, the per-node sum at EON (written by layer-2 count heads)
(WI, MT, WX, NUM, VAR, DEIC, UIN, ACT, NARR, STATUS, SUMM, DEM, PROMO, UNIT, KICK, META, HID, TINY, AUX) = both(
    "wi", "mt", "wx", "num", "var", "deic", "uin", "act", "narr", "status", "summ", "dem", "promo", "unit", "kick",
    "meta", "hid", "tiny", "aux")
# A count head sums at most 16 dims. These 16 are summed by layer-2 head 4 (the L2 MLP turns the sums into
# booleans); VAR, HID and AUX go with the token features in layer-2 head 0. (Until late in round 2 the 16 were
# summed in layer 1; nothing before layer 3 reads the booleans, and the move freed a layer-1 head for m_pn.)
STATIC_A = [WI, MT, WX, NUM, DEIC, UIN, ACT, NARR, STATUS, SUMM, DEM, PROMO, UNIT, KICK, META, TINY]
STATIC_B = [VAR, HID, AUX]
# token-only lexical dims
WN, WS, IL, NOHIT = tok("wn", "ws", "il", "nohit")
# match outputs (tokens only)
M_P, M_PH, M_PL, M_PC, M_N, M_NC, M_N1H, M_G = tok("m_p", "m_ph", "m_pl", "m_pc", "m_n", "m_nc", "m_n1h", "m_g")
M_PN = tok("m_pn")  # the concept is in a rank 1-3 node among the two nodes just before (a subtitle's own anchor)
# layer 2: the concept occurs in some other node of the window. A dimension of its own (not shared with an EON
# boolean): in layer 2 an EON carries its node's code sum, so the "anywhere" head also writes here at EON
# positions (a meaningless value that nothing reads).
M_ANY = both("m_any")
# pointer key flags (tokens only)
RF, RB, F_ANY = tok("rf", "rb", "f_any")
# matched code: at a token, its code (12 bits) if the token found its concept in a node this node could be
# restating, else 0; at EON the sum over the node (the pointer's query: "the concepts I repeat")
NMC = 11  # the pointer compares 11 bits
MC = [both(f"mc{i}") for i in range(NMC)]
# token features computed by the L1 MLP; per-node sums at EON (L2 count head)
# hit_p: a word shared with the heading (or error title, or for a tooltip the control) before; hit_pl: with the
# label before
NOV_P, NOV_N, NOV_X, HIT_P, HIT_PL, HIT_N, HIT_NC, HIT_NC0 = both(
    "nov_p", "nov_n", "nov_x", "hit_p", "hit_pl", "hit_n", "hit_nc", "hit_nc0")
DYN_COUNTED = [NOV_P, NOV_N, NOV_X, HIT_P, HIT_PL, HIT_N, HIT_NC, HIT_NC0]
# role groups (role embedding; on every token of the node and its EON)
(G_EYEBROW, G_SUB, G_HELP, G_CAP, G_PH, G_TIP, G_BODY, G_LI, G_SBODY, R_DLG, G_LABEL, G_H1, G_H2, G_LINK, G_BADGE,
 R_BUTTON, R_ERR) = both("g_eyebrow", "g_sub", "g_help", "g_cap", "g_ph", "g_tip", "g_body", "g_li", "g_sbody",
                         "r_dlg", "g_label", "g_h1", "g_h2", "g_link", "g_badge", "r_button", "r_err")
# flags
(F_DYN, F_STF, F_COND, F_NAVTAB, F_PILL, F_SR, F_LAST, F_FIRST, F_CARD, F_CTL, F_TRK, F_SMALL, F_STATE,
 F_REP) = both("f_dyn", "f_stf", "f_cond", "f_navtab", "f_pill", "f_sr", "f_last", "f_first", "f_card", "f_ctl",
               "f_trk", "f_small", "f_state", "f_rep")
# EON booleans, static (L1 MLP)
(B_STATUS, B_DEIC, B_UINNUM, B_UIN, B_UIN2, B_ACT, B_DEM, B_NUM, B_SUMM, B_PROMO, B_MT2, B_MT3, B_WORD,
 B_WI2U, B_XS, B_XP, B_KICK, B_METAH, B_TINY) = eon(
    "b_status", "b_deic", "b_uinnum", "b_uin", "b_uin2", "b_act", "b_dem", "b_num", "b_summ", "b_promo",
    "b_mt2", "b_mt3", "b_word", "b_wi2u", "b_xs", "b_xp", "b_kick", "b_metah", "b_tiny")
# layer 3: information weight of the node's words that occur nowhere else in the window.
# b_list: the node is a list of at least three things and has no clause marker (from the signed sum in `aux`)
NOVG, B_LIST = eon("novg", "b_list")
# EON booleans, dynamic (L2 MLP)
(B_HITP, B_HITPL, B_HITN, B_HITNC, B_HITNC0, B_NOVP, B_NOVPL, B_NOVP2, B_NOVN, B_NOVN2, B_NOVX1, B_NOVX2,
 ) = eon("b_hitp", "b_hitpl", "b_hitn", "b_hitnc", "b_hitnc0", "b_novp", "b_novpl", "b_novp2", "b_novn",
         "b_novn2", "b_novx1", "b_novx2")
# narration novelty against the controls below only: per token (tok dim, L1 MLP) the narration weight of a word
# that no button or link in the next three nodes has; its per-node sum at EON (eon dim, L2 count head)
NOV_CT = tok("nov_ct")
NOVC = eon("novc")
# layer 3 attention ("named control" head): one of the next three nodes is a control whose whole label is
# repeated by the narrating lines before it
B_CTL = eon("b_ctl")
DL1, NORM2 = eon("dl1", "norm2")  # duplicate distance; squared length of the node's code sum (duplicate head)
O_ECHO, O_NARR, O_DECOR = eon("o_echo", "o_narr", "o_decor")
finish_alloc()

T = mdl.zero_tensors(2)
E = T["tok_emb.weight"]

# ---------------------------------------------------------------------------------------------------------------
# 3. Token embedding
# ---------------------------------------------------------------------------------------------------------------


def code_vec(idx):
    """A 15-bit +-1 code, distinct for every index below 16384.

    The first 14 bits are a bijective scramble of the index (odd multiplier mod 2^14, so neighbouring indices do
    not get neighbouring codes); the 15th is their parity. The matching heads compare the first 14 bits (any two
    different codes differ in at least one of them); the duplicate comparison uses all 15.
    """
    x = (idx * 5167 + 1234) % 16384
    bits = [(x >> i) & 1 for i in range(14)]
    bits.append(sum(bits) & 1)
    return np.array([1.0 if b else -1.0 for b in bits], dtype=np.float64)


# Concept indices are handed out in order: numbers, the two matchable number classes, vocabulary concepts, hash
# buckets. Layout 2 has 25,088 rows but a 14-bit code has 16,384 values; synonym groups share one index, so as
# long as numbers + groups + buckets stay below 16,384 every concept has its own code (checked below; beyond
# that, indices wrap and two concepts would share a code, which is the same kind of collision as a hash bucket).
_concept = {}


def concept_index(key):
    if key not in _concept:
        _concept[key] = len(_concept)
    return _concept[key]


_members = {}  # concept key -> how many stems share it so far


def set_code(tid, key):
    """Concept code in bits 0-13 (shared by synonyms). Bit 14 is NOT shared: the j-th stem of a synonym group
    gets 1 + 4 j there. Only the duplicate distance DL1 reads bit 14 (the matching heads use bits 0-13, the
    duplicate head's attention and NORM2 bits 0-9, the pointer bits 0-10), so two nodes count as "the same bag"
    only when they use the same stems: a section "Members" on a page titled "Team" shares a concept with its
    title but is not a repeated title (LABELLING 8: the synonym case is KEEP when the page has other sections),
    and with thesaurus-scale groups that case would otherwise fire often. A difference of one synonym gives
    DL1 >= 4, which takes the whole of R_EXACT = 3 away."""
    v = code_vec(concept_index(key) % 16384)
    j = _members.get(key, 0)
    _members[key] = j + 1
    v[14] = 1.0 + 4.0 * j
    E[tid, CODE] = v
    E[tid, MT] = 1.0


# specials
E[tk.EON, IS_EON] = 1.0
# VAR: interpolated runtime text is always new information, and vetoes every rule. ALT also vetoes (K14).
E[tk.VAR, [VAR, WI, WX, int(WN)]] = 1.0
E[tk.ALT, VAR] = 1.0
# The degree sign (and other odd symbols) is a quarter of a decorative coordinate (see META in section 7).
E[tk.GLYPH_IDS["SYMBOL"], META] = 0.25
# list cues: the comma and the ampersand (ASCII punctuation ids run from 96 in code-point order: '!' = 96)
for ch in ",&":
    E[96 + ord(ch) - ord("!"), AUX] = 1.0
# numbers 0-99: matchable (the same number in two nodes is the same fact), informative when unmatched,
# never counted against narration ("Choose from 4 options below").
for n in range(100):
    tid = tk.NUMBER_BASE + n
    set_code(tid, ("num", n))
    E[tid, [NUM, WI, WX]] = 1.0
# YEAR stands for many different years: never matches, always informative.
E[tk.YEAR, [NUM, WI, WX]] = 1.0
# NUM (grouped and large numbers) and DEC (decimals): one code per class, so that a figure repeated beside its
# stat ("Foot passengers: 4,753" under the stat "4,753") is covered. Two different large numbers in adjacent
# nodes would also match, but a number is never a hit (below), so this alone never makes an echo: the words
# around it must be covered too.
for tid in (tk.NUM, tk.DEC):
    set_code(tid, ("numclass", tid))
    E[tid, [NUM, WI, WX]] = 1.0
# vocabulary
for st in vocab:
    tid = vocab_id[st]
    a = attrs[st]
    # "above" is half a layout word: it is as often "or higher" ("on the Plus plan and above") as a direction
    E[tid, DEIC] = 0.5 if st == "abov" else 1.0 if "deic" in a else 0.0
    for attr, dim in (("uin", UIN), ("narr", NARR), ("status", STATUS),
                      ("dem", DEM), ("promo", PROMO), ("unit", UNIT), ("hid", HID),
                      ("tiny", TINY), ("il", IL)):
        if attr in a:
            E[tid, dim] = 1.0
    # `aux` is one signed sum: +1 for each list cue (a comma, "and", "&": set below and here), -100 for each
    # clause marker (auxiliary, modal, relative pronoun). It is >= 2 exactly when the node enumerates at least
    # three things ("A, B and C") and contains no clause marker.
    E[tid, AUX] = -100.0 if "aux" in a else 1.0 if st == "and" else 0.0
    # summary cue weight: any cue (>= 1) marks a closing line (R17, layer 3); a whole summary frame (>= 4) is
    # what the layer-4 rule for subtitles asks for
    E[tid, SUMM] = 4.0 if "summ4" in a else 3.0 if "summ3" in a else 2.0 if "summ2" in a else \
        1.0 if "summ1" in a else 0.0
    # narration cue weight: a gesture or control-mechanics verb is a whole cue; "choose", "pick", "turn", "switch"
    # are ordinary verbs half the time ("Our class picked it up"), so they need a second cue ("... on or off")
    E[tid, ACT] = 1.0 if "act" in a else 0.5 if "actw" in a else 0.25 if "actp" in a else 0.0
    # section-genre kicker weight: 1 for a word that is a genre kicker on its own, 0.5 for half of a two-word one
    E[tid, KICK] = 1.0 if "kick1" in a else 0.5 if "kickh" in a else 0.0
    # decorative meta strip: a meta word 1, "local" 0.5, "time" 0.25, compass letters 0.5 / 0.25
    E[tid, META] = (1.0 if "meta1" in a else 0.5 if st == "local" else 0.25 if "metah" in a else 0.0) \
        + (COMPASS.get(st, 0.0) if "compass" in a else 0.0)
    if "function" in a and "marker" not in a:
        continue  # no code, no weight: ignored by matching and by novelty
    key = ("num", numword_of[st]) if "numword" in a else ("stem", group_of.get(st, st))
    set_code(tid, key)
    if "marker" in a:
        wi = 1.0  # a consequence, condition, time or audience marker is news whenever the anchor lacks it
    elif a & {"frame", "deic", "scope", "particle"}:
        wi = 0.0  # filler: restating with these adds nothing
    elif "semi" in a:
        wi = 0.5  # half weight: alone it does not make a line informative, with a second one it does
    elif a & {"summ1", "summ2", "summ3", "summ4"}:
        wi = 0.0  # a summary cue is not itself information (unless it is also a marker: "plus", see marker.txt)
    else:
        wi = 1.0
    if "numword" in a:
        E[tid, NUM] = 1.0
    E[tid, WI] = wi
    # scope words (all, every, everything): in a confirmation body they say the action reaches further than the
    # title's object (K7), elsewhere they are the frame "all your ..." (weight 0)
    E[tid, WS] = 1.0 if "scope" in a else 0.0
    # particles carry a code so that "Sign in" and "Sign up" are different bags, but a shared "in" is never a hit
    E[tid, NOHIT] = 1.0 if "particle" in a else 0.0
    # narration novelty: words that only point at the interface do not count as new information, nor does the
    # padding of a narrating line ("... when you are ready", "... to continue"). The markers "when" / "once" keep
    # their weight, so such a tail still uses up the one word a narrating line may add, and a real condition
    # ("... when the upload has finished") is over it.
    # A particle is never narration novelty either: after a gesture it is the gesture's direction ("Scroll down
    # to see more boats.", "Swipe up", "Turn it off"); only "down" and "off" (weight 1 as markers) are affected.
    E[tid, WN] = 0.0 if a & {"uin", "act", "actw", "actp", "narr", "numword", "npad", "particle"} else wi
    # class-external information: genre-kicker, status, meta-strip and navigation-genre words do not count
    # (the pure-genre / pure-status / meta-strip rules ask for "nothing besides these")
    E[tid, WX] = 0.0 if a & {"kick1", "kickh", "status", "meta1", "metah", "compass", "tiny"} else wi
    if "numword" in a:
        E[tid, WX] = 1.0
    if "timeword" in a:
        # "today", "now": filler inside a sentence (wi = 0), but as a kicker on their own they are a date
        # category (K1: a sidebar group "TODAY"), so they count as class-external information
        E[tid, WX] = 1.0
# hash buckets: an unknown stem is a content word: own code, fully informative.
for tid in range(LAY.hash_base, LAY.vocab_size):
    set_code(tid, ("hash", tid))
    E[tid, [WI, int(WN), WX]] = 1.0
if len(_concept) > 16384:
    notes.append(f"{len(_concept)} concepts for 16384 codes: {len(_concept) - 16384} hash buckets share a code "
                 f"with an earlier concept")

# ---------------------------------------------------------------------------------------------------------------
# 4. Offset, role and flag embeddings
# ---------------------------------------------------------------------------------------------------------------

T["off_emb.weight"][:, ONE] = 1.0  # a constant 1 at every position except BOS
T["off_emb.weight"][0, FIRST] = 1.0  # the node's first token

ROLE_DIMS = {
    "h1": [G_H1], "dialog_title": [G_H1], "empty_title": [G_H1],
    "h2": [G_H2], "h3": [G_H2], "h4": [G_H2], "h5": [G_H2], "h6": [G_H2],
    "eyebrow": [G_EYEBROW], "subtitle": [G_SUB], "helper": [G_HELP], "caption": [G_CAP],
    "placeholder": [G_PH], "tooltip": [G_TIP], "body": [G_BODY], "list_item": [G_LI],
    "empty_body": [G_SBODY], "dialog_body": [G_SBODY, R_DLG], "error": [G_SBODY, R_ERR],
    "label": [G_LABEL], "link": [G_LINK], "badge": [G_BADGE], "button": [R_BUTTON],
    # A stat shares the button's dimension: like a button it has no rule of its own (it is never flagged) and it
    # is an anchor that a lower-rank line can repeat ("Foot passengers: 4,753" under the stat "4,753"). The
    # places where "a control" is asked for by name (tooltips, "click X") need a matched word, and a stat's
    # tokens are numbers, which never count as a hit.
    "stat": [R_BUTTON],
}
for role, rid in tk.ROLE_IDS.items():
    for d in ROLE_DIMS.get(role, []):  # every other role has no rule at all: never flagged
        T["role_emb.weight"][rid, d] = 1.0

FLAG_DIMS = {"dynamic": F_DYN, "stateful": F_STF, "conditional": F_COND, "in_nav": F_NAVTAB, "in_table": F_NAVTAB,
             "pill": F_PILL, "dot": F_PILL, "sr_only": F_SR, "section_last": F_LAST, "section_first": F_FIRST,
             "in_card": F_CARD, "in_control": F_CTL, "tracked": F_TRK, "small": F_SMALL, "in_state": F_STATE,
             "repeated": F_REP}
for flag, d in FLAG_DIMS.items():
    T["flag_emb.weight"][tk.FLAG_IDS[flag], d] = 1.0

# key-side role classes (sums of role dims)
K_HEAD = [G_H1, G_H2]
K_HEADE = [G_H1, G_H2, R_ERR]  # headings, and an error line (the first of two error lines acts as the title)
K_CTRL = [R_BUTTON, G_LINK]
K_LOC = [G_H1, G_H2, G_LABEL, R_BUTTON, G_LINK, G_BODY, R_ERR]  # ranks 1-3: what a lower-rank node can restate
LOWRANK = [G_SUB, G_HELP, G_CAP, G_PH, G_TIP, G_BODY, G_LI, G_SBODY]
TEXT3 = [G_SUB, G_HELP, G_CAP]  # the rank-4 text lines
NARR_ROLES = [G_SUB, G_HELP, G_CAP, G_SBODY, G_BODY]

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
        T[self.p + "attn.q.weight"][16 * head + hd, int(dim)] += val

    def qb(self, head, hd, val):
        T[self.p + "attn.q.bias"][16 * head + hd] += val

    def k(self, head, hd, dim, val):
        T[self.p + "attn.k.weight"][16 * head + hd, int(dim)] += val

    def v(self, head, hd, dim, val):
        T[self.p + "attn.v.weight"][16 * head + hd, int(dim)] += val

    def o(self, dim, head, hd, val):
        T[self.p + "attn.o.weight"][int(dim), 16 * head + hd] += val

    def bias(self, head, cols, val):
        T[self.p + "attn.rel_bias"][head, cols] = val

    def unit(self, win, bias, wout):
        u = self.units
        assert u < mdl.M, "MLP layer full"
        for d, c in win.items():
            T[self.p + "mlp.fc1.weight"][u, int(d)] += c
        T[self.p + "mlp.fc1.bias"][u] = bias
        for d, c in wout.items():
            T[self.p + "mlp.fc2.weight"][int(d), u] += c
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
    """A token-level MLP unit: switched off at EON (where the same dims hold per-node sums or booleans)."""
    w = {}
    for d, c in win.items():
        w[int(d)] = w.get(int(d), 0.0) + c
    w[IS_EON] = w.get(IS_EON, 0.0) - G
    L.unit(w, bias, wout)


def ge(L, x, lo, hi, out):
    """EON boolean: 0 when x <= lo, 1 when x >= hi (linear between), from two ReLUs. x is {dim: coef}."""
    s = 1.0 / (hi - lo)
    w = {int(d): c * s for d, c in x.items()}
    w[IS_EON] = w.get(IS_EON, 0.0) + G
    L.unit(w, -lo * s - G, {out: 1.0})
    L.unit(w, -lo * s - 1.0 - G, {out: -1.0})


def rule(L, pos, need, neg, outs, soft=(), negw=None):
    """Rule unit at EON: relu(sum(pos) - (need - 1) - G * sum(neg) - sum(soft)): 1 when `need` of the positive
    booleans hold, no veto is set and the soft terms are 0. A veto may be a boolean or a raw non-negative
    per-node sum whose smallest non-zero value is 0.5 (G * 0.5 = 50 is still far above any sum of positives).
    `negw` adds arbitrary coefficients (used for the veto "information besides numbers": WX - NUM).
    The unit only fires at EON (tok-only dimensions share the booleans' physical dimensions)."""
    w = {IS_EON: G}
    for d in pos:
        w[int(d)] = w.get(int(d), 0.0) + 1.0
    for d in neg:
        w[int(d)] = w.get(int(d), 0.0) - G
    for d in soft:
        w[int(d)] = w.get(int(d), 0.0) - 1.0
    for d, c in (negw or {}).items():
        w[int(d)] = w.get(int(d), 0.0) + c
    L.unit(w, -(need - 1.0) - G, outs)


L1, L2, L3, L4 = Layer(0), Layer(1), Layer(2), Layer(3)

# ---------------------------------------------------------------------------------------------------------------
# 6. Layer 1 attention: matching, code sums
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
match_head(L1, 3, dist_cols(-63, -1), elig(K_HEAD + [R_BUTTON] + TEXT3,
                                           banned=[F_SR, F_DYN, F_COND, F_NAVTAB]), [([ONE], M_G)])
# H4 / H7 reverse matches, used only as pointer keys: is this token repeated by a node that could be restating it?
# H4: any later lower-rank or duplicate-class node; H7: an eyebrow, tooltip or narrating line in the 3 nodes before.
match_head(L1, 4, dist_cols(1, 63), elig(LOWRANK + [G_LABEL, G_H1, G_H2, G_LINK, G_BADGE], banned=[F_SR, F_DYN]),
           [([ONE], RF)])
match_head(L1, 7, dist_cols(-3, -1), elig([G_EYEBROW, G_TIP] + NARR_ROLES, banned=[F_SR, F_DYN]), [([ONE], RB)])
# H5 the two nodes just before (-2..-1), same keys as H0: a subtitle's own anchor. A subtitle sits right under
# its heading (or one node on, after a badge); a heading three or four nodes back belongs to the section or card
# before. The layer-1 MLP weighs a subtitle's words against these two nodes and every other role's against H0.
match_head(L1, 5, dist_cols(-2, -1), elig(K_LOC, bonus=K_HEADE + [G_LABEL] + K_CTRL, banned=[F_SR, F_DYN]),
           [([ONE], M_PN)], state_pair=True)
# H6 the node's code sum (written into the code dims of its EON)
count_head(L1, 6, [(CODE[i], CODE[i]) for i in range(NCODE)])

# ---------------------------------------------------------------------------------------------------------------
# 7. Layer 1 MLP: token features (and, further down, the static EON booleans of the layer-2 MLP)
# ---------------------------------------------------------------------------------------------------------------

# novel = informative weight, cancelled when the concept is covered (m ~ 1; -2*m also cancels a half-split match):
# covered by a rank 1-3 node in the four nodes before (m_p); for a subtitle only by the two nodes before (m_pn).
# A subtitle's heading is the node directly before it (or one further, after a badge), so anything three or
# four back is another section or card: "Everything in Free, plus:" under the plan "Pro" does not restate the
# heading "Free" of the card before. The other roles sit inside a form, card, list or state whose heading can
# be three or four back and is part of what they can restate (placeholder "Name of your project" under the
# label "Name" in the card "Create project"; "The tutors list has nothing in it." under "Nothing here yet" on
# the page "Tutors"). (A first version used the two-node reach for every role but list lines and lost exactly
# those.)
tok_unit(L1, {WI: 1.0, M_P: -2.0, G_SUB: -G}, 0.0, {NOV_P: 1.0})
tok_unit(L1, {WI: 1.0, M_PN: -2.0, G_SUB: 1.0}, -1.0, {NOV_P: 1.0})
# nov_n (novelty against the nodes after) is only used by eyebrows and tooltips; for subtitles, body lines and
# list lines the same dimension is written in layer 2 with the novelty against the whole window (section 9)
tok_unit(L1, {WI: 1.0, G_EYEBROW: 1.0, G_TIP: 1.0, M_N: -2.0}, -1.0, {NOV_N: 1.0})
tok_unit(L1, {WN: 1.0, M_P: -2.0, M_N: -2.0}, 0.0, {NOV_X: 1.0})
# the same against the controls below only (m_nc): what a line says besides naming a button or link under it.
# A word that the heading above also has still counts here ("photo" in "Choose a photo to upload." under
# "Profile photo": it is the object of the instruction, not the name of the button).
tok_unit(L1, {WN: 1.0, M_NC: -2.0}, 0.0, {NOV_CT: 1.0})
# a tooltip must repeat its control word for word: filler words count too (wi + (mt - wi) = every coded word)
tok_unit(L1, {MT: 1.0, WI: -1.0, G_TIP: 1.0, M_P: -2.0}, -1.0, {NOV_P: 1.0})
tok_unit(L1, {MT: 1.0, WI: -1.0, G_TIP: 1.0, M_N: -2.0}, -1.0, {NOV_N: 1.0})
# K7: a scope word ("all", "every", "everything") in a confirmation body says the action reaches further than
# the title's object, unless the title already has it ("Delete all data?" / "... all your data")
tok_unit(L1, {WS: 1.0, R_DLG: 1.0, M_P: -2.0}, -1.0, {NOV_P: 1.0})
# K8 / U14: an imperative. A node that opens with a low-weight instruction verb (add, create, choose, pick,
# enter, try ...) tells the reader what to do next; that is information unless the verb itself is the anchor's.
# The unit adds IMP = 64 to the node's novelty instead of 1, so that two readers can tell it apart from ordinary
# novelty (which is at most 31): against a heading or title the imperative is news (b_novp: novelty >= 0.75);
# against a field label it is only an input lead ("Enter your postcode" under "Postcode", "Choose a size." under
# "Size": R5, R8) and b_novpl ignores the 64.
IMP = 64.0
#  - in an empty, error or dialog body: unless the title has the verb (K8 keeps a next action even when a button
#    repeats it)
tok_unit(L1, {IL: 1.0, FIRST: 1.0, G_SBODY: 1.0, M_PH: -2.0}, -2.0, {NOV_P: IMP})
#  - in a subtitle, body line, list line, caption, helper or placeholder: unless a node just before or a control
#    just after has the verb ("Save your changes." above "Save changes" is the control's own label)
#    ("just before" has the reach of the role's novelty unit: m_p, and m_pn for a subtitle)
IMP_ROLES = {G_BODY: 1.0, G_LI: 1.0, G_CAP: 1.0, G_HELP: 1.0, G_PH: 1.0}
tok_unit(L1, {IL: 1.0, FIRST: 1.0, M_P: -2.0, M_NC: -2.0, **IMP_ROLES}, -2.0, {NOV_P: IMP})
tok_unit(L1, {IL: 1.0, FIRST: 1.0, M_PN: -2.0, M_NC: -2.0, G_SUB: 1.0}, -2.0, {NOV_P: IMP})
# The lead verb's own weight is taken out again in these roles (and in state bodies), so that the 64 is all it
# contributes: against a label "Upload a photo for your profile." under "Profile picture" leaves nothing.
# relu(wi + il + first + role - 3 - 2 m) equals the role's novelty unit exactly when il = first = role = 1.
tok_unit(L1, {WI: 1.0, IL: 1.0, FIRST: 1.0, M_P: -2.0, G_SBODY: 1.0, **IMP_ROLES}, -3.0, {NOV_P: -1.0})
tok_unit(L1, {WI: 1.0, IL: 1.0, FIRST: 1.0, M_PN: -2.0, G_SUB: 1.0}, -3.0, {NOV_P: -1.0})


def hit_unit(roles, ms, out, extra=None, need=1.0):
    """1 when a token of a node in `roles` matched a key of the listed anchor classes (numbers and particles
    never count: a shared "3" or "in" is not a shared concept)."""
    w = {NUM: -1.0, NOHIT: -1.0}
    for d in roles:
        w[d] = 1.0
    for d in ms:
        w[d] = 1.0
    if extra:
        w.update(extra)
    tok_unit(L1, w, -need, {out: 1.0})


hit_unit([G_SUB, G_PH, G_BODY, G_HELP, G_CAP], [M_PH], HIT_P)  # restates the heading before it
hit_unit([G_SUB, G_PH, G_BODY, G_HELP, G_CAP], [M_PL], HIT_PL)  # restates the label before it
hit_unit([G_TIP], [M_PC, M_PL], HIT_P)  # tooltip: the control before it
hit_unit([G_LI, G_SBODY, G_LABEL], [M_PH], HIT_P)  # list line, state body, card label: the heading
hit_unit([G_EYEBROW], [M_N1H], HIT_N)  # eyebrow: the heading directly after it
hit_unit([G_TIP], [M_NC], HIT_N)  # tooltip: the control after it
hit_unit(NARR_ROLES, [M_N], HIT_NC)  # narration naming a control, label or heading below
hit_unit([G_SUB], [M_NC], HIT_NC0, extra={FIRST: 1.0}, need=2.0)  # ... as its first word ("Resume does ...")
# A control's own tokens (button, link; the hit_n dimension is otherwise only written for eyebrows and tooltips):
# 1 for each coded word of the label that no narrating line in the three nodes before repeats (rb, from H7). The
# per-node sum at the control's EON is 0 exactly when its whole label is named just before it ("Choose Save
# changes to continue." above [Save changes]); [New playlist] after "Pick a playlist." keeps "new" uncovered.
tok_unit(L1, {MT: 1.0, R_BUTTON: 1.0, G_LINK: 1.0, RB: -2.0}, -1.0, {HIT_N: 1.0})

# matched code: mc_i = code_i when the token is matched through a gate that fits its node's role, else 0.
# relu(s * code_i + 3 * m (+ 3 * role - 3) - 3) is 1 only for code_i = s and m = 1; two gates open: 2 * code.
MC_GATES = [  # (gate, roles it applies to, a match that closes the gate)
    (M_P, LOWRANK + [G_LABEL], None),  # a lower-rank node's word found in the nodes just before (not an eyebrow's)
    (M_N1H, [G_EYEBROW], None),  # an eyebrow's word in the heading after it
    (M_NC, [G_TIP] + NARR_ROLES, None),  # a tooltip's or a narrating line's word in a control below
    # a repeated title / call to action: its earlier copy
    (M_G, [G_H1, G_H2, G_LINK], None),
    # a repeated fact (R14 text): a helper / caption / subtitle word found in an earlier text line, but only when
    # it was NOT found just before: a word that the label or heading right above already gives must not pull the
    # pointer to some far heading that happens to share it (round 1 had no target for repeated facts for that
    # reason; the m_p condition is what was missing)
    (M_G, TEXT3, M_P),
]
for i in range(NMC):
    for sgn in (1.0, -1.0):
        for gate, roles, closed_by in MC_GATES:
            w = {CODE[i]: sgn, gate: 3.0, NOHIT: -6.0}
            w.update({d: 3.0 for d in roles})
            bias = -6.0
            if closed_by is not None:
                # ... and only for a full-weight word: a frame verb ("view", "see") matched in some far text line
                # pulled "View and manage your team." to a field label "Display name"
                w[closed_by] = -6.0
                w[WI] = 3.0
                bias = -9.0
            tok_unit(L1, w, bias, {MC[i]: sgn})

# pointer gate: the token is repeated by some node that could be restating it (rf or rb), clamped to 1
tok_unit(L1, {RF: 1.0, RB: 1.0}, 0.0, {F_ANY: 1.0})
tok_unit(L1, {RF: 1.0, RB: 1.0}, -1.0, {F_ANY: -1.0})

# static EON booleans (layer-2 MLP, from the sums of layer-2 head 4; they stand here with the other lexical
# features). Counts are integers (x0.997), half weights 0.5: thresholds sit between the values.
ge(L2, {WI: 1.0, STATUS: -1.0}, 0.3, 0.45, B_XS)  # any informative word that is not a status word
ge(L2, {STATUS: 1.0}, 0.4, 0.8, B_STATUS)
ge(L2, {DEIC: 1.0}, 0.6, 0.9, B_DEIC)  # a whole layout word ("above" alone is half)
ge(L2, {UIN: 1.0, NUM: 1.0}, 0.4, 0.8, B_UINNUM)
ge(L2, {UIN: 1.0}, 0.4, 0.8, B_UIN)
ge(L2, {UIN: 1.0, NARR: 1.0}, 1.4, 1.8, B_UIN2)
ge(L2, {ACT: 1.0}, 0.55, 0.7, B_ACT)  # cue weights are multiples of 0.25: 0.5 (one weak verb) is not a cue, 0.75 is
ge(L2, {PROMO: 1.0}, 0.4, 0.8, B_PROMO)
ge(L2, {WI: 1.0, PROMO: -1.0}, 0.3, 0.45, B_XP)  # any informative word that is not a promo-pill word
ge(L2, {DEM: 1.0}, 0.4, 0.8, B_DEM)
ge(L2, {NUM: 1.0}, 0.4, 0.8, B_NUM)
ge(L2, {SUMM: 1.0}, 0.6, 0.9, B_SUMM)  # any summary cue
ge(L2, {MT: 1.0}, 1.4, 1.8, B_MT2)
ge(L2, {MT: 1.0}, 2.4, 2.8, B_MT3)
ge(L2, {MT: 1.0, NUM: -1.0}, 0.4, 0.8, B_WORD)  # a coded word that is not a number ("12" as a heading is a figure)
ge(L2, {WI: 1.0, UNIT: -1.0}, 1.4, 1.8, B_WI2U)  # two informative tokens that are not units / qualifiers (K13)
# genre kicker: the kicker weights add up to a whole kicker (one full word, or two halves: "how" + "works")
ge(L2, {KICK: 1.0}, 0.6, 0.9, B_KICK)
# meta strip: a whole meta word ("GMT", "SECTOR"), or half of one beside a number ("N 50.74", "Local time 16:45",
# "12.5° S"); values are multiples of 0.25
ge(L2, {META: 1.0}, 0.3, 0.45, B_METAH)
ge(L2, {TINY: 1.0}, 0.4, 0.8, B_TINY)

# squared length of the code sum over the duplicate head's bits (EON). Each entry x is an integer, and
# x^2 = |x| + 2 * sum_{k>=1} relu(|x| - k) exactly for |x| <= 11 (linear beyond: only very long nodes).
DUP_BITS = 10
for i in range(DUP_BITS):
    for sgn in (1.0, -1.0):
        for kk in range(0, 11):
            L1.unit({CODE[i]: sgn, IS_EON: G}, -kk - G, {NORM2: 1.0 if kk == 0 else 2.0})

# ---------------------------------------------------------------------------------------------------------------
# 8. Layer 2 attention: per-node sums, duplicate head
# ---------------------------------------------------------------------------------------------------------------

count_head(L2, 0, [(d, d) for d in DYN_COUNTED + STATIC_B] + [(NOV_CT, NOVC)])
count_head(L2, 4, [(d, d) for d in STATIC_A])  # per-node sums of the static lexical classes

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
    (14, TEXT3, TEXT3 + [G_BODY], False),  # a text line repeats an earlier one (R14)
]
count_head(L2, 2, [(d, d) for d in MC])  # the node's matched-code sum (pointer query)
# H3 "anywhere" match: does this token's concept occur in any other node of the window (any distance, any role,
# static text only)? Feeds the layer-4 rule for summary lines whose content is all elsewhere on the screen.
# EON keys are banned: in this layer they hold code sums, whose dot product with a code is noise. No state
# pairing here: a noun that the page's empty state or dialog also uses ("No recipes") is still a noun the page
# is about, which is all the two layer-4 rules ask.
# The ban on EON keys must be larger than any code-sum product (31 tokens x 14 bits x 6 = 2604), not the usual
# -60: with -60 the EON of a long node whose code sum happened to line up with a word's code (about 1 in 100 for
# an eight-word node) counted as "found elsewhere", and an unknown word ("studio") was then covered by chance;
# which words, depended on the order in which codes are handed out.
_any = elig([ONE], banned=[F_SR, F_DYN])
_any[IS_EON] = -4000.0
match_head(L2, 3, dist_cols(-63, -1) + dist_cols(1, 63), _any, [([ONE], M_ANY)])
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

# A word found in both the heading and the label before it splits its attention between the two keys (0.5 each;
# a third anchor: 0.33), so these two thresholds sit below a third; attention leaks are under 0.02 per token.
ge(L2, {HIT_P: 1.0}, 0.15, 0.3, B_HITP)
ge(L2, {HIT_PL: 1.0}, 0.15, 0.3, B_HITPL)
ge(L2, {HIT_N: 1.0}, 0.5, 0.9, B_HITN)
# b_novpl: the same test as b_novp with the imperative's 64 taken out. Ordinary novelty is below 40 and the
# count head scales sums by 0.997..1, so: on from 0.75, off again above 50-55 (an imperative is present: 63.8-64
# alone, at most 64.5 with one half-weight word), on again from 64.8 (imperative plus a content word).
ge(L2, {NOV_P: 1.0}, 0.6, 0.85, B_NOVPL)
ge(L2, {NOV_P: -1.0}, -55.0, -50.0, B_NOVPL)  # +1 below 50, 0 above 55 ...
L2.unit({IS_EON: G}, -G + 1.0, {B_NOVPL: -1.0})  # ... minus the constant 1: net -1 from 55 upwards
ge(L2, {NOV_P: 1.0}, IMP + 0.55, IMP + 0.75, B_NOVPL)
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
ge(L2, {NOV_X: 1.0}, 0.6, 0.85, B_NOVX1)
ge(L2, {AUX: 1.0}, 1.4, 1.8, B_LIST)  # two list cues and no clause marker (each marker is -100)
# Token unit (layer 2): for subtitles, body lines and list lines, the novelty against the whole window: the
# information weight of a word found neither just before (m_p) nor anywhere else (m_any). It goes into the nov_n
# dimension, which layer 1 leaves empty for these roles; the layer-3 count head sums it into NOVG.
tok_unit(L2, {WI: 1.0, G_SUB: 1.0, G_BODY: 1.0, G_LI: 1.0, M_P: -2.0, M_ANY: -2.0}, -1.0, {NOV_N: 1.0})
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
VETO = [VAR, F_DYN]  # runtime text or alternative branches are never flagged (VAR is the raw marker count)
# narration wording makes NARRATE the class even when echo rules fire too (two of them can: -10 outweighs both)
NARR_OUT = {O_NARR: R_LEX, O_ECHO: -10.0}
XNUM = {WX: -G, NUM: G}  # veto: any class-external information besides numbers (WX - NUM >= 0.5)


def rule3(pos, neg, outs, soft=(), negw=None, alt=()):
    """All of `pos` must hold, plus one of `alt` when given (role alternatives: a node has exactly one role)."""
    need = len(pos) + (1 if alt else 0)
    rule(L3, list(pos) + list(alt), need, list(neg), outs, soft=soft, negw=negw)


# ECHO, local
rule3([G_EYEBROW, B_HITN], VETO + [B_NOVN2], {O_ECHO: R_HIT})  # R1
# Against a heading (b_hitp) an imperative lead counts as news (b_novp); against a field label (b_hitpl) it is an
# input lead and does not (b_novpl).
# (not a placeholder: one that shares a word with the section heading and has no label of its own, "Recipe name"
# under "New recipe", is the field's only name: the rule layer's R8b, never an echo)
rule3([B_HITP], VETO + [B_NOVP], {O_ECHO: R_HIT}, alt=[G_SUB, G_HELP])  # R3 R5, heading before
rule3([B_HITPL], VETO + [B_NOVPL], {O_ECHO: R_HIT}, alt=[G_SUB, G_HELP, G_PH])  # R5 R8, label before
# A caption needs a second coded word: a lone noun under a heading that contains it ("Bed" under "Busiest beds")
# is a column header or a field-style label (K1, K5), not a restatement.
rule3([G_CAP, B_HITP, B_MT2], VETO + [B_NOVP], {O_ECHO: R_HIT})  # R5
rule3([G_CAP, B_HITPL], VETO + [B_NOVPL], {O_ECHO: R_HIT})  # R5 (under a field label one word is enough)
rule3([G_TIP, B_HITP], VETO + [B_NOVP], {O_ECHO: R_EXACT})  # R5 tooltip after its control
rule3([G_TIP, B_HITN], VETO + [B_NOVN], {O_ECHO: R_EXACT})  # R5 tooltip before its control
rule3([G_BODY, F_SMALL, B_HITP], VETO + [B_NOVP], {O_ECHO: R_HIT})  # R5 stat-card note
rule3([G_BODY, F_SMALL, B_HITPL], VETO + [B_NOVPL], {O_ECHO: R_HIT})  # R5 stat-card note under its label
# R4 first line of a section. Body text is where real explanations live, so here not even a half-weight word may
# be left over (NOV_P read raw): "Your recent activity will show up here." under "Recent activity" is the
# section's only statement that it is empty.
rule3([G_BODY, F_FIRST, B_HITP], VETO + [NOV_P], {O_ECHO: R_HIT})
rule3([G_LI, F_CARD, B_HITP], VETO + [B_NOVP], {O_ECHO: R_HIT})  # R4 plan-card line
rule3([G_SBODY, B_HITP], VETO + [B_NOVP, F_CTL], {O_ECHO: R_HIT})  # R4 (not an inline field error: K12)
# R3 / R17 a subtitle, or the closing line of a section, made of filler words only says nothing (WI is the raw
# sum of information weights: any half-weight word vetoes)
rule3([G_SUB, B_MT2], VETO + [WI], {O_ECHO: R_LEX})
# (a closing line inside a conditional branch or a state cluster is usually live status text, "One moment…":
# the two closing-line rules leave those alone)
rule3([F_LAST, B_MT2], VETO + [WI, F_COND, F_STATE], {O_ECHO: R_LEX}, alt=[G_BODY, G_LI])
# The same with three or more coded words: then one half-weight word is tolerated, and words already given by
# the nodes just before do not count ("Everything you've saved, in one spot."; a closing line that only repeats
# the section). b_novp is the novelty against the previous four nodes, which is the whole weight when nothing
# matches. Not on repeated rows (their words repeat by design, K5).
rule3([G_SUB, B_MT3], VETO + [B_NOVP], {O_ECHO: R_LEX})
rule3([F_LAST, B_MT3], VETO + [B_NOVP, F_REP, F_COND, F_STATE], {O_ECHO: R_LEX}, alt=[G_BODY, G_LI])
# Near miss (ranking only, never a call): a shared word and exactly one informative word left over. 0.25 stays
# under the 0.5 a call needs; it orders KEEP nodes for the precision-recall curve (margin -0.25 instead of -0.5).
rule3([B_HITP], VETO + [B_NOVP2, F_CTL], {O_ECHO: 0.25}, alt=[G_SUB, G_HELP, G_CAP, G_PH, G_SBODY])
# ECHO, same bag of concepts as an earlier node (DL1 = 0)
rule3([B_WORD], VETO, {O_ECHO: R_EXACT}, soft=[DL1], alt=[G_H1, G_H2])  # R9 (two equal figures are not a title)
# R5 a card label equal to the card heading just before it (a word must also match a heading within 4 nodes:
# a field label far below a heading that happens to use the same words is not an echo)
rule3([G_LABEL, B_HITP], VETO + [F_CTL, F_NAVTAB], {O_ECHO: R_EXACT}, soft=[DL1])
rule3([G_LINK, B_MT2], VETO + [F_COND, F_NAVTAB], {O_ECHO: R_EXACT}, soft=[DL1])  # R14 call to action
rule3([B_WI2U], VETO + [F_COND, F_REP], {O_ECHO: R_EXACT}, soft=[DL1], alt=TEXT3)  # R14 fact
# ECHO, closing line (R17): last body line of a section, a summary word, at most one informative word
# (WI is read raw: 4 * (1.25 - WI) is 1 at one informative word and negative from one and a half; the L4 cap
# takes care of the larger values below one)
L3.unit({IS_EON: G, F_LAST: G, int(B_SUMM): G, G_BODY: G, G_LI: G, WI: -4.0, VAR: -G, F_DYN: -G},
        -4.0 * G + 5.0, {O_ECHO: R_LEX})
# DECOR
# R2 an eyebrow / kicker-styled label whose words add up to a section-genre kicker and nothing else (WX: raw sum
# of class-external information)
rule3([G_EYEBROW, B_KICK], VETO + [WX], {O_DECOR: R_LEX})
rule3([G_LABEL, F_TRK, B_KICK], VETO + [WX, F_CTL, F_NAVTAB], {O_DECOR: R_LEX})  # not a nav group label
# R2 decorative meta strip (coordinates, local time, sector / node / index dressing): meta words and numbers only
rule3([B_METAH, B_NUM], VETO, {O_DECOR: R_LEX}, negw=XNUM, alt=[G_CAP, G_HELP, G_EYEBROW])
def graded_rule(musts, graded, neg, outs):
    """Fires `graded` (a flag dimension that two flags share, so it can be 2: pill + dot, in_nav + in_table)
    when every boolean in `musts` holds (one of each inner list) and no veto is set. A plain rule would let a 2
    stand in for a missing boolean (an empty badge with both flags was a "single-letter pill")."""
    w = {IS_EON: G, int(graded): 1.0}
    for m in musts:
        for d in (m if isinstance(m, (list, tuple)) else [m]):
            w[int(d)] = w.get(int(d), 0.0) + G
    for d in neg:
        w[int(d)] = w.get(int(d), 0.0) - G
    L3.unit(w, -G * (len(musts) + 1), outs)


# R2 a heading inside navigation that only names what follows ("Contents", "On this page", "Menu")
graded_rule([G_H2, B_TINY], F_NAVTAB, VETO + [WX, F_REP], {O_DECOR: R_LEX})
# R11 a static pill holding a single letter (a monogram): no data, no state
graded_rule([G_BADGE, B_TINY], F_PILL, VETO + [WX, F_STF, F_COND, F_REP], {O_DECOR: R_LEX})
# R11 static status. Not on repeated sibling items (a status there usually differs between siblings: K2), not on
# list lines (dotted list lines are a legend: K2), never on controls, headings, cells or stats.
ST_VETO = VETO + [B_XS, F_STF, F_COND, F_NAVTAB, F_REP]
rule3([G_BADGE, B_STATUS], ST_VETO, {O_DECOR: R_LEX})
graded_rule([B_STATUS, [G_BODY, G_CAP, G_HELP, G_SUB, G_LABEL, G_EYEBROW]], F_PILL, ST_VETO, {O_DECOR: R_LEX})
# R11 the same promo pill ("New", "Fresh", "Popular") repeated on sibling cards. Only promo words: a repeated
# tag ("TypeScript" on two project cards) is data that differs between siblings (K2).
rule3([G_BADGE, B_PROMO], VETO + [F_STF, F_NAVTAB, B_XP], {O_DECOR: R_EXACT}, soft=[DL1])
# NARRATE (HID: a hidden-input cue such as "hold", "long-press", "shortcut" makes the line a K4 hint, never narration)
NV = VETO + [HID]
rule3([B_DEIC, B_UINNUM], NV + [B_NOVX2], NARR_OUT, alt=NARR_ROLES)  # R16 "four options below"
rule3([B_DEIC, B_HITNC], NV + [B_NOVX2, B_UINNUM], NARR_OUT, alt=NARR_ROLES)  # "enter your email below"
# "Below you'll find our pricing." under "Pricing": a layout word and nothing new at all (LABELLING 3: "below
# you'll find"). Stricter novelty than the other narration rules because nothing else anchors it.
rule3([B_DEIC], NV + [B_NOVX1, B_UINNUM, B_HITNC], NARR_OUT, alt=NARR_ROLES)
# R16 "this page lists ..." / "here are your key numbers": graded on 2 - NUM like the count rule below, so that
# two numbers ("four of your 16": a partial count, a fact) switch it off
_w = {IS_EON: G, int(B_DEM): G, int(B_UIN2): G, NUM: -1.0, int(B_NOVX2): -G}
for _d in NARR_ROLES:
    _w[_d] = G
for _d in NV:
    _w[_d] = -G
L3.unit(_w, -4.0 * G + 2.0, NARR_OUT)
# R16 "here are three numbers": a literal count is ONE number. "Here are four of your 16 key numbers" is a partial
# count, which is a fact (LABELLING 8: "340 invoices" above a paged table). The unit is graded on 2 - NUM behind
# G-weighted booleans: 1 with one number token, 0 with two.
_w = {IS_EON: G, int(B_DEM): G, int(B_NUM): G, int(B_UIN): G, NUM: -1.0, int(B_NOVX2): -G}
for _d in NARR_ROLES:
    _w[_d] = G
for _d in NV:
    _w[_d] = -G
L3.unit(_w, -5.0 * G + 2.0, NARR_OUT)
# R6 "click the button below": a whole action cue plus an interface noun, a layout word, or a word of a node
# below. Three rules instead of one OR-boolean (the layer-4 cap makes two of them firing count once).
rule3([B_ACT, B_UIN], NV + [B_NOVX2], NARR_OUT, alt=NARR_ROLES)
rule3([B_ACT, B_DEIC], NV + [B_NOVX2], NARR_OUT, alt=NARR_ROLES)
rule3([B_ACT, B_HITNC], NV + [B_NOVX2], NARR_OUT, alt=NARR_ROLES)
# R6 / codex U11 ("a button label named inside the body plus a narration verb"): half a cue is enough when the
# line names a control below in full (b_ctl, from the layer-3 "named control" head) and says NOTHING else:
# "Choose Resume to finish.", "Hit Save to proceed.", "Select Cancel to continue." "Choose", "select", "pick",
# "hit" are not interface words by themselves, so the evidence has to be that the verb's object is the control;
# word order is invisible here, and the test that stands in for it is that the line has no word of narration
# weight outside the controls below it (NOVC read raw: even a half-weight word vetoes; a word the heading
# above shares still counts). Any other content word could be the real object, and then the line is an
# instruction (K8): "Choose a photo to upload." under "Profile photo" above [Upload], "Choose a plan to
# upgrade." above [Upgrade], "Pick a playlist." above [New playlist] (label not named in full). A line with a
# figure is a fact ("Switch to annual and save 20%"). Graded on 4 * ACT - 1 behind G-weighted booleans: 0 at a
# quarter cue ("on"), 1 at a half (larger values are cut by the layer-4 cap).
_w = {IS_EON: G, int(B_CTL): G, int(B_HITNC): G, ACT: 4.0, int(NOVC): -G, int(B_NUM): -G}
for _d in NARR_ROLES:
    _w[_d] = G
for _d in NV:
    _w[_d] = -G
L3.unit(_w, -4.0 * G - 1.0, NARR_OUT)
# The same condition at exactly half a cue (b_act still 0) sets b_act for the pointer head in layer 4, which gives
# a button +3 when the line has an action cue: the line names the button, so that is its target, not a title
# above that happens to hold the same word ("Choose Delete to get started." under "Delete batch?").
_w = dict(_w)
_w[int(B_ACT)] = -G
L3.unit(_w, -4.0 * G - 1.0, {B_ACT: 1.0})
rule3([B_ACT, B_HITPL], NV + [B_NOVX2], NARR_OUT, alt=[G_HELP, G_CAP])  # R6 "toggle <label>"
rule3([B_ACT, B_HITP], NV + [B_NOVX2], NARR_OUT, alt=[G_HELP, G_CAP])  # R6 the same under a heading
# R6 "<Button> does ...": a subtitle whose first word is the name of a control below, with at most two and a half
# words of narration novelty. Graded on 11 - 4 * NOV_X behind G-weighted booleans (novelty comes in halves:
# 1 at 2.5, nothing from 2.75; larger values are cut by the layer-4 cap), which decides exactly as the boolean
# "nov_x >= 3" did and leaves its dimension to NOVC.
_w = {IS_EON: G, G_SUB: G, int(B_HITNC0): G, NOV_X: -4.0, int(B_NUM): -G}
for _d in NV:
    _w[_d] = -G
L3.unit(_w, -3.0 * G + 11.0, NARR_OUT)

# ---------------------------------------------------------------------------------------------------------------
# 11. Layer 4 attention: the pointer head (attention pattern only; it writes nothing)
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
# The head sits in layer 4 (round 2; layer 3 until then, with the same weights): everything it reads is unchanged
# between the two layers (matched-code sums and static booleans at EON; codes, flags and roles at tokens), and in
# layer 4 the b_act it reads has been raised by the layer-3 rule "half a cue + a control named in full", so that
# "Choose Delete to get started." points at the button [Delete] and not at the title "Delete batch?".
PL = L4
PTR = 0
PTR_BITS = 11
for i in range(PTR_BITS):
    PL.q(PTR, i, MC[i], math.sqrt(8.0))
    PL.k(PTR, i, CODE[i], math.sqrt(8.0))
PL.qb(PTR, 11, 1.0)
for d in (G_H1, G_H2, G_LABEL, R_BUTTON, G_LINK, R_ERR):
    PL.k(PTR, 11, d, 12.0)
PL.k(PTR, 11, IS_EON, -4000.0)
PL.q(PTR, 12, B_ACT, 1.0)
PL.q(PTR, 12, B_DEIC, 1.0)  # "the button below" names a control as much as "click" does
# (Tried and reverted: the same bonus for a line with a number, to send a stat note to its stat instead of the
# stat's label. Most lines with a number restate a heading that carries the number, and the bonus pulled them
# to a button that shares one word: dev pointer accuracy 91.0% -> 89.9%.)
PL.k(PTR, 12, R_BUTTON, 12.0)
# (Tried and taken out: +2 more for a button token that a narrating line just before it repeats (rb), so that of
# two buttons with the same word the one right after the line wins over an earlier one. It changed no call on
# dev or the probes: where it would matter, the earlier button is usually narrated as well.)
for d in (G_H1, G_H2, G_LABEL, G_LINK):
    PL.q(PTR, 13, d, 1.0)
for d in K_HEAD + [R_BUTTON]:
    PL.k(PTR, 13, d, 24.0)
# gate: a key token that no candidate restater repeats loses 24. An eyebrow's target comes after it, so for an
# eyebrow only "repeated just before" (rb) counts; for every other role either flag does.
for d in (G_SUB, G_HELP, G_CAP, G_PH, G_TIP, G_BODY, G_LI, G_SBODY, G_LABEL, G_H1, G_H2, G_LINK, G_BADGE):
    PL.q(PTR, 14, d, 1.0)
PL.k(PTR, 14, F_ANY, 96.0)
PL.k(PTR, 14, ONE, -96.0)
PL.q(PTR, 15, G_EYEBROW, 1.0)
PL.k(PTR, 15, RB, 96.0)
PL.k(PTR, 15, ONE, -96.0)
for d in range(-63, 64):
    PL.bias(PTR, d + 63, max(-0.6, -0.05 * abs(d)) - (1.0 if d > 0 else 0.0))
PL.bias(PTR, 63, BIG_OFF)  # never the node itself
PL.bias(PTR, mdl.BOS_COL, 7.0)

# ---------------------------------------------------------------------------------------------------------------
# 11b. Layer 3 attention, head 2: the "named control" head. EON of a line -> the EON of a control (button, link)
#      in the next three nodes whose whole label is named by the narrating lines before it: it has a word that is
#      not a number (b_word) and no uncovered word (b_hitn = 0 at a control's EON: the sum written by the layer-1
#      unit for control tokens). Scores are exact: an eligible key 0, any other key -60 or less, BOS -30; the head
#      writes the constant 1 of the key, so b_ctl is 1 when such a control exists and 0 otherwise. Never dynamic
#      or sr-only controls; a line outside a state cluster does not see a control inside one (LABELLING 2a).
#      The layer-3 MLP reads b_ctl in the same layer (rule "half a cue + a control named in full").
# ---------------------------------------------------------------------------------------------------------------
CTLH = 2
L3.qb(CTLH, 0, 1.0)
for d, c in ((ONE, -180.0), (IS_EON, 60.0), (R_BUTTON, 60.0), (G_LINK, 60.0), (B_WORD, 60.0), (B_HITN, -60.0),
             (F_DYN, -60.0), (F_SR, -60.0)):
    L3.k(CTLH, 0, d, 4.0 * c)  # in score units: 60 * (is_eon + control + b_word - 3) - 60 * (uncovered, dyn, sr)
L3.qb(CTLH, 1, 4.0)
L3.q(CTLH, 1, IS_EON, -4.0)  # 4 * (1 - is_eon): token queries ...
L3.k(CTLH, 1, ONE, -200.0)  # ... see -200 on every key but BOS and write nothing
L3.qb(CTLH, 14, 1.0)
L3.q(CTLH, 14, F_STATE, -1.0)  # 1 when the query's node is not in a state cluster
L3.k(CTLH, 14, F_STATE, -240.0)  # -60 for a key inside one
L3.bias(CTLH, slice(None), OFF)
L3.bias(CTLH, dist_cols(1, 3), 0.0)
L3.bias(CTLH, mdl.BOS_COL, -30.0)
L3.v(CTLH, 0, ONE, 1.0)
L3.o(B_CTL, CTLH, 0, 1.0)

# ---------------------------------------------------------------------------------------------------------------
# 12. Layer 4 MLP: caps. Two layer-3 rules firing on one node are not twice the evidence: each class is capped
#     at the level of its strongest kind of rule (3 for ECHO and DECOR, 2 for NARRATE, whose rules are all
#     lexical). The two layer-4 rules below sit in the same MLP, so what they add (2 each) comes on top of the
#     capped layer-3 evidence: a subtitle that restates its heading and is also a summary line of the whole
#     screen ends at 4.5. That only orders nodes that are flagged anyway.
# ---------------------------------------------------------------------------------------------------------------
for dim, cap in ((O_ECHO, R_EXACT), (O_DECOR, R_EXACT), (O_NARR, R_LEX)):
    L4.unit({int(dim): 1.0, IS_EON: G}, -cap - G, {dim: -1.0})

# ---------------------------------------------------------------------------------------------------------------
# 12b. Layers 3-4: summary lines whose content is all elsewhere on the screen (LABELLING 3: ECHO with no single
#      target "when the information is spread over several"; R16 "copy describing the screen itself"; R17).
#      A subtitle that carries a whole summary frame ("... at a glance", "... in one place", "an overview /
#      a summary of ...", "everything about ...", "... all together": cue weight >= 4) and whose informative words
#      all occur in other nodes of the window says nothing the screen does not show: "A summary of your orders."
#      on a dashboard whose cards are about orders. The frame is required: a sentence that merely shares its nouns
#      with the rest of the screen can still say something new about them (who may do what, what happens when),
#      which a bag of concepts cannot see. A first version also took closing lines and any loose cue ("that",
#      "covers"): on pricing pages, where every noun recurs, it flagged plan descriptions ("For small clubs that
#      share games.") and closing facts ("And that's on every plan, with no setup fee.").
#      L2 H3 gives m_any per token, the L2 MLP the per-token novelty against the window (in the nov_n dimension),
#      the L3 count head below sums it into NOVG, and the L4 unit reads NOVG as a raw veto (any uncovered
#      half-weight word keeps the node) and the raw cue sum SUMM against its threshold of 4.
# ---------------------------------------------------------------------------------------------------------------
count_head(L3, 1, [(NOV_N, NOVG)])
def clamped(L, w, bias, outs):
    """A graded unit clamped to [0, 1]: relu(x) - relu(x - 1)."""
    L.unit(w, bias, outs)
    L.unit(w, bias - 1.0, {d: -c for d, c in outs.items()})


clamped(L4, {IS_EON: G, G_SUB: G, SUMM: 1.0, VAR: -G, F_DYN: -G, int(NOVG): -G}, -2.0 * G - 3.0, {O_ECHO: R_LEX})
# 12c. The enumerating intro: a subtitle that lists what the nodes around it show
#      ("Academic foundations across business, computer science, systems engineering and networking."
#      over an education list whose entries name those subjects). Conditions: the shape of a list
#      (b_list: at least two of comma / "and" / "&", and no auxiliary, modal or relative pronoun: a sentence that
#      says who can do what, or what happens, has one); no number (a figure is a fact); and enough of the list
#      is on the screen: covered - 2 * uncovered >= 4 informative words, where covered = WI - NOVG is given
#      elsewhere in the window and uncovered = NOVG is not (three or fewer covered words is an ordinary subtitle
#      that shares nouns with the page; each word found nowhere else must be outweighed by two that are). An
#      imperative lead costs 2 (its 64 in nov_p, / 32). No heading hit is required (bias -3G - 3: three
#      booleans and a threshold of 4): what makes an enumeration redundant is that the screen shows the listed
#      things, not that the line shares a word with its heading.
#      A first version without the list shape and the number veto fired on pricing-page facts whose nouns recur
#      across the page ("Cancel free until 24 hours before.", "7,882 plot holders used Furrow last month.").
clamped(L4, {IS_EON: G, G_SUB: G, int(B_LIST): G, WI: 1.0, int(NOVG): -3.0, NOV_P: -1.0 / 32.0,
             NUM: -G, VAR: -G, F_DYN: -G}, -3.0 * G - 3.0, {O_ECHO: R_LEX})

# ---------------------------------------------------------------------------------------------------------------
# 13. Output head
# ---------------------------------------------------------------------------------------------------------------

T["head.weight"][1, int(O_ECHO)] = 1.0
T["head.weight"][2, int(O_NARR)] = 1.0
T["head.weight"][3, int(O_DECOR)] = 1.0
T["head.bias"][1:] = -0.5  # a rule must fire to at least a quarter before a node is flagged

sha = mdl.save_artifact(WORK, T, vocab, (4, PTR))  # pointer head: layer 4 (1-based), head 0
# for ./inspect: a short list of EON features (the full name -> dim table goes to dims.json)
SHOW = {"wi": WI, "mt": MT, "wx": WX, "kick": KICK, "meta": META, "nov_p": NOV_P, "nov_n": NOV_N, "nov_x": NOV_X,
        "hit_p": HIT_P, "hit_n": HIT_N, "hit_nc": HIT_NC, "dl1(eon)": DL1, "o_echo(eon)": O_ECHO,
        "o_narr(eon)": O_NARR, "o_decor(eon)": O_DECOR}
(WORK / "directions.json").write_text(json.dumps({n: [int(d)] for n, d in SHOW.items()}) + "\n")
(WORK / "dims.json").write_text(json.dumps({NAMES[i]: i for i in sorted(NAMES)}, indent=0) + "\n")
nz = sum(int(np.count_nonzero(v)) for v in T.values())
print(f"vocab {len(vocab)} stems ({len(HAND)} hand, {n_thes} thesaurus), {len(_concept)} concepts, "
      f"{_next[0]} residual dims, units {[L.units for L in (L1, L2, L3, L4)]}, non-zero params {nz}, "
      f"artefact {sha[:12]}")
for n in notes:
    print("note:", n)
