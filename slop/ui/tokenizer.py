"""Slop tokeniser (SPEC.md §6) and sequence builder (§4, §7). Mirrored exactly by ``lint/tokenizer.js``.

Standalone: standard library plus ``porter.py`` and the two JSON tables beside this file (``ids.json``: the fixed
id table and the closed role and flag lists; ``unicode_tables.json``: pinned Unicode data, see
``gen_unicode_tables.py``).

Per node text, in order:

1. NFKC, except that code points unassigned in Python's Unicode version (15.0) and the two marker code points
   are left alone; unassigned ones become ``SYMBOL``.
2. One left-to-right scan. At each position the first rule that applies wins:
   U+E000 is the ``VAR`` marker and U+E001 the ``ALT`` marker (the extractor writes them for an interpolation and
   between the branches of a conditional; no other private-use code point is a marker); a run of three or more
   full stops is ``ELLIPSIS`` (NFKC turns ``…`` into ``...``); a glyph listed in ``ids.json`` is its class (``DOT``,
   ``MIDDOT``, ``ARROW``, ``DASH``, ``ELLIPSIS``, ``CHECK``, ``STAR``, ``SPARKLE``); a pictograph
   (``Extended_Pictographic`` from U+2300 up, or a regional indicator) is one ``EMOJI`` together with its variation
   selectors, skin-tone modifiers, tags, keycap mark, zero-width-joined pictographs or second regional indicator;
   an unassigned code point is ``SYMBOL``; white space and control characters separate tokens; format characters
   and stray combining marks are dropped; a letter run (letters and marks, with apostrophes ``'`` or ``’`` allowed
   between letters, and format characters such as a soft hyphen or zero-width space inside it dropped) is words; ASCII digits are a number; ``_`` separates; ASCII punctuation has its own id; U+2010
   (hyphen) is ``-``; anything else is ``SYMBOL``.
3. Every letter run (``VAR`` and ``ALT`` spelt in letters are ordinary words) counts towards the ``caps`` flag (at
   least two capitals and no lower-case letter) and the ``title_case`` flag (not caps, at least two runs, every run
   starting with a capital), and is split into words: between a lower-case and an upper-case
   letter, and before the last capital of an upper-case run that is followed by a lower-case letter
   (``HTTPServer`` -> ``HTTP Server``). Each word is case folded, NFKD-decomposed with non-spacing and enclosing
   marks dropped, stripped of apostrophes, and stemmed with ``porter.stem``. The tokeniser's ``caps`` and
   ``title_case`` are ORed with the extractor's flags of the same name (see ``encode_node``).
4. Numbers (``1,234`` grouping and ``.5`` decimals allowed): a decimal is ``DEC``; grouped digits are ``NUM``;
   otherwise the value without leading zeros is its own id when 0-99, ``YEAR`` when 1900-2099, else ``NUM``. An
   integer part of two or more digits that starts with ``0`` is preceded by ``LZ``.
5. A node keeps at most 31 tokens; a longer one is cut and gets the ``truncated`` flag.

Ids: see ``ids.json`` and SPEC.md §6. A stem in ``vocab.txt`` (line n) is 512 + n; any other stem is
4096 + (first 4 bytes of sha256(stem), big-endian, mod 6144).

A screen (``slop.screen/1``) becomes one or more windows. A window is ``BOS`` and then, per node, its tokens and
``EON``; at most 64 nodes and 1,024 positions. Consecutive nodes with the same cluster index form a cluster run;
runs are packed greedily into windows, and a run that does not fit in an empty window is split at node boundaries.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

try:
    from . import porter
except ImportError:  # top-level module, as in the author kit's lib/
    import porter

_HERE = Path(__file__).resolve().parent
IDS = json.loads((_HERE / "ids.json").read_text(encoding="utf-8"))
_TABLES = json.loads((_HERE / "unicode_tables.json").read_text(encoding="utf-8"))

ROLES: tuple[str, ...] = tuple(IDS["roles"])
FLAGS: tuple[str, ...] = tuple(IDS["flags"])
CONTEXT_ROLES: frozenset[str] = frozenset(IDS["context_roles"])
CLASSES: tuple[str, ...] = tuple(IDS["classes"])
ROLE_IDS: dict[str, int] = {r: i for i, r in enumerate(ROLES)}
FLAG_IDS: dict[str, int] = {f: i for i, f in enumerate(FLAGS)}
assert len(ROLES) == 36 and len(FLAGS) == 28

PAD = IDS["special"]["PAD"]
BOS = IDS["special"]["BOS"]
EON = IDS["special"]["EON"]
VAR = IDS["special"]["VAR"]
ALT = IDS["special"]["ALT"]
LZ = IDS["special"]["LZ"]
MARKERS: dict[str, int] = {IDS["markers"]["VAR"]: VAR, IDS["markers"]["ALT"]: ALT}  # U+E000, U+E001
assert sorted(MARKERS) == ["\ue000", "\ue001"]

GLYPH_CLASSES: tuple[str, ...] = tuple(IDS["glyph_classes"])
GLYPH_IDS: dict[str, int] = {g: IDS["glyph_base"] + i for i, g in enumerate(GLYPH_CLASSES)}
GLYPH_OF_CHAR: dict[str, str] = {ch: cls for cls, chars in IDS["glyph_chars"].items() for ch in chars}
PUNCT_IDS: dict[str, int] = {ch: IDS["punct_base"] + i for i, ch in enumerate(IDS["punct_chars"])}
NUMBER_BASE = IDS["number_base"]
YEAR = IDS["number_tokens"]["YEAR"]
NUM = IDS["number_tokens"]["NUM"]
DEC = IDS["number_tokens"]["DEC"]
VOCAB_BASE = IDS["vocab_base"]
VOCAB_MAX = IDS["vocab_max"]
HASH_BASE = IDS["hash_base"]
HASH_BUCKETS = IDS["hash_buckets"]
V = IDS["vocab_size"]
MAX_NODE_TOKENS = IDS["max_node_tokens"]
MAX_WINDOW_NODES = IDS["max_window_nodes"]
MAX_WINDOW_POSITIONS = IDS["max_window_positions"]
assert HASH_BASE + HASH_BUCKETS == V and VOCAB_BASE + VOCAB_MAX == HASH_BASE

_ASSIGNED_STARTS = [a for a, _ in _TABLES["assigned"]]
_ASSIGNED_ENDS = [b for _, b in _TABLES["assigned"]]
_EMOJI_STARTS = [a for a, _ in _TABLES["emoji"]]
_EMOJI_ENDS = [b for _, b in _TABLES["emoji"]]

_APOSTROPHES = "'’"
_HYPHEN = "‐"
_NUMBER = re.compile(r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")


class TokenizerError(ValueError):
    pass


def _in_ranges(cp: int, starts: list[int], ends: list[int]) -> bool:
    i = bisect.bisect_right(starts, cp) - 1
    return i >= 0 and cp <= ends[i]


def is_assigned(ch: str) -> bool:
    return _in_ranges(ord(ch), _ASSIGNED_STARTS, _ASSIGNED_ENDS)


def _is_regional(cp: int) -> bool:
    return 0x1F1E6 <= cp <= 0x1F1FF


def is_pictograph(ch: str) -> bool:
    cp = ord(ch)
    return _is_regional(cp) or _in_ranges(cp, _EMOJI_STARTS, _EMOJI_ENDS)


def _cat(ch: str) -> str:
    return unicodedata.category(ch)


def _is_upper(ch: str) -> bool:
    return _cat(ch) in ("Lu", "Lt")


def _is_lower(ch: str) -> bool:
    return _cat(ch) == "Ll"


def _nfkc_protected(text: str) -> str:
    """NFKC on runs of assigned code points; unassigned ones and the markers pass through untouched."""
    out: list[str] = []
    run: list[str] = []
    for ch in text:
        if is_assigned(ch) and ch not in MARKERS:
            run.append(ch)
        else:
            if run:
                out.append(unicodedata.normalize("NFKC", "".join(run)))
                run = []
            out.append(ch)
    if run:
        out.append(unicodedata.normalize("NFKC", "".join(run)))
    return "".join(out)


def camel_split(run: str) -> list[str]:
    """Split one letter run at lower-to-upper changes and before the last capital of an upper-case run."""
    parts: list[str] = []
    start = 0
    for p in range(1, len(run)):
        a, b = run[p - 1], run[p]
        if (_is_lower(a) and _is_upper(b)) or (
            _is_upper(a) and _is_upper(b) and p + 1 < len(run) and _is_lower(run[p + 1])
        ):
            parts.append(run[start:p])
            start = p
    parts.append(run[start:])
    return parts


def fold_word(part: str) -> str:
    """Case folding, NFKD with non-spacing and enclosing marks dropped, apostrophes removed."""
    s = unicodedata.normalize("NFKD", part.casefold())
    return "".join(ch for ch in s if _cat(ch) not in ("Mn", "Me") and ch not in _APOSTROPHES)


def hash_bucket(stem: str) -> int:
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    return HASH_BASE + int.from_bytes(digest[:4], "big") % HASH_BUCKETS


def number_tokens(text: str) -> list[int]:
    """Ids for one matched number string."""
    out: list[int] = []
    int_part = text.split(".")[0]
    if len(int_part) >= 2 and int_part[0] == "0":
        out.append(LZ)
    if "." in text:
        out.append(DEC)
    elif "," in text:
        out.append(NUM)
    else:
        digits = text.lstrip("0") or "0"
        if len(digits) <= 2:
            out.append(NUMBER_BASE + int(digits))
        elif len(digits) == 4 and IDS["year_range"][0] <= int(digits) <= IDS["year_range"][1]:
            out.append(YEAR)
        else:
            out.append(NUM)
    return out


# -- vocabulary ---------------------------------------------------------------------------------------------------


class Vocab:
    """``vocab.txt``: one Porter stem per line; line n is id 512 + n."""

    def __init__(self, stems: list[str]) -> None:
        if len(stems) > VOCAB_MAX:
            raise TokenizerError(f"vocab has {len(stems)} stems (max {VOCAB_MAX})")
        seen: dict[str, int] = {}
        for n, s in enumerate(stems):
            if not s or s != s.lower() or any(ch.isspace() for ch in s):
                raise TokenizerError(f"vocab line {n + 1}: {s!r} is empty, not lower-case or has white space")
            if s in seen:
                raise TokenizerError(f"vocab line {n + 1}: {s!r} duplicates line {seen[s] + 1}")
            seen[s] = n
        self.stems = list(stems)
        self.ids = {s: VOCAB_BASE + n for n, s in enumerate(stems)}

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return cls(lines)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def word_id(self, stem: str) -> int:
        return self.ids.get(stem, hash_bucket(stem))


EMPTY_VOCAB = Vocab([])


# -- one node -----------------------------------------------------------------------------------------------------


@dataclass
class Token:
    id: int
    kind: str  # word | number | punct | glyph | marker
    surface: str
    word: str | None = None  # folded word before stemming (words only)
    stem: str | None = None


@dataclass
class TextTokens:
    tokens: list[Token]
    flags: list[str]  # computed: caps, title_case, truncated (in FLAGS order)
    dropped: int  # tokens cut by the 31-token cap

    @property
    def ids(self) -> list[int]:
        return [t.id for t in self.tokens]


def _emoji_end(text: str, i: int) -> int:
    """End (exclusive) of the emoji sequence that starts at ``i``."""
    n = len(text)
    first = ord(text[i])
    j = i + 1
    if _is_regional(first) and j < n and _is_regional(ord(text[j])):
        return j + 1
    while j < n:
        cp = ord(text[j])
        if cp in (0xFE0E, 0xFE0F, 0x20E3) or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F:
            j += 1
        elif cp == 0x200D and j + 1 < n and is_pictograph(text[j + 1]):
            j += 2
        else:
            break
    return j


def tokenize_text(text: str, vocab: Vocab = EMPTY_VOCAB) -> TextTokens:
    s = _nfkc_protected(text)
    n = len(s)
    toks: list[Token] = []
    runs: list[str] = []
    i = 0
    while i < n:
        ch = s[i]
        if ch in MARKERS:
            toks.append(Token(MARKERS[ch], "marker", ch))
            i += 1
            continue
        if ch == "." and s.startswith("...", i):
            j = i
            while j < n and s[j] == ".":
                j += 1
            toks.append(Token(GLYPH_IDS["ELLIPSIS"], "glyph", s[i:j]))
            i = j
            continue
        if ch in GLYPH_OF_CHAR:
            toks.append(Token(GLYPH_IDS[GLYPH_OF_CHAR[ch]], "glyph", ch))
            i += 1
            continue
        if is_pictograph(ch):
            j = _emoji_end(s, i)
            toks.append(Token(GLYPH_IDS["EMOJI"], "glyph", s[i:j]))
            i = j
            continue
        if not is_assigned(ch):
            toks.append(Token(GLYPH_IDS["SYMBOL"], "glyph", ch))
            i += 1
            continue
        c = _cat(ch)
        if c[0] == "Z" or c == "Cc":
            i += 1
            continue
        if c == "Cf" or c[0] == "M":
            i += 1
            continue
        if c[0] == "L":
            j = i + 1
            while j < n:
                cj = _cat(s[j]) if is_assigned(s[j]) else "Cn"
                if cj[0] in ("L", "M") or cj == "Cf":
                    j += 1
                elif s[j] in _APOSTROPHES and j + 1 < n and is_assigned(s[j + 1]) and _cat(s[j + 1])[0] == "L":
                    j += 1
                else:
                    break
            run = "".join(ch for ch in s[i:j] if _cat(ch) != "Cf")
            i = j
            runs.append(run)
            for part in camel_split(run):
                word = fold_word(part)
                if not word:
                    continue
                st = porter.stem(word)
                toks.append(Token(vocab.word_id(st), "word", part, word, st))
            continue
        if "0" <= ch <= "9":
            m = _NUMBER.match(s, i)
            assert m is not None
            for tid in number_tokens(m.group(0)):
                toks.append(Token(tid, "number", m.group(0)))
            i = m.end()
            continue
        if ch == "_":
            i += 1
            continue
        if ch in PUNCT_IDS:
            toks.append(Token(PUNCT_IDS[ch], "punct", ch))
            i += 1
            continue
        if ch == _HYPHEN:
            toks.append(Token(PUNCT_IDS["-"], "punct", ch))
            i += 1
            continue
        toks.append(Token(GLYPH_IDS["SYMBOL"], "glyph", ch))
        i += 1

    flags: list[str] = []
    n_upper = sum(1 for r in runs for ch in r if _is_upper(ch))
    n_lower = sum(1 for r in runs for ch in r if _is_lower(ch))
    caps = n_upper >= 2 and n_lower == 0
    if caps:
        flags.append("caps")
    if not caps and len(runs) >= 2 and all(_is_upper(r[0]) for r in runs):
        flags.append("title_case")
    dropped = max(0, len(toks) - MAX_NODE_TOKENS)
    if dropped:
        toks = toks[:MAX_NODE_TOKENS]
        flags.append("truncated")
    return TextTokens(toks, flags, dropped)


# -- screens and windows ------------------------------------------------------------------------------------------


@dataclass
class NodeEnc:
    index: int  # position in the screen's node list
    i: int  # the node's own ``i``
    k: int  # cluster index
    role: str
    role_id: int
    flags: list[str]  # extractor flags plus computed ones, in FLAGS order
    flag_mask: int
    text: TextTokens


@dataclass
class Window:
    tokens: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    roles: list[int] = field(default_factory=list)  # -1 at BOS
    clusters: list[int] = field(default_factory=list)  # raw cluster index; -1 at BOS
    flags: list[int] = field(default_factory=list)  # bit mask over FLAGS; 0 at BOS
    node: list[int] = field(default_factory=list)  # 0 at BOS, 1.. for the window's nodes
    nodes: list[int] = field(default_factory=list)  # screen node list index per window node
    eon: list[int] = field(default_factory=list)  # EON position per window node

    def to_json(self) -> dict:
        return {
            "tokens": self.tokens, "offsets": self.offsets, "roles": self.roles, "clusters": self.clusters,
            "flags": self.flags, "node": self.node, "nodes": self.nodes, "eon": self.eon,
        }


@dataclass
class EncodedScreen:
    id: str | None
    nodes: list[NodeEnc]
    windows: list[Window]


def encode_node(index: int, node: dict, vocab: Vocab = EMPTY_VOCAB) -> NodeEnc:
    role = node.get("role")
    if role not in ROLE_IDS:
        raise TokenizerError(f"node {index}: unknown role {role!r}")
    given = node.get("flags") or []
    for f in given:
        if f not in FLAG_IDS:
            raise TokenizerError(f"node {index}: unknown flag {f!r}")
    text = node.get("text")
    if not isinstance(text, str):
        raise TokenizerError(f"node {index}: text must be a string")
    tt = tokenize_text(text, vocab)
    names = set(given) | set(tt.flags)
    flags = [f for f in FLAGS if f in names]
    mask = 0
    for f in flags:
        mask |= 1 << FLAG_IDS[f]
    k = node.get("k")
    if not isinstance(k, int) or k < 0:
        raise TokenizerError(f"node {index}: cluster index k must be a non-negative integer")
    return NodeEnc(index, int(node.get("i", index)), k, role, ROLE_IDS[role], flags, mask, tt)


def split_windows(nodes: list[NodeEnc]) -> list[list[int]]:
    """Node list indices per window (SPEC.md §4 Windows)."""
    runs: list[list[int]] = []
    for n in nodes:
        if runs and nodes[runs[-1][-1]].k == n.k:
            runs[-1].append(n.index)
        else:
            runs.append([n.index])

    def cost(idx: list[int]) -> int:
        return sum(len(nodes[x].text.tokens) + 1 for x in idx)

    windows: list[list[int]] = []
    cur: list[int] = []
    for run in runs:
        if len(cur) + len(run) <= MAX_WINDOW_NODES and 1 + cost(cur) + cost(run) <= MAX_WINDOW_POSITIONS:
            cur.extend(run)
            continue
        if cur:
            windows.append(cur)
            cur = []
        if len(run) <= MAX_WINDOW_NODES and 1 + cost(run) <= MAX_WINDOW_POSITIONS:
            cur = list(run)
            continue
        for x in run:  # a cluster that does not fit an empty window: split at node boundaries
            if len(cur) + 1 > MAX_WINDOW_NODES or 1 + cost(cur) + cost([x]) > MAX_WINDOW_POSITIONS:
                windows.append(cur)
                cur = []
            cur.append(x)
    if cur:
        windows.append(cur)
    return windows


def build_window(nodes: list[NodeEnc], idx: list[int]) -> Window:
    w = Window(tokens=[BOS], offsets=[0], roles=[-1], clusters=[-1], flags=[0], node=[0])
    for wn, x in enumerate(idx, start=1):
        ne = nodes[x]
        for off, tid in enumerate(ne.text.ids + [EON]):
            w.tokens.append(tid)
            w.offsets.append(off)
            w.roles.append(ne.role_id)
            w.clusters.append(ne.k)
            w.flags.append(ne.flag_mask)
            w.node.append(wn)
        w.nodes.append(x)
        w.eon.append(len(w.tokens) - 1)
    return w


def encode_screen(screen: dict, vocab: Vocab = EMPTY_VOCAB) -> EncodedScreen:
    if screen.get("schema") != "slop.screen/1":
        raise TokenizerError(f"not a slop.screen/1 document (schema {screen.get('schema')!r})")
    raw = screen.get("nodes")
    if not isinstance(raw, list):
        raise TokenizerError("screen has no node list")
    nodes = [encode_node(x, nd, vocab) for x, nd in enumerate(raw)]
    windows = [build_window(nodes, idx) for idx in split_windows(nodes)]
    return EncodedScreen(screen.get("id"), nodes, windows)


def load_screen(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
