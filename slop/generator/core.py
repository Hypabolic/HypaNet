"""Shared pieces of the dev generator: text slots with their planted labels, the abstract UI tree, the source
writer that records each slot's span, and the seeded random source.

A screen is built in three steps. A composer (``compose.py``) builds an abstract tree of ``E`` elements whose text
lives in ``Slot`` objects, and decides each slot's label as it plants (or does not plant) a pattern. A renderer
(``render_tsx.py``, ``render_html.py``, ``render_swift.py``) writes real source code for the tree through a
``Writer``, which records the span of every slot's visible characters in the extractor's coordinates (1-based
lines, code-point columns, exclusive end). ``build.py`` then runs the real extractor and joins labels to nodes by
span.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

# LABELLING.md §9 scored patterns (R18 is track C, deferred: the extractor emits no comment nodes in r1)
PATTERN_CLASS = {
    "R1": "ECHO", "R3": "ECHO", "R4": "ECHO", "R5": "ECHO", "R8": "ECHO", "R9": "ECHO", "R14": "ECHO", "R17": "ECHO",
    "R18": "ECHO", "R6": "NARRATE", "R16": "NARRATE", "R2": "DECOR", "R11": "DECOR",
}
PLANTABLE = [p for p in PATTERN_CLASS if p != "R18"]
K_RULES = [f"K{i}" for i in range(1, 15)]
PLANTABLE_K = [k for k in K_RULES if k != "K10"]  # K10 is about code comments (track C)


class GenError(RuntimeError):
    """A generator invariant failed. The build stops: nothing is written."""


@dataclass(eq=False)
class Slot:
    """One piece of visible copy and its label (SPEC.md §5).

    ``role`` is the role the composer means the text to have; the extractor decides the real one. ``cls`` None
    with ``aside`` UNRESOLVED marks a node with no literal words. ``context`` slots (aria, alt) get no label record.
    """

    text: str
    role: str
    cls: str | None = "KEEP"
    pattern: str | None = None
    target: "Slot | None" = None
    equiv: list = field(default_factory=list)
    conf: str = "clear"
    ct: bool = False
    partial: bool = False
    aside: list = field(default_factory=list)
    context: bool = False
    span: tuple | None = None

    def mark(self, cls, pattern=None, target=None, conf="clear", ct=False, aside=None):
        if cls != "KEEP" and pattern is not None and PATTERN_CLASS.get(pattern) != cls:
            raise GenError(f"pattern {pattern} is not a {cls} pattern")
        self.cls, self.pattern, self.target, self.conf, self.ct = cls, pattern, target, conf, ct
        if aside:
            self.add_aside(*aside)
        return self

    def echo(self, target, pattern, conf="clear"):
        return self.mark("ECHO", pattern, target, conf)

    def narrate(self, pattern, target=None, ct=False, conf="clear"):
        return self.mark("NARRATE", pattern, target, conf, ct)

    def decor(self, pattern, conf="clear"):
        return self.mark("DECOR", pattern, None, conf)

    def keep(self, krule=None, ct=False, conf="clear", partial=False, aside=None):
        self.mark("KEEP", krule, None, conf, ct, aside)
        self.partial = partial
        return self

    def add_aside(self, *codes):
        for c in codes:
            if c not in self.aside:
                self.aside.append(c)
        return self

    @property
    def planted(self) -> bool:
        return self.cls not in (None, "KEEP")


def S(text: str, role: str, **kw) -> Slot:
    return Slot(text, role, **kw)


def ctx(text: str, role: str) -> Slot:
    """A context-only slot (aria, alt): read by the model, never labelled."""
    return Slot(text, role, cls=None, context=True)


def dyn(words: str, role: str) -> Slot:
    """An interpolated value with no literal words around it: labelled class null, aside UNRESOLVED."""
    return Slot(words, role, cls=None, aside=["UNRESOLVED"])


def stack(top: Slot, copies: list, patterns: list, conf="clear"):
    """Three or more nodes naming the same thing: keep ``top``, flag each copy, list the others in ``equiv``."""
    for c, p in zip(copies, patterns):
        c.echo(top, p, conf)
    for c in copies:
        c.equiv = [o for o in copies if o is not c]


class E:
    """An element of the abstract UI tree: ``kind``, child elements and properties (slots among them)."""

    __slots__ = ("kind", "kids", "p")

    def __init__(self, kind: str, *kids, **props):
        self.kind = kind
        self.kids = [k for k in kids if k is not None]
        self.p = props

    def __getitem__(self, key):
        return self.p.get(key)

    def walk(self):
        yield self
        for k in self.kids:
            yield from k.walk()


@dataclass
class Page:
    """A whole screen: the component (or document, or view) name, its kind and its top-level elements."""

    name: str
    kind: str
    body: list
    product: str = ""
    r19: list = field(default_factory=list)  # lists of slots that repeat one device (screen aside R19)
    props: list = field(default_factory=list)  # component props used by bound expressions (TSX)
    nav_title: Slot | None = None  # SwiftUI .navigationTitle

    def slots(self) -> list:
        """Every slot in the tree, each once, in tree order (properties, then children)."""
        out = []
        seen = set()

        def visit(v):
            if isinstance(v, Slot):
                if id(v) not in seen:
                    seen.add(id(v))
                    out.append(v)
            elif isinstance(v, E):
                for x in v.p.values():
                    visit(x)
                for k in v.kids:
                    visit(k)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    visit(x)

        visit(self.nav_title)
        for top in self.body:
            visit(top)
        return out


class Writer:
    """Source text with a cursor in the extractor's coordinates; ``put`` records a slot's span."""

    def __init__(self):
        self.parts: list[str] = []
        self.line = 1
        self.col = 1
        self.written: list[Slot] = []

    def raw(self, s: str):
        if not s:
            return
        self.parts.append(s)
        nl = s.count("\n")
        if nl:
            self.line += nl
            self.col = len(s) - s.rfind("\n")
        else:
            self.col += len(s)

    def put(self, slot: Slot, raw: str):
        """Write the visible characters of ``slot`` (already escaped for the language) and record their span."""
        if slot.span is not None:
            raise GenError(f"slot {slot.text!r} written twice")
        if raw != raw.strip():
            raise GenError(f"slot text has outer whitespace: {raw!r}")
        l1, c1 = self.line, self.col
        self.raw(raw)
        slot.span = (l1, c1, self.line, self.col)
        self.written.append(slot)

    def shift(self, lines: int):
        """Move every span written here down by ``lines`` (when this text is preceded by other text)."""
        for s in self.written:
            l1, c1, l2, c2 = s.span
            s.span = (l1 + lines, c1, l2 + lines, c2)

    def ln(self, indent: int, *pieces):
        """One line: indentation, then strings and (slot, raw) pairs."""
        self.raw("  " * indent)
        for p in pieces:
            if isinstance(p, tuple):
                self.put(*p)
            else:
                self.raw(p)
        self.raw("\n")

    def text(self) -> str:
        return "".join(self.parts)


class Rng(random.Random):
    """Seeded randomness. A screen's stream depends only on (seed, namespace, index, attempt)."""

    @classmethod
    def for_screen(cls, seed: int, namespace: str, index: int, attempt: int = 0) -> "Rng":
        h = hashlib.sha256(f"slop-gen|{seed}|{namespace}|{index}|{attempt}".encode()).digest()
        return cls(int.from_bytes(h[:8], "big"))

    def chance(self, p: float) -> bool:
        return self.random() < p

    def pick(self, xs):
        return xs[self.randrange(len(xs))]

    def wpick(self, pairs):
        """Pick from [(value, weight), ...]."""
        total = sum(w for _, w in pairs)
        x = self.random() * total
        for v, w in pairs:
            x -= w
            if x < 0:
                return v
        return pairs[-1][0]

    def some(self, xs, n):
        n = max(0, min(n, len(xs)))
        idx = sorted(self.sample(range(len(xs)), n))
        return [xs[i] for i in idx]

    def shuffled(self, xs):
        ys = list(xs)
        self.shuffle(ys)
        return ys


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def lower_first(s: str) -> str:
    """Lower-case the first letter unless the first word is an acronym or a name-like token."""
    if not s:
        return s
    w = s.split(" ")[0]
    if len(w) > 1 and w.isupper():
        return s
    return s[:1].lower() + s[1:]


def title_case(s: str) -> str:
    small = {"a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "at", "by", "or"}
    out = []
    for i, w in enumerate(s.split(" ")):
        out.append(w if (i and w in small) else (w[:1].upper() + w[1:]))
    return " ".join(out)


def ident(s: str, upper_first=True) -> str:
    """A PascalCase or camelCase identifier from words."""
    words = [w for w in "".join(ch if ch.isalnum() else " " for ch in s).split() if w]
    if not words:
        return "Thing"
    out = "".join(w[:1].upper() + (w[1:] if any(ch.isupper() for ch in w[1:]) and not w.isupper() else w[1:].lower())
                  for w in words)
    if out[0].isdigit():
        out = "N" + out
    return out if upper_first else out[:1].lower() + out[1:]


def slug(s: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split()) or "x"
