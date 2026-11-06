"""v0.5 edits reader: the author's replies -> a v0.3 state dict and an apply report. Harness only.

The author writes the weights of the unchanged v0.3 transformer (``v03.model.PARAM_SHAPES``, torch
``state_dict`` names and layouts, Linear weight ``[out, in]``) as statements of the Exp 5 "edits"
language inside fenced blocks opened by a line of three backticks followed by ``edits``. Every
weight starts at exactly 0.0; the network is never run here. No ``eval``/``exec``: a small lexer
and a recursive-descent parser read the statements.

**Blocks (multi-reply deviation from Exp 5).** A run is an ordered list of replies. EVERY edits
block of EVERY reply is read and applied, in the order written, so a later statement (in the same
or a later reply) overwrites an earlier one. Block rules, per reply (a block never continues into
the next reply; the author must open a new block):

* a block opens at a fence line (three or more backticks or tildes, optional leading whitespace)
  whose info string's first word is ``edits`` (any case); other fenced blocks are not tracked;
* it closes at the next fence line of any kind (``closed_by`` ``fence``, or ``fence_with_info``
  when that line has an info string; a line opening another edits block closes this one and opens
  the next), at a line that is exactly ``END OF WEIGHTS`` (``end_marker``), or at the end of the
  reply: ``end_of_reply`` (the block is ``cut_off``) when the reply was cut off by the output limit
  or it is not known whether it was, ``end_of_reply_not_cut`` when the reply is known to have ended
  normally (the block is simply unclosed, and its last statement is read like any other).

**Replies.** A reply is one API message of the author. A run directory is read through
``replies/INDEX.json`` when the launcher wrote one: every turn is read as its parts
(``NNN.p<K>.md``, one per API message) when Claude Code continued a cut reply inside the turn, else
as ``NNN.md``, each with whether it was cut off (``cut_off`` true / false / null = unknown). So the
cut-off rule below applies at every cut, also at one inside a turn. Without an index, every
``NNN.md`` is a reply whose cut-off is unknown (treated as possibly cut).

**Statements.** One per line; ``#`` starts a comment anywhere (also inside a list). A statement
continues on the next line while a ``[`` is open or after a trailing ``=``/``+=``/``-=``/``*=``
or ``,``. A line that starts with a tensor name always starts a new statement, so an unclosed
bracket costs only its own statement. Forms::

    name[i, j] = v                  one entry
    name[a:b, j] = [v, ...]         explicit values along a range (as many values as entries)
    name[a:b] = v                   scalar fill
    name[a:b:s, j] = [...]          every s-th index (a:b half-open; ':' alone = whole axis;
                                    a:, :b, ::s and a::s also accepted)
    name[a:b, c:d] = [[...], ...]   2-D list for a 2-D block ('[ [' with spaces is fine)
    name[...] *= v                  scale; also += and -= (v a scalar, or a list of the block shape)

Policies (decided for v0.5, recorded in every report under ``reader``):

* an integer index drops its axis (``W[0:6, 3]`` is a 6-vector); missing trailing indices mean
  the whole axis (``rel_bias[3]`` is head 3's row; a bare name is the whole tensor);
* a list must have the selected block's shape, or the same shape once length-1 axes are removed
  (``W[0:1, 0:3] = [1, 2, 3]`` is accepted); ragged, empty or mismatched lists are rejected;
* **negative indices are rejected** (no Python wrap-around); integer indices must be ``< n``;
  a range needs ``0 <= a < b <= n`` and step ``s >= 1`` (no silent clamping);
* numbers: plain decimals with an optional sign (``-0.5``, ``.5``, ``3.``, ``+2``; a Unicode minus
  is read as ``-``). Exponent notation (``1e-3``, ``2.5E+2``) is unambiguous and **accepted**.
  ``nan``, ``inf``, fractions, expressions and hex are not numbers. Every literal and every result
  of ``*=``/``+=``/``-=`` must be finite with ``|v| <= 1e6``; values are stored as float32;
* a statement that cannot be read or applied (unknown tensor, too many indices, out-of-range or
  negative index, shape mismatch, bad value, stray character) is skipped and recorded with its
  reply, line and reason; it is never fatal and the others still apply;
* lines outside edits blocks are never applied; the ones that look like a statement (a tensor
  name followed by ``[`` or an operator) or like the rest of a list are counted per reply
  (``outside_block_statement_lines``, with the first few under ``outside_block_examples``) so that
  weights written outside a block are visible in the report;
* in a cut-off block, the trailing statement is **incomplete** (skipped, recorded under
  ``cut_off_statements``, not under ``errors``) when a ``[`` is still open, when it ends with an
  operator or comma, when it does not parse, or when its last line has no newline after it (so a
  number may be truncated) and does not end in ``]`` or a comment. Launchers must therefore write
  each reply's text verbatim, without stripping a final newline.

``END OF WEIGHTS``: a reply "ends with the marker" when its last non-blank line, stripped, is
exactly ``END OF WEIGHTS`` (:func:`ends_with_end_marker`, which the launcher should reuse).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..v03.model import PARAM_COUNT, PARAM_SHAPES

READER_VERSION = "v05-edits-2"
END_MARKER = "END OF WEIGHTS"
MAX_ABS_VALUE = 1e6
MAX_LIST_DEPTH = 8
MAX_ERRORS_LISTED = 2000
OUTSIDE_EXAMPLES = 5
INDEX_FILE = "INDEX.json"
STATEMENT_TEXT_CHARS = 300

REPLIES_DIR = "replies"
ARTIFACT_FILE = "artifact.dlw"
ARTIFACT_SHA_FILE = "artifact.sha256"
APPLY_FILE = "apply.json"

POLICY = {
    "name": "agent_loop_lab.v05.edits",
    "version": READER_VERSION,
    "blocks": "every edits block of every reply, in order; a block ends at a fence, an END OF WEIGHTS "
              "line or the end of its reply",
    "max_abs_value": MAX_ABS_VALUE,
    "exponent_numbers": "accepted",
    "negative_indices": "rejected",
    "range_bounds": "0 <= a < b <= n, step >= 1, no clamping",
    "missing_trailing_indices": "whole axis",
    "list_shape": "equal to the block shape, or equal once length-1 axes are removed",
    "cut_off_trailing_statement": "skipped when open, ending in an operator or comma, unparsable, or on an "
                                  "unterminated last line not ending in ']' or a comment; applied only to a reply "
                                  "that was cut off or whose ending is unknown",
    "replies": "one per API message: replies/INDEX.json parts when a turn has several, else NNN.md",
    "outside_block_lines": "never applied; statement-like ones are counted per reply",
    "dtype": "float32",
}

__all__ = [
    "END_MARKER",
    "READER_VERSION",
    "ApplyResult",
    "Block",
    "EditError",
    "apply_replies",
    "build_run",
    "ends_with_end_marker",
    "extract_blocks",
    "reading_units",
    "reply_files",
    "split_statements",
]


class EditError(ValueError):
    """One statement cannot be read or applied (it is skipped; never fatal)."""


class RunDirError(ValueError):
    """The run directory cannot be read (missing or misnumbered replies)."""


# -- blocks -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    number: int  # 1-based line number in the reply
    text: str
    terminated: bool  # a newline follows this line in the reply


@dataclass
class Block:
    reply: int  # 1-based reply index in the run
    index_in_reply: int  # 1-based
    open_line: int
    close_line: int | None = None
    closed_by: str = "end_of_reply"
    lines: list[Line] = field(default_factory=list)

    @property
    def cut_off(self) -> bool:
        return self.closed_by == "end_of_reply"


END_OF_REPLY_NOT_CUT = "end_of_reply_not_cut"


_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(.*?)\s*$")


def _split_lines(text: str) -> list[Line]:
    parts = text.split("\n")
    out = []
    for i, p in enumerate(parts):
        if p.endswith("\r"):
            p = p[:-1]
        out.append(Line(i + 1, p, i < len(parts) - 1))
    return out


def _is_edits_info(info: str) -> bool:
    words = info.split()
    return bool(words) and words[0].lower() == "edits"


def extract_blocks(text: str, reply: int = 1, cut_off: bool | None = None,
                   outside: list[Line] | None = None) -> list[Block]:
    """Every edits block of one reply, in order (see the module docstring for the rules).

    ``cut_off``: whether the reply ended at the output limit (``None``: unknown, treated as possibly cut).
    ``outside``: when given, collects the lines that are in no edits block (fence and marker lines excluded)."""
    blocks: list[Block] = []
    current: Block | None = None
    for line in _split_lines(text):
        m = _FENCE.match(line.text)
        if current is None:
            if m and _is_edits_info(m.group(2)):
                current = Block(reply, len(blocks) + 1, line.number)
            elif outside is not None and not m and line.text.strip() != END_MARKER:
                outside.append(line)
            continue
        if m or line.text.strip() == END_MARKER:
            current.close_line = line.number
            current.closed_by = "end_marker" if m is None else ("fence_with_info" if m.group(2) else "fence")
            blocks.append(current)
            current = None
            if m and _is_edits_info(m.group(2)):
                current = Block(reply, len(blocks) + 1, line.number)
            continue
        current.lines.append(line)
    if current is not None:
        if cut_off is False:
            current.closed_by = END_OF_REPLY_NOT_CUT
        blocks.append(current)
    return blocks


def _statement_like(line: Line) -> bool:
    """A line outside any edits block that reads like a statement or like the rest of a list."""
    toks = _lex(line.text.partition("#")[0], line.number)
    if not toks or any(t.kind == "bad" for t in toks):
        return False
    if toks[0].kind == "name":
        return toks[0].text in PARAM_SHAPES and len(toks) > 1 and (toks[1].text == "[" or toks[1].kind == "op")
    kinds = {t.kind for t in toks}
    texts = {t.text for t in toks}
    return kinds <= {"num", "punct"} and "num" in kinds and ":" not in texts and bool(texts & {",", "[", "]"})


def ends_with_end_marker(text: str) -> bool:
    """True when the reply's last non-blank line, stripped, is exactly ``END OF WEIGHTS``."""
    for line in reversed(text.split("\n")):
        if line.strip():
            return line.strip() == END_MARKER
    return False


# -- lexer and statement splitter -------------------------------------------------------------------


@dataclass(frozen=True)
class Tok:
    kind: str  # op, num, name, punct, bad
    text: str
    line: int


_TOKEN = re.compile(
    r"(?P<op>\+=|-=|\*=|=)"
    r"|(?P<num>[+\-−]?[ \t]*(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)"
    r"|(?P<punct>[\[\],:])"
)


def _lex(code: str, line: int) -> list[Tok]:
    toks: list[Tok] = []
    pos = 0
    while pos < len(code):
        if code[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(code, pos)
        if m is None:
            toks.append(Tok("bad", code[pos], line))
            pos += 1
            continue
        toks.append(Tok(m.lastgroup or "bad", m.group(), line))
        pos = m.end()
    return toks


@dataclass
class RawStatement:
    tokens: list[Tok]
    first_line: int
    last_line: int
    source: list[str]
    last_terminated: bool = True  # its last code line has a newline or a comment after the code
    broken: str | None = None  # a reason found while splitting (unclosed bracket, ...)
    incomplete: bool = False  # trailing statement of a cut-off block

    @property
    def text(self) -> str:
        s = "\n".join(x.rstrip() for x in self.source).strip()
        return s if len(s) <= STATEMENT_TEXT_CHARS else s[: STATEMENT_TEXT_CHARS - 3] + "..."


def _open_reason(depth: int) -> str:
    return "unclosed '['" if depth > 0 else "incomplete statement (nothing after '=', an operator or ',')"


def split_statements(block: Block) -> list[RawStatement]:
    """Group a block's lines into statements; mark the trailing one of a cut-off block incomplete."""
    out: list[RawStatement] = []
    cur: RawStatement | None = None
    depth = 0

    def finish(reason: str | None = None) -> None:
        nonlocal cur, depth
        assert cur is not None
        cur.broken = reason
        out.append(cur)
        cur, depth = None, 0

    for line in block.lines:
        code, hash_, _ = line.text.partition("#")
        toks = _lex(code, line.number)
        if not toks:
            if cur is not None:
                cur.source.append(line.text)
            continue
        if cur is not None and toks[0].kind == "name":
            finish(_open_reason(depth))  # a new statement starts while the previous one is open
        if cur is None:
            cur = RawStatement([], line.number, line.number, [])
        cur.tokens.extend(toks)
        cur.source.append(line.text)
        cur.last_line = line.number
        cur.last_terminated = line.terminated or bool(hash_)
        for t in toks:
            if t.text == "[":
                depth += 1
            elif t.text == "]":
                depth -= 1
        if depth <= 0 and toks[-1].kind != "op" and toks[-1].text != ",":
            finish()
    if cur is not None:
        if block.cut_off:
            cur.incomplete = True
            out.append(cur)
        else:
            finish(_open_reason(depth))
    elif block.cut_off and out and not out[-1].broken and not out[-1].last_terminated:
        last = out[-1]  # it ends on the reply's last line, with no newline after it
        if last.tokens[-1].text != "]":
            last.incomplete = True  # a number or name at the very end of the reply may be truncated
        else:
            try:
                parse_statement(last.tokens)
            except EditError:
                last.incomplete = True
    return out


# -- parser -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Index:
    start: int | None
    stop: int | None = None
    step: int | None = None
    is_slice: bool = False


@dataclass(frozen=True)
class Statement:
    name: str
    indices: tuple[Index, ...]
    op: str
    value: Any  # float or nested list of floats


def _describe(t: Tok | None) -> str:
    return "the end of the statement" if t is None else f"'{t.text}'"


class _Parser:
    def __init__(self, toks: list[Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> Tok | None:
        t = self.peek()
        self.i += 1
        return t

    def at(self, text: str) -> bool:
        t = self.peek()
        return t is not None and t.kind != "bad" and t.text == text

    def statement(self) -> Statement:
        for t in self.toks:
            if t.kind == "bad":
                raise EditError(f"unexpected character '{t.text}'")
        t = self.take()
        if t is None or t.kind != "name":
            raise EditError(f"a statement starts with a tensor name, got {_describe(t)}")
        indices: list[Index] = []
        if self.at("["):
            self.take()
            indices.append(self.index())
            while self.at(","):
                self.take()
                indices.append(self.index())
            if not self.at("]"):
                raise EditError(f"expected ',' or ']' in the index, got {_describe(self.peek())}")
            self.take()
            if self.at("["):
                raise EditError("write one [ ] with comma-separated indices, e.g. name[i, j]")
        op = self.take()
        if op is None or op.kind != "op":
            raise EditError(f"expected '=', '+=', '-=' or '*=' after the target, got {_describe(op)}")
        if self.peek() is None:
            raise EditError("missing value after the operator")
        value = self.value(0)
        if self.peek() is not None:
            raise EditError(f"unexpected {_describe(self.peek())} after the value")
        return Statement(t.text, tuple(indices), op.text, value)

    def _int(self) -> int | None:
        t = self.peek()
        if t is None or t.kind != "num":
            return None
        self.take()
        s = t.text.replace(" ", "").replace("\t", "")
        if s[0] in "-−":
            raise EditError(f"negative index '{t.text}' is not accepted (indices count from 0)")
        if not re.fullmatch(r"\+?\d+", s):
            raise EditError(f"index '{t.text}' is not a whole number")
        return int(s.lstrip("+"))

    def index(self) -> Index:
        a = self._int()
        if self.at(":"):
            self.take()
            b = self._int()
            s = None
            if self.at(":"):
                self.take()
                s = self._int()
            return Index(a, b, s, True)
        if a is None:
            raise EditError(f"expected an index, got {_describe(self.peek())}")
        return Index(a)

    def value(self, depth: int) -> Any:
        t = self.take()
        if t is None:
            raise EditError("missing value")
        if t.kind == "num":
            return _number(t.text)
        if t.text != "[":
            if t.kind == "name":
                raise EditError(f"'{t.text}' is not a plain number")
            raise EditError(f"expected a number or a list, got '{t.text}'")
        if depth >= MAX_LIST_DEPTH:
            raise EditError("list nested too deeply")
        items: list[Any] = []
        while True:
            if self.at("]"):
                self.take()
                break
            items.append(self.value(depth + 1))
            if self.at(","):
                self.take()
                continue
            if self.at("]"):
                self.take()
                break
            raise EditError(f"expected ',' or ']' in the list, got {_describe(self.peek())}")
        if not items:
            raise EditError("empty list")
        return items


def _number(text: str) -> float:
    s = text.replace(" ", "").replace("\t", "").replace("−", "-")
    v = float(s)
    if not math.isfinite(v):
        raise EditError(f"value '{text}' is not finite")
    if abs(v) > MAX_ABS_VALUE:
        raise EditError(f"value {text} exceeds |v| <= 1e6")
    return v


def parse_statement(toks: list[Tok]) -> Statement:
    return _Parser(toks).statement()


# -- applying ---------------------------------------------------------------------------------------


def zero_state() -> "OrderedDict[str, np.ndarray]":
    return OrderedDict((k, np.zeros(s, dtype=np.float32)) for k, s in PARAM_SHAPES.items())


def _list_shape(v: Any) -> tuple[int, ...]:
    if isinstance(v, float):
        return ()
    shapes = {_list_shape(x) for x in v}
    if len(shapes) != 1:
        raise EditError("ragged list (rows of different lengths, or numbers mixed with lists)")
    return (len(v),) + shapes.pop()


def _selection(name: str, indices: tuple[Index, ...]) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    shape = PARAM_SHAPES[name]
    if len(indices) > len(shape):
        raise EditError(f"{name} has {len(shape)} axis/axes {list(shape)}, {len(indices)} indices given")
    key: list[Any] = []
    sel: list[int] = []
    for axis, n in enumerate(shape):
        ix = indices[axis] if axis < len(indices) else Index(None, None, None, True)
        if not ix.is_slice:
            assert ix.start is not None
            if ix.start >= n:
                raise EditError(f"index {ix.start} out of range for axis {axis} of {name} (length {n})")
            key.append(ix.start)
            continue
        a = 0 if ix.start is None else ix.start
        b = n if ix.stop is None else ix.stop
        s = 1 if ix.step is None else ix.step
        if s < 1:
            raise EditError(f"step {s} on axis {axis} of {name}: the step must be >= 1")
        if b > n:
            raise EditError(f"range end {b} exceeds axis {axis} of {name} (length {n})")
        if a >= b:
            raise EditError(f"empty range {a}:{b} on axis {axis} of {name}")
        key.append(slice(a, b, s))
        sel.append(len(range(a, b, s)))
    return tuple(key), tuple(sel)


def _apply(state: "OrderedDict[str, np.ndarray]", st: Statement) -> int:
    """Apply one statement in place; return the number of entries written."""
    if st.name not in PARAM_SHAPES:
        raise EditError(f"unknown tensor '{st.name}'")
    key, sel = _selection(st.name, st.indices)
    if isinstance(st.value, float):
        val: np.ndarray = np.full(sel, st.value, dtype=np.float64)
    else:
        lshape = _list_shape(st.value)
        val = np.array(st.value, dtype=np.float64)
        if lshape != sel:
            if val.size == math.prod(sel) and tuple(d for d in lshape if d != 1) == tuple(d for d in sel if d != 1):
                val = val.reshape(sel)
            else:
                raise EditError(f"list shape {list(lshape)} does not match the selected block {list(sel)} of {st.name}")
    arr = state[st.name]
    old = np.asarray(arr[key], dtype=np.float64)
    if st.op == "=":
        new = val
    elif st.op == "+=":
        new = old + val
    elif st.op == "-=":
        new = old - val
    else:
        new = old * val
    if not bool(np.isfinite(new).all()) or float(np.abs(new).max(initial=0.0)) > MAX_ABS_VALUE:
        raise EditError(f"'{st.op}' would make a value non-finite or larger than 1e6 in magnitude")
    arr[key] = new.astype(np.float32)
    return int(math.prod(sel))


# -- runs -------------------------------------------------------------------------------------------


@dataclass
class ApplyResult:
    state_dict: "OrderedDict[str, torch.Tensor]"
    report: dict[str, Any]


def apply_replies(replies: Sequence[str], names: Sequence[str] | None = None,
                  cut_off: Sequence[bool | None] | None = None) -> ApplyResult:
    """Read every edits block of every reply, in order, into a zero v0.3 state dict.

    ``cut_off[r]``: True if reply r ended at the output limit, False if it ended normally, None if unknown
    (the default for every reply; treated as possibly cut)."""
    if names is not None and len(names) != len(replies):
        raise ValueError("names and replies differ in length")
    if cut_off is not None and len(cut_off) != len(replies):
        raise ValueError("cut_off and replies differ in length")
    state = zero_state()
    errors: list[dict[str, Any]] = []
    cut: list[dict[str, Any]] = []
    touched: set[str] = set()
    n_err = 0
    totals = {"statements": 0, "applied": 0, "entries_written": 0}
    reply_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for r, text in enumerate(replies, start=1):
        known = cut_off[r - 1] if cut_off is not None else None
        outside: list[Line] = []
        blocks = extract_blocks(text, r, known, outside)
        like = [ln for ln in outside if _statement_like(ln)]
        rrow = {"reply": r, "file": names[r - 1] if names else None,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "reply_cut_off": known,
                "blocks": 0, "statements": 0, "applied": 0, "cut_off": False,
                "ends_with_end_of_weights": ends_with_end_marker(text),
                "outside_block_statement_lines": len(like),
                "outside_block_examples": [{"line": ln.number, "text": ln.text[:STATEMENT_TEXT_CHARS]}
                                           for ln in like[:OUTSIDE_EXAMPLES]]}
        for blk in blocks:
            bno = len(block_rows) + 1
            brow = {"block": bno, "reply": r, "index_in_reply": blk.index_in_reply, "open_line": blk.open_line,
                    "close_line": blk.close_line, "closed_by": blk.closed_by, "cut_off": blk.cut_off,
                    "statements": 0, "applied": 0, "errors": 0, "cut_off_statements": 0}
            for raw in split_statements(blk):
                brow["statements"] += 1
                where = {"reply": r, "block": bno, "line": raw.first_line, "end_line": raw.last_line,
                         "statement": raw.text}
                if raw.incomplete:
                    brow["cut_off_statements"] += 1
                    cut.append({**where, "reason": "cut off by the end of the reply"})
                    continue
                try:
                    if raw.broken:
                        raise EditError(raw.broken)
                    st = parse_statement(raw.tokens)
                    written = _apply(state, st)
                except EditError as exc:
                    brow["errors"] += 1
                    n_err += 1
                    if len(errors) < MAX_ERRORS_LISTED:
                        errors.append({**where, "reason": str(exc)})
                    continue
                brow["applied"] += 1
                totals["entries_written"] += written
                touched.add(st.name)
            block_rows.append(brow)
            rrow["blocks"] += 1
            rrow["statements"] += brow["statements"]
            rrow["applied"] += brow["applied"]
            rrow["cut_off"] = rrow["cut_off"] or blk.cut_off
        totals["statements"] += rrow["statements"]
        totals["applied"] += rrow["applied"]
        reply_rows.append(rrow)

    marked = [row["reply"] for row in reply_rows if row["ends_with_end_of_weights"]]
    nonzero = {k: int(np.count_nonzero(a)) for k, a in state.items()}
    max_abs = {k: float(np.abs(a).max(initial=0.0)) for k, a in state.items()}
    report: dict[str, Any] = {
        "reader": POLICY,
        "edit_report": {
            "block": bool(block_rows),
            "blocks": len(block_rows),
            "statements": totals["statements"],
            "applied": totals["applied"],
            "skipped": n_err,
            "entries_written": totals["entries_written"],
            "errors": errors,
            "errors_listed": len(errors),
            "cut_off": any(b["cut_off"] for b in block_rows),
            "cut_off_statements": cut,
            "tensors_touched": [k for k in PARAM_SHAPES if k in touched],
            "outside_block_statement_lines": sum(r["outside_block_statement_lines"] for r in reply_rows),
        },
        "replies": reply_rows,
        "n_replies": len(reply_rows),
        "end_of_weights": bool(reply_rows) and reply_rows[-1]["ends_with_end_of_weights"],
        "end_of_weights_reply": marked[0] if marked else None,
        "blocks": block_rows,
        "nonzero": nonzero,
        "nonzero_total": sum(nonzero.values()),
        "parameters": PARAM_COUNT,
        "max_abs": max_abs,
    }
    sd = OrderedDict((k, torch.from_numpy(a.copy())) for k, a in state.items())
    return ApplyResult(sd, report)


# -- run directories (scripts/v05_apply_reply.py) ---------------------------------------------------


def reply_files(run_dir: str | Path) -> list[Path]:
    """``RUN_DIR/replies/NNN.md`` in numeric order; numbering must be contiguous from 0 or 1."""
    d = Path(run_dir) / REPLIES_DIR
    if not d.is_dir():
        raise RunDirError(f"{d}: no replies directory")
    found: dict[int, Path] = {}
    for p in d.iterdir():
        m = re.fullmatch(r"(\d+)\.md", p.name)
        if not m or not p.is_file():
            continue
        n = int(m.group(1))
        if n in found:
            raise RunDirError(f"{d}: two reply files numbered {n} ({found[n].name}, {p.name})")
        found[n] = p
    if not found:
        raise RunDirError(f"{d}: no NNN.md reply files")
    nums = sorted(found)
    if nums[0] not in (0, 1) or nums != list(range(nums[0], nums[0] + len(nums))):
        raise RunDirError(f"{d}: reply numbering is not contiguous from 0 or 1: {nums}")
    return [found[n] for n in nums]


def reading_units(run_dir: str | Path) -> tuple[list[Path], list[bool | None], dict[str, Any]]:
    """The replies of a run directory in reading order, with whether each was cut off, and how they were found.

    With ``replies/INDEX.json`` (written by the v0.5 launcher): each turn's parts when it has several (one API
    message each), else its ``NNN.md``; the index must list exactly the ``NNN.md`` files on disk, their sha256s
    must match, and a turn's parts joined by a newline must be its ``NNN.md``. Without an index: every
    ``NNN.md``, cut-off unknown."""
    run_dir = Path(run_dir)
    files = reply_files(run_dir)
    d = run_dir / REPLIES_DIR
    index = d / INDEX_FILE
    if not index.is_file():
        return files, [None] * len(files), {"mode": "reply_files", "index": None}
    try:
        idx = json.loads(index.read_text(encoding="utf-8"))
        units = sorted(idx["units"], key=lambda u: int(u["turn"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise RunDirError(f"{index}: unreadable ({exc})") from exc
    if [u.get("file") for u in units] != [p.name for p in files]:
        raise RunDirError(f"{index} lists {[u.get('file') for u in units]}, "
                          f"the directory has {[p.name for p in files]}")
    out_files: list[Path] = []
    cuts: list[bool | None] = []
    rows = []
    for u, f in zip(units, files):
        text = f.read_bytes().decode("utf-8")
        if u.get("sha256") not in (None, hashlib.sha256(text.encode("utf-8")).hexdigest()):
            raise RunDirError(f"{f}: sha256 differs from {index}")
        parts = u.get("parts") or []
        unit_cut = u.get("cut_off")
        if not parts:
            out_files.append(f)
            cuts.append(unit_cut if isinstance(unit_cut, bool) else None)
            rows.append({"turn": u["turn"], "files": [f.name]})
            continue
        texts, names = [], []
        for k, part in enumerate(parts):
            name = part["file"] if isinstance(part, dict) else str(part)
            pf = d / name
            if not re.fullmatch(r"\d+\.p\d+\.md", name) or not pf.is_file():
                raise RunDirError(f"{index}: part {name!r} of turn {u['turn']} missing")
            texts.append(pf.read_bytes().decode("utf-8"))
            names.append(name)
            if isinstance(part, dict):
                c = part.get("cut_off")
            else:  # an older index: every part but the last was cut (a recovery followed it)
                c = True if k < len(parts) - 1 else unit_cut
            out_files.append(pf)
            cuts.append(c if isinstance(c, bool) else None)
        if "\n".join(t for t in texts if t != "") != text:
            raise RunDirError(f"{f}: is not its parts {names} joined by a newline")
        rows.append({"turn": u["turn"], "files": names})
    info = {"mode": "index", "index": INDEX_FILE,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "units": rows}
    return out_files, cuts, info


def _canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_run(run_dir: str | Path) -> tuple[bytes, str, dict[str, Any]]:
    """Read a run directory; return (DLW3 artefact bytes, its sha256, the apply report).

    Deterministic: the same reply files give the same bytes (no timestamps, no paths in the
    artefact metadata).
    """
    from ..v03.artifact import serialize_state_dict

    run_dir = Path(run_dir)
    files, cuts, reading = reading_units(run_dir)
    texts = [p.read_bytes().decode("utf-8") for p in files]
    res = apply_replies(texts, [p.name for p in files], cuts)
    replies_sha = _canonical_sha256([[r["file"], r["sha256"]] for r in res.report["replies"]])
    metadata = {
        "experiment": "v0.5",
        "builder": "edits reader",
        "reader_version": READER_VERSION,
        "run": run_dir.resolve().name,
        "n_replies": str(len(files)),
        "reading": reading["mode"],
        "replies_sha256": replies_sha,
        "end_of_weights": "true" if res.report["end_of_weights"] else "false",
    }
    data = serialize_state_dict(res.state_dict, metadata)
    sha = hashlib.sha256(data).hexdigest()
    report = {"run": metadata["run"], "replies_sha256": replies_sha, "reading": reading, **res.report,
              "artifact": {"file": ARTIFACT_FILE, "sha256": sha, "metadata": metadata}}
    return data, sha, report


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=1, ensure_ascii=False) + "\n"
