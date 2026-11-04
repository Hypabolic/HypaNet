"""v0.3 tokeniser (spec v0.3 §5). Frozen; ships in the authoring packet.

Standalone: standard library only, no relative imports.

    enc = encode_session(blocks)   # blocks: one rendered string per event (renderer.render_session)
    enc.tokens, enc.event_ord, enc.offset, enc.eoe_index, enc.surface

A session is ``BOS`` followed, for each event block, by the block's tokens and then ``EOE``.
``event_ord`` is 0 for ``BOS`` and k+1 for every token of block k, including its ``EOE``.
``offset`` is the 0-based position within the block (``BOS`` has offset 0); an offset >= ``O`` raises.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

V = 10240
O = 192

PAD, BOS, EOE, NL = 0, 1, 2, 3
NL_AT, NL_BANG, NL_BRACKET = 4, 5, 6
LINE_MARKERS = {"@": NL_AT, "!": NL_BANG, "[": NL_BRACKET}

TOOL_NAMES = (
    "read_file", "search", "list_dir", "edit_file", "write_file", "revert", "run_tests", "build", "lint",
    "http_call", "poll_status", "plan", "ask", "yield",
)
FIELD_AND_STATUS_WORDS = (
    "path", "lines", "scope", "query", "limit", "bytes", "paths", "suite", "tests", "target", "body", "req",
    "resource", "backoff", "action", "in", "out", "ok", "error", "FAILED", "passed", "ready", "pending", "at",
    "pid", "workspace", "untracked", "TRUNC", "https", "GET", "POST", "PUT", "PATCH", "DELETE",
)
RESERVED_WORDS: tuple[str, ...] = TOOL_NAMES + FIELD_AND_STATUS_WORDS
RESERVED_BASE = 7
assert RESERVED_BASE + len(RESERVED_WORDS) <= 128
RESERVED_IDS: dict[str, int] = {w: RESERVED_BASE + i for i, w in enumerate(RESERVED_WORDS)}

PUNCT_BASE = 128
PUNCT_CHARS = "".join(chr(c) for c in range(33, 127) if not chr(c).isalnum() and chr(c) != "_")
assert len(PUNCT_CHARS) <= 96
PUNCT_IDS: dict[str, int] = {ch: PUNCT_BASE + i for i, ch in enumerate(PUNCT_CHARS)}

DIGIT_BASE = 224
DIGIT_GROUPS: tuple[str, ...] = (
    tuple(f"{i}" for i in range(10)) + tuple(f"{i:02d}" for i in range(100)) + tuple(f"{i:03d}" for i in range(1000))
)
DIGIT_IDS: dict[str, int] = {g: DIGIT_BASE + i for i, g in enumerate(DIGIT_GROUPS)}
assert DIGIT_BASE + len(DIGIT_GROUPS) == 1334

HASH_BASE = 2048
HASH_BUCKETS = 8192
assert HASH_BASE + HASH_BUCKETS == V

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]*")
_NUM = re.compile(r"[0-9]+")


class TokenizerError(ValueError):
    pass


def word_id(word: str) -> int:
    """Id of a WORD token: reserved id if reserved, otherwise its hashed bucket."""
    if word in RESERVED_IDS:
        return RESERVED_IDS[word]
    return HASH_BASE + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % HASH_BUCKETS


def digit_groups(num: str) -> list[str]:
    """Split a digit string left to right into groups of 3 (the last may be shorter)."""
    return [num[i : i + 3] for i in range(0, len(num), 3)]


def tokenize_spans(block: str) -> list[tuple[int, str, int, int]]:
    """(id, surface, start, end) per token of one event block (without EOE); ``block[start:end]`` is the
    matched source text (for a line marker, the newline, any spaces and the marker character)."""
    out: list[tuple[int, str, int, int]] = []
    i, n = 0, len(block)
    while i < n:
        ch = block[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "\n":
            j = i + 1
            while j < n and block[j] in " \t":
                j += 1
            if j < n and block[j] in LINE_MARKERS:
                out.append((LINE_MARKERS[block[j]], "\n" + block[j], i, j + 1))
                i = j + 1
            else:
                out.append((NL, "\n", i, i + 1))
                i += 1
            continue
        m = _WORD.match(block, i)
        if m:
            w = m.group(0)
            out.append((word_id(w), w, i, m.end()))
            i = m.end()
            continue
        m = _NUM.match(block, i)
        if m:
            for k, g in enumerate(digit_groups(m.group(0))):
                out.append((DIGIT_IDS[g], g, i + 3 * k, i + 3 * k + len(g)))
            i = m.end()
            continue
        if ch in PUNCT_IDS:
            out.append((PUNCT_IDS[ch], ch, i, i + 1))
            i += 1
            continue
        raise TokenizerError(f"unrenderable character {ch!r} at {i}")
    return out


def tokenize_block(block: str) -> list[tuple[int, str]]:
    """(id, surface) pairs for one event block (without EOE)."""
    return [(tid, s) for tid, s, _, _ in tokenize_spans(block)]


@dataclass(frozen=True)
class Encoded:
    tokens: list[int]
    event_ord: list[int]
    offset: list[int]
    eoe_index: list[int]
    surface: list[str]


def encode_session(blocks: list[str]) -> Encoded:
    tokens, event_ord, offset, eoe, surface = [BOS], [0], [0], [], ["<BOS>"]
    for k, block in enumerate(blocks):
        pairs = tokenize_block(block)
        pairs.append((EOE, "<EOE>"))
        if len(pairs) > O:
            raise TokenizerError(f"event block {k} has {len(pairs)} tokens (offset >= O={O})")
        for off, (tid, s) in enumerate(pairs):
            tokens.append(tid)
            event_ord.append(k + 1)
            offset.append(off)
            surface.append(s)
        eoe.append(len(tokens) - 1)
    return Encoded(tokens, event_ord, offset, eoe, surface)


def prefix(enc: Encoded, t: int) -> Encoded:
    """The encoding of blocks 0..t (inclusive): a prefix ending at EOE_t."""
    end = enc.eoe_index[t] + 1
    return Encoded(enc.tokens[:end], enc.event_ord[:end], enc.offset[:end], enc.eoe_index[: t + 1], enc.surface[:end])
