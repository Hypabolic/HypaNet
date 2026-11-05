"""v0.3 log renderer (spec v0.3 §4). Frozen; ships in the authoring packet.

Standalone: standard library only, plus the sibling packet module ``tokenizer`` (imported lazily so
the harness can also import this file as ``agent_loop_lab.v03.renderer``).

    blocks = render_session(rawops, session_id)   # rawops: list of RawOp dicts (RawOp.to_dict())

One block per event: a header line, then at most 6 body lines (hunk lines, error line, workspace line).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

try:  # packet layout: tokenizer.py next to this file
    import tokenizer as _tok  # type: ignore
except ImportError:  # harness layout
    from . import tokenizer as _tok  # type: ignore

HEADER_MAX_TOKENS = 64
BODY_MAX_LINES = 6
LINE_MAX_TOKENS = 20  # excluding the line-marker token
HUNK_MAX_LINES = 5
TRUNC = "TRUNC"

FIELD_BUDGETS = {"tests": 24, "body": 16, "paths": 16, "query": 12}

VALIDATE_FAMILIES = ("run_tests", "build", "lint")
DIGEST_DROPS = ("digest_before", "digest_after")


def _get(op: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(op, Mapping):
        return op.get(key, default)
    return getattr(op, key, default)


def _ascii(text: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(text))


def _ntok(text: str) -> int:
    return len(_tok.tokenize_block(text))


def _fit(text: str, budget: int) -> str:
    """ASCII-sanitise ``text`` and cut it to at most ``budget`` tokens, ending with TRUNC when cut.

    The cut keeps the source text up to the end of the last kept token, so the kept part reads
    exactly as it would uncut; a space before TRUNC keeps it a separate word."""
    text = _ascii(text)
    spans = _tok.tokenize_spans(text)
    if len(spans) <= budget:
        return text
    keep = max(0, budget - 1)
    kept = text[: spans[keep - 1][3]] if keep else ""
    return (kept + " " + TRUNC) if kept else TRUNC


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def render_rng(op: Mapping[str, Any] | Any) -> random.Random:
    d = op if isinstance(op, Mapping) else op.to_dict()
    seed = int(hashlib.sha256((canonical_json(d) + ":render").encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def _clock(ts: str | None) -> str:
    if not ts:
        return "?"
    t = datetime.fromisoformat(ts)
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc)
    return f"{t:%H:%M:%S}.{t.microsecond // 1000:03d}"


def _q(v: Any) -> str:
    return "?" if v is None else str(v)


def hunk_id(session_id: str, path: str, lines: Sequence[str]) -> str:
    if not lines:
        return "h0"
    text = "\n".join(lines)
    return "h" + hashlib.sha256(f"{session_id}:hunk:{path}:{text}".encode("utf-8")).hexdigest()[:10]


def diff_hunks(before: str | None, after: str | None) -> list[tuple[list[str], list[str]]]:
    """(removed lines, added lines) per hunk of the n=0 unified diff."""
    a = [] if before is None else before.split("\n")
    b = [] if after is None else after.split("\n")
    lines = list(difflib.unified_diff(a, b, "a", "b", lineterm="", n=0))[2:]
    hunks: list[tuple[list[str], list[str]]] = []
    for ln in lines:
        if ln.startswith("@@"):
            hunks.append(([], []))
        elif hunks and ln.startswith("-"):
            hunks[-1][0].append(ln[1:])
        elif hunks and ln.startswith("+"):
            hunks[-1][1].append(ln[1:])
    return hunks


def _status(op: Any) -> str:
    family, outcome = _get(op, "family"), _get(op, "outcome")
    dropped = set(_get(op, "dropped") or ())
    val = _get(op, "validation")
    if family == "poll_status":
        status = "ready" if outcome == "SUCCESS" else ("pending" if outcome == "NEUTRAL" else "error")
    elif family in VALIDATE_FAMILIES:
        status = "FAILED" if outcome == "ERROR" else "ok"
    else:
        status = "error" if outcome == "ERROR" else "ok"
    if val is not None:
        p = "?" if "validation_passed" in dropped else _q(val.get("passed"))
        t = "?" if "validation_total" in dropped else _q(val.get("total"))
        status += f" {p}/{t} passed"
    tin = "?" if "input_tokens" in dropped else _q(_get(op, "input_tokens"))
    tout = "?" if "output_tokens" in dropped else _q(_get(op, "output_tokens"))
    return f"-> {status} in={tin} out={tout}"


def _fields(op: Any, rng: random.Random) -> str:
    family = _get(op, "family")
    target = _get(op, "target")
    args = _get(op, "args") or {}
    dropped = set(_get(op, "dropped") or ())
    parts: list[str] = []
    if family in ("read_file", "list_dir", "edit_file"):
        parts.append(f"path={target}")
        if family == "read_file" and (args.get("line_start") is not None or args.get("line_end") is not None):
            parts.append(f"lines={_q(args.get('line_start'))}-{_q(args.get('line_end'))}")
    elif family == "write_file":
        parts.append(f"path={target}")
    elif family == "search":
        parts.append(f"scope={target}")
        parts.append('query="' + _fit(str(args.get("query", "")), FIELD_BUDGETS["query"]) + '"')
        if args.get("limit") is not None:
            parts.append(f"limit={args['limit']}")
    elif family == "revert":
        paths = target if isinstance(target, (list, tuple)) else [target]
        parts.append("paths=" + _fit(",".join(str(p) for p in paths), FIELD_BUDGETS["paths"]))
    elif family == "run_tests":
        suite = "?" if "validation_suite" in dropped else target
        parts.append(f"suite={suite}")
        if args.get("tests"):
            parts.append("tests=" + _fit(",".join(str(t) for t in args["tests"]), FIELD_BUDGETS["tests"]))
    elif family in ("build", "lint"):
        suite_dropped = "validation_suite" in dropped and _get(op, "validation") is not None
        parts.append(f"target={'?' if suite_dropped else target}")
    elif family == "http_call":
        t = target or {}
        parts.append(f"{t.get('method', '?')} https://{t.get('host', '?')}{t.get('path_template', '')}")
        if args.get("body") is not None:
            parts.append("body=" + _fit(str(args["body"]), FIELD_BUDGETS["body"]))
        parts.append(f"req={rng.getrandbits(32):08x}")
    elif family == "poll_status":
        parts.append(f"resource={target}")
        if args.get("backoff") is not None:
            parts.append(f"backoff={args['backoff']}")
    elif family in ("plan", "ask", "yield"):
        parts.append(f"action={args.get('action_type', '?')}")
    else:
        parts.append(f"target={target}")
    return " ".join(parts)


def _header(op: Any, rng: random.Random) -> str:
    head = f"{_clock(_get(op, 'timestamp'))} {_get(op, 'family')}"
    suffix = _status(op)
    fields = _ascii(_fields(op, rng))
    room = HEADER_MAX_TOKENS - _ntok(head) - _ntok(suffix)
    if _ntok(fields) > room:
        fields = _fit(fields, room)
    return _ascii(f"{head} {fields} {suffix}")


def _line(marker: str, text: str) -> str:
    return "  " + marker + " " + _fit(_ascii(text), LINE_MAX_TOKENS)


def render_event(op: Any, session_id: str) -> str:
    rng = render_rng(op)
    header = _header(op, rng)
    hunk_lines: list[str] = []
    for ch in sorted(_get(op, "changed") or [], key=lambda c: c["path"]):
        # One line per changed file (spec v0.3 r2 §4.3): all removed lines vs all added lines of the
        # n=0 diff, so the number of hunks inside a file is not shown.
        hunks = diff_hunks(ch.get("before"), ch.get("after"))
        if not hunks:
            continue
        removed = [ln for r, _ in hunks for ln in r]
        added = [ln for _, a in hunks for ln in a]
        path = ch["path"]
        hunk_lines.append(f"{path} {hunk_id(session_id, path, removed)} > {hunk_id(session_id, path, added)}")
    tail: list[str] = []
    if _get(op, "outcome") == "ERROR":
        dropped = set(_get(op, "dropped") or ())
        loc = _get(op, "error_location")
        if "failure" in dropped:
            cls, loc_s, line = "?", "?", "?"
        else:
            cls = _q(_get(op, "error_class"))
            loc_s = _q(loc)
            line = "?" if loc is None else str(int(hashlib.sha256(f"{session_id}:{loc}".encode()).hexdigest()[:4], 16) % 900 + 1)
        noise = f"pid {rng.randint(1000, 99999)}" if rng.random() < 0.5 else f"0x{rng.getrandbits(32):08x}"
        tail.append(("!", f"{cls} at {loc_s}:{line} ({noise})"))
    if set(_get(op, "dropped") or ()) & set(DIGEST_DROPS):
        tail.append(("[", "workspace untracked]"))
    max_hunks = min(HUNK_MAX_LINES, BODY_MAX_LINES - len(tail))
    if len(hunk_lines) > max_hunks:
        omitted = len(hunk_lines) - (max_hunks - 1)
        hunk_lines = hunk_lines[: max_hunks - 1] + [f"{TRUNC} {omitted}"]
    body = [_line("@", h) for h in hunk_lines] + [_line(m, t) for m, t in tail]
    return "\n".join([header] + body)


def render_session(rawops: Sequence[Any], session_id: str) -> list[str]:
    return [render_event(op, session_id) for op in rawops]
