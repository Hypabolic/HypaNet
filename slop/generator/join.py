"""Running the real extractor over generated files, and joining planted labels to its nodes by source span.

A slot's span is where the renderer wrote its visible characters; a node's ``loc`` is where the extractor found
them. The join is exact: every labelled slot must be found as a node (0 unmatched spans, SPEC.md §12.4), every node
must have come from a slot, and the extracted roles must agree with each planted pattern and with the precedence
of every pointer (LABELLING.md §5). Any failure stops the build.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .core import GenError, Slot

SLOP = Path(__file__).resolve().parents[1]
BATCH = SLOP / "generator" / "extract_batch.ts"

# LABELLING.md §2: context only, never labelled
CONTEXT_ROLES = {"aria", "alt", "code"}

# LABELLING.md §5 precedence, 1 = kept first. A label in a form or control counts as rank 1.
_RANK = {r: 1.0 for r in ("button", "link", "nav", "tab", "menu_item", "breadcrumb", "h1", "dialog_title", "empty_title")}
_RANK.update({f"h{n}": 2.0 + (n - 2) / 10 for n in range(2, 7)})
_RANK.update({r: 3.0 for r in ("body", "list_item", "dialog_body", "empty_body", "toast", "error", "stat", "cell",
                                "input_value", "other")})
_RANK.update({r: 4.0 for r in ("subtitle", "helper", "caption")})
_RANK.update({r: 5.0 for r in ("eyebrow", "badge", "tooltip", "placeholder")})


def rank(role: str, flags) -> float:
    if role == "label":
        return 1.0 if {"in_form", "in_control"} & set(flags) else 3.0
    return _RANK.get(role, 3.0)


# Extracted roles each planted pattern may carry (the pattern describes the node's role)
PATTERN_ROLES = {
    "R1": {"eyebrow"},
    "R2": {"eyebrow", "caption", "body", "label", "other", "badge"},
    "R3": {"subtitle"},
    "R4": {"body", "empty_body", "dialog_body", "list_item", "caption", "error"},
    "R5": {"helper", "tooltip", "subtitle", "body", "caption", "label"},
    "R6": {"subtitle", "body", "caption", "helper", "empty_body", "dialog_body"},
    "R8": {"placeholder"},
    "R9": {"h1", "h2", "h3", "h4", "dialog_title", "empty_title"},
    "R11": {"badge", "body", "caption", "label", "other", "eyebrow"},
    "R14": {"caption", "body", "button", "link", "subtitle", "helper", "stat", "label", "list_item", "h3"},
    "R16": {"subtitle", "body", "caption", "helper"},
    "R17": {"body", "subtitle", "caption", "list_item"},
}


def run_extractor(requests: list[dict]) -> list[dict]:
    """Extract every request ({repo, path, src, css}) with the real extractor; one screen per request, in order."""
    if not requests:
        return []
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
    res = subprocess.run(["node", str(BATCH)], input=payload.encode("utf-8"), capture_output=True, cwd=SLOP)
    if res.returncode != 0:
        raise GenError(f"extractor batch failed: {res.stderr.decode('utf-8', 'replace')[-2000:]}")
    lines = res.stdout.decode("utf-8").splitlines()
    if len(lines) != len(requests):
        raise GenError(f"extractor returned {len(lines)} screens for {len(requests)} files")
    out = []
    for req, line in zip(requests, lines):
        obj = json.loads(line)
        if "error" in obj:
            raise GenError(f"extractor failed on {req['path']}: {obj['error'][:500]}")
        out.append({"line": line, "screen": obj})
    return out


def loc_of(span) -> list:
    return [int(x) for x in span]


def join_screen(screen: dict, slots: list[Slot]) -> dict:
    """Match slots to nodes by span. Returns {"pairs": [(node, slot)], "problems": [...]} with every problem kind
    counted; the caller decides what is fatal."""
    nodes = {tuple(n["loc"]): n for n in screen["nodes"]}
    by_span = {}
    problems = []
    for s in slots:
        if s.span is None:
            problems.append(("unwritten", s.text))
            continue
        if s.span in by_span:
            problems.append(("duplicate_span", s.text))
        by_span[s.span] = s
    pairs = []
    for loc, n in nodes.items():
        s = by_span.get(loc)
        if s is None:
            problems.append(("node_without_slot", f"{n['role']} {n['text']!r} {list(loc)}"))
        else:
            pairs.append((n, s))
    for span, s in by_span.items():
        if span not in nodes:
            kind = "unmatched_planted" if s.planted or s.pattern else "unmatched_keep"
            problems.append((kind, f"{s.role} {s.text!r} {list(span)}"))
    return {"pairs": pairs, "problems": problems}


def check_labels(pairs, screen) -> list:
    """Role and pointer checks on joined pairs (fatal problems)."""
    problems = []
    node_of = {id(s): n for n, s in pairs}
    for n, s in pairs:
        role = n["role"]
        is_ctx = role in CONTEXT_ROLES or "sr_only" in n["flags"]
        if is_ctx and (s.planted or s.pattern):
            problems.append(("planted_on_context", f"{role} {s.text!r}"))
            continue
        if s.planted:
            allowed = PATTERN_ROLES.get(s.pattern)
            if allowed and role not in allowed:
                problems.append(("pattern_role", f"{s.pattern} on {role} {s.text!r}"))
            for ref in ([s.target] if s.target is not None else []) + list(s.equiv):
                if id(ref) not in node_of:
                    problems.append(("pointer_missing", f"{s.pattern} {s.text!r} -> {ref.text!r}"))
            if s.cls == "ECHO" and s.target is not None and id(s.target) in node_of:
                t = node_of[id(s.target)]
                rn, rt = rank(role, n["flags"]), rank(t["role"], t["flags"])
                if rn < rt or (rn == rt and n["i"] < t["i"]):
                    problems.append(("precedence", f"{s.pattern} {role} {s.text!r} -> {t['role']} {t['text']!r}"))
                if t["role"] in CONTEXT_ROLES or "sr_only" in t["flags"]:
                    problems.append(("target_context", f"{s.text!r}"))
    return problems


def node_record(n: dict, s: Slot, screen_id: str, meta: dict) -> dict | None:
    """The SPEC.md §5 label record for one node, or None for a context-only node."""
    if n["role"] in CONTEXT_ROLES or "sr_only" in n["flags"]:
        return None
    if s.cls is None:
        return {"screen": screen_id, "loc": n["loc"], "class": None, "target": None, "equiv": [], "pattern": None,
                "aside": ["UNRESOLVED"], "partial": False, "ct": False, "conf": "clear", "role_ok": True, **meta}
    return {
        "screen": screen_id, "loc": n["loc"], "class": s.cls,
        "target": loc_of(s.target.span) if s.target is not None else None,
        "equiv": [loc_of(x.span) for x in s.equiv], "pattern": s.pattern, "aside": list(s.aside),
        "partial": s.partial, "ct": s.ct, "conf": s.conf, "role_ok": True, **meta,
    }

