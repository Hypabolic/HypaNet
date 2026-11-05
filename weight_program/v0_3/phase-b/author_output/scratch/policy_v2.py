"""Candidate revised policy (Phase B).  Same interface as reference_policy; exposes the per-block
feature tables for analysis.  Every quantity is defined so the network can compute it."""

import hashlib
import re

PRODUCTIVE, RECOVERY, STALLED, OSCILLATING = 0, 1, 2, 3

_TOOLS = (
    "read_file", "search", "list_dir", "edit_file", "write_file", "revert", "run_tests", "build", "lint",
    "http_call", "poll_status", "plan", "ask", "yield",
)
_FIELD_WORDS = (
    "path", "lines", "scope", "query", "limit", "bytes", "paths", "suite", "tests", "target", "body", "req",
    "resource", "backoff", "action", "in", "out", "ok", "error", "FAILED", "passed", "ready", "pending", "at",
    "pid", "workspace", "untracked", "TRUNC", "https", "GET", "POST", "PUT", "PATCH", "DELETE",
)
_RESERVED = frozenset(_TOOLS + _FIELD_WORDS)
_WORD_PATTERN = r"[A-Za-z_][A-Za-z0-9_.\-/]*\Z"

_INSP = frozenset(("read_file", "search", "list_dir"))
_VAL = frozenset(("run_tests", "build", "lint"))
_MUT = frozenset(("edit_file", "write_file", "revert"))
_NET = frozenset(("http_call", "poll_status"))
_STATUS = frozenset(("ok", "error", "FAILED", "ready", "pending"))

WINDOW = 8
N_STALL = 5
N_REV = 2
K_REC = 4
RETRY_LIMIT = 4  # a failing block with >= 4 earlier failures of the same episode key is out of policy
POLL_LIMIT = 6  # polls: >= 6 earlier pending polls of the same resource


def _bucket(word):
    if word in _RESERVED or re.match(_WORD_PATTERN, word) is None:
        return None
    return 2048 + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % 8192


def _tool_class(tool):
    if tool in _INSP:
        return "insp"
    if tool in _MUT:
        return "mut"
    if tool in _VAL:
        return "val"
    if tool in _NET:
        return "net"
    return None


def _block_features(tb):
    n = len(tb)
    tool = tb[7] if n > 7 and tb[7] in _TOOLS else None
    status = None
    for s in tb:
        if s in _STATUS:
            status = s
    s_ok = status in ("ok", "ready")
    s_fail = status in ("error", "pending") or status == "FAILED"
    has_hunk = "\n@" in tb
    jp = -1
    for i, s in enumerate(tb):
        if s == "passed":
            jp = i
    pv = 0
    if jp >= 3 and tb[jp - 3].isdigit():
        pv = int(tb[jp - 3])
    # intent: the first hashed token at offset >= 10 (>= 14 for search: the first query word)
    lo = 14 if tool == "search" else 10
    intent = None
    for i in range(lo, n):
        h = _bucket(tb[i])
        if h is not None:
            intent = h
            break
    # error location: the token 3 after the error-line marker
    loc = None
    for p, s in enumerate(tb):
        if s == "\n!" and p + 3 < n:
            loc = _bucket(tb[p + 3])
    cls = _tool_class(tool)
    holds = set()
    hnews = set()
    for p, s in enumerate(tb):
        if s != ">" or p == 0 or p + 1 >= n:
            continue
        if tb[p - 1] != "h0":
            h = _bucket(tb[p - 1])
            if h is not None:
                holds.add(h)
        h = _bucket(tb[p + 1])
        if h is not None:
            hnews.add(h)
    return {
        "tool": tool, "cls": cls, "insp": tool in _INSP, "val": tool in _VAL, "mut": tool in _MUT,
        "revert": tool == "revert", "poll": tool == "poll_status", "s_ok": s_ok, "s_fail": s_fail,
        "has_hunk": has_hunk, "pv": pv, "intent": intent, "loc": loc, "holds": holds, "hnews": hnews,
        "key": (cls, intent) if (cls is not None and intent is not None) else None,
    }


def _relational(feats, b):
    f = feats[b]
    earlier = feats[:b]
    lo = max(0, b - (WINDOW - 1))
    rev = any(h in e["holds"] for e in feats[lo:b] for h in f["hnews"])
    same = [e for e in earlier if f["key"] is not None and e["key"] == f["key"]]
    c_fail = sum(1 for e in same if e["s_fail"] and e["loc"] == f["loc"])
    prev = same[-1] if same else None
    maxpv_prev = max([e["pv"] for e in same], default=0)
    improved = f["val"] and prev is not None and f["pv"] > maxpv_prev
    resolv = f["s_ok"] and prev is not None and prev["s_fail"]
    limit = POLL_LIMIT if f["poll"] else RETRY_LIMIT
    kill = f["s_fail"] and c_fail >= limit
    inpol = f["s_fail"] and 1 <= c_fail < limit
    redundant_ok = f["val"] and f["s_ok"] and prev is not None and prev["s_ok"] and not improved
    stagnant = f["val"] and ((f["s_fail"] and c_fail >= 1 and not improved) or redundant_ok)
    # nearest earlier validation block (any suite): is it stagnant?
    nearest_val_stagnant = False
    for e in reversed(earlier):
        if e["val"]:
            nearest_val_stagnant = e["_stagnant"]
            break
    futile = f["mut"] and f["has_hunk"] and not f["s_fail"] and nearest_val_stagnant
    noop = f["mut"] and not f["has_hunk"] and not f["s_fail"]
    redinsp = f["insp"] and prev is not None and not resolv
    si_revert = f["revert"] and not rev
    si = rev or si_revert or noop or futile or redinsp or kill or redundant_ok
    f["_stagnant"] = stagnant
    return {"rev": rev, "si": si, "inpol": inpol, "resolv": resolv, "kill": kill, "improved": improved,
            "stagnant": stagnant, "futile": futile, "noop": noop, "redinsp": redinsp, "redok": redundant_ok,
            "fail": f["s_fail"], "c_fail": c_fail}


def analyse(blocks):
    feats = [_block_features(tb) for tb in blocks]
    rel = []
    for b in range(len(blocks)):
        rel.append(_relational(feats, b))
    return feats, rel


def decide(feats, rel, t):
    lo = max(0, t - (WINDOW - 1))
    n_rev = sum(1 for b in range(lo, t + 1) if rel[b]["rev"])
    n_si = sum(1 for b in range(lo, t + 1) if rel[b]["si"])
    prog = any(rel[b]["improved"] or rel[b]["resolv"] for b in range(lo, t + 1))
    # the most recent failing or resolving block anywhere in the prefix
    nr = None
    for b in range(t, -1, -1):
        if rel[b]["fail"] or rel[b]["resolv"]:
            nr = b
            break
    active = nr is not None and rel[nr]["inpol"]
    recent_resolv = any(rel[b]["resolv"] for b in range(max(lo, t - (K_REC - 1)), t + 1))
    if n_rev >= N_REV and not prog:
        return OSCILLATING
    if n_si >= N_STALL and rel[t]["si"] and not prog and not active:
        return STALLED
    if active or recent_resolv:
        return RECOVERY
    return PRODUCTIVE


def reference_policy(tokens, event_ord):
    blocks = []
    cur = None
    cur_ord = None
    for s, e in zip(tokens, event_ord):
        if e == 0:
            continue
        if e != cur_ord:
            cur = []
            blocks.append(cur)
            cur_ord = e
        cur.append(s)
    if not blocks:
        return PRODUCTIVE
    feats, rel = analyse(blocks)
    return decide(feats, rel, len(blocks) - 1)
