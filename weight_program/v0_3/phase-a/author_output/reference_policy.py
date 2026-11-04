"""Reference policy for the v0.3 doom-loop detector (Phase A, hand-written, zero labels).

Per-block features (tool at header offset 7, last status word, hunk lines, passed count, intent
token) and relations over the previous seven blocks (undo of an earlier hunk id, same-intent
repeats, nearest validation / hunk block) feed a first-match-wins window rule: OSCILLATING,
STALLED, RECOVERY, else PRODUCTIVE.  Every quantity is defined in CIRCUITS.md section 1.
"""

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
RETRY_LIMIT = 4  # a failing block with >= 4 earlier same-intent failures in the window is out of policy
POLL_LIMIT = 5  # polls: >= 5 earlier pending/error polls of the same resource


def _bucket(word):
    """The tokeniser's hashed id of a non-reserved word, else None (reserved words, digits,
    punctuation and the marker tokens are never identifiers)."""
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
    s_fail = status in ("error", "pending")
    s_failed = status == "FAILED"
    has_hunk = "\n@" in tb
    jp = -1
    for i, s in enumerate(tb):
        if s == "passed":
            jp = i
    has_passed = jp >= 0
    pv = 0.0
    if jp >= 3 and tb[jp - 3].isdigit():
        pv = int(tb[jp - 3]) / 100.0
    if tool == "http_call":
        slot = 13
    elif tool == "search":
        slot = 14
    else:
        slot = 10
    intent = _bucket(tb[slot]) if slot < n else None
    cls = _tool_class(tool)
    holds = set()
    hnews = set()
    for p, s in enumerate(tb):
        if s != "\n@":
            continue
        if p + 2 < n and tb[p + 2] != "h0":
            h = _bucket(tb[p + 2])
            if h is not None:
                holds.add(h)
        if p + 4 < n:
            h = _bucket(tb[p + 4])
            if h is not None:
                hnews.add(h)
    return {
        "tool": tool,
        "cls": cls,
        "insp": tool in _INSP,
        "val": tool in _VAL,
        "mut": tool in _MUT,
        "revert": tool == "revert",
        "poll": tool == "poll_status",
        "s_ok": s_ok,
        "s_fail": s_fail,
        "s_failed": s_failed,
        "has_hunk": has_hunk,
        "has_passed": has_passed,
        "pv": pv,
        "intent": intent,
        "holds": holds,
        "hnews": hnews,
    }


def _relational(feats, b):
    """Per-block effect flags of block b given the features of all blocks (uses b-7..b-1)."""
    f = feats[b]
    lo = max(0, b - (WINDOW - 1))
    earlier = feats[lo:b]
    rev = any(h in e["holds"] for e in earlier for h in f["hnews"])
    same = [
        e for e in earlier
        if f["intent"] is not None and f["cls"] is not None and e["intent"] == f["intent"] and e["cls"] == f["cls"]
    ]
    c_all = len(same)
    c_fail = sum(1 for e in same if e["s_fail"])
    nearest = 0
    for c in range(1, WINDOW):
        if b - c < 0:
            break
        e = feats[b - c]
        if e["val"] or e["has_hunk"]:
            nearest = int(e["val"]) - int(e["has_hunk"])
            break
    redval = f["val"] and nearest == 1
    u1 = f["s_fail"] and (not f["poll"]) and c_fail >= RETRY_LIMIT
    u2 = f["s_fail"] and f["poll"] and c_fail >= POLL_LIMIT
    redretry = u1 or u2
    inpol = f["s_fail"] and not redretry
    resolv = f["s_ok"] and c_fail >= 1
    noop = f["mut"] and not f["has_hunk"] and not f["s_fail"]
    redinsp = f["insp"] and (c_all - c_fail) >= 1
    si_revert = f["revert"] and not rev
    si = rev or si_revert or noop or redinsp or redval or redretry
    return {"rev": rev, "si": si, "inpol": inpol, "resolv": resolv, "kill": redretry}


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
    # Only the last 2*WINDOW-1 blocks can influence the answer (window + its look-back).
    blocks = blocks[max(0, len(blocks) - (2 * WINDOW - 1)):]
    t = len(blocks) - 1
    lo = max(0, t - (WINDOW - 1))
    feats = [_block_features(tb) for tb in blocks]
    rel = {b: _relational(feats, b) for b in range(lo, t + 1)}

    n_rev = sum(1 for b in range(lo, t + 1) if rel[b]["rev"])
    n_si = sum(1 for b in range(lo, t + 1) if rel[b]["si"])
    # RECOVERY evidence: the most recent block among the last K_REC that is either an in-policy
    # failure / resolution (recovery) or an out-of-policy retry (which ends the in-policy episode).
    any_rec = False
    for b in range(t, max(lo, t - (K_REC - 1)) - 1, -1):
        if rel[b]["inpol"] or rel[b]["resolv"]:
            any_rec = True
            break
        if rel[b]["kill"]:
            break

    vals = [b for b in range(lo, t + 1) if feats[b]["has_passed"]]
    prog = False
    if vals:
        first, last = feats[vals[0]], feats[vals[-1]]
        prog = (last["pv"] > first["pv"] + 0.005) or (first["s_failed"] and last["s_ok"])

    if n_rev >= N_REV and not prog:
        return OSCILLATING
    if n_si >= N_STALL and rel[t]["si"] and not prog and not rel[t]["inpol"]:
        return STALLED
    if any_rec:
        return RECOVERY
    return PRODUCTIVE
