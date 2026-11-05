"""Reference policy, Phase B revision of the v0.3 doom-loop detector (hand-written).

Per-block features (tool, last status word, hunk lines, passed count, intent token) and
session-wide relations (same-intent repeats, retries on an unchanged workspace, resolutions,
undo of a hunk id within the window) feed a first-match-wins window rule: OSCILLATING, STALLED,
RECOVERY, else PRODUCTIVE.  CIRCUITS.md section 1 defines every quantity; build_weights.py
computes the same.
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
_MARKERS = frozenset(("<BOS>", "<EOE>", "\n", "\n@", "\n!", "\n["))
_PUNCT = "".join(chr(c) for c in range(33, 127) if not chr(c).isalnum() and chr(c) != "_")

_INSP = frozenset(("read_file", "search", "list_dir", "plan", "ask", "yield"))
_VAL = frozenset(("run_tests", "build", "lint"))
_MUT = frozenset(("edit_file", "write_file", "revert"))
_NET = frozenset(("http_call", "poll_status"))
_STATUS = frozenset(("ok", "error", "FAILED", "ready", "pending"))

WINDOW = 8  # blocks t-7..t
N_STALL = 5  # stall-ineffective blocks for STALLED
N_REV = 2  # reversions for OSCILLATING
K_REC = 4  # a resolution within the last 4 blocks is RECOVERY evidence
KILL_COUNT = 4  # a retry with >= 4 earlier same-key failures is out of policy (R_max = 3)
KILL_COUNT_POLL = 6  # polls: >= 6 earlier pending polls of the resource (P_max = 5)
PV_CAP = 100  # passed counts are compared up to this value


def _token_id(s):
    """13-bit identity pattern of a non-reserved surface token (hashed word, digit group or
    punctuation character), exactly as the network's token-embedding bits encode it; None for
    reserved words and markers."""
    if s in _RESERVED or s in _MARKERS:
        return None
    if re.match(_WORD_PATTERN, s) is not None:
        tid = 2048 + int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16) % 8192
    elif s.isdigit():
        if len(s) == 1:
            tid = 224 + int(s)
        elif len(s) == 2:
            tid = 234 + int(s)
        else:
            tid = 334 + int(s)
    elif len(s) == 1 and s in _PUNCT:
        tid = 128 + _PUNCT.index(s)
    else:
        return None
    return (tid - 2048) % 8192


def _hashed(s):
    """A hashed-word token other than the empty-side hunk id h0."""
    return s != "h0" and s not in _RESERVED and re.match(_WORD_PATTERN, s) is not None


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
    pv = 0
    for i, s in enumerate(tb):
        if s in _STATUS:
            status = s
            if i + 1 < n and tb[i + 1].isdigit():
                pv = min(int(tb[i + 1]), PV_CAP)
    s_ok = status in ("ok", "ready")
    s_fail = status in ("error", "pending", "FAILED")
    has_hunk = "\n@" in tb
    intent = None
    first_any = None
    for i in range(10, n):
        s = tb[i]
        if _hashed(s):
            intent = _token_id(s)
            break
        if first_any is None:
            first_any = _token_id(s)
    if intent is None:
        intent = first_any
    cls = _tool_class(tool)
    holds = set()
    hnews = set()
    for p, s in enumerate(tb):
        if s != ">" or p == 0 or p + 1 >= n:
            continue
        if _hashed(tb[p - 1]):
            holds.add(_token_id(tb[p - 1]))
        if _hashed(tb[p + 1]):
            hnews.add(_token_id(tb[p + 1]))
    return {
        "tool": tool, "insp": tool in _INSP, "val": tool in _VAL, "mut": tool in _MUT,
        "revert": tool == "revert", "poll": tool == "poll_status",
        "s_ok": s_ok, "s_fail": s_fail, "has_hunk": has_hunk, "pv": pv, "holds": holds, "hnews": hnews,
        "key": (cls, intent) if (cls is not None and intent is not None) else None,
    }


def _relations(feats):
    """Per-block effect flags, each using only the blocks before it (one pass, with running
    per-key statistics: last index, number of failures, largest passed count)."""
    rel = []
    by_key = {}
    last_vh = None  # index of the last block that is a validation or has a hunk line
    last_val = None  # index of the last validation block
    for b, f in enumerate(feats):
        lo = max(0, b - (WINDOW - 1))
        rev = any(h in feats[e]["holds"] for e in range(lo, b) for h in f["hnews"])
        stats = by_key.get(f["key"]) if f["key"] is not None else None
        prev_i = stats[0] if stats is not None else None
        prev = feats[prev_i] if prev_i is not None else None
        exists = prev is not None
        exists7 = exists and b - prev_i <= WINDOW - 1
        maxpv = stats[2] if stats is not None else 0
        c_fail = stats[1] if stats is not None else 0
        same_status = exists and prev["s_ok"] == f["s_ok"] and prev["s_fail"] == f["s_fail"]
        nearest = 0
        if last_vh is not None:
            nearest = int(feats[last_vh]["val"]) - int(feats[last_vh]["has_hunk"])
        better = exists and f["pv"] > maxpv
        improved = f["val"] and better and (f["s_fail"] or (f["s_ok"] and not prev["s_fail"]))
        # a retry repeats a recent same-key failure; validations also need an unchanged workspace
        # (no hunk block since the last validation) and at least as many passes as before
        retry_base = f["s_fail"] and exists7 and prev["s_fail"] and (nearest == 1 if f["val"] else True)
        retry = retry_base and ((f["pv"] >= prev["pv"] and not rev) if f["val"] else True)
        within = c_fail < (KILL_COUNT_POLL if f["poll"] else KILL_COUNT)
        inpol = retry and within
        kill = retry and not within
        if f["val"]:
            resolv = f["s_ok"] and exists and prev["s_fail"] and rel[prev_i]["retry_base"]
        else:
            resolv = f["s_ok"] and exists and prev["s_fail"]
        # a stagnant validation repeats its predecessor's status without a new high
        stagnant = f["val"] and same_status and not better and not rev
        nv_same = rel[last_val]["same_status"] if last_val is not None else False
        futile = f["mut"] and f["has_hunk"] and not f["s_fail"] and nv_same and not rev
        noop = f["mut"] and not f["has_hunk"] and not f["s_fail"]
        redinsp = f["insp"] and f["s_ok"] and exists
        si_revert = f["revert"] and not rev
        si = rev or si_revert or noop or futile or redinsp or (stagnant and not inpol) or (kill and not f["val"])
        rel.append({"rev": rev, "si": si, "inpol": inpol, "resolv": resolv, "improved": improved,
                    "same_status": same_status, "retry_base": retry_base, "fail": f["s_fail"]})
        if f["key"] is not None:
            by_key[f["key"]] = (b, c_fail + int(f["s_fail"]), max(maxpv, f["pv"]))
        if f["val"] or f["has_hunk"]:
            last_vh = b
        if f["val"]:
            last_val = b
    return rel


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
    feats = [_block_features(tb) for tb in blocks]
    rel = _relations(feats)
    t = len(blocks) - 1
    lo = max(0, t - (WINDOW - 1))
    n_rev = sum(1 for b in range(lo, t + 1) if rel[b]["rev"])
    n_si = sum(1 for b in range(lo, t + 1) if rel[b]["si"])
    prog = any(rel[b]["improved"] for b in range(lo, t + 1))
    nearest_episode = None
    for b in range(t, -1, -1):
        if rel[b]["fail"] or rel[b]["resolv"]:
            nearest_episode = b
            break
    active = nearest_episode is not None and rel[nearest_episode]["inpol"]
    recent_resolv = any(rel[b]["resolv"] for b in range(max(lo, t - (K_REC - 1)), t + 1))
    if n_rev >= N_REV and not prog:
        return OSCILLATING
    if n_si >= N_STALL and rel[t]["si"] and not prog and not active:
        return STALLED
    if active or recent_resolv:
        return RECOVERY
    return PRODUCTIVE
