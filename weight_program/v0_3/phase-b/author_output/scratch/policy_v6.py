"""Candidate v6: exactly the per-block / window computation the revised network implements.

Compared with v5: the intent of every tool is the first hashed token at header offset >= 10
(else the first other non-reserved token), a retry must repeat its predecessor's result and link
to a same-key failure within the last 7 blocks, one retry is already in policy, a FAILED -> ok
transition is a resolution rather than measured progress, and the passed count is the digit
group right after the status word (capped at 100)."""

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

WINDOW = 8
N_STALL = 5
N_REV = 2
K_REC = 4
KILL_COUNT = 4
KILL_COUNT_POLL = 6
PV_CAP = 100


def _token_id(s):
    if s in _RESERVED or s in _MARKERS:
        return None
    if re.match(_WORD_PATTERN, s) is not None:
        return 2048 + int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16) % 8192
    if s.isdigit():
        if len(s) == 1:
            return 224 + int(s)
        if len(s) == 2:
            return 234 + int(s)
        return 334 + int(s)
    if len(s) == 1 and s in _PUNCT:
        return 128 + _PUNCT.index(s)
    return None


def _hashed(s):
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
            pv = 0
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
        "tool": tool, "cls": cls, "insp": tool in _INSP, "val": tool in _VAL, "mut": tool in _MUT,
        "revert": tool == "revert", "poll": tool == "poll_status", "net": tool in _NET,
        "s_ok": s_ok, "s_fail": s_fail, "has_hunk": has_hunk, "pv": pv, "holds": holds, "hnews": hnews,
        "key": (cls, intent) if (cls is not None and intent is not None) else None,
    }


def analyse(blocks):
    feats = [_block_features(tb) for tb in blocks]
    rel = []
    for b, f in enumerate(feats):
        lo = max(0, b - (WINDOW - 1))
        rev = any(h in feats[e]["holds"] for e in range(lo, b) for h in f["hnews"])
        same = [i for i in range(b) if f["key"] is not None and feats[i]["key"] == f["key"]]
        prev_i = same[-1] if same else None
        prev = feats[prev_i] if prev_i is not None else None
        exists = prev is not None
        exists7 = exists and b - prev_i <= WINDOW - 1
        maxpv = max([feats[i]["pv"] for i in same], default=0)
        c_fail = sum(1 for i in same if feats[i]["s_fail"])
        notid = (not exists) or prev["s_ok"] != f["s_ok"] or prev["s_fail"] != f["s_fail"] or prev["pv"] != f["pv"]
        nearest = 0
        for i in range(b - 1, -1, -1):
            if feats[i]["val"] or feats[i]["has_hunk"]:
                nearest = int(feats[i]["val"]) - int(feats[i]["has_hunk"])
                break
        better = exists and f["pv"] > maxpv
        improved = f["val"] and better and (f["s_fail"] or (f["s_ok"] and not prev["s_fail"]))
        if f["val"]:
            retry = f["s_fail"] and exists7 and prev["s_fail"] and nearest == 1 and not notid
        else:
            retry = f["s_fail"] and exists7 and prev["s_fail"]
        kill = retry and c_fail >= (KILL_COUNT_POLL if f["poll"] else KILL_COUNT)
        inpol = retry and not kill
        if f["val"]:
            resolv = f["s_ok"] and exists and prev["s_fail"] and rel[prev_i]["retry"]
        else:
            resolv = f["s_ok"] and exists and prev["s_fail"]
        stagnant = f["val"] and exists and not notid
        nv_stag = False
        for i in range(b - 1, -1, -1):
            if feats[i]["val"]:
                nv_stag = rel[i]["stagnant"]
                break
        futile = f["mut"] and f["has_hunk"] and not f["s_fail"] and nv_stag
        noop = f["mut"] and not f["has_hunk"] and not f["s_fail"]
        redinsp = f["insp"] and f["s_ok"] and exists
        si_revert = f["revert"] and not rev
        si = rev or si_revert or noop or futile or redinsp or (stagnant and not inpol) or (kill and not f["val"])
        rel.append({"rev": rev, "si": si, "inpol": inpol, "resolv": resolv, "kill": kill, "improved": improved,
                    "stagnant": stagnant, "futile": futile, "redinsp": redinsp, "noop": noop, "retry": retry,
                    "r": int(retry), "fail": f["s_fail"], "ident": not notid})
    return feats, rel


def decide(feats, rel, t):
    lo = max(0, t - (WINDOW - 1))
    n_rev = sum(1 for b in range(lo, t + 1) if rel[b]["rev"])
    n_si = sum(1 for b in range(lo, t + 1) if rel[b]["si"])
    prog = any(rel[b]["improved"] for b in range(lo, t + 1))
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
