"""Per-window observable feature table (analysis only; nothing here feeds weights).

For each disclosed window, compute from the rendered blocks:
  kerr   : for the last failing (FAILED/error) event in the window, the number of EARLIER events in the whole
           session with the same error-location token (the WORD after 'at' in the '!' line)
  kpoll  : for the last pending poll in the window, the number of earlier pending polls of the same resource
  res    : distance from t back to the most recent "resolution": an ok/ready event whose family had an
           error/pending (http/poll) or a retried FAILED (validation with the same error location seen before)
           within the 4 events before it; None if no such event in the window
  act    : the most recent failing event (FAILED/error/pending) in the window has 1<=k<=3 (polls <=5) and no
           ok/ready event of the same family comes after it in the window
  I_*    : counts over the window of candidate ineffective events:
             cnt  = validation whose passed count equals an earlier (whole session) validation's passed count
             rep  = read/list/search/build/lint/plan/http/poll target token repeated from an earlier event
             repL = such a repeat that carries a lines= field (read_file)
             noop = mutation without hunk
             rev  = h_new equals an earlier hunk id
             ed3  = edit/write/revert with a hunk, with a cnt-validation among the previous 1..3 events
             out  = failing event beyond policy (kerr>=4) or pending poll with k>=6
  fin    : the final event's description and which of the flags it carries
"""
import os
import re
import sys
from collections import Counter, defaultdict

import v05_load

OUT = os.path.join(v05_load.OUTPUT, "scratch")
VAL = ("run_tests", "build", "lint")
MUT = ("edit_file", "write_file", "revert")


def parse(block):
    lines = block.split("\n")
    h = lines[0]
    toks = h.split(" ")
    fam = toks[1]
    m = re.search(r" -> (\S+)(?: (\S+)/(\S+) passed)? in=(\S+) out=(\S+)$", h)
    status = m.group(1)
    passed = m.group(2)
    fields = h[len(toks[0]) + len(fam) + 2: m.start()].strip()
    # target token as the network would see it
    target = None
    if fam in ("read_file", "list_dir", "edit_file", "write_file"):
        target = fields.split(" ")[0][len("path="):]
    elif fam == "search":
        mq = re.search(r'query="([^"]*)"', fields)
        q = mq.group(1) if mq else ""
        mw = re.match(r"[A-Za-z_][A-Za-z0-9_.\-/]*", q)
        target = mw.group(0) if mw else None
    elif fam in ("build", "lint"):
        target = fields[len("target="):].split(" ")[0]
    elif fam == "run_tests":
        target = fields[len("suite="):].split(" ")[0]
    elif fam == "http_call":
        mu = re.search(r"https://(\S+)", fields)
        u = mu.group(1) if mu else ""
        mw = re.match(r"[A-Za-z_][A-Za-z0-9_.\-/]*", u)
        target = mw.group(0) if mw else None
    elif fam == "poll_status":
        target = fields[len("resource="):].split(" ")[0]
    elif fam in ("plan", "ask", "yield"):
        target = fields[len("action="):]
    haslines = " lines=" in h
    hunks = []
    errloc = None
    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith("@ "):
            parts = s[2:].split(" ")
            if parts[0] != "TRUNC":
                hunks.append((parts[0], parts[1], parts[3]))
        elif s.startswith("! "):
            me = re.match(r"! (\S+) at (.*)", s)
            rest = me.group(2)
            mw = re.match(r"[A-Za-z_][A-Za-z0-9_.\-/]*", rest)
            errloc = mw.group(0) if mw else None
    return dict(fam=fam, status=status, passed=passed, target=target, haslines=haslines, hunks=hunks,
                errloc=errloc, fields=fields)


def analyse(blocks, start):
    evs = [parse(b) for b in blocks]
    n = len(evs)
    t = n - 1
    # per-event flags over the whole session
    seen_target = defaultdict(int)  # (fam-group, target) -> count
    seen_counts = Counter()
    seen_hunk = defaultdict(set)
    errcount = Counter()
    pollcount = Counter()
    flags = []
    for i, e in enumerate(evs):
        f = set()
        fam = e["fam"]
        grp = "rl" if fam in ("read_file", "list_dir") else fam
        if fam in VAL:
            if e["passed"] is not None and seen_counts[e["passed"]] > 0:
                f.add("cnt")
            if e["passed"] is not None:
                seen_counts[e["passed"]] += 1
        if fam in ("read_file", "list_dir", "search", "build", "lint", "plan", "ask", "yield", "http_call", "poll_status"):
            key = (grp if fam not in ("build", "lint") else "bl", e["target"])
            if e["target"] is not None and seen_target[key] > 0:
                f.add("rep")
                if e["haslines"]:
                    f.add("repL")
            if e["target"] is not None:
                seen_target[key] += 1
        if fam in MUT:
            if not e["hunks"]:
                f.add("noop")
            for p, ho, hn in e["hunks"]:
                if hn in seen_hunk[p]:
                    f.add("rev")
                seen_hunk[p].add(ho)
                seen_hunk[p].add(hn)
            if e["hunks"]:
                for d in (1, 2, 3):
                    if i - d >= 0 and "cnt" in flags[i - d]:
                        f.add("ed3")
                        break
        k = None
        if e["status"] in ("FAILED", "error"):
            k = errcount[e["errloc"]] if e["errloc"] else 0
            if e["errloc"]:
                errcount[e["errloc"]] += 1
            if k >= 4:
                f.add("out")
        if fam == "poll_status" and e["status"] == "pending":
            k = pollcount[e["target"]]
            pollcount[e["target"]] += 1
            if k >= 6:
                f.add("out")
        e["k"] = k
        flags.append(f)
    # resolution events
    res_idx = []
    for i, e in enumerate(evs):
        if e["status"] in ("ok", "ready") and e["fam"] in ("http_call", "poll_status") + VAL:
            for d in (1, 2, 3, 4):
                j = i - d
                if j < 0:
                    break
                p = evs[j]
                if p["fam"] == e["fam"] and p["status"] in ("error", "pending", "FAILED"):
                    if e["fam"] in VAL and (p["k"] or 0) < 1:
                        continue
                    res_idx.append(i)
                    break
    res = None
    for i in res_idx:
        if i >= start:
            res = t - i
    # active
    act = False
    last_fail = None
    for i in range(start, n):
        e = evs[i]
        if e["status"] in ("FAILED", "error", "pending"):
            last_fail = i
    if last_fail is not None:
        e = evs[last_fail]
        k = e["k"] or 0
        ok_after = any(evs[j]["status"] in ("ok", "ready") and evs[j]["fam"] == e["fam"] for j in range(last_fail + 1, n))
        lim = 5 if e["fam"] == "poll_status" else 3
        act = (1 <= k <= lim) and not ok_after
        kinfo = f"{e['fam'][:4]}k{k}"
    else:
        kinfo = "-"
    cnts = Counter()
    for i in range(start, n):
        for fl in flags[i]:
            cnts[fl] += 1
    fin = evs[t]["fam"][:4] + ":" + ",".join(sorted(flags[t]))
    return dict(kinfo=kinfo, res=res, act=act, cnts=cnts, fin=fin, flags=flags, evs=evs)


def main():
    ws = v05_load.windows()
    rows = []
    for w in ws:
        a = analyse(w["blocks"], w["start"])
        c = a["cnts"]
        I = c["cnt"] + c["rep"] + c["noop"] + c["rev"] + c["ed3"] + c["out"]
        rows.append((w["id"], w["oracle_label"][:4], a["kinfo"], a["res"], a["act"], I,
                     f"cnt{c['cnt']} rep{c['rep']}(L{c['repL']}) noop{c['noop']} rev{c['rev']} ed3{c['ed3']} out{c['out']}",
                     a["fin"]))
    with open(os.path.join(OUT, "features.txt"), "w") as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]} lastfail={r[2]:10s} res={str(r[3]):5s} act={int(r[4])} I={r[5]:2d} {r[6]:50s} final={r[7]}\n")
    print("wrote features.txt")


if __name__ == "__main__":
    main()
