"""Refined per-window observable feature table (analysis only; nothing here feeds weights).

Changes from features.py:
  kline  : retry count keyed on the error LINE number (the digits before ' (' in the '!' line); for pending polls
           keyed on the resource token
  cntok  : ok/ready validation whose passed count equals an earlier validation's passed count (session-wide)
  rep    : read/list path, search first word, plan/ask/yield action repeated (session-wide, same family group)
  edr    : mutation with a hunk whose previous 1..3 events include a retried FAILED (kline>=1) or a cntok validation
  out    : FAILED/error with kline>=4, or pending poll with k>=6
  res    : distance back to the latest ok/ready whose family matches an error/pending (http/poll) within the
           previous 4 events, or (validations) an ok whose total equals the total of a retried FAILED within the
           previous 4 events
  act    : the latest failing event in the window has 1<=k<=3 (polls<=5) and no later ok/ready of its family
Usage: ./py features2.py [Vnnn ...]   (with ids: print those windows' events with flags)
"""
import os
import re
import sys
from collections import Counter, defaultdict

import v05_load

OUT = os.path.join(v05_load.OUTPUT, "scratch")
VAL = ("run_tests", "build", "lint")
MUT = ("edit_file", "write_file", "revert")
WORD = r"[A-Za-z_][A-Za-z0-9_.\-/]*"


def parse(block):
    lines = block.split("\n")
    h = lines[0]
    toks = h.split(" ")
    fam = toks[1]
    m = re.search(r" -> (\S+)(?: (\S+)/(\S+) passed)? in=(\S+) out=(\S+)$", h)
    status, passed, total = m.group(1), m.group(2), m.group(3)
    fields = h[len(toks[0]) + len(fam) + 2: m.start()].strip()
    target = None
    if fam in ("read_file", "list_dir", "edit_file", "write_file"):
        target = fields.split(" ")[0][len("path="):]
    elif fam == "search":
        mq = re.search(r'query="([^"]*)"', fields)
        mw = re.match(WORD, mq.group(1) if mq else "")
        target = mw.group(0) if mw else None
    elif fam in ("build", "lint"):
        target = fields[len("target="):].split(" ")[0]
    elif fam == "run_tests":
        target = fields[len("suite="):].split(" ")[0]
    elif fam == "http_call":
        mu = re.search(r"https://(\S+)", fields)
        mw = re.match(WORD, mu.group(1) if mu else "")
        target = mw.group(0) if mw else None
    elif fam == "poll_status":
        target = fields[len("resource="):].split(" ")[0]
    elif fam in ("plan", "ask", "yield"):
        target = fields[len("action="):]
    hunks = []
    line = None
    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith("@ "):
            parts = s[2:].split(" ")
            if parts[0] != "TRUNC":
                hunks.append((parts[0], parts[1], parts[3]))
        elif s.startswith("! "):
            ml = re.search(r":(\S+) \(", s)
            line = ml.group(1) if ml else None
    return dict(fam=fam, status=status, passed=passed, total=total, target=target, hunks=hunks, line=line,
                fields=fields, haslines=" lines=" in h)


def analyse(blocks, start):
    evs = [parse(b) for b in blocks]
    n = len(evs)
    t = n - 1
    seen_target = Counter()
    seen_counts = Counter()
    seen_hunk = defaultdict(set)
    linecount = Counter()
    pollcount = Counter()
    flags = []
    for i, e in enumerate(evs):
        f = set()
        fam = e["fam"]
        if fam in VAL and e["passed"] is not None:
            if e["status"] == "ok" and seen_counts[e["passed"]] > 0:
                f.add("cntok")
            seen_counts[e["passed"]] += 1
        grp = {"read_file": "rl", "list_dir": "rl", "search": "s", "plan": "p", "ask": "p", "yield": "p"}.get(fam)
        if grp and e["target"] is not None:
            if seen_target[(grp, e["target"])] > 0:
                f.add("rep")
            seen_target[(grp, e["target"])] += 1
        k = None
        if e["status"] in ("FAILED", "error"):
            if e["line"] is not None and e["line"] != "?":
                k = linecount[e["line"]]
                linecount[e["line"]] += 1
            else:
                k = 0
            if k >= 1:
                f.add("retry")
            if k >= 4:
                f.add("out")
        if fam == "poll_status" and e["status"] == "pending":
            k = pollcount[e["target"]]
            pollcount[e["target"]] += 1
            if k >= 1:
                f.add("retry")
            if k >= 6:
                f.add("out")
        e["k"] = k
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
                    j = i - d
                    if j >= 0 and (("retry" in flags[j] and evs[j]["fam"] in VAL) or "cntok" in flags[j]):
                        f.add("edr")
                        break
        flags.append(f)
    res_idx = []
    for i, e in enumerate(evs):
        if e["status"] in ("ok", "ready") and e["fam"] in ("http_call", "poll_status") + VAL:
            for d in (1, 2, 3, 4):
                j = i - d
                if j < 0:
                    break
                p = evs[j]
                if e["fam"] in VAL:
                    if p["fam"] in VAL and p["status"] == "FAILED" and "retry" in flags[j] and p["total"] == e["total"]:
                        res_idx.append(i)
                        break
                elif p["fam"] == e["fam"] and p["status"] in ("error", "pending"):
                    res_idx.append(i)
                    break
    res = None
    for i in res_idx:
        if i >= start:
            res = t - i
    last_fail = None
    for i in range(start, n):
        if evs[i]["status"] in ("FAILED", "error", "pending"):
            last_fail = i
    act = False
    kinfo = "-"
    if last_fail is not None:
        e = evs[last_fail]
        k = e["k"] or 0
        ok_after = any(evs[j]["status"] in ("ok", "ready") and evs[j]["fam"] == e["fam"] for j in range(last_fail + 1, n))
        lim = 5 if e["fam"] == "poll_status" else 3
        act = (1 <= k <= lim) and not ok_after
        kinfo = f"{e['fam'][:4]}k{k}"
    cnts = Counter()
    for i in range(start, n):
        for fl in flags[i]:
            cnts[fl] += 1
    fin = evs[t]["fam"][:4] + ":" + ",".join(sorted(flags[t]))
    return dict(kinfo=kinfo, res=res, act=act, cnts=cnts, fin=fin, flags=flags, evs=evs)


def main():
    ws = v05_load.windows()
    want = set(sys.argv[1:])
    rows = []
    for w in ws:
        a = analyse(w["blocks"], w["start"])
        c = a["cnts"]
        I = c["cntok"] + c["rep"] + c["noop"] + c["rev"] + c["edr"] + c["out"]
        fin_flags = a["flags"][-1]
        ineff_t = bool(fin_flags & {"cntok", "rep", "noop", "rev", "edr", "out"})
        rows.append((w["id"], w["oracle_label"][:4], a["kinfo"], a["res"], a["act"], I, ineff_t,
                     f"cntok{c['cntok']} rep{c['rep']} noop{c['noop']} rev{c['rev']} edr{c['edr']} out{c['out']} retry{c['retry']}",
                     a["fin"]))
        if w["id"] in want:
            print(f"=== {w['id']} {w['oracle_label']} t={w['t']} start={w['start']}")
            for i in range(max(0, w["start"] - 3), len(a["evs"])):
                e = a["evs"][i]
                mark = "*" if i >= w["start"] else " "
                desc = f"{e['fam']} {e['status']}" + (f" {e['passed']}/{e['total']}" if e["passed"] else "")
                print(f" {mark}{i:2d} {desc:28s} {str(e['target'])[:30]:30s} line={e['line']} k={e['k']} {' '.join(sorted(a['flags'][i]))}")
    with open(os.path.join(OUT, "features2.txt"), "w") as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]} lastfail={r[2]:10s} res={str(r[3]):5s} act={int(r[4])} I={r[5]:2d} inT={int(r[6])} "
                    f"{r[7]:56s} final={r[8]}\n")
    # summary of the candidate rule
    conf = Counter()
    for r in rows:
        wid, lab, kinfo, res, act, I, ineff_t, _, fin = r
        rev = int(re.search(r"rev(\d+)", r[7]).group(1))
        if rev >= 2:
            pred = "OSCI"
        elif act:
            pred = "RECO"
        elif I >= 5 and ineff_t:
            pred = "STAL"
        elif res is not None and res <= 3:
            pred = "RECO"
        else:
            pred = "PROD"
        conf[(lab, pred)] += 1
        if pred != lab:
            print("rule-miss", wid, lab, "->", pred, f"act={int(act)} res={res} I={I} inT={int(ineff_t)} {r[7]} fin={fin}")
    print("confusion (oracle -> rule):")
    for lab in ("PROD", "RECO", "STAL", "OSCI"):
        print(lab, {p: conf[(lab, p)] for p in ("PROD", "RECO", "STAL", "OSCI")})


if __name__ == "__main__":
    main()
