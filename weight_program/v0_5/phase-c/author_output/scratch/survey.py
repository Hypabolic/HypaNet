"""Compact survey of disclosed windows: one line per event with observable flags and the base network's class.
Usage: ./py survey.py WHICH [LABEL ...] [edits]   -> scratch/survey_<WHICH>_<LABEL>.txt
Flags: rep (target seen earlier in session), cnt (count == previous validation's count), cntw (count seen earlier),
rev (h_new equals an earlier hunk id of that file), noop (mutation without hunk), nomut (validation without a
file-changing mutation since the previous validation), kN (consecutive identical failures: same family+fields+count+err),
res (ok/ready after a failure of the same key)."""
import os
import re
import sys

import model
import v05_load

CN = model.CLASS_NAMES
OUT = os.path.join(v05_load.OUTPUT, "scratch")
FAM = {"read_file": "R", "list_dir": "L", "search": "S", "edit_file": "E", "write_file": "W", "revert": "V",
       "run_tests": "T", "build": "B", "lint": "N", "http_call": "H", "poll_status": "P", "plan": "p", "ask": "a",
       "yield": "y"}


def parse(block):
    lines = block.split("\n")
    h = lines[0]
    toks = h.split(" ")
    fam = toks[1]
    m = re.search(r" -> (\S+)(?: (\S+)/(\S+) passed)? in=(\S+) out=(\S+)$", h)
    status = m.group(1)
    count = (m.group(2) + "/" + m.group(3)) if m.group(2) else None
    fields = h[len(toks[0]) + len(fam) + 2: m.start()]
    hunks = []
    err = None
    untracked = False
    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith("@ "):
            parts = s[2:].split(" ")
            if parts[0] != "TRUNC":
                hunks.append((parts[0], parts[1], parts[3]))
            else:
                hunks.append(("TRUNC", parts[1], ""))
        elif s.startswith("! "):
            err = s[2:]
        elif s.startswith("["):
            untracked = True
    return fam, fields, status, count, hunks, err, untracked


def main():
    which = sys.argv[1]
    rest = sys.argv[2:]
    edits = None
    labels = []
    for a in rest:
        if a.endswith(".edits"):
            edits = a
        else:
            labels.append(a)
    labels = labels or ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
    m = v05_load.load_b() if edits is None else v05_load.apply_edits_file(edits)[0]
    ws = v05_load.windows(which)
    for lab in labels:
        path = os.path.join(OUT, f"survey_{which}_{lab}.txt")
        with open(path, "w") as f:
            for w in ws:
                if w["oracle_label"] != lab:
                    continue
                blocks = w["blocks"]
                lg = v05_load.eoe_logits(m, blocks)
                cls = [CN[int(model.predict(lg[i:i + 1])[0])][:4] for i in range(len(blocks))]
                seen_targets = set()
                seen_hunks = {}
                last_val_count = None
                counts_seen = set()
                last_fail_key = None
                nfail = 0
                mut_since_val = True
                rows = []
                for i, b in enumerate(blocks):
                    fam, fields, status, count, hunks, err, untracked = parse(b)
                    flags = []
                    key = fam + " " + fields
                    if fam in ("read_file", "list_dir", "search", "http_call", "poll_status", "run_tests", "build", "lint", "plan", "ask", "yield"):
                        tkey = key if fam != "http_call" else re.sub(r" req=\S+", "", key)
                        if tkey in seen_targets:
                            flags.append("rep")
                        seen_targets.add(tkey)
                    if fam in ("edit_file", "write_file", "revert"):
                        if not hunks:
                            flags.append("noop")
                        else:
                            mut_since_val = True
                        for p, ho, hn in hunks:
                            ids = seen_hunks.setdefault(p, set())
                            if hn in ids:
                                flags.append("rev")
                            ids.add(ho)
                            ids.add(hn)
                    if fam in ("run_tests", "build", "lint"):
                        if count is not None and count == last_val_count:
                            flags.append("cnt")
                        if count is not None and count in counts_seen:
                            flags.append("cntw")
                        if not mut_since_val:
                            flags.append("nomut")
                        mut_since_val = False
                        last_val_count = count
                        if count is not None:
                            counts_seen.add(count)
                    if status in ("FAILED", "error", "pending"):
                        fk = re.sub(r" req=\S+", "", key) + "|" + str(count) + "|" + str(err)
                        if fk == last_fail_key:
                            nfail += 1
                        else:
                            nfail = 0
                            last_fail_key = fk
                        flags.append(f"k{nfail}")
                    elif status in ("ok", "ready"):
                        if last_fail_key is not None and last_fail_key.startswith(re.sub(r" req=\S+", "", key).split("|")[0][:20]):
                            flags.append("res")
                        last_fail_key = None
                        nfail = 0
                    if untracked:
                        flags.append("UNTR")
                    st = {"ok": "+", "error": "!", "FAILED": "F", "ready": "r", "pending": "p"}[status]
                    desc = f"{FAM[fam]}{st}"
                    if count:
                        desc += count
                    short_fields = fields.replace("path=", "").replace("scope=", "").replace("resource=", "").replace("action=", "")[:44]
                    hk = " ".join(f"{p.split('/')[-1][:14]}:{ho[:5]}>{hn[:5]}" for p, ho, hn in hunks)[:60]
                    er = ""
                    if err:
                        er = re.sub(r" \(.*\)$", "", err)[:40]
                    rows.append((i, desc, short_fields, hk, er, " ".join(flags), cls[i]))
                f.write(f"=== {w['id']} {lab} net={w['network_class']} t={w['t']}\n")
                for i, desc, sf, hk, er, fl, c in rows[max(0, w["start"] - 4):]:
                    mark = "*" if i >= w["start"] else " "
                    f.write(f" {mark}{i:2d} {c} {desc:9s} {sf:44s} {hk:28s} {er:40s} {fl}\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
