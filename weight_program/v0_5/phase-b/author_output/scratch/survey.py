"""Survey the disclosed windows: one compact line per event with observable features, for reading.
This is analysis only (nothing here feeds the weights). Usage: ./py survey.py [LABEL ...]  -> scratch/survey_<LABEL>.txt

Per event: index, family code, status, validation count, and flags:
  rep   = the target (read/list path, search query, http url, poll resource, test suite+tests) occurred in an earlier event
  cnt   = validation count equals the count of the previous validation event in the session
  cntw  = count equals the count of any earlier validation in the session
  rev   = a hunk line's h_new equals an earlier hunk id (h_old or h_new) of the same file in the session
  noop  = edit/write/revert with no hunk line
  nomut = validation with no mutation (edit/write/revert with a hunk) since the previous validation
  k     = number of identical consecutive failures of the same validation key so far (retries)
"""
import os
import re
import sys

import v05_load

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
    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith("@ "):
            parts = s[2:].split(" ")
            if parts[0] != "TRUNC":
                hunks.append((parts[0], parts[1], parts[3]))
        elif s.startswith("! "):
            err = s[2:]
    return fam, fields, status, count, hunks, err


def main():
    labels = sys.argv[1:] or ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
    ws = v05_load.windows()
    for lab in labels:
        path = os.path.join(OUT, f"survey_{lab}.txt")
        with open(path, "w") as f:
            for w in ws:
                if w["oracle_label"] != lab:
                    continue
                blocks = w["blocks"]
                seen_targets = set()
                seen_hunks = {}  # path -> set of ids
                last_val_count = None
                counts_seen = set()
                last_fail_key = None
                nfail = 0
                mut_since_val = True
                rows = []
                for i, b in enumerate(blocks):
                    fam, fields, status, count, hunks, err = parse(b)
                    flags = []
                    key = fam + " " + fields
                    if fam in ("read_file", "list_dir", "search", "http_call", "poll_status", "run_tests", "build", "lint"):
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
                    # retry counter for failures (same family+fields+count/err)
                    if status in ("FAILED", "error", "pending"):
                        fk = key + "|" + str(count) + "|" + str(err)
                        if fk == last_fail_key:
                            nfail += 1
                        else:
                            nfail = 0
                            last_fail_key = fk
                        flags.append(f"k{nfail}")
                    elif status in ("ok", "ready"):
                        if last_fail_key is not None and last_fail_key.startswith(re.sub(r" req=\S+", "", key)):
                            flags.append("resolved")
                        last_fail_key = None
                        nfail = 0
                    st = {"ok": "+", "error": "!", "FAILED": "F", "ready": "r", "pending": "p"}[status]
                    desc = f"{FAM[fam]}{st}"
                    if count:
                        desc += count
                    short_fields = fields.replace("path=", "").replace("scope=", "").replace("resource=", "")[:38]
                    rows.append((i, desc, short_fields, " ".join(flags)))
                f.write(f"=== {w['id']} {lab} net={w['network_class']} t={w['t']}\n")
                for i, desc, sf, fl in rows[max(0, w["start"] - 3):]:
                    mark = "*" if i >= w["start"] else " "
                    f.write(f" {mark}{i:2d} {desc:10s} {sf:40s} {fl}\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
