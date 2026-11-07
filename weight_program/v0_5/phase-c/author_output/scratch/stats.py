"""Descriptive statistics over the disclosed windows (analysis only).
- mutations with h_old == h0 vs real hunks, per label, inside the window
- fresh reads (first read of a path in the session) inside windows, per label
- plan action token structure
- http_call events with hunks
- validation events: does the error line vary within a session (V vs W)?
"""
import re
from collections import Counter, defaultdict

import v05_load


def parse(block):
    lines = block.split("\n")
    h = lines[0]
    toks = h.split(" ")
    fam = toks[1]
    m = re.search(r" -> (\S+)(?: (\S+)/(\S+) passed)? in=(\S+) out=(\S+)$", h)
    status = m.group(1)
    count = (m.group(2), m.group(3)) if m.group(2) else None
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
            mm = re.match(r"(\S+) at (\S+):(\S+) \(", s[2:])
            err = (mm.group(1), mm.group(2), mm.group(3)) if mm else s
    return fam, fields, status, count, hunks, err


def main():
    for which in ("V", "W"):
        ws = v05_load.windows(which)
        h0 = Counter()
        real = Counter()
        fresh = Counter()
        fresh_final = Counter()
        http_hunk = 0
        plan_shapes = Counter()
        line_vary = Counter()
        for w in ws:
            lab = w["oracle_label"]
            blocks = w["blocks"]
            seen_read = set()
            lines_seen = set()
            for i, b in enumerate(blocks):
                fam, fields, status, count, hunks, err = parse(b)
                inwin = i >= w["start"]
                if fam in ("edit_file", "write_file", "revert") and inwin:
                    for p, ho, hn in hunks:
                        if ho == "h0":
                            h0[lab] += 1
                        else:
                            real[lab] += 1
                if fam == "http_call" and hunks:
                    http_hunk += 1
                if fam == "read_file":
                    path = fields.split(" ")[0]
                    if path not in seen_read:
                        if inwin:
                            fresh[lab] += 1
                            if i == w["t"]:
                                fresh_final[lab] += 1
                        seen_read.add(path)
                if fam in ("plan", "ask", "yield"):
                    act = fields.replace("action=", "")
                    shape = re.sub(r"[A-Za-z_][A-Za-z0-9_.\-/]*", "W", act)
                    shape = re.sub(r"[0-9]+", "D", shape)
                    plan_shapes[shape] += 1
                if fam in ("run_tests", "build", "lint") and status == "FAILED" and err:
                    lines_seen.add(err[2])
            line_vary[(lab, len(lines_seen))] += 1
        print(f"=== {which}")
        print(" h0 mutations in window by label:", dict(h0))
        print(" real-hunk mutations in window by label:", dict(real))
        print(" fresh reads in window by label:", dict(fresh), " as final event:", dict(fresh_final))
        print(" http_call events with hunks:", http_hunk)
        print(" plan action shapes:", dict(plan_shapes))
        print(" distinct failing lines per session (label, n):", sorted(line_vary.items()))


if __name__ == "__main__":
    main()
