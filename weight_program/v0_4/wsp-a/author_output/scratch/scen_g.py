"""Scenario set G: matched pairs for two remaining distinctions.
  IMPROV:    [edit(fresh), FAILED k/n] x4 with k improving (positive) vs constant (negative); ends on a failing test
             (variant 'e' adds a final edit so the window ends on an edit). Only the pass counts differ.
  EDITCOUNT: [edit, FAILED x7] (negative: one edit) vs [edit, FAILED] x4 with fresh ids (positive: four edits),
             constant counts; ends on a failing test."""
from mk2 import Session, Ctx, explore, hnum, variant, PROJECTS

PAIRS = []


def mk(sid, pre, proj):
    s = Session(sid, start=["2026-06-01T09:00:00.000+00:00", "2026-06-02T15:20:40.000+00:00", "2026-06-03T21:33:05.000+00:00"][proj % 3])
    c = Ctx(s)
    c.P = PROJECTS[proj % len(PROJECTS)]
    if pre:
        explore(c, pre)
    return s, c


def improv(pid, pre, proj, end_on_edit=False):
    n = 9 + hnum(pid, hi=8)
    out = []
    for tag, improving in (("p", True), ("n", False)):
        s, c = mk(pid + tag, pre, proj)
        p = c.path(); txt = c.base_text(p)
        for i in range(4):
            txt = variant(txt, i + 1, f"fix{i}")
            s.edit(p, txt, dp=0.1 if improving else 0.0, eff="PROGRESS" if improving else "NONE")
            left = (4 - i) if improving else 3
            s.tests(c.suite(), n - left, n, "PROGRESS" if improving else "NONE", dp=0.1 if improving else 0.0,
                    error_class=c.err(), error_location=p)
        if end_on_edit:
            txt = variant(txt, 5, "fix4")
            s.edit(p, txt, dp=0.1 if improving else 0.0, eff="PROGRESS" if improving else "NONE")
        out.append(s)
    PAIRS.append((pid, "IMPROV", +1, out[0])); PAIRS.append((pid, "IMPROV", -1, out[1]))


def editcount(pid, pre, proj):
    n = 9 + hnum(pid, hi=8)
    s, c = mk(pid + "p", pre, proj)
    p = c.path(); txt = c.base_text(p)
    for i in range(4):
        txt = variant(txt, i + 1, f"try{i}")
        s.edit(p, txt, eff="NONE")
        s.tests(c.suite(), n - 3, n, "NONE", error_class=c.err(), error_location=p)
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path(); txt2 = d.base_text(p2)
    t.edit(p2, variant(txt2, 1, "try0"), eff="NONE")
    for i in range(7):
        t.tests(d.suite(), n - 3, n, "NONE" if i == 0 else "REDUNDANT", error_class=d.err(), error_location=p2)
    PAIRS.append((pid, "EDITCOUNT", +1, s)); PAIRS.append((pid, "EDITCOUNT", -1, t))


def build_all():
    PAIRS.clear()
    j = 0
    for pre in (0, 2, 3):
        for proj in range(8):
            improv(f"g{j}", pre, proj, end_on_edit=(proj % 2 == 1)); j += 1
            if (proj + pre) % 2 == 0:
                editcount(f"g{j}", pre, proj); j += 1
    return list(PAIRS)
