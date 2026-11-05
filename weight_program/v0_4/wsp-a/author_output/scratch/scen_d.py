"""Scenario set D: matched groups for OSCI proxies and count boundaries.
Groups (same prefix, same project):
  D1a ends on failing test: osc [e,t]*4 | stal5 [t]*8 | stal6 [n,t]*4   (e = A/B edit, n = no-op edit)
  D1b ends on edit:        osc [t,e]*4 | stal6 [t,n]*4 | prod [t,f]*4 (fresh edits, tests improving)
  D1c revert cycles:       osc [e,b,rv,b]*2 | prod fresh edits with builds ok
  D1d error-free:          osc edits A/B x8 | stal reads x8
  D2  reads:               stal [r]*7 | stal2 [r1,r2]*4 | prod distinct reads
  D3  http chains:         reco pre4 + err x3 | stal pre4 + err x9
  D4  polls:               reco pre3 + pending x4 | stal pre3 + pending x11
  D5  diagnosed repair resolved then k explore ops (k = 1..5), one session (windows labelled by the oracle)
"""
from mk2 import Session, Ctx, explore, hnum, ftext, variant, PROJECTS

GROUPS = []  # (group id, variant tag, Session)


def mk(sid, pre, proj):
    s = Session(sid, start=["2026-03-05T08:00:00.000+00:00", "2026-03-06T16:30:12.000+00:00", "2026-03-07T11:11:11.000+00:00"][proj % 3])
    c = Ctx(s)
    c.P = PROJECTS[proj % len(PROJECTS)]
    if pre:
        explore(c, pre)
    return s, c


def failing(ctx, p, n, eff):
    ctx.s.tests(ctx.suite(), n - 2, n, eff, error_class=ctx.err(), error_location=p)


def d1a(gid, pre, proj):
    n = 9 + hnum(gid, hi=8)
    # osc
    s, c = mk(gid + "_osc", pre, proj); p = c.path(); A = c.base_text(p); B = variant(A, 3, "alt")
    for i in range(4):
        s.edit(p, B if i % 2 == 0 else A, eff="NONE" if i == 0 else "REVERSION")
        failing(c, p, n, "NONE" if i == 0 else "REDUNDANT")
    GROUPS.append((gid, "osc", s))
    # stal5
    s, c = mk(gid + "_stal5", pre, proj); p = c.path()
    for i in range(8):
        failing(c, p, n, "NONE" if i == 0 else "REDUNDANT")
    GROUPS.append((gid, "stal5", s))
    # stal6
    s, c = mk(gid + "_stal6", pre, proj); p = c.path(); A = c.base_text(p); txt = variant(A, 2, "attempt")
    for i in range(4):
        s.edit(p, txt, eff="NONE" if i == 0 else "REDUNDANT")
        failing(c, p, n, "NONE" if i == 0 else "REDUNDANT")
    GROUPS.append((gid, "stal6", s))


def d1b(gid, pre, proj):
    n = 9 + hnum(gid, hi=8)
    s, c = mk(gid + "_osc", pre, proj); p = c.path(); A = c.base_text(p); B = variant(A, 3, "alt")
    for i in range(4):
        failing(c, p, n, "NONE" if i == 0 else "REDUNDANT")
        s.edit(p, B if i % 2 == 0 else A, eff="NONE" if i == 0 else "REVERSION")
    GROUPS.append((gid, "osc", s))
    s, c = mk(gid + "_stal6", pre, proj); p = c.path(); A = c.base_text(p); txt = variant(A, 2, "attempt")
    for i in range(4):
        failing(c, p, n, "NONE" if i == 0 else "REDUNDANT")
        s.edit(p, txt, eff="NONE" if i == 0 else "REDUNDANT")
    GROUPS.append((gid, "stal6", s))
    s, c = mk(gid + "_prod", pre, proj); p = c.path(); txt = c.base_text(p)
    for i in range(4):
        left = max(0, 4 - i)
        if left:
            s.tests(c.suite(), n - left, n, "NONE" if i == 0 else "PROGRESS", dp=0.0 if i == 0 else 0.1, error_class=c.err(), error_location=p)
        else:
            s.tests(c.suite(), n, n, "PROGRESS", dp=0.1)
        txt = variant(txt, i + 1, f"fix{i}")
        s.edit(p, txt, dp=0.1)
    GROUPS.append((gid, "prod", s))


def d1c(gid, pre, proj):
    s, c = mk(gid + "_osc", pre, proj); p = c.path(); A = c.base_text(p); B = variant(A, 4, "try")
    for i in range(2):
        s.edit(p, B, eff="NONE" if i == 0 else "REVERSION")
        s.build(c.target(), True, "NONE" if i == 0 else "REDUNDANT")
        s.revert({p: A})
        s.build(c.target(), True, "REDUNDANT")
    GROUPS.append((gid, "osc", s))
    s, c = mk(gid + "_prod", pre, proj); p = c.path(); txt = c.base_text(p)
    for i in range(4):
        txt = variant(txt, i + 1, f"v{i}")
        s.edit(p, txt, dp=0.05)
        s.build(c.target(), True, "PROGRESS", dp=0.05)
    GROUPS.append((gid, "prod", s))


def d1d(gid, pre, proj):
    s, c = mk(gid + "_osc", pre, proj); p = c.path(); A = c.base_text(p); B = variant(A, 3, "alt")
    for i in range(8):
        s.edit(p, B if i % 2 == 0 else A, eff="NONE" if i == 0 else "REVERSION")
    GROUPS.append((gid, "osc", s))
    s, c = mk(gid + "_stal", pre, proj); p = c.path()
    for i in range(8):
        s.read(p, eff="NEW_FACT" if i == 0 else "REDUNDANT")
    GROUPS.append((gid, "stal", s))


def d2(gid, pre, proj):
    s, c = mk(gid + "_stal", pre, proj); p = c.path()
    for i in range(7):
        s.read(p, eff="NEW_FACT" if i == 0 else "REDUNDANT")
    GROUPS.append((gid, "stal", s))
    s, c = mk(gid + "_stal2", pre, proj); p1, p2 = c.path(), c.path()
    for i in range(8):
        s.read(p1 if i % 2 == 0 else p2, eff="NEW_FACT" if i < 2 else "REDUNDANT")
    GROUPS.append((gid, "stal2", s))
    s, c = mk(gid + "_prod", pre, proj)
    for i in range(7):
        s.read(c.path(), lines=(10 * i + 1, 10 * i + 50) if i % 2 else None)
    GROUPS.append((gid, "prod", s))


def d3(gid, pre, proj):
    for tag, k in (("reco", 3), ("stal", 9)):
        s, c = mk(gid + "_" + tag, pre, proj); loc = c.path(0)
        for i in range(k + 1):
            s.http("POST", c.P["host"], "/v1/images/{name}/push", False, "FAIL0" if i == 0 else "RETRY", error_class=c.err(1), error_location=loc, body={"ref": "main"})
        GROUPS.append((gid, tag, s))


def d4(gid, pre, proj):
    for tag, k in (("reco", 4), ("stal", 11)):
        s, c = mk(gid + "_" + tag, pre, proj)
        for i in range(k + 1):
            s.poll(c.P["res"], "pending", "FAIL0" if i == 0 else "POLL", backoff=min(2 ** i, 64))
        GROUPS.append((gid, tag, s))


def d5(gid, pre, proj):
    s, c = mk(gid + "_reco", pre, proj); p = c.path(); base = c.base_text(p)
    s.build(c.target(), False, "FAIL0", error_class=c.err(2), error_location=p)
    s.read(p, lines=(80, 140))
    s.edit(p, variant(base, 3, "repair"), dp=0.0)
    s.build(c.target(), True, "RESOLVES")
    explore(c, 5, j0=3)
    GROUPS.append((gid, "reco", s))


def build_all():
    GROUPS.clear()
    j = 0
    for pre in (0, 2, 4):
        for proj in range(8):
            r = (proj + pre) % 4
            if r == 0:
                d1a(f"d{j}", pre, proj); j += 1
                d2(f"d{j}", pre, proj); j += 1
            if r == 1:
                d1b(f"d{j}", pre, proj); j += 1
                d3(f"d{j}", pre, proj); j += 1
            if r == 2:
                d1c(f"d{j}", pre, proj); j += 1
                d4(f"d{j}", pre, proj); j += 1
                d5(f"d{j}", pre, proj); j += 1
            if r == 3:
                d1d(f"d{j}", pre, proj); j += 1
                d1a(f"d{j}", pre, proj); j += 1
    return list(GROUPS)
