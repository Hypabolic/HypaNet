"""Scenario set C: matched pairs. Each pair is two sessions with the same prefix that differ in one property:
REP (recent blocks repeat), REV (hunk ids return to earlier states), RES (latest status is a success after
errors), ERR (errors present), PROG (pass counts improving). Positive member first."""
from mk2 import Session, Ctx, explore, hnum, ftext, variant, PROJECTS

PAIRS = []  # (pair id, feature, sign, Session)


def mk(sid, pre, proj):
    s = Session(sid, start=["2026-03-01T09:14:03.250+00:00", "2026-03-02T14:02:11.000+00:00", "2026-03-04T03:21:45.100+00:00"][proj % 3])
    c = Ctx(s)
    c.P = PROJECTS[proj % len(PROJECTS)]
    if pre:
        explore(c, pre)
    return s, c


def pair(pid, feat, pos, neg):
    PAIRS.append((pid, feat, +1, pos))
    PAIRS.append((pid, feat, -1, neg))


# ---------------- REP: repeats vs distinct -------------------------------------------------------------
def rep_reads(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    s.read(p)
    for _ in range(k - 1):
        s.read(p, eff="REDUNDANT")
    t, d = mk(pid + "n", pre, proj)
    for i in range(k):
        d.s.read(d.path(), lines=(5 * i + 1, 5 * i + 60) if i % 2 else None)
    pair(pid, "REP", s, t)


def rep_tests(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    n = 9 + hnum(pid, hi=8)
    s.tests(c.suite(), n - 2, n, "NONE", error_class=c.err(), error_location=p)
    for _ in range(k - 1):
        s.tests(c.suite(), n - 2, n, "REDUNDANT", error_class=c.err(), error_location=p)
    t, d = mk(pid + "n", pre, proj)
    for i in range(k):
        suite = d.P["suites"][i % 2]
        m = 6 + (i * 5) % 11
        if i % 3 == 1:
            d.s.tests(suite, m, m, "PROGRESS", dp=0.1)
        else:
            d.s.tests(suite, m - 1 - (i % 2), m, "NONE", error_class=d.err(i), error_location=d.path(i))
    pair(pid, "REP", s, t)


def rep_builds(pid, pre, proj, k, family="build"):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    s.build(c.target(), False, "NONE", error_class=c.err(2), error_location=p, family=family)
    for _ in range(k - 1):
        s.build(c.target(), False, "REDUNDANT", error_class=c.err(2), error_location=p, family=family)
    t, d = mk(pid + "n", pre, proj)
    for i in range(k):
        tg = d.target() + ("" if i % 2 == 0 else "_" + d.P["suites"][1])
        d.s.build(tg, i % 3 == 1, "PROGRESS" if i % 3 == 1 else "NONE", dp=0.05 if i % 3 == 1 else 0,
                  error_class=None if i % 3 == 1 else d.err(i), error_location=d.path(i), family=family)
    pair(pid, "REP", s, t)


def rep_http(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    loc = c.path(0)
    for i in range(k):
        s.http("PUT", c.P["host"], "/buckets/{b}/objects/{k}", False, "FAIL0" if i == 0 else "RETRY", error_class=c.err(1), error_location=loc)
    t, d = mk(pid + "n", pre, proj)
    paths = ["/v1/images/{name}/push", "/jobs", "/v2/users/{id}", "/pipelines", "/v1/status/{id}", "/buckets/{b}", "/health", "/v3/items"]
    for i in range(k):
        ok = i % 3 != 1
        d.s.http(["GET", "POST", "PUT"][i % 3], d.P["host"], paths[i % len(paths)], ok, "NEW_FACT" if ok else "NONE",
                 error_class=None if ok else d.err(i), error_location=None if ok else d.path(i))
    pair(pid, "REP", s, t)


def rep_polls(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    for i in range(k):
        s.poll(c.P["res"], "pending", "FAIL0" if i == 0 else "POLL", backoff=min(2 ** i, 64))
    t, d = mk(pid + "n", pre, proj)
    for i in range(k):
        d.s.poll(d.P["res"] + f"_{i}", "ready" if i % 2 else "pending", "NEW_FACT" if i % 2 else "NONE", backoff=1 if i % 2 == 0 else None)
    pair(pid, "REP", s, t)


def rep_search(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    q = c.query(5)
    s.search(c.P["dirs"][0], q, limit=10)
    for _ in range(k - 1):
        s.search(c.P["dirs"][0], q, eff="REDUNDANT", limit=10)
    t, d = mk(pid + "n", pre, proj)
    for i in range(k):
        d.s.search(d.P["dirs"][i % 2], d.query(i + 1), limit=10 if i % 2 == 0 else None)
    pair(pid, "REP", s, t)


def rep_noop_edits(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    base = c.base_text(p)
    txt = variant(base, 2, "attempt")
    s.edit(p, txt, eff="NONE")
    n = 7 + hnum(pid, hi=6)
    s.tests(c.suite(), n - 2, n, "NONE", error_class=c.err(), error_location=p)
    for _ in range(k):
        s.edit(p, txt, eff="REDUNDANT")
        s.tests(c.suite(), n - 2, n, "REDUNDANT", error_class=c.err(), error_location=p)
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path()
    b2 = d.base_text(p2)
    txt2 = b2
    for i in range(k + 1):
        txt2 = variant(txt2, i + 1, f"try{i}")
        d.s.edit(p2, txt2, dp=0.1)
        left = max(0, 2 - i)
        if left:
            d.s.tests(d.suite(), n - left, n, "PROGRESS", dp=0.1, error_class=d.err(), error_location=p2)
        else:
            d.s.tests(d.suite(), n, n, "PROGRESS", dp=0.1)
    pair(pid, "REP", s, t)


# ---------------- REV: reversal vs fresh ids ------------------------------------------------------------
def rev_ab(pid, pre, proj, k, between=None):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    A = c.base_text(p)
    B = variant(A, 3, "alt")
    n = 8 + hnum(pid, hi=8)

    def val(ss, cc, pp, eff, ok=False):
        if between == "tests":
            if ok:
                ss.tests(cc.suite(), n, n, "PROGRESS", dp=0.1)
            else:
                ss.tests(cc.suite(), n - 2, n, eff, error_class=cc.err(), error_location=pp)
        elif between == "build_ok":
            ss.build(cc.target(), True, "NONE" if eff == "NONE" else "REDUNDANT")

    s.edit(p, B, eff="NONE")
    if between:
        val(s, c, p, "NONE")
    for i in range(k):
        s.edit(p, A if i % 2 == 0 else B, eff="REVERSION")
        if between:
            val(s, c, p, "REDUNDANT")
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path()
    txt = d.base_text(p2)
    for i in range(k + 1):
        txt = variant(txt, i + 1, f"v{i}")
        d.s.edit(p2, txt, dp=0.05)
        if between:
            val(d.s, d, p2, "NONE", ok=(between == "tests" and i == k))
    pair(pid, "REV", s, t)


def rev_revert(pid, pre, proj, k, between=None):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    A = c.base_text(p)
    B = variant(A, 4, "try")
    s.edit(p, B, eff="NONE")
    if between == "build_ok":
        s.build(c.target(), True, "NONE")
    for i in range(k):
        s.revert({p: A})
        s.edit(p, B, eff="REVERSION")
        if between == "build_ok":
            s.build(c.target(), True, "REDUNDANT")
    # negative: one revert, then fresh edits (V = 1 only)
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path()
    A2 = d.base_text(p2)
    txt = variant(A2, 4, "try")
    d.s.edit(p2, txt, eff="NONE")
    if between == "build_ok":
        d.s.build(d.target(), True, "NONE")
    d.s.revert({p2: A2}, eff="NONE")
    txt = A2
    for i in range(2 * k - 1):
        txt = variant(txt, i + 1, f"w{i}")
        d.s.edit(p2, txt, dp=0.05)
        if between == "build_ok":
            d.s.build(d.target(), True, "PROGRESS", dp=0.02)
    pair(pid, "REV", s, t)


def rev_write2(pid, pre, proj, k):
    s, c = mk(pid + "p", pre, proj)
    p1, p2 = c.path(), c.path()
    a1, a2 = c.base_text(p1), c.base_text(p2)
    s.write(p1, variant(a1, 1, "v2"), eff="NONE")
    s.write(p2, variant(a2, 1, "v2"), eff="NONE")
    for i in range(k):
        s.revert({p1: a1, p2: a2})
        s.write(p1, variant(a1, 1, "v2"), eff="REVERSION")
        s.write(p2, variant(a2, 1, "v2"), eff="REVERSION")
    t, d = mk(pid + "n", pre, proj)
    q1, q2 = d.path(), d.path()
    b1, b2 = d.base_text(q1), d.base_text(q2)
    for i in range(3 * k + 2):
        if i % 2 == 0:
            b1 = variant(b1, i + 1, f"u{i}"); d.s.write(q1, b1, dp=0.05)
        else:
            b2 = variant(b2, i + 1, f"u{i}"); d.s.write(q2, b2, dp=0.05)
    pair(pid, "REV", s, t)


def rev_abc(pid, pre, proj):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    A = c.base_text(p)
    B, C = variant(A, 1, "b"), variant(A, 2, "c")
    s.edit(p, B, eff="NONE")
    s.edit(p, C, eff="NONE")
    for txt in (A, B, C, A):
        s.edit(p, txt, eff="REVERSION")
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path()
    txt = d.base_text(p2)
    for i in range(6):
        txt = variant(txt, i + 1, f"s{i}")
        d.s.edit(p2, txt, dp=0.05)
    pair(pid, "REV", s, t)


# ---------------- RES: resolved vs still failing ----------------------------------------------------------
def res_http(pid, pre, proj, nfail, tail=0):
    s, c = mk(pid + "p", pre, proj)
    loc = c.path(0)
    for i in range(nfail):
        s.http("POST", c.P["host"], "/v1/images/{name}/push", False, "FAIL0" if i == 0 else "RETRY", error_class=c.err(1), error_location=loc, body={"ref": "main"})
    s.http("POST", c.P["host"], "/v1/images/{name}/push", True, "RESOLVES", body={"ref": "main"})
    t, d = mk(pid + "n", pre, proj)
    loc2 = d.path(0)
    for i in range(nfail + 1):
        d.s.http("POST", d.P["host"], "/v1/images/{name}/push", False, "FAIL0" if i == 0 else "RETRY", error_class=d.err(1), error_location=loc2, body={"ref": "main"})
    for ss, cc in ((s, c), (t, d)):
        if tail:
            explore(cc, tail, j0=3)
    pair(pid, "RES", s, t)


def res_poll(pid, pre, proj, npend, tail=0):
    s, c = mk(pid + "p", pre, proj)
    for i in range(npend):
        s.poll(c.P["res"], "pending", "FAIL0" if i == 0 else "POLL", backoff=2 ** i)
    s.poll(c.P["res"], "ready", "RESOLVES", backoff=2 ** npend)
    t, d = mk(pid + "n", pre, proj)
    for i in range(npend + 1):
        d.s.poll(d.P["res"], "pending", "FAIL0" if i == 0 else "POLL", backoff=2 ** i)
    for ss, cc in ((s, c), (t, d)):
        if tail:
            explore(cc, tail, j0=3)
    pair(pid, "RES", s, t)


def res_tests(pid, pre, proj, tail=0):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    base = c.base_text(p)
    n = 10 + hnum(pid, hi=9)
    s.tests(c.suite(), n - 1, n, "FAIL0", error_class=c.err(2), error_location=p)
    s.read(p, lines=(80, 140))
    s.edit(p, variant(base, 3, "repair"), dp=0.0)
    s.tests(c.suite(), n, n, "RESOLVES")
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path()
    b2 = d.base_text(p2)
    d.s.tests(d.suite(), n - 1, n, "FAIL0", error_class=d.err(2), error_location=p2)
    d.s.read(p2, lines=(80, 140))
    d.s.edit(p2, variant(b2, 3, "repair"), dp=0.0)
    d.s.tests(d.suite(), n - 1, n, "RETRY", error_class=d.err(2), error_location=p2)
    for ss, cc in ((s, c), (t, d)):
        if tail:
            explore(cc, tail, j0=3)
    pair(pid, "RES", s, t)


def res_build(pid, pre, proj, family="build", tail=0):
    s, c = mk(pid + "p", pre, proj)
    p = c.path()
    base = c.base_text(p)
    s.build(c.target(), False, "FAIL0", error_class=c.err(2), error_location=p, family=family)
    s.edit(p, variant(base, 3, "repair"), dp=0.0)
    s.build(c.target(), True, "RESOLVES", family=family)
    t, d = mk(pid + "n", pre, proj)
    p2 = d.path()
    b2 = d.base_text(p2)
    d.s.build(d.target(), False, "FAIL0", error_class=d.err(2), error_location=p2, family=family)
    d.s.edit(p2, variant(b2, 3, "repair"), dp=0.0)
    d.s.build(d.target(), False, "RETRY", error_class=d.err(2), error_location=p2, family=family)
    for ss, cc in ((s, c), (t, d)):
        if tail:
            explore(cc, tail, j0=3)
    pair(pid, "RES", s, t)


# ---------------- ERR: error present vs ok ------------------------------------------------------------------
def err_pair(pid, pre, proj, kind):
    s, c = mk(pid + "p", pre, proj)
    t, d = mk(pid + "n", pre, proj)
    p, p2 = c.path(), d.path()
    n = 11 + hnum(pid, hi=7)
    if kind == "tests":
        s.tests(c.suite(), n - 2, n, "NONE", error_class=c.err(), error_location=p)
        d.s.tests(d.suite(), n, n, "PROGRESS", dp=0.1)
    elif kind == "build":
        s.build(c.target(), False, "NONE", error_class=c.err(2), error_location=p)
        d.s.build(d.target(), True, "PROGRESS", dp=0.05)
    elif kind == "http":
        s.http("GET", c.P["host"], "/v1/status/{id}", False, "NONE", error_class=c.err(1), error_location=p)
        d.s.http("GET", d.P["host"], "/v1/status/{id}", True, "NEW_FACT")
    elif kind == "poll":
        s.poll(c.P["res"], "pending", "NONE", backoff=2)
        d.s.poll(d.P["res"], "ready", "NEW_FACT", backoff=2)
    elif kind == "read":
        s.read("missing/" + p, eff="NONE", outcome="ERROR", error_class="FileNotFound")
        d.s.read(p2)
    pair(pid, "ERR", s, t)


# ---------------- PROG: improving counts vs flat counts (fresh edits in both) -------------------------------
def prog_pair(pid, pre, proj, steps=3):
    s, c = mk(pid + "p", pre, proj)
    t, d = mk(pid + "n", pre, proj)
    n = 9 + hnum(pid, hi=8)
    for ss, cc, improving in ((s, c, True), (t, d, False)):
        p = cc.path()
        txt = cc.base_text(p)
        f0 = 4
        ss.tests(cc.suite(), n - f0, n, "NONE", error_class=cc.err(), error_location=p)
        for i in range(steps):
            txt = variant(txt, i + 1, f"e{i}")
            ss.edit(p, txt, dp=0.1 if improving else 0.0, eff="PROGRESS" if improving else "NONE")
            f = (f0 - (i + 1) * (f0 // steps + 1)) if improving else f0
            f = max(0, f) if i < steps - 1 or not improving else 0
            if f == 0:
                ss.tests(cc.suite(), n, n, "PROGRESS", dp=0.2)
            else:
                ss.tests(cc.suite(), n - f, n, "PROGRESS" if improving else "NONE", dp=0.1 if improving else 0.0,
                         error_class=cc.err(), error_location=p)
    pair(pid, "PROG", s, t)


def build_all():
    PAIRS.clear()
    j = 0
    for pre in (0, 2, 4):
        for proj in range(8):
            if (proj + pre) % 3 == 0:
                rep_reads(f"c{j}", pre, proj, 6 if proj % 2 else 4); j += 1
                rep_tests(f"c{j}", pre, proj, 6 if proj % 2 else 4); j += 1
                rev_ab(f"c{j}", pre, proj, 4, between=None if proj % 2 else "tests"); j += 1
                res_http(f"c{j}", pre, proj, 2, tail=proj % 3); j += 1
                err_pair(f"c{j}", pre, proj, ["tests", "build", "http", "poll", "read"][proj % 5]); j += 1
            if (proj + pre) % 3 == 1:
                rep_builds(f"c{j}", pre, proj, 6 if proj % 2 else 4, family="lint" if proj % 4 == 1 else "build"); j += 1
                rep_http(f"c{j}", pre, proj, 8 if proj % 2 else 4); j += 1
                rev_revert(f"c{j}", pre, proj, 3, between=None if proj % 2 else "build_ok"); j += 1
                res_poll(f"c{j}", pre, proj, 3, tail=proj % 3); j += 1
                prog_pair(f"c{j}", pre, proj, 3); j += 1
                rep_search(f"c{j}", pre, proj, 5); j += 1
            if (proj + pre) % 3 == 2:
                rep_polls(f"c{j}", pre, proj, 8 if proj % 2 else 4); j += 1
                rep_noop_edits(f"c{j}", pre, proj, 3); j += 1
                rev_write2(f"c{j}", pre, proj, 2); j += 1
                rev_abc(f"c{j}", pre, proj); j += 1
                res_tests(f"c{j}", pre, proj, tail=proj % 3); j += 1
                res_build(f"c{j}", pre, proj, family="lint" if proj % 2 else "build", tail=proj % 2); j += 1
                err_pair(f"c{j}", pre, proj, ["tests", "build", "http", "poll", "read"][(proj + 2) % 5]); j += 1
    return list(PAIRS)
