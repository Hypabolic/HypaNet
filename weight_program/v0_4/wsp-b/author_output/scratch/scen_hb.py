"""Sets H1b and H2b: the matched pairs of scen_h and scen_h2 with both members sharing the history (seed = pair id).
Features: NF1, NF2, REPF, OSCOK, RESH (H1b); PEND, REPH, REPP, NF3, RDS (H2b). Positive first."""
from mk4 import R4
from mk1 import hnum, variant

PAIRS = []


def base_pair(pid, vocab_idx, hist):
    out = []
    for tag in ("p", "n"):
        r = R4(pid + tag, vocab_idx, seed=pid); r.history(hist); out.append(r)
    return out


def nf1(pid, vocab_idx, hist, end):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, distinct in ((p, True), (n, False)):
        f = r.fresh_path(); r.base(f)
        first = True
        seq = ["test_fail", "read", "search", "test_fail", "read", "list", "test_fail", "read"] if end == "read" else ["read", "test_fail", "search", "read", "test_fail", "list", "read", "test_fail"]
        for i, kind in enumerate(seq):
            if kind == "test_fail":
                r.tests(6, 8, loc=f)
            elif kind == "read":
                if distinct or first:
                    r.read(lines=bool(i % 2)); first = False
                else:
                    r.read(r.used_paths[-1], new=False, eff="REDUNDANT")
            elif kind == "search":
                r.search() if distinct else r.search(q=r.V["queries"][0], new=False, eff="REDUNDANT")
            elif kind == "list":
                r.list_dir() if distinct else r.s.list_dir(r.V["dirs"][0], eff="REDUNDANT")
    PAIRS.append((pid, "NF1", +1, p)); PAIRS.append((pid, "NF1", -1, n))


def nf2(pid, vocab_idx, hist, end):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, reads in ((p, True), (n, False)):
        f = r.fresh_path(); r.base(f)
        seq = ["test_fail", "X", "X", "test_fail", "X", "X", "test_fail", "X"] if end == "read" else ["X", "test_fail", "X", "X", "test_fail", "X", "X", "test_fail"]
        for i, kind in enumerate(seq):
            if kind == "test_fail":
                r.tests(6, 8, loc=f)
            elif reads:
                [r.read, r.search, r.list_dir, r.read, r.search][i % 5]()
            else:
                [r.plan, lambda: r.poll("ready", j=i % 3, eff="NEW_FACT"), r.plan, lambda: r.http(True), r.plan][i % 5]()
    PAIRS.append((pid, "NF2", +1, p)); PAIRS.append((pid, "NF2", -1, n))


def repf(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, same in ((p, True), (n, False)):
        f = r.fresh_path(); r.base(f)
        for i in range(8):
            if i in (2, 5):
                r.read(r.used_paths[0], new=False, eff="REDUNDANT")
            elif same:
                r.tests(6, 8, loc=f)
            else:
                r.tests(4 + (i % 3), 8 + (i % 2), suite=r.V["suites"][i % 3], err=r.V["errs"][i % 6], loc=r.V["paths"][i % 12])
    PAIRS.append((pid, "REPF", +1, p)); PAIRS.append((pid, "REPF", -1, n))


def oscok(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    f = p.fresh_path(); A = p.base(f); B = variant(A, 3, "alt")
    for i in range(4):
        p.edit(f, B if i % 2 == 0 else A, eff="NONE" if i == 0 else "REVERSION", family="write_file" if i % 2 else "edit_file")
        p.tests(8, 8, eff="REDUNDANT")
    for i in range(4):
        q = n.fresh_path(); bq = n.base(q)
        n.edit(q, variant(bq, i + 1, f"v{i}"), dp=0.1, eff="PROGRESS", family="write_file" if i % 2 else "edit_file")
        n.tests(8, 8, eff="PROGRESS", dp=0.1)
    PAIRS.append((pid, "OSCOK", +1, p)); PAIRS.append((pid, "OSCOK", -1, n))


def resh(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, resolved in ((p, True), (n, False)):
        r.read()
        for i in range(3):
            r.http(False, j=1, body={"ref": "main"})
        if resolved:
            r.http(True, j=1, eff="RESOLVES", body={"ref": "main"})
        else:
            r.http(False, j=1, body={"ref": "main"})
        r.search(); r.read(); r.plan()
    PAIRS.append((pid, "RESH", +1, p)); PAIRS.append((pid, "RESH", -1, n))


def pend(pid, vocab_idx, hist, k):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, polls in ((p, True), (n, False)):
        r.read(); r.search()
        for i in range(k):
            if polls:
                r.poll("pending", backoff=2 ** i if i else None)
            else:
                [r.plan, lambda: r.http(True, j=i % 3), r.plan, lambda: r.http(True, j=(i + 1) % 3)][i % 4]()
        r.read()
        if polls:
            r.poll("ready", eff="RESOLVES")
        else:
            r.plan()
        for _ in range(8 - 4 - k):
            r.read(lines=True)
    PAIRS.append((pid, "PEND", +1, p)); PAIRS.append((pid, "PEND", -1, n))


def reph(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, same in ((p, True), (n, False)):
        r.search(); r.read()
        for i in range(3):
            if same:
                r.http(False, j=1, body={"ref": "main"})
            else:
                r.http(i == 1, j=i, body={"ref": "main"} if i == 1 else None)
        r.http(True, j=1, eff="RESOLVES", body={"ref": "main"})
        r.read(); r.plan()
    PAIRS.append((pid, "REPH", +1, p)); PAIRS.append((pid, "REPH", -1, n))


def repp(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, same in ((p, True), (n, False)):
        r.read(); r.plan()
        for i in range(4):
            if same:
                r.poll("pending", j=0, backoff=2 ** i if i else None)
            else:
                r.poll("ready" if i % 2 else "pending", j=i % 3, backoff=None, eff="NEW_FACT" if i % 2 else "POLL")
        r.search(); r.read()
    PAIRS.append((pid, "REPP", +1, p)); PAIRS.append((pid, "REPP", -1, n))


def nf3(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, reads in ((p, True), (n, False)):
        for i in range(8):
            if i in (1, 3, 5):
                r.http(False, j=1, body={"ref": "main"})
            elif reads:
                [r.read, r.search, r.list_dir, r.read, r.search][i % 5]()
            else:
                [r.plan, lambda: r.http(True, j=2), r.plan, lambda: r.poll("ready", j=2, eff="NEW_FACT"), r.plan][i % 5]()
    PAIRS.append((pid, "NF3", +1, p)); PAIRS.append((pid, "NF3", -1, n))


def rds(pid, vocab_idx, hist):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, distinct in ((p, True), (n, False)):
        old = r.used_paths[0]
        r.plan()
        for i in range(4):
            if distinct:
                r.read(lines=bool(i % 2))
            else:
                r.read(old, new=False, eff="REDUNDANT", lines=bool(i % 2))
        r.plan(); r.http(True); r.list_dir() if distinct else r.s.list_dir(r.V["dirs"][0], eff="REDUNDANT")
    PAIRS.append((pid, "RDS", +1, p)); PAIRS.append((pid, "RDS", -1, n))


def build_all():
    PAIRS.clear()
    j = 0
    for vocab_idx in range(4):
        for k in range(2):
            hist = 10 + hnum("hb", vocab_idx, k, hi=10)
            nf1(f"j{j}", vocab_idx, hist, "read" if k == 0 else "test"); j += 1
            nf2(f"j{j}", vocab_idx, hist + 1, "read" if k == 1 else "test"); j += 1
            repf(f"j{j}", vocab_idx, hist + 2); j += 1
            oscok(f"j{j}", vocab_idx, hist + 3); j += 1
            resh(f"j{j}", vocab_idx, hist + 4); j += 1
            pend(f"j{j}", vocab_idx, hist + 5, 3 + k); j += 1
            reph(f"j{j}", vocab_idx, hist + 6); j += 1
            repp(f"j{j}", vocab_idx, hist + 7); j += 1
            nf3(f"j{j}", vocab_idx, hist + 8); j += 1
            rds(f"j{j}", vocab_idx, hist + 9); j += 1
    return list(PAIRS)
