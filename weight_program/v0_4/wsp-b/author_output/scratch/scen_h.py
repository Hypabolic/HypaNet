"""Set H: matched pairs under realistic context (same busy history, 8-block windows differing in one property):
  NF1  distinct new reads/searches interleaved with failing tests  vs  the same file re-read / same query repeated
  NF2  distinct reads/searches interleaved with failing tests      vs  plans / polls-ready in their place (no reads)
  REPF identical failing test repeated                             vs  different failing tests (suite/count differ)
  OSCOK A/B edits with passing tests                               vs  fresh edits on different files with passing tests
  RESH http retries then ok then reads                             vs  http retries continuing then reads
Positive first."""
from mk3 import R
from mk1 import hnum, variant

PAIRS = []


def base_pair(pid, vocab_idx, hist):
    out = []
    for tag in ("p", "n"):
        r = R(pid + tag, vocab_idx)
        r.history(hist)
        out.append(r)
    return out


def nf1(pid, vocab_idx, hist, end):
    p, n = base_pair(pid, vocab_idx, hist)
    for r, distinct in ((p, True), (n, False)):
        f = r.fresh_path(); r.base(f)
        first = True
        for i, kind in enumerate(["test_fail", "read", "search", "test_fail", "read", "list", "test_fail", "read"][: 8 if end == "read" else 7] + ([] if end == "read" else ["test_fail"])):
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


def build_all():
    PAIRS.clear()
    j = 0
    for vocab_idx in range(4):
        for k in range(2):
            hist = 10 + hnum("h", vocab_idx, k, hi=10)
            nf1(f"h{j}", vocab_idx, hist, "read" if k == 0 else "test"); j += 1
            nf2(f"h{j}", vocab_idx, hist + 1, "read" if k == 1 else "test"); j += 1
            repf(f"h{j}", vocab_idx, hist + 2); j += 1
            oscok(f"h{j}", vocab_idx, hist + 3); j += 1
            resh(f"h{j}", vocab_idx, hist + 4); j += 1
    return list(PAIRS)
