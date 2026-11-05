"""Set H2: more matched pairs under realistic context.
  PEND   pending polls (3-4, then a ready) among reads   vs  plans / http ok in their place
  REPH   identical http error repeated 3x then ok        vs  three different http calls (mixed ok/error) then ok
  REPP   identical poll pending x4                       vs  four different polls (ready/pending mixed)
  NF3    distinct reads/searches among http retries      vs  plans in their place
  RDS    4 distinct reads of new files                    vs  4 re-reads of a file read in the history (no failures)
Positive first."""
from mk3 import R
from mk1 import hnum

PAIRS = []


def base_pair(pid, vocab_idx, hist):
    out = []
    for tag in ("p", "n"):
        r = R(pid + tag, vocab_idx); r.history(hist); out.append(r)
    return out


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
            hist = 11 + hnum("h2", vocab_idx, k, hi=10)
            pend(f"i{j}", vocab_idx, hist, 3 + k); j += 1
            reph(f"i{j}", vocab_idx, hist + 1); j += 1
            repp(f"i{j}", vocab_idx, hist + 2); j += 1
            nf3(f"i{j}", vocab_idx, hist + 3); j += 1
            rds(f"i{j}", vocab_idx, hist + 4); j += 1
    return list(PAIRS)
