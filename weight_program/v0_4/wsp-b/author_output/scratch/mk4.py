"""mk3 with an explicit seed for the start time and the busy history, so that matched-pair members share them.
R4(sid, vocab_idx, seed): identical to mk3.R except the history and start time derive from `seed` (default sid)."""
from mk3 import R, VOCAB, pick
from mk1 import Session, ftext, variant, hnum


class R4(R):
    def __init__(self, sid, vocab_idx=None, seed=None):
        self.sid = sid
        self.seed = seed or sid
        self.V = VOCAB[hnum(self.seed, "vocab", hi=len(VOCAB)) if vocab_idx is None else vocab_idx]
        starts = ["2026-07-01T03:07:10.000+00:00", "2026-07-02T06:01:31.000+00:00", "2026-07-03T08:19:56.000+00:00", "2026-07-04T05:53:47.000+00:00"]
        self.s = Session(sid, start=starts[hnum(self.seed, "start", hi=4)])
        self.texts = {}
        self.used_paths = []
        self.used_queries = []
        self.k = 0

    def history(self, n):
        seed = self.seed + "hist"
        p = self.fresh_path(); self.base(p)
        plan = ["read", "search", "list", "plan", "edit_ok", "test_fail", "test_ok", "build_ok", "http_ok", "read", "search", "edit_ok", "lint_fail", "read", "test_ok"]
        for i in range(n):
            kind = plan[(hnum(seed, i, hi=len(plan)) + i) % len(plan)]
            if kind == "read":
                self.read(lines=bool(i % 2))
            elif kind == "search":
                self.search()
            elif kind == "list":
                self.list_dir()
            elif kind == "plan":
                self.plan()
            elif kind == "edit_ok":
                q = self.fresh_path(); b = self.base(q); self.edit(q, variant(b, i + 1, f"h{i}"), dp=0.1, eff="PROGRESS")
            elif kind == "test_fail":
                self.tests(5 + i % 3, 8 + i % 3)
            elif kind == "test_ok":
                self.tests(8 + i % 3, 8 + i % 3, eff="PROGRESS", dp=0.1)
            elif kind == "build_ok":
                self.build(True, eff="PROGRESS", dp=0.05)
            elif kind == "lint_fail":
                self.build(False, "lint")
            elif kind == "http_ok":
                self.http(True)
