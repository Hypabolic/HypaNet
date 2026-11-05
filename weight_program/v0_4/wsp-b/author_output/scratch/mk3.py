"""WSP-B scenario library: realistic sessions (long busy history, then an 8-block window shaped like the disclosed
windows). Deterministic (hashlib). Builds on mk1.Session for rendering."""
import hashlib
from mk1 import Session, ftext, variant, hnum

VOCAB = [
    {"paths": ["src/billing/archive.py", "src/billing/ledger.py", "src/notify/pagination.py", "src/metrics/notifier.py", "src/core/config.py", "src/api/handlers.py", "tests/smoke/test_pagination.py", "src/billing/quota.py", "src/notify/router.py", "scripts/migrate.py", "src/core/cache.py", "src/api/auth.py"],
     "dirs": ["src", "tests", "scripts", "docs", "src/billing", "src/notify"], "suites": ["tests/regression", "tests/smoke", "tests/unit"], "builds": ["go-build", "py-wheel"], "lints": ["staticcheck", "ruff"],
     "hosts": [("GET", "ci.acme.dev", "/api/v4/jobs/{job_id}/trace"), ("POST", "maps.googleapis.com", "/v1/embeddings"), ("GET", "registry.acme.dev", "/v2/{name}/tags")],
     "res": ["deploy/staging/ledger", "replica/sync/7308", "build/queue/4471"], "errs": ["IndexError", "SyntaxError", "CompileError", "ineffassign", "KeyError", "TimeoutError"],
     "queries": ["flush_quota", "copyright", "import json", "metrics.increment", "def main", "manifest off_by_one", "retry_backoff", "TODO"], "plans": ["verify_plan", "choose_next_step", "checkpoint", "summarise_findings"]},
    {"paths": ["internal/media/rate_limiter.go", "internal/search/manifest.go", "internal/auth/token.go", "cmd/server/main.go", "internal/store/pg.go", "internal/media/encoder.go", "internal/search/index.go", "pkg/util/retry.go", "internal/auth/session.go", "cmd/cli/root.go", "internal/store/cache.go", "pkg/util/log.go"],
     "dirs": ["internal", "cmd", "pkg", "docs", "internal/media", "internal/search"], "suites": ["contract", "integration", "unit"], "builds": ["go-build", "docker-image"], "lints": ["staticcheck", "golangci"],
     "hosts": [("GET", "ci.corp.example", "/pipelines/{id}/status"), ("POST", "hooks.corp.example", "/v1/notify"), ("PUT", "blob.corp.example", "/buckets/{b}/objects/{k}")],
     "res": ["deploy/prod/canary", "index/rebuild/19", "cache/warm/3"], "errs": ["CompileError", "NilPointer", "DataRace", "ineffassign", "AssertionError", "HTTP503"],
     "queries": ["rate_limit", "manifest", "func New", "context.WithTimeout", "// FIXME", "token expiry", "encoder flush", "cache miss"], "plans": ["checkpoint", "verify_plan", "triage_failure", "summarise_findings"]},
    {"paths": ["web/src/components/Cart.tsx", "web/src/store/cart.ts", "web/src/api/client.ts", "web/src/pages/checkout.tsx", "web/src/utils/money.ts", "web/src/components/Nav.tsx", "web/src/__tests__/cart.test.ts", "web/src/store/user.ts", "web/src/api/types.ts", "web/vite.config.ts", "web/src/hooks/useCart.ts", "web/src/utils/format.ts"],
     "dirs": ["web/src", "web", "web/src/store", "web/src/api", "web/src/components", "docs"], "suites": ["web/unit", "web/e2e", "web/contract"], "builds": ["vite-build", "tsc"], "lints": ["eslint", "prettier"],
     "hosts": [("GET", "api.shop.example", "/v2/cart/{id}"), ("POST", "api.shop.example", "/v2/checkout"), ("GET", "cdn.shop.example", "/assets/{hash}")],
     "res": ["preview/deploy/88", "e2e/grid/2", "cdn/purge/7"], "errs": ["TypeError", "ReferenceError", "HydrationError", "LintError", "RangeError", "FetchError"],
     "queries": ["useCart", "formatMoney", "TODO remove", "fetchCart", "cart total", "checkout flow", "import type", "console.log"], "plans": ["choose_next_step", "checkpoint", "verify_plan", "summarise_findings"]},
    {"paths": ["zork/blat/fizzle.qz", "zork/blat/wump.qz", "zork/snarf.qz", "zork/blat/throb.qz", "zork/quib.qz", "zork/blat/oont.qz", "zork/tests/gorp.qz", "zork/blat/muft.qz", "zork/plugh.qz", "zork/xyzzy.qz", "zork/blat/vorp.qz", "zork/frotz.qz"],
     "dirs": ["zork", "zork/blat", "zork/tests", "docs", "zork/tools", "zork/lib"], "suites": ["gamma", "delta", "epsilon"], "builds": ["blat_bundle", "zork_image"], "lints": ["quuxlint", "fizzfmt"],
     "hosts": [("GET", "snarfhost.zork", "/v9/gadgets/{gid}"), ("POST", "snarfhost.zork", "/v9/jobs"), ("GET", "wumphost.zork", "/status/{id}")],
     "res": ["zork_mill", "zeta_hopper", "gronk/bake/12"], "errs": ["WumpError", "ThrobFault", "QuibPanic", "GrueError", "FrobError", "QuuxError"],
     "queries": ["frobnicate", "wump_level", "def snarf", "oont flag", "TODO plugh", "quib ratio", "throb timer", "fizzle"], "plans": ["sketch_zeta_zigzag", "ponder_options", "checkpoint", "verify_plan"]},
]


def pick(seed, seq, j=0):
    return seq[hnum(seed, "pick", j, hi=len(seq))]


class R:
    """One realistic session: a vocabulary, a busy history, then a class-specific window."""

    def __init__(self, sid, vocab_idx=None):
        self.sid = sid
        self.V = VOCAB[hnum(sid, "vocab", hi=len(VOCAB)) if vocab_idx is None else vocab_idx]
        starts = ["2026-07-01T03:07:10.000+00:00", "2026-07-02T06:01:31.000+00:00", "2026-07-03T08:19:56.000+00:00", "2026-07-04T05:53:47.000+00:00"]
        self.s = Session(sid, start=starts[hnum(sid, "start", hi=4)])
        self.texts = {}
        self.used_paths = []
        self.used_queries = []
        self.k = 0

    # ---- primitives --------------------------------------------------------------------------
    def fresh_path(self):
        for p in self.V["paths"]:
            if p not in self.used_paths:
                self.used_paths.append(p)
                return p
        p = self.V["paths"][self.k % len(self.V["paths"])]; self.k += 1
        return p

    def base(self, p):
        if p not in self.texts:
            self.texts[p] = ftext(p.split("/")[-1].split(".")[0], n=5 + hnum(self.sid, p, hi=4))
            self.s.set_file(p, self.texts[p])
        return self.texts[p]

    def read(self, p=None, new=True, lines=False, eff="NEW_FACT"):
        if p is None:
            p = self.fresh_path() if new else (self.used_paths[0] if self.used_paths else self.fresh_path())
        elif p not in self.used_paths:
            self.used_paths.append(p)
        kw = {}
        if lines:
            a = hnum(self.sid, "ln", len(self.s.ops), lo=1, hi=60); kw["lines"] = (a, a + hnum(self.sid, "lw", len(self.s.ops), lo=4, hi=40))
        return self.s.read(p, eff=eff, **kw)

    def search(self, q=None, new=True, eff="NEW_FACT"):
        if q is None:
            qs = [x for x in self.V["queries"] if x not in self.used_queries]
            q = (qs[0] if qs else self.V["queries"][len(self.s.ops) % len(self.V["queries"])]) if new else (self.used_queries[0] if self.used_queries else self.V["queries"][0])
        if q not in self.used_queries:
            self.used_queries.append(q)
        scope = pick(self.sid + str(len(self.s.ops)), self.V["dirs"])
        lim = hnum(self.sid, "lim", len(self.s.ops), hi=3)
        return self.s.search(scope, q, eff=eff, limit=[None, 5, 10][lim])

    def list_dir(self, eff="NEW_FACT"):
        return self.s.list_dir(pick(self.sid + str(len(self.s.ops)), self.V["dirs"]), eff=eff)

    def plan(self):
        return self.s.plan(pick(self.sid + str(len(self.s.ops)), self.V["plans"]))

    def http(self, ok=True, j=None, eff="NEW_FACT", body=None):
        m, h, p = self.V["hosts"][(hnum(self.sid, "host", len(self.s.ops), hi=3) if j is None else j) % 3]
        return self.s.http(m, h, p, ok, eff if ok else "NONE", body=body, error_class=None if ok else pick(self.sid, self.V["errs"], 5), error_location=None if ok else self.used_paths[0] if self.used_paths else self.V["paths"][0])

    def poll(self, state, j=0, backoff=None, eff="POLL"):
        return self.s.poll(self.V["res"][j % 3], state, eff, backoff=backoff)

    def tests(self, passed, total, suite=None, eff="NONE", dp=0.0, err=None, loc=None, tests=None):
        suite = suite or self.V["suites"][0]
        if passed < total:
            return self.s.tests(suite, passed, total, eff, dp=dp, error_class=err or self.V["errs"][0], error_location=loc or (self.used_paths[0] if self.used_paths else self.V["paths"][0]), tests=tests)
        return self.s.tests(suite, passed, total, eff, dp=dp, tests=tests)

    def build(self, ok, kind="build", eff="NONE", dp=0.0):
        tgt = self.V["builds"][0] if kind == "build" else self.V["lints"][0]
        return self.s.build(tgt, ok, eff, dp=dp, error_class=None if ok else self.V["errs"][2 if kind == "build" else 3], error_location=None if ok else (self.used_paths[0] if self.used_paths else self.V["paths"][0]), family=kind)

    def edit(self, p, txt, dp=0.0, eff="NONE", family="edit_file"):
        if family == "write_file":
            return self.s.write(p, txt, eff=eff, dp=dp)
        return self.s.edit(p, txt, eff=eff, dp=dp)

    def revert(self, p, txt):
        return self.s.revert({p: txt})

    # ---- a busy history of n blocks (mixed, with a couple of failures and a fix) ---------------
    def history(self, n):
        seed = self.sid + "hist"
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

    # ---- windows (8 blocks) ---------------------------------------------------------------------
    def win_prod(self, v):
        """PRODUCTIVE: >= 3 distinct new reads/searches, some failures, no repeats."""
        p = self.fresh_path(); self.base(p)
        seq = [
            ["read", "search", "read", "search", "build_fail", "read", "list", "search"],
            ["search", "build_fail", "lint_fail", "search", "search", "search", "http", "list"],
            ["test_fail", "read", "read", "read", "search", "http", "plan", "plan"],
            ["list", "list", "read", "list", "search", "edit", "test_fail", "read"],
            ["edit", "test_fail", "read", "search", "edit2", "test_ok", "test_fail2", "read"],
            ["read", "lint_fail", "read", "http", "read", "read", "search", "read"],
            ["poll_ok", "search", "read", "search", "read", "edit", "test_ok", "build_ok"],
            ["lint_fail", "read", "edit", "edit2", "test_ok", "test_fail2", "read", "build_fail"],
        ][v % 8]
        self._run_seq(seq, p)

    def win_reco(self, v):
        """RECOVERY: a bounded retry chain (<=3) or polls (<=5) with resolution, plus distinct reads."""
        p = self.fresh_path(); self.base(p)
        seq = [
            ["http_fail", "http_fail", "http_ok", "search", "read", "edit", "test_fail", "read"],
            ["poll_p", "poll_p", "poll_p", "poll_p", "plan", "poll_p", "plan", "poll_r"],
            ["test_fail", "read", "edit", "test_ok", "search", "http", "poll_p", "poll_r"],
            ["read", "lint_fail", "edit", "test_ok", "poll_p", "poll_p", "search", "poll_p"],
            ["list", "search", "read", "plan", "http_fail", "http_fail", "http_fail", "http_ok"],
            ["read", "http_fail", "http_fail", "plan", "http_ok", "search", "read", "lint_fail"],
            ["plan", "http", "http_fail", "poll_p", "poll_p", "poll_p", "poll_r", "test_fail"],
            ["search", "list", "plan", "read", "build_fail", "build_fail", "plan", "build_fail"],
        ][v % 8]
        self._run_seq(seq, p)

    def win_stal(self, v):
        """STALLED: >= 5 repeats of identical blocks, no new reads (re-reads only), no reversal."""
        p = self.fresh_path(); self.base(p)
        seq = [
            ["test_fail", "reread", "test_fail", "test_fail", "test_fail", "reread", "test_fail", "test_fail"],
            ["edit_f", "test_fail", "edit_f", "test_fail", "edit_f", "test_fail", "edit_f", "test_fail"],
            ["test_fail", "test_fail", "test_fail", "test_fail", "test_fail", "test_fail", "test_fail", "test_fail"],
            ["reread", "reread", "reread", "reread", "list_r", "edit_f", "reread", "reread"],
            ["poll_p", "poll_p", "reread", "poll_p", "reread", "poll_p", "poll_p", "test_fail"],
            ["plan_r", "test_fail", "plan_r", "test_fail", "test_fail", "reread", "test_fail", "test_fail"],
            ["test_okr", "edit_f", "build_okr", "edit_f", "test_okr", "edit_f", "test_okr", "edit_f"],
            ["search_r", "test_fail", "search_r", "test_fail", "plan_r", "list_r", "test_fail", "search_r"],
        ][v % 8]
        self._run_seq(seq, p)

    def win_osc(self, v):
        """OSCILLATING: >= 2 reversals (ids return to earlier states), with or without validation."""
        p = self.fresh_path(); self.base(p)
        seq = [
            ["ab", "ab", "ab", "ab", "ab", "ab", "ab", "ab"],
            ["test_fail", "ab", "test_fail", "ab", "test_fail", "ab", "test_fail", "ab"],
            ["abw", "test_okr", "ab", "test_okr", "abw", "test_okr", "ab", "test_okr"],
            ["lint_okr", "ab", "lint_okr", "ab", "lint_okr", "ab", "lint_okr", "ab"],
            ["build_okr", "rev", "build_okr", "ab", "build_okr", "rev", "build_okr", "ab"],
            ["rev", "ab", "rev", "ab", "rev", "ab", "rev", "ab"],
            ["read", "search", "read", "ab", "ab", "ab", "ab", "ab"],
            ["ab", "test_fail", "ab", "test_fail", "ab", "test_fail", "lint_fail", "ab"],
        ][v % 8]
        self._run_seq(seq, p)

    def _run_seq(self, seq, p):
        b = self.base(p)
        A, B = b, variant(b, 3, "alt")
        state = {"ab": 0, "n_fail": 7 + hnum(self.sid, "nf", hi=6), "fresh": 0, "suite": self.V["suites"][hnum(self.sid, "suite", hi=3)]}
        tf = state["n_fail"]
        for i, kind in enumerate(seq):
            if kind == "read":
                self.read(lines=bool(i % 2))
            elif kind == "reread":
                self.read(p, new=False, eff="REDUNDANT")
            elif kind == "search":
                self.search()
            elif kind == "search_r":
                self.search(q=self.V["queries"][0], new=False, eff="REDUNDANT")
            elif kind == "list":
                self.list_dir()
            elif kind == "list_r":
                self.s.list_dir(self.V["dirs"][0], eff="REDUNDANT")
            elif kind == "plan":
                self.plan()
            elif kind == "plan_r":
                self.s.plan(self.V["plans"][2], eff="REDUNDANT")
            elif kind == "http":
                self.http(True)
            elif kind == "http_ok":
                self.http(True, j=1, eff="RESOLVES", body={"ref": "main"})
            elif kind == "http_fail":
                self.http(False, j=1, body={"ref": "main"})
            elif kind == "poll_p":
                self.poll("pending", backoff=min(2 ** i, 64) if i % 2 else None)
            elif kind == "poll_r":
                self.poll("ready", eff="RESOLVES")
            elif kind == "poll_ok":
                self.poll("ready", j=1, eff="NEW_FACT")
            elif kind == "test_fail":
                self.tests(tf - 2, tf, suite=state["suite"], loc=p)
            elif kind == "test_fail2":
                self.tests(tf - 1, tf, suite=state["suite"], loc=p)
            elif kind == "test_ok":
                self.tests(tf, tf, suite=state["suite"], eff="PROGRESS", dp=0.1)
            elif kind == "test_okr":
                self.tests(tf, tf, suite=state["suite"], eff="REDUNDANT")
            elif kind == "build_fail":
                self.build(False)
            elif kind == "build_ok":
                self.build(True, eff="PROGRESS", dp=0.05)
            elif kind == "build_okr":
                self.build(True, eff="REDUNDANT")
            elif kind == "lint_fail":
                self.build(False, "lint")
            elif kind == "lint_okr":
                self.build(True, "lint", eff="REDUNDANT")
            elif kind == "edit":
                state["fresh"] += 1
                self.edit(p, variant(self.s.files[p], state["fresh"], f"fix{state['fresh']}"), dp=0.1, eff="PROGRESS")
            elif kind == "edit2":
                q = self.fresh_path(); bq = self.base(q)
                self.edit(q, variant(bq, 2, "fix"), dp=0.1, eff="PROGRESS")
            elif kind == "edit_f":  # fresh-id edits that do not change validation (stall family 6)
                state["fresh"] += 1
                self.edit(p, variant(self.s.files[p], state["fresh"], f"try{state['fresh']}"), eff="REDUNDANT")
            elif kind == "ab":
                state["ab"] += 1
                self.edit(p, B if state["ab"] % 2 else A, eff="REVERSION" if state["ab"] > 1 else "NONE")
            elif kind == "abw":
                state["ab"] += 1
                self.edit(p, B if state["ab"] % 2 else A, eff="REVERSION" if state["ab"] > 1 else "NONE", family="write_file")
            elif kind == "rev":
                self.revert(p, A if self.s.files[p] != A else B)
            else:
                raise ValueError(kind)

    def blocks(self):
        return self.s.blocks()
