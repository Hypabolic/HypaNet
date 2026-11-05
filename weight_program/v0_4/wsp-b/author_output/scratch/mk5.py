"""mk3/mk4 with a Session whose numeric noise (token counts, timestamps, line ranges, picks) derives from an
explicit seed, so that matched-pair members (different session ids) render identical histories."""
from datetime import datetime, timedelta
from mk1 import Session, ftext, variant, hnum
from mk3 import VOCAB


class Session5(Session):
    def __init__(self, sid, seed, start):
        super().__init__(sid, start=start)
        self.seed = seed
        self.in_tok = hnum(seed, "in0", lo=1200, hi=2600)

    def _stamp(self, i):
        self.t += timedelta(seconds=hnum(self.seed, "dt", i, lo=2, hi=40), milliseconds=hnum(self.seed, "ms", i, lo=0, hi=999))
        return self.t.isoformat(timespec="milliseconds")

    def add(self, family, target, eff, dp=0.0, args=None, outcome="SUCCESS", error_class=None, error_location=None,
            validation=None, changed=None, out_tok=None, extra=None, dropped=None):
        i = len(self.ops)
        self.in_tok += hnum(self.seed, "din", i, lo=40, hi=400)
        if out_tok is None:
            out_tok = hnum(self.seed, "out", i, lo=12, hi=400)
        op = {"family": family, "target": target, "args": args or {}, "outcome": outcome,
              "error_class": error_class, "error_location": error_location, "validation": validation,
              "changed": changed or [], "extra": extra, "input_tokens": self.in_tok, "output_tokens": out_tok,
              "timestamp": self._stamp(i), "dropped": sorted(dropped or [])}
        self.ops.append(op)
        self.ann.append({"eff": eff, "dp": dp})
        return op

    # per-tool out_tok ranges as in mk1 but seeded by self.seed
    def read(self, path, eff="NEW_FACT", lines=None, outcome="SUCCESS", error_class=None):
        args = {"line_start": lines[0], "line_end": lines[1]} if lines else {}
        return self.add("read_file", path, eff, args=args, outcome=outcome, error_class=error_class,
                        error_location=path if outcome == "ERROR" else None, out_tok=hnum(self.seed, "rd", len(self.ops), lo=40, hi=900))

    def search(self, scope, query, eff="NEW_FACT", limit=None):
        args = {"query": query}
        if limit:
            args["limit"] = limit
        return self.add("search", scope, eff, args=args, out_tok=hnum(self.seed, "se", len(self.ops), lo=10, hi=120))

    def list_dir(self, path, eff="NEW_FACT"):
        return self.add("list_dir", path, eff, out_tok=hnum(self.seed, "ls", len(self.ops), lo=10, hi=60))

    def plan(self, action, eff="NONE", family="plan"):
        return self.add(family, None, eff, args={"action_type": action}, out_tok=hnum(self.seed, "pl", len(self.ops), lo=200, hi=900))

    def edit(self, path, new_text, eff="PROGRESS", dp=0.0, outcome="SUCCESS"):
        before = self.files.get(path)
        changed = [] if new_text == before else [{"path": path, "before": before, "after": new_text}]
        if new_text != before:
            self.files[path] = new_text
        return self.add("edit_file", path, eff, dp=dp, changed=changed, outcome=outcome, out_tok=hnum(self.seed, "ed", len(self.ops), lo=120, hi=600))

    def write(self, path, new_text, eff="PROGRESS", dp=0.0):
        before = self.files.get(path)
        changed = [] if new_text == before else [{"path": path, "before": before, "after": new_text}]
        self.files[path] = new_text
        return self.add("write_file", path, eff, dp=dp, args={"content": "x"}, changed=changed, out_tok=hnum(self.seed, "wr", len(self.ops), lo=200, hi=900))

    def revert(self, paths_texts, eff="REVERSION"):
        changed = []
        for p, txt in paths_texts.items():
            before = self.files.get(p)
            if txt != before:
                changed.append({"path": p, "before": before, "after": txt})
            self.files[p] = txt
        return self.add("revert", list(paths_texts), eff, changed=changed, out_tok=hnum(self.seed, "rv", len(self.ops), lo=8, hi=40))

    def tests(self, suite, passed, total, eff, dp=0.0, error_class=None, error_location=None, tests=None):
        ok = passed == total
        args = {"tests": tests} if tests else {}
        return self.add("run_tests", suite, eff, dp=dp, args=args, outcome="SUCCESS" if ok else "ERROR",
                        error_class=None if ok else error_class, error_location=None if ok else error_location,
                        validation={"suite": suite, "passed": passed, "total": total}, out_tok=hnum(self.seed, "te", len(self.ops), lo=60, hi=400))

    def build(self, target, ok, eff, dp=0.0, error_class=None, error_location=None, family="build"):
        return self.add(family, target, eff, dp=dp, outcome="SUCCESS" if ok else "ERROR",
                        error_class=None if ok else error_class, error_location=None if ok else error_location, out_tok=hnum(self.seed, "bd", len(self.ops), lo=20, hi=200))

    def http(self, method, host, path, ok, eff, dp=0.0, body=None, error_class=None, error_location=None):
        args = {"body": body} if body is not None else {}
        return self.add("http_call", {"method": method, "host": host, "path_template": path}, eff, dp=dp, args=args,
                        outcome="SUCCESS" if ok else "ERROR", error_class=None if ok else error_class, error_location=None if ok else error_location,
                        out_tok=hnum(self.seed, "ht", len(self.ops), lo=10, hi=200))

    def poll(self, resource, state, eff, dp=0.0, backoff=None):
        args = {"backoff": backoff} if backoff is not None else {}
        outcome = {"ready": "SUCCESS", "pending": "NEUTRAL", "error": "ERROR"}[state]
        return self.add("poll_status", resource, eff, dp=dp, args=args, outcome=outcome, out_tok=hnum(self.seed, "po", len(self.ops), lo=8, hi=40))


def pick(seed, seq, j=0):
    return seq[hnum(seed, "pick", j, hi=len(seq))]


class R5:
    """Realistic session whose every choice derives from `seed` (shared within a pair); `sid` only names it."""

    def __init__(self, sid, vocab_idx=None, seed=None):
        self.sid = sid
        self.seed = seed or sid
        self.V = VOCAB[hnum(self.seed, "vocab", hi=len(VOCAB)) if vocab_idx is None else vocab_idx]
        starts = ["2026-07-01T03:07:10.000+00:00", "2026-07-02T06:01:31.000+00:00", "2026-07-03T08:19:56.000+00:00", "2026-07-04T05:53:47.000+00:00"]
        self.s = Session5(sid, self.seed, start=starts[hnum(self.seed, "start", hi=4)])
        self.texts = {}
        self.used_paths = []
        self.used_queries = []
        self.k = 0

    def fresh_path(self):
        for p in self.V["paths"]:
            if p not in self.used_paths:
                self.used_paths.append(p)
                return p
        p = self.V["paths"][self.k % len(self.V["paths"])]; self.k += 1
        return p

    def base(self, p):
        if p not in self.texts:
            self.texts[p] = ftext(p.split("/")[-1].split(".")[0], n=5 + hnum(self.seed, p, hi=4))
            self.s.set_file(p, self.texts[p])
        return self.texts[p]

    def read(self, p=None, new=True, lines=False, eff="NEW_FACT"):
        if p is None:
            p = self.fresh_path() if new else (self.used_paths[0] if self.used_paths else self.fresh_path())
        elif p not in self.used_paths:
            self.used_paths.append(p)
        kw = {}
        if lines:
            a = hnum(self.seed, "ln", len(self.s.ops), lo=1, hi=60); kw["lines"] = (a, a + hnum(self.seed, "lw", len(self.s.ops), lo=4, hi=40))
        return self.s.read(p, eff=eff, **kw)

    def search(self, q=None, new=True, eff="NEW_FACT"):
        if q is None:
            qs = [x for x in self.V["queries"] if x not in self.used_queries]
            q = (qs[0] if qs else self.V["queries"][len(self.s.ops) % len(self.V["queries"])]) if new else (self.used_queries[0] if self.used_queries else self.V["queries"][0])
        if q not in self.used_queries:
            self.used_queries.append(q)
        scope = pick(self.seed + str(len(self.s.ops)), self.V["dirs"])
        lim = hnum(self.seed, "lim", len(self.s.ops), hi=3)
        return self.s.search(scope, q, eff=eff, limit=[None, 5, 10][lim])

    def list_dir(self, eff="NEW_FACT"):
        return self.s.list_dir(pick(self.seed + str(len(self.s.ops)), self.V["dirs"]), eff=eff)

    def plan(self):
        return self.s.plan(pick(self.seed + str(len(self.s.ops)), self.V["plans"]))

    def http(self, ok=True, j=None, eff="NEW_FACT", body=None):
        m, h, p = self.V["hosts"][(hnum(self.seed, "host", len(self.s.ops), hi=3) if j is None else j) % 3]
        return self.s.http(m, h, p, ok, eff if ok else "NONE", body=body, error_class=None if ok else pick(self.seed, self.V["errs"], 5),
                           error_location=None if ok else self.used_paths[0] if self.used_paths else self.V["paths"][0])

    def poll(self, state, j=0, backoff=None, eff="POLL"):
        return self.s.poll(self.V["res"][j % 3], state, eff, backoff=backoff)

    def tests(self, passed, total, suite=None, eff="NONE", dp=0.0, err=None, loc=None, tests=None):
        suite = suite or self.V["suites"][0]
        if passed < total:
            return self.s.tests(suite, passed, total, eff, dp=dp, error_class=err or self.V["errs"][0], error_location=loc or (self.used_paths[0] if self.used_paths else self.V["paths"][0]), tests=tests)
        return self.s.tests(suite, passed, total, eff, dp=dp, tests=tests)

    def build(self, ok, kind="build", eff="NONE", dp=0.0):
        tgt = self.V["builds"][0] if kind == "build" else self.V["lints"][0]
        return self.s.build(tgt, ok, eff, dp=dp, error_class=None if ok else self.V["errs"][2 if kind == "build" else 3],
                            error_location=None if ok else (self.used_paths[0] if self.used_paths else self.V["paths"][0]), family=kind)

    def edit(self, p, txt, dp=0.0, eff="NONE", family="edit_file"):
        if family == "write_file":
            return self.s.write(p, txt, eff=eff, dp=dp)
        return self.s.edit(p, txt, eff=eff, dp=dp)

    def revert(self, p, txt):
        return self.s.revert({p: txt})

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

    def blocks(self):
        return self.s.blocks()
