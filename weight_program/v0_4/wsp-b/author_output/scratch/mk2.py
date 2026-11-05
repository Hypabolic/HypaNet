"""Scenario library v2: mk1's Session plus a phase library and name tables, so that many varied sessions
can be composed from recipes. Deterministic (hashlib only)."""
from mk1 import Session, ftext, variant, hnum, write_request, CLASS_IDX  # noqa: F401

PROJECTS = [
    {"paths": ["src/core/parser.py", "src/core/lexer.py", "src/core/ast.py", "src/core/printer.py", "src/core/tokens.py", "src/core/errors.py"],
     "dirs": ["src/core", "src"], "suites": ["core", "core_parse"], "targets": ["core_lib"], "host": "api.internal.example",
     "errs": ["AssertionError", "SyntaxError", "KeyError"], "res": "core_job"},
    {"paths": ["lib/net/client.go", "lib/net/retry.go", "lib/net/server.go", "lib/net/dial.go", "lib/net/pool.go", "lib/net/conn.go"],
     "dirs": ["lib/net", "lib"], "suites": ["net", "net_integration"], "targets": ["netd"], "host": "registry.example.net",
     "errs": ["TimeoutError", "ConnectionRefused", "EOFError"], "res": "netd_rollout"},
    {"paths": ["app/models/user.rb", "app/controllers/users_controller.rb", "app/serializers/user_serializer.rb", "app/jobs/mailer.rb", "app/models/account.rb", "app/helpers/format.rb"],
     "dirs": ["app/models", "app"], "suites": ["models", "controllers"], "targets": ["web"], "host": "deploy.example.org",
     "errs": ["NoMethodError", "ActiveRecordError", "ArgumentError"], "res": "staging_job"},
    {"paths": ["pkg/sched/queue.rs", "pkg/sched/worker.rs", "pkg/sched/timer.rs", "pkg/sched/deadline.rs", "pkg/sched/lib.rs", "pkg/sched/tests/queue_test.rs"],
     "dirs": ["pkg/sched", "pkg"], "suites": ["sched", "sched_unit"], "targets": ["schedd"], "host": "ci.example.io",
     "errs": ["PanicError", "BorrowError", "OverflowError"], "res": "pipeline_4471"},
    {"paths": ["zork/zeta/widget.qx", "zork/zeta/gizmo_plinth.qx", "zork/frotz.qx", "zork/plugh.qx", "zork/gnusto.qx", "zork/xyzzy.qx"],
     "dirs": ["zork/zeta", "zork"], "suites": ["alpha", "beta"], "targets": ["zeta_bale"], "host": "quuxhost.zeta",
     "errs": ["QuuxError", "FrobError", "GrueError"], "res": "zeta_hopper"},
    {"paths": ["web/forms/validate.ts", "web/forms/submit.ts", "web/forms/schema.ts", "web/forms/index.ts", "web/api/client.ts", "web/util/log.ts"],
     "dirs": ["web/forms", "web"], "suites": ["forms", "web_e2e"], "targets": ["web_bundle"], "host": "blob.example.com",
     "errs": ["TypeError", "RangeError", "ValidationError"], "res": "bundle_build"},
    {"paths": ["cmd/tool/main.c", "cmd/tool/args.c", "cmd/tool/util.c", "cmd/tool/io.c", "include/tool.h", "cmd/tool/fmt.c"],
     "dirs": ["cmd/tool", "cmd"], "suites": ["tool", "tool_cli"], "targets": ["tool"], "host": "artifacts.example.dev",
     "errs": ["CompileError", "LinkError", "SegfaultError"], "res": "tool_release"},
    {"paths": ["svc/deploy/push.py", "svc/deploy/manifest.yaml", "svc/deploy/rollout.py", "svc/sync/upload.py", "svc/common/auth.py", "svc/deploy/config.py"],
     "dirs": ["svc/deploy", "svc"], "suites": ["deploy", "sync"], "targets": ["svc_image"], "host": "deploy.example.org",
     "errs": ["HTTP503", "AuthError", "ManifestError"], "res": "deploy_7f2"},
]
QUERIES = ["def parse_expr", "retry_backoff", "UserSerializer", "fn pop_ready", "frobnicate", "onSubmit", "parse_args", "rollout_wait",
           "TODO remove", "class Account", "timeout=", "ready_flag"]


class Ctx:
    """Name tables for one session, chosen by hash of the session id, with a cursor over fresh paths."""

    def __init__(self, s):
        self.s = s
        self.P = PROJECTS[hnum(s.sid, "proj", hi=len(PROJECTS))]
        self.k = hnum(s.sid, "k0", hi=6)
        self.texts = {}

    def path(self, i=None):
        if i is None:
            self.k += 1
            i = self.k
        return self.P["paths"][i % len(self.P["paths"])]

    def suite(self):
        return self.P["suites"][hnum(self.s.sid, "suite", hi=2)]

    def target(self):
        return self.P["targets"][0]

    def err(self, j=0):
        return self.P["errs"][(hnum(self.s.sid, "err") + j) % 3]

    def query(self, j):
        return QUERIES[(hnum(self.s.sid, "q") + j) % len(QUERIES)]

    def base_text(self, p):
        if p not in self.texts:
            self.texts[p] = ftext(p.split("/")[-1].split(".")[0], n=5 + hnum(self.s.sid, p, hi=4))
            self.s.set_file(p, self.texts[p])
        return self.texts[p]


# ---- phases: each appends ops to ctx.s ---------------------------------------------------------------

def explore(ctx, k=3, j0=0):
    s = ctx.s
    for j in range(k):
        kind = hnum(s.sid, "ex", j0 + j, hi=4)
        if kind == 0:
            s.list_dir(ctx.P["dirs"][j % 2])
        elif kind == 1:
            s.search(ctx.P["dirs"][j % 2], ctx.query(j0 + j), limit=(5 + 5 * (j % 2)) if j % 3 == 0 else None)
        elif kind == 2:
            p = ctx.path()
            s.read(p, lines=(10 * (j + 1), 10 * (j + 1) + 60))
        else:
            s.read(ctx.path())


def fix_and_pass(ctx, total=None, with_build=False, j=0):
    s = ctx.s
    p = ctx.path()
    base = ctx.base_text(p)
    s.edit(p, variant(base, j + 1, f"fix{j}"), dp=0.2)
    n = total or (8 + hnum(s.sid, "tot", j, hi=20))
    s.tests(ctx.suite(), n, n, "PROGRESS", dp=0.2)
    if with_build:
        s.build(ctx.target(), True, "PROGRESS", dp=0.05, family="lint" if j % 2 else "build")


def incremental_repair(ctx, steps=2, j=0):
    """Failing tests whose count improves after each edit; treated as progress, not as an episode."""
    s = ctx.s
    p = ctx.path()
    base = ctx.base_text(p)
    n = 9 + hnum(s.sid, "ir", j, hi=12)
    fails = [min(n - 1, 2 + steps + hnum(s.sid, "f0", j, hi=3))]
    s.tests(ctx.suite(), n - fails[0], n, "NONE", error_class=ctx.err(), error_location=p)
    txt = base
    for k in range(steps):
        if k == 0:
            s.read(p)
        txt = variant(txt, k + 2, f"step{k}")
        s.edit(p, txt, dp=0.15)
        left = fails[-1] - 1 - (k % 2)
        if k == steps - 1:
            left = 0
        fails.append(max(0, left))
        if fails[-1] == 0:
            s.tests(ctx.suite(), n, n, "PROGRESS", dp=0.2)
        else:
            s.tests(ctx.suite(), n - fails[-1], n, "PROGRESS", dp=0.15, error_class=ctx.err(), error_location=p)


def transient_http(ctx, nfail=2, method="POST", j=0):
    s = ctx.s
    path = ["/v1/images/{name}/push", "/jobs", "/v2/users/{id}", "/pipelines"][j % 4]
    loc = ctx.path(0)
    for k in range(nfail):
        s.http(method, ctx.P["host"], path, False, "FAIL0" if k == 0 else "RETRY", error_class=ctx.err(1), error_location=loc,
               body={"ref": "main"} if method == "POST" else None)
    s.http(method, ctx.P["host"], path, True, "RESOLVES", body={"ref": "main"} if method == "POST" else None)


def polls_ok(ctx, npend=3):
    s = ctx.s
    for k in range(npend):
        s.poll(ctx.P["res"], "pending", "FAIL0" if k == 0 else "POLL", backoff=2 ** k)
    s.poll(ctx.P["res"], "ready", "RESOLVES", backoff=2 ** npend)


def diagnosed_repair(ctx, kind="build", j=0):
    s = ctx.s
    p = ctx.path()
    base = ctx.base_text(p)
    if kind == "build":
        s.build(ctx.target(), False, "FAIL0", error_class=ctx.err(2), error_location=p)
    elif kind == "lint":
        s.build(ctx.target(), False, "FAIL0", error_class=ctx.err(2), error_location=p, family="lint")
    else:
        n = 10 + hnum(s.sid, "dr", j, hi=10)
        s.tests(ctx.suite(), n - 1, n, "FAIL0", error_class=ctx.err(2), error_location=p)
    s.read(p, lines=(80, 140))
    s.edit(p, variant(base, j + 3, "repair"), dp=0.1)
    if kind == "build":
        s.build(ctx.target(), True, "RESOLVES")
    elif kind == "lint":
        s.build(ctx.target(), True, "RESOLVES", family="lint")
    else:
        s.tests(ctx.suite(), n, n, "RESOLVES")


def retry_active(ctx, n=2, kind="http", j=0):
    """An episode still open at the end, within policy."""
    s = ctx.s
    loc = ctx.path(1)
    for k in range(n):
        if kind == "http":
            s.http("GET", ctx.P["host"], "/v1/status/{id}", False, "FAIL0" if k == 0 else "RETRY", error_class=ctx.err(1), error_location=loc)
        elif kind == "read":
            s.read("missing/" + loc, eff="FAIL0" if k == 0 else "RETRY", outcome="ERROR", error_class="FileNotFound")
        else:
            s.poll(ctx.P["res"], "pending", "FAIL0" if k == 0 else "POLL", backoff=2 ** k)


def stall_tests(ctx, k=6, with_read=False):
    s = ctx.s
    p = ctx.path()
    n = 7 + hnum(s.sid, "st", hi=10)
    f = 1 + hnum(s.sid, "sf", hi=4)
    s.tests(ctx.suite(), n - f, n, "NONE", error_class=ctx.err(), error_location=p)
    for i in range(k):
        if with_read and i % 2 == 0:
            s.read(p, eff="REDUNDANT")
        s.tests(ctx.suite(), n - f, n, "REDUNDANT", error_class=ctx.err(), error_location=p)


def stall_reads(ctx, k=7, two=False):
    s = ctx.s
    p1, p2 = ctx.path(), ctx.path()
    s.read(p1)
    if two:
        s.read(p2)
    for i in range(k):
        s.read(p1 if (not two or i % 2 == 0) else p2, eff="REDUNDANT")


def stall_search(ctx, k=6):
    s = ctx.s
    q = ctx.query(3)
    s.search(ctx.P["dirs"][0], q)
    for i in range(k):
        s.search(ctx.P["dirs"][0], q, eff="REDUNDANT")


def stall_noop_edits(ctx, k=4, validate="tests"):
    s = ctx.s
    p = ctx.path()
    base = ctx.base_text(p)
    txt = variant(base, 2, "attempt")
    s.edit(p, txt, eff="NONE")
    n = 6 + hnum(s.sid, "ne", hi=8)
    if validate == "tests":
        s.tests(ctx.suite(), n - 2, n, "NONE", error_class=ctx.err(), error_location=p)
    else:
        s.build(ctx.target(), False, "NONE", error_class=ctx.err(), error_location=p)
    for i in range(k):
        s.edit(p, txt, eff="REDUNDANT")  # no change: no hunk line
        if validate == "tests":
            s.tests(ctx.suite(), n - 2, n, "REDUNDANT", error_class=ctx.err(), error_location=p)
        else:
            s.build(ctx.target(), False, "REDUNDANT", error_class=ctx.err(), error_location=p)


def stall_http(ctx, k=9, method="PUT"):
    s = ctx.s
    loc = ctx.path(2)
    for i in range(k + 1):
        s.http(method, ctx.P["host"], "/buckets/{b}/objects/{k}", False, "FAIL0" if i == 0 else "RETRY", error_class=ctx.err(1), error_location=loc)


def stall_polls(ctx, k=11):
    s = ctx.s
    for i in range(k + 1):
        s.poll(ctx.P["res"], "pending", "FAIL0" if i == 0 else "POLL", backoff=min(2 ** i, 64))


def stall_build(ctx, k=6, with_read=True, family="build"):
    s = ctx.s
    p = ctx.path()
    s.build(ctx.target(), False, "NONE", error_class=ctx.err(2), error_location=p, family=family)
    for i in range(k):
        if with_read and i % 2 == 0:
            s.read(p, eff="REDUNDANT")
        s.build(ctx.target(), False, "REDUNDANT", error_class=ctx.err(2), error_location=p, family=family)


def osc_ab(ctx, k=4, between="tests"):
    """Alternating edits A->B->A->B; optional validation between them."""
    s = ctx.s
    p = ctx.path()
    A = ctx.base_text(p)
    B = variant(A, 3, "alt")
    n = 8 + hnum(s.sid, "oab", hi=8)

    def val(eff):
        if between == "tests":
            s.tests(ctx.suite(), n - 2, n, eff, error_class=ctx.err(), error_location=p)
        elif between == "build_ok":
            s.build(ctx.target(), True, "NONE" if eff == "NONE" else "REDUNDANT")
        elif between == "lint":
            s.build(ctx.target(), False, eff, error_class=ctx.err(), error_location=p, family="lint")

    s.edit(p, B, eff="NONE")
    if between:
        val("NONE")
    for i in range(k):
        s.edit(p, A if i % 2 == 0 else B, eff="REVERSION")
        if between:
            val("REDUNDANT")


def osc_revert(ctx, k=3, between=None):
    s = ctx.s
    p = ctx.path()
    A = ctx.base_text(p)
    B = variant(A, 4, "try")
    s.edit(p, B, eff="NONE")
    if between == "build_ok":
        s.build(ctx.target(), True, "NONE")
    elif between == "tests":
        s.tests(ctx.suite(), 5, 7, "NONE", error_class=ctx.err(), error_location=p)
    for i in range(k):
        s.revert({p: A})
        s.edit(p, B, eff="REVERSION")
        if between == "build_ok":
            s.build(ctx.target(), True, "REDUNDANT")
        elif between == "tests":
            s.tests(ctx.suite(), 5, 7, "REDUNDANT", error_class=ctx.err(), error_location=p)


def osc_write2(ctx, k=2):
    s = ctx.s
    p1, p2 = ctx.path(), ctx.path()
    a1, a2 = ctx.base_text(p1), ctx.base_text(p2)
    s.write(p1, variant(a1, 1, "v2"), eff="NONE")
    s.write(p2, variant(a2, 1, "v2"), eff="NONE")
    for i in range(k):
        s.revert({p1: a1, p2: a2})
        s.write(p1, variant(a1, 1, "v2"), eff="REVERSION")
        s.write(p2, variant(a2, 1, "v2"), eff="REVERSION")


def osc_abc(ctx):
    s = ctx.s
    p = ctx.path()
    A = ctx.base_text(p)
    B, C = variant(A, 1, "b"), variant(A, 2, "c")
    s.edit(p, B, eff="NONE")
    s.edit(p, C, eff="NONE")
    s.build(ctx.target(), True, "NONE")
    for txt in (A, B, C, A):
        s.edit(p, txt, eff="REVERSION")
    s.build(ctx.target(), True, "REDUNDANT")


def flaky_once(ctx, j=0):
    s = ctx.s
    p = ctx.path()
    n = 12 + hnum(s.sid, "fl", j, hi=9)
    s.tests(ctx.suite(), n - 1, n, "FAIL0", error_class="FlakyTimeout", error_location=p, tests=[ctx.suite() + "/spin_wobble"])
    s.tests(ctx.suite(), n, n, "RESOLVES", tests=[ctx.suite() + "/spin_wobble"])


def plan(ctx, j=0):
    ctx.s.plan(["outline_fix", "summarise_changes", "sketch_zeta_zigzag", "ponder_options"][j % 4])
