"""Iteration 2 inputs: stress sessions (long prefixes crossing the 128-row block boundary, many
hunk lines with TRUNC, error and workspace lines, dropped fields, http bodies, search limits,
multi-file reverts, large passed counts, mixed tools)."""
import sys

from common import (ask, build, compare, edit, http, lint, list_dir, op, plan, poll, read, revert, search,
                    tests, write)

A = "x = 1\ny = 2\n"
B = "x = 1\ny = 3\n"
C = "x = 1\ny = 4\n"

SESSIONS = {}

# long session: exploration, then a stall, then recovery, then oscillation (crosses several 128-token blocks)
long = []
for i in range(6):
    long.append(read(f"pkg/mod{i}.qx", lines=(10 + i, 40 + i)))
long += [search("pkg", "widget frobnicate plinth", limit=5), list_dir("pkg/sub")]
long += [tests("core", 4, 12)] * 7
long += [http("POST", "svc.example", "/v2/things/{tid}", ok=False, body='{"k":"v"}')] * 3
long += [http("POST", "svc.example", "/v2/things/{tid}", ok=True, body='{"k":"v"}')]
long += [read("pkg/mod1.qx"), edit("pkg/mod1.qx", A, B), tests("core", 6, 12), edit("pkg/mod1.qx", B, A),
         tests("core", 4, 12), edit("pkg/mod1.qx", A, B), tests("core", 6, 12), edit("pkg/mod1.qx", B, A)]
long += [plan("regroup"), ask("which"), read("pkg/mod2.qx"), edit("pkg/mod2.qx", A, C), tests("core", 12, 12),
         lint("core"), build("core"), read("pkg/mod3.qx")]
SESSIONS["long"] = long

# many files changed at once (TRUNC line), then reverted (multi-path revert), repeated
files = [f"pkg/f{i}.qx" for i in range(7)]
big_edit = op("edit_file", "pkg/f0.qx", changed=[{"path": p, "before": A, "after": B} for p in files])
big_revert = op("revert", files, changed=[{"path": p, "before": B, "after": A} for p in files])
SESSIONS["multi_revert"] = [big_edit, big_revert, big_edit, big_revert, big_edit, tests("core", 3, 9)]

# dropped observations and untracked workspace (unscored in truth, must not crash)
d1 = tests("core", 5, 9)
d1["dropped"] = ["validation_passed", "input_tokens"]
d2 = tests("core", 5, 9)
d2["dropped"] = ["failure", "validation_suite", "digest_after"]
d3 = edit("pkg/a.qx", A, B)
d3["dropped"] = ["digest_before", "output_tokens"]
d4 = build("core", ok=False)
d4["validation"] = {"suite": "core", "passed": 0, "total": 1}
d4["dropped"] = ["validation_total"]
SESSIONS["dropped"] = [d1, d2, d3, d4, d1, d2, read("pkg/a.qx"), d1, d1]

# large passed counts (3-digit groups) and improvement
SESSIONS["big_counts"] = [
    tests("all", 120, 350), edit("pkg/a.qx", A, B), tests("all", 121, 350), edit("pkg/b.qx", A, B),
    tests("all", 999, 1000), edit("pkg/c.qx", A, B), tests("all", 1000, 1000),
]

# error edits, read errors with retries (diagnosed repair style), then success
SESSIONS["read_retry"] = [
    read("pkg/missing.qx", ok=False), read("pkg/missing.qx", ok=False), list_dir("pkg"), read("pkg/missing.qx", ok=True),
    edit("pkg/missing.qx", A, B), tests("core", 9, 9), read("pkg/x.qx"), read("pkg/y.qx"), read("pkg/z.qx"),
]

# write_file create / delete cycles (h0 ids must not match)
SESSIONS["create_delete"] = [
    write("pkg/new.qx", None, A), op("edit_file", "pkg/new.qx", changed=[{"path": "pkg/new.qx", "before": A, "after": None}]),
    write("pkg/other.qx", None, B), op("edit_file", "pkg/other.qx", changed=[{"path": "pkg/other.qx", "before": B, "after": None}]),
    write("pkg/new.qx", None, A), tests("core", 9, 9),
]

# polls with errors and a resolution, then the window slides out
SESSIONS["poll_error"] = [
    poll("res-1", "error"), poll("res-1", "pending", 1), poll("res-1", "pending", 2), poll("res-1", "ready"),
    read("pkg/a.qx"), read("pkg/b.qx"), read("pkg/c.qx"), read("pkg/d.qx"), read("pkg/e.qx"),
]

# mixed stall: no-op edits and repeated searches
SESSIONS["mixed_stall"] = [
    search("pkg", "needle"), edit("pkg/a.qx", A, A), search("pkg", "needle"), edit("pkg/a.qx", A, A),
    search("pkg", "needle"), edit("pkg/a.qx", A, A), search("pkg", "needle limit"), tests("core", 2, 9),
]

if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = True if "-v" in args else "dis"
    only = [a for a in args if a != "-v"] or list(SESSIONS)
    tot_d = tot_n = 0
    for name in only:
        print("=" * 100)
        print("SESSION", name)
        d, n = compare(SESSIONS[name], sid="sid-" + name, verbose=verbose)
        tot_d += d
        tot_n += n
    print("=" * 100)
    print(f"disagreements {tot_d} / {tot_n}")
